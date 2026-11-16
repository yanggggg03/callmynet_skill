'use strict';
const $ = id => document.getElementById(id);
const IDS = ['residual_mlp', 'fourier_resmlp', 'conditional_flow'];
const LABELS = {residual_mlp:'残差 MLP', fourier_resmlp:'Fourier ResMLP', conditional_flow:'条件 Flow'};
const COLORS = {residual_mlp:'#4a84dd', fourier_resmlp:'#d99a39', conditional_flow:'#129679'};
const S = {bundle:null, batch:null, analysis:null, visible:new Set(IDS), api:localStorage.getItem('callmynet-api') || '', busy:false};
const escapeHTML = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const num = (value, digits=2) => value === null || value === undefined ? '—' : Number(value).toFixed(digits);
const configurationKeys = ['max_cv_pct','max_ripple_pct','max_error_pct','max_drift_pct','max_overshoot_pct','steady_start_fraction'];
function notice(message, kind='') { $('notice').textContent = message; $('notice').className = 'notice' + (kind ? ' '+kind : ''); }
function base() { return S.api.replace(/\/$/, ''); }
async function api(path, payload, blob=false) {
  const controller = new AbortController();
  const timeout = setTimeout(()=>controller.abort(), 800000);
  try {
    const response = await fetch(base()+path, {method:payload ? 'POST' : 'GET', headers:payload ? {'Content-Type':'application/json'} : {}, body:payload ? JSON.stringify(payload) : undefined, signal:controller.signal});
    if (!response.ok) {
      const error = await response.json().catch(()=>({error:'服务返回 '+response.status}));
      throw Error(error.error || '请求失败');
    }
    return blob ? response.blob() : response.json();
  } catch(error) {
    if (error.name === 'AbortError') throw Error('服务请求超时，请检查本地推理进程。');
    if (error instanceof TypeError) throw Error('无法连接推理服务。请启动本地服务；GitHub页面可先使用示例，或打开本地面板。');
    throw error;
  } finally { clearTimeout(timeout); }
}
async function health() {
  try {
    const health = await api('/api/health');
    const count = health.models.filter(m=>m.checkpoint_ready).length;
    $('connection').textContent = `本地服务已连接 · 权重 ${count}/3`;
    $('connection').className = 'connection ready';
    $('model-chips').innerHTML = health.models.map(m=>`<span class="${m.checkpoint_ready?'ready':''}">${escapeHTML(m.label)} ${m.checkpoint_ready?'✓':'待训练'}</span>`).join('');
    return true;
  } catch(error) {
    $('connection').textContent = '未连接本地服务'; $('connection').className = 'connection';
    return false;
  }
}
function download(blob, filename) {
  const link = document.createElement('a'); link.href=URL.createObjectURL(blob); link.download=filename;
  link.click(); setTimeout(()=>URL.revokeObjectURL(link.href),5000);
}
async function upload(file) {
  if(file.size > 15_000_000) throw Error('单个文件请控制在15MB以内。');
  const encoded = await new Promise((resolve,reject)=>{
    const reader=new FileReader(); reader.onload=()=>resolve(reader.result.split(',')[1]);reader.onerror=reject;reader.readAsDataURL(file);
  });
  return {filename:file.name,content_base64:encoded};
}
function options() {
  const result=Object.fromEntries(configurationKeys.map(key=>[key,Number($(key).value)]));
  for(const key of ['evaluation_start_s','evaluation_end_s'])if($(key).value.trim()!=='')result[key]=Number($(key).value);
  return result;
}
function useBundle(bundle) {
  S.bundle=bundle;S.analysis=bundle.waveform_analysis || null;
  if(bundle.mode==='demo'){
    $('unit').value=bundle.magnitude_unit;
    if(!S.batch){$('mag').value=bundle.records[0].mag;$('deg').value=bundle.records[0].deg;}
  }
  for(const key of ['evaluation_start_s','evaluation_end_s'])$(key).value=S.analysis?.settings[key]??'';
  if(S.analysis)for(const key of configurationKeys)if(S.analysis.settings[key]!==undefined)$(key).value=S.analysis.settings[key];
  $('sample-select').innerHTML=bundle.records.map(r=>`<option value="${escapeHTML(r.sample_id)}">${escapeHTML(r.sample_id)}</option>`).join('');
  $('result-empty').classList.add('hidden');$('result-data').classList.remove('hidden');$('export').disabled=false;
  $('result-mode').textContent=bundle.mode==='demo'?'合成示例':'模型推理';$('result-mode').className='badge '+(bundle.mode==='demo'?'demo':'success');
  $('run-id').textContent='RUN '+bundle.run_id;
  notice(bundle.notice,bundle.mode==='demo'?'demo':'');
  renderResults();renderAnalysis();
}
function renderResults() {
  if(!S.bundle)return;
  const id=$('sample-select').value;
  const records=S.bundle.predictions.filter(r=>String(r.sample_id)===id);
  const provisional=S.bundle.estimated_recommendations.find(r=>String(r.sample_id)===id);
  const verified=S.analysis && S.analysis.run_id===S.bundle.run_id && String(S.analysis.sample_id)===id;
  const winner=verified ? S.analysis.winner : null;
  const prefix=S.bundle.mode==='demo'?'示例 · ':'';
  let title,caption,detail;
  let coefficient=null;
  if(winner){
    const result=records.find(r=>r.model_id===winner);
    coefficient=result;
    title=prefix+LABELS[winner];caption='波形比较中满足稳定性判据的最佳方案';
    detail=`L1 ${num(result.L1)} ${result.length_unit} · L2 ${num(result.L2)} ${result.length_unit} · ${S.analysis.source==='measured'?'实测':S.analysis.source==='simulation'?'仿真':'预设示意'}功率波形`;
  } else if(verified) {
    title='三套方案均未满足稳定性判据';caption=prefix+'当前工况的波形比较';detail='请检查波形、调节方案与设备要求，不将最低分方案标记为稳定方案。';
  } else {
    title=prefix+LABELS[provisional.model_id];caption='预计匹配推荐 · 等待功率波形验证';detail=`L1 ${num(provisional.L1)} ${provisional.length_unit} · L2 ${num(provisional.L2)} ${provisional.length_unit} · 共用响应代理评分`;
    coefficient=provisional;
  }
  const coefficientMarkup=coefficient?`<div class="coefficient-output"><span>预计匹配后反射系数（线性幅值）</span><b>|Γ| = ${num(coefficient.estimated_matched_mag,5)}</b><span>相位 ∠Γ = ${num(coefficient.estimated_matched_deg)}° · 无量纲，非百分比</span></div>`:'';
  $('recommendation').innerHTML=`<span class="caption">${escapeHTML(caption)}</span><strong>${escapeHTML(title)}</strong>${coefficientMarkup}<p>${escapeHTML(detail)}</p>`;
  document.querySelectorAll('.length-unit').forEach(el=>el.textContent=records[0].length_unit==='deg'?'°':records[0].length_unit);
  $('result-rows').innerHTML=records.map(r=>`<tr class="${r.model_id===(winner || provisional.model_id)?'best':''}"><td>${escapeHTML(LABELS[r.model_id])}${r.outside_training_input_box?'<small>超出训练输入范围</small>':''}</td><td>${num(r.L1)}</td><td>${num(r.L2)}</td><td>${num(r.estimated_matched_mag,5)}</td><td>${num(r.estimated_matched_deg)}</td></tr>`).join('');
  $('result-note').textContent=S.bundle.mode==='demo'?'示例反射系数与动作均为预设值，不能用来判断三个网络的真实优劣。':'返回匹配后反射系数幅值|Γ|及相位，幅值是0～1的线性值，不是反射功率百分比。数值为统一响应代理预测，并非新调节后的实测；低幅值时相位对噪声敏感。';
}
function renderAnalysis() {
  const analysis=S.analysis;
  $('wave-empty').classList.toggle('hidden',Boolean(analysis));$('chart').classList.toggle('hidden',!analysis);$('zoom-control').classList.toggle('hidden',!analysis);
  $('reanalyze').disabled=!analysis;
  if(!analysis){
    $('wave-source').textContent='待接入';$('wave-source').className='badge';
    $('metrics').innerHTML='<div><span>稳态波动 CV</span><strong>—</strong></div><div><span>峰峰纹波 / 目标</span><strong>—</strong></div><div><span>稳态偏差</span><strong>—</strong></div><div><span>调节判定</span><strong>待验证</strong></div>';
    $('metric-table').innerHTML='';return;
  }
  const source={demo:'合成示例',measured:'实测波形',simulation:'仿真波形'}[analysis.source];
  const hasChannels=analysis.waveforms.every(r=>r.incident_power_w!==undefined&&r.reflected_power_w!==undefined);
  for(const option of $('power-channel').querySelectorAll('option'))if(option.value!=='power_w')option.disabled=!hasChannels;
  if(!hasChannels)$('power-channel').value='power_w';
  $('wave-source').textContent=source+' · 样本 '+analysis.sample_id;$('wave-source').className='badge '+(analysis.source==='demo'?'demo':'success');
  const best=analysis.metrics[0];
  $('metrics').innerHTML=`<div><span>首选方案 CV</span><strong>${num(best.cv_pct)}<small>%</small></strong></div><div><span>峰峰纹波 / 目标</span><strong>${num(best.ripple_pct)}<small>%</small></strong></div><div><span>稳态偏差</span><strong>${num(best.error_pct)}<small>%</small></strong></div><div><span>${analysis.winner?escapeHTML(LABELS[analysis.winner]):'全部方案'}</span><strong class="${analysis.winner?'positive':'negative'}">${analysis.winner?'满足判据':'未达标'}</strong></div>`;
  $('metric-table').innerHTML='<table><thead><tr><th>模型</th><th>CV %</th><th>纹波 %</th><th>偏差 %</th><th>漂移 %</th><th>超调 %</th><th>评分↓</th><th>判定</th></tr></thead><tbody>'+analysis.metrics.map(m=>`<tr><td>${escapeHTML(LABELS[m.model_id])}</td><td>${num(m.cv_pct)}</td><td>${num(m.ripple_pct)}</td><td>${num(m.error_pct)}</td><td>${num(m.drift_pct)}</td><td>${num(m.overshoot_pct)}</td><td>${num(m.score,3)}</td><td>${m.stable?'达标':'未达标'}</td></tr>`).join('')+'</tbody></table>';
  renderChart();
}
function renderChart() {
  const analysis=S.analysis;if(!analysis)return;
  const all=analysis.waveforms; const svg=$('chart');
  const channel=$('power-channel').value||'power_w';
  const field=row=>row[channel]??row.power_w;
  const times=all.filter(r=>r.model_id===IDS[0]).map(r=>r.time_s);
  const begin=times[0], end=times[times.length-1], duration=end-begin;
  const zoom=Number($('zoom').value), span=duration/zoom;
  const low=begin+(duration-span)*Number($('pan').value)/100, high=low+span;
  const windowed=all.filter(r=>r.time_s>=low && r.time_s<=high && S.visible.has(r.model_id));
  const target=channel==='reflected_power_w'?0:all[0].target_w;
  let min=target,max=target;
  for(const row of windowed){min=Math.min(min,field(row));max=Math.max(max,field(row));}
  const padding=Math.max((max-min)*.14,all[0].target_w*.005);min-=padding;max+=padding;
  const box={left:60,right:700,top:22,bottom:265};
  const x=t=>box.left+(t-low)/span*(box.right-box.left);
  const y=p=>box.bottom-(p-min)/(max-min)*(box.bottom-box.top);
  const evaluationStart=analysis.settings.evaluation_start_s??begin;
  const evaluationEnd=analysis.settings.evaluation_end_s??end;
  const steadyStart=evaluationStart+(evaluationEnd-evaluationStart)*analysis.settings.steady_start_fraction;
  const shadeStart=Math.max(low,steadyStart),shadeEnd=Math.min(high,evaluationEnd);
  let markup=`<rect x="${x(shadeStart)}" y="${box.top}" width="${Math.max(0,x(shadeEnd)-x(shadeStart))}" height="${box.bottom-box.top}" fill="#f2f8f5"/>`;
  for(let n=0;n<=4;n++){
    const py=box.top+(box.bottom-box.top)*n/4;const value=max-(max-min)*n/4;
    markup+=`<line x1="${box.left}" y1="${py}" x2="${box.right}" y2="${py}" stroke="#e9eef1"/><text x="${box.left-9}" y="${py+4}" text-anchor="end" fill="#98a6af" font-size="10">${num(value,0)}</text>`;
  }
  for(let n=0;n<=5;n++){
    const time=low+span*n/5;markup+=`<text x="${x(time)}" y="${box.bottom+21}" text-anchor="middle" fill="#98a6af" font-size="10">${num(time,2)}</text>`;
  }
  markup+=`<text x="17" y="15" fill="#8b9ca6" font-size="10">功率 / W</text><text x="700" y="305" text-anchor="end" fill="#8b9ca6" font-size="10">时间 / s</text><line x1="${box.left}" y1="${y(target)}" x2="${box.right}" y2="${y(target)}" stroke="#9aa8b1" stroke-dasharray="5 5"/>`;
  if(channel==='reflected_power_w'&&all[0].target_w*.05>=min&&all[0].target_w*.05<=max){const py=y(all[0].target_w*.05);markup+=`<line x1="${box.left}" y1="${py}" x2="${box.right}" y2="${py}" stroke="#c49b63" stroke-dasharray="3 4"/><text x="${box.right-4}" y="${py-6}" text-anchor="end" fill="#ad8a55" font-size="9">5%目标功率参考线</text>`;}
  for(const id of IDS){
    if(!S.visible.has(id))continue;
    const points=all.filter(r=>r.model_id===id&&r.time_s>=low&&r.time_s<=high);
    // At most 1400 displayed points per line. Export always preserves the full input.
    const stride=Math.max(1,Math.ceil(points.length/1400));
    const selected=points.filter((_,i)=>i%stride===0||i===points.length-1);
    markup+=`<path d="${selected.map((r,i)=>(i?'L':'M')+x(r.time_s).toFixed(2)+','+y(field(r)).toFixed(2)).join(' ')}" fill="none" stroke="${COLORS[id]}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>`;
  }
  if(steadyStart>=low&&steadyStart<=high)markup+=`<text x="${x(steadyStart)+7}" y="${box.top+13}" font-size="9" fill="#91ac9f">稳态观察窗口</text>`;
  svg.innerHTML=markup;
  svg.onpointermove=event=>{
    const ratio=(event.clientX-svg.getBoundingClientRect().left)/svg.getBoundingClientRect().width;
    const time=low+Math.max(0,Math.min(1,(ratio*720-box.left)/(box.right-box.left)))*span;
    let html=`<b>时间 ${num(time,3)} s</b>`;
    for(const id of IDS){
      if(!S.visible.has(id))continue;
      const values=all.filter(r=>r.model_id===id);
      let l=0,h=values.length-1;while(l<h){const mid=(l+h)>>1;if(values[mid].time_s<time)l=mid+1;else h=mid;}
      const row=l>0 && Math.abs(values[l-1].time_s-time)<Math.abs(values[l].time_s-time)?values[l-1]:values[l];
      html+=`<br><span style="color:${COLORS[id]}">●</span> ${LABELS[id]} ${num(field(row))} W`;
    }
    $('chart-tooltip').innerHTML=html;$('chart-tooltip').classList.remove('hidden');
  };
  svg.onpointerleave=()=>$('chart-tooltip').classList.add('hidden');
}
// Small uncompressed ZIP/XLSX writer for the public GitHub Pages demo, offline-capable.
const utf8=new TextEncoder();
const xml=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&apos;'}[c]));
function crc32(bytes){let crc=0xffffffff;for(const byte of bytes){crc^=byte;for(let n=0;n<8;n++)crc=(crc>>>1)^((crc&1)?0xedb88320:0);}return (crc^0xffffffff)>>>0;}
function zipped(files){
  const parts=[],central=[];let offset=0,centralSize=0;
  for(const [filename,text] of files){const name=utf8.encode(filename),data=utf8.encode(text),crc=crc32(data);
    const local=new Uint8Array(30),v=new DataView(local.buffer);v.setUint32(0,0x04034b50,true);v.setUint16(4,20,true);v.setUint32(14,crc,true);v.setUint32(18,data.length,true);v.setUint32(22,data.length,true);v.setUint16(26,name.length,true);
    parts.push(local,name,data);
    const c=new Uint8Array(46),cv=new DataView(c.buffer);cv.setUint32(0,0x02014b50,true);cv.setUint16(4,20,true);cv.setUint16(6,20,true);cv.setUint32(16,crc,true);cv.setUint32(20,data.length,true);cv.setUint32(24,data.length,true);cv.setUint16(28,name.length,true);cv.setUint32(42,offset,true);central.push(c,name);offset+=30+name.length+data.length;centralSize+=46+name.length;
  }
  const end=new Uint8Array(22),ev=new DataView(end.buffer);ev.setUint32(0,0x06054b50,true);ev.setUint16(8,files.length,true);ev.setUint16(10,files.length,true);ev.setUint32(12,centralSize,true);ev.setUint32(16,offset,true);
  return new Blob([...parts,...central,end],{type:'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'});
}
function localWorkbook(sheets){
  const ns='http://schemas.openxmlformats.org/spreadsheetml/2006/main',rel='http://schemas.openxmlformats.org/officeDocument/2006/relationships';
  const column=i=>{let out='';for(i++;i;i=Math.floor((i-1)/26))out=String.fromCharCode(65+(i-1)%26)+out;return out;};
  const files=[['_rels/.rels',`<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="${rel}/officeDocument" Target="xl/workbook.xml"/></Relationships>`]];
  files.push(['[Content_Types].xml',`<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>${sheets.map((_,i)=>`<Override PartName="/xl/worksheets/sheet${i+1}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>`).join('')}</Types>`]);
  files.push(['xl/workbook.xml',`<workbook xmlns="${ns}" xmlns:r="${rel}"><sheets>${sheets.map((s,i)=>`<sheet name="${xml(s.name)}" sheetId="${i+1}" r:id="rId${i+1}"/>`).join('')}</sheets></workbook>`]);
  files.push(['xl/_rels/workbook.xml.rels',`<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">${sheets.map((_,i)=>`<Relationship Id="rId${i+1}" Type="${rel}/worksheet" Target="worksheets/sheet${i+1}.xml"/>`).join('')}</Relationships>`]);
  sheets.forEach((s,i)=>{const rows=[s.headers,...s.rows];files.push([`xl/worksheets/sheet${i+1}.xml`,`<worksheet xmlns="${ns}"><sheetData>${rows.map((row,r)=>`<row r="${r+1}">${row.map((value,c)=>{const address=column(c)+(r+1);if(typeof value==='number'&&Number.isFinite(value))return `<c r="${address}"><v>${value}</v></c>`;return `<c r="${address}" t="inlineStr"><is><t>${xml(typeof value==='object'?JSON.stringify(value):value)}</t></is></c>`;}).join('')}</row>`).join('')}</sheetData></worksheet>`]);});
  return zipped(files);
}
function exportLocally(){
  const b=S.bundle,a=S.analysis || {},headers=['sample_id','model_id','mag','deg','L1','L2','length_unit','estimated_matched_mag','estimated_matched_deg','estimated_matched_real','estimated_matched_imag','estimated_reflected_power_ratio','evaluator'];
  const metric=['model_id','stable','score','mean_power_w','cv_pct','ripple_pct','error_pct','drift_pct','overshoot_pct','settling_time_s'];
  const wave=['run_id','sample_id','scenario_id','model_id','time_s','power_w','target_w','source','incident_power_w','reflected_power_w','reflected_power_ratio'];
  const sheet=(name,keys,rows)=>({name,headers:keys,rows:rows.map(r=>keys.map(k=>r[k]??''))});
  const confirmed=b.predictions.filter(r=>r.model_id===a.winner&&String(r.sample_id)===String(a.sample_id)&&b.run_id===a.run_id);
  return localWorkbook([sheet('三模型结果',headers,b.predictions),sheet('预计推荐_待波形验证',headers,b.estimated_recommendations),sheet('稳定性评估',metric,a.metrics||[]),sheet('功率波形',wave,a.waveforms||[]),sheet('波形选优结果',headers,confirmed),{name:'说明',headers:['项目','内容'],rows:[['模式',b.mode],['run_id',b.run_id],['输入幅值单位',b.magnitude_unit],['主要返回量','estimated_matched_mag=线性反射系数幅值|Γ|，无量纲非百分比；estimated_matched_deg=相位°；real/imag=实部/虚部；reflected_power_ratio=辅助功率比|Γ|²'],['波形来源',a.source||'未接入'],['优胜模型',a.winner||'无已验证稳定方案'],['阈值',JSON.stringify(a.settings||{})],['波形参考',JSON.stringify(b.waveform_reference||{})],['说明',b.notice]]}]);
}
async function getDemo(){
  const response=await fetch('assets/demo.json');if(!response.ok)throw Error('未找到示例文件');return response.json();
}
async function busy(action){if(S.busy)return;S.busy=true;$('run').disabled=true;try{await action();}catch(error){notice(error.message,'error');}finally{S.busy=false;$('run').disabled=false;}}
$('settings-toggle').onclick=()=>$('settings').classList.toggle('hidden');$('api-url').value=S.api;
$('connect').onclick=()=>busy(async()=>{S.api=$('api-url').value.trim().replace(/\/$/,'');if(S.api&&!/^https?:\/\//.test(S.api))throw Error('服务地址须以http://或https://开头');localStorage.setItem('callmynet-api',S.api);if(await health())notice('已连接推理服务。');else throw Error('连接失败，请检查服务地址。');});
$('run').onclick=()=>busy(async()=>{notice('正在载入三个网络并比较候选方案，请稍候……');const records=S.batch||[{mag:Number($('mag').value),deg:Number($('deg').value)}];useBundle(await api('/api/predict',{records,magnitude_unit:$('unit').value}));});
$('demo').onclick=()=>busy(async()=>{useBundle(await getDemo());});
$('input-file').onchange=event=>busy(async()=>{const file=event.target.files[0];if(!file)return;const payload=await upload(file);const response=await api('/api/read-input',{...payload,magnitude_unit:$('unit').value});S.batch=response.records;$('input-summary').textContent=file.name+' · '+S.batch.length+'条';$('clear-batch').classList.remove('hidden');$('mag').disabled=$('deg').disabled=true;notice('输入表已载入。点击“运行三个模型”开始。');event.target.value='';});
$('clear-batch').onclick=()=>{S.batch=null;$('input-summary').textContent='单条输入 · mag, deg';$('clear-batch').classList.add('hidden');$('mag').disabled=$('deg').disabled=false;};
$('sample-select').onchange=renderResults;
for(const name of ['results','example'])$('tab-'+name).onclick=()=>{for(const tab of ['results','example']){$('tab-'+tab).classList.toggle('active',tab===name);$('tab-'+tab).setAttribute('aria-selected',tab===name?'true':'false');$(tab==='results'?'results-content':'example-content').classList.toggle('hidden',tab!==name);}};
$('wave-file').onchange=event=>busy(async()=>{const file=event.target.files[0];if(!file)return;const payload=await upload(file);const expected=S.bundle?.run_id;
  const analysis=await api('/api/analyze-waveforms',{...payload,source:$('wave-source-select').value,options:options(),expected_run_id:expected});
  if(S.bundle&&!S.bundle.records.some(r=>String(r.sample_id)===String(analysis.sample_id)))throw Error('波形sample_id不在当前推理批次内');
  if(S.bundle&&S.bundle.mode==='inference'&&analysis.source==='demo')throw Error('合成示例波形不能验证真实推理结果');
  S.analysis=analysis;if(S.bundle)S.bundle.waveform_analysis=analysis;else{S.bundle={mode:'waveforms_only',run_id:analysis.run_id,records:[],predictions:[],estimated_recommendations:[],notice:'独立导入的功率波形评估'};$('export').disabled=false;}
  renderAnalysis();if(S.bundle.predictions.length)renderResults();notice(analysis.winner?'功率波形比较完成，已选出满足当前稳定性判据的方案。':'功率波形比较完成：三套方案均未满足当前稳定性判据。',analysis.source==='demo'?'demo':'');event.target.value='';});
$('reanalyze').onclick=()=>busy(async()=>{if(!S.analysis)return;const response=await api('/api/analyze-waveforms',{rows:S.analysis.waveforms,source:S.analysis.source,expected_run_id:S.bundle?.run_id,options:options()});S.analysis=response;S.bundle.waveform_analysis=response;renderAnalysis();if(S.bundle.predictions.length)renderResults();notice('已按更新后的判据重新比较。',response.source==='demo'?'demo':'');});
$('export').onclick=()=>busy(async()=>{if(!S.bundle)return;download(exportLocally(),(S.bundle.mode==='demo'?'DEMO_':'')+'matching_results.xlsx');});
$('input-template').onclick=()=>download(localWorkbook([{name:'input',headers:['mag','deg'],rows:[]}]),'prediction_template.xlsx');
$('wave-template').onclick=()=>download(localWorkbook([{name:'waveforms',headers:['run_id','sample_id','scenario_id','model_id','time_s','power_w','target_w','source','incident_power_w','reflected_power_w'],rows:[]}]),'waveform_template.xlsx');
document.querySelectorAll('#legend button').forEach(button=>button.onclick=()=>{const id=button.dataset.model;if(S.visible.has(id)){if(S.visible.size===1)return;S.visible.delete(id);}else S.visible.add(id);button.classList.toggle('off',!S.visible.has(id));renderChart();});
$('zoom').oninput=$('pan').oninput=renderChart;
$('power-channel').onchange=renderChart;
$('reset-zoom').onclick=()=>{$('zoom').value=1;$('pan').value=0;renderChart();};
health();
if(new URLSearchParams(location.search).get('demo')==='1')busy(async()=>useBundle(await getDemo()));
