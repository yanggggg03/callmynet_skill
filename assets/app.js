'use strict';
const $ = id => document.getElementById(id);
let IDS = ['residual_mlp', 'fourier_resmlp', 'conditional_flow'];
const LABELS = {residual_mlp:'残差 MLP', fourier_resmlp:'Fourier ResMLP', conditional_flow:'条件 Flow'};
const COLORS = {residual_mlp:'#4a84dd', fourier_resmlp:'#d99a39', conditional_flow:'#129679'};
const S = {bundle:null, batch:null, filename:'', inputMode:'file', models:[], selected:new Set(), trainingActive:false, connected:false, readyModels:0, job:'idle', analysis:null, visible:new Set(IDS), api:localStorage.getItem('callmynet-api') || '', busy:false};
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
    if (error instanceof TypeError) throw Error('无法连接推理服务。请启动本地服务并检查连接设置，或打开 http://127.0.0.1:8765/ 本地面板。');
    throw error;
  } finally { clearTimeout(timeout); }
}
async function health() {
  try {
    const health = await api('/api/health');
    const count = health.models.filter(m=>m.checkpoint_ready).length;
    S.connected=true;S.readyModels=count;S.models=health.models;
    if(typeof refreshModelSelection==='function')refreshModelSelection(health);
    $('connection').textContent = `本地服务已连接 · 权重 ${count}/${health.models.length}`;
    $('connection').className = 'connection ready';
    $('model-chips').innerHTML = health.models.map(m=>`<span class="${m.checkpoint_ready?'ready':''}">${escapeHTML(m.label)} ${m.checkpoint_ready?'✓':'待训练'}</span>`).join('');
    syncControls();return true;
  } catch(error) {
    S.connected=false;S.readyModels=0;
    $('connection').textContent = '未连接本地服务'; $('connection').className = 'connection';syncControls();
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
    const reader=new FileReader(); reader.onload=()=>resolve(reader.result.split(',')[1]);reader.onerror=()=>reject(Error('无法读取所选文件，请重新选择。'));reader.readAsDataURL(file);
  });
  return {filename:file.name,content_base64:encoded};
}
function options() {
  const result=Object.fromEntries(configurationKeys.map(key=>[key,Number($(key).value)]));
  for(const key of ['evaluation_start_s','evaluation_end_s'])if($(key).value.trim()!=='')result[key]=Number($(key).value);
  return result;
}
function selectedModels(){return S.models.filter(m=>S.selected.has(m.id));}
function syncControls(){
  const count=S.inputMode==='file'?(S.batch?.length||0):1;
  $('run').disabled=S.busy||!count||!S.connected||!selectedModels().length||selectedModels().some(m=>!m.checkpoint_ready)||S.trainingActive;
  $('run-label').textContent=S.job==='running'?'正在计算…':`开始计算${count?' · '+count+' 条':''}`;
  for(const id of ['input-file','wave-file','unit','mode-file','mode-manual','clear-batch','mag','deg','connect'])$(id).disabled=S.busy;
  $('evaluation-model').disabled=S.busy;
  document.querySelectorAll('#inference-models input').forEach(input=>input.disabled=S.busy);
  $('upload-zone').classList.toggle('disabled',S.busy);
  let hint='文件校验通过后，即可开始计算。';
  if(S.job==='running')hint=`正在处理 ${count} 条工况，等待所选模型返回结果，请保持页面打开。`;
  else if(!S.connected)hint='请先启动本地推理服务，在右上角“连接设置”中连接。';
  else if(S.trainingActive)hint='正在训练，请等待结束或取消训练后计算。';
  else if(!selectedModels().length)hint='请先勾选参与预测的模型。';
  else if(selectedModels().some(m=>!m.checkpoint_ready))hint='所选模型尚无权重，请先训练并启用。';
  else if(count)hint=`${S.inputMode==='file'?'文件已就绪':'单条输入已就绪'}，点击“开始计算”运行 ${selectedModels().length} 个模型。`;
  $('run-hint').textContent=hint;
  $('job-state').textContent=({idle:S.inputMode==='file'?'等待上传':'单条输入',ready:'待处理',running:'正在处理',done:'处理完成',error:'处理失败'})[S.job];
  for(const id of ['flow-input','flow-results','flow-wave'])$(id).className='';
  $('flow-input').className=count?'complete':'current';
  $('flow-results').className=S.bundle?.predictions?.length?'complete':count?'current':'';
  $('flow-wave').className=S.analysis?'complete':S.bundle?.predictions?.length?'current':'';
}
function clearResults(message='上传并校验工况数据后，点击“开始计算”。'){
  S.bundle=null;S.analysis=null;S.job='idle';
  $('result-data').classList.add('hidden');$('result-empty').classList.remove('hidden');
  $('result-empty-title').textContent='尚无计算结果';$('result-empty-detail').textContent=message;
  $('result-mode').textContent='待运行';$('result-mode').className='badge';$('result-meta').textContent='';
  $('export').disabled=true;$('sample-select').innerHTML='';$('result-rows').innerHTML='';$('recommendation').innerHTML='';
  $('wave-upload-status').textContent='尚未导入波形文件。功率波形用于稳定性验证，不由模型直接生成。';
  $('wave-upload-status').className='wave-upload-status';renderAnalysis();
}
function fileStatus(state,title,detail,file){
  $('upload-status').dataset.state=state;$('upload-status').setAttribute('aria-busy',state==='reading'?'true':'false');
  $('upload-icon').textContent=({idle:'—',reading:'↻',success:'✓',error:'!'})[state];
  $('upload-title').textContent=title;$('upload-detail').textContent=detail;
  $('input-summary').textContent=file?`${file.name} · ${file.size>=1024?(file.size/1024).toFixed(1)+' KB':file.size+' B'}`:'';
  $('clear-batch').classList.toggle('hidden',state==='idle');
  $('upload-label').textContent=state==='idle'?'选择文件':'更换文件';
}
function resetFile(detail='上传后将在此显示校验结果、有效条数及数据预览。'){
  S.batch=null;S.filename='';$('input-file').value='';$('input-preview').classList.add('hidden');$('preview-rows').innerHTML='';
  fileStatus('idle','尚未上传文件',detail);clearResults();syncControls();
}
async function acceptInputFile(file){
  if(!file||S.busy)return;
  await busy(async()=>{
    S.batch=null;S.filename='';clearResults('正在校验新上传的文件，之前的计算结果已清除。');
    $('input-preview').classList.add('hidden');$('preview-rows').innerHTML='';
    fileStatus('reading','正在上传并校验…','正在检查文件格式、字段和数值，请稍候。',file);
    try{
      if(!/\.(xlsx|csv)$/i.test(file.name))throw Error('仅支持 .xlsx 或 .csv 文件。');
      const payload=await upload(file);
      const response=await api('/api/read-input',{...payload,magnitude_unit:$('unit').value});
      if(!response.records?.length)throw Error('文件中没有可用的工况数据。');
      S.batch=response.records;S.filename=file.name;S.job='ready';
      fileStatus('success','上传成功 · 数据校验通过',`已识别 ${S.batch.length} 条有效工况 · 两列数值校验通过 · 尚未开始计算`,file);
      $('preview-caption').textContent=`数据预览 · 前 ${Math.min(5,S.batch.length)} 条 / 共 ${S.batch.length} 条`;
      $('preview-mag-header').textContent=$('unit').value==='db'?'初始 S11 / dB':'反射系数幅值 |Γ|';
      $('preview-rows').innerHTML=S.batch.slice(0,5).map((r,i)=>`<tr><td>${i+1}</td><td>${num(r.mag,6)}</td><td>${num(r.deg,4)}</td></tr>`).join('');
      $('input-preview').classList.remove('hidden');
      $('result-empty-detail').textContent=`${file.name} 已就绪。点击“开始计算”处理 ${S.batch.length} 条工况。`;
      notice(`上传成功：${file.name}，共 ${S.batch.length} 条工况，校验已通过。`,'success');
    }catch(error){
      S.batch=null;S.filename='';fileStatus('error','上传失败 · 文件未载入',`${error.message} 请检查后重新上传。`,file);
      $('result-empty-detail').textContent='文件未通过校验，请重新上传；本次不会使用先前的输入数据。';
      throw error;
    }finally{$('input-file').value='';}
  });
}
function setInputMode(mode){
  if(S.busy||S.inputMode===mode)return;
  S.inputMode=mode;resetFile();
  for(const name of ['file','manual']){$('mode-'+name).classList.toggle('active',name===mode);$('mode-'+name).setAttribute('aria-pressed',String(name===mode));}
  $('batch-input').classList.toggle('hidden',mode!=='file');$('manual-input').classList.toggle('hidden',mode!=='manual');
  notice(mode==='file'?'请选择待处理的 Excel 或 CSV 文件。':'单条输入模式：填写幅值和相位后开始计算。');syncControls();
}
function useBundle(bundle) {
  S.bundle=bundle;S.analysis=bundle.waveform_analysis || null;
  updatePlotModels(bundle.models || [...new Set(bundle.predictions.map(r=>r.model_id))].map(id=>({id,label:bundle.predictions.find(r=>r.model_id===id)?.model_label||LABELS[id]||id})));
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
  $('result-meta').textContent=`${S.filename||'单条输入'} · ${bundle.records.length} 条工况 · ${bundle.predictions.length} 个模型方案`;
  S.job='done';syncControls();
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
    title='所选方案均未满足稳定性判据';caption=prefix+'当前工况的波形比较';detail='请检查波形、调节方案与设备要求，不将最低分方案标记为稳定方案。';
  } else {
    title=prefix+LABELS[provisional.model_id];caption='预计匹配推荐 · 等待功率波形验证';detail=`L1 ${num(provisional.L1)} ${provisional.length_unit} · L2 ${num(provisional.L2)} ${provisional.length_unit} · 共用响应代理评分`;
    coefficient=provisional;
  }
  const coefficientMarkup=coefficient?`<div class="coefficient-output"><span>预计匹配后反射系数（线性幅值）</span><b>|Γ| = ${num(coefficient.estimated_matched_mag,5)}</b><span>相位 ∠Γ = ${num(coefficient.estimated_matched_deg)}° · 无量纲，非百分比</span></div>`:'';
  $('recommendation').innerHTML=`<span class="caption">${escapeHTML(caption)}</span><strong>${escapeHTML(title)}</strong>${coefficientMarkup}<p>${escapeHTML(detail)}</p>`;
  document.querySelectorAll('.length-unit').forEach(el=>el.textContent=records[0].length_unit==='deg'?'°':records[0].length_unit);
  $('result-rows').innerHTML=records.map(r=>`<tr class="${r.model_id===(winner || provisional.model_id)?'best':''}"><td>${escapeHTML(LABELS[r.model_id])}${r.outside_training_input_box?'<small>超出训练输入范围</small>':''}</td><td>${num(r.L1)}</td><td>${num(r.L2)}</td><td>${num(r.estimated_matched_mag,5)}</td><td>${num(r.estimated_matched_deg)}</td></tr>`).join('');
  $('result-note').textContent=S.bundle.mode==='demo'?'示例反射系数与动作均为预设值，不能用来判断所选网络的真实优劣。':'返回匹配后反射系数幅值|Γ|及相位，幅值是0～1的线性值，不是反射功率百分比。数值为统一响应代理预测，并非新调节后的实测；低幅值时相位对噪声敏感。';
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
function updatePlotModels(models){
  IDS=models.map(m=>m.id);S.visible=new Set(IDS);
  const palette=['#4a84dd','#d99a39','#129679','#9869cb','#cf617a','#327d9b'];
  models.forEach((m,i)=>{LABELS[m.id]=m.label||m.id;COLORS[m.id]=palette[i%palette.length];});
  $('legend').innerHTML=models.map(m=>`<button data-model="${escapeHTML(m.id)}"><i class="dot" style="background:${COLORS[m.id]}"></i>${escapeHTML(m.label||m.id)}</button>`).join('')+'<span><i class="target-dash"></i>目标</span>';
  document.querySelectorAll('#legend button').forEach(button=>button.onclick=()=>{const id=button.dataset.model;if(S.visible.has(id)){if(S.visible.size===1)return;S.visible.delete(id);}else S.visible.add(id);button.classList.toggle('off',!S.visible.has(id));renderChart();});
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
      html+=`<br><span style="color:${COLORS[id]}">●</span> ${escapeHTML(LABELS[id])} ${num(field(row))} W`;
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
  return localWorkbook([sheet('模型预测结果',headers,b.predictions),sheet('预计推荐_待波形验证',headers,b.estimated_recommendations),sheet('稳定性评估',metric,a.metrics||[]),sheet('功率波形',wave,a.waveforms||[]),sheet('波形选优结果',headers,confirmed),{name:'说明',headers:['项目','内容'],rows:[['模式',b.mode],['run_id',b.run_id],['输入幅值单位',b.magnitude_unit],['主要返回量','estimated_matched_mag=线性反射系数幅值|Γ|，无量纲非百分比；estimated_matched_deg=相位°；real/imag=实部/虚部；reflected_power_ratio=辅助功率比|Γ|²'],['波形来源',a.source||'未接入'],['优胜模型',a.winner||'无已验证稳定方案'],['阈值',JSON.stringify(a.settings||{})],['波形参考',JSON.stringify(b.waveform_reference||{})],['说明',b.notice]]}]);
}
async function busy(action){if(S.busy)return;S.busy=true;syncControls();if(typeof syncTrainingControls==='function')syncTrainingControls();try{await action();}catch(error){notice(error.message,'error');}finally{S.busy=false;syncControls();if(typeof syncTrainingControls==='function')syncTrainingControls();}}
$('settings-toggle').onclick=()=>$('settings').classList.toggle('hidden');$('api-url').value=S.api;
$('connect').onclick=()=>busy(async()=>{const address=$('api-url').value.trim().replace(/\/$/,'');if(address&&!/^https?:\/\//.test(address))throw Error('服务地址须以http://或https://开头');S.api=address;localStorage.setItem('callmynet-api',S.api);resetFile();S.selected.clear();T.selectedInitialized=false;if(await health()){notice('已连接本地服务。');pollTraining();}else throw Error('连接失败，请检查服务地址。');});
$('run').onclick=()=>busy(async()=>{
  if(S.inputMode==='file'&&!S.batch)throw Error('请先上传并校验工况数据文件。');
  if(!S.connected||!selectedModels().length||selectedModels().some(m=>!m.checkpoint_ready)||S.trainingActive)throw Error('请先连接本地服务，勾选已训练的模型，等待训练任务结束。');
  const records=S.inputMode==='file'?S.batch:[{mag:Number($('mag').value),deg:Number($('deg').value)}];
  if(S.inputMode==='manual'&&(!$('mag').value.trim()||!$('deg').value.trim()))throw Error('请填写幅值与相位。');
  clearResults();S.job='running';syncControls();
  $('result-mode').textContent='计算中';$('result-empty-title').textContent='正在比较所选模型';
  $('result-empty-detail').textContent=`正在处理 ${records.length} 条工况，结果将在全部计算完成后显示。`;
  notice(`正在处理 ${records.length} 条工况，运行所选网络并比较候选方案…`);
  try{useBundle(await api('/api/predict',{records,magnitude_unit:$('unit').value,model_ids:selectedModels().map(m=>m.id),evaluation_model:$('evaluation-model').value}));notice(`处理完成：${records.length} 条工况的模型结果已生成，可下载 Excel。`,'success');}
  catch(error){S.job='error';$('result-mode').textContent='计算失败';$('result-mode').className='badge error';$('result-empty-title').textContent='本次计算未完成';$('result-empty-detail').textContent=error.message;throw error;}
});
$('input-file').onchange=event=>acceptInputFile(event.target.files[0]);
$('clear-batch').onclick=()=>{if(S.busy)return;resetFile();notice('文件已移除，请重新上传待处理数据。');};
$('mode-file').onclick=()=>setInputMode('file');$('mode-manual').onclick=()=>setInputMode('manual');
$('unit').onchange=()=>{
  const hadFile=!!S.batch;
  resetFile(hadFile?'幅值单位已更改，请重新上传文件以校验新单位。':'上传后将在此显示校验结果、有效条数及数据预览。');
  $('unit-help').textContent=$('unit').value==='db'?'幅值为不大于 0 的 S11 dB 数值，相位单位为度。':'幅值为 0～1 的线性反射系数，相位单位为度。';
  notice(hadFile?'幅值单位已更改，请重新上传并校验文件。':'输入幅值单位已更新。');
};
for(const id of ['mag','deg'])$(id).oninput=()=>{clearResults('单条工况已修改，请重新开始计算。');syncControls();};
$('upload-zone').ondragover=event=>{event.preventDefault();if(!S.busy)$('upload-zone').classList.add('dragover');};
$('upload-zone').ondragleave=()=>$('upload-zone').classList.remove('dragover');
$('upload-zone').ondrop=event=>{event.preventDefault();$('upload-zone').classList.remove('dragover');if(S.busy)return;if(event.dataTransfer.files.length!==1){notice('每次请上传一个待处理文件。','error');return;}return acceptInputFile(event.dataTransfer.files[0]);};
$('sample-select').onchange=renderResults;
$('wave-file').onchange=event=>busy(async()=>{const file=event.target.files[0];if(!file)return;
  $('wave-upload-status').textContent=`正在上传并校验：${file.name}`;$('wave-upload-status').className='wave-upload-status reading';
  try{const payload=await upload(file);const expected=S.bundle?.run_id;
  const analysis=await api('/api/analyze-waveforms',{...payload,source:$('wave-source-select').value,options:options(),expected_run_id:expected,model_ids:S.bundle?.models?.map(m=>m.id),model_labels:Object.fromEntries((S.bundle?.models||[]).map(m=>[m.id,m.label]))});
  if(S.bundle&&!S.bundle.records.some(r=>String(r.sample_id)===String(analysis.sample_id)))throw Error('波形sample_id不在当前推理批次内');
  if(S.bundle&&S.bundle.mode==='inference'&&analysis.source==='demo')throw Error('合成示例波形不能验证真实推理结果');
  S.analysis=analysis;if(S.bundle)S.bundle.waveform_analysis=analysis;else{S.bundle={mode:'waveforms_only',run_id:analysis.run_id,records:[],predictions:[],estimated_recommendations:[],notice:'独立导入的功率波形评估'};$('export').disabled=false;}
  updatePlotModels(analysis.metrics.map(m=>({id:m.model_id,label:m.model_label})));renderAnalysis();if(S.bundle.predictions.length)renderResults();notice(analysis.winner?'功率波形比较完成，已选出满足当前稳定性判据的方案。':'功率波形比较完成：所选方案均未满足当前稳定性判据。',analysis.source==='demo'?'demo':'');
  $('wave-upload-status').textContent=`导入成功：${file.name} · ${analysis.waveforms.length} 个采样点 · 所选模型波形已完成稳定性比较`;$('wave-upload-status').className='wave-upload-status success';
  }catch(error){$('wave-upload-status').textContent=`导入失败：${file.name} · ${error.message}${S.analysis?'。图中仍为上一次成功导入的波形。':''}`;$('wave-upload-status').className='wave-upload-status error';throw error;}
  finally{event.target.value='';}
});
$('reanalyze').onclick=()=>busy(async()=>{if(!S.analysis)return;const response=await api('/api/analyze-waveforms',{rows:S.analysis.waveforms,source:S.analysis.source,expected_run_id:S.bundle?.run_id,options:options(),model_ids:S.analysis.metrics.map(m=>m.model_id),model_labels:Object.fromEntries(S.analysis.metrics.map(m=>[m.model_id,m.model_label]))});S.analysis=response;S.bundle.waveform_analysis=response;renderAnalysis();if(S.bundle.predictions.length)renderResults();notice('已按更新后的判据重新比较。',response.source==='demo'?'demo':'');});
$('export').onclick=()=>busy(async()=>{if(!S.bundle)return;download(exportLocally(),(S.bundle.mode==='demo'?'DEMO_':'')+'matching_results.xlsx');});
$('input-template').onclick=()=>download(localWorkbook([{name:'input',headers:['mag0','deg0'],rows:[]}]),'prediction_template.xlsx');
$('wave-template').onclick=()=>download(localWorkbook([{name:'waveforms',headers:['run_id','sample_id','scenario_id','model_id','time_s','power_w','target_w','source','incident_power_w','reflected_power_w'],rows:[]}]),'waveform_template.xlsx');

$('zoom').oninput=$('pan').oninput=renderChart;
$('power-channel').onchange=renderChart;
$('reset-zoom').onclick=()=>{$('zoom').value=1;$('pan').value=0;renderChart();};
syncControls();
