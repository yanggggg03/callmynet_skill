'use strict';
const T={types:[],drafts:[],file:null,job:null,polling:false,selectedInitialized:false};
function refreshModelSelection(info){
  T.types=info.model_types||[];
  if(!T.drafts.length&&T.types.length)T.drafts=[{id:'model_1',type:T.types[0].id,label:T.types[0].label}];
  if(!T.selectedInitialized){S.selected=new Set(info.models.filter(m=>m.checkpoint_ready).map(m=>m.id));T.selectedInitialized=true;}
  else S.selected=new Set([...S.selected].filter(id=>info.models.some(m=>m.id===id)));
  $('inference-models').innerHTML=info.models.map(m=>`<label class="inference-model ${m.checkpoint_ready?'ready':''}"><input type="checkbox" value="${escapeHTML(m.id)}" ${S.selected.has(m.id)?'checked':''}><span><strong>${escapeHTML(m.label)}</strong><small>${escapeHTML(m.type||m.id)} · ${m.checkpoint_ready?'权重就绪':'待训练'}</small></span></label>`).join('');
  document.querySelectorAll('#inference-models input').forEach(input=>input.onchange=()=>{
    if(S.busy){input.checked=S.selected.has(input.value);return;}
    if(input.checked)S.selected.add(input.value);else S.selected.delete(input.value);
    clearResults('参与预测的模型已变更，请重新计算。');refreshEvaluator();syncControls();
  });
  refreshEvaluator(info.evaluator);renderTrainingDrafts();syncTrainingControls();syncControls();
}
function refreshEvaluator(preferred){
  const current=preferred||$('evaluation-model').value;
  $('evaluation-model').innerHTML=selectedModels().map(m=>`<option value="${escapeHTML(m.id)}">${escapeHTML(m.label)}</option>`).join('');
  if(S.selected.has(current))$('evaluation-model').value=current;
  $('selected-count').textContent=`已选 ${selectedModels().length} 个 / 共 ${S.models.length} 个`;
}
function renderTrainingDrafts(){
  $('training-models').innerHTML=T.drafts.map((m,i)=>`<div class="training-model-row"><span class="draft-index">${i+1}</span><label>网络类型<select data-field="type" data-index="${i}">${T.types.map(kind=>`<option value="${escapeHTML(kind.id)}" ${m.type===kind.id?'selected':''}>${escapeHTML(kind.label)}</option>`).join('')}</select></label><label>模型名称<input data-field="label" data-index="${i}" value="${escapeHTML(m.label)}" maxlength="80"></label><button class="link" data-remove="${i}" aria-label="删除模型 ${i+1}">移除</button></div>`).join('');
  document.querySelectorAll('#training-models [data-field]').forEach(input=>input.onchange=()=>{T.drafts[Number(input.dataset.index)][input.dataset.field]=input.value;});
  document.querySelectorAll('#training-models [data-remove]').forEach(button=>button.onclick=()=>{if(S.trainingActive)return;T.drafts.splice(Number(button.dataset.remove),1);renderTrainingDrafts();syncTrainingControls();});
}
function syncTrainingControls(){
  const active=S.trainingActive,editingBlocked=active||S.busy;
  $('training-start').disabled=!S.connected||active||S.busy||!T.file||!T.drafts.length;
  $('training-cancel').disabled=!active||T.job?.status==='cancelling';
  $('training-activate').disabled=active||S.busy||T.job?.status!=='completed'||T.job?.activated;
  $('training-activate').textContent=T.job?.activated?'本次权重已启用':'启用本次权重';
  for(const id of ['training-file','training-input-unit','training-output-unit','add-training-model','train-epochs','train-surrogate-epochs','train-patience','train-batch-size','train-seed','train-device'])$(id).disabled=editingBlocked;
  document.querySelectorAll('#training-models input,#training-models select,#training-models button').forEach(el=>el.disabled=editingBlocked);
}
function renderTrainingJob(job){
  T.job=job;S.trainingActive=['preparing','training','cancelling'].includes(job.status);
  const labels={idle:'尚未开始',preparing:'准备数据',training:'正在训练',cancelling:'正在取消',cancelled:'已取消',failed:'训练失败',interrupted:'训练中断',completed:'训练完成'};
  $('training-badge').textContent=labels[job.status]||job.status;
  $('training-badge').className='badge '+(job.status==='completed'?'success':['failed','interrupted'].includes(job.status)?'error':'');
  $('training-job').classList.toggle('hidden',job.status==='idle');
  if(job.status!=='idle'){
    $('training-progress').value=job.progress||0;
    $('training-progress-title').textContent=`${labels[job.status]||job.status} · ${Math.round((job.progress||0)*100)}%`;
    const splits=job.data_summary?.split_rows;
    $('training-summary').textContent=job.error||`${job.filename} · ${job.rows} 条 · ${job.models.length} 个模型${splits?` · 训练 ${splits.train} / 验证 ${splits.val} / 测试 ${splits.test}`:''}${job.status==='completed'?' · 全部权重已保存在本机，可启用后预测。':''}`;
    const states={queued:'排队中',training:'训练中',completed:'已完成',failed:'失败',cancelled:'已取消'};
    $('training-job-models').innerHTML=job.models.map(m=>`<tr><td>${escapeHTML(m.label)}<small>${escapeHTML(m.type)}</small></td><td>${states[m.status]||m.status}${m.status==='training'&&m.epoch?` · ${m.phase==='forward'?'响应代理':'网络'} ${m.epoch}/${m.max_epochs}`:''}</td><td>${m.best_epoch??'—'}</td><td class="checkpoint-path">${escapeHTML(m.checkpoint||'完成后生成 best.pt')}</td></tr>`).join('');
    $('training-log').textContent=(job.logs||[]).join('\n');
  }
  syncTrainingControls();syncControls();
}
async function pollTraining(){
  if(T.polling)return;
  T.polling=true;
  try{renderTrainingJob(await api('/api/training/status'));}
  catch(error){$('training-badge').textContent='训练状态连接失败';$('training-summary').textContent=error.message;}
  finally{T.polling=false;}
  if(S.trainingActive)setTimeout(pollTraining,1800);
}
$('add-training-model').onclick=()=>{
  if(S.trainingActive||!T.types.length)return;
  if(T.drafts.length>=16){notice('单次最多配置16个模型。','error');return;}
  const id='model_'+Date.now().toString(36)+'_'+T.drafts.length;
  T.drafts.push({id,type:T.types[0].id,label:T.types[0].label+' '+(T.drafts.length+1)});renderTrainingDrafts();syncTrainingControls();
};
$('training-file').onchange=event=>{
  const file=event.target.files[0];if(!file)return;
  if(!/\.(xlsx|csv)$/i.test(file.name)||file.size>15_000_000){T.file=null;$('training-file-status').textContent='文件不符合要求：请选择15MB以内的 .xlsx 或 .csv。';}
  else{T.file=file;$('training-file-status').textContent=`已选择：${file.name} · ${(file.size/1024).toFixed(1)} KB。点击开始训练后校验六列数据并准备固定划分。`;}
  event.target.value='';syncTrainingControls();
};
$('training-start').onclick=()=>busy(async()=>{
  if(!T.file||!T.drafts.length)throw Error('请选择训练数据并添加模型。');
  $('training-file-status').textContent=`正在上传和校验：${T.file.name}`;
  const payload=await upload(T.file);
  const parameters={epochs:Number($('train-epochs').value),surrogate_epochs:Number($('train-surrogate-epochs').value),patience:Number($('train-patience').value),batch_size:Number($('train-batch-size').value),seed:Number($('train-seed').value),device:$('train-device').value};
  try{
    const job=await api('/api/training/start',{...payload,models:T.drafts,magnitude_unit:$('training-input-unit').value,matched_magnitude_unit:$('training-output-unit').value,parameters});
    $('training-file-status').textContent=`训练数据校验通过：${job.filename} · ${job.rows} 条。`;
    renderTrainingJob(job);notice('训练任务已启动，权重和日志将保存在本机。');setTimeout(pollTraining,500);
  }catch(error){$('training-file-status').textContent='训练未启动：'+error.message;throw error;}
});
$('training-cancel').onclick=()=>busy(async()=>{renderTrainingJob(await api('/api/training/cancel',{}));notice('正在取消训练，已经存在的权重不受影响。');});
$('training-activate').onclick=()=>busy(async()=>{
  const job=await api('/api/training/activate',{job_id:T.job.id});renderTrainingJob(job);
  S.selected=new Set(job.models.map(m=>m.id));clearResults('本次训练权重已启用，请上传预测工况或重新计算。');
  await health();notice(`已启用 ${job.models.length} 个模型的本地权重，可上传 mag0 / deg0 Excel 进行预测。`,'success');
});
$('evaluation-model').onchange=()=>{clearResults('共用响应评估器已变更，请重新计算。');syncControls();};
$('training-template').onclick=()=>download(localWorkbook([{name:'training',headers:['mag0','deg0','L1','L2','matched_mag','matched_deg','group_id'],rows:[]}]),'training_template.xlsx');
health().then(connected=>{
  syncTrainingControls();
  if(connected){notice('本地服务已连接。可先训练模型，或选择已有权重进行预测。');pollTraining();}
  else notice('未连接本地服务。请启动项目中的 start_panel.ps1，或在连接设置中填写地址。','error');
});
