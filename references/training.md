# 本地训练、任务接口和模型扩展

## 任务接口

POST /api/training/start 接收filename、content_base64和以下字段：

```json
{"models":[{"id":"mlp_a","type":"residual_mlp","label":"MLP A"},
           {"id":"flow_b","type":"conditional_flow","label":"Flow B"}],
 "magnitude_unit":"db","matched_magnitude_unit":"linear",
 "parameters":{"epochs":150,"surrogate_epochs":100,"patience":25,
               "batch_size":128,"seed":42,"device":"cpu"}}
```

返回202，异步运行。models支持1～16项，type可重复但id唯一；id以小写字母开始，仅含小写字母、数字、下划线。只运行服务端注册类型。高级参数还包括width、blocks、layers、fourier_features、fourier_scale、surrogate_width、lr、response_weight、threads、candidates。

- GET /api/training/status：当前/最近任务、阶段、进度、末尾日志、划分和权重路径。
- POST /api/training/cancel，正文{}：终止当前训练子进程，不启用未完成权重。
- POST /api/training/activate，正文含job_id：仅完整成功任务可启用，写入基础配置旁的*.active.local.json，不覆盖基础配置或旧权重。

训练和推理互斥。任务状态、完整日志和准备数据保存在任务目录。刷新可继续查看；服务重启时未完成任务标记interrupted，不自动续训。

训练至少六列，初始/匹配幅值分别声明单位，统一线性化。只准备一次数据，所有网络使用同一个prepared.npz。固定分组按约70%训练、15%验证、15%测试划分，实际行数取决于实验组。采集独立性由真实实验和group_id保证。

## 可选预测模型

POST /api/predict 支持records、magnitude_unit、model_ids、evaluation_model：

```json
{"records":[{"mag":0.5,"deg":-30}],"magnitude_unit":"linear",
 "model_ids":["mlp_a","flow_b"],"evaluation_model":"mlp_a"}
```

至少一个模型，评估器须选自其中。权重和准备文件只从服务端配置读取。bundle.models记录实际参与模型；预测、推荐、Excel不固定数量。统一响应代理为所有候选评分。

POST /api/analyze-waveforms 使用model_ids明确预期列表，要求全部齐全。可传model_labels。面板自动从当前bundle传入并检查run_id、sample_id；独立波形可从model_id列推断列表。每次比较一个样本、一个工况。没有动态模型或外部波形就不能返回真实时间响应。

## 注册本地模型类型

内置residual_mlp、fourier_resmlp、conditional_flow。要接入别的网络，在基础配置增加：

```json
"model_types":[{"id":"my_network","label":"我的网络","runtime_dir":"models/my_network"}]
```

运行目录由本地开发者维护，服务器不接受浏览器指定路径或上传Python源码。相对路径按Skill根目录解析，目录至少提供data.py、engine.py、model.py，协议如下：

1. 兼容measured-double-stub-v1数据。所有网络共用内置准备器产生的x/y/after、groups、row_ids和三份索引，不能重新划分。
2. engine.train(args)接收training_service.parameters中定义的命名空间参数。在args.output写入best.pt、history.json、training_summary.json，后者包含best_epoch。
3. engine.Predictor(checkpoint,device)提供ck、mean/std、low/span、context_min/max、forward(context,normalized_lengths)、support_distance、candidates(x,count,seed)。candidates返回与内置实现一致的六元组，第三项为每样本的候选电长度。
4. data.to_complex(magnitude,phase,unit)、engine.initialize(seed,threads,device)、engine.tensor(array,device)签名与内置一致。检查点保存metadata、bounds、data_sha256、best_epoch、training_args，供单位与训练一致性检查。
5. 可更换逆向网络，响应代理仍须输出同测量参考面的复反射系数。物理量不同须先开发适配器，任意pt不能直接当兼容权重。

注册后刷新面板，新类型会出现在训练下拉框。额外依赖需安装到服务Python中。训练Worker在独立子进程加载对应目录，避免同名模块污染。
