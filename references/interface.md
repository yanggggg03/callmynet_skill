# 模型、功率波形和 HTTP 接口

## 本地运行

安装 Python 3.10+、numpy 和 PyTorch。已有训练环境可直接复用，不需要 DeepSeek/OpenAI API key。面板的 HTML、CSS、JavaScript 不依赖 npm、CDN 或联网字体。

```powershell
./start_panel.ps1 -Python 'D:\Anaconda\CondaData\env_dirs\WDCNN\python.exe'
```

也可从任意工作目录运行 `python -B /完整路径/scripts/server.py --config /完整路径/models.local.json`。

默认只监听 `127.0.0.1:8765`，用于本机访问。所有本地文件路径只从服务端配置读取，浏览器请求不能指定加载任意权重或运行任意脚本。

## 权重配置

`config/models.json` 的三个条目对应三个源项目。默认正式权重路径为：

| id | 源项目 | 权重 |
| --- | --- | --- |
| residual_mlp | competition_01_physics_residual | runs/measured/best.pt |
| fourier_resmlp | competition_02_fourier_transformer | runs/measured/best.pt |
| conditional_flow | competition_03_conditional_flow | runs/measured/best.pt |

可以把本人的训练权重放到此技能 `models/<id>/best.pt`，然后把配置改成对应相对路径。仓库默认忽略全部 `.pt/.pth`，因此公开代码时不会自动上传训练权重或训练数据。

`evaluation_model` 指定唯一共用响应代理，默认 `residual_mlp`。这只是可用的统一评分入口，应通过独立验证集评估响应代理的准确性后再选定。推理中的 `estimated_recommendations` 为共用代理候选评分结果，最终动态验证来自外部波形。

`python_executable` 留空则复用启动服务的 Python；`flow_candidates` 默认32，固定随机种子保证相同输入、配置、权重时可重复。三个生成进程和一个评分进程隔离运行，避免原项目同名 Python 模块互相污染。

训练一致性校验首先比较检查点记录的NPZ文件哈希。若三者不同（例如分别准备时元数据保存了不同项目路径），服务会定位各检查点训练时的准备文件，先核对它与检查点原始文件哈希一致，再对训练数组、标签、分组、行号、训练/验证/测试索引、单位和配置计算规范化内容指纹。描述性路径不参与内容指纹；真实数据或划分差异仍然拒绝比较。

迁移到另一台电脑时，如原准备数据路径失效，可在每个模型配置项增加 `prepared_data`，指向它训练时使用的那份 `measured_matching.npz`。必须保留原文件；换一份重新生成的NPZ不能用来代替检查点训练证据。三个检查点若引用同一个原始NPZ哈希，则无需准备文件即可确认一致。

## 网络输入

预测 Excel/CSV：

```csv
mag,deg
0.4,60
0.6,-45
```

线性幅值 0～1，dB幅值≤0；相位为度。网络内部使用复反射系数实部与虚部。返回 `sample_id` 为从1开始的输入记录顺序。

## 功率波形

单次上传包含同一调节批次、样本与工况的三个模型波形，可用 `.xlsx` 或 `.csv`：

```csv
run_id,sample_id,scenario_id,model_id,time_s,power_w,target_w,source
本次推理返回的编号,1,experiment-001,residual_mlp,0,900,1000,measured
```

| 字段 | 含义 |
| --- | --- |
| run_id | 当前推理响应返回的编号；独立波形比较可以自定，但不能用于验证不匹配的模型推理 |
| sample_id | 本次输入样本序号，例如1 |
| scenario_id | 实验/仿真工况编号，三个方案一致 |
| model_id | residual_mlp / fourier_resmlp / conditional_flow |
| time_s | 单位秒，非负；每个模型的时间戳唯一 |
| power_w | 同一位置测量、同一意义的功率，单位W，非负 |
| target_w | 目标功率，单位W，必须>0；本版本单次比较固定目标 |
| source | measured / simulation / demo；同一批数据须同一来源 |

每个模型必须使用完全相同的时间网格、目标功率与工况。程序不擅自插值、裁剪出不同区间或将行号当时间轴。阶跃目标可拆分为固定目标段；高频噪声或稀疏采样需要在数据采集端合理处理。程序检查字段一致性，真实的初始条件一致性仍由实验或仿真流程保证。

可选列 `incident_power_w,reflected_power_w` 必须同时覆盖全部采样，且满足 `0≤reflected≤incident` 和 `power_w=incident-reflected`。本版本将 `power_w` 解释为同一测量参考面上的净输入功率；没有损耗模型时，不把它解释为等离子体实际吸收功率。面板支持切换显示三种通道，评分仍针对净输入功率。

`options.evaluation_start_s` 和 `options.evaluation_end_s` 可显式设置共同有功率窗口。全时段曲线仍保留；稳态起点从此窗口按比例计算，超调和调节时间也在此窗口计算。示例使用1.00～5.95秒，不把6秒后的关断段计入调节稳定性。修改窗口必须由实验过程解释，不能为了获得达标结果任意挑选区间。

示例参考论文《ICRF阻抗匹配反馈控制系统的设计与实现》图3.41（正文71页，PDF91页）和图3.42（正文72页，PDF92页）：约20kW入射平台、启动高反射、亚秒级衰减及低反射稳态。图4.8（正文83页）用于了解约0.4秒调节量级。示例不是论文曲线数字化结果，三模型动作/速度/稳态值均为预设，不代表优劣验证。论文中部分VSWR与功率数字未满足简单同参考面换算，本项目不据此生成VSWR标签。

默认最后40%时间为稳态窗口，使用总体标准差计算CV；峰峰值、目标误差和漂移均按目标功率归一化。漂移用稳态样本线性回归斜率乘稳态窗口时长计算。超调使用整个观察窗口。各指标除以各自阈值后的均值为评分，越低越好。稳定选优优先于评分；全部不达标时 `winner=null`。

## HTTP API

接口不依赖任何大语言模型厂商，可供 DeepSeek Agent、Codex 或其他客户端调用。

| 请求 | 功能 |
| --- | --- |
| GET /api/health | 模型权重是否存在、服务版本、统一评估器 |
| GET /api/demo | 明确标记demo的合成演示 |
| POST /api/predict | 三网络生成候选并统一评分 |
| POST /api/read-input | 读取预测Excel/CSV |
| POST /api/analyze-waveforms | 波形检查、指标计算与选优 |
| POST /api/export | 将结果包导出为XLSX二进制 |
| GET /api/templates/input.xlsx | 空输入模板 |
| GET /api/templates/waveforms.xlsx | 空功率模板 |

预测请求示例：

```json
{"magnitude_unit":"linear","records":[{"mag":0.4,"deg":60}]}
```

返回含 `run_id,records,predictions,estimated_recommendations,metadata`。`predictions` 保留全部三模型，每条含 `sample_id,model_id,L1,L2,length_unit,estimated_matched_mag,estimated_reflected_power_ratio,evaluator` 等。

波形分析请求：

```json
{
  "source":"measured",
  "expected_run_id":"当前推理编号",
  "options":{"max_cv_pct":2,"max_error_pct":2},
  "rows":[{"run_id":"当前推理编号","sample_id":"1","scenario_id":"case-1","model_id":"residual_mlp","time_s":0,"power_w":1000,"target_w":1000,"source":"measured"}]
}
```

上面仅展示一条字段格式，实际需提交全部三个模型且包含足够样本。也可用 `filename,content_base64` 替代 `rows` 上传表格。返回 `metrics,waveforms,winner,selection_status,settings`。将它作为原推理包的 `waveform_analysis` 字段，再POST给 `/api/export`。

## GitHub与本地服务

GitHub仓库可下载整个Skill；支持Skill的Agent可安装后运行其脚本。网页公开访问由GitHub Pages提供，默认地址在启用部署后为 `https://yanggggg03.github.io/callmynet_skill/`。

该页面无需后端即可载入合成示例、交互查看曲线、下载完整示例Excel和空模板。真实预测与波形文件分析调用本地服务；每位访客需要自行下载仓库、准备权重并启动服务。

从GitHub Pages连接本地服务时，在本地配置中设置：

```json
"allowed_origins": ["https://yanggggg03.github.io"]
```

然后在面板“连接设置”填写 `http://127.0.0.1:8765`。浏览器可能受本地网络访问授权或HTTPS混合内容策略影响，因此本地完整面板 `http://127.0.0.1:8765/` 是首选。`allowed_origins`仅允许指定来源访问本地API。

如果后续希望任何访客不用下载、直接通过你电脑里的权重运行，需要另行设置你电脑服务的公网HTTPS入口、访问控制和限流，并保持电脑在线。当前交付不包含此公网入口，GitHub也不会代为运行Python。
