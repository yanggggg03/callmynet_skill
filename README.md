# CallMyNet Skill

在本机完成阻抗匹配模型训练、权重管理、Excel批量预测和外部功率波形比较。用户选择模型种类及数量，每个模型独立生成best.pt。支持工具调用的Codex、DeepSeek等Agent可读取SKILL.md，或直接使用面板与HTTP API。

## 启动

需要Python 3.10+、PyTorch、numpy。训练和推理均在用户自己的电脑运行，不依赖大语言模型API。

```bash
python -m pip install -r requirements.txt
python -B scripts/server.py
```

Windows也可运行 `./start_panel.ps1`；已有训练环境时指定 `-Python '你的Python完整路径'`。打开 **http://127.0.0.1:8765/**。

## 在窗口中训练

1. 在“本地模型训练”中添加模型，选择网络类型和名称。内置残差MLP、Fourier ResMLP和条件Flow；数量从1个开始，单次最多16个模型项。同一种网络可以重复添加，各自训练、保存。
2. 上传自己的训练Excel/CSV，至少包含以下六列，可下载空白训练模板。

| 字段 | 含义 |
| --- | --- |
| mag0 | 初始反射系数幅值 |
| deg0 | 初始相位，° |
| L1、L2 | 已知支节电长度，° |
| matched_mag | 调节后的反射系数幅值 |
| matched_deg | 调节后的相位，° |

支持Mag0/Deg0、mag/deg、相位别名meg0和match_mag/match_deg。可增加group_id标记真实实验组，同组不跨训练/验证/测试划分。至少20条数据、10个独立输入或实验组，两个电长度都须有变化。合成增强数据不能当作独立实测测试数据。

3. 分别选择初始幅值和匹配后幅值单位，分别支持linear或dB。系统统一转换为线性幅值，避免负数S11与线性匹配幅值混用。
4. 设置最大轮数、响应代理轮数、提前停止、批大小、种子和设备，点击“开始训练所选模型”。只准备一次数据，所有模型使用相同NPZ和固定分组划分。模型依次训练，显示阶段、轮次、日志和权重路径；训练过程中不并发预测。
5. 全部完成后点击“启用本次权重”。失败或取消不会替换已启用权重。刷新页面可以恢复当前任务；重启服务后未完成任务标记为中断。

每次训练保存在独立目录：

```text
runs/training/<任务编号>/
  prepared.npz
  training_linear.csv
  training.log
  job.json
  <模型标识>/best.pt
  <模型标识>/history.json
  <模型标识>/training_summary.json
```

不把多个模型合并成一个best.pt。每个检查点保存该模型结构配置、逆向网络、响应代理、标准化、单位及训练指纹。

## 预测与比较

1. 在“本次预测与比较”勾选一个或多个已训练模型，选择共用响应评估器。所有候选由同一个代理评分；多个模型的训练数据和划分须一致。
2. 上传仅包含mag0、deg0两列的Excel/CSV，mag0、meg0也支持。选择输入幅值单位。成功后显示文件名、有效条数与前5条预览；最多1000条。也可切换单条输入。
3. 点击开始计算，查看每个模型的电长度和**匹配后反射系数**：线性幅值|Γ|、相位。Excel/API保留实部、虚部；辅助反射功率比为|Γ|²。预测系数来自共用响应代理，不是调节后的实测值。
4. 下载结果Excel，包含全部样本和模型结果。更换输入、单位、模型或评估器会清除旧结果。

若使用已有权重，在config/models.json配置每项的唯一id、网络type、label和checkpoint。面板启用训练权重时写入同目录models.active.local.json，优先于基础配置。基础配置使用可迁移的相对路径，不引用作者电脑。迁移时保留对应准备NPZ；必要时设置prepared_data。

## 波形

现有网络处理独立幅相样本，不能由两列输入生成真实动态功率曲线。需要导入同工况下的实测或仿真序列，并与本次run_id、sample_id绑定。模型数量随预测选择变化，单模型也可评估。

支持入射/反射/净输入功率、时间缩放和稳定阈值。所选模型波形须齐全、目标和时间网格一致。只有所有阈值通过才参与稳定选优；没有达标方案会明确显示。没有波形时仅报告预计匹配推荐，不填充预设曲线。

完整格式与扩展接口见 [references/interface.md](references/interface.md) 和 [references/training.md](references/training.md)。

## 其他模型与其他电脑

本地开发者可在model_types注册兼容的运行目录，新类型出现在面板下拉框中。需要实现训练、检查点、候选生成及响应评分协议；任意网络代码或任意pt不能直接通用。浏览器不能上传Python源码或指定任意运行路径，详见训练扩展文档。

仓库：[yanggggg03/callmynet_skill](https://github.com/yanggggg03/callmynet_skill)。别人下载、安装依赖并启动服务，即可在自己的电脑训练。训练数据、任务、机器配置和权重不随公开仓库或分发ZIP提供。

GitHub Pages仅托管面板，不能训练或运行PyTorch。公开网址需要连接用户本地服务，使用本地页面最直接。服务默认仅监听127.0.0.1，不提供公网推理或设备控制。

## 验证

```bash
python -B -m unittest discover -s tests -v
```

另用隔离的小规模合成数据验证训练、生成权重、启用、单/双模型预测和波形导入，不代表实际比赛精度。见 [references/validation.md](references/validation.md)。
