# CallMyNet Skill

把三套比赛网络的训练权重接入同一个调节工作台：推荐L1/L2，展示三模型结果，导入实测或仿真功率波形比较稳定性，并下载Excel。可作为Codex、DeepSeek或其他支持本地脚本/HTTP工具的Agent技能使用。

## 运行

```powershell
# 使用已经安装PyTorch的训练环境
./start_panel.ps1 -Python 'D:\Anaconda\CondaData\env_dirs\WDCNN\python.exe'
```

打开 **http://127.0.0.1:8765/**。Mac/Linux或其他Python环境：

```bash
python -m pip install -r requirements.txt
python -B scripts/server.py
```

未训练时点击 **载入示例**，即可查看两区面板、功率波形、稳定指标与Excel下载。示例完全是合成数据，不代表真实模型性能。

## 训练完成后接入

1. 三套模型用同一份准备数据与固定划分训练，分别保存正式 `runs/measured/best.pt`。
2. 修改 `config/models.json` 中三个 `checkpoint` 路径。默认已经指向当前电脑的 `D:/Projects/competition_*/runs/measured/best.pt`。
3. 使用面板输入 `mag,deg` 或导入预测Excel。点击运行，三个模型分别给出电长度方案；共用一个响应代理比较全部候选。
4. 采集或仿真三种方案在相同工况下的功率时间序列，按模板填入本次 `run_id`、样本编号和工况编号，导入功率区。
5. 查看稳定性判断并下载Excel。没有波形时，只报告预计匹配推荐；三者都未稳定时，不宣称已有稳定方案。

现有网络为 **残差MLP、Fourier ResMLP、条件RealNVP Flow**。它们是幅相到支节电长度的独立样本模型，不能凭空生成真实动态功率曲线。完整接口、判据和输入单位见 [references/interface.md](references/interface.md)。

## 面板

- **结果区**：输出结果/示例说明、样本切换、三模型表格、预计或波形验证后的推荐、Excel下载。
- **波形区**：三方案与目标功率曲线、入射/反射/净输入功率通道、悬浮读数、时间缩放、波形导入、共同有功率窗口、阈值配置和稳定性比较。
- **Excel**：三模型结果、预计推荐、稳定指标、完整功率样本、波形选优结果和来源说明。

默认稳定性阈值为演示用工程判据，可按装置要求调整。统一目标和采样网格防止不公平比较；CV、纹波、误差、漂移、超调全部通过后才参与稳定选优。判定只适用于本次观察窗口。

功率示例采用20kW入射平台和亚秒级反射衰减，形态参考ICRF论文图3.41/3.42，所有示例动作及三模型曲线均为预设合成值。示例中的优胜者没有真实模型比较意义。支持显式设置脉冲有功率窗口，完整启动/关断曲线仍保存在Excel。用户的原始测量及2500条增强数据仅保存在本地，不包含在公开仓库中。

## 作为Skill使用

此仓库根目录的 `SKILL.md` 为技能入口。支持此格式的Agent可以将整个仓库放入自己的技能目录；也可以自行读取 `SKILL.md` 并通过本地命令或HTTP调用。仅有普通聊天窗口且没有脚本/工具执行能力时，需要用户自己启动运行环境。

技能本身不包含大语言模型权重。PyTorch权重独立保存在本机，使用者需取得模型权重后配置路径。代码可跨模型复用，执行效果取决于Agent是否提供文件、命令或HTTP工具。

## 发布到GitHub

目标仓库：[yanggggg03/callmynet_skill](https://github.com/yanggggg03/callmynet_skill)。仓库提供Skill源码和运行脚本；本地训练权重、训练数据不会自动公开。

仓库包含GitHub Pages工作流。在仓库 **Settings → Pages → Source** 选择 **GitHub Actions**，推送到main/master后部署静态面板。部署成功后地址预期为 **https://yanggggg03.github.io/callmynet_skill/**；以Actions实际部署地址为准。

公开面板能直接演示与下载示例Excel。真实推理在本地运行：访客下载项目并启动自己的服务，再使用本地面板。若从公开面板连接本机API，需要配置 `allowed_origins` 与服务地址，详见接口文档。当前方案没有让外部访客通过同一网址调用你的私人电脑。

## 验证

```bash
python -B -m unittest discover -s tests -v
```

针对原项目测试权重的集成核验使用单独的本地配置完成，不将测试权重作为正式权重分发。模型兼容代码保存在 `models/<id>/`；发布包不依赖原来的 `D:/Projects` 源代码目录，只需要相应正式检查点。
