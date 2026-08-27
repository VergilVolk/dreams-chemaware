---
title: DreaMS-based ChemAware Showcase
emoji: 🧪
colorFrom: purple
colorTo: violet
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: true
---

# DreaMS-based ChemAware Showcase

这是一个基于 DreaMS 的扩展项目展示面板，不等同于上游 DreaMS。它提供多格式质谱上传、官方 DreaMS embedding、候选分子检索、结构信息展示和批量导出。

## 功能

- 上传 `.mgf`、`.mzML`、`.mzXML`、`.hdf5`、`.h5`、`.hd5`、`.json`
- 多谱图选择、峰表和 stick plot 预览
- 使用官方 `embedding_model.ckpt` + `ssl_model.ckpt` 生成真实 1024 维 DreaMS 表示
- MassSpecGym 候选分子检索、结构去重、SMILES/Formula 和 RDKit 性质展示
- 单谱和批量 JSON/CSV 导出
- 启发式化学规则提示

候选排名是相似度排序，不是结构鉴定结论或鉴定置信度。

## 资源要求

代码随 Git 同步，但以下资源不提交到仓库：

- 官方模型：`embedding_model.ckpt`、`ssl_model.ckpt`
- 候选数据库：`MassSpecGym_DreaMS.hdf5`

官方模式还需要主仓库中的 `dreams/` Python 包。模型和数据库建议放在仓库之外，例如：

```text
D:\DreaMSData\models\embedding_model.ckpt
D:\DreaMSData\models\ssl_model.ckpt
D:\DreaMSData\MassSpecGym_DreaMS.hdf5
```

## Windows 台式机：首次部署

推荐使用 Python 3.11 x64。以下命令在 `dreams-chemaware` 仓库根目录执行：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip wheel
.\.venv\Scripts\python.exe -m pip install -r showspace\requirements.txt
```

复制配置模板并编辑路径：

```powershell
Copy-Item showspace\.env.example showspace\.env
notepad showspace\.env
```

`.env` 至少应包含：

```text
DREAMS_EMBEDDING_CKPT=D:/DreaMSData/models/embedding_model.ckpt
DREAMS_SSL_CKPT=D:/DreaMSData/models/ssl_model.ckpt
DREAMS_MOLECULE_DB=D:/DreaMSData/MassSpecGym_DreaMS.hdf5
GRADIO_SERVER_NAME=127.0.0.1
GRADIO_SERVER_PORT=7860
GRADIO_SHARE=false
GRADIO_FRONTEND_CHECK=false
```

- `127.0.0.1`：仅台式机本机访问。
- `0.0.0.0`：允许局域网访问，需配置 Windows 防火墙。
- `GRADIO_SHARE=true`：请求 Gradio 临时公网 `gradio.live` 链接；需要台式机能连接 Gradio 服务。

检查并启动：

```powershell
.\showspace\start_showspace.ps1
.\showspace\healthcheck_showspace.ps1
```

打开：

```text
http://127.0.0.1:7860
```

停止服务：

```powershell
.\showspace\stop_showspace.ps1
```

查看日志：

```powershell
Get-Content showspace\logs\showspace.out.log -Wait
Get-Content showspace\logs\showspace.err.log -Wait
```

## 台式机：以后 Git 更新

```powershell
cd C:\path\to\dreams-chemaware
.\showspace\stop_showspace.ps1
git pull
.\.venv\Scripts\python.exe -m pip install -r showspace\requirements.txt
.\showspace\start_showspace.ps1
.\showspace\healthcheck_showspace.ps1
```

`.env`、模型、数据库和日志都被忽略，不会被 `git pull` 覆盖。

## 前台调试启动

```powershell
cd C:\path\to\dreams-chemaware
Get-Content showspace\.env | ForEach-Object {
    if ($_ -match '^\s*([^#=]+)\s*=\s*(.*)\s*$') { [Environment]::SetEnvironmentVariable($matches[1].Trim(), $matches[2].Trim(), 'Process') }
}
.\.venv\Scripts\python.exe showspace\app.py
```

## 重要说明

- 默认 `official_dreams` 使用真实官方 DreaMS checkpoint；`demo` 仅用于没有权重时测试界面流程。
- `.mgf` 示例可用于手工验收；大 HDF5、模型权重和虚拟环境不进入 Git。
- 当前面板不是正式 ChemAware checkpoint 推理，也不会把 demo 向量当作科学结果。
- `Dockerfile` 只适合在额外提供 `dreams/` 源码、模型和数据库的构建环境中使用；台式机优先使用上述 Python/PowerShell 方案。
