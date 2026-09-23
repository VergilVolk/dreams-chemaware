# DreaMS ChemAware Showspace：本机算力 + 公网面板

Showspace 是自托管应用，不需要 Hugging Face Spaces。浏览器只负责上传谱图和显示结果；DreaMS、E4/E8、MassSpecGym 检索与 P2b 全部在本机 GPU/磁盘上运行。

```text
browser -> HTTPS/domain -> Cloudflare Tunnel -> 127.0.0.1:7860 -> local GPU/model/index
```

本机没有入站端口，也不公开模型权重和参考谱库。生产环境使用 named tunnel；随机 `trycloudflare.com` quick tunnel 只适合临时测试。

## 当前算法边界

- `Official DreaMS`、`E4-A shared embedding`、`E8 shared embedding` 是互斥的共享编码器模式。
- 每个查询 checkpoint 只能访问由同一 checkpoint SHA256 生成的完整 reference index；模型、索引或 HDF5 指纹不一致会直接停止。
- P2b 是 embedding 之后的谱学融合预览，不是 E4/E8 的一部分，也不默认覆盖主排序。
- BioAware 是可弃权的 phenotype-blind 上下文证据，不是身份、通量或酶活结论。

## 1. 本机环境

在仓库根目录执行：

```powershell
powershell -ExecutionPolicy Bypass -File showspace\setup_showspace_env.ps1 -Recreate
Copy-Item showspace\.env.example showspace\.env
notepad showspace\.env
```

默认脚本复用 `D:\dreams_env` 里的 Torch/RDKit 科学栈，并在 `.venv_showspace` 固定兼容的 Web 依赖。若基础 Python 在别处：

```powershell
powershell -ExecutionPolicy Bypass -File showspace\setup_showspace_env.ps1 -Recreate -BasePython D:\path\python.exe
```

## 2. 构建模型对齐的参考索引

本机目前只有官方 checkpoint 时，先构建官方完整索引：

```powershell
$env:DREAMS_OFFICIAL_SLIM_CKPT='D:/DreaMS/data/e1/official_embedding_slim.pt'
$env:DREAMS_ARCHITECTURE_CKPT='D:/DreaMS/dreams/models/pretrained/ssl_model_server.pt'
.\.venv_showspace\Scripts\python.exe showspace\build_embedding_index.py `
  --model official_dreams `
  --data D:/DreaMS/data/models/MassSpecGym_MurckoHist_split.hdf5 `
  --output D:/DreaMS/models/showspace/index/official.npy `
  --device cuda --batch-size 128
```

E4-A/E8 必须先选定可部署的完整 shared checkpoint，再分别设置 `DREAMS_E4_CKPT`/`DREAMS_E8_CKPT` 并重编码整个参考库。严禁把 E4/E8 查询 embedding 与官方 reference index 混用。

## 3. 仅本机启动

编辑 `showspace/.env`，至少确保官方模型、HDF5、官方 `.npy/.json` 索引存在，并保持：

```text
GRADIO_SERVER_NAME=127.0.0.1
GRADIO_SHARE=false
SHOWSPACE_PUBLIC_MODE=false
```

然后：

```powershell
.\showspace\start_showspace.ps1
.\showspace\healthcheck_showspace.ps1
```

打开 `http://127.0.0.1:7860`。日志位于 `showspace/logs/`，停止命令是：

```powershell
.\showspace\stop_showspace.ps1
```

## 4. 域名公开访问（仍使用本机算力）

本机已安装 Tailscale 时，可先用 Funnel 获得稳定的 `*.ts.net` HTTPS 公网域名：

```powershell
.\showspace\start_tailscale_funnel.ps1
```

停止公网转发：

```powershell
.\showspace\stop_tailscale_funnel.ps1
```

首次使用需在 Tailscale 管理页允许该节点使用 Funnel。应用仍只监听 `127.0.0.1`，公网访问必须保留 Showspace 登录保护。

推荐在 Cloudflare Dashboard 创建 remotely-managed named tunnel，并把 public hostname（例如 `dreams.example.org`）路由到 `http://localhost:7860`。在本机安装 `cloudflared` 后，把 tunnel token 单独保存为：

```text
showspace/secrets/cloudflare-tunnel-token.txt
```

该目录被 Git 忽略。随后把 `.env` 改为：

```text
SHOWSPACE_PUBLIC_MODE=true
SHOWSPACE_USERNAME=your_user
SHOWSPACE_PASSWORD=a-long-random-password
CLOUDFLARE_TUNNEL_TOKEN_FILE=D:/DreaMS/dreams-chemaware/showspace/secrets/cloudflare-tunnel-token.txt
```

先启动本机应用，再启动 named tunnel：

```powershell
.\showspace\start_showspace.ps1
.\showspace\healthcheck_showspace.ps1
.\showspace\start_public_tunnel.ps1
```

停止：

```powershell
.\showspace\stop_public_tunnel.ps1
.\showspace\stop_showspace.ps1
```

正式公开时建议再在 Cloudflare Access 添加登录策略和速率限制；Gradio 自身账号只是第二道最小保护。不要把 tunnel token、模型或索引提交 Git。

## 5. 上线前验收

1. 本机 healthcheck 返回 HTTP 200。
2. 官方 checkpoint SHA256 与官方索引 manifest 完全一致。
3. HDF5 SHA256 与索引 manifest 完全一致，且 `complete=true`。
4. 未下载 E4/E8 checkpoint 或未构建匹配索引时，UI 必须显示未就绪，不能静默回退到官方或 demo。
5. `GRADIO_SHARE=false`、origin 为 `127.0.0.1`。
6. 公网启用 Cloudflare Access 或 Showspace 账号，不匿名开放 GPU 队列。

候选排序是检索证据，不是 MSI Level 1/2 鉴定结论；跨任务性能数字不能混用。
