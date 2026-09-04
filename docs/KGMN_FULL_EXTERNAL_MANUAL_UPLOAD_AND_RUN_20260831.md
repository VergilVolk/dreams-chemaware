# KGMN/DreaMS 外部主线：手工上传与单一执行入口

## 唯一提交命令

完成下列上传后，只提交：

```text
sbatch tasks/run_kgmn_full_external_pipeline.sbatch
```

Slurm 日志直接写入提交目录（通常为仓库根目录），不依赖预先存在的子目录：

```text
/data02/run01/scv7tsl/DreaMS/kgmn_full_external_<JOBID>.out
/data02/run01/scv7tsl/DreaMS/kgmn_full_external_<JOBID>.err
```

该作业在同一张 GPU 内依次完成：作者 200STD 基线、DreaMS 边校准、OEP003284 作者输入冻结、hidden-seed 外测。已完成且报告有效的阶段自动复用；有目录但缺正式报告的半成品会 fail-closed，绝不覆盖。

## 必须手工同步/上传的依赖

### 本地已打包的小依赖

仓库根目录已经生成：

```text
kgmn_external_dependencies_small_20260831.tar.gz
```

- 大小：118,562,677 B（113.07 MiB）
- SHA256：`feecb487166021b2848b283ee121ad274a1a43c7cefc67f7d687d439c12aa576`
- 校验契约：`tasks/contracts/kgmn_external_dependencies_small_20260831.sha256`

它包含冻结 MetDNA2 源码及 `.git`、两个 DreaMS 边校准 manifest 文件、三份 Zenodo 补充表，并保留服务器所需的相对目录结构。将压缩包上传到服务器仓库根目录后执行：

```text
tar -xzf kgmn_external_dependencies_small_20260831.tar.gz
```

该压缩包不包含 5.77 GB NODE 原始 mzXML，也不包含 Linux R 运行时。

| 服务器相对路径 | 大小 | 固定校验 |
|---|---:|---|
| `third_party/MetDNA2/` | 完整源码与 `inst/extdata` | git commit `5685ab219269c2f35cd5087655b0470b2da4d93c`，工作树必须 clean |
| `data/models/MassSpecGym_MurckoHist_split.hdf5` | 582,897,368 B | SHA256 `ccda2c4114d9b21413977df03376ca0fc097956a7fa304b861a3154a2b81e64f` |
| `data/e1/official_embedding_slim.pt` | 468,427,031 B | SHA256 `8928f908606c0bd652c5a4107d3c35102f660622958c225a1f625abe4b1ba245` |
| `dreams/models/pretrained/ssl_model_server.pt` | 464,254,162 B | SHA256 `9884b62ecadf4bd441d22fec79b6787e5ffef168e15e7d8d5804dbdea08b38b2` |
| `data/validation/kgmn_dreams_edge_calibration_manifest_20260831/report.json` | 2,107 B | SHA256 `dd2b2914d6c3a50187fd1976e17ed8e8e873a2cd6a8c6e7829c1d2f6b5ff1968` |
| `data/validation/kgmn_dreams_edge_calibration_manifest_20260831/paired_reaction_decoy_triples.csv.gz` | 21,345 B | SHA256 `11d41bfc3c6404fd1d9abfc0a1ce1473f19128c356b195c50eda93a86048c7d2` |
| `data/reference/kgmn_zenodo_7089991/Supplementary data1.xlsx` | 748,578 B | MD5 `8eadc3821d6e6973cc81cb3596ef414b` |
| `data/reference/kgmn_zenodo_7089991/Supplementary data2.xlsx` | 122,746 B | MD5 `9a047288772908c6bb0d34573bb3b2f8` |
| `data/reference/kgmn_zenodo_7089991/Supplementary data3.xlsx` | 5,914,502 B | MD5 `3e936cbbb22863371213ff8825c9f006` |
| `data/reference/OEP003284_raw/*.mzXML` | 24 文件；合计 5,769,447,979 B | 每文件 MD5 见 `tasks/contracts/kgmn_oep003284_node_files_20260831.csv` |

OEP003284 原始数据必须从 NODE 通过 SFTP 上传：

- host: `fms.biosino.org`
- port: `44398`
- remote path: `/Public/byRun/OER00/OER0025/OER002533/OER00253320`
- authentication: 用户自己的 NODE 账户；脚本不会读取、保存或打印密码。

必须保持 24 个原始文件的名字不变，直接放入 `data/reference/OEP003284_raw/`，不能再套一层目录。

## 不需要手工上传的派生物

以下均由单一作业生成，禁止提前拼接或伪造：

- `data/validation/kgmn_metdna2_200std_author_baseline/`
- `data/validation/kgmn_dreams_edge_calibration_official_20260831/`
- `data/validation/kgmn_external_validation_contract_20260831/`
- `data/validation/kgmn_oep003284_author_identifier_contract_20260831/`
- `data/reference/OEP003284/`
- `data/validation/kgmn_oep003284_hidden_seed_20260831/`

## 运行时依赖

服务器 `dreams` 环境必须提供 Python 模块 `numpy/pandas/scipy/sklearn/torch/h5py/openpyxl`，并可执行 `Rscript/git/sha256sum`。MetDNA2 的 R 依赖由既有环境与 `preflight_kgmn_metdna2_runtime.R` 逐项核验；正式作业只把冻结源码安装到 commit 专属 R library，不使用未知的全局 MetDNA2 包。

特别注意：公共 MetDNA2 仓库明确删除了受版权限制的 `zhuMetlib` 等库对象。当前冻结作者协议要求 `MetLib::loadLibData` 以及非空、结构正确的 `zhuMetlib`。仅安装公开 GitHub 包不能通过这一门。必须满足以下二者之一，且不能静默互换：

1. 在 Linux R 环境中安装用户合法持有、包含 `zhuMetlib` 的 MetLib 运行库，从而执行作者协议；
2. 正式改变研究协议为“公开固定种子网络传播”，不再声称复现完整作者 KGMN/MetDNA2 流程，并重建基线和外测契约。

## 结果与停止规则

最终只读取：

```text
data/validation/kgmn_oep003284_hidden_seed_20260831/final_decision.json
```

- `kgmn_oep003284_hidden_seed_external_passed`：允许进入 network-teacher 到共享 embedding 的下一阶段；
- `kgmn_oep003284_hidden_seed_external_failed`：作为真实负结果封存，不用次要臂替换主模型，不继续 embedding 蒸馏；
- 其他状态或缺文件：工程未闭环，不能解释性能。

依赖预检报告为 `data/validation/kgmn_full_external_preflight_<jobid>.json`。若失败，`problems` 会逐项列出缺文件、哈希错误、错误 commit、缺运行库或额外 mzXML。
