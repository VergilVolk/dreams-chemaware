# Chem-aware DreaMS 注释平台 — 服务器部署清单

> 2026-08-18。本地机器已把工具/脚本/数据就绪，你把这些移到服务器后按下面顺序跑。
> 本地机器 = Windows + conda `D:\dreams_env`；服务器建议 Linux + conda `dreams`（Python 3.11）。

---

## 1. 要移过去的文件（按大小）

### 必须移（代码 + 权重 + 参考库，约 2 GB）

| 路径 | 大小 | 说明 |
|---|---|---|
| `annotation/` | 236 KB | 注释管线 14 个 .py（M1–M9 核心） |
| `dreams/` | 3.4 GB* | 核心库（含 backbone 权重 `dreams/models/pretrained/ssl_model_server.pt` 464 MB） |
| `tasks/` | — | 含 `encode_msv100574_spectra.py`、`convert_met_neg_hdf5.py` |
| `data/e1/official_embedding_slim.pt` | 468 MB | 线性 head 权重 |
| `data/models/mona_neg_full.mgf` | 141 MB | MONA-neg 参考谱库 |
| `data/models/mona_pos_full.mgf` | 304 MB | MONA-pos 参考谱库 |
| `data/models/mona_neg_dreams_emb/` | 155 MB | MONA-neg 预嵌入（embeddings.npy + manifest.csv） |
| `dreams/models/chem_aware/chem_rules_data.json` | 100 KB | 335 主规则 |
| `requirements.txt` | — | Python 依赖 |

\* `dreams/` 3.4 GB 大部分是 `models/pretrained/ssl_model_server.pt`（464 MB）和 `models/chem_aware/` 里的杂项。如果只想跑注释管线，可只移 `dreams/utils/`、`dreams/models/pretrained/ssl_model_server.pt`、`dreams/models/chem_aware/chem_rules_data.json` + `dreams/__init__.py`，不必整包搬。**但稳妥起见先整包搬，跑通再精简。**

### 数据（看你选哪种策略，见第 3 节）

| 选项 | 大小 | 说明 |
|---|---|---|
| A. 移 `.mzML`（本地转好） | Met/neg ~7 GB，全 4 组 ~60 GB | 本地用 ThermoRawFileParser 转完再搬 |
| B. 移 `.raw` | 全 4 组 120 GB | 服务器上用 Linux 版 ThermoRawFileParser 转 |
| C. 移 `.hdf5`（本地转好） | 同 mzML 量级 | 跳过服务器上的 mzML 读取 |

### 不用移（可选/未接入）
- `data/models/MS2DeepScore_allGNPSpositive_10k_500_500_200.pt`（21 MB）+ `ms2deepscore_model.pt`（416 MB）—— MS2DeepScore 打分还没接入，暂不需要。
- `dreams/models/chem_aware/chem_rules_massbank.json`（2.15 MB）—— 3151 条 MassBank 噪声规则，已排除不用。

---

## 2. 服务器环境搭建

```bash
conda create -n dreams python=3.11 -y
conda activate dreams
# GPU 服务器：先把 requirements.txt 里 torch==2.2.1 改成 torch==2.2.1+cu121
pip install -r requirements.txt
```

验证（缺一不可）：
```bash
python -c "import torch, numpy, pandas, h5py, pyteomics, scipy, sklearn; print('ok')"
```

---

## 3. 数据策略建议

**如果有 GPU**（强烈建议）：选 A 或 B，把 `.mzML`/`.raw` 搬到服务器，在服务器上做 hdf5 + 嵌入（GPU 上 87 万谱从 27 小时缩到 ~1-2 小时），再跑注释/FDR/差异。

**如果服务器也只有 CPU**：那 27 小时瓶颈一样存在，建议本地就把嵌入做完，只搬 `.hdf5` 或嵌入结果 `.npy`（87 万×1024 float32 ≈ 3.5 GB，比 60 GB mzML 小得多）。

---

## 4. 服务器上的运行顺序（Met/neg 为例）

```bash
# (1) 若搬的是 .raw，先转 mzML（Linux 版 ThermoRawFileParser，-f=1）
ThermoRawFileParser.exe -d=raw/Metabolomics/neg -o=Metabolomics/neg -f=1   # 路径按实际

# (2) mzML -> hdf5（断点续传）
python tasks/convert_met_neg_hdf5.py --mzml-dir data/msv100574/Metabolomics/neg

# (3) 嵌入（--hdf5 传全部或子集 30 文件）
python tasks/encode_msv100574_spectra.py \
    --hdf5 $(ls data/msv100574/Metabolomics/neg/*.hdf5) \
    --out data/msv100574/embeddings/met_neg

# (4) 注释（+规则注入）
python -m annotation.cli annotate \
    --query data/msv100574/embeddings/met_neg \
    --library data/models/mona_neg_dreams_emb \
    --out data/msv100574/annotation/met_neg

# (5) FDR（诱饵嵌入 68 min，nohup + 断点续传）
python tasks/run_fdr_met_neg.py

# (6) 差异分析 PF vs HF（run 脚本待补，见下）
```

---

## 5. 代码状态 + 服务器提交方式

**代码已全部就绪**（本轮补齐了最后两块）：
- `tasks/run_diff.py`（差异分析 run 脚本，烟雾测试通过：162 化合物 / 4 个 q<0.05）
- `tasks/run_fdr_met_neg.py`（分块诱饵嵌入 + 断点续传 + `--device`）
- `run_annotate_met_neg.sbatch`（服务器提交脚本，照仓库现有 `run_e1_identity.sbatch` 写法）

**服务器用 sbatch 提交**（仓库已有大量 `run_*.sbatch`，模式统一）：
```bash
sbatch run_annotate_met_neg.sbatch
```
脚本头：`#SBATCH --gpus=1` / `--time=24:00:00` / `--output=...%j.out`，正文 `cd /data02/run01/scv7tsl/DreaMS` + `export PYTHONPATH=$PWD` + `python -u ... --device cuda`。GPU 用 `#SBATCH --gpus=1`，多卡/array 用 `--gpus 8` / `--array=0-N`（见 `run_e1_identity.sbatch`）。

**FDR 本地重跑（验证代码能跑通，n=1 数据，~70 min）**：
> 本地是 Windows cmd.exe（Anaconda Prompt `(D:\dreams_env) D:\DreaMS>`），**不要用 bash 的 `nohup`/`$!`/`tail`**（cmd 不认）。用下面 cmd 命令。

后台新窗口跑（推荐，主窗口能继续用，结束后自动关）：
```bat
start "FDR" cmd /c "python -u tasks\run_fdr_met_neg.py > data\models\fdr_met_neg.log 2>&1"
```

或前台跑（卡住 ~70 min，但中断后重跑自动续传）：
```bat
python -u tasks\run_fdr_met_neg.py
```

看进度（另开一个窗口）：
```bat
type data\models\fdr_met_neg.log
```
实时刷新（PowerShell）：`Get-Content data\models\fdr_met_neg.log -Wait -Tail 20`

分块断点续传：中断后重跑自动跳过已存的 `data/models/mona_neg_decoy_chunks/chunk_*.npy`。
