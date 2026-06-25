# AGENTS.md — 基于 LLM 的下一兴趣点推荐 (Next POI Recommendation) 论文复现项目

## 1. 项目总览

本项目复现并对比多篇 **基于 LLM 的下一兴趣点推荐 (Next POI Recommendation)** 论文，核心目标是复现 **LRSA**（Zhu et al., Information Processing & Management 2026）。

### 涉及论文

| 论文 | 发表 | 角色 | 复现状态 |
|------|------|------|---------|
| **LRSA** (目标) | IP&M 2026 | 🎯 **最终目标** | 未开始 |
| **ROTAN** (ID-based SOTA) | KDD 2024 | LRSA 的 POI 嵌入来源 + 基线 | 未开始 |
| **LLM4POI** (LLM-based SOTA) | SIGIR 2024 | 最重要的 LLM 基线 | 未开始（需自实现） |
| **PEPLER** (LLM-based) | ACM TOIS 2023 | LLM 基线 | 未开始 |
| **LLMMove** (Zero-shot) | IEEE CAI 2024 | Zero-shot 基线 | ✅ 已完成 (1400/1400) |

**建议复现顺序**（依赖关系驱动）：
```
ROTAN → PEPLER → LLMMove → LLM4POI → LRSA
(验证数据)  (GPT-2 基线)  (ChatGPT 基线)  (需自实现)  (最终目标)
```

---

## 2. 仓库结构

```
paper/
├── AGENTS.md              # 本文件 — 项目约定
├── README.md              # 项目总览（手动维护）
├── repro/                  # 各论文复现入口（每个子目录独立）
│   ├── LLMMove/            # Zero-shot ChatGPT 基线
│   ├── ROTAN/              # ID-based SOTA 基线
│   ├── PEPLER/             # GPT-2 prompt-learning 基线
│   ├── LLM4POI/            # LLM-based SOTA 基线（自实现）
│   ├── LRSA/               # 最终目标论文
│   └── _template/          # 复现项目模板（目录骨架 + 说明）
├── docs/                   # 每篇论文的复现说明文档
│   ├── repro-ROTAN.md
│   ├── repro-PEPLER.md
│   ├── repro-LLMMove.md
│   ├── repro-LLM4POI.md
│   └── repro-LRSA.md
├── lib/                    # 共享工具库
│   ├── data_loader.py      # 数据加载/标准化
│   ├── metrics.py          # 评估指标（Acc@k, MRR, ValidRatio）
│   └── api_utils.py        # LLM API 统一调用
├── data/                   # 数据集（只读，不混入代码）
│   ├── NYC/                # Foursquare NYC
│   │   ├── NYC_train.csv
│   │   └── NYC_val.csv
│   ├── TKY/                # Foursquare Tokyo
│   │   ├── TKY_train.csv
│   │   └── TKY_val.csv
│   └── CA/                 # Gowalla California
│       ├── CA_train.csv
│       └── CA_val.csv
├── results/                # 实验结果
│   ├── results.csv         # 汇总表（所有实验的横向对比）
│   └── <Paper>_<Dataset>/  # 各实验详细日志
├── logs/                   # 运行日志
│   └── errors.log          # Agent 自动记录的错误日志
├── .venv/                  # uv 虚拟环境（本地开发用）
└── .gitignore
```

### 目录规范原则

- `repro/<Paper>/` 下只放**可执行代码**，每个论文完全隔离
- `docs/repro-<Paper>.md` 放复现步骤、数据准备、关键参数、预期结果、已知坑
- `lib/` 放所有论文共享的数据处理/评估工具，避免重复
- `data/` 视为只读，不允许在数据目录中生成中间文件
- `results/` 不提交二进制/大文件，只提交汇总 CSVs 和配置文件

---

## 3. 技术栈与环境

| 项目 | 要求 | 说明 |
|------|------|------|
| **Python** | ≥3.10 | 项目当前使用的版本 |
| **包管理** | `uv` | 所有依赖通过 `uv pip install` 管理 |
| **深度学习** | PyTorch ≥2.0 | ROTAN/PEPLER/LRSA 依赖 |
| **LLM API** | `openai` SDK (≥1.0) | 统一调用 OpenAI 兼容接口 |
| **本地 LLM 推理** | vLLM | 在云平台暴露 OpenAI 兼容 API |
| **代码风格** | ruff + black | 运行 `ruff check . && black .` 后再提交 |
| **CUDA** | 11.8+ | 云平台 GPU 环境 |

### 环境管理

- **本地环境**：`uv venv` 创建，`uv pip install -r requirements.txt` 安装依赖
- **云端环境**：通过 rsync/git 同步代码和 data，在云平台运行 `repro/<Paper>/run.sh`
- **API 配置**：通过环境变量或 `.env` 文件控制（不在仓库中提交 API Key）
  - `LLM_API_BASE` — API 地址（本地→远端 GPT，云端→本地 vLLM）
  - `LLM_API_KEY` — API Key
  - `LLM_MODEL` — 模型名

### 各论文依赖

每个 `repro/<Paper>/` 目录下应有自己的 `requirements.txt`，在运行前通过 `uv pip install -r requirements.txt` 安装。

---

## 4. 数据管理

### 标准数据格式

`data/` 下的 CSV 文件列顺序统一为：
```
,time,trajectory_id,check-in_id,user_id,latitude,longitude,poi_id,category,...
```

### 数据预处理

- `data/` 只存放原始标准格式数据，**不做任何修改**
- 每个 `repro/<Paper>/` 目录下必须包含 `prepare_data.py`
- `prepare_data.py` 从 `../../data/<Dataset>/` 读取标准数据，转换为该论文需要的格式，输出到 `./data/`（该 repro 目录下的本地数据目录）
- 所有 repro 共用同一份原始数据，确保对比公平

### 数据集来源记录

| 数据集 | 城市 | 来源 | 引用 |
|--------|------|------|------|
| NYC | 纽约 Foursquare | Foursquare 数据集 | 详见各论文 |
| TKY | 东京 Foursquare | Foursquare 数据集 | 详见各论文 |
| CA | 加州 Gowalla | Gowalla 数据集 | 详见各论文 |

---

## 5. 实验结果

### LLMmove — NYC 数据集 (qwen3.7-plus)

**配置:** 1400 测试轨迹, 101 候选集（100 负采样 + 1 GT），Prompt 含 40 长程 + 5 近期 check-in + Haversine 距离排序

| 指标 | K=1 | K=5 | K=10 |
|------|:---:|:---:|:---:|
| **Acc@K** | 0.5050 | 0.6386 | 0.7486 |
| **Recall@K** | 0.5050 | 0.6386 | 0.7486 |
| **Precision@K** | 0.5050 | 0.1277 | 0.0749 |
| **NDCG@K** | 0.5050 | 0.5779 | 0.6120 |
| **MRR** | **0.5708** | — | — |
| **ValidRatio** | **1.0000** | — | — |

**与论文 GPT-3.5 对比:** Acc@1 -2.9%, MRR +2.1%, ValidRatio 1.0000

---

## 6. 复现一篇论文的工作流

每篇论文从零到可运行遵循以下步骤：

1. **阅读论文** — 从 `docs/` 下的 PDF 提取文档（或已有 `.md` 提取稿）中理解方法
2. **编写复现文档** — 在 `docs/repro-<Paper>.md` 中记录复现计划、关键参数、预期结果
3. **准备数据** — 在 `repro/<Paper>/prepare_data.py` 中实现数据预处理
4. **实现模型** — 在 `repro/<Paper>/` 下编写模型代码
5. **运行验证** — 在本地用小数据集/少量样本验证代码通顺（见 [9.6 小规模测试](#96-小规模测试)）
6. **云平台跑全量** — 上传到云平台执行 `run.sh`
7. **记录结果** — 写入 `results/results.csv` 并保留日志到 `results/<Paper>_<Dataset>/`

---

## 6. 运行约定

### 统一入口

每个 `repro/<Paper>/` 必须提供：

| 文件 | 必须 | 说明 |
|------|------|------|
| `run.sh` | ✅ | 一键运行脚本，带头参数示例 |
| `README.md` | ✅ | 简短说明：怎么跑、依赖、预期结果 |
| `prepare_data.py` | ✅ | 数据预处理脚本 |
| `requirements.txt` | ✅ | 依赖列表 |

### run.sh 规范

```bash
#!/bin/bash
# 示例 — LLMMove 的 run.sh
cd "$(dirname "$0")"
uv pip install -r requirements.txt
python main.py --dataset nyc --llm gpt-4o-mini --cases 10
```

### 命令行参数规范

所有论文的入口脚本**尽可能**保持一致风格：
- `--dataset` / `-d`：数据集名（nyc/tky/ca）
- `--cases`：测试样本数（省略则跑全量）
- `--debug`：小规模调试模式，自动将所有参数设为最小（如 cases=1, epochs=1, batch_size=2），用于快速验证代码可运行
- `--llm`：模型名
- `--api-base`：API 地址（默认指向配置文件中的地址）

---

## 7. 实验结果记录

### 汇总表 `results/results.csv`

格式：
```
paper,dataset,Acc@1,Acc@5,Acc@10,MRR,ValidRatio,timestamp,notes,run_id
LLMMove,nyc,0.5000,0.6500,0.7700,0.5835,1.0000,2026-06-22T10:30,10 trajectories,llmmove-nyc-001
```

### 详细日志

每个实验的详细输出、配置文件、seed 等存入 `results/<Paper>_<Dataset>/<run_id>/`。

---

## 8. Git 工作流

### 分支策略

- `main` — 保持干净，只合入已完成的完整论文复现
- `repro/<paper-name>` — 每篇论文的独立开发分支（如 `repro/rotan`、`repro/lrsa`）
- 论文完成后通过 PR / 直接 merge 到 `main`

### 提交规范

每次 commit 只涉及**一篇论文**的代码。commit message 格式：

```
<Paper>-<action>: <简短描述>

示例：
LLMMove-add: add data preprocessing script
ROTAN-fix: correct data loader directory path
PEPLER-init: initial reproduction scaffolding
LRSA-impl: implement Influence-based Trajectory Correction module
```

不允许在同一 commit 中混合多个论文的修改。

### 忽略的文件

`.gitignore` 应包含：
- `.venv/`
- `__pycache__/`
- `*.pyc`
- `.env`
- `repro/*/data/`（数据预处理生成的临时数据）
- `results/*/` 中的二进制/大文件
- `logs/*.log`

---

## 9. Agent 行为规则

当 AI agent（如 Reasonix）在本项目中工作时，遵循以下约束：

### 9.1 修改范围

- **一次只修改一个论文复现项目**（`repro/<Paper>/` 下的单个项目）
- 修改完成并确认后再进入下一个论文
- 跨论文的共享修改（`lib/`、`data/`、`docs/`）在单个论文修改时附带处理

### 9.2 工作顺序

严格按依赖链推进：
```
ROTAN → PEPLER → LLMMove → LLM4POI → LRSA
```
除非用户明确指定跳过某个环节。

### 9.3 错误处理

1. 遇到错误时，记录上下文到 `logs/errors.log`
2. 尝试自动修复一次
3. 修复失败则询问用户

### 9.4 代码质量

- 运行 `ruff check .` 和 `black .` 进行格式化
- 每个函数应有 docstring（中英文均可，简明扼要）
- 关键超参数应有注释说明来源（论文中的哪个表格/段落）

### 9.5 请询问用户的情况

- 需要决策算法设计选择时（如 LRSA 的模块参数）
- 遇到论文中未明确的超参数时
- 实验结果偏离论文预期时（可能论文有未公开的训练细节）

### 9.6 小规模测试

**所有较大任务在运行全量之前 MUST 先通过小规模测试。**

#### 触发条件

以下情况视为"较大任务"，必须进行小规模测试：

- `--cases` 参数省略（即跑全量数据集）
- 模型训练任务（ROTAN/PEPLER/LRSA），未使用 `--debug` 或等价的最小参数组合
- `prepare_data.py` 首次运行时（应先用最小数据子集验证）

#### 小规模测试的最低标准

- 推理任务：`--cases 1`（或 `--debug`）跑通，无 import 错误、路径错误、shape mismatch、API 调用失败
- 训练任务：`--debug`（等价于 `--epochs 1 --batch-size 2` 等最小配置）完成至少一个 batch 的训练和一次评估，无报错
- 数据预处理：`prepare_data.py` 用单个数据集的最小子集验证格式转换正确

目标不是验证结果正确性，而是验证代码和依赖完整可运行。

#### 测试通过后的 gate 机制

小规模测试未通过时，**禁止**启动全量运行。失败处理复用 [9.3 错误处理](#93-错误处理) 流程（记录日志 → 自动修复一次 → 修复失败则询问用户）。测试通过后方可进入全量。

#### 例外

以下情况可以跳过小规模测试：

1. **用户明确要求跑全量** — agent 应提醒一次，用户坚持则跳过
2. **同一代码已通过小规模测试且代码无变更** — 无需重复测试
3. **测试涉及付费 API（如 ChatGPT）且用户未明确授权** — agent 不得自行决定调用付费 API；应先用本地/免费模型验证，或询问用户

#### `--debug` 参数约定

所有 `repro/<Paper>/` 的入口脚本应支持 `--debug` flag，效果为自动将所有可调节参数设为最小值：

- 推理任务：`cases=1`
- 训练任务：`epochs=1, batch_size=2`，以及其他训练相关参数的最小合法值
- 数据预处理：处理最少用户/轨迹数

agent 在小规模测试时优先使用 `--debug`，无需了解每个任务的"最小参数组合"。

---

## 10. 迁移计划

当前项目中的代码需要迁移到新结构：

| 当前路径 | 目标路径 | 操作 |
|---------|---------|------|
| `LLMpoi/LLMMove/` | `repro/LLMMove/` | 迁移代码到新结构 |
| `data/NYC/` | `data/NYC/` | 不动 |
| `data/TKY/` | `data/TKY/` | 不动 |
| `data/CA/` | `data/CA/` | 不动 |
| `LLMpoi/LLMMove/output/` | `results/LLMMove_<Dataset>/` | 日志迁移 |
| `LLMpoi/LLMMove/results/` | `results/results.csv` | 汇总 |
| 各 PDF/MD | `docs/` | 整理 |

迁移分阶段进行，在每个论文分支中完成对应部分的迁移。
