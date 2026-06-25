# ZeroShot_Qwen — 零样本下一兴趣点推荐（Next POI Recommendation）

通过 API 调用 `qwen/qwen-2.5-7b-instruct`（OpenRouter），无需任何微调，直接对 NYC 数据集做零样本下一 POI 推荐。

## 结果总览（NYC, 412 test samples, candidate_size=10）

| 指标 | Zero-Shot Qwen-2.5-7B | MoE 微调（实验说明） | 差距 |
|------|:---------------------:|:------------------:|:----:|
| Acc@1 | **0.3495** (144/412) | 0.4587 | −0.1092 |
| Acc@5 | **0.8544** (352/412) | 0.8932 | −0.0388 |
| Acc@10 | **0.9976** (411/412) | 1.0000 | −0.0024 |
| MRR | **0.5628** | 0.6359 | −0.0731 |
| NDCG@5 | **0.5436** | 0.6889 | −0.1453 |
| NDCG@10 | **0.5628** | 0.7241 | −0.1613 |
| ValidRatio | **1.0000** | 1.0000 | 0.0000 |
| Mean Rank | **2.92** | 2.55 | +0.37 |

**效率统计**
- 总 LLM 调用：1,214 次（平均 2.95 次/样本）
- 早停率：99.8%（仅 1 个样本耗尽全部候选）
- API 错误：0
- 预估费用：~$0.03（按 OpenRouter $0.04/1M input + $0.10/1M output）

---

## 方法

### 推理策略
采用**迭代猜测**机制，与实验说明中的 MoE 微调方法一致：

1. 构建候选池：1 个正确答案 + 9 个全局随机负样本 = 10 个候选
2. 模型从候选池中选出最可能的 1 个 POI
3. 猜对 → 记录命中位置，停止
4. 猜错 → 移除该错误候选，用剩余候选重新提问
5. 重复直到命中或候选池空

### Prompt 结构（纯文本版，不使用额外 embedding）

```
system: Identify the most plausible next POI from the provided Candidate Pool
        based on the user's historical preferences, movement trajectory,
        and current spatial-temporal constraints.

user:
### [User Long-term Preference]
- POI Check-in Categories: Daily routine: Train Station (21%); ...

### [User Short-term Trajectory]
 POI Check-in Sequences:
- December 16th afternoon at 15:08 (sunday) | Department Store | geohash: dr5rzg
- ...

### [Spatio-Temporal Context]
- Target Time: December 19th morning at 07:00 (wednesday)
- Displacement: Δd=0.15km, Δt=60min from last visit.

### [Candidate Pool]
Which of the following POIs is the user most likely to visit next?
- [ID: 3410 | Category: Coffee Shop]
- ...
```

### 数据过滤
- 测试轨迹：NYC 验证集，过滤条件：
  - 轨迹长度 ≥ 5（最后一条签到作为 ground truth）
  - 用户在训练数据中有 ≥ 2 个不同的 POI 类别（排除冷启动用户）
- 最终样本数：**412**

### 关键设计选择
| 决策 | 选择 |
|------|------|
| 数据集 | NYC |
| 模型 | `qwen/qwen-2.5-7b-instruct` via OpenRouter |
| 历史轨迹上限 | 50 条（cap at 50） |
| 候选池大小 | 10（1 GT + 9 随机负采样） |
| 推理方式 | 迭代猜测（每次输出 1 个 POI ID，直到命中） |
| 输出格式 | JSON mode (`response_format={"type": "json_object"}`) |
| 输出 key | `predicted_poi_id` |

---

## 复现步骤

### 前置条件
- Python ≥ 3.10
- OpenRouter API Key（[获取](https://openrouter.ai/keys)）
- 网络可访问 `https://openrouter.ai/api/v1`

### 1. 克隆仓库
```bash
git clone https://github.com/JialeZhouxin/LLMmove-reproduction.git
cd LLMmove-reproduction
```

### 2. 创建虚拟环境（推荐 uv）
```bash
uv venv
uv pip install openai geohash2 tqdm
```

### 3. 运行评估
```bash
cd repro/ZeroShot_Qwen

# 设置 API Key
export OPENROUTER_API_KEY="sk-or-v1-..."

# 调试模式（1 个样本）
python main.py --debug

# 全量 NYC 评估（412 样本）
python main.py --dataset nyc

# 指定参数
python main.py --dataset nyc --candidate-size 10 --min-traj-len 5
```

### 4. 查看结果
- 终端直接输出指标汇总
- 详细结果保存到 `output/nyc_eval_cand_10_mintrj_5_results.json`
- 每轨迹缓存保存到 `output/nyc/`（可断电续跑）
- 汇总追加到 `../../results/results.csv`

### 参数说明
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `-d, --dataset` | `nyc` | 数据集（nyc/tky/ca） |
| `--model` | `qwen/qwen-2.5-7b-instruct` | 模型名 |
| `--api-base` | `https://openrouter.ai/api/v1` | API 地址 |
| `--candidate-size` | 10 | 候选池大小 |
| `--min-traj-len` | 5 | 最小轨迹长度 |
| `--cases` | 全部 | 测试样本数 |
| `--debug` | — | 调试模式（1 样本） |

---

## 文件结构
```
repro/ZeroShot_Qwen/
├── main.py              # 完整实现：数据加载、Prompt 构建、API 调用、迭代推理、指标计算
├── requirements.txt     # 依赖：openai, geohash2, tqdm
├── run.sh               # 一键运行脚本
├── output/
│   ├── nyc/             # 每轨迹缓存（412 个文件，可恢复）
│   └── nyc_eval_cand_10_mintrj_5_results.json  # 详细结果
```

## 依赖
- `openai` — OpenAI 兼容 API 调用（OpenRouter）
- `geohash2` — 坐标转 geohash 编码
- `tqdm` — 进度条

## 注意事项
- API Key 通过环境变量 `OPENROUTER_API_KEY` 传入，**不要提交到 GitHub**
- 输出目录 `output/` 已经在 `.gitignore` 中忽略
- 首次运行会自动下载数据（在 `data/` 目录下），或手动放置标准 CSV 格式数据
- CA 数据集尚未测试（类别列为 JSON 格式，需额外处理）
