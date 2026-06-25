# LLMmove 复现教程

## 环境要求

- Python 3.10+
- 一个兼容 OpenAI API 的 LLM 端点（支持 chat/completions）
- 约 50 GB 磁盘空间（含数据集和缓存）

## 步骤 1：准备环境

```bash
# 创建虚拟环境
uv venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# 安装依赖
uv pip install openai tqdm tenacity
```

## 步骤 2：准备数据

数据集使用 Foursquare NYC 和 TKY 数据，已按 STHGCN 格式预处理。

文件格式（CSV）：

```
check_ins_id,local_time,UTCTimeOffset,pseudo_session_trajectory_id,query_pseudo_session_trajectory_id,
UserId,latitude,longitude,PoiId,POI_catid,PoiCategoryName,last_checkin_epoch_time,trajectory_id,
POI_catid_code,kg_user_id
```

将数据放在以下路径：

```
data/
├── NYC/
│   ├── NYC_train.csv
│   └── NYC_val.csv
├── TKY/
│   ├── TKY_train.csv
│   └── TKY_val.csv
└── CA/
    ├── CA_train.csv
    └── CA_val.csv
```

## 步骤 3：配置 API Key

```bash
# Linux/Mac
export OPENAI_API_KEY="your-api-key"

# Windows CMD
set OPENAI_API_KEY="your-api-key"

# Windows PowerShell
$env:OPENAI_API_KEY="your-api-key"
```

支持任意兼容 OpenAI 格式的 API 端点（OpenAI、DeepSeek、通义千问等）。

## 步骤 4：运行

### 快速测试（1 条轨迹，验证环境）

```bash
cd LLMpoi/LLMMove
python main.py -d nyc --llm qwen3.7-plus --api-base https://opencode.ai/zen/go/v1 --cases 1
```

### 跑完整测试集

```bash
python main.py -d nyc --llm qwen3.7-plus --api-base https://opencode.ai/zen/go/v1
```

完整 NYC 测试集（1400 轨迹）耗时约 15-22h，具体取决于 API 响应速度。

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-m, --model` | 模型名称（固定 `LLMMove`） | 必填 |
| `-d, --datasetName` | 数据集：`nyc` 或 `tky` | `nyc` |
| `--llm` | LLM 模型名称（如 `qwen3.7-plus`, `gpt-3.5-turbo`, `deepseek-v4-flash`） | `gpt-3.5-turbo` |
| `--api-base` | API 端点 URL | `https://api.openai.com/v1` |
| `--cases` | 限制测试轨迹数（调试用） | 全部 |

### 示例：不同 API 端点

```bash
# OpenAI
python main.py -d nyc --llm gpt-3.5-turbo --api-base https://api.openai.com/v1

# DeepSeek 官方
python main.py -d nyc --llm deepseek-v4-flash --api-base https://api.deepseek.com/v1

# 通义千问 (opencode.ai)
python main.py -d nyc --llm qwen3.7-plus --api-base https://opencode.ai/zen/go/v1
```

## 步骤 5：查看结果

运行结束后自动输出：

```
=== Results (nyc) ===
  acc@1:      0.5050
  acc@5:      0.6386
  acc@10:     0.7486
  mrr:        0.5708
  validRatio: 1.0000
```

结果同时保存到 `results/LLMMove_nyc`。

### 手动计算全部指标

```bash
python -c "
import json, os, math, sys
sys.path.insert(0, 'LLMpoi/LLMMove')
from models.LLMMove import parse_response, normalize_prediction

d = 'LLMpoi/LLMMove/output/LLMMove/nyc'
files = [f for f in os.listdir(d) if os.path.isfile(os.path.join(d, f))]
valid = len(files)
hits = {k:0 for k in [1,5,10]}
ndcg = {k:0.0 for k in [1,5,10]}
rr_sum = 0.0

for fname in files:
    with open(os.path.join(d, fname)) as f:
        data = json.load(f)
    gt = str(data['groundTruth'][0]) if isinstance(data['groundTruth'], (list,tuple)) else str(data['groundTruth'])
    resp = data.get('response')
    if isinstance(resp, str):
        parsed = json.loads(resp) if resp.startswith('{') else resp
    else:
        parsed = resp
    pred = normalize_prediction(parsed.get('recommendation', []))
    if gt in pred:
        rank = pred.index(gt) + 1
        rr_sum += 1.0/rank
        for k in [1,5,10]:
            if rank <= k:
                hits[k] += 1
                ndcg[k] += 1.0/math.log2(rank+1)

print(f'{\"Metric\":<15} {\"K=1\":>10} {\"K=5\":>10} {\"K=10\":>10}')
print('-'*47)
for m in ['Acc@K','Recall@K','Precision@K','NDCG@K']:
    v=[]
    for k in [1,5,10]:
        if m=='Precision@K': v.append(hits[k]/(valid*k))
        elif m=='NDCG@K': v.append(ndcg[k]/valid)
        else: v.append(hits[k]/valid)
    print(f'{m:<15} {v[0]:>10.4f} {v[1]:>10.4f} {v[2]:>10.4f}')
print(f'MRR:       {rr_sum/valid:.4f}')
print(f'ValidRatio: {valid}/1400 = {valid/1400:.4f}')
"
```

### 指标说明

| 指标 | 说明 |
|------|------|
| **Acc@K** | Ground Truth 是否在 Top-K 推荐中 |
| **Recall@K** | 相关 POI 被召回的比例（= Acc@K，因每例只有 1 个 GT） |
| **Precision@K** | Top-K 中相关 POI 的比例 |
| **NDCG@K** | 归一化折损累计增益，考虑排名位置 |
| **MRR** | Mean Reciprocal Rank，第一个命中的倒数排名均值 |
| **ValidRatio** | LLM 返回有效 JSON 的比例 |

## 缓存机制

代码自动缓存每个 API 响应到 `output/LLMMove/{dataset}/{trajectory_id}`。

- **断点续跑**：重跑会自动跳过已有缓存
- **清理缓存**：删除对应目录即可重新生成单条

## 已知问题

1. **Cloudflare 524 超时**（opencode.ai 端点）：某些请求超过 120s 时触发，代码自动跳过不缓存，需重跑
2. **数据集版本差异**：不同预处理版本可能导致轨迹数略有不同，不影响指标趋势
3. **LLM 非确定性**：不同 LLM 返回结果有差异，建议使用同一模型和 seed
