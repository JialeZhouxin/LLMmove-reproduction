# LLMmove 复现结果 — NYC 数据集

## 实验配置

| 配置项 | 值 |
|--------|-----|
| **模型** | qwen3.7-plus (via opencode.ai) |
| **数据集** | NYC (Foursquare) |
| **训练集** | 72,207 check-ins, 1,047 users, 4,937 POIs |
| **测试轨迹** | 1,400 |
| **候选集** | 101 (100 随机负采样 + 1 ground truth) |
| **推荐数** | 10 |
| **随机种子** | `int(trajectory_id)` (确定性) |
| **运行时间** | ~22h (含 3 轮 API 错误重试) |
| **API 成功率** | **100%** (0 条错误) |

## 最终指标

| 指标 | K=1 | K=5 | K=10 |
|------|:---:|:---:|:---:|
| **Acc@K** | **0.505** | **0.639** | **0.749** |
| **Recall@K** | 0.505 | 0.639 | 0.749 |
| **Precision@K** | 0.505 | 0.128 | 0.075 |
| **NDCG@K** | 0.505 | 0.578 | 0.612 |
| **MRR** | **0.571** | — | — |
| **ValidRatio** | **1.000** | — | — |

> 注：Recall@K 与 Acc@K 相同是因为 next POI 推荐每个测试用例只有 **1 个 ground truth**，属于该任务的标准现象。

## 与论文对比

论文 (Feng et al., 2024) 使用 GPT-3.5-turbo 在 NYC 数据集上报告以下指标：

| 指标 | 论文 GPT-3.5 (1364条) | 本复现 qwen3.7-plus (1400条) | 差距 |
|------|:-------------------:|:--------------------------:|:---:|
| **Acc@1** | 0.520 | 0.505 | -2.9% |
| **Acc@10** | 0.665 | 0.749 | **+12.6%** |
| **MRR** | 0.559 | 0.571 | **+2.1%** |
| **ValidRatio** | 0.999 | 1.000 | +0.1% |

## 文件结构

```
LLMpoi/LLMMove/
├── main.py                    # 入口脚本
├── models/
│   └── LLMMove.py             # 模型实现 (prompt构造 + API调用 + 指标计算)
├── diagnose_api.py            # API 连通性诊断工具
├── data/                      # 数据集 (需自行下载)
│   ├── NYC/NYC_train.csv
│   ├── NYC/NYC_val.csv
│   └── ...
├── output/LLMMove/nyc/        # 缓存 (API 响应)
└── results/
    └── LLMMove_nyc             # 最终结果
```

## 复现步骤

见 [REPRODUCTION.md](REPRODUCTION.md)
