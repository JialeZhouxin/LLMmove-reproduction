# LLMmove Reproduction — Next POI Recommendation

基于 [LLMmove (Feng et al., 2024)](https://arxiv.org/abs/2408.13464) 论文的复现实现。

## Results

**NYC 数据集 (qwen3.7-plus)**

| 指标 | K=1 | K=5 | K=10 |
|------|:---:|:---:|:---:|
| Acc@K | 0.505 | 0.639 | 0.749 |
| Recall@K | 0.505 | 0.639 | 0.749 |
| Precision@K | 0.505 | 0.128 | 0.075 |
| NDCG@K | 0.505 | 0.578 | 0.612 |
| MRR | 0.571 | — | — |
| ValidRatio | 1.000 | — | — |

详见 [NYC_RESULTS.md](NYC_RESULTS.md) 和 [REPRODUCTION.md](REPRODUCTION.md)。

## 快速开始

```bash
pip install openai tqdm tenacity
cd LLMpoi/LLMMove
python main.py -d nyc --llm qwen3.7-plus --api-base https://opencode.ai/zen/go/v1 --cases 1
```
