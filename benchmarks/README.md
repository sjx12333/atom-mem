# 测试结果快照

这些文件是从原工程评测目录复制的**结果快照**，用于支撑根目录 README 的结论；不包含 API key、角色私有数据或原始对话。

| 文件 | 内容 | 口径 |
| --- | --- | --- |
| `locomo-equal-token.md` | LoCoMo conv0–4，999 题，8 档 token 的四系统对比 | 同一 reader/judge、同一 local prompt、同一 `cl100k_base` 打包 |
| `locomo-equal-token.metrics.json` | 上表的机器可读指标 | 逐题配对 / McNemar / bootstrap CI |
| `chinese-chatbot-corpus.md` | 中文闲聊 + 字幕噪声：112 回合 / 74 查询 | 批次封存 → 抽取 → 查询；严格@3 |
| `graph-boost-negative-result.md` | 两种自动图增强在 LoCoMo 单跳 418 题上的负结果 | BM25→BFS、谓词→DFS；含随机事实对照 |
| `graph-tool-accuracy.md` | 图查询工具 10 题合成多跳 | 不开工具 3/10；开工具 10/10 |

## 关键数字

- 等 token 相对 Mem0：64–2048 档稳定领先 **+7.6 ~ +11.6 pp**（p<0.0001）。
- 纯图自动召回：LoCoMo 单跳 gold@30 **52.4% → 52.4%**，MRR 反而下降；Graphiti Core 混合检索 **4.1–4.6 s、0 admitted**。
- 图作为工具：10 题从 **3/10** 提升到 **10/10**，所以本包只把 PPR 放在 `operator="ppr"` 的工具调用里。

> 这些数字来自本工程同一套记忆核心的历史实测，不是在本目录重新跑出的新榜单；逐题产物和脚本保留在原工程 `docs/reports/`、`_rawrag_bench/`、`_locomo_bench/`。不同数据集的分数不可横向相加。
