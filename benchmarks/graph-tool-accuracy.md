# 图谱查询工具（Method A + C）实现与准确率对比

日期：2026-11-08  
范围：当前工作区（含未提交改动）

## 1. 做了什么

### Method A：实体锚定召回

- 新增 graphiti_memory/entity_linking.py：
  - 从 query 解析所有白名单实体，而不是只返回一个布尔值；
  - 支持完整名、别名、中文名片段（"爱迪生" -> "托马斯·爱迪生"）和英文事实名回退（"Jon" -> 节点"主人"）；
  - 属性描述题的回退（"美国第37任总统" -> 尼克松）只在没有名字锚点且 query 含中文时启用。
- NodeMemoryRecall：
  - 每个 anchor 预留最多 anchor_facts_per_entity 条事实（默认 6）；
  - 多 anchor 时允许多个 BFS seed（上限 anchor_seed_limit，默认 3）；
  - 预留只是候选，仍走同一套 rerank/门槛。
- RetrievalPolicy / runtime / graphiti_config：
  - entity_anchoring_enabled=true
  - anchor_facts_per_entity=6
  - anchor_seed_limit=3

### Method C：图谱查询工具

- 新增 graphiti_memory/graph_query.py，确定性执行：
  - facts / list / union / intersection / path / count
  - 锚点解析、谓词推断（visited / wrote / met / helped / lives / works / owns）
  - 交集按共同实体和共同谓词；路径为有界最短路径；计数只数不同对象。
- 新增 POST /graphiti/query。
- 新增 LLM 工具 memos_graph_query，默认随只读查询工具暴露。
- 工具只返回图证据，不生成答案；最终回答仍由模型完成。

## 2. 评测设计

因为 LoCoMo 的角色图在本次工作区已被清空（只剩 recall/judge 产物，topology 不在），无法直接重跑 LoCoMo 图检索。为了隔离"工具是否有效"，使用一张**合成图**：

- 15 个实体、33 条事实、3 条关系；
- 10 道多跳/集合题：交集、分别、共同对象、路径、计数；
- 实体名是虚构的（Aria/Borin/Zephyria/Emberstone 等），模型无法靠参数化知识猜答案；
- 自动召回使用真实 NodeMemoryRecall：fact-first + Method A，top_k=8；
- 两个条件：
  - no_tool：只给自动召回；
  - active_tool：开放 memos_graph_query，由模型自行决定调用（OpenAI function calling），工具结果返回后回答；
- 模型 deepseek-v4-flash，temperature=0，max_tokens=1200；
- 严格 LLM judge：答非所问、说"未提及"、与 gold 矛盾、两段答案只答一段都判 WRONG。

脚本：_locomo_bench/tool_accuracy_eval.py  
输出：_locomo_bench/.graph-tool-eval-result.json

## 3. 结果

两次独立运行一致：

| 条件 | 正确 | 准确率 | 工具调用题数 |
| --- | ---: | ---: | ---: |
| 不使用工具 | 3 / 10 | 30% | 0 |
| 使用工具（模型主动调用） | 10 / 10 | 100% | 9 / 10 |

逐题：

| 问题 | 类型 | 不用工具 | 用工具 |
| --- | --- | :---: | :---: |
| Which city did both Aria and Borin visit? | 交集 | ✗ | ✓ |
| Did both Aria and Borin visit Zephyria? | 交集 | ✗ | ✓ |
| What object do Aria and Borin both own? | 交集 | ✗ | ✓ |
| How many cities did Borin visit? | 计数 | ✓ | ✓ |
| Who connects Aria and Dorin? | 路径 | ✗ | ✓ |
| What does Aria own? | 单事实 | ✓ | ✓ |
| What do Aria and Celia both like? | 交集 | ✗ | ✓ |
| Which organization do Aria and Borin both work at? | 交集 | ✗ | ✓ |
| What did Celia and Dorin respectively visit? | 分别 | ✗ | ✓ |
| How many things does Aria own? | 计数 | ✓ | ✓ |

不用工具时，模型只有 8 条扁平事实：8 条里通常只覆盖一个实体，或者覆盖了两个实体但缺少"共同/关系"那一条，所以交集、路径、分别题大量答"未提及"。工具把这些变成显式算子：

- 交集：A 和 B 都提到 Zephyria / Emberstone / Moonpetal / Star Guild；
- 路径：Aria -> Celia -> Dorin；
- 分别：每个 anchor 的事实分组；
- 计数：只数不同的城市/对象。

### 真实图冒烟

- 在 CCC 角色缓存图上，工具能解析"托马斯·爱迪生 / 詹姆斯·瓦特"这类全名与短名的对应，并返回两端事实；
- 但真实 CCC 题的 no_tool 基线会被模型世界知识抬高（例如它知道爱迪生/瓦特），不能用来衡量工具增益；
- 这也是本次改用虚构实体做准确率对比的原因。

## 4. 已知限制

1. **合成图不是 LoCoMo。** 数字证明的是"结构化图算子能补上扁平 top-k 缺失的多实体证据"，不是线上 LoCoMo 的最终准确率。
2. **实体链接仍是瓶颈。** 查询用属性描述实体（"第 37 任总统"）而没有名字时，只有中文且完全没有名字锚点时才启用重叠回退；英文短名/昵称可能解析失败。
3. **工具输出会占上下文。** 默认每 anchor 最多 12 条事实；两个 anchor 时可能较长。当前预算下 10 题都正常，但真实长图需要更严格的 per-anchor 配额。
4. **模型不一定调用。** 本次 9/10 题调用了工具；单事实题（"What does Aria own?"）没有调用也答对了。生产提示词需要继续强调"多实体/分别/共同/路径/计数"才调用。
5. **真实 LoCoMo 需重新建图后复跑。** 复跑时应把本脚本的合成图替换为 LoCoMo 角色 topology，并保留同一套 no_tool/active_tool 配对。

## 5. 复现

~~~bash
python _locomo_bench/tool_accuracy_eval.py
~~~

需要 memos_system/config/graphiti_config.json 中可用的 chat key；脚本会写 _locomo_bench/.graph-tool-eval-result.json。

## 6. 相关文件

- graphiti_memory/entity_linking.py
- graphiti_memory/graph_query.py
- graphiti_memory/node_recall.py
- graphiti_memory/retrieval.py
- graphiti_memory/runtime.py
- graphiti_memory/gateway.py
- graphiti_memory/service.py
- api/graphiti_routes.py
- live-2d/plugins/built-in/memos/tools.js
- live-2d/plugins/built-in/memos/index.test.js
- graphiti_memory/test_entity_anchoring.py
- graphiti_memory/test_graph_query.py
