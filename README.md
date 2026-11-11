# atom-mem

**一个可独立部署的角色记忆服务：单进程、单端口（8003），自动召回走向量，图检索只在工具调用时做 PPR。**

atom-mem 把原工程里的长期记忆核心拆成了一个独立目录：后端、管理 WebUI、启动脚本和数据目录都在这里，不再依赖 `live-2d`、Flask 8004 或原仓库的前端构建流程。

- WebUI / API / OpenAPI 文档：`http://127.0.0.1:8003/`
- 现有客户端协议：`/graphiti/*`（原路径和请求体保持不变）
- 存储：SQLite 权威库 + `topology.json` 图 + Neo4j 向量/身份投影 + 温层 JSONL
- 默认自动召回：向量匹配；**不会**在后台偷偷跑图扩散
- 图检索增强：只有模型显式调用 `operator="ppr"` 的多跳工具时，才做 Personalized PageRank

---

## 1. 项目介绍

### 1.1 它解决什么问题

角色聊天里真正难的不是“存一段文本”，而是：

1. 每轮对话都要低延迟地找到相关记忆；
2. 近期原文不能丢，否则“上次说的那个”会失忆；
3. 旧对话要能沉淀成实体、事实和关系，支持“谁和谁是什么关系”；
4. 多跳、交集、路径这类结构问题，扁平 top-k 天然答不好；
5. 以上能力不能靠一个每轮都跑的图检索拖慢普通聊天。

atom-mem 的取舍是：**自动召回以向量为主，图作为工具按需调用**。

### 1.2 三层记忆

| 层 | 内容 | 生命周期 | 召回方式 |
| --- | --- | --- | --- |
| Live Context | 当前会话最近几轮 | 会话内，2 分钟检查点裁剪 | 直接在工作记忆里 |
| Warm Layer | 角色级完整近期正文，按 7 天 TTL 保留 | 角色级，不随 session 关闭而删除 | **RAW-RAG 风格**：256 token 窗口、dense top-128、rerank top-15 |
| Cold Layer | 实体、事实、关系、谓词、hook | 发布后长期保留 | 向量 + 事实优先 + 有界扩展；PPR 只走工具 |

### 1.3 存储形式

所有运行态都在包根目录的 `data/` 下，升级程序时不会覆盖：

```text
atom-mem/
├─ app/memos/                       # 代码（后端 + vendored graphiti_core）
├─ webui/                           # 已构建的 WebUI 静态资源
├─ config/
│  ├─ graphiti_config.json.example  # 模板，密钥为空
│  └─ graphiti_config.json          # 本地实际配置（.gitignore，不提交）
├─ data/                            # 首次启动生成，不提交
│  └─ graphiti/
│     ├─ memory.sqlite3             # 权威库：角色/批次/发布/工作记忆/审计
│     ├─ shared/topology.json       # 共享节点、关系、角色白名单
│     ├─ warm/<role_uuid>.jsonl     # 温层正文（append-only）
│     ├─ warm/<role_uuid>.vec.jsonl # 温层向量副文件
│     ├─ warm/<role_uuid>.state.json# 批次与抽取状态
│     └─ characters/<role_uuid>/    # 角色私有认知/提示词/备份
└─ 4.MEMOS-API.bat / 5.MEMOS-WebUI.bat / start_memos.ps1
```

分工：

- **SQLite** 是在线权威存储。角色、批次、发布 Saga、工作记忆、审计先落这里。
- **`topology.json`** 是可读的图投影：节点、关系、谓词、`has_fact`、角色白名单。图查询工具直接读它，不依赖 Neo4j 在线。
- **Neo4j** 承担向量索引、身份节点和事件投影；Graphiti Core 的写入/去重也走它。Neo4j 挂了，图查询和 SQLite 仍可读，但语义向量召回会报组件错误。
- **温层 JSONL** 是正文的 append-only 日志；向量是缓存，缺失时自动回退词法，不会把正文弄丢。

### 1.4 工作流程

```text
用户/AI 一轮对话
   │
   ├─ POST /graphiti/sessions/turn
   │     ├─ 写工作记忆
   │     └─ 立刻把完整回合追加到温层（正文先落盘）
   │
   ├─ 自动召回（每轮，低延迟）
   │     ├─ 长期层：向量 top-k → 事实优先/实体锚定 → 有界扩展 → 门槛 → 预算
   │     └─ 温层：RAW-RAG 原始窗口 → dense → rerank → 独立门槛
   │     （这一路没有 PPR）
   │
   ├─ 模型需要多实体/多跳时，显式调用 memos_graph_query
   │     └─ POST /graphiti/query  operator="ppr"
   │           └─ 只在这次调用里做 PPR，返回实体质量 + 事实证据
   │
   └─ 批次空闲/7 天 TTL/手动 checkpoint
         └─ 冻结批次 → 抽取实体/事实/关系 → 发布 → 温层正文按批次逐出
```

关键点：

- **温层写入是同步的**：回合结束时正文已经 durable，不依赖下一次检查点。
- **封批 ≠ 抽取**：`seal` 只定义批次边界；抽取由 TTL 巡检或显式 checkpoint 触发。
- **发布成功前不删正文**：抽取失败时正文留在温层，批次标记可重试。
- **自动召回不跑 PPR**：普通单跳问题不为图扩散付延迟；图只在工具调用时付费。

### 1.5 检索策略

#### 自动召回：向量为主

长期层保留原有链路：查询 embedding → 候选召回 → 事实优先/实体锚定 → 有界扩展 → 门槛 → token 预算。这里没有新增 PPR 通道，也没有把 `core_search_enabled` 打开。

温层换成 RAW-RAG 的取数形状：

- 窗口：256 token，32 token overlap，按完整回合拼接，不把一句话切成两半；
- dense：top-128；
- rerank：top-15（配置了 rerank 适配器时；失败自动退回 dense 顺序）；
- 准入仍用温层自己的相似度地板和层内相对地板，rerank 只改排序，不放宽门槛。

> 与论文原版的一处工程差异：窗口向量用已有回合向量的 token 加权均值，而不是每次检索重新嵌入整段历史。窗口文本仍是原始正文；这样每轮只多一次 query embedding，不会把温层检索变成 N 次云端调用。

#### 图工具：只在调用时 PPR

`POST /graphiti/query` 新增 `operator="ppr"`：

- 种子：查询解析出的白名单实体；
- 图：只含该角色可见的实体/关系边，事实不作为中转点；
- 传播：damping 0.85，最多 30 次迭代，纯本地计算，不调用 LLM；
- 输出：`evidence.ppr_nodes`（实体质量）和 `evidence.ppr_facts`（按父实体质量排序的事实），带 `ppr_score`；
- 诊断：`diagnostics.ppr` 记录耗时、节点数、种子和事实候选数。

事实只做终点：先传播实体，再按父实体质量给事实打分。这保证一条长事实不会变成两个无关实体之间的桥。

---

## 2. 使用说明

### 2.1 环境要求

| 组件 | 要求 | 本次验证 |
| --- | --- | --- |
| Python | **3.11 及以上** | 3.14.6 |
| Neo4j | **5.x**（Bolt 7687） | Neo4j Community 5.26.20 |
| 操作系统 | Windows 10/11（启动脚本为 bat/ps1） | Windows |
| 模型 | OpenAI 兼容的 chat + embedding；可选 rerank | DashScope `text-embedding-v4` / `qwen3-rerank`，DeepSeek chat |

Python 依赖见 [`requirements.txt`](requirements.txt)。不需要 Flask，WebUI 由 FastAPI 同源提供。

### 2.2 安装

```bat
cd /d E:\atom-mem
python -m pip install -r requirements.txt
```

### 2.3 准备 Neo4j

1. 安装 Neo4j 5.x（Community 即可），启动 Bolt，默认 `bolt://127.0.0.1:7687`。
2. 设置 `neo4j` 用户密码；不需要预建向量索引，服务首次使用时会创建。
3. 如果希望服务自动拉起本机 Neo4j，把 `config/graphiti_config.json` 的 `database.neo4j_home` 指向 Neo4j 安装目录；否则留空，手工启动 Neo4j。

### 2.4 准备配置

```bat
copy config\graphiti_config.json.example config\graphiti_config.json
```

然后编辑 `config/graphiti_config.json`：

| 字段 | 说明 |
| --- | --- |
| `database.neo4j_uri` | 例如 `bolt://127.0.0.1:7687` |
| `database.neo4j_user` / `neo4j_password` | Neo4j 凭据 |
| `embedding.base_url` / `model` / `api_key` | 1024 维 embedding；维度必须与 `embedding.dimension`、`vector.dimension` 一致 |
| `chat.base_url` / `model` / `api_key` | 抽取和日记使用的 OpenAI 兼容 chat |
| `rerank.base_url` / `model` / `api_key` | 可选；温层 rerank 使用它 |
| `retrieval.warm_rag_enabled` | 默认 `true` |
| `retrieval.warm_rag_chunk_tokens` | 默认 `256` |
| `retrieval.warm_rag_chunk_overlap` | 默认 `32` |
| `retrieval.warm_rag_dense_top_k` | 默认 `128` |
| `retrieval.warm_rag_rerank_top_k` | 默认 `15` |

**不要把 `config/graphiti_config.json` 提交到 git。** 仓库只保留密钥为空的 `.example`。

### 2.5 启动

双击 [`4.MEMOS-API.bat`](4.MEMOS-API.bat)，或：

```bat
4.MEMOS-API.bat
```

启动后：

- WebUI：`http://127.0.0.1:8003/`
- OpenAPI：`http://127.0.0.1:8003/docs`
- 健康检查：`http://127.0.0.1:8003/health`

[`5.MEMOS-WebUI.bat`](5.MEMOS-WebUI.bat) 只负责打开控制台；服务未启动时会自动拉起 4 号脚本。停止服务用 [`stop.bat`](stop.bat)。

### 2.6 安装后的必要操作

1. **建角色**：打开 WebUI，创建一个角色，记下 `role_uuid`；所有记忆 API 都按角色隔离。
2. **配模型**：在 WebUI 的模型设置里确认 embedding 维度和 chat 可用；用“测试连接”验证 Neo4j / 模型。
3. **写一轮对话**：

```http
POST /graphiti/sessions/turn
Content-Type: application/json

{
  "role_uuid": "role_xxx",
  "conversation_id": "demo-001",
  "user_content": "我养了一只橘猫，叫大橘。",
  "assistant_content": "记住了，大橘是只黏人的橘猫。",
  "complete": true
}
```

4. **自动召回**：

```http
POST /graphiti/search
Content-Type: application/json

{
  "query": "大橘是什么猫？",
  "role_uuid": "role_xxx",
  "top_k": 5
}
```

5. **多跳工具（只有这里才跑 PPR）**：

```http
POST /graphiti/query
Content-Type: application/json

{
  "query": "Alice 通过朋友和谁有间接关联？",
  "role_uuid": "role_xxx",
  "operator": "ppr",
  "anchors": ["Alice"],
  "top_k": 12
}
```

响应里看 `evidence.ppr_nodes` 和 `evidence.ppr_facts`；`diagnostics.ppr.elapsed_ms` 是这次扩散的本地耗时。

6. **封批与抽取**：正常对话会按空闲时间封批；也可以在 WebUI 的“温层 / 批次”页手动 seal/checkpoint。封批后正文仍在温层，抽取成功并发布后才逐出。

### 2.7 给模型接入的工具

把图查询工具描述为“多实体、多跳、交集、路径、计数”专用，并把 `operator` 枚举加上 `ppr`。示例：

```json
{
  "name": "memos_graph_query",
  "description": "多实体/多跳/交集/路径/计数时调用；普通单事实问题不要调用。",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {"type": "string"},
      "operator": {
        "type": "string",
        "enum": ["facts", "list", "union", "intersection", "path", "count", "ppr"]
      },
      "anchors": {"type": "array", "items": {"type": "string"}},
      "top_k": {"type": "integer"}
    },
    "required": ["query"]
  }
}
```

`ppr` 不要求模型每次都用；它的价值是当模型明确需要“沿关系找间接证据”时，有一个可控、可解释、按次付费的算子。

---

## 3. 测试结果

完整报告和逐题产物在 [`benchmarks/`](benchmarks/)：

- [`locomo-equal-token.md`](benchmarks/locomo-equal-token.md)：LoCoMo conv0–4，999 题，8 档 token，四系统同 reader/judge。
- [`chinese-chatbot-corpus.md`](benchmarks/chinese-chatbot-corpus.md)：中文闲聊 + 字幕噪声，112 回合 / 74 查询。
- [`graph-boost-negative-result.md`](benchmarks/graph-boost-negative-result.md)：为什么自动图增强没有收益。
- [`graph-tool-accuracy.md`](benchmarks/graph-tool-accuracy.md)：图工具从 3/10 到 10/10。

### 3.1 等 token 下相对 Mem0 的准确率

LoCoMo conv0–4，999 题；同一 reader、同一 judge、同一 `cl100k_base` 打包，按实际召回 token 对齐。下表只取 64–2048 档，因为 4096/8192 档 Mem0 会用到 8.1k token，已经不是等预算。

| 预算 | atom-mem | Mem0 | 差值 | McNemar |
| ---: | ---: | ---: | ---: | --- |
| 64 | **54.0%** | 43.4% | **+10.6 pp** | p<0.0001 |
| 128 | **61.2%** | 49.6% | **+11.6 pp** | p<0.0001 |
| 256 | **65.5%** | 54.8% | **+10.7 pp** | p<0.0001 |
| 512 | **67.8%** | 59.2% | **+8.6 pp** | p<0.0001 |
| 1024 | **71.6%** | 61.8% | **+9.8 pp** | p<0.0001 |
| 2048 | **72.1%** | 64.5% | **+7.6 pp** | p<0.0001 |

在等 token 区间内，优势稳定在 **+7.6 ~ +11.6 pp**；短预算（64/128）优势最大，因为 atom-mem 返回的是短事实而不是整段原文，同样的预算能装下更多有效证据。

补充：这套系统不是“全面强于原生 MemOS”。在中文语料上，原生 MemOS 2.0 的严格召回@3 是 82.4%，atom-mem 是 77.9%；但 atom-mem 的 MRR 更高（0.948 对 0.908）、查询更快（0.306 s 对 0.585 s）、top-3 更短（43 字对 125 字）。这是“精度/延迟/上下文密度”的取舍，不是单指标碾压。

### 3.2 为什么不用纯 GraphRAG

“纯 GraphRAG”指：所有召回都先构图、再从图扩展，自动链路依赖图而不是向量。当前系统明确不这么做，原因是有实测：

| 观察 | 数据 | 含义 |
| --- | --- | --- |
| Graphiti Core 混合图检索直接打开 | 候选 20 条，**0 条 admitted**；查询 **4.1–4.6 s** | 它的结果没有标定分数，网关全丢；延迟却是自动召回的 10 倍以上 |
| LoCoMo 单跳 418 题，BM25→BFS 图增强 | gold@30 **52.4% → 52.4%**，MRR **0.309 → 0.283** | 排序净变差；图扩展没有带回 gold |
| 同上，谓词→DFS | gold@30 **52.4% → 52.4%**，MRR **0.309 → 0.299** | 同样打平/变差 |
| 端到端 +2.4~2.6 pp | 随机事实对照同样 +5.3 pp；同 prompt 复测 100 题翻转 6 题 | 涨幅来自上下文扰动/测量噪声，不能归因于图 |
| 图结构本身 | **51.4% 的事实只挂在两个说话人 hub 上**；基线漏掉的 199 道单跳题里只有 **7 道**在 BM25 种子 2 跳内 | 大量事实从实体出发根本到不了，纯图召回会系统性漏 |
| 建图成本 | 112 回合抽取 **713.7 s**，74 facts / 39 父实体，285 节点 / 394 边 | 冷启动和增量成本都高，不适合每轮自动跑 |

所以纯 GraphRAG 的问题不是“图没用”，而是：

1. **自动链路付不起**：普通单跳问题每轮多 4 秒，收益却是 0 admitted。
2. **图覆盖不全**：一半事实挂在 hub 上，图路径到不了；向量反而能直接命中原文。
3. **图擅长的是结构问题**：多实体、交集、路径、计数。把它做成工具，收益是 3/10 → 10/10；把它做成默认召回，收益是负的。

atom-mem 因此保留向量自动召回，把 PPR 限制在 `operator="ppr"` 的工具调用里。这样普通聊天不为图付费，模型需要多跳时再显式付费。

### 3.3 其它现有结果

| 数据集 | 结果 | 对照 |
| --- | --- | --- |
| 中文语料（112 回合 / 74 查询） | 严格@3 **77.9%**，MRR **0.948**，avg **0.306 s**，top-3 **43 字** | 原生 MemOS 2.0：82.4% / 0.908 / 0.585 s / 125 字 |
| 蓝色大肥鱼（合成多跳） | 严格@3 **68.4%**，宽松@3 **94.7%**，多跳 **6/8** | 原生 MemOS 逐轮写入：63.2% / 100% / 4/8；整批写入 73.7%，但 top-3 上下文是 4.5 倍 |
| Douban MTRS | 严格@3 **13.3%**，top-6 **21.7%**，avg **0.362 s**，top-3 **64 字** | 原生 MemOS：13.3% / 16.7% / 0.609 s / 178 字 |
| 图谱查询工具（合成 10 题） | 不开工具 **3/10**；开工具 **10/10**（9/10 主动调用） | 说明图算子的价值在结构题，不在自动召回 |

> 口径提醒：不同数据集的难度和判定不同，数字不能横向相加；LoCoMo 的 999 题表是唯一一组按等 token 统一打包的结果。以上均为单机、指定模型/版本下的实测，不是排行榜。

---

## 4. 目录与实现说明

| 路径 | 作用 |
| --- | --- |
| `app/memos/memos_system/api/graphiti_server.py` | 单进程入口；挂 `/graphiti/*`、`/health`、`/docs` 和 WebUI 静态站 |
| `app/memos/memos_system/api/graphiti_routes.py` | 原 88 条 `/graphiti` 路由，路径和请求体保持不变 |
| `app/memos/memos_system/graphiti_memory/warm_store.py` | 温层正文、批次、TTL、向量回退、RAW-RAG 检索入口 |
| `app/memos/memos_system/graphiti_memory/warm_rag.py` | 256/32 窗口、dense top-128、rerank top-15 |
| `app/memos/memos_system/graphiti_memory/graph_query.py` | `facts/list/union/intersection/path/count/ppr` 工具算子 |
| `app/memos/memos_system/graphiti_memory/runtime.py` | 配置加载、模型适配器、Neo4j 生命周期 |
| `app/memos/memos_system/graphiti_memory/paths.py` | 用 `ATOM_MEM_HOME` 把 `data/`、`config/` 固定在包根目录 |
| `webui/` | 构建后的 Vue 管理台；同源加载，不再需要 8004 |
| `tests/` | 离线测试：PPR 只在工具里跑、自动算子不跑 PPR、温层 RAW-RAG 形状 |

启动脚本会设置 `ATOM_MEM_HOME` 和 `PYTHONPATH`，因此数据不会写进代码目录；`python -E` 会忽略这些变量，脚本刻意没有使用它。

## 5. 安全与提交边界

- `config/graphiti_config.json`、`data/`、日志、SQLite、向量文件全部在 [`.gitignore`](.gitignore) 中。
- 仓库只提交 `config/graphiti_config.json.example`，其中 API key 和 Neo4j 密码为空。
- 服务默认只监听 `127.0.0.1`，不是公网服务；管理接口沿用 loopback + session token。
- 不要用真实密钥覆盖 `.example` 后再提交；如果曾经提交过，立即轮换密钥。

## 6. 已知限制

1. **抽取仍是瓶颈**：中文语料的严格召回失分主要来自抽取没有生成对应 fact，不是向量排序。
2. **PPR 依赖实体链接**：查询没有点名白名单实体时，工具只能返回候选实体建议，不会瞎扩散。
3. **温层窗口向量是近似**：用回合向量加权均值代替逐窗口重嵌入，换来每轮一次 embedding；如果追求论文原版的逐窗口向量，需要额外建索引任务。
4. **Neo4j 仍是外部依赖**：本包不内置数据库；SQLite-only 向量后端不在本轮范围。
5. **8192 档不是等预算**：Mem0 会用满 8.1k，atom-mem 在 3.9k 左右饱和；比较时只看 64–2048。

## 7. 复现

```bat
python -m pytest tests -q -p no:cacheprovider
```

评测脚本和逐题产物保留在原工程 `docs/reports/` 与 `_rawrag_bench/`；本包 `benchmarks/` 里是结果快照，不包含 API key、角色数据或原始私有对话。
