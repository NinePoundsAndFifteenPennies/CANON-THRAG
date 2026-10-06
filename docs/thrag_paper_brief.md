# TH-RAG 论文精读摘要

用途：后续会话按需查询论文关键事实，替代重复读取整篇 PDF。核验于 2026-10-05；**这里记录论文报告，不代表本地已复现**。用户方案另见 [idea_digest.md](idea_digest.md)，公开实现差异另见 [resource_audit.md](resource_audit.md)。

## 1. 身份与证据

- 全名：TH-RAG: Topic-Based Hierarchical Knowledge Graphs for Robust Multi-hop Reasoning in Graph-based RAG Systems。
- 五位作者：JungHyoun Kim、Soohyeong Kim、Seok Jun Hwang、Jeonghyeon Park、Yong Suk Choi。Zotero 搜索 helper 也列出了四位 proceedings editors，他们不是本文作者。
- ACL 2026 Volume 1: Long Papers，页 37508–37531，24 页；DOI `10.18653/v1/2026.acl-long.1740`；正式 BibTeX key 可用 `kim-etal-2026-th`。
- [ACL 官方条目](https://aclanthology.org/2026.acl-long.1740/)；[正式 PDF](https://aclanthology.org/2026.acl-long.1740.pdf)；[作者代码仓库](https://github.com/Tiabet/TH-RAG)。
- 本次实际精读 Zotero 附件 `TK74E7LL`；父条目 `KVZEBPPB`。本地路径、SHA256、提取方式见 [来源 manifest](sources/thrag_manifest.json)，全文缓存见 `sources/thrag_fulltext.txt`，默认不加载。
- 下文 `pN` 是 PDF 的第 N 页，印刷页为 `37507+N`。正文与 Limitations 为 p1–9，参考文献延续到 p12，附录 p13–24。

## 2. 问题与贡献

LLM 从不同文档抽出的实体三元组经常成为互不相连的小图。路径检索无法沿实体关系覆盖分散的证据；社区摘要/聚类等补图过程又可能增加成本或丢失细节。

TH-RAG 为实体上方增加 Topic/Subtopic 语义层，借这些上层节点跨越实体图的碎片，再按原始证据句检索并恢复局部完整上下文。多跳推理主要由最终 LLM 综合证据完成；它不是增加实体图 traversal 深度的算法，也不显式要求找到完整多跳实体路径。（§1、§3、Figure 1/2，p1–5）

三项机制：同次抽取事实与层级标签；LLM 从显式候选表做层次选择；source-sentence cosine 过滤与来源 chunk 扩展。保留原句能改善事实 grounding，但不能保证 LLM 抽取覆盖所有原文事实。（§3.4、Appendix E.2）

## 3. 建图：数据对象及关系

语料切成 chunks；每个 chunk 一次 LLM 调用，返回多条记录。每条记录包含 `(subject, relation, object)`、对应 source sentence、**subject 各自的** `subtopic/main_topic`、**object 各自的** `subtopic/main_topic`。例如同一关系的两端可以分别归入 Person/Entertainment 与 Record label/Music industry；不能给整条三元组只配一套标签。（§3.1、Algorithm 1，p3–4；Table 11 p15、Table 17 p21）

图的三种节点及三种有向边：

| 对象 | 意义 |
|---|---|
| Topic / Subtopic / Entity | 三类互不混用的节点 |
| Topic → Subtopic | 上层语义包含关系 |
| Subtopic → Entity | 实体所属语义类别 |
| Entity → Entity | 原始事实关系，附 source sentence |

Entity 可连多个 Subtopic，Subtopic 可连多个 Topic，**不是严格单父节点树**。检索需保留 sentence 对应的来源 chunk；具体文件/schema 以代码审计为准。相同标签可复用，语义相同但字符串不同的标签仍可能成为不同节点。

## 4. 查询：论文定义的精确语义

```text
query
  → LLM 从完整 Topic label 列表选 NT 个
  → 每个已选 Topic：LLM 从其完整邻接 Subtopic 列表选 NST 个
  → 合并所选 Subtopics，取得其 Entity
  → 收集 Entity 的 1-hop 邻居及相关关系边/source sentences
  → 计算 query 与候选 source sentence 的 cosine similarity
  → 同一排名的 Top-K1 sentences 作为 primary context
  → primary context 的前 K2 sentences 找回完整来源 chunks
  → primary + extended context 输入回答 LLM
```

依据：§3.2–3.3、Algorithm 2、Figure 2（p3–5）。Algorithm 2 的“all sentences”须结合 §3.3 理解为 traversal 得到的候选证据句。

关键区别：

- `NT/NST` 是 prompt 中的选取数量范围，由 LLM 返回实际数量；不是固定一次只取一个分支。正常路由调用数约 `1 + 实际选择的 Topic 数`，另加最终回答调用；重试等属于实现行为。
- **K1 排的是 source sentence；K2 仍沿用该排名，不计算 query–chunk score。**前 K2 句子可来自同一 chunk，去重后不同 chunk 数可能少于 K2。
- `K2≤K1`。句子和 chunk 同时给生成器，source sentence 不等于抽象三元组字符串。
- 论文语义是候选子图内部 Top-K1；公开 Retriever 的全局预召回再过滤近似行为及不同 chunk 扫描规则须单独记录，不可混为论文定义。

## 5. 论文配置与 prompts

| 项目 | 论文报告 | 原文定位 |
|---|---|---|
| Chunk size / overlap | 1200 / 100 tokens，所有方法统一 | Appendix B，p18 |
| 抽取、回答及所需 LLM | GPT-4o-mini | Appendix B |
| Embedding | OpenAI small embedding；B 写成 `text-embedding-small-3`，D.3 写 `text-embedding-3-small`，前者疑似名称笔误 | p18、p20 |
| 向量存储与打分 | NaiveRAG/TH-RAG 用 FAISS；按 cosine similarity | Appendix B |
| K1 / K2 | 30 / 5 | §4.4，p6 |
| NT / NST | 5–10 / 10–25，由 prompt 范围约束 | §4.4、Tables 12/13 |
| 回答温度 / gleaning | 0 / 0；抽取、路由、judge 温度未明确 | Appendix B |

论文没有给出可直接锁定的 API 模型快照/完整调用时间、随机抽样种子与 1000 个问题 IDs，也没有给出一个完整实验配置 manifest；以上未知项后续复现需补齐。

Prompt 要点（Appendix A，p13–17）：

- 长回答 Table 9：限定检索信息、同问题语言、多段落、证据不足承认未知。
- 短回答 Table 10：用于两个具体 QA 集；最简短答案、最多 5 个词，yes/no 仅输出相应词，信息不足返回指定 abstention 文本。
- 抽取 Table 11：事实有句子依据，结合上下文消解代词，subject/object 分别分类，仅返回合法 JSON。
- Topic Table 12：从给定列表精确选择原标签；不创造标签；即使无明显相关项仍要求选最接近的最小数量。
- Subtopic Table 13：仅从当前 Topic 的列表选择；要求 min/max，列表较短时允许不足 min；保留候选列表原顺序。

## 6. 数据、基线与评测定义

Table 1（p5）报告的 **论文实验子集**，不是完整原始 benchmark：

| 数据集 | Tokens | Passages | QA |
|---|---:|---:|---:|
| Agriculture | 1.9M | 12 | 125 |
| CS | 2.0M | 10 | 125 |
| Legal | 4.7M | 94 | 125 |
| Mix | 602K | 61 | 125 |
| HotpotQA | 1.2M | 9827 | 1000 |
| MultiHop-RAG | 991K | 435 | 1000 |

UltraDomain 为长文本、开放式问题，以上四域各报告 125 QA；HotpotQA 用 Distractor setting（通常每题十个段落，其中两个支持证据），随机取 1000 QA 及其 passages；MultiHop-RAG 为新闻、2–4 文档的多跳问题，类型有 Inference/Comparison/Temporal/Null，也随机取 1000 QA。正文/Table 1 与 Appendix C.1 的“125 questions total”措辞并不完全一致，应以表中的各域数区分，原抽样仍待恢复。（§4.1，p5–6；C.1，p18）

基线：NaiveRAG、GraphRAG-G/L、LightRAG、PathRAG、HyperGraphRAG；HippoRAG2 只在附录做 HotpotQA matched comparison。NaiveRAG 检索 8 chunks，TH-RAG 主设置 K2=5；论文承认统一大 chunk 可能不利于通常使用小 chunk 的 NaiveRAG。（C.3，p18–19）

评测（§4.2、C.4，p6、p19）：

- Answer F1/Precision/Recall 使用 HotpotQA 风格的传统答案评测；**论文 Accuracy 是预测答案包含 gold answer 的判定，不能直接当 exact match**。公开脚本是否完全实现该协议另行核验。
- Retrieval Recall/F1/Recall@5/NDCG@5 的对象是 **chunk IDs**。Gold chunks 是包含 supporting evidence 的 chunks；不是 Topic、Entity 或三元组召回。source-sentence→chunk 的映射影响评测。
- UltraDomain 用 LLM 对两个答案逐题做 pairwise 比较，维度为 Comprehensive、Diversity、Empowerment 和 Overall win rate；没有可比的客观金答案分数。GraphRAG-G 的跨大量 chunks 摘要不纳入 chunk 检索评测。

## 7. 主结果与证据强度

以下均为 **paper-reported**，本地未跑。Table 3（p6）：

| 数据集 / 方法 | Answer F1 | Precision | Recall | Accuracy |
|---|---:|---:|---:|---:|
| MultiHop / TH-RAG | .712 | .711 | .720 | .722 |
| MultiHop / HyperGraphRAG | .526 | .503 | .619 | .621 |
| Hotpot / TH-RAG | .671 | .692 | .685 | .612 |
| Hotpot / HyperGraphRAG | .674 | .703 | .683 | .599 |

| 数据集 / 方法 | Chunk Recall | F1 | Recall@5 | NDCG@5 |
|---|---:|---:|---:|---:|
| MultiHop / TH-RAG | .405 | .271 | .405 | .555 |
| MultiHop / HyperGraphRAG | .426 | .283 | .402 | .460 |
| Hotpot / TH-RAG | .886 | .383 | .886 | .893 |
| Hotpot / HyperGraphRAG | .848 | .382 | .848 | .763 |

结论应限定：TH-RAG 的 MultiHop 答案质量、Hotpot Accuracy 和 NDCG@5 强；**不是所有指标最优**。HyperGraphRAG 的 Hotpot Answer F1 与 MultiHop Chunk Recall/F1 更高。UltraDomain Table 2 Overall 对 HyperGraphRAG 的 win rate 是 Agriculture 52.3%、CS 49.1%、Legal 50.9%、Mix 57.9%，主要是接近/略优而非全面大幅领先。

Table 5（p7）按问题类型给 TH-RAG / HyperGraphRAG：Comparison .625/.538、Temporal .509/.197、Inference .954/.938、Null .786/.664。**表头没有明确 metric**；按上下文推测为 Accuracy，引用时须标“推断/待评测脚本确认”。

### 结构（Table 4/14，p7/p19）

Legal 最大连通分量占比：TH-RAG 99.98%、LightRAG 56.11%；子图数 3 vs 8805。论文没有交代强/弱连通分量计算协议。Table 14 的 Legal TH-RAG：Topic 424、Subtopic 14319、Entity 35419，三类边为 19212/41268/34027，总边 94507。Table 14 给六域节点/边数，但没有为每域提供连通分量指标。

**Table 4 内部矛盾，已视觉核对原表**：LightRAG 总边 24707，小于其 Entity–Entity 边 34027；TH-RAG 的 Entity–Entity 边又写 21906，无法与其余边相加得到 94507。Table 14 的 Legal TH-RAG 三类边合计自洽。保留矛盾标记，不替作者改表；LightRAG 数据须本地重算。

### 效率与消融（Tables 6–8，p8）

- Table 6 在 MultiHop-RAG 上报告 TH-RAG 索引 902 次 LLM calls、2.3M tokens；查询 3.54 秒，Context Token 列 7.4K（caption 说明 token 计数包含 prompt 与 context，不能当纯证据长度）。前两者是整个语料索引总量；querying time 不能认定为含最终回答的端到端时间。GraphRAG-L 查询 .77 秒，比 TH-RAG 更快。
- Table 6 正文将 13.6K 错称为 HyperGraphRAG 对照，表中它属于 GraphRAG-L；HyperGraphRAG 是 20K。成本比较以表格和本地重跑为准。
- Table 7（Acc/F1）：完整 .722/.712；去完整 chunks .580/.576；去 triplet 层信息 .692/.680；去 traversal（全局句子过滤）.624/.620。说明恢复完整上下文最关键，但不能拿它证明 CANON 新模块有效。
- Table 8 是 MultiHop Accuracy 的 K1/K2 sweep：默认 30/5 为 .720，30/10 为该网格最佳 .743，50/10 为 .726。并非预算越大越好；不是跨数据集最优参数；.720 与主表 .722 不视为同一次实验。

### 大图、标签一致性及错误案例（Appendix D/E）

- Table 15（p19–20）：六语料合并约 11M tokens，只测 1000 个 HotpotQA 问题。TH-RAG Main F1/Acc .671/.612；TH-Large .631/.587；将 max(NT)/max(NST) 从 10/25 调为 20/50 后 .668/.614，K1/K2 不变。结果是基本恢复，不能说所有指标完全恢复。
- D.2（p20）称合并语料 Topic 数 3744→2499、Subtopic 64843→55832，支持相同标签一定程度复用。作者对 Wikidata 类别对齐的抽样分析和计数收敛，不是严格的语义合并 precision/recall 评测，也未证明消除了近义重复。
- Table 16：HippoRAG2 vs TH-RAG 的 Hotpot F1 .653/.671、Acc .582/.611；此处 .611 与主表 .612 略不同，属于附录 matched comparison；两表的运行关系及差异原因未说明，不能拼作同一次结果或断言 TH-RAG 独立重跑。D.3 说保持 HippoRAG2 的 chunk size 不变，应与 B 的统一 `1200/100` 主实验设置区别记录。
- E.2（p22）：chunk expansion 有时能补回 KG 缺失的实体证据；但关键实体漏抽仍会失败。另一个失败案例路由遗漏 Commercial Complex 分支，导致无法接入 Rockefeller Center 证据。规范化/预筛未必能解决漏抽，预筛还可能丢掉必要分支。

D.2 的“Topic/Subtopic 比例稳定在 1:10–1:30”与 Table 14 的 Agriculture（约 1:7.83）、Legal（约 1:33.77）不完全一致；“冗余检索”段将选择数量误写为 K1/K2，按主方法应为 NT/NST。论文以 LLM 调用数讨论 O(NT)，不应据此认定整个检索计算与语料规模无关。

## 8. 与 CANON 直接相关的边界

Limitations（p9）**明确承认 Topic/Subtopic normalization 不充分**，举近义/单复数标签，预定义 schema 尝试未满意，建议 embedding-based clustering。论文有这一未来工作建议，结构感知规范化本身的进一步创新性还需后续文献与实验建立，当前不作结论。

完整候选列表 + 多次 LLM 选择提供 Top-M 预筛的直接接口。第一版方案维持原选择器/选择预算、sentence/chunk 排名和生成器，才便于隔离收益。所有新旧方法须先在用户 API 下重跑 TH-RAG；论文旧数字仅作历史参考。

原方法刻意没有 query expansion 或额外 context reranker，仅评测 GPT-4o-mini；换成用户 API 会改变模型条件。baseline 与 CANON 应共享缓存的抽取 JSON 和固定协议；具体实验安排留到复现阶段。

## 9. 定向查阅索引

| 后续问题 | 原文位置 |
|---|---|
| 图 schema、1-hop、K1/K2 | §3.1–3.3、Algorithms 1/2，p3–5 |
| 数据规模、主参数、主结果 | Tables 1–3、§4.4，p5–6 |
| 连通性、类型、效率、消融、预算 | Tables 4–8，p7–8 |
| 作者承认的缺陷 | Limitations，p9 |
| prompts、subject/object 分别分类 | Tables 9–13 p13–17；Table 17 p21 |
| 模型、温度、chunking、数据、指标 | B/C，p18–19 |
| 大图、语义一致性、HippoRAG2 | Tables 14–16、D，p19–20 |
| 成功/失败案例、图示 | E.2–E.4，p22–24 |
