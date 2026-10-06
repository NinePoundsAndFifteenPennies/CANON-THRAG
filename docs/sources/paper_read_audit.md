# TH-RAG 附录审读核查（只依据论文 PDF）

页码为 PDF 页序；对应印刷页为 37507 + 页序。主要阅读 PDF 9–24 页，补查方法 PDF 3–6、8 页。此文件记录论文声称，不将其当作作者代码已实现的保证。

## 附录 A / E.1：提示词的实质约束

- PDF 13，A.1 / Table 9：UltraDomain 使用较长的多段落回答；仅依据上下文、同用户问题语言，缺证据时承认不知道，Markdown 输出。没有给最大 token 数。
- PDF 14，A.2 / Table 10：HotpotQA / MultiHop-RAG 使用最多 5 词的最简回答；证据不足输出 `Insufficient information.`，判断题只能 `Yes` / `No`。这些约束会直接影响答案指标，复现时须记录使用的提示词版本。
- PDF 15，A.3 / Table 11：一次读取整个 chunk，提取有具体原句依据的事实三元组；允许关系的合理改写，要求用上下文消解三元组中的代词；subject 与 object 各自带 subtopic/main_topic，输出合法 JSON。原句文本作为关系边属性，而不是额外概括后的证据。
- PDF 16，A.4 / Table 12：从允许话题列表精确选择标签，不可创建/改写；选择数量在 min_topics/max_topics 之间；即使无直接相关项也必须选至少 min_topics 个最相关项。列表不足时怎样处理没有明确说明。
- PDF 17，A.5 / Table 13：每个选中话题分别从其 subtopic 列表选择，精确保留标签并按候选列表原顺序输出；无相关项仍选最相关的 min_subtopics，候选不足时允许少于下限。
- PDF 21，E.1 / Table 17：每条记录字段示例为 `triplet: [s,r,o]`、`sentence`、`subject: {subtopic,main_topic}`、`object: {subtopic,main_topic}`。论文示例展示的是单条对象，完整多条输出容器形式须查代码。若干例句以省略号截断，不能据此保证示例 sentence 是逐字原句，也不能仅据截断断言其捏造。
- 附录并未给出 judge 完整提示词、125 个问题生成提示词、错误 JSON 的修复/重试流程。

## 算法与实现口径（PDF 3–6，Algorithms 1/2）

1. 图构建：每个 chunk 一次 LLM 调用提取 triplets + 两端实体各自的话题/子话题；图节点为 topic、subtopic、entity 三种，边为 topic→subtopic、subtopic→entity、entity→entity 关系边，关系边保留来源原句。一个实体/子话题可多归属。Algorithm 1 的“connect s and o to each st”是简写，实际分类应对照 Table 17 的 subject/object 分别注释。
2. 检索：一次调用从所有 topic 中选 NT 个；每个选中 topic 再调用一次选 NST 个 subtopic；收集选中 subtopic 下所有实体，扩展每个实体的 1-hop 邻居，收集相关关系边的句子。不是迭代式多跳推理/路径搜索。
3. 在上述候选句中按 query–sentence 余弦相似度排名，top-K1=30 句为主证据；再取其中 top-K2=5 句所属原始 chunks 扩展上下文。论文摘要性描述说使用 chunk 与 triplet 信息，但 Algorithm 2 明确最终上下文为句子和来源 chunk，未完整说明三元组字符串如何序列化。
4. NT / NST 为话题/子话题选择数，主设置范围为 5–10 / 10–25；不是 K1 / K2。subtopic 范围看提示词与逐话题调用，应为每个选中 topic 的数量限制。
5. 来源 chunk 去重、同一原句被多条关系复用时的去重、top-K2 句共享同一 chunk 时是否补满 5 个不同 chunk，论文未明确。不能把论文的 K2 直接认定为“恰好 5 个不同 chunk”。
6. 所有句子/关系原始位置映射、跨 chunk 边属性合并策略、实体/标签规范化方式、cosine 的工程实现需另查代码。

## Appendix B：环境/共同配置（PDF 18）

- NaiveRAG 与 TH-RAG 用 Faiss 作为向量库；相似度用自实现 cosine，不用 Faiss 内置 L2 / inner product。
- 所有方法固定 chunk=1200 tokens、overlap=100；答案 generation temperature=0、gleaning=0；生成提示词统一，其余 baseline 设置采用各自默认值。
- 包括图构建和答案生成，按需要调用 OpenAI GPT-4o-mini；句子/chunk embeddings 使用 OpenAI small embedding。Appendix B 写 `text-embedding-small-3`，而 Appendix D.3 写正确常用模型名 `text-embedding-3-small`，宜标为论文拼写差异并以代码核实。
- NaiveRAG top-8 chunks；TH-RAG 的 K2=5。作者承认 1200-token chunk 对常用约 300-token 的 NaiveRAG 可能不利，以 8 对 5 个 chunks 补偿，不是严格相同上下文预算。
- 论文未披露 OS/GPU/CPU/RAM、依赖版本、精确 GPT 模型快照、随机种子、QA 样本 ID、延迟重复次数/方差。只报均值/单一结果，未给显著性检验。

## 数据与指标（PDF 5–6、18–19，Table 1 / C.1 / C.4）

| 使用数据 | token | 原始 passages | QA 数 | tokens/passage |
|---|---:|---:|---:|---:|
| Agriculture | 1.9M | 12 | 125 | 158K |
| CS | 2.0M | 10 | 125 | 200K |
| Legal | 4.7M | 94 | 125 | 50K |
| Mix | 602K | 61 | 125 | 9.9K |
| HotpotQA | 1.2M | 9,827 | 1,000 | 122 |
| MultiHop-RAG | 991K | 435 | 1,000 | 2.3K |

- 原 UltraDomain 有 20 个领域，本实验选四个领域/混合集。Table 1 明确每个子集 125 QA（总计 500）；正文/附录的“a total of 125 questions”有表述歧义，不能擅自把四个领域当作总共 125。
- HotpotQA 使用 Distractor；每题 10 段中 8 段不相关。作者随机抽取 1000 QA 及对应 passages 构建实验语料，不是整个 HotpotQA FullWiki。
- MultiHop-RAG 来自英文新闻，跨 2–4 个 documents，问题类型 Inference / Comparison / Temporal / Null；作者同样抽取 1000 QA 对及对应 passages。
- UltraDomain 以 LLM 1-vs-1 judge 比较 Comprehensive / Diversity / Empowerment 三维与 Overall 胜率。完整评价 prompt、judge 的独立精确快照、答案顺序偏差处理及平局处理未在附录披露。
- HotpotQA 与 MultiHop-RAG 均声称遵循 HotpotQA 官方评估：答案 token F1 / Precision / Recall；另定义 Accuracy 为预测答案包含 gold answer（不是明确的严格 exact match）。代码的大小写、标点、别名、空答案与 contains 匹配规则须核实。
- Retrieval 用 gold supporting evidence 所在 chunks 作真值；方法输入生成的特定 chunks/注明来源 chunk IDs 作预测 chunks，计算 Recall、F1、Recall@5、NDCG@5。精确 chunk 映射、同源去重、无证据 Null 题分母/排序、NDCG 相关度定义未披露。GraphRAG-G 在主检索比较中排除，因为 global summaries 覆盖过多 chunks。

## Table 14：全量实验图规模（PDF 19）

| 统计项 | Agri | CS | Legal | Mix | HotpotQA | MultiHop-RAG |
|---|---:|---:|---:|---:|---:|---:|
| 总节点 | 44,588 | 45,921 | 50,162 | 19,806 | 50,256 | 26,250 |
| Topic | 1,568 | 531 | 424 | 401 | 374 | 446 |
| Subtopic | 12,280 | 15,142 | 14,319 | 5,993 | 9,188 | 7,921 |
| Entity | 30,740 | 30,248 | 35,419 | 13,412 | 40,694 | 17,883 |
| 总边 | 76,946 | 76,598 | 94,507 | 31,580 | 87,757 | 42,857 |
| Topic–Subtopic | 18,424 | 20,675 | 19,212 | 7,436 | 12,672 | 9,825 |
| Subtopic–Entity | 35,219 | 34,017 | 41,268 | 14,312 | 43,843 | 19,680 |
| Entity–Entity | 23,303 | 21,906 | 34,027 | 9,832 | 31,242 | 13,352 |

各表列的节点/边子项加总与总量相符。

## D.1 / Table 15：扩大语料（PDF 19–20）

- 合并四个 UltraDomain 子集、HotpotQA、MultiHop-RAG（约 11M tokens）构成大且含更多干扰的语料；仍评估 1000 条 HotpotQA。
- TH-Large 保留原主超参：F1 .631 / P .651 / R .647 / Accuracy .587。
- TH-Large-Tune 仅公开把 max NT 10→20、max NST 25→50：F1 .668 / P .689 / R .683 / Accuracy .614。
- 对照原主 TH-RAG：F1 .671 / P .692 / R .685 / Accuracy .612。因此是增加候选话题/子话题范围可恢复该样本表现的案例，不足以推出普遍规模复杂度结论。

## D.2：语义一致性与收敛（PDF 20）

- 合并语料 topic：单独六集加总 3744 → 合并 2499，少约 33%；subtopic：64843→55832，少约 14%。Table 14 的六集加总与这两个起始数一致。
- 该结果说明标签在六个语料间复用/重合；只有一个合并点，不是随 N 多点观测的收敛曲线，不能证明话题数量渐近有界或整体检索只需 O(NT)。候选实体/边和排序同样有计算成本，且 NT 在文中有选中数/总 topic 数混用倾向。
- 声称 topic:subtopic 稳定在 1:10–1:30，与 Table 14 不完全一致：Agriculture 1:7.83；Legal 1:33.77；其余约 1:14.94、1:17.76、1:24.57、1:28.52。
- 声称 HotpotQA/MultiHop-RAG 的 36 个生成 topics 中 33 个直接匹配 Wikidata schema；对应 Table 14 全图 topics 为 374 / 446，36 的采样范围、评审方式未说明，不能记为全量 91.7% 本体对齐率。
- “冗余检索”段误把多 topic/subtopic 选择称作 K1/K2；方法主文中这应是 NT/NST，K1/K2 指句子数和扩展数。

## D.3 / Table 16：HippoRAG2 补充基线（PDF 19–20）

- 作者仅报告 HotpotQA；NER / triplet extraction 用 GPT-4o-mini，embedding 用 text-embedding-3-small，同 TH-RAG 答案 prompt；HippoRAG2 原有 chunk size 不改，故不遵从主实验所有方法 1200/100 的共同设定。
- HippoRAG2：F1 .653 / P .685 / R .657 / Accuracy .582。
- TH-RAG：F1 .671 / P .692 / R .685 / Accuracy .611。此 .611 与 Tables 3/15 的 .612 有 0.001 差异，表中没有解释。
- 附录声称 HippoRAG 家族在 UltraDomain/MultiHop-RAG 较差，归因其小 passage 设计，但未给对应两个数据集实测表。不要将此记作本论文已完成的实证。

## Table 8 与效率（PDF 8）

- Table 8 是 MultiHop-RAG Accuracy；K1列=5/10/30/50，K2行=1/3/5/10：
  - K2=1：.536 / .565 / .622 / .630
  - K2=3：.662 / .679 / .697 / .702
  - K2=5：.697 / .685 / .720 / .719
  - K2=10：不可用 / .714 / .743 / .726
- 主设置30/5在该表为 .720，Table 3 / Table 7 主结果 .722；可能是精度/运行差异，未解释。最高 .743 是30/10，不能把它记为作者主结果。
- Table 6 TH-RAG：indexing 902 calls / 2.3M tokens；query latency3.54s；context tokens7.4K（caption说包括 prompt + context）。LightRAG 5978 /8M /2.66s /25K；HyperGraphRAG 2772 /20.3M /9.78s /20K；GraphRAG-L4354 /15M /.77s /13.6K。
- 正文称7.4K为HyperGraphRAG13.6K的54%，比较对象错置：13.6K应为GraphRAG-L；相对HyperGraphRAG20K约37%。
- “topic nodes rarely exceeds1000”与Agriculture1568及合并2499并非普遍成立。
- 检索 LLM calls=1+选中topics数量，而非1+整个图库topic节点数；在主设定通常6–11 calls。作者的查询延迟数未披露并行化/重试和测试硬件，不能只由 calls 推出可比性能。

## Limitations / E.2 失败模式（PDF 9、22）

- 作者承认 topic/subtopic 同义、单复数与不同命名导致图膨胀；曾尝试预定义 schema，效果不满意，未来可试 embedding clustering。query expansion 与 context reranking 在本工作有意省略。
- 仅用GPT-4o-mini做主评估；作者称试过更强/更弱LLM改变表现但幅度近似NaiveRAG，因此未展示实验。不能据此确认跨模型鲁棒性。
- E.2展示：缺少关键实体但可借来源chunk expansion找回；也有 Fastjet Tanzania/Fly540 实体缺失而扩展失败；Rockefeller Center的Commercial Complex子话题未选导致关键证据断路。后者直接支持“选择层漏召回”是可诊断问题。
- 源句保留意味着已抽取边有原文依据，并不保证全部原文句子均进入图或全部信息可达；论文的“no semantic information lost”应作为作者强表述，不能写成已证实的性质。Figure 5对实体文本质量不依赖的叙述同样不是保证，检索仍依赖抽取和层级标签质量。
