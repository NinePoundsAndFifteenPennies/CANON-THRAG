# CANON-THRAG：用户想法速读

来源：工作区 [note.md](../note.md)。本文件压缩用户方案，事实核验见 [论文摘要](thrag_paper_brief.md)、[资源审计](resource_audit.md)，状态见 [快速上下文](../PROJECT_CONTEXT.md)。原笔记保留不动。

## 目标与边界

以 TH-RAG 为直接基线，探索足以支撑 CCF-C 投稿的独立改进；发表结果不能预先保证。下一阶段先用用户自己的 API 复现核心实验，再加入模块，成功后补实验。本轮只整理论文、想法和持续会话上下文，不启动复现、不定 API/模型和预算。

核心只有两项：①规范化 Topic/Subtopic 的近义标签；②先向量 Top-M 预筛候选，再让原 LLM 选择器从短列表多选。实体一跳扩展深度不变。Top-M 主要减少路由提示词和干扰项，本身不会减少 LLM 调用次数。

## 核心模块

**结构规范化。**复用同一份原始 extraction JSON，在 `graph_construction → json_to_gexf` 之间加入 `hierarchy_normalize`，产出规范化 JSON 副本和 `canonical_map.json`。只改 subject/object 各自的 Topic/Subtopic 字段；不能把整条三元组当成只有一个层次标签。

先规范化 Topic，再规范化 Subtopic。label embedding 只召回同层少量候选，结合实体邻域、关系分布及 Subtopic 的父 Topic 相容性决定合并；有冲突或无法判断则保留。避免全节点两两比较；避免仅用阈值加 union-find 导致传递误合并，新增成员须检查整个 cluster。规范标签优先选已有别名，保留旧 ID、aliases、来源和合并记录；撤销还需原始 JSON/图。路由向量包含 canonical label 与 aliases。

当前方案不拆分已经合成同一节点的完全同名异义标签。节点变少也不保证候选变少：邻域取并集可能扩张实体和证据范围，增加噪声，收益必须实测。

**候选预筛。**规范化后另建 Topic/Subtopic 向量索引，小图可直接矩阵 cosine。Topic 在全图预筛；Subtopic 仅在当前所选 Topic 的直接邻居内预筛。分别设置 `M_topic/M_subtopic`，M 应大于最终选择上限。验证集检查 baseline 所选节点保留率与 gold supporting chunks/edges 对应路由召回；前者不是 gold。用候选召回与 prompt tokens 的折中选择 M。

## 不变量与归因

原始 Entity 集合、Entity–Entity 三元组、source sentences、chunk IDs 必须完全一致。TH-RAG 与 CANON 共享抽取缓存、数据划分、API/模型版本、回答 prompt、评测和检索协议；核心实验固定 K1/K2，比较「原版／仅规范化／仅预筛／两者」四组。记录 merge precision、保留率、来源完整率、触达实体/候选句数、实际 unique chunks、路由 calls/tokens、延迟与回答/检索质量。压缩率和连通性改善不能替代正确性证据。

原论文数字只能作标明来源的历史参考，直接基线和关键消融必须本地重跑。外部方法保留各自建图流程，不强迫共享 TH-RAG 的三元组。回答质量与 chunk 检索质量分开报告；无答案问题单独看拒答正确率和错误作答率。若未来做动态预算，须补实际上下文成本匹配的固定预算对照。

## 暂缓的模块

来源文本特征、灰区 LLM 合并判定、直接路由与完整列表 fallback、动态 N_T/N_ST、动态 K1/K2、query expansion、cross-encoder、反思及同名异义拆分均属后续可选，须逐项消融。softmax/entropy/margin 只是待校准启发式，不是真实概率、拒答判断或问题难度。合并阈值用 merge-pair 开发集；路由参数用验证集，测试前冻结。

## 待核验与未决事项

Table 4 计数矛盾已由论文原表核验；Table 5 指标仍只是上下文推断。公开 Retriever 的预召回、fallback、unique chunks 与默认参数以资源审计记录的上游 commit 为准；复现前确认参数与依赖。论文忠实协议与公开代码协议尚需择一并固定；笔记后文要求严格原文，与前文允许二选一有张力。K1 是证据句预算；论文语义下 K2 沿同一排名回取 chunk，不是 chunk 打分，实际数量可能小于 K2。API、具体数据版本、协议、M、阈值与实现状态均未因这份想法笔记而确定。

## 当前有效性验证的优先顺序（2026-10-06，建议）

优先对照原论文 Table 3 的答案与 chunk 检索质量、Table 6 的效率，并按 Table 7 的消融思路新增「TH-RAG／仅规范化／仅预筛／两者」四组。使用用户自己的 LLM API 时，以同一环境重跑的 TH-RAG 为直接对照，论文旧分数仅为参考；四组共享原始抽取 JSON、embedding、回答模型/prompt、数据与检索协议，核心固定 K1/K2。优先看 Answer F1/Accuracy、Chunk Recall/NDCG@5，保留其他完整指标；同时记 routing tokens/calls、延迟、实际 context、离线规范化成本。检索指标需要 gold supporting evidence→chunk 映射，当前尚未备齐。

Table 14（以及有原表矛盾的 Table 4）辅助分析节点压缩与图结构；需 merge precision 和实体/证据保留检查，不能用节点减少代替任务有效性。Table 15 大图、Table 5 问题类型及 Table 2 主观评测留待核心验证后。Top-M 若在相近质量下节省路由 tokens，支持效率收益；不能预先宣称调用数减少或质量提高。

Cross-encoder 把 query 与候选文本一起输入模型打相关性分，常用于向量初召回后的重排，不能挽回初召回遗漏的文本。参见 [Sentence Transformers 官方说明](https://www.sbert.net/examples/cross_encoder/applications/README.html)。当前保持后续可选，先隔离验证规范化和 Top-M。

## 投稿充分性的客观评估（2026-10-06，尚无实验）

Table 2 的 UltraDomain 长答案 pairwise 评测可后置；若最终声称跨领域长文本/全局理解，则应补直接 CANON-vs-TH-RAG 评测。它扩展适用范围，不能替代方法区别和因果证据。

当前第一版足以启动方向验证，不能据此认定已足够 CCF-C 投稿。复现/适配有工程工作量，但 label embedding 聚类与 Top-M 本身新意较弱；TH-RAG Limitations 也已提出 embedding clustering。已有 [CESI](https://arxiv.org/abs/1902.00172) 探索 embedding+side information 的知识库规范化，[LeanRAG](https://arxiv.org/abs/2508.10391) 与 [ArchRAG](https://arxiv.org/abs/2502.09891) 分别涉及语义聚合/结构检索及层次社区索引。仅初步读取这些摘要，不断言它们与 CANON 完全相同；完整创新性对照尚未完成。

潜在核心贡献：结构约束如何比简单标签聚类更少误合并，并保留路由证据覆盖、降低成本。投稿前需明确的算法/约束；除四组主消融，还需 label-only clustering 对照，尽可能匹配压缩程度/候选或上下文成本；人工 merge precision、实体/证据不变量、gold chunk recall 和失败案例；在两个具体 QA 集上的稳定质量–成本收益及配对置信区间；合理强基线，以及与主张匹配的规模或标签噪声分析。它们是本项目评估标准建议，不是某会议官方录用规则。若结构法没有超过简单聚类，应收窄贡献或调整方案，不能用额外 reranker/反思模块掩盖缺口。
