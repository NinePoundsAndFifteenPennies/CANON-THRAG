# CANON-THRAG 方案与实验设计

来源为用户原始 [note.md](../note.md)，原文保留。本文只维护用户提案、算法边界和验证设计；当前目标与实施进度见 [项目上下文](../PROJECT_CONTEXT.md)，论文事实见 [论文摘要](thrag_paper_brief.md)，实现差异见 [资源审计](resource_audit.md)。

<a id="modules"></a>
## 核心模块

核心限定为同层 Topic/Subtopic 规范化，以及 embedding Top-M 预筛后由原 LLM 选择器多选。Entity 一跳扩展深度不变；预筛主要减少选择提示词和干扰项，本身不减少选择器调用次数。

**结构规范化。**在 `graph_construction → json_to_gexf` 之间加入 `hierarchy_normalize`，从原始 extraction JSON 生成规范化副本和 `canonical_map.json`。只改 subject/object 各自的 Topic/Subtopic 字段。

先规范化 Topic，再规范化 Subtopic。label embedding 只召回同层少量候选，结合实体邻域、关系分布及 Subtopic 的父 Topic 相容性决定合并；冲突或不确定时保留。避免全节点两两比较，也不只靠阈值加 union-find 做传递合并：新增成员须检查整个 cluster。规范标签优先选已有别名，保留旧 ID、aliases、来源和合并记录；撤销依赖原始 JSON/图。用于候选选择的向量包含 canonical label 与 aliases。

本方案不拆分已经合为同一节点的完全同名异义标签。节点减少也不保证候选减少：邻域取并集可能扩大实体和证据范围，增加噪声。

**候选预筛。**规范化后另建 Topic/Subtopic 向量索引；小图可直接做矩阵 cosine。Topic 在全图预筛，Subtopic 仅在当前已选 Topic 的直接邻居内预筛。分别设置 `M_topic/M_subtopic`，M 大于最终选择上限。

在验证集同时检查 baseline 所选节点保留率和 gold supporting chunks/edges 对应的证据覆盖；baseline 的选择结果不是 gold。依据候选召回与提示词成本的折中选 M。合并阈值用 merge-pair 开发集，选择参数用验证集，测试前冻结。

<a id="invariants"></a>
## 不变量与因果归因

原始 Entity 集合、Entity–Entity 三元组、source sentences、chunk IDs 必须一致。直接对照和消融共享原始抽取缓存、数据划分、模型版本、embedding、回答 prompt、评测和检索协议；固定 K1/K2，比较四组：原版、仅规范化、仅预筛、两者结合。

外部基线保留各自建图流程，不强迫共享 TH-RAG 的三元组。论文旧分数只作历史参考，直接对照须在同一环境重跑。原笔记对“严格论文协议”和“可选公开代码协议”的口径存在张力，应按 [项目上下文](../PROJECT_CONTEXT.md)中的最终决策执行；技术差异集中在资源审计。

<a id="validation"></a>
## 验证设计

先验证答案与 chunk 检索质量、效率以及四组消融，再补与贡献主张匹配的规模、问题类型和跨领域实验。论文相应表格与指标定义由 [论文摘要](thrag_paper_brief.md)维护。

| 证据类别 | 需要报告的内容 |
|---|---|
| 答案与检索 | Answer F1/Accuracy、Chunk Recall/NDCG@5 及完整辅助指标；需要 gold evidence→chunk 映射 |
| 规范化正确性 | 人工 merge precision、来源完整率、实体/证据不变量；不能用图压缩率或连通性替代 |
| 选择覆盖 | 候选保留率、触达实体/候选句数、实际 unique chunks、失败案例 |
| 成本 | 主题/子主题选择的 calls/tokens、延迟、实际 context、离线规范化成本 |
| 稳定性与归因 | 两个具体 QA 集的质量–成本结果、配对置信区间；增加 label-only clustering 对照，尽量匹配压缩程度或候选/上下文成本 |

无答案问题单独评估拒答正确率和错误作答率；当前纳入哪些问题见项目上下文。未来若做动态预算，补实际上下文成本匹配的固定预算对照。Top-M 在相近质量下节省提示词可以支持效率收益，不能预先宣称准确率提高或调用数减少。

UltraDomain 的长答案 pairwise 可后置；若主张跨领域长文本或全局理解能力，应补直接 CANON-vs-TH-RAG 对照。它扩展适用范围，不能替代方法区别和因果证据。

<a id="contribution"></a>
## 贡献判断

第一版可用于验证方向，不足以预先认定能支撑投稿。label embedding 聚类和 Top-M 本身的新意有限；原论文的未来工作建议见 [论文局限](thrag_paper_brief.md#paper-limitations)。

潜在贡献在于：结构约束如何比简单标签聚类更少误合并，同时保留证据覆盖、降低成本。需要明确的算法和约束，以及上节所列的质量、成本、正确性和对照证据。若结构法未优于简单聚类，应收窄贡献或调整方案，不能靠额外模块掩盖缺口。这些是研究设计建议，不是会议的官方录用规则。

初步相关工作包括 [CESI](https://arxiv.org/abs/1902.00172) 的 embedding+side information 规范化、[LeanRAG](https://arxiv.org/abs/2508.10391) 的语义聚合/结构检索、[ArchRAG](https://arxiv.org/abs/2502.09891) 的层次社区索引。此处依据仅到摘要，不能断言与 CANON 完全相同；完整创新性对照仍须建立。

## 非核心扩展

来源文本特征、灰区 LLM 合并判定、直接依据向量选择、完整候选列表 fallback、动态 N_T/N_ST、动态 K1/K2、query expansion、cross-encoder、反思及同名异义拆分均不自动进入核心版本，扩展时须逐项消融。

softmax/entropy/margin 是待校准启发式，不是真实概率、拒答判断或问题难度。Cross-encoder 对 query 与候选文本一起打分，无法挽回初召回漏掉的文本；参见 [Sentence Transformers 说明](https://www.sbert.net/examples/cross_encoder/applications/README.html)。
