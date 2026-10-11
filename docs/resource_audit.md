# TH-RAG 资源与实现差异

本文件负责公开资源范围，以及论文、上游代码和本地适配之间的语义差异。配置项的实际作用见 [配置参考](config_reference.md)，默认值见 [配置模板](../thrag/.env.example)，命令与路径布局见 [服务器说明](../thrag/SETUP_GUIDE.md)，验证结果与待办只在 [项目上下文](../PROJECT_CONTEXT.md) 维护。

## 来源与能力范围

上游仓库、固定 commit、导入日期及许可证以 [UPSTREAM.md](../thrag/UPSTREAM.md) 为准；历史获取证据见 [来源 manifest](sources/upstream_web_manifest.json)，其中旧 runtime 字段是当时快照，不代表当前配置。代码入口与阶段对应关系见 [代码导航](../thrag/README.md#pipeline-stages)。

仓库提供图构建、FAISS 证据检索、短答/长答和答案评测，以及两条 QA 的最小示例。正式论文子集、问题 IDs、gold evidence→chunk 映射及预建图没有随上游发布。数据准备脚本的转换范围见 [脚本说明](../scripts/README.md)。

已有答案 F1 和 UltraDomain 风格 pairwise judge；完整 chunk 检索指标、端到端成本采集、外部基线和 CANON 消融编排不属于已完成的上游能力。哪些工作已补齐，以项目上下文为准。

<a id="protocol"></a>
## 参数与实验协议

[论文报告的配置](thrag_paper_brief.md)与[本地模板](../thrag/.env.example)分别代表两个设置，不能混称为同一次复现。聊天模型、embedding、tokenizer 和部分 chunk/K 参数已有本地适配；实际生效条件由[配置参考](config_reference.md)解释。

复现实验必须注明采用论文语义还是公开代码语义，并固定数据、模型和检索协议。该选择及实验进度不在本文重复维护。

<a id="extraction"></a>
## 抽取数量与本地适配

论文提示词要求见 [论文摘要的 prompts 部分](thrag_paper_brief.md)。上游固定版本的 [抽取提示词](https://github.com/Tiabet/TH-RAG/blob/6543fc2a8f5aadecd3032352d7d31dcd5791ec73/prompt/extract_graph.py)未规定每块 20 条上限。

本地 commit `336eaa484efa58ffd301d06a98992cc0e07c6c5b` 曾加入“最多 20 条、优先重要事实”。随后按用户授权在 [抽取提示词](../thrag/prompt/extract_graph.py)和 [system prompt](../thrag/index/graph_construction.py)两处取消，恢复要求抽取全部有原句依据的非重复事实。解析器没有按条数裁剪列表；输出预算与截断处理见 [生成参数](config_reference.md#generation)。

抽取记录的三元组、证据句和两端层级标签定义由 [论文图 schema](thrag_paper_brief.md)维护，不将完整图的节点或边数量当作抽取条数。

<a id="retrieval"></a>
## 候选过滤、回退与 K2

- 主题和子主题选择失败后，取对应候选列表开头的若干项。数量限制的实际生效方式见 [选择参数](config_reference.md#selection)。
- 有 entity filter 时，FAISS 先取全局 `K1 × OVERRETRIEVE_FACTOR`，再保留任一端点属于 Entity 集合的边；这不同于直接在候选子图内部排名，返回数可能少于 K1。
- 带 entity filter 的检索没有结果时，回退到全图搜索。
- K2 沿返回边的排名扫描并去重恢复 chunk，直到得到 K2 个不同 chunk 或候选耗尽。论文 K2 定义见 [论文查询流程](thrag_paper_brief.md)，两种扫描口径不能混用。
- 归一化向量使用 FAISS `IndexFlatIP`，对非零向量等价于 cosine 排名；与论文附录描述的实现方式不同。

实现依据：[`Retriever.py`](../thrag/generate/Retriever.py)、[`edge_embedding.py`](../thrag/index/edge_embedding.py)。

## 来源映射

`sentence -> chunk_id` 使用单值字典，重复证据句可能覆盖来源。Edge record 又按句子全局去重，只保留首次遇到的关系记录。这会影响来源追踪与 gold chunk 对齐；CANON 的保留约束见 [方案定义](idea_digest.md)。

<a id="evaluation"></a>
## 答案评测口径

[`judge_F1.py`](../thrag/evaluate/judge_F1.py)输出 normalized exact match、F1、precision、recall 和 normalized gold substring accuracy；论文指标定义另见 [论文评测部分](thrag_paper_brief.md)，不可把 accuracy 当作 exact match。

`missing/compared/total` 单独统计。缺失或 `[Error]` 预测不进入当前脚本的均值分母，因此模型失败率必须同时报告。预测和 gold 按 query 建字典，重复 query 会覆盖。

[`judge_Ultradomain.py`](../thrag/evaluate/judge_Ultradomain.py)对两个答案文件进行逐题 pairwise LLM 比较；其模型与参数含义见 [配置参考](config_reference.md)，操作命令见 [运行说明](../thrag/SETUP_GUIDE.md#run)。
