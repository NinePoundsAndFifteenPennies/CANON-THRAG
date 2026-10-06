# TH-RAG 公开资源与实现核验

核验日期：2026-10-06。作者 [公开仓库](https://github.com/Tiabet/TH-RAG) 已下载到 `thrag/`，上游 commit 为 `6543fc2a8f5aadecd3032352d7d31dcd5791ec73`，原许可证为 Apache-2.0。源码由工作区根 Git 仓库统一管理，来源见 `thrag/UPSTREAM.md`。尚未安装依赖或运行实验。

## 可用资源的确切范围

[README](https://github.com/Tiabet/TH-RAG/blob/main/README.md) 确认提供建图、FAISS 索引、短/长答案生成和答案/pairwise 评测 pipeline，要求 Python 3.10+ 与 API key；许可为 Apache-2.0。输入为每个数据目录的 `contexts.txt` 和 `qa.json`，图、KV、向量索引及回答由 pipeline 生成。

本地源码只有 `data/test_dataset/` 的两条示例 QA 和 `contexts_example.txt`，缺 README/pipeline 所需的 `contexts.txt`，原样 quick start 会缺输入。仓库未附论文六语料、1000-QA 抽样 IDs、gold chunk 映射、预建图/embedding/答案结果或本地模型权重；其他发布渠道仍可后续查找。论文通过 API 调用模型。依赖采用版本下限而非锁定环境。

## 文件入口

| 环节 | 仓库路径 |
|---|---|
| 配置/编排 | `config.py`、`pipeline.py` |
| 抽取/图转换 | `index/graph_construction.py`、`index/json_to_gexf.py` |
| 证据向量索引 | `index/edge_embedding.py` |
| 层次选择 | `index/topic_choice.py`、`index/subtopic_choice.py`、`prompt/` |
| 检索/答案 | `generate/Retriever.py`、`generate/answer_generation_short.py`、`generate/answer_generation_long.py` |
| 评测 | `evaluate/judge_F1.py`、`evaluate/judge_Ultradomain.py` |

这是导航图，只有下文引用的文件本次逐行读取；未读的建图/生成文件不据此宣称行为已验证。

## 论文与当前网页代码的差异

**默认参数。**[config.py](../thrag/config.py) 与 [.env.example](../thrag/.env.example) 给 `MAX_TOKENS/OVERLAP=3000/300`、`TOP_K1/TOP_K2=50/10`、long K 为 25/5、`ANSWER_TEMPERATURE=.3`。主建图入口实际使用 `MAX_TOKENS/OVERLAP`，`ALT_MAX_TOKENS/ALT_OVERLAP` 仅被配置类读取，未用于主流程。短答案生成与三元组抽取温度硬编码为 0；`ANSWER_TEMPERATURE` 用于层次选择和长答案，`EVAL_TEMPERATURE` 用于 pairwise judge。不能把默认环境文件当成论文完整配置。Embedding 默认名为 `text-embedding-3-small`；NT/NST min/max 为 5/10 和 10/25；overretrieve factor 为 5。

**候选先全局召回再过滤。**[edge_embedding.py](https://raw.githubusercontent.com/Tiabet/TH-RAG/main/index/edge_embedding.py) 的 `search`（约 149–192 行）在存在 Entity filter 时，先全局取 `K1×factor`，再保留任一端点在 Entity 集合内的边，最多返回 K1；因此可能不足 K1，也不保证等于候选子图内部 exact Top-K1。代码归一化向量后用 FAISS `IndexFlatIP`，数学上是非零向量的 cosine 排名；与 Appendix B 声称“不用 FAISS inner product、自实现 cosine”的工程描述不同。

**fallback 与 K2。**[Retriever.py](https://raw.githubusercontent.com/Tiabet/TH-RAG/main/generate/Retriever.py) 的 `retrieve`（约 101–145 行）：Entity filter 空时直接无过滤；有 filter 但无结果时再搜全图。K2 沿返回的 edges 扫描，跳过重复 chunk，直到取得 K2 个不同 chunk 或候选耗尽，可能越过第 K2 句。论文 Algorithm 2 则从前 K2 句恢复 chunks；不能把两者混作同一协议。`_collect_entity_filter` 并行进行各 Topic 的 Subtopic 选择。

**选择数量并非强制达到 min。**[topic_choice.py](https://raw.githubusercontent.com/Tiabet/TH-RAG/main/index/topic_choice.py)（约 53–96 行）和 [subtopic_choice.py](https://raw.githubusercontent.com/Tiabet/TH-RAG/main/index/subtopic_choice.py)（约 53–97 行）把完整 label 列表放进 prompt，限制到合法标签、检查非空并截断 max，却不强制验证返回数量达到 min；失败后取列表开头若干项。它们使用 `answer_temperature`。因此 min/max prompt、parser 和失败回退是三个需分开记录的行为。

**来源映射风险。**`edge_embedding.py` 的 `build_sent2chunk` 用 sentence 字符串作为单值 key，重复句子的后出现 chunk 可覆盖前者；`_collect_edge_records` 全局去重 sentence，仅保留首先遍历到的关系记录。尚未实测影响，但这不是天然完整的一对多 provenance 保证。CANON 不应无记录地改变原 evidence/provenance 协议。

## 评测已核实与缺口

[judge_F1.py](https://raw.githubusercontent.com/Tiabet/TH-RAG/main/evaluate/judge_F1.py) 的 `compute_metrics`（约 23–46 行）做小写、冠词/标点清理与 token overlap，同时算 normalized exact match。`main`（约 69–100 行）另以 normalized gold substring 判 `accuracy`，支持论文“包含 gold”定义；**accuracy 与 exact_match 是两个输出字段**。

值得后续记录：缺失预测或 `[Error]` 预测从均值分母排除，另报 `missing/compared/total`；因此失败题多时不能只看均值。预测/金答案按 query 字符串建字典，重复 query 会覆盖；空答案和 yes/no/Null 等特殊情况是否完全遵守官方 HotpotQA 协议尚需单测核验，不能仅凭文件名认定完全一致。

README 的默认 evaluation 步骤是答案 F1-style 评测，并展示 UltraDomain pairwise CLI。本次未找到已核实的完整 chunk Recall/F1/Recall@5/NDCG@5 pipeline；gold supporting evidence→chunk mapping 及这些指标的具体脚本仍待确认。Table 5 的问题类型 metric 也未由这个通用答案脚本直接确认。

## 接续时需要解决

上游 commit 已固定。接下来确认运行环境、复核实现差异，准备论文数据子集与 gold evidence，再选择统一协议，在用户 API 下验证 baseline。仓库未附完整实验数据，其他数据发布渠道仍需查询；尚未运行实验。当前进度只维护在根目录 `PROJECT_CONTEXT.md`。

## 硬件估算（2026-10-06，未实测）

假设 LLM 和 embedding 均使用远程 API，本地不加载其模型权重：小样本 4 核/16GB RAM/SSD 预留10GB；两个具体 QA 主实验顺序运行，4–8 核/16–32GB/SSD预留20GB；11M-token 大图建议32–64GB，64GB更适合并行或多份图缓存，SSD预留50GB。均不要求独显；这是工程预留建议，不是作者硬件要求或本地测量。

[FAISS 官方索引说明](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes)给 IndexFlatIP 每向量4d字节：假设10万条1536维向量，仅向量约0.61GB；另有 NetworkX、文本、payload、构建时副本与操作系统开销，不能等同于峰值 RAM。API限流、延迟和预算也会影响实验速度。改用本地模型时重新评估。

## 运行准备：实际配置项与适配点

本节依据固定源码的静态检查；没有安装依赖、导入项目模块、调用模型或运行实验。

- 环境：Python ≥3.10，建议服务器用独立 Python 3.11/3.12 环境；依赖见 `thrag/requirements.txt`，均为版本下限，正式运行后需锁定实际版本。索引用 `faiss-cpu`，没有 GPU 训练入口。
- 配置位置：`thrag/.env`，由 `.env.example` 复制；密钥只放运行机器，不入 Git。
- `OPENAI_API_KEY`：当前全部模型客户端共用一个 key。OpenAI SDK 支持 `OPENAI_BASE_URL`，可补入 `.env` 指向兼容服务；[SDK 官方说明](https://github.com/openai/openai-python#configuring-the-http-client)。若 LLM 和 embedding 来自不同服务/凭据，必须拆分客户端，现有源码未分开这两组配置。
- `DEFAULT_MODEL`：三元组抽取及 Topic/Subtopic 选择；`CHAT_MODEL`：最终回答；`EMBED_MODEL`：证据句/问题向量；`EVAL_MODEL`：仅后续长答案 pairwise judge 使用。答案 F1/Accuracy 计算不调用 judge API。
- 自定义服务需兼容 `chat.completions`、相应 `response_format` 与 embeddings 接口。路由要求 `json_object`；抽取请求 `text` 并自行解析 JSON。
- **自定义模型名的 tokenizer 问题**：建图 `chunk_text` 直接调用 `tiktoken.encoding_for_model(DEFAULT_MODEL)`，无未知模型 fallback；换成不认识的服务模型名可能在 API 调用前抛 KeyError。需要显式固定/配置分词器并与 API 模型名解耦。回答 token 计数有 fallback，但不能替代建图适配。
- 论文主参数应显式设 `MAX_TOKENS=1200`、`OVERLAP=100`、`TOP_K1=30`、`TOP_K2=5`、Topic 5–10、Subtopic 10–25；若跑长答案，另确认 `TOP_K1_LONG/TOP_K2_LONG`，不能让其默认 25/5 混入主协议。`ANSWER_TEMPERATURE=0` 可固定路由/长答随机性，但论文只明确回答温度 0，路由设置应记录为本次配置。
- 并发先小后大，例如 `MAX_WORKERS=2` 起步，依据服务限流和内存实测调整。问题批处理与每题 Subtopic 选择都有线程池；各问题工作线程创建各自的 GraphRAG/图/索引，会放大并发和内存。
- `ALT_MAX_TOKENS/ALT_OVERLAP`、`TEMPERATURE`、`MAX_CONTEXT_LENGTH`、`BATCH_SIZE`、`TIMEOUT_SECONDS`、`ENABLE_CACHE/CACHE_TTL` 当前仅解析、未接入相应主逻辑，不能靠修改它们期待限制上下文、超时或自动缓存失效。
- 缓存按数据集文件名/输出是否存在复用，抽取续跑只检查 chunk ID 和 triples 容器。换模型、chunk 参数或语料需隔离输出/显式重建；错误记录中的空 triples 也可能被当成已完成，需检查失败统计。
- 建图转换使用 `nx.Graph()` 无向图，同一实体对的关系合并在一个边属性中；论文描述为带类型的有向边。短答 prompt 没有附录中严格的最多 5 词/统一拒答文本要求。应将这些与既有 Retriever 差异一起纳入复现协议。

## 论文实验覆盖

| 论文表/实验 | 固定源码已有 | 仍需准备或补写 |
|---|---|---|
| Table 3，TH-RAG 回答质量 | 完整核心链路、短答、F1/Precision/Recall/EM/contains-gold Accuracy | 数据转换、问题 IDs、失败题处理协议；不代表已复现论文分数 |
| Table 3，chunk 检索质量 | 记录返回 chunk IDs 和 sentence 对应 chunk IDs | gold evidence→chunk 映射；Recall/F1/Recall@5/NDCG@5 计算与排名输出 |
| Table 2，UltraDomain pairwise | 长答、三维+Overall judge、按 seed 均衡 A/B 顺序 | 四领域对应问题集、另一方法答案、批量比较组织；judge 独立 CLI，不在默认 pipeline 内 |
| Table 4/14，图统计 | GEXF/NetworkX 图 | 节点/边/连通性统计脚本；LightRAG 对照不在仓库 |
| Table 6，效率 | 每题检索耗时及证据 context token 日志 | 索引和路由 calls/tokens、生成耗时、离线与总费用统计 |
| Table 5/7/8/15 | 可改参数并多次调用核心流程 | 分类型报告、去组件开关、预算 sweep、大图构建/编排与汇总均未封装 |
| Table 16 与其他外部基线 | 无外部方法实现 | 单独获取与运行各基线；代码里的 GraphRAG 类名是 TH-RAG 包装 |

`meta.total_spent` 实际只计 `Retriever.retrieve` 耗时；`context_tokens` 只计拼接证据文本，不含路由/回答 prompt 或最终生成，不能直接当端到端耗时或完整 API token 成本。

## 数据获取与转换

| 数据 | 原始资源 | 获取难点 |
|---|---|---|
| MultiHop-RAG | [作者仓库](https://github.com/yixuantt/MultiHop-RAG)、[作者 HF 文件](https://huggingface.co/datasets/yixuantt/MultiHopRAG/tree/main)：2556 queries，`corpus.json` 6.79MB，`MultiHopRAG.json` 5.17MB | 下载容易；保留新闻元数据和 gold evidence，转换当前输入格式 |
| HotpotQA | [官方页面](https://hotpotqa.github.io/)：distractor dev 标注约44MB，带候选 paragraphs 与 supporting facts | 下载容易；按论文 distractor setting 保留候选段落并建立 chunk 映射，初期不需 FullWiki 全维基语料 |
| UltraDomain | [HF 文件](https://huggingface.co/datasets/TommyChien/UltraDomain/tree/main)：Agriculture/CS/Legal/Mix 可分开取，共约276MB，20领域全量约2.17GB | 语料公开；本篇论文各域125题的具体问题/生成设置不在源码，需要恢复或按统一协议重建 |

当前接口只有 `data/<name>/contexts.txt` 与 `qa.json`，不会自动下载或转换这些 benchmark。`contexts.txt` 被读成整段文本后连续 chunking；`qa.json` 最低字段为 query/answer，官方问题 ID、文档 ID、证据位置与题型必须另外保留，不能在转换时丢弃。需要补预处理脚本和 provenance/gold chunk 映射。

TH-RAG 作者随机抽取的两个 1000-QA 子集未在固定源码中附带，抽样 seed/IDs 也未给出可直接恢复的文件。因此公开原数据容易获取，精确复刻原论文同一子集仍有缺口。可以建立并公布自己的固定子集，用同一 API 重跑 TH-RAG/CANON，但不能声称已经使用作者原子集。

## DeepSeek 与服务器 E5 的组合

用户已确定使用 DeepSeek 官方 V4.1 Flash。当前官方调用名为 `deepseek-flash`，base URL 为 `https://api.deepseek.com`；[官方发布说明](https://api-docs.deepseek.com/updates/)、[JSON 接口示例](https://api-docs.deepseek.com/guides/json_mode/)。运行前显式固定 thinking 模式、输出预算和 API 模型标识，记录日期/版本；尚未进行 API 兼容测试。

用户报告服务器有 `/data/panzheng/bearrag/models/e5-base-v2`。建议先复用此 embedding，是否确为标准模型及文件完整性尚未连接服务器核验。LLM 服务与 embedding 可以独立，原源码共用 API 客户端的实现需要改为独立后端，不能只将 `EMBED_MODEL` 填成一个本地路径。

Tokenizer 将文本转成 token IDs；离线 chunking 用它确定分块长度，API 内部和 E5 输入又分别使用各自的 tokenizer。这三项职责应分开。原默认 GPT-4o-mini 在 tiktoken 中映射到 `o200k_base`；为保持原 chunking 条件，可显式固定这个分块编码，不随 DeepSeek 模型名改变。它只定义本实验的分块协议，不能冒充 DeepSeek 实际 token 数或账单，实际费用统计以 API usage 为准。该分块编码是建议，尚未实施或冻结。

标准 [intfloat/e5-base-v2 模型卡](https://huggingface.co/intfloat/e5-base-v2) 报告：英语模型、768维、最多512输入tokens；检索问题加 `query: `，证据句加 `passage: `，按 mask mean pooling/官方 SentenceTransformer 配置得到并归一化向量。语义聚类的两侧可按官方建议统一用 query 前缀。本地推理需要 PyTorch 和 Transformers/SentenceTransformers，无需重新训练；GPU 可加速，CPU 也能运行。

当前 TH-RAG 对 source sentences 而非1200-token完整chunk做向量化，所以 E5 的512上限不要求把建图chunk改为512。仍应记录超长证据/问题的截断情况；原始句子和来源chunk继续保留。切换 E5 要重建向量索引，所有问题向量也用同一 E5/前缀协议。TH-RAG、CANON及消融共享 DeepSeek/E5 和固定chunk协议；属于换模型条件下的方法复现，原论文数字仅作历史参考。
