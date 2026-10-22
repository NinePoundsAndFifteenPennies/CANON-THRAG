# TH-RAG 配置参考

本文件负责配置项的实际作用、参数覆盖关系、分词器与 embedding 行为、并发及重试。默认值统一见 [`.env.example`](../thrag/.env.example)，实现依据为 [`config.py`](../thrag/config.py)；此处不再复制一份默认值表。

运行和迁移步骤见 [服务器使用说明](../thrag/SETUP_GUIDE.md)，论文与代码差异见 [资源审计](resource_audit.md)，运行验证状态见 [项目上下文](../PROJECT_CONTEXT.md)。以下结论来自静态代码审查，不是服务实测。

<a id="loading"></a>
## 配置加载

配置固定读取代码目录的 `thrag/.env`，进程环境变量优先。修改文件后需重新启动运行进程。当前客户端面向 vLLM 的 OpenAI-compatible API，旧 `DEEPSEEK_*` 字段不再读取。真实密钥的填写步骤只在 [环境配置](../thrag/SETUP_GUIDE.md#environment) 维护。

<a id="paths"></a>
## 服务、模型与路径

| 配置 | 实际作用 |
|---|---|
| `THRAG_STORAGE_ROOT` | 数据与结果根目录。相对值以 `thrag/` 为基准；未设置时 Linux 使用 `config.py` 的服务器默认路径，其他平台使用 `thrag/`。不改变代码目录的 tokenizer 和 pipeline state 位置。 |
| `LLM_API_KEY`、`LLM_BASE_URL` | 传给 API 客户端的密钥与地址。 |
| `LLM_THINKING` | 接受 `enabled/disabled`，通过 `chat_template_kwargs.enable_thinking` 作用于所有模型调用。 |
| `DEFAULT_MODEL` | 建图抽取、主题及子主题选择使用的模型 ID。 |
| `CHAT_MODEL` | 最终短答、长答使用的模型 ID；未设置时继承 `DEFAULT_MODEL`。 |
| `EVAL_MODEL` | Pairwise LLM judge 使用的模型 ID；未设置时继承 `DEFAULT_MODEL`。F1 评测不调用模型。 |
| `CHAT_TOKENIZER_PATH` | 原文切块与 evidence context 计数所用的本地 tokenizer 目录；相对值以 `thrag/` 为基准。 |
| `E5_MODEL_PATH` | 本地 CPU E5 模型及其 WordPiece tokenizer 目录。 |

模型 ID 必须是服务 `/v1/models` 接受的名称。思考开关采用 [Qwen 的 vLLM 参数形式](https://huggingface.co/Qwen/Qwen3.5-9B#instruct-or-non-thinking-mode)，不是提示词中的 `/nothink`。

<a id="generation"></a>
## 生成参数

| 配置 | 实际作用与限制 |
|---|---|
| `TEMPERATURE` | 只读取，未用于模型调用。建图抽取和最终短答的温度均硬编码为 0。 |
| `MAX_TOKENS_RESPONSE` | 仅建图抽取的输出上限；响应因长度截断时会报错，不缓存不完整抽取。 |
| `ANSWER_TEMPERATURE` | 主题选择、子主题选择及长答的温度；不控制短答。 |
| `ANSWER_MAX_TOKENS` | 长答直接使用；短答实际使用 `max(64, min(256, value))`。 |
| `EVAL_TEMPERATURE` | 仅 pairwise LLM judge 的温度。 |

主题/子主题选择使用独立的候选 ID 输出预算，不受上述两个输出预算控制；pairwise judge 未显式设置该输出上限。抽取数量规则及其来源见 [抽取差异](resource_audit.md#extraction)。

调用依据：[`graph_construction.py`](../thrag/index/graph_construction.py)、[短答封装](../thrag/generate/graph_based_rag_short.py)、[长答封装](../thrag/generate/graph_based_rag_long.py)、[主题选择](../thrag/index/topic_choice.py)、[子主题选择](../thrag/index/subtopic_choice.py)、[pairwise judge](../thrag/evaluate/judge_Ultradomain.py)。

<a id="tokenization"></a>
## 切块与 embedding

| 配置 | 实际作用与限制 |
|---|---|
| `MAX_TOKENS`、`OVERLAP` | 实际使用的原文滑动窗口大小与重叠，以聊天模型 tokenizer 的 token 计数。 |
| `ALT_MAX_TOKENS`、`ALT_OVERLAP` | 只读取，没有启用入口，也没有失败后自动切换逻辑。 |
| `E5_BATCH_SIZE` | E5 `encode()` 的批大小，不控制 LLM 请求并发。 |
| `E5_MAX_LENGTH` | E5 输入截断长度，配置要求 1–512。 |

聊天 tokenizer 通过 `AutoTokenizer.from_pretrained(..., local_files_only=True, use_fast=True)` 加载，原文切块不添加 BOS/EOS、不套 chat template。模型权重位于推理服务；本地 tokenizer 的来源、各文件 revision 和 SHA256 见其目录内的 `SOURCE.json`，当前文件为 [Qwen tokenizer 来源](../thrag/tokenizers/qwen3.5-9b/SOURCE.json)。旧 DeepSeek tokenizer 作为既有资源保留，不再是默认配置。

E5 通过 Sentence-Transformers 仅在 CPU 本地加载。query 加 `query: `，证据句加 `passage: `；输出归一化 float32、768 维向量。其 WordPiece 与切块 tokenizer 独立，嵌入对象是证据句，不是完整 chunk。索引维数不匹配时拒绝加载。

本地 context 计数仅覆盖证据文本；API usage 才包含完整请求格式。更换 tokenizer 即使保持切块参数，也会改变 chunk 边界；缓存处理见 [迁移与重建](../thrag/SETUP_GUIDE.md#migration)。

<a id="selection"></a>
## 主题、子主题选择与检索

| 配置 | 实际作用与限制 |
|---|---|
| `TOPIC_CHOICE_MIN`、`SUBTOPIC_CHOICE_MIN` | 进入提示词并按候选数调整；非空合法返回即使不足最小数量，也会被代码接受。 |
| `TOPIC_CHOICE_MAX`、`SUBTOPIC_CHOICE_MAX` | 进入提示词并限制最终列表长度；有效值会按候选数和 min 调整。 |
| `TOP_K1`、`TOP_K2` | 短答的证据句/边数量和不同来源 chunk 数量上限。 |
| `TOP_K1_LONG`、`TOP_K2_LONG` | 长答对应的检索上限。 |
| `EMBEDDING_TOP_K` | 底层 `EdgeEmbedderFAISS.search()` 未指定 `top_k` 时的默认值；主流程显式传 K1，因此通常不会用到。 |
| `OVERRETRIEVE_FACTOR` | 有 entity filter 时的全局预召回倍率；无 filter 时不使用。 |
| `MAX_CONTEXT_LENGTH` | 主题选择计入聊天模板、输出预留及安全余量；超长候选先压缩再分批选择。CANON 联合选择也使用它检查输入预算；不修改 vLLM 启动设置，不自动裁剪最终回答上下文。 |

候选过滤、失败回退、K2 去重扫描与来源映射的具体语义集中在 [实现差异](resource_audit.md#retrieval)。

<a id="concurrency"></a>
## 并发、重试与服务负载

| 配置 | 实际作用与限制 |
|---|---|
| `MAX_WORKERS` | 线程池上限；经 `create_answer_completion` 的主题/子主题选择和回答还共享进程内请求信号量。抽取和独立 judge 的请求不由该信号量统一协调。 |
| `MAX_RETRIES` | 主题/子主题选择的尝试次数（含首轮），另受选择器内部有限上限约束；选择器关闭 SDK 重试以避免预算相乘。 |
| `RETRY_BACKOFF` | 选择器失败尝试之间的指数退避起点。 |

一次问答先选择主题，再为各主题并行选择子主题，最后生成答案。虽然外层和内层线程池嵌套，但当前问答请求经共享信号量限流，不应再将两个线程数相乘当作正常在途 HTTP 请求数。抽取和 pairwise judge 只有一层线程池。pipeline 阶段依次执行；另开多个进程会叠加负载。

当前没有跨进程请求限流或服务负载自适应。原入口每个问答线程独立加载图与 FAISS；CANON 入口在进程内共享只读证据、图和向量，每个回答线程单独保存本题状态。E5 encoder 继续共享并用锁串行 encode。代码依据：[问答外层](../thrag/generate/answer_generation_short.py)、[子主题内层](../thrag/generate/Retriever.py)、[模型服务](../thrag/model_services.py)。

当前主题与子主题选择复用带 JSON Schema 的有界 ID 选择器；暂时性 API 错误和响应校验失败可重试，永久性 API 错误直接上报。抽取、最终回答和 judge 没有接入这组选择器重试参数，继续沿用其 SDK 默认超时及重试行为。

客户端线程数与 vLLM 的调度并发无需相等。过大的负载可能排队、增加延迟、触发重算或服务拒绝；容量取决于 GPU、KV cache、请求长度和服务启动参数，不能由单请求上下文窗口推算。[vLLM 调优说明](https://docs.vllm.ai/en/latest/configuration/optimization/)、[调度参数](https://docs.vllm.ai/en/latest/configuration/engine_args/#schedulerconfig)。

<a id="legacy"></a>
## 未接入运行的系统配置

| 配置 | 当前情况 |
|---|---|
| `LOG_LEVEL`、`LOG_FILE` | 只读取，没有配置 logger 或文件日志。 |
| `BATCH_SIZE` | 只读取，没有用于运行批量；E5 使用自身的批量配置。 |
| `TIMEOUT_SECONDS` | 只读取，没有传给 API 客户端。 |
| `ENABLE_CACHE`、`CACHE_TTL` | 只读取，不控制缓存复用或 TTL 清理。实际缓存规则及强制重建步骤见 [迁移与重建](../thrag/SETUP_GUIDE.md#migration)。 |

<a id="canon"></a>
## CANON 独立模块配置

实现为 [canon/core/config.py](../thrag/canon/core/config.py)。这些变量只被新入口读取，不影响原 pipeline；优先级为新入口显式 CLI 开关、选定任务清单中的方法配置、进程环境、`.env`、代码默认值。可复制初值只维护在 [配置模板](../thrag/.env.example)。CLI 和缓存操作见 [服务器说明](../thrag/SETUP_GUIDE.md#canon)。

| 配置 | 作用 |
|---|---|
| `CANON_HIERARCHY` | `none`、`split`、`split_merge`；关闭规范化不关闭其他模块。 |
| `CANON_SOFT_LINKS` | 构建软连接；只有 A/融合路由使用它传播，B/原选择器不读软边分数。 |
| `CANON_ROUTING` | `baseline`、`a`、`b`、`hybrid`；`baseline` 仅指复用原选择器，不代表作者源码实验版本。 |
| `CANON_BRIDGE`、`CANON_BALANCE` | 分别开启桥梁接力加权和分支均衡分配。 |
| `CANON_MERGE_NEIGHBORS` | 每个标签召回的近邻数量；子主题硬合并只在相同 canonical 父主题组内召回。 |
| `CANON_MERGE_LABEL_THRESHOLD`、`CANON_MERGE_CONTEXT_THRESHOLD` | 硬合并须同时满足的标签及上下文门槛；检查所有跨簇原始成员对。 |
| `CANON_SENTENCE_PROTOTYPE_WEIGHT` | 上下文相似度中句子原型的权重；其余质量分给实体名称原型。 |
| `CANON_SOFT_LABEL_THRESHOLD`、`CANON_SOFT_CONTEXT_THRESHOLD` | 软边标签及上下文门槛；Subtopic 还检查父主题相容性。 |
| `CANON_SOFT_NEIGHBORS` | 每节点最多保留的互选软邻居数量。 |
| `CANON_SOFT_TRANSITION_WEIGHT` | 扩散时软边关系组相对其他关系组的质量；为零时软边不传播。 |
| `CANON_SEED_SENTENCES`、`CANON_SEED_LABELS` | A 的底层种子句数量、每种层级节点的标签种子数量。 |
| `CANON_BOTTOM_UP_WEIGHT` | 底层种子的混合权重；余下质量用于顶层种子，再均分给 Topic/Subtopic。 |
| `CANON_RESTART` | 每轮重新注入初始查询种子的质量比例。 |
| `CANON_DIFFUSION_ITERATIONS`、`CANON_DIFFUSION_TOLERANCE` | 迭代上限与全体节点分数的 L1 停止门槛；记录未收敛截断状态。 |
| `CANON_ACTIVATION_LABEL_WEIGHT`、`CANON_ACTIVATION_TOP_N` | B 的标签/证据混合权重和证据 Top-n 均值窗口；融合版候选排名来自 A。 |
| `CANON_CANDIDATE_TOPICS`、`CANON_CANDIDATE_SUBTOPICS` | B/融合的主题、每主题子主题候选上限；零表示相应原选择上限的两倍，提示词预算可能进一步缩减。 |
| `CANON_PREVIEW_INPUT_TOKENS` | 联合选择输入预算，还受服务窗口扣除输出预算和余量后的限制。 |
| `CANON_PREVIEW_SENTENCE_TOKENS`、`CANON_PREVIEW_ENTITIES` | 每张卡片的证据片段 token 上限及实体预览数。 |
| `CANON_ROUTE_ATTEMPTS` | 联合选择总尝试次数（含首轮），还受 `MAX_RETRIES` 限制。正常完成只需一次请求。 |
| `CANON_BRIDGE_SEED_SENTENCES`、`CANON_BRIDGE_WEIGHT` | 桥梁候选的高分句数量和加权强度；只传播一次，不扩大路由子图。 |
| `CANON_COVERAGE_WEIGHT` | chunk 选择中分支覆盖边际收益的权重。 |

A/B/融合的最终主题、每主题子主题数量上限复用现有 `*_CHOICE_MAX`，不强制补足原提示词的最小选择数；`baseline` 路由沿用原选择器行为。K1/K2 按短答/长答配置读取；`retrieve` 阶段使用短答预算，并要求 `1 <= K2 <= K1`。

新入口不使用 `OVERRETRIEVE_FACTOR`，因为它在实体候选池内部精确排序。API usage 缺失时记录 null，不伪造为零；调用计数是客户端 `create()` 次数，最终回答 SDK 内部重试不属于额外可观测调用。模型服务在同一个 ID 下更换权重无法从 ID 自动识别，需主动重新运行对应结果。
