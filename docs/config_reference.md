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

主题选择、子主题选择和 pairwise judge 未显式传 `max_tokens`，不受上述两个输出预算控制。抽取数量规则及其来源见 [抽取差异](resource_audit.md#extraction)。

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
| `MAX_CONTEXT_LENGTH` | 仅作为服务窗口元数据，并由配置自检展示；不检查请求长度、不裁剪证据，也不修改 vLLM 启动设置。 |

候选过滤、失败回退、K2 去重扫描与来源映射的具体语义集中在 [实现差异](resource_audit.md#retrieval)。

<a id="concurrency"></a>
## 并发、重试与服务负载

| 配置 | 实际作用与限制 |
|---|---|
| `MAX_WORKERS` | 每个线程池的线程上限，作用于抽取、问答外层、每题子主题选择内层和 pairwise judge；不是全局 HTTP 并发上限。 |
| `MAX_RETRIES` | 主题/子主题选择的最多尝试次数，含首轮；不配置 SDK 的 HTTP 重试。 |
| `RETRY_BACKOFF` | 仅子主题选择失败尝试之间的固定等待秒数；主题选择不等待。 |

一次问答先选择主题，再为各主题并行选择子主题，最后生成答案。外层最多同时处理 W 道题，每题内层最多 `min(W, 实际选中主题数)` 个请求；例如 W=10 且每题选中 10 个主题，客户端正常在途请求峰值可达 100。抽取和 pairwise judge 只有一层线程池，各自最多 W 个请求。pipeline 的阶段依次执行；另开多个进程会叠加负载。

当前没有全局 HTTP semaphore、请求速率限制或服务负载自适应。每个问答线程独立加载图与 FAISS，影响 pipeline 所在机器的内存；E5 encoder 则在进程内共享，并用锁串行执行 encode。降低 W 会同时降低两层并发，例如 W=2 时问答正常峰值约为 4。代码依据：[问答外层](../thrag/generate/answer_generation_short.py)、[子主题内层](../thrag/generate/Retriever.py)、[共享 encoder](../thrag/model_services.py)。

主题选择的 API 调用在自定义 `try` 外，因此该层主要重试 JSON/标签校验失败，不捕获 API 调用异常；子主题选择则把 API 调用放在 `try` 内。抽取、回答和 judge 没有接入这组自定义重试参数。客户端构造只传 `api_key/base_url`，HTTP 超时和底层重试仍使用 SDK 自身默认设置。

客户端线程数与 vLLM 的调度并发无需相等。过大的负载可能排队、增加延迟、触发重算或服务拒绝；容量取决于 GPU、KV cache、请求长度和服务启动参数，不能由单请求上下文窗口推算。[vLLM 调优说明](https://docs.vllm.ai/en/latest/configuration/optimization/)、[调度参数](https://docs.vllm.ai/en/latest/configuration/engine_args/#schedulerconfig)。

<a id="legacy"></a>
## 未接入运行的系统配置

| 配置 | 当前情况 |
|---|---|
| `LOG_LEVEL`、`LOG_FILE` | 只读取，没有配置 logger 或文件日志。 |
| `BATCH_SIZE` | 只读取，没有用于运行批量；E5 使用自身的批量配置。 |
| `TIMEOUT_SECONDS` | 只读取，没有传给 API 客户端。 |
| `ENABLE_CACHE`、`CACHE_TTL` | 只读取，不控制缓存复用或 TTL 清理。实际缓存规则及强制重建步骤见 [迁移与重建](../thrag/SETUP_GUIDE.md#migration)。 |
