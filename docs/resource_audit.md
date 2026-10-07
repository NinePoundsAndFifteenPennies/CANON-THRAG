# TH-RAG 资源与实现核验

核验日期：2026-10-07

## 资源范围

- 公开仓库：<https://github.com/Tiabet/TH-RAG>
- 上游版本：`6543fc2a8f5aadecd3032352d7d31dcd5791ec73`
- 许可证：Apache-2.0
- 代码目录：[thrag/](../thrag/)
- 入口：[pipeline.py](../thrag/pipeline.py)
- 服务器存储根目录：`/data/panzheng/canon-thrag/`，配置项 `THRAG_STORAGE_ROOT`
- 输入格式：`/data/panzheng/canon-thrag/data/<dataset>/contexts.txt` 与 `qa.json`
- 输出目录：`/data/panzheng/canon-thrag/results/` 下的 `index/`、`generated/`、`chunks/`、`evaluated/`
- 流程状态：代码目录内的 `thrag/temp/pipeline_state.json`

代码覆盖：

| 环节 | 文件 |
|---|---|
| 配置与编排 | `config.py`、`pipeline.py` |
| 图构建 | `index/graph_construction.py`、`index/json_to_gexf.py` |
| Edge embedding | `index/edge_embedding.py` |
| Topic/Subtopic 选择 | `index/topic_choice.py`、`index/subtopic_choice.py`、`prompt/` |
| 检索与回答 | `generate/Retriever.py`、`generate/answer_generation_short.py`、`generate/answer_generation_long.py` |
| 评测 | `evaluate/judge_F1.py`、`evaluate/judge_Ultradomain.py` |

仓库包含两条示例 QA 和 `contexts_example.txt`。正式论文数据、问题 ID、gold supporting evidence 到 chunk 的映射和预建结果不在仓库中。

## 论文描述与实现差异

### 参数

配置默认值：

```text
MAX_TOKENS / OVERLAP = 3000 / 300
TOP_K1 / TOP_K2 = 50 / 10
TOP_K1_LONG / TOP_K2_LONG = 25 / 5
TOPIC_CHOICE_MIN / MAX = 5 / 10
SUBTOPIC_CHOICE_MIN / MAX = 10 / 25
EMBEDDING_TOP_K = 5
OVERRETRIEVE_FACTOR = 5
```

主建图流程使用 `MAX_TOKENS` 和 `OVERLAP`；`ALT_MAX_TOKENS` 与 `ALT_OVERLAP` 仅由配置读取。选择数量的 `min/max` 进入提示词和解析限制，但不强制模型返回不少于 `min` 个标签。失败回退取候选列表开头的若干项。

### 候选过滤、fallback 与 K2

- Entity filter 存在时，FAISS 先取全局 `K1 × OVERRETRIEVE_FACTOR`，再保留任一端点属于 Entity 集合的边，因此返回数可能少于 K1。
- Entity filter 无结果时，检索器回退到全图搜索。
- K2 沿返回边的顺序去重恢复 chunk，直到达到 K2 个不同 chunk 或候选耗尽；实现行为与“只从前 K2 条证据恢复 chunk”的论文算法描述不同。
- 向量归一化后使用 FAISS `IndexFlatIP`，等价于非零向量的 cosine 排名；论文附录对 cosine 计算方式的描述不同。

### 来源映射

`sentence -> chunk_id` 使用单值字典；重复证据句可能覆盖来源。Edge record 又会按句子全局去重，只保留首次遇到的关系记录。CANON 的规范化和预筛需要保留证据句、来源 chunk 与实体关系的映射。

### 评测

`judge_F1.py` 同时输出 normalized exact match、F1、precision、recall 和基于 normalized gold substring 的 accuracy。`missing`、`compared`、`total` 单独统计，缺失或 `[Error]` 预测不进入均值分母。预测和 gold 按 query 建字典，重复 query 会覆盖。

仓库提供短答案 F1 和 UltraDomain 风格 pairwise judge；没有已核实的 chunk Recall/F1、Recall@5、NDCG@5 pipeline，也没有 gold evidence→chunk 映射。

## 服务器运行环境

运行环境：

```text
Conda environment: canon-thrag
Python: 3.10.22
PyTorch: 2.14.1+cpu
faiss-cpu: 1.15.1
sentence-transformers: 5.7.0
transformers: 5.19.0
openai: 3.26.0
numpy: 2.2.6
pytest: 9.1.1
```

依赖来源：

```bash
pip install -r requirements-cpu.txt
PIP_CONFIG_FILE=/dev/null python -m pip install --index-url https://pypi.org/simple -r requirements.txt
PIP_CONFIG_FILE=/dev/null python -m pip install --index-url https://pypi.org/simple pytest
```

`PIP_CONFIG_FILE=/dev/null` 用于绕过系统 NVIDIA PyPI extra index 和不可用的清华 PyPI 配置，不改变全局 pip 配置。`pip check` 无 broken requirements。

模型与服务：

```text
LLM API: https://api.deepseek.com
Chat model: deepseek-flash
Thinking: disabled
Storage root: /data/panzheng/canon-thrag
Data: /data/panzheng/canon-thrag/data
Results: /data/panzheng/canon-thrag/results
Embedding model: /data/panzheng/bearrag/models/e5-base-v2
Embedding device: CPU
E5 batch / max length: 32 / 512
E5 dimension: 768
```

密钥通过 `thrag/.env` 中的 `DEEPSEEK_API_KEY` 提供，不写入仓库。E5 仅从本地路径加载，不使用 embedding API。

Linux 在未设置 `THRAG_STORAGE_ROOT` 时也使用上述存储根目录；其他平台未设置时使用 `thrag/`。相对路径以 `thrag/` 为基准解析，`THRAG_STORAGE_ROOT=.` 可用于本地目录。所有建图、检索、生成、F1 评测和数据集扫描均通过同一配置读取数据与结果路径。目录内结构与文件名固定，已有产物放入对应目录后可继续复用；搬放命令见 `thrag/SETUP_GUIDE.md`。模板数据仍在仓库的 `thrag/data/test_dataset/`。

## Tokenizer 与 embedding 协议

DeepSeek tokenizer 位于：

```text
thrag/tokenizers/deepseek-v4.1-flash/
```

包含 `tokenizer.json`、`tokenizer_config.json`、`SOURCE.json` 和许可证。项目通过 `AutoTokenizer.from_pretrained(..., local_files_only=True, use_fast=True)` 加载，不下载 DeepSeek 权重，不添加 BOS/EOS，不套 chat template。

该 tokenizer 用于：

1. `index/graph_construction.py` 的文档切块；
2. `generate/graph_rag.py` 的 evidence context token 计数。

API 返回的 usage 是完整请求 token 和计费的依据；本地计数只覆盖 evidence context 字符串。

E5 使用本地 `intfloat/e5-base-v2` 权重：

- query 加 `query: ` 前缀；
- evidence sentence 加 `passage: ` 前缀；
- `normalize_embeddings=True`；
- 输出 float32、768 维向量；
- 输入超过 512 个 WordPiece token 时截断。

当前 edge embedding 针对 evidence sentence，不是完整 chunk。更换 DeepSeek tokenizer、E5 模型或 chunk 参数都需要重建抽取、图、FAISS 和答案缓存。

## 已验证运行

服务器使用 `canon-thrag` 环境完成：

```bash
cd thrag
python test_config.py
python -m pytest -q
```

结果：

```text
5 passed
```

使用示例 `test_dataset` 完成 graph construction、JSON→GEXF、CPU E5/FAISS、短答案生成和 F1 评估。该次输入只有一个 chunk、两个 QA：

```text
F1: 0.8333333333333333
Accuracy: 1.0
Exact match: 0.5
```

该次验证的工作区产物快照位于 `thrag/results/` 和 `thrag/temp/`，包含两条有效答案、一个 chunk 和全部短答流程产物；没有长答案。服务器的运行数据与结果按上述存储配置定位。这是流程验证，不构成论文复现实验结论。

外置存储配置已通过本地隔离验证：数据集发现、图与答案产物复用、F1 评测读写及相对路径解析正确。复用服务器的两条预测进行离线重评，F1 仍为 0.8333333333333333，missing=0；原始产物与状态文件保留。

## 运行入口与待补内容

```bash
cd /home/panzheng/projects/CANON-THRAG/thrag
conda activate canon-thrag

python test_config.py
python pipeline.py --list-datasets
python pipeline.py --dataset <dataset> --steps graph_build
python pipeline.py --dataset <dataset> --steps answer_generation_short
python pipeline.py --dataset <dataset> --steps answer_generation_long
python pipeline.py --dataset <dataset> --steps evaluation
```

待补：

- 论文数据子集和问题 ID；
- gold supporting evidence→chunk 映射；
- chunk 检索指标；
- API usage、延迟和成本的完整采集；
- 原版、仅规范化、仅预筛、完整 CANON 的统一消融编排；
- 论文协议与公开代码协议的最终选择。
