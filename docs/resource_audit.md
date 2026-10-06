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

**默认参数。**[config.py](https://raw.githubusercontent.com/Tiabet/TH-RAG/main/config.py) 的 `_load_environment`（源文件约 32–61 行）与 [.env.example](https://github.com/Tiabet/TH-RAG/blob/main/.env.example) 给 `MAX_TOKENS/OVERLAP=3000/300`、`TOP_K1/TOP_K2=50/10`、long K 为 25/5、`ANSWER_TEMPERATURE=.3`。同时存在 `ALT_MAX_TOKENS/ALT_OVERLAP=1200/100`，但未逐入口确认哪种 chunk 配置实际被使用；不能把默认环境文件当成论文完整配置。Embedding 默认名为 `text-embedding-3-small`；NT/NST min/max 为 5/10 和 10/25；overretrieve factor 为 5。

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
