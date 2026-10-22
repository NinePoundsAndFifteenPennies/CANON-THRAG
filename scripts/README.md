# 数据准备脚本

本文负责抽样与转换规则；TH-RAG 输入约定见 [服务器说明](../thrag/SETUP_GUIDE.md#dataset)，已生成数据及筛选结果只在 [项目上下文](../PROJECT_CONTEXT.md#data)记录。

各入口只读取本地原始数据、抽样和转换。不会下载数据、安装依赖、调用 API 或启动 TH-RAG。Python 3.10+；HotpotQA 读取 Parquet 需要服务器环境已有 `pyarrow`，其余入口仅使用标准库。`_prepare_common.py` 是所有入口共用的本地辅助模块；UltraDomain 四个入口另共用 `_prepare_ultradomain.py`，应一同拉取。

| 入口 | 默认输入 | 默认输出目录 |
|---|---|---|
| `prepare_hotpotqa_1000.py` | `/data/panzheng/canon-thrag/datasets/distractor/validation-00000-of-00001.parquet` | `/data/panzheng/canon-thrag/data/hotpotqa_1000/` |
| `prepare_multihop_1000.py` | `/data/panzheng/canon-thrag/datasets/MultiHop/MultiHopRAG.json` 和同目录 `corpus.json` | `/data/panzheng/canon-thrag/data/multihop_1000/` |
| `prepare_ultradomain_agriculture.py` | `/data/panzheng/canon-thrag/datasets/UltraDomain/agriculture.jsonl` | `/data/panzheng/canon-thrag/data/ultradomain_agriculture/` |
| `prepare_ultradomain_cs.py` | `/data/panzheng/canon-thrag/datasets/UltraDomain/cs.jsonl` | `/data/panzheng/canon-thrag/data/ultradomain_cs/` |
| `prepare_ultradomain_legal.py` | `/data/panzheng/canon-thrag/datasets/UltraDomain/legal.jsonl` | `/data/panzheng/canon-thrag/data/ultradomain_legal/` |
| `prepare_ultradomain_mix.py` | `/data/panzheng/canon-thrag/datasets/UltraDomain/mix.jsonl` | `/data/panzheng/canon-thrag/data/ultradomain_mix/` |

## 服务器运行命令

在服务器进入仓库根目录，即 [服务器说明中的代码目录](../thrag/SETUP_GUIDE.md#environment) 的上一级。下面的脚本路径相对仓库根目录；输入和输出仍按上表定位。

```bash
python scripts/prepare_hotpotqa_1000.py --seed 42
python scripts/prepare_multihop_1000.py --seed 42

python scripts/prepare_ultradomain_agriculture.py --seed 42
python scripts/prepare_ultradomain_cs.py --seed 42
python scripts/prepare_ultradomain_legal.py --seed 42
python scripts/prepare_ultradomain_mix.py --seed 42
```

UltraDomain 入口固定写出 **100** 条 QA；`--seed` 默认 **42**。

指定其他输入、输出和 seed：

```bash
python scripts/prepare_hotpotqa_1000.py \
  --input /absolute/path/validation.parquet \
  --output-dir /absolute/path/hotpotqa_1000 --seed 42

python scripts/prepare_multihop_1000.py \
  --queries /absolute/path/MultiHopRAG.json \
  --corpus /absolute/path/corpus.json \
  --output-dir /absolute/path/multihop_1000 --seed 42

python scripts/prepare_ultradomain_legal.py \
  --input /absolute/path/legal.jsonl \
  --output-dir /absolute/path/ultradomain_legal --seed 42 --force
```

自定义路径必须是绝对路径，可使用 `~`。HotpotQA 与 MultiHop-RAG 入口的默认 seed 均为 **42**，固定抽取 **1,000** 条，不提供可变样本数。使用独立的 `random.Random(seed).sample(range(总数), 1000)`，无放回抽样，保留返回的随机顺序，不按难度或问题类型过滤。相同源文件、源记录顺序、seed 和 Python 版本会产生相同抽样与输出。metadata 每条记录保存 `source_index`（源文件中从 0 开始的行号）和 `sampling_seed`；HotpotQA 同时保存原始 id，MultiHop 有原始 id 时也保留。

输出目录已存在时默认报错，包括空目录。确实需要重写时，在对应命令末尾显式加上 `--force`，例如：

```bash
python scripts/prepare_hotpotqa_1000.py --seed 42 --force
python scripts/prepare_multihop_1000.py --seed 42 --force
```

`--force` 只替换目标目录的 `contexts.txt`、`qa.json` 和 `metadata.json`，不清理其他文件。源数据检查、抽样和 QA/metadata 对齐检查完成后，脚本才开始写文件。

## 三个输出文件

- `qa.json`：按 [TH-RAG 输入约定](../thrag/SETUP_GUIDE.md#dataset)写出的问题；HotpotQA 的原始 `question` 转为 `query`。样本数和抽样规则见上文。
- `contexts.txt`：UTF-8 文本，只含原始文本；段落或文档间插入空行，没有问题编号、人工标签或 metadata。字符串内容不做 strip、改写、合并句子或 Unicode 规范化。
- `metadata.json`：UTF-8 JSON 数组，与 QA 的顺序及 query/answer 完全一致。**仅用于后续检索评测，不是当前 TH-RAG pipeline 的必需输入。** 本次转换不生成 chunk IDs 或 gold evidence 到 chunk 的映射。

### HotpotQA

依次写入抽样问题的全部候选段落，包括 distractor。每段为原始 title，换行后按原顺序逐行写入全部 sentences。段间和相邻问题的段落组间都使用空行。即使不同问题共享段落，也不去重。

metadata 保留原始 `id/query/answer/type/level`，`supporting_facts` 中每条为 `title/sent_id/text`；`text` 按精确 title 和从 0 开始的 sent_id 从原始 context 取出。另存原始 `supporting_facts_raw`、完整性标志和异常列表。

运行时检查整个源数据的数量、id 唯一性、非空问题/答案、context 的 title/sentences 长度和字符串结构，以及每个 supporting fact 的结构、标题和句号范围。重复 id、坏 QA 或不能完整写出的 context 结构会报错并停止。supporting facts 长度不一致、标题缺失/歧义、非法 sent_id 等会逐条报告：**保留抽中的样本，不补抽，只将能明确提取的原始支持句放入 `supporting_facts`**，其余原始信息和原因保存在 metadata；`supporting_facts_complete=false`。

### MultiHop-RAG

每条抽样 query 的全部 evidence 先按原始 title 精确匹配 corpus。唯一 title 直接匹配；重复 title 只允许用 evidence 的原始 URL 在同标题候选中精确消歧。没有唯一匹配时不任取第一篇，也不进行模糊匹配。

contexts 写入匹配文档的完整 `title + 换行 + body`，不使用 fact 代替正文。按 corpus 的记录索引全局去重，文档顺序是其在抽样 query/evidence 顺序中的首次出现。每个 query 的 metadata 仍保留自己完整的 evidence 对应关系及未改写的 fact；重复引用不会被从 metadata 删除。

metadata 保留 `query/answer/question_type`、原始 `evidence_list_raw` 和增强后的 `evidence`。增强条目保留原始 evidence 的所有字段，并补充：

- `corpus_found`：是否找到唯一对应文档；找不到或无法消歧时为 false。
- `corpus_record_index`、`corpus_candidate_indices`、`corpus_match_method` 和 `corpus_match_status`：记录匹配依据和结果。
- `corpus_body_nonempty`：完整正文是否可写入；唯一文档存在但 body 为空/非法时，`corpus_found=true`，此标志为 false。
- `context_document_index`：文档在输出中的从 0 开始的顺序索引；未写入为 null。此索引不是 TH-RAG chunk ID。

运行时检查源 query 数量、精确重复 query、重复完整记录以及可用原始 id 的唯一性；检查非空 query/answer、evidence_list 结构、空 evidence、corpus 重复标题、每个 evidence 的精确匹配和文档正文。重复 query/id/完整记录、坏 QA 或无法读取的结构会报错并停止。空 evidence、未匹配/歧义标题、空/非法 body 会报告：**保留抽中的 query，不补抽，保留其原始 evidence 和问题级异常；只写入能唯一匹配且正文非空的文档**，`evidence_complete=false`。

### UltraDomain

四个入口分别处理 `agriculture.jsonl`、`cs.jsonl`、`legal.jsonl` 和 `mix.jsonl`。每行一个 JSON 对象；每个域固定写出 **100** 条 QA。源文件恰好 100 条时（Agri、CS）保留文件顺序；超过 100 条时（Legal 438、Mix 130）使用 `random.Random(seed).sample(range(总数), 100)` 无放回抽样，并保留返回的随机顺序。论文每域 125 条问题的设置不采用，因此这些输出不能直接复现论文的问题集。

- `qa.json`：源 `input` 原样写为 `query`，`answers[0]` 原样写为 `answer`；四份源文件的 `answers` 都恰好只有一项。
- `contexts.txt`：写入该域全部 released 文档，不只写被抽中问题引用的文档。按 `context` 的精确文本去重，保留首次出现顺序。`context` 本身已包含原文标题和正文，因此不再追加标题。字符串不 strip、不改写。Agri/CS 的 `context_id` 只用于一致性核对，不作为去重键。
- `metadata.json`：每条保存 `domain`、`source_index`、`source_record_id`（源 `_id`）、`sampling_seed`、`sample_retained`、`context_document_index`、`context_sha256`、`context_chars` 和 `answers_raw`；源文件存在时还保留 `context_id`、`length`（源 token 长度）、`label`、`dataset`、`meta`。`duplicate_query_in_source`、`duplicate_query_source_indices` 和 `duplicate_query_in_sample` 标记同一文件内及本次样本中的重复问题。
- Legal 源文件有 11 组共 23 条重复问题（同一问题文本对应不同文档，答案相同）。seed 42 的 100 条样本包含其中 1 组 2 条（源索引 189、432）；重复记录全部保留并标记，不去重、不补抽，评测时需自行决定是否按问题去重。
- 不生成 chunk ID 或 gold evidence→chunk 映射；UltraDomain 使用 pairwise 答案评测。

运行时检查 JSONL 结构、`_id` 唯一性、非空问题/答案/正文、单答案列表和重复问题；结构错误会报错并停止。相同 `context` 文本若出现多个 `context_id`、同一 `context_id` 对应多个正文，或重复正文的 `length` 不一致，会输出警告但不丢弃记录。

## 运行后统计

正常结束时打印源样本数、唯一性统计、seed、抽样数、源数据和抽样数据的 evidence 异常统计、实际写入的段落/文档数及三个文件路径。MultiHop 额外报告重复标题数、匹配状态计数和去重文档引用数。UltraDomain 报告源记录数、seed、写出 QA/文档数、样本引用的文档数、重复问题统计与上下文 token 合计。异常输出到 stderr；QA 与 metadata 的数量、顺序和内容对齐在写文件前检查。
