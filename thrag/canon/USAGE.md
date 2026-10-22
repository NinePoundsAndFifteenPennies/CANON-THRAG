# CANON 服务器运行指南

本文件只维护增强包的运行操作。模型与依赖的安装、凭据及原始数据准备见 [原系统环境说明](../SETUP_GUIDE.md#environment)；模块算法见 [方案文档](../../docs/idea_digest.md)，开关含义见 [配置参考](../../docs/config_reference.md#canon)。

以下命令在服务器仓库的 `thrag/` 目录及现有环境中执行。修改现有 `.env` 时，仅按 [共享模板](../.env.example) 补充 CANON 字段，保留原服务地址、密钥和实际路径。

<a id="prepare"></a>
## 1. 准备输入与派生产物

需要完整、相互一致的 `<dataset>_graph.json` 和 `<dataset>_kv_store.json`，且没有待处理的抽取失败。请先通过原抽取/修复入口完成这一阶段；增强入口不会自动重抽事实。

新入口不要求已有 GEXF 或旧句子 FAISS 索引。可以单独准备派生产物：

```bash
python -m canon.pipeline --dataset my_dataset --steps prepare
```

将 `my_dataset` 换成实际数据集名。准备阶段编码句子与标签/实体词表，构建选定的层级视图并记录输入与模型指纹，不调用 LLM。复用缓存时也会读取模型文件计算指纹。

<a id="full"></a>
## 2. 运行“基础优化＋三个模块”

下面显式开启三个模块，使用 A+B 融合路由，避免现有 `.env` 中不同的模块开关影响组合：

```bash
python -m canon.pipeline --dataset my_dataset --hierarchy split_merge --soft-links on --routing hybrid --bridge on --balance on --steps generate_short
```

这条命令使用增强版完整来源关联和精确子图检索，并叠加层次规范化、融合路由、桥梁与均衡选择。需要的派生产物会按当前配置准备或复用。

单独使用 A 或 B 时，保留上面其余参数，只把 `--routing hybrid` 改为 `--routing a` 或 `--routing b`。最终答案生成继续复用原短答/长答提示词。

| 运行阶段 | 用途 |
|---|---|
| `--steps generate_short` | 生成短答案 |
| `--steps generate_long` | 生成长答案，使用长答 K1/K2 |
| `--steps retrieve` | 只记录检索结果，不生成最终答案，使用短答 K1/K2 |
| 不传 `--steps` | 准备派生产物并生成短答案 |

只做 A 路由检索时无需 LLM 密钥；B、融合及现有选择器仍会调用路由模型，生成答案也需要模型服务。

限定运行范围可加 `--limit 10`，或用 `--question-indices 0 3 7` 指定原 `qa.json` 的零起始行号，两者互斥。重复问题文本保留各自的行号和问题 ID。

<a id="switches"></a>
## 3. 分别切换模块

只拆分并加软边，不做硬合并：

```bash
python -m canon.pipeline --dataset my_dataset --hierarchy split --soft-links on --routing hybrid
```

关闭层次规范化，使用 A 和仅桥梁加权的证据分配：

```bash
python -m canon.pipeline --dataset my_dataset --hierarchy none --soft-links off --routing a --bridge on --balance off
```

`--routing baseline` 只表示在增强系统内复用现有主题/子主题选择器。`--bridge off --balance off` 关闭两个证据分配子功能，使用普通 cosine 排序及不同 chunk 扫描。

**这些模块开关不会关闭基础优化。**作者源码的原版对照仍由用户单独运行；原 `pipeline.py` 不会自动启用本包。

<a id="ablations"></a>
## 4. 可选消融任务清单

先生成任务清单，不执行任何实验：

```bash
python -m canon.ablations --dataset my_dataset --matrix core --output /absolute/path/core_tasks.json
```

矩阵可选 `core`、`routing`、`hierarchy`、`allocation`。任务的 `environment` 保存当时的方法配置，部署和模型服务配置仍取自服务器环境。

选中一个任务后再运行：

```bash
python -m canon.pipeline --dataset my_dataset --task-file /absolute/path/core_tasks.json --task h1_r1_e1 --steps generate_short
```

显式 CLI 模块开关优先于任务配置。该入口不自动运行作者对照、不比较实验结果、不生成修复贡献报告或评测指标。

<a id="outputs"></a>
## 5. 产物、缓存与续跑

所有增强版产物位于 `<storage_root>/results/canon/`：

| 相对位置 | 内容 |
|---|---|
| `artifacts/<dataset>/evidence-<fingerprint>/` | 完整句子/事实来源、句子与词表向量、校验清单 |
| `artifacts/<dataset>/hierarchy-<fingerprint>/` | 层级图、逐条路径、`canonical_map.json`、标签向量 |
| `runs/<dataset>/<run-id>/manifest.json` | 实际配置、代码与输入身份、模型、问题 ID、阶段状态 |
| `runs/<dataset>/<run-id>/<stage>/questions/` | 原子写入的逐题检查点，含错误和检索详情 |
| `runs/<dataset>/<run-id>/<stage>/answers.json` | 按输入顺序保存的短答或长答，附问题 ID 与运行信息 |
| `runs/<dataset>/<run-id>/<stage>/retrievals.jsonl` | 路径、句子/来源、chunk、回退、usage 和耗时 |

默认运行 ID 根据有效配置与输入生成。显式 `--run-id NAME` 不能复用于另一套输入或配置。重复同一命令会复用成功的逐题检查点，并重试失败题。

`--force` 仅重建 CANON 派生产物和选定结果阶段，不覆盖原抽取/KV 或原入口结果。旧的损坏缓存会被拒绝，并提示重建。`.building` 文件用于阻止同一产物的并发写入；崩溃后须先确认没有仍在写入的进程，再移除遗留锁。

服务在相同 URL 和模型 ID 下更换权重无法自动识别，更新部署后需显式重跑相关结果。manifest 中原抽取 SHA256 标识输入快照；当前模型配置不能倒推原始抽取时的模型或 commit。

这些文件是运行记录，不是自动评测报告。现有答案评测按问题文本对齐等限制见 [评测审计](../../docs/resource_audit.md#evaluation)。

<a id="tests"></a>
## 6. 服务器回归检查

同步代码和配置后，可在服务器执行：

```bash
python -m pytest tests/test_canon.py -q
python -m pytest -q
```

此处只提供命令。实际执行及验证状态统一见 [PROJECT_CONTEXT.md](../../PROJECT_CONTEXT.md)。
