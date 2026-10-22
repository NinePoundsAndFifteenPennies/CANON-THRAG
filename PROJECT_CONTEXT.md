# CANON-THRAG 项目上下文与进度

更新：2026-10-22（Asia/Shanghai）。本文件是唯一进度入口；文档职责见 [导航](README.md#documents)。

## 目标与范围

基于 TH-RAG 做 CANON-THRAG，目标是形成支持 CCF-C 投稿的研究结果。当前增强方案包含层次拆分/合并与软连接、可独立选择的 A/B/融合路由、桥梁接力与分支均衡证据分配；定义和模块边界以 [方案文档](docs/idea_digest.md)为准。最终与作者源码运行结果的比较由用户自行组织，不设置修复版基线或自动贡献归因。当前没有正式复现实验或方法有效性结论。

## 已完成与当前状态

- 论文事实、方案、实现差异和配置说明已各有独立维护位置；研究笔记及源文证据保留原文。
- 模型适配已改为用户部署的 Qwen3.5-9B，本版采用非思考模式；抽取协议调整和分词器切换已完成本地修改并推送。部署参数的唯一文档来源是 [配置模板](thrag/.env.example)，行为说明见 [配置参考](docs/config_reference.md)。
- 抽取已统一为每 chunk 至多 110 条三元组，使用请求级 JSON Schema 与严格落盘校验；首轮失败按 dataset 记录并继续其他 chunk。独立修复入口为 [repair_graph.py](thrag/index/repair_graph.py)，沿用原 KV 身份合并；pipeline 在抽取后及后续阶段前检查失败，自动修复一次，未解决则停止并保留成功结果与剩余记录。已移除新增的请求级显式超时，正常抽取与修复均恢复使用原 SDK 默认超时。断点进度只由 KV 与 graph 判断，与 fail 逻辑完全分离：不带 `--force` 时保留非连续合法块、只补缺块并从落盘数量继续进度；带 `--force` 时所有选定阶段从头重建。续建加载原 fail 状态且不做对账、筛选或补写，之后仍只在成功时移除对应条目、失败时更新对应条目。
- 配置与并发已按现有源码静态核对，详细行为只在配置参考维护；新入口复用现有模型服务与请求信号量，本轮未改原并发调度或清理遗留配置。
- UltraDomain 四个子集的原始数据与 TH-RAG 输入文件已就绪，并通过源数据一致性检查；尚未建图、生成答案或做任何评测。
- 三个增强模块及独立入口已写入 [canon/](thrag/canon/)，算法与原检索器分开；原生成器仅增加可选依赖注入及完整来源字段兼容。支持模块独立开关、硬规范化/仅软连接路线及 A/B/融合选择；完整来源关联与子图内精确检索只用于增强版。
- 新入口支持配置和输入指纹隔离、逐题断点、调用/usage/耗时记录及可选消融任务清单；不自动运行作者对照或评测。用法仅在 [服务器说明](thrag/SETUP_GUIDE.md#canon) 维护，参数初值在配置模板，含义在配置参考。
- 回归用例已编写，尚未执行。正式问题 IDs、检索评测映射、预算和实验协议仍由用户后续确定；当前参数是实现初值，未调优。

<a id="validation"></a>
## 已有验证记录

2026-10-22：本轮仅完成源码逐段阅读、差异及空白检查；未执行 Python、pytest、编译、pipeline、模型调用或任何服务器试跑，不能记为运行验证通过。新增测试覆盖来源关联、作用域拆分/合并、路由调用与错误预算、桥梁/覆盖、配置缓存和逐题检查点；待用户在服务器运行。


当时记录的环境版本：Python 3.10.22、PyTorch 2.14.1+cpu、faiss-cpu 1.15.1、sentence-transformers 5.7.0、transformers 5.19.0、openai 3.26.0、numpy 2.2.6、pytest 9.1.1。安装步骤以 [服务器说明](thrag/SETUP_GUIDE.md#environment)为准，不在这里复制命令。



<a id="data"></a>
## 数据准备状态

- MultiHop：`contexts.txt` 约 4.7 MB，含 440 篇去重 corpus 文档；当前可直接运行的 `qa.json` 与 `metadata.json` 已恢复为完整 1,000 条，119 条 `null_query` 的答案保留为 `Insufficient information.`。此前的 881 条版本保存在 `contexts881.txt`、`qa881.json` 和 `metadata881.json`，`qa_all.json` 仍保留为原始 QA 备份。
- 1,000 条版本的 `metadata.json` 与 `qa.json` 顺序完全对应；119 条 `null_query` 的 evidence 为空并标记为 preparation issue，但其答案可用于判断模型是否正确回答 `Insufficient information.`。
- HotpotQA：1,000 条抽样已检查，无空答案、空 supporting facts 或不完整 supporting fact。
- UltraDomain：原始 JSONL 位于 `/data/panzheng/canon-thrag/datasets/UltraDomain/`（Agriculture、CS、Legal、Mix 四份），来自 Hugging Face `TommyChien/UltraDomain`（dataset commit `aa8a51d523f8fc3c5a0ab90dd16b7f6b9dbb5d0d`，apache-2.0）。已生成可直接运行的 `/data/panzheng/canon-thrag/data/ultradomain_agriculture`、`ultradomain_cs`、`ultradomain_legal`、`ultradomain_mix`：每个域固定 100 条 QA。Agri/CS 源文件恰好 100 条，保留全部记录和文件顺序；Legal 从 438 条、Mix 从 130 条中用 seed 42 无放回抽样。`contexts.txt` 写入各域全部 released 文档，去重后为 12/10/94/61 篇；Agri/CS 的 `context_id` 完整，四个子集的源 `length` 合计为 1,949,526/2,047,866/4,773,793/611,161，与 Table 1 的 1.9M/2.0M/4.7M/602K 一致。统一采用 100 题/域，不采用论文每域 125 题。Legal 源文件有 11 组重复问题（23 条），seed 42 样本含其中 1 组 2 条，已保留并在 metadata 标记。

转换算法、metadata 字段和脚本命令仅在 [数据脚本说明](scripts/README.md)维护；运行目录及产物布局见 [服务器说明](thrag/SETUP_GUIDE.md#dataset)。

## 待解决与下一步

1. 用户将新源码与配置同步到服务器，按 [运行说明](thrag/SETUP_GUIDE.md#canon) 执行回归用例，再检查真实语料派生产物与小规模问答；本轮尚未运行，服务容量和性能也未验证。
2. 用户自行组织作者源码对照以及 A/B/融合与模块消融，确定问题 IDs、预算及最终组合；不自动加入修复版对照或修复收益报告。
3. 完整 gold evidence→chunk 映射和 chunk 检索指标仍未实现；相关缺口见 [资源审计](docs/resource_audit.md)。UltraDomain 每域 100 条与论文采样口径的差异、Legal 重复问题处理仍需在用户实验协议中说明。

## 仓库与会话约束

上游版本只在 [UPSTREAM.md](thrag/UPSTREAM.md)维护。本地 `main` 跟踪 `origin/main`，私有仓库为 [CANON-THRAG](https://github.com/NinePoundsAndFifteenPennies/CANON-THRAG)。本机该仓库的 HTTPS 认证仍引用 `tmp/` 中的便携 CLI 与配置，切换认证后才能清理剩余工具。

交流使用“主题/子主题选择”。本轮已授权三个增强模块与消融入口的代码实现；用户明确要求不执行项目代码或测试，运行由用户在服务器进行。不自动开展完整数据集实验、不新增其他研究模块、不提交或推送代码。保留用户对本文件既有内容的删除，不恢复旧验证记录。论文事实、用户提案、代码行为与运行实测须分开；证据记录要求见 [AGENTS.md](AGENTS.md)。
