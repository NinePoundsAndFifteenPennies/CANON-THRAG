# CANON-THRAG 项目上下文与进度

更新：2026-10-12（Asia/Shanghai）。本文件是唯一进度入口；文档职责见 [导航](README.md#documents)。

## 目标与范围

基于 TH-RAG 做 CANON-THRAG，目标是形成支持 CCF-C 投稿的研究结果。先重跑 baseline，再验证 Topic/Subtopic 结构规范化与 Top-M 候选预筛；核心定义、约束和实验设计以 [方案文档](docs/idea_digest.md)为准。当前没有正式复现实验或方法有效性结论。

## 已完成与当前状态

- 论文事实、方案、实现差异和配置说明已各有独立维护位置；研究笔记及源文证据保留原文。
- 模型适配已改为用户部署的 Qwen3.5-9B，本版采用非思考模式；抽取协议调整和分词器切换已完成本地修改并推送。部署参数的唯一文档来源是 [配置模板](thrag/.env.example)，行为说明见 [配置参考](docs/config_reference.md)。
- 抽取已统一为每 chunk 至多 110 条三元组，使用请求级 JSON Schema 与严格落盘校验；首轮失败按 dataset 记录并继续其他 chunk。独立修复入口为 [repair_graph.py](thrag/index/repair_graph.py)，沿用原 KV 身份合并；pipeline 在抽取后及后续阶段前检查失败，自动修复一次，未解决则停止并保留成功结果与剩余记录。
- 配置生效范围与并发问题已经静态核对；详细结论只在配置参考维护，尚未实施并发结构或遗留配置清理。
- 核心改进模块尚未实现；正式问题 IDs、预算与复现协议仍需固定。

<a id="validation"></a>
## 已有验证记录

以下均为此前 DeepSeek 配置下的记录，不代表当前 Qwen 版本已验证，也不构成论文复现结果。

- 服务器 `canon-thrag` 环境通过当时的 5 个 pytest 测试，`pip check` 无 broken requirements；最小示例完成建图、GEXF、CPU E5/FAISS、短答和 F1 评估。
- 示例只有 1 个 chunk、2 条 QA：F1 = 0.8333333333333333，Accuracy = 1.0，Exact match = 0.5。已有产物快照位置为 `thrag/results/` 与 `thrag/temp/`，没有长答。
- 外置存储路径曾通过隔离验证：数据集发现、产物复用和评测读写正常；对同两条预测的离线重评 F1 不变、missing=0。

当时记录的环境版本：Python 3.10.22、PyTorch 2.14.1+cpu、faiss-cpu 1.15.1、sentence-transformers 5.7.0、transformers 5.19.0、openai 3.26.0、numpy 2.2.6、pytest 9.1.1。安装步骤以 [服务器说明](thrag/SETUP_GUIDE.md#environment)为准，不在这里复制命令。

<a id="data"></a>
## 数据准备状态

- MultiHop：`contexts.txt` 约 4.7 MB，含 440 篇去重 corpus 文档。原始抽样 1,000 条 QA 保存在 `qa_all.json`；正式 `qa.json` 与 `metadata.json` 已同步筛为 881 条有 evidence 的问题。
- 119 条 `null_query` 的 evidence 为空，答案为 `Insufficient information.`，不进入当前正式回答与评测。
- HotpotQA：1,000 条抽样已检查，无空答案、空 supporting facts 或不完整 supporting fact。

转换算法、metadata 字段和脚本命令仅在 [数据脚本说明](scripts/README.md)维护；运行目录及产物布局见 [服务器说明](thrag/SETUP_GUIDE.md#dataset)。

## 待解决与下一步

1. 确认服务可承受的并发，决定是否处理 [配置参考中的并发与重试问题](docs/config_reference.md#concurrency)。
2. 固定正式问题 IDs、预算和论文忠实/公开代码协议，补齐 [实现审计列出的评测能力缺口](docs/resource_audit.md)，启动 baseline。
3. 依 [实验设计](docs/idea_digest.md#validation)实现并验证两个核心模块；M、合并阈值在开发/验证集选择，成功后补扩展实验。

## 仓库与会话约束

上游版本只在 [UPSTREAM.md](thrag/UPSTREAM.md)维护。本地 `main` 跟踪 `origin/main`，私有仓库为 [CANON-THRAG](https://github.com/NinePoundsAndFifteenPennies/CANON-THRAG)。本机该仓库的 HTTPS 认证仍引用 `tmp/` 中的便携 CLI 与配置，切换认证后才能清理剩余工具。

交流使用“主题/子主题选择”。当前授权不包括完整数据集运行或新增可选研究模块；论文事实、用户提案、代码行为与运行实测须分开。源文只按需定位读取，证据记录要求见 [AGENTS.md](AGENTS.md)。
