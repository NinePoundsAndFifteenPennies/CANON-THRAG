# CANON-THRAG 项目上下文与进度

更新：2026-10-09（Asia/Shanghai）。本文件是唯一进度入口，只保留当前状态、已完成和下一步。

## 项目与范围

基于 ACL 2026 长文 TH-RAG（`2026.acl-long.1740`）开展 CANON-THRAG，目标是形成支持 CCF-C 投稿的研究结果；当前尚无复现实验或有效性结论。先用用户自己的 API 重跑 TH-RAG，再添加核心模块，验证成立后补后续实验。

两个核心模块：①同层 Topic/Subtopic 的语义与结构感知规范化，保留别名和来源映射；②Topic 全局、Subtopic 在已选 Topic 的直接邻居内做 embedding Top-M 预筛，再由原 LLM 选择器多选。预筛主要减少候选列表和提示词，调用数及准确率收益须实测。

保持 Entity、Entity–Entity 三元组、证据句、chunk IDs 和一跳深度不变。K1 排证据句，K2 沿同一排名恢复来源 chunks，没有独立 chunk score。公开代码与论文的候选过滤、fallback、K2 和默认参数有差异，先统一协议再比较。动态预算、直接路由、cross-encoder 等均为后续可选项，不自动加入第一版。

## 已完成

- 论文要素、用户方案及资源差异已有按需摘要，原始 `note.md` 保留。
- 有效性验证顺序已明确：Table 3 质量与 chunk 检索、Table 6 效率、自建四组核心消融；Table 2 后置。
- 第一版目前用于验证方向，投稿充分性未证实。需证明结构约束优于简单 label-only 聚类，不能用模块数量或图压缩率代替贡献证据。
- 进度记录统一到本文件，不另维护过程日志。
- 核心代码与数据来源已核对：支持 TH-RAG 主流程、短答指标及长答 pairwise；chunk 检索指标、完整成本统计、外部基线和消融编排需补。配置与数据入口见资源审计。
- 模型服务适配已完成：DeepSeek 官方 API与官方V4.1文本tokenizer、本地CPU E5及独立WordPiece；query/passage前缀、归一化768维向量与512-token截断。
- 候选模型评估已区分原论文 GPT-4o-mini 与当前 DeepSeek V4.1 Flash 两个参照；三元组 20 条上限的本地修改来源及论文未量化的换模试验已补入按需摘要。尚未进行换模或模型质量对照。

## 当前状态

原 TH-RAG 源码已下载到 `thrag/`，上游 commit 为 `6543fc2a8f5aadecd3032352d7d31dcd5791ec73`。根目录已完成首次提交，源码、文档、配置模板和示例数据统一管理；密钥、缓存和实验产物已忽略。源码与文档已推送到私有仓库 https://github.com/NinePoundsAndFifteenPennies/CANON-THRAG，本地 `main` 跟踪 `origin/main`。

临时下载包、校验文件与上游 Git 备份已清理。本机仅此仓库的 GitHub HTTPS 认证暂引用 `tmp/` 中的便携 CLI 与配置；切换认证后才能移除剩余工具。ChatGPT GitHub 连接已可读取本仓库。

服务器 `canon-thrag` 环境已安装依赖并通过 5 个 pytest 测试；已用示例 `contexts.txt` 成功跑通 graph construction、GEXF、CPU E5/FAISS、短答案生成和 F1 评估。
已生成并抽查 MultiHop 数据：`contexts.txt` 约 4.7 MB，包含 440 篇去重后的 corpus 文档；原始 1,000 条 QA 保存在 `qa_all.json`，正式使用的 `qa.json` 与 `metadata.json` 已同步过滤为 881 条有 evidence 的 query。119 条 `null_query` 的 `evidence_list` 为空且答案为 `Insufficient information.`，不参与当前正式建图后的回答生成与评测。HotpotQA 的 1,000 条抽样无空答案、空 supporting facts 或不完整 supporting fact。


服务器数据与结果配置为 `/data/panzheng/canon-thrag/data/` 和 `/data/panzheng/canon-thrag/results/`，由 `THRAG_STORAGE_ROOT` 统一设置，Linux 默认即该根目录；内部结构和文件名不变。状态文件保留在 `thrag/temp/pipeline_state.json`，tokenizer 与代码一同管理。

已确定并写入配置：LLM 使用 DeepSeek 官方 V4.1 Flash（`deepseek-flash`、`https://api.deepseek.com`），第一版显式关闭 thinking，可配置启用；密钥通过服务器 `DEEPSEEK_API_KEY` 提供。本地 E5 路径为 `/data/panzheng/bearrag/models/e5-base-v2`，固定CPU、仅本地加载、默认batch 32/最大512。文档切块与raw context计数已改用项目附带的官方V4.1 tokenizer（约6MB，仅tokenizer文件，无模型权重），由 `DEEPSEEK_TOKENIZER_PATH` 指定；E5 WordPiece仅用于embedding。chunk/K数值未改，切块边界会随tokenizer改变，后续不能复用旧分块缓存。


用户考虑让全部 baseline 与新方法统一使用 Qwen3.5-9B（区别于 Qwen3-8B），重点判断绝对质量是否显著下降。用户明确允许开启推理模式、调整输出预算；能力讨论不再以现有 token 限制作为障碍。重点是建图的实体/关系/主题抽取和查询时的主题/子主题选择；交流直接使用这些名称，不另加“路由”术语或混入最终答案生成的风险。模型替换尚未决定，当前配置仍为 DeepSeek V4.1 Flash non-thinking。

研究判断（非本地实测）：以原论文已使用的 GPT-4o-mini 为能力参照，对 Qwen3.5-9B thinking 作为统一实验模型持偏积极判断；与当前 DeepSeek V4.1 Flash 是否接近仍是另一问题。[LongBench v2 官方榜](https://longbench2.github.io/)报告 GPT-4o-mini 直接回答/提示 CoT 为 29.3/32.4，[Qwen 官方模型卡](https://huggingface.co/Qwen/Qwen3.5-9B)报告 9B 为 55.2；来自不同评测运行，只支持长文理解和推理的方向性判断，不能直接预测 TH-RAG 分数。“主题选择比抽取更适合替换”只是基于任务结构的推断，并无本项目实验证据确立风险排序。

当前每块最多 20 triples 是本地 commit `336eaa4` 新增的提示词上限，论文 Table 11 要求抽取全部有依据事实，上游对应版本也没有 20 条上限。当前建议取消该人为上限，保留句子依据和去重要求；尚未修改代码。当前切块为 3000/300；不同模型对照先冻结相同 chunk 文本/IDs、embedding 和检索协议，抽取缓存按模型隔离，各模型内四组核心消融共享其原始抽取缓存。

未定：是否采用 Qwen3.5-9B thinking、是否取消 20 条三元组上限、可接受的质量下降幅度；预算、首批数据和问题 IDs、论文忠实或公开代码协议。论文数据子集、gold supporting evidence→chunk 映射与检索评测脚本仍需确认。

## 下一步

1. 确定抽取覆盖协议与候选模型后，若推进换模验证，在相同语料背景和问题 IDs 上分开比较只换抽取、只换主题/子主题选择、同时替换；固定最终回答模型以定位这两个环节的影响。记录关键事实覆盖、完整支持证据召回与 Answer F1；小样本只筛明显退化，不作为正式有效性结论。
2. 在服务器完成数据与结果目录搬放并确认数据集与已有产物可见。等用户开始正式数据准备，再补数据转换、gold chunk 映射和评测/成本采集，固定模型与协议后启动 TH-RAG baseline。
3. 使用同一抽取缓存比较原版、仅规范化、仅预筛、完整核心，检查合并精度与不变量，报告质量和成本；成功后再补规模与外部基线实验。

## 按需阅读

论文事实：`docs/thrag_paper_brief.md`；方案：`docs/idea_digest.md`；代码、数据和协议：`docs/resource_audit.md`。来源与上游版本放在 `docs/sources/`，默认不读全文或重读 `note.md`。实验记录需包含 commit、配置、模型、数据 IDs、缓存和结果路径，密钥不入库。
