# CANON-THRAG 项目上下文与进度

更新：2026-10-06（Asia/Shanghai）。本文件是唯一进度入口，只保留当前状态、已完成和下一步。

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

## 当前状态

原 TH-RAG 源码已下载到 `thrag/`，上游 commit 为 `6543fc2a8f5aadecd3032352d7d31dcd5791ec73`。根目录已完成首次提交，源码、文档、配置模板和示例数据统一管理；密钥、缓存和实验产物已忽略。源码与文档已推送到私有仓库 https://github.com/NinePoundsAndFifteenPennies/CANON-THRAG，本地 `main` 跟踪 `origin/main`。

临时下载包、校验文件与上游 Git 备份已清理。本机仅此仓库的 GitHub HTTPS 认证暂引用 `tmp/` 中的便携 CLI 与配置；切换认证后才能移除剩余工具。ChatGPT GitHub 连接已可读取本仓库。

没有安装依赖、运行 baseline 或产生研究结果。本地编码，服务器运行且可能改代码，双方通过 GitHub 同步；本轮不配置服务器。上游仅有两条示例 QA 和 `contexts_example.txt`，缺 pipeline 要求的 `contexts.txt`；正式数据及 gold chunk 映射需另行准备。

已确定：LLM 使用 DeepSeek 官方 V4.1 Flash，当前官方 API 名为 `deepseek-flash`、地址 `https://api.deepseek.com`；用户已有 key，密钥只放服务器环境配置。Embedding 建议复用服务器 `/data/panzheng/bearrag/models/e5-base-v2`，尚待确认模型文件、版本和运行环境，未实际接入。

未定：embedding 最终配置、预算、首批数据和问题 IDs、论文忠实或公开代码协议。论文数据子集、gold supporting evidence→chunk 映射与检索评测脚本仍需确认。

## 下一步

1. 接入 DeepSeek，并把切块分词器与 API 模型名解耦；确认并接入本地 E5，分开 LLM 与 embedding 后端。准备数据转换、gold chunk 映射和评测/成本采集，固定协议后在服务器启动 TH-RAG baseline。
2. 使用同一抽取缓存比较原版、仅规范化、仅预筛、完整核心，检查合并精度与不变量，报告质量和成本；成功后再补规模与外部基线实验。

## 按需阅读

论文事实：`docs/thrag_paper_brief.md`；方案：`docs/idea_digest.md`；代码、数据和协议：`docs/resource_audit.md`。来源与上游版本放在 `docs/sources/`，默认不读全文或重读 `note.md`。实验记录需包含 commit、配置、模型、数据 IDs、缓存和结果路径，密钥不入库。
