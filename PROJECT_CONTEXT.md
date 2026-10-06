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

## 当前状态

原 TH-RAG 源码已下载到 `thrag/`，上游 commit 为 `6543fc2a8f5aadecd3032352d7d31dcd5791ec73`。根目录已建立 Git 仓库，并准备 README、忽略规则和跨平台换行规则。目标远程是用户 `NinePoundsAndFifteenPennies` 的私有仓库 `CANON-THRAG`，尚待登录后创建并推送。

没有安装依赖、运行 baseline 或产生研究结果。本地编码，服务器运行且可能改代码，双方通过 GitHub 同步；本轮不配置服务器。上游仅有两条示例 QA 和 `contexts_example.txt`，缺 pipeline 要求的 `contexts.txt`；正式数据及 gold chunk 映射需另行准备。

未定：API 服务商/base URL、LLM 与 embedding、预算、首批数据和问题 IDs、论文忠实或公开代码协议。论文数据子集、gold supporting evidence→chunk 映射与检索评测脚本仍需确认。

## 下一步

1. 完成 GitHub 登录、首次提交与私有仓库上传，记录远程地址。
2. 确认 API、数据和统一协议，做小规模 TH-RAG baseline。
3. 使用同一抽取缓存比较原版、仅规范化、仅预筛、完整核心，检查合并精度与不变量，报告质量和成本；成功后再补规模与外部基线实验。

## 按需阅读

论文事实：`docs/thrag_paper_brief.md`；方案：`docs/idea_digest.md`；代码、数据和协议：`docs/resource_audit.md`。来源与上游版本放在 `docs/sources/`，默认不读全文或重读 `note.md`。实验记录需包含 commit、配置、模型、数据 IDs、缓存和结果路径，密钥不入库。
