# CANON-THRAG

基于 ACL 2026 TH-RAG 的研究工作区。目标是先用自己的 API 复现 TH-RAG，再验证 Topic/Subtopic 结构规范化与候选 Top-M 预筛。

## 目录

- `thrag/`：原始 TH-RAG 源码及示例数据；版本见 `thrag/UPSTREAM.md`，原作者说明见 `thrag/README.md`。
- `note.md`：原始研究笔记。
- `PROJECT_CONTEXT.md`：唯一项目进度文件，记录已完成、当前状态和下一步。
- `docs/`：论文、方案和公开实现差异的参考资料。

## 开发与同步

整个工作区由根目录的 Git 仓库管理，`thrag/` 是普通源码目录。修改原始代码时直接在这里提交。

本地与服务器使用同一远程仓库。开始修改前拉取，结束后提交并推送：

```bash
git pull --ff-only
git add <changed-files>
git commit -m "Describe the change"
git push
```

两端同时修改时可各建分支；合并前检查差异，避免强制覆盖对方修改。

## 配置与产物

依赖与 pipeline 说明见 `thrag/README.md`。当前配置为 DeepSeek 官方 API 加服务器本地 CPU E5；在独立服务器环境先安装 `thrag/requirements-cpu.txt`，再安装 `thrag/requirements.txt`。从 `thrag/.env.example` 创建服务器的 `.env` 并填写 `DEEPSEEK_API_KEY`；数据和复现协议待后续确定。

服务器运行数据位于 `/data/panzheng/canon-thrag/data/`，结果位于 `/data/panzheng/canon-thrag/results/`，由 `THRAG_STORAGE_ROOT` 统一指定；子目录与文件名见 `thrag/SETUP_GUIDE.md`。流程状态仍位于代码目录的 `thrag/temp/`。服务器已跑通两条问题的短答示例和 5 个 pytest 测试，正式论文 baseline 尚未开始。

Git 同步源码、文档、配置模板和小型示例数据。密钥、虚拟环境、下载数据、图/向量缓存及实验输出留在各运行机器；实验结论对应的小型指标与配置应另行整理成可提交文件。

原 TH-RAG 源码遵循 Apache-2.0，保留 `thrag/LICENSE` 和上游来源。
