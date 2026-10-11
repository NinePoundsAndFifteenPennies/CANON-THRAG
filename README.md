# CANON-THRAG

基于 TH-RAG 的研究工作区。会话接续先读 [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md)。

<a id="documents"></a>
## 文档导航与维护职责

每类内容只在下列指定位置详细维护；其他文档用相对链接引用，不复制配置表、操作步骤或实验结果。

| 要解决的问题 | 唯一详细说明位置 |
|---|---|
| 研究目标、当前进度、已验证结果、待办 | [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) |
| 核心方案、模块边界、实验设计与贡献判断 | [docs/idea_digest.md](docs/idea_digest.md) |
| 论文方法、参数、结果和原文定位 | [docs/thrag_paper_brief.md](docs/thrag_paper_brief.md) |
| 上游/本地实现差异、协议与资源缺口 | [docs/resource_audit.md](docs/resource_audit.md) |
| 各环境变量是否生效、生成参数、并发和重试 | [docs/config_reference.md](docs/config_reference.md) |
| 可复制的配置默认值 | [thrag/.env.example](thrag/.env.example) |
| 服务器安装、填密钥、运行、迁移和排错 | [thrag/SETUP_GUIDE.md](thrag/SETUP_GUIDE.md) |
| 源码入口、pipeline 阶段与目录职责 | [thrag/README.md](thrag/README.md) |
| 原始数据抽样、转换规则和脚本用法 | [scripts/README.md](scripts/README.md) |
| 上游版本与许可证 | [thrag/UPSTREAM.md](thrag/UPSTREAM.md) |

[note.md](note.md) 是用户原始笔记，不随文档整理改写。[docs/sources/](docs/sources/) 保存源文证据和历史快照，按需查阅，不当作当前配置或进度入口。

## 开发与同步

根目录 Git 仓库统一管理整个工作区，`thrag/` 是普通源码目录。两端同步时先检查工作区差异；有未提交改动时先处理，避免直接覆盖。常用命令：

```bash
git pull --ff-only
git add <changed-files>
git commit -m "Describe the change"
git push
```

两端同时修改时可使用独立分支，再审查和合并。提交与推送按当前授权执行。

Git 管理源码、文档、配置模板和小型示例；真实密钥、环境、下载数据、图/向量缓存与实验输出不入库。复现实验的记录要求见 [工作区规则](AGENTS.md)。
