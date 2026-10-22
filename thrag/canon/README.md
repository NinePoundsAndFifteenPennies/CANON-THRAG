# CANON 增强模块

本包读取完整的原始抽取 JSON/KV，独立保存增强版派生产物和结果。原检索入口保留，作者源码对照由用户自行运行。

**增强入口采用“基础优化＋按开关启用的模块”。**同句的完整实体/三元组关联、全部 chunk 来源和子图内精确检索已写入增强版；这些优化不会随研究模块开关关闭。完整开启三个模块时，运行的就是优化与模块的组合。

## 文档

- [服务器运行、模块切换与消融](USAGE.md)
- [方法与模块边界](../../docs/idea_digest.md)
- [配置含义](../../docs/config_reference.md#canon)
- [配置默认值](../.env.example)
- [项目进度与验证状态](../../PROJECT_CONTEXT.md)

## 代码结构

| 位置 | 职责 |
|---|---|
| [core/](core/) | 方法配置、完整来源存储、向量、缓存身份与调用记录 |
| [hierarchy/](hierarchy/) | 父域拆分、保守原型合并、可选软连接 |
| [routing/](routing/) | 现有选择器适配、A 扩散、B 证据卡片、融合 |
| [allocation/](allocation/) | 独立开关的桥梁加权与分支覆盖 |
| [retriever.py](retriever.py) | 组合模块并输出句子、路径与 chunk |
| [pipeline.py](pipeline.py) | 增强版运行入口与逐题断点 |
| [ablations.py](ablations.py) | 生成可选模块任务清单，不自动执行实验 |
| [回归用例](../tests/test_canon.py) | 使用模拟客户端和小型向量的测试 |

内部使用 `(topic_id, subtopic_id)` 路径，区分不同父主题下的同名子主题。原 [GraphRAG](../generate/graph_rag.py) 接受可选的检索器、客户端、配置与 chunk 表注入，原调用方仍使用默认路径。

本包不设置修复版基线预设，不自动比较实验结果或生成修复贡献报告。
