# TH-RAG code guide

This directory contains the locally adapted TH-RAG implementation. For the
upstream revision and license, see [UPSTREAM.md](UPSTREAM.md).

## Documentation

- [Server setup and commands](SETUP_GUIDE.md)
- [Configuration behavior and concurrency](../docs/config_reference.md)
- [Configuration defaults](.env.example)
- [Paper-to-code differences](../docs/resource_audit.md)
- [Paper facts and source references](../docs/thrag_paper_brief.md)
- [Research progress and validation results](../PROJECT_CONTEXT.md)
- [Workspace documentation map](../README.md#documents)

## Pipeline stages

[`pipeline.py`](pipeline.py) resolves requested stages and runs them in the
following order. The pairwise judge is a separate entry point.

| Stage | Responsibility | Implementation |
|---|---|---|
| `graph_construction` | Split source text, extract facts, and write the chunk lookup | [graph_construction.py](index/graph_construction.py) |
| `json_to_gexf` | Convert extraction records to the hierarchical graph | [json_to_gexf.py](index/json_to_gexf.py) |
| `edge_embedding` | Embed source sentences and build FAISS artifacts | [edge_embedding.py](index/edge_embedding.py) |
| `answer_generation_short` | Answer dataset questions in short form | [answer_generation_short.py](generate/answer_generation_short.py) |
| `answer_generation_long` | Answer dataset questions in long form | [answer_generation_long.py](generate/answer_generation_long.py) |
| `evaluation_f1` | Score short answers against the gold QA file | [judge_F1.py](evaluate/judge_F1.py) |
| Separate pairwise judge | Compare two answer files using an LLM | [judge_Ultradomain.py](evaluate/judge_Ultradomain.py) |

Stage aliases:

| Alias | Expansion |
|---|---|
| `graph_build` | `graph_construction json_to_gexf edge_embedding` |
| `answer_generation` | `answer_generation_short answer_generation_long` |
| `evaluation` | `evaluation_f1` |

## Supporting modules

| File or directory | Responsibility |
|---|---|
| [config.py](config.py) | Environment loading, artifact paths, and pipeline state |
| [model_services.py](model_services.py) | Chat client, local text tokenizer, shared CPU E5 encoder |
| [Retriever.py](generate/Retriever.py) | Theme selection and evidence/chunk retrieval |
| [graph_rag.py](generate/graph_rag.py) | Shared answer generation |
| [topic_choice.py](index/topic_choice.py), [subtopic_choice.py](index/subtopic_choice.py) | Select labels from graph candidates |
| [prompt/](prompt/) | Extraction, selection, answer, and evaluation prompts |
| [test_config.py](test_config.py) | Inspect effective settings without displaying the API key |
| [tests/](tests/) | Automated checks |
| [data/test_dataset/](data/test_dataset/) | Minimal input templates |
| [run_pipeline.bat](run_pipeline.bat) | Optional Windows CLI launcher |

The code uses English comments and console messages. Runtime instructions,
input/output layouts, and troubleshooting are maintained only in the
[setup guide](SETUP_GUIDE.md).

<a id="canon"></a>
CANON 增强版的中文说明见 [包内文档](canon/README.md)。
