# TH-RAG

Reference implementation for the ACL 2026 paper **TH-RAG: Topic-Based Hierarchical Knowledge Graphs for Robust Multi-hop Reasoning in Graph-based RAG Systems**.
You can check the paper here: [Paper](https://aclanthology.org/2026.acl-long.1740/)

This repository contains the end-to-end research pipeline used to:

1. build hierarchical knowledge-graph artifacts from raw contexts,
2. index graph evidence with FAISS,
3. generate short-form and long-form answers, and
4. evaluate model outputs.

This workspace uses the official DeepSeek API for chat and a local CPU-only
E5-base-v2 model for embeddings. See `.env.example` for the current settings.

## Repository Status

The codebase has been cleaned for public release with these constraints:

- English-only comments, documentation, and user-facing messages
- no decorative emoji or informal console output
- removal of empty placeholder files and unrelated cleanup scripts
- unified configuration and path handling across graph construction, retrieval, generation, and evaluation

## Requirements

- Python 3.10+
- A DeepSeek API key exposed through `DEEPSEEK_API_KEY`
- Local E5-base-v2 weights and tokenizer files, configured with `E5_MODEL_PATH`
- Platform support: Windows, Linux, and macOS through the Python CLI

Install dependencies:

```bash
pip install -r requirements-cpu.txt
pip install -r requirements.txt
```

Optionally, install the project metadata and test dependencies:

```bash
pip install -e .[dev]
```

## Dataset Format

Each runtime dataset lives under `/data/panzheng/canon-thrag/data/<dataset_name>/`
on the server. `THRAG_STORAGE_ROOT` selects the parent of `data/` and `results/`.
Linux defaults to `/data/panzheng/canon-thrag`; other platforms default to the
`thrag/` directory when the setting is absent. Relative values resolve against
`thrag/`, so `THRAG_STORAGE_ROOT=.` uses repository-local storage.

Required files:

- `contexts.txt`: source passages used to build the hierarchical graph
- `qa.json`: evaluation questions and gold answers

Expected `qa.json` format:

```json
[
  {
    "query": "What does TH-RAG use to index graph evidence?",
    "answer": "FAISS"
  }
]
```

The repository includes example templates at `data/test_dataset/`. Copy them
to the runtime data directory before running the example.

## Quick Start

1. Copy the environment template.

```bash
cp .env.example .env
```

2. Set `DEEPSEEK_API_KEY` in `.env`. The default official endpoint is
   `https://api.deepseek.com`, and the model name is `deepseek-flash`.

3. Prepare the example in the server storage directory. If the dataset already
   exists there, use those files. See `SETUP_GUIDE.md` for relocating existing
   runtime data and results.

```bash
mkdir -p /data/panzheng/canon-thrag/data/test_dataset
cp -n data/test_dataset/qa.json /data/panzheng/canon-thrag/data/test_dataset/
cp -n data/test_dataset/contexts_example.txt \
  /data/panzheng/canon-thrag/data/test_dataset/contexts.txt
```

4. Validate the configuration and effective storage paths.

```bash
python test_config.py
```

5. List available datasets.

```bash
python pipeline.py --list-datasets
```

6. Run the short-answer example.

```bash
python pipeline.py --dataset test_dataset --steps graph_build answer_generation_short evaluation
```

## Pipeline Stages

### Model configuration

- `DEFAULT_MODEL`, `CHAT_MODEL`, and `EVAL_MODEL` select the DeepSeek chat models.
- `DEEPSEEK_THINKING=disabled` explicitly selects non-thinking mode for this
  first version; set it to `enabled` if needed. Temperature is ignored by the
  official API when thinking is enabled.
- `E5_MODEL_PATH` defaults to `/data/panzheng/bearrag/models/e5-base-v2`.
  Sentence-Transformers loads it with `device="cpu"` and `local_files_only=True`,
  including the local BERT WordPiece tokenizer. No model download is attempted.
- `E5_MAX_LENGTH=512` controls input truncation; `E5_BATCH_SIZE=32` controls CPU
  encoding batches. Queries use `query: ` and evidence sentences use `passage: `.
  All embeddings are normalized float32 vectors with 768 dimensions.
- `DEEPSEEK_TOKENIZER_PATH` selects the bundled official V4.1 Flash tokenizer
  for document chunking and raw context token counts. It loads locally with
  AutoTokenizer and is separate from E5 WordPiece. API usage includes message
  formatting and is authoritative for request totals and billing.

Only tokenizer files (about 6 MB) and their license are bundled, not DeepSeek
weights. Their fixed source revision and hashes are recorded in
`tokenizers/deepseek-v4.1-flash/SOURCE.json`. Changing from the old OpenAI
tokenization changes chunk boundaries; rebuild graph/embedding/answer caches
when preparing data. Chunk-size and retrieval-budget numbers are unchanged.

Use a fresh environment and install the CPU-only PyTorch requirement first.
Previously generated embedding indexes must be rebuilt after changing the
embedding model; a non-768-dimensional index is rejected when loaded.

### Pipeline steps

The unified pipeline exposes the following canonical steps:

- `graph_construction`: chunk `contexts.txt`, extract triples, and create the chunk KV store
- `json_to_gexf`: convert extracted triples into a hierarchical GEXF graph
- `edge_embedding`: embed predicate-edge evidence and build the FAISS index
- `answer_generation_short`: produce concise answers
- `answer_generation_long`: produce detailed answers
- `evaluation_f1`: score the short-answer output against the gold answers in `qa.json`

You can also use step aliases:

- `graph_build` -> `graph_construction json_to_gexf edge_embedding`
- `answer_generation` -> `answer_generation_short answer_generation_long`
- `evaluation` -> `evaluation_f1`

Examples:

```bash
python pipeline.py --dataset test_dataset --steps graph_build
python pipeline.py --dataset test_dataset --steps answer_generation_short
python pipeline.py --dataset test_dataset --steps evaluation
python pipeline.py --dataset test_dataset --steps graph_build answer_generation_short --force
```

## Output Layout

Server artifacts are written under `/data/panzheng/canon-thrag/results/`.
The directory names and artifact filenames are:

- `/data/panzheng/canon-thrag/results/index/`: extracted graph JSON, KV store, GEXF graph, FAISS index, and payloads
- `/data/panzheng/canon-thrag/results/generated/`: model answers
- `/data/panzheng/canon-thrag/results/chunks/`: chunk usage logs for answer generation
- `/data/panzheng/canon-thrag/results/evaluated/`: evaluation summaries
- `thrag/temp/`: pipeline state bookkeeping in the code checkout

`THRAG_STORAGE_ROOT` affects only `data/` and `results/`. Pipeline state and the
bundled DeepSeek tokenizer stay in the code checkout. Steps reuse the expected
artifacts found under the configured result directory; file contents and
filenames can be preserved when relocating existing artifacts.

## Pairwise Evaluation

For pairwise LLM-based comparison between two answer files, use the UltraDomain-style evaluator:

```bash
python evaluate/judge_Ultradomain.py \
  --answer-a /data/panzheng/canon-thrag/results/generated/test_dataset_answers_long.json \
  --answer-b other_system_answers.json \
  --output /data/panzheng/canon-thrag/results/evaluated/test_dataset_pairwise.json \
  --label-a TH-RAG \
  --label-b Baseline \
  --plot /data/panzheng/canon-thrag/results/evaluated/test_dataset_pairwise.png
```

## Windows Helper

A menu-driven Windows launcher is available:

```cmd
run_pipeline.bat
```

## Repository Structure

```text
THRAG/
|-- config.py
|-- pipeline.py
|-- test_config.py
|-- README.md
|-- SETUP_GUIDE.md
|-- requirements.txt
|-- pyproject.toml
|-- run_pipeline.bat
|-- data/
|-- index/
|   |-- build_graph.py
|   |-- graph_construction.py
|   |-- json_to_gexf.py
|   |-- edge_embedding.py
|   |-- topic_choice.py
|   |-- subtopic_choice.py
|-- generate/
|   |-- Retriever.py
|   |-- graph_rag.py
|   |-- graph_based_rag_short.py
|   |-- graph_based_rag_long.py
|   |-- answer_generation_short.py
|   |-- answer_generation_long.py
|-- evaluate/
|   |-- judge_F1.py
|   |-- judge_Ultradomain.py
|-- prompt/
|-- tests/
```

## Notes

- The pipeline expects `qa.json` to contain gold answers if you want to run `evaluation_f1`.
- `answer_generation_short` is the output consumed by the F1 evaluator by default.
- Public-release cleanup removed unfinished wrappers and utility files that were not part of the research pipeline.

## License

Apache License 2.0
