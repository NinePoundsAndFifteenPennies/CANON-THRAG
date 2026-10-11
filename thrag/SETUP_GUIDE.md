# TH-RAG server setup and operation

This document owns installation, credential entry, dataset layout, execution,
and migration procedures. Parameter semantics are in the
[configuration reference](../docs/config_reference.md); values come from
[.env.example](.env.example). Actual completion and test results belong only in
[PROJECT_CONTEXT.md](../PROJECT_CONTEXT.md).

Run the commands below in the server environment. Do not execute the project
on the local machine without its dependencies.

<a id="environment"></a>
## 1. Prepare the environment and credentials

Enter the server checkout; adjust the prefix if the repository is elsewhere:

```bash
cd /home/panzheng/projects/CANON-THRAG/thrag
conda activate canon-thrag
```

If that environment does not exist, create a Python 3.10+ environment first.
A standalone alternative is:

```bash
python -m venv .venv
source .venv/bin/activate
```

For a new environment, install CPU PyTorch before the runtime dependencies:

```bash
pip install -r requirements-cpu.txt
pip install -r requirements.txt
```

If an inherited pip extra index interferes, use this server-specific fallback
for the runtime dependencies, without changing global pip configuration:

```bash
PIP_CONFIG_FILE=/dev/null python -m pip install --index-url https://pypi.org/simple -r requirements.txt
```

Install optional development dependencies with `pip install -e .[dev]`.

Create the private environment file if absent:

```bash
cp -n .env.example .env
```

Edit `.env` in this `thrag/` directory and replace the `LLM_API_KEY` placeholder
with the real service key. For an existing file, update its model, endpoint,
thinking, tokenizer, and generation fields from the latest template; preserve
actual storage and E5 path overrides. Also update any previously exported
settings; see [configuration loading](../docs/config_reference.md#loading).
Never put the real key in the template or commit it.

Sync the complete local tokenizer directory selected by the template. Its
model weights remain on the inference server. The E5 model must already exist
at its configured local path; [tokenizer and embedding requirements](../docs/config_reference.md#tokenization)
are maintained separately.

Check the resolved paths and settings:

```bash
python test_config.py
```

<a id="dataset"></a>
## 2. Prepare input files

`<storage_root>` below means the resolved `THRAG_STORAGE_ROOT` displayed by the
configuration check. Use [data preparation scripts](../scripts/README.md) for
benchmark-specific sampling, conversion, and metadata rules.

```text
<storage_root>/
|-- data/
|   `-- <dataset>/
|       |-- contexts.txt
|       `-- qa.json
`-- results/
    |-- index/
    |-- generated/
    |-- chunks/
    `-- evaluated/
```

`contexts.txt` is UTF-8 source text. `qa.json` is a JSON array with `query` and
`answer` fields:

```json
[
  {"query": "Which component builds the graph?", "answer": "Graph construction"}
]
```

Graph building requires the text file; answering and F1 evaluation require the
QA file. For a minimal example, copy `qa.json` and `contexts_example.txt` from
[data/test_dataset/](data/test_dataset/) into the runtime dataset directory,
renaming the latter to `contexts.txt`. Do not overwrite an existing benchmark
dataset with these examples.

<a id="run"></a>
## 3. Run the pipeline

List available datasets:

```bash
python pipeline.py --list-datasets
```

Build the graph, generate short answers, and evaluate them:

```bash
python pipeline.py --dataset my_dataset --steps graph_build answer_generation_short evaluation
```

Replace `my_dataset` with the runtime dataset directory name. Use individual
stage names or aliases from the [pipeline map](README.md#pipeline-stages) to
run a subset. For long answers:

```bash
python pipeline.py --dataset my_dataset --steps answer_generation_long
```

Omitting `--steps` runs all canonical stages, including both answer modes.
[`index/build_graph.py`](index/build_graph.py) is an alternative graph-only
wrapper; [`run_pipeline.bat`](run_pipeline.bat) is the optional Windows launcher.

For pairwise LLM evaluation, replace the example file paths with actual files:

```bash
python evaluate/judge_Ultradomain.py \
  --answer-a /absolute/path/my_dataset_answers_long.json \
  --answer-b /absolute/path/baseline_answers.json \
  --output /absolute/path/my_dataset_pairwise.json \
  --label-a TH-RAG \
  --label-b Baseline
```

Add `--plot /absolute/path/my_dataset_pairwise.png` to save a chart. Score
semantics and exclusions are documented in the [implementation audit](../docs/resource_audit.md#evaluation).

<a id="outputs"></a>
## 4. Locate outputs

All paths in this table are relative to `<storage_root>/results/`:

| Directory | Files |
|---|---|
| `index/` | `<dataset>_graph.json`, `<dataset>_kv_store.json`, `<dataset>_graph.gexf`, `<dataset>_edge_index.faiss`, `<dataset>_edge_payloads.npy` |
| `generated/` | `<dataset>_answers_short.json`, `<dataset>_answers_long.json` |
| `chunks/` | `<dataset>_chunks_short.jsonl`, `<dataset>_chunks_long.jsonl` |
| `evaluated/` | `<dataset>_eval_f1.json`; pairwise output uses the explicit CLI path |

Pipeline bookkeeping remains at `thrag/temp/pipeline_state.json` in the code
checkout. The tokenizer also remains in the checkout, outside runtime storage.

<a id="migration"></a>
## 5. Reuse or rebuild artifacts

The pipeline primarily checks whether expected files exist. It does not detect
changes to model, tokenizer, extraction prompt, or configuration automatically.
Changing any of these can invalidate downstream artifacts; tokenizer changes
can alter chunk boundaries even when the chunk-size settings stay unchanged.

Archive old results that must be retained, then force a fresh short-answer run:

```bash
python pipeline.py --dataset my_dataset --steps graph_build answer_generation_short evaluation --force
```

This replaces that dataset's extraction, chunk map, graph, embeddings, short
answers, and F1 results. Add `answer_generation_long` when long answers are also
needed. To rerun every canonical stage, omit `--steps` and retain `--force`.
Do not combine rebuilt graph artifacts with old answers or chunk mappings.

The environment cache flags do not currently control this behavior; their
status is listed in [unused system settings](../docs/config_reference.md#legacy).

<a id="troubleshooting"></a>
## 6. Checks and troubleshooting

Run automated checks only in the configured server environment:

```bash
python -m pytest -q
```

| Symptom | Where to check |
|---|---|
| Missing or rejected API key | The private `.env` created in [step 1](#environment); match the configured service and key |
| Wrong model or tokenizer | Template values and the [configuration reference](../docs/config_reference.md#paths) |
| Dataset not found | Resolved storage path and [input layout](#dataset) |
| Existing outputs skipped | [Reuse and rebuild rules](#migration) |
| FAISS input missing or wrong dimension | Graph-building stages and [embedding requirements](../docs/config_reference.md#tokenization) |
| Slow requests, queueing, or timeouts | [Concurrency and retry behavior](../docs/config_reference.md#concurrency); inspect server capacity before increasing parallelism |
