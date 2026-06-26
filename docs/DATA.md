# Data & Workload Reproduction

Datasets and model weights are **not** shipped with this repository (they exceed
GitHub size limits and carry their own licenses). This document explains how to
obtain them and regenerate the serving workloads.

## 1. Datasets

EnergyServe is evaluated on two synthetic instruction-following benchmarks and one
real-world production trace:

| Dataset       | Source (Hugging Face)                          | Role                                  |
| :------------ | :--------------------------------------------- | :------------------------------------ |
| Alpaca        | `tatsu-lab/alpaca`                             | Synthetic instruction following       |
| Dolly-15k     | `databricks/databricks-dolly-15k`              | Synthetic instruction following       |
| LMSYS-Chat-1M | `lmsys/lmsys-chat-1m`                          | Real-world human–AI conversation trace |

```bash
# Example: download with the Hugging Face CLI
huggingface-cli download databricks/databricks-dolly-15k --repo-type dataset \
    --local-dir data/raw/databricks-dolly-15k
```

> LMSYS-Chat-1M is gated; request access on its Hugging Face page first.

## 2. Expected Layout

The scripts assume the following directory structure under `data/` (ignored by git):

```text
data/
├── raw/                         # As downloaded from Hugging Face
├── processed/                   # Cleaned prediction files
│   ├── alpaca_pred_test.jsonl   #   each line: {"prompt": ..., "output_len": ...}
│   ├── dolly_pred_test.jsonl
│   └── lmsys_pred_test.jsonl
└── workload/                    # Generated serving workloads (see step 4)
    ├── alpaca_workload.jsonl
    ├── dolly_workload.jsonl
    └── lmsys_workload.jsonl
```

Each `*_pred_test.jsonl` record holds at least a `prompt` string and an
`output_len` integer (the reference generation length used for the Oracle SLO).

## 3. Preprocessing

Clean and tokenize each raw dataset into the `data/processed/` format above
(English-only filtering, prompt extraction, output-length labelling). Bucketed
output-length labels follow DynamoLLM: input S/M/L at `<256 / <1024 / <8192`
tokens and output S/M/L at `<100 / <350 / >350` tokens.

## 4. Generate Serving Workloads

`generate_workload.py` samples requests and assigns Poisson arrival times
(default λ = 50 req/s, 2000 requests):

```bash
python scripts/generate_workload.py --dataset lmsys --rate 50 --num_requests 2000
```

This reads `data/processed/<dataset>_pred_test.jsonl` and writes
`data/workload/<dataset>_workload.jsonl` with records of the form:

```json
{"prompt": "...", "output_len": 128, "arrival_time": 12.34}
```

The serving scripts (`run_serving.py`, `run_baselines.py`) load these workload
files via the `workloads` section of `configs/core_config.yaml`.

## 5. Model Weights

| Component        | Model                          | Notes                                   |
| :--------------- | :----------------------------- | :-------------------------------------- |
| Serving engine   | `Meta-Llama-3-8B-Instruct`     | Set `system.base_model_path` in config  |
| Resource profiler| `google/gemma-2b` + LoRA       | Trained via `scripts/train_predictor.py`|

LoRA adapters are written to `checkpoints/` (git-ignored). Download base weights
from Hugging Face and point the config at the local path.
