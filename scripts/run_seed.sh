#!/usr/bin/env bash
set -euo pipefail
if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    cat <<'USAGE'
Usage: scripts/run_seed.sh MODEL REVISION DATA_DIR SEED OUTPUT_ROOT

From the repository root, install requirements.txt and pull python:3.11-slim into Docker.
Prepare the SQL and code splits with python -m src.grpo_data sql --help and
python -m src.grpo_data code --help. Use the same split for seeds 2, 42 and 2026.
DATA_DIR must contain sql_train.jsonl, sql_anchor.jsonl, code_train.jsonl and sql_eval.jsonl.

Example:
  scripts/run_seed.sh Qwen/Qwen3-0.6B "$HUB_COMMIT" data/bird 2 runs/bird-qwen

SQL SFT and GRPO use eight visible GPUs on one node by default. Set NPROC_PER_NODE=1
for single-device training. Decoding and evaluation run once after each branch.
The full GRPO batch is 256 SQL and 256 code prompts globally, with 16 responses per prompt.
For a local backbone, use its directory as MODEL and pass an empty REVISION.
Set SPIDER_EVALUATOR_DIR and SPIDER_TEST_SUITE_DB_DIR for official Spider TS.
Set SPIDER_TABLES to tables.json for Spider exact match and install the
tokenizer data using python -m nltk.downloader punkt_tab.
For the verl loss kernel, install requirements-verl.txt and set GRPO_BACKEND=verl.
USAGE
    exit 0
fi
if [[ $# -ne 5 ]]; then
    echo 'Usage: bash scripts/run_seed.sh MODEL REVISION DATA_DIR SEED OUTPUT_ROOT' >&2
    exit 2
fi
model=$1
revision=$2
data=$3
seed=$4
run_root=$5/seed-$seed
workers=${NPROC_PER_NODE:-8}
[[ "$workers" =~ ^[1-9][0-9]*$ ]] || { echo 'NPROC_PER_NODE must be a positive integer.' >&2; exit 2; }
train=(python -m)
if [[ "$workers" != 1 ]]; then
    train=(torchrun --standalone --nnodes=1 --nproc-per-node="$workers" --module)
fi
if [[ ! -d "$model" && ! "$revision" =~ ^[a-fA-F0-9]{40}$ ]]; then
    echo 'For Hub models supply a full immutable revision.' >&2
    exit 2
fi
[[ ! -e "$run_root" ]] || { echo "Run directory already exists: $run_root" >&2; exit 1; }
image=${CODE_IMAGE:-python:3.11-slim}
image_id=$(docker image inspect --format '{{.Id}}' "$image")
evaluation=(--data "$data/sql_eval.jsonl")
if [[ -n "${SPIDER_EVALUATOR_DIR:-}" || -n "${SPIDER_TEST_SUITE_DB_DIR:-}" ]]; then
    evaluation+=(--spider-evaluator-dir "${SPIDER_EVALUATOR_DIR:?set both Spider paths}")
    evaluation+=(--spider-test-suite-db-dir "${SPIDER_TEST_SUITE_DB_DIR:?set both Spider paths}")
fi
if [[ -n "${SPIDER_TABLES:-}" ]]; then
    : "${SPIDER_EVALUATOR_DIR:?SPIDER_TABLES requires the official Spider evaluator}"
    evaluation+=(--spider-tables "$SPIDER_TABLES")
fi
shared=(--model "$model" --data-dir "$data" --seed "$seed")
if [[ -n "$revision" ]]; then shared+=(--model-revision "$revision"); fi
if [[ -n "${LORA_RANK:-}" ]]; then shared+=(--rank "$LORA_RANK"); fi
mkdir -p "$run_root"
printf '%s\n' "$image_id" > "$run_root/verifier-image.txt"
git rev-parse HEAD > "$run_root/source-commit.txt"
python -m pip freeze > "$run_root/environment.txt"
"${train[@]}" src.train_sql_sft "${shared[@]}" --output-dir "$run_root/sft"
for branch in sql_only full; do
    dest=$run_root/$branch
    "${train[@]}" src.train_grpo --config configs/revised_grpo.json "${shared[@]}" \
        --branch "$branch" --backend "${GRPO_BACKEND:-torch}" --code-image "$image_id" \
        --sft-checkpoint "$run_root/sft/common_sql_trainable.pt" --output-dir "$dest"
    python -m src.train_grpo "${shared[@]}" --sft-checkpoint "$dest/trainable.pt" \
        --eval-file "$data/sql_eval.jsonl" --predict "$dest/predictions.jsonl"
    python -m src.evaluate_grpo "${evaluation[@]}" --predictions "$dest/predictions.jsonl" \
        --out-dir "$dest/evaluation"
done
