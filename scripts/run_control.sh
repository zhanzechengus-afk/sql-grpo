#!/usr/bin/env bash
set -euo pipefail
if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    cat <<'USAGE'
Usage: scripts/run_control.sh MODE MODEL REVISION DATA_DIR SEED SFT_CHECKPOINT OUT_DIR [SOURCE_RL_DIR]

Baselines: sql-only, full, zero-code, naive, retention, pcgrad, sql-only-double-prompts
Ablations: no-projection, no-norm-match, no-gate, no-kl
Sensitivity: weight-024, weight-036, kl-0008, kl-0012, gate-016, gate-024
Capacity: table2-shared-r8, table2-adapter-r8, table2-adapter-r16
Interpolation: alpha-06, alpha-08, alpha-10 (also supply the saved post-RL run directory)

Use an SFT checkpoint with the same seed, backbone, adapter configuration and training split.
USAGE
    exit 0
fi
if [[ $# -lt 7 || $# -gt 8 ]]; then
    echo 'Usage: run_control.sh MODE MODEL REVISION DATA_DIR SEED SFT_CHECKPOINT OUT_DIR [SOURCE_RL_DIR]' >&2
    exit 2
fi
mode=$1 model=$2 revision=$3 data=$4 seed=$5 sft=$6 dest=$7
[[ -f "$sft" && ! -e "$dest" ]] || { echo 'Supply an existing SFT checkpoint and a new output directory.' >&2; exit 2; }
[[ -d "$model" || "$revision" =~ ^[0-9a-fA-F]{40}$ ]] || { echo 'Pin Hub models to the full commit ID.' >&2; exit 2; }
base=(--model "$model" --data-dir "$data" --seed "$seed")
if [[ -n "$revision" ]]; then base+=(--model-revision "$revision"); fi
arch=$(python - "$sft" "$model" "$revision" "$data" "$seed" <<'PY'
import json, sys
from pathlib import Path
from src.grpo_data import digest
checkpoint, model, revision, data, seed = sys.argv[1:]
meta = json.loads((Path(checkpoint).parent/"sft_config.json").read_text())
if meta["model"] != model or (meta.get("model_revision") or "") != revision or meta["seed"] != int(seed):
    sys.exit("SFT model, revision or seed does not match this run")
if digest(Path(data)/"sql_train.jsonl") != meta["input_sha256"]:
    sys.exit("SFT training partition does not match")
print(meta["rank"], meta.get("adapter_type", "output"), meta["adapter_dim"], meta.get("adapter_top_k", 6))
PY
)
read -r rank adapter dim top_k <<< "$arch"
base+=(--rank "$rank" --adapter-type "$adapter" --adapter-dim "$dim" --adapter-top-k "$top_k")
branch=full
extra=()
alpha=
case "$mode" in
    full) ;;
    sql-only) branch=sql_only ;;
    zero-code) branch=zero_code ;;
    naive) extra=(--no-project --no-norm-match --no-gate --kl 0 --alpha 1) ;;
    retention) extra=(--no-project --no-norm-match --no-gate) ;;
    pcgrad) extra=(--project --no-norm-match --no-gate) ;;
    no-projection) extra=(--no-project) ;;
    no-norm-match) extra=(--no-norm-match) ;;
    no-gate) extra=(--no-gate) ;;
    no-kl) extra=(--kl 0) ;;
    sql-only-double-prompts) branch=sql_only; extra=(--sql-prompts 512) ;;
    weight-024) extra=(--code-weight 0.24) ;;
    weight-036) extra=(--code-weight 0.36) ;;
    kl-0008) extra=(--kl 0.0008) ;;
    kl-0012) extra=(--kl 0.0012) ;;
    gate-016) extra=(--gate-denominator 0.016) ;;
    gate-024) extra=(--gate-denominator 0.024) ;;
    table2-shared-r8|table2-adapter-r8|table2-adapter-r16)
        expected_rank=8 expected_adapter=output
        if [[ "$mode" == table2-adapter-r16 ]]; then expected_rank=16; fi
        if [[ "$mode" == table2-shared-r8 ]]; then expected_adapter=none; fi
        if [[ "$rank" != "$expected_rank" || "$adapter" != "$expected_adapter" || ("$adapter" == output && "$dim" != 96) ]]; then
            echo "SFT checkpoint architecture is incompatible with $mode" >&2; exit 2
        fi
        extra=(--kl 0 --alpha 1) ;;
    alpha-06) alpha=0.6 ;;
    alpha-08) alpha=0.8 ;;
    alpha-10) alpha=1.0 ;;
    *) echo "Unknown control: $mode" >&2; exit 2 ;;
esac

evaluation=(--data "$data/sql_eval.jsonl")
if [[ -n "${SPIDER_EVALUATOR_DIR:-}" || -n "${SPIDER_TEST_SUITE_DB_DIR:-}" ]]; then
    evaluation+=(--spider-evaluator-dir "${SPIDER_EVALUATOR_DIR:?set both Spider paths}"
                 --spider-test-suite-db-dir "${SPIDER_TEST_SUITE_DB_DIR:?set both Spider paths}")
fi
if [[ -n "${SPIDER_TABLES:-}" ]]; then
    : "${SPIDER_EVALUATOR_DIR:?SPIDER_TABLES requires the official Spider evaluator}"
    evaluation+=(--spider-tables "$SPIDER_TABLES")
fi
if [[ -n "$alpha" ]]; then
    [[ $# -eq 8 ]] || { echo 'Alpha evaluation also requires SOURCE_RL_DIR.' >&2; exit 2; }
    source_run=$8
    source_info=$(python - "$source_run" "$sft" "$seed" "$alpha" <<'PY'
import json, sys
from pathlib import Path
from src.grpo_data import digest
run, checkpoint, seed, alpha = sys.argv[1:]
meta = json.loads((Path(run)/"resolved_config.json").read_text())
sft_hash = digest(Path(checkpoint))
if meta["seed"] != int(seed) or meta["sha256"]["sft_checkpoint"] != sft_hash:
    sys.exit("Post-RL checkpoint was trained with a different seed or SFT initialization")
print(json.dumps(dict(source_grpo_dir=str(Path(run).resolve()), seed=int(seed),
                      sft_sha256=sft_hash, alpha=float(alpha)), indent=2))
PY
)
    mkdir -p "$(dirname "$dest")"
    mkdir "$dest"
    printf '%s\n' "$source_info" > "$dest/interpolation_config.json"
    checkpoint="$dest/interpolated_for_eval.pt"
    python -m src.interpolate_checkpoint --sft "$sft" --rl "$source_run/post_rl_trainable.pt" \
        --alpha "$alpha" --output "$checkpoint"
else
    [[ $# -eq 7 ]] || { echo 'Only alpha modes accept a source RL directory.' >&2; exit 2; }
    if [[ "$branch" != sql_only ]]; then
        image_id=$(docker image inspect --format '{{.Id}}' "${CODE_IMAGE:-python:3.11-slim}")
        extra+=(--code-image "$image_id")
    fi
    python -m src.train_grpo --config configs/revised_grpo.json "${base[@]}" \
        --branch "$branch" --backend "${GRPO_BACKEND:-torch}" --sft-checkpoint "$sft" \
        --output-dir "$dest" ${extra[@]+"${extra[@]}"}
    checkpoint="$dest/trainable.pt"
fi
python -m src.train_grpo "${base[@]}" --sft-checkpoint "$checkpoint" \
    --eval-file "$data/sql_eval.jsonl" --predict "$dest/predictions.jsonl"
python -m src.evaluate_grpo "${evaluation[@]}" --predictions "$dest/predictions.jsonl" --out-dir "$dest/evaluation"
