from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def _read_scored(path: Path, comparison: dict) -> tuple[str, dict]:
    scored = {}
    metric = comparison.get("metric")
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            qid = row["id"]
            if not isinstance(qid, str) or not qid.strip() or qid in scored:
                raise ValueError(f"{path}:{number}: missing or duplicate question ID {qid!r}")
            if not isinstance(row.get("correct"), bool):
                raise ValueError(f"{path}:{number}: expected a boolean correct field")
            if not isinstance(row.get("metric"), str) or not row["metric"].strip():
                raise ValueError(f"{path}:{number}: missing evaluation metric")
            if metric is None:
                metric = row["metric"]
            if row["metric"] != metric or row.get("dataset") != comparison["dataset"]:
                raise ValueError(f"{path}:{number}: dataset or metric mismatch")
            for key in ("model", "seed"):
                if key in row and row[key] != comparison[key]:
                    raise ValueError(f"{path}:{number}: {key} disagrees with manifest")
            scored[qid] = (row["correct"], row.get("gold_sql"), row.get("db_id"))
    if not scored:
        raise ValueError(f"{path}: empty predictions")
    return metric, scored


def exact_mcnemar(full_only: int, baseline_only: int) -> float:
    n = full_only + baseline_only
    k = min(full_only, baseline_only)
    if 2 * k == n:
        return 1.
    term = 1
    tail = term
    for j in range(1, k + 1):
        term = term * (n - j + 1) // j
        tail += term
    return 2 * tail / (1 << n)


def holm_adjust(p_values: list[float]) -> list[float]:
    adjusted = [0.] * len(p_values)
    floor = 0.
    for rank, index in enumerate(sorted(range(len(p_values)), key=p_values.__getitem__)):
        floor = max(floor, min(1., (len(p_values) - rank) * p_values[index]))
        adjusted[index] = floor
    return adjusted


def compare(comparison: dict, directory: Path, samples: int, seed: int):
    outputs = [
        _read_scored(directory / comparison[key], comparison)
        for key in ("baseline_predictions", "full_predictions")
    ]
    (metric, baseline), (other_metric, full) = outputs
    if metric != other_metric or baseline.keys() != full.keys():
        raise ValueError("Paired evaluation requires the same metric and exactly the same question IDs")

    counts = {"both_correct": 0, "full_only_correct": 0, "baseline_only_correct": 0, "both_wrong": 0}
    for key, (f, full_sql, full_db) in full.items():
        b, base_sql, base_db = baseline[key]
        if (full_sql != base_sql and None not in (base_sql, full_sql)) or (
            full_db != base_db and None not in (base_db, full_db)
        ):
            raise ValueError(f"Gold SQL or database identity differs for {key!r}")
        category = ("both_correct" if b else "full_only_correct") if f else (
            "baseline_only_correct" if b else "both_wrong"
        )
        counts[category] += 1

    family_key = tuple(comparison[key] for key in ("model", "dataset", "baseline", "seed")) + (metric,)
    digest = hashlib.sha256(json.dumps([seed, *family_key]).encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:16], "big"))
    a, b = counts["full_only_correct"], counts["baseline_only_correct"]
    n = len(full)
    bootstrap = rng.multinomial(n, [a/n, b/n, (n-a-b)/n], size=samples)
    lo, hi = np.percentile(100. * (bootstrap[:, 0] - bootstrap[:, 1]) / n, [2.5, 97.5])
    full_digest = hashlib.sha256()
    for qid in sorted(full):
        full_digest.update(json.dumps([qid, full[qid][0]], ensure_ascii=True).encode() + b"\n")
    return family_key, {
        "full_outcome_sha256": full_digest.hexdigest(),
        **{k: comparison[k] for k in ("model", "dataset", "baseline", "seed")}, "metric": metric,
        "n": n, **counts, "baseline_accuracy_pct": 100. * (counts["both_correct"] + b) / n,
        "full_accuracy_pct": 100. * (counts["both_correct"] + a) / n,
        "difference_pp": 100. * (a - b) / n, "difference_ci95_pp": [float(lo), float(hi)],
        "mcnemar_p": exact_mcnemar(a, b),
    }


def analyze(manifest: Path, samples: int = 10000, random_seed: int = 42) -> dict:
    if samples < 100 or random_seed < 0:
        raise ValueError("Use at least 100 bootstrap samples and a nonnegative random seed")
    family = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(family, list) or not family:
        raise ValueError("Family manifest must be a nonempty JSON array")
    results, keys = [], set()
    used_files, full_by_seed = {}, {}
    for entry in family:
        if not isinstance(entry, dict) or not all(isinstance(entry.get(k), str) and entry[k].strip()
                for k in ("model", "dataset", "baseline", "baseline_predictions", "full_predictions")):
            raise ValueError("Each comparison needs model, dataset, baseline, and both prediction paths")
        if type(entry.get("seed")) is not int:
            raise ValueError("Specify one integer training seed per comparison")
        key, result = compare(entry, manifest.parent, samples, random_seed)
        for file_key in ("baseline_predictions", "full_predictions"):
            path = (manifest.parent / entry[file_key]).resolve()
            if used_files.setdefault(path, entry["seed"]) != entry["seed"]:
                raise ValueError(f"One scored prediction file was reused for different seeds: {path}")
        full_key = (entry["model"], entry["dataset"], entry["seed"], result["metric"])
        if full_by_seed.setdefault(full_key, result["full_outcome_sha256"]) != result["full_outcome_sha256"]:
            raise ValueError("Full-method outcomes differ across baselines for the same model, dataset and seed")
        if key in keys:
            raise ValueError(f"Duplicate test in Holm family: {key}")
        keys.add(key)
        results.append(result)
    for result, p in zip(results, holm_adjust([r["mcnemar_p"] for r in results])):
        result["holm_p"] = p
    grouped = {}
    for row in results:
        key = tuple(row[k] for k in ("model", "dataset", "baseline", "metric"))
        grouped.setdefault(key, []).append(row)
    summaries = []
    for key, rows in grouped.items():
        summary = dict(zip(("model", "dataset", "baseline", "metric"), key))
        summary["seeds"] = sorted(row["seed"] for row in rows)
        for field in ("baseline_accuracy_pct", "full_accuracy_pct", "difference_pp"):
            values = [row[field] for row in rows]
            summary[field] = {"mean": float(np.mean(values)),
                              "sample_sd": float(np.std(values, ddof=1)) if len(values) > 1 else None}
        summaries.append(summary)
    return {
        "holm_family_size": len(results), "bootstrap_samples": samples, "random_seed": random_seed,
        "interval": ("Per-seed 95% percentile interval for full minus baseline accuracy in percentage "
                     "points. Each replicate resamples paired questions. It conditions on the fitted models. "
                     "Database clustering, variability across training seeds and simultaneous coverage "
                     "are outside this interval."),
        "comparisons": results, "across_seeds": summaries,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compare two models' scored questions for each independent training seed.\n\nUsage: python -m src.paired_eval FAMILY.json --output paired.json\nFAMILY.json is a JSON array. Each entry specifies model, dataset, baseline, seed,\nbaseline_predictions, and full_predictions. Prediction paths refer to scored\nquestions.jsonl files and are relative to the manifest. Set metric explicitly\nto require e.g. test-suite accuracy for Spider.\n", formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args(argv)
    result = analyze(args.manifest, args.bootstrap_samples, args.random_seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as out:
        json.dump(result, out, indent=2, allow_nan=False)
        out.write("\n")


if __name__ == "__main__":
    main()
