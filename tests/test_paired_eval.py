import json
import random

import pytest

from src.paired_eval import analyze, exact_mcnemar, holm_adjust


def family(tmp_path):
    rows = [{"id":str(i), "dataset":"bird", "metric":"execution", "gold_sql":f"SELECT {i}"}
            for i in range(10)]
    for name, correct in [("base", {4,5,6,7}), ("full", {0,1,2,3,5,6,7})]:
        scored = [dict(row, correct=i in correct) for i, row in enumerate(rows)]
        random.Random(12 if name=="full" else 19).shuffle(scored)
        (tmp_path/(name+".jsonl")).write_text("".join(json.dumps(row)+"\n" for row in scored))
    entry = {"model":"m", "baseline":"sql-only", "dataset":"bird", "seed":2,
             "baseline_predictions":"base.jsonl", "full_predictions":"full.jsonl"}
    path = tmp_path/"family.json"
    path.write_text(json.dumps([entry, dict(entry, baseline="control")]))
    return path


def test_paired_question_ids_and_family_statistics(tmp_path):
    p = family(tmp_path)
    report = analyze(p)
    r = report["comparisons"][0]
    assert report["holm_family_size"] == 2
    assert [r[k] for k in ("both_correct","full_only_correct","baseline_only_correct","both_wrong")] == [3,4,1,2]
    assert (r["baseline_accuracy_pct"], r["full_accuracy_pct"], r["difference_pp"]) == (40.,70.,30.)
    assert r["mcnemar_p"] == .375 and r["holm_p"] == .75
    assert r["difference_ci95_pp"][0] < 30. < r["difference_ci95_pp"][1]
    p.write_text(json.dumps(json.loads(p.read_text())[::-1]))
    assert analyze(p)["comparisons"][1] == r


@pytest.mark.parametrize("corruption", ["duplicate","missing","metric","seed","wrong_gold","nonboolean"])
def test_rejects_unpairable_predictions(tmp_path, corruption):
    p = family(tmp_path)
    target = tmp_path/"full.jsonl"
    rows = [json.loads(line) for line in target.read_text().splitlines()]
    if corruption == "duplicate": rows.append(rows[0])
    elif corruption == "missing": rows.pop()
    elif corruption == "metric": rows[0]["metric"] = "spider_single_db_execution"
    elif corruption == "seed": rows[0]["seed"] = 2026
    elif corruption == "wrong_gold": rows[0]["gold_sql"] = "SELECT 99"
    elif corruption == "nonboolean": rows[0]["correct"] = "false"
    target.write_text("".join(json.dumps(row)+"\n" for row in rows))
    with pytest.raises(ValueError):
        analyze(p)


def test_holm_monotonicity_capping_and_empty_discordance():
    assert holm_adjust([.8,.002,.01,.008]) == [.8,.008,.024,.024]
    assert exact_mcnemar(0,0) == exact_mcnemar(2,2) == 1.
    assert exact_mcnemar(0,6) == exact_mcnemar(6,0) == .03125


def test_reports_sample_sd_of_unrounded_seed_scores(tmp_path):
    manifest = family(tmp_path)
    first = json.loads(manifest.read_text())[0]
    second = dict(first, seed=42, full_predictions="full42.jsonl", baseline_predictions="base42.jsonl")
    for name in ("base", "full"):
        rows = [json.loads(line) for line in (tmp_path / f"{name}.jsonl").read_text().splitlines()]
        if name == "full":
            next(row for row in rows if not row["correct"])["correct"] = True
        (tmp_path / f"{name}42.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    manifest.write_text(json.dumps([first, second]))
    summary, = analyze(manifest)["across_seeds"]
    assert summary["seeds"] == [2, 42]
    assert summary["full_accuracy_pct"]["mean"] == 75.
    assert summary["full_accuracy_pct"]["sample_sd"] == pytest.approx(50. ** .5)


def test_same_file_cannot_create_two_independent_seeds(tmp_path):
    p = family(tmp_path)
    entries = json.loads(p.read_text())
    entries[1]["seed"] = 42
    p.write_text(json.dumps(entries))
    with pytest.raises(ValueError, match="different seeds"):
        analyze(p)


def test_matched_full_predictions_must_agree_across_baselines(tmp_path):
    p = family(tmp_path)
    entries = json.loads(p.read_text())
    rows = [json.loads(line) for line in (tmp_path/"full.jsonl").read_text().splitlines()]
    rows[0]["correct"] = not rows[0]["correct"]
    (tmp_path/"changed.jsonl").write_text("".join(json.dumps(row)+"\n" for row in rows))
    entries[1]["full_predictions"] = "changed.jsonl"
    p.write_text(json.dumps(entries))
    with pytest.raises(ValueError, match="Full-method outcomes"):
        analyze(p)
