from __future__ import annotations

import argparse
from contextlib import closing
import importlib
import json
from pathlib import Path
import sqlite3
import sys

from .grpo_data import digest, iter_records, load_rows, write_rows
from .grpo_rewards import _allow, _execute, extract, score_sql


def _upstream(checkout, name):
    source = (checkout / (name + ".py")).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    sys.path.insert(0, str(checkout.resolve()))
    try:
        module = importlib.import_module(name)
    finally:
        sys.path.pop(0)
    if Path(module.__file__).resolve() != source:
        raise RuntimeError(f"loaded {name} differs from the requested Spider checkout")
    return module


def _string_exact(row, pred):
    def normalize(query):
        return " ".join(query.strip().removesuffix(";").lower().split())
    return normalize(pred) == normalize(row["gold_sql"])


def spider_exact_matcher(checkout, tables):
    _upstream(checkout, "process_sql")
    _upstream(checkout, "parse")
    _upstream(checkout, "exec_eval")
    evaluator = _upstream(checkout, "evaluation")
    kmaps = evaluator.build_foreign_key_map_from_json(str(tables))
    schemas = {}

    def canonical(tree, schema, kmap):
        valid = evaluator.build_valid_col_units(tree["from"]["table_units"], schema)
        return evaluator.rebuild_sql_col(valid, evaluator.rebuild_sql_val(tree), kmap)

    def match(row, pred):
        db = row["db_id"]
        if db not in schemas:
            schemas[db] = evaluator.Schema(evaluator.get_schema(row["db_path"]))
        schema = schemas[db]
        gold = canonical(evaluator.get_sql(schema, row["gold_sql"].rstrip(";")), schema, kmaps[db])
        try:
            candidate = evaluator.get_sql(schema, pred.replace("value", "1"))
        except (AssertionError, ValueError, KeyError, IndexError, TypeError, RecursionError):
            return False
        return bool(evaluator.Evaluator().eval_exact_match(canonical(candidate, schema, kmaps[db]), gold))

    return match


def spider_suite_matcher(checkout, root, rows):
    for name in ("exec_eval.py", "parse.py"):
        if not (checkout / name).is_file():
            raise FileNotFoundError(checkout / name)
    ids = {row["db_id"] for row in rows}
    for key in ids:
        if Path(key).name != key or not (root / key / (key + ".sqlite")).is_file():
            raise ValueError(f"invalid or missing Spider suite directory for {key}")
    if not any(len(list((root / key).glob("*.sqlite"))) > 1 for key in ids):
        raise ValueError("no generated test databases found; cannot label this score Spider TS")
    _upstream(checkout, "parse")
    evaluator = _upstream(checkout, "exec_eval")

    async def query(db_path, sql):
        with closing(sqlite3.connect(Path(db_path).resolve().as_uri() + "?mode=ro", uri=True)) as conn:
            conn.execute("PRAGMA query_only=ON")
            conn.execute("PRAGMA trusted_schema=OFF")
            conn.execute("PRAGMA temp_store=MEMORY")
            conn.text_factory = lambda value: value.decode(errors="ignore")
            if hasattr(conn, "setlimit"):
                conn.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, 2_000_000)
            conn.set_authorizer(_allow)
            return "result", _execute(conn, evaluator.replace_cur_year(sql))

    def match(row, pred):
        if not pred:
            return False
        db = root / row["db_id"] / (row["db_id"] + ".sqlite")
        original = evaluator.exec_on_db_
        evaluator.exec_on_db_ = query
        try:
            return bool(evaluator.eval_exec_match(str(db), pred, row["gold_sql"], False, False, False))
        finally:
            evaluator.exec_on_db_ = original
    return match


def evaluate(data, predictions, out_dir, evaluator_dir=None, suite_dir=None, spider_tables=None):
    rows = load_rows(data, "sql")
    preds = {}
    for record in iter_records(predictions):
        key = record["id"]
        if key in preds or not isinstance(record["prediction"], str):
            raise ValueError(f"invalid or duplicated prediction for {key}")
        if "seed" in record and type(record["seed"]) is not int:
            raise ValueError(f"prediction seed must be an integer at {key}")
        preds[key] = record
    seeds = {record.get("seed") for record in preds.values()}
    if len(seeds) > 1:
        raise ValueError("all predictions must have the same training seed")
    seed = next(iter(seeds), None)
    if {row["id"] for row in rows} != set(preds):
        raise ValueError("evaluation and prediction IDs do not match exactly")
    datasets = {row["dataset"] for row in rows}
    if len(datasets) != 1:
        raise ValueError("evaluate one dataset at a time")
    dataset = datasets.pop()
    metric = {"bird":"execution", "wikisql":"denotation", "spider":"spider_single_db_execution"}[dataset]
    suite = None
    exact = _string_exact if dataset in ("wikisql", "bird") else None
    exact_metric = f"{dataset}_normalized_sql_exact_match"
    if any((evaluator_dir, suite_dir, spider_tables)):
        if dataset != "spider" or not evaluator_dir or not (suite_dir or spider_tables):
            raise ValueError("supply a Spider evaluator checkout and a generated suite or tables.json")
    if suite_dir:
        suite = spider_suite_matcher(evaluator_dir, suite_dir, rows)
        metric = "spider_test_suite_accuracy"
    if spider_tables:
        exact = spider_exact_matcher(evaluator_dir, spider_tables)
        exact_metric = "spider_official_exact_set_match"
    outcomes = []
    for row in rows:
        pred = preds[row["id"]]
        if pred.get("dataset",dataset) != dataset:
            raise ValueError(f"prediction dataset mismatch at {row['id']}")
        sql = extract(pred["prediction"], "sql")
        ok = suite(row, sql) if suite else score_sql(sql, row)["correct"]
        result = dict(id=row["id"], dataset=dataset, metric=metric, correct=bool(ok),
                      prediction=sql, gold_sql=row["gold_sql"])
        if exact:
            result.update(exact_metric=exact_metric, exact_correct=exact(row, sql))
        if seed is not None:
            result["seed"] = seed
        if "db_id" in row:
            result["db_id"] = row["db_id"]
        outcomes.append(result)
    count = sum(r["correct"] for r in outcomes)
    stats = dict(dataset=dataset, metric=metric, n=len(rows), correct=count,
                 accuracy_percent=count * 100 / len(rows), data_sha256=digest(data),
                 predictions_sha256=digest(predictions))
    if seed is not None:
        stats["seed"] = seed
    if exact:
        n_exact = sum(r["exact_correct"] for r in outcomes)
        stats.update(exact_metric=exact_metric, exact_correct=n_exact,
                     exact_accuracy_percent=n_exact * 100 / len(rows))
    if spider_tables:
        stats.update(spider_tables_sha256=digest(spider_tables),
                     official_evaluation_sha256=digest(evaluator_dir / "evaluation.py"),
                     official_process_sql_sha256=digest(evaluator_dir / "process_sql.py"))
    if suite:
        stats["official_exec_eval_sha256"] = digest(evaluator_dir / "exec_eval.py")
        stats["official_parse_sha256"] = digest(evaluator_dir / "parse.py")
    write_rows(out_dir / "questions.jsonl", outcomes)
    (out_dir / "metrics.json").write_text(json.dumps(stats, indent=2) + "\n")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Score ID-aligned GRPO predictions and emit one paired outcome per question.\n\nFor Spider TS, supply a trusted checkout of taoyds/test-suite-sql-eval\n(e97acc546ecbee8fa27fa8dbf025ef61493a876c) and its generated database suite.\nThe default Spider metric measures execution on a single database.\nFor official Spider exact-set match, also supply its tables.json and install\nthe upstream parser's NLTK tokenizer data. WikiSQL and BIRD Exact compare\nthe prepared gold and predicted SQL strings with case and whitespace normalized.\n")
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--spider-evaluator-dir", type=Path)
    parser.add_argument("--spider-test-suite-db-dir", type=Path)
    parser.add_argument("--spider-tables", type=Path, help="Official Spider tables.json for structural exact-set match")
    args = parser.parse_args()
    print(json.dumps(evaluate(args.data, args.predictions, args.out_dir, args.spider_evaluator_dir,
                              args.spider_test_suite_db_dir, args.spider_tables)))


if __name__ == "__main__":
    main()
