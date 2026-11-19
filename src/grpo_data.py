from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

from .data_protocol import split_training_indices

SOURCE_COUNTS = {"mbpp": 369, "apps": 493, "codecontests": 500}


def iter_records(path):
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        if path.suffix == ".jsonl":
            for line in f:
                if line.strip():
                    yield json.loads(line)
        else:
            records = json.load(f)
            if not isinstance(records, list):
                raise ValueError(f"expected JSON array or JSONL at {path}")
            yield from records


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as f:
        try:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.close()
            os.replace(f.name, path)
        finally:
            Path(f.name).unlink(missing_ok=True)


def load_rows(path, domain):
    path = Path(path)
    if domain not in ("sql", "code"):
        raise ValueError("domain must be sql or code")
    rows = list(iter_records(path))
    seen = set()
    for row in rows:
        key = row.get("id")
        if not isinstance(key, str) or not key or key in seen:
            raise ValueError(f"missing or duplicate id in {path}: {key!r}")
        seen.add(key)
        if not isinstance(row.get("prompt"), str) or not row["prompt"].strip():
            raise ValueError(f"missing prompt for {key}")
        if domain == "code":
            if not isinstance(row.get("tests"), list) or not row["tests"]:
                raise ValueError(f"missing code tests for {key}")
            continue
        if row.get("dataset") not in ("wikisql", "spider", "bird") or not row.get("gold_sql"):
            raise ValueError(f"missing SQL metadata for {key}")
        if "table" not in row:
            db = (path.parent / row["db_path"]).resolve()
            if not db.is_file():
                raise FileNotFoundError(db)
            row["db_path"] = str(db)
    if not rows:
        raise ValueError(f"empty dataset at {path}")
    return rows


def _wiki_literal(value, column_type):
    if column_type != "real":
        return "'" + str(value).replace("'", "''") + "'"
    try:
        number = float(str(value).replace(",", ""))
        if math.isfinite(number):
            return str(number)
    except ValueError:
        pass
    return "'" + str(value).replace("'", "''") + "'"


def wiki_row(example, tables):
    table = example.get("table") or tables[example["table_id"]]
    spec = example["sql"]
    col = f'"c{int(spec["sel"])}"'
    agg = ("", "MAX", "MIN", "COUNT", "SUM", "AVG")[int(spec["agg"])]
    conds = spec["conds"]
    if isinstance(conds, dict):
        conds = zip(conds["column_index"], conds["operator_index"], conds["condition"])
    predicates = []
    for index, op, value in conds:
        operator = ("=", ">", "<", "!=")[int(op)]
        predicates.append(f'"c{int(index)}" {operator} {_wiki_literal(value, table["types"][int(index)])}')
    golden = f"SELECT {agg + '(' + col + ')' if agg else col} FROM data"
    if predicates:
        golden += " WHERE " + " AND ".join(predicates)
    schema = ", ".join(f"c{i} ({kind}; {h})" for i, (h, kind) in enumerate(zip(table["header"], table["types"])))
    prompt = ("Write one SQLite query for the question. Use table data and the c0, c1, ... "
              f"column identifiers shown below. Return SQL only.\nColumns: {schema}\nQuestion: {example['question']}")
    return dict(prompt=prompt, gold_sql=golden + ";", table=table)


def _sql_rows(dataset, files, tables_path, db_dir, split, out_dir):
    from . import bird_data, spider_data
    tables = {}
    if tables_path:
        table_key = "id" if dataset == "wikisql" else "db_id"
        tables = {str(t[table_key]): t for t in iter_records(tables_path)}
    schemas = {}
    rows = []
    for path in files:
        for raw in iter_records(path):
            if dataset == "wikisql":
                row = wiki_row(raw, tables)
            else:
                if not db_dir:
                    raise ValueError(f"{dataset} preparation requires a database directory")
                db_id = str(raw["db_id"])
                if Path(db_id).name != db_id:
                    raise ValueError("invalid database identifier")
                db = Path(db_dir).resolve() / db_id / (db_id + ".sqlite")
                if not db.is_file():
                    raise FileNotFoundError(db)
                mod = bird_data if dataset == "bird" else spider_data
                if db_id not in schemas:
                    schemas[db_id] = mod.schema_text(tables[db_id]) if db_id in tables else bird_data.schema_text_from_sqlite(db)
                example = dict(raw, _schema_text=schemas[db_id])
                prompt = bird_data.bird_prompt(example) if dataset == "bird" else spider_data.spider_prompt(example)
                gold = bird_data.gold_bird(example) if dataset == "bird" else spider_data.gold_spider(example)
                row = dict(prompt=prompt, gold_sql=gold, db_id=db_id, db_path=os.path.relpath(db, out_dir))
            rows.append(dict(id=f"{dataset}:{split}:{len(rows)}", dataset=dataset, **row))
    return rows


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(131072), b""):
            h.update(block)
    return h.hexdigest()


def prepare_sql(args):
    train_files = [args.train_file, *args.extra_train_file]
    if any(p.resolve() == args.eval_file.resolve() for p in train_files):
        raise ValueError("evaluation file must be distinct from the training files")
    train = _sql_rows(args.dataset, train_files, args.train_tables, args.train_db_dir, "train", args.out_dir)
    test = _sql_rows(args.dataset, [args.eval_file], args.eval_tables, args.eval_db_dir, "eval", args.out_dir)
    split = split_training_indices(len(train), args.anchor_examples, args.split_seed)
    if len(split.anchor) != args.anchor_examples or not test:
        raise ValueError("insufficient examples for requested anchor/evaluation split")
    parts = {"train": [train[i] for i in split.optimization], "anchor": [train[i] for i in split.anchor],
             "selection": [train[i] for i in split.model_selection], "eval": test}
    paths = [*train_files, args.eval_file, args.train_tables, args.eval_tables]
    manifest = {"dataset": args.dataset, "split_seed": args.split_seed,
                "source_sha256": [dict(file=p.name, sha256=digest(p)) for p in paths if p], "parts": {}}
    for name, records in parts.items():
        path = args.out_dir / ("sql_" + name + ".jsonl")
        write_rows(path, records)
        manifest["parts"][name] = dict(count=len(records), sha256=digest(path))
    (args.out_dir / "sql_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def prepare_code(args):
    from .code_data import SourceRows, VerificationResult, build_code_pool
    from .grpo_rewards import code_test_outcomes
    def verify(task):
        outcomes, _ = code_test_outcomes(task.reference_solution, task.to_dict(), image=args.image)
        return VerificationResult({str(t["test_id"]): ok for t, ok in zip(task.tests, outcomes)})
    caps = SOURCE_COUNTS if args.cap is None else dict.fromkeys(SOURCE_COUNTS, args.cap)
    sources = [SourceRows(name, "train", iter_records(getattr(args, name))) for name in SOURCE_COUNTS]
    built = build_code_pool(sources, verify, pilot_cap=caps)
    seen = Counter(task.source for task in built.tasks)
    if seen != caps:
        raise ValueError(f"training sources yielded {dict(seen)}, expected {caps}")
    def serialize(task):
        instruction = "Return one complete Python 3 program." if task.verifier == "stdin" else "Return the complete Python function implementation."
        return dict(id=task.uid, prompt=f"{instruction} Do not use Markdown or explanation.\nTask: {task.statement}",
                    source=task.source, tests=task.tests, test_setup=task.test_setup)
    path = args.out_dir / "code_train.jsonl"
    write_rows(path, map(serialize, built.tasks))
    manifest = dict(built.manifest, image=args.image, source_caps=caps, output_sha256=digest(path),
                    source_sha256={source: digest(getattr(args, source)) for source in SOURCE_COUNTS})
    (args.out_dir / "code_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description='Prepare and load GRPO JSONL files from official training and development splits.')
    sub = parser.add_subparsers(dest="domain", required=True)
    sql = sub.add_parser("sql", description="Read official training and development rows in their original order. For Spider include train_spider.json then --extra-train-file train_others.json.")
    sql.add_argument("--dataset", choices=("wikisql", "bird", "spider"), required=True)
    for name in ("train-file", "eval-file", "train-tables", "eval-tables", "train-db-dir", "eval-db-dir"):
        sql.add_argument("--" + name, type=Path, required=name in ("train-file", "eval-file"))
    sql.add_argument("--extra-train-file", action="append", type=Path, default=[])
    sql.add_argument("--split-seed", type=int, required=True)
    sql.add_argument("--anchor-examples", type=int, default=256)
    code = sub.add_parser("code", description="Read official training rows in source order from JSON or JSONL, normalized by code_data and verified inside Docker.")
    for name in SOURCE_COUNTS:
        code.add_argument("--" + name, type=Path, required=True)
    code.add_argument("--cap", type=int, help="accepted tasks per source, for small checks")
    code.add_argument("--image", default="python:3.11-slim", help="locally available Linux Python image or digest")
    for cmd in (sql, code):
        cmd.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    (prepare_sql if args.domain == "sql" else prepare_code)(args)


if __name__ == "__main__":
    main()
