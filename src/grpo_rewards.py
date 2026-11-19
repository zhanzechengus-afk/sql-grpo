from __future__ import annotations

import ast
import json
import re
import sqlite3
import subprocess
import time
import uuid
from contextlib import closing
from pathlib import Path

import sqlglot
from sqlglot import exp


def extract(text, lang):
    match = re.search(r"```(?:" + lang + r")?\s*([\s\S]*?)```", text, re.I)
    return (match.group(1) if match else text).strip()


def _connection(row):
    if "table" not in row:
        return sqlite3.connect(Path(row["db_path"]).resolve().as_uri() + "?mode=ro", uri=True)
    table = row["table"]
    conn = sqlite3.connect(":memory:")
    try:
        columns = [f'c{i} {"REAL" if kind == "real" else "TEXT COLLATE NOCASE"}' for i, kind in enumerate(table["types"])]
        conn.execute("CREATE TABLE data (" + ",".join(columns) + ")")
        def cells(line):
            for value, kind in zip(line, table["types"]):
                if kind == "real" and isinstance(value, str):
                    try:
                        value = float(value.replace(",", ""))
                    except ValueError:
                        pass
                yield value
        conn.executemany("INSERT INTO data VALUES (" + ",".join("?" for _ in columns) + ")",
                         (tuple(cells(line)) for line in table["rows"]))
        return conn
    except BaseException:
        conn.close()
        raise


def _allow(op, arg1, arg2, db, origin):
    if op == sqlite3.SQLITE_FUNCTION and (arg2 or "").lower() in {
        "load_extension", "readfile", "writefile", "fts3_tokenizer", "randomblob", "zeroblob",
    }:
        return sqlite3.SQLITE_DENY
    allowed = (sqlite3.SQLITE_READ, sqlite3.SQLITE_SELECT, sqlite3.SQLITE_FUNCTION, getattr(sqlite3, "SQLITE_RECURSIVE", 33))
    return sqlite3.SQLITE_OK if op in allowed else sqlite3.SQLITE_DENY


def _execute(conn, sql):
    deadline = time.monotonic() + 5
    steps = 0

    def stop():
        nonlocal steps
        steps += 1000
        return int(steps > 5_000_000 or time.monotonic() > deadline)

    conn.set_progress_handler(stop, 1000)
    with closing(conn.execute(sql)) as cur:
        rows = []
        nbytes = 0
        for batch in iter(lambda: cur.fetchmany(100), []):
            nbytes += sum(len(v) for line in batch for v in line if isinstance(v, (str, bytes)))
            if len(rows) + len(batch) > 100_000 or nbytes > 16_000_000:
                raise sqlite3.DataError("SQL result exceeds configured size limit")
            rows.extend(batch)
    return rows


def score_sql(candidate, row):
    from .spider_data import denotations_equal

    candidate = extract(candidate, "sql")
    try:
        exprs = sqlglot.parse(candidate, read="sqlite")
        parsed = len(exprs) == 1 and isinstance(exprs[0], exp.Query)
    except (sqlglot.errors.SqlglotError, RecursionError, ValueError):
        parsed = False
    result = dict(prediction=candidate, parsed=parsed, executable=False, correct=False, reward=0.0)
    if not parsed:
        return result
    with closing(_connection(row)) as conn:
        conn.execute("PRAGMA trusted_schema=OFF")
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute("PRAGMA query_only=ON")
        if hasattr(conn, "setlimit"):
            conn.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, 2_000_000)
        conn.set_authorizer(_allow)
        gold = _execute(conn, row["gold_sql"])
        try:
            pred = _execute(conn, candidate)
        except sqlite3.Error:
            result["reward"] = 0.05
            return result
    if row["dataset"] == "bird":
        correct = set(gold) == set(pred)
    elif row["dataset"] == "spider":
        correct = denotations_equal(gold, pred, bool(re.search(r"\border\s+by\b", row["gold_sql"], re.I)))
    elif row["dataset"] == "wikisql":
        def canonical(rows):
            return sorted(tuple(str(v).strip().lower() for v in line) for line in rows)
        correct = canonical(gold) == canonical(pred)
    else:
        raise ValueError("unknown SQL dataset")
    result.update(executable=True, correct=correct, reward=1.0 if correct else 0.20)
    return result


def sql_reward(candidate, row):
    return score_sql(candidate, row)["reward"]


def code_test_outcomes(candidate, row, *, image="python:3.11-slim"):
    candidate = extract(candidate, "python")
    tests = row["tests"]
    if not tests:
        raise ValueError("code task must have tests")
    if any(t.get("kind") not in ("assert", "function", "stdin") for t in tests):
        raise ValueError("unsupported code test type")
    try:
        ast.parse(candidate)
    except (SyntaxError, ValueError, RecursionError):
        return [False] * len(tests), False
    if not candidate:
        return [False] * len(tests), False
    worker = Path(__file__).with_name("grpo_container_worker.py").read_text()
    name = "sql-grpo-" + uuid.uuid4().hex
    cmd = [
        "docker", "run", "--rm", "--pull=never", "-i", "--name", name,
        "--network=none", "--read-only", "--cap-drop=ALL", "--security-opt=no-new-privileges",
        "--user=65534:65534", "--pids-limit=32", "--cpus=1", "--memory=256m",
        "--memory-swap=256m", "--ulimit", "nofile=64:64", "--tmpfs", "/tmp:rw,noexec,nosuid,size=16m,mode=1777",
        "--log-driver=none", "--entrypoint", "python", image, "-I", "-B", "-c", worker,
    ]
    payload = json.dumps({"candidate": candidate, "setup": row.get("test_setup", ""), "tests": tests})
    try:
        run = subprocess.run(cmd, input=payload, capture_output=True, text=True,
                             timeout=30 + 6 * len(tests), check=True)
    except FileNotFoundError as exc:
        raise RuntimeError("Docker CLI required for isolated code verification") from exc
    except subprocess.TimeoutExpired as exc:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=12, check=False)
        raise RuntimeError("Docker verification worker timed out") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Docker verifier failed ({exc.returncode}): {exc.stderr[-600:]}") from exc
    outputs = json.loads(run.stdout)
    if not isinstance(outputs, list) or len(outputs) != len(tests) or any(type(ok) is not bool for ok in outputs):
        raise RuntimeError("incomplete isolated verification result")
    return outputs, True


def code_reward(candidate, row, *, image="python:3.11-slim"):
    outcomes, parsed = code_test_outcomes(candidate, row, image=image)
    return sum(outcomes) / len(outcomes) if any(outcomes) else 0.05 if parsed else 0.0
