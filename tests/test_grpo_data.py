import json
import os
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from src.evaluate_grpo import evaluate
from src.grpo_data import wiki_row, write_rows, load_rows
from src.grpo_rewards import sql_reward, code_reward


class TestRewardAndEvaluation(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        f = self.root / "example.sqlite"
        conn = sqlite3.connect(f)
        conn.executescript("CREATE TABLE t(x int); INSERT INTO t VALUES (1),(1),(2);")
        conn.close()
        self.row = dict(id="example", dataset="bird", gold_sql="SELECT x FROM t", prompt="list x", db_path=str(f))

    def test_rewards_and_read_only(self):
        self.assertEqual(sql_reward("SELECT DISTINCT x FROM t", self.row), 1)
        self.assertEqual(sql_reward("SELECT count(x) FROM t", self.row), .2)
        self.assertEqual(sql_reward("SELECT missing_column FROM t", self.row), .05)
        self.assertEqual(sql_reward("DELETE FROM t", self.row), 0)
        self.assertEqual(sql_reward("SELECT 1; DELETE FROM t", self.row), 0)
        self.assertEqual(sql_reward("SELECT DISTINCT x FROM t", dict(self.row, dataset="spider")), .2)
        self.assertEqual(sql_reward("SELECT x FROM t", self.row), 1)

    def test_wikisql_original_conditions(self):
        row = wiki_row({"table_id":"a", "question":"count?", "sql":{"sel":0,"agg":3,"conds":[[1,0,"cat"]]}},
                       {"a":{"header":["n","animal"],"types":["real","text"],"rows":[[1,"cat"],[2,"dog"]]}})
        self.assertEqual(sql_reward("SELECT 1", dict(row, dataset="wikisql")), 1)


    def test_wikisql_text_identifiers_and_numeric_amounts(self):
        t = {"header": ["unit", "cost"], "types":["text", "real"],
             "rows":[["003", "1,000"], ["3", 2], ["Team", 2]]}
        record = {"table_id": "a", "question": "which cost?", "sql":{"sel":1, "agg":0, "conds":[[0,0,"003"]]}}
        row = dict(wiki_row(record, {"a":t}), dataset="wikisql")
        self.assertEqual(sql_reward("SELECT 1000.0", row), 1)
        self.assertEqual(sql_reward("SELECT 2.0", row), .2)
        record["sql"] = {"sel":0, "agg":0, "conds":[[1,0,"1,000"]]}
        row = dict(wiki_row(record, {"a":t}), dataset="wikisql")
        self.assertEqual(sql_reward("SELECT '003'", row), 1)
        record["sql"] = {"sel":1, "agg":0, "conds":[[0,0,"team"]]}
        row = dict(wiki_row(record, {"a":t}), dataset="wikisql")
        self.assertEqual(sql_reward("SELECT 2.0", row), 1)

    def test_per_question_output_and_missing_ids(self):
        f = self.root / "eval.jsonl"
        p = self.root / "pred.jsonl"
        write_rows(f,[self.row])
        write_rows(p,[dict(id="example", prediction="SELECT x FROM t")])
        stats = evaluate(f,p,self.root / "result")
        self.assertEqual(stats["accuracy_percent"], 100)
        self.assertEqual(stats["exact_accuracy_percent"], 100)
        self.assertEqual(stats["exact_metric"], "bird_normalized_sql_exact_match")
        pairs = json.loads((self.root/"result"/"questions.jsonl").read_text())
        self.assertTrue(pairs["correct"])
        write_rows(p,[dict(id="other",prediction="SELECT x FROM t")])
        with self.assertRaisesRegex(ValueError,"IDs"):
            evaluate(f,p,self.root/"bad")

    def test_exact_differs_from_execution(self):
        data = self.root / "data.jsonl"
        predictions = self.root / "pred.jsonl"
        for dataset in ("wikisql", "bird"):
            row = dict(self.row, dataset=dataset)
            write_rows(data, [row, dict(row, id="second")])
            write_rows(predictions, [
                dict(id="example", prediction="select   x  FROM t ;"),
                dict(id="second", prediction="SELECT (x) FROM t")
            ])
            stats = evaluate(data, predictions, self.root / dataset)
            self.assertEqual(stats["accuracy_percent"], 100)
            self.assertEqual(stats["exact_accuracy_percent"], 50)
            self.assertEqual(stats["exact_metric"], dataset + "_normalized_sql_exact_match")
            pairs = [json.loads(s) for s in (self.root / dataset / "questions.jsonl").read_text().splitlines()]
            self.assertEqual([r["exact_correct"] for r in pairs], [True, False])
        write_rows(data, [dict(self.row, dataset="spider", db_id="test")])
        write_rows(predictions, [dict(id="example", prediction=self.row["gold_sql"])])
        self.assertNotIn("exact_accuracy_percent", evaluate(data, predictions, self.root / "spider"))

    @unittest.skipUnless(os.environ.get("SPIDER_EVALUATOR_DIR"), "official Spider checkout not configured")
    def test_official_spider_exact(self):
        checkout = Path(os.environ["SPIDER_EVALUATOR_DIR"])
        data, preds, tables = (self.root / f for f in ("data.jsonl", "preds.jsonl", "tables.json"))
        tables.write_text(json.dumps([dict(db_id="test", table_names_original=["t"], foreign_keys=[],
                                          column_names_original=[[-1, "*"], [0, "x"]])]))
        row = dict(self.row, dataset="spider", db_id="test", gold_sql="SELECT x FROM t WHERE x = 1")
        write_rows(data, [row, dict(row, id="wrong"), dict(row, id="invalid")])
        write_rows(preds, [dict(id="example", prediction="SELECT x FROM t WHERE x = 2"),
                           dict(id="wrong", prediction="SELECT count(x) FROM t"),
                           dict(id="invalid", prediction="not SQL")])
        stats = evaluate(data, preds, self.root / "official", checkout, None, tables)
        self.assertEqual(stats["exact_metric"], "spider_official_exact_set_match")
        self.assertEqual(stats["exact_correct"], 1)
        self.assertEqual(stats["correct"], 0)

    def test_seed_propagation_and_mixed_seed_rejection(self):
        data = self.root / "eval.jsonl"
        predictions = self.root / "predictions.jsonl"
        rows = [self.row, dict(self.row, id="second")]
        write_rows(data, rows)
        pred = [dict(id=r["id"], seed=42, prediction=r["gold_sql"]) for r in rows]
        write_rows(predictions, pred)
        stats = evaluate(data, predictions, self.root / "out")
        self.assertEqual(stats["seed"], 42)
        self.assertEqual([json.loads(line)["seed"] for line in
                          (self.root/"out"/"questions.jsonl").read_text().splitlines()], [42, 42])
        for replacement in (dict(seed=2), {}):
            pred[1] = dict(id="second", prediction=self.row["gold_sql"], **replacement)
            write_rows(predictions, pred)
            with self.assertRaisesRegex(ValueError, "same training seed"):
                evaluate(data, predictions, self.root/"bad_seed")

    def test_no_host_fallback_for_generated_code(self):
        task = {"tests":[dict(kind="assert", code="assert add(1) == 2")]}
        with patch("src.grpo_rewards.subprocess.run",side_effect=FileNotFoundError):
            with self.assertRaisesRegex(RuntimeError,"Docker"):
                code_reward("def add(n): return n+1", task)
        self.assertEqual(code_reward("def ???",task), 0.0)


if __name__ == "__main__":
    unittest.main()
