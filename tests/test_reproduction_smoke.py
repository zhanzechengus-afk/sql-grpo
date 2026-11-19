import json
from pathlib import Path
import sqlite3
import subprocess
import sys

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast


def make_inputs(root):
    model_dir = root / "base"
    model_dir.mkdir()
    tokens = ["<unk>", "<eos>", "<pad>", "SELECT", "1", "2", "SQL", "answer", ":", "?", "."]
    tok = Tokenizer(WordLevel(dict(zip(tokens, range(len(tokens)))), unk_token="<unk>"))
    tok.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token=tokens[0],
                                       eos_token=tokens[1], pad_token=tokens[2])
    tokenizer.chat_template = "{% for m in messages %}{{ m.content }} {% endfor %}{% if add_generation_prompt %}answer: {% endif %}"
    tokenizer.save_pretrained(model_dir)
    torch.manual_seed(42)
    LlamaForCausalLM(LlamaConfig(vocab_size=len(tokens), hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=64,
        bos_token_id=0, eos_token_id=1, pad_token_id=2)).save_pretrained(model_dir)
    data = root / "data"
    data.mkdir()
    db = data / "fixture.sqlite"
    with sqlite3.connect(db) as connection:
        connection.execute("create table flags (value integer)")
    for split, val in [("train", 1), ("anchor", 2), ("eval", 3)]:
        row = dict(id=f"fixture-{split}",dataset="bird",db_id="fixture",db_path=str(db),prompt=f"SQL {val}?",
                   gold_sql=f"SELECT {val}")
        (data / f"sql_{split}.jsonl").write_text(json.dumps(row) + "\n")
    return model_dir, data


def cli(module, *args):
    subprocess.run([sys.executable, "-m", f"src.{module}", *map(str, args)], check=True,
                   cwd=Path(__file__).resolve().parents[1], timeout=180)


def test_sft_grpo_sql(tmp_path):
    model, data = make_inputs(tmp_path)
    sft = tmp_path / "sft"
    rl = tmp_path / "rl"
    shared = ["--model", model, "--data-dir", data, "--device", "cpu", "--rank", "2", "--adapter-dim", "8",
              "--max-length", "32", "--seed", "2"]
    cli("train_sql_sft", *shared, "--batch-size", "1", "--output-dir", sft)
    init = torch.load(sft / "common_sql_trainable.pt", weights_only=True)
    assert init and any("lora_" in k for k in init)
    control = tmp_path / "continued"
    cli("train_sql_sft", *shared, "--batch-size", "1", "--steps", "2", "--continue-from",
        sft / "common_sql_trainable.pt", "--output-dir", control)
    assert len((control / "sft_steps.jsonl").read_text().splitlines()) == 2
    assert (control / "continued_sql_trainable.pt").is_file()
    cli("train_grpo", *shared, "--sft-checkpoint", sft / "common_sql_trainable.pt", "--output-dir", rl,
        "--branch", "sql_only", "--group-size", "2", "--sql-prompts", "1", "--steps", "1",
        "--sql-new-tokens", "3", "--code-new-tokens", "3", "--anchor-batch", "1", "--backend", "torch")
    trained = torch.load(rl / "post_rl_trainable.pt", weights_only=True)
    deploy = torch.load(rl / "trainable.pt", weights_only=True)
    assert set(init) == set(trained) and set(deploy).issubset(init)
    for k, value in deploy.items():
        assert torch.isfinite(value).all()
        torch.testing.assert_close(value.float(), init[k].float().lerp(trained[k].float(), .8), atol=1e-5, rtol=1e-4)
    predictions = rl / "predictions.jsonl"
    held_out = data / "sql_eval.jsonl"
    cli("train_grpo", *shared, "--sft-checkpoint", rl / "trainable.pt", "--eval-file", held_out,
        "--sql-new-tokens", "3", "--code-new-tokens", "3", "--predict", predictions)
    evaluation = rl / "evaluation"
    cli("evaluate_grpo", "--data", held_out, "--predictions", predictions, "--out-dir", evaluation)
    result = json.loads((evaluation / "questions.jsonl").read_text())
    stats = json.loads((evaluation / "metrics.json").read_text())
    assert result["id"] == "fixture-eval" and result["metric"] == "execution"
    assert result["seed"] == stats["seed"] == 2
    assert stats["n"] == 1 and stats["correct"] == int(result["correct"])
