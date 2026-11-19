from contextlib import contextmanager
import copy
import json
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from src.train_grpo import CODE_PREFIX, parse_config, train


class ToyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Parameter(torch.randn(7, 7) / 10)
        self.private = nn.ParameterDict(
            {d: nn.Parameter(torch.zeros(7, 7)) for d in ("sql", "code")}
        )
        self.active = "sql"

    @property
    def policy(self):
        return self

    @contextmanager
    def domain(self, name):
        old, self.active = self.active, name
        try:
            yield
        finally:
            self.active = old

    def shared_parameters(self):
        return [self.shared]

    def trainable_state(self):
        return {
            "lora_weight": self.shared.detach().clone(),
            **{
                "private_adapters.domain_adapters." + d + ".weight": p.detach().clone()
                for d, p in self.private.items()
            },
        }

    def forward(self, input_ids, **kwargs):
        return SimpleNamespace(
            logits=self.shared[input_ids] + self.private[self.active][input_ids]
        )


class ToyTokenizer:
    eos_token_id = 6
    pad_token_id = 0

    def apply_chat_template(self, *args, **kwargs):
        return [1, 2]

    def encode(self, *args, **kwargs):
        return [3]

    def decode(self, ids, **kwargs):
        return " ".join(str(i) for i in ids)


def test_three_grpo_controls_and_deployment(tmp_path):
    torch.manual_seed(8)
    init = ToyPolicy()
    init_state = init.trainable_state()
    outputs = {}
    initial_code = init.private["code"].clone().detach()

    def oracle(text, row, **kwargs):
        return sum(map(int, text.split())) % 7 / 6

    sql = [{"id": "train", "prompt": "sql", "gold_sql": "select 1"}]
    anchors = [{"id": "other", "prompt": "anchor", "gold_sql": "select 1"}]
    code = [{"id": "code", "prompt": "code"}]
    with (
        patch("src.grpo_rewards.sql_reward", oracle),
        patch("src.grpo_rewards.code_reward", oracle),
    ):
        for branch in ("sql_only", "zero_code", "full"):
            device_model = copy.deepcopy(init)
            path = tmp_path / branch
            path.mkdir()
            args = parse_config(
                [
                    "--sft-checkpoint",
                    "unused",
                    "--data-dir",
                    "unused",
                    "--output-dir",
                    str(path),
                    "--branch",
                    branch,
                    "--steps",
                    "1",
                    "--sql-prompts",
                    "1",
                    "--code-prompts",
                    "1",
                    "--group-size",
                    "8",
                    "--anchor-batch",
                    "1",
                    "--rollout-batch",
                    "4",
                    "--micro-batch",
                    "2",
                    "--lr",
                    "0.001",
                    "--max-length",
                    "6",
                    "--sql-new-tokens",
                    "2",
                    "--code-new-tokens",
                    "2",
                ]
            )
            train(
                device_model,
                copy.deepcopy(init),
                ToyTokenizer(),
                init_state,
                sql,
                anchors,
                code,
                args,
                path,
            )
            trained = torch.load(path / "post_rl_trainable.pt", weights_only=True)
            deployed = torch.load(path / "trainable.pt", weights_only=True)
            assert all(not key.startswith(CODE_PREFIX) for key in deployed)
            assert deployed.keys() == {
                key for key in init_state if not key.startswith(CODE_PREFIX)
            }
            for key in deployed:
                torch.testing.assert_close(
                    deployed[key], torch.lerp(init_state[key], trained[key], args.alpha)
                )
            metrics = json.loads((path / "metrics.jsonl").read_text())
            assert metrics["sql_responses"] == 8 and 8 <= metrics["sql_tokens"] <= 16
            if branch == "full":
                assert metrics["code_responses"] == 8 and "anchor_cosine" in metrics
                assert not torch.equal(device_model.private["code"], initial_code)
            outputs[branch] = deployed
    for key in outputs["sql_only"]:
        torch.testing.assert_close(
            outputs["sql_only"][key], outputs["zero_code"][key], rtol=0, atol=0
        )


def test_prediction_atomicity_and_seed(tmp_path):
    import pytest
    from src.train_grpo import write_predictions

    output = tmp_path / "pred.jsonl"
    rows = [
        dict(id="a", prompt="sql", dataset="wikisql"),
        dict(id="b", prompt="sql", dataset="wikisql"),
    ]
    config = SimpleNamespace(seed=42)
    sample = [SimpleNamespace(tokens=[3])]
    with patch(
        "src.train_grpo.generate_batch",
        side_effect=[sample, RuntimeError("decode failed")],
    ):
        with pytest.raises(RuntimeError, match="decode failed"):
            write_predictions(output, rows, None, ToyTokenizer(), config)
    assert list(tmp_path.iterdir()) == []
    with patch("src.train_grpo.generate_batch", return_value=sample):
        write_predictions(output, rows, None, ToyTokenizer(), config)
        with pytest.raises(FileExistsError):
            write_predictions(output, rows, None, ToyTokenizer(), config)
    assert [json.loads(line)["seed"] for line in output.read_text().splitlines()] == [
        42,
        42,
    ]


def test_base_and_adapter_architectures(tmp_path):
    import pytest
    from test_reproduction_smoke import make_inputs
    from src.train_grpo import load_policy, load_state, main, validate_checkpoint_config

    base, data = make_inputs(tmp_path)
    shared = load_policy(str(base), None, 2, 8, "cpu", "none")
    assert shared.trainable_state() and all(
        "lora_" in k for k in shared.trainable_state()
    )
    assert shared.private_parameters("sql") == shared.private_parameters("code") == []
    shared.eval()
    ids = torch.tensor([[3, 4]])
    with torch.no_grad():
        logits = shared(input_ids=ids).logits
        with shared.policy.disable_adapter():
            torch.testing.assert_close(
                logits, shared(input_ids=ids).logits, rtol=0, atol=0
            )
    weights = tmp_path / "shared.pt"
    torch.save(shared.trainable_state(), weights)
    load_state(shared, weights, deploy=True)
    top = load_policy(str(base), None, 2, 8, "cpu", "top_k", 1)
    assert top.private_adapters.selected_layer_indices == (0,)
    with pytest.raises(ValueError, match="missing="):
        load_state(top, weights, deploy=True)
    (tmp_path / "sft_config.json").write_text(
        json.dumps(dict(adapter_type="none", rank=2))
    )
    config = SimpleNamespace(
        model=str(base),
        model_revision=None,
        rank=2,
        seed=2,
        adapter_type="top_k",
        adapter_dim=8,
        adapter_top_k=1,
    )
    with pytest.raises(ValueError, match="adapter_type"):
        validate_checkpoint_config(config, weights)
    prediction = tmp_path / "base-pred.jsonl"
    main(
        [
            "--model",
            str(base),
            "--device",
            "cpu",
            "--base-only",
            "--rank",
            "2",
            "--eval-file",
            str(data / "sql_eval.jsonl"),
            "--predict",
            str(prediction),
            "--sql-new-tokens",
            "2",
            "--code-new-tokens",
            "2",
            "--max-length",
            "32",
        ]
    )
    assert json.loads(prediction.read_text())["id"] == "fixture-eval"
