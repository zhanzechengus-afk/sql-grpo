from __future__ import annotations

import argparse
import functools
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
import time

import torch
import torch.nn.functional as F

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.distributed_training import (
    barrier, distributed_training, max_tensor_, rank, reduce_gradients, shard,
    sum_tensor_, world_size,
)
from src.grpo_objective import align_gradients, update_anchor
from src.grpo_rollouts import (
    backward_domain,
    collect_domain,
    generate_batch,
    prompt_ids,
)
from src.interpolate_checkpoint import interpolate

CODE_PREFIX = "private_adapters.domain_adapters.code."


def load_policy(
    model_name,
    revision,
    rank,
    adapter_dim,
    device,
    adapter_type="output",
    adapter_top_k=6,
):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from src.run_pilot import DomainModel

    model = DomainModel(
        model_name,
        adapter_dim,
        rank,
        adapter_type=adapter_type,
        adapter_top_k=adapter_top_k,
        model_revision=revision or "",
    )
    model = model.to(device)
    if torch.device(device).type == "cpu":
        model.float()
    return model


def load_state(model, path, *, deploy=False):
    state = torch.load(path, map_location="cpu", weights_only=True)
    needed = set(model.trainable_state())
    if deploy:
        needed = {key for key in needed if not key.startswith(CODE_PREFIX)}
    fixed = {
        ("private_adapters." + key if key.startswith("domain_adapters.") else key): val
        for key, val in state.items()
    }
    missing = needed - set(fixed)
    extra = set(fixed) - set(model.trainable_state())
    if missing or extra:
        raise ValueError(
            f"trainable state does not match model: missing={sorted(missing)}, extra={sorted(extra)}"
        )
    model.load_state_dict(fixed, strict=False)
    return fixed


def save_state(path, state):
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
        temp = Path(f.name)
    try:
        torch.save(state, temp)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def checkpoint_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for data in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def validate_checkpoint_config(config, checkpoint=None):
    checkpoint_dir = Path(checkpoint or config.sft_checkpoint).parent
    for name in ("resolved_config.json", "sft_config.json"):
        path = checkpoint_dir / name
        if not path.is_file():
            continue
        recorded = json.loads(path.read_text())
        keys = ["model", "seed", "rank", "adapter_type"]
        if config.adapter_type != "none":
            keys.append("adapter_dim")
        if config.adapter_type == "top_k":
            keys.append("adapter_top_k")
        if recorded.get("model_revision") and not config.model_revision:
            config.model_revision = recorded["model_revision"]
        if recorded.get("model_revision"):
            keys.append("model_revision")
        for key in keys:
            current = getattr(config, key)
            expected = recorded.get(key)
            if (
                key == "model"
                and expected
                and Path(expected).is_dir()
                and Path(current).is_dir()
            ):
                expected, current = Path(expected).resolve(), Path(current).resolve()
            if expected is not None and current != expected:
                raise ValueError(
                    f"{key} {current!r} differs from checkpoint {key} {expected!r}"
                )
        return


def anchor_direction(model, tokenizer, rows, config, previous):
    device = next(model.parameters()).device
    shared = model.shared_parameters()
    sums = [torch.zeros_like(p, dtype=torch.float32) for p in shared]
    for row in rows:
        prompt = prompt_ids(tokenizer, row["prompt"])
        target = tokenizer.encode(row["gold_sql"], add_special_tokens=False)
        if tokenizer.eos_token_id is not None:
            target = list(target[: config.max_length - 2]) + [tokenizer.eos_token_id]
        target = target[: config.max_length - 1]
        if not target:
            raise ValueError(f"empty SQL target for {row['id']}")
        prompt = prompt[-(config.max_length - len(target)) :]
        seq = torch.tensor([prompt + target], device=device)
        with model.domain("sql"):
            logits = model(input_ids=seq, use_cache=False).logits
            start = len(prompt) - 1
            loss = F.cross_entropy(logits[0, start:-1].float(), seq[0, start + 1 :])
            gradient = torch.autograd.grad(loss, shared)
        with torch.no_grad():
            for accum, grad in zip(sums, gradient):
                accum.add_(grad)
    for total in sums:
        sum_tensor_(total)
    return update_anchor(previous, sums, config.anchor_ema)


def take_shared(parameters):
    collected = []
    for param in parameters:
        if param.grad is None:
            raise RuntimeError("missing shared LoRA gradient")
        collected.append(param.grad.detach())
        param.grad = None
    return collected


def domain_step(model, reference, tokenizer, domain, examples, scorer, config,
                generator, reward_rng=None):
    device = next(model.parameters()).device
    auxiliary_stats = {} if domain == "code" else None
    local_tokens = local_responses = 0
    reward = 0.0
    if examples:
        batch, reward = collect_domain(
            model, tokenizer, domain, examples, scorer, config, generator,
            reward_rng=reward_rng, reward_stats=auxiliary_stats,
        )
        local_responses = len(batch)
        local_tokens = backward_domain(model, reference, tokenizer, domain, batch, config)

    counts = [local_tokens, local_responses, reward * local_responses]
    auxiliary_keys = ("aux_prompt_groups", "aux_variable_reward_groups",
                      "aux_responses", "aux_changed_rewards")
    if auxiliary_stats is not None:
        counts.extend(auxiliary_stats.get(key, 0) for key in auxiliary_keys)
    totals = sum_tensor_(torch.tensor(counts, device=device, dtype=torch.float64)).tolist()
    tokens, responses, reward_sum = totals[:3]
    if world_size() > 1:
        active = model.shared_parameters() + model.private_parameters(domain)
        reduce_gradients(active, local_tokens / tokens)
    metrics = {
        f"{domain}_tokens": int(tokens),
        f"{domain}_responses": int(responses),
        f"{domain}_reward": reward_sum / responses,
    }
    if auxiliary_stats is not None:
        metrics.update(zip(auxiliary_keys, map(int, totals[3:])))
    return metrics


def train(model, reference, tokenizer, sft_state, sql, anchors, code, config, output):
    from src.grpo_rewards import sql_reward, code_reward

    if len(sql) < config.sql_prompts or (
        config.branch != "sql_only" and len(code) < config.code_prompts
    ):
        raise ValueError(
            "not enough distinct prompts in prepared split for an iteration"
        )
    use_gate = config.branch == "full" and config.gate
    track_anchor = use_gate or getattr(config, "log_anchor", False)
    if track_anchor and len(anchors) < config.anchor_batch:
        raise ValueError("not enough distinct reserved SQL anchor examples")
    if {r["id"] for r in sql} & {r["id"] for r in anchors}:
        raise ValueError("SQL training and anchor IDs must be disjoint")
    shared = model.shared_parameters()
    model.eval()
    if reference is not None:
        reference.eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=config.lr,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )
    device = next(model.parameters()).device
    streams = {
        name: random.Random(config.seed + offset)
        for name, offset in (("sql", 0), ("anchor", 1), ("code", 2))
    }
    worker_seed = config.seed + 1000003 * rank()
    generators = {
        name: torch.Generator(device=device).manual_seed(worker_seed + offset)
        for name, offset in (("sql", 0), ("code", 1))
    }
    auxiliary_mode = getattr(config, "auxiliary_mode", "verified")
    if auxiliary_mode == "arithmetic":
        from src.arithmetic_control import arithmetic_reward

        code_scorer = arithmetic_reward
    else:
        code_scorer = functools.partial(code_reward, image=config.code_image)
    reward_rng = random.Random(worker_seed + 3) if auxiliary_mode == "shuffled" else None
    anchor = None
    for i in range(config.steps):
        started = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        if track_anchor and i % config.anchor_refresh == 0:
            selected = streams["anchor"].sample(anchors, config.anchor_batch)
            anchor = anchor_direction(model, tokenizer, shard(selected), config, anchor)

        selected = streams["sql"].sample(sql, config.sql_prompts)
        metrics = domain_step(
            model, reference, tokenizer, "sql", shard(selected), sql_reward,
            config, generators["sql"],
        )
        metrics["step"] = i + 1
        if config.branch != "sql_only":
            sql_gradient = take_shared(shared)
            selected = streams["code"].sample(code, config.code_prompts)
            metrics.update(domain_step(
                model, None, tokenizer, "code", shard(selected), code_scorer,
                config, generators["code"], reward_rng=reward_rng,
            ))
            code_gradient = take_shared(shared)
            combined, diagnostics = align_gradients(
                sql_gradient,
                code_gradient,
                anchor,
                weight=config.code_weight,
                project=config.project,
                norm_match=config.norm_match,
                gate=use_gate,
                gate_denominator=config.gate_denominator,
                max_scale=config.max_scale,
            )
            for parameter, grad in zip(shared, combined):
                parameter.grad = grad.to(dtype=parameter.dtype)
            metrics.update(diagnostics)
            del combined, code_gradient, sql_gradient
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            config.grad_clip,
            error_if_nonfinite=True,
        )
        optimizer.step()
        peak_bytes = 0
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak_bytes = torch.cuda.max_memory_allocated(device)
        seconds, peak_bytes = max_tensor_(torch.tensor(
            [time.monotonic() - started, peak_bytes], dtype=torch.float64, device=device
        )).tolist()
        if rank() == 0:
            if device.type == "cuda":
                metrics["peak_cuda_allocated_bytes"] = int(peak_bytes)
            metrics.update(grad_norm=float(grad_norm), seconds=seconds)
            with (output / "metrics.jsonl").open("a") as f:
                f.write(json.dumps(metrics, allow_nan=False) + "\n")
            if config.save_every and (i + 1) % config.save_every == 0:
                save_state(output / f"step_{i + 1:04d}.pt", model.trainable_state())

    if rank() != 0:
        return
    final_state = model.trainable_state()
    save_state(output / "post_rl_trainable.pt", final_state)
    mixed = interpolate(sft_state, final_state, config.alpha)
    save_state(
        output / "trainable.pt",
        {k: v for k, v in mixed.items() if not k.startswith(CODE_PREFIX)},
    )


def parser():
    p = argparse.ArgumentParser(
        description="Train GRPO on one device or with torchrun, or decode a SQL checkpoint."
    )
    p.add_argument(
        "--config",
        help="JSON object of command-line defaults; CLI arguments override it",
    )
    p.add_argument(
        "--model",
        default="Qwen/Qwen3-0.6B",
        help="Hugging Face model ID or local model directory",
    )
    p.add_argument("--model-revision", help="pinned backbone and tokenizer commit")
    p.add_argument(
        "--sft-checkpoint",
        help="trainable .pt state from SQL SFT; for prediction, use deployed trainable.pt",
    )
    p.add_argument("--data-dir", help="directory containing prepared JSONL partitions")
    p.add_argument("--output-dir", help="new directory for one training run")
    p.add_argument(
        "--predict", help="write greedy SQL predictions to this new JSONL file"
    )
    p.add_argument(
        "--base-only",
        action="store_true",
        help="with --predict, decode the pretrained backbone before SQL SFT",
    )
    p.add_argument(
        "--eval-file", help="prepared held-out SQL JSONL file for prediction"
    )
    p.add_argument(
        "--branch", choices=["sql_only", "full", "zero_code"], default="full"
    )
    p.add_argument(
        "--auxiliary-mode", choices=["verified", "shuffled", "arithmetic"],
        default="verified",
        help="Use verified code rewards, shuffle rewards within each code prompt, or use graded arithmetic.",
    )
    p.add_argument("--backend", choices=["torch", "verl"], default="torch")
    p.add_argument("--code-image", default="python:3.11-slim")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--project", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--norm-match", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--gate", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--log-anchor", action="store_true",
                   help="Measure anchor agreement even when the full branch's gate is disabled.")
    p.add_argument("--rank", type=int, help="defaults to 8 (Qwen) or 16 (SmolLM3)")
    p.add_argument(
        "--adapter-type",
        choices=["none", "output", "top_k"],
        default="output",
        help="Shared LoRA only, output residual adapters, or top-k layer adapters",
    )
    for name, default in {
        "steps": 142,
        "sql-prompts": 256,
        "code-prompts": 256,
        "group-size": 16,
        "anchor-batch": 16,
        "anchor-refresh": 8,
        "seed": 2,
        "adapter-dim": 96,
        "adapter-top-k": 6,
        "max-length": 512,
        "sql-new-tokens": 256,
        "code-new-tokens": 128,
        "rollout-batch": 4,
        "micro-batch": 1,
        "save-every": 0,
    }.items():
        p.add_argument("--" + name, type=int, default=default)
    for name, default in {
        "lr": 1e-5,
        "clip": 0.2,
        "kl": 0.001,
        "alpha": 0.8,
        "code-weight": 0.30,
        "gate-denominator": 0.02,
        "max-scale": 2.0,
        "anchor-ema": 0.90,
        "grad-clip": 1.0,
        "temperature": 0.8,
    }.items():
        p.add_argument("--" + name, type=float, default=default)
    return p


def parse_config(argv=None):
    p = parser()
    preliminary, _ = p.parse_known_args(argv)
    if preliminary.config:
        options = json.loads(Path(preliminary.config).read_text())
        known = {action.dest for action in p._actions} - {"config", "help"}
        if not isinstance(options, dict) or set(options) - known:
            p.error("config has an unknown key or is not a JSON object")
        for action in p._actions:
            if (
                action.dest in options
                and action.choices
                and options[action.dest] not in action.choices
            ):
                p.error(f"invalid config choice for {action.dest}")
        p.set_defaults(**options)
    args = p.parse_args(argv)
    if args.auxiliary_mode != "verified" and args.branch != "full":
        p.error("auxiliary reward controls require --branch full")
    if args.log_anchor and args.branch != "full":
        p.error("--log-anchor requires --branch full")
    if args.rank is None:
        args.rank = 16 if "smollm3" in args.model.lower() else 8
    if args.device != "cpu" and not args.device.startswith("cuda"):
        p.error("use --device cpu or cuda[:index]")
    if args.base_only and (not args.predict or args.sft_checkpoint):
        p.error("--base-only requires --predict and excludes --sft-checkpoint")
    if not args.sft_checkpoint and not args.base_only:
        p.error("--sft-checkpoint is required")
    if args.predict:
        if not args.eval_file:
            p.error("prediction requires --eval-file")
    elif not args.output_dir or not args.data_dir:
        p.error("training requires --data-dir and --output-dir")
    for name in (
        "steps",
        "sql_prompts",
        "code_prompts",
        "anchor_batch",
        "anchor_refresh",
        "adapter_dim",
        "rank",
        "rollout_batch",
        "micro_batch",
        "sql_new_tokens",
        "code_new_tokens",
    ):
        if getattr(args, name) <= 0:
            p.error(f"{name} must be positive")
    if args.adapter_type == "top_k" and args.adapter_top_k < 1:
        p.error("--adapter-top-k must be positive for top_k adapters")
    if args.group_size < 2 or args.max_length <= max(
        args.sql_new_tokens, args.code_new_tokens
    ):
        p.error(
            "group_size must be >=2 and max_length must exceed both completion budgets"
        )
    if not (
        0 <= args.alpha <= 1
        and 0 <= args.anchor_ema < 1
        and args.kl >= 0
        and args.code_weight >= 0
        and 0 < args.clip < 1
        and args.lr > 0
        and args.grad_clip > 0
        and args.temperature > 0
        and args.save_every >= 0
        and args.gate_denominator > 0
        and args.max_scale > 0
    ):
        p.error("invalid optimizer, alignment, sampling or interpolation setting")
    if any(not math.isfinite(x) for x in vars(args).values() if isinstance(x, float)):
        p.error("non-finite numeric setting")
    return args


def write_predictions(destination, rows, model, tokenizer, config):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=destination.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        try:
            for row in rows:
                [response] = generate_batch(
                    model, tokenizer, "sql", row["prompt"], 1, config, greedy=True
                )
                record = {
                    "id": row["id"],
                    "dataset": row["dataset"],
                    "seed": config.seed,
                    "prediction": tokenizer.decode(
                        response.tokens, skip_special_tokens=True
                    ),
                }
                handle.write(json.dumps(record) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            os.link(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)


def write_run_config(output, paths, model, c):
    specs = {
        **vars(c),
        "sha256": {key: checkpoint_hash(path) for key, path in paths.items()},
    }
    specs["model_commit"] = getattr(model.policy.config, "_commit_hash", None)
    specs["packages"] = {
        key: importlib.metadata.version(key)
        for key in ("torch", "transformers", "peft", "datasets", "sqlglot")
    }
    if c.backend == "verl":
        specs["packages"]["verl"] = importlib.metadata.version("verl")
    uses_cuda = torch.device(c.device).type == "cuda"
    specs["hardware"] = {
        "device": c.device,
        "world_size": world_size(),
        "accelerators_used": world_size() if uses_cuda else 0,
    }
    if uses_cuda:
        specs["hardware"].update(
            gpu=torch.cuda.get_device_name(c.device), cuda_version=torch.version.cuda
        )
    specs["sha256"]["sft_checkpoint"] = checkpoint_hash(c.sft_checkpoint)
    (output / "resolved_config.json").write_text(json.dumps(specs, indent=2) + "\n")


def run(c):
    from transformers import AutoTokenizer
    from src.grpo_data import load_rows, write_rows

    tokenizer = AutoTokenizer.from_pretrained(c.model, revision=c.model_revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise ValueError("tokenizer needs a pad token or EOS token")
    torch.manual_seed(c.seed)
    model_args = (
        c.model,
        c.model_revision,
        c.rank,
        c.adapter_dim,
        c.device,
        c.adapter_type,
        c.adapter_top_k,
    )
    model = load_policy(*model_args)
    if c.sft_checkpoint:
        state = load_state(model, c.sft_checkpoint, deploy=bool(c.predict))
    model.eval()
    if c.predict:
        rows = load_rows(Path(c.eval_file), "sql")
        destination = Path(c.predict)
        write_predictions(destination, rows, model, tokenizer, c)
        return
    source = Path(c.data_dir)
    paths = {"sql": source / "sql_train.jsonl", "anchor": source / "sql_anchor.jsonl"}
    if c.branch != "sql_only" and c.auxiliary_mode != "arithmetic":
        paths["code"] = source / "code_train.jsonl"
    datasets = {
        name: load_rows(path, "code" if name == "code" else "sql")
        for name, path in paths.items()
    }
    reference = None
    if c.kl:
        reference = load_policy(*model_args)
        reference.load_state_dict(state, strict=False)
    output = Path(c.output_dir)
    if c.auxiliary_mode == "arithmetic":
        from src.arithmetic_control import prepare_arithmetic_pool

        datasets["code"] = prepare_arithmetic_pool(count=1362, seed=0)
        paths["arithmetic"] = output / "arithmetic_pool.jsonl"
    if rank() == 0:
        output.mkdir(parents=True, exist_ok=False)
        if c.auxiliary_mode == "arithmetic":
            write_rows(paths["arithmetic"], datasets["code"])
        write_run_config(output, paths, model, c)
    barrier()
    train(
        model,
        reference,
        tokenizer,
        state,
        datasets["sql"],
        datasets["anchor"],
        datasets.get("code", []),
        c,
        output,
    )
    barrier()


def main(argv=None):
    c = parse_config(argv)
    if c.sft_checkpoint:
        validate_checkpoint_config(c)
    with distributed_training(c.device, predict=bool(c.predict)) as device:
        c.device = device
        run(c)


if __name__ == "__main__":
    main()
