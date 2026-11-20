import argparse
import json
import random
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from transformers import AutoTokenizer

from . import distributed_training as distributed
from . import run_full_experiment as legacy
from .grpo_data import digest, load_rows
from .train_grpo import load_state, validate_checkpoint_config


def training_groups(records, batch_size, steps, seed):
    rng = random.Random(seed + 11)
    if steps is None:
        rng.shuffle(records)
        for offset in range(0, len(records), batch_size):
            yield records[offset:offset + batch_size]
    else:
        for _ in range(steps):
            yield rng.sample(records, batch_size)


def main():
    parser = argparse.ArgumentParser(description="Train the SQL initialization or continue a seed's SFT checkpoint on SQL.\n\nFor the matched continued-SFT control, pass --continue-from COMMON_SQL_STATE\n--steps 142 --batch-size 256. Score continued_sql_trainable.pt directly, then\nuse src.interpolate_checkpoint with COMMON_SQL_STATE and the continued checkpoint\nat --alpha 0.8 to evaluate the interpolated control without retraining.\n")
    parser.add_argument("--model", required=True, help="Local model directory or Hugging Face ID")
    parser.add_argument("--model-revision", default="", help="Immutable Hub commit for a remote model")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--rank", type=int)
    parser.add_argument("--adapter-dim", type=int, default=96)
    parser.add_argument("--adapter-type", choices=["none", "output", "top_k"], default="output",
                        help="Shared LoRA only, output residual adapters, or top-k layer adapters")
    parser.add_argument("--adapter-top-k", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=16, help="Global examples per optimizer step")
    parser.add_argument("--micro-batch", type=int, default=1, help="Examples per device forward pass")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--continue-from", type=Path, help="SQL-SFT initialization for continued SFT")
    parser.add_argument("--steps", type=int, help="Optimizer updates for continued SFT")
    args = parser.parse_args()
    if min(args.batch_size, args.micro_batch, args.adapter_dim, args.rank if args.rank is not None else 1) < 1 or args.max_length < 2 or args.learning_rate <= 0:
        parser.error("Expected positive training settings and --max-length >= 2")
    train_file = args.data_dir / "sql_train.jsonl"
    records = load_rows(train_file, "sql")
    if (args.steps is None) != (args.continue_from is None):
        parser.error("Provide --continue-from and --steps together")
    if args.steps is not None and (args.steps < 1 or args.batch_size > len(records)):
        parser.error("Continued SFT needs positive steps and a batch no larger than the SQL split")
    with distributed.distributed_training(args.device) as device:
        train(args, records, train_file, torch.device(device))


def train(args, records, train_file, device):
    primary = distributed.rank() == 0
    if primary:
        if args.output_dir.exists() and any(args.output_dir.iterdir()):
            raise ValueError("Choose an empty output directory")
        args.output_dir.mkdir(parents=True, exist_ok=True)
    distributed.barrier()
    legacy.core.seed_everything(args.seed)
    args.rank = args.rank if args.rank is not None else legacy.paper_lora_rank(args.model)
    if args.continue_from is not None:
        validate_checkpoint_config(args, args.continue_from)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.model_revision or None)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = legacy.core.DomainModel(args.model, args.adapter_dim, args.rank, adapter_type=args.adapter_type,
                                   adapter_top_k=args.adapter_top_k, model_revision=args.model_revision)
    model.to(device)
    if device.type == "cpu":
        model.float()
    if args.continue_from is not None:
        load_state(model, args.continue_from)
    if distributed.world_size() > 1:
        for param in model.parameters():
            if param.requires_grad:
                torch.distributed.broadcast(param.detach(), src=0)
        torch.manual_seed(args.seed + distributed.rank())
    trainable = model.shared_parameters() + model.private_parameters("sql")
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, betas=(0.9, 0.95),
                                  weight_decay=0.01)
    if primary:
        config = {**vars(args), "world_size": distributed.world_size(),
                  "device": str(device), "epochs": 1 if args.steps is None else None,
                  "examples": len(records), "input_sha256": digest(train_file),
                  "model_commit": getattr(model.policy.config, "_commit_hash", None)}
        if args.continue_from is not None:
            config["initial_checkpoint_sha256"] = digest(args.continue_from)
        (args.output_dir / "sft_config.json").write_text(json.dumps(config, default=str, indent=2) + "\n")
    model.train()
    log_context = (args.output_dir / "sft_steps.jsonl").open("w") if primary else nullcontext()
    with model.domain("sql"), log_context as log:
        groups = training_groups(records, args.batch_size, args.steps, args.seed)
        for step, group in enumerate(groups, 1):
            start = time.monotonic()
            shard = distributed.shard(group)
            batches = []
            for i in range(0, len(shard), args.micro_batch):
                micro = shard[i:i + args.micro_batch]
                batches.append(legacy.encode_supervised_batch(
                    tokenizer, [r["prompt"] for r in micro], [r["gold_sql"] for r in micro],
                    args.max_length, torch.device("cpu")))
            counts = [int((batch["labels"][:, 1:] != -100).sum()) for batch in batches]
            total = distributed.sum_tensor_(torch.tensor(sum(counts), device=device)).item()
            if total == 0:
                raise ValueError("No supervised target tokens in this batch")
            optimizer.zero_grad(set_to_none=True)
            loss_total = torch.zeros((), dtype=torch.float64, device=device)
            for batch, count in zip(batches, counts):
                if count == 0:
                    continue
                inputs = {k: v.to(device) for k, v in batch.items()}
                loss = model(**inputs, use_cache=False).loss * (count / total)
                loss.backward()
                loss_total.add_(loss.detach())
            if distributed.world_size() > 1:
                distributed.reduce_gradients(trainable)
            torch.nn.utils.clip_grad_norm_(trainable, 1., error_if_nonfinite=True)
            optimizer.step()
            distributed.sum_tensor_(loss_total)
            if primary:
                log.write(json.dumps(dict(step=step, loss=loss_total.item(), examples=len(group),
                                          target_tokens=total, seconds=time.monotonic()-start)) + "\n")
                log.flush()
    name = "common_sql_trainable.pt" if args.continue_from is None else "continued_sql_trainable.pt"
    if primary:
        legacy.atomic_save_trainable(model, args.output_dir / name)


if __name__ == "__main__":
    main()
