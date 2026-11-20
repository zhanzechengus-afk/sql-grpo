from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .grpo_objective import group_advantages, grpo_loss


@dataclass
class Response:
    prompt: list[int]
    tokens: list[int]
    behavior_logp: list[float]
    advantage: float = 0.0


def prompt_ids(
    tokenizer, prompt, system_instruction="Return only an exact, executable solution."
):
    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": prompt},
    ]
    tokens = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, enable_thinking=False
    )
    if hasattr(tokens, "keys"):
        tokens = tokens["input_ids"]
    if isinstance(tokens, torch.Tensor):
        tokens = tokens.tolist()
    if not tokens or not isinstance(tokens[0], int):
        raise ValueError("chat template must produce a non-empty, flat sequence")
    return tokens


@torch.no_grad()
def generate_batch(
    model, tokenizer, domain, prompt, count, config, generator=None, greedy=False
):
    budget = config.sql_new_tokens if domain == "sql" else config.code_new_tokens
    if domain == "code" and getattr(config, "auxiliary_mode", "verified") == "arithmetic":
        prefix = prompt_ids(tokenizer, prompt, "Give the exact integer answer.")
    else:
        prefix = prompt_ids(tokenizer, prompt)
    prefix = prefix[-(config.max_length - budget) :]
    device = next(model.parameters()).device
    eos = set()
    for ids in (
        tokenizer.eos_token_id,
        getattr(getattr(model.policy, "generation_config", None), "eos_token_id", None),
    ):
        if ids is not None:
            eos.update(ids if isinstance(ids, (list, tuple)) else [ids])
    generated = []
    with model.domain(domain):
        for offset in range(0, count, config.rollout_batch):
            size = min(config.rollout_batch, count - offset)
            seq = torch.tensor([prefix] * size, device=device)
            ends = torch.full((size,), budget, device=device, dtype=torch.long)
            past = None
            actions, old_logp = [], []
            for step in range(budget):
                inputs = seq if past is None else seq[:, -1:]
                output = model(
                    input_ids=inputs,
                    attention_mask=torch.ones_like(seq),
                    past_key_values=past,
                    use_cache=True,
                )
                past = getattr(output, "past_key_values", None)
                logp = F.log_softmax(
                    output.logits[:, -1, :].float() / config.temperature, dim=-1
                )
                token = (
                    logp.argmax(-1, keepdim=True)
                    if greedy
                    else torch.multinomial(logp.exp(), 1, generator=generator)
                )
                actions.append(token[:, 0])
                old_logp.append(logp.gather(-1, token)[:, 0])
                seq = torch.cat((seq, token), dim=1)
                if eos:
                    stop = torch.zeros(size, device=device, dtype=torch.bool)
                    for end in eos:
                        stop |= token[:, 0].eq(end)
                    ends = torch.where(stop & ends.eq(budget), step + 1, ends)
                    if ends.le(step + 1).all().item():
                        break
            ids = torch.stack(actions, dim=1).tolist()
            scores = torch.stack(old_logp, dim=1).tolist()
            for row, scores_row, length in zip(ids, scores, ends.tolist()):
                generated.append(Response(prefix, row[:length], scores_row[:length]))
    return generated


def collect_domain(
    model, tokenizer, domain, examples, scorer, config, generator, *,
    reward_rng=None, reward_stats=None,
):
    responses, means = [], []
    changed_rewards = 0
    variable_groups = 0
    for example in examples:
        group = generate_batch(
            model,
            tokenizer,
            domain,
            example["prompt"],
            config.group_size,
            config,
            generator,
        )
        rewards = [
            scorer(tokenizer.decode(r.tokens, skip_special_tokens=True), example)
            for r in group
        ]
        if reward_stats is not None:
            variable_groups += min(rewards) != max(rewards)
        if reward_rng is not None:
            assigned = rewards.copy()
            reward_rng.shuffle(assigned)
            changed_rewards += sum(a != b for a, b in zip(rewards, assigned))
        else:
            assigned = rewards
        adv = group_advantages(torch.tensor([assigned], dtype=torch.float32))[0]
        for row, value in zip(group, adv.tolist()):
            row.advantage = (
                0.0 if domain == "code" and config.branch == "zero_code" else value
            )
        responses.extend(group)
        means.append(sum(rewards) / len(rewards))
    if reward_stats is not None:
        reward_stats.update(
            aux_prompt_groups=len(means),
            aux_variable_reward_groups=variable_groups,
            aux_responses=len(responses),
            aux_changed_rewards=changed_rewards,
        )
    return responses, sum(means) / len(means)


def backward_domain(model, reference, tokenizer, domain, responses, config):
    device = next(model.parameters()).device
    total = sum(len(r.tokens) for r in responses)
    if not total:
        raise ValueError("no response tokens")
    for offset in range(0, len(responses), config.micro_batch):
        group = responses[offset : offset + config.micro_batch]
        width = max(len(r.prompt) + len(r.tokens) for r in group)
        ids = torch.full(
            (len(group), width), tokenizer.pad_token_id, dtype=torch.long, device=device
        )
        attn = torch.zeros_like(ids)
        mask = torch.zeros_like(ids[:, :-1], dtype=torch.bool)
        old = torch.zeros_like(mask, dtype=torch.float32)
        for i, r in enumerate(group):
            end = len(r.prompt) + len(r.tokens)
            ids[i, :end] = torch.tensor(r.prompt + r.tokens, device=device)
            attn[i, :end] = 1
            selected = slice(len(r.prompt) - 1, end - 1)
            mask[i, selected] = True
            old[i, selected] = torch.tensor(r.behavior_logp, device=device)

        def token_logp(policy):
            with policy.domain(domain):
                logits = policy(
                    input_ids=ids, attention_mask=attn, use_cache=False
                ).logits
            scores = logits[:, :-1][mask].float() / config.temperature
            targets = ids[:, 1:][mask]
            selected = (
                F.log_softmax(scores, dim=-1).gather(1, targets[:, None]).squeeze(1)
            )
            return torch.zeros_like(old).masked_scatter(mask, selected)

        ref_logp = None
        if reference is not None:
            with torch.no_grad():
                ref_logp = token_logp(reference)
        adv = torch.tensor([r.advantage for r in group], device=device)
        loss = grpo_loss(
            token_logp(model),
            old,
            adv,
            mask,
            ref_logp=ref_logp,
            kl_coef=config.kl if ref_logp is not None else 0.0,
            clip_eps=config.clip,
            normalizer=total,
            backend=config.backend,
        )
        if not torch.isfinite(loss.detach()).item():
            raise FloatingPointError(f"non-finite {domain} policy loss")
        loss.backward()
    return total
