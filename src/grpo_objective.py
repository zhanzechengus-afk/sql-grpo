from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor


@torch.no_grad()
def group_advantages(rewards: Tensor, eps: float = 1e-6) -> Tensor:
    if rewards.ndim != 2 or rewards.shape[1] < 2:
        raise ValueError("Expected rewards [prompts, responses] with at least two responses per prompt")
    scores = rewards if rewards.dtype == torch.float64 else rewards.float()
    if not torch.isfinite(scores).all():
        raise ValueError("Non-finite rollout reward")
    return (scores - scores.mean(-1, keepdim=True)) / (scores.std(-1, keepdim=True) + eps)


def grpo_loss(
    logp: Tensor, old_logp: Tensor, advantages: Tensor, mask: Tensor, *,
    clip_eps: float = 0.2, ref_logp: Tensor | None = None, kl_coef: float = 0.,
    normalizer: int | None = None, backend: str = "torch",
) -> Tensor:
    if logp.ndim != 2 or old_logp.shape != logp.shape or mask.shape != logp.shape:
        raise ValueError("Expected matching log-probability and mask shapes [responses, tokens]")
    if advantages.shape != (logp.shape[0],):
        raise ValueError("Expected one advantage per response")
    if not 0. <= clip_eps < 1. or kl_coef < 0.:
        raise ValueError("Invalid clipping or KL coefficient")
    if kl_coef and ref_logp is None:
        raise ValueError("A positive KL coefficient requires reference log probabilities")
    if ref_logp is not None and ref_logp.shape != logp.shape:
        raise ValueError("Reference log probabilities must match the actor shape")
    if backend not in {"torch", "verl"}:
        raise ValueError("backend must be torch or verl")
    keep = mask.bool()
    dtype = torch.float64 if logp.dtype == torch.float64 else torch.float32
    current = logp[keep].to(dtype)
    count = current.numel()
    if normalizer is not None and (normalizer <= 0 or normalizer < count):
        raise ValueError("Domain-wide token count must be positive and cover this microbatch")
    if count == 0:
        return current.sum()
    denominator = count if normalizer is None else normalizer
    old = old_logp.detach()[keep].to(dtype)
    adv = advantages.detach()[:, None].expand_as(logp)[keep].to(dtype)
    if backend == "verl":
        from verl.trainer.ppo.core_algos import compute_policy_loss, kl_penalty
        loss, *_ = compute_policy_loss(
            old[None, :], current[None, :], adv[None, :], torch.ones_like(current)[None, :],
            cliprange=clip_eps, loss_agg_mode="token-mean",
        )
        loss_sum = loss * count
    else:
        ratio = (current - old).clamp(-20, 20).exp()
        clipped = ratio.clamp(1. - clip_eps, 1. + clip_eps)
        surrogate = torch.minimum(ratio * adv, clipped * adv)
        surrogate = torch.where(adv < 0, torch.maximum(surrogate, 3. * adv), surrogate)
        loss_sum = -surrogate.sum()

    if kl_coef:
        reference = ref_logp.detach()[keep].to(dtype)
        if backend == "verl":
            penalty = kl_penalty(current, reference, "low_var_kl")
        else:
            gap = (reference - current).clamp(-20, 20)
            penalty = (torch.expm1(gap) - gap).clamp(0., 10.)
        loss_sum = loss_sum + kl_coef * penalty.sum()
    return loss_sum / denominator


def _check_layout(a: Sequence[Tensor], b: Sequence[Tensor]) -> None:
    if not a or len(a) != len(b) or any(x.shape != y.shape or x.device != y.device for x, y in zip(a, b)):
        raise ValueError("Gradient lists must follow the same non-empty parameter layout")


def _dot(a: Sequence[Tensor], b: Sequence[Tensor]) -> Tensor:
    return torch.stack([torch.dot(x.reshape(-1).float(), y.reshape(-1).float()) for x, y in zip(a, b)]).sum()


@torch.no_grad()
def update_anchor(previous: Sequence[Tensor] | None, grads: Sequence[Tensor], decay: float = .90) -> list[Tensor]:
    if not 0. <= decay < 1.:
        raise ValueError("Anchor EMA decay must be in [0, 1)")
    _check_layout(grads, grads if previous is None else previous)
    inverse_norm = _dot(grads, grads).sqrt().clamp_min(1e-12).reciprocal()
    direction = [g.float() * inverse_norm for g in grads]
    if previous is not None:
        for d, p in zip(direction, previous):
            d.mul_(1. - decay).add_(p, alpha=decay)
        inverse_norm = _dot(direction, direction).sqrt().clamp_min(1e-12).reciprocal()
        for d in direction:
            d.mul_(inverse_norm)
    return direction


@torch.no_grad()
def align_gradients(
    sql: Sequence[Tensor], code: Sequence[Tensor], anchor: Sequence[Tensor] | None, *,
    weight: float = .30, project: bool = True, norm_match: bool = True, gate: bool = True,
    gate_denominator: float = .02, max_scale: float = 2.,
) -> tuple[list[Tensor], dict[str, float]]:
    _check_layout(sql, code)
    if gate and anchor is None:
        raise ValueError("Anchor gating requires an anchor direction")
    if anchor is not None:
        _check_layout(sql, anchor)
    if gate_denominator <= 0 or max_scale <= 0 or weight < 0:
        raise ValueError("Invalid gradient-alignment hyperparameters")
    eps = 1e-12
    sql_sq = _dot(sql, sql)
    sql_norm = sql_sq.sqrt()
    code_norm = _dot(code, code).sqrt()
    dot = _dot(sql, code)
    projection = dot.clamp_max(0.) / (sql_sq + eps) if project else dot.new_zeros(())
    combined = [c.float() - projection * s.float() for s, c in zip(sql, code)]
    projected_norm = _dot(combined, combined).sqrt()
    scale = (sql_norm / (projected_norm + eps)).clamp_max(max_scale) if norm_match else dot.new_ones(())
    diagnostics = {
        "sql_grad_norm": sql_norm, "code_grad_norm": code_norm,
        "sql_code_cosine": dot / (sql_norm * code_norm + eps),
        "conflict": (dot < 0).float(), "projected_code_grad_norm": projected_norm,
        "projected_sql_code_cosine": _dot(combined, sql) / (projected_norm * sql_norm + eps),
        "norm_scale": scale,
    }
    if anchor is not None:
        agreement = _dot(combined, anchor) / (projected_norm * _dot(anchor, anchor).sqrt() + eps)
        diagnostics["anchor_cosine"] = agreement
    gate_value = (agreement / gate_denominator).clamp(0., 1.) if gate else dot.new_ones(())
    diagnostics["anchor_gate"] = gate_value
    factor = weight * scale * gate_value
    for mixed, s in zip(combined, sql):
        mixed.mul_(factor).add_(s)
    values = torch.stack(list(diagnostics.values())).tolist()
    return combined, dict(zip(diagnostics, values))
