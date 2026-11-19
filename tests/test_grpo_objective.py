import math

import pytest
import torch

from src.grpo_objective import align_gradients, grpo_loss, group_advantages, update_anchor


def test_group_baseline_uses_only_the_other_rollouts_for_that_prompt():
    scores = torch.tensor([[8., 8., 8.], [4., 5., 6.]])
    torch.testing.assert_close(group_advantages(scores), torch.tensor([[0., 0., 0.], [-1., 0., 1.]]))


def test_padding_and_microbatching_preserve_token_mean_and_update():
    old = torch.tensor([[-2., float('nan'), float('nan')], [-2., -2., -2.]], requires_grad=True)
    current = old.detach().clone().requires_grad_()
    mask = torch.tensor([[1, 0, 0], [1, 1, 1]])
    adv = torch.tensor([1., -1.])
    loss = grpo_loss(current, old, adv, mask)
    micro = sum(grpo_loss(current[i:i+1], old[i:i+1], adv[i:i+1], mask[i:i+1], normalizer=4) for i in (0, 1))
    assert loss.item() == micro.item() == .5
    loss.backward()
    assert old.grad is None
    torch.testing.assert_close(current.grad, torch.tensor([[-.25, 0., 0.], [.25, .25, .25]]))
    current.grad.zero_()
    micro.backward()
    torch.testing.assert_close(current.grad, torch.tensor([[-.25, 0., 0.], [.25, .25, .25]]))


def test_dual_clip_and_reference_kl():
    old = torch.full((2, 1), -3.)
    current = torch.zeros_like(old)
    adv = torch.tensor([1., -1.])
    mask = torch.ones_like(old)
    assert grpo_loss(current, old, adv, mask).item() == pytest.approx((3. - 1.2) / 2.)
    zero = torch.zeros_like(adv)
    val = grpo_loss(current, old, zero, mask, ref_logp=old, kl_coef=.1)
    assert val.item() == pytest.approx(.1 * (math.exp(-3.) + 2.))


def test_flat_geometry_across_multiple_parameter_tensors_and_gate_ablation():
    sql = [torch.tensor([3.]), torch.tensor([0.])]
    code = [torch.tensor([-2.]), torch.tensor([1.])]
    anchor = [torch.tensor([0.]), torch.tensor([1.])]
    grad, log = align_gradients(sql, code, anchor)
    torch.testing.assert_close(torch.cat(grad), torch.tensor([3., .6]))
    assert log['conflict'] == 1. and log['norm_scale'] == 2. and log['anchor_gate'] == 1.
    opposite = [-v for v in anchor]
    grad, log = align_gradients(sql, code, opposite)
    assert log['anchor_gate'] == 0.
    torch.testing.assert_close(torch.cat(grad), torch.cat(sql))
    grad, _ = align_gradients(sql, code, None, project=False, norm_match=False, gate=False)
    torch.testing.assert_close(torch.cat(grad), torch.tensor([2.4, .3]))


def test_anchor_ema_and_degenerate_directions():
    initial = update_anchor(None, [torch.tensor([0., 4.])])
    refreshed = update_anchor(initial, [torch.tensor([8., 0.])], decay=.5)
    torch.testing.assert_close(refreshed[0], torch.tensor([.5**.5, .5**.5]))
    z = [torch.zeros(2)]
    grad, diagnostics = align_gradients(z, z, update_anchor(None, z))
    assert all(math.isfinite(v) for v in diagnostics.values())
    torch.testing.assert_close(grad[0], z[0])


def test_native_verl_matches_portable_kernel_when_installed():
    pytest.importorskip('verl.trainer.ppo.core_algos')
    current = torch.tensor([[-.1, -.2], [-1.4, -.6]], requires_grad=True)
    old = torch.full_like(current, -1.2)
    ref = torch.full_like(current, -1.)
    adv = torch.tensor([-1., 1.])
    mask = torch.tensor([[1, 0], [1, 1]])
    outputs = [grpo_loss(current, old, adv, mask, ref_logp=ref, kl_coef=.001, backend=b) for b in ('torch', 'verl')]
    torch.testing.assert_close(outputs[0], outputs[1])
    torch.testing.assert_close(*[torch.autograd.grad(x, current)[0] for x in outputs])


def test_empty_microbatch_contributes_zero_without_contaminating_gradients():
    current = torch.full((2, 3), float('nan'), requires_grad=True)
    old = torch.full_like(current, float('nan'))
    loss = grpo_loss(current, old, torch.zeros(2), torch.zeros_like(current),
                     ref_logp=old, kl_coef=.001, normalizer=7)
    assert loss.item() == 0.
    loss.backward()
    torch.testing.assert_close(current.grad, torch.zeros_like(current))
