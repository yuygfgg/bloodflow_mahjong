from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from training.iql import (
    ActionValueNetwork,
    CriticConfig,
    IndependentCritics,
    StateValueNetwork,
    awr_actor_loss,
    awr_weights,
    critic_losses,
    double_q_huber_loss,
    expectile_loss,
    legal_cql_loss,
    policy_reference_kl,
)
from training.model import ACTION_SPACE_SIZE, BloodFlowTransformer


def tiny_config() -> CriticConfig:
    return CriticConfig(
        d_model=32,
        num_heads=4,
        static_layers=1,
        history_layers=1,
        ffn_dim=64,
        head_dim=48,
        max_history=16,
    )


def state(batch: int = 3, history: int = 6) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(71)
    tile_obs = torch.randint(
        0, 5, (batch, 10, 27), generator=generator, dtype=torch.uint8
    )
    melds = torch.full((batch, 4, 4, 3), 255, dtype=torch.uint8)
    meta = torch.zeros((batch, 34), dtype=torch.int32)
    meta[:, 4] = 40
    meta[:, 9] = history
    meta[:, 12:16] = 10_000
    events = torch.zeros((batch, history, 8), dtype=torch.int32)
    events[:, :, 0] = torch.arange(history) % 11
    lengths = torch.full((batch,), history, dtype=torch.int64)
    legal = torch.zeros((batch, ACTION_SPACE_SIZE), dtype=torch.bool)
    legal[:, :5] = True
    return tile_obs, melds, meta, events, lengths, legal


def test_default_critics_are_compact_and_fully_independent() -> None:
    actor_parameters = sum(parameter.numel() for parameter in BloodFlowTransformer().parameters())
    critics = IndependentCritics()
    components = (critics.q1, critics.q2, critics.v)
    counts = [sum(parameter.numel() for parameter in model.parameters()) for model in components]
    assert all(count < actor_parameters for count in counts)
    assert sum(counts) < actor_parameters

    parameter_ids = [{id(parameter) for parameter in model.parameters()} for model in components]
    assert parameter_ids[0].isdisjoint(parameter_ids[1])
    assert parameter_ids[0].isdisjoint(parameter_ids[2])
    assert parameter_ids[1].isdisjoint(parameter_ids[2])


def test_critic_forward_masks_illegal_q_actions_and_backpropagates() -> None:
    critics = IndependentCritics(tiny_config()).train()
    inputs = state()
    output = critics(*inputs)
    assert output.q1.values.shape == (3, ACTION_SPACE_SIZE)
    assert output.q2.values.shape == (3, ACTION_SPACE_SIZE)
    assert output.value.shape == (3,)
    assert torch.isfinite(output.q1.raw_values).all()
    assert torch.isfinite(output.q2.raw_values).all()
    assert torch.isfinite(output.value).all()
    minimum = torch.finfo(output.q1.values.dtype).min
    assert torch.all(output.q1.values[~inputs[-1]] == minimum)
    assert torch.all(output.q2.values[~inputs[-1]] == minimum)

    loss = (
        output.q1.values[inputs[-1]].mean()
        + output.q2.values[inputs[-1]].mean()
        + output.value.mean()
    )
    loss.backward()
    for model in (critics.q1, critics.q2, critics.v):
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.grad is not None
        ]
        assert gradients
        assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_individual_critic_network_shapes() -> None:
    inputs = state(batch=2)
    q = ActionValueNetwork(tiny_config()).eval()
    v = StateValueNetwork(tiny_config()).eval()
    with torch.no_grad():
        q_output = q(*inputs)
        value = v(*inputs[:-1])
    assert q_output.values.shape == q_output.raw_values.shape == (2, 115)
    assert value.shape == (2,)


def test_double_q_huber_and_expectile_match_hand_calculation() -> None:
    q1 = torch.zeros((2, ACTION_SPACE_SIZE))
    q2 = torch.zeros_like(q1)
    q1[0, 1], q1[1, 2] = 2.0, -2.0
    q2[0, 1], q2[1, 2] = 1.0, -1.0
    actions = torch.tensor([1, 2])
    returns = torch.tensor([0.5, -0.5])
    q1_loss, q2_loss = double_q_huber_loss(q1, q2, actions, returns)
    torch.testing.assert_close(
        q1_loss, F.huber_loss(torch.tensor([2.0, -2.0]), returns)
    )
    torch.testing.assert_close(
        q2_loss, F.huber_loss(torch.tensor([1.0, -1.0]), returns)
    )

    prediction = torch.tensor([0.0, 0.0, 1.0])
    target = torch.tensor([2.0, -2.0, 1.0])
    actual = expectile_loss(prediction, target, expectile=0.75, reduction="none")
    torch.testing.assert_close(actual, torch.tensor([3.0, 1.0, 0.0]))


def test_cql_uses_only_legal_actions() -> None:
    values = torch.zeros((1, ACTION_SPACE_SIZE))
    values[0, :3] = torch.tensor([1.0, 2.0, 3.0])
    values[0, 80] = 1_000.0
    legal = torch.zeros_like(values, dtype=torch.bool)
    legal[0, :3] = True
    loss = legal_cql_loss(values, legal, torch.tensor([1]))
    expected = torch.logsumexp(torch.tensor([1.0, 2.0, 3.0]), dim=0) - 2.0
    torch.testing.assert_close(loss, expected)


def test_critic_loss_detaches_q_target_from_value_regression() -> None:
    q1 = torch.zeros((2, ACTION_SPACE_SIZE), requires_grad=True)
    q2 = torch.zeros((2, ACTION_SPACE_SIZE), requires_grad=True)
    values = torch.tensor([0.25, -0.25], requires_grad=True)
    legal = torch.zeros_like(q1, dtype=torch.bool)
    legal[:, :3] = True
    actions = torch.tensor([0, 1])
    returns = torch.tensor([1.0, -1.0])
    losses = critic_losses(
        q1,
        q2,
        values,
        actions,
        returns,
        legal,
        expectile=0.7,
        cql_scale=0.5,
    )
    losses.value_loss.backward()
    assert values.grad is not None
    assert q1.grad is None
    assert q2.grad is None
    torch.testing.assert_close(losses.value_target, torch.zeros(2))
    assert torch.isfinite(losses.q_loss)


def test_mc_auxiliary_rows_only_change_q_regression() -> None:
    q1 = torch.zeros((2, ACTION_SPACE_SIZE))
    q2 = torch.zeros_like(q1)
    values = torch.zeros(2)
    legal = torch.zeros_like(q1, dtype=torch.bool)
    legal[:, :2] = True
    actions = torch.tensor([0, 1])
    returns = torch.tensor([0.0, 10.0])
    regular = torch.tensor([True, False])
    q1[1, 0] = 100.0
    q2[1, 0] = 100.0
    losses = critic_losses(
        q1,
        q2,
        values,
        actions,
        returns,
        legal,
        cql_scale=1.0,
        cql_sample_mask=regular,
        value_sample_mask=regular,
    )
    expected_cql = torch.log(torch.tensor(2.0))
    torch.testing.assert_close(losses.q1_cql, expected_cql)
    torch.testing.assert_close(losses.q2_cql, expected_cql)
    torch.testing.assert_close(losses.value_loss, torch.tensor(0.0))
    assert losses.q1_regression > 0
    assert losses.q2_regression > 0


def test_awr_clips_weights_and_only_updates_actor() -> None:
    actor_logits = torch.zeros((3, ACTION_SPACE_SIZE), requires_grad=True)
    reference_logits = torch.zeros_like(actor_logits, requires_grad=True)
    q1 = torch.zeros_like(actor_logits, requires_grad=True)
    q2 = torch.zeros_like(actor_logits, requires_grad=True)
    values = torch.zeros(3, requires_grad=True)
    actions = torch.tensor([0, 1, 2])
    q1.data[0, 0], q1.data[1, 1], q1.data[2, 2] = 1.0, 0.0, -1.0
    q2.data.copy_(q1.data)
    legal = torch.zeros_like(actor_logits, dtype=torch.bool)
    legal[:, :4] = True

    output = awr_actor_loss(
        actor_logits,
        actions,
        q1,
        q2,
        values,
        legal,
        reference_logits=reference_logits,
        beta=0.5,
        max_weight=3.0,
        reference_kl_scale=1.0,
    )
    expected_weights = torch.tensor([3.0, 1.0, math.exp(-2.0)])
    torch.testing.assert_close(output.weights, expected_weights)
    torch.testing.assert_close(output.reference_kl, torch.tensor(0.0))
    assert 1.0 <= output.effective_sample_size <= 3.0
    output.loss.backward()
    assert actor_logits.grad is not None
    assert torch.isfinite(actor_logits.grad).all()
    assert reference_logits.grad is None
    assert q1.grad is None
    assert q2.grad is None
    assert values.grad is None


def test_policy_reference_kl_detects_post_update_change() -> None:
    actor = torch.zeros((1, ACTION_SPACE_SIZE))
    reference = torch.zeros_like(actor)
    legal = torch.zeros_like(actor, dtype=torch.bool)
    legal[:, :3] = True
    torch.testing.assert_close(
        policy_reference_kl(actor, reference, legal), torch.tensor(0.0)
    )
    actor[0, 0] = 2.0
    assert policy_reference_kl(actor, reference, legal) > 0

    direct_weights = awr_weights(
        torch.tensor([100.0, 0.0, -100.0]), beta=0.1, max_weight=7.0
    )
    torch.testing.assert_close(direct_weights[0], torch.tensor(7.0))
    torch.testing.assert_close(direct_weights[1], torch.tensor(1.0))
    assert torch.isfinite(direct_weights).all()


def test_losses_reject_states_without_legal_actions() -> None:
    values = torch.zeros((1, ACTION_SPACE_SIZE))
    legal = torch.zeros_like(values, dtype=torch.bool)
    with pytest.raises(RuntimeError, match="at least one legal action"):
        legal_cql_loss(values, legal, torch.tensor([0]))
