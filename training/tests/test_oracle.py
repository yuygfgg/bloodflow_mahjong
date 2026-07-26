from __future__ import annotations

import torch

from training.iql import CriticConfig
from training.model import ACTION_SPACE_SIZE, BloodFlowTransformer
from training.oracle import ORACLE_TILE_PLANES, OracleCritics, distillation_loss


def _inputs(batch: int = 2) -> tuple[torch.Tensor, ...]:
    tile_obs = torch.zeros((batch, 10, 27), dtype=torch.uint8)
    melds = torch.full((batch, 4, 4, 3), 255, dtype=torch.uint8)
    meta = torch.zeros((batch, 34), dtype=torch.int32)
    events = torch.zeros((batch, 2, 8), dtype=torch.int32)
    lengths = torch.ones(batch, dtype=torch.int64)
    oracle = torch.zeros((batch, ORACLE_TILE_PLANES, 27), dtype=torch.uint8)
    legal = torch.zeros((batch, ACTION_SPACE_SIZE), dtype=torch.bool)
    legal[:, :3] = True
    return tile_obs, melds, meta, events, lengths, oracle, legal


def test_oracle_critics_are_separate_from_actor_and_mask_actions() -> None:
    config = CriticConfig(
        d_model=32,
        num_heads=4,
        static_layers=1,
        history_layers=1,
        ffn_dim=64,
        head_dim=48,
        max_history=8,
    )
    actor = BloodFlowTransformer()
    oracle = OracleCritics(config)
    actor_ids = {id(parameter) for parameter in actor.parameters()}
    oracle_ids = {id(parameter) for parameter in oracle.parameters()}
    assert actor_ids.isdisjoint(oracle_ids)
    output = oracle(*_inputs())
    assert output.q1.values.shape == (2, ACTION_SPACE_SIZE)
    assert output.value.shape == (2,)
    assert torch.all(output.q1.values[:, 3:] == torch.finfo(torch.float32).min)


def test_distillation_detaches_oracle_teacher() -> None:
    student1 = torch.zeros((1, ACTION_SPACE_SIZE), requires_grad=True)
    student2 = torch.zeros_like(student1, requires_grad=True)
    teacher1 = torch.ones_like(student1, requires_grad=True)
    teacher2 = torch.ones_like(student1, requires_grad=True)
    legal = torch.zeros_like(student1, dtype=torch.bool)
    legal[:, :2] = True
    actions = torch.tensor([1])
    loss = distillation_loss(
        student1, student2, teacher1, teacher2, legal, actions
    )
    loss.backward()
    assert student1.grad is not None or student2.grad is not None
    student_grad = student1.grad if student1.grad is not None else student2.grad
    assert student_grad is not None
    assert torch.count_nonzero(student_grad[:, :1]) == 0
    assert teacher1.grad is None
    assert teacher2.grad is None
