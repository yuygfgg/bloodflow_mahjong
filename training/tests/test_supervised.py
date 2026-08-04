from __future__ import annotations

from contextlib import closing
from dataclasses import replace
import threading

import numpy as np
import pytest
import torch

from training.contracts import TRAINING_INPUT_SCHEMA, validate_engine_contract
from training.model import BloodFlowTransformer, TransformerConfig
from training.policy import (
    actor_parameters,
    is_actor_parameter,
    load_actor_checkpoint,
    save_actor_checkpoint,
)
from training.supervised import (
    RuleEvCollector,
    SupervisedConfig,
    _batch_sizes,
    _prefetched_batches,
    supervised_update,
)


def tiny_model() -> BloodFlowTransformer:
    return BloodFlowTransformer(
        TransformerConfig(
            d_model=48,
            num_heads=4,
            static_layers=1,
            history_layers=1,
            ffn_dim=96,
            max_history=192,
            value_atoms=17,
        )
    )


def tiny_config(**changes: object) -> SupervisedConfig:
    config = SupervisedConfig(
        envs=4,
        batch_labels=64,
        minibatch=4,
        microbatch=2,
    )
    return replace(config, **changes)


def test_current_engine_matches_training_contract() -> None:
    validate_engine_contract()


def test_actor_checkpoint_round_trip_excludes_critic(tmp_path) -> None:
    torch.manual_seed(3)
    model = tiny_model()
    with torch.no_grad():
        for parameter in model.critic.parameters():
            parameter.fill_(7.0)
    path = tmp_path / "actor.pt"
    save_actor_checkpoint(path, model, metadata={"source": "test"})

    restored = load_actor_checkpoint(path, torch.device("cpu"))

    for name, expected in model.state_dict().items():
        if is_actor_parameter(name):
            torch.testing.assert_close(restored.state_dict()[name], expected)
    assert not torch.equal(restored.critic[1].weight, model.critic[1].weight)


def test_actor_checkpoint_rejects_an_input_schema_change(tmp_path) -> None:
    path = tmp_path / "actor.pt"
    save_actor_checkpoint(path, tiny_model())
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert payload["training_input_schema"] == TRAINING_INPUT_SCHEMA
    payload["training_input_schema"] = "old-private-hands"
    torch.save(payload, path)

    with pytest.raises(ValueError, match="input schema"):
        load_actor_checkpoint(path, torch.device("cpu"))


def test_rule_ev_collector_produces_legal_actor_labels() -> None:
    collector = RuleEvCollector(4, seed=11)
    batch = collector.collect(20, exploration=0.25)

    assert len(batch) == 20
    assert batch.events.shape == (20, 192, 8)
    assert np.all(batch.event_lengths <= 192)
    assert np.all(batch.actions < 115)
    assert np.all(batch.legal[np.arange(len(batch)), batch.actions])
    assert np.all(batch.legal.sum(axis=1) > 1)


@pytest.mark.parametrize(
    ("labels", "expected"),
    (
        (64, [32, 32]),
        (68, [32, 36]),
        (69, [32, 32, 5]),
    ),
)
def test_supervised_batch_sizes_preserve_the_final_validation_split(
    labels: int, expected: list[int]
) -> None:
    assert _batch_sizes(labels, batch_labels=32, minibatch=4) == expected


def test_supervised_batch_sizes_reject_a_too_small_final_batch() -> None:
    with pytest.raises(ValueError, match="final batch"):
        _batch_sizes(4, batch_labels=32, minibatch=4)


def test_supervised_collection_prefetches_exactly_one_batch() -> None:
    second_started = threading.Event()
    release_second = threading.Event()
    third_started = threading.Event()

    class FakeCollector:
        def __init__(self) -> None:
            self.calls: list[int] = []
            self.active = 0
            self.max_active = 0
            self.lock = threading.Lock()

        def collect(self, labels: int, exploration: float) -> int:
            assert exploration == 0.25
            with self.lock:
                self.calls.append(labels)
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            if labels == 11:
                second_started.set()
                assert release_second.wait(timeout=2.0)
            elif labels == 12:
                third_started.set()
            with self.lock:
                self.active -= 1
            return labels

    collector = FakeCollector()
    with closing(
        _prefetched_batches(collector, [10, 11, 12], 0.25)  # type: ignore[arg-type]
    ) as batches:
        first = next(batches)
        assert first.batch == 10
        assert second_started.wait(timeout=1.0)
        with collector.lock:
            assert collector.calls == [10, 11]
        release_second.set()
        second = next(batches)
        assert second.batch == 11
        assert third_started.wait(timeout=1.0)
        third = next(batches)
        assert third.batch == 12
        with pytest.raises(StopIteration):
            next(batches)

    assert collector.calls == [10, 11, 12]
    assert collector.max_active == 1


def test_supervised_collection_propagates_worker_failure() -> None:
    class FailingCollector:
        def __init__(self) -> None:
            self.calls: list[int] = []

        def collect(self, labels: int, exploration: float) -> int:
            self.calls.append(labels)
            if labels == 11:
                raise RuntimeError("collection failed")
            return labels

    collector = FailingCollector()
    with closing(
        _prefetched_batches(collector, [10, 11], 0.0)  # type: ignore[arg-type]
    ) as batches:
        assert next(batches).batch == 10
        with pytest.raises(RuntimeError, match="collection failed"):
            next(batches)

    assert collector.calls == [10, 11]


def test_prefetched_supervised_batches_do_not_share_storage() -> None:
    collector = RuleEvCollector(4, seed=23)
    with closing(_prefetched_batches(collector, [8, 8], 0.0)) as batches:
        first = next(batches).batch
        second = next(batches).batch

    for first_array, second_array in zip(
        vars(first).values(), vars(second).values(), strict=True
    ):
        assert not np.shares_memory(first_array, second_array)


def test_supervised_update_changes_only_policy_parameters() -> None:
    torch.manual_seed(13)
    config = tiny_config()
    collector = RuleEvCollector(config.envs, seed=17)
    batch = collector.collect(config.batch_labels, config.exploration)
    model = tiny_model()
    optimizer = torch.optim.AdamW(
        list(actor_parameters(model)), lr=config.learning_rate
    )
    actor_before = model.actor[-1].weight.detach().clone()
    non_actor_before = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
        if not is_actor_parameter(name)
    }

    metrics = supervised_update(
        model,
        optimizer,
        batch,
        config,
        torch.device("cpu"),
        np.random.default_rng(19),
    )

    assert not torch.equal(actor_before, model.actor[-1].weight)
    for name, expected in non_actor_before.items():
        torch.testing.assert_close(model.state_dict()[name], expected)
    assert metrics["optimizer_steps"] > 0.0
    assert all(np.isfinite(value) for value in metrics.values())
