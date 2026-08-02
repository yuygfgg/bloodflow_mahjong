from __future__ import annotations

from typing import Callable

import bloodflow_mahjong as bm

DIFFICULTIES = ("简单", "标准", "困难")

_default_ev_config = bm.RuleEvConfig.standard()
_default_planner_config = bm.RulePlannerConfig()


def make_policy(difficulty: str) -> Callable[[bm.Game], int | None]:
    """返回一个 (game) -> action_id 的策略函数,动作保证在 legal mask 内。"""
    if difficulty == "简单":
        return lambda game: game.simple_rule_action()
    if difficulty == "标准":
        return lambda game: game.rule_ev_action(_default_ev_config)
    if difficulty == "困难":
        return lambda game: game.rule_planner_action(_default_planner_config)
    raise ValueError(f"未知难度: {difficulty!r}")
