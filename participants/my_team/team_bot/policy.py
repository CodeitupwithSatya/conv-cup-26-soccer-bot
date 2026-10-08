"""Rollout-planner policy: exact forward model + opponent-model search, tactical fallback."""
from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

from .baseline import tactical_action  # noqa: F401  (re-exported: evaluate.py uses it as an opponent)
from .planner import Planner
from .sim import MOVE_NAMES, Cfg, Sim


def _to_dict(action: tuple[int, int, int]) -> dict[str, Any]:
    move, kick_dir, power = action
    result: dict[str, Any] = {"move": MOVE_NAMES[move]}
    if kick_dir:
        result["kick"] = {"direction": MOVE_NAMES[kick_dir], "power": int(power)}
    return result


class Policy:
    def __init__(self, params: dict | None = None, seed: int = 0, time_budget: float = 0.9) -> None:
        self.params = params or {}
        self.model_seed = seed
        self.time_budget = time_budget
        self.random = random.Random(seed)
        self.match_seeded = False
        self._reset_state()

    def _reset_state(self) -> None:
        # Always rebind (never mutate) so shallow copies of this object stay independent.
        self._planner = Planner(self.params)
        self._cfg = None
        self._seed = None
        self._last_it = -1
        self._errors = 0

    @classmethod
    def load(cls, path: Path) -> "Policy":
        try:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:
            raw = {}
        return cls(params=raw.get("planner", {}), seed=int(raw.get("seed", 0)), time_budget=float(raw.get("time_budget", 0.9)))

    def choose_action(self, observation: dict[str, Any]) -> dict[str, Any]:
        try:
            state = observation["state"]
            iteration = int(state["iteration"])
            if self._cfg is None or iteration <= self._last_it or state.get("seed") != self._seed:
                self._planner = Planner(self.params)
                self._cfg = Cfg.from_observation(observation)
                self._seed = state.get("seed")
            self._last_it = iteration
            me = 0 if observation["player_id"] == "player_1" else 1
            sim = Sim.from_observation(observation, self._cfg)
            action = self._planner.plan(sim, me, budget=self.time_budget)
            return _to_dict(action)
        except Exception as error:  # never crash the match
            self._errors += 1
            import sys
            print(f"planner error: {error!r}", file=sys.stderr, flush=True)
            return tactical_action(observation)
