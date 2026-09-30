from __future__ import annotations

import json
import math
import random
import sqlite3
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Protocol

import numpy as np


ACTION_NAMES = {
    0: "A0_basic_top5",
    1: "A1_wide_rerank",
    2: "A2_rewrite_rerank",
}


@dataclass(frozen=True)
class RLState:
    question_length: float
    top1_score: float
    score_gap: float
    score_std: float

    def vector(self) -> np.ndarray:
        # Keep features small and interpretable. Bias is encoded via question_length floor.
        return np.asarray([
            max(0.0, min(self.question_length / 200.0, 2.0)),
            max(-1.0, min(self.top1_score, 1.0)),
            max(-1.0, min(self.score_gap, 1.0)),
            max(0.0, min(self.score_std, 1.0)),
        ], dtype=float)


@dataclass(frozen=True)
class OfflineOutcome:
    relevant_rank: int | None
    should_fallback: bool
    did_fallback: bool
    grounded: bool
    latency_ms: float


@dataclass(frozen=True)
class OnlineFeedback:
    thumb: int = 0  # -1 / 0 / +1
    source_opened: bool = False
    reasked: bool = False


def offline_reward(outcome: OfflineOutcome) -> float:
    """Reward only from labels known in an offline benchmark.

    Deliberately separate from user behavior reward to avoid evaluation leakage.
    """
    reward = 0.0
    if outcome.relevant_rank is not None:
        reward += 1.0 / max(1, outcome.relevant_rank)
    elif not outcome.should_fallback:
        reward -= 1.0

    if outcome.should_fallback == outcome.did_fallback:
        reward += 0.5
    else:
        reward -= 0.5

    if outcome.grounded:
        reward += 0.25
    else:
        reward -= 1.0

    # Small latency penalty, capped so quality dominates.
    reward -= min(max(outcome.latency_ms, 0.0) / 5000.0, 0.3)
    return float(reward)


def online_reward(feedback: OnlineFeedback) -> float:
    """Reward from actual user behavior only."""
    reward = float(max(-1, min(feedback.thumb, 1)))
    if feedback.source_opened:
        reward += 0.2
    if feedback.reasked:
        reward -= 0.2
    return reward


class Policy(Protocol):
    name: str
    def choose(self, state: RLState) -> int: ...
    def update(self, action: int, state: RLState, reward: float) -> None: ...


class RandomPolicy:
    name = "random"
    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)
    def choose(self, state: RLState) -> int:
        return self.rng.randrange(3)
    def update(self, action: int, state: RLState, reward: float) -> None:
        return None


class RulePolicy:
    name = "rule"
    def choose(self, state: RLState) -> int:
        # Low score or uncertainty -> stronger retrieval strategy.
        if state.top1_score < 0.35:
            return 2
        if state.score_gap < 0.08 or state.score_std > 0.20:
            return 1
        return 0
    def update(self, action: int, state: RLState, reward: float) -> None:
        return None


class LinUCBPolicyV2:
    name = "linucb"
    def __init__(self, n_features: int = 4, n_actions: int = 3, alpha: float = 0.8):
        self.n_features = n_features
        self.n_actions = n_actions
        self.alpha = float(alpha)
        self.A = [np.eye(n_features, dtype=float) for _ in range(n_actions)]
        self.b = [np.zeros(n_features, dtype=float) for _ in range(n_actions)]

    def choose(self, state: RLState) -> int:
        x = state.vector()
        scores: list[float] = []
        for action in range(self.n_actions):
            inv = np.linalg.inv(self.A[action])
            theta = inv @ self.b[action]
            uncertainty = math.sqrt(max(0.0, float(x @ inv @ x)))
            scores.append(float(theta @ x + self.alpha * uncertainty))
        return int(np.argmax(scores))

    def update(self, action: int, state: RLState, reward: float) -> None:
        x = state.vector()
        self.A[action] += np.outer(x, x)
        self.b[action] += float(reward) * x

    def to_dict(self) -> dict:
        return {
            "schema_version": 1,
            "name": self.name,
            "n_features": self.n_features,
            "n_actions": self.n_actions,
            "alpha": self.alpha,
            "A": [x.tolist() for x in self.A],
            "b": [x.tolist() for x in self.b],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LinUCBPolicyV2":
        obj = cls(
            n_features=int(data["n_features"]),
            n_actions=int(data["n_actions"]),
            alpha=float(data["alpha"]),
        )
        obj.A = [np.asarray(x, dtype=float) for x in data["A"]]
        obj.b = [np.asarray(x, dtype=float) for x in data["b"]]
        return obj

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> "LinUCBPolicyV2":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


class RLLogDB:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(path) as c:
            c.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS policy_decisions(
                    decision_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    seed INTEGER NOT NULL,
                    step INTEGER NOT NULL,
                    policy_name TEXT NOT NULL,
                    action_id INTEGER NOT NULL,
                    state_json TEXT NOT NULL,
                    offline_reward REAL,
                    online_reward REAL,
                    created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS feedback_log(
                    feedback_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    thumb INTEGER NOT NULL,
                    source_opened INTEGER NOT NULL,
                    reasked INTEGER NOT NULL,
                    reward REAL NOT NULL,
                    created_at REAL NOT NULL
                );
                """
            )

    def log_decision(self, *, decision_id: str, run_id: str, seed: int, step: int,
                     policy_name: str, action_id: int, state: RLState,
                     offline: float | None = None, online: float | None = None) -> None:
        with sqlite3.connect(self.path) as c:
            c.execute(
                "INSERT OR REPLACE INTO policy_decisions VALUES(?,?,?,?,?,?,?,?,?,?)",
                (
                    decision_id, run_id, seed, step, policy_name, action_id,
                    json.dumps(asdict(state), ensure_ascii=False), offline, online, time.time(),
                ),
            )


@dataclass(frozen=True)
class Scenario:
    state: RLState
    action_rewards: tuple[float, float, float]


class OfflineBanditEnv:
    """Deterministic synthetic environment for implementation verification.

    This validates learning mechanics only; it is not evidence of real user preference.
    """
    def __init__(self, seed: int):
        self.rng = random.Random(seed)

    def sample(self) -> Scenario:
        # Three broad query conditions, each with a known best action.
        kind = self.rng.randrange(3)
        if kind == 0:  # easy/clear -> A0 cheapest and sufficient
            state = RLState(self.rng.uniform(20, 80), self.rng.uniform(.65, .95), self.rng.uniform(.15, .40), self.rng.uniform(.02, .10))
            rewards = (1.10, 0.90, 0.70)
        elif kind == 1:  # ambiguous -> rerank helps
            state = RLState(self.rng.uniform(30, 120), self.rng.uniform(.40, .65), self.rng.uniform(.01, .07), self.rng.uniform(.15, .30))
            rewards = (0.55, 1.15, 0.90)
        else:  # low confidence -> rewrite + rerank helps
            state = RLState(self.rng.uniform(50, 180), self.rng.uniform(.10, .34), self.rng.uniform(.00, .05), self.rng.uniform(.20, .40))
            rewards = (0.15, 0.65, 1.25)
        noise = tuple(self.rng.uniform(-.05, .05) for _ in range(3))
        return Scenario(state, tuple(float(a+b) for a, b in zip(rewards, noise)))

    def reward(self, scenario: Scenario, action: int) -> float:
        return scenario.action_rewards[action]


@dataclass
class RunSummary:
    policy: str
    seed: int
    steps: int
    reward_mean: float
    reward_std: float
    action_counts: dict[str, int]
    completed_steps: int
    resumed: bool


def _checkpoint_path(out_dir: Path, policy_name: str, seed: int) -> Path:
    return out_dir / policy_name / f"seed_{seed}" / "checkpoint.json"


def _policy_path(out_dir: Path, policy_name: str, seed: int) -> Path:
    return out_dir / policy_name / f"seed_{seed}" / "policy.json"


def simulate_policy(policy: Policy, *, seed: int, steps: int, out_dir: Path,
                    checkpoint_every: int = 50, resume: bool = True,
                    logger: RLLogDB | None = None) -> RunSummary:
    run_dir = out_dir / policy.name / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt = _checkpoint_path(out_dir, policy.name, seed)
    result_path = run_dir / "result.json"

    start_step = 0
    rewards: list[float] = []
    counts = {name: 0 for name in ACTION_NAMES.values()}
    resumed = False

    # Deterministic resume: replay the environment and restore state exactly.
    if resume and ckpt.exists():
        data = json.loads(ckpt.read_text(encoding="utf-8"))
        start_step = int(data["completed_steps"])
        rewards = [float(x) for x in data["rewards"]]
        counts = {str(k): int(v) for k, v in data["action_counts"].items()}
        if isinstance(policy, LinUCBPolicyV2) and data.get("policy"):
            restored = LinUCBPolicyV2.from_dict(data["policy"])
            policy.A, policy.b, policy.alpha = restored.A, restored.b, restored.alpha
        resumed = start_step > 0

    env = OfflineBanditEnv(seed)
    # Advance environment state to the checkpoint point.
    for _ in range(start_step):
        env.sample()

    for step in range(start_step, steps):
        scenario = env.sample()
        action = int(policy.choose(scenario.state))
        reward = float(env.reward(scenario, action))
        policy.update(action, scenario.state, reward)
        rewards.append(reward)
        counts[ACTION_NAMES[action]] += 1
        if logger:
            logger.log_decision(
                decision_id=f"{policy.name}-{seed}-{step}", run_id=f"week7-{policy.name}-{seed}",
                seed=seed, step=step, policy_name=policy.name, action_id=action,
                state=scenario.state, offline=reward,
            )
        if (step + 1) % checkpoint_every == 0 or step + 1 == steps:
            payload = {
                "schema_version": 1,
                "policy_name": policy.name,
                "seed": seed,
                "completed_steps": step + 1,
                "rewards": rewards,
                "action_counts": counts,
                "policy": policy.to_dict() if isinstance(policy, LinUCBPolicyV2) else None,
            }
            ckpt.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if isinstance(policy, LinUCBPolicyV2):
        policy.save(_policy_path(out_dir, policy.name, seed))

    summary = RunSummary(
        policy=policy.name,
        seed=seed,
        steps=steps,
        reward_mean=float(statistics.fmean(rewards)) if rewards else 0.0,
        reward_std=float(statistics.pstdev(rewards)) if len(rewards) > 1 else 0.0,
        action_counts=counts,
        completed_steps=len(rewards),
        resumed=resumed,
    )
    result_path.write_text(json.dumps(asdict(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def aggregate_runs(summaries: Iterable[RunSummary]) -> dict:
    items = list(summaries)
    grouped: dict[str, list[RunSummary]] = {}
    for item in items:
        grouped.setdefault(item.policy, []).append(item)
    report = {"schema_version": 1, "policies": {}}
    for policy, rows in sorted(grouped.items()):
        means = [x.reward_mean for x in rows]
        report["policies"][policy] = {
            "seed_count": len(rows),
            "reward_mean": float(statistics.fmean(means)) if means else 0.0,
            "reward_std_across_seeds": float(statistics.pstdev(means)) if len(means) > 1 else 0.0,
            "seeds": [x.seed for x in rows],
            "runs": [asdict(x) for x in rows],
        }
    return report


def choose_deploy_policy(report: dict, *, margin: float = 0.0) -> dict:
    policies = report["policies"]
    rule = float(policies["rule"]["reward_mean"])
    lin = float(policies["linucb"]["reward_mean"])
    selected = "linucb" if lin >= rule + margin else "rule"
    return {
        "selected_policy": selected,
        "rule_reward_mean": rule,
        "linucb_reward_mean": lin,
        "margin": margin,
        "rollback_required": selected == "rule",
        "reason": "linucb_not_better_than_rule" if selected == "rule" else "linucb_meets_promotion_threshold",
    }
