"""
memory.py — Experience Memory & Policy Adaptation
===================================================
Stores agent experiences and implements lightweight
reward-weighted policy adaptation (no heavy RL libraries).
"""

from __future__ import annotations
import json
import os
from collections import defaultdict
from typing import Optional
from models import Experience
import config


def _bucket(deal_value: float, risk_score: float, urgency: float, acq_cost: float) -> str:
    """Map continuous lead features to a discrete context bucket."""
    high_value = deal_value > 50_000
    high_risk  = risk_score > 0.55
    urgent     = urgency > 0.65
    expensive  = acq_cost > deal_value * 0.35

    if urgent:
        return "urgent_lead"
    if expensive:
        return "expensive_lead"
    if high_value and not high_risk:
        return "high_value_low_risk"
    if high_value and high_risk:
        return "high_value_high_risk"
    if not high_value and not high_risk:
        return "low_value_low_risk"
    return "low_value_high_risk"


class PolicyWeights:
    """
    Per-agent action preference weights, indexed by context bucket.
    Updated after each episode using a reward-weighted gradient.
    """

    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        # weights[bucket][action] = preference score (init uniform)
        self.weights: dict[str, dict[str, float]] = {
            bucket: {action: 1.0 for action in config.ACTIONS}
            for bucket in config.CONTEXT_BUCKETS
        }

    def get(self, bucket: str) -> dict[str, float]:
        return self.weights.get(bucket, {a: 1.0 for a in config.ACTIONS})

    def best_action(self, bucket: str) -> str:
        w = self.get(bucket)
        return max(w, key=w.get)

    def update(self, bucket: str, action: str, reward: float, next_bucket: Optional[str] = None):
        """
        True Q-Learning via the Bellman Equation:
        Q(s,a) = Q(s,a) + alpha * (R + gamma * max_Q(s') - Q(s,a))
        
        This enables agents to look ahead and predict future state value,
        rather than just reacting to immediate rewards.
        """
        if bucket not in self.weights:
            self.weights[bucket] = {a: 1.0 for a in config.ACTIONS}

        alpha = config.LEARNING_RATE
        gamma = 0.95  # Discount factor for future rewards
        
        # Calculate max Q value for the next state (if episode continues)
        max_q_next = 0.0
        if next_bucket and next_bucket in self.weights:
            max_q_next = max(self.weights[next_bucket].values())
            
        current_q = self.weights[bucket][action]
        
        # The Bellman Update
        new_q = current_q + alpha * (reward + gamma * max_q_next - current_q)
        
        # Clip weights to prevent numeric explosion
        self.weights[bucket][action] = max(0.1, min(10.0, new_q))

    def snapshot(self) -> dict[str, dict[str, float]]:
        return {b: dict(self.weights[b]) for b in self.weights}


class ExperienceMemory:
    """
    Stores all agent experiences and exposes retrieval by context bucket.
    Also manages per-agent PolicyWeights and persists state to disk.
    """

    def __init__(self):
        self.experiences: list[Experience]      = []
        self.policy_weights: dict[str, PolicyWeights] = {}
        self._load()

    # ── Policy Weights ─────────────────────────────────────────────────────

    def get_policy(self, agent_name: str) -> PolicyWeights:
        if agent_name not in self.policy_weights:
            self.policy_weights[agent_name] = PolicyWeights(agent_name)
        return self.policy_weights[agent_name]

    def update_policies(self, agent_name: str, bucket: str, action: str, reward: float, next_bucket: Optional[str] = None):
        self.get_policy(agent_name).update(bucket, action, reward, next_bucket)

    def all_policy_snapshots(self) -> dict[str, dict]:
        return {
            name: pw.snapshot()
            for name, pw in self.policy_weights.items()
        }

    # ── Experience Storage ─────────────────────────────────────────────────

    def store(self, experience: Experience):
        self.experiences.append(experience)
        if len(self.experiences) > config.MAX_MEMORY_SIZE:
            self.experiences = self.experiences[-config.MAX_MEMORY_SIZE:]

    def retrieve_similar(self, bucket: str, n: int = config.MEMORY_WINDOW) -> list[dict]:
        """Return the N most recent experiences matching this context bucket."""
        matching = [
            e for e in reversed(self.experiences)
            if e.context_bucket == bucket
        ][:n]
        return [
            {
                "context_bucket": e.context_bucket,
                "final_action":   e.final_action,
                "reward":         e.reward,
                "outcome":        e.outcome,
                "explanation":    e.explanation,
                "episode":        e.episode,
            }
            for e in matching
        ]

    def recent(self, n: int = 20) -> list[dict]:
        """Return the N most recent experiences as plain dicts."""
        return [e.model_dump() for e in self.experiences[-n:]]

    def bucket_for_lead(
        self,
        deal_value: float,
        risk_score: float,
        urgency:    float,
        acq_cost:   float,
    ) -> str:
        return _bucket(deal_value, risk_score, urgency, acq_cost)

    def summary_stats(self) -> dict:
        if not self.experiences:
            return {}
        rewards   = [e.reward for e in self.experiences]
        actions   = [e.final_action for e in self.experiences]
        outcomes  = [e.outcome for e in self.experiences]
        from collections import Counter
        return {
            "total_experiences": len(self.experiences),
            "avg_reward":        round(sum(rewards) / len(rewards), 4),
            "max_reward":        round(max(rewards), 4),
            "min_reward":        round(min(rewards), 4),
            "action_dist":       dict(Counter(actions)),
            "outcome_dist":      dict(Counter(outcomes)),
        }

    # ── Persistence ────────────────────────────────────────────────────────

    def save(self):
        os.makedirs(config.OUTPUTS_DIR, exist_ok=True)
        payload = {
            "experiences": [e.model_dump() for e in self.experiences],
            "policy_weights": self.all_policy_snapshots(),
        }
        with open(config.MEMORY_PATH, "w") as f:
            json.dump(payload, f, indent=2)

    def _load(self):
        if not os.path.exists(config.MEMORY_PATH):
            return
        try:
            with open(config.MEMORY_PATH) as f:
                data = json.load(f)
            for e_dict in data.get("experiences", []):
                try:
                    self.experiences.append(Experience(**e_dict))
                except Exception:
                    pass
            for agent_name, buckets in data.get("policy_weights", {}).items():
                pw = PolicyWeights(agent_name)
                for bucket, actions in buckets.items():
                    pw.weights[bucket] = actions
                self.policy_weights[agent_name] = pw
        except Exception:
            pass   # fresh start if corrupt
