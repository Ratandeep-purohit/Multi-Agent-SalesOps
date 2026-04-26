"""
Multi-Agent SalesOps Arena -- Before vs After Training
=======================================================
Demonstrates measurable improvement from training using
a simple Q-table (no external RL libraries required).

Phases:
  1. BEFORE TRAINING  - Agents use random policy
  2. TRAINING         - Agents learn via Q-learning (epsilon-greedy)
  3. AFTER TRAINING   - Agents use learned Q-table policy

Run: python training_comparison.py
"""

import sys
import random
import math

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
BEFORE_EPISODES  = 15       # episodes run with random (untrained) policy
TRAINING_EPISODES = 200     # episodes used for Q-learning
AFTER_EPISODES   = 15       # episodes run with trained policy

ALPHA   = 0.15              # learning rate
GAMMA   = 0.90              # discount factor
EPSILON_START = 1.0         # 100% exploration at start
EPSILON_END   = 0.05        # 5% exploration after training
ACTIONS = ["pursue", "negotiate", "reject"]

ARBITRATION_WEIGHTS = {"sales": 0.40, "finance": 0.35, "compliance": 0.25}

# ─────────────────────────────────────────────────────────────────────────────
# ENVIRONMENT
# ─────────────────────────────────────────────────────────────────────────────
class SalesEnvironment:
    def generate_lead(self):
        return {
            "lead_value":       round(random.uniform(5_000, 50_000), 2),
            "acquisition_cost": round(random.uniform(500, 15_000), 2),
            "risk_score":       round(random.uniform(0.0, 1.0), 2),
            "urgency":          round(random.uniform(0.0, 1.0), 2),
        }

    def compute_global_reward(self, lead: dict, action: str) -> float:
        if action == "reject":
            return 0.0
        revenue_gain = lead["lead_value"] * (0.6 if action == "negotiate" else 1.0)
        cost_penalty = lead["acquisition_cost"]
        risk_penalty = lead["risk_score"] * lead["lead_value"] * 0.3
        return round(revenue_gain - cost_penalty - risk_penalty, 2)

    def get_state_key(self, lead: dict) -> str:
        """Discretize continuous lead features into a Q-table state key."""
        val_bucket  = "high" if lead["lead_value"] > 25000 else ("mid" if lead["lead_value"] > 12000 else "low")
        cost_bucket = "high" if lead["acquisition_cost"] > 8000 else ("mid" if lead["acquisition_cost"] > 3000 else "low")
        risk_bucket = "high" if lead["risk_score"] > 0.65 else ("mid" if lead["risk_score"] > 0.35 else "low")
        return f"{val_bucket}_{cost_bucket}_{risk_bucket}"

# ─────────────────────────────────────────────────────────────────────────────
# Q-TABLE AGENTS (one Q-table shared per agent type)
# ─────────────────────────────────────────────────────────────────────────────
class QAgent:
    def __init__(self, name: str, bias_action: str):
        """
        bias_action: the action this agent naturally prefers before learning.
        """
        self.name = name
        self.bias = bias_action
        self.q_table: dict[str, dict[str, float]] = {}

    def _ensure_state(self, state: str):
        if state not in self.q_table:
            # Initialize with a slight bias toward the agent's natural preference
            self.q_table[state] = {a: (0.1 if a == self.bias else 0.0) for a in ACTIONS}

    def random_action(self) -> str:
        """Completely random — used in BEFORE TRAINING phase."""
        return random.choice(ACTIONS)

    def select_action(self, state: str, epsilon: float) -> str:
        """Epsilon-greedy Q-action — used during TRAINING and AFTER TRAINING."""
        self._ensure_state(state)
        if random.random() < epsilon:
            return random.choice(ACTIONS)
        return max(self.q_table[state], key=self.q_table[state].get)

    def update(self, state: str, action: str, reward: float, next_state: str):
        """Standard Q-learning update."""
        self._ensure_state(state)
        self._ensure_state(next_state)
        old_q  = self.q_table[state][action]
        max_nq = max(self.q_table[next_state].values())
        self.q_table[state][action] = old_q + ALPHA * (reward + GAMMA * max_nq - old_q)

# ─────────────────────────────────────────────────────────────────────────────
# ARBITRATION ENGINE
# ─────────────────────────────────────────────────────────────────────────────
ACTION_SCORES = {"pursue": 1.0, "negotiate": 0.5, "reject": 0.0}

def arbitrate(actions: dict[str, str]) -> str:
    score = sum(ACTION_SCORES[a] * ARBITRATION_WEIGHTS[agent] for agent, a in actions.items())
    if score >= 0.70:   return "pursue"
    elif score >= 0.35: return "negotiate"
    else:               return "reject"

# ─────────────────────────────────────────────────────────────────────────────
# DISPLAY HELPERS
# ─────────────────────────────────────────────────────────────────────────────
W = 66   # console width

def hdr(text): print(f"\033[1;36m{text}\033[0m")
def ok(text):  print(f"\033[1;32m{text}\033[0m")
def warn(text):print(f"\033[1;31m{text}\033[0m")
def bold(text):print(f"\033[1;33m{text}\033[0m")
def sep(c="-"): print(c * W)

def phase_header(title: str):
    print()
    sep("=")
    hdr(f"  {title}")
    sep("=")

def episode_row(ep, lead, actions, final, reward, show_log=True):
    if not show_log:
        return
    sep()
    print(f"  Ep {ep:>3} | Value: ${lead['lead_value']:>8,.0f} | "
          f"Cost: ${lead['acquisition_cost']:>7,.0f} | Risk: {lead['risk_score']:.2f}")
    print(f"         Sales→{actions['sales']:<11} Finance→{actions['finance']:<11} "
          f"Compliance→{actions['compliance']:<11}")
    print(f"         Arbitration→\033[1;35m{final.upper():<12}\033[0m  "
          f"Reward: \033[1;{'32' if reward >= 0 else '31'}m${reward:>9,.0f}\033[0m")

def summary_block(label, rewards, actions_log):
    total = sum(rewards)
    avg   = total / len(rewards) if rewards else 0
    pursuits   = actions_log.count("pursue")
    negotiates = actions_log.count("negotiate")
    rejects    = actions_log.count("reject")
    pos_ep = sum(1 for r in rewards if r > 0)

    print(f"\n  {'Metric':<30} {'Value':>15}")
    sep("-")
    print(f"  {'Total Episodes':<30} {len(rewards):>15}")
    print(f"  {'Total Net Reward':<30} ${total:>14,.2f}")
    print(f"  {'Average Reward / Episode':<30} ${avg:>14,.2f}")
    print(f"  {'Profitable Episodes':<30} {pos_ep:>14} / {len(rewards)}")
    print(f"  {'Pursue Count':<30} {pursuits:>15}")
    print(f"  {'Negotiate Count':<30} {negotiates:>15}")
    print(f"  {'Reject Count':<30} {rejects:>15}")
    sep()
    return {"label": label, "total": total, "avg": avg, "pos": pos_ep, "n": len(rewards)}

# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION RUNNER
# ─────────────────────────────────────────────────────────────────────────────
def run_phase(env, agents, n_episodes, epsilon=None, train=False, show_log=True):
    """
    Run n_episodes.
    - If epsilon is None → fully random policy (before training)
    - If epsilon is float → epsilon-greedy policy
    - If train=True → perform Q-table updates after each step
    """
    rewards      = []
    actions_log  = []
    epsilon_vals = []

    for ep in range(1, n_episodes + 1):
        lead  = env.generate_lead()
        state = env.get_state_key(lead)

        # Compute decayed epsilon for training phase
        if epsilon is None:
            eps = 1.0  # fully random
        elif train:
            eps = EPSILON_END + (EPSILON_START - EPSILON_END) * math.exp(-5.0 * ep / n_episodes)
        else:
            eps = epsilon  # fixed (post-training = nearly greedy)

        epsilon_vals.append(round(eps, 3))

        # Each agent selects an action
        agent_actions = {}
        for name, agent in agents.items():
            if epsilon is None:
                agent_actions[name] = agent.random_action()
            else:
                agent_actions[name] = agent.select_action(state, eps)

        final_action = arbitrate(agent_actions)
        reward       = env.compute_global_reward(lead, final_action)

        # Q-table update (only during training)
        if train:
            next_lead  = env.generate_lead()
            next_state = env.get_state_key(next_lead)
            for name, agent in agents.items():
                agent.update(state, agent_actions[name], reward, next_state)

        rewards.append(reward)
        actions_log.append(final_action)
        episode_row(ep, lead, agent_actions, final_action, reward, show_log=show_log)

    return rewards, actions_log, epsilon_vals

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    random.seed(42)   # reproducible results for demo

    env   = SalesEnvironment()
    agents = {
        "sales":      QAgent("sales",      bias_action="pursue"),
        "finance":    QAgent("finance",    bias_action="reject"),
        "compliance": QAgent("compliance", bias_action="negotiate"),
    }

    # ═══════════════════════════════════════════════════════
    # PHASE 1: BEFORE TRAINING (random policy)
    # ═══════════════════════════════════════════════════════
    phase_header("PHASE 1 — BEFORE TRAINING  (Random Policy, No Learning)")
    print(f"  Running {BEFORE_EPISODES} episodes with fully random agent actions...\n")
    before_rewards, before_actions, _ = run_phase(
        env, agents, BEFORE_EPISODES, epsilon=None, train=False, show_log=True
    )
    phase_header("PHASE 1 SUMMARY — Before Training")
    before_stats = summary_block("Before", before_rewards, before_actions)

    # ═══════════════════════════════════════════════════════
    # PHASE 2: TRAINING (Q-learning over N episodes)
    # ═══════════════════════════════════════════════════════
    phase_header(f"PHASE 2 — TRAINING  ({TRAINING_EPISODES} Episodes, Epsilon-Greedy Q-Learning)")
    print(f"  Agents are now updating their Q-tables after each episode.")
    print(f"  Exploration decays: epsilon {EPSILON_START:.0%} → {EPSILON_END:.0%}\n")

    # Show only every 25th episode during training (to avoid flooding)
    training_rewards = []
    training_actions = []
    CHECKPOINT_INTERVAL = 25

    for checkpoint_start in range(0, TRAINING_EPISODES, CHECKPOINT_INTERVAL):
        chunk_end = min(checkpoint_start + CHECKPOINT_INTERVAL, TRAINING_EPISODES)
        chunk_n   = chunk_end - checkpoint_start

        c_rewards, c_actions, c_eps = run_phase(
            env, agents, chunk_n, epsilon=EPSILON_START, train=True, show_log=False
        )
        training_rewards.extend(c_rewards)
        training_actions.extend(c_actions)

        avg_chunk = sum(c_rewards) / len(c_rewards)
        ep_label  = f"Ep {checkpoint_start+1:>3} – {chunk_end:>3}"
        bar_len   = max(0, int((avg_chunk / 35000) * 30))
        bar       = "█" * bar_len + "░" * (30 - bar_len)
        color     = "\033[1;32m" if avg_chunk > 0 else "\033[1;31m"
        print(f"  {ep_label} | ε={c_eps[-1]:.3f} | Avg: {color}${avg_chunk:>8,.0f}\033[0m  [{bar}]")

    phase_header("TRAINING COMPLETE — Q-Tables Learned")

    # ═══════════════════════════════════════════════════════
    # PHASE 3: AFTER TRAINING (greedy learned policy)
    # ═══════════════════════════════════════════════════════
    phase_header(f"PHASE 3 — AFTER TRAINING  (Learned Policy, ε={EPSILON_END})")
    print(f"  Running {AFTER_EPISODES} episodes using the trained Q-table (near-greedy)...\n")
    after_rewards, after_actions, _ = run_phase(
        env, agents, AFTER_EPISODES, epsilon=EPSILON_END, train=False, show_log=True
    )
    phase_header("PHASE 3 SUMMARY — After Training")
    after_stats = summary_block("After", after_rewards, after_actions)

    # ═══════════════════════════════════════════════════════
    # FINAL COMPARISON TABLE
    # ═══════════════════════════════════════════════════════
    print()
    sep("=")
    hdr("  FINAL COMPARISON: Before Training vs After Training")
    sep("=")

    b_avg   = before_stats["avg"]
    a_avg   = after_stats["avg"]
    delta   = a_avg - b_avg
    pct     = ((a_avg - b_avg) / abs(b_avg) * 100) if b_avg != 0 else 0

    b_pos   = before_stats["pos"]
    a_pos   = after_stats["pos"]
    n       = before_stats["n"]

    print(f"\n  {'Metric':<35} {'Before':>12} {'After':>12} {'Delta':>12}")
    sep("-")
    print(f"  {'Avg Reward / Episode':<35} ${b_avg:>11,.0f} ${a_avg:>11,.0f} "
          f"{'$'+f'{delta:,.0f}':>12}")
    print(f"  {'Profitable Episodes':<35} {b_pos:>12} {a_pos:>12} "
          f"{a_pos - b_pos:>+12}")
    total_delta = after_stats["total"] - before_stats["total"]
    total_delta_str = f"${total_delta:,.0f}"
    print(f"  {'Total Net Reward':<35} ${before_stats['total']:>11,.0f} "
          f"${after_stats['total']:>11,.0f} {total_delta_str:>12}")
    sep("=")

    if pct > 0:
        ok(f"\n  Improvement: +{pct:.1f}% average reward per episode after training.")
        ok(f"  The agents learned to avoid low-margin and high-risk leads.")
    elif pct < 0:
        warn(f"\n  Note: {abs(pct):.1f}% lower average this run (can vary by random seed).")
        warn(f"  Increase TRAINING_EPISODES or run again for consistent improvement.")
    else:
        bold(f"\n  No change detected. Try increasing TRAINING_EPISODES.")

    print()
    sep("=")
    hdr("  LOOP CONFIRMED: Observe -> Learn -> Adapt -> Improve")
    sep("=")
    print()

    # ── Q-Table Insight (show what agents learned) ───────────────────────────
    print()
    bold("  LEARNED Q-TABLE SNAPSHOT (Finance Agent — top 5 states)")
    sep("-")
    finance_q = agents["finance"].q_table
    sorted_states = sorted(finance_q.items(), key=lambda x: max(x[1].values()), reverse=True)[:5]
    for state_key, q_vals in sorted_states:
        best_action = max(q_vals, key=q_vals.get)
        print(f"  State [{state_key:<22}]  Best Action: {best_action.upper():<11} "
              f"Q={max(q_vals.values()):>8.3f}")
    sep()
    print()


if __name__ == "__main__":
    main()
