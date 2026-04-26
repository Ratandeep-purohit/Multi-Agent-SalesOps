"""
Multi-Agent SalesOps Arena -- Proof of Concept Demo
====================================================
A minimal, dependency-free simulation demonstrating:
  - Stochastic sales environment
  - Sales Agent (maximize revenue)
  - Finance Agent (minimize cost)
  - Compliance Agent (reduce risk)
  - Weighted Arbitration Engine
  - Global Reward Function
  - Explainable Decision Logs

Run: python demo.py
"""

import random
import sys
import time

# Force UTF-8 output on Windows terminals
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ─────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────
EPISODES = 8
ARBITRATION_WEIGHTS = {
    "sales":      0.40,   # Revenue drive
    "finance":    0.35,   # Cost discipline
    "compliance": 0.25,   # Risk gate
}

# ─────────────────────────────────────────────────────────
# ENVIRONMENT
# ─────────────────────────────────────────────────────────
class SalesEnvironment:
    """Simulates a single lead arriving each episode."""

    ACTIONS = ["pursue", "negotiate", "reject"]

    def generate_lead(self) -> dict:
        return {
            "lead_value":    round(random.uniform(5_000, 50_000), 2),   # potential revenue
            "acquisition_cost": round(random.uniform(500, 15_000), 2),  # cost to close
            "risk_score":    round(random.uniform(0.0, 1.0), 2),        # 0=safe, 1=high risk
            "urgency":       round(random.uniform(0.0, 1.0), 2),        # 0=low, 1=critical
        }

    def compute_global_reward(self, lead: dict, action: str) -> float:
        """
        Global reward = Revenue contribution − Cost − Risk Penalty
        Reward is action-gated: reject yields zero, pursue yields full.
        """
        if action == "reject":
            return 0.0

        revenue_gain  = lead["lead_value"] * (0.6 if action == "negotiate" else 1.0)
        cost_penalty  = lead["acquisition_cost"]
        risk_penalty  = lead["risk_score"] * lead["lead_value"] * 0.3

        return round(revenue_gain - cost_penalty - risk_penalty, 2)


# ─────────────────────────────────────────────────────────
# AGENTS
# ─────────────────────────────────────────────────────────
class SalesAgent:
    """Maximizes revenue. Prefers to pursue; falls back to negotiate."""

    def recommend(self, lead: dict) -> tuple[str, str]:
        if lead["lead_value"] >= 10_000:
            action = "pursue"
            reason = f"High-value lead (${lead['lead_value']:,.0f}). Worth aggressive pursuit."
        elif lead["lead_value"] >= 5_000:
            action = "negotiate"
            reason = f"Mid-value lead. Negotiate to protect conversion rate."
        else:
            action = "negotiate"
            reason = f"Low value but salvageable via negotiation."
        return action, reason


class FinanceAgent:
    """Minimizes cost. Rejects or negotiates when margins are thin."""

    MARGIN_THRESHOLD = 0.35  # minimum acceptable margin

    def recommend(self, lead: dict) -> tuple[str, str]:
        margin = (lead["lead_value"] - lead["acquisition_cost"]) / max(lead["lead_value"], 1)
        if margin < self.MARGIN_THRESHOLD:
            action = "reject"
            reason = f"Margin too thin ({margin:.1%} < {self.MARGIN_THRESHOLD:.0%}). CAC too high."
        elif margin < 0.55:
            action = "negotiate"
            reason = f"Margin acceptable ({margin:.1%}) but negotiate to protect unit economics."
        else:
            action = "pursue"
            reason = f"Strong margin ({margin:.1%}). Approved for full pursuit."
        return action, reason


class ComplianceAgent:
    """Reduces risk. Flags or blocks high-risk leads."""

    RISK_THRESHOLD = 0.65

    def recommend(self, lead: dict) -> tuple[str, str]:
        if lead["risk_score"] >= self.RISK_THRESHOLD:
            action = "reject"
            reason = f"Risk score {lead['risk_score']:.2f} exceeds threshold ({self.RISK_THRESHOLD}). Flagged."
        elif lead["risk_score"] >= 0.40:
            action = "negotiate"
            reason = f"Moderate risk ({lead['risk_score']:.2f}). Proceed with contractual safeguards."
        else:
            action = "pursue"
            reason = f"Low risk ({lead['risk_score']:.2f}). No compliance concerns."
        return action, reason


# ─────────────────────────────────────────────────────────
# ARBITRATION ENGINE
# ─────────────────────────────────────────────────────────
ACTION_SCORES = {"pursue": 1.0, "negotiate": 0.5, "reject": 0.0}
SCORE_TO_ACTION = {1.0: "pursue", 0.5: "negotiate", 0.0: "reject"}

def arbitrate(recommendations: dict[str, str], weights: dict[str, float]) -> tuple[str, str]:
    """
    Weighted voting arbitration.
    Each agent's action is mapped to a score, weighted, and summed.
    Final action = closest bracket to weighted sum.
    """
    weighted_score = sum(
        ACTION_SCORES[rec] * weights[agent]
        for agent, rec in recommendations.items()
    )

    # Determine final action by bracket
    if weighted_score >= 0.70:
        final_action = "pursue"
    elif weighted_score >= 0.35:
        final_action = "negotiate"
    else:
        final_action = "reject"

    reason = (
        f"Weighted arbitration score: {weighted_score:.3f}. "
        f"Weights → Sales:{weights['sales']}, Finance:{weights['finance']}, "
        f"Compliance:{weights['compliance']}. "
        f"Decision threshold mapped to '{final_action}'."
    )
    return final_action, reason


# ─────────────────────────────────────────────────────────
# DISPLAY HELPERS
# ─────────────────────────────────────────────────────────
COLORS = {
    "header":  "\033[1;36m",   # Cyan bold
    "agent":   "\033[1;33m",   # Yellow bold
    "arb":     "\033[1;35m",   # Magenta bold
    "reward":  "\033[1;32m",   # Green bold
    "neg":     "\033[1;31m",   # Red bold
    "reset":   "\033[0m",
}

def c(color_key: str, text: str) -> str:
    return f"{COLORS[color_key]}{text}{COLORS['reset']}"

def print_separator(char="-", width=64):
    print(char * width)


# ─────────────────────────────────────────────────────────
# MAIN SIMULATION LOOP
# ─────────────────────────────────────────────────────────
def run_simulation():
    env        = SalesEnvironment()
    sales_a    = SalesAgent()
    finance_a  = FinanceAgent()
    compliance = ComplianceAgent()

    episode_log = []
    total_reward = 0.0

    print()
    print_separator("=")
    print(c("header", "  [ARENA]  Multi-Agent SalesOps Arena -- Proof of Concept"))
    print_separator("=")
    time.sleep(0.3)

    for ep in range(1, EPISODES + 1):

        lead = env.generate_lead()

        # ── Agent Recommendations ──────────────────────────────
        s_action, s_reason  = sales_a.recommend(lead)
        f_action, f_reason  = finance_a.recommend(lead)
        cp_action, cp_reason = compliance.recommend(lead)

        recommendations = {
            "sales":      s_action,
            "finance":    f_action,
            "compliance": cp_action,
        }

        # ── Arbitration ────────────────────────────────────────
        final_action, arb_reason = arbitrate(recommendations, ARBITRATION_WEIGHTS)

        # ── Global Reward ──────────────────────────────────────
        reward = env.compute_global_reward(lead, final_action)
        total_reward += reward

        episode_log.append({
            "episode": ep,
            "action":  final_action,
            "reward":  reward,
        })

        # ── Decision Log Output ────────────────────────────────
        print_separator("=")
        print(c("header", f"  Episode {ep}/{EPISODES}"))
        print_separator()
        print(f"  Lead Value     : ${lead['lead_value']:>10,.2f}")
        print(f"  Acq. Cost      : ${lead['acquisition_cost']:>10,.2f}")
        print(f"  Risk Score     :  {lead['risk_score']:.2f}  |  Urgency: {lead['urgency']:.2f}")
        print_separator()

        print(c("agent", f"  [Sales Agent]")      + f"       → {s_action.upper():<10} | {s_reason}")
        print(c("agent", f"  [Finance Agent]")    + f"     → {f_action.upper():<10} | {f_reason}")
        print(c("agent", f"  [Compliance Agent]") + f"  → {cp_action.upper():<10} | {cp_reason}")

        print_separator()
        print(c("arb", f"  [Arbitration]") + f"       → {final_action.upper():<10} | {arb_reason}")
        print_separator()

        reward_color = "reward" if reward >= 0 else "neg"
        print(c(reward_color, f"  [Global Reward]     → ${reward:>10,.2f}"))
        print_separator()
        time.sleep(0.2)

    # ── Final Summary ──────────────────────────────────────────
    print()
    print_separator("=")
    print(c("header", "  [SUMMARY] Simulation Results"))
    print_separator("=")
    print(f"  Total Episodes    : {EPISODES}")
    print(f"  Total Net Reward  : ${total_reward:>10,.2f}")
    print(f"  Average / Episode : ${total_reward / EPISODES:>10,.2f}")
    print_separator("-")

    pursuits   = sum(1 for e in episode_log if e["action"] == "pursue")
    negotiates = sum(1 for e in episode_log if e["action"] == "negotiate")
    rejects    = sum(1 for e in episode_log if e["action"] == "reject")

    print(f"  Pursue    : {pursuits} episodes")
    print(f"  Negotiate : {negotiates} episodes")
    print(f"  Reject    : {rejects} episodes")
    print_separator("=")
    print()
    print(c("header", "  [OK] Loop confirmed: Observe -> Recommend -> Arbitrate -> Reward"))
    print()


if __name__ == "__main__":
    run_simulation()
