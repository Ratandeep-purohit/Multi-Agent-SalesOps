# ⚙️ Technical Deep Dive

This document outlines the core Machine Learning mathematics and algorithms powering the Multi-Agent SalesOps Arena. It is intended for technical judges evaluating the rigor of the system.

---

## 1. True Q-Learning Integration (The Bellman Equation)

Most LLM-based agents rely entirely on prompt engineering and immediate context window memory. This system is different: it integrates a true Reinforcement Learning Q-table updater under the hood to handle long-term state prediction.

When a decision is made, the system doesn't just look at the immediate reward. It uses the **Bellman Equation** to estimate the future value of the resulting state.

### The Math:
```python
Q(s,a) = Q(s,a) + α * [ R + γ * max_Q(s') - Q(s,a) ]
```

*   **`s` (Current State):** The context bucket of the current lead (e.g., `high_value_low_risk`).
*   **`a` (Action):** The action chosen by the Strategy Manager (e.g., `offer_discount`).
*   **`α` (Learning Rate):** Set to `0.1`. Controls how much new information overrides old information.
*   **`R` (Reward):** The immediate global reward emitted by the Reward Engine.
*   **`γ` (Discount Factor):** Set to `0.95`. This is the key to lookahead—it heavily weights the predicted value of the *next* state.
*   **`max_Q(s')`:** The maximum known Q-value for any action in the *next* state the system finds itself in.

By injecting these Q-values into the LLM's context window as "Policy Weights," the LLM is mathematically grounded. It doesn't hallucinate strategies; it reads a converged Q-table.

---

## 2. Dynamic Economy-Aware Reward Shaping

The reward engine does not use static, linear coefficients. To mimic real-world corporate finance, the penalty space is non-linear and dynamic based on the `budget_remaining`.

### The Global Reward Formula:
```python
R = (Revenue * 0.40) 
  - (Exponential Cost Penalty) 
  - (Exponential Risk Penalty) 
  + (Speed Bonus) 
  + (Oracle Alignment Bonus)
```

### The Exponential Dampener:
When the `budget_ratio` (Budget Remaining / Total Starting Budget) drops below **20%**, the system applies an exponential scaling function to costs and risks:

```python
if budget_ratio < 0.2:
    survival_multiplier = exp(10 * (0.2 - budget_ratio))
    cost_penalty *= survival_multiplier
    risk_penalty *= survival_multiplier
```

**Why this matters:** A static AI will happily spend its last $10,000 on a high-risk lead because the linear math says the Expected Value is positive. Our system's exponential shaping forces the AI to dynamically shift its entire policy from "Aggressive Growth" to "Conservative Survival" as capital dries up.

---

## 3. Experience Memory & CEO Override (RLHF)

The memory system is an indexed experience replay buffer. Instead of randomly sampling past experiences (like DQN), it retrieves memory based on **Contextual Buckets**.

### Discretized State Space:
Continuous lead variables (deal value, risk score) are discretized into a finite state space:
*   `high_value_low_risk`
*   `expensive_lead`
*   `urgent_lead`
*   etc.

### Human-in-the-Loop (CEO Override)
When the AI makes a catastrophic error, the system features a **CEO Override** module. This is a form of **Reinforcement Learning from Human Feedback (RLHF)**. 

When a human overrides an action:
1. The memory buffer purges the AI's bad decision.
2. The human's chosen action is injected.
3. The reward signal is multiplied by **10x** (Human Priority Multiplier).
4. The Bellman update mathematically locks this decision into the Q-table, guaranteeing the AI will not repeat the mistake in that specific context bucket.
