---
title: Multi-Agent SalesOps Arena
emoji: ⚡
colorFrom: purple
colorTo: pink
sdk: docker
app_port: 7860
pinned: false
---

# Multi-Agent SalesOps Arena

> **"Multi-Agent SalesOps Arena is not a CRM app. It is an advanced Enterprise Reinforcement Learning environment where AI agents develop optimal business decision policies through interaction, Bellman Q-Learning, and Human-in-the-Loop conflict resolution."**

---

## Problem Statement

Enterprise sales operations are not single-objective problems. At any given moment:
- **Sales** wants to close every deal possible
- **Finance** wants to minimize CAC and protect margins
- **Compliance** wants to flag risky or non-compliant leads

A single AI agent — no matter how smart — cannot simultaneously optimize all three objectives. The real world is a system of competing incentives. Modeling it as such produces far better outcomes.

---

## Why Single-Agent Systems Fail

| Limitation | Impact |
|---|---|
| Single objective function | Ignores organizational friction |
| No conflict resolution | Blind to cost or risk consequences |
| No memory | Can't learn from past mistakes |
| No explainability | Black-box decisions rejected by stakeholders |

---

## System Architecture

```
┌──────────────────────────────────────────────────────────┐
│            STOCHASTIC SALESOPS ENVIRONMENT               │
│   (Lead Pool | Budget Constraints | Market Volatility)   │
└──────────────────────┬───────────────────────────────────┘
                       │  STATE: Lead observation
                       ▼
       ┌───────────────────────────────────────┐
       │             AGENT LAYER               │
       │  [Sales] [Finance] [Compliance]       │
       │  Each calls LLM + uses Policy Weights │
       └──────────────┬────────────────────────┘
                      │  Recommendations + Confidence
                      ▼
       ┌───────────────────────────────────────┐
       │        ARBITRATION ENGINE             │
       │  Weighted voting + risk/budget dampeners │
       │  → Final action + explanation         │
       └──────────────┬────────────────────────┘
                      │  Final Action
                      ▼
       ┌───────────────────────────────────────┐
       │        REWARD ENGINE                  │
       │  R = Revenue - Exponential Cost - Risk│
       └──────────────┬────────────────────────┘
                      │  Reward signal
                      ▼
       ┌───────────────────────────────────────┐
       │  MEMORY + Q-LEARNING ADAPTATION       │
       │  Bellman Equation State Prediction    │
       │  Human-in-the-loop CEO Override       │
       └───────────────────────────────────────┘
```

---

## Enterprise Features (Hackathon Highlights)

1. **True Q-Learning (Bellman Equation):** The memory engine doesn't just react to immediate rewards. It predicts future state values using a discounted factor (`gamma=0.95`), allowing agents to look ahead and optimize long-term Net Present Value.
2. **Economy-Aware Dynamic Reward Shaping:** The reward engine acts like a real corporate finance department. If the corporate budget drops below 20%, the penalty for taking risks or spending money scales exponentially, forcing the AI into a conservative survival policy.
3. **CEO Override (RLHF):** A Human-in-the-Loop Reinforcement Learning interface. Executives can pause the simulation on edge cases and inject their logic. This applies a 10x multiplier to the memory bucket, permanently overriding the AI's future behavior.
4. **CRM CSV Ingestion:** Stop simulating. Upload a CSV export from Salesforce, HubSpot, or Dynamics to run the multi-agent arbitration on real-world enterprise data.
5. **Oracle Accuracy Score:** We calculate the mathematically perfect "Ground Truth" decision for every lead in the background and compare it to the AI's decision, giving judges a hard percentage metric of the system's accuracy.

---

## Action Space

| Action | Effect |
|---|---|
| `pursue_lead` | Full acquisition cost, high conversion probability |
| `nurture_lead` | Low cost, builds long-term relationship |
| `reject_lead` | Zero cost, zero revenue |
| `escalate_to_enterprise` | Higher CLV, enterprise sales cycle |
| `offer_discount` | Increases conversion, reduces CLV |
| `request_more_info` | Low cost, reduces risk score |
| `notify_compliance` | Eliminates risk, blocks conversion |
| `schedule_demo` | Medium cost, high conversion |

---

## Learning Mechanism

**Context Buckets** classify each lead:
- `high_value_low_risk`, `high_value_high_risk`
- `low_value_low_risk`, `low_value_high_risk`
- `urgent_lead`, `expensive_lead`

**Policy Weights** use the Bellman Equation:
```python
Q(s,a) = Q(s,a) + alpha * (Reward + gamma * max_Q(s') - Q(s,a))
```

Agents retrieve similar past experiences before deciding, making every decision context-aware and mathematically predicting the optimal long-term strategy.

---

## How to Run

### 1. Run via Docker (Recommended)
```bash
docker-compose up --build
```
The Streamlit UI will be available at `http://localhost:8501`.

### 2. Run Locally
```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env and add your HF_TOKEN
python train.py 50
streamlit run app.py
```

---

## Demo Flow

1. Open Streamlit UI → `🔌 CRM Data Import`
2. Upload a custom lead list or download the sample CSV to inject real-world leads.
3. Navigate to `🎯 Run Episode` and click **Run New Episode**.
4. Check the **Oracle Accuracy** metric to see how the agents compare to the mathematical ground truth.
5. If the AI makes a mistake, go to `👑 CEO Override (RLHF)` and inject human logic to fix the policy.
6. Navigate to `🏋️ Train & Compare` and run full training to generate a beautiful reward curve against baselines.
7. Go to `💬 Strategy Chat` and ask the LLM *why* it made its decisions.

---

*Built for hackathon demonstration. Zero GPU required. Containerized for immediate deployment.*
