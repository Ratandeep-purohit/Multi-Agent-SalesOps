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

**Built using OpenEnv latest release.**

> **"Multi-Agent SalesOps Arena is not a CRM app. It is an advanced Enterprise Reinforcement Learning environment where AI agents develop optimal business decision policies through interaction, Bellman Q-Learning, and Human-in-the-Loop conflict resolution."**

🚀 **[Live Demo on Hugging Face Spaces](https://huggingface.co/spaces/Ratandeep-purohit/Multi-Agent-SalesOps)**  
🎥 **[Watch Full Dashboard Video Demo](https://1drv.ms/v/c/2027d75a724b735f/IQAVoe62lsjIQq8p4GIcHqXuAXb4u94MmDYh21HWbq6pPFo?e=SL2c9I)**

---

## Problem Motivation

Enterprise sales operations are not single-objective problems. At any given moment:
- **Sales** wants to close every deal possible to maximize revenue.
- **Finance** wants to minimize Customer Acquisition Cost (CAC) and protect budget margins.
- **Compliance** wants to flag risky or non-compliant leads to prevent regulatory fines.

A single AI agent — no matter how smart — cannot simultaneously optimize all three competing objectives effectively. The real world is a system of friction and competing incentives. We built the **SalesOps Arena** to model this exact friction using Multi-Agent Reinforcement Learning (MARL).

---

## What the Environment Does

The **Multi-Agent SalesOps Arena** simulates a high-stakes corporate CRM. Leads enter the system with varying values, risk scores, and acquisition costs. 

Three distinct AI agents (Sales, Finance, Compliance) evaluate the lead and argue for different actions based on their specific utility functions. An **Arbitration Engine** weighs their inputs against the current corporate budget and risk tolerance. Finally, a **Reward Engine** calculates the actual financial impact of the decision, and the agents update their **Policy Weights using Bellman Q-Learning** to make better decisions in the future.

---

## Observation / Action / Reward

### Observation Space
The environment emits a state dictionary representing a CRM Lead:
- `deal_value`: Continuous (Potential revenue of the lead)
- `risk_score`: Continuous [0.0, 1.0] (Probability of compliance failure)
- `lead_score`: Continuous [0.0, 1.0] (Probability of conversion)
- `urgency`: Continuous [0.0, 1.0] (Time sensitivity)
- `acquisition_cost`: Continuous (Cost to pursue the lead)

### Action Space (Discrete 8)
1. `pursue_lead`
2. `nurture_lead`
3. `reject_lead`
4. `escalate_to_enterprise`
5. `offer_discount`
6. `request_more_info`
7. `notify_compliance`
8. `schedule_demo`

### Reward Signal
The reward is a dynamically shaped, multi-signal economy calculation:
`Reward = Revenue - (Exponential Cost based on remaining budget) - (Risk Penalty) + Alignment Bonus`

---

## Tasks: Easy, Medium, Hard

The environment dynamically scales difficulty based on corporate conditions:
- **🟢 Easy Task:** Abundant corporate budget (>80%) and low-risk leads. The AI simply learns to maximize conversion rates.
- **🟡 Medium Task:** Declining budget (<50%). The AI must learn to balance pursuit costs against potential revenue (Cost-Aware RL).
- **🔴 Hard Task:** Budget crisis (<20%) mixed with high-risk leads. The AI must discover the optimal survival policy: rejecting risky leads entirely and only pursuing guaranteed, cheap conversions to avoid exponential bankruptcy penalties.

---

## Training Setup

- **Frameworks:** Python 3.11, OpenEnv, Hugging Face `Mistral-7B-Instruct`.
- **Algorithm:** Multi-Agent Bellman Q-Learning (Tabular Experience Replay).
- **Memory:** JSON-backed Q-Table state persistence mapping context buckets to policy weights.
- **Hardware:** CPU-friendly (uses HF Inference API for LLM routing).

### How to Reproduce Training

You can reproduce the exact training benchmark (comparing Random vs. Greedy vs. Multi-Agent RL) in two ways:

1. **Colab Notebook (Recommended for Judges):**
   Open the included Jupyter Notebook in Google Colab to run the training cloud-natively.
   👉 **[notebooks/training_colab.ipynb](notebooks/training_colab.ipynb)**

2. **Terminal Script:**
   ```bash
   python train.py 100
   ```
   *This will run 100 episodes and automatically generate the `reward_curve.png`, `loss_curve.png`, and `comparison_chart.png` in the `outputs/` folder.*

---

## Results

After running the Reinforcement Learning benchmark, the Multi-Agent system converges and learns to avoid the massive financial penalties that destroy the greedy baseline. 

| Baseline Mode | Avg Reward Score | Avg Conversions | Avg Risk Incidents |
|---|---|---|---|
| **Random Baseline** | -24,267.75 | 0.70 | 0.80 |
| **Greedy Heuristic** | -16,327.23 | 0.80 | 0.40 |
| **Multi-Agent / RL Agent** | **+4,800.00** | **1.60** | **0.30** |

*Note: The Multi-Agent RL model actively minimizes risk incidents while doubling conversion rates.*

**Training Proofs:**
When you run `train.py`, it generates:
- `outputs/reward_curve.png`
- `outputs/loss_curve.png`
- `outputs/comparison_chart.png`

*(Run the training script locally or in Colab to generate and view the latest curve graphics).*

---

## How to Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/Ratandeep-purohit/Multi-Agent-SalesOps.git
cd Multi-Agent-SalesOps

# 2. Install dependencies
pip install -r requirements.txt

# 3. Setup Environment Variables
cp .env.example .env
# Edit .env and add your HF_TOKEN

# 4. Run the Streamlit Dashboard
streamlit run app.py
```
*(The dashboard will be available at `http://localhost:8501`)*

---

## Video / Blog / Slides Links

- 🎥 **Video Demo:** [Watch Dashboard Walkthrough on OneDrive](https://1drv.ms/v/c/2027d75a724b735f/IQAVoe62lsjIQq8p4GIcHqXuAXb4u94MmDYh21HWbq6pPFo?e=SL2c9I)
- 🚀 **Hugging Face Space:** [Live Demo](https://huggingface.co/spaces/Ratandeep-purohit/Multi-Agent-SalesOps)
- 📊 **Architecture Doc:** See `system_architecture.md` inside the repo.

---

## Team / Submission Notes

- **Team Name:** Ratandeep Purohit
- **Track:** OpenEnv Hackathon Submission
- **Notes for Judges:** The `openenv.yaml` file is included in the root directory. This project relies on the HF Inference API for the agent reasoning layer. For the best experience, please use the Hugging Face Spaces Live Demo. If running locally, ensure you provide a valid `HF_TOKEN`.
