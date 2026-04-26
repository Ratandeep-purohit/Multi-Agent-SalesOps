"""
config.py — Central Configuration
===================================
All constants, weights, API settings, and paths live here.
Import this module in every other module.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── LLM / HuggingFace ─────────────────────────────────────────────────────
HF_TOKEN      = os.getenv("HF_TOKEN", "")
HF_MODEL      = os.getenv("HF_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
HF_ROUTER_URL = os.getenv("HF_ROUTER_URL", "https://router.huggingface.co/v1/chat/completions")
USE_HF        = os.getenv("USE_HF", "true").lower() != "false"
HF_TIMEOUT    = 30   # seconds

# ── Episode / Simulation ──────────────────────────────────────────────────
LEADS_PER_EPISODE          = 5
DEFAULT_TRAINING_EPISODES  = 50
INITIAL_BUDGET             = 100_000.0
MAX_RISK_TOLERANCE         = 0.75   # global risk ceiling

# ── Action Space ──────────────────────────────────────────────────────────
ACTIONS = [
    "pursue_lead",
    "nurture_lead",
    "reject_lead",
    "escalate_to_enterprise",
    "offer_discount",
    "request_more_info",
    "notify_compliance",
    "schedule_demo",
]

# ── Arbitration Base Weights ──────────────────────────────────────────────
ARBITRATION_WEIGHTS = {
    "sales":      0.30,
    "finance":    0.30,
    "compliance": 0.20,
    "strategy":   0.20,
}

# ── Reward Signal Weights ─────────────────────────────────────────────────
REWARD_WEIGHTS = {
    "revenue":   0.40,
    "cost":      0.25,
    "risk":      0.20,
    "speed":     0.10,
    "alignment": 0.05,
}

# ── Policy Learning ───────────────────────────────────────────────────────
LEARNING_RATE   = 0.15
MEMORY_WINDOW   = 8       # past experiences passed to LLM prompt
MAX_MEMORY_SIZE = 500

# ── Context Buckets (used by memory & learning) ──────────────────────────
CONTEXT_BUCKETS = [
    "high_value_low_risk",
    "high_value_high_risk",
    "low_value_low_risk",
    "low_value_high_risk",
    "urgent_lead",
    "expensive_lead",
]

# ── Simulation Data Pools ─────────────────────────────────────────────────
INDUSTRIES         = ["SaaS", "FinTech", "HealthTech", "Retail", "Manufacturing", "EdTech", "Logistics"]
MARKET_CONDITIONS  = ["bull", "bear", "stable", "volatile"]
COMPANY_SIZES      = ["startup", "smb", "mid-market", "enterprise"]

# ── Paths ─────────────────────────────────────────────────────────────────
OUTPUTS_DIR            = "outputs"
TRAINING_RESULTS_PATH  = f"{OUTPUTS_DIR}/training_results.json"
LOGS_PATH              = f"{OUTPUTS_DIR}/logs.json"
MEMORY_PATH            = f"{OUTPUTS_DIR}/memory.json"
REWARD_CURVE_PATH      = f"{OUTPUTS_DIR}/reward_curve.png"
COMPARISON_CHART_PATH  = f"{OUTPUTS_DIR}/comparison_chart.png"
