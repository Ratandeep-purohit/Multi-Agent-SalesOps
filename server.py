"""
server.py — FastAPI REST Server
================================
Exposes the full Multi-Agent SalesOps Arena as a REST API.

Endpoints:
  GET  /             — project info
  GET  /health       — health check
  POST /reset        — reset environment
  POST /step         — single environment step
  POST /run-episode  — run one full episode
  POST /train        — run N-episode training
  GET  /metrics      — current metrics
  GET  /memory       — recent experiences
  GET  /logs         — explainable decision logs
"""

from __future__ import annotations
import json
import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from env import SalesOpsEnvironment
from agents import build_agents
from arbitration import ArbitrationEngine
from memory import ExperienceMemory
from metrics import MetricsTracker
from models import Experience
from hf_client import HFClient
from train import run_training, run_multi_agent_episode
import config

# ── App setup ──────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "Multi-Agent SalesOps Arena",
    description = "A learning environment where AI agents develop better business decision policies through interaction, feedback, and conflict resolution.",
    version     = "1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global singletons ──────────────────────────────────────────────────────
_env     = SalesOpsEnvironment()
_client  = HFClient()
_memory  = ExperienceMemory()
_arbiter = ArbitrationEngine(hf_client=_client)
_agents  = build_agents(_client, _memory)
_tracker = MetricsTracker()
_episode_counter = 0


# ── Request / Response schemas ─────────────────────────────────────────────
class StepRequest(BaseModel):
    action: str

class TrainRequest(BaseModel):
    n_episodes: int = config.DEFAULT_TRAINING_EPISODES
    verbose:    bool = False


# ══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/")
def root():
    return {
        "project":     "Multi-Agent SalesOps Arena",
        "version":     "1.0.0",
        "description": (
            "A real multi-agent RL environment for business decision-making. "
            "Agents: Sales, Finance, Compliance, Strategy Manager."
        ),
        "agents":   ["Sales Agent", "Finance Agent", "Compliance Agent", "Strategy Manager"],
        "actions":  config.ACTIONS,
        "docs_url": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status":        "ok",
        "hf_token_set":  bool(config.HF_TOKEN),
        "hf_model":      config.HF_MODEL,
        "memory_size":   len(_memory.experiences),
        "episodes_run":  _episode_counter,
    }


@app.post("/reset")
def reset_env():
    global _episode_counter
    lead = _env.reset()
    _episode_counter += 1
    return {
        "message":    "Environment reset. New episode started.",
        "episode":    _episode_counter,
        "first_lead": lead.model_dump(),
    }


@app.post("/step")
def step_env(req: StepRequest):
    if _env.done:
        raise HTTPException(
            status_code=400,
            detail="Episode is done. Call POST /reset to start a new episode."
        )
    if req.action not in config.ACTIONS:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid action '{req.action}'. Valid: {config.ACTIONS}"
        )

    lead        = _env.current_lead
    recs        = {key: agent.recommend(lead) for key, agent in _agents.items()}
    budget_ratio = _env.budget / config.INITIAL_BUDGET
    arb         = _arbiter.decide(recs, lead.risk_score, budget_ratio)
    result      = _env.step(req.action)

    return {
        "lead":          lead.model_dump(),
        "recommendations": {k: r.model_dump() for k, r in recs.items()},
        "arbitration":   arb.model_dump(),
        "reward":        result.reward.model_dump(),
        "done":          result.done,
        "next_lead":     result.next_state.model_dump() if result.next_state else None,
        "info":          result.info,
    }


@app.post("/run-episode")
def run_episode():
    global _episode_counter
    _episode_counter += 1
    result = run_multi_agent_episode(
        _env, _agents, _arbiter, _memory, _episode_counter
    )
    _tracker.record(
        mode="multi_agent",
        episode=_episode_counter,
        total_reward=result["total_reward"],
        conversions=result["conversions"],
        risk_incidents=result["risk_incidents"],
        budget_spent=result["budget_spent"],
        action_dist=result["action_dist"],
        policy_snapshots=_memory.all_policy_snapshots(),
    )
    _memory.save()
    return {
        "episode":          _episode_counter,
        "total_reward":     round(result["total_reward"], 4),
        "conversions":      result["conversions"],
        "risk_incidents":   result["risk_incidents"],
        "budget_spent":     round(result["budget_spent"], 2),
        "action_dist":      result["action_dist"],
        "experiences":      [e.model_dump() for e in result["experiences"]],
    }


@app.post("/train")
def train(req: TrainRequest):
    summaries = run_training(n_episodes=req.n_episodes, verbose=req.verbose)
    return {
        "message":    f"Training complete. {req.n_episodes} episodes per mode.",
        "summaries":  summaries,
        "outputs": {
            "training_results": config.TRAINING_RESULTS_PATH,
            "logs":             config.LOGS_PATH,
            "reward_curve":     config.REWARD_CURVE_PATH,
            "comparison_chart": config.COMPARISON_CHART_PATH,
        },
    }


@app.get("/metrics")
def get_metrics():
    return {
        "tracker_summaries": _tracker.all_summaries(),
        "memory_summary":    _memory.summary_stats(),
        "policy_weights":    _memory.all_policy_snapshots(),
    }


@app.get("/memory")
def get_memory(n: int = 20):
    return {
        "count":   len(_memory.experiences),
        "recent":  _memory.recent(n),
        "summary": _memory.summary_stats(),
    }


@app.get("/logs")
def get_logs(n: int = 50):
    return {
        "logs": _env.render_logs(n),
        "total": len(_env._all_logs),
    }


# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
