"""
hf_client.py — HuggingFace Inference Router Client
====================================================
Uses the HF Inference Router (OpenAI-compatible format):
  POST https://router.huggingface.co/v1/chat/completions

All config loaded from .env:
  HF_TOKEN       — your HuggingFace token
  HF_MODEL       — model ID (default: meta-llama/Llama-3.1-8B-Instruct)
  HF_ROUTER_URL  — override router URL (optional)
  USE_HF         — set to "false" to force fallback mode

Fallback policy:
  Any HTTP error (401/402/403/404/429/5xx), timeout, HTML response,
  or JSON parse failure → rule-based heuristic fallback.
  The system NEVER crashes during a demo.

Caching:
  Successful API responses cached to outputs/hf_cache.json by prompt hash.
  Same prompt never calls the API twice.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import requests
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

# ── Config (all from .env) ─────────────────────────────────────────────────
HF_TOKEN      = os.getenv("HF_TOKEN", "")
HF_MODEL      = os.getenv("HF_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
HF_ROUTER_URL = os.getenv("HF_ROUTER_URL", "https://router.huggingface.co/v1/chat/completions")
USE_HF        = os.getenv("USE_HF", "true").lower() != "false"

TIMEOUT    = 30
CACHE_PATH = "outputs/hf_cache.json"

SYSTEM_PROMPT = (
    "You are a precise AI agent in a Multi-Agent SalesOps decision system. "
    "Always respond ONLY with a valid JSON object as specified. "
    "Do not include any markdown, code fences, or explanation outside the JSON."
)


# ══════════════════════════════════════════════════════════════════════════════
# CORE ROUTER CALL (as specified)
# ══════════════════════════════════════════════════════════════════════════════

def call_hf_router(
    prompt: str,
    system_prompt: str = SYSTEM_PROMPT,
) -> dict:
    """
    Call HF Inference Router with OpenAI-compatible chat format.
    Returns parsed JSON dict on success.
    Raises ValueError if HF_TOKEN is missing.
    Raises HTTPError / other exceptions on network failure.
    """
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN is missing. Set it in your .env file.")

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type":  "application/json",
    }

    payload = {
        "model":    HF_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens":  400,
    }

    response = requests.post(
        HF_ROUTER_URL,
        headers = headers,
        json    = payload,
        timeout = TIMEOUT,
    )

    response.raise_for_status()
    data    = response.json()
    content = data["choices"][0]["message"]["content"]

    try:
        return json.loads(content)
    except Exception:
        return {"raw_response": content, "parse_error": True}

def call_hf_router_text(prompt: str, system_prompt: str) -> str:
    """Same as call_hf_router but returns raw text instead of parsing JSON."""
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN is missing.")
    
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model":    HF_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": prompt},
        ],
        "temperature": 0.4,
        "max_tokens":  600,
    }
    
    response = requests.post(
        HF_ROUTER_URL, headers=headers, json=payload, timeout=TIMEOUT
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]



# ══════════════════════════════════════════════════════════════════════════════
# DISK CACHE
# ══════════════════════════════════════════════════════════════════════════════

def _load_cache() -> dict:
    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_cache(cache: dict):
    os.makedirs("outputs", exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)


def _cache_key(prompt: str) -> str:
    return hashlib.sha256(f"{HF_MODEL}::{prompt}".encode()).hexdigest()[:16]


# ══════════════════════════════════════════════════════════════════════════════
# PROMPT BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def build_agent_prompt(
    agent_name:        str,
    objective:         str,
    lead_state:        dict,
    memory_summary:    list[dict],
    available_actions: list[str],
) -> str:
    lead_str    = json.dumps(lead_state, indent=2)
    actions_str = "\n".join(f"  - {a}" for a in available_actions)

    mem_str = (
        "\n".join(
            f"  - Context: {m.get('context_bucket','?')} | "
            f"Action: {m.get('final_action','?')} | "
            f"Reward: {m.get('reward', 0):.3f} | "
            f"Outcome: {m.get('outcome','?')}"
            for m in memory_summary[-5:]
        )
        if memory_summary
        else "  No past experience yet."
    )

    return f"""You are {agent_name} in a Multi-Agent SalesOps Arena.

Role Objective:
{objective}

Current Lead State:
{lead_str}

Recent Learning Memory:
{mem_str}

Available Actions:
{actions_str}

Rules:
- Choose only one valid action from the list above.
- Consider your role objective and recent rewards.
- Return JSON only.
- Do not include markdown or explanation outside JSON.

Return:
{{
  "agent": "{agent_name}",
  "recommended_action": "...",
  "confidence": 0.0,
  "reason": "..."
}}"""


def build_strategy_prompt(
    lead_state:            dict,
    agent_recommendations: dict,
    memory_summary:        list[dict],
    available_actions:     list[str],
) -> str:
    lead_str    = json.dumps(lead_state, indent=2)
    actions_str = "\n".join(f"  - {a}" for a in available_actions)

    recs_str = "\n".join(
        f"  {agent}: action={rec.get('recommended_action','?')} | "
        f"confidence={rec.get('confidence', 0):.2f} | "
        f"reason={rec.get('reason','')}"
        for agent, rec in agent_recommendations.items()
    )

    mem_str = (
        "\n".join(
            f"  - Context: {m.get('context_bucket','?')} | "
            f"Action: {m.get('final_action','?')} | "
            f"Reward: {m.get('reward', 0):.3f}"
            for m in memory_summary[-5:]
        )
        if memory_summary
        else "  No past experience yet."
    )

    return f"""You are the Strategy Manager in a Multi-Agent SalesOps Arena.

Your job is to choose the final action by balancing revenue, cost, risk, and long-term value.

Lead State:
{lead_str}

Agent Recommendations:
{recs_str}

Recent Learning Memory:
{mem_str}

Available Actions:
{actions_str}

Return JSON only:
{{
  "final_action": "...",
  "confidence": 0.0,
  "reason": "...",
  "conflict_detected": true
}}"""


# ══════════════════════════════════════════════════════════════════════════════
# JSON PARSER (robust)
# ══════════════════════════════════════════════════════════════════════════════

def _parse_json(text: str) -> Optional[dict]:
    """Extracts JSON from LLM output, handles markdown fences."""
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass
    return None


# ══════════════════════════════════════════════════════════════════════════════
# RULE-BASED FALLBACK (always available, no API needed)
# ══════════════════════════════════════════════════════════════════════════════

def _fallback_agent(agent_name: str, lead_state: dict) -> dict:
    """High-quality heuristic fallback per agent role."""
    deal_value = lead_state.get("deal_value", 0)
    risk_score = lead_state.get("risk_score", 0)
    lead_score = lead_state.get("lead_score", 0)
    acq_cost   = lead_state.get("acquisition_cost", 0)
    urgency    = lead_state.get("urgency", 0)
    flags      = lead_state.get("compliance_flags", [])
    margin     = (deal_value - acq_cost) / max(deal_value, 1)
    name       = agent_name.lower()

    if "sales" in name:
        if lead_score >= 0.7 and deal_value >= 20_000:
            a, c, r = "pursue_lead", 0.88, f"Strong lead score ({lead_score:.2f}) and deal value (${deal_value:,.0f}). Full pursuit."
        elif urgency >= 0.7:
            a, c, r = "schedule_demo", 0.74, f"High urgency ({urgency:.2f}). Demo is the fastest path to conversion."
        elif lead_score >= 0.5:
            a, c, r = "nurture_lead", 0.62, f"Moderate signal ({lead_score:.2f}). Nurture before committing full budget."
        else:
            a, c, r = "request_more_info", 0.50, "Insufficient signal. Gather qualifying data first."

    elif "finance" in name:
        if margin < 0.25:
            a, c, r = "reject_lead", 0.88, f"Margin ({margin:.1%}) below 25% floor. CAC unacceptable."
        elif margin < 0.50:
            a, c, r = "nurture_lead", 0.72, f"Thin margin ({margin:.1%}). Nurture to improve deal economics."
        elif risk_score >= 0.65:
            a, c, r = "request_more_info", 0.68, f"Risk ({risk_score:.2f}) requires financial due diligence."
        else:
            a, c, r = "pursue_lead", 0.80, f"Healthy margin ({margin:.1%}), acceptable risk. Finance approved."

    elif "compliance" in name:
        if risk_score >= 0.75 or "blacklisted_region" in flags:
            a, c, r = "notify_compliance", 0.94, f"Risk {risk_score:.2f} or blacklist flag. Compliance review mandatory."
        elif risk_score >= 0.55 or flags:
            a, c, r = "request_more_info", 0.78, f"Elevated risk ({risk_score:.2f}) or flags {flags}. Request documentation."
        else:
            a, c, r = "pursue_lead", 0.68, f"Clean risk profile ({risk_score:.2f}). No compliance issues."

    else:   # strategy manager
        if risk_score >= 0.70:
            a, c, r = "request_more_info", 0.72, "Balancing upside vs elevated risk. Gather info before committing."
        elif lead_score >= 0.65 and margin >= 0.40:
            a, c, r = "pursue_lead", 0.84, "Strong quality and margin support global pursuit objective."
        elif urgency >= 0.75:
            a, c, r = "schedule_demo", 0.70, "High urgency. Demo maximises conversion speed."
        else:
            a, c, r = "nurture_lead", 0.57, "Mixed signals. Nurture to reduce uncertainty."

    return {"agent": agent_name, "recommended_action": a, "confidence": c, "reason": r}


def _fallback_strategy(agent_recommendations: dict, lead_state: dict) -> dict:
    """Heuristic majority-vote strategy with risk gate."""
    from collections import Counter
    risk  = lead_state.get("risk_score", 0)
    votes = [v.get("recommended_action", "nurture_lead") for v in agent_recommendations.values()]
    best, count = Counter(votes).most_common(1)[0]
    if risk >= 0.75 and best == "pursue_lead":
        best = "request_more_info"
    return {
        "final_action":     best,
        "confidence":       0.65,
        "reason":           f"Fallback majority vote: {best} ({count}/{len(votes)}). Risk={risk:.2f}.",
        "conflict_detected": len(set(votes)) > 1,
    }


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC CLIENT CLASS
# ══════════════════════════════════════════════════════════════════════════════

class HFClient:
    """
    Production-safe HF Inference Router client.

    Decision tree per call:
      1. USE_HF=false or no HF_TOKEN → fallback immediately
      2. Cache hit                   → return cached result
      3. Router call succeeds + JSON → cache + return
      4. Any failure                 → log + return fallback
    """

    def __init__(self):
        self._cache:  dict          = _load_cache()
        self._api_ok: Optional[bool] = None

    # ── Public: Agent Recommend ────────────────────────────────────────────

    def recommend(
        self,
        agent_name:     str,
        agent_role:     str,
        state_summary:  dict,
        actions:        list[str],
        memory_context: list[dict],
    ) -> dict:
        """Called by Sales, Finance, Compliance agents."""
        if not HF_TOKEN or not USE_HF:
            return _fallback_agent(agent_name, state_summary)

        prompt = build_agent_prompt(
            agent_name        = agent_name,
            objective         = agent_role,
            lead_state        = state_summary,
            memory_summary    = memory_context,
            available_actions = actions,
        )
        result = self._safe_call(prompt, agent_name)
        if result and "recommended_action" in result:
            if result["recommended_action"] not in actions:
                result["recommended_action"] = actions[0]
            result.setdefault("agent", agent_name)
            return result

        return _fallback_agent(agent_name, state_summary)

    # ── Public: Strategy Manager ───────────────────────────────────────────

    def strategy_decide(
        self,
        lead_state:            dict,
        agent_recommendations: dict,
        memory_context:        list[dict],
        actions:               list[str],
    ) -> dict:
        """Called by the Strategy Manager arbitration engine."""
        if not HF_TOKEN or not USE_HF:
            return _fallback_strategy(agent_recommendations, lead_state)

        prompt = build_strategy_prompt(
            lead_state            = lead_state,
            agent_recommendations = agent_recommendations,
            memory_summary        = memory_context,
            available_actions     = actions,
        )
        result = self._safe_call(prompt, "Strategy Manager")
        if result and "final_action" in result:
            if result["final_action"] not in actions:
                result["final_action"] = actions[0]
            return result

        return _fallback_strategy(agent_recommendations, lead_state)

    # ── Public: Strategy Manager Chat ──────────────────────────────────────

    def chat_strategy_manager(self, message: str, memory_summary: dict, policy_snapshots: dict) -> str:
        """Interactive free-text chat with the Strategy Manager about its policies."""
        if not HF_TOKEN or not USE_HF:
            return "HF API is in fallback mode. I cannot chat interactively right now, but I am still using heuristic rules to resolve conflicts."

        sys_prompt = (
            "You are the Strategy Manager of a Multi-Agent SalesOps Arena. "
            "Your job is to explain the system's behavior to the human user. "
            "You have access to the current memory stats and the policy weights of the agents. "
            "Keep your answers concise, professional, and directly address the user's question."
        )

        prompt = f"""
Current Memory Stats:
{json.dumps(memory_summary, indent=2)}

Current Policy Weights:
{json.dumps(policy_snapshots, indent=2)}

User Question:
{message}
"""
        try:
            return call_hf_router_text(prompt, sys_prompt)
        except Exception as e:
            return f"Error communicating with LLM: {str(e)}"

    # ── Internal: Cache + Safe Wrapper ─────────────────────────────────────

    def _safe_call(self, prompt: str, caller: str = "") -> Optional[dict]:
        """Cache-wrapped call to call_hf_router with full error handling."""
        key = _cache_key(prompt)

        # Cache hit
        if key in self._cache:
            print(f"    [HF Cache] {caller} — key={key}")
            return self._cache[key]

        # Live call
        try:
            result = call_hf_router(prompt)

            # Handle parse_error flag
            if result.get("parse_error"):
                raw = result.get("raw_response", "")
                parsed = _parse_json(raw)
                if parsed:
                    result = parsed
                else:
                    print(f"    [HF] {caller} — unparseable response: {raw[:80]}")
                    self._api_ok = False
                    return None

            self._cache[key] = result
            _save_cache(self._cache)
            self._api_ok = True
            return result

        except requests.exceptions.HTTPError as e:
            code = e.response.status_code if e.response else "?"
            msgs = {
                401: "Unauthorized — invalid HF_TOKEN.",
                402: "Payment required — free-tier token has no Inference Router access.",
                403: "Forbidden — token lacks permission for this model.",
                404: "Model not found on HF Inference Router.",
                429: "Rate limit exceeded — too many requests.",
            }
            msg = msgs.get(code, f"HTTP {code} error.")
            print(f"    [HF] {caller} — {msg} Fallback active.")
            self._api_ok = False

        except requests.exceptions.Timeout:
            print(f"    [HF] {caller} — Timeout after {TIMEOUT}s. Fallback active.")
            self._api_ok = False

        except requests.exceptions.ConnectionError:
            print(f"    [HF] {caller} — Network unreachable. Fallback active.")
            self._api_ok = False

        except ValueError as e:
            print(f"    [HF] {caller} — Config error: {e}")
            self._api_ok = False

        except Exception as e:
            print(f"    [HF] {caller} — Unexpected: {e}. Fallback active.")
            self._api_ok = False

        return None

    @property
    def is_api_available(self) -> bool:
        return bool(HF_TOKEN) and USE_HF and self._api_ok is not False

    @property
    def status(self) -> str:
        if not HF_TOKEN:
            return "NO_TOKEN"
        if not USE_HF:
            return "DISABLED"
        if self._api_ok is True:
            return "LIVE"
        if self._api_ok is False:
            return "FALLBACK"
        return "UNTESTED"
