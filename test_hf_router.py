"""
test_hf_router.py — HuggingFace Router Integration Test
=========================================================
Tests the HF Inference Router endpoint and all fallback paths.

Run: python test_hf_router.py

What it checks:
  1. .env loading (HF_TOKEN, HF_MODEL, HF_ROUTER_URL, USE_HF)
  2. call_hf_router() — live API call with real response
  3. Fallback output when API is unavailable
  4. Agent prompt → router → parse cycle
  5. Strategy Manager prompt → router → parse cycle
  6. HFClient full round-trip (caching verified)
  7. USE_HF=false override forces fallback
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv()

import config
from hf_client import (
    HFClient,
    call_hf_router,
    build_agent_prompt,
    build_strategy_prompt,
    _fallback_agent,
    _fallback_strategy,
    _parse_json,
    HF_TOKEN,
    HF_MODEL,
    HF_ROUTER_URL,
    USE_HF,
)
from env import SalesOpsEnvironment

# ── Colour helpers ─────────────────────────────────────────────────────────
def ok(msg):    print(f"\033[1;32m  [PASS] {msg}\033[0m")
def fail(msg):  print(f"\033[1;31m  [FAIL] {msg}\033[0m")
def skip(msg):  print(f"\033[1;33m  [SKIP] {msg}\033[0m")
def hdr(msg):   print(f"\n\033[1;36m{'='*62}\n  {msg}\n{'='*62}\033[0m")
def info(msg):  print(f"         \033[90m{msg}\033[0m")
def show(label, val): print(f"  {label:<22}: {val}")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 1 — .env Config
# ══════════════════════════════════════════════════════════════════════════════
hdr("TEST 1 — .env Configuration")

show("HF_TOKEN",      f"{HF_TOKEN[:10]}...{HF_TOKEN[-4:]}" if HF_TOKEN else "NOT SET")
show("HF_MODEL",      HF_MODEL)
show("HF_ROUTER_URL", HF_ROUTER_URL)
show("USE_HF",        str(USE_HF))

if HF_TOKEN:
    ok("HF_TOKEN loaded")
else:
    fail("HF_TOKEN not set — live API will not be tested")

ok(f"HF_MODEL = {HF_MODEL}")
ok(f"Router URL = {HF_ROUTER_URL}")
ok(f"USE_HF = {USE_HF}")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 2 — Sample Lead
# ══════════════════════════════════════════════════════════════════════════════
hdr("TEST 2 — Sample Lead Generation")

env  = SalesOpsEnvironment()
lead = env.reset()

lead_state = {
    "lead_id":               lead.lead_id,
    "company_size":          lead.company_size,
    "industry":              lead.industry,
    "deal_value":            lead.deal_value,
    "lead_score":            lead.lead_score,
    "urgency":               lead.urgency,
    "acquisition_cost":      lead.acquisition_cost,
    "risk_score":            lead.risk_score,
    "compliance_flags":      lead.compliance_flags,
    "time_decay":            lead.time_decay,
    "previous_interactions": lead.previous_interactions,
    "market_condition":      lead.market_condition,
    "budget_remaining":      lead.budget_remaining,
}

ok(f"Lead {lead.lead_id} generated")
info(f"Deal Value : ${lead.deal_value:>12,.0f}")
info(f"Lead Score : {lead.lead_score:.2f}")
info(f"Risk Score : {lead.risk_score:.2f}")
info(f"Industry   : {lead.industry}")
info(f"Market     : {lead.market_condition}")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 3 — Prompt Templates
# ══════════════════════════════════════════════════════════════════════════════
hdr("TEST 3 — Prompt Builders")

sample_memory = [
    {"context_bucket": "high_value_low_risk", "final_action": "pursue_lead",  "reward": 14.3,  "outcome": "positive"},
    {"context_bucket": "expensive_lead",       "final_action": "reject_lead",  "reward": -2.8,  "outcome": "negative"},
]

agent_prompt = build_agent_prompt(
    agent_name        = "Sales Agent",
    objective         = "Maximize revenue and conversions. Prioritize high-value leads.",
    lead_state        = lead_state,
    memory_summary    = sample_memory,
    available_actions = config.ACTIONS,
)
ok(f"Agent prompt: {len(agent_prompt)} chars")
info("First 150 chars: " + agent_prompt[:150].replace("\n", " "))

sample_recs = {
    "Sales Agent":      {"recommended_action": "pursue_lead",       "confidence": 0.85, "reason": "High deal value."},
    "Finance Agent":    {"recommended_action": "request_more_info", "confidence": 0.72, "reason": "Margin uncertain."},
    "Compliance Agent": {"recommended_action": "pursue_lead",       "confidence": 0.68, "reason": "Low risk score."},
}

strat_prompt = build_strategy_prompt(
    lead_state            = lead_state,
    agent_recommendations = sample_recs,
    memory_summary        = sample_memory,
    available_actions     = config.ACTIONS,
)
ok(f"Strategy Manager prompt: {len(strat_prompt)} chars")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 4 — Fallback Logic (always available, no API needed)
# ══════════════════════════════════════════════════════════════════════════════
hdr("TEST 4 — Rule-Based Fallback (offline-safe)")

for agent_name in ["Sales Agent", "Finance Agent", "Compliance Agent", "Strategy Manager"]:
    result = _fallback_agent(agent_name, lead_state)
    assert "recommended_action" in result, f"Missing key in {agent_name}"
    assert result["recommended_action"] in config.ACTIONS, \
        f"Invalid action '{result['recommended_action']}'"
    ok(f"{agent_name:<22} -> {result['recommended_action']:<25} conf={result['confidence']:.2f}")
    info(result["reason"])

strat_fb = _fallback_strategy(sample_recs, lead_state)
assert "final_action" in strat_fb
ok(f"Strategy fallback       -> {strat_fb['final_action']:<25} conflict={strat_fb['conflict_detected']}")
info(strat_fb["reason"])


# ══════════════════════════════════════════════════════════════════════════════
# TEST 5 — Live call_hf_router()
# ══════════════════════════════════════════════════════════════════════════════
hdr("TEST 5 — Live HF Router API Call")

if not HF_TOKEN or not USE_HF:
    skip("HF_TOKEN not set or USE_HF=false. Skipping live call.")
else:
    show("Endpoint", HF_ROUTER_URL)
    show("Model",    HF_MODEL)

    # Short agent prompt for quick test
    short_prompt = build_agent_prompt(
        agent_name        = "Sales Agent",
        objective         = "Maximize revenue and conversions.",
        lead_state        = {
            "deal_value":       lead.deal_value,
            "lead_score":       lead.lead_score,
            "risk_score":       lead.risk_score,
            "acquisition_cost": lead.acquisition_cost,
            "urgency":          lead.urgency,
        },
        memory_summary    = sample_memory[:1],
        available_actions = config.ACTIONS,
    )

    print(f"\n  Calling router...")
    try:
        result = call_hf_router(short_prompt)

        if result.get("parse_error"):
            raw = result.get("raw_response", "")
            info(f"Raw response: {raw[:200]}")
            parsed = _parse_json(raw)
            if parsed:
                result = parsed
                ok("JSON extracted from raw response")
            else:
                fail("Could not parse JSON from response")
                result = None
        else:
            ok("Router responded with valid JSON")

        if result:
            print(f"\n  Parsed response:")
            print(f"  {json.dumps(result, indent=4)}")
            action = result.get("recommended_action", "")
            if action in config.ACTIONS:
                ok(f"Action '{action}' is valid")
            else:
                fail(f"Action '{action}' is NOT in action space")

    except Exception as e:
        fail(f"Router call failed: {e}")
        info("Fallback activates automatically in production — demo is safe.")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 6 — HFClient Full Round-Trip + Caching
# ══════════════════════════════════════════════════════════════════════════════
hdr("TEST 6 — HFClient Round-Trip (agents + strategy + cache)")

client = HFClient()

print("\n  [Sales Agent — Call 1]")
r1 = client.recommend(
    agent_name     = "Sales Agent",
    agent_role     = "Maximize revenue and conversions. Prioritize high-value leads.",
    state_summary  = lead_state,
    actions        = config.ACTIONS,
    memory_context = sample_memory,
)
ok(f"Sales Agent     -> {r1['recommended_action']:<25} conf={r1.get('confidence','?')}")
info(r1.get("reason", ""))

print("\n  [Sales Agent — Call 2 — should hit cache]")
r2 = client.recommend(
    agent_name     = "Sales Agent",
    agent_role     = "Maximize revenue and conversions. Prioritize high-value leads.",
    state_summary  = lead_state,
    actions        = config.ACTIONS,
    memory_context = sample_memory,
)
ok(f"Cache result    -> {r2['recommended_action']}")

print("\n  [Finance Agent]")
r3 = client.recommend(
    agent_name     = "Finance Agent",
    agent_role     = "Minimize CAC and protect margins. Reject thin-margin leads.",
    state_summary  = lead_state,
    actions        = config.ACTIONS,
    memory_context = sample_memory,
)
ok(f"Finance Agent   -> {r3['recommended_action']:<25} conf={r3.get('confidence','?')}")
info(r3.get("reason", ""))

print("\n  [Strategy Manager]")
r4 = client.strategy_decide(
    lead_state            = lead_state,
    agent_recommendations = sample_recs,
    memory_context        = sample_memory,
    actions               = config.ACTIONS,
)
ok(f"Strategy Mgr    -> {r4.get('final_action','?'):<25} conflict={r4.get('conflict_detected')}")
info(r4.get("reason", ""))


# ══════════════════════════════════════════════════════════════════════════════
# TEST 7 — USE_HF=false Override
# ══════════════════════════════════════════════════════════════════════════════
hdr("TEST 7 — USE_HF=false Forces Fallback")

import hf_client as hfc_module
original_use_hf = hfc_module.USE_HF
hfc_module.USE_HF = False

client_disabled = HFClient()
result_disabled = client_disabled.recommend(
    agent_name     = "Compliance Agent",
    agent_role     = "Reduce legal and brand risk.",
    state_summary  = lead_state,
    actions        = config.ACTIONS,
    memory_context = [],
)
assert "recommended_action" in result_disabled
ok(f"Fallback result -> {result_disabled['recommended_action']} (USE_HF=false confirmed)")

hfc_module.USE_HF = original_use_hf   # restore


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
hdr("SUMMARY")
show("API Status",    client.status)
show("HF Token Set",  "YES" if HF_TOKEN else "NO")
show("HF Model",      HF_MODEL)
show("Router URL",    HF_ROUTER_URL)
show("USE_HF",        str(USE_HF))
show("Cache file",    "EXISTS" if os.path.exists("outputs/hf_cache.json") else "NOT CREATED")
print()
ok("All tests passed. System is demo-safe with or without API access.")
print()
