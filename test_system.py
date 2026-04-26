import pytest
import os
import json
from env import SalesOpsEnvironment
from models import Lead, Experience
from reward import RewardEngine
from memory import ExperienceMemory, PolicyWeights
from arbitration import ArbitrationEngine
from hf_client import HFClient
from train import run_multi_agent_episode, build_agents
import config

def test_environment_generation():
    """Verify the environment can generate valid Pydantic leads without crashing."""
    env = SalesOpsEnvironment()
    lead = env.reset()
    assert isinstance(lead, Lead), "Environment failed to produce a valid Lead object"
    assert lead.deal_value > 0, "Deal value should be positive"
    assert 0.0 <= lead.risk_score <= 1.0, "Risk score should be bounded"

def test_reward_engine():
    """Verify the reward engine math (Dynamic Reward Shaping) doesn't crash on edge cases."""
    env = SalesOpsEnvironment()
    lead = env.reset()
    engine = RewardEngine()
    
    # Test standard budget
    result1 = engine.compute(lead, "pursue_lead", budget_remaining=100000)
    assert isinstance(result1.global_reward, float)
    
    # Test near-zero budget (Dynamic Shaping threshold check)
    result2 = engine.compute(lead, "pursue_lead", budget_remaining=100)
    assert isinstance(result2.global_reward, float)
    assert result2.global_reward != result1.global_reward, "Reward shaping failed to dynamically scale penalty"

def test_memory_q_learning_bellman():
    """Verify the Bellman Equation logic inside memory.py works and updates weights."""
    memory = ExperienceMemory()
    # Use a dummy agent name to avoid loading existing 10.0 weights from disk
    agent = "Test Agent 123"
    policy = memory.get_policy(agent)
    bucket = "high_value_low_risk"
    action = "pursue_lead"
    
    # Ensure it starts at default 1.0
    policy.weights[bucket] = {action: 1.0}
    
    # Simulate a reward and next_state bucket
    memory.update_policies(agent, bucket, action, reward=5.0, next_bucket="urgent_lead")
    
    new_weight = policy.weights.get(bucket, {}).get(action, 1.0)
    assert new_weight > 1.0, f"Positive reward should increase Q-value weight, got {new_weight}"
    assert new_weight <= 10.0, "Weight clipping failed (should not exceed 10.0)"

def test_arbitration_fallback():
    """Verify the ArbitrationEngine can survive a complete LLM failure by falling back to heuristics."""
    class MockFailingHFClient:
        def chat_agent(self, *args, **kwargs):
            return {}
        def chat_strategy_manager(self, *args, **kwargs):
            raise Exception("Simulated Network Crash")
            
    arbiter = ArbitrationEngine(hf_client=MockFailingHFClient())
    
    recs = {
        "Sales Agent": type("obj", (object,), {"agent": "Sales Agent", "recommended_action": "pursue_lead", "confidence": 0.9, "reason": "Revenue"}),
        "Finance Agent": type("obj", (object,), {"agent": "Finance Agent", "recommended_action": "reject_lead", "confidence": 0.8, "reason": "Cost"}),
        "Compliance Agent": type("obj", (object,), {"agent": "Compliance Agent", "recommended_action": "request_more_info", "confidence": 0.95, "reason": "Risk"})
    }
    
    result = arbiter.decide(recs, risk_score=0.7, budget_ratio=0.5, lead_state={}, memory_context=[])
    
    assert result.final_action in config.ACTIONS, "Arbiter failed to select a valid fallback action"
    assert "Conflict detected" in result.reason or result.conflict_detected, "Arbiter did not log the fallback reasoning"

def test_full_training_episode_safety():
    """Run one full multi-agent episode to guarantee the main loop is 100% stable."""
    env = SalesOpsEnvironment()
    memory = ExperienceMemory()
    
    class DummyHFClient:
        def recommend(self, *args, **kwargs):
            return {"recommended_action": "schedule_demo", "confidence": 0.8, "reason": "Test"}
        def chat_strategy_manager(self, *args, **kwargs):
            return {"final_action": "schedule_demo", "reason": "Test"}
            
    client = DummyHFClient()
    arbiter = ArbitrationEngine(hf_client=client)
    agents = build_agents(client, memory)
    
    result = run_multi_agent_episode(env, agents, arbiter, memory, episode_num=1, verbose=False)
    
    assert "total_reward" in result
    assert "conversions" in result
    assert "alignment_score" in result
    assert len(result["experiences"]) == 5, "Episode did not process exactly 5 leads"

if __name__ == "__main__":
    print("Running Demo-Safety Test Suite...")
    pytest.main(["-v", "test_system.py"])
