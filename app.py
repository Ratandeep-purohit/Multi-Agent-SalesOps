"""
app.py — Streamlit Dashboard
=============================
Interactive UI for the Multi-Agent SalesOps Arena.
Runs standalone (no server dependency).
"""

from __future__ import annotations
import json
import os
import sys
import time
import random

import streamlit as st


# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multi-Agent SalesOps Arena",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── WORLD-CLASS CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;600&display=swap');

/* === RESET & BASE === */
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
* { box-sizing: border-box; }

/* === GLOBAL TEXT VISIBILITY === */
body, .stApp, [data-testid="stAppViewContainer"],
[data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li, [data-testid="stMarkdownContainer"] span,
.stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown span,
[data-testid="stText"], .stText,
p, li, span:not(.live-dot), label, div:not([class*="hero"]):not([class*="stat"]):not([class*="nav"]) > span {
  color: rgba(255, 255, 255, 0.85) !important;
}
h1, h2, h3, h4, h5, h6 { color: #ffffff !important; }
strong, b { color: #ffffff !important; }
code { color: #e2e8f0 !important; }
/* Metric labels */
[data-testid="stMetricLabel"] { color: rgba(255,255,255,0.55) !important; font-size: 0.8rem !important; }
[data-testid="stMetricValue"] { color: #ffffff !important; }
/* Widget labels */
.stSelectbox label, .stSlider label, .stRadio label,
.stTextInput label, .stNumberInput label, .stFileUploader label {
  color: rgba(255,255,255,0.75) !important;
  font-weight: 500 !important;
}
/* Expander text */
.streamlit-expanderContent p, .streamlit-expanderContent li,
.streamlit-expanderContent span { color: rgba(255,255,255,0.8) !important; }
/* Dataframe */
[data-testid="stDataFrame"] { color: #e2e8f0 !important; }
/* Info / alert text */
.stAlert p, .stAlert span { color: rgba(255,255,255,0.9) !important; }
/* Sidebar (if ever shown) */
[data-testid="stSidebar"] * { color: rgba(255,255,255,0.8) !important; }

/* === MARKDOWN OVERRIDES (DOCS) === */
[data-testid="stMarkdownContainer"] pre, [data-testid="stMarkdownContainer"] code {
  background-color: rgba(255,255,255,0.06) !important;
  color: #e2e8f0 !important;
  border-radius: 6px !important;
}
[data-testid="stMarkdownContainer"] blockquote {
  border-left: 4px solid #a78bfa !important;
  background: rgba(255,255,255,0.03) !important;
  color: rgba(255,255,255,0.8) !important;
  padding: 10px 20px !important;
  margin: 1.5rem 0 !important;
}

/* === ANIMATED BACKGROUND === */
[data-testid="stAppViewContainer"] {
  background: #050508 !important;
  background-image:
    radial-gradient(ellipse 80% 50% at 20% -10%, rgba(120,40,200,0.25) 0%, transparent 60%),
    radial-gradient(ellipse 60% 40% at 80% 110%, rgba(20,120,255,0.2) 0%, transparent 60%),
    radial-gradient(ellipse 40% 40% at 60% 50%, rgba(236,72,153,0.08) 0%, transparent 60%) !important;
}

/* === HIDE DEFAULT STREAMLIT CHROME === */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stSidebar"] { display: none !important; }
[data-testid="stToolbar"] { display: none !important; }
.block-container { padding: 0 2rem 3rem 2rem !important; max-width: 1400px !important; }

/* === TOP NAVBAR === */
.top-nav {
  display: flex;
  align-items: center;
  gap: 0.4rem;
  padding: 1rem 0 0.5rem 0;
  flex-wrap: wrap;
}
.nav-logo {
  font-size: 1.4rem;
  font-weight: 900;
  background: linear-gradient(135deg, #a78bfa, #ec4899);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  margin-right: 1.5rem;
  white-space: nowrap;
  letter-spacing: -0.5px;
}
.stButton > button {
  background: rgba(255,255,255,0.05) !important;
  color: rgba(255,255,255,0.6) !important;
  border: 1px solid rgba(255,255,255,0.1) !important;
  border-radius: 999px !important;
  padding: 6px 18px !important;
  font-weight: 500 !important;
  font-size: 0.82rem !important;
  letter-spacing: 0.3px !important;
  transition: all 0.2s ease !important;
  white-space: nowrap !important;
  min-height: 36px !important;
  line-height: 1 !important;
  backdrop-filter: blur(8px) !important;
  box-shadow: none !important;
}
.stButton > button:hover {
  background: rgba(167,139,250,0.15) !important;
  border-color: rgba(167,139,250,0.4) !important;
  color: #a78bfa !important;
  transform: none !important;
  box-shadow: 0 0 16px rgba(167,139,250,0.2) !important;
}
.stButton > button[kind="primary"] {
  background: linear-gradient(135deg, #7c3aed, #a855f7) !important;
  color: white !important;
  border: none !important;
  box-shadow: 0 0 20px rgba(168,85,247,0.5), inset 0 1px 0 rgba(255,255,255,0.2) !important;
}
.stButton > button[kind="primary"]:hover {
  box-shadow: 0 0 30px rgba(168,85,247,0.7) !important;
  color: white !important;
}

/* === HERO BANNER === */
.hero-banner {
  position: relative;
  overflow: hidden;
  background: linear-gradient(135deg, rgba(124,58,237,0.15) 0%, rgba(236,72,153,0.1) 50%, rgba(20,120,255,0.1) 100%);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 24px;
  padding: 2.5rem 3rem;
  margin: 1rem 0 2rem 0;
}
.hero-banner::before {
  content: '';
  position: absolute;
  top: -50%; left: -50%;
  width: 200%; height: 200%;
  background: conic-gradient(from 0deg at 50% 50%, transparent 0%, rgba(167,139,250,0.05) 25%, transparent 50%, rgba(236,72,153,0.05) 75%, transparent 100%);
  animation: spin 20s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }
.hero-title {
  font-size: 2.6rem;
  font-weight: 900;
  line-height: 1.1;
  background: linear-gradient(135deg, #ffffff 0%, #a78bfa 50%, #ec4899 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  position: relative;
  z-index: 1;
  letter-spacing: -1px;
}
.hero-sub {
  font-size: 1rem;
  color: rgba(255,255,255,0.5);
  margin-top: 0.6rem;
  position: relative;
  z-index: 1;
  font-weight: 400;
  max-width: 600px;
}
.hero-badge {
  display: inline-block;
  background: rgba(16,185,129,0.15);
  border: 1px solid rgba(16,185,129,0.3);
  color: #10b981;
  font-size: 0.72rem;
  font-weight: 600;
  padding: 3px 10px;
  border-radius: 999px;
  letter-spacing: 1px;
  text-transform: uppercase;
  margin-bottom: 0.8rem;
  position: relative;
  z-index: 1;
}
.hero-badge::before {
  content: '● ';
  animation: pulse-dot 1.5s ease infinite;
}
@keyframes pulse-dot { 0%,100%{opacity:1} 50%{opacity:0.3} }

/* === STAT CARDS === */
.stat-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 1rem;
  margin: 1.5rem 0;
}
.stat-card {
  background: rgba(255,255,255,0.03);
  border: 1px solid rgba(255,255,255,0.07);
  border-radius: 16px;
  padding: 1.2rem 1.4rem;
  position: relative;
  overflow: hidden;
  transition: all 0.25s ease;
}
.stat-card::after {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 2px;
  background: var(--accent, linear-gradient(90deg, #7c3aed, #a855f7));
  border-radius: 2px 2px 0 0;
}
.stat-card:hover {
  background: rgba(255,255,255,0.06);
  border-color: rgba(255,255,255,0.15);
  transform: translateY(-3px);
  box-shadow: 0 16px 40px rgba(0,0,0,0.4);
}
.stat-label {
  font-size: 0.72rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 1.2px;
  color: rgba(255,255,255,0.4);
  margin-bottom: 0.6rem;
}
.stat-value {
  font-size: 2rem;
  font-weight: 800;
  color: #fff;
  line-height: 1;
  font-family: 'JetBrains Mono', monospace !important;
}
.stat-delta {
  font-size: 0.78rem;
  margin-top: 0.4rem;
  font-weight: 500;
}

/* === AGENT CARDS === */
.agent-card {
  background: rgba(255,255,255,0.025);
  border: 1px solid rgba(255,255,255,0.07);
  border-left: 3px solid;
  border-radius: 12px;
  padding: 1rem 1.2rem;
  margin-bottom: 0.7rem;
  transition: all 0.2s ease;
}
.agent-card:hover { background: rgba(255,255,255,0.05); transform: translateX(4px); }
.agent-card b { font-size: 0.9rem; }
.agent-card code { background: rgba(255,255,255,0.1); padding: 2px 8px; border-radius: 6px; font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; }
.agent-sales      { border-left-color: #10b981; }
.agent-finance    { border-left-color: #3b82f6; }
.agent-compliance { border-left-color: #ef4444; }
.agent-strategy   { border-left-color: #f59e0b; }

/* === ARBITRATION CARD === */
.arb-card {
  background: linear-gradient(135deg, rgba(124,58,237,0.12) 0%, rgba(15,23,42,0.6) 100%);
  border: 1px solid rgba(167,139,250,0.25);
  border-radius: 16px;
  padding: 1.5rem 2rem;
  margin: 1.2rem 0;
  box-shadow: 0 0 40px rgba(167,139,250,0.08);
}
.arb-card b { color: #c4b5fd; }
.final-action {
  display: inline-block;
  background: linear-gradient(135deg, #7c3aed, #ec4899);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  font-size: 1.3rem;
  font-weight: 800;
  letter-spacing: 1px;
  text-transform: uppercase;
  margin: 6px 0;
  font-family: 'JetBrains Mono', monospace;
}

/* === SECTION HEADINGS === */
.section-heading {
  font-size: 0.72rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 2px;
  color: rgba(255,255,255,0.35);
  margin: 2rem 0 1rem 0;
  display: flex;
  align-items: center;
  gap: 0.6rem;
}
.section-heading::after {
  content: '';
  flex: 1;
  height: 1px;
  background: rgba(255,255,255,0.07);
}

/* === GLASS PANEL === */
.glass-panel {
  background: rgba(255,255,255,0.03);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 20px;
  padding: 2rem;
  margin-bottom: 1.5rem;
}

/* === RISK BADGE === */
.risk-high { color: #ef4444; font-weight: 700; }
.risk-med  { color: #f59e0b; font-weight: 700; }
.risk-low  { color: #10b981; font-weight: 700; }

/* === EXPANDERS === */
details summary, .streamlit-expanderHeader {
  background: rgba(255,255,255,0.03) !important;
  border: 1px solid rgba(255,255,255,0.07) !important;
  border-radius: 10px !important;
  font-weight: 600 !important;
  font-size: 0.9rem !important;
  color: rgba(255,255,255,0.75) !important;
  padding: 0.75rem 1rem !important;
}

/* === METRICS === */
[data-testid="stMetricValue"] {
  font-family: 'JetBrains Mono', monospace !important;
  font-weight: 700 !important;
  font-size: 1.8rem !important;
}
[data-testid="metric-container"] {
  background: rgba(255,255,255,0.03);
  border: 1px solid rgba(255,255,255,0.07);
  border-radius: 12px;
  padding: 1rem;
}

/* === DATAFRAME === */
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }

/* === DIVIDER === */
hr { border-color: rgba(255,255,255,0.07) !important; margin: 1.5rem 0 !important; }

/* === CHAT MESSAGES === */
[data-testid="stChatMessage"] {
  background: rgba(255,255,255,0.03) !important;
  border: 1px solid rgba(255,255,255,0.07) !important;
  border-radius: 14px !important;
  margin-bottom: 0.75rem !important;
}

/* === ALERTS === */
.stAlert { border-radius: 12px !important; border: none !important; }
.stSuccess { background: rgba(16,185,129,0.1) !important; border: 1px solid rgba(16,185,129,0.25) !important; }
.stWarning { background: rgba(245,158,11,0.1) !important; border: 1px solid rgba(245,158,11,0.25) !important; }
.stError   { background: rgba(239,68,68,0.1)  !important; border: 1px solid rgba(239,68,68,0.25)  !important; }
.stInfo    { background: rgba(59,130,246,0.1)  !important; border: 1px solid rgba(59,130,246,0.25)  !important; }

/* === FILE UPLOADER === */
[data-testid="stFileUploader"] {
  background: rgba(255,255,255,0.02) !important;
  border: 2px dashed rgba(255,255,255,0.1) !important;
  border-radius: 16px !important;
}

/* === SELECTBOX / RADIO === */
[data-testid="stRadio"] label { font-size: 0.88rem !important; }
.stSelectbox > div > div {
  background: rgba(255,255,255,0.05) !important;
  border: 1px solid rgba(255,255,255,0.1) !important;
  border-radius: 10px !important;
}

/* === SLIDER === */
[data-testid="stSlider"] .stSlider { color: #a78bfa !important; }

/* === SCROLLBAR === */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.15); border-radius: 99px; }

/* === LIVE DOT ANIMATION === */
@keyframes liveDot { 0%,100%{transform:scale(1);opacity:1} 50%{transform:scale(1.5);opacity:0.6} }
.live-dot { display:inline-block; width:8px; height:8px; background:#10b981; border-radius:50%; animation:liveDot 1.5s infinite; margin-right:6px; }
</style>
""", unsafe_allow_html=True)

# ── Imports (after sys.path tweak) ─────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from env import SalesOpsEnvironment
from agents import build_agents
from arbitration import ArbitrationEngine
from memory import ExperienceMemory
from hf_client import HFClient, HF_TOKEN, HF_MODEL, USE_HF, HF_ROUTER_URL
from metrics import MetricsTracker
from train import run_training, run_random_episode, run_greedy_episode, run_multi_agent_episode
import config

# ── Session state init ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_singletons():
    env     = SalesOpsEnvironment()
    client  = HFClient()
    memory  = ExperienceMemory()
    arbiter = ArbitrationEngine(hf_client=client)
    agents  = build_agents(client, memory)
    tracker = MetricsTracker()
    return env, client, memory, arbiter, agents, tracker

env, client, memory, arbiter, agents, tracker = get_singletons()

if "episode_num"     not in st.session_state: st.session_state.episode_num     = 0
if "episode_history" not in st.session_state: st.session_state.episode_history = []
if "trained"         not in st.session_state: st.session_state.trained         = False
if "chat_messages"   not in st.session_state: st.session_state.chat_messages   = [{"role": "assistant", "content": "Hello! I am the Strategy Manager. Ask me anything about the current policy weights or recent decisions!"}]


# ══════════════════════════════════════════════════════════════════════════════
# TOP NAV BAR
# ══════════════════════════════════════════════════════════════════════════════
PAGES = ["🎯 Run", "🔌 Import", "👑 Override", "🏋️ Train", "📊 Metrics", "🧠 Memory", "💬 Chat", "📄 Docs"]
PAGE_MAP = {
    "🎯 Run": "run", "🔌 Import": "import", "👑 Override": "override",
    "🏋️ Train": "train", "📊 Metrics": "metrics", "🧠 Memory": "memory",
    "💬 Chat": "chat", "📄 Docs": "docs"
}

if "current_page" not in st.session_state:
    st.session_state.current_page = PAGES[0]

# Status bar
llm_status = "🟢 LLM LIVE" if (HF_TOKEN and USE_HF and client.status == "LIVE") else "🟡 Heuristic Mode"
ep_count = st.session_state.episode_num
mem_count = len(memory.experiences)

st.markdown(f"""
<div style="display:flex;align-items:center;justify-content:space-between;padding:1.2rem 0 0.4rem 0;border-bottom:1px solid rgba(255,255,255,0.06);margin-bottom:0.8rem;">
  <div style="display:flex;align-items:center;gap:0.6rem;">
    <span style="font-size:1.5rem;font-weight:900;background:linear-gradient(135deg,#a78bfa,#ec4899);-webkit-background-clip:text;-webkit-text-fill-color:transparent;letter-spacing:-0.5px;">⚡ SalesOps Arena</span>
    <span style="font-size:0.68rem;background:rgba(124,58,237,0.2);border:1px solid rgba(124,58,237,0.4);color:#a78bfa;padding:2px 10px;border-radius:999px;font-weight:600;letter-spacing:1px;text-transform:uppercase;">v3.0 Enterprise</span>
  </div>
  <div style="display:flex;align-items:center;gap:1.2rem;font-size:0.78rem;color:rgba(255,255,255,0.4);">
    <span><span class="live-dot"></span>{llm_status}</span>
    <span>📁 {mem_count} memories</span>
    <span>🎬 {ep_count} episodes</span>
  </div>
</div>
""", unsafe_allow_html=True)

# Nav buttons
nav_cols = st.columns(len(PAGES))
for idx, p_name in enumerate(PAGES):
    is_active = (st.session_state.current_page == p_name)
    if nav_cols[idx].button(p_name, type="primary" if is_active else "secondary", use_container_width=True):
        st.session_state.current_page = p_name
        st.rerun()

logical_page = PAGE_MAP[st.session_state.current_page]
st.markdown("<div style='margin-top:1rem'></div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: RUN EPISODE
# ══════════════════════════════════════════════════════════════════════════════
if logical_page == "run":
    # Hero Banner
    st.markdown(f"""
    <div class="hero-banner">
      <div class="hero-badge">Live Simulation</div>
      <div class="hero-title">Multi-Agent SalesOps Arena</div>
      <div class="hero-sub">Three AI agents — Sales, Finance, Compliance — compete, argue, and converge on optimal enterprise decisions via Bellman Q-Learning.</div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("⚡  Run New Episode", use_container_width=True):
        st.session_state.episode_num += 1
        ep_num = st.session_state.episode_num
        with st.spinner("Agents observing environment and making decisions..."):
            result = run_multi_agent_episode(env, agents, arbiter, memory, ep_num, verbose=False)
        tracker.record(
            mode="multi_agent", episode=ep_num,
            total_reward=result["total_reward"], conversions=result["conversions"],
            risk_incidents=result["risk_incidents"], budget_spent=result["budget_spent"],
            action_dist=result["action_dist"], policy_snapshots=memory.all_policy_snapshots(),
            alignment_score=result.get("alignment_score", 0.0),
        )
        memory.save()
        result["experiences"] = [
            exp.model_dump() if hasattr(exp, "model_dump") else exp
            for exp in result.get("experiences", [])
        ]
        st.session_state.episode_history.append(result)
        st.rerun()

    if st.session_state.episode_history:
        result = st.session_state.episode_history[-1]
        exps   = result.get("experiences", [])
        ep_num = st.session_state.episode_num

        reward_color = "#10b981" if result['total_reward'] >= 0 else "#ef4444"
        risk_color   = "#ef4444" if result['risk_incidents'] > 0 else "#10b981"
        acc_pct      = result.get('alignment_score', 0.0) * 100

        st.markdown(f"""
        <div class="stat-grid">
          <div class="stat-card" style="--accent: linear-gradient(90deg,{reward_color},{reward_color}88);">
            <div class="stat-label">Total Reward</div>
            <div class="stat-value" style="color:{reward_color}">{result['total_reward']:+.2f}</div>
            <div class="stat-delta" style="color:{reward_color}">Episode {ep_num}</div>
          </div>
          <div class="stat-card" style="--accent: linear-gradient(90deg,#a78bfa,#ec4899);">
            <div class="stat-label">Conversions</div>
            <div class="stat-value">{result['conversions']}</div>
            <div class="stat-delta" style="color:rgba(255,255,255,0.4)">leads closed</div>
          </div>
          <div class="stat-card" style="--accent: linear-gradient(90deg,{risk_color},{risk_color}88);">
            <div class="stat-label">Risk Incidents</div>
            <div class="stat-value" style="color:{risk_color}">{result['risk_incidents']}</div>
            <div class="stat-delta" style="color:rgba(255,255,255,0.4)">compliance flags</div>
          </div>
          <div class="stat-card" style="--accent: linear-gradient(90deg,#06b6d4,#3b82f6);">
            <div class="stat-label">Oracle Accuracy</div>
            <div class="stat-value" style="color:#06b6d4">{acc_pct:.1f}%</div>
            <div class="stat-delta" style="color:rgba(255,255,255,0.4)">vs ground truth</div>
          </div>
          <div class="stat-card" style="--accent: linear-gradient(90deg,#f59e0b,#f97316);">
            <div class="stat-label">Budget Spent</div>
            <div class="stat-value" style="font-size:1.5rem;color:#f59e0b">${result['budget_spent']:,.0f}</div>
            <div class="stat-delta" style="color:rgba(255,255,255,0.4)">acquisition cost</div>
          </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<div class='section-heading'>Lead-by-Lead Decisions</div>", unsafe_allow_html=True)

        # ── Per-lead detail ───────────────────────────────────────────────
        st.markdown("### Lead-by-Lead Decisions")
        for exp in result.get("experiences", []):
            outcome_color = "🟢" if exp["outcome"] == "positive" else ("🔴" if exp["outcome"] == "negative" else "🟡")
            with st.expander(
                f"{outcome_color} | {exp['lead_id']} | "
                f"Action: {exp['final_action']} | Reward: {exp['reward']:+.3f}"
            ):
                # Lead info
                ss = exp["state_summary"]
                lc1, lc2, lc3 = st.columns(3)
                lc1.metric("Deal Value",   f"${ss['deal_value']:,.0f}")
                lc2.metric("Lead Score",   f"{ss['lead_score']:.2f}")
                lc3.metric("Risk Score",   f"{ss['risk_score']:.2f}")

                # Agent recommendations
                st.markdown("**Agent Recommendations**")
                agent_colors = {
                    "Sales Agent":      ("🟢", "agent-sales"),
                    "Finance Agent":    ("🔵", "agent-finance"),
                    "Compliance Agent": ("🔴", "agent-compliance"),
                    "Strategy Manager": ("🟡", "agent-strategy"),
                }
                for agent_name, action in exp["recommendations"].items():
                    conf = exp["confidences"].get(agent_name, 0)
                    emoji, cls = agent_colors.get(agent_name, ("⚪", "agent-strategy"))
                    st.markdown(
                        f'<div class="agent-card {cls}">'
                        f'<b>{emoji} {agent_name}</b> → '
                        f'<code>{action}</code> '
                        f'<span style="color:#aaa;">(conf: {conf:.2f})</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                # Arbitration result
                st.markdown(
                    f'<div class="arb-card">'
                    f'<b>⚖️ Strategy Manager Decision</b><br>'
                    f'<div class="final-action">{exp["final_action"]}</div><br>'
                    f'<span style="color:#94a3b8; font-size:0.95em; line-height: 1.5;">{exp["explanation"]}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                # Reward breakdown
                bd = exp["reward_breakdown"]
                col_a, col_b = st.columns(2)
                col_a.markdown("**Reward Breakdown**")
                for k, v in bd.items():
                    color = "green" if v >= 0 else "red"
                    col_a.markdown(f"- {k}: `{v:+.4f}`")
                col_b.metric("Global Reward", f"{exp['reward']:+.4f}",
                             delta=f"{'POSITIVE' if exp['reward'] >= 0 else 'NEGATIVE'}")
    else:
        st.info("Click **Run New Episode** to start the simulation.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: CRM DATA IMPORT (Hackathon Real-World Feature)
# ══════════════════════════════════════════════════════════════════════════════
elif logical_page == "import":
    st.markdown("""
    <div class="hero-banner" style="padding:2rem 2.5rem;">
      <div class="hero-badge">Enterprise Integration</div>
      <div class="hero-title" style="font-size:2rem;">🔌 CRM Data Import</div>
      <div class="hero-sub">Upload a CSV export from Salesforce, HubSpot, or Dynamics 365. The Multi-Agent Engine will immediately arbitrate on your real leads.</div>
    </div>
    """, unsafe_allow_html=True)

    import pandas as pd
    import io
    from models import Lead
    import uuid

    # Provide a sample CSV download for the pitch
    sample_csv = "Lead_ID,Company_Size,Industry,Deal_Value,Lead_Score,Urgency,Acquisition_Cost,Risk_Score,Compliance_Flags\n" \
                 "SF-1001,enterprise,FinTech,850000,0.85,0.9,45000,0.82,high_churn_probability\n" \
                 "SF-1002,startup,EdTech,25000,0.4,0.2,12000,0.1,none\n" \
                 "SF-1003,smb,HealthTech,90000,0.7,0.5,15000,0.6,gdpr_concern\n"
    
    st.download_button(
        label="📥 Download Sample CRM Export",
        data=sample_csv,
        file_name="salesforce_leads_export.csv",
        mime="text/csv",
    )
    
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.dataframe(df.head(), use_container_width=True)
            
            if st.button("▶ Run Simulation on CRM Leads", use_container_width=True):
                # Parse DataFrame to Pydantic Leads
                custom_leads = []
                for _, row in df.iterrows():
                    flags = str(row.get("Compliance_Flags", "none")).split("|")
                    if flags == ["none"]: flags = []
                    
                    l = Lead(
                        lead_id=str(row.get("Lead_ID", f"L-{uuid.uuid4().hex[:6].upper()}")),
                        company_size=str(row.get("Company_Size", "smb")),
                        industry=str(row.get("Industry", "SaaS")),
                        deal_value=float(row.get("Deal_Value", 50000)),
                        lead_score=float(row.get("Lead_Score", 0.5)),
                        urgency=float(row.get("Urgency", 0.5)),
                        acquisition_cost=float(row.get("Acquisition_Cost", 5000)),
                        risk_score=float(row.get("Risk_Score", 0.1)),
                        compliance_flags=flags,
                        time_decay=0.02,
                        previous_interactions=3,
                        market_condition="stable",
                        budget_remaining=config.INITIAL_BUDGET
                    )
                    custom_leads.append(l)
                
                st.session_state.episode_num += 1
                ep_num = st.session_state.episode_num

                with st.spinner("Multi-Agent Arbitration Engine is processing imported leads..."):
                    result = run_multi_agent_episode(env, agents, arbiter, memory, ep_num, custom_leads=custom_leads)

                    # Serialize Experience objects
                    result["experiences"] = [
                        exp.model_dump() if hasattr(exp, "model_dump") else exp
                        for exp in result.get("experiences", [])
                    ]
                    st.session_state.episode_history.append(result)
                
                st.success("Successfully processed CRM data! Check the Run Episode tab for the detailed breakdown.")
                st.rerun()
                
        except Exception as e:
            st.error(f"Error processing CSV: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: CEO OVERRIDE (RLHF)
# ══════════════════════════════════════════════════════════════════════════════
elif logical_page == "override":
    st.markdown("""
    <div class="hero-banner" style="padding:2rem 2.5rem;background:linear-gradient(135deg,rgba(239,68,68,0.12),rgba(245,158,11,0.08));border-color:rgba(239,68,68,0.2);">
      <div class="hero-badge" style="background:rgba(239,68,68,0.15);border-color:rgba(239,68,68,0.3);color:#ef4444;">RLHF Active</div>
      <div class="hero-title" style="font-size:2rem;">👑 CEO Override</div>
      <div class="hero-sub">Real-Time Human-in-the-Loop Reinforcement Learning. Your executive decision is injected with a 10x weight multiplier, permanently altering AI policy.</div>
    </div>
    """, unsafe_allow_html=True)

    # Generate a massive high-stakes lead
    st.error("⚠️ **CRITICAL EDGE CASE DETECTED: REQUIRES HUMAN ESCALATION**")
    
    st.markdown(f"""
    <div class="arb-card" style="border-color: #ef4444; background: rgba(239, 68, 68, 0.1);">
        <h3 style="color: #fca5a5;">Deal ID: SF-999-MEGA</h3>
        <b>Deal Value:</b> <span style="color: #10b981; font-size: 1.5em;">$1,500,000</span><br>
        <b>Risk Score:</b> <span style="color: #ef4444; font-size: 1.5em;">0.92</span> (High Regulatory Risk)<br>
        <b>Urgency:</b> 0.85<br>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🤖 Agents are in extreme conflict:")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            '<div class="agent-card agent-sales">'
            '<b>🟢 Sales Agent</b><br>'
            '<i>"We must PURSUE this! It is a massive commission and they want to close today. Do not let this slip!"</i>'
            '</div>', unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            '<div class="agent-card agent-compliance">'
            '<b>🔴 Compliance Agent</b><br>'
            '<i>"Absolutely REJECT. The regulatory risk is over 90%. If this fails, we face multi-million dollar fines."</i>'
            '</div>', unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown("### ⚖️ Make the Executive Decision")
    st.markdown("Your choice will be injected into the AI's core memory with a **10x Executive Weight**, altering all future automated decisions for similar profiles.")

    ac1, ac2, ac3, ac4 = st.columns(4)
    
    def apply_rlhf(action):
        bucket = "high_value_high_risk"
        # Massive manual policy boost
        for agent in ["Sales Agent", "Finance Agent", "Compliance Agent"]:
            memory.update_policies(agent, bucket, action, reward=5.0) # 10x normal reward impact
        st.session_state["rlhf_done"] = action

    if ac1.button("🟢 PURSUE", use_container_width=True): apply_rlhf("pursue_lead")
    if ac2.button("🟡 NURTURE", use_container_width=True): apply_rlhf("nurture")
    if ac3.button("🟠 REQUEST DISCOUNT", use_container_width=True): apply_rlhf("offer_discount")
    if ac4.button("🔴 REJECT", use_container_width=True): apply_rlhf("reject")

    if st.session_state.get("rlhf_done"):
        st.success(f"**Executive Override Accepted!** You chose to `{st.session_state['rlhf_done'].upper()}`.")
        st.info("🧠 The Strategy Manager has successfully absorbed this RLHF feedback. Policy weights for 'High Value, High Risk' deals have been permanently updated across all agents.")
        if st.button("Clear Override"):
            del st.session_state["rlhf_done"]
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: TRAIN & COMPARE
# ══════════════════════════════════════════════════════════════════════════════
elif logical_page == "train":
    st.markdown("""
    <div class="hero-banner" style="padding:2rem 2.5rem;">
      <div class="hero-badge">Benchmark Suite</div>
      <div class="hero-title" style="font-size:2rem;">🏋️ Training &amp; Comparison</div>
      <div class="hero-sub">Run Multi-Agent, Greedy, and Random baselines side-by-side. Watch the Bellman Q-Learning advantage emerge over 30-100 episodes.</div>
    </div>
    """, unsafe_allow_html=True)

    n_eps = st.slider("Episodes per mode", min_value=10, max_value=100, value=30, step=5)
    col1, col2 = st.columns(2)

    with col1:
        if st.button("🚀 Run Full Training", use_container_width=True):
            progress = st.progress(0, text="Starting training...")
            log_box  = st.empty()
            
            log_lines = []
            def update_logs(msg):
                log_lines.append(msg)
                # render as a code block
                log_box.markdown("```text\n" + "\n".join(log_lines) + "\n```")

            with st.spinner(f"Running {n_eps * 3} total episodes across 3 modes..."):
                # Run training synchronously but updating UI
                summaries = run_training(n_episodes=n_eps, progress_callback=update_logs)
                st.session_state.trained = True

            st.success("Training complete!")
            st.rerun()

    with col2:
        if st.button("🔄 Clear Results", use_container_width=True):
            st.session_state.trained = False
            st.rerun()

    # Show charts if trained
    if st.session_state.trained or os.path.exists(config.TRAINING_RESULTS_PATH):
        st.markdown("---")

        # Load results
        try:
            with open(config.TRAINING_RESULTS_PATH) as f:
                results = json.load(f)
            summaries = results.get("summaries", {})

            # KPI comparison table
            st.markdown("### Performance Comparison")
            import pandas as pd
            rows = []
            for mode, s in summaries.items():
                rows.append({
                    "Mode":              mode.replace("_", " ").title(),
                    "Avg Reward":        round(s.get("avg_reward", 0), 4),
                    "Max Reward":        round(s.get("max_reward", 0), 4),
                    "Avg Conversions":   round(s.get("avg_conversions", 0), 2),
                    "Avg Risk Incidents":round(s.get("avg_risk_incidents", 0), 2),
                    "Budget Efficiency": round(s.get("avg_budget_efficiency", 0), 6),
                })
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True)

        except Exception as e:
            st.warning(f"Could not load training results: {e}")

        # Reward curve chart
        if os.path.exists(config.REWARD_CURVE_PATH):
            st.markdown("### Reward Curve")
            st.image(config.REWARD_CURVE_PATH, use_container_width=True)

        # Comparison chart
        if os.path.exists(config.COMPARISON_CHART_PATH):
            st.markdown("### Metric Comparison")
            st.image(config.COMPARISON_CHART_PATH, use_container_width=True)
    else:
        st.info("Click **Run Full Training** to generate comparison results.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: METRICS
# ══════════════════════════════════════════════════════════════════════════════
elif logical_page == "metrics":
    st.markdown("""
    <div class="hero-banner" style="padding:1.5rem 2.5rem;">
      <div class="hero-badge">Real-Time Analytics</div>
      <div class="hero-title" style="font-size:1.8rem;">📊 Live Performance Metrics</div>
    </div>
    """, unsafe_allow_html=True)

    ep_data = tracker.data.get("multi_agent", [])
    mem_stats = memory.summary_stats()

    if ep_data:
        import pandas as pd
        df = pd.DataFrame(ep_data)

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Episodes",     len(ep_data))
        c2.metric("Avg Reward",         f"{df['total_reward'].mean():+.3f}")
        c3.metric("Avg Conversions",    f"{df['conversions'].mean():.2f}")
        c4.metric("Avg Risk Incidents", f"{df['risk_incidents'].mean():.2f}")
        c5.metric("Oracle Accuracy",    f"{df.get('alignment_score', pd.Series([0])).mean() * 100:.1f}%")
        st.markdown("---")

        # Reward over time
        st.markdown("### Reward Over Episodes")
        st.line_chart(df.set_index("episode")["total_reward"])

        # Action distribution
        if "action_dist" in df.columns:
            st.markdown("### Action Distribution (last episode)")
            last_dist = df["action_dist"].iloc[-1]
            if isinstance(last_dist, dict):
                dist_df = pd.DataFrame(
                    list(last_dist.items()), columns=["Action", "Count"]
                ).sort_values("Count", ascending=False)
                st.bar_chart(dist_df.set_index("Action")["Count"])

        # Policy weight evolution
        st.markdown("### Policy Weights (Finance Agent — expensive_lead bucket)")
        pw = memory.all_policy_snapshots()
        if "Finance Agent" in pw:
            bucket_weights = pw["Finance Agent"].get("expensive_lead", {})
            if bucket_weights:
                pw_df = pd.DataFrame(
                    list(bucket_weights.items()), columns=["Action", "Weight"]
                ).sort_values("Weight", ascending=False)
                st.bar_chart(pw_df.set_index("Action")["Weight"])
    else:
        st.info("Run some episodes to see live metrics.")

    # Memory stats
    if mem_stats:
        st.markdown("---")
        st.markdown("<div class='section-heading'>Memory Summary</div>", unsafe_allow_html=True)

        avg_r = mem_stats.get("avg_reward", 0)
        avg_color = "#10b981" if avg_r >= 0 else "#ef4444"

        st.markdown(f"""
        <div class="stat-grid" style="margin-bottom:1rem;">
          <div class="stat-card" style="--accent:linear-gradient(90deg,#a78bfa,#7c3aed);">
            <div class="stat-label">Total Experiences</div>
            <div class="stat-value">{mem_stats.get("total_experiences", 0)}</div>
            <div class="stat-delta" style="color:rgba(255,255,255,0.4)">in replay buffer</div>
          </div>
          <div class="stat-card" style="--accent:linear-gradient(90deg,{avg_color},{avg_color}88);">
            <div class="stat-label">Avg Reward</div>
            <div class="stat-value" style="color:{avg_color}">{avg_r:+.4f}</div>
            <div class="stat-delta" style="color:rgba(255,255,255,0.4)">per decision</div>
          </div>
          <div class="stat-card" style="--accent:linear-gradient(90deg,#10b981,#06b6d4);">
            <div class="stat-label">Max Reward</div>
            <div class="stat-value" style="color:#10b981">{mem_stats.get("max_reward", 0):+.4f}</div>
          </div>
          <div class="stat-card" style="--accent:linear-gradient(90deg,#ef4444,#f97316);">
            <div class="stat-label">Min Reward</div>
            <div class="stat-value" style="color:#ef4444">{mem_stats.get("min_reward", 0):+.4f}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("<div class='section-heading'>Action Distribution</div>", unsafe_allow_html=True)
            action_dist = mem_stats.get("action_dist", {})
            total_acts = sum(action_dist.values()) or 1
            for act, cnt in sorted(action_dist.items(), key=lambda x: -x[1]):
                pct = cnt / total_acts * 100
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:0.8rem;margin-bottom:0.5rem;">
                  <div style="flex:0 0 140px;font-size:0.82rem;color:rgba(255,255,255,0.7);font-family:'JetBrains Mono',monospace;">{act}</div>
                  <div style="flex:1;background:rgba(255,255,255,0.07);border-radius:999px;height:8px;overflow:hidden;">
                    <div style="width:{pct:.0f}%;height:100%;background:linear-gradient(90deg,#7c3aed,#a855f7);border-radius:999px;"></div>
                  </div>
                  <div style="flex:0 0 40px;font-size:0.82rem;color:#a78bfa;font-weight:600;text-align:right;">{cnt}</div>
                </div>
                """, unsafe_allow_html=True)

        with col_b:
            st.markdown("<div class='section-heading'>Outcome Distribution</div>", unsafe_allow_html=True)
            outcome_dist = mem_stats.get("outcome_dist", {})
            outcome_colors = {"positive": "#10b981", "negative": "#ef4444", "neutral": "#f59e0b"}
            total_out = sum(outcome_dist.values()) or 1
            for outcome, cnt in sorted(outcome_dist.items(), key=lambda x: -x[1]):
                pct = cnt / total_out * 100
                col = outcome_colors.get(outcome, "#a78bfa")
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:0.8rem;margin-bottom:0.5rem;">
                  <div style="flex:0 0 80px;font-size:0.82rem;color:{col};font-weight:600;text-transform:capitalize;">{outcome}</div>
                  <div style="flex:1;background:rgba(255,255,255,0.07);border-radius:999px;height:8px;overflow:hidden;">
                    <div style="width:{pct:.0f}%;height:100%;background:{col};border-radius:999px;"></div>
                  </div>
                  <div style="flex:0 0 50px;font-size:0.82rem;color:rgba(255,255,255,0.6);text-align:right;">{pct:.0f}%</div>
                </div>
                """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MEMORY & LOGS
# ══════════════════════════════════════════════════════════════════════════════
elif logical_page == "memory":
    st.markdown("""
    <div class="hero-banner" style="padding:1.5rem 2.5rem;">
      <div class="hero-badge">Q-Table &amp; Replay Buffer</div>
      <div class="hero-title" style="font-size:1.8rem;">🧠 Agent Memory &amp; Decision Logs</div>
    </div>
    """, unsafe_allow_html=True)

    # Environment logs
    st.markdown("### Environment Action Logs")
    logs = env.render_logs(50)
    if logs:
        import pandas as pd
        log_df = pd.DataFrame(logs)
        st.dataframe(log_df, use_container_width=True)
    else:
        st.info("No environment logs yet. Run some episodes first.")

    st.markdown("---")

    # Experience memory
    st.markdown("### Recent Experiences")
    recent_exps = memory.recent(10)
    if recent_exps:
        for exp in reversed(recent_exps):
            outcome_color = "🟢" if exp["outcome"] == "positive" else ("🔴" if exp["outcome"] == "negative" else "🟡")
            with st.expander(
                f"{outcome_color} Ep {exp['episode']} | {exp['lead_id']} | "
                f"{exp['final_action']} | Reward: {exp['reward']:+.3f}"
            ):
                st.markdown(f"**Context Bucket:** `{exp['context_bucket']}`")
                st.markdown(f"**Explanation:** {exp['explanation']}")
                st.markdown("**Agent Votes:**")
                for agent, action in exp["recommendations"].items():
                    st.markdown(f"- {agent}: `{action}`")
                st.markdown(f"**Outcome:** `{exp['outcome']}`")
    else:
        st.info("Memory is empty. Run episodes to populate it.")

    st.markdown("---")

    # Policy weights full view
    st.markdown("### Policy Weights Snapshot (All Agents)")
    pw = memory.all_policy_snapshots()
    if pw:
        agent_sel = st.selectbox("Select Agent", list(pw.keys()))
        bucket_sel = st.selectbox("Select Context Bucket", config.CONTEXT_BUCKETS)
        if agent_sel in pw and bucket_sel in pw[agent_sel]:
            weights = pw[agent_sel][bucket_sel]
            import pandas as pd
            w_df = pd.DataFrame(
                sorted(weights.items(), key=lambda x: -x[1]),
                columns=["Action", "Weight"]
            )
            st.dataframe(w_df, use_container_width=True)
            st.bar_chart(w_df.set_index("Action")["Weight"])
    else:
        st.info("No policy weights yet. Train the system first.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: STRATEGY CHAT (Interactive Explainability)
# ══════════════════════════════════════════════════════════════════════════════
elif logical_page == "chat":
    st.markdown("""
    <div class="hero-banner" style="padding:1.5rem 2.5rem;">
      <div class="hero-badge">LLM Explainability</div>
      <div class="hero-title" style="font-size:1.8rem;">💬 Strategy Manager Chat</div>
      <div class="hero-sub">Ask the AI why it made specific decisions. It reads live memory and policy weights to answer.</div>
    </div>
    """, unsafe_allow_html=True)

    # Render chat history
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if user_query := st.chat_input("E.g., Why are we rejecting EdTech leads?"):
        st.session_state.chat_messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing memory and policy weights..."):
                mem_stats = memory.summary_stats()
                policy_snapshots = memory.all_policy_snapshots()
                
                response_text = client.chat_strategy_manager(
                    message=user_query,
                    memory_summary=mem_stats,
                    policy_snapshots=policy_snapshots
                )
            st.markdown(response_text)
            st.session_state.chat_messages.append({"role": "assistant", "content": response_text})
# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DOCUMENTS
# ══════════════════════════════════════════════════════════════════════════════
elif logical_page == "docs":
    st.markdown("""
    <div class="hero-banner" style="padding:1.5rem 2.5rem;">
      <div class="hero-badge">Knowledge Base</div>
      <div class="hero-title" style="font-size:1.8rem;">📄 Project Documentation</div>
      <div class="hero-sub">Architecture, technical deep dives, and project summaries — all rendered live within the dashboard.</div>
    </div>
    """, unsafe_allow_html=True)

    docs = {
        "Technical Deep Dive (For Judges)": "technical_doc.md",
        "System Architecture (Whiteboard)": "system_architecture.md",
        "Readme (System Architecture)": "README.md",
        "Project Summary": "project_summary.md"
    }

    selected_doc = st.radio("Select Document to View:", list(docs.keys()), horizontal=True)
    st.markdown("---")
    
    file_path = docs[selected_doc]
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Render actual markdown content
        with st.container():
            st.markdown(content)
    else:
        st.error(f"Could not find document: `{file_path}`")
