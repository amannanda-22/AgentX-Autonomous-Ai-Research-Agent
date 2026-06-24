"""
AgentX — Streamlit Frontend
Premium dark-theme UI with:
  - Typewriter streaming effect
  - Animated agent activity log with pulsing dot
  - Live cost tracker widget
  - Research history sidebar with hover effects
  - Glass-morphism result cards
  - Skeleton loading animation
  - Export as .md button
  - Mobile responsive layout
  - Inter / Space Grotesk fonts
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests
import streamlit as st
from streamlit.components.v1 import html as st_html

# ── Config ─────────────────────────────────────────────────────────────────────

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")
POLL_INTERVAL = 2  # seconds between status polls

st.set_page_config(
    page_title="AgentX — AI Research Agent",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS Injection ──────────────────────────────────────────────────────────────

CUSTOM_CSS = """
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

/* ── Reset & base ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0a0a0f !important;
    color: #e0e0f0 !important;
}

/* ── App container ── */
.stApp {
    background: linear-gradient(135deg, #0a0a0f 0%, #0d0d1a 50%, #0a0f1a 100%) !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: rgba(14,14,26,0.95) !important;
    border-right: 1px solid rgba(108,99,255,0.2) !important;
    backdrop-filter: blur(20px);
}
section[data-testid="stSidebar"] * {
    color: #c0c0e0 !important;
}

/* ── Header ── */
.agentx-header {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.6rem;
    font-weight: 700;
    background: linear-gradient(135deg, #6c63ff 0%, #a855f7 50%, #06b6d4 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-align: center;
    letter-spacing: -0.5px;
    animation: fadeInDown 0.8s ease-out;
}

.agentx-subtitle {
    text-align: center;
    color: #6c7080 !important;
    font-size: 0.95rem;
    margin-bottom: 2rem;
    animation: fadeInDown 1s ease-out;
}

/* ── Animations ── */
@keyframes fadeInDown {
    from { opacity: 0; transform: translateY(-20px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes slideInLeft {
    from { opacity: 0; transform: translateX(-30px); }
    to   { opacity: 1; transform: translateX(0); }
}
@keyframes pulse {
    0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(108,99,255,0.5); }
    50%       { opacity: 0.7; box-shadow: 0 0 0 6px rgba(108,99,255,0); }
}
@keyframes shimmer {
    0%   { background-position: -600px 0; }
    100% { background-position: 600px 0; }
}
@keyframes typewriter {
    from { width: 0; }
    to   { width: 100%; }
}
@keyframes blink-cursor {
    0%, 100% { border-right-color: #6c63ff; }
    50%       { border-right-color: transparent; }
}
@keyframes spin {
    from { transform: rotate(0deg); }
    to   { transform: rotate(360deg); }
}
@keyframes glow {
    0%, 100% { box-shadow: 0 0 5px rgba(108,99,255,0.3); }
    50%       { box-shadow: 0 0 20px rgba(108,99,255,0.8), 0 0 40px rgba(108,99,255,0.4); }
}

/* ── Glass morphism card ── */
.glass-card {
    background: rgba(255,255,255,0.03) !important;
    backdrop-filter: blur(20px);
    border: 1px solid rgba(108,99,255,0.18) !important;
    border-radius: 18px !important;
    padding: 1.5rem !important;
    margin: 0.8rem 0 !important;
    animation: fadeInUp 0.6s ease-out;
    transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
}
.glass-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(108,99,255,0.15) !important;
    border-color: rgba(108,99,255,0.4) !important;
}

/* ── Skeleton loader ── */
.skeleton-line {
    height: 16px;
    border-radius: 8px;
    background: linear-gradient(90deg, #1a1a2e 25%, #252545 50%, #1a1a2e 75%);
    background-size: 600px 100%;
    animation: shimmer 1.5s infinite;
    margin: 10px 0;
}
.skeleton-line.short  { width: 40%; }
.skeleton-line.medium { width: 70%; }
.skeleton-line.long   { width: 95%; }
.skeleton-title {
    height: 24px;
    width: 55%;
    border-radius: 8px;
    background: linear-gradient(90deg, #1a1a2e 25%, #252545 50%, #1a1a2e 75%);
    background-size: 600px 100%;
    animation: shimmer 1.5s infinite;
    margin-bottom: 16px;
}

/* ── Agent log ── */
.agent-log-container {
    max-height: 340px;
    overflow-y: auto;
    padding: 0.5rem;
    scrollbar-width: thin;
    scrollbar-color: #6c63ff #1a1a2e;
}
.log-entry {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    padding: 8px 12px;
    margin: 4px 0;
    border-radius: 10px;
    background: rgba(255,255,255,0.02);
    border-left: 3px solid #6c63ff;
    animation: slideInLeft 0.3s ease-out;
    font-size: 0.85rem;
    font-family: 'Inter', monospace;
}
.log-entry.done    { border-left-color: #10b981; }
.log-entry.error   { border-left-color: #ef4444; background: rgba(239,68,68,0.05); }
.log-entry.warning { border-left-color: #f59e0b; }
.pulse-dot {
    width: 10px; height: 10px;
    border-radius: 50%;
    background: #6c63ff;
    flex-shrink: 0;
    margin-top: 3px;
    animation: pulse 1.5s infinite;
}
.pulse-dot.done    { background: #10b981; animation: none; }
.pulse-dot.error   { background: #ef4444; animation: none; }
.pulse-dot.warning { background: #f59e0b; animation: none; }
.log-agent {
    font-weight: 600;
    color: #a78bfa;
    min-width: 120px;
}
.log-msg { color: #c0c0d8; }
.log-time { color: #4a4a6a; font-size: 0.75rem; margin-left: auto; white-space: nowrap; }

/* ── Cost tracker ── */
.cost-tracker {
    background: rgba(108,99,255,0.07);
    border: 1px solid rgba(108,99,255,0.25);
    border-radius: 14px;
    padding: 1rem 1.2rem;
    animation: glow 3s infinite;
}
.cost-label { font-size: 0.75rem; color: #6c7080; text-transform: uppercase; letter-spacing: 1px; }
.cost-value { font-size: 1.4rem; font-weight: 700; color: #6c63ff; font-family: 'Space Grotesk', sans-serif; }
.cost-tokens { font-size: 0.8rem; color: #8880b0; }

/* ── History item ── */
.history-item {
    padding: 10px 14px;
    border-radius: 10px;
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(108,99,255,0.1);
    margin: 6px 0;
    cursor: pointer;
    transition: all 0.2s ease;
    animation: slideInLeft 0.4s ease-out;
}
.history-item:hover {
    background: rgba(108,99,255,0.1);
    border-color: rgba(108,99,255,0.4);
    transform: translateX(4px);
}
.history-topic { font-weight: 500; color: #c0c0e0; font-size: 0.87rem; }
.history-meta  { font-size: 0.73rem; color: #5a5a7a; margin-top: 3px; }
.status-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
}
.badge-completed { background: rgba(16,185,129,0.15); color: #10b981; border: 1px solid rgba(16,185,129,0.3); }
.badge-running   { background: rgba(108,99,255,0.15); color: #a78bfa; border: 1px solid rgba(108,99,255,0.3); }
.badge-failed    { background: rgba(239,68,68,0.15);  color: #ef4444; border: 1px solid rgba(239,68,68,0.3); }
.badge-pending   { background: rgba(245,158,11,0.15); color: #f59e0b; border: 1px solid rgba(245,158,11,0.3); }

/* ── Input & buttons ── */
.stTextInput input, .stTextArea textarea {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(108,99,255,0.3) !important;
    border-radius: 12px !important;
    color: #e0e0f0 !important;
    font-size: 1rem !important;
    padding: 12px 16px !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: #6c63ff !important;
    box-shadow: 0 0 0 3px rgba(108,99,255,0.2) !important;
}
.stButton > button {
    background: linear-gradient(135deg, #6c63ff 0%, #a855f7 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.7rem 2rem !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    font-family: 'Space Grotesk', sans-serif !important;
    letter-spacing: 0.3px !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 15px rgba(108,99,255,0.35) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(108,99,255,0.5) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── Selectbox & misc ── */
.stSelectbox div[data-baseweb="select"] > div {
    background: rgba(255,255,255,0.04) !important;
    border-color: rgba(108,99,255,0.3) !important;
    border-radius: 10px !important;
    color: #e0e0f0 !important;
}
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3 { color: #c0b8ff !important; }
.stMarkdown a { color: #6c63ff !important; }
hr { border-color: rgba(108,99,255,0.15) !important; }

/* ── Confidence bar ── */
.conf-bar-bg {
    background: rgba(255,255,255,0.06);
    border-radius: 6px;
    height: 8px;
    overflow: hidden;
    width: 100%;
    margin-top: 4px;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 6px;
    background: linear-gradient(90deg, #6c63ff, #10b981);
    transition: width 0.8s ease;
}

/* ── Section divider ── */
.section-divider {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(108,99,255,0.4), transparent);
    margin: 1.5rem 0;
}

/* ── Mobile responsive ── */
@media (max-width: 768px) {
    .agentx-header { font-size: 1.8rem; }
    .glass-card { padding: 1rem !important; }
    .log-agent { min-width: 80px; }
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────

def api_get(endpoint: str, timeout: int = 10) -> Optional[Dict[str, Any]]:
    try:
        r = requests.get(f"{API_BASE}{endpoint}", timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return None


def api_post(endpoint: str, payload: Dict[str, Any], timeout: int = 30) -> Optional[Dict[str, Any]]:
    try:
        r = requests.post(f"{API_BASE}{endpoint}", json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def _badge(status: str) -> str:
    cls = f"badge-{status}"
    return f'<span class="status-badge {cls}">{status}</span>'


def _confidence_bar(score: float) -> str:
    pct = int(score * 100)
    color = "#10b981" if pct >= 75 else "#f59e0b" if pct >= 50 else "#ef4444"
    return (
        f'<div class="conf-bar-bg">'
        f'<div class="conf-bar-fill" style="width:{pct}%;background:{color}"></div>'
        f'</div>'
        f'<small style="color:{color};font-weight:600">{pct}% confidence</small>'
    )


def _skeleton_block() -> str:
    return """
    <div class="glass-card">
      <div class="skeleton-title"></div>
      <div class="skeleton-line long"></div>
      <div class="skeleton-line medium"></div>
      <div class="skeleton-line long"></div>
      <div class="skeleton-line short"></div>
      <div class="skeleton-line medium"></div>
      <div class="skeleton-line long"></div>
    </div>
    """


# ── Typewriter JS component ────────────────────────────────────────────────────

def typewriter_component(text: str, speed_ms: int = 12) -> None:
    escaped = json.dumps(text)
    st_html(
        f"""
        <div id="tw-container" style="
            font-family: 'Inter', sans-serif;
            font-size: 0.92rem;
            line-height: 1.7;
            color: #d0d0f0;
            white-space: pre-wrap;
            word-break: break-word;
            padding: 0;
        "></div>
        <script>
        (function() {{
            const container = document.getElementById('tw-container');
            const text = {escaped};
            let i = 0;
            function type() {{
                if (i < text.length) {{
                    container.textContent += text[i];
                    i++;
                    setTimeout(type, {speed_ms});
                }}
            }}
            type();
        }})();
        </script>
        """,
        height=max(200, min(600, len(text) // 3)),
        scrolling=True,
    )


# ── Log renderer ──────────────────────────────────────────────────────────────

def render_agent_log(log: List[Dict[str, Any]]) -> str:
    entries_html = ""
    for entry in log[-20:]:  # show last 20
        status = entry.get("status", "running")
        agent = entry.get("agent", "Agent")
        msg = entry.get("message", "")
        ts = entry.get("timestamp", "")
        time_str = ts[11:19] if len(ts) >= 19 else ts

        entries_html += f"""
        <div class="log-entry {status}">
            <div class="pulse-dot {status}"></div>
            <span class="log-agent">{agent}</span>
            <span class="log-msg">{msg}</span>
            <span class="log-time">{time_str}</span>
        </div>
        """

    return f"""
    <div style="margin:0.5rem 0">
        <div style="font-size:0.8rem;color:#6c7080;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px">
            ● Agent Activity
        </div>
        <div class="agent-log-container">
            {entries_html if entries_html else '<div class="log-entry"><div class="pulse-dot"></div><span class="log-msg" style="color:#555">Waiting for activity…</span></div>'}
        </div>
    </div>
    """


# ── Cost tracker widget ────────────────────────────────────────────────────────

def render_cost_tracker(cost: Dict[str, Any]) -> str:
    usd = cost.get("usd_cost", 0.0)
    tokens = cost.get("total_tokens", 0)
    model = cost.get("model", "N/A")
    return f"""
    <div class="cost-tracker">
        <div class="cost-label">Live Cost Tracker</div>
        <div class="cost-value">${usd:.6f}</div>
        <div class="cost-tokens">
            {tokens:,} tokens · {cost.get('input_tokens',0):,} in + {cost.get('output_tokens',0):,} out
        </div>
        <div style="font-size:0.72rem;color:#4a4a6a;margin-top:4px">Model: {model}</div>
    </div>
    """


# ── Sidebar: history ──────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.markdown(
            '<div style="font-family:\'Space Grotesk\',sans-serif;font-size:1.3rem;'
            'font-weight:700;color:#a78bfa;padding:0.5rem 0 1rem">🔬 AgentX</div>',
            unsafe_allow_html=True,
        )
        st.markdown("**Research History**", unsafe_allow_html=True)

        hist_data = api_get("/history")
        if hist_data and hist_data.get("history"):
            for item in hist_data["history"]:
                topic = item.get("topic", "Unknown")
                status = item.get("status", "unknown")
                created = item.get("created_at", "")[:10]
                wc = item.get("word_count", 0)
                cost_str = f"${item.get('usd_cost', 0):.4f}"

                col_clicked = st.button(
                    f"📄 {topic[:32]}{'…' if len(topic) > 32 else ''}",
                    key=f"hist_{item['task_id']}",
                    use_container_width=True,
                )
                st.markdown(
                    f'<div class="history-meta">'
                    f'{_badge(status)} &nbsp; {created} &nbsp; {wc:,}w &nbsp; {cost_str}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                if col_clicked:
                    st.session_state["view_task_id"] = item["task_id"]
                    st.rerun()
        else:
            st.markdown(
                '<div style="color:#4a4a6a;font-size:0.85rem;padding:1rem 0">'
                "No research history yet.<br>Start your first search above."
                "</div>",
                unsafe_allow_html=True,
            )

        st.divider()
        health = api_get("/health")
        if health:
            st.markdown(
                f'<div style="font-size:0.75rem;color:#4a4a6a">'
                f'🟢 API connected &nbsp;|&nbsp; {health.get("active_tasks",0)} active tasks'
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="font-size:0.75rem;color:#ef4444">🔴 API not reachable</div>',
                unsafe_allow_html=True,
            )


# ── Main page layout ──────────────────────────────────────────────────────────

def render_main():
    # Header
    st.markdown(
        '<div class="agentx-header">AgentX</div>'
        '<div class="agentx-subtitle">Autonomous AI Research Agent · '
        'LangGraph + CrewAI · Gemini 2.5 Flash</div>',
        unsafe_allow_html=True,
    )

    # Research form
    with st.form("research_form", clear_on_submit=False):
        col_topic, col_depth = st.columns([3, 1])
        with col_topic:
            topic = st.text_input(
                "Research Topic",
                placeholder="e.g. The impact of quantum computing on cryptography",
                label_visibility="collapsed",
            )
        with col_depth:
            depth = st.selectbox(
                "Depth",
                ["medium", "quick", "deep"],
                label_visibility="collapsed",
            )

        col_email, col_btn = st.columns([3, 1])
        with col_email:
            email = st.text_input(
                "Email (optional)",
                placeholder="you@example.com — receive report via Gmail",
                label_visibility="collapsed",
            )
        with col_btn:
            submitted = st.form_submit_button(
                "🚀 Research",
                use_container_width=True,
            )

    if submitted and topic.strip():
        payload = {"topic": topic.strip(), "depth": depth, "email": email or None}
        result = api_post("/research", payload)
        if result:
            st.session_state["active_task_id"] = result["task_id"]
            st.session_state["active_topic"] = topic.strip()
            st.session_state["view_task_id"] = None
            st.rerun()
        else:
            st.error("Failed to start research. Is the API running?")

    elif submitted:
        st.warning("Please enter a research topic.")


# ── Research progress view ────────────────────────────────────────────────────

def render_research_progress(task_id: str):
    task_data = api_get(f"/research/{task_id}")
    cost_data = api_get(f"/cost/{task_id}")

    if not task_data:
        st.error("Could not fetch task data.")
        return

    status = task_data.get("status", "unknown")
    topic = task_data.get("topic", "")

    st.markdown(f"### 🔬 Researching: *{topic}*", unsafe_allow_html=True)
    st.markdown(
        f'<span style="font-size:0.85rem;color:#6c7080">Task ID: {task_id}</span>',
        unsafe_allow_html=True,
    )

    col_log, col_cost = st.columns([2, 1])

    with col_log:
        log_placeholder = st.empty()
        log = task_data.get("log", [])
        log_placeholder.markdown(render_agent_log(log), unsafe_allow_html=True)

    with col_cost:
        cost_placeholder = st.empty()
        cost = cost_data or task_data.get("cost", {})
        cost_placeholder.markdown(render_cost_tracker(cost), unsafe_allow_html=True)

    # Show skeleton while running
    if status in ("pending", "running"):
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.markdown(
            '<div style="font-size:0.85rem;color:#6c63ff;animation:pulse 1.5s infinite">⏳ Research in progress…</div>',
            unsafe_allow_html=True,
        )
        skeleton_ph = st.empty()
        skeleton_ph.markdown(_skeleton_block(), unsafe_allow_html=True)

        # Poll loop
        while status in ("pending", "running"):
            time.sleep(POLL_INTERVAL)
            task_data = api_get(f"/research/{task_id}") or task_data
            cost_data = api_get(f"/cost/{task_id}")
            status = task_data.get("status", "unknown")
            log = task_data.get("log", [])
            log_placeholder.markdown(render_agent_log(log), unsafe_allow_html=True)
            cost = cost_data or task_data.get("cost", {})
            cost_placeholder.markdown(render_cost_tracker(cost), unsafe_allow_html=True)

        skeleton_ph.empty()
        st.rerun()

    elif status == "completed":
        result = task_data.get("result") or {}
        report = result.get("report", "")

        # Stats bar
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Words", f"{result.get('word_count', 0):,}")
        m2.metric("Sources", result.get("source_count", 0))
        m3.metric("Subtasks", result.get("subtask_count", 0))
        avg_conf = result.get("avg_confidence", 0)
        m4.metric("Confidence", f"{avg_conf:.0%}")

        st.markdown(
            render_cost_tracker(task_data.get("cost", {})), unsafe_allow_html=True
        )

        # Report
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.markdown(
            '<div style="font-family:\'Space Grotesk\',sans-serif;font-size:1.1rem;'
            'font-weight:600;color:#a78bfa;margin-bottom:1rem">📋 Research Report</div>',
            unsafe_allow_html=True,
        )

        report_tab, raw_tab = st.tabs(["🎨 Formatted", "📝 Markdown"])

        with report_tab:
            st.markdown(report)

        with raw_tab:
            st.code(report, language="markdown")

        # Export
        st.download_button(
            label="⬇️  Export as .md",
            data=report,
            file_name=f"agentx_report_{topic[:30].replace(' ','_')}_{task_id[:8]}.md",
            mime="text/markdown",
            use_container_width=False,
        )

        # Subtask summaries (expandable)
        summaries = result.get("summaries", [])
        if summaries:
            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
            st.markdown(
                '<div style="font-family:\'Space Grotesk\',sans-serif;font-size:1rem;'
                'font-weight:600;color:#a78bfa;margin-bottom:0.5rem">🔍 Subtask Summaries</div>',
                unsafe_allow_html=True,
            )
            for i, s in enumerate(summaries):
                with st.expander(
                    f"Subtask {i+1}: {s.get('subtask','')[:60]}", expanded=False
                ):
                    conf = s.get("overall_confidence", 0)
                    st.markdown(s.get("summary", ""))
                    st.markdown(_confidence_bar(conf), unsafe_allow_html=True)
                    facts = s.get("key_facts", [])
                    if facts:
                        st.markdown("**Key Facts:**")
                        for f in facts[:5]:
                            flag = " ⚠️" if f.get("confidence", 1) < 0.6 else ""
                            st.markdown(f"- {f.get('fact','')}{flag}")
                    uncertain = s.get("uncertain_claims", [])
                    if uncertain:
                        st.markdown("**Uncertain Claims:**")
                        for u in uncertain:
                            st.warning(f"⚠️ {u}")
                    sources = s.get("sources", [])
                    if sources:
                        st.markdown("**Sources:**")
                        for src in sources[:5]:
                            st.markdown(f"- [{src}]({src})")

    elif status == "failed":
        st.error(f"Research failed: {task_data.get('error', 'Unknown error')}")
        if st.button("Retry"):
            topic = task_data.get("topic", "")
            if topic:
                result = api_post("/research", {"topic": topic})
                if result:
                    st.session_state["active_task_id"] = result["task_id"]
                    st.rerun()


# ── Viewing past task ─────────────────────────────────────────────────────────

def render_past_task(task_id: str):
    st.markdown(f"### 📂 Past Research: `{task_id[:8]}…`")
    task_data = api_get(f"/research/{task_id}")
    if not task_data:
        st.error("Task not found — it may have expired or the API restarted.")
        if st.button("← Back to Research", key="back_notfound"):
            st.session_state["view_task_id"] = None
            st.rerun()
        return

    status = task_data.get("status", "unknown")
    topic = task_data.get("topic", "")
    # Use `or {}` — .get("result", {}) returns None when key exists but value is None
    result = task_data.get("result") or {}
    report = result.get("report", "")

    st.markdown(f"**Topic:** {topic}")

    if status == "failed":
        error_msg = task_data.get("error") or "Unknown error"
        st.error(f"Research failed: {error_msg}")
        log = task_data.get("log", [])
        if log:
            with st.expander("Agent activity log", expanded=False):
                st.markdown(render_agent_log(log), unsafe_allow_html=True)
        col_retry, col_back = st.columns([1, 3])
        with col_retry:
            if st.button("🔄 Retry", key="retry_failed"):
                new_result = api_post("/research", {"topic": topic})
                if new_result:
                    st.session_state["active_task_id"] = new_result["task_id"]
                    st.session_state["view_task_id"] = None
                    st.rerun()

    elif status in ("pending", "running"):
        st.info("This research task is still in progress.")
        if st.button("📡 View Live Progress", key="view_live"):
            st.session_state["active_task_id"] = task_id
            st.session_state["view_task_id"] = None
            st.rerun()

    elif report:
        # Stats
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Words", f"{result.get('word_count', 0):,}")
        m2.metric("Sources", result.get("source_count", 0))
        m3.metric("Subtasks", result.get("subtask_count", 0))
        m4.metric("Confidence", f"{result.get('avg_confidence', 0):.0%}")

        cost = task_data.get("cost", {})
        if cost:
            st.markdown(render_cost_tracker(cost), unsafe_allow_html=True)

        st.markdown(report)
        st.download_button(
            label="⬇️ Export as .md",
            data=report,
            file_name=f"agentx_{topic[:30].replace(' ','_')}_{task_id[:8]}.md",
            mime="text/markdown",
        )

    else:
        st.info("No report available for this task.")

    if st.button("← Back to Research", key="back_past"):
        st.session_state["view_task_id"] = None
        st.rerun()


# ── App entry point ────────────────────────────────────────────────────────────

def main():
    render_sidebar()
    render_main()

    # Route: viewing a past task from history click
    if st.session_state.get("view_task_id"):
        render_past_task(st.session_state["view_task_id"])
        return

    # Route: active task progress
    if st.session_state.get("active_task_id"):
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        render_research_progress(st.session_state["active_task_id"])
        if st.button("🔄 New Research"):
            st.session_state["active_task_id"] = None
            st.session_state["active_topic"] = None
            st.rerun()


if __name__ == "__main__":
    main()
