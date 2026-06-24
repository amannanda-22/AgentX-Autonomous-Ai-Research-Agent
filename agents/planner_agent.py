"""
Planner Agent — LangGraph ReAct loop that breaks a research topic into
3-5 subtasks and self-reflects until quality >= 0.75 or max iterations hit.

LLM chain: gemini-2.5-flash → gemini-2.5-flash-lite → gpt-4o-mini
"""

from __future__ import annotations

import json
import logging
import os
from typing import Annotated, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

logger = logging.getLogger(__name__)


# ── State ──────────────────────────────────────────────────────────────────────

class PlannerState(TypedDict):
    topic: str
    subtasks: List[str]
    reflection: str
    iteration: int
    max_iterations: int
    plan_quality: float
    messages: Annotated[list, add_messages]


# ── LLM fallback chain ─────────────────────────────────────────────────────────

def _is_rate_limited(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(tok in msg for tok in ("429", "resource_exhausted", "503", "overloaded", "rate limit", "quota"))


def _invoke_with_fallback(messages: list, temperature: float = 0.3, max_tokens: int = 2048):
    """Try gemini-2.5-flash → gemini-2.5-flash-lite → gpt-4o-mini in sequence."""
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    last_exc: Optional[Exception] = None

    if gemini_key and gemini_key != "your_gemini_api_key_here":
        from langchain_google_genai import ChatGoogleGenerativeAI

        for model in [os.getenv("GEMINI_MODEL", "gemini-2.5-flash"), "gemini-2.5-flash-lite"]:
            try:
                llm = ChatGoogleGenerativeAI(
                    model=model,
                    google_api_key=gemini_key,
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    retries=0,  # Surface quota errors immediately; we handle fallback
                )
                response = llm.invoke(messages)
                logger.info("PlannerAgent: succeeded with %s", model)
                return response
            except Exception as exc:
                last_exc = exc
                if _is_rate_limited(exc):
                    logger.warning("PlannerAgent: %s quota exceeded, trying next. (%s)", model, str(exc)[:80])
                    continue
                logger.error("PlannerAgent: %s non-quota error: %s", model, exc)
                break

    openai_key = os.getenv("OPENAI_API_KEY", "")
    if openai_key and openai_key != "your_openai_api_key_here":
        try:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_key, temperature=temperature)
            response = llm.invoke(messages)
            logger.info("PlannerAgent: succeeded with OpenAI gpt-4o-mini")
            return response
        except Exception as exc:
            last_exc = exc
            logger.error("PlannerAgent: OpenAI fallback failed: %s", exc)

    raise RuntimeError(f"All LLMs unavailable for planning: {last_exc}")


# ── Nodes ──────────────────────────────────────────────────────────────────────

def plan_node(state: PlannerState) -> PlannerState:
    """Generate 3-5 research subtasks from the topic."""
    system = (
        "You are an expert research planner. Break the given research topic into "
        "3-5 specific, web-searchable subtasks. Each subtask must be distinct, "
        "actionable, and together they must cover the topic comprehensively.\n\n"
        "Return ONLY valid JSON:\n"
        '{"subtasks": ["...", "...", "..."], "rationale": "...", "complexity": "low|medium|high"}'
    )

    user_content = f"Topic: {state['topic']}"
    if state.get("reflection"):
        user_content += f"\n\nPrevious reflection (use to improve): {state['reflection']}"

    subtasks: List[str] = []
    try:
        response = _invoke_with_fallback(
            [SystemMessage(content=system), HumanMessage(content=user_content)]
        )
        raw = response.content.strip()
        if "```" in raw:
            raw = raw.split("```")[1].lstrip("json").strip()
            raw = raw.split("```")[0].strip()
        parsed = json.loads(raw)
        subtasks = [s.strip() for s in parsed.get("subtasks", []) if s.strip()][:5]
    except json.JSONDecodeError:
        # If response came back but wasn't JSON, parse numbered lines
        if "response" in dir():
            for line in response.content.splitlines():
                clean = line.strip().lstrip("0123456789.-•* ").strip()
                if len(clean) > 15:
                    subtasks.append(clean)
            subtasks = subtasks[:5]
    except Exception as exc:
        logger.error("PlannerAgent plan_node failed: %s", exc)

    if not subtasks:
        subtasks = [
            f"Overview of {state['topic']}",
            f"Recent developments in {state['topic']}",
            f"Key challenges and opportunities in {state['topic']}",
        ]

    return {
        **state,
        "subtasks": subtasks,
        "messages": [
            AIMessage(content=f"Planned {len(subtasks)} subtasks for '{state['topic']}'.")
        ],
    }


def reflect_node(state: PlannerState) -> PlannerState:
    """Evaluate subtask quality; set plan_quality score."""
    system = (
        "You are a senior research quality reviewer. Evaluate the following subtasks "
        "for a research topic. Assess coverage, specificity, and search-friendliness.\n\n"
        "Return ONLY valid JSON:\n"
        '{"quality_score": 0.0-1.0, "approved": true|false, "issues": ["..."], "suggestions": ["..."]}'
    )

    numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(state["subtasks"]))
    user_content = f"Topic: {state['topic']}\n\nSubtasks:\n{numbered}"

    quality: float = 0.75
    reflection = "Plan reviewed."
    try:
        response = _invoke_with_fallback(
            [SystemMessage(content=system), HumanMessage(content=user_content)]
        )
        raw = response.content.strip()
        if "```" in raw:
            raw = raw.split("```")[1].lstrip("json").strip()
            raw = raw.split("```")[0].strip()
        parsed = json.loads(raw)
        quality = float(parsed.get("quality_score", 0.75))
        issues = parsed.get("issues", [])
        suggestions = parsed.get("suggestions", [])
        reflection = (
            f"Quality: {quality:.2f}. "
            + (f"Issues: {'; '.join(issues)}. " if issues else "No issues. ")
            + (f"Suggestions: {'; '.join(suggestions)}" if suggestions else "")
        )
    except Exception as exc:
        logger.warning("PlannerAgent reflect_node fallback (quality=0.75): %s", exc)

    return {
        **state,
        "plan_quality": quality,
        "reflection": reflection,
        "iteration": state.get("iteration", 0) + 1,
        "messages": [
            AIMessage(content=f"Reflection done. Quality score: {quality:.2f}")
        ],
    }


def _should_replan(state: PlannerState) -> str:
    if state.get("plan_quality", 0) >= 0.75:
        return "done"
    if state.get("iteration", 0) >= state.get("max_iterations", 3):
        return "done"
    return "replan"


# ── Graph ──────────────────────────────────────────────────────────────────────

def _build_graph() -> StateGraph:
    g = StateGraph(PlannerState)
    g.add_node("plan", plan_node)
    g.add_node("reflect", reflect_node)
    g.set_entry_point("plan")
    g.add_edge("plan", "reflect")
    g.add_conditional_edges("reflect", _should_replan, {"replan": "plan", "done": END})
    return g.compile()


# ── Public API ─────────────────────────────────────────────────────────────────

class PlannerAgent:
    """
    Orchestrates the LangGraph ReAct planning loop.
    Call .run(topic) → dict with subtasks and metadata.
    """

    def __init__(self) -> None:
        self._graph = _build_graph()

    def run(self, topic: str) -> dict:
        initial: PlannerState = {
            "topic": topic,
            "subtasks": [],
            "reflection": "",
            "iteration": 0,
            "max_iterations": int(os.getenv("PLANNER_MAX_ITER", "1")),
            "plan_quality": 0.0,
            "messages": [HumanMessage(content=f"Research topic: {topic}")],
        }
        result = self._graph.invoke(initial)
        return {
            "topic": topic,
            "subtasks": result["subtasks"],
            "plan_quality": result.get("plan_quality", 0.75),
            "iterations": result.get("iteration", 1),
            "reflection": result.get("reflection", ""),
        }
