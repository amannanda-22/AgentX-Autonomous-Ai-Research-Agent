"""
Editor Agent — CrewAI-based agent that compiles all subtask summaries into
a single, professional research report with introduction, thematic sections,
conclusion, and numbered citations.

LLM chain: gemini-2.5-flash → gemini-2.5-flash-lite → gpt-4o-mini → fallback report
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _is_rate_limited(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(tok in msg for tok in ("429", "resource_exhausted", "503", "overloaded", "rate limit", "quota"))


# ── LLM factory ───────────────────────────────────────────────────────────────

def _make_crew_llm(model: str, max_tokens: int = 4096):
    """Create a crewai.LLM for the given model string."""
    from crewai import LLM

    gemini_key = os.getenv("GEMINI_API_KEY", "")
    if model.startswith("gemini") and gemini_key and gemini_key != "your_gemini_api_key_here":
        return LLM(
            model=f"gemini/{model}",
            api_key=gemini_key,
            temperature=0.4,
            max_tokens=max_tokens,
        )
    openai_key = os.getenv("OPENAI_API_KEY", "")
    if openai_key and openai_key != "your_openai_api_key_here":
        return LLM(
            model="openai/gpt-4o",
            api_key=openai_key,
            temperature=0.4,
            max_tokens=max_tokens,
        )
    raise RuntimeError("No valid LLM API key available for EditorAgent.")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _build_summary_digest(summaries: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for i, s in enumerate(summaries, 1):
        lines.append(f"## Subtask {i}: {s.get('subtask', 'N/A')}")
        lines.append(f"**Summary:** {s.get('summary', '')}")
        lines.append(f"**Confidence:** {s.get('overall_confidence', 0):.0%}")

        facts = s.get("key_facts", [])
        if facts:
            lines.append("**Key Facts:**")
            for f in facts[:5]:
                conf = f.get("confidence", 0)
                flag = " ⚠️" if conf < 0.6 else ""
                lines.append(f"  - {f.get('fact', '')}{flag}")

        uncertain = s.get("uncertain_claims", [])
        if uncertain:
            lines.append("**Uncertain Claims:**")
            for u in uncertain[:3]:
                lines.append(f"  - ⚠️ WARNING: {u} ⚠️")

        sources = s.get("sources", [])
        if sources:
            lines.append(f"**Sources:** {', '.join(sources[:3])}")

        lines.append("")
    return "\n".join(lines)


def _collect_all_sources(summaries: List[Dict[str, Any]]) -> List[str]:
    seen: set = set()
    sources: List[str] = []
    for s in summaries:
        for url in s.get("sources", []):
            if url and url not in seen:
                seen.add(url)
                sources.append(url)
        for fact in s.get("key_facts", []):
            url = fact.get("source_url", "")
            if url and url not in seen:
                seen.add(url)
                sources.append(url)
    return sources


# ── CrewAI setup ───────────────────────────────────────────────────────────────

def _build_editor_crew(topic: str, digest: str, sources: List[str], model: str):
    from crewai import Agent, Crew, Process, Task

    llm = _make_crew_llm(model)
    date_str = datetime.utcnow().strftime("%B %d, %Y")

    editor = Agent(
        role="Senior Research Editor",
        goal=(
            f"Compile a comprehensive, professional research report on '{topic}' "
            "using the provided subtask summaries."
        ),
        backstory=(
            "You are an award-winning research editor at a top-tier think tank. "
            "Your reports are known for clarity, depth, accurate citations, and "
            "actionable insights. You structure information logically and write "
            "in a professional yet accessible tone."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    sources_text = "\n".join(f"[{i+1}] {url}" for i, url in enumerate(sources))

    task = Task(
        description=(
            f"Write a professional research report on: **{topic}**\n"
            f"Report Date: {date_str}\n\n"
            "---\n"
            "INPUT SUMMARIES:\n"
            f"{digest}\n\n"
            "---\n"
            "AVAILABLE SOURCES:\n"
            f"{sources_text}\n\n"
            "---\n"
            "INSTRUCTIONS:\n"
            "Structure the report using Markdown with these exact sections:\n\n"
            "# Research Report: [topic]\n\n"
            "**Date:** [date] | **Confidence:** [overall %] | **Sources:** [count]\n\n"
            "## Executive Summary\n"
            "(2-3 sentences capturing the most important finding)\n\n"
            "## Introduction\n"
            "(Why this topic matters, scope of this research)\n\n"
            "## [Section for each major subtask theme]\n"
            "(2-4 paragraphs per section, cite sources inline as [N])\n"
            "(Flag uncertain claims with ⚠️)\n\n"
            "## Key Findings\n"
            "(Bullet list of 5-7 most important findings)\n\n"
            "## Conclusion\n"
            "(Synthesis, implications, recommendations)\n\n"
            "## References\n"
            "[1] URL\n"
            "[2] URL\n"
            "...\n\n"
            "---\n"
            "Rules:\n"
            "- Use only the facts provided in the summaries.\n"
            "- Mark uncertain claims with ⚠️.\n"
            "- Write at least 800 words total.\n"
            "- End with a clean References section.\n"
        ),
        expected_output=(
            "A complete, well-structured Markdown research report with all "
            "required sections, inline citations, and a References list."
        ),
        agent=editor,
    )

    return Crew(
        agents=[editor],
        tasks=[task],
        process=Process.sequential,
        verbose=False,
    )


# ── Public API ─────────────────────────────────────────────────────────────────

class EditorAgent:
    """
    Compiles all summaries into one cohesive Markdown research report.
    LLM chain: gemini-2.5-flash → gemini-2.5-flash-lite → gpt-4o → fallback report.
    """

    def run(
        self,
        topic: str,
        summaries: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        logger.info("EditorAgent compiling report for topic: %s", topic)

        digest = _build_summary_digest(summaries)
        sources = _collect_all_sources(summaries)

        avg_confidence = (
            sum(s.get("overall_confidence", 0.5) for s in summaries) / len(summaries)
            if summaries
            else 0.5
        )

        report_text: Optional[str] = None
        model_used: str = "fallback"

        # Try Gemini models first, then OpenAI
        gemini_key = os.getenv("GEMINI_API_KEY", "")
        candidates = []
        if gemini_key and gemini_key != "your_gemini_api_key_here":
            candidates += [
                os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
                "gemini-2.5-flash-lite",
            ]
        openai_key = os.getenv("OPENAI_API_KEY", "")
        if openai_key and openai_key != "your_openai_api_key_here":
            candidates.append("openai-gpt4o")  # sentinel handled in _make_crew_llm

        for model in candidates:
            try:
                actual_model = "gpt-4o" if model == "openai-gpt4o" else model
                crew = _build_editor_crew(topic, digest, sources, actual_model)
                result = crew.kickoff(inputs={"topic": topic})
                text = getattr(result, "raw", None) or str(result) or ""
                if text.strip():
                    report_text = text
                    model_used = actual_model
                    logger.info("EditorAgent: succeeded with %s", actual_model)
                    break
            except Exception as exc:
                if _is_rate_limited(exc):
                    logger.warning("EditorAgent: %s quota exceeded, trying next. (%s)", model, str(exc)[:80])
                    continue
                logger.error("EditorAgent: %s failed: %s", model, exc)
                break

        if not report_text:
            logger.warning("EditorAgent: all LLMs failed — using fallback report")
            report_text = _fallback_report(topic, summaries, sources)
            model_used = "fallback"

        return {
            "topic": topic,
            "report": report_text,
            "word_count": len(report_text.split()),
            "source_count": len(sources),
            "avg_confidence": round(avg_confidence, 2),
            "subtask_count": len(summaries),
            "model_used": model_used,
        }


def _fallback_report(
    topic: str, summaries: List[Dict[str, Any]], sources: List[str]
) -> str:
    """Emergency fallback: build a minimal report without LLM."""
    date_str = datetime.utcnow().strftime("%B %d, %Y")
    lines = [
        f"# Research Report: {topic}\n",
        f"**Date:** {date_str}\n",
        "## Executive Summary\n",
        "This report was compiled using automated research agents with direct search data.\n",
        "## Research Findings\n",
    ]
    for s in summaries:
        lines.append(f"### {s.get('subtask', 'Research Area')}\n")
        lines.append(s.get("summary", "") + "\n")
        for fact in s.get("key_facts", [])[:3]:
            lines.append(f"- {fact.get('fact', '')}")
        lines.append("")

    lines.append("## References\n")
    for i, url in enumerate(sources, 1):
        lines.append(f"[{i}] {url}")

    return "\n".join(lines)
