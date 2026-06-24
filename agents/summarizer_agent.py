"""
Summarizer Agent — Condenses raw search results into structured summaries
with extracted key facts, confidence scores, and flagged uncertain claims.

LLM chain: gemini-2.5-flash → gemini-2.5-flash-lite → gpt-4o-mini
Switches models immediately on quota exhaustion (no long waits).
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)


def _is_rate_limited(exc: BaseException) -> bool:
    """True for quota/rate-limit errors where switching models helps."""
    msg = str(exc).lower()
    return any(tok in msg for tok in ("429", "resource_exhausted", "503", "overloaded", "rate limit", "quota"))


_SYSTEM_PROMPT = """\
You are an expert research summarizer and fact-checker. Your job is to:
1. Read raw web search results for a specific subtask.
2. Extract the most important, accurate facts.
3. Assign a confidence score (0.0-1.0) to the overall summary.
4. Flag any claims that seem uncertain, unverified, or potentially outdated
   by wrapping them in ⚠️ WARNING: ... ⚠️

Return ONLY valid JSON with this exact schema:
{
  "subtask": "...",
  "summary": "3-5 sentence comprehensive summary",
  "key_facts": [
    {"fact": "...", "confidence": 0.0-1.0, "source_url": "..."},
    ...
  ],
  "overall_confidence": 0.0-1.0,
  "uncertain_claims": ["claim 1", "claim 2"],
  "sources": ["url1", "url2"]
}

Rules:
- overall_confidence < 0.6 means the evidence is weak or conflicting.
- List at most 5 key facts.
- Do NOT invent facts not present in the search results.
- Mark any claim older than 2 years as uncertain if recency matters.
- Use ONLY source URLs provided in the search results — do NOT invent URLs.
"""


def _extract_source_urls(search_data: Dict[str, Any]) -> List[str]:
    """Extract all real URLs from search results BEFORE calling LLM."""
    urls: List[str] = []
    seen: set = set()
    for r in search_data.get("results", []):
        url = (r.get("url") or "").strip()
        if url and url not in seen and url.startswith("http"):
            seen.add(url)
            urls.append(url)
    return urls


def _format_search_results(search_data: Dict[str, Any]) -> str:
    lines: List[str] = [f"Subtask: {search_data.get('subtask', 'N/A')}\n"]
    for i, r in enumerate(search_data.get("results", [])[:5], 1):
        lines.append(
            f"[Result {i}]\n"
            f"  Title: {r.get('title', 'N/A')}\n"
            f"  URL: {r.get('url', 'N/A')}\n"
            f"  Content: {r.get('snippet', '')[:800]}\n"
            f"  Relevance: {r.get('relevance_score', 0):.2f}\n"
        )
    if search_data.get("key_finding"):
        lines.append(f"\nKey Finding from Search: {search_data['key_finding']}")
    return "\n".join(lines)


def _parse_summary_response(
    raw: str,
    subtask: str,
    fallback_urls: List[str],
) -> Dict[str, Any]:
    clean = raw.strip()
    if "```" in clean:
        match = re.search(r"```(?:json)?\s*([\s\S]+?)```", clean)
        if match:
            clean = match.group(1).strip()

    try:
        parsed = json.loads(clean)
    except json.JSONDecodeError:
        parsed = {
            "subtask": subtask,
            "summary": clean[:600],
            "key_facts": [],
            "overall_confidence": None,
            "uncertain_claims": ["Full structured response unavailable — raw text returned."],
            "sources": [],
        }

    parsed.setdefault("subtask", subtask)
    parsed.setdefault("summary", clean[:300])
    parsed.setdefault("key_facts", [])
    parsed.setdefault("uncertain_claims", [])

    # Coerce confidence — setdefault() won't override existing 0.0.
    raw_conf = parsed.get("overall_confidence")
    if raw_conf is None or not isinstance(raw_conf, (int, float)):
        parsed["overall_confidence"] = 0.65 if parsed.get("key_facts") else 0.5
    else:
        parsed["overall_confidence"] = max(0.1, min(1.0, float(raw_conf)))

    # Prefer LLM-returned URLs; fall back to Tavily URLs when LLM omits them.
    llm_sources = [
        u for u in (parsed.get("sources") or [])
        if isinstance(u, str) and u.startswith("http")
    ]
    parsed["sources"] = llm_sources if llm_sources else fallback_urls

    # Fix per-fact source_urls that were hallucinated or omitted.
    for fact in parsed.get("key_facts", []):
        fact_url = (fact.get("source_url") or "").strip()
        if not fact_url or not fact_url.startswith("http"):
            fact["source_url"] = fallback_urls[0] if fallback_urls else ""

    return parsed


def _extract_token_usage(response: Any) -> Dict[str, int]:
    """Extract token counts across LangChain/API versions."""
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        meta = response.usage_metadata
        # Newer langchain_google_genai returns usage_metadata as a plain dict
        if isinstance(meta, dict):
            return {
                "input_tokens": int(meta.get("input_tokens", 0)),
                "output_tokens": int(meta.get("output_tokens", 0)),
            }
        # Older versions return an object with attributes
        inp = (
            getattr(meta, "input_tokens", None)
            or getattr(meta, "prompt_token_count", None)
            or getattr(meta, "prompt_tokens", None)
            or 0
        )
        out = (
            getattr(meta, "output_tokens", None)
            or getattr(meta, "candidates_token_count", None)
            or getattr(meta, "completion_tokens", None)
            or 0
        )
        return {"input_tokens": int(inp), "output_tokens": int(out)}

    if hasattr(response, "response_metadata"):
        meta = response.response_metadata or {}
        usage = meta.get("token_usage") or meta.get("usage") or {}
        inp = (
            usage.get("prompt_tokens")
            or usage.get("input_tokens")
            or usage.get("prompt_token_count")
            or 0
        )
        out = (
            usage.get("completion_tokens")
            or usage.get("output_tokens")
            or usage.get("candidates_token_count")
            or 0
        )
        return {"input_tokens": int(inp), "output_tokens": int(out)}

    return {"input_tokens": 0, "output_tokens": 0}


class SummarizerAgent:
    """
    Summarizes a single search result dict into a structured summary.

    LLM chain: gemini-2.5-flash → gemini-2.5-flash-lite → gpt-4o-mini.
    Switches immediately on quota exhaustion — no wasted wait time.
    """

    def run(self, search_data: Dict[str, Any]) -> Dict[str, Any]:
        subtask = search_data.get("subtask", "Unknown subtask")
        logger.info("SummarizerAgent processing: %s", subtask)

        fallback_urls = _extract_source_urls(search_data)
        formatted = _format_search_results(search_data)
        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(
                content=f"Please summarize the following search results:\n\n{formatted}"
            ),
        ]

        response: Optional[Any] = None
        last_exc: Optional[Exception] = None
        model_used: str = "unknown"

        # --- Gemini chain: 2.5-flash → 1.5-flash ---
        gemini_key = os.getenv("GEMINI_API_KEY", "")
        if gemini_key and gemini_key != "your_gemini_api_key_here":
            from langchain_google_genai import ChatGoogleGenerativeAI

            for model in [os.getenv("GEMINI_MODEL", "gemini-2.5-flash"), "gemini-2.5-flash-lite"]:
                try:
                    llm = ChatGoogleGenerativeAI(
                        model=model,
                        google_api_key=gemini_key,
                        temperature=0.2,
                        max_output_tokens=2048,
                        retries=0,  # Surface quota errors immediately; we handle fallback
                    )
                    response = llm.invoke(messages)
                    model_used = model
                    logger.info("SummarizerAgent: succeeded with %s", model)
                    break
                except Exception as exc:
                    last_exc = exc
                    if _is_rate_limited(exc):
                        logger.warning(
                            "SummarizerAgent: %s quota exceeded, trying next model. (%s)",
                            model, str(exc)[:120],
                        )
                        continue
                    # Non-quota error (e.g. bad key, network) — skip remaining Gemini
                    logger.error("SummarizerAgent: %s non-quota error: %s", model, exc)
                    break

        # --- OpenAI fallback ---
        if response is None:
            openai_key = os.getenv("OPENAI_API_KEY", "")
            if openai_key and openai_key != "your_openai_api_key_here":
                try:
                    from langchain_openai import ChatOpenAI

                    llm = ChatOpenAI(
                        model="gpt-4o-mini",
                        api_key=openai_key,
                        temperature=0.2,
                    )
                    response = llm.invoke(messages)
                    model_used = "gpt-4o-mini"
                    logger.info("SummarizerAgent: succeeded with OpenAI gpt-4o-mini")
                except Exception as exc:
                    last_exc = exc
                    logger.error("SummarizerAgent: OpenAI fallback also failed: %s", exc)

        if response is None:
            raise RuntimeError(
                f"All LLMs failed for subtask '{subtask}': {last_exc}"
            )

        usage = _extract_token_usage(response)
        parsed = _parse_summary_response(response.content, subtask, fallback_urls)
        parsed["token_usage"] = usage
        parsed["model_used"] = model_used
        return parsed

    def run_batch(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        summaries = []
        for sr in search_results:
            try:
                summaries.append(self.run(sr))
            except Exception as exc:
                logger.error("SummarizerAgent failed for subtask: %s", exc)
                summaries.append(
                    {
                        "subtask": sr.get("subtask", "Unknown"),
                        "summary": f"Summarization failed: {exc}",
                        "key_facts": [],
                        "overall_confidence": 0.0,
                        "uncertain_claims": [str(exc)],
                        "sources": _extract_source_urls(sr),
                        "token_usage": {},
                        "model_used": "none",
                    }
                )
        return summaries
