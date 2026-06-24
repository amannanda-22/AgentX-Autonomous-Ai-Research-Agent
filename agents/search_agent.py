"""
Search Agent — Direct Tavily search guarantees real URLs, then uses a
lightweight CrewAI task for the key_finding sentence.  Apify is optional
enrichment for the top result.

LLM chain for key_finding: gemini-2.5-flash → gemini-2.5-flash-lite → Tavily snippet
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _is_rate_limited(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(tok in msg for tok in ("429", "resource_exhausted", "503", "overloaded", "rate limit", "quota"))


# ── LLM factory ───────────────────────────────────────────────────────────────

def _make_crew_llm(model: str):
    """Create a crewai.LLM for the given model name."""
    from crewai import LLM

    gemini_key = os.getenv("GEMINI_API_KEY", "")
    if model.startswith("gemini") and gemini_key and gemini_key != "your_gemini_api_key_here":
        return LLM(
            model=f"gemini/{model}",
            api_key=gemini_key,
            temperature=0.1,
            max_tokens=512,
        )
    openai_key = os.getenv("OPENAI_API_KEY", "")
    if openai_key and openai_key != "your_openai_api_key_here":
        return LLM(
            model="openai/gpt-4o-mini",
            api_key=openai_key,
            temperature=0.1,
            max_tokens=512,
        )
    raise RuntimeError("No LLM key available.")


# ── Direct Tavily search ───────────────────────────────────────────────────────

def _direct_tavily_search(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Call Tavily directly and return structured results with real URLs preserved.
    Bypasses CrewAI tool string formatting which loses URL data.
    """
    from tavily import TavilyClient

    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    response = client.search(
        query=query,
        max_results=max_results,
        search_depth="advanced",
        include_answer=True,
        include_raw_content=False,
    )

    results: List[Dict[str, Any]] = []
    for r in response.get("results", [])[:max_results]:
        url = (r.get("url") or "").strip()
        if not url:
            continue
        results.append(
            {
                "title": r.get("title", ""),
                "url": url,
                "snippet": (r.get("content") or "")[:800],
                "relevance_score": float(r.get("score", 0.5)),
                "source_type": "news" if "news" in url.lower() else "web",
            }
        )

    return results


# ── Key finding via CrewAI ─────────────────────────────────────────────────────

def _get_key_finding(
    subtask: str, topic: str, results: List[Dict[str, Any]]
) -> str:
    """
    Extract the key finding from Tavily results.

    Uses the highest-relevance Tavily snippet by default to preserve LLM
    quota for the Summarizer (where it provides far more value).
    Set SEARCH_LLM_KEY_FINDING=true in .env to enable LLM-based generation.
    """
    if not results:
        return f"No results found for: {subtask}"

    best = max(results, key=lambda r: r.get("relevance_score", 0))
    snippet_fallback = (best["snippet"] or results[0]["snippet"])[:300]

    if os.getenv("SEARCH_LLM_KEY_FINDING", "false").lower() != "true":
        return snippet_fallback

    # ── Optional LLM path (only if SEARCH_LLM_KEY_FINDING=true) ──────────────
    digest = "\n".join(
        f"- {r['title']}: {r['snippet'][:200]}" for r in results[:3]
    )
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    models_to_try: List[str] = []
    if gemini_key and gemini_key != "your_gemini_api_key_here":
        models_to_try += [
            os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            "gemini-2.5-flash-lite",
        ]

    for model in models_to_try:
        try:
            from crewai import Agent, Crew, Process, Task

            llm = _make_crew_llm(model)
            analyst = Agent(
                role="Research Analyst",
                goal="Extract the single most important finding from search results",
                backstory="You are a concise research analyst who distills information into one clear, factual sentence.",
                llm=llm,
                verbose=False,
                allow_delegation=False,
            )
            task = Task(
                description=(
                    f"Based on these search results about '{subtask}' (topic: '{topic}'):\n\n{digest}\n\n"
                    "Write ONE sentence summarising the single most important finding. Be specific and factual."
                ),
                expected_output="One sentence summarising the key finding.",
                agent=analyst,
            )
            result = Crew(agents=[analyst], tasks=[task], process=Process.sequential, verbose=False).kickoff()
            raw = getattr(result, "raw", None) or str(result) or ""
            sentence = raw.strip().split("\n")[0].strip()
            if sentence:
                return sentence[:300]
        except Exception as exc:
            if _is_rate_limited(exc):
                logger.warning("SearchAgent key_finding: %s quota exceeded, trying next.", model)
                continue
            logger.warning("SearchAgent key_finding: %s failed (%s) — using snippet.", model, exc)
            break

    return snippet_fallback


# ── Optional Apify enrichment ──────────────────────────────────────────────────

def _apify_scrape(url: str) -> Optional[str]:
    """Deep-scrape a URL with Apify. Returns None if disabled or failed."""
    apify_token = os.getenv("APIFY_API_TOKEN", "")
    if not apify_token or apify_token == "your_apify_api_token_here":
        return None

    try:
        from apify_client import ApifyClient

        client = ApifyClient(apify_token)
        run_input = {
            "startUrls": [{"url": url}],
            "maxCrawlPages": 1,
            "maxCrawlDepth": 0,
            "pageFunction": (
                "async function pageFunction(context) {"
                "  const { page, request } = context;"
                "  const title = await page.title();"
                "  const text = await page.evaluate(() => document.body.innerText);"
                "  return { url: request.url, title, text: text.slice(0, 2000) };"
                "}"
            ),
        }
        run = client.actor("apify/web-scraper").call(run_input=run_input)
        items = list(client.dataset(run["defaultDatasetId"]).iterate_items())
        if items:
            return (items[0].get("text") or "")[:1500]
    except Exception as exc:
        logger.warning("Apify scrape failed for %s: %s", url, exc)

    return None


# ── Public API ─────────────────────────────────────────────────────────────────

class SearchAgent:
    """
    For each subtask:
    1. Calls Tavily directly → real URLs guaranteed in structured output
    2. Uses a lightweight CrewAI task to generate key_finding (falls back to snippet)
    3. Optionally enriches top result snippet with Apify
    """

    def run(self, subtask: str, topic: str) -> Dict[str, Any]:
        logger.info("SearchAgent running subtask: %s", subtask)

        try:
            results = _direct_tavily_search(subtask, max_results=5)
        except Exception as exc:
            logger.error("Tavily search failed for '%s': %s", subtask, exc)
            return {
                "subtask": subtask,
                "results": [],
                "key_finding": f"Search failed: {exc}",
                "error": str(exc),
            }

        if not results:
            return {
                "subtask": subtask,
                "results": [],
                "key_finding": f"No results found for: {subtask}",
            }

        top_url = results[0]["url"]
        enriched = _apify_scrape(top_url)
        if enriched:
            results[0]["snippet"] = enriched[:800]

        key_finding = _get_key_finding(subtask, topic, results)

        return {
            "subtask": subtask,
            "results": results[:5],
            "key_finding": key_finding,
        }

    def run_batch(self, subtasks: List[str], topic: str) -> List[Dict[str, Any]]:
        """Run search for each subtask sequentially."""
        return [self.run(st, topic) for st in subtasks]
