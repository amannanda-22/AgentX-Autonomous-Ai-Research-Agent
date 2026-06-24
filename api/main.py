"""
AgentX FastAPI Backend
Endpoints:
  POST /research                — start new research task
  GET  /research/{task_id}      — poll task status + result
  GET  /research/{task_id}/stream — SSE live event stream
  GET  /cost/{task_id}          — token cost breakdown
  GET  /history                 — last 10 research sessions
  GET  /health                  — health check
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import uuid
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional

import httpx
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

load_dotenv()

# ── Remove GOOGLE_API_KEY conflict ─────────────────────────────────────────────
# google-genai SDK prefers GOOGLE_API_KEY over GEMINI_API_KEY when both are set,
# which silently routes all traffic through the wrong quota bucket.
# We pass keys explicitly in every call so the ambient env var must be cleared.
if os.environ.get("GOOGLE_API_KEY") and os.environ.get("GEMINI_API_KEY"):
    os.environ.pop("GOOGLE_API_KEY", None)

# ── LangSmith tracing (optional) ───────────────────────────────────────────────
if os.getenv("LANGCHAIN_API_KEY"):
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_PROJECT", "agentx")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.editor_agent import EditorAgent
from agents.memory_agent import MemoryAgent
from agents.planner_agent import PlannerAgent
from agents.search_agent import SearchAgent
from agents.summarizer_agent import SummarizerAgent

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ── Pricing constants (USD per 1M tokens, June 2026 estimates) ─────────────────
PRICING = {
    "gemini-2.5-flash":      {"input": 0.075, "output": 0.30},
    "gemini-2.5-flash-lite": {"input": 0.01,  "output": 0.04},
    "gpt-4o-mini":           {"input": 0.15,  "output": 0.60},
    "gpt-4o":                {"input": 2.50,  "output": 10.00},
}
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


# ── In-memory task store ───────────────────────────────────────────────────────

class TaskRecord:
    def __init__(self, task_id: str, topic: str, email: Optional[str] = None):
        self.task_id = task_id
        self.topic = topic
        self.email = email
        self.status: str = "pending"          # pending | running | completed | failed
        self.created_at: str = datetime.utcnow().isoformat()
        self.completed_at: Optional[str] = None
        self.result: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None
        self.log: List[Dict[str, Any]] = []   # agent activity log entries
        self.cost: Dict[str, Any] = {
            "model": DEFAULT_MODEL,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "usd_cost": 0.0,
        }
        self._sse_queue: asyncio.Queue = asyncio.Queue()

    def push_log(self, agent: str, message: str, status: str = "running") -> None:
        entry = {
            "agent": agent,
            "message": message,
            "status": status,
            "timestamp": datetime.utcnow().isoformat(),
        }
        self.log.append(entry)
        try:
            self._sse_queue.put_nowait({"type": "log", "data": entry})
        except asyncio.QueueFull:
            pass

    def push_token_usage(self, input_tokens: int, output_tokens: int, model: Optional[str] = None) -> None:
        if model and model not in ("none", "unknown", "fallback"):
            self.cost["model"] = model
        model = self.cost["model"]
        price = PRICING.get(model, PRICING["gemini-2.5-flash"])
        self.cost["input_tokens"] += input_tokens
        self.cost["output_tokens"] += output_tokens
        self.cost["total_tokens"] += input_tokens + output_tokens
        self.cost["usd_cost"] = round(
            self.cost["input_tokens"] / 1_000_000 * price["input"]
            + self.cost["output_tokens"] / 1_000_000 * price["output"],
            6,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "topic": self.topic,
            "status": self.status,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "result": self.result,
            "error": self.error,
            "log": self.log,
            "cost": self.cost,
        }


_tasks: Dict[str, TaskRecord] = {}
_history: Deque[str] = deque(maxlen=50)  # keeps task_ids in order


# ── Research orchestration (runs in background) ────────────────────────────────

async def _run_research(task_id: str) -> None:
    task = _tasks[task_id]
    task.status = "running"
    task.push_log("Orchestrator", f"Starting research on: {task.topic}", "running")

    try:
        loop = asyncio.get_running_loop()

        # 1. Planner
        task.push_log("PlannerAgent", "Generating research plan…", "running")
        try:
            plan = await loop.run_in_executor(
                None, lambda: PlannerAgent().run(task.topic)
            )
            subtasks: List[str] = plan.get("subtasks", [task.topic])
            task.push_log(
                "PlannerAgent",
                f"Plan ready: {len(subtasks)} subtasks (quality {plan.get('plan_quality', 0):.0%})",
                "done",
            )
        except Exception as planner_exc:
            logger.warning("PlannerAgent failed (%s), using fallback subtasks.", planner_exc)
            task.push_log("PlannerAgent", f"Planner error — using fallback plan: {planner_exc}", "warning")
            subtasks = [
                f"Overview of {task.topic}",
                f"Recent developments in {task.topic}",
                f"Key challenges and opportunities in {task.topic}",
            ]
            plan = {"subtasks": subtasks, "plan_quality": 0.5}

        # Past memory context
        task.push_log("MemoryAgent", "Checking past research context…", "running")
        mem_agent = MemoryAgent()
        past_context = await loop.run_in_executor(
            None, lambda: mem_agent.get_context_for_topic(task.topic)
        )
        if past_context:
            task.push_log("MemoryAgent", "Found relevant past research.", "done")
        else:
            task.push_log("MemoryAgent", "No prior context found.", "done")

        # 2. Search
        task.push_log("SearchAgent", f"Searching {len(subtasks)} subtasks…", "running")
        search_agent = SearchAgent()
        search_results: List[Dict[str, Any]] = []
        for i, st in enumerate(subtasks):
            task.push_log("SearchAgent", f"Searching: {st[:60]}…", "running")
            sr = await loop.run_in_executor(
                None, lambda s=st: search_agent.run(s, task.topic)
            )
            search_results.append(sr)
            task.push_log(
                "SearchAgent",
                f"Subtask {i+1}/{len(subtasks)} done — {len(sr.get('results', []))} results",
                "done",
            )

        # 3. Summarizer
        task.push_log("SummarizerAgent", "Summarizing search results…", "running")
        summarizer = SummarizerAgent()
        summaries: List[Dict[str, Any]] = []
        for i, sr in enumerate(search_results):
            task.push_log(
                "SummarizerAgent", f"Summarizing subtask {i+1}/{len(search_results)}…", "running"
            )
            try:
                summary = await loop.run_in_executor(None, lambda s=sr: summarizer.run(s))
            except Exception as sum_exc:
                logger.warning("SummarizerAgent failed for subtask %d: %s", i + 1, sum_exc)
                task.push_log("SummarizerAgent", f"Subtask {i+1} summarization failed — using Tavily fallback.", "warning")
                # Preserve Tavily URLs and snippets so sources > 0 even when LLM is unavailable
                fallback_urls = [
                    r.get("url", "") for r in sr.get("results", [])
                    if r.get("url", "").startswith("http")
                ]
                fallback_facts = [
                    {"fact": r.get("snippet", "")[:200], "confidence": r.get("relevance_score", 0.5), "source_url": r.get("url", "")}
                    for r in sr.get("results", [])[:3] if r.get("snippet")
                ]
                fallback_conf = round(
                    sum(r.get("relevance_score", 0.5) for r in sr.get("results", [])[:3]) / max(1, min(3, len(sr.get("results", [])))),
                    2,
                ) if sr.get("results") else 0.0
                summary = {
                    "subtask": sr.get("subtask", f"Subtask {i+1}"),
                    "summary": sr.get("key_finding", f"Summarization unavailable (LLM quota): {str(sum_exc)[:120]}"),
                    "key_facts": fallback_facts,
                    "overall_confidence": fallback_conf,
                    "uncertain_claims": ["LLM summarization unavailable — confidence derived from Tavily relevance scores."],
                    "sources": fallback_urls,
                    "token_usage": {},
                }
            summaries.append(summary)
            # Accumulate token usage with actual model used
            usage = summary.get("token_usage", {})
            task.push_token_usage(
                usage.get("input_tokens", 0),
                usage.get("output_tokens", 0),
                model=summary.get("model_used"),
            )
        task.push_log("SummarizerAgent", "All summaries complete.", "done")

        # 4. Editor
        task.push_log("EditorAgent", "Compiling final research report…", "running")
        editor = EditorAgent()
        report_data = await loop.run_in_executor(
            None, lambda: editor.run(task.topic, summaries)
        )
        task.push_log(
            "EditorAgent",
            f"Report compiled: {report_data.get('word_count', 0):,} words",
            "done",
        )

        # 5. Memory store
        task.push_log("MemoryAgent", "Storing report to long-term memory…", "running")
        stored = await loop.run_in_executor(
            None,
            lambda: mem_agent.store(
                task_id=task_id,
                topic=task.topic,
                report=report_data.get("report", ""),
                metadata={
                    "avg_confidence": report_data.get("avg_confidence", 0),
                    "word_count": report_data.get("word_count", 0),
                },
            ),
        )
        task.push_log(
            "MemoryAgent",
            "Stored to ChromaDB." if stored else "Memory store failed (non-critical).",
            "done" if stored else "warning",
        )

        # Inject past context into result
        report_with_context = report_data.get("report", "")
        if past_context:
            report_with_context = past_context + "\n\n---\n\n" + report_with_context

        # Finalise task
        task.result = {
            "report": report_with_context,
            "word_count": report_data.get("word_count", 0),
            "avg_confidence": report_data.get("avg_confidence", 0),
            "source_count": report_data.get("source_count", 0),
            "subtask_count": len(summaries),
            "summaries": summaries,
            "plan": plan,
        }
        task.status = "completed"
        task.completed_at = datetime.utcnow().isoformat()
        task.push_log("Orchestrator", "Research complete!", "done")

        # Direct email delivery (independent of n8n)
        recipient = task.email or os.getenv("DEFAULT_RECIPIENT_EMAIL", "")
        if recipient:
            task.push_log("EmailAgent", f"Sending report to {recipient}…", "running")
            try:
                from utils.email_sender import send_report_email

                email_sent = send_report_email(
                    to_email=recipient,
                    topic=task.topic,
                    report=report_with_context,
                    task_id=task_id,
                    word_count=report_data.get("word_count", 0),
                    avg_confidence=report_data.get("avg_confidence", 0.0),
                    cost_usd=task.cost.get("usd_cost", 0.0),
                )
                task.push_log(
                    "EmailAgent",
                    f"Email sent to {recipient}." if email_sent
                    else "Email skipped — configure GMAIL_USER and GMAIL_APP_PASSWORD in .env.",
                    "done" if email_sent else "warning",
                )
            except Exception as email_exc:
                logger.warning("EmailAgent error: %s", email_exc)
                task.push_log("EmailAgent", f"Email error: {email_exc}", "warning")

        # Notify n8n webhook
        await _notify_n8n(task)

    except Exception as exc:
        logger.exception("Research task %s failed", task_id)
        task.status = "failed"
        task.error = str(exc)
        task.push_log("Orchestrator", f"Research failed: {exc}", "error")
        await _notify_n8n(task)
    finally:
        try:
            task._sse_queue.put_nowait({"type": "done", "status": task.status})
        except asyncio.QueueFull:
            pass


# ── n8n webhook notification ───────────────────────────────────────────────────

async def _notify_n8n(task: TaskRecord) -> None:
    webhook_url = os.getenv("N8N_WEBHOOK_URL", "")
    if not webhook_url:
        logger.info("N8N_WEBHOOK_URL not set — skipping notification.")
        return

    payload = {
        "task_id": task.task_id,
        "topic": task.topic,
        "status": task.status,
        "email": task.email or os.getenv("DEFAULT_RECIPIENT_EMAIL", ""),
        "report_snippet": (
            (task.result or {}).get("report", "")[:500] if task.status == "completed"
            else f"Research failed: {task.error}"
        ),
        "word_count": (task.result or {}).get("word_count", 0),
        "avg_confidence": (task.result or {}).get("avg_confidence", 0),
        "cost_usd": task.cost.get("usd_cost", 0),
    }

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(webhook_url, json=payload)
            logger.info("n8n notified (status %d) for task %s", resp.status_code, task.task_id)
    except Exception as exc:
        logger.warning("n8n notification failed: %s", exc)


# ── FastAPI app ────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("AgentX API starting up…")
    yield
    logger.info("AgentX API shutting down…")


app = FastAPI(
    title="AgentX — Autonomous AI Research Agent",
    description="Multi-agent research pipeline powered by LangGraph + CrewAI",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic models ────────────────────────────────────────────────────────────

class ResearchRequest(BaseModel):
    topic: str = Field(..., min_length=3, max_length=500, description="Research topic")
    email: Optional[str] = Field(None, description="Email to receive the report")
    depth: Optional[str] = Field("medium", description="Research depth: quick|medium|deep")


class ResearchStartResponse(BaseModel):
    task_id: str
    status: str
    message: str
    estimated_seconds: int


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "active_tasks": sum(1 for t in _tasks.values() if t.status == "running"),
        "total_tasks": len(_tasks),
        "version": "1.0.0",
    }


@app.post("/research", response_model=ResearchStartResponse)
async def start_research(
    request: ResearchRequest,
    background_tasks: BackgroundTasks,
):
    task_id = str(uuid.uuid4())
    task = TaskRecord(task_id=task_id, topic=request.topic, email=request.email)
    _tasks[task_id] = task
    _history.appendleft(task_id)

    background_tasks.add_task(_run_research, task_id)

    depth_time = {"quick": 60, "medium": 180, "deep": 360}
    return ResearchStartResponse(
        task_id=task_id,
        status="pending",
        message=f"Research started on: {request.topic}",
        estimated_seconds=depth_time.get(request.depth or "medium", 180),
    )


@app.get("/research/{task_id}")
async def get_research(task_id: str):
    task = _tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    return task.to_dict()


@app.get("/research/{task_id}/stream")
async def stream_research(task_id: str):
    """Server-Sent Events stream for real-time agent activity."""
    task = _tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    async def event_generator():
        # Replay existing log entries
        for entry in task.log:
            yield f"data: {json.dumps({'type': 'log', 'data': entry})}\n\n"

        if task.status in ("completed", "failed"):
            yield f"data: {json.dumps({'type': 'done', 'status': task.status})}\n\n"
            return

        # Live events
        while True:
            try:
                event = await asyncio.wait_for(task._sse_queue.get(), timeout=30)
                yield f"data: {json.dumps(event)}\n\n"
                if event.get("type") == "done":
                    break
            except asyncio.TimeoutError:
                yield "data: {\"type\": \"heartbeat\"}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/cost/{task_id}")
async def get_cost(task_id: str):
    task = _tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    model = task.cost["model"]
    price = PRICING.get(model, PRICING["gemini-2.5-flash"])

    return {
        "task_id": task_id,
        "topic": task.topic,
        "model": model,
        "input_tokens": task.cost["input_tokens"],
        "output_tokens": task.cost["output_tokens"],
        "total_tokens": task.cost["total_tokens"],
        "usd_cost": task.cost["usd_cost"],
        "price_per_1m_input": price["input"],
        "price_per_1m_output": price["output"],
        "status": task.status,
    }


@app.get("/history")
async def get_history():
    results = []
    for tid in list(_history)[:10]:
        task = _tasks.get(tid)
        if task:
            results.append(
                {
                    "task_id": task.task_id,
                    "topic": task.topic,
                    "status": task.status,
                    "created_at": task.created_at,
                    "completed_at": task.completed_at,
                    "word_count": (task.result or {}).get("word_count", 0),
                    "usd_cost": task.cost.get("usd_cost", 0),
                }
            )
    return {"history": results, "total": len(results)}


@app.delete("/research/{task_id}")
async def delete_task(task_id: str):
    if task_id not in _tasks:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    del _tasks[task_id]
    return {"message": f"Task {task_id} deleted"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=int(os.getenv("API_PORT", "8000")),
        reload=os.getenv("DEBUG", "false").lower() == "true",
        log_level="info",
    )
