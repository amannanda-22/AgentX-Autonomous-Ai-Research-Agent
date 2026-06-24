# AgentX — Autonomous AI Research Agent

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   █████╗  ██████╗ ███████╗███╗   ██╗████████╗██╗  ██╗                       ║
║  ██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝╚██╗██╔╝                       ║
║  ███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║    ╚███╔╝                        ║
║  ██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║    ██╔██╗                        ║
║  ██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║   ██╔╝ ██╗                       ║
║  ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝   ╚═╝  ╚═╝                       ║
║                                                                              ║
║            Autonomous AI Research Agent  v1.0.0                              ║
║       LangGraph + CrewAI · Gemini 2.5 Flash · ChromaDB · n8n                ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

## Architecture

```
                        ┌─────────────────────────────────────────────────┐
                        │              Streamlit Frontend (8501)           │
                        │   Dark theme · Typewriter · Cost tracker         │
                        └──────────────────┬──────────────────────────────┘
                                           │ HTTP REST
                        ┌──────────────────▼──────────────────────────────┐
                        │           FastAPI Backend (8000)                 │
                        │   POST /research · GET /stream · /cost · /history│
                        └──┬──────────────────────────────────────────────┘
                           │ Background async task (LangGraph)
          ┌────────────────▼──────────────────────────────────────────────┐
          │                    Research Pipeline                           │
          │                                                                │
          │  ┌─────────────┐   ┌──────────────┐   ┌──────────────────┐   │
          │  │PlannerAgent │──▶│ Search Agent │──▶│SummarizerAgent   │   │
          │  │ (LangGraph) │   │  (CrewAI)    │   │(Gemini 2.5 Flash)│   │
          │  │ ReAct loop  │   │Tavily+Apify  │   │confidence score  │   │
          │  └─────────────┘   └──────────────┘   └────────┬─────────┘   │
          │                                                  │             │
          │  ┌─────────────┐   ┌──────────────────────────┐│             │
          │  │MemoryAgent  │◀──│       EditorAgent        ││             │
          │  │ (ChromaDB)  │   │    (CrewAI — compile)    ││             │
          │  │ vector store│   │  Professional MD report   │◀────────────┘
          │  └─────────────┘   └──────────────────────────┘
          │         │
          │ On complete: POST to n8n webhook
          └──────────────────────────────────────────────────────────────┘
                     │
          ┌──────────▼─────────────────────────────────────────────────────┐
          │                  n8n Workflow (5678)                            │
          │  Webhook → Check status → Poll API → Gmail send                 │
          │  Error branch → Gmail error alert                                │
          └──────────────────────────────────────────────────────────────── ┘
```

## Agents

| Agent | Framework | Role |
|-------|-----------|------|
| **PlannerAgent** | LangGraph (ReAct) | Breaks topic into 3-5 subtasks with self-reflection quality loop |
| **SearchAgent** | CrewAI | Searches web (Tavily) + deep scrapes (Apify), filters top 5 results |
| **SummarizerAgent** | LangChain | Extracts key facts + confidence scores, flags uncertain claims |
| **EditorAgent** | CrewAI | Compiles professional Markdown report with citations |
| **MemoryAgent** | ChromaDB | Stores sessions, semantic retrieval of past research context |

## Quick Start

### Prerequisites
- Python 3.11+
- Docker + Docker Compose
- API keys (see below)

### 1. Clone & Configure
```bash
cd agentx
cp .env.example .env
# Edit .env with your API keys
```

### 2. Run with Docker
```bash
docker-compose up --build
```
Services start at:
- **Frontend**: http://localhost:8501
- **API**: http://localhost:8000
- **n8n**: http://localhost:5678

### 3. Run Locally (Development)
```bash
python -m venv venv
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate

pip install -r requirements.txt

# Terminal 1 — API
python -m uvicorn api.main:app --reload --port 8000

# Terminal 2 — Frontend
streamlit run frontend/app.py
```

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/research` | Start a research task |
| `GET` | `/research/{task_id}` | Poll task status + full result |
| `GET` | `/research/{task_id}/stream` | SSE live event stream |
| `GET` | `/cost/{task_id}` | Token usage + USD cost breakdown |
| `GET` | `/history` | Last 10 research sessions |
| `GET` | `/health` | API health check |

### POST /research

```json
{
  "topic": "The impact of quantum computing on cryptography",
  "email": "you@example.com",
  "depth": "medium"
}
```

Response:
```json
{
  "task_id": "3f8a1b2c-...",
  "status": "pending",
  "message": "Research started on: The impact...",
  "estimated_seconds": 180
}
```

## n8n Workflow Setup

1. Open n8n at http://localhost:5678 (admin / agentx2026)
2. Go to **Workflows → Import from file**
3. Import `workflows/n8n_workflow.json`
4. In the **Gmail** nodes, reconnect the **Gmail OAuth2** credential:
   - Click either Gmail node → Credentials → Create new → follow OAuth flow
5. Update the `N8N_WEBHOOK_URL` in your `.env` to the webhook URL shown in the "Webhook Trigger" node
6. **Activate** the workflow (toggle in top right)

## Required API Keys

You must obtain these keys manually:

| Key | Service | Where to Get | Free Tier |
|-----|---------|-------------|-----------|
| `GEMINI_API_KEY` | Google Gemini | https://aistudio.google.com/app/apikey | Yes (60 rpm) |
| `OPENAI_API_KEY` | OpenAI (fallback) | https://platform.openai.com/api-keys | Pay-per-use |
| `TAVILY_API_KEY` | Tavily Search | https://app.tavily.com/ | 1000 req/month |
| `APIFY_API_TOKEN` | Apify Scraping | https://console.apify.com/account/integrations | $5 free credit |
| `LANGCHAIN_API_KEY` | LangSmith tracing | https://smith.langchain.com/ | Free tier |

> **Apify is optional.** If `APIFY_API_TOKEN` is blank, deep scraping is skipped gracefully — Tavily results are used alone.

> **LangSmith is optional.** Set `LANGCHAIN_TRACING_V2=false` to disable.

## Manual Steps After Download

1. **Copy env file**: `cp .env.example .env` and fill in all API keys
2. **Gmail OAuth in n8n**: Connect your Gmail account via OAuth2 inside the n8n UI (cannot be automated)
3. **Gemini model**: Verify the model name in `.env` — as of June 2026, `gemini-2.5-flash` is current. Check https://ai.google.dev/models for the latest
4. **n8n webhook URL**: After importing the workflow, copy the Webhook URL from the trigger node and update `N8N_WEBHOOK_URL` in `.env`
5. **n8n encryption key**: Change `N8N_ENCRYPTION_KEY` in `.env` to a random 32-character string for security
6. **Activate n8n workflow**: Toggle the workflow to **Active** in the n8n UI

## Project Structure

```
agentx/
├── agents/
│   ├── __init__.py
│   ├── planner_agent.py      ← LangGraph ReAct loop
│   ├── search_agent.py       ← CrewAI + Tavily + Apify
│   ├── summarizer_agent.py   ← Gemini/OpenAI summarizer
│   ├── editor_agent.py       ← CrewAI report compiler
│   └── memory_agent.py       ← ChromaDB wrapper
├── api/
│   ├── __init__.py
│   └── main.py               ← FastAPI app + orchestration
├── memory/
│   ├── __init__.py
│   └── memory_manager.py     ← ChromaDB singleton
├── frontend/
│   └── app.py                ← Streamlit premium UI
├── workflows/
│   └── n8n_workflow.json     ← Import into n8n
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── .env.example
└── README.md
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Orchestration | LangGraph 0.4.x + CrewAI 0.130.x |
| Tool calling / Memory | LangChain 0.3.x |
| Primary LLM | Gemini 2.5 Flash |
| Fallback LLM | OpenAI GPT-4o-mini / GPT-4o |
| Web Search | Tavily API |
| Deep Scraping | Apify (web-scraper actor) |
| Vector Memory | ChromaDB 0.6.x |
| Embeddings | OpenAI text-embedding-3-small (fallback: sentence-transformers) |
| API Backend | FastAPI 0.115.x + Uvicorn |
| Frontend | Streamlit 1.45.x |
| Workflow Automation | n8n latest |
| Email Delivery | Gmail via n8n OAuth2 |
| Observability | LangSmith |
| Containerization | Docker + Docker Compose |
| Python | 3.11 |

## Environment Variables Reference

See `.env.example` for the full list with descriptions.

## Troubleshooting

**API not reachable from frontend**
- Local dev: ensure `API_BASE_URL=http://localhost:8000`
- Docker: ensure `API_BASE_URL=http://api:8000`

**ChromaDB errors on startup**
- Ensure `CHROMA_PERSIST_DIR` directory is writable
- Delete `./chroma_data` to reset the vector store

**Gemini 429 rate limit**
- You've hit the free tier RPM limit. Wait 60s or upgrade your Gemini plan.
- The fallback to OpenAI activates automatically.

**n8n Gmail credential error**
- Reconnect the Gmail OAuth2 credential in the n8n UI.
- Ensure your Google account has Gmail API enabled (it's on by default).

**Research task stuck in "running"**
- Check API logs: `docker-compose logs api`
- Common cause: missing/invalid API keys in `.env`

## License

MIT — feel free to use, modify, and distribute.
