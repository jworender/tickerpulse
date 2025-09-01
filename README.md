# Ticker Research (Render demo)

A FastAPI backend + static HTML/JS UI that calls an OpenAI-compatible LLM to analyze a ticker
(stock or crypto) and produce near/medium/long term drivers and a time-bound likelihood assessment.
Optionally enriches with recent web sources (via Tavily).

## Local dev

1) Python 3.10+
2) `python -m venv .venv && source .venv/bin/activate`
3) `pip install -r requirements.txt`
4) Set env:
   - `OPENAI_API_KEY=...`
   - `OPENAI_API_BASE=https://api.openai.com/v1` (or your vLLM/llama server)
   - `OPENAI_MODEL=gpt-4o-mini` (or your model)
   - Optional `TAVILY_API_KEY=...`
5) `uvicorn app:app --reload`

Open http://localhost:8000

## Deploy to Render

1. Push this repo to GitHub.
2. In Render: "New +" → "Web Service" → connect repo.
3. Environment: Python
4. Build command: `pip install -r requirements.txt`
5. Start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`
6. Add environment variables in Render:
   - `OPENAI_API_KEY` (required)
   - `OPENAI_API_BASE` (defaults to OpenAI; set your own if needed)
   - `OPENAI_MODEL` (e.g., `gpt-4o-mini` or your model name)
   - `TAVILY_API_KEY` (optional)
7. Deploy.

## API

POST `/api/research`
```json
{
  "ticker": "AAPL",
  "asset_type": "auto",
  "question": "Focus on Vision Pro and services margin.",
  "use_search": true
}
