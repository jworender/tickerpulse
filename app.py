import os, json, time, math
from datetime import datetime, timezone
from typing import Optional, Literal, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from openai import OpenAI

from research import gather_context
from prompts import build_messages

APP_NAME = "Ticker Research API"

# ---- LLM client (OpenAI-compatible) ----
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

if not OPENAI_API_KEY:
    # Render won't start until you set this; local dev can set via .env
    print("WARNING: OPENAI_API_KEY not set. The app will error on first LLM call.")

client = OpenAI(base_url=OPENAI_API_BASE, api_key=OPENAI_API_KEY)

# ---- FastAPI app ----
app = FastAPI(title=APP_NAME)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to your domain.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static UI
app.mount("/static", StaticFiles(directory="static"), name="static")

class ResearchRequest(BaseModel):
    ticker: str = Field(..., description="Ticker symbol, e.g. AAPL or BTC")
    asset_type: Optional[Literal["equity","crypto","auto"]] = "auto"
    question: Optional[str] = Field(None, description="Optional free-form question")
    use_search: Optional[bool] = True   # Uses Tavily if TAVILY_API_KEY is set
    model: Optional[str] = None         # Override OPENAI_MODEL

@app.get("/")
def root():
    return FileResponse("static/index.html")

@app.post("/api/research")
def research(req: ResearchRequest):
    started = time.time()
    ticker = req.ticker.strip().upper()
    if not ticker or len(ticker) > 15:
        raise HTTPException(status_code=400, detail="Invalid ticker.")
    # Gather structured context (prices, quick facts, optional news/search)
    try:
        ctx = gather_context(ticker=ticker, asset_type=req.asset_type, use_search=req.use_search)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Context gathering failed: {e}")

    # Build messages for the LLM
    messages, response_format = build_messages(ctx=ctx, user_question=req.question)

    # Choose model
    model = req.model or OPENAI_MODEL

    # Call LLM (request JSON if supported)
    try:
        completion = client.chat.completions.create(
            model=model,
            temperature=0.3,
            messages=messages,
            response_format=response_format,  # {"type":"json_object"} on OpenAI / compatible
        )
        content = completion.choices[0].message.content
        # Ensure parseable JSON
        try:
            payload = json.loads(content)
        except Exception:
            # Fallback: wrap raw text
            payload = {
                "ticker": ticker,
                "as_of": datetime.now(timezone.utc).isoformat(),
                "summary": content,
                "raw_text": content,
                "disclaimer": "This output could not be parsed into the expected JSON shape. Treat as unstructured notes. Not financial advice."
            }

    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM call failed: {e}")

    payload["_meta"] = {
        "ticker": ticker,
        "timing_ms": int((time.time() - started)*1000),
        "model": model,
        "ctx": {
            "entity_name": ctx.get("entity_name"),
            "asset_type": ctx.get("asset_type"),
            "has_search": bool(ctx.get("sources_index")),
        }
    }
    return JSONResponse(payload)
