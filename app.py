import os, json, time, math
import logging
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

logger = logging.getLogger("app")

def llm_complete(messages, model, temperature=0.3, try_json=True):
    """
    Call the Chat Completions API with graceful fallbacks:
    - If the server rejects response_format, retry without it.
    - If the server rejects temperature, retry without it (use provider default).
    """
    kwargs = dict(model=model, messages=messages)

    # Optional: let env pin max_tokens (some providers require it)
    mx = os.getenv("OPENAI_MAX_TOKENS")
    if mx and mx.isdigit():
        kwargs["max_tokens"] = int(mx)

    if try_json:
        kwargs["response_format"] = {"type": "json_object"}
    if temperature is not None:
        kwargs["temperature"] = temperature

    for _ in range(3):  # up to 3 adaptive attempts
        try:
            resp = client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content
        except Exception as e:
            msg = str(e)
            low = msg.lower()

            # If temperature is not accepted, drop it and retry
            if ("temperature" in low and ("unsupported" in low or "does not support" in low)) and "temperature" in kwargs:
                logger.warning("Provider rejected temperature; retrying without it. Error: %s", msg)
                kwargs.pop("temperature", None)
                continue

            # If response_format not accepted, drop it and retry
            if ("response_format" in low or "unrecognized request argument" in low) and "response_format" in kwargs:
                logger.warning("Provider rejected response_format; retrying without it. Error: %s", msg)
                kwargs.pop("response_format", None)
                continue

            # If max_tokens is required, add a safe default and retry once
            if ("max_tokens" in low and "required" in low) and "max_tokens" not in kwargs:
                logger.warning("Provider requires max_tokens; retrying with 1024. Error: %s", msg)
                kwargs["max_tokens"] = 1024
                continue

            # Otherwise, bubble up the error
            raise

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

    # Gather context
    try:
        ctx = gather_context(ticker=ticker, asset_type=req.asset_type, use_search=req.use_search)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Context gathering failed: {e}")

    messages, response_format = build_messages(ctx=ctx, user_question=req.question)

    # Choose model
    model = req.model or OPENAI_MODEL
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not set on the server.")
    if not model:
        raise HTTPException(status_code=500, detail="No model configured. Set OPENAI_MODEL or pass 'model' in the request.")

    # Call LLM with compatibility retry
    try:
        content = llm_complete(messages=messages, model=model, try_json=True)
        try:
            payload = json.loads(content)
        except Exception:
            payload = {
                "ticker": ctx.get("ticker"),
                "entity_name": ctx.get("entity_name"),
                "asset_type": ctx.get("asset_type"),
                "as_of": datetime.now(timezone.utc).isoformat(),
                "summary": content,
                "drivers": {"near_term": {"positives": [], "negatives": []},
                            "medium_term": {"positives": [], "negatives": []},
                            "long_term": {"positives": [], "negatives": []}},
                "timeline_assessment": [],
                "risks_and_mitigants": [],
                "sources_index": ctx.get("sources_index", []),
                "disclaimer": "This output was unstructured text; JSON parse failed. Treat as notes. Not financial advice."
            }
    except Exception as e:
        # Improve the message for common cases
        msg = str(e)
        if "does not exist" in msg and "model" in msg.lower():
            raise HTTPException(status_code=502, detail=f"LLM model not found on server ('{model}'). Set OPENAI_MODEL to a valid model for your OPENAI_API_BASE.")
        if "invalid api key" in msg.lower() or "authentication" in msg.lower():
            raise HTTPException(status_code=502, detail="Authentication failed with the LLM provider. Check OPENAI_API_KEY.")
        raise HTTPException(status_code=502, detail=f"LLM call failed: {msg}")

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

@app.get("/diag/llm")
def diag_llm():
    try:
        content = llm_complete(
            messages=[{"role":"user","content":"Reply with a tiny valid JSON: {\"ok\":true}"}],
            model=OPENAI_MODEL,
            try_json=False  # plain text to minimize surface area
        )
        return {"ok": True, "model": OPENAI_MODEL, "sample": content[:200]}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=502)
