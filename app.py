import os, json, time, math, httpx
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, Literal, Dict, Any, List
from fastapi import FastAPI, HTTPException, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from openai import OpenAI
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from research import gather_context
from prompts import build_messages

APP_NAME = "Ticker Research API"

# ---- LLM client (OpenAI-compatible) ----
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "")
TRANSCRIPT_LANGS = os.getenv("TRANSCRIPT_LANGS", "en,en-US,en-GB,es,pt,fr,de,ru,hi,ja,ko,zh-Hans,zh-Hant").split(",")
YOUTUBE_TRANSLATE_TO_EN = os.getenv("YOUTUBE_TRANSLATE_TO_EN", "1") == "1"
YOUTUBE_COMMENTS_FALLBACK = os.getenv("YOUTUBE_COMMENTS_FALLBACK", "0") == "1"
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", None)
try:
    if TAVILY_API_KEY:
        from tavily import TavilyClient
        tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
    else:
        tavily_client = None
except Exception:
    tavily_client = None

if not OPENAI_API_KEY:
    # Render won't start until you set this; local dev can set via .env
    print("WARNING: OPENAI_API_KEY not set. The app will error on first LLM call.")

client = OpenAI(base_url=OPENAI_API_BASE, api_key=OPENAI_API_KEY)

logger = logging.getLogger("app")

def _utcnow():
    return datetime.now(timezone.utc)

def _to_rfc3339(dt: datetime) -> str:
    # Ensure RFC3339 with 'Z'
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

def _shorten(txt: str, limit: int = 12000) -> str:
    if not txt: return ""
    return txt if len(txt) <= limit else txt[:limit] + " …"

def _flatten_driver_labels(drivers_obj: Dict[str, Any]) -> List[str]:
    labels = []
    if not drivers_obj: return labels
    for bucket in ("near_term", "medium_term", "long_term"):
        if drivers_obj.get(bucket):
            for side in ("positives", "negatives"):
                for d in drivers_obj[bucket].get(side, []):
                    name = (d.get("driver") or "").strip()
                    if name and name not in labels:
                        labels.append(name)
    return labels[:24]  # keep prompt compact

def _rank_score(views_list: List[int], subs_list: List[int], v: int, s: int) -> float:
    # Normalize by sample max; blend views (0.6) + subs (0.4); log-scale to damp outliers
    vmax = max(views_list) if views_list else 1
    smax = max(subs_list) if subs_list else 1
    v_norm = math.log10(max(v,1)) / math.log10(max(vmax,10))
    s_norm = math.log10(max(s,1)) / math.log10(max(smax,10))
    return 0.6 * v_norm + 0.4 * s_norm

def _yt_get(client: httpx.Client, path: str, params: dict) -> dict:
    base = "https://www.googleapis.com/youtube/v3"
    qp = {**params, "key": YOUTUBE_API_KEY}
    r = client.get(f"{base}/{path}", params=qp, timeout=20)
    r.raise_for_status()
    return r.json()

def _search_and_rank_videos(ticker: str, entity_name: str, lookback_days: int = 7, max_candidates: int = 60) -> List[dict]:
    if not YOUTUBE_API_KEY:
        raise RuntimeError("YOUTUBE_API_KEY is not set")

    queries = []
    t = ticker.upper()
    if entity_name and entity_name.lower() != t.lower():
        # prioritize stock/crypto context
        queries = [f"{t} stock", f"{entity_name} stock", f"{t} {entity_name}", t]
    else:
        queries = [f"{t} stock", t]

    published_after = _to_rfc3339(_utcnow() - timedelta(days=lookback_days))

    items = {}
    with httpx.Client() as client:
        for q in queries:
            try:
                data = _yt_get(client, "search", {
                    "part": "snippet",
                    "q": q,
                    "type": "video",
                    "order": "viewCount",
                    "publishedAfter": published_after,
                    "maxResults": 50,
                    "relevanceLanguage": "en",
                    "regionCode": "US",
                    "safeSearch": "none"
                })
            except httpx.HTTPStatusError as e:
                # Quota or invalid key etc.
                raise RuntimeError(f"YouTube search failed for '{q}': {e.response.text}") from e

            for it in data.get("items", []):
                vid = it["id"]["videoId"]
                if vid not in items:
                    items[vid] = {
                        "videoId": vid,
                        "title": it["snippet"]["title"],
                        "channelId": it["snippet"]["channelId"],
                        "channelTitle": it["snippet"]["channelTitle"],
                        "publishedAt": it["snippet"]["publishedAt"],
                        "description": it["snippet"].get("description", "")
                    }
                if len(items) >= max_candidates:
                    break

        if not items:
            return []

        # Enrich: video stats
        video_ids = list(items.keys())
        videos_stats = {}
        for i in range(0, len(video_ids), 50):
            chunk = ",".join(video_ids[i:i+50])
            data = _yt_get(client, "videos", {"part":"statistics,snippet", "id": chunk})
            for it in data.get("items", []):
                vid = it["id"]
                stats = it.get("statistics", {})
                views = int(stats.get("viewCount", 0))
                like_count = int(stats.get("likeCount", 0)) if "likeCount" in stats else None
                videos_stats[vid] = {"views": views, "likeCount": like_count}

        # Enrich: channel subs
        ch_ids = list({it["channelId"] for it in items.values()})
        chan_stats = {}
        for i in range(0, len(ch_ids), 50):
            chunk = ",".join(ch_ids[i:i+50])
            data = _yt_get(client, "channels", {"part":"statistics", "id": chunk})
            for it in data.get("items", []):
                ch = it["id"]
                s = it.get("statistics", {})
                subs = int(s.get("subscriberCount", 0)) if s.get("hiddenSubscriberCount") is False else 0
                chan_stats[ch] = {"subs": subs}

        # Rank
        views_all = [v.get("views", 0) for v in videos_stats.values()]
        subs_all = [c.get("subs", 0) for c in chan_stats.values()]

        ranked = []
        for vid, meta in items.items():
            vstats = videos_stats.get(vid, {})
            views = int(vstats.get("views", 0))
            subs = int(chan_stats.get(meta["channelId"], {}).get("subs", 0))
            score = _rank_score(views_all, subs_all, views, subs)
            meta.update({
                "views": views, "subs": subs, "score": score,
                "url": f"https://www.youtube.com/watch?v={vid}"
            })
            ranked.append(meta)

        # Sort by our blended score, desc
        ranked.sort(key=lambda x: (x["score"], x["views"], x["subs"]), reverse=True)
        return ranked

def _fetch_transcript_text(video_id: str) -> str | None:
    try:
        # Prefer English; fall back to auto subs if needed
        lines = YouTubeTranscriptApi.get_transcript(video_id, languages=["en", "en-US", "en-GB"])
        text = " ".join(seg.get("text","") for seg in lines if seg.get("text"))
        return text.strip() or None
    except (TranscriptsDisabled, NoTranscriptFound):
        return None
    except Exception:
        # Network or throttling; just skip this video
        return None

def _fetch_transcript_text_flexible(video_meta: dict) -> tuple[Optional[str], dict]:
    """
    Try multiple legit sources for transcript text in priority:
    1) YouTube manual/auto English
    2) YouTube manual/auto in allowed langs (+ optional translate to EN)
    3) External transcript/show notes via Tavily (if available)
    Returns (text, meta) where meta includes source/lang/translated/url (if external).
    """
    video_id = video_meta["videoId"]

    # 1–2) YouTube transcript objects
    try:
        listing = YouTubeTranscriptApi.list_transcripts(video_id)
    except Exception:
        listing = None

    if listing:
        # Manual English first
        try:
            tr = listing.find_manually_created_transcript(["en", "en-US", "en-GB"])
            segs = tr.fetch()
            text = " ".join(s.get("text","") for s in segs)
            if text.strip():
                return text, {"source":"youtube_manual", "lang": tr.language_code, "translated": False}
        except Exception:
            pass
        # Auto English
        try:
            tr = listing.find_generated_transcript(["en", "en-US", "en-GB"])
            segs = tr.fetch()
            text = " ".join(s.get("text","") for s in segs)
            if text.strip():
                return text, {"source":"youtube_auto", "lang": tr.language_code, "translated": False}
        except Exception:
            pass
        # Any other allowed language (manual → auto), optionally translate to EN
        for finder in ("find_manually_created_transcript", "find_generated_transcript"):
            try:
                tr = getattr(listing, finder)(TRANSCRIPT_LANGS)
                segs = tr.fetch()
                text = " ".join(s.get("text","") for s in segs)
                if text.strip():
                    if YOUTUBE_TRANSLATE_TO_EN and tr.language_code not in ("en","en-US","en-GB"):
                        try:
                            tr_en = tr.translate("en")
                            segs_en = tr_en.fetch()
                            text_en = " ".join(s.get("text","") for s in segs_en)
                            if text_en.strip():
                                return text_en, {"source":"youtube_translated", "lang": tr.language_code, "translated": True}
                        except Exception:
                            # fall through to return non‑EN as‑is
                            return text, {"source":"youtube_non_en", "lang": tr.language_code, "translated": False}
                    else:
                        return text, {"source":"youtube_non_en", "lang": tr.language_code, "translated": False}
            except Exception:
                pass
        # Generic translation if available
        if YOUTUBE_TRANSLATE_TO_EN:
            try:
                for tr in listing:
                    if any((l.get("language_code") in ("en","en-US","en-GB")) for l in tr.translation_languages):
                        tr_en = tr.translate("en")
                        segs_en = tr_en.fetch()
                        text_en = " ".join(s.get("text","") for s in segs_en)
                        if text_en.strip():
                            return text_en, {"source":"youtube_translated", "lang": tr.language_code, "translated": True}
            except Exception:
                pass

    # 3) External transcript/show-notes pages via Tavily
    if tavily_client:
        title = video_meta.get("title") or ""
        channel = video_meta.get("channelTitle") or ""
        queries = [
            f'"{title}" transcript {channel}',
            f'{channel} "{title}" "show notes"',
            f'{channel} "{title}" "full transcript"',
        ]
        seen = set()
        for q in queries:
            try:
                res = tavily_client.search(query=q, max_results=5, search_depth="basic", days=365)
                for it in res.get("results", []):
                    url = it.get("url")
                    if not url or url in seen:
                        continue
                    seen.add(url)
                    content = (it.get("content") or "").strip()
                    # Heuristic: look for transcript-y pages with enough text
                    if len(content) > 800 and any(k in content.lower() for k in ["transcript", "show notes", "full text"]):
                        return content[:20000], {"source":"external_page", "lang":"unknown", "translated": False, "url": url}
            except Exception:
                continue

    return None, {"source":"none"}

def _fetch_comments_text(video_id: str, max_threads: int = 80) -> Optional[str]:
    try:
        with httpx.Client() as client:
            data = _yt_get(client, "commentThreads", {
                "part": "snippet",
                "videoId": video_id,
                "order": "relevance",
                "maxResults": 100,
                "textFormat": "plainText"
            })
        texts = []
        for it in data.get("items", [])[:max_threads]:
            sn = it.get("snippet", {}).get("topLevelComment", {}).get("snippet", {})
            txt = sn.get("textDisplay") or sn.get("textOriginal")
            if txt:
                texts.append(txt)
        blob = " ".join(texts)
        return blob[:20000] if blob else None
    except Exception:
        return None

def _build_youtube_messages(tkr: str, entity_name: str, drivers: List[str], transcript: str, video_meta: dict, tmeta: dict) -> List[dict]:
    source_note = f"(transcript source: {tmeta.get('source')}, lang: {tmeta.get('lang','?')}, translated: {tmeta.get('translated',False)})"
    if tmeta.get("source") == "external_page" and tmeta.get("url"):
        source_note += f", external_url: {tmeta['url']}"

    system = {
        "role":"system",
        "content":(
            "You are an equity/crypto research analyst. "
            "Given a transcript-like text for a YouTube video, classify the CREATOR'S stance (bullish/bearish/neutral/mixed), "
            "and assess whether any provided price drivers are suggested to materialize in the NEAR TERM (0–3 months). "
            "Use only the provided text; do not invent facts. Return strict JSON."
        )
    }
    schema = {
        "role":"system",
        "content":(
            "Respond with a JSON object:\n"
            "{\n"
            '  "sentiment": "bullish|bearish|neutral|mixed|uncertain",\n'
            '  "sentiment_score": 0.0-1.0,\n'
            '  "confidence": "low|medium|high",\n'
            '  "evidence_quotes": ["<<=30 words>>", "..."],\n'
            '  "drivers_assessed": [\n'
            '     {"driver":"<label>","mentioned":true/false,\n'
            '      "near_term_hint":"yes|maybe|no|unclear",\n'
            '      "rationale":"<<=40 words>>"}\n'
            '  ]\n'
            "}"
        )
    }
    user = {
        "role":"user",
        "content":(
            f"Ticker: {tkr}\n"
            f"Name: {entity_name or tkr}\n"
            f"Video: {video_meta.get('title')} (channel: {video_meta.get('channelTitle')})\n"
            f"{source_note}\n"
            f"Drivers to check: {drivers}\n\n"
            f"TRANSCRIPT (truncated):\n{_shorten(transcript, 12000)}"
        )
    }
    return [system, schema, user]

class YTSentimentReq(BaseModel):
    ticker: str
    entity_name: Optional[str] = None
    asset_type: Optional[str] = "auto"
    drivers: Optional[Dict[str, Any]] = None
    lookback_days: Optional[int] = 7
    max_videos: Optional[int] = 5

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

@app.post("/api/youtube/sentiment")
def youtube_sentiment(req: YTSentimentReq, x_client_plan: Optional[str] = Header(default=None)):
    # Feature toggle
    if os.getenv("FEATURE_YOUTUBE", "1") != "1":
        raise HTTPException(status_code=403, detail="YouTube feature is disabled.")

    # Simple “Pro” gate (flip with envs; replace with real auth later)
    require_pro = os.getenv("YOUTUBE_REQUIRE_PLAN", "0") == "1"
    plan = (x_client_plan or os.getenv("CLIENT_PLAN", "free")).lower()
    if require_pro and plan != "pro":
        raise HTTPException(status_code=402, detail="YouTube scan is a Pro feature. Upgrade to enable.")

    if not YOUTUBE_API_KEY:
        raise HTTPException(status_code=500, detail="YOUTUBE_API_KEY is not set on the server.")

    t0 = time.time()
    ticker = req.ticker.strip().upper()
    entity = (req.entity_name or ticker).strip()
    lookback = max(1, min(req.lookback_days or 7, 14))
    max_videos = max(1, min(req.max_videos or 5, 10))
    drivers_list = _flatten_driver_labels(req.drivers or {})

    ranked = _search_and_rank_videos(ticker, entity, lookback_days=lookback, max_candidates=60)
    if not ranked:
        return {
            "ticker": ticker,
            "lookback_days": lookback,
            "videos_considered": 0,
            "videos_analyzed": 0,
            "videos": [],
            "table_rows": [],
            "summary": {"note": "No recent videos matched the query in the lookback window."},
            "_meta": {"elapsed_ms": int((time.time()-t0)*1000)}
        }

    top = ranked[:max_videos]
    results = []
    agg_scores = []
    analyzed_count = 0

    for v in top:
        transcript, tmeta = _fetch_transcript_text_flexible(v)
        analysis = None

        if transcript:
            try:
                messages = _build_youtube_messages(ticker, entity, drivers_list, transcript, v, tmeta)
                content = llm_complete(messages=messages, model=(os.getenv("OPENAI_MODEL") or OPENAI_MODEL), try_json=True)
                try:
                    analysis = json.loads(content)
                except Exception:
                    analysis = {"sentiment":"uncertain","sentiment_score":0.5,"confidence":"low",
                                "evidence_quotes":[], "drivers_assessed":[], "_raw": content}
            except Exception as e:
                analysis = {"error": f"LLM analysis failed: {e}"}
        elif YOUTUBE_COMMENTS_FALLBACK:
            comments_text = _fetch_comments_text(v["videoId"])
            if comments_text and len(comments_text) > 400:
                try:
                    # Reuse the same builder but make it very clear in the prompt that this is audience sentiment
                    cmessages = [
                        {"role":"system","content":"You are an analyst estimating AUDIENCE sentiment from YouTube comments about an asset. Output JSON with fields: audience_sentiment (bullish/bearish/neutral/mixed), sentiment_score (0..1), confidence, evidence_quotes, drivers_assessed[]."},
                        {"role":"user","content": f"Ticker: {ticker}\nName: {entity}\nVideo: {v.get('title')} (channel: {v.get('channelTitle')})\nCOMMENTS (sample, truncated):\n{_shorten(comments_text, 12000)}\nDrivers to check: {drivers_list}"}
                    ]
                    content = llm_complete(messages=cmessages, model=(os.getenv("OPENAI_MODEL") or OPENAI_MODEL), try_json=True)
                    analysis = json.loads(content)
                    analysis["_note"] = "Derived from audience comments, not creator speech."
                except Exception as e:
                    analysis = {"note":"No transcript and comment analysis failed.", "error": str(e)}
            else:
                analysis = {"note":"No transcript available; insufficient comments for fallback."}
        else:
            analysis = {"note":"No transcript available; skipped stance analysis."}

        results.append({
            "video_id": v["videoId"],
            "title": v["title"],
            "channel_title": v["channelTitle"],
            "channel_id": v["channelId"],
            "published_at": v["publishedAt"],
            "url": v["url"],
            "views": v["views"],
            "subscribers": v["subs"],
            "transcript_available": bool(transcript),
            "transcript_source": tmeta.get("source"),
            "analysis": analysis
        })


    # Aggregate sentiment (simple avg of analyzed)
    avg_score = sum(agg_scores)/len(agg_scores) if agg_scores else None
    agg_label = None
    if avg_score is not None:
        agg_label = "bearish" if avg_score < 0.4 else "neutral" if avg_score < 0.6 else "bullish"

    # Build table rows
    table_rows = [{
        "date": r["published_at"][:10],
        "time": r["published_at"][11:19] + "Z",
        "title": r["title"],
        "creator": r["channel_title"],
        "url": r["url"]
    } for r in results]

    return {
        "ticker": ticker,
        "entity_name": entity,
        "lookback_days": lookback,
        "videos_considered": len(ranked),
        "videos_analyzed": analyzed_count,
        "aggregate_sentiment": {"avg_score": avg_score, "label": agg_label},
        "videos": results,
        "table_rows": table_rows,
        "_meta": {"elapsed_ms": int((time.time()-t0)*1000)}
    }

@app.get("/api/config")
def get_config():
    return {
        "features": {
            # Toggle visibility of the YouTube checkbox in the UI
            "youtube": os.getenv("FEATURE_YOUTUBE", "1") == "1"
        },
        # If true, the backend will require a "pro" plan for /api/youtube/sentiment
        "youtube_requires_plan": os.getenv("YOUTUBE_REQUIRE_PLAN", "0") == "1",
        # Optional: what plan the *current* deployment represents (free|pro), for demo
        "plan": os.getenv("CLIENT_PLAN", "free").lower()
    }