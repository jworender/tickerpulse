import os, math, statistics
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
import httpx
import yfinance as yf

# Optional Tavily search
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", None)
try:
    if TAVILY_API_KEY:
        from tavily import TavilyClient
        tavily = TavilyClient(api_key=TAVILY_API_KEY)
    else:
        tavily = None
except Exception:
    tavily = None

def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default

def _moving_avg(series: List[float], window: int) -> Optional[float]:
    if not series or len(series) < window:
        return None
    return sum(series[-window:]) / window

# ---------- Equity helpers (Yahoo Finance via yfinance) ----------
def _equity_snapshot(ticker: str) -> Dict[str, Any]:
    t = yf.Ticker(ticker)
    hist = None
    try:
        hist = t.history(period="6mo", interval="1d")
    except Exception:
        hist = None

    closes = hist["Close"].tolist() if hist is not None and "Close" in hist else []
    ma50 = _moving_avg(closes, 50)
    ma200 = _moving_avg(closes, 200)

    # fast_info tends to be more reliable than .info these days
    fi = {}
    try:
        fi = t.fast_info or {}
    except Exception:
        fi = {}

    last_price = _safe_float(fi.get("last_price"), closes[-1] if closes else None)

    # Earnings (yfinance calendar sometimes returns dataframe; be resilient)
    next_earnings = None
    try:
        cal = t.get_calendar()  # new yfinance
        if cal is not None and "Earnings Date" in cal:
            ed = cal["Earnings Date"]
            if isinstance(ed, list) or isinstance(ed, tuple):
                next_earnings = str(ed[0])
            else:
                next_earnings = str(ed)
    except Exception:
        next_earnings = None

    # Basic identity
    entity_name = None
    try:
        # yfinance 0.2.40 adds get_info()
        info = t.get_info()
        entity_name = info.get("longName") or info.get("shortName")
        sector = info.get("sector")
        industry = info.get("industry")
    except Exception:
        info, sector, industry = {}, None, None

    return {
        "asset_type": "equity",
        "ticker": ticker,
        "entity_name": entity_name or ticker,
        "as_of": _utcnow_iso(),
        "last_price": last_price,
        "currency": fi.get("currency"),
        "market": fi.get("market"),
        "year_high": _safe_float(fi.get("year_high")),
        "year_low": _safe_float(fi.get("year_low")),
        "market_cap": _safe_float(fi.get("market_cap")),
        "ma50": ma50,
        "ma200": ma200,
        "next_earnings": next_earnings,
        "sector": sector,
        "industry": industry,
    }

# ---------- Crypto helpers (CoinGecko public API) ----------
CG_BASE = "https://api.coingecko.com/api/v3"

async def _cg_get(client, path, params=None):
    url = f"{CG_BASE}{path}"
    r = await client.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()

def _crypto_snapshot_sync(symbol: str) -> Dict[str, Any]:
    symbol = symbol.lower()
    # Try simple resolves for common ones to avoid big search:
    symbol_map = {
        "btc": "bitcoin",
        "eth": "ethereum",
        "sol": "solana",
        "ada": "cardano",
        "xrp": "ripple",
        "doge": "dogecoin",
        "matic": "matic-network",
        "avax": "avalanche-2"
    }
    candidate_ids = []
    if symbol in symbol_map:
        candidate_ids.append(symbol_map[symbol])

    # Use search endpoint to find candidates
    with httpx.Client() as client:
        try:
            s = client.get(f"{CG_BASE}/search", params={"query": symbol}, timeout=20)
            s.raise_for_status()
            data = s.json()
            for c in data.get("coins", []):
                if c.get("symbol", "").lower() == symbol:
                    candidate_ids.append(c.get("id"))
        except Exception:
            pass

        cg_id = candidate_ids[0] if candidate_ids else None
        if not cg_id:
            # Fallback: treat unknown symbol generically
            return {
                "asset_type": "crypto",
                "ticker": symbol.upper(),
                "entity_name": symbol.upper(),
                "as_of": _utcnow_iso(),
            }

        # Markets data
        try:
            m = client.get(f"{CG_BASE}/coins/markets",
                           params={"vs_currency":"usd", "ids": cg_id, "price_change_percentage":"24h,7d,30d"},
                           timeout=20)
            m.raise_for_status()
            row = (m.json() or [{}])[0]
        except Exception:
            row = {}

        # Basic identity/details
        entity_name = row.get("name") or cg_id.replace("-", " ").title()

        return {
            "asset_type": "crypto",
            "ticker": symbol.upper(),
            "entity_name": entity_name,
            "as_of": _utcnow_iso(),
            "last_price": row.get("current_price"),
            "currency": "USD",
            "market_cap": row.get("market_cap"),
            "price_change_24h": row.get("price_change_percentage_24h"),
            "price_change_7d": row.get("price_change_percentage_7d_in_currency"),
            "price_change_30d": row.get("price_change_percentage_30d_in_currency"),
            "coingecko_id": cg_id,
        }

# ---------- Search enrichment (Tavily) ----------
def _search_snippets(queries: List[str], max_results: int = 8) -> List[Dict[str, Any]]:
    if not tavily:
        return []
    seen = set()
    results = []
    for q in queries:
        try:
            res = tavily.search(
                query=q,
                search_depth="advanced",
                max_results=5,
                include_domains=None,
                exclude_domains=None,
                days=365  # look back up to 1 year
            )
            for item in res.get("results", []):
                url = item.get("url")
                if not url or url in seen:
                    continue
                seen.add(url)
                results.append({
                    "title": item.get("title"),
                    "url": url,
                    "content": item.get("content"),
                    "published": item.get("published_date"),
                })
                if len(results) >= max_results:
                    return results
        except Exception:
            continue
    return results

# ---------- Public function ----------
def gather_context(ticker: str, asset_type: Optional[str] = "auto", use_search: bool = True) -> Dict[str, Any]:
    ticker = ticker.upper()
    ctx: Dict[str, Any] = {}

    # Decide asset type
    chosen_type = asset_type
    if chosen_type == "auto":
        # Heuristic: try equity first, then crypto
        try:
            eq = _equity_snapshot(ticker)
            if eq.get("last_price") is not None:
                ctx = eq
                chosen_type = "equity"
            else:
                raise RuntimeError("No equity price")
        except Exception:
            # Try crypto
            cr = _crypto_snapshot_sync(ticker)
            ctx = cr
            chosen_type = "crypto"
    elif chosen_type == "equity":
        ctx = _equity_snapshot(ticker)
    else:
        ctx = _crypto_snapshot_sync(ticker)

    # Build search queries
    sources_index: List[Dict[str, Any]] = []
    if use_search and tavily:
        name = ctx.get("entity_name") or ticker
        if ctx["asset_type"] == "equity":
            queries = [
                f"{name} {ticker} catalysts drivers risks",
                f"{name} {ticker} earnings guidance outlook",
                f"{name} {ticker} regulatory risk {ctx.get('industry') or ''}",
                f"{name} {ticker} product launch roadmap",
                f"{name} {ticker} supply chain contracts customer demand",
            ]
        else:
            sym = ctx.get("ticker")
            queries = [
                f"{name} {sym} roadmap upgrade fork proposal risks",
                f"{name} {sym} tokenomics supply unlocks emissions schedule",
                f"{name} {sym} regulation SEC ETF approval denial delay",
                f"{name} {sym} ecosystem adoption developers addresses active users",
                f"{name} {sym} catalysts drivers halving staking yields",
            ]
        snippets = _search_snippets(queries)
        # Index and trim content to keep prompt size reasonable
        for i, s in enumerate(snippets, start=1):
            sources_index.append({
                "id": f"S{i}",
                "title": s.get("title"),
                "url": s.get("url"),
                "published": s.get("published"),
                "snippet": (s.get("content") or "")[:800]
            })

    ctx["sources_index"] = sources_index
    return ctx
