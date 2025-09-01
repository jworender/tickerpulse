from typing import Dict, Any, List, Tuple
import json

SYSTEM_INSTRUCTIONS = """
You are a disciplined sell-side research analyst. Produce a sober, structured brief for the given ticker.
Focus on (a) near/medium/long-term positive & negative price drivers and (b) which drivers are likely to occur and when.
Do not give investment advice or price targets. Avoid sensational language.
Where specific claims rely on web sources, cite them using the provided source IDs (e.g., "source_ids": ["S1","S3"]).
"""

SCHEMA_SPEC = """
Return a single JSON object with this exact shape:

{
  "ticker": "<e.g., AAPL>",
  "entity_name": "<resolved name if available>",
  "asset_type": "equity" | "crypto",
  "as_of": "<ISO8601 UTC timestamp>",
  "summary": "<3-6 sentence overview tailored to this asset and the current context.>",
  "drivers": {
    "near_term": {
      "positives": [ {"driver": "...", "why_it_matters": "...", "source_ids": ["S1","S2"] } ],
      "negatives": [ {"driver": "...", "why_it_matters": "...", "source_ids": [] } ]
    },
    "medium_term": {
      "positives": [ ... ],
      "negatives": [ ... ]
    },
    "long_term": {
      "positives": [ ... ],
      "negatives": [ ... ]
    }
  },
  "timeline_assessment": [
    {
      "driver": "<short label>",
      "timeframe": "near-term (0-3 months)" | "medium-term (3-12 months)" | "long-term (1-3+ years)",
      "expected_window": "<e.g., Oct–Dec 2025>",
      "likelihood": 0.0-1.0,
      "confidence": "low" | "medium" | "high",
      "rationale": "<succinct reasoning>",
      "leading_indicators": ["<signal 1>", "<signal 2>"],
      "source_ids": ["S3"]
    }
  ],
  "risks_and_mitigants": [
    {"risk": "...", "mitigants": ["..."], "source_ids": []}
  ],
  "sources_index": [
    {"id":"S1","title":"...","url":"...","published":"<date>"}
  ],
  "disclaimer": "For education only. Not investment advice."
}
"""

GUIDANCE = """
Time buckets: near-term = 0–3 months; medium-term = 3–12 months; long-term = 1–3+ years.
If formal sources are limited, lean on industry knowledge and make that explicit; still keep structure.
Be specific about mechanisms (unit economics, margins, regulation, roadmap, demand/supply, catalysts).
Never fabricate sources; only include IDs that appear in sources_index.
"""

def _render_sources_for_prompt(sources_index: List[dict]) -> str:
    if not sources_index:
        return "No sources provided."
    lines = []
    for s in sources_index:
        lines.append(f"- {s.get('id')}: {s.get('title')} | {s.get('url')} | published: {s.get('published')}\n  snippet: {s.get('snippet')}")
    return "\n".join(lines)

def build_messages(ctx: Dict[str, Any], user_question: str = None) -> Tuple[List[dict], dict]:
    """Prepare chat messages & response_format for a JSON answer."""
    # Context block
    ctx_lines = []
    ctx_lines.append(f"Ticker: {ctx.get('ticker')}")
    ctx_lines.append(f"Asset type: {ctx.get('asset_type')}")
    if ctx.get("entity_name"):
        ctx_lines.append(f"Name: {ctx.get('entity_name')}")
    if ctx.get("sector"):
        ctx_lines.append(f"Sector: {ctx.get('sector')}")
    if ctx.get("industry"):
        ctx_lines.append(f"Industry: {ctx.get('industry')}")
    if ctx.get("last_price") is not None:
        ctx_lines.append(f"Last price: {ctx.get('last_price')} {ctx.get('currency') or ''}".strip())
    if ctx.get("market_cap"):
        ctx_lines.append(f"Market cap: {ctx.get('market_cap')}")
    if ctx.get("ma50") or ctx.get("ma200"):
        ctx_lines.append(f"MA50: {ctx.get('ma50')} | MA200: {ctx.get('ma200')}")
    if ctx.get("next_earnings"):
        ctx_lines.append(f"Next earnings: {ctx.get('next_earnings')}")
    if ctx.get("price_change_24h") is not None:
        ctx_lines.append(f"24h change: {ctx.get('price_change_24h')}%")
    if ctx.get("price_change_7d") is not None:
        ctx_lines.append(f"7d change: {ctx.get('price_change_7d')}%")
    if ctx.get("price_change_30d") is not None:
        ctx_lines.append(f"30d change: {ctx.get('price_change_30d')}%")

    sources_str = _render_sources_for_prompt(ctx.get("sources_index", []))
    user_q = user_question.strip() if user_question else "Focus on price drivers and when they might occur."

    system_msg = {"role":"system","content": SYSTEM_INSTRUCTIONS}
    dev_msg = {"role":"system","content": GUIDANCE}
    schema_msg = {"role":"system","content": f"Follow this exact JSON schema:\n{SCHEMA_SPEC}"}
    context_msg = {
        "role":"user",
        "content": f"""CONTEXT (as-of {ctx.get('as_of')} UTC)
{chr(10).join(ctx_lines)}

SOURCES:
{sources_str}

TASK:
- Analyze {ctx.get('ticker')} as described.
- {user_q}
- Output ONLY a single JSON object matching the specified schema."""
    }

    # Request JSON if the provider supports it (OpenAI-style)
    response_format = {"type":"json_object"}

    return [system_msg, dev_msg, schema_msg, context_msg], response_format
