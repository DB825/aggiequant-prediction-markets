"""Kalshi adapter: fetch resolved markets and map them onto the pipeline schema.

The public Kalshi REST API exposes read endpoints at
`https://api.elections.kalshi.com/trade-api/v2` with no authentication
required for the market-read paths we use here:

- ``GET /markets`` returns live/recent markets (by status: open, closed,
  settled, ...).
- ``GET /historical/markets`` returns archived markets.
- ``GET /events`` lists events (a group of related binary markets).

Each market is a binary contract. We keep the markets whose lifecycle has
reached a terminal state with a known ``result`` of yes or no
and project them onto the pipeline's schema:

    event_id, category, question, market_prob, market_spread,
    open_ts, resolve_ts, resolved, [feat_signal, feat_momentum, feat_dispersion]

The "market_prob" column is the last-observed YES midpoint. This is the
cleanest snapshot we can take from a single REST call per market. A
future extension can pull ``/markets/candlesticks`` at T minus k days
before close to backtest pre-resolution edge; v1 keeps the wire format
narrow so the first end-to-end run is easy to audit.

Prosperity-style feature engineering lives in ``build_orderbook_features``:
microstructure signals (signed order-flow proxy, spread-normalized
momentum, dispersion) computed from the snapshot fields Kalshi exposes
per market. Those are the features a Prosperity-round market-making bot
would lean on before committing capital.

No authentication is required for the public reads used here. Kalshi
applies rate limits, so this module caches responses to disk by default.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from .data import CATEGORIES

DEFAULT_BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
DEFAULT_CACHE_DIR = Path("data/kalshi_cache")
DEFAULT_USER_AGENT = "aggie-pm/0.2 (+https://github.com/DB825/aggiequant-prediction-markets)"

# Terminal statuses where ``result`` is known. Kalshi's current public
# market status filter uses ``settled``; older exports may carry
# ``finalized`` or ``determined``.
RESOLVED_STATUSES = frozenset({"settled", "finalized", "determined"})

KALSHI_CATEGORY_MAP: dict[str, str] = {
    "business": "earnings",
    "climate": "weather",
    "companies": "earnings",
    "crypto": "crypto",
    "culture": "entertainment",
    "economics": "macro",
    "elections": "policy",
    "entertainment": "entertainment",
    "financials": "earnings",
    "government": "policy",
    "health": "policy",
    "international": "geopolitics",
    "markets": "macro",
    "politics": "policy",
    "science": "policy",
    "sports": "sports",
    "technology": "earnings",
    "weather": "weather",
    "world": "geopolitics",
}

_CATEGORY_KEYWORDS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("nba", "nfl", "nhl", "mlb", "soccer", "tennis", "golf", "ufc"), "sports"),
    (("temperature", "rain", "snow", "hurricane", "weather", "climate"), "weather"),
    (("bitcoin", "ethereum", "crypto", "btc", "eth"), "crypto"),
    (("election", "president", "senate", "congress", "approval", "fed chair"), "policy"),
    (("war", "ukraine", "russia", "israel", "china", "nato"), "geopolitics"),
    (("earnings", "revenue", "profit", "stock", "s&p", "nasdaq", "dow"), "earnings"),
    (("cpi", "inflation", "gdp", "fed", "rates", "unemployment"), "macro"),
    (("oscar", "grammy", "movie", "album", "box office", "streaming"), "entertainment"),
)


class KalshiAPIError(RuntimeError):
    """Raised when a Kalshi REST call fails after retries."""


def canonicalize_category(raw: Any, *, text: str = "") -> str:
    """Map Kalshi/event taxonomy to the package's stable category families."""
    raw_s = str(raw or "").strip().lower()
    if raw_s in CATEGORIES:
        return raw_s
    if raw_s in KALSHI_CATEGORY_MAP:
        return KALSHI_CATEGORY_MAP[raw_s]

    blob = f"{raw_s} {text}".lower()
    for keywords, category in _CATEGORY_KEYWORDS:
        if any(k in blob for k in keywords):
            return category
    return "other"


@dataclass(frozen=True)
class KalshiClient:
    """Minimal Kalshi REST client for public read endpoints.

    Parameters
    ----------
    base_url : str
        Trading API v2 base. The production host also serves the
        "elections" subdomain and routes public reads there.
    timeout : float
        Per-request timeout in seconds.
    max_retries : int
        Retries on transient HTTP errors (500, 502, 503, 504) and
        rate-limit responses (429) with exponential backoff.
    """

    base_url: str = DEFAULT_BASE_URL
    timeout: float = 15.0
    max_retries: int = 3

    def get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        url = self.base_url.rstrip("/") + path
        if params:
            url = f"{url}?{urlencode({k: v for k, v in params.items() if v is not None})}"
        req = Request(url, headers={"User-Agent": DEFAULT_USER_AGENT, "Accept": "application/json"})
        delay = 1.0
        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with urlopen(req, timeout=self.timeout) as resp:
                    body = resp.read().decode("utf-8")
                    return json.loads(body)
            except HTTPError as err:
                last_err = err
                if err.code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                    time.sleep(delay)
                    delay *= 2.0
                    continue
                raise KalshiAPIError(f"HTTP {err.code} from {url}: {err.reason}") from err
            except URLError as err:
                last_err = err
                if attempt < self.max_retries:
                    time.sleep(delay)
                    delay *= 2.0
                    continue
                raise KalshiAPIError(f"Network error calling {url}: {err.reason}") from err
        raise KalshiAPIError(f"Exhausted retries calling {url}") from last_err

    def fetch_markets(
        self,
        *,
        status: str | None = "settled",
        series_ticker: str | None = None,
        event_ticker: str | None = None,
        tickers: str | None = None,
        source: str = "historical",
        max_pages: int = 10,
        page_limit: int = 200,
    ) -> list[dict[str, Any]]:
        """Page through Kalshi market endpoints collecting binary markets.

        Parameters
        ----------
        status : str | None
            Kalshi lifecycle status for the live endpoint. ``settled`` is
            the terminal state with a known ``result``. Historical markets
            do not accept a status filter, so this is ignored there.
        series_ticker, event_ticker : str | None
            Optional filters. Kalshi treats them as mutually exclusive.
        tickers : str | None
            Optional comma-separated market tickers.
        source : {"historical", "live"}
            ``historical`` calls ``GET /historical/markets`` for archived
            settled markets. ``live`` calls ``GET /markets`` for recent
            markets, usually with ``status="settled"``.
        max_pages : int
            Safety cap on pagination so a bad filter does not run forever.
        page_limit : int
            Page size. Kalshi caps this (currently 1000), 200 is friendly.
        """
        if source not in {"historical", "live"}:
            raise ValueError("source must be 'historical' or 'live'")
        endpoint = "/historical/markets" if source == "historical" else "/markets"
        out: list[dict[str, Any]] = []
        cursor: str | None = None
        for _ in range(max_pages):
            params = {
                "limit": page_limit,
                "cursor": cursor,
                "series_ticker": series_ticker,
                "event_ticker": event_ticker,
                "tickers": tickers,
            }
            if source == "live":
                params["status"] = status
            payload = self.get(endpoint, params=params)
            markets = payload.get("markets") or []
            out.extend(markets)
            cursor = payload.get("cursor") or None
            if not cursor or not markets:
                break
        return out


# ---------------------------------------------------------------------------
# Fixed-point parsing
# ---------------------------------------------------------------------------


def _parse_dollars(val: Any) -> float:
    """Parse Kalshi's FixedPointDollars into a float in [0, 1] for YES-side prices.

    Kalshi returns prices as strings like ``"0.5400"`` (dollars per contract).
    For binary YES markets that equals the implied probability. We clamp to
    [1e-4, 1 - 1e-4] so downstream logit transforms never blow up.
    """
    if val is None or val == "":
        return float("nan")
    try:
        x = float(val)
    except (TypeError, ValueError):
        return float("nan")
    if not np.isfinite(x):
        return float("nan")
    return float(np.clip(x, 1e-4, 1 - 1e-4))


def _parse_count(val: Any) -> float:
    if val is None or val == "":
        return 0.0
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


def _parse_ts(val: Any) -> pd.Timestamp | None:
    if not val:
        return None
    try:
        return pd.Timestamp(val)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Pipeline mapping
# ---------------------------------------------------------------------------


def _midpoint(yes_bid: float, yes_ask: float, last_price: float) -> float:
    """Best-available snapshot of the YES probability.

    Prefer the bid/ask midpoint when both sides are present. Fall back to
    last trade if the book is one-sided, which happens on thinly-traded
    markets just before close.
    """
    if np.isfinite(yes_bid) and np.isfinite(yes_ask):
        return 0.5 * (yes_bid + yes_ask)
    if np.isfinite(last_price):
        return float(last_price)
    if np.isfinite(yes_bid):
        return float(yes_bid)
    if np.isfinite(yes_ask):
        return float(yes_ask)
    return float("nan")


def _spread(yes_bid: float, yes_ask: float) -> float:
    """Round-trip spread, floored at 1 cent so fees stay realistic."""
    if np.isfinite(yes_bid) and np.isfinite(yes_ask):
        return float(max(yes_ask - yes_bid, 0.01))
    return 0.05


def _question(market: dict[str, Any]) -> str:
    for key in ("yes_sub_title", "title", "subtitle"):
        val = market.get(key)
        if val:
            return str(val)
    return market.get("ticker", "")


def _result_to_binary(result: Any) -> int | None:
    """Map Kalshi's string result to 0/1. Unknown/scalar results drop."""
    if not isinstance(result, str):
        return None
    r = result.strip().lower()
    if r == "yes":
        return 1
    if r == "no":
        return 0
    return None


def kalshi_markets_to_dataframe(
    markets: Iterable[dict[str, Any]],
    *,
    drop_unresolved: bool = True,
) -> pd.DataFrame:
    """Project a list of raw Kalshi market dicts onto the pipeline schema.

    Returns a DataFrame with the columns the rest of ``aggie_pm``
    expects: ``event_id, category, question, market_prob,
    market_spread, open_ts, resolve_ts, resolved`` plus the raw
    microstructure fields under ``raw_*`` so downstream feature
    engineering can use them.
    """
    rows: list[dict[str, Any]] = []
    for m in markets:
        result = _result_to_binary(m.get("result"))
        if drop_unresolved and result is None:
            continue

        yes_bid = _parse_dollars(m.get("yes_bid_dollars"))
        yes_ask = _parse_dollars(m.get("yes_ask_dollars"))
        last_price = _parse_dollars(m.get("last_price_dollars"))
        prev_price = _parse_dollars(m.get("previous_price_dollars"))
        volume = _parse_count(m.get("volume_fp"))
        volume_24h = _parse_count(m.get("volume_24h_fp"))
        open_interest = _parse_count(m.get("open_interest_fp"))

        mkt_prob = _midpoint(yes_bid, yes_ask, last_price)
        spread = _spread(yes_bid, yes_ask)

        open_time = _parse_ts(m.get("open_time")) or _parse_ts(m.get("created_time"))
        close_time = (
            _parse_ts(m.get("settlement_ts"))
            or _parse_ts(m.get("close_time"))
            or _parse_ts(m.get("expiration_time"))
            or _parse_ts(m.get("expected_expiration_time"))
        )

        question = _question(m)
        raw_category = str(m.get("category") or "").strip().lower()
        category = canonicalize_category(
            raw_category,
            text=" ".join(
                str(m.get(k) or "")
                for k in ("ticker", "event_ticker", "title", "subtitle", "yes_sub_title")
            ),
        )

        rows.append(
            {
                "event_id": m.get("ticker", ""),
                "event_ticker": m.get("event_ticker", ""),
                "raw_category": raw_category,
                "category": category,
                "question": question,
                "market_prob": mkt_prob,
                "market_spread": spread,
                "open_time": open_time,
                "close_time": close_time,
                "resolved": result if result is not None else -1,
                "raw_yes_bid": yes_bid,
                "raw_yes_ask": yes_ask,
                "raw_last_price": last_price,
                "raw_prev_price": prev_price,
                "raw_volume": volume,
                "raw_volume_24h": volume_24h,
                "raw_open_interest": open_interest,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Drop rows with unusable prices, but keep track of how many.
    df = df.dropna(subset=["market_prob"]).reset_index(drop=True)

    # Day indices ordered by close_time. Walk-forward backtests need a
    # strict temporal order; if close_time is missing we fall back to the
    # row order, which is already API-returned "most recent first" and
    # gets reversed below.
    if df["close_time"].notna().any():
        df = df.sort_values("close_time", kind="mergesort").reset_index(drop=True)
        base = df["close_time"].min()
        df["resolve_ts"] = ((df["close_time"] - base).dt.total_seconds() / 86400.0).astype(int)
    else:
        df = df.iloc[::-1].reset_index(drop=True)
        df["resolve_ts"] = np.arange(len(df))

    if df["open_time"].notna().any() and df["close_time"].notna().any():
        df["open_ts"] = (
            (df["open_time"] - df["close_time"].min()).dt.total_seconds() / 86400.0
        ).fillna(df["resolve_ts"] - 7).astype(int)
    else:
        df["open_ts"] = (df["resolve_ts"] - 7).clip(lower=0)

    return df


# ---------------------------------------------------------------------------
# Prosperity-style orderbook features
# ---------------------------------------------------------------------------


def build_orderbook_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the pipeline's ``feat_signal / feat_momentum / feat_dispersion``
    columns from Kalshi microstructure fields.

    The logic is borrowed from Prosperity-style market-making heuristics:

    - **feat_signal**: signed deviation of last trade vs midpoint,
      scaled by the half-spread. When fills are printing through the
      ask the signal is positive (informed buyers); through the bid,
      negative (informed sellers). This is a classic order-flow
      imbalance proxy (Cartea, Jaimungal, Penalva 2015,
      *Algorithmic and High-Frequency Trading*).
    - **feat_momentum**: change in last price relative to the
      previous-day close, scaled by spread. A larger move per unit
      spread is rarer and more informative than one move alone.
    - **feat_dispersion**: spread divided by how far the midpoint sits
      from 0.5. Wide spreads on priced-out markets are cheap; wide
      spreads on tossups are where liquidity hates to be and edges
      tend to cluster.
    - **feat_liquidity**: log-scaled volume. Retained as an extra
      column (not a mandated feature) so notebooks can slice on it.

    All features are computed per row with no cross-row leakage.
    """
    if df.empty:
        out = df.copy()
        for col in ("feat_signal", "feat_momentum", "feat_dispersion", "feat_liquidity"):
            out[col] = []
        return out

    last = df["raw_last_price"].to_numpy(dtype=float)
    prev = df["raw_prev_price"].to_numpy(dtype=float)
    mid = df["market_prob"].to_numpy(dtype=float)
    spread = df["market_spread"].to_numpy(dtype=float)
    half_spread = np.maximum(spread / 2.0, 1e-3)
    volume = df["raw_volume"].to_numpy(dtype=float)

    # Informed-flow proxy. nanmean if last is missing.
    last_filled = np.where(np.isfinite(last), last, mid)
    feat_signal = (last_filled - mid) / half_spread

    prev_filled = np.where(np.isfinite(prev), prev, last_filled)
    feat_momentum = (last_filled - prev_filled) / half_spread

    extremity = np.abs(mid - 0.5) + 0.05
    feat_dispersion = spread / extremity

    feat_liquidity = np.log1p(np.maximum(volume, 0.0))

    out = df.copy()
    out["feat_signal"] = np.clip(feat_signal, -6.0, 6.0)
    out["feat_momentum"] = np.clip(feat_momentum, -6.0, 6.0)
    out["feat_dispersion"] = np.clip(feat_dispersion, 0.0, 10.0)
    out["feat_liquidity"] = feat_liquidity
    return out


# ---------------------------------------------------------------------------
# High-level fetch-or-cache helper
# ---------------------------------------------------------------------------


def load_kalshi_resolved(
    *,
    source: str = "historical",
    status: str | None = "settled",
    series_ticker: str | None = None,
    event_ticker: str | None = None,
    tickers: str | None = None,
    max_pages: int = 20,
    page_limit: int = 200,
    cache_dir: Path | str | None = DEFAULT_CACHE_DIR,
    refresh: bool = False,
    client: KalshiClient | None = None,
) -> pd.DataFrame:
    """Fetch resolved Kalshi markets and project them onto the pipeline schema.

    ``source="historical"`` is the default because settled markets older
    than Kalshi's live-data cutoff live under ``GET /historical/markets``.
    Use ``source="live"`` for recently settled markets. Caches the raw
    JSON response under ``cache_dir`` keyed by the filter args. Pass
    ``refresh=True`` to ignore the cache and re-fetch.

    Returns a DataFrame ready to feed into ``build_features``, with
    Prosperity-style orderbook features already attached.
    """
    cache_dir_p = Path(cache_dir) if cache_dir else None
    cache_file: Path | None = None
    if cache_dir_p is not None:
        cache_dir_p.mkdir(parents=True, exist_ok=True)
        key_parts = [
            source,
            status or "any",
            series_ticker or "any",
            event_ticker or "any",
            tickers or "any",
            str(page_limit),
            str(max_pages),
        ]
        key = "_".join(p.replace("/", "-") for p in key_parts)
        cache_file = cache_dir_p / f"kalshi_{key}.json"

    markets: list[dict[str, Any]]
    if cache_file is not None and cache_file.exists() and not refresh:
        markets = json.loads(cache_file.read_text(encoding="utf-8"))
    else:
        cli = client or KalshiClient()
        markets = cli.fetch_markets(
            source=source,
            status=status,
            series_ticker=series_ticker,
            event_ticker=event_ticker,
            tickers=tickers,
            max_pages=max_pages,
            page_limit=page_limit,
        )
        if cache_file is not None:
            cache_file.write_text(json.dumps(markets), encoding="utf-8")

    df = kalshi_markets_to_dataframe(markets, drop_unresolved=True)
    if df.empty:
        return df
    df = build_orderbook_features(df)
    return df
