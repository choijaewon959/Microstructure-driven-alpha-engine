import pandas as pd
import numpy as np
import os 
from massive import RESTClient
from dotenv import load_dotenv
from pathlib import Path

BASE_DIR = Path(__file__).resolve()
DATA_DIR = BASE_DIR.parent / "data"

load_dotenv()
API_KEY = os.getenv("POLYGON_API_KEY")


def get_ohlcv(
    ticker: str,
    timespan: str,
    date: str
) -> list:
    client = RESTClient(API_KEY)

    aggs = []
    print(f"Fetching {ticker} prices for {date}...")
    start = date
    end = (pd.Timestamp(date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    for a in client.list_aggs(
        ticker,
        1,
        timespan,
        start,
        end,
        adjusted="true",
        sort="asc",
        limit=50000,
    ):
        aggs.append(a)

    return aggs


def parse_aggs(aggs, symbol: str) -> pd.DataFrame:
    records = []
    for a in aggs:
        records.append({
            "ts": pd.to_datetime(a.timestamp, unit="ms", utc=True),
            "symbol": symbol,
            "open": a.open,
            "high": a.high,
            "low": a.low,
            "close": a.close,
            "volume": a.volume,
            "vwap": getattr(a, "vwap", None),
            "n_trades": getattr(a, "transactions", None),
        })

    df = pd.DataFrame(records)

    return df


def localize_to_eastern(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ts"] = df["ts"].dt.tz_convert("US/Eastern")
    return df


def filter_rth(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        (df["ts"].dt.time >= pd.to_datetime("09:30").time()) &
        (df["ts"].dt.time <= pd.to_datetime("16:00").time())
    ]


def finalize_bars(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("ts").set_index("ts")
    return df


def reindex_1s(df: pd.DataFrame) -> pd.DataFrame:
    if df.index.tz is None:
        raise ValueError("df.index must be timezone-aware")

    sym = None
    if "symbol" in df.columns:
        # grab the first non-null symbol
        sym_series = df["symbol"].dropna()
        sym = sym_series.iloc[0] if len(sym_series) else None

    tz = df.index.tz
    out = []

    for day in df.index.normalize().unique():
        start = pd.Timestamp(day).tz_convert(tz) + pd.Timedelta(hours=9, minutes=30)
        end   = pd.Timestamp(day).tz_convert(tz) + pd.Timedelta(hours=16)

        full_idx = pd.date_range(start, end, freq="1s", inclusive="left")

        day_df = df.loc[(df.index >= start) & (df.index < end)].copy()
        day_df = day_df.reindex(full_idx)

        price_cols = [c for c in ["open","high","low","close","vwap"] if c in day_df.columns]
        if price_cols:
            day_df[price_cols] = day_df[price_cols].ffill()

        for col in ["volume", "n_trades"]:
            if col in day_df.columns:
                day_df[col] = day_df[col].fillna(0)

        if sym is not None:
            day_df["symbol"] = sym

        out.append(day_df)

    return pd.concat(out).sort_index()


def collect_price_data(tickers, timespan, start, end):
    for ticker in tickers:
        for d in pd.date_range(start, end, inclusive='left'):
            date = d.strftime("%Y-%m-%d")
            aggs = get_ohlcv(ticker, timespan, date)
            if aggs:
                df = (
                    parse_aggs(aggs, ticker)
                    .pipe(localize_to_eastern)
                    .pipe(filter_rth)
                    .pipe(finalize_bars)
                    .pipe(reindex_1s)
                )

                prices_dir = DATA_DIR / "prices"
                prices_dir.mkdir(parents=True, exist_ok=True)

                out_fp = prices_dir / f"{ticker}_1s_{date}.parquet"
                df.index.name = "ts"
                df.reset_index().to_parquet(
                    out_fp,
                    engine="pyarrow",
                    compression="zstd"
                )

                print("Saved:", out_fp)


def get_nbbo(
    ticker: str, 
    date: str
) -> list:
    """
    start_date/end_date: 'YYYY-MM-DD'
    end_date is exclusive (standard Python range style)
    """
    client = RESTClient(API_KEY)

    all_quotes = []
    print(f"Fetching {ticker} quotes for {date}...")

    for q in client.list_quotes(
        ticker=ticker,
        timestamp=date,       
        order="asc",
        sort="timestamp",
        limit=50000,
    ):
        all_quotes.append(q)

    return all_quotes


def quotes_to_df(symbol: str, quotes: pd.DataFrame) -> pd.DataFrame:
    """
    quotes: list/iterable of quote objects (or dicts) from client.list_quotes(...)
    Returns a raw DataFrame with best-effort column extraction.
    """
    rows = []
    for q in quotes:
        # Support both object-like and dict-like
        get = (lambda k, default=None: q.get(k, default)) if isinstance(q, dict) else (lambda k, default=None: getattr(q, k, default))

        rows.append({
            "symbol": symbol,
            "sip_timestamp": get("sip_timestamp"),
            "participant_timestamp": get("participant_timestamp"),
            "sequence_number": get("sequence_number"),
            "bid_price": get("bid_price"),
            "ask_price": get("ask_price"),
            "bid_size": get("bid_size"),
            "ask_size": get("ask_size"),
            "bid_exchange": get("bid_exchange"),
            "ask_exchange": get("ask_exchange"),
            "tape": get("tape"),
        })

    df = pd.DataFrame(rows)

    # Choose best available timestamp
    if "sip_timestamp" in df.columns and df["sip_timestamp"].notna().any():
        df["ts"] = pd.to_datetime(df["sip_timestamp"], unit="ns", utc=True)
    elif "participant_timestamp" in df.columns and df["participant_timestamp"].notna().any():
        df["ts"] = pd.to_datetime(df["participant_timestamp"], unit="ns", utc=True)
    else:
        raise ValueError("No usable timestamp field found (sip_timestamp / participant_timestamp).")

    df = df.sort_values(["ts", "sequence_number"], kind="stable").reset_index(drop=True)
    return df


def reconstruct_nbbo_state(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Many quote records are incremental. For example ask_price=0 means 'ask unchanged'.
    This function forward-fills bid/ask prices and sizes ONLY when the incoming value is > 0.
    """
    df = df_raw.copy()

    # Treat zeros as "no update" for these fields
    for col in ["bid_price", "ask_price", "bid_size", "ask_size"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df.loc[df[col] == 0, col] = np.nan

    # Forward-fill state through time
    df[["bid_price", "ask_price", "bid_size", "ask_size"]] = df[["bid_price", "ask_price", "bid_size", "ask_size"]].ffill()

    # Drop any rows before both sides exist
    df = df.dropna(subset=["bid_price", "ask_price"]).copy()

    # Basic derived fields
    df["mid"] = 0.5 * (df["bid_price"] + df["ask_price"])
    df["spread"] = df["ask_price"] - df["bid_price"]

    # L1 imbalance (top-of-book)
    denom = (df["bid_size"].fillna(0) + df["ask_size"].fillna(0)).replace(0, np.nan)
    df["imbalance_l1"] = (df["bid_size"] - df["ask_size"]) / denom

    # Microprice (requires sizes; if missing sizes, result is NaN)
    size_denom = (df["bid_size"] + df["ask_size"]).replace(0, np.nan)
    df["microprice"] = (df["ask_price"] * df["bid_size"] + df["bid_price"] * df["ask_size"]) / size_denom
    df["microprice_dev"] = df["microprice"] - df["mid"]

    return df


def resample_nbbo_1s(df_state: pd.DataFrame) -> pd.DataFrame:
    """
    Resamples event-driven NBBO to 1-second bars using last-observation-carried-forward.
    """
    df = df_state.copy().set_index("ts").sort_index()

    keep_cols = ["symbol", "bid_price", "ask_price", "bid_size", "ask_size", "mid", "spread", "imbalance_l1", "microprice", "microprice_dev"]
    keep_cols = [c for c in keep_cols if c in df.columns]

    out = df[keep_cols].resample("1s").last().ffill()
    out.index.name = "ts"
    return out


def collect_nbbo_data(
    tickers: str, 
    start: str,
    end: str
):
    for ticker in tickers:
        for d in pd.date_range(start, end, inclusive='left'):
            date = d.strftime("%Y-%m-%d")
            quotes = get_nbbo(ticker, date)
            if quotes:
                df_quotes = (
                    quotes_to_df(ticker, quotes)
                    .pipe(reconstruct_nbbo_state)
                    .pipe(localize_to_eastern)
                    .pipe(filter_rth)
                    .pipe(resample_nbbo_1s)
                )

                prices_dir = DATA_DIR / "quotes"
                prices_dir.mkdir(parents=True, exist_ok=True)

                out_fp = prices_dir / f"{ticker}_quotes_1s_{date}.parquet"
                df_quotes.reset_index().to_parquet(
                    out_fp,
                    engine="pyarrow",
                    compression="zstd"
                )

                print("Saved:", out_fp)


def get_trades(
    ticker: str, 
    date: str
) -> list:
    """
    start_date/end_date: 'YYYY-MM-DD'
    """
    client = RESTClient(API_KEY)

    trades = []
    print(f"Fetching {ticker} trades for {date}...")

    for t in client.list_trades(
        ticker=ticker,
        timestamp=date,
        order="asc",
        sort="timestamp",
        limit=50000
    ):
        trades.append(t)

    return trades


def trades_to_df(
    trades, 
    symbol: str, 
    ts_preference: str = "sip"
) -> pd.DataFrame:
    """
    trades: list/iterable of trade objects
    ts_preference: "sip" or "participant"
      - sip_timestamp: when SIP received trade
      - participant_timestamp: when exchange generated trade
    """

    rows = []
    for t in trades:
        get = (lambda k, default=None: t.get(k, default)) if isinstance(t, dict) else (lambda k, default=None: getattr(t, k, default))

        rows.append({
            "symbol": symbol,
            "price": get("price"),
            "size": get("size"),

            # timestamps (ns unix)
            "sip_timestamp": get("sip_timestamp"),
            "participant_timestamp": get("participant_timestamp"),
            "trf_timestamp": get("trf_timestamp"),

            # identifiers / ordering
            "id": get("id"),
            "sequence_number": get("sequence_number"),

            # venue / meta
            "exchange": get("exchange"),
            "tape": get("tape"),
            "trf_id": get("trf_id"),

            # quality / filtering
            "conditions": get("conditions"),
            "correction": get("correction"),
        })

    df = pd.DataFrame(rows)

    # numeric coercion
    for col in ["price", "size", "sip_timestamp", "participant_timestamp", "trf_timestamp", 
                "sequence_number", "exchange", "tape", "trf_id", "correction"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # normalize conditions to tuple (hashable / consistent)
    if "conditions" in df.columns:
        def _norm_conditions(x):
            if x is None or (isinstance(x, float) and pd.isna(x)):
                return tuple()
            if isinstance(x, list):
                return tuple(int(v) for v in x if v is not None)
            if isinstance(x, (int, float)) and not pd.isna(x):
                return (int(x),)
            return tuple()
        df["conditions"] = df["conditions"].map(_norm_conditions)
    
    # choose timestamp
    if ts_preference.lower() == "participant" and df["participant_timestamp"].notna().any():
        df["ts"] = pd.to_datetime(df["participant_timestamp"], unit="ns", utc=True)
    elif df["sip_timestamp"].notna().any():
        df["ts"] = pd.to_datetime(df["sip_timestamp"], unit="ns", utc=True)          
    elif df["participant_timestamp"].notna().any():
        df["ts"] = pd.to_datetime(df["participant_timestamp"], unit="ns", utc=True)
    else:
        raise ValueError("No usable timestamp field found (sip_timestamp / participant_timestamp).")

    df = df.sort_values(["ts", "sequence_number"], kind="stable").reset_index(drop=True)
    return df


def clean_trades(
    df: pd.DataFrame,
    keep_only_uncorrected: bool = True,
    drop_nonpositive: bool = True,
    dedupe: bool = True,
) -> pd.DataFrame:
    """
    - filters invalid price/size
    - optionally removes corrected prints (correction != 0)
    - robust de-duplication using (symbol, exchange, trf_id, id) since id is unique per that combo
    """
    # drop missing ts
    df = df.dropna(subset=["ts"])

    if drop_nonpositive:
        if "price" in df.columns:
            df = df[df["price"] > 0]
        if "size" in df.columns:
            df = df[df["size"] > 0]

    if keep_only_uncorrected and "correction" in df.columns:
        df = df[(df["correction"].isna()) | (df["correction"] == 0)]

    if dedupe:
        # Preferred dedupe keys
        keys = [c for c in ["symbol", "exchange", "trf_id", "id"] if c in df.columns]
        if "id" not in keys:
            # fallback if some payload lacks id
            keys = [c for c in ["symbol", "exchange", "sequence_number", "sip_timestamp", "participant_timestamp", "price", "size"] if c in df.columns]
        df = df.drop_duplicates(subset=keys, keep="last")

    df = df.sort_values(["ts", "sequence_number"], kind="stable").reset_index(drop=True)
    return df


def resample_trades_1s(
    df_trades: pd.DataFrame
) -> pd.DataFrame:
    """
    - resamples trades data at 1s interval
    - enriches VWAP per second
    """
    df = df_trades.set_index("ts").sort_index()
    
    out = pd.DataFrame(index=df.resample("1s").size().index)
    out["last"] = df["price"].resample("1s").last()
    out["volume"] = df["size"].resample("1s").sum().fillna(0)
    out["n_trades"] = df["size"].resample("1s").count().fillna(0).astype(int)

    # VWAP per second
    dollar = (df["price"] * df["size"]).resample("1s").sum()
    vol = df["size"].resample("1s").sum()
    out["vwap"] = (dollar / vol.replace(0, np.nan))

    # keep symbol
    sym = None
    if "symbol" in df.columns:
        sym_series = df["symbol"].dropna()
        sym = sym_series.iloc[0] if len(sym_series) else None
    if sym is not None:
        out["symbol"] = sym

    out.index.name = "ts"
    return out


def collect_trades_data(
    tickers: str,
    start: str,
    end: str,
    ts_preference: str = "sip"
):
    for ticker in tickers:
        for d in pd.date_range(start, end, inclusive='left'):
            date = d.strftime("%Y-%m-%d")
            trades = get_trades(ticker, date)
            if trades:
                df = (
                    trades_to_df(trades, ticker, ts_preference=ts_preference)
                    .pipe(clean_trades)
                    .pipe(localize_to_eastern)
                    .pipe(filter_rth)
                    .pipe(resample_trades_1s)
                )

                trades_dir = DATA_DIR / "trades"
                trades_dir.mkdir(parents=True, exist_ok=True)

                out_fp = trades_dir / f"{ticker}_trades_1s_{date}.parquet"
                df.index.name = "ts"
                df.reset_index().to_parquet(
                    out_fp,
                    engine="pyarrow",
                    compression="zstd"
                )

                print("Saved:", out_fp)



if __name__ == "__main__":
    # query params
    tickers = [ "XLF"]
    timespan = "second"
    start = "2025-01-01"
    end = "2025-03-31"

    # ohlcv
    collect_price_data(tickers, timespan, start, end)

    # nbbo
    collect_nbbo_data(tickers, start, end)

    # trades
    collect_trades_data(tickers, start, end)
