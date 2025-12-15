import pandas as pd
import os 
from massive import RESTClient
from dotenv import load_dotenv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"

load_dotenv()
API_KEY = os.getenv("POLYGON_API_KEY")


def get_ohlcv(ticker: str, timespan: str, start: str, end: str):
    client = RESTClient(API_KEY)

    aggs = []
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


if __name__ == "__main__":
    # query params
    tickers = ["GS", "MS"]
    timespan = "second"
    start = "2024-01-02"
    end = "2024-01-31"

    for ticker in tickers:
        aggs = get_ohlcv(ticker, timespan, start, end)
        df = (
            parse_aggs(aggs, ticker)
            .pipe(localize_to_eastern)
            .pipe(filter_rth)
            .pipe(finalize_bars)
            .pipe(reindex_1s)
        )

        DATA_DIR.mkdir(parents=True, exist_ok=True)

        out_fp = DATA_DIR / f"{ticker}_1s_{start}_{end}.csv"
        df.reset_index().to_csv(out_fp, index=False)
        print("Saved:", out_fp)