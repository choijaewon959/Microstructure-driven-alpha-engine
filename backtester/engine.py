from portfolio import Portfolio
from strategy import *
from execution import ExecutionSimulator
from features import MicrostructureModule, RVModule, CompositeFeatureEngine
from models import QuoteEvent, TradeEvent, MarketState
from pathlib import Path
import pandas as pd


class Engine:
    def __init__(
        self,
        preprocessor: CompositeFeatureEngine,
        strategy: Strategy,
        portfolio: Portfolio,
        exec_sim: ExecutionSimulator,
    ):
        self.preprocessor = preprocessor
        self.strategy = strategy
        self.portfolio = portfolio
        self.exec_sim = exec_sim
        self.pending_fills = []

    def run(
        self,
        quotes: pd.DataFrame,
        trades: pd.DataFrame
    ) -> None:
        """
        quotes: 1s tick data for quotes
        trades: 1s tick data for trades

        Ideally, these quotes and trades are already preprocessed such that the shape[0] is matching
        """
        quotes_e = quotes.copy()
        quotes_e["event_type"] = "quote"

        trades_e = trades.copy()
        trades_e["event_type"] = "trade"

        events = (
            pd.concat([quotes_e, trades_e], axis=0)
            .sort_index()  # assumes ts is index
        )
        
        sigs = []
        for ts, row in events.iterrows():
            # 1) apply fills due at ts
            due = [f for f in self.pending_fills if f.ts == ts]
            for fill in due:
                self.portfolio.apply_fill(fill)
            self.pending_fills = [f for f in self.pending_fills if f.ts > ts]

            # 2) construct event + enrich features
            if row["event_type"] == "quote":
                qe = QuoteEvent(
                    ts=ts,
                    symbol=row["q_symbol"],
                    bid=row["q_bid_price"],
                    ask=row["q_ask_price"],
                    bsz=row["q_bid_size"],
                    asz=row["q_ask_size"],
                )
                features = self.preprocessor.on_event(qe)
            else:
                te = TradeEvent(
                    ts=ts,
                    symbol=row["t_symbol"],
                    last=row["t_last"],
                    volume=row["t_volume"],
                    n_trades=row["t_n_trades"],
                    vwap=row["t_vwap"],
                )
                features = self.preprocessor.on_event(te)

            # 3. generate signal
            sig: List[Signal] = self.strategy.generate_signals(features, self.portfolio.snapshot())
            if sig:
                sigs.append(sig)
        
        print(len(sigs))
            



if __name__ == "__main__":
    # build engine
    # modules = [MicrostructureModule()]
    # strategy = MicrostructureStrategy()

    modules =[RVModule(prefix="RV", stocks=["MS"])]
    feature_engine = CompositeFeatureEngine(modules, ["SPY", "MS"])

    engine = Engine(
        feature_engine,
        RVStrategy(),
        Portfolio(),
        ExecutionSimulator()
    )

    # get data
    BASE = Path("../data")
    ticker = "MS"
    date = '2025-06-02'

    quotes_fp = BASE / "quotes" / f"{ticker}_quotes_1s_{date}.parquet"
    prices_fp = BASE / "prices" / f"{ticker}_1s_{date}.parquet"
    trades_fp = BASE / "trades" / f"{ticker}_trades_1s_{date}.parquet"
    q_s = pd.read_parquet(quotes_fp).set_index('ts').add_prefix("q_")
    t_s = pd.read_parquet(trades_fp).set_index('ts').add_prefix("t_")
    q_s['symbol'] = ticker
    t_s['symbol'] = ticker

    ticker = "SPY"
    quotes_fp = BASE / "quotes" / f"{ticker}_quotes_1s_{date}.parquet"
    prices_fp = BASE / "prices" / f"{ticker}_1s_{date}.parquet"
    trades_fp = BASE / "trades" / f"{ticker}_trades_1s_{date}.parquet"
    q_spy = pd.read_parquet(quotes_fp).set_index('ts').add_prefix("q_")
    t_spy = pd.read_parquet(trades_fp).set_index('ts').add_prefix("t_")
    q_spy['symbol'] = ticker
    t_spy['symbol'] = ticker

    ts = q_s.index.intersection(q_spy.index)
    q = pd.concat([q_s, q_spy]).loc[ts]
    q.sort_index(inplace=True)
    
    t = pd.concat([t_s, t_spy]).loc[ts]
    t.sort_index(inplace=True)

    # run
    engine.run(q, t)
