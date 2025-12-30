from portfolio import Portfolio
from strategy import *
from execution import ExecutionSimulator
from features import MicrostructureModule, CompositeFeatureEngine
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
        all_ts = quotes.index.union(trades.index)

        for ts in all_ts:
            q = quotes.loc[ts]
            t = trades.loc[ts]

            # 1. apply fills - handle pending fills that need to be executed at current ts
            for fill in [f for f in self.pending_fills if f.ts == ts]:
                self.portfolio.apply_fill(fill)
            
            self.pending_fills = [f for f in self.pending_fills if f.ts > ts]

            # 2. enrich features
            qe = QuoteEvent(
                    ts = ts,
                    symbol = t["t_symbol"],
                    bid = q["q_bid_price"],
                    ask = q["q_ask_price"],
                    bsz = q["q_bid_size"],
                    asz = q["q_ask_size"],
                )
            te = TradeEvent(
                    ts = ts,
                    symbol = t["t_symbol"],
                    last = t["t_last"],
                    volume = t["t_volume"],
                    n_trades = t["t_n_trades"],
                    vwap = t['t_vwap']
                )
            
            features = {}
            for event in [qe, te]:
                features = self.preprocessor.on_event(event)


            # 3. generate signal
            sig: Signal = self.strategy.generate_signals(features, self.portfolio.snapshot())
            

if __name__ == "__main__":
    # build engine
    market_state = MarketState()
    modules = [MicrostructureModule()]
    feature_engine = CompositeFeatureEngine(market_state, modules)
    strategy = MicrostructureStrategy()
    exec_sim = ExecutionSimulator()

    engine = Engine(
        feature_engine,
        strategy,
        Portfolio(),
        ExecutionSimulator()
    )

    # get data
    BASE = Path("../data")

    ticker = "IVV"
    date = '2025-05-02'

    quotes_fp = BASE / "quotes" / f"{ticker}_quotes_1s_{date}.parquet"
    prices_fp = BASE / "prices" / f"{ticker}_1s_{date}.parquet"
    trades_fp = BASE / "trades" / f"{ticker}_trades_1s_{date}.parquet"
    q = pd.read_parquet(quotes_fp).set_index('ts').add_prefix("q_")
    t = pd.read_parquet(trades_fp).set_index('ts').add_prefix("t_")

    # run
    print(engine.run(q, t))            
