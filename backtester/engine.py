from portfolio import Portfolio
from strategy import *
from execution import ExecutionSimulator
from features import MicrostructureModule, RVModule, CompositeFeatureEngine
from models import QuoteEvent, TradeEvent, MarketState
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import numpy as np


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
        self.last_mid: Dict[str, float] = {}

    def _apply_due_fills(
        self,
        ts: pd.Timestamp,
    ) -> None:
        due = [f for f in self.pending_fills if f.ts <= ts]
        for fill in due:
            self.portfolio.apply_fill(fill)
        self.pending_fills = [f for f in self.pending_fills if f.ts > ts]

    def _mid_from_ms(
        self,
        ms: MarketState,
    ) -> float:
        b = float(ms.quote.bid) if ms.quote.bid is not None else np.nan
        a = float(ms.quote.ask) if ms.quote.ask is not None else np.nan
        if np.isfinite(b) and np.isfinite(a) and b > 0 and a > 0:
            return 0.5 * (b + a)
        return np.nan

    def _update_mid_cache(
        self,
    ) -> None:
        symbols = self.preprocessor.market_states().keys()
        if symbols is None:
            # fallback: mark-to-market only open positions
            symbols = list(self.portfolio.snapshot().keys())

        for sym in symbols:
            ms = self.preprocessor.market_states()[sym]
            mid = self._mid_from_ms(ms)
            if np.isfinite(mid):
                self.last_mid[sym] = float(mid)

    def _execute_signals(
        self, 
        ts: pd.Timestamp,
        signals: List[Any]
    ) -> None:
        port_state = self.portfolio.snapshot()

        for s in signals:
            sym = s.symbol
            target = int(s.target_pos)
            cur = int(port_state.get(sym, 0))
            order_qty = target - cur
            if order_qty == 0:
                continue

            ms = self.preprocessor.market_states().get(sym)
            b = float(ms.quote.bid) if ms.quote.bid is not None else np.nan
            a = float(ms.quote.ask) if ms.quote.ask is not None else np.nan
            if not (np.isfinite(b) and np.isfinite(a)):
                continue

            exec_row = {"q_bid_price": b, "q_ask_price": a}

            new_fills = self.exec_sim.generate_fills(
                ts=ts,
                symbol=sym,
                order_qty=order_qty,
                row=exec_row,
                style=s.style,
                urgency=float(s.urgency),
            )
            self.pending_fills.extend(new_fills)

    def _mark_to_market(
        self,
        ts: pd.Timestamp,
    ) -> None:
        # mark only what we have
        marks = {sym: px for sym, px in self.last_mid.items() if np.isfinite(px)}
        if marks:
            self.portfolio.mark_to_market(ts, marks)
        
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
        
        features = None
        records: List[Dict[str, Any]] = []

        for ts, row in events.iterrows():
            # 1) apply fills due at ts
            self._apply_due_fills(ts)

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
            
            # 3) update MTM cache from preprocessor states, then MTM
            self._update_mid_cache()
            self._mark_to_market(ts)
            
            # 4)  only act on bar-flush features (aligned)
            if not features:
                continue
            signals: List[Signal] = self.strategy.generate_signals(features, self.portfolio.positions_snapshot())

            # 5) execute + schedule fill
            if signals:
                self._execute_signals(ts, signals)
            
            # 6) optional logging row (features + nav)
            rec = {"ts": ts, "nav": self.portfolio.nav()}
            rec["features"] = features
            records.append(rec)

        # flush last bar
        last = self.preprocessor.close()
        if last:
            ts_last = last.get("ts", events.index[-1] if len(events.index) else None)
            if ts_last is not None:
                self._apply_due_fills(ts_last)
                self._update_mid_cache()
                self._mark_to_market(ts_last)

                signals = self.strategy.generate_signals(last, self.portfolio.positions_snapshot())
                if signals:
                    self._execute_signals(ts_last, signals)
                
                records.append({"ts": ts_last, "nav": self.portfolio.nav(), "features": last})

        return pd.DataFrame(records)


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
    records = engine.run(q, t)
    print(records)
