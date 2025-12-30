import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from models import QuoteEvent, TradeEvent, MarketState, UnknownEventError 
from typing import Any, Dict


EPS = 1e-12

def _safe_float(x, default=np.nan):
    if x is None:
        return default
    try:
        return default if np.isnan(x) else float(x)
    except TypeError:
        return float(x)

def _safe_size(x, default=0.0):
    # sizes: default to 0 if missing or NaN
    if x is None:
        return default
    try:
        return default if np.isnan(x) else float(x)
    except TypeError:
        return float(x)


class MarketStateUpdater:
    @staticmethod
    def apply(
        ms: MarketState, 
        event: QuoteEvent | TradeEvent,
    ) -> None:
        if isinstance(event, QuoteEvent):
            ms.quote.symbol = event.symbol
            ms.quote.bid = event.bid
            ms.quote.ask = event.ask
            ms.quote.bsz = event.bsz
            ms.quote.asz = event.asz
            ms.quote.last_ts = event.ts
            return

        if isinstance(event, TradeEvent):
            ms.trade.symbol = event.symbol
            ms.trade.last = event.last
            ms.trade.volume = event.volume
            ms.trade.n_trades = event.n_trades
            ms.trade.last_ts = event.ts
            return

        raise UnknownEventError("Unknown event detected.", ts=getattr(event, "ts", None), symbol=getattr(event, "symbol", None))


class FeatureModule(ABC):
    @abstractmethod
    def on_event(self, event, ms) -> None:
        """Update any internal rolling state needed by this module."""
        pass

    @abstractmethod
    def snapshot(self, ms: MarketState) -> Dict[str, Any]:
        """Return current feature values."""
        pass


class MicrostructureModule(FeatureModule):
    def __init__(self, prefix: str = "micro"):
        self.prefix = prefix

    def on_event(self, event: QuoteEvent | TradeEvent, ms: MarketState) -> None:
        return  # no rolling state here

    def snapshot(self, ms: MarketState) -> Dict[str, Any]:
        # symbol fallback
        symbol = ms.quote.symbol or ms.trade.symbol

        # quotes
        b   = _safe_float(ms.quote.bid, np.nan)
        a   = _safe_float(ms.quote.ask, np.nan)
        bsz = _safe_size(ms.quote.bsz, 0.0)
        asz = _safe_size(ms.quote.asz, 0.0)

        has_quote = not (np.isnan(b) or np.isnan(a))
        if has_quote:
            mid = 0.5 * (b + a)
            spread = a - b
            denom = bsz + asz
            imbalance_l1 = (bsz - asz) / (denom + EPS)
            microprice = (a * bsz + b * asz) / (denom + EPS)
            microprice_dev = microprice - mid
        else:
            mid = np.nan
            spread = np.nan
            imbalance_l1 = 0.0
            microprice = np.nan
            microprice_dev = np.nan

        # trades (1s bar)
        volume = _safe_size(ms.trade.volume, 0.0)
        n_trades = int(ms.trade.n_trades or 0)
        last = _safe_float(ms.trade.last, np.nan)
        vwap  = _safe_float(ms.trade.vwap, 0.0)
        
        return {
            "symbol": symbol,
            f"{self.prefix}_has_quote": has_quote,
            f"{self.prefix}_bid_price": b,
            f"{self.prefix}_ask_price": a,
            f"{self.prefix}_bid_size": bsz,
            f"{self.prefix}_ask_size": asz,
            f"{self.prefix}_mid": mid,
            f"{self.prefix}_spread": spread,
            f"{self.prefix}_imbalance_l1": imbalance_l1,
            f"{self.prefix}_microprice": microprice,
            f"{self.prefix}_microprice_dev": microprice_dev,
            f"{self.prefix}_n_trades": n_trades,
            f"{self.prefix}_market_volume": volume,
            f"{self.prefix}_last": last,
            f"{self.prefix}_vwap_proxy": vwap,
        }


class CompositeFeatureEngine:
    def __init__(
        self, 
        market_state: MarketState, 
        modules: list[FeatureModule]
    ) -> None:
        self.ms = market_state
        self.updater = MarketStateUpdater()
        self.modules = modules

    def on_event(
        self,
        event: QuoteEvent | TradeEvent,
    ) -> Dict[str, Any]:
        # 1) update raw state once
        self.updater.apply(self.ms, event)

        # 2) let each module update its rolling memory (if any)
        for m in self.modules:
            m.on_event(event, self.ms)

        # 3) produce a unified feature snapshot
        feats: Dict[str, Any] = {}
        for m in self.modules:
            feats.update(m.snapshot(self.ms))

        return feats


