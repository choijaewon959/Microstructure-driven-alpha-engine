import pandas as pd
import numpy as np
from collections import deque
from abc import ABC, abstractmethod
from models import QuoteEvent, TradeEvent, MarketState, UnknownEventError, UnknownSymbolError
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
    

class RVModule(FeatureModule):
    def __init__(
        self,
        prefix: str = "RV",
        stocks: list[str] = [],
        market: str = "SPY",
        window: int = 300,
    ):
        self.prefix = prefix
        self.stocks = stocks
        self.market = market
        self.window = window

        # rolling mid buffers per symbol
        self.mid_hist: Dict[str, deque] = {
            s: deque(maxlen=window + 1) for s in [market] + self.stocks
        }

        # latest snapshot of rv
        self._latest: Dict[str, Any] = {}

    def on_event(
        self, 
        event: QuoteEvent | TradeEvent,
        ms: MarketState
    ) -> None:
        # symbol fallback
        symbol = ms.quote.symbol or ms.trade.symbol

        # quotes
        b   = _safe_float(ms.quote.bid, np.nan)
        a   = _safe_float(ms.quote.ask, np.nan)

        has_quote = not (np.isnan(b) or np.isnan(a))
        if has_quote:
            mid = 0.5 * (b + a)
        else:
            mid = np.nan

        # update historical mid
        if symbol not in self.mid_hist:
            self.mid_hist[symbol] = deque(maxlen=self.window + 1)

        self.mid_hist[symbol].append(mid)

        # compute relative value
        if all(len(self.mid_hist[sym]) >= self.window + 1 for sym in self.mid_hist):
            self._compute_rv()

    def _compute_log_ret(
        self,
        sym: str
    ) -> float:
        px = np.asarray(self.mid_hist[sym], dtype=float)
        return np.diff(np.log(px))
    
    def _ols_alpha_beta(
        self,
        y: np.ndarray,
        x: np.ndarray
    ) -> tuple[float, float]:
        x_bar, y_bar = x.mean(), y.mean()
        dx = x - x_bar
        dy = y - y_bar
        
        var_x = (dx * dx).mean()
        if var_x < 1e-12:
            return 0.0, 0.0
        
        cov_xy = (dx * dy).mean()
        b = cov_xy / (var_x + 1e-12)
        a = y_bar - b * x_bar
        return a, b

    def _compute_rv(
        self,
    ) -> None:
        r_m = self._compute_log_ret(self.market)[-self.window:]
        var_m = r_m.var(ddof=1)
        if var_m < 1e-10:
            return
        
        out: Dict[str, Dict[str, Any]] = {}
        for s in self.stocks:
            r = self._compute_log_ret(s)[-self.window:]
            alpha, beta = self._ols_alpha_beta(r, r_m)            
            epsilon = r - (alpha + beta * r_m)
            sigma_eps = epsilon.std(ddof=1)
            z = float(epsilon[-1] / (sigma_eps + 1e-12)) if sigma_eps > 1e-12 else 0.0
            
            out[s] = {
                "mid": self.mid_hist[s][-1],
                "mid_market": self.mid_hist[self.market][-1],
                "alpha": alpha,
                "beta": beta,
                "eps": float(epsilon[-1]),
                "z": z
            }

        self._latest = out

    def snapshot(
        self,
        ms: MarketState
    ) -> Dict[str, Any]:
        return self._latest


class CompositeFeatureEngine:
    def __init__(
        self,
        modules: list[FeatureModule]
    ):
        self.updater = MarketStateUpdater()
        self.modules = modules
        self.ms_by_symbol: Dict[str, MarketState] = {}

    def on_event(
        self,
        event: QuoteEvent | TradeEvent
    ) -> Dict[str, Any]:
        sym = event.symbol
        ms = self.ms_by_symbol.setdefault(sym, MarketState())

        # 1) update raw state for THIS symbol only
        self.updater.apply(ms, event)

        # 2) update rolling state per module (module can manage per-symbol memory)
        for m in self.modules:
            m.on_event(event, ms)

        # 3) snapshot features
        feats: Dict[str, Any] = {}
        for m in self.modules:
            feats.update(m.snapshot(ms))

        return feats

    def market_state(self, symbol: str) -> MarketState:
        return self.ms_by_symbol.setdefault(symbol, MarketState())


