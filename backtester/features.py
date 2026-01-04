import pandas as pd
import numpy as np
from collections import deque
from abc import ABC, abstractmethod
from models import QuoteEvent, TradeEvent, MarketState, UnknownEventError, UnknownSymbolError
from typing import Any, Dict, List, Optional


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
    def on_bar(self, event, ms) -> None:
        """Update any internal rolling state needed by this module."""
        pass

    @abstractmethod
    def snapshot(self, ms: MarketState) -> Dict[str, Any]:
        """Return current feature values."""
        pass


class MicrostructureModule(FeatureModule):
    def __init__(self, prefix: str = "micro", universe: Optional[List[str]] = None):
        self.prefix = prefix
        self.universe = universe or []
        self._latest: Dict[str, Any] = {}

    def on_bar(self, ts: pd.Timestamp, ms_by_symbol: Dict[str, MarketState]) -> None:
        out: Dict[str, Any] = {"ts": ts}
        for sym in self.universe:
            ms = ms_by_symbol.get(sym)
            if ms is None:
                continue

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

            volume = _safe_size(ms.trade.volume, 0.0)
            n_trades = int(ms.trade.n_trades or 0)
            last = _safe_float(ms.trade.last, np.nan)
            vwap  = _safe_float(getattr(ms.trade, "vwap", None), 0.0)

            # namespaced by symbol
            out[f"{sym}_{self.prefix}_has_quote"] = has_quote
            out[f"{sym}_{self.prefix}_bid_price"] = b
            out[f"{sym}_{self.prefix}_ask_price"] = a
            out[f"{sym}_{self.prefix}_bid_size"] = bsz
            out[f"{sym}_{self.prefix}_ask_size"] = asz
            out[f"{sym}_{self.prefix}_mid"] = mid
            out[f"{sym}_{self.prefix}_spread"] = spread
            out[f"{sym}_{self.prefix}_imbalance_l1"] = imbalance_l1
            out[f"{sym}_{self.prefix}_microprice"] = microprice
            out[f"{sym}_{self.prefix}_microprice_dev"] = microprice_dev
            out[f"{sym}_{self.prefix}_n_trades"] = n_trades
            out[f"{sym}_{self.prefix}_market_volume"] = volume
            out[f"{sym}_{self.prefix}_last"] = last
            out[f"{sym}_{self.prefix}_vwap_proxy"] = vwap

        self._latest = out

    def snapshot(self) -> Dict[str, Any]:
        return self._latest
    

class RVModule(FeatureModule):
    def __init__(
        self,
        prefix: str = "RV",
        stocks: Optional[List[str]] = None,
        market: str = "SPY",
        window: int = 300,
    ):
        self.prefix = prefix
        self.stocks = stocks or []
        self.market = market
        self.window = window
        self.symbols = [self.market] + self.stocks

        # rolling mid buffers per symbol: (window+1) sized prices -> (window) sized returns
        self.mid_hist: Dict[str, deque] = {s: deque(maxlen=window+1) for s in self.symbols}
        self._latest: Dict[str, Any] = {}

    def on_bar(
        self,
        ts: pd.Timestamp,
        ms_by_symbol: Dict[str, MarketState]
    ) -> None:
        # append one aligned sample per ts for ALL symbols
        for sym in self.symbols:
            ms = ms_by_symbol.get(sym)
            if ms is None:
                self.mid_hist[sym].append(np.nan)
                continue

            b = _safe_float(ms.quote.bid, np.nan)
            a = _safe_float(ms.quote.ask, np.nan)
            has_quote = not (np.isnan(b) or np.isnan(a))
            mid = 0.5 * (b + a) if has_quote else np.nan
            self.mid_hist[sym].append(mid)

        # compute only if everyone has window+1 prices (=> window returns)
        if all(len(self.mid_hist[s]) >= self.window for s in self.symbols):
            self._compute_rv(ts)

    @staticmethod
    def _compute_log_ret_from_prices(
        px: np.ndarray
    ) -> np.ndarray:
        px = np.asarray(px, dtype=float)
        px = np.where(np.isfinite(px) & (px > 0), px, np.nan)

        # forward-fill in-array (requires at least one finite)
        if np.isnan(px).all():
            return np.array([], dtype=float)

        mask = np.isnan(px)
        if mask.any():
            idx = np.where(~mask, np.arange(len(px)), 0)
            np.maximum.accumulate(idx, out=idx)
            px = px[idx]

        return np.diff(np.log(px))
    
    @staticmethod
    def _ols_alpha_beta(
        y: np.ndarray,
        x: np.ndarray
    ) -> tuple[float, float]:
        x_bar, y_bar = x.mean(), y.mean()
        dx = x - x_bar
        dy = y - y_bar
        
        var_x = (dx * dx).mean()
        if var_x < EPS:
            return 0.0, 0.0
        
        cov_xy = (dx * dy).mean()
        b = cov_xy / (var_x + EPS)
        a = y_bar - b * x_bar
        return a, b

    def _compute_rv(
        self, 
        ts: pd.Timestamp
    ) -> None:
        # market returns
        px_m = np.asarray(self.mid_hist[self.market], dtype=float)
        r_m = self._compute_log_ret_from_prices(px_m)

        if len(r_m) < self.window:
            return
        
        r_m = r_m[-self.window:]

        if np.var(r_m, ddof=1) < 1e-10:
            return

        out: Dict[str, Dict[str, Any]] = {}
        market_mid = float(px_m[-1])

        for s in self.stocks:
            px = np.asarray(self.mid_hist[s], dtype=float)
            r = self._compute_log_ret_from_prices(px)
            if len(r) < self.window:
                continue

            r = r[-self.window:]  # aligned length = window
            alpha, beta = self._ols_alpha_beta(r, r_m)
            eps = r - (alpha + beta * r_m)

            # series = np.cumsum(eps) if self.use_cum_residual else eps
            series = eps
            mu = series.mean()
            sd = series.std(ddof=1)

            z = float((series[-1] - mu) / (sd + EPS)) if sd > EPS else 0.0

            out[s] = {
                "ts": ts,
                "mid": float(px[-1]),
                "mid_market": market_mid,
                "alpha": float(alpha),
                "beta": float(beta),
                "eps": float(eps[-1]),
                "z": z,
                # "z_type": "cum_eps" if self.use_cum_residual else "eps",
                "z_type": "eps",
                "window": self.window,
            }

        self._latest = out

    def snapshot(
        self,
    ) -> Dict[str, Any]:
        return self._latest


class CompositeFeatureEngine:
    def __init__(
        self,
        modules: List[FeatureModule],
        universe: List[str],
    ):
        self.updater = MarketStateUpdater()
        self.modules = modules
        self.ms_by_symbol: Dict[str, MarketState] = {s: MarketState() for s in universe}
        self._cur_ts: Optional[pd.Timestamp] = None

    def on_event(
        self,
        event: QuoteEvent | TradeEvent
    ) -> Optional[Dict[str, Any]]:
        ts = event.ts
        sym = event.symbol

        # ignore symbols outside universe
        if sym not in self.ms_by_symbol:
            return None

        # flush bar when ts changes
        if self._cur_ts is None:
            self._cur_ts = ts
        elif ts != self._cur_ts:
            # process ts-1 events first
            feats = self._flush_bar(self._cur_ts)

            # now process the new event after flushing
            # (fall through to apply event below)
            self._cur_ts = ts
            self.updater.apply(self.ms_by_symbol[sym], event)

            return feats

        # same ts: just update state
        self.updater.apply(self.ms_by_symbol[sym], event)
        return None

    def _flush_bar(
        self,
        ts: pd.Timestamp
    ) -> Dict[str, Any]:
        for m in self.modules:
            m.on_bar(ts, self.ms_by_symbol)

        feats: Dict[str, Any] = {}
        for m in self.modules:
            feats.update(m.snapshot())
        return feats

    def close(self) -> Optional[Dict[str, Any]]:
        # call at end of stream to flush last ts
        if self._cur_ts is None:
            return None
        
        return self._flush_bar(self._cur_ts)



