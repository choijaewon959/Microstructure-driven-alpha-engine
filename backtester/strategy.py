from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List
import pandas as pd
import numpy as np


@dataclass(frozen=True)
class Signal:
    symbol: str
    target_pos: int
    style: str  # "taker" | "maker" | "none"
    urgency: float  # 0..1 (used by exec model to adjust slippage/impact)


class Strategy(ABC):
    def __init__(self):
        pass

    def on_start(self):
        """Called once before backtest starts"""
        pass

    @abstractmethod
    def generate_signals(
        self,
        features: dict,
        portfolio_state: dict
    ) -> List[Signal]:
        """
        Return target position (NOT order qty)

        Examples:
        - +100 shares
        - -1 futures contract
        - 0 flat
        """
        pass

    def on_end(self):
        """Called once after backtest ends"""
        pass


class MicrostructureStrategy(Strategy):
    def __init__(self, style: str = "taker", max_pos: int = 200):
        self.max_pos = max_pos
        self.style = style
        self.last_trade_ts = None

    def generate_signals(
        self,
        features: dict,
        portfolio_state: dict
    ) -> List[Signal]:
        mid = features['micro_mid']
        spread = features['micro_spread']
        microprice = features['micro_microprice']
        sig = (microprice - mid) / max(spread, 1e-9)
        sym = features['symbol']

        # 2) cost gate (simplified)
        taker_fee = 0.0005   # $0.0005/share = 0.05 cents
        maker_fee = -0.0001  # small rebate example (optional)

        fees = taker_fee if self.style == "taker" else maker_fee
        expected_edge = abs(microprice - mid)
        min_edge = spread/2 + fees  # rough "must beat" threshold

        # 3) risk/inventory
        pos = portfolio_state["num_positions"]

        if spread <= 0:
            return [Signal(sym, pos, "none", 0.0)]

        # 4) decision logic
        if expected_edge < min_edge:
            return [Signal(sym, pos, "none", 0.0)]

        if sig > 0.3 and pos < self.max_pos:
            # book pressure up: want more long
            target = min(self.max_pos, pos + 50)
            return [Signal(sym, target, "taker", urgency=min(1.0, sig))]
        elif sig < -0.3 and pos > -self.max_pos:
            target = max(-self.max_pos, pos - 50)
            return [Signal(sym, target, "taker", urgency=min(1.0, -sig))]

        return [Signal(sym, pos, "none", 0.0)]


class RVStrategy(Strategy):
    def __init__(
        self,
        market: str = "SPY",
        z_enter: float = 2.0,
        z_exit: float = 0.5
    ):
        self.market = market
        self.z_enter = z_enter
        self.z_exit = z_exit
        
        # @TODO: update size determination
        self.base_size = 5000

        # @TODO: Erase
        self.signals = []

    def generate_signals(self, features: dict, portfolio_state: dict) -> List[Signal]:
        if not features:
            return []

        out: List[Signal] = []
        spy_target_total = 0
        urgency_max = 0.0

        for sym, info in features.items():
            z = info.get("z", None)
            beta = info.get("beta", None)
            mid = info.get("mid", None)
            market_mid = info.get("mid_market", None)

            if z is None or beta is None or mid is None or market_mid is None:
                continue
            if not (np.isfinite(z) and np.isfinite(beta) and np.isfinite(mid) and np.isfinite(market_mid)):
                continue
            if mid <= 0 or market_mid <= 0:
                continue

            current_qty = int(portfolio_state.get(sym, 0))
            target = None

            if current_qty == 0:
                if z >= self.z_enter:
                    target = -self.base_size
                elif z <= -self.z_enter:
                    target = +self.base_size
                else:
                    continue
            else:
                if abs(z) <= self.z_exit:
                    target = 0
                else:
                    continue  # hold

            hedge_target = int(round(-beta * target * (mid / market_mid)))
            spy_target_total += hedge_target

            urgency = min(1.0, abs(z) / (self.z_enter + 1e-12))
            urgency_max = max(urgency_max, urgency)

            out.append(Signal(sym, int(target), "taker", urgency))

        if out:
            out.append(Signal(self.market, int(spy_target_total), "taker", urgency_max))

        return out

