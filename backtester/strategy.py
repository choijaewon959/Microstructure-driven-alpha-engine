from abc import ABC, abstractmethod
from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class Signal:
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
        ts: pd.Timestamp,
        row: pd.Series,
        portfolio_state: dict
    ) -> Signal:
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

    def generate_signals(self, features: dict, portfolio_state: dict) -> Signal:
        mid = features['micro_mid']
        spread = features['micro_spread']
        microprice = features['micro_microprice']
        sig = (microprice - mid) / max(spread, 1e-9)

        # 2) cost gate (simple version)
        taker_fee = 0.0005   # $0.0005/share = 0.05 cents
        maker_fee = -0.0001  # small rebate example (optional)

        fees = taker_fee if self.style == "taker" else maker_fee
        expected_edge = abs(microprice - mid)
        min_edge = spread/2 + fees  # rough "must beat" threshold

        # 3) risk/inventory
        pos = portfolio_state["num_positions"]

        if spread <= 0:
            return Signal(pos, "none", 0.0)

        # 4) decision logic
        if expected_edge < min_edge:
            return Signal(pos, "none", 0.0)

        if sig > 0.3 and pos < self.max_pos:
            # book pressure up: want more long
            target = min(self.max_pos, pos + 50)
            return Signal(target, "taker", urgency=min(1.0, sig))
        elif sig < -0.3 and pos > -self.max_pos:
            target = max(-self.max_pos, pos - 50)
            return Signal(target, "taker", urgency=min(1.0, -sig))

        return Signal(pos, "none", 0.0)
