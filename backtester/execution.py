from typing import Literal
import pandas as pd
from models import Fill

class ExecutionSimulator:
    """
    Immediate full-fill execution simulator.
    Assumes top-of-book execution with fixed latency and fee.
    """

    def __init__(
        self,
        latency: int | pd.Timedelta = 1,
        exec_style: Literal["taker", "maker"] = "taker",
        fee_per_share: float = 0.001,
    ):
        if isinstance(latency, int):
            latency = pd.Timedelta(milliseconds=latency)

        self.latency = latency
        self.exec_style = exec_style
        self.fee_per_share = fee_per_share

    def generate_fills(
        self,
        ts: pd.Timestamp,
        symbol: str,
        order_qty: float,
        row: pd.Series,
        style: str,
        urgency: float,
    ) -> list[Fill]:
        if order_qty == 0:
            return []

        if pd.isna(row["q_bid_price"]) or pd.isna(row["q_ask_price"]):
            return []

        if self.exec_style == "taker":
            px = row["q_ask_price"] if order_qty > 0 else row["q_bid_price"]
        else:  # maker
            px = row["q_bid_price"] if order_qty > 0 else row["q_ask_price"]

        return [
            Fill(
                ts=ts + self.latency,
                symbol=symbol,
                qty=order_qty,
                price=px,
                fee=abs(order_qty) * self.fee_per_share,
            )
        ]
