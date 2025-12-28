import pandas as pd
from dataclasses import dataclass


@dataclass
class Position:
    symbol: str
    qty: float
    avg_px: float
    realized_pnl: float
    unrealized_pnl: float
    last_mark: float        # last mid price 


@dataclass
class Fill:
    ts: pd.Timestamp        # execution time
    symbol: str            
    qty: float              # signed quantity (+buy, -sell)
    price: float            # executed price
    fee: float = 0.0        # commissions / fees