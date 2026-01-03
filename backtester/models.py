import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


### ========
### Events
### ========

@dataclass(frozen=True)
class QuoteEvent:
    ts: pd.Timestamp
    symbol: str
    bid: float
    ask: float
    bsz: float
    asz: float


@dataclass(frozen=True)
class TradeEvent:
    ts: pd.Timestamp
    symbol: str
    last: float
    volume: int | float
    n_trades: int
    vwap: float

### ========
### States
### ========

@dataclass
class QuoteState:
    symbol: str = ''
    bid: float = np.nan
    ask: float = np.nan
    bsz: float = 0.0
    asz: float = 0.0
    last_ts: Optional[pd.Timestamp] = None

    @property
    def mid(self) -> float:
        return 0.5 * (self.bid + self.ask)

    @property
    def spread(self) -> float:
        return self.ask - self.bid


@dataclass
class TradeState:
    symbol: str = ''
    last: float = np.nan
    volume: int | float = 0
    n_trades: int = 0
    vwap: float = 0.0
    last_ts: Optional[pd.Timestamp] = None

@dataclass
class MarketState:
    quote: QuoteState = field(default_factory=QuoteState)
    trade: TradeState = field(default_factory=TradeState)


### ========
### Objects
### ========

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


### ============
### Exceptions
### ============
class UnknownEventError(Exception):
    """Unknown event type error."""
    pass

class UnknownSymbolError(Exception):
    """Unexpected symbol type error."""
    pass