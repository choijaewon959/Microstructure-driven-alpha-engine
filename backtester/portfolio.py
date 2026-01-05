from models import Position, Fill
import numpy as np

class Portfolio:
    def __init__(self, initial_cash: int | float = 1_000_000, currency: str = 'USD'):
        self.start_cash = initial_cash
        self.cash = initial_cash
        self.positions = {}  # symbol -> Position
        self.base_ccy = currency
        self.fees_paid: float = 0.0
        self.turnover: float = 0.0  # sum(|qty|*price)

        # history
        self.history_ts = []
        self.history_equity = []
        self.history_cash = []
        self.history_gross = []
        self.history_net = []

    def mark_to_market(self, ts, marks: dict[str, float]):
        gross = 0.0
        net = 0.0
        unrealized_total = 0.0
        realized_total = sum(p.realized_pnl for p in self.positions.values())

        for symbol, pos in self.positions.items():
            # update mark if provided, else keep previous
            if symbol in marks:
                mark = float(marks[symbol])
                if np.isfinite(mark):
                    pos.last_mark = mark

            # if still no valid mark, skip valuation for this position
            if pos.last_mark is None or (isinstance(pos.last_mark, float) and not np.isfinite(pos.last_mark)):
                continue

            mark = float(pos.last_mark)

            pos.unrealized_pnl = float(pos.qty) * (mark - float(pos.avg_px))

            gross += abs(float(pos.qty) * mark)
            net += float(pos.qty) * mark
            unrealized_total += float(pos.unrealized_pnl)

        equity = float(self.cash) + unrealized_total + realized_total

        self.history_ts.append(ts)
        self.history_equity.append(equity)
        self.history_cash.append(float(self.cash))
        self.history_gross.append(gross)
        self.history_net.append(net)

    def apply_fill(self, fill: Fill):
        """
        Updates cash/positions/realized PnL given an executed fill.
        Conventions:
          - fill.qty > 0 is buy, < 0 is sell
          - avg_px is the cost basis of the OPEN position
          - realized PnL is recognized when reducing/closing/reversing
        """
        if fill.qty == 0:
            return

        pos = self.positions.get(fill.symbol)
        if pos is None:
            pos = Position(symbol=fill.symbol)
            self.positions[fill.symbol] = pos

        q0 = pos.qty
        qf = fill.qty
        px = float(fill.price)

        # accounting: cash decreases on buys, increases on sells (since qf is signed)
        self.cash -= qf * px
        self.cash -= float(fill.fee)
        self.fees_paid += float(fill.fee)

        # turnover
        self.turnover += abs(qf) * px

        # last mark to execution price
        pos.last_mark = px

        # If no existing position, this fill opens a new one
        if q0 == 0:
            pos.qty = qf
            pos.avg_px = px
            return

        # Helper: same direction (both long or both short)
        same_dir = (q0 > 0 and qf > 0) or (q0 < 0 and qf < 0)

        if same_dir:
            # Adding to an existing position: update weighted average price
            new_qty = q0 + qf
            # (q0 and qf have same sign; new_qty keeps that sign and is non-zero)
            pos.avg_px = (q0 * pos.avg_px + qf * px) / new_qty
            pos.qty = new_qty
            return

        # Otherwise, this fill is reducing/closing/reversing the position
        # Closed quantity is the overlap between existing position and incoming opposite trade
        # This qty is reflected into realized_pnl 
        closed_qty = min(abs(q0), abs(qf))

        # Realized PnL:
        # Long reduced by sell: +closed*(sell_px - avg_px)
        # Short reduced by buy: +closed*(avg_px - buy_px)
        if q0 > 0 and qf < 0:
            pos.realized_pnl += closed_qty * (px - pos.avg_px)
        elif q0 < 0 and qf > 0:
            pos.realized_pnl += closed_qty * (pos.avg_px - px)

        new_qty = q0 + qf  # can be 0 (closed) or opposite sign (reversal) or same sign (partial close)

        if new_qty == 0:
            # Fully closed: reset cost basis
            pos.qty = 0.0
            pos.avg_px = 0.0
            return

        # If position direction flips, the remaining open position's avg_px becomes the fill price
        if (q0 > 0 and new_qty < 0) or (q0 < 0 and new_qty > 0):
            pos.qty = new_qty
            pos.avg_px = px
        else:
            # Partial close without flipping: qty shrinks, avg_px unchanged
            pos.qty = new_qty
            # pos.avg_px stays the same
    
    def nav(self) -> float:
        """
        Net Asset Value (equity) = cash + sum(qty * last_mark)
        """
        equity = float(self.cash)
        for p in self.positions.values():
            px = getattr(p, "last_mark", None)
            if px is None or (isinstance(px, float) and not np.isfinite(px)):
                continue
            equity += float(p.qty) * float(px)
        return float(equity)

    def snapshot(self) -> dict:
        """
        Return current snapshot of portfolio state
        """
        gross_exposure = 0.0
        net_exposure = 0.0
        num_positions = 0

        for p in self.positions.values():
            qty = float(p.qty)
            px = getattr(p, "last_mark", None)
            if px is None or (isinstance(px, float) and not np.isfinite(px)):
                continue

            px = float(px)
            gross_exposure += abs(qty) * px
            net_exposure += qty * px
            if qty != 0:
                num_positions += 1

        equity = self.nav()  # single source of truth

        return {
            "cash": float(self.cash),
            "nav": float(equity),
            "gross_exposure": float(gross_exposure),
            "net_exposure": float(net_exposure),
            "num_positions": int(num_positions),
        }
    
    def positions_snapshot(
        self
    ) -> dict[str, int]:
        """
        Symbol → current position quantity
        """
        return {
            sym: p.qty
            for sym, p in self.positions.items()
        }



            