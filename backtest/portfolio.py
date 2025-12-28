from models import Position, Fill


class Portfolio:
    def __init__(self, initial_cash, currency: str = 'USD'):
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
        """
        ts: pd.Timestamp
        marks: {symbol: mark_price}  (usually mid price)
        """
        gross = 0.0
        net = 0.0
        unrealized_total = 0.0

        for symbol, pos in self.positions.items():
            if symbol not in marks:
                continue  # cannot mark without price

            mark = marks[symbol]
            pos.last_mark = mark

            # unrealized PnL
            pos.unrealized_pnl = pos.qty * (mark - pos.avg_px)

            gross += abs(pos.qty * mark)
            net += pos.qty * mark
            unrealized_total += pos.unrealized_pnl

        equity = self.cash + unrealized_total + sum(p.realized_pnl for p in self.positions.values())

        # record history
        self.history_ts.append(ts)
        self.history_equity.append(equity)
        self.history_cash.append(self.cash)
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
    
    def stats(self) -> dict:
        """
        Return current snapshot of portfolio state
        """

        gross_exposure = sum(
            abs(p.qty) * p.last_mark
            for p in self.positions.values()
        )

        net_exposure = sum(
            p.qty * p.last_mark
            for p in self.positions.values()
        )

        equity = self.cash + net_exposure

        return {
            "cash": self.cash,
            "equity": equity,
            "gross_exposure": gross_exposure,
            "net_exposure": net_exposure,
            "num_positions": sum(1 for p in self.positions.values() if p.qty != 0),
        }


            