import pandas as pd
import numpy as np
from typing import Dict
from datetime import datetime


class BacktestingEngine:
    def __init__(
        self,
        initial_capital: float = 10.0,
        commission: float = 0.001,
        slippage: float = 0.0005,
        leverage: int = 3
    ):
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.leverage = leverage
        self.reset()

    def reset(self):
        self.capital = float(self.initial_capital)
        self.trades = []
        self.equity_curve = []
        self.current_position = None

    def run_backtest(
        self,
        df: pd.DataFrame,
        signals: np.ndarray,
        atr_values: np.ndarray,
        stop_loss_multiplier: float = 1.5,
        take_profit_multiplier: float = 2.5,
        risk_per_trade: float = 0.05,
        max_holding_periods: int = 96
    ) -> Dict:
        """
        Backtest with next-candle open execution.

        - Signal at index i is generated from candle i
        - Entry happens at candle i+1 open
        - Exits are checked using candle high/low (conservative ordering)
        """
        self.reset()

        n = len(df)
        if n < 10:
            raise ValueError('Not enough data for backtest')

        for i in range(n):
            ts = df.iloc[i]['open_time'] if 'open_time' in df.columns else i

            self.equity_curve.append({
                'timestamp': ts,
                'equity': self.capital,
                'position': 1 if self.current_position else 0
            })

            if self.current_position is not None:
                self._check_exit_conditions(
                    candle=df.iloc[i],
                    timestamp=ts,
                    max_holding_periods=max_holding_periods
                )

            if self.current_position is None and i + 1 < n:
                sig = int(signals[i])
                if sig != 0:
                    entry_candle = df.iloc[i + 1]
                    entry_price = float(entry_candle['open'])
                    entry_time = entry_candle['open_time'] if 'open_time' in df.columns else ts

                    atr = float(atr_values[i]) if i < len(atr_values) else entry_price * 0.01
                    if not np.isfinite(atr) or atr <= 0:
                        atr = entry_price * 0.01

                    self._enter_position(
                        signal=sig,
                        entry_price=entry_price,
                        timestamp=entry_time,
                        atr=atr,
                        risk_per_trade=risk_per_trade,
                        stop_loss_multiplier=stop_loss_multiplier,
                        take_profit_multiplier=take_profit_multiplier
                    )

        if self.current_position is not None:
            last_price = float(df.iloc[-1]['close'])
            last_ts = df.iloc[-1]['open_time'] if 'open_time' in df.columns else n - 1
            self._exit_position(last_price, last_ts, 'End of data')

        return self._calculate_metrics()

    def _enter_position(
        self,
        signal: int,
        entry_price: float,
        timestamp,
        atr: float,
        risk_per_trade: float,
        stop_loss_multiplier: float,
        take_profit_multiplier: float
    ):
        risk_amount = self.capital * float(risk_per_trade)
        stop_distance = float(atr) * float(stop_loss_multiplier)
        if stop_distance <= 0:
            return

        position_notional = (risk_amount / stop_distance) * self.leverage
        position_notional = min(position_notional, self.capital * self.leverage)

        fees = position_notional * (self.commission + self.slippage)
        if fees > self.capital:
            return

        self.capital -= fees

        direction = 'long' if signal > 0 else 'short'
        if direction == 'long':
            stop_loss = entry_price - (atr * stop_loss_multiplier)
            take_profit = entry_price + (atr * take_profit_multiplier)
        else:
            stop_loss = entry_price + (atr * stop_loss_multiplier)
            take_profit = entry_price - (atr * take_profit_multiplier)

        self.current_position = {
            'direction': direction,
            'entry_price': entry_price,
            'entry_time': timestamp,
            'size': position_notional,
            'stop_loss': float(stop_loss),
            'take_profit': float(take_profit),
            'holding_periods': 0,
            'atr': float(atr)
        }

    def _check_exit_conditions(self, candle: pd.Series, timestamp, max_holding_periods: int):
        pos = self.current_position
        if pos is None:
            return

        pos['holding_periods'] += 1

        high = float(candle['high'])
        low = float(candle['low'])
        close = float(candle['close'])

        if pos['direction'] == 'long':
            if low <= pos['stop_loss']:
                self._exit_position(pos['stop_loss'], timestamp, 'Stop Loss')
                return
            if high >= pos['take_profit']:
                self._exit_position(pos['take_profit'], timestamp, 'Take Profit')
                return
        else:
            if high >= pos['stop_loss']:
                self._exit_position(pos['stop_loss'], timestamp, 'Stop Loss')
                return
            if low <= pos['take_profit']:
                self._exit_position(pos['take_profit'], timestamp, 'Take Profit')
                return

        if pos['holding_periods'] >= int(max_holding_periods):
            self._exit_position(close, timestamp, 'Max Holding Period')

    def _exit_position(self, exit_price: float, timestamp, reason: str):
        pos = self.current_position
        if pos is None:
            return

        if pos['direction'] == 'long':
            pnl = (exit_price - pos['entry_price']) * pos['size']
        else:
            pnl = (pos['entry_price'] - exit_price) * pos['size']

        fees = pos['size'] * (self.commission + self.slippage)
        pnl -= fees

        self.capital += pnl + (pos['size'] / self.leverage)

        self.trades.append({
            'entry_time': pos['entry_time'],
            'exit_time': timestamp,
            'direction': pos['direction'],
            'entry_price': pos['entry_price'],
            'exit_price': float(exit_price),
            'size': pos['size'],
            'pnl': float(pnl),
            'pnl_percent': float((pnl / (pos['size'] / self.leverage)) * 100) if pos['size'] > 0 else 0.0,
            'holding_periods': int(pos['holding_periods']),
            'exit_reason': reason
        })

        self.current_position = None

    def _calculate_metrics(self) -> Dict:
        if len(self.trades) == 0:
            return {
                'total_return': 0.0,
                'total_return_pct': 0.0,
                'final_capital': float(self.capital),
                'total_trades': 0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'max_drawdown': 0.0,
                'trades_df': pd.DataFrame(),
                'equity_df': pd.DataFrame(self.equity_curve)
            }

        trades_df = pd.DataFrame(self.trades)
        equity_df = pd.DataFrame(self.equity_curve)

        total_return = float(self.capital - self.initial_capital)
        total_return_pct = float((total_return / self.initial_capital) * 100) if self.initial_capital > 0 else 0.0

        wins = trades_df[trades_df['pnl'] > 0]
        losses = trades_df[trades_df['pnl'] < 0]

        win_rate = float(len(wins) / len(trades_df) * 100)

        total_wins = float(wins['pnl'].sum()) if len(wins) else 0.0
        total_losses = float(abs(losses['pnl'].sum())) if len(losses) else 0.0
        profit_factor = float(total_wins / total_losses) if total_losses > 0 else float('inf')

        cummax = equity_df['equity'].cummax()
        drawdown = (equity_df['equity'] - cummax) / cummax
        max_drawdown = float(abs(drawdown.min()) * 100)

        return {
            'total_return': total_return,
            'total_return_pct': total_return_pct,
            'final_capital': float(self.capital),
            'total_trades': int(len(trades_df)),
            'winning_trades': int(len(wins)),
            'losing_trades': int(len(losses)),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'avg_win': float(wins['pnl'].mean()) if len(wins) else 0.0,
            'avg_loss': float(losses['pnl'].mean()) if len(losses) else 0.0,
            'largest_win': float(wins['pnl'].max()) if len(wins) else 0.0,
            'largest_loss': float(losses['pnl'].min()) if len(losses) else 0.0,
            'avg_holding_periods': float(trades_df['holding_periods'].mean()),
            'trades_df': trades_df,
            'equity_df': equity_df
        }
