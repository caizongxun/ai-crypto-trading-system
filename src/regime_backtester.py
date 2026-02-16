import pandas as pd
import numpy as np
from typing import Dict
from src.rule_based_entry import RuleBasedEntry


class RegimeBacktester:
    """
    Backtester for regime-based trading system
    """
    
    def __init__(
        self,
        initial_capital: float = 10.0,
        commission: float = 0.001,
        slippage: float = 0.0005,
        leverage: int = 3,
        risk_per_trade: float = 0.05
    ):
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.leverage = leverage
        self.risk_per_trade = risk_per_trade
        self.reset()
    
    def reset(self):
        self.capital = float(self.initial_capital)
        self.trades = []
        self.equity_curve = []
        self.current_position = None
        self.regime_stats = {i: {'entries': 0, 'wins': 0, 'total_pnl': 0.0} for i in range(5)}
    
    def run_backtest(
        self,
        df: pd.DataFrame,
        regimes: np.ndarray,
        regime_probas: np.ndarray
    ) -> Dict:
        """
        Run backtest with regime-based entries
        
        Args:
            df: DataFrame with OHLCV + features
            regimes: Predicted regimes for each candle
            regime_probas: Regime probability distributions
        
        Returns:
            Backtest metrics
        """
        self.reset()
        entry_system = RuleBasedEntry()
        
        n = len(df)
        if n < 10:
            raise ValueError('Not enough data')
        
        for i in range(n):
            row = df.iloc[i]
            ts = row.get('open_time', i)
            
            self.equity_curve.append({
                'timestamp': ts,
                'equity': self.capital,
                'position': 1 if self.current_position else 0,
                'regime': int(regimes[i]) if i < len(regimes) else -1
            })
            
            # Check exit conditions for existing position
            if self.current_position is not None:
                self._check_exit_conditions(row, ts)
            
            # Generate new entry signal
            if self.current_position is None and i + 1 < n:
                regime = int(regimes[i])
                regime_proba = regime_probas[i] if i < len(regime_probas) else np.zeros(5)
                
                signal_data = entry_system.generate_signal(row, regime, regime_proba)
                
                if signal_data['signal'] != 0:
                    entry_candle = df.iloc[i + 1]
                    entry_price = float(entry_candle['open'])
                    atr = float(row.get('atr', entry_price * 0.01))
                    
                    if not np.isfinite(atr) or atr <= 0:
                        atr = entry_price * 0.01
                    
                    self._enter_position(
                        signal=signal_data['signal'],
                        entry_price=entry_price,
                        timestamp=entry_candle.get('open_time', ts),
                        atr=atr,
                        regime=regime,
                        entry_system=entry_system,
                        reason=signal_data['reason']
                    )
        
        # Close any remaining position
        if self.current_position is not None:
            last_price = float(df.iloc[-1]['close'])
            last_ts = df.iloc[-1].get('open_time', n - 1)
            self._exit_position(last_price, last_ts, 'End of data')
        
        return self._calculate_metrics()
    
    def _enter_position(
        self,
        signal: int,
        entry_price: float,
        timestamp,
        atr: float,
        regime: int,
        entry_system: RuleBasedEntry,
        reason: str
    ):
        direction = 'long' if signal > 0 else 'short'
        
        # Calculate stops based on regime
        stop_loss = entry_system.calculate_stop_loss(entry_price, atr, direction, regime)
        take_profit = entry_system.calculate_take_profit(entry_price, atr, direction, regime)
        
        # Position sizing
        risk_amount = self.capital * self.risk_per_trade
        stop_distance = abs(entry_price - stop_loss)
        
        if stop_distance <= 0:
            return
        
        position_notional = (risk_amount / stop_distance) * self.leverage
        position_notional = min(position_notional, self.capital * self.leverage)
        
        fees = position_notional * (self.commission + self.slippage)
        if fees > self.capital:
            return
        
        self.capital -= fees
        
        self.current_position = {
            'direction': direction,
            'entry_price': float(entry_price),
            'entry_time': timestamp,
            'size': float(position_notional),
            'stop_loss': float(stop_loss),
            'take_profit': float(take_profit),
            'regime': int(regime),
            'entry_reason': reason,
            'holding_periods': 0,
            'atr': float(atr)
        }
        
        self.regime_stats[regime]['entries'] += 1
    
    def _check_exit_conditions(self, row: pd.Series, timestamp):
        pos = self.current_position
        if pos is None:
            return
        
        pos['holding_periods'] += 1
        
        high = float(row.get('high', 0))
        low = float(row.get('low', 0))
        close = float(row.get('close', 0))
        
        # Check stops
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
        
        # Max holding period: 96 candles (24 hours on 15m)
        if pos['holding_periods'] >= 96:
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
        
        # Track regime stats
        regime = pos['regime']
        if pnl > 0:
            self.regime_stats[regime]['wins'] += 1
        self.regime_stats[regime]['total_pnl'] += pnl
        
        self.trades.append({
            'entry_time': pos['entry_time'],
            'exit_time': timestamp,
            'direction': pos['direction'],
            'entry_price': pos['entry_price'],
            'exit_price': float(exit_price),
            'size': pos['size'],
            'pnl': float(pnl),
            'pnl_percent': float((pnl / (pos['size'] / self.leverage)) * 100),
            'holding_periods': int(pos['holding_periods']),
            'regime': int(regime),
            'entry_reason': pos['entry_reason'],
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
                'regime_stats': self.regime_stats,
                'trades_df': pd.DataFrame(),
                'equity_df': pd.DataFrame(self.equity_curve)
            }
        
        trades_df = pd.DataFrame(self.trades)
        equity_df = pd.DataFrame(self.equity_curve)
        
        total_return = float(self.capital - self.initial_capital)
        total_return_pct = float((total_return / self.initial_capital) * 100)
        
        wins = trades_df[trades_df['pnl'] > 0]
        losses = trades_df[trades_df['pnl'] < 0]
        
        win_rate = float(len(wins) / len(trades_df) * 100)
        
        total_wins = float(wins['pnl'].sum()) if len(wins) else 0.0
        total_losses = float(abs(losses['pnl'].sum())) if len(losses) else 0.0
        profit_factor = float(total_wins / total_losses) if total_losses > 0 else float('inf')
        
        cummax = equity_df['equity'].cummax()
        drawdown = (equity_df['equity'] - cummax) / cummax
        max_drawdown = float(abs(drawdown.min()) * 100)
        
        # Calculate regime-specific metrics
        regime_performance = {}
        for regime_id, stats in self.regime_stats.items():
            if stats['entries'] > 0:
                regime_performance[f'Regime_{regime_id}'] = {
                    'entries': stats['entries'],
                    'wins': stats['wins'],
                    'win_rate': float(stats['wins'] / stats['entries'] * 100),
                    'total_pnl': float(stats['total_pnl'])
                }
        
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
            'avg_holding_periods': float(trades_df['holding_periods'].mean()),
            'regime_performance': regime_performance,
            'trades_df': trades_df,
            'equity_df': equity_df
        }
