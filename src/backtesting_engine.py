import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json

class BacktestingEngine:
    def __init__(self,
                 initial_capital: float = 10.0,
                 commission: float = 0.001,
                 slippage: float = 0.0005,
                 leverage: int = 3):
        """
        Initialize backtesting engine
        
        Args:
            initial_capital: Starting capital in USDT
            commission: Trading fee (0.001 = 0.1%)
            slippage: Slippage rate
            leverage: Leverage multiplier
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.leverage = leverage
        
        self.reset()
    
    def reset(self):
        """
        Reset backtest state
        """
        self.capital = self.initial_capital
        self.positions = []
        self.trades = []
        self.equity_curve = []
        self.current_position = None
        
    def run_backtest(self,
                    df: pd.DataFrame,
                    signals: np.ndarray,
                    atr_values: np.ndarray,
                    stop_loss_multiplier: float = 1.5,
                    take_profit_multiplier: float = 2.5,
                    risk_per_trade: float = 0.05,
                    max_holding_periods: int = 96) -> Dict:
        """
        Run complete backtest with risk management
        
        Args:
            df: DataFrame with OHLCV data
            signals: Trading signals array
            atr_values: ATR values for position sizing
            stop_loss_multiplier: ATR multiplier for stop loss
            take_profit_multiplier: ATR multiplier for take profit
            risk_per_trade: Risk percentage per trade
            max_holding_periods: Maximum candles to hold position
            
        Returns:
            Dictionary with backtest results
        """
        self.reset()
        
        for i in range(len(df)):
            timestamp = df.iloc[i]['open_time']
            price = df.iloc[i]['close']
            signal = signals[i]
            atr = atr_values[i] if not np.isnan(atr_values[i]) else price * 0.01
            
            self.equity_curve.append({
                'timestamp': timestamp,
                'equity': self.capital,
                'position': 1 if self.current_position else 0
            })
            
            if self.current_position:
                self._check_exit_conditions(
                    i, price, timestamp,
                    stop_loss_multiplier,
                    take_profit_multiplier,
                    max_holding_periods
                )
            
            if self.current_position is None and signal != 0:
                self._enter_position(
                    signal, price, timestamp, atr,
                    risk_per_trade, stop_loss_multiplier,
                    take_profit_multiplier
                )
        
        if self.current_position:
            last_price = df.iloc[-1]['close']
            last_timestamp = df.iloc[-1]['open_time']
            self._exit_position(last_price, last_timestamp, 'End of data')
        
        results = self._calculate_metrics()
        return results
    
    def _enter_position(self,
                       signal: int,
                       price: float,
                       timestamp: datetime,
                       atr: float,
                       risk_per_trade: float,
                       stop_loss_multiplier: float,
                       take_profit_multiplier: float):
        """
        Enter new position
        """
        risk_amount = self.capital * risk_per_trade
        stop_distance = atr * stop_loss_multiplier
        
        position_size = (risk_amount / stop_distance) * self.leverage
        position_size = min(position_size, self.capital * self.leverage)
        
        entry_cost = position_size * self.commission
        slippage_cost = position_size * self.slippage
        total_cost = entry_cost + slippage_cost
        
        if total_cost > self.capital:
            return
        
        self.capital -= total_cost
        
        direction = 'long' if signal > 0 else 'short'
        
        if direction == 'long':
            stop_loss = price - (atr * stop_loss_multiplier)
            take_profit = price + (atr * take_profit_multiplier)
        else:
            stop_loss = price + (atr * stop_loss_multiplier)
            take_profit = price - (atr * take_profit_multiplier)
        
        self.current_position = {
            'direction': direction,
            'entry_price': price,
            'entry_time': timestamp,
            'size': position_size,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'entry_index': len(self.trades),
            'holding_periods': 0,
            'atr': atr
        }
    
    def _exit_position(self,
                      price: float,
                      timestamp: datetime,
                      reason: str):
        """
        Exit current position
        """
        if self.current_position is None:
            return
        
        pos = self.current_position
        
        if pos['direction'] == 'long':
            pnl = (price - pos['entry_price']) * pos['size']
        else:
            pnl = (pos['entry_price'] - price) * pos['size']
        
        exit_cost = pos['size'] * self.commission
        slippage_cost = pos['size'] * self.slippage
        pnl -= (exit_cost + slippage_cost)
        
        self.capital += pnl + (pos['size'] / self.leverage)
        
        trade_record = {
            'entry_time': pos['entry_time'],
            'exit_time': timestamp,
            'direction': pos['direction'],
            'entry_price': pos['entry_price'],
            'exit_price': price,
            'size': pos['size'],
            'pnl': pnl,
            'pnl_percent': (pnl / (pos['size'] / self.leverage)) * 100,
            'holding_periods': pos['holding_periods'],
            'exit_reason': reason
        }
        
        self.trades.append(trade_record)
        self.current_position = None
    
    def _check_exit_conditions(self,
                              index: int,
                              price: float,
                              timestamp: datetime,
                              stop_loss_multiplier: float,
                              take_profit_multiplier: float,
                              max_holding_periods: int):
        """
        Check if exit conditions are met
        """
        if self.current_position is None:
            return
        
        pos = self.current_position
        pos['holding_periods'] += 1
        
        if pos['direction'] == 'long':
            if price <= pos['stop_loss']:
                self._exit_position(price, timestamp, 'Stop Loss')
                return
            elif price >= pos['take_profit']:
                self._exit_position(price, timestamp, 'Take Profit')
                return
        else:
            if price >= pos['stop_loss']:
                self._exit_position(price, timestamp, 'Stop Loss')
                return
            elif price <= pos['take_profit']:
                self._exit_position(price, timestamp, 'Take Profit')
                return
        
        if pos['holding_periods'] >= max_holding_periods:
            self._exit_position(price, timestamp, 'Max Holding Period')
            return
    
    def _calculate_metrics(self) -> Dict:
        """
        Calculate backtest performance metrics
        """
        if len(self.trades) == 0:
            return {
                'total_return': 0,
                'total_return_pct': 0,
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'largest_win': 0,
                'largest_loss': 0
            }
        
        trades_df = pd.DataFrame(self.trades)
        equity_df = pd.DataFrame(self.equity_curve)
        
        total_return = self.capital - self.initial_capital
        total_return_pct = (total_return / self.initial_capital) * 100
        
        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] < 0]
        
        win_rate = len(winning_trades) / len(trades_df) if len(trades_df) > 0 else 0
        
        total_wins = winning_trades['pnl'].sum() if len(winning_trades) > 0 else 0
        total_losses = abs(losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 1
        profit_factor = total_wins / total_losses if total_losses > 0 else 0
        
        returns = equity_df['equity'].pct_change().dropna()
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if len(returns) > 0 and returns.std() > 0 else 0
        
        cummax = equity_df['equity'].cummax()
        drawdown = (equity_df['equity'] - cummax) / cummax
        max_drawdown = abs(drawdown.min()) * 100
        
        avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
        
        metrics = {
            'total_return': total_return,
            'total_return_pct': total_return_pct,
            'final_capital': self.capital,
            'total_trades': len(trades_df),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate * 100,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'largest_win': winning_trades['pnl'].max() if len(winning_trades) > 0 else 0,
            'largest_loss': losing_trades['pnl'].min() if len(losing_trades) > 0 else 0,
            'avg_holding_periods': trades_df['holding_periods'].mean(),
            'trades_df': trades_df,
            'equity_df': equity_df
        }
        
        return metrics
    
    def get_trades_summary(self) -> pd.DataFrame:
        """
        Get summary of all trades
        """
        if len(self.trades) == 0:
            return pd.DataFrame()
        
        return pd.DataFrame(self.trades)
    
    def get_equity_curve(self) -> pd.DataFrame:
        """
        Get equity curve data
        """
        if len(self.equity_curve) == 0:
            return pd.DataFrame()
        
        return pd.DataFrame(self.equity_curve)
