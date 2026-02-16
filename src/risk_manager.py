import numpy as np
import pandas as pd
from typing import Dict, Optional

class RiskManager:
    def __init__(self,
                 initial_capital: float,
                 risk_per_trade: float = 0.05,
                 max_positions: int = 2,
                 leverage: int = 3):
        """
        Initialize risk management system
        
        Args:
            initial_capital: Starting capital
            risk_per_trade: Maximum risk per trade (as decimal)
            max_positions: Maximum concurrent positions
            leverage: Trading leverage
        """
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.max_positions = max_positions
        self.leverage = leverage
        self.current_capital = initial_capital
        self.active_positions = []
        
    def calculate_position_size(self,
                               entry_price: float,
                               stop_loss: float,
                               atr: float) -> float:
        """
        Calculate position size based on ATR and risk parameters
        
        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            atr: Average True Range
            
        Returns:
            Position size in USDT
        """
        risk_amount = self.current_capital * self.risk_per_trade
        
        stop_distance = abs(entry_price - stop_loss)
        
        if stop_distance == 0:
            stop_distance = atr * 1.5
        
        position_size = risk_amount / stop_distance
        
        max_size = self.current_capital * self.leverage
        position_size = min(position_size, max_size)
        
        min_size = 10
        if position_size < min_size:
            return 0
        
        return position_size
    
    def can_open_position(self) -> bool:
        """
        Check if new position can be opened
        
        Returns:
            True if position can be opened
        """
        if len(self.active_positions) >= self.max_positions:
            return False
        
        if self.current_capital < self.initial_capital * 0.1:
            return False
        
        return True
    
    def calculate_stop_loss(self,
                          entry_price: float,
                          atr: float,
                          direction: str,
                          multiplier: float = 1.5) -> float:
        """
        Calculate stop loss price
        
        Args:
            entry_price: Entry price
            atr: Average True Range
            direction: 'long' or 'short'
            multiplier: ATR multiplier
            
        Returns:
            Stop loss price
        """
        stop_distance = atr * multiplier
        
        if direction == 'long':
            stop_loss = entry_price - stop_distance
        else:
            stop_loss = entry_price + stop_distance
        
        return stop_loss
    
    def calculate_take_profit(self,
                            entry_price: float,
                            atr: float,
                            direction: str,
                            multiplier: float = 2.5) -> float:
        """
        Calculate take profit price
        
        Args:
            entry_price: Entry price
            atr: Average True Range
            direction: 'long' or 'short'
            multiplier: ATR multiplier
            
        Returns:
            Take profit price
        """
        tp_distance = atr * multiplier
        
        if direction == 'long':
            take_profit = entry_price + tp_distance
        else:
            take_profit = entry_price - tp_distance
        
        return take_profit
    
    def update_trailing_stop(self,
                           position: Dict,
                           current_price: float,
                           atr: float) -> Optional[float]:
        """
        Update trailing stop loss
        
        Args:
            position: Position dictionary
            current_price: Current market price
            atr: Average True Range
            
        Returns:
            New stop loss price or None
        """
        entry_price = position['entry_price']
        direction = position['direction']
        current_stop = position['stop_loss']
        
        if direction == 'long':
            profit_percent = (current_price - entry_price) / entry_price
            if profit_percent > 0.015:
                new_stop = current_price - (atr * 1.0)
                if new_stop > current_stop:
                    return new_stop
        else:
            profit_percent = (entry_price - current_price) / entry_price
            if profit_percent > 0.015:
                new_stop = current_price + (atr * 1.0)
                if new_stop < current_stop:
                    return new_stop
        
        return None
    
    def check_volume_filter(self,
                          current_volume: float,
                          volume_ma: float,
                          threshold: float = 0.5) -> bool:
        """
        Check if volume is sufficient for trading
        
        Args:
            current_volume: Current candle volume
            volume_ma: Volume moving average
            threshold: Minimum volume ratio
            
        Returns:
            True if volume is sufficient
        """
        if volume_ma == 0:
            return False
        
        volume_ratio = current_volume / volume_ma
        return volume_ratio >= threshold
    
    def check_atr_filter(self,
                        atr: float,
                        atr_ma: float,
                        threshold: float = 0.3) -> bool:
        """
        Check if market volatility is suitable
        
        Args:
            atr: Current ATR
            atr_ma: ATR moving average
            threshold: Minimum ATR ratio
            
        Returns:
            True if volatility is suitable
        """
        if atr_ma == 0:
            return False
        
        atr_ratio = atr / atr_ma
        return atr_ratio >= threshold
    
    def update_capital(self, new_capital: float):
        """
        Update current capital
        
        Args:
            new_capital: New capital amount
        """
        self.current_capital = new_capital
    
    def add_position(self, position: Dict):
        """
        Add position to active positions
        
        Args:
            position: Position dictionary
        """
        self.active_positions.append(position)
    
    def remove_position(self, position_id: str):
        """
        Remove position from active positions
        
        Args:
            position_id: Position identifier
        """
        self.active_positions = [
            p for p in self.active_positions 
            if p.get('id') != position_id
        ]
    
    def get_risk_metrics(self) -> Dict:
        """
        Get current risk metrics
        
        Returns:
            Dictionary of risk metrics
        """
        total_risk = sum(
            p.get('risk_amount', 0) 
            for p in self.active_positions
        )
        
        return {
            'current_capital': self.current_capital,
            'active_positions': len(self.active_positions),
            'max_positions': self.max_positions,
            'total_risk': total_risk,
            'risk_percent': (total_risk / self.current_capital) * 100 if self.current_capital > 0 else 0,
            'capital_utilization': (total_risk / (self.current_capital * self.leverage)) * 100 if self.current_capital > 0 else 0
        }
