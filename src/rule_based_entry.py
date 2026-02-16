import pandas as pd
import numpy as np
from typing import Dict, Optional


class RuleBasedEntry:
    """
    Rule-based entry system with RELAXED conditions for more opportunities
    
    Key changes:
    - Lower confidence threshold (0.4)
    - Wider RSI ranges
    - Lower volume requirements
    - Enable trading in regime 2 (mean reversion) and 4 (early breakout)
    - Tighter stops for better risk/reward
    """
    
    def __init__(self):
        self.last_regime = None
        self.regime_confidence_threshold = 0.4  # Lower from 0.5
    
    def generate_signal(
        self,
        df_row: pd.Series,
        regime: int,
        regime_proba: np.ndarray
    ) -> Dict:
        max_proba = float(np.max(regime_proba))
        
        if max_proba < self.regime_confidence_threshold:
            return {
                'signal': 0,
                'regime': int(regime),
                'confidence': max_proba,
                'reason': 'Low regime confidence'
            }
        
        if regime == 0:
            return self._strong_uptrend_logic(df_row, max_proba)
        elif regime == 1:
            return self._strong_downtrend_logic(df_row, max_proba)
        elif regime == 2:
            return self._choppy_logic(df_row, max_proba)
        elif regime == 3:
            return self._high_volatility_logic(df_row, max_proba)
        elif regime == 4:
            return self._low_volatility_logic(df_row, max_proba)
        else:
            return {
                'signal': 0,
                'regime': int(regime),
                'confidence': max_proba,
                'reason': 'Unknown regime'
            }
    
    def _strong_uptrend_logic(self, row: pd.Series, confidence: float) -> Dict:
        """Strong uptrend: Long on ANY pullback - RELAXED"""
        close = float(row.get('close', 0))
        ema9 = float(row.get('ema_9', close))
        ema21 = float(row.get('ema_21', close))
        rsi = float(row.get('rsi_14', 50))
        macd_diff = float(row.get('macd_diff', 0))
        volume_ratio = float(row.get('volume_ratio_20', 1))
        adx = float(row.get('adx', 20))
        
        # RELAXED: Just need price above EMA21 and not overbought
        price_ok = close > ema21 * 0.99
        rsi_ok = rsi < 70  # Just not overbought
        volume_ok = volume_ratio > 0.5  # Half of average is enough
        trend_ok = adx > 15  # Lower ADX
        
        if price_ok and rsi_ok and volume_ok and trend_ok:
            return {
                'signal': 1,
                'regime': 0,
                'confidence': confidence,
                'reason': 'Uptrend entry'
            }
        
        return {'signal': 0, 'regime': 0, 'confidence': confidence, 'reason': 'Wait'}
    
    def _strong_downtrend_logic(self, row: pd.Series, confidence: float) -> Dict:
        """Strong downtrend: Short on ANY rally - RELAXED"""
        close = float(row.get('close', 0))
        ema21 = float(row.get('ema_21', close))
        rsi = float(row.get('rsi_14', 50))
        volume_ratio = float(row.get('volume_ratio_20', 1))
        adx = float(row.get('adx', 20))
        
        # RELAXED: Just need price below EMA21 and not oversold
        price_ok = close < ema21 * 1.01
        rsi_ok = rsi > 30  # Just not oversold
        volume_ok = volume_ratio > 0.5
        trend_ok = adx > 15
        
        if price_ok and rsi_ok and volume_ok and trend_ok:
            return {
                'signal': -1,
                'regime': 1,
                'confidence': confidence,
                'reason': 'Downtrend entry'
            }
        
        return {'signal': 0, 'regime': 1, 'confidence': confidence, 'reason': 'Wait'}
    
    def _choppy_logic(self, row: pd.Series, confidence: float) -> Dict:
        """Choppy: NOW TRADE mean reversion at BB extremes"""
        close = float(row.get('close', 0))
        bb_percent = float(row.get('bb_percent', 0.5))
        rsi = float(row.get('rsi_14', 50))
        
        # Buy near lower BB (oversold)
        if bb_percent < 0.25 and rsi < 45:
            return {
                'signal': 1,
                'regime': 2,
                'confidence': confidence,
                'reason': 'Choppy mean reversion long'
            }
        
        # Sell near upper BB (overbought)
        if bb_percent > 0.75 and rsi > 55:
            return {
                'signal': -1,
                'regime': 2,
                'confidence': confidence,
                'reason': 'Choppy mean reversion short'
            }
        
        return {'signal': 0, 'regime': 2, 'confidence': confidence, 'reason': 'Wait'}
    
    def _high_volatility_logic(self, row: pd.Series, confidence: float) -> Dict:
        """High vol: Trade breakouts - RELAXED volume"""
        close = float(row.get('close', 0))
        bb_upper = float(row.get('bb_upper', close * 1.02))
        bb_lower = float(row.get('bb_lower', close * 0.98))
        bb_percent = float(row.get('bb_percent', 0.5))
        volume_ratio = float(row.get('volume_ratio_20', 1))
        macd_diff = float(row.get('macd_diff', 0))
        rsi = float(row.get('rsi_14', 50))
        
        # RELAXED: Lower volume requirement
        volume_ok = volume_ratio > 1.2  # Down from 2.0
        
        # Breakout above
        if bb_percent > 0.8 and macd_diff > 0 and rsi > 50 and volume_ok:
            return {
                'signal': 1,
                'regime': 3,
                'confidence': confidence,
                'reason': 'High vol breakout up'
            }
        
        # Breakdown below
        if bb_percent < 0.2 and macd_diff < 0 and rsi < 50 and volume_ok:
            return {
                'signal': -1,
                'regime': 3,
                'confidence': confidence,
                'reason': 'High vol breakout down'
            }
        
        return {'signal': 0, 'regime': 3, 'confidence': confidence, 'reason': 'Wait'}
    
    def _low_volatility_logic(self, row: pd.Series, confidence: float) -> Dict:
        """Low vol: NOW TRADE early breakout signals"""
        close = float(row.get('close', 0))
        bb_upper = float(row.get('bb_upper', close * 1.02))
        bb_lower = float(row.get('bb_lower', close * 0.98))
        macd_diff = float(row.get('macd_diff', 0))
        volume_ratio = float(row.get('volume_ratio_20', 1))
        
        # Trade when price touches BB edges with momentum
        if close >= bb_upper * 0.995 and macd_diff > 0 and volume_ratio > 0.7:
            return {
                'signal': 1,
                'regime': 4,
                'confidence': confidence,
                'reason': 'Low vol breakout attempt up'
            }
        
        if close <= bb_lower * 1.005 and macd_diff < 0 and volume_ratio > 0.7:
            return {
                'signal': -1,
                'regime': 4,
                'confidence': confidence,
                'reason': 'Low vol breakout attempt down'
            }
        
        return {'signal': 0, 'regime': 4, 'confidence': confidence, 'reason': 'Wait'}
    
    def calculate_stop_loss(
        self,
        entry_price: float,
        atr: float,
        direction: str,
        regime: int
    ) -> float:
        """Tighter stops for better risk/reward"""
        if regime in [0, 1]:  # Trend
            multiplier = 1.5  # Tighter (was 2.0)
        elif regime == 2:  # Choppy mean reversion
            multiplier = 1.0  # Tight
        elif regime == 3:  # Breakout
            multiplier = 1.0  # Tight
        else:  # Low vol
            multiplier = 1.2
        
        if direction == 'long':
            return entry_price - (atr * multiplier)
        else:
            return entry_price + (atr * multiplier)
    
    def calculate_take_profit(
        self,
        entry_price: float,
        atr: float,
        direction: str,
        regime: int
    ) -> float:
        """Reasonable targets for decent win rate"""
        if regime in [0, 1]:  # Trend
            multiplier = 3.0  # Good target
        elif regime == 2:  # Choppy mean reversion
            multiplier = 2.0  # Quick profit
        elif regime == 3:  # Breakout
            multiplier = 2.5  # Medium
        else:  # Low vol
            multiplier = 2.5  # Medium
        
        if direction == 'long':
            return entry_price + (atr * multiplier)
        else:
            return entry_price - (atr * multiplier)
