import pandas as pd
import numpy as np
from typing import Dict, Optional


class RuleBasedEntry:
    """
    Rule-based entry system that uses regime predictions + technical conditions
    
    Entry logic per regime:
    
    Regime 0 (Strong Uptrend):
    - Enter long on pullbacks to EMA9/21
    - RSI < 60 (not overbought)
    - MACD histogram positive
    - Volume surge on bounce
    
    Regime 1 (Strong Downtrend):
    - Enter short on rallies to EMA9/21
    - RSI > 40 (not oversold)
    - MACD histogram negative
    - Volume surge on rejection
    
    Regime 2 (Choppy/Ranging):
    - No entries (too risky)
    
    Regime 3 (High Volatility Breakout):
    - Enter in direction of breakout
    - Price breaks above/below Bollinger Bands
    - Volume > 2x average
    - ADX rising
    
    Regime 4 (Low Volatility Consolidation):
    - No entries (waiting for breakout)
    """
    
    def __init__(self):
        self.last_regime = None
        self.regime_confidence_threshold = 0.5
    
    def generate_signal(
        self,
        df_row: pd.Series,
        regime: int,
        regime_proba: np.ndarray
    ) -> Dict:
        """
        Generate trading signal based on regime and technical conditions
        
        Args:
            df_row: Current candle data with features
            regime: Predicted regime (0-4)
            regime_proba: Probability distribution over regimes
        
        Returns:
            {
                'signal': -1/0/1,
                'regime': int,
                'confidence': float,
                'reason': str
            }
        """
        max_proba = float(np.max(regime_proba))
        
        # Skip if regime confidence too low
        if max_proba < self.regime_confidence_threshold:
            return {
                'signal': 0,
                'regime': int(regime),
                'confidence': max_proba,
                'reason': 'Low regime confidence'
            }
        
        # Route to regime-specific logic
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
        """
        Strong uptrend: Long on pullbacks
        """
        close = float(row.get('close', 0))
        ema9 = float(row.get('ema_9', close))
        ema21 = float(row.get('ema_21', close))
        rsi = float(row.get('rsi_14', 50))
        macd_diff = float(row.get('macd_diff', 0))
        volume_ratio = float(row.get('volume_ratio_20', 1))
        bb_percent = float(row.get('bb_percent', 0.5))
        
        # Check for pullback setup
        near_ema = close < ema9 * 1.005 and close > ema21 * 0.995
        rsi_ok = 30 < rsi < 60
        macd_positive = macd_diff > 0
        volume_ok = volume_ratio > 0.8
        not_overbought = bb_percent < 0.8
        
        if near_ema and rsi_ok and macd_positive and volume_ok and not_overbought:
            return {
                'signal': 1,
                'regime': 0,
                'confidence': confidence,
                'reason': 'Uptrend pullback to EMA'
            }
        
        return {
            'signal': 0,
            'regime': 0,
            'confidence': confidence,
            'reason': 'Waiting for pullback setup'
        }
    
    def _strong_downtrend_logic(self, row: pd.Series, confidence: float) -> Dict:
        """
        Strong downtrend: Short on rallies
        """
        close = float(row.get('close', 0))
        ema9 = float(row.get('ema_9', close))
        ema21 = float(row.get('ema_21', close))
        rsi = float(row.get('rsi_14', 50))
        macd_diff = float(row.get('macd_diff', 0))
        volume_ratio = float(row.get('volume_ratio_20', 1))
        bb_percent = float(row.get('bb_percent', 0.5))
        
        # Check for rally setup
        near_ema = close > ema9 * 0.995 and close < ema21 * 1.005
        rsi_ok = 40 < rsi < 70
        macd_negative = macd_diff < 0
        volume_ok = volume_ratio > 0.8
        not_oversold = bb_percent > 0.2
        
        if near_ema and rsi_ok and macd_negative and volume_ok and not_oversold:
            return {
                'signal': -1,
                'regime': 1,
                'confidence': confidence,
                'reason': 'Downtrend rally to EMA'
            }
        
        return {
            'signal': 0,
            'regime': 1,
            'confidence': confidence,
            'reason': 'Waiting for rally setup'
        }
    
    def _choppy_logic(self, row: pd.Series, confidence: float) -> Dict:
        """
        Choppy market: No trades
        """
        return {
            'signal': 0,
            'regime': 2,
            'confidence': confidence,
            'reason': 'Choppy market - no entry'
        }
    
    def _high_volatility_logic(self, row: pd.Series, confidence: float) -> Dict:
        """
        High volatility breakout: Trade breakouts with strong volume
        """
        close = float(row.get('close', 0))
        bb_upper = float(row.get('bb_upper', close * 1.02))
        bb_lower = float(row.get('bb_lower', close * 0.98))
        volume_ratio = float(row.get('volume_ratio_20', 1))
        adx = float(row.get('adx', 0))
        macd_diff = float(row.get('macd_diff', 0))
        
        volume_surge = volume_ratio > 2.0
        adx_strong = adx > 25
        
        # Breakout above BB
        if close > bb_upper and volume_surge and adx_strong and macd_diff > 0:
            return {
                'signal': 1,
                'regime': 3,
                'confidence': confidence,
                'reason': 'High vol breakout up'
            }
        
        # Breakdown below BB
        if close < bb_lower and volume_surge and adx_strong and macd_diff < 0:
            return {
                'signal': -1,
                'regime': 3,
                'confidence': confidence,
                'reason': 'High vol breakout down'
            }
        
        return {
            'signal': 0,
            'regime': 3,
            'confidence': confidence,
            'reason': 'Waiting for breakout confirmation'
        }
    
    def _low_volatility_logic(self, row: pd.Series, confidence: float) -> Dict:
        """
        Low volatility consolidation: Wait for breakout
        """
        return {
            'signal': 0,
            'regime': 4,
            'confidence': confidence,
            'reason': 'Low volatility - waiting for breakout'
        }
    
    def calculate_stop_loss(
        self,
        entry_price: float,
        atr: float,
        direction: str,
        regime: int
    ) -> float:
        """
        Calculate stop loss based on regime
        
        Regime 0/1 (Trend): Wider stops (2.0 ATR)
        Regime 3 (Breakout): Tight stops (1.0 ATR)
        Others: Normal stops (1.5 ATR)
        """
        if regime in [0, 1]:
            multiplier = 2.0
        elif regime == 3:
            multiplier = 1.0
        else:
            multiplier = 1.5
        
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
        """
        Calculate take profit based on regime
        
        Regime 0/1 (Trend): Larger targets (4.0 ATR)
        Regime 3 (Breakout): Quick targets (2.0 ATR)
        Others: Normal targets (2.5 ATR)
        """
        if regime in [0, 1]:
            multiplier = 4.0
        elif regime == 3:
            multiplier = 2.0
        else:
            multiplier = 2.5
        
        if direction == 'long':
            return entry_price + (atr * multiplier)
        else:
            return entry_price - (atr * multiplier)
