import pandas as pd
import numpy as np
from typing import Optional
import ta
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, ROCIndicator, WilliamsRIndicator
from ta.volatility import BollingerBands, AverageTrueRange, KeltnerChannel
from ta.volume import OnBalanceVolumeIndicator, VolumeWeightedAveragePrice, ChaikinMoneyFlowIndicator

class FeatureEngineer:
    def __init__(self):
        self.feature_columns = []
    
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive feature set for model training
        All features use only historical data (no look-ahead bias)
        
        Args:
            df: DataFrame with OHLCV columns
            
        Returns:
            DataFrame with added features
        """
        df = df.copy()
        
        df = self._add_price_features(df)
        df = self._add_momentum_indicators(df)
        df = self._add_trend_indicators(df)
        df = self._add_volatility_indicators(df)
        df = self._add_volume_indicators(df)
        df = self._add_pattern_features(df)
        df = self._add_time_features(df)
        
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        return df
    
    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Basic price-based features
        """
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        for period in [5, 10, 20, 50]:
            df[f'return_{period}'] = df['close'].pct_change(period)
            df[f'price_change_{period}'] = df['close'] - df['close'].shift(period)
        
        df['high_low_range'] = (df['high'] - df['low']) / df['close']
        df['close_open_range'] = (df['close'] - df['open']) / df['open']
        
        df['upper_shadow'] = (df['high'] - np.maximum(df['open'], df['close'])) / df['close']
        df['lower_shadow'] = (np.minimum(df['open'], df['close']) - df['low']) / df['close']
        df['body_size'] = np.abs(df['close'] - df['open']) / df['close']
        
        return df
    
    def _add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Momentum-based technical indicators
        """
        for period in [7, 14, 21]:
            rsi = RSIIndicator(close=df['close'], window=period)
            df[f'rsi_{period}'] = rsi.rsi()
        
        stoch = StochasticOscillator(
            high=df['high'], 
            low=df['low'], 
            close=df['close'],
            window=14,
            smooth_window=3
        )
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        williams = WilliamsRIndicator(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            lbp=14
        )
        df['williams_r'] = williams.williams_r()
        
        for period in [10, 20]:
            roc = ROCIndicator(close=df['close'], window=period)
            df[f'roc_{period}'] = roc.roc()
        
        return df
    
    def _add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Trend-following indicators
        """
        for period in [9, 21, 50, 200]:
            ema = EMAIndicator(close=df['close'], window=period)
            df[f'ema_{period}'] = ema.ema_indicator()
            df[f'price_to_ema_{period}'] = (df['close'] - df[f'ema_{period}']) / df[f'ema_{period}']
        
        macd_indicator = MACD(
            close=df['close'],
            window_slow=26,
            window_fast=12,
            window_sign=9
        )
        df['macd'] = macd_indicator.macd()
        df['macd_signal'] = macd_indicator.macd_signal()
        df['macd_diff'] = macd_indicator.macd_diff()
        
        adx = ADXIndicator(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            window=14
        )
        df['adx'] = adx.adx()
        df['adx_pos'] = adx.adx_pos()
        df['adx_neg'] = adx.adx_neg()
        
        for period in [20, 50]:
            df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
            df[f'price_to_sma_{period}'] = (df['close'] - df[f'sma_{period}']) / df[f'sma_{period}']
        
        return df
    
    def _add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Volatility-based indicators
        """
        bb = BollingerBands(
            close=df['close'],
            window=20,
            window_dev=2
        )
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_middle'] = bb.bollinger_mavg()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_width'] = bb.bollinger_wband()
        df['bb_percent'] = bb.bollinger_pband()
        
        atr = AverageTrueRange(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            window=14
        )
        df['atr'] = atr.average_true_range()
        df['atr_percent'] = df['atr'] / df['close']
        
        kc = KeltnerChannel(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            window=20
        )
        df['kc_upper'] = kc.keltner_channel_hband()
        df['kc_middle'] = kc.keltner_channel_mband()
        df['kc_lower'] = kc.keltner_channel_lband()
        df['kc_width'] = kc.keltner_channel_wband()
        
        for period in [5, 10, 20]:
            df[f'volatility_{period}'] = df['returns'].rolling(window=period).std()
        
        return df
    
    def _add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Volume-based indicators
        """
        obv = OnBalanceVolumeIndicator(close=df['close'], volume=df['volume'])
        df['obv'] = obv.on_balance_volume()
        
        vwap = VolumeWeightedAveragePrice(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            volume=df['volume']
        )
        df['vwap'] = vwap.volume_weighted_average_price()
        df['price_to_vwap'] = (df['close'] - df['vwap']) / df['vwap']
        
        cmf = ChaikinMoneyFlowIndicator(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            volume=df['volume'],
            window=20
        )
        df['cmf'] = cmf.chaikin_money_flow()
        
        for period in [5, 20, 50]:
            df[f'volume_ma_{period}'] = df['volume'].rolling(window=period).mean()
            df[f'volume_ratio_{period}'] = df['volume'] / df[f'volume_ma_{period}']
        
        df['volume_price_trend'] = df['volume'] * df['returns']
        
        return df
    
    def _add_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Candlestick pattern features
        """
        df['is_bullish'] = (df['close'] > df['open']).astype(int)
        df['is_bearish'] = (df['close'] < df['open']).astype(int)
        df['is_doji'] = (np.abs(df['close'] - df['open']) < (df['high'] - df['low']) * 0.1).astype(int)
        
        df['bullish_streak'] = (df['is_bullish']
                                .groupby((df['is_bullish'] != df['is_bullish'].shift()).cumsum())
                                .cumsum())
        df['bearish_streak'] = (df['is_bearish']
                                .groupby((df['is_bearish'] != df['is_bearish'].shift()).cumsum())
                                .cumsum())
        
        df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
        df['lower_low'] = (df['low'] < df['low'].shift(1)).astype(int)
        df['higher_low'] = (df['low'] > df['low'].shift(1)).astype(int)
        df['lower_high'] = (df['high'] < df['high'].shift(1)).astype(int)
        
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Time-based features
        """
        df['hour'] = df['open_time'].dt.hour
        df['day_of_week'] = df['open_time'].dt.dayofweek
        df['day_of_month'] = df['open_time'].dt.day
        df['month'] = df['open_time'].dt.month
        
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        return df
    
    def create_target(self, df: pd.DataFrame, 
                     forward_periods: int = 1,
                     threshold: float = 0.002) -> pd.DataFrame:
        """
        Create target variable for classification
        Uses future price movement (only for training, not prediction)
        
        Args:
            df: DataFrame with features
            forward_periods: Number of periods to look ahead
            threshold: Minimum price change to trigger signal
            
        Returns:
            DataFrame with target column
        """
        df = df.copy()
        
        df['future_return'] = df['close'].shift(-forward_periods) / df['close'] - 1
        
        df['target'] = 0
        df.loc[df['future_return'] > threshold, 'target'] = 1
        df.loc[df['future_return'] < -threshold, 'target'] = -1
        
        df = df.drop('future_return', axis=1)
        df = df[:-forward_periods]
        
        return df
    
    def get_feature_columns(self, df: pd.DataFrame) -> list:
        """
        Get list of feature column names
        """
        exclude_cols = [
            'open_time', 'close_time', 'open', 'high', 'low', 'close', 'volume',
            'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
            'taker_buy_quote_asset_volume', 'ignore', 'target'
        ]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        return feature_cols
