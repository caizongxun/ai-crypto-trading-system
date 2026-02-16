from huggingface_hub import hf_hub_download
import pandas as pd
import numpy as np
from typing import Optional, Tuple
from datetime import datetime, timedelta
import pytz

class CryptoDataLoader:
    def __init__(self, dataset_id: str = 'zongowo111/v2-crypto-ohlcv-data'):
        self.dataset_id = dataset_id
        
    def load_klines(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        Load kline data from HuggingFace dataset
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            timeframe: Time period ('15m', '1h', '1d')
            
        Returns:
            DataFrame with OHLCV data
        """
        base = symbol.replace('USDT', '')
        filename = f"{base}_{timeframe}.parquet"
        path_in_repo = f"klines/{symbol}/{filename}"
        
        try:
            local_path = hf_hub_download(
                repo_id=self.dataset_id,
                filename=path_in_repo,
                repo_type='dataset'
            )
            df = pd.read_parquet(local_path)
            
            df['open_time'] = pd.to_datetime(df['open_time'])
            df['close_time'] = pd.to_datetime(df['close_time'])
            df = df.sort_values('open_time').reset_index(drop=True)
            df = df.drop_duplicates(subset=['open_time'], keep='last')
            
            return df
            
        except Exception as e:
            raise Exception(f"Failed to load {symbol} {timeframe}: {str(e)}")
    
    def prepare_training_data(self, 
                            symbol: str, 
                            timeframe: str,
                            start_date: Optional[str] = None,
                            end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Load and prepare data for model training
        Uses only completed candles to avoid look-ahead bias
        
        Args:
            symbol: Trading pair
            timeframe: Time period
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            
        Returns:
            Prepared DataFrame
        """
        df = self.load_klines(symbol, timeframe)
        
        current_time = datetime.now(pytz.UTC)
        df = df[df['close_time'] < current_time]
        
        if start_date:
            start = pd.to_datetime(start_date, utc=True)
            df = df[df['open_time'] >= start]
            
        if end_date:
            end = pd.to_datetime(end_date, utc=True)
            df = df[df['open_time'] <= end]
        
        if len(df) < 500:
            raise ValueError(f"Insufficient data: only {len(df)} candles available")
            
        return df
    
    def get_latest_completed_candle(self, symbol: str, timeframe: str) -> pd.Series:
        """
        Get the most recent completed candle
        Critical for live trading to avoid using incomplete data
        
        Args:
            symbol: Trading pair
            timeframe: Time period
            
        Returns:
            Series containing the latest completed candle
        """
        df = self.load_klines(symbol, timeframe)
        current_time = datetime.now(pytz.UTC)
        
        completed = df[df['close_time'] < current_time]
        
        if len(completed) == 0:
            raise ValueError("No completed candles available")
            
        return completed.iloc[-1]
    
    def split_train_test(self, 
                        df: pd.DataFrame, 
                        train_ratio: float = 0.7) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into training and testing sets chronologically
        
        Args:
            df: Input DataFrame
            train_ratio: Proportion of data for training
            
        Returns:
            Tuple of (train_df, test_df)
        """
        split_idx = int(len(df) * train_ratio)
        train = df.iloc[:split_idx].copy()
        test = df.iloc[split_idx:].copy()
        
        return train, test
    
    def create_walk_forward_splits(self, 
                                  df: pd.DataFrame,
                                  n_splits: int = 5,
                                  train_size: int = 2000,
                                  test_size: int = 500) -> list:
        """
        Create walk-forward validation splits
        
        Args:
            df: Input DataFrame
            n_splits: Number of splits
            train_size: Size of training window
            test_size: Size of test window
            
        Returns:
            List of (train_indices, test_indices) tuples
        """
        splits = []
        step = test_size
        
        for i in range(n_splits):
            start_idx = i * step
            train_end = start_idx + train_size
            test_end = train_end + test_size
            
            if test_end > len(df):
                break
                
            train_idx = list(range(start_idx, train_end))
            test_idx = list(range(train_end, test_end))
            splits.append((train_idx, test_idx))
        
        return splits
    
    def load_multiple_timeframes(self, 
                                symbol: str, 
                                timeframes: list = ['15m', '1h', '1d']) -> dict:
        """
        Load data for multiple timeframes for multi-timeframe analysis
        
        Args:
            symbol: Trading pair
            timeframes: List of timeframes
            
        Returns:
            Dictionary with timeframe as key and DataFrame as value
        """
        data = {}
        
        for tf in timeframes:
            try:
                data[tf] = self.load_klines(symbol, tf)
            except Exception as e:
                print(f"Warning: Could not load {tf} data: {str(e)}")
        
        return data
