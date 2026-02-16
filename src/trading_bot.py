import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import pytz
from typing import Dict, Optional, List
import logging

class TradingBot:
    def __init__(self,
                 api_key: str,
                 api_secret: str,
                 model,
                 risk_manager,
                 symbol: str = 'BTC/USDT',
                 timeframe: str = '15m',
                 paper_trading: bool = True):
        """
        Initialize trading bot
        
        Args:
            api_key: Binance API key
            api_secret: Binance API secret
            model: Trained ML model
            risk_manager: Risk manager instance
            symbol: Trading pair
            timeframe: Time period
            paper_trading: Use paper trading mode
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.model = model
        self.risk_manager = risk_manager
        self.paper_trading = paper_trading
        
        if paper_trading:
            self.exchange = ccxt.binance({
                'apiKey': api_key,
                'secret': api_secret,
                'enableRateLimit': True,
                'options': {'defaultType': 'future'}
            })
        else:
            self.exchange = ccxt.binance({
                'apiKey': api_key,
                'secret': api_secret,
                'enableRateLimit': True,
                'options': {'defaultType': 'future'}
            })
        
        self.positions = {}
        self.trade_history = []
        self.last_candle_time = None
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def fetch_ohlcv(self, limit: int = 500) -> pd.DataFrame:
        """
        Fetch OHLCV data from exchange
        
        Args:
            limit: Number of candles to fetch
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(
                self.symbol,
                self.timeframe,
                limit=limit
            )
            
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            
            df['open_time'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df['close_time'] = df['open_time'] + pd.Timedelta(self.timeframe)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching OHLCV: {str(e)}")
            return pd.DataFrame()
    
    def is_new_candle(self, current_candle_time) -> bool:
        """
        Check if new candle has formed
        
        Args:
            current_candle_time: Current candle timestamp (pd.Timestamp or datetime)
            
        Returns:
            True if new candle
        """
        if self.last_candle_time is None:
            self.last_candle_time = current_candle_time
            return True
        
        if pd.Timestamp(current_candle_time) > pd.Timestamp(self.last_candle_time):
            self.last_candle_time = current_candle_time
            return True
        
        return False
    
    def get_completed_candles(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get only completed candles (exclude current forming candle)
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with completed candles only
        """
        current_time = pd.Timestamp.now(tz='UTC')
        completed = df[df['close_time'] < current_time].copy()
        return completed
    
    def generate_signal(self, features: pd.DataFrame) -> Dict:
        """
        Generate trading signal from features
        
        Args:
            features: Feature DataFrame
            
        Returns:
            Signal dictionary
        """
        try:
            if len(features) == 0:
                return {'signal': 0, 'probability': 0}
            
            latest_features = features.iloc[[-1]]
            
            probability = self.model.predict_proba(latest_features)[0, 1]
            
            signal = 0
            if probability >= 0.6:
                signal = 1
            elif probability <= 0.4:
                signal = -1
            
            return {
                'signal': signal,
                'probability': probability,
                'timestamp': pd.Timestamp.now(tz='UTC')
            }
            
        except Exception as e:
            self.logger.error(f"Error generating signal: {str(e)}")
            return {'signal': 0, 'probability': 0}
    
    def execute_trade(self,
                     signal: int,
                     price: float,
                     atr: float):
        """
        Execute trade based on signal
        
        Args:
            signal: Trading signal (1: long, -1: short, 0: no trade)
            price: Current price
            atr: Current ATR
        """
        if signal == 0:
            return
        
        if not self.risk_manager.can_open_position():
            self.logger.info("Cannot open position: max positions reached or insufficient capital")
            return
        
        direction = 'long' if signal > 0 else 'short'
        
        stop_loss = self.risk_manager.calculate_stop_loss(price, atr, direction)
        take_profit = self.risk_manager.calculate_take_profit(price, atr, direction)
        
        position_size = self.risk_manager.calculate_position_size(
            price, stop_loss, atr
        )
        
        if position_size == 0:
            self.logger.info("Position size too small, skipping trade")
            return
        
        if self.paper_trading:
            self._execute_paper_trade(
                direction, price, position_size,
                stop_loss, take_profit, atr
            )
        else:
            self._execute_live_trade(
                direction, price, position_size,
                stop_loss, take_profit
            )
    
    def _execute_paper_trade(self,
                            direction: str,
                            price: float,
                            size: float,
                            stop_loss: float,
                            take_profit: float,
                            atr: float):
        """
        Execute paper trade (simulated)
        """
        position = {
            'id': f"{self.symbol}_{pd.Timestamp.now().timestamp()}",
            'symbol': self.symbol,
            'direction': direction,
            'entry_price': price,
            'entry_time': pd.Timestamp.now(tz='UTC'),
            'size': size,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'atr': atr,
            'status': 'open'
        }
        
        self.positions[position['id']] = position
        self.risk_manager.add_position(position)
        
        self.logger.info(f"Paper trade executed: {direction} {size} {self.symbol} at {price}")
    
    def _execute_live_trade(self,
                           direction: str,
                           price: float,
                           size: float,
                           stop_loss: float,
                           take_profit: float):
        """
        Execute live trade on exchange
        """
        try:
            side = 'buy' if direction == 'long' else 'sell'
            
            order = self.exchange.create_market_order(
                self.symbol,
                side,
                size
            )
            
            if order['status'] == 'closed':
                position = {
                    'id': order['id'],
                    'symbol': self.symbol,
                    'direction': direction,
                    'entry_price': order['average'],
                    'entry_time': pd.Timestamp.now(tz='UTC'),
                    'size': size,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'status': 'open'
                }
                
                self.positions[position['id']] = position
                self.risk_manager.add_position(position)
                
                self.logger.info(f"Live trade executed: {direction} {size} {self.symbol} at {order['average']}")
                
        except Exception as e:
            self.logger.error(f"Error executing live trade: {str(e)}")
    
    def check_positions(self, current_price: float, current_atr: float):
        """
        Check and manage open positions
        
        Args:
            current_price: Current market price
            current_atr: Current ATR
        """
        for position_id, position in list(self.positions.items()):
            if position['status'] != 'open':
                continue
            
            should_close, reason = self._check_exit_conditions(
                position, current_price
            )
            
            if should_close:
                self._close_position(position_id, current_price, reason)
            else:
                new_stop = self.risk_manager.update_trailing_stop(
                    position, current_price, current_atr
                )
                if new_stop:
                    position['stop_loss'] = new_stop
                    self.logger.info(f"Trailing stop updated for {position_id}: {new_stop}")
    
    def _check_exit_conditions(self,
                              position: Dict,
                              current_price: float) -> tuple:
        """
        Check if position should be closed
        
        Returns:
            Tuple of (should_close, reason)
        """
        if position['direction'] == 'long':
            if current_price <= position['stop_loss']:
                return True, 'Stop Loss'
            elif current_price >= position['take_profit']:
                return True, 'Take Profit'
        else:
            if current_price >= position['stop_loss']:
                return True, 'Stop Loss'
            elif current_price <= position['take_profit']:
                return True, 'Take Profit'
        
        current_time = pd.Timestamp.now(tz='UTC')
        entry_time = pd.Timestamp(position['entry_time'])
        holding_time = current_time - entry_time
        
        if holding_time > pd.Timedelta(hours=24):
            return True, 'Max Holding Time'
        
        return False, ''
    
    def _close_position(self,
                       position_id: str,
                       exit_price: float,
                       reason: str):
        """
        Close position
        """
        position = self.positions[position_id]
        
        if position['direction'] == 'long':
            pnl = (exit_price - position['entry_price']) * position['size']
        else:
            pnl = (position['entry_price'] - exit_price) * position['size']
        
        trade_record = {
            'symbol': position['symbol'],
            'direction': position['direction'],
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'entry_time': position['entry_time'],
            'exit_time': pd.Timestamp.now(tz='UTC'),
            'size': position['size'],
            'pnl': pnl,
            'pnl_percent': (pnl / position['size']) * 100,
            'exit_reason': reason
        }
        
        self.trade_history.append(trade_record)
        position['status'] = 'closed'
        self.risk_manager.remove_position(position_id)
        
        self.logger.info(f"Position closed: {position_id}, PnL: {pnl:.2f}, Reason: {reason}")
    
    def get_account_balance(self) -> float:
        """
        Get current account balance
        
        Returns:
            Balance in USDT
        """
        try:
            balance = self.exchange.fetch_balance()
            return balance['USDT']['free']
        except:
            return 0
    
    def run(self, feature_engineer):
        """
        Main trading loop
        
        Args:
            feature_engineer: Feature engineering instance
        """
        self.logger.info(f"Starting trading bot for {self.symbol} on {self.timeframe}")
        
        while True:
            try:
                df = self.fetch_ohlcv(limit=500)
                
                if len(df) == 0:
                    time.sleep(60)
                    continue
                
                completed_df = self.get_completed_candles(df)
                
                if len(completed_df) < 200:
                    time.sleep(60)
                    continue
                
                latest_candle_time = completed_df.iloc[-1]['open_time']
                
                if not self.is_new_candle(latest_candle_time):
                    time.sleep(30)
                    continue
                
                features_df = feature_engineer.create_all_features(completed_df)
                feature_cols = feature_engineer.get_feature_columns(features_df)
                features = features_df[feature_cols]
                
                signal_data = self.generate_signal(features)
                
                current_price = completed_df.iloc[-1]['close']
                current_atr = features_df.iloc[-1].get('atr', current_price * 0.01)
                
                self.check_positions(current_price, current_atr)
                
                if signal_data['signal'] != 0:
                    self.execute_trade(
                        signal_data['signal'],
                        current_price,
                        current_atr
                    )
                
                time.sleep(60)
                
            except KeyboardInterrupt:
                self.logger.info("Bot stopped by user")
                break
            except Exception as e:
                self.logger.error(f"Error in trading loop: {str(e)}")
                time.sleep(60)
