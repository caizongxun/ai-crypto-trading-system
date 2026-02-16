import ccxt
import pandas as pd
import numpy as np
import time
from datetime import timedelta
from typing import Dict
import logging


class TradingBot:
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        model,
        risk_manager,
        symbol: str = 'BTC/USDT',
        timeframe: str = '15m',
        paper_trading: bool = True,
        proba_threshold: float = 0.45
    ):
        self.symbol = symbol
        self.timeframe = timeframe
        self.model = model
        self.risk_manager = risk_manager
        self.paper_trading = paper_trading
        self.proba_threshold = float(proba_threshold)

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
        try:
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['open_time'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df['close_time'] = df['open_time'] + pd.Timedelta(self.timeframe)
            return df
        except Exception as e:
            self.logger.error(f"Error fetching OHLCV: {str(e)}")
            return pd.DataFrame()

    def is_new_candle(self, current_candle_time) -> bool:
        if self.last_candle_time is None:
            self.last_candle_time = current_candle_time
            return True
        if pd.Timestamp(current_candle_time) > pd.Timestamp(self.last_candle_time):
            self.last_candle_time = current_candle_time
            return True
        return False

    def get_completed_candles(self, df: pd.DataFrame) -> pd.DataFrame:
        current_time = pd.Timestamp.now(tz='UTC')
        return df[df['close_time'] < current_time].copy()

    def generate_signal(self, features: pd.DataFrame) -> Dict:
        try:
            if len(features) == 0:
                return {'signal': 0, 'probas': None}

            latest_features = features.iloc[[-1]]
            probas = self.model.predict_proba(latest_features)[0]

            cls = int(np.argmax(probas))
            max_p = float(np.max(probas))

            signal = 0
            if max_p >= self.proba_threshold:
                if cls == 0:
                    signal = -1
                elif cls == 2:
                    signal = 1

            return {
                'signal': int(signal),
                'probas': probas,
                'timestamp': pd.Timestamp.now(tz='UTC')
            }
        except Exception as e:
            self.logger.error(f"Error generating signal: {str(e)}")
            return {'signal': 0, 'probas': None}

    def execute_trade(self, signal: int, price: float, atr: float):
        if signal == 0:
            return
        if not self.risk_manager.can_open_position():
            self.logger.info('Cannot open position: constraints hit')
            return

        direction = 'long' if signal > 0 else 'short'
        stop_loss = self.risk_manager.calculate_stop_loss(price, atr, direction)
        take_profit = self.risk_manager.calculate_take_profit(price, atr, direction)

        position_size = self.risk_manager.calculate_position_size(price, stop_loss, atr)
        if position_size == 0:
            self.logger.info('Position size too small, skipping trade')
            return

        self._execute_paper_trade(direction, price, position_size, stop_loss, take_profit, atr)

    def _execute_paper_trade(self, direction: str, price: float, size: float, stop_loss: float, take_profit: float, atr: float):
        position = {
            'id': f"{self.symbol}_{pd.Timestamp.now().timestamp()}",
            'symbol': self.symbol,
            'direction': direction,
            'entry_price': float(price),
            'entry_time': pd.Timestamp.now(tz='UTC'),
            'size': float(size),
            'stop_loss': float(stop_loss),
            'take_profit': float(take_profit),
            'atr': float(atr),
            'status': 'open'
        }

        self.positions[position['id']] = position
        self.risk_manager.add_position(position)
        self.logger.info(f"Paper trade executed: {direction} {size} {self.symbol} at {price}")

    def run(self, feature_engineer):
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

                current_price = float(completed_df.iloc[-1]['close'])
                current_atr = float(features_df.iloc[-1].get('atr', current_price * 0.01))

                if signal_data['signal'] != 0:
                    self.execute_trade(signal_data['signal'], current_price, current_atr)

                time.sleep(60)

            except KeyboardInterrupt:
                self.logger.info('Bot stopped by user')
                break
            except Exception as e:
                self.logger.error(f"Error in trading loop: {str(e)}")
                time.sleep(60)
