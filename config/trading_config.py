import os
from dotenv import load_dotenv

load_dotenv()

class TradingConfig:
    # Data Configuration
    HF_DATASET_ID = os.getenv('HF_DATASET_ID', 'zongowo111/v2-crypto-ohlcv-data')
    
    SYMBOLS = [
        'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'DOGEUSDT',
        'XRPUSDT', 'DOTUSDT', 'UNIUSDT', 'LINKUSDT', 'LTCUSDT',
        'SOLUSDT', 'MATICUSDT', 'AVAXUSDT', 'ATOMUSDT', 'ALGOUSDT',
        'ARBUSDT', 'OPUSDT', 'NEARUSDT', 'FILUSDT', 'AAVEUSDT',
        'MKRUSDT', 'SNXUSDT', 'COMPUSDT', 'CRVUSDT', 'ENSUSDT',
        'GRTUSDT', 'SANDUSDT', 'MANAUSDT', 'GALAUSDT', 'IMXUSDT',
        'BALUSDT', 'BATUSDT', 'BCHUSDT', 'ETCUSDT', 'ENJUSDT',
        'KAVAUSDT', 'SPELLUSDT', 'ZRXUSDT'
    ]
    
    TIMEFRAMES = ['15m', '1h', '1d']
    PRIMARY_TIMEFRAME = '15m'
    
    # Trading Parameters
    INITIAL_CAPITAL = float(os.getenv('INITIAL_CAPITAL', 10))
    LEVERAGE = int(os.getenv('LEVERAGE', 3))
    TRADING_MODE = os.getenv('TRADING_MODE', 'paper')
    
    # Risk Management
    RISK_PER_TRADE = float(os.getenv('RISK_PER_TRADE', 0.05))
    MAX_POSITIONS = int(os.getenv('MAX_POSITIONS', 2))
    STOP_LOSS_ATR_MULTIPLIER = float(os.getenv('STOP_LOSS_ATR_MULTIPLIER', 1.5))
    TAKE_PROFIT_ATR_MULTIPLIER = float(os.getenv('TAKE_PROFIT_ATR_MULTIPLIER', 2.5))
    TRAILING_STOP_ACTIVATION = 1.5
    MAX_HOLDING_CANDLES = 96
    
    # Model Configuration
    PREDICTION_THRESHOLD = 0.60
    MIN_VOLUME_PERCENTILE = 30
    MIN_ATR_PERCENTILE = 20
    
    # Binance API
    BINANCE_API_KEY = os.getenv('BINANCE_API_KEY', '')
    BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET', '')
    
    # Backtesting
    COMMISSION_RATE = 0.001
    SLIPPAGE_RATE = 0.0005
    TRAIN_TEST_SPLIT = 0.7
    WALK_FORWARD_WINDOWS = 5
    
    # Feature Engineering
    FEATURE_WINDOWS = [5, 10, 20, 50, 100, 200]
    RSI_PERIODS = [7, 14, 21]
    EMA_PERIODS = [9, 21, 50, 200]
    MACD_PARAMS = [(12, 26, 9), (5, 35, 5)]
