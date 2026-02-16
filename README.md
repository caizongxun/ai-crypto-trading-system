# AI Crypto Trading System

An advanced AI-powered cryptocurrency trading system featuring XGBoost machine learning models, comprehensive backtesting engine, and real-time automated trading capabilities with a professional web-based dashboard.

## Features

### Core Capabilities
- **AI-Driven Predictions**: XGBoost ensemble models trained on historical OHLCV data with 70+ technical indicators
- **Multi-Timeframe Support**: Optimized for 15m scalping, supports 1h and 1d timeframes
- **Complete Backtesting Engine**: Walk-forward validation with realistic slippage and commission modeling
- **Real-Time Trading**: Automated execution via Binance API with position management
- **Professional Dashboard**: Streamlit-based GUI for model training, backtesting, and live monitoring
- **Risk Management**: ATR-based position sizing, dynamic stop-loss, and profit targets

### Technical Highlights
- Uses completed candles only (no look-ahead bias)
- HuggingFace dataset integration for 38 cryptocurrency pairs
- Feature engineering with momentum, volatility, volume, and trend indicators
- Model versioning and performance tracking
- Paper trading mode for strategy validation

## Project Structure

```
ai-crypto-trading-system/
├── src/
│   ├── data_loader.py       # HuggingFace data loading and preprocessing
│   ├── feature_engineering.py # Technical indicator calculation
│   ├── model_trainer.py      # XGBoost model training pipeline
│   ├── backtesting_engine.py # Complete backtesting system
│   ├── trading_bot.py        # Live trading execution
│   └── risk_manager.py       # Position sizing and risk controls
├── app.py                   # Streamlit dashboard application
├── config/
│   ├── trading_config.py    # Trading parameters
│   └── .env.example         # API keys template
├── models/                  # Trained model storage
├── requirements.txt
└── README.md
```

## Installation

### Prerequisites
- Python 3.9 or higher
- Git
- Binance account (for live trading)

### Setup Steps

1. Clone the repository:
```bash
git clone https://github.com/caizongxun/ai-crypto-trading-system.git
cd ai-crypto-trading-system
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure API keys:
```bash
cp config/.env.example config/.env
# Edit config/.env with your Binance API credentials
```

## Usage

### 1. Launch Dashboard
```bash
streamlit run app.py
```

### 2. Train Models
- Navigate to "Model Training" tab
- Select cryptocurrency pair (e.g., BTCUSDT)
- Choose timeframe (15m recommended)
- Configure training parameters
- Click "Train Model"

### 3. Run Backtests
- Go to "Backtesting" tab
- Load trained model
- Set date range and initial capital
- Execute backtest to evaluate performance

### 4. Start Live Trading
- Switch to "Live Trading" tab
- Enable paper trading for testing
- Monitor positions and performance in real-time
- Review trade history and analytics

## Trading Strategy

### Signal Generation
The system uses XGBoost classification to predict price movements:
- **Long Signal**: Model predicts upward movement with >60% probability
- **Short Signal**: Model predicts downward movement with >60% probability
- **No Trade**: Prediction confidence below threshold

### Entry Rules
- Only trade on completed candles (no intra-candle decisions)
- Minimum volume filter to ensure liquidity
- ATR-based volatility check to avoid choppy markets
- Maximum 3 concurrent positions

### Exit Rules
- Dynamic stop-loss: 1.5x ATR from entry
- Take-profit: 2.5x ATR (risk-reward ratio 1:1.67)
- Trailing stop activation at 1.5x ATR profit
- Maximum holding period: 24 candles

### Position Sizing
- Risk per trade: 2% of account balance
- Position size calculated using ATR volatility
- Leverage: 1x-3x (configurable)

## Performance Optimization

### For 15m Timeframe
- Target: 20-40 trades per day
- Win rate objective: 70%+
- Average hold time: 2-4 hours
- Sharpe ratio target: >2.0

### Model Features (70+ indicators)
- **Momentum**: RSI, Stochastic, Williams %R, ROC
- **Trend**: EMA (9,21,50,200), MACD, ADX, Supertrend
- **Volatility**: Bollinger Bands, ATR, Keltner Channels
- **Volume**: OBV, VWAP, Volume MA, CMF
- **Price Action**: Support/Resistance, Pivot Points, Candle patterns

## Risk Warning

Cryptocurrency trading involves substantial risk of loss. This system is provided for educational purposes only. Always:
- Start with paper trading
- Test thoroughly on historical data
- Never risk more than you can afford to lose
- Monitor positions actively
- Use appropriate stop-losses

## 10U Challenge Configuration

For aggressive growth targeting (high risk):
```python
INITIAL_CAPITAL = 10  # USDT
RISK_PER_TRADE = 0.05  # 5% per trade
LEVERAGE = 3  # 3x leverage
MAX_POSITIONS = 2
MIN_WIN_RATE = 0.70
```

## License

MIT License - See LICENSE file for details

## Contributing

Contributions welcome. Please open an issue first to discuss proposed changes.

## Support

For issues or questions, please open a GitHub issue.

## Disclaimer

This software is for educational and research purposes only. Use at your own risk. The authors assume no liability for financial losses incurred through use of this system.
