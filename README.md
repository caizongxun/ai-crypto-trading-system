# AI Crypto Trading System

An AI-powered cryptocurrency trading research system featuring:
- Multi-class ML models (XGBoost) trained on OHLCV + technical indicators
- A realistic backtesting engine (commission, slippage, next-candle execution)
- A Streamlit web dashboard for training and backtesting
- A trading bot module for paper/live execution (Binance via CCXT)

Repository: [ai-crypto-trading-system](https://github.com/caizongxun/ai-crypto-trading-system)

## Key Principles

- Uses completed candles only for feature generation and prediction
- Backtest executes entries on the next candle open to reduce look-ahead bias
- Labels are generated using a volatility-scaled triple-barrier method (ATR-based)

## Project Structure

```
ai-crypto-trading-system/
├── src/
│   ├── data_loader.py            # HuggingFace data loading and preprocessing
│   ├── feature_engineering.py    # Technical indicator calculation
│   ├── labeling.py               # Triple-barrier labeling (ATR-based)
│   ├── model_trainer.py          # XGBoost multi-class training pipeline
│   ├── backtesting_engine.py     # Backtesting engine with next-open execution
│   ├── trading_bot.py            # Paper/live execution loop (CCXT)
│   └── risk_manager.py           # Position sizing and risk controls
├── app.py                        # Streamlit dashboard
├── config/
│   ├── trading_config.py         # Trading parameters
│   └── .env.example              # API keys template
├── models/
├── requirements.txt
└── README.md
```

## Installation

- Python 3.9+

```bash
git clone https://github.com/caizongxun/ai-crypto-trading-system.git
cd ai-crypto-trading-system
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run the Web Dashboard

```bash
streamlit run app.py
```

## Notes on Performance

- A 70%+ win rate is not guaranteed in live trading.
- You can tune trade frequency vs win rate mainly through: labeling horizon, barriers, and probability thresholds.
- Always validate with walk-forward and out-of-sample segments before risking capital.

## License

MIT License.
