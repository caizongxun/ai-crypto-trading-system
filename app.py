import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import CryptoDataLoader
from src.feature_engineering import FeatureEngineer
from src.labeling import triple_barrier_labeling
from src.model_trainer import ModelTrainer
from src.backtesting_engine import BacktestingEngine
from config.trading_config import TradingConfig

st.set_page_config(
    page_title='AI Crypto Trading System',
    layout='wide'
)

st.title('AI Crypto Trading System')
st.markdown('---')

@st.cache_resource
def load_data_loader():
    return CryptoDataLoader(TradingConfig.HF_DATASET_ID)

@st.cache_resource
def load_feature_engineer():
    return FeatureEngineer()

tab1, tab2, tab3 = st.tabs([
    'Model Training',
    'Backtesting',
    'Analytics'
])

with tab1:
    st.header('Train AI Model')

    col1, col2, col3 = st.columns(3)
    with col1:
        symbol = st.selectbox('Select Symbol', TradingConfig.SYMBOLS, index=TradingConfig.SYMBOLS.index('BTCUSDT'))
    with col2:
        timeframe = st.selectbox('Timeframe', TradingConfig.TIMEFRAMES, index=TradingConfig.TIMEFRAMES.index('15m'))
    with col3:
        train_test_split = st.slider('Train/Test Split', 0.5, 0.9, 0.7, 0.05)

    st.subheader('Labeling (Triple Barrier)')
    col1, col2, col3 = st.columns(3)
    with col1:
        horizon = st.number_input('Horizon Candles', min_value=4, max_value=96, value=12, step=1)
    with col2:
        pt_mult = st.number_input('Profit Barrier (ATR Mult)', min_value=0.5, max_value=5.0, value=2.0, step=0.5)
    with col3:
        sl_mult = st.number_input('Stop Barrier (ATR Mult)', min_value=0.5, max_value=5.0, value=1.5, step=0.5)

    st.subheader('Model Parameters')
    col1, col2, col3 = st.columns(3)
    with col1:
        n_estimators = st.number_input('N Estimators', 200, 2000, 600, 50)
    with col2:
        max_depth = st.number_input('Max Depth', 3, 15, 7, 1)
    with col3:
        learning_rate = st.number_input('Learning Rate', 0.01, 0.3, 0.05, 0.01)

    if st.button('Train Model', type='primary'):
        with st.spinner('Loading data and training model...'):
            try:
                data_loader = load_data_loader()
                feature_engineer = load_feature_engineer()

                df = data_loader.prepare_training_data(symbol, timeframe)
                st.success(f'Loaded {len(df)} candles')

                df_features = feature_engineer.create_all_features(df)
                df_labeled = triple_barrier_labeling(
                    df_features,
                    horizon=int(horizon),
                    pt_atr_mult=float(pt_mult),
                    sl_atr_mult=float(sl_mult),
                    atr_col='atr'
                )

                feature_cols = feature_engineer.get_feature_columns(df_labeled)
                X = df_labeled[feature_cols]
                y = df_labeled['target']

                split_idx = int(len(X) * train_test_split)
                X_train = X.iloc[:split_idx]
                y_train = y.iloc[:split_idx]
                X_test = X.iloc[split_idx:]
                y_test = y.iloc[split_idx:]

                params = {
                    'n_estimators': int(n_estimators),
                    'max_depth': int(max_depth),
                    'learning_rate': float(learning_rate),
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'objective': 'multi:softprob',
                    'num_class': 3,
                    'eval_metric': 'mlogloss',
                    'random_state': 42,
                    'n_jobs': -1
                }

                trainer = ModelTrainer()
                trainer.train_model(X_train, y_train, X_test, y_test, params)
                metrics = trainer.evaluate_model(X_test, y_test)

                st.success('Model trained successfully')

                col1, col2, col3 = st.columns(3)
                col1.metric('Accuracy', f"{metrics['accuracy']*100:.2f}%")
                col2.metric('Macro F1', f"{metrics['macro_f1']*100:.2f}%")
                col3.metric('Test Samples', f"{metrics['test_samples']}")

                model_path = trainer.save_model(symbol, timeframe)
                st.success(f'Model saved to {model_path}')

                st.subheader('Top 20 Important Features')
                top_features = trainer.get_top_features(20)
                fig = px.bar(top_features, x='importance', y='feature', orientation='h', title='Feature Importance')
                st.plotly_chart(fig, use_container_width=True)

                st.session_state['trained_model'] = trainer
                st.session_state['model_symbol'] = symbol
                st.session_state['model_timeframe'] = timeframe
                st.session_state['feature_cols'] = feature_cols
                st.session_state['label_params'] = {
                    'horizon': int(horizon),
                    'pt_mult': float(pt_mult),
                    'sl_mult': float(sl_mult)
                }

            except Exception as e:
                st.error(f'Error: {str(e)}')

with tab2:
    st.header('Backtest Strategy')

    if 'trained_model' not in st.session_state:
        st.warning('Please train a model first in the Model Training tab')
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            initial_capital = st.number_input('Initial Capital (USDT)', min_value=10.0, value=10.0, step=10.0)
        with col2:
            leverage = st.number_input('Leverage', 1, 10, 3, 1)
        with col3:
            risk_per_trade = st.slider('Risk Per Trade (%)', 1, 10, 5, 1) / 100

        col1, col2, col3 = st.columns(3)
        with col1:
            stop_loss_mult = st.number_input('Stop Loss (ATR Mult)', 0.5, 5.0, 1.5, 0.5)
        with col2:
            take_profit_mult = st.number_input('Take Profit (ATR Mult)', 1.0, 10.0, 2.5, 0.5)
        with col3:
            proba_threshold = st.slider('Probability Threshold', 0.30, 0.80, 0.45, 0.01)

        if st.button('Run Backtest', type='primary'):
            with st.spinner('Running backtest...'):
                try:
                    data_loader = load_data_loader()
                    feature_engineer = load_feature_engineer()

                    symbol = st.session_state['model_symbol']
                    timeframe = st.session_state['model_timeframe']
                    lp = st.session_state['label_params']

                    df = data_loader.prepare_training_data(symbol, timeframe)
                    df_features = feature_engineer.create_all_features(df)
                    df_labeled = triple_barrier_labeling(
                        df_features,
                        horizon=int(lp['horizon']),
                        pt_atr_mult=float(lp['pt_mult']),
                        sl_atr_mult=float(lp['sl_mult']),
                        atr_col='atr'
                    )

                    feature_cols = st.session_state['feature_cols']
                    X = df_labeled[feature_cols]

                    trainer = st.session_state['trained_model']
                    signals, probas = trainer.predict_signals(X, proba_threshold=float(proba_threshold))

                    backtester = BacktestingEngine(
                        initial_capital=float(initial_capital),
                        commission=0.001,
                        slippage=0.0005,
                        leverage=int(leverage)
                    )

                    atr_values = df_labeled['atr'].to_numpy(dtype=float)

                    results = backtester.run_backtest(
                        df_labeled,
                        signals,
                        atr_values,
                        stop_loss_multiplier=float(stop_loss_mult),
                        take_profit_multiplier=float(take_profit_mult),
                        risk_per_trade=float(risk_per_trade)
                    )

                    st.success('Backtest completed')

                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric('Total Return', f"{results['total_return_pct']:.2f}%", f"{results['total_return']:.2f} USDT")
                    col2.metric('Total Trades', results['total_trades'])
                    col3.metric('Win Rate', f"{results['win_rate']:.2f}%")
                    col4.metric('Profit Factor', f"{results['profit_factor']:.2f}")

                    if 'equity_df' in results and len(results['equity_df']) > 0:
                        st.subheader('Equity Curve')
                        equity_df = results['equity_df']
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=equity_df['timestamp'], y=equity_df['equity'], mode='lines', name='Equity'))
                        fig.update_layout(title='Portfolio Equity Over Time', xaxis_title='Date', yaxis_title='Equity (USDT)', hovermode='x unified')
                        st.plotly_chart(fig, use_container_width=True)

                    if 'trades_df' in results and len(results['trades_df']) > 0:
                        st.subheader('Trade History')
                        trades_df = results['trades_df']
                        st.dataframe(
                            trades_df[['entry_time', 'exit_time', 'direction', 'entry_price', 'exit_price', 'pnl', 'pnl_percent', 'exit_reason']],
                            use_container_width=True
                        )
                        st.download_button('Download Trades CSV', trades_df.to_csv(index=False), 'trades.csv', 'text/csv')

                except Exception as e:
                    st.error(f'Error: {str(e)}')

with tab3:
    st.header('Analytics')

    if os.path.exists('models'):
        model_files = [f for f in os.listdir('models') if f.endswith('.pkl')]
        if model_files:
            st.subheader('Available Models')
            models_data = []
            for model_file in model_files:
                parts = model_file.replace('.pkl', '').split('_')
                if len(parts) >= 3:
                    models_data.append({
                        'Model': model_file,
                        'Symbol': parts[0],
                        'Timeframe': parts[1],
                        'Version': parts[2] if len(parts) > 2 else 'v1'
                    })
            if models_data:
                st.dataframe(pd.DataFrame(models_data), use_container_width=True)
        else:
            st.info('No trained models found. Train a model first.')
    else:
        st.info('Models directory not found.')

st.markdown('---')
st.markdown('AI Crypto Trading System. For research purposes only.')
