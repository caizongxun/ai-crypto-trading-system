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
from src.regime_classifier import MarketRegimeClassifier
from src.rule_based_entry import RuleBasedEntry
from src.regime_backtester import RegimeBacktester
from config.trading_config import TradingConfig

st.set_page_config(
    page_title='AI Regime-Based Trading',
    layout='wide'
)

st.title('AI Market Regime Trading System')
st.markdown('Model predicts market state, rules decide entry')
st.markdown('---')

@st.cache_resource
def load_data_loader():
    return CryptoDataLoader(TradingConfig.HF_DATASET_ID)

@st.cache_resource
def load_feature_engineer():
    return FeatureEngineer()

tab1, tab2, tab3 = st.tabs([
    'Regime Training',
    'Regime Backtest',
    'Regime Analysis'
])

with tab1:
    st.header('Train Market Regime Classifier')
    
    st.info('This model classifies market into 5 regimes: Strong Uptrend, Strong Downtrend, Choppy, High Volatility Breakout, Low Volatility Consolidation')
    
    col1, col2, col3 = st.columns(3)
    with col1:
        symbol = st.selectbox('Symbol', TradingConfig.SYMBOLS, index=0)
    with col2:
        timeframe = st.selectbox('Timeframe', TradingConfig.TIMEFRAMES, index=1)
    with col3:
        train_split = st.slider('Train Split', 0.5, 0.9, 0.7, 0.05)
    
    if st.button('Train Regime Classifier', type='primary'):
        with st.spinner('Training regime classifier...'):
            try:
                data_loader = load_data_loader()
                feature_engineer = load_feature_engineer()
                
                df = data_loader.prepare_training_data(symbol, timeframe)
                st.success(f'Loaded {len(df)} candles')
                
                df_features = feature_engineer.create_all_features(df)
                
                regime_classifier = MarketRegimeClassifier()
                regime_features = regime_classifier.create_regime_features(df_features)
                
                st.info('Auto-labeling market regimes using clustering...')
                regime_labels = regime_classifier.auto_label_regimes(df_features)
                
                split_idx = int(len(regime_features) * train_split)
                X_train = regime_features.iloc[:split_idx]
                y_train = regime_labels[:split_idx]
                X_test = regime_features.iloc[split_idx:]
                y_test = regime_labels[split_idx:]
                
                regime_classifier.train_model(X_train, y_train, X_test, y_test)
                
                from sklearn.metrics import accuracy_score
                y_pred, _ = regime_classifier.predict_regime(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                st.success(f'Regime classifier trained. Test accuracy: {accuracy*100:.2f}%')
                
                model_path = regime_classifier.save_model(symbol, timeframe)
                st.success(f'Model saved to {model_path}')
                
                st.subheader('Regime Distribution')
                regime_dist = pd.Series(regime_labels).value_counts().sort_index()
                fig = px.bar(
                    x=regime_dist.index,
                    y=regime_dist.values,
                    labels={'x': 'Regime', 'y': 'Count'},
                    title='Regime Distribution'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader('Top Features')
                top_features = regime_classifier.feature_importance.head(15)
                fig = px.bar(
                    top_features,
                    x='importance',
                    y='feature',
                    orientation='h',
                    title='Top 15 Features'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.session_state['regime_classifier'] = regime_classifier
                st.session_state['regime_symbol'] = symbol
                st.session_state['regime_timeframe'] = timeframe
                st.session_state['regime_features_cols'] = regime_features.columns.tolist()
                
            except Exception as e:
                st.error(f'Error: {str(e)}')
                import traceback
                st.code(traceback.format_exc())

with tab2:
    st.header('Backtest Regime-Based Strategy')
    
    if 'regime_classifier' not in st.session_state:
        st.warning('Train a regime classifier first')
    else:
        st.info('Entry rules: Uptrend=long pullbacks, Downtrend=short rallies, Breakout=momentum, Choppy/Consolidation=no trade')
        
        col1, col2, col3 = st.columns(3)
        with col1:
            initial_capital = st.number_input('Initial Capital', 10.0, 10000.0, 10.0, 10.0)
        with col2:
            leverage = st.number_input('Leverage', 1, 10, 3, 1)
        with col3:
            risk_per_trade = st.slider('Risk Per Trade (%)', 1, 10, 5, 1) / 100
        
        col1, col2 = st.columns(2)
        with col1:
            regime_confidence = st.slider('Min Regime Confidence', 0.3, 0.8, 0.5, 0.05)
        
        if st.button('Run Regime Backtest', type='primary'):
            with st.spinner('Running backtest...'):
                try:
                    data_loader = load_data_loader()
                    feature_engineer = load_feature_engineer()
                    
                    symbol = st.session_state['regime_symbol']
                    timeframe = st.session_state['regime_timeframe']
                    
                    df = data_loader.prepare_training_data(symbol, timeframe)
                    df_features = feature_engineer.create_all_features(df)
                    
                    regime_classifier = st.session_state['regime_classifier']
                    regime_features = regime_classifier.create_regime_features(df_features)
                    
                    regimes, regime_probas = regime_classifier.predict_regime(regime_features)
                    
                    entry_system = RuleBasedEntry()
                    entry_system.regime_confidence_threshold = float(regime_confidence)
                    
                    backtester = RegimeBacktester(
                        initial_capital=float(initial_capital),
                        commission=0.001,
                        slippage=0.0005,
                        leverage=int(leverage),
                        risk_per_trade=float(risk_per_trade)
                    )
                    
                    results = backtester.run_backtest(df_features, regimes, regime_probas)
                    
                    st.success('Backtest completed')
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric('Total Return', f"{results['total_return_pct']:.2f}%", f"{results['total_return']:.2f} USDT")
                    col2.metric('Total Trades', results['total_trades'])
                    col3.metric('Win Rate', f"{results['win_rate']:.2f}%")
                    col4.metric('Profit Factor', f"{results['profit_factor']:.2f}")
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric('Max Drawdown', f"{results['max_drawdown']:.2f}%")
                    col2.metric('Avg Win', f"{results['avg_win']:.2f} USDT")
                    col3.metric('Avg Loss', f"{results['avg_loss']:.2f} USDT")
                    
                    if 'regime_performance' in results:
                        st.subheader('Performance by Regime')
                        regime_perf = results['regime_performance']
                        if regime_perf:
                            perf_df = pd.DataFrame(regime_perf).T
                            st.dataframe(perf_df, use_container_width=True)
                    
                    if 'equity_df' in results and len(results['equity_df']) > 0:
                        st.subheader('Equity Curve')
                        equity_df = results['equity_df']
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=equity_df['timestamp'],
                            y=equity_df['equity'],
                            mode='lines',
                            name='Equity',
                            line=dict(color='cyan')
                        ))
                        fig.update_layout(
                            title='Portfolio Equity Over Time',
                            xaxis_title='Date',
                            yaxis_title='Equity (USDT)',
                            hovermode='x unified'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    if 'trades_df' in results and len(results['trades_df']) > 0:
                        st.subheader('Recent Trades')
                        trades_df = results['trades_df']
                        
                        display_df = trades_df[[
                            'entry_time', 'exit_time', 'direction', 'regime',
                            'entry_price', 'exit_price', 'pnl', 'pnl_percent',
                            'entry_reason', 'exit_reason'
                        ]].tail(50)
                        
                        st.dataframe(display_df, use_container_width=True)
                        
                        st.download_button(
                            'Download All Trades',
                            trades_df.to_csv(index=False),
                            'regime_trades.csv',
                            'text/csv'
                        )
                        
                        st.subheader('PnL Distribution')
                        fig = px.histogram(
                            trades_df,
                            x='pnl_percent',
                            nbins=50,
                            title='Trade Returns Distribution'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f'Error: {str(e)}')
                    import traceback
                    st.code(traceback.format_exc())

with tab3:
    st.header('Regime Analysis')
    
    if 'regime_classifier' in st.session_state:
        st.subheader('Regime Definitions')
        
        regime_info = pd.DataFrame([
            {
                'Regime': 0,
                'Name': 'Strong Uptrend',
                'Entry': 'Long on pullbacks to EMA9/21',
                'Stop/Target': '2.0 ATR / 4.0 ATR'
            },
            {
                'Regime': 1,
                'Name': 'Strong Downtrend',
                'Entry': 'Short on rallies to EMA9/21',
                'Stop/Target': '2.0 ATR / 4.0 ATR'
            },
            {
                'Regime': 2,
                'Name': 'Choppy/Ranging',
                'Entry': 'No trades (too risky)',
                'Stop/Target': 'N/A'
            },
            {
                'Regime': 3,
                'Name': 'High Vol Breakout',
                'Entry': 'Trade breakouts with volume',
                'Stop/Target': '1.0 ATR / 2.0 ATR'
            },
            {
                'Regime': 4,
                'Name': 'Low Vol Consolidation',
                'Entry': 'Wait for breakout',
                'Stop/Target': 'N/A'
            }
        ])
        
        st.dataframe(regime_info, use_container_width=True)
        
        st.subheader('Key Concepts')
        st.markdown("""
        **Why Regime-Based Trading?**
        
        - **Separation of Concerns**: Model predicts market state, not trade direction
        - **Reduces Overfitting**: Rules are consistent across market conditions
        - **Better Risk Management**: Different regimes use different stop/target ratios
        - **Interpretable**: You know WHY each trade was taken
        
        **How to Use:**
        
        1. Train regime classifier on your symbol/timeframe
        2. Run backtest to see performance by regime
        3. Adjust regime confidence threshold to filter entries
        4. Focus on regimes with highest win rate
        5. Disable poor-performing regimes by modifying rules
        """)
    else:
        st.info('Train a regime classifier to see analysis')

st.markdown('---')
st.markdown('Regime-Based Trading System - Research Tool Only')
