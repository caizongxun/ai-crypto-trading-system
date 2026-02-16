import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.cluster import KMeans
import joblib
from datetime import datetime
import json
import os
from typing import Dict, Optional


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        return super().default(obj)


class MarketRegimeClassifier:
    """
    Market regime classifier for cryptocurrency trading.
    
    Classifies market into regimes based on:
    - Trend strength (ADX, EMA alignment)
    - Volatility (ATR, Bollinger width)
    - Volume profile
    - Momentum (RSI, MACD)
    
    Regimes:
    0: Strong Uptrend (high ADX, price > EMAs, positive momentum)
    1: Strong Downtrend (high ADX, price < EMAs, negative momentum)
    2: Choppy/Ranging (low ADX, price oscillating)
    3: High Volatility Breakout (expanding BB, high volume)
    4: Low Volatility Consolidation (narrow BB, low volume)
    """
    
    def __init__(self, model_dir: str = 'models'):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.model = None
        self.feature_importance = None
        self.regime_labels = None
        
    def create_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features specifically for regime classification
        """
        features = pd.DataFrame(index=df.index)
        
        # Trend features
        if 'adx' in df.columns:
            features['adx'] = df['adx']
            features['adx_strength'] = (df['adx'] > 25).astype(int)
        
        # Price position relative to EMAs
        for period in [9, 21, 50, 200]:
            if f'ema_{period}' in df.columns:
                features[f'price_above_ema{period}'] = (df['close'] > df[f'ema_{period}']).astype(int)
        
        # Volatility features
        if 'atr_percent' in df.columns:
            features['atr_pct'] = df['atr_percent']
            features['atr_high'] = (df['atr_percent'] > df['atr_percent'].rolling(50).mean()).astype(int)
        
        if 'bb_width' in df.columns:
            features['bb_width'] = df['bb_width']
            features['bb_squeeze'] = (df['bb_width'] < df['bb_width'].rolling(50).quantile(0.2)).astype(int)
            features['bb_expansion'] = (df['bb_width'] > df['bb_width'].rolling(50).quantile(0.8)).astype(int)
        
        # Volume features
        if 'volume_ratio_20' in df.columns:
            features['volume_surge'] = (df['volume_ratio_20'] > 1.5).astype(int)
            features['volume_dry'] = (df['volume_ratio_20'] < 0.5).astype(int)
        
        # Momentum features
        if 'rsi_14' in df.columns:
            features['rsi'] = df['rsi_14']
            features['rsi_overbought'] = (df['rsi_14'] > 70).astype(int)
            features['rsi_oversold'] = (df['rsi_14'] < 30).astype(int)
        
        if 'macd_diff' in df.columns:
            features['macd_diff'] = df['macd_diff']
            features['macd_positive'] = (df['macd_diff'] > 0).astype(int)
        
        # Directional movement
        if 'adx_pos' in df.columns and 'adx_neg' in df.columns:
            features['di_diff'] = df['adx_pos'] - df['adx_neg']
        
        # Price action
        features['returns_5'] = df['close'].pct_change(5)
        features['returns_20'] = df['close'].pct_change(20)
        features['volatility_20'] = df['close'].pct_change().rolling(20).std()
        
        return features.fillna(method='ffill').fillna(0)
    
    def auto_label_regimes(self, df: pd.DataFrame) -> np.ndarray:
        """
        Automatically label market regimes using clustering
        """
        regime_features = self.create_regime_features(df)
        
        # Select key features for clustering
        cluster_features = []
        for col in ['adx', 'atr_pct', 'bb_width', 'rsi', 'macd_diff', 'returns_20', 'volatility_20']:
            if col in regime_features.columns:
                cluster_features.append(col)
        
        if len(cluster_features) < 3:
            raise ValueError('Not enough features for regime clustering')
        
        X = regime_features[cluster_features].fillna(0)
        
        # Normalize features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # KMeans clustering
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        
        # Interpret clusters based on centroids
        centroids = pd.DataFrame(
            scaler.inverse_transform(kmeans.cluster_centers_),
            columns=cluster_features
        )
        
        self.regime_labels = labels
        return labels
    
    def train_model(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[np.ndarray] = None
    ) -> XGBClassifier:
        """
        Train regime classifier
        """
        params = {
            'n_estimators': 300,
            'max_depth': 5,
            'learning_rate': 0.1,
            'objective': 'multi:softprob',
            'num_class': 5,
            'eval_metric': 'mlogloss',
            'random_state': 42,
            'n_jobs': -1
        }
        
        self.model = XGBClassifier(**params)
        
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
        
        self.model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
        
        self.feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.model.feature_importances_.astype(float)
        }).sort_values('importance', ascending=False)
        
        return self.model
    
    def predict_regime(self, X: pd.DataFrame) -> tuple:
        """
        Predict market regime with probabilities
        
        Returns:
            (regime_class, probabilities)
        """
        if self.model is None:
            raise ValueError('Model not trained yet')
        
        regime = self.model.predict(X)
        probas = self.model.predict_proba(X)
        
        return regime, probas
    
    def save_model(self, symbol: str, timeframe: str, version: Optional[str] = None) -> str:
        if self.model is None:
            raise ValueError('No model to save')
        
        if version is None:
            version = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        model_name = f"regime_{symbol}_{timeframe}_v{version}"
        model_path = os.path.join(self.model_dir, f"{model_name}.pkl")
        
        model_data = {
            'model': self.model,
            'feature_importance': self.feature_importance,
            'symbol': symbol,
            'timeframe': timeframe,
            'version': version,
            'trained_at': datetime.now().isoformat()
        }
        
        joblib.dump(model_data, model_path)
        
        metadata = {
            'symbol': symbol,
            'timeframe': timeframe,
            'version': version,
            'trained_at': model_data['trained_at'],
            'top_features': self.feature_importance.head(10).to_dict('records')
        }
        
        metadata_path = os.path.join(self.model_dir, f"{model_name}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, cls=NumpyEncoder)
        
        return model_path
    
    def load_model(self, model_path: str) -> Dict:
        model_data = joblib.load(model_path)
        self.model = model_data['model']
        self.feature_importance = model_data.get('feature_importance')
        return model_data
