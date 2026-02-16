import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib
from datetime import datetime
import json
import os
from typing import Dict, Tuple, Optional

class NumpyEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for numpy types
    """
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (pd.Timestamp, pd.DatetimeTZDtype)):
            return obj.isoformat()
        return super().default(obj)

class ModelTrainer:
    def __init__(self, model_dir: str = 'models'):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.model = None
        self.feature_importance = None
        self.metrics = {}
        
    def train_model(self,
                   X_train: pd.DataFrame,
                   y_train: pd.Series,
                   X_val: Optional[pd.DataFrame] = None,
                   y_val: Optional[pd.Series] = None,
                   params: Optional[Dict] = None) -> XGBClassifier:
        """
        Train XGBoost classification model
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            params: Model hyperparameters
            
        Returns:
            Trained model
        """
        if params is None:
            params = self._get_default_params()
        
        y_train_binary = self._convert_to_binary(y_train)
        
        if X_val is not None and y_val is not None:
            y_val_binary = self._convert_to_binary(y_val)
            eval_set = [(X_train, y_train_binary), (X_val, y_val_binary)]
        else:
            eval_set = [(X_train, y_train_binary)]
        
        self.model = XGBClassifier(**params)
        
        self.model.fit(
            X_train, 
            y_train_binary,
            eval_set=eval_set,
            verbose=False
        )
        
        self.feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return self.model
    
    def _convert_to_binary(self, y: pd.Series) -> pd.Series:
        """
        Convert multi-class target to binary (trade or no trade)
        1: Long, -1: Short both converted to 1 (trade signal)
        0: No trade remains 0
        """
        return (y != 0).astype(int)
    
    def _get_default_params(self) -> Dict:
        """
        Get default XGBoost parameters optimized for crypto trading
        """
        return {
            'n_estimators': 300,
            'max_depth': 7,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'random_state': 42,
            'n_jobs': -1,
            'tree_method': 'hist'
        }
    
    def _convert_metrics_to_serializable(self, metrics: Dict) -> Dict:
        """
        Convert numpy types to Python native types for JSON serialization
        
        Args:
            metrics: Dictionary with metrics
            
        Returns:
            Dictionary with serializable types
        """
        serializable = {}
        for key, value in metrics.items():
            if isinstance(value, (np.integer, np.int32, np.int64)):
                serializable[key] = int(value)
            elif isinstance(value, (np.floating, np.float32, np.float64)):
                serializable[key] = float(value)
            elif isinstance(value, np.ndarray):
                serializable[key] = value.tolist()
            else:
                serializable[key] = value
        return serializable
    
    def evaluate_model(self,
                      X_test: pd.DataFrame,
                      y_test: pd.Series,
                      threshold: float = 0.5) -> Dict:
        """
        Evaluate model performance
        
        Args:
            X_test: Test features
            y_test: Test labels
            threshold: Prediction probability threshold
            
        Returns:
            Dictionary of metrics
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        y_test_binary = self._convert_to_binary(y_test)
        
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        metrics = {
            'accuracy': float(accuracy_score(y_test_binary, y_pred)),
            'precision': float(precision_score(y_test_binary, y_pred, zero_division=0)),
            'recall': float(recall_score(y_test_binary, y_pred, zero_division=0)),
            'f1': float(f1_score(y_test_binary, y_pred, zero_division=0)),
            'test_samples': int(len(y_test)),
            'positive_predictions': int(y_pred.sum()),
            'positive_rate': float(y_pred.sum() / len(y_pred))
        }
        
        self.metrics = metrics
        return metrics
    
    def predict_signal(self,
                      X: pd.DataFrame,
                      y_original: pd.Series,
                      threshold: float = 0.6) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate trading signals from predictions
        
        Args:
            X: Features
            y_original: Original multi-class labels
            threshold: Probability threshold for signal
            
        Returns:
            Tuple of (signals, probabilities)
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        proba = self.model.predict_proba(X)[:, 1]
        
        signals = np.zeros(len(proba))
        
        high_confidence_idx = proba >= threshold
        signals[high_confidence_idx] = y_original[high_confidence_idx]
        
        return signals, proba
    
    def cross_validate(self,
                      X: pd.DataFrame,
                      y: pd.Series,
                      n_splits: int = 5) -> Dict:
        """
        Perform time series cross-validation
        
        Args:
            X: Features
            y: Labels
            n_splits: Number of CV folds
            
        Returns:
            Dictionary of CV metrics
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        scores = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train_fold = X.iloc[train_idx]
            y_train_fold = y.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
            y_val_fold = y.iloc[val_idx]
            
            self.train_model(X_train_fold, y_train_fold)
            metrics = self.evaluate_model(X_val_fold, y_val_fold)
            
            for key in scores.keys():
                scores[key].append(metrics[key])
        
        cv_results = {
            f'{key}_mean': float(np.mean(values)) for key, values in scores.items()
        }
        cv_results.update({
            f'{key}_std': float(np.std(values)) for key, values in scores.items()
        })
        
        return cv_results
    
    def save_model(self,
                  symbol: str,
                  timeframe: str,
                  version: Optional[str] = None) -> str:
        """
        Save trained model and metadata
        
        Args:
            symbol: Trading pair
            timeframe: Time period
            version: Model version
            
        Returns:
            Path to saved model
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        if version is None:
            version = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        model_name = f"{symbol}_{timeframe}_v{version}"
        model_path = os.path.join(self.model_dir, f"{model_name}.pkl")
        
        model_data = {
            'model': self.model,
            'feature_importance': self.feature_importance,
            'metrics': self.metrics,
            'symbol': symbol,
            'timeframe': timeframe,
            'version': version,
            'trained_at': datetime.now().isoformat(),
            'n_features': len(self.feature_importance)
        }
        
        joblib.dump(model_data, model_path)
        
        metadata_path = os.path.join(self.model_dir, f"{model_name}_metadata.json")
        
        top_features = self.feature_importance.head(10).copy()
        top_features['importance'] = top_features['importance'].astype(float)
        
        metadata = {
            'symbol': symbol,
            'timeframe': timeframe,
            'version': version,
            'metrics': self._convert_metrics_to_serializable(self.metrics),
            'trained_at': model_data['trained_at'],
            'n_features': int(model_data['n_features']),
            'top_10_features': top_features.to_dict('records')
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, cls=NumpyEncoder)
        
        return model_path
    
    def load_model(self, model_path: str) -> Dict:
        """
        Load trained model from file
        
        Args:
            model_path: Path to model file
            
        Returns:
            Model metadata dictionary
        """
        model_data = joblib.load(model_path)
        
        self.model = model_data['model']
        self.feature_importance = model_data['feature_importance']
        self.metrics = model_data['metrics']
        
        return model_data
    
    def get_top_features(self, n: int = 20) -> pd.DataFrame:
        """
        Get top N most important features
        
        Args:
            n: Number of features to return
            
        Returns:
            DataFrame with top features
        """
        if self.feature_importance is None:
            raise ValueError("Model not trained yet")
        
        return self.feature_importance.head(n)
