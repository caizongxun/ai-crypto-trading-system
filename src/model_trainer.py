import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score
import joblib
from datetime import datetime
import json
import os
from typing import Dict, Tuple, Optional


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


class ModelTrainer:
    """
    Multi-class XGBoost trainer.

    Target convention:
    - -1: short
    -  0: no-trade
    -  1: long

    Internal class mapping:
    - short -> 0
    - no-trade -> 1
    - long -> 2
    """

    def __init__(self, model_dir: str = 'models'):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.model = None
        self.feature_importance = None
        self.metrics = {}

    @staticmethod
    def _map_target(y: pd.Series) -> np.ndarray:
        yv = y.to_numpy()
        mapped = np.full_like(yv, fill_value=1, dtype=int)
        mapped[yv < 0] = 0
        mapped[yv > 0] = 2
        return mapped

    @staticmethod
    def _unmap_class(cls: int) -> int:
        if cls == 0:
            return -1
        if cls == 2:
            return 1
        return 0

    def _get_default_params(self) -> Dict:
        return {
            'n_estimators': 500,
            'max_depth': 7,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'objective': 'multi:softprob',
            'num_class': 3,
            'eval_metric': 'mlogloss',
            'random_state': 42,
            'n_jobs': -1,
            'tree_method': 'hist'
        }

    def train_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        params: Optional[Dict] = None
    ) -> XGBClassifier:
        if params is None:
            params = self._get_default_params()

        y_train_m = self._map_target(y_train)

        eval_set = None
        if X_val is not None and y_val is not None:
            y_val_m = self._map_target(y_val)
            eval_set = [(X_train, y_train_m), (X_val, y_val_m)]
        else:
            eval_set = [(X_train, y_train_m)]

        params = dict(params)
        params.setdefault('objective', 'multi:softprob')
        params.setdefault('num_class', 3)
        params.setdefault('eval_metric', 'mlogloss')

        self.model = XGBClassifier(**params)
        self.model.fit(X_train, y_train_m, eval_set=eval_set, verbose=False)

        self.feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.model.feature_importances_.astype(float)
        }).sort_values('importance', ascending=False)

        return self.model

    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        if self.model is None:
            raise ValueError('Model not trained yet')

        y_true = self._map_target(y_test)
        y_pred = self.model.predict(X_test).astype(int)

        metrics = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'macro_f1': float(f1_score(y_true, y_pred, average='macro')),
            'test_samples': int(len(y_test))
        }

        self.metrics = metrics
        return metrics

    def predict_signals(
        self,
        X: pd.DataFrame,
        proba_threshold: float = 0.45
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert model probabilities into {-1,0,1} trading signals.

        Rules:
        - If max class probability < proba_threshold -> no trade
        - Else choose argmax class and map to -1/0/1

        Returns:
            (signals, probas) where probas is (n, 3): [short, no-trade, long]
        """
        if self.model is None:
            raise ValueError('Model not trained yet')

        probas = self.model.predict_proba(X)
        cls = np.argmax(probas, axis=1)
        max_p = np.max(probas, axis=1)

        signals = np.array([self._unmap_class(int(c)) for c in cls], dtype=int)
        signals[max_p < proba_threshold] = 0

        return signals, probas

    def save_model(self, symbol: str, timeframe: str, version: Optional[str] = None) -> str:
        if self.model is None:
            raise ValueError('No model to save')

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
            'n_features': int(len(self.feature_importance))
        }

        joblib.dump(model_data, model_path)

        metadata_path = os.path.join(self.model_dir, f"{model_name}_metadata.json")
        metadata = {
            'symbol': symbol,
            'timeframe': timeframe,
            'version': version,
            'metrics': self.metrics,
            'trained_at': model_data['trained_at'],
            'n_features': model_data['n_features'],
            'top_10_features': self.feature_importance.head(10).to_dict('records')
        }

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, cls=NumpyEncoder)

        return model_path

    def load_model(self, model_path: str) -> Dict:
        model_data = joblib.load(model_path)
        self.model = model_data['model']
        self.feature_importance = model_data['feature_importance']
        self.metrics = model_data.get('metrics', {})
        return model_data

    def get_top_features(self, n: int = 20) -> pd.DataFrame:
        if self.feature_importance is None:
            raise ValueError('Model not trained yet')
        return self.feature_importance.head(n)
