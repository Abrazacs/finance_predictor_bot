"""
Random Forest модель для прогнозирования
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from models.base_model import BaseModel
from config import config


class RandomForestModel(BaseModel):
    """Random Forest модель с лаговыми признаками"""

    def __init__(self):
        super().__init__("Random Forest")
        self.n_lags = config.RF_N_LAGS
        self.last_data = None
        self.data = None

    def create_lag_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Создание лаговых признаков"""
        df = data.copy()

        # Лаговые признаки
        for i in range(1, self.n_lags + 1):
            df[f'lag_{i}'] = df['price'].shift(i)

        # Скользящие средние
        df['ma_7'] = df['price'].rolling(window=7).mean()
        df['ma_30'] = df['price'].rolling(window=30).mean()

        # Скорость изменения
        df['price_change'] = df['price'].pct_change()

        return df.dropna()

    def train(self, data: pd.DataFrame, train_size: float = 0.8) -> float:
        """Обучение Random Forest"""
        self.data = data
        df = self.create_lag_features(data)

        split_idx = int(len(df) * train_size)
        train = df.iloc[:split_idx]
        test = df.iloc[split_idx:]

        feature_cols = [col for col in df.columns if col != 'price']
        X_train = train[feature_cols]
        y_train = train['price']
        X_test = test[feature_cols]
        y_test = test['price']

        self.model = RandomForestRegressor(
            n_estimators=config.RF_N_ESTIMATORS,
            max_depth=config.RF_MAX_DEPTH,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_train, y_train)

        predictions = self.model.predict(X_test)
        rmse = root_mean_squared_error(y_test, predictions)

        # Сохраняем последние данные для прогноза
        self.last_data = df.iloc[-1:].copy()
        self.trained = True

        return rmse

    def predict(self, steps: int) -> np.ndarray:
        """Прогнозирование на будущее"""
        if not self.trained:
            raise ValueError("Модель не обучена")

        predictions = []
        current_data = self.last_data.copy()

        df = self.create_lag_features(self.data)
        feature_cols = [col for col in df.columns if col != 'price']

        for _ in range(steps):
            pred = self.model.predict(current_data[feature_cols])[0]
            predictions.append(pred)

            # Обновляем лаги
            new_row = current_data.copy()
            new_row['price'] = pred
            for i in range(self.n_lags, 1, -1):
                if f'lag_{i}' in new_row.columns:
                    new_row[f'lag_{i}'] = current_data[f'lag_{i-1}'].values[0]
            new_row['lag_1'] = pred

            current_data = new_row

        return np.array(predictions)