"""
Сервис для прогнозирования цен акций
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from models.random_forest import RandomForestModel
from models.arima_model import ARIMAModel
from models.lstm_model import LSTMModel
from models.base_model import BaseModel
from config import config
import logging

logger = logging.getLogger(__name__)


class PredictionService:
    """Сервис для обучения моделей и прогнозирования"""

    def __init__(self):
        self.models = {
            'Random Forest': RandomForestModel(),
            'ARIMA': ARIMAModel(),
            'LSTM': LSTMModel()
        }
        self.best_model_name: Optional[str] = None
        self.best_rmse: float = float('inf')
        self.results: Dict[str, float] = {}

    def train_all_models(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Обучение всех моделей

        Args:
            data: DataFrame с историческими данными

        Returns:
            Словарь {название_модели: RMSE}
        """
        results = {}

        for name, model in self.models.items():
            logger.info(f"Обучение модели {name}...")
            try:
                rmse = model.train(data, train_size=config.TRAIN_SIZE)
                results[name] = rmse
                logger.info(f"{name}: RMSE = {rmse:.2f}")
            except Exception as e:
                logger.error(f"Ошибка обучения {name}: {e}")
                results[name] = float('inf')

        # Выбор лучшей модели
        self.results = results
        self.best_model_name = min(results.keys(), key=lambda k: results[k])
        self.best_rmse = results[self.best_model_name]

        logger.info(f"Лучшая модель: {self.best_model_name} (RMSE={self.best_rmse:.2f})")

        return results

    def get_best_model(self) -> BaseModel:
        """Получить лучшую модель"""
        if self.best_model_name is None:
            raise ValueError("Модели не обучены")
        return self.models[self.best_model_name]

    def predict(self, steps: int = 30) -> np.ndarray:
        """
        Прогнозирование на будущее

        Args:
            steps: Количество дней для прогноза

        Returns:
            Массив прогнозируемых цен
        """
        best_model = self.get_best_model()
        predictions = best_model.predict(steps)
        return predictions

    def get_results_summary(self) -> Dict[str, any]:
        """Получить сводку результатов"""
        return {
            'best_model': self.best_model_name,
            'best_rmse': self.best_rmse,
            'all_results': self.results
        }