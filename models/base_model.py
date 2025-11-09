"""
Базовый класс для всех моделей
"""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


class BaseModel(ABC):
    """Базовый класс для моделей прогнозирования"""

    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.trained = False

    @abstractmethod
    def train(self, data: pd.DataFrame, train_size: float) -> float:
        """
        Обучение модели

        Args:
            data: DataFrame с историческими данными
            train_size: Размер обучающей выборки (0-1)

        Returns:
            RMSE на тестовой выборке
        """
        pass

    @abstractmethod
    def predict(self, steps: int) -> np.ndarray:
        """
        Прогнозирование на будущее

        Args:
            steps: Количество шагов для прогноза

        Returns:
            Массив прогнозируемых значений
        """
        pass

    def get_name(self) -> str:
        """Получить название модели"""
        return self.name

    def is_trained(self) -> bool:
        """Проверить, обучена ли модель"""
        return self.trained