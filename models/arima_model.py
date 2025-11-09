"""
ARIMA модель для прогнозирования
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import root_mean_squared_error
from models.base_model import BaseModel
from config import config
import logging

logger = logging.getLogger(__name__)


class ARIMAModel(BaseModel):
    """ARIMA модель для временных рядов"""

    def __init__(self):
        super().__init__("ARIMA")
        self.order = config.ARIMA_ORDER
        self.model_fit = None

    def train(self, data: pd.DataFrame, train_size: float = 0.8) -> float:
        """Обучение ARIMA"""
        split_idx = int(len(data) * train_size)
        train = data.iloc[:split_idx]['price']
        test = data.iloc[split_idx:]['price']

        try:
            model = SARIMAX(
                train,
                order=self.order,
                seasonal_order=(0, 0, 0, 0),
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            self.model_fit = model.fit(disp=False, maxiter=100)

            predictions = self.model_fit.forecast(steps=len(test))
            rmse = root_mean_squared_error(test, predictions)

            self.trained = True
            return rmse

        except Exception as e:
            logger.error(f"Ошибка обучения ARIMA: {e}")
            return float('inf')

    def predict(self, steps: int) -> np.ndarray:
        """Прогнозирование на будущее"""
        if not self.trained or self.model_fit is None:
            raise ValueError("Модель не обучена")

        predictions = self.model_fit.forecast(steps=steps)
        return predictions.values
