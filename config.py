"""
Конфигурация приложения
"""

import os
from dataclasses import dataclass


@dataclass
class Config:
    """Конфигурация бота"""

    # Telegram
    BOT_TOKEN: str = 'YOUR_BOT_TOKEN'

    # Параметры данных
    HISTORY_DAYS: int = 730  # 2 года
    FORECAST_DAYS: int = 30

    # Параметры обучения
    TRAIN_SIZE: float = 0.8
    LSTM_EPOCHS: int = 50
    LSTM_BATCH_SIZE: int = 32
    LSTM_LOOK_BACK: int = 60
    LSTM_HIDDEN_SIZE: int = 50
    LSTM_NUM_LAYERS: int = 2

    # Random Forest
    RF_N_ESTIMATORS: int = 100
    RF_MAX_DEPTH: int = 10
    RF_N_LAGS: int = 30

    # ARIMA
    ARIMA_ORDER: tuple = (5, 1, 2)

    # Логирование
    LOG_FILE: str = 'logs.txt'
    LOG_FORMAT: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    # Визуализация
    FIGURE_SIZE: tuple = (14, 7)
    DPI: int = 100


config = Config()