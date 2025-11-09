"""
Утилиты для логирования
"""

import logging
from datetime import datetime
from config import config


def setup_logging():
    """Настройка логирования"""
    logging.basicConfig(
        format=config.LOG_FORMAT,
        level=logging.INFO,
        handlers=[
            logging.FileHandler('bot.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )


def log_user_request(
        user_id: int,
        ticker: str,
        amount: float,
        model: str,
        metric: float,
        profit: float
):
    """
    Логирование пользовательского запроса

    Args:
        user_id: ID пользователя
        ticker: Тикер компании
        amount: Сумма инвестиции
        model: Название модели
        metric: Метрика качества (RMSE)
        profit: Прибыль
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = (
        f"{timestamp}|{user_id}|{ticker}|{amount:.2f}|"
        f"{model}|{metric:.2f}|{profit:.2f}\n"
    )

    with open(config.LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(log_entry)