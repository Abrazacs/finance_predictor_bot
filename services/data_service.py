"""
Сервис для загрузки и обработки данных
"""

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from config import config
import logging

logger = logging.getLogger(__name__)


class DataService:
    """Сервис для работы с данными акций"""

    @staticmethod
    def load_stock_data(ticker: str) -> pd.DataFrame:
        """
        Загрузка исторических данных акций

        Args:
            ticker: Тикер компании (например, AAPL)

        Returns:
            DataFrame с ценами закрытия или None при ошибке
        """
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=config.HISTORY_DAYS)

            data = yf.download(ticker, start=start_date, end=end_date, progress=False)

            if data.empty:
                logger.error(f"Данные для {ticker} не найдены")
                return None

            # Извлекаем только цены закрытия
            df = data[['Close']].copy()
            df.columns = ['price']

            logger.info(f"Загружено {len(df)} записей для {ticker}")
            return df

        except Exception as e:
            logger.error(f"Ошибка загрузки данных для {ticker}: {e}")
            return None

    @staticmethod
    def validate_ticker(ticker: str) -> bool:
        """Проверка валидности тикера"""
        if not ticker or len(ticker) > 10:
            return False
        return ticker.isalnum()