"""
Сервис для визуализации данных и прогнозов
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List
from config import config


class VisualizationService:
    """Сервис для создания графиков"""

    @staticmethod
    def plot_prediction(
            ticker: str,
            historical: pd.DataFrame,
            predictions: np.ndarray,
            buy_days: List[int],
            sell_days: List[int]
    ) -> str:
        """
        Создание графика с прогнозом

        Args:
            ticker: Тикер компании
            historical: Исторические данные
            predictions: Прогнозируемые цены
            buy_days: Дни для покупки
            sell_days: Дни для продажи

        Returns:
            Путь к сохраненному файлу
        """
        plt.figure(figsize=config.FIGURE_SIZE)

        # Исторические данные
        plt.plot(
            historical.index,
            historical['price'],
            label='Исторические данные',
            linewidth=2,
            color='#2E86AB'
        )

        # Прогноз
        future_dates = pd.date_range(
            start=historical.index[-1] + timedelta(days=1),
            periods=len(predictions)
        )
        plt.plot(
            future_dates,
            predictions,
            label='Прогноз',
            linewidth=2,
            linestyle='--',
            color='#F77F00'
        )

        # Сигналы покупки
        if buy_days:
            plt.scatter(
                [future_dates[i] for i in buy_days],
                [predictions[i] for i in buy_days],
                color='#06A77D',
                s=150,
                marker='^',
                label='Покупать',
                zorder=5,
                edgecolors='black',
                linewidths=1
            )

        # Сигналы продажи
        if sell_days:
            plt.scatter(
                [future_dates[i] for i in sell_days],
                [predictions[i] for i in sell_days],
                color='#D62828',
                s=150,
                marker='v',
                label='Продавать',
                zorder=5,
                edgecolors='black',
                linewidths=1
            )

        plt.xlabel('Дата', fontsize=12, fontweight='bold')
        plt.ylabel('Цена ($)', fontsize=12, fontweight='bold')
        plt.title(
            f'Прогноз цены акций {ticker} на {config.FORECAST_DAYS} дней',
            fontsize=14,
            fontweight='bold'
        )
        plt.legend(fontsize=10, loc='best')
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()

        filename = f'{ticker}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(filename, dpi=config.DPI, bbox_inches='tight')
        plt.close()

        return filename