"""
Утилиты для определения торговых сигналов
"""

import numpy as np
from scipy.signal import argrelextrema
from typing import List, Tuple


class TradingSignals:
    """Класс для определения торговых сигналов"""

    @staticmethod
    def find_extrema(predictions: np.ndarray, order: int = 5) -> Tuple[List[int], List[int]]:
        """
        Поиск локальных минимумов и максимумов

        Args:
            predictions: Массив прогнозируемых цен
            order: Порядок для определения экстремумов

        Returns:
            Кортеж (дни покупки, дни продажи)
        """
        local_min = argrelextrema(predictions, np.less, order=order)[0]
        local_max = argrelextrema(predictions, np.greater, order=order)[0]

        return local_min.tolist(), local_max.tolist()

    @staticmethod
    def calculate_profit(
            predictions: np.ndarray,
            investment: float,
            buy_days: List[int],
            sell_days: List[int]
    ) -> Tuple[float, str]:
        """
        Расчет потенциальной прибыли

        Args:
            predictions: Массив прогнозируемых цен
            investment: Сумма инвестиции
            buy_days: Дни для покупки
            sell_days: Дни для продажи

        Returns:
            Кортеж (общая прибыль, описание стратегии)
        """
        if not buy_days or not sell_days:
            return 0.0, "Недостаточно сигналов для расчета стратегии"

        strategy = []
        total_profit = 0

        for buy_day in buy_days:
            # Находим следующий день продажи после покупки
            sell_candidates = [d for d in sell_days if d > buy_day]

            if sell_candidates:
                sell_day = sell_candidates[0]
                buy_price = predictions[buy_day]
                sell_price = predictions[sell_day]

                # Расчет прибыли от одной сделки
                shares = investment / buy_price
                profit = shares * (sell_price - buy_price)
                total_profit += profit

                strategy.append(
                    f"📅 День {buy_day+1}: Купить по ${buy_price:.2f}\n"
                    f"📅 День {sell_day+1}: Продать по ${sell_price:.2f}\n"
                    f"💵 Прибыль от сделки: ${profit:.2f}"
                )

        if not strategy:
            return 0.0, "Нет выгодных точек для покупки и продажи"

        strategy_text = "\n\n".join(strategy)
        return total_profit, strategy_text