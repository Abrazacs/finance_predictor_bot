"""
Обработчики Telegram бота
"""

import os
import logging
from telegram import Update
from telegram.ext import ContextTypes, ConversationHandler
from services.data_service import DataService
from services.prediction_service import PredictionService
from services.visualization_service import VisualizationService
from utils.trading_signals import TradingSignals
from utils.logger import log_user_request
from config import config

logger = logging.getLogger(__name__)

# Состояния диалога
TICKER, AMOUNT = range(2)


class BotHandlers:
    """Класс с обработчиками бота"""

    @staticmethod
    async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Начало диалога"""
        await update.message.reply_text(
            "👋 <b>Добро пожаловать!</b>\n\n"
            "Я бот для прогнозирования цен акций с использованием "
            "машинного обучения 🤖\n\n"
            "Я проанализирую исторические данные и построю прогноз на "
            f"{config.FORECAST_DAYS} дней с помощью трех разных моделей:\n"
            "• Random Forest 🌳\n"
            "• ARIMA 📊\n"
            "• LSTM (нейросеть) 🧠\n\n"
            "Введите тикер компании (например, AAPL, MSFT, TSLA):",
            parse_mode='HTML'
        )
        return TICKER

    @staticmethod
    async def ticker_received(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Обработка тикера"""
        ticker = update.message.text.strip().upper()

        # Валидация тикера
        if not DataService.validate_ticker(ticker):
            await update.message.reply_text(
                "❌ Некорректный тикер. Пожалуйста, введите валидный тикер "
                "(например, AAPL, MSFT, GOOGL):"
            )
            return TICKER

        context.user_data['ticker'] = ticker

        await update.message.reply_text(
            f"✅ Тикер: <b>{ticker}</b>\n\n"
            "Теперь введите сумму для условной инвестиции в долларах "
            "(например, 10000):",
            parse_mode='HTML'
        )
        return AMOUNT

    @staticmethod
    async def amount_received(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Обработка суммы и запуск анализа"""
        try:
            amount = float(update.message.text.strip().replace(',', ''))

            if amount <= 0:
                await update.message.reply_text(
                    "❌ Сумма должна быть положительной. Попробуйте еще раз:"
                )
                return AMOUNT

            if amount > 1000000000:
                await update.message.reply_text(
                    "❌ Сумма слишком большая. Введите реальную сумму:"
                )
                return AMOUNT

            context.user_data['amount'] = amount
            ticker = context.user_data['ticker']

            # Уведомление о начале анализа
            await update.message.reply_text(
                f"💼 <b>Начинаю анализ акций {ticker}</b>\n\n"
                "⏳ Загружаю данные за последние 2 года...\n"
                "🤖 Обучаю модели машинного обучения...\n"
                "📈 Строю прогноз...\n\n"
                "⏱ Это займет 1-2 минуты, пожалуйста, подождите...",
                parse_mode='HTML'
            )

            # Загрузка данных
            data_service = DataService()
            data = data_service.load_stock_data(ticker)

            if data is None:
                await update.message.reply_text(
                    f"❌ <b>Ошибка загрузки данных</b>\n\n"
                    f"Не удалось загрузить данные для тикера <b>{ticker}</b>.\n"
                    "Возможные причины:\n"
                    "• Неверный тикер\n"
                    "• Проблемы с подключением к Yahoo Finance\n"
                    "• Тикер не торгуется на бирже\n\n"
                    "Используйте /start для новой попытки.",
                    parse_mode='HTML'
                )
                return ConversationHandler.END

            # Обучение моделей
            prediction_service = PredictionService()
            prediction_service.train_all_models(data)

            # Прогнозирование
            predictions = prediction_service.predict(steps=config.FORECAST_DAYS)

            # Определение торговых сигналов
            trading_signals = TradingSignals()
            buy_days, sell_days = trading_signals.find_extrema(predictions)
            profit, strategy = trading_signals.calculate_profit(
                predictions, amount, buy_days, sell_days
            )

            # Создание графика
            viz_service = VisualizationService()
            chart_file = viz_service.plot_prediction(
                ticker, data, predictions, buy_days, sell_days
            )

            # Отправка графика
            with open(chart_file, 'rb') as photo:
                await update.message.reply_photo(photo=photo)

            # Удаление временного файла
            os.remove(chart_file)

            # Формирование отчета
            results = prediction_service.get_results_summary()
            current_price = data['price'].iloc[-1]
            predicted_price = predictions[-1]
            price_change = ((predicted_price - current_price) / current_price) * 100

            # Эмодзи для изменения цены
            trend_emoji = "📈" if price_change > 0 else "📉"
            trend_text = "вырастет" if price_change > 0 else "упадет"

            report = (
                f"📊 <b>ОТЧЕТ ПО АКЦИЯМ {ticker}</b>\n"
                f"{'='*40}\n\n"
                f"🤖 <b>Модели машинного обучения:</b>\n"
            )

            # Добавляем результаты всех моделей
            for model_name, rmse in results['all_results'].items():
                if rmse == float('inf'):
                    report += f"   • {model_name}: ❌ Ошибка обучения\n"
                else:
                    best_mark = " ⭐" if model_name == results['best_model'] else ""
                    report += f"   • {model_name}: RMSE = {rmse:.2f}{best_mark}\n"

            report += (
                f"\n🏆 <b>Лучшая модель:</b> {results['best_model']}\n"
                f"📏 <b>Точность (RMSE):</b> {results['best_rmse']:.2f}\n\n"
                f"{'='*40}\n"
                f"💵 <b>АНАЛИЗ ЦЕН:</b>\n"
                f"   • Текущая цена: <b>${current_price:.2f}</b>\n"
                f"   • Прогноз через {config.FORECAST_DAYS} дней: <b>${predicted_price:.2f}</b>\n"
                f"   • Изменение: {trend_emoji} <b>{abs(price_change):.2f}%</b> ({trend_text})\n\n"
                f"{'='*40}\n"
                f"💰 <b>ИНВЕСТИЦИОННАЯ СТРАТЕГИЯ:</b>\n"
                f"   • Сумма инвестиции: <b>${amount:,.2f}</b>\n"
                f"   • Потенциальная прибыль: <b>${profit:,.2f}</b>\n"
            )

            if profit > 0:
                roi = (profit / amount) * 100
                report += f"   • ROI: <b>{roi:.2f}%</b>\n"

            report += f"\n{'='*40}\n📍 <b>ТОРГОВЫЕ РЕКОМЕНДАЦИИ:</b>\n\n"

            if strategy:
                report += strategy
            else:
                report += "⚠️ Недостаточно четких сигналов для торговли"

            report += (
                f"\n\n{'='*40}\n"
                "⚠️ <b>Важное предупреждение:</b>\n"
                "Этот прогноз создан для образовательных целей. "
                "Не используйте его как единственную основу для "
                "инвестиционных решений.\n\n"
                "Используйте /start для нового анализа."
            )

            await update.message.reply_text(report, parse_mode='HTML')

            # Логирование запроса
            log_user_request(
                user_id=update.effective_user.id,
                ticker=ticker,
                amount=amount,
                model=results['best_model'],
                metric=results['best_rmse'],
                profit=profit
            )

            logger.info(
                f"Успешный анализ для пользователя {update.effective_user.id}: "
                f"{ticker}, ${amount:.2f}, прибыль ${profit:.2f}"
            )

        except ValueError:
            await update.message.reply_text(
                "❌ Некорректное число. Пожалуйста, введите сумму в долларах "
                "(например, 10000):"
            )
            return AMOUNT

        except Exception as e:
            logger.error(f"Ошибка при обработке запроса: {e}", exc_info=True)
            await update.message.reply_text(
                f"❌ <b>Произошла ошибка</b>\n\n"
                f"Детали: {str(e)}\n\n"
                "Попробуйте:\n"
                "• Проверить правильность тикера\n"
                "• Попробовать другую компанию\n"
                "• Повторить попытку через минуту\n\n"
                "Используйте /start для новой попытки.",
                parse_mode='HTML'
            )

        return ConversationHandler.END

    @staticmethod
    async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Отмена диалога"""
        await update.message.reply_text(
            "❌ Операция отменена.\n\n"
            "Используйте /start для начала нового анализа."
        )
        return ConversationHandler.END

    @staticmethod
    async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Команда помощи"""
        help_text = (
            "📖 <b>СПРАВКА</b>\n\n"
            "<b>Доступные команды:</b>\n"
            "/start - Начать анализ акций\n"
            "/help - Показать эту справку\n"
            "/cancel - Отменить текущую операцию\n\n"
            "<b>Как использовать бота:</b>\n"
            "1️⃣ Отправьте /start\n"
            "2️⃣ Введите тикер компании (например, AAPL)\n"
            "3️⃣ Введите сумму для инвестиции (например, 10000)\n"
            "4️⃣ Получите прогноз и рекомендации!\n\n"
            "<b>Популярные тикеры:</b>\n"
            "• AAPL - Apple\n"
            "• MSFT - Microsoft\n"
            "• GOOGL - Google\n"
            "• TSLA - Tesla\n"
            "• AMZN - Amazon\n"
            "• NVDA - NVIDIA\n"
            "• META - Meta (Facebook)\n\n"
            "<b>О моделях:</b>\n"
            "Бот использует три модели машинного обучения:\n"
            "🌳 Random Forest - ансамбль деревьев решений\n"
            "📊 ARIMA - статистическая модель временных рядов\n"
            "🧠 LSTM - рекуррентная нейронная сеть\n\n"
            "Автоматически выбирается лучшая модель по метрике RMSE."
        )
        await update.message.reply_text(help_text, parse_mode='HTML')
