"""
Главный файл для запуска бота
"""

import logging
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ConversationHandler
from bot.handlers import BotHandlers, TICKER, AMOUNT
from utils.logger import setup_logging
from config import config

# Настройка логирования
setup_logging()
logger = logging.getLogger(__name__)


def main():
    """Главная функция запуска бота"""

    logger.info("=" * 60)
    logger.info("Запуск Telegram-бота для прогнозирования акций")
    logger.info("=" * 60)

    # Проверка токена
    if config.BOT_TOKEN == 'YOUR_BOT_TOKEN':
        logger.error("Необходимо установить токен бота!")
        logger.error("Получите токен у @BotFather и установите в config.py")
        return

    # Создание приложения
    application = Application.builder().token(config.BOT_TOKEN).build()

    # Обработчик диалога
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', BotHandlers.start)],
        states={
            TICKER: [
                MessageHandler(
                    filters.TEXT & ~filters.COMMAND,
                    BotHandlers.ticker_received
                )
            ],
            AMOUNT: [
                MessageHandler(
                    filters.TEXT & ~filters.COMMAND,
                    BotHandlers.amount_received
                )
            ],
        },
        fallbacks=[CommandHandler('cancel', BotHandlers.cancel)],
    )

    # Добавление обработчиков
    application.add_handler(conv_handler)
    application.add_handler(CommandHandler('help', BotHandlers.help_command))

    # Запуск бота
    logger.info("Бот успешно запущен и готов к работе!")
    logger.info(f"Прогноз на {config.FORECAST_DAYS} дней")
    logger.info(f"История данных: {config.HISTORY_DAYS} дней")
    logger.info("-" * 60)

    application.run_polling(allowed_updates=None)


if __name__ == '__main__':
    main()
