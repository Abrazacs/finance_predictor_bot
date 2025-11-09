## «Телеграм-бот для анализа и прогнозирования акций на основе временных рядов»

### Структура проекта:

stock_prediction_bot\
├── main.py                      
├── config.py                
├── requirements.txt             
├── README.md                   
├── models\
│   ├── base_model.py           
│   ├── random_forest.py       
│   ├── arima_model.py          
│   └── lstm_model.py           
├── services\
│   ├── data_service.py         
│   ├── prediction_service.py   
│   └── visualization_service.py 
├── utils\
│   ├── logger.py               
│   └── trading_signals.py     
└── bot\
└──  └── handlers.py            

Логи:\
├── logs.txt                     
└── bot.log   

### Конфигурирование

Конфигурирование происходит в файле `config.py` в корне проекта.


### Запуск бота

1. Клонируем репозиторий

```bash
git clone https://github.com/Abrazacs/finance_predictor_bot.git
cd finance_predictor_bot
```

2. Устанавливаем зависимости

```bash
pip install -r requirements.txt
```

3. Заменяем токен бота в файле `config.py`

```python
BOT_TOKEN: str = 'YOUR_BOT_TOKEN'  # Замените YOUR_BOT_TOKEN на ваш токен бота
```

4. Переходим в корень проекта и запускаем бота

```bash
cd finance_predictor_bot
python main.py
```
