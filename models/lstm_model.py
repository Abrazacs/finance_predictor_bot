"""
LSTM модель для прогнозирования
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error
from models.base_model import BaseModel
from config import config


class LSTMNetwork(nn.Module):
    """Архитектура LSTM сети"""

    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1):
        super(LSTMNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        self.fc1 = nn.Linear(hidden_size, 25)
        self.fc2 = nn.Linear(25, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out


class LSTMModel(BaseModel):
    """LSTM модель для временных рядов"""

    def __init__(self):
        super().__init__("LSTM")
        self.look_back = config.LSTM_LOOK_BACK
        self.scaler = StandardScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data = None

    def prepare_sequences(self, data: np.ndarray) -> tuple:
        """Подготовка последовательностей для LSTM"""
        X, y = [], []
        for i in range(self.look_back, len(data)):
            X.append(data[i-self.look_back:i])
            y.append(data[i])
        return np.array(X), np.array(y)

    def train(self, data: pd.DataFrame, train_size: float = 0.8) -> float:
        """Обучение LSTM"""
        self.data = data
        prices = data['price'].values.reshape(-1, 1)
        scaled_prices = self.scaler.fit_transform(prices)

        split_idx = int(len(scaled_prices) * train_size)
        train = scaled_prices[:split_idx]
        test = scaled_prices[split_idx:]

        X_train, y_train = self.prepare_sequences(train)
        X_test, y_test = self.prepare_sequences(test)

        if len(X_train) == 0 or len(X_test) == 0:
            return float('inf')

        # Конвертация в PyTorch tensors
        # X уже имеет форму (samples, seq_len), добавляем размерность фичей
        X_train = torch.FloatTensor(X_train).to(self.device)
        if X_train.dim() == 2:
            X_train = X_train.unsqueeze(-1)  # (samples, seq_len, 1)
        y_train = torch.FloatTensor(y_train).to(self.device)

        X_test = torch.FloatTensor(X_test).to(self.device)
        if X_test.dim() == 2:
            X_test = X_test.unsqueeze(-1)  # (samples, seq_len, 1)
        y_test = torch.FloatTensor(y_test).to(self.device)

        # DataLoader (shuffle=False для временных рядов!)
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.LSTM_BATCH_SIZE,
            shuffle=False
        )

        # Создание модели
        self.model = LSTMNetwork(
            input_size=1,
            hidden_size=config.LSTM_HIDDEN_SIZE,
            num_layers=config.LSTM_NUM_LAYERS,
            output_size=1
        ).to(self.device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        # Обучение
        self.model.train()
        for epoch in range(config.LSTM_EPOCHS):
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

        # Оценка
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_test).cpu().numpy().reshape(-1, 1)
            predictions = self.scaler.inverse_transform(predictions)
            y_test_inv = self.scaler.inverse_transform(y_test.cpu().numpy().reshape(-1, 1))

        rmse = root_mean_squared_error(y_test_inv, predictions)
        self.trained = True

        return rmse

    def predict(self, steps: int) -> np.ndarray:
        """Прогнозирование на будущее"""
        if not self.trained:
            raise ValueError("Модель не обучена")

        self.model.eval()
        predictions = []

        prices = self.data['price'].values.reshape(-1, 1)
        scaled_prices = self.scaler.transform(prices)
        last_sequence = scaled_prices[-self.look_back:]

        with torch.no_grad():
            for _ in range(steps):
                # Подготовка входа: (1, seq_len, 1)
                input_seq = torch.FloatTensor(last_sequence).unsqueeze(0).to(self.device)
                if input_seq.dim() == 2:
                    input_seq = input_seq.unsqueeze(-1)

                # Прогноз
                pred_scaled = self.model(input_seq).cpu().numpy().reshape(-1, 1)
                pred = self.scaler.inverse_transform(pred_scaled)[0, 0]
                predictions.append(pred)

                # Обновление последовательности
                last_sequence = np.append(last_sequence[1:], pred_scaled.reshape(-1, 1), axis=0)

        return np.array(predictions)
