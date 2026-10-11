"""Predictor LSTM con análisis de momentum."""

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


class LSTMMomentumPredictor:
    """Modelo LSTM con análisis de momentum para predicciones de goles.
    
    Utiliza redes recurrentes para capturar patrones temporales
    en el histórico de goles y complementa con análisis de momentum.
    """

    def __init__(self, lookback=5, lstm_units=32, epochs=50, batch_size=8):
        """Inicializa el predictor LSTM.
        
        Args:
            lookback: Número de pasos hacia atrás para secuencias
            lstm_units: Unidades en capas LSTM
            epochs: Épocas de entrenamiento
            batch_size: Tamaño de batch
        """
        self.lookback = lookback
        self.lstm_units = lstm_units
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.scaler_mean = 0
        self.scaler_std = 1

    def crear_modelo(self):
        """Crea arquitectura del modelo LSTM.
        
        Returns:
            keras.Model: Modelo compilado
        """
        modelo = keras.Sequential([
            layers.LSTM(self.lstm_units, activation='relu', 
                       input_shape=(self.lookback, 1), return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(self.lstm_units // 2, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(1, activation='linear')
        ])
        
        modelo.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        return modelo

    def preparar_datos(self, goals_history):
        """Prepara datos para entrenamiento con normalización.
        
        Args:
            goals_history: Histórico de goles
            
        Returns:
            tuple: (X, y) secuencias y targets normalizados
        """
        goals_array = np.array(goals_history, dtype=float)
        self.scaler_mean = np.mean(goals_array)
        self.scaler_std = np.std(goals_array) + 1e-8
        goals_normalized = (goals_array - self.scaler_mean) / self.scaler_std

        X, y = [], []
        for i in range(len(goals_normalized) - self.lookback):
            X.append(goals_normalized[i:i + self.lookback])
            y.append(goals_normalized[i + self.lookback])

        return np.array(X).reshape(-1, self.lookback, 1), np.array(y)

    def entrenar(self, goals_history):
        """Entrena el modelo LSTM con histórico de goles.
        
        Args:
            goals_history: Histórico de goles
            
        Returns:
            bool: True si entrenamiento exitoso, False en caso contrario
        """
        if len(goals_history) < self.lookback + 1:
            return False
        
        X, y = self.preparar_datos(goals_history)
        if len(X) == 0:
            return False
        
        self.model = self.crear_modelo()
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
        return True

    def predecir(self, goals_history):
        """Realiza predicción y calcula momentum.
        
        Args:
            goals_history: Histórico de goles
            
        Returns:
            tuple: (predicción_goles, momentum) o (None, None) si no está listo
        """
        if self.model is None or len(goals_history) < self.lookback:
            return None, None
        
        goals_array = np.array(goals_history[-self.lookback:], dtype=float)
        goals_normalized = (goals_array - self.scaler_mean) / self.scaler_std
        X_pred = goals_normalized.reshape(1, self.lookback, 1)
        
        y_pred_normalized = self.model.predict(X_pred, verbose=0)[0][0]
        y_pred = y_pred_normalized * self.scaler_std + self.scaler_mean
        
        # Calcular momentum de últimos 3 partidos
        momentum = np.mean(np.diff(goals_array[-3:]))
        
        return float(y_pred), float(momentum)
