"""Tests para modelos de predicción."""

import numpy as np
import pytest
from models.poisson import PoissonBivariadoModel
from models.lstm import LSTMMomentumPredictor


class TestPoissonBivariadoModel:
    """Tests para modelo Poisson Bivariado."""

    def test_fit_basic(self):
        """Verifica que fit actualice parámetros."""
        model = PoissonBivariadoModel()
        goals_home = [1, 2, 1, 3]
        goals_away = [0, 1, 2, 1]
        
        model.fit(goals_home, goals_away)
        
        assert model.lambda_home == np.mean(goals_home)
        assert model.lambda_away == np.mean(goals_away)
        assert model.lambda_covariance >= 0

    def test_pmf_basic(self):
        """Verifica que PMF retorne valores válidos."""
        model = PoissonBivariadoModel()
        model.lambda_home = 1.5
        model.lambda_away = 1.0
        model.lambda_covariance = 0.1
        
        pmf_0_0 = model.pmf(0, 0)
        assert 0 <= pmf_0_0 <= 1

    def test_predict_over_05(self):
        """Verifica probabilidad de Over 0.5."""
        model = PoissonBivariadoModel()
        model.lambda_home = 1.5
        model.lambda_away = 1.0
        model.lambda_covariance = 0.1
        
        prob = model.predict_over_05()
        assert 0 <= prob <= 1
        # Si hay lambda > 0, prob debe ser > 0
        assert prob > 0

    def test_zero_lambda(self):
        """Verifica comportamiento con lambda cero."""
        model = PoissonBivariadoModel()
        model.lambda_home = 0
        model.lambda_away = 1.0
        
        pmf = model.pmf(1, 1)
        assert pmf == 0


class TestLSTMMomentumPredictor:
    """Tests para predictor LSTM."""

    def test_init(self):
        """Verifica inicialización."""
        predictor = LSTMMomentumPredictor(lookback=5, lstm_units=32)
        assert predictor.lookback == 5
        assert predictor.lstm_units == 32
        assert predictor.model is None

    def test_preparar_datos(self):
        """Verifica preparación de datos."""
        predictor = LSTMMomentumPredictor(lookback=3)
        goals = [1, 2, 1, 2, 3]
        
        X, y = predictor.preparar_datos(goals)
        
        # Con lookback=3 y 5 datos, generamos 2 muestras (5-3=2)
        assert len(X) == 2
        assert X.shape == (2, 3, 1)
        assert len(y) == 2

    def test_entrenar_insufficient_data(self):
        """Verifica que entrenamiento falla con pocos datos."""
        predictor = LSTMMomentumPredictor(lookback=5)
        goals = [1, 2, 1]  # Muy pocos datos
        
        result = predictor.entrenar(goals)
        assert result is False

    def test_entrenar_success(self):
        """Verifica entrenamiento exitoso con suficientes datos."""
        predictor = LSTMMomentumPredictor(lookback=3, epochs=1, batch_size=2)
        goals = [1, 2, 1, 2, 1, 2, 1]  # 7 datos, mínimo es 3+1=4
        
        result = predictor.entrenar(goals)
        assert result is True
        assert predictor.model is not None

    def test_predecir_without_model(self):
        """Verifica que predicción retorna None sin modelo."""
        predictor = LSTMMomentumPredictor()
        goals = [1, 2, 1]
        
        pred, mom = predictor.predecir(goals)
        assert pred is None
        assert mom is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
