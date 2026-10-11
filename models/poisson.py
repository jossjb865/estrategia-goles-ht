"""Modelo de Distribución Bivariada de Poisson."""

import numpy as np
from scipy.special import factorial


class PoissonBivariadoModel:
    """Modelo de Poisson Bivariado para correlación de goles.
    
    Captura la dependencia entre goles del equipo local y visitante
    mediante parámetros de covarianza.
    """

    def __init__(self):
        """Inicializa parámetros del modelo."""
        self.lambda_home = 0
        self.lambda_away = 0
        self.lambda_covariance = 0

    def fit(self, goals_home, goals_away):
        """Ajusta el modelo con datos históricos.
        
        Args:
            goals_home: Array de goles del local
            goals_away: Array de goles del visitante
        """
        goals_home = np.array(goals_home, dtype=float)
        goals_away = np.array(goals_away, dtype=float)
        
        self.lambda_home = np.mean(goals_home)
        self.lambda_away = np.mean(goals_away)
        self.lambda_covariance = max(0, np.cov(goals_home, goals_away)[0, 1])

    def pmf(self, x, y):
        """Calcula la función de masa de probabilidad P(X=x, Y=y).
        
        Args:
            x: Goles del local
            y: Goles del visitante
            
        Returns:
            float: Probabilidad P(X=x, Y=y)
        """
        if self.lambda_home <= 0 or self.lambda_away <= 0:
            return 0
        
        try:
            term1 = np.exp(-(self.lambda_home + self.lambda_away + self.lambda_covariance))
            term2 = (self.lambda_home ** x) / factorial(x)
            term3 = (self.lambda_away ** y) / factorial(y)
            
            sum_term = 0
            for k in range(min(x, y) + 1):
                coeff = (factorial(x) * factorial(y) * (self.lambda_covariance ** k)) / (
                    factorial(k) * factorial(x - k) * factorial(y - k)
                )
                sum_term += coeff / (self.lambda_home ** k * self.lambda_away ** k)
            
            return term1 * term2 * term3 * sum_term
        except (OverflowError, ZeroDivisionError):
            return 0

    def predict_over_05(self):
        """Predice probabilidad de Over 0.5 goles (X > 0).
        
        Returns:
            float: P(X > 0.5) = 1 - P(X=0)
        """
        prob_0_0 = self.pmf(0, 0)
        return 1 - prob_0_0
