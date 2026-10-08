import os
import datetime
import requests
import pandas as pd
import numpy as np
from scipy.special import factorial, gammaln
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')

# 1. Configuración de URL base y credenciales oficiales
API_KEY = os.environ.get("THE_STATS_API_KEY")
BASE_URL = "https://api.thestatsapi.com/api"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

# Configuración de modelos ML
LSTM_LOOKBACK = 5  # Ventana de historia para LSTM
LSTM_UNITS = 32
EPOCHS = 50
BATCH_SIZE = 8

class PoissonBivariadoModel:
    """
    Modelo de Distribución Bivariada de Poisson para predecir correlación de goles.
    Modela la dependencia entre goles del equipo local y visitante.
    """
    
    def __init__(self):
        self.lambda_home = 0
        self.lambda_away = 0
        self.lambda_covariance = 0
    
    def fit(self, goals_home, goals_away):
        """
        Estima los parámetros lambda de la distribución bivariada de Poisson.
        """
        goals_home = np.array(goals_home, dtype=float)
        goals_away = np.array(goals_away, dtype=float)
        
        # Estimación de máxima verosimilitud
        self.lambda_home = np.mean(goals_home)
        self.lambda_away = np.mean(goals_away)
        
        # Covarianza como indicador de dependencia
        self.lambda_covariance = np.cov(goals_home, goals_away)[0, 1]
        self.lambda_covariance = max(0, self.lambda_covariance)
    
    def pmf(self, x, y):
        """
        Calcula la probabilidad conjunta P(X=x, Y=y) usando Poisson Bivariado.
        """
        if self.lambda_home <= 0 or self.lambda_away <= 0:
            return 0
        
        try:
            # Función de masa de probabilidad bivariada de Poisson
            term1 = np.exp(-(self.lambda_home + self.lambda_away + self.lambda_covariance))
            term2 = (self.lambda_home ** x) / factorial(x)
            term3 = (self.lambda_away ** y) / factorial(y)
            
            # Sumatorio de la distribución de Poisson bivariada
            sum_term = 0
            for k in range(min(x, y) + 1):
                coeff = (factorial(x) * factorial(y) * (self.lambda_covariance ** k)) / (
                    factorial(k) * factorial(x - k) * factorial(y - k)
                )
                sum_term += coeff / (self.lambda_home ** k * self.lambda_away ** k)
            
            return term1 * term2 * term3 * sum_term
        except:
            return 0
    
    def predict_over_05(self):
        """
        Calcula la probabilidad de Over 0.5 (al menos 1 gol) usando Poisson Bivariado.
        """
        prob_0_0 = self.pmf(0, 0)
        return 1 - prob_0_0


class LSTMMomentumPredictor:
    """
    Modelo LSTM con análisis de momentum para predicciones de goles.
    Captura tendencias temporales y patrones de rendimiento.
    """
    
    def __init__(self, lookback=LSTM_LOOKBACK):
        self.lookback = lookback
        self.model = None
        self.scaler_mean = 0
        self.scaler_std = 1
    
    def crear_modelo(self):
        """Construye la arquitectura LSTM con momentum."""
        modelo = keras.Sequential([
            layers.LSTM(LSTM_UNITS, activation='relu', input_shape=(self.lookback, 1), return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(LSTM_UNITS // 2, activation='relu'),
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
        """
        Prepara datos históricos para LSTM.
        Normaliza y crea secuencias con ventana móvil.
        """
        goals_array = np.array(goals_history, dtype=float)
        
        # Normalización
        self.scaler_mean = np.mean(goals_array)
        self.scaler_std = np.std(goals_array) + 1e-8
        goals_normalized = (goals_array - self.scaler_mean) / self.scaler_std
        
        # Crear secuencias
        X, y = [], []
        for i in range(len(goals_normalized) - self.lookback):
            X.append(goals_normalized[i:i + self.lookback])
            y.append(goals_normalized[i + self.lookback])
        
        return np.array(X).reshape(-1, self.lookback, 1), np.array(y)
    
    def entrenar(self, goals_history):
        """Entrena el modelo LSTM con historial de goles."""
        if len(goals_history) < self.lookback + 1:
            return False
        
        X, y = self.preparar_datos(goals_history)
        
        if len(X) == 0:
            return False
        
        self.model = self.crear_modelo()
        self.model.fit(
            X, y,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=0
        )
        
        return True
    
    def predecir(self, goals_history):
        """
        Predice próximos goles usando LSTM con momentum.
        Retorna predicción y tendencia (momentum).
        """
        if self.model is None or len(goals_history) < self.lookback:
            return None, None
        
        goals_array = np.array(goals_history[-self.lookback:], dtype=float)
        goals_normalized = (goals_array - self.scaler_mean) / self.scaler_std
        
        X_pred = goals_normalized.reshape(1, self.lookback, 1)
        y_pred_normalized = self.model.predict(X_pred, verbose=0)[0][0]
        y_pred = y_pred_normalized * self.scaler_std + self.scaler_mean
        
        # Calcular momentum (tendencia)
        momentum = np.mean(np.diff(goals_array[-3:]))  # Cambio promedio últimos 3 partidos
        
        return float(y_pred), float(momentum)


def obtener_rango_fechas():
    """Determina las fechas de consulta basándose en los inputs manuales o usa HOY."""
    env_from = os.environ.get("INPUT_DATE_FROM", "").strip()
    env_to = os.environ.get("INPUT_DATE_TO", "").strip()
    
    date_from = env_from if env_from else datetime.date.today().isoformat()
    date_to = env_to if env_to else datetime.date.today().isoformat()
    
    return date_from, date_to

def obtener_partidos_jornada(date_from, date_to):
    """Obtiene la lista de partidos programados usando el paso seguro de parámetros."""
    url = f"{BASE_URL}/football/matches"
    params = {
        "date_from": date_from,
        "date_to": date_to,
        "status": "scheduled",
        "per_page": 100
    }
    
    response = requests.get(url, headers=HEADERS, params=params)
    
    print(f"[DEBUG] URL de Jornada: {response.url}")
    print(f"[DEBUG] Código de Estado: {response.status_code}")
    print(f"[DEBUG] Respuesta (primeros 500 chars): {response.text[:500]}")
    
    response.raise_for_status()
    
    res_json = response.json()
    meta = res_json.get("meta", {})
    print(f"[DIAGNÓSTICO JORNADA] Total partidos en rango: {meta.get('total', 0)} | Total páginas: {meta.get('total_pages', 0)}")
    
    return res_json.get("data", [])

def obtener_historico_equipo(team_id, condicion, limit=20):
    """
    Obtiene historial de partidos para un equipo con rol específico.
    """
    url = f"{BASE_URL}/football/matches"
    params = {
        "team_id": team_id,
        "status": "finished",
        "per_page": limit
    }
    
    response = requests.get(url, headers=HEADERS, params=params)
    
    if response.status_code != 200:
        print(f"[ERROR] Código {response.status_code} para equipo {team_id}.")
        response.raise_for_status()
    
    partidos = response.json().get("data", [])
    partidos.sort(key=lambda x: x.get("utc_date", ""), reverse=True)
    
    partidos_filtrados = []
    for p in partidos:
        es_local = p["home_team"]["id"] == team_id
        if (condicion == "home" and es_local) or (condicion == "away" and not es_local):
            partidos_filtrados.append(p)
    
    return partidos_filtrados[:limit]

def extraer_goles_ht(partido, team_id, rol):
    """Extrae goles de primer tiempo según rol del equipo."""
    try:
        score = partido.get("score", {})
        if rol == "home":
            return int(score.get("half_time_home", 0))
        else:
            return int(score.get("half_time_away", 0))
    except:
        return None

def calcular_metrica_ht_mejorada(team_id, condicion, usar_lstm=True):
    """
    Calcula probabilidad de HT Over 0.5 usando:
    1. Distribución Bivariada de Poisson
    2. LSTM Momentum (opcional)
    """
    
    partidos = obtener_historico_equipo(team_id, condicion, limit=20)
    
    if len(partidos) < 3:
        return 0, 0, "Datos insuficientes"
    
    # Obtener detalles completos
    goles_ht = []
    goles_ft = []
    
    for p in partidos:
        match_id = p["id"]
        detalle_url = f"{BASE_URL}/football/matches/{match_id}"
        detalle_res = requests.get(detalle_url, headers=HEADERS)
        
        if detalle_res.status_code != 200:
            continue
        
        score = detalle_res.json().get("data", {}).get("score", {})
        
        if condicion == "home":
            ht_goles = score.get("half_time_home")
        else:
            ht_goles = score.get("half_time_away")
        
        if ht_goles is not None:
            goles_ht.append(int(ht_goles))
    
    if len(goles_ht) < 3:
        return 0, 0, "Datos insuficientes post-filtrado"
    
    # 1. Modelo Bivariado de Poisson
    modelo_poisson = PoissonBivariadoModel()
    modelo_poisson.fit(goles_ht, goles_ht)
    
    prob_poisson = modelo_poisson.predict_over_05() * 100
    
    # 2. LSTM Momentum
    prob_lstm = 0
    momentum = 0
    
    if usar_lstm and len(goles_ht) >= LSTM_LOOKBACK + 1:
        lstm_predictor = LSTMMomentumPredictor(lookback=LSTM_LOOKBACK)
        if lstm_predictor.entrenar(goles_ht):
            pred, mom = lstm_predictor.predecir(goles_ht)
            if pred is not None:
                # Convertir predicción a probabilidad de Over 0.5
                prob_lstm = (pred > 0.5) * 100 if pred > 0 else 0
                momentum = mom
    
    # Promedio ponderado
    if usar_lstm:
        prob_final = (prob_poisson * 0.6 + prob_lstm * 0.4)
    else:
        prob_final = prob_poisson
    
    estadisticas = f"[μ={np.mean(goles_ht):.2f}|σ={np.std(goles_ht):.2f}|m={momentum:.2f}]"
    
    return prob_final, prob_lstm, estadisticas

def ejecutar_pipeline():
    """Pipeline mejorado con Poisson Bivariado y LSTM Momentum."""
    d_from, d_to = obtener_rango_fechas()
    print(f"Buscando jornadas programadas desde {d_from} hasta {d_to}...")
    
    partidos_jornada = obtener_partidos_jornada(d_from, d_to)
    partidos_filtrados = []
    
    print(f"Iniciando análisis con Poisson Bivariado + LSTM Momentum para {len(partidos_jornada)} partidos...")
    
    for i, partido in enumerate(partidos_jornada):
        print(f"\n[Proceso {i+1}/{len(partidos_jornada)}] Analizando partido...")
        
        match_id = partido["id"]
        home_id = partido["home_team"]["id"]
        away_id = partido["away_team"]["id"]
        home_name = partido["home_team"]["name"]
        away_name = partido["away_team"]["name"]
        
        # Calcular métricas mejoradas
        home_pct, home_lstm, home_stats = calcular_metrica_ht_mejorada(home_id, "home", usar_lstm=True)
        away_pct, away_lstm, away_stats = calcular_metrica_ht_mejorada(away_id, "away", usar_lstm=True)
        
        UMBRAL = 70.0  # Ajustado para mayor precisión con nuevos modelos
        
        if home_pct >= UMBRAL and away_pct >= UMBRAL:
            partidos_filtrados.append({
                "ID Partido": match_id,
                "Fecha UTC": partido["utc_date"],
                "Liga ID": partido.get("competition_id"),
                "Local": home_name,
                "Visitante": away_name,
                "% HT Over 0.5 Local (Poisson)": round(home_pct, 2),
                "% HT Over 0.5 Visitante (Poisson)": round(away_pct, 2),
                "% HT Over 0.5 Local (LSTM)": round(home_lstm, 2),
                "% HT Over 0.5 Visitante (LSTM)": round(away_lstm, 2),
                "Confianza Combinada": round((home_pct + away_pct) / 2, 2),
                "Stats Local": home_stats,
                "Stats Visitante": away_stats
            })
            
            print(f"✓ PARTIDO VÁLIDO: {home_name} vs {away_name}")
            print(f"  Local: Poisson={home_pct:.2f}% | LSTM={home_lstm:.2f}% | {home_stats}")
            print(f"  Visitante: Poisson={away_pct:.2f}% | LSTM={away_lstm:.2f}% | {away_stats}")
    
    if partidos_filtrados:
        df = pd.DataFrame(partidos_filtrados)
        df.to_csv("partidos_del_dia.csv", index=False)
        print(f"\n✓ Proceso finalizado. Se guardaron {len(partidos_filtrados)} partidos óptimos.")
    else:
        columnas = [
            "ID Partido", "Fecha UTC", "Liga ID", "Local", "Visitante",
            "% HT Over 0.5 Local (Poisson)", "% HT Over 0.5 Visitante (Poisson)",
            "% HT Over 0.5 Local (LSTM)", "% HT Over 0.5 Visitante (LSTM)",
            "Confianza Combinada", "Stats Local", "Stats Visitante"
        ]
        pd.DataFrame(columns=columnas).to_csv("partidos_del_dia.csv", index=False)
        print(f"\nFinalizado. Ningún partido cumplió las condiciones para {d_from} / {d_to}.")

if __name__ == "__main__":
    if not API_KEY:
        print("Error crítico: No se detectó la variable de entorno THE_STATS_API_KEY.")
    else:
        ejecutar_pipeline()
