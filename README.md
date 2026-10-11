# Estrategia Goles HT

**Un predictor de goles en el primer tiempo (HT) basado en Poisson Bivariado y LSTM con momentum.**

Analiza partidos programados utilizando datos históricos y modelos de machine learning para identificar encuentros con alta probabilidad de goles en el primer tiempo. Integración automática con Telegram.

## 🎯 Características

- **Modelo de Poisson Bivariado**: Captura la correlación entre goles del equipo local y visitante
- **LSTM + Momentum**: Red neuronal recurrente con análisis de tendencias para predicciones dinámicas
- **Scoring Combinado**: Fusión de ambos modelos (60% Poisson + 40% LSTM)
- **Notificaciones en Telegram**: Envío automático de resultados a través de bot
- **API Integration**: Conexión con [The Stats API](https://www.thestatsapi.com/) para datos en tiempo real

## 🚀 Quick Start

### Requisitos
- Python 3.8+
- Variables de entorno configuradas

### Instalación

```bash
git clone https://github.com/jossjb865/estrategia-goles-ht.git
cd estrategia-goles-ht
pip install -r requirements.txt
```

### Configuración de Variables de Entorno

Crea un archivo `.env` o configura en tu sistema:

```env
# API de The Stats API
THE_STATS_API_KEY=tu_api_key_aqui

# Telegram (opcional)
TELEGRAM_BOT_TOKEN=tu_bot_token_aqui
TELEGRAM_CHAT_ID=tu_chat_id_aqui

# Rango de fechas (opcional, por defecto hoy)
INPUT_DATE_FROM=2026-10-11
INPUT_DATE_TO=2026-10-11
```

### Ejecución

```bash
python main.py
```

El script genera `partidos_del_dia.csv` con los partidos que cumplen los criterios (confianza >= 70%).

## 📊 Output

Resultado en CSV con columnas:
- **ID Partido**: Identificador único
- **Fecha UTC**: Hora del partido
- **Local / Visitante**: Nombres de los equipos
- **% HT Over 0.5 Local/Visitante**: Probabilidad de >0.5 goles en primer tiempo (Poisson)
- **% HT Over 0.5 Local/Visitante (LSTM)**: Probabilidad según modelo LSTM
- **Confianza Combinada**: Promedio ponderado de ambos modelos
- **Stats Local/Visitante**: Media, desv. estándar y momentum de goles

## 🏗️ Estructura del Proyecto

```
estrategia-goles-ht/
├── main.py                    # Script principal
├── models/
│   ├── __init__.py
│   ├── poisson.py            # Modelo de Poisson Bivariado
│   └── lstm.py               # Predictor LSTM + Momentum
├── api/
│   ├── __init__.py
│   └── stats_api.py          # Integración con The Stats API
├── utils/
│   ├── __init__.py
│   ├── logger.py             # Sistema de logging mejorado
│   └── telegram.py           # Utilidades de Telegram
├── tests/
│   ├── __init__.py
│   ├── test_models.py        # Tests de modelos
│   └── test_api.py           # Tests de API
├── requirements.txt           # Dependencias
├── .env.example              # Template de variables de entorno
├── .gitignore                # Archivos ignorados
└── README.md                 # Este archivo
```

## 🔧 Modelos

### Modelo de Poisson Bivariado

Captura correlación entre goles de ambos equipos:

```python
P(X=x, Y=y) = P(X_home=x) * P(X_away=y) * factor_covarianza
```

Donde:
- `λ_home`: Media de goles del local
- `λ_away`: Media de goles del visitante
- `λ_covariance`: Covarianza para capturar dependencia

### LSTM con Momentum

Red neuronal recurrente que:
1. Normaliza el histórico de goles
2. Entrena en secuencias de 5 partidos (`LSTM_LOOKBACK`)
3. Predice goles futuros
4. Calcula momentum (tendencia) de últimos 3 partidos
5. Pesa la predicción según tendencia

## 📈 Parámetros Configurables

En `main.py`:

```python
LSTM_LOOKBACK = 5        # Ventana de histórico para LSTM
LSTM_UNITS = 32          # Unidades en capas LSTM
EPOCHS = 50              # Épocas de entrenamiento
BATCH_SIZE = 8           # Batch size para entrenamiento
UMBRAL = 70.0            # Umbral de confianza mínima
```

## 🧪 Testing

```bash
# Ejecutar todos los tests
python -m pytest tests/

# Con cobertura
python -m pytest tests/ --cov=.
```

## 🐛 Troubleshooting

### "API Key not found"
```bash
# Verifica la variable de entorno
echo $THE_STATS_API_KEY
```

### "No matches found"
Verifica que:
- Las fechas sean correctas
- Haya partidos programados en ese rango
- La API esté disponible

### "Insufficient data"
Algunos equipos pueden no tener suficiente histórico. El modelo requiere mínimo 3 partidos.

## 📝 Logs

El sistema genera logs detallados en stdout:
- `[DEBUG]`: Información de requests HTTP
- `[DIAGNÓSTICO JORNADA]`: Estadísticas de partidos
- `[ERROR]`: Errores capturados
- `[TELEGRAM]`: Estado de envíos

## 🔗 Recursos

- [The Stats API Docs](https://www.thestatsapi.com/docs)
- [TensorFlow Keras Docs](https://keras.io/)
- [Telegram Bot API](https://core.telegram.org/bots/api)

## 📄 Licencia

Sin licencia especificada (todos los derechos reservados).

## 👤 Autor

[@jossjb865](https://github.com/jossjb865)
