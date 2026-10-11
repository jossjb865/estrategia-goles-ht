"""Pipeline principal para análisis de goles en primer tiempo (HT).

Combina modelos de Poisson Bivariado y LSTM para predecir probabilidades
de goles en el primer tiempo de partidos programados.
"""

import os
import datetime
import sys

from models.poisson import PoissonBivariadoModel
from models.lstm import LSTMMomentumPredictor
from api.stats_api import StatsAPIClient
from utils.logger import setup_logger
from utils.telegram import enviar_telegram_mensaje, formatear_mensaje_resultado


# Logger
logger = setup_logger(__name__)

# Configuración de hiperparámetros
LSTM_LOOKBACK = 5
LSTM_UNITS = 32
EPOCHS = 50
BATCH_SIZE = 8
UMBRAL_CONFIANZA = 70.0
LIMITE_HISTORICO = 20


def obtener_rango_fechas():
    """Extrae rango de fechas de variables de entorno.
    
    Returns:
        tuple: (date_from, date_to) en formato ISO
    """
    env_from = os.environ.get("INPUT_DATE_FROM", "").strip()
    env_to = os.environ.get("INPUT_DATE_TO", "").strip()
    hoy = datetime.date.today().isoformat()
    return env_from or hoy, env_to or hoy


def calcular_metrica_ht_mejorada(api_client, team_id, condicion, usar_lstm=True):
    """Calcula probabilidad de Over 0.5 goles en primer tiempo.
    
    Args:
        api_client: Instancia de StatsAPIClient
        team_id: ID del equipo
        condicion: 'home' o 'away'
        usar_lstm: Si incluir modelo LSTM
        
    Returns:
        tuple: (prob_final, prob_lstm, estadisticas_str)
    """
    try:
        partidos = api_client.obtener_historico_equipo(team_id, condicion, limit=LIMITE_HISTORICO)
        if len(partidos) < 3:
            return 0, 0, "Datos insuficientes"

        goles_ht = []
        for p in partidos:
            try:
                score = api_client.obtener_detalle_partido(p["id"])
                ht_goles = score.get("half_time_home") if condicion == "home" else score.get("half_time_away")
                if ht_goles is not None:
                    goles_ht.append(int(ht_goles))
            except Exception as e:
                logger.debug(f"Error obteniendo detalle para partido {p['id']}: {e}")
                continue

        if len(goles_ht) < 3:
            return 0, 0, "Datos insuficientes post-filtrado"

        # Modelo Poisson Bivariado
        import numpy as np
        modelo_poisson = PoissonBivariadoModel()
        modelo_poisson.fit(goles_ht, goles_ht)
        prob_poisson = modelo_poisson.predict_over_05() * 100

        # Modelo LSTM (opcional)
        prob_lstm = 0
        momentum = 0
        if usar_lstm and len(goles_ht) >= LSTM_LOOKBACK + 1:
            lstm_predictor = LSTMMomentumPredictor(lookback=LSTM_LOOKBACK)
            if lstm_predictor.entrenar(goles_ht):
                pred, mom = lstm_predictor.predecir(goles_ht)
                if pred is not None:
                    prob_lstm = (pred > 0.5) * 100 if pred > 0 else 0
                    momentum = mom

        prob_final = (prob_poisson * 0.6 + prob_lstm * 0.4) if usar_lstm else prob_poisson
        estadisticas = f"[μ={np.mean(goles_ht):.2f}|σ={np.std(goles_ht):.2f}|m={momentum:.2f}]"
        return prob_final, prob_lstm, estadisticas

    except Exception as e:
        logger.error(f"Error calculando métrica para equipo {team_id}: {e}")
        return 0, 0, f"Error: {str(e)}"


def ejecutar_pipeline():
    """Pipeline principal de análisis."""
    logger.info("=" * 70)
    logger.info("INICIANDO PIPELINE: Estrategia Goles HT")
    logger.info("=" * 70)

    # Validar API Key
    api_key = os.environ.get("THE_STATS_API_KEY", "").strip()
    if not api_key:
        logger.error("Error crítico: THE_STATS_API_KEY no configurada")
        return

    api_client = StatsAPIClient(api_key)
    d_from, d_to = obtener_rango_fechas()
    logger.info(f"Buscando partidos desde {d_from} hasta {d_to}...")

    try:
        partidos_jornada = api_client.obtener_partidos_jornada(d_from, d_to)
        logger.info(f"Se encontraron {len(partidos_jornada)} partidos en el rango")
    except Exception as e:
        logger.error(f"Error obteniendo jornada: {e}")
        return

    partidos_filtrados = []
    logger.info(f"Iniciando análisis con {len(partidos_jornada)} partidos...\n")

    for i, partido in enumerate(partidos_jornada, 1):
        try:
            match_id = partido["id"]
            home_id = partido["home_team"]["id"]
            away_id = partido["away_team"]["id"]
            home_name = partido["home_team"]["name"]
            away_name = partido["away_team"]["name"]

            logger.info(f"[{i}/{len(partidos_jornada)}] Analizando: {home_name} vs {away_name}")

            home_pct, home_lstm, home_stats = calcular_metrica_ht_mejorada(api_client, home_id, "home", usar_lstm=True)
            away_pct, away_lstm, away_stats = calcular_metrica_ht_mejorada(api_client, away_id, "away", usar_lstm=True)

            if home_pct >= UMBRAL_CONFIANZA and away_pct >= UMBRAL_CONFIANZA:
                confianza_combinada = (home_pct + away_pct) / 2
                partidos_filtrados.append({
                    "ID Partido": match_id,
                    "Fecha UTC": partido["utc_date"],
                    "Liga ID": partido.get("competition_id"),
                    "Local": home_name,
                    "Visitante": away_name,
                    "% HT Over 0.5 Local": round(home_pct, 2),
                    "% HT Over 0.5 Visitante": round(away_pct, 2),
                    "% HT Over 0.5 Local (LSTM)": round(home_lstm, 2),
                    "% HT Over 0.5 Visitante (LSTM)": round(away_lstm, 2),
                    "Confianza Combinada": round(confianza_combinada, 2),
                    "Stats Local": home_stats,
                    "Stats Visitante": away_stats
                })
                logger.info(f"  ✓ VÁLIDO: Local {home_pct:.2f}% | Visitante {away_pct:.2f}% | Confianza: {confianza_combinada:.2f}%\n")
            else:
                logger.debug(f"  ✗ No cumple umbral: Local {home_pct:.2f}% | Visitante {away_pct:.2f}%\n")

        except Exception as e:
            logger.error(f"Error procesando partido {partido.get('id', 'UNKNOWN')}: {e}")
            continue

    # Guardar resultados
    import pandas as pd
    if partidos_filtrados:
        df = pd.DataFrame(partidos_filtrados)
        df.to_csv("partidos_del_dia.csv", index=False)
        logger.info(f"\n✓ Proceso finalizado. {len(partidos_filtrados)} partidos óptimos guardados en 'partidos_del_dia.csv'")
    else:
        columnas = [
            "ID Partido", "Fecha UTC", "Liga ID", "Local", "Visitante",
            "% HT Over 0.5 Local", "% HT Over 0.5 Visitante",
            "% HT Over 0.5 Local (LSTM)", "% HT Over 0.5 Visitante (LSTM)",
            "Confianza Combinada", "Stats Local", "Stats Visitante"
        ]
        pd.DataFrame(columns=columnas).to_csv("partidos_del_dia.csv", index=False)
        logger.info(f"\nNingún partido cumplió criterios para {d_from} / {d_to}")

    # Enviar notificación Telegram
    mensaje = formatear_mensaje_resultado(partidos_filtrados, d_from, d_to)
    enviar_telegram_mensaje(mensaje)

    logger.info("=" * 70)
    logger.info("PIPELINE FINALIZADO")
    logger.info("=" * 70)


if __name__ == "__main__":
    try:
        ejecutar_pipeline()
    except KeyboardInterrupt:
        logger.info("\nPipeline interrumpido por usuario")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error no controlado en pipeline: {e}", exc_info=True)
        sys.exit(1)
