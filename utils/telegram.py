"""Utilidades para integración con Telegram."""

import os
import requests
from utils.logger import setup_logger


logger = setup_logger(__name__)


def enviar_telegram_mensaje(mensaje):
    """Envía mensaje a Telegram si está configurado.
    
    Args:
        mensaje: Texto del mensaje (acepta HTML)
        
    Returns:
        bool: True si se envió exitosamente, False en caso contrario
    """
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    
    if not bot_token or not chat_id:
        logger.info("[TELEGRAM] Bot Token o Chat ID no configurados. Se omite envío.")
        return False

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": mensaje,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }

    try:
        response = requests.post(url, data=payload, timeout=20)
        response.raise_for_status()
        result = response.json()
        
        if result.get("ok"):
            logger.info("[TELEGRAM] Mensaje enviado correctamente.")
            return True
        else:
            logger.warning(f"[TELEGRAM] Error en respuesta: {result}")
            return False
            
    except requests.exceptions.Timeout:
        logger.error("[TELEGRAM] Timeout al enviar mensaje")
        return False
    except Exception as e:
        logger.error(f"[TELEGRAM] Excepción: {e}")
        return False


def formatear_mensaje_resultado(partidos_filtrados, date_from, date_to):
    """Formatea resultado para Telegram.
    
    Args:
        partidos_filtrados: Lista de partidos que cumplen criterios
        date_from: Fecha inicio
        date_to: Fecha fin
        
    Returns:
        str: Mensaje formateado en HTML
    """
    if not partidos_filtrados:
        return (
            "<b>Estrategia Goles HT</b>\n"
            f"<b>Rango:</b> {date_from} - {date_to}\n"
            "<b>Resultado:</b> No se encontraron partidos que cumplan los criterios."
        )

    lineas = [
        "<b>Estrategia Goles HT</b>",
        f"<b>Rango:</b> {date_from} - {date_to}",
        f"<b>Partidos encontrados:</b> {len(partidos_filtrados)}",
        "",
    ]

    # Mostrar solo los primeros 5
    for i, p in enumerate(partidos_filtrados[:5], start=1):
        local = p["Local"]
        visitante = p["Visitante"]
        confianza = p["Confianza Combinada"]
        lineas.append(
            f"{i}. <b>{local}</b> vs <b>{visitante}</b> - "
            f"Confianza: <b>{confianza}%</b>"
        )

    if len(partidos_filtrados) > 5:
        lineas.append(f"... y {len(partidos_filtrados) - 5} más.")

    return "\n".join(lineas)
