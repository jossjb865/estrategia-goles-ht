import os
import datetime
import requests
import pandas as pd

# 1. Configuración de URL base y credenciales oficiales
API_KEY = os.environ.get("THE_STATS_API_KEY")
BASE_URL = "https://www.thestatsapi.com"  # URL corregida de forma estricta
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

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
    
    # Si hay un error de rate limit (429) o credenciales (401), se interrumpe aquí de forma clara
    response.raise_for_status()
    
    res_json = response.json()
    meta = res_json.get("meta", {})
    print(f"[DIAGNÓSTICO JORNADA] Total partidos en rango: {meta.get('total', 0)} | Total páginas: {meta.get('total_pages', 0)}")
    
    return res_json.get("data", [])

def calcular_porcentaje_ht_over05(team_id, condicion):
    """
    Consulta una muestra controlada de 10 partidos del equipo.
    Cualquier respuesta HTTP fallida detendrá el script para auditoría.
    """
    url = f"{BASE_URL}/football/matches"
    params = {
        "team_id": team_id,
        "status": "finished",
        "per_page": 10  # Reducido estrictamente a los últimos 10 partidos globales
    }
    
    response = requests.get(url, headers=HEADERS, params=params)
    
    if response.status_code != 200:
        print(f"[ERROR CRÍTICO HISTORIAL] Código {response.status_code} para el equipo {team_id}.")
        response.raise_for_status()
        
    partidos_historicos = response.json().get("data", [])
    if not partidos_historicos:
        return 0

    # Orden cronológico manual (más recientes primero)
    partidos_historicos.sort(key=lambda x: x.get("utc_date", ""), reverse=True)
        
    partidos_validos_procesados = 0
    partidos_con_goles_ht = 0
    MUESTRA_OBJETIVO = 3  # Evaluamos una muestra pequeña de 3 partidos válidos en rol para mitigar ráfagas
    
    for p_resumido in partidos_historicos:
        if partidos_validos_procesados >= MUESTRA_OBJETIVO:
            break
            
        es_local = p_resumido["home_team"]["id"] == team_id
        if condicion == "home" and not es_local:
            continue
        if condicion == "away" and es_local:
            continue
            
        match_id = p_resumido["id"]
        detalle_url = f"{BASE_URL}/football/matches/{match_id}"
        detalle_res = requests.get(detalle_url, headers=HEADERS)
        
        # Validación de integridad de la cuota en llamadas secundarias individuales
        if detalle_res.status_code != 200:
            print(f"[ERROR CRÍTICO DETALLE] Falló el partido {match_id}. Status: {detalle_res.status_code}")
            detalle_res.raise_for_status()
            
        score = detalle_res.json().get("data", {}).get("score", {})
        ht_home = score.get("half_time_home")
        ht_away = score.get("half_time_away")
        
        if ht_home is not None and ht_away is not None:
            partidos_validos_procesados += 1
            goles_primer_tiempo = int(ht_home) + int(ht_away)
            if goles_primer_tiempo > 0:
                partidos_con_goles_ht += 1
                    
    if partidos_validos_procesados == 0:
        return 0
        
    return (partidos_con_goles_ht / partidos_validos_procesados) * 100

def ejecutar_pipeline():
    d_from, d_to = obtener_rango_fechas()
    print(f"Buscando jornadas programadas desde {d_from} hasta {d_to}...")
    
    partidos_jornada = obtener_partidos_jornada(d_from, d_to)
    partidos_filtrados = []
    
    print(f"Iniciando análisis profundo para {len(partidos_jornada)} partidos encontrados...")
    
    for partido in partidos_jornada:
        match_id = partido["id"]
        home_id = partido["home_team"]["id"]
        away_id = partido["away_team"]["id"]
        home_name = partido["home_team"]["name"]
        away_name = partido["away_team"]["name"]
        
        # Pipeline secuencial seguro controlado por raise_for_status()
        home_pct = calcular_porcentaje_ht_over05(home_id, "home")
        away_pct = calcular_porcentaje_ht_over05(away_id, "away")
        
        UMBRAL = 66.0  # Coherente con al menos 2 partidos de 3 válidos con gol al descanso
        if home_pct >= UMBRAL and away_pct >= UMBRAL:
            partidos_filtrados.append({
                "ID Partido": match_id,
                "Fecha UTC": partido["utc_date"],
                "Liga ID": partido.get("competition_id"),
                "Local": home_name,
                "Visitante": away_name,
                "% HT Over 0.5 Local": round(home_pct, 2),
                "% HT Over 0.5 Visitante": round(away_pct, 2)
            })
            
    if partidos_filtrados:
        df = pd.DataFrame(partidos_filtrados)
        df.to_csv("partidos_del_dia.csv", index=False)
        print(f"Proceso finalizado con éxito. Se guardaron {len(partidos_filtrados)} partidos óptimos.")
    else:
        columnas = ["ID Partido", "Fecha UTC", "Liga ID", "Local", "Visitante", "% HT Over 0.5 Local", "% HT Over 0.5 Visitante"]
        pd.DataFrame(columns=columnas).to_csv("partidos_del_dia.csv", index=False)
        print(f"Finalizado. Ningún partido cumplió las condiciones para el rango {d_from} / {d_to}.")

if __name__ == "__main__":
    if not API_KEY:
        print("Error crítico: No se detectó la variable de entorno THE_STATS_API_KEY.")
    else:
        ejecutar_pipeline()
