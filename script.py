import os
import datetime
import requests
import pandas as pd

# Configuración de URL base y credenciales oficiales
API_KEY = os.environ.get("THE_STATS_API_KEY")
BASE_URL = "https://thestatsapi.com"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

def obtener_rango_fechas():
    """
    Determina las fechas de consulta basándose en los inputs manuales de GitHub 
    o por defecto calcula el día de hoy para la automatización diaria.
    """
    env_from = os.environ.get("INPUT_DATE_FROM", "").strip()
    env_to = os.environ.get("INPUT_DATE_TO", "").strip()
    
    # Si hay inputs manuales, los prioriza. Si no, usa la fecha de hoy.
    date_from = env_from if env_from else datetime.date.today().isoformat()
    date_to = env_to if env_to else datetime.date.today().isoformat()
    
    return date_from, date_to

def obtener_partidos_jornada(date_from, date_to):
    """Obtiene la lista de partidos programados en el rango de fechas especificado."""
    url = f"{BASE_URL}/football/matches?date_from={date_from}&date_to={date_to}&status=scheduled&per_page=100"
    
    response = requests.get(url, headers=HEADERS)
    if response.status_code == 200:
        return response.json().get("data", [])
    print(f"Error al conectar con TheStatsAPI: Code {response.status_code}")
    return []

def calcular_porcentaje_ht_over05(team_id, condicion):
    """
    Filtra los partidos del equipo por rol e inspecciona el detalle 
    individual para obtener el marcador de la primera mitad.
    """
    url = f"{BASE_URL}/football/matches?team_id={team_id}&status=finished&per_page=70"
    response = requests.get(url, headers=HEADERS)
    
    if response.status_code != 200:
        return 0
        
    partidos_historicos = response.json().get("data", [])
    if not partidos_historicos:
        return 0

    partidos_historicos.sort(key=lambda x: x.get("utc_date", ""), reverse=True)
        
    partidos_validos_procesados = 0
    partidos_con_goles_ht = 0
    MUESTRA_OBJETIVO = 5  
    
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
        
        if detalle_res.status_code == 200:
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
    # 1. Resolver el rango temporal dinámico
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
        
        home_pct = calcular_porcentaje_ht_over05(home_id, "home")
        away_pct = calcular_porcentaje_ht_over05(away_id, "away")
        
        UMBRAL = 80.0
        if home_pct >= UMBRAL and away_pct >= UMBRAL:
            partidos_filtrados.append({
                "ID Partido": match_id,
                "Fecha UTC": partido["utc_date"],
                "Liga ID": partido.get("competition_id"),
                "Local": home_name,
                "Visitante": away_name,
                "% HT Over 0.5 Local (Últimos 5)": round(home_pct, 2),
                "% HT Over 0.5 Visitante (Últimos 5)": round(away_pct, 2)
            })
            
    if partidos_filtrados:
        df = pd.DataFrame(partidos_filtrados)
        df.to_csv("partidos_del_dia.csv", index=False)
        print(f"Proceso finalizado. Se guardaron {len(partidos_filtrados)} partidos óptimos.")
    else:
        columnas = ["ID Partido", "Fecha UTC", "Liga ID", "Local", "Visitante", "% HT Over 0.5 Local (Últimos 5)", "% HT Over 0.5 Visitante (Últimos 5)"]
        pd.DataFrame(columns=columnas).to_csv("partidos_del_dia.csv", index=False)
        print(f"Finalizado. Ningún partido cumplió las condiciones para el rango {d_from} / {d_to}.")

if __name__ == "__main__":
    if not API_KEY:
        print("Error: No se detectó la variable de entorno THE_STATS_API_KEY.")
    else:
        ejecutar_pipeline()
