import os
import datetime
import requests
import pandas as pd

# Configuración de URL base y credenciales oficiales según la documentación técnica
API_KEY = os.environ.get("THE_STATS_API_KEY")
BASE_URL = "https://thestatsapi.com"  # URL corregida con prefijo /api
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

def obtener_partidos_hoy():
    """Obtiene la lista de partidos programados para el día actual."""
    hoy = datetime.date.today().isoformat()
    url = f"{BASE_URL}/football/matches?date_from={hoy}&date_to={hoy}&status=scheduled&per_page=100"
    
    response = requests.get(url, headers=HEADERS)
    if response.status_code == 200:
        return response.json().get("data", [])
    return []

def calcular_porcentaje_ht_over05(team_id, condicion):
    """
    Extrae una muestra amplia de partidos finalizados del equipo,
    los filtra por rol (local/visitante) y consulta el detalle individual 
    para obtener de forma segura el marcador de la primera mitad.
    """
    # Solicitamos una muestra lo suficientemente amplia para asegurar historial por rol
    url = f"{BASE_URL}/football/matches?team_id={team_id}&status=finished&per_page=70"
    response = requests.get(url, headers=HEADERS)
    
    if response.status_code != 200:
        return 0
        
    partidos_historicos = response.json().get("data", [])
    if not partidos_historicos:
        return 0

    # Ordenamos explícitamente de forma descendente por fecha (más recientes primero)
    partidos_historicos.sort(key=lambda x: x.get("utc_date", ""), reverse=True)
        
    partidos_validos_procesados = 0
    partidos_con_goles_ht = 0
    MUESTRA_OBJETIVO = 5  # Analizaremos con precisión los últimos 5 partidos en esa condición
    
    for p_resumido in partidos_historicos:
        if partidos_validos_procesados >= MUESTRA_OBJETIVO:
            break
            
        # Validar condición estricta de Local o Visitante
        es_local = p_resumido["home_team"]["id"] == team_id
        if condicion == "home" and not es_local:
            continue
        if condicion == "away" and es_local:
            continue
            
        # Realizar llamada obligatoria al detalle para obtener el HT Score de forma segura
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
    partidos_hoy = obtener_partidos_hoy()
    partidos_filtrados = []
    
    print(f"Iniciando análisis profundo para {len(partidos_hoy)} partidos del día...")
    
    for partido in partidos_hoy:
        match_id = partido["id"]
        home_id = partido["home_team"]["id"]
        away_id = partido["away_team"]["id"]
        home_name = partido["home_team"]["name"]
        away_name = partido["away_team"]["name"]
        
        # Procesamiento secuencial con cálculo algorítmico estricto
        home_pct = calcular_porcentaje_ht_over05(home_id, "home")
        away_pct = calcular_porcentaje_ht_over05(away_id, "away")
        
        # Umbral: Ambos deben registrar al menos un 80% en sus últimos 5 partidos bajo su rol actual
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
            
    # Gestión del archivo de salida para la persistencia en el repositorio GitHub
    if partidos_filtrados:
        df = pd.DataFrame(partidos_filtrados)
        df.to_csv("partidos_del_dia.csv", index=False)
        print(f"Proceso finalizado. Se guardaron {len(partidos_filtrados)} partidos óptimos.")
    else:
        columnas = ["ID Partido", "Fecha UTC", "Liga ID", "Local", "Visitante", "% HT Over 0.5 Local (Últimos 5)", "% HT Over 0.5 Visitante (Últimos 5)"]
        pd.DataFrame(columns=columnas).to_csv("partidos_del_dia.csv", index=False)
        print("Finalizado. Ningún partido cumplió las condiciones el día de hoy.")

if __name__ == "__main__":
    if not API_KEY:
        print("Error: No se detectó la variable de entorno THE_STATS_API_KEY.")
    else:
        ejecutar_pipeline()
