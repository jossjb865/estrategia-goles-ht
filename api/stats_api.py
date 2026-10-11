"""Cliente para The Stats API."""

import requests
from utils.logger import setup_logger


logger = setup_logger(__name__)


class StatsAPIClient:
    """Cliente para interactuar con The Stats API.
    
    Proporciona métodos para obtener partidos, históricos y detalles.
    """

    def __init__(self, api_key):
        """Inicializa cliente con API key.
        
        Args:
            api_key: Token de autenticación para The Stats API
        """
        self.api_key = api_key
        self.base_url = "https://api.thestatsapi.com/api"
        self.headers = {"Authorization": f"Bearer {api_key}"}

    def obtener_partidos_jornada(self, date_from, date_to):
        """Obtiene partidos programados en un rango de fechas.
        
        Args:
            date_from: Fecha inicio (ISO format)
            date_to: Fecha fin (ISO format)
            
        Returns:
            list: Partidos en el rango
            
        Raises:
            requests.HTTPError: Si la respuesta no es 200
        """
        url = f"{self.base_url}/football/matches"
        params = {
            "date_from": date_from,
            "date_to": date_to,
            "status": "scheduled",
            "per_page": 100
        }

        response = requests.get(url, headers=self.headers, params=params)
        logger.debug(f"[API] GET {response.url}")
        logger.debug(f"[API] Status: {response.status_code}")
        
        response.raise_for_status()
        res_json = response.json()
        meta = res_json.get("meta", {})
        logger.info(f"[API] Total partidos: {meta.get('total', 0)} | Páginas: {meta.get('total_pages', 0)}")
        
        return res_json.get("data", [])

    def obtener_historico_equipo(self, team_id, condicion, limit=20):
        """Obtiene histórico de partidos de un equipo.
        
        Args:
            team_id: ID del equipo
            condicion: 'home' (local) o 'away' (visitante)
            limit: Número máximo de partidos
            
        Returns:
            list: Partidos filtrados y ordenados por fecha
            
        Raises:
            requests.HTTPError: Si la respuesta no es 200
        """
        url = f"{self.base_url}/football/matches"
        params = {
            "team_id": team_id,
            "status": "finished",
            "per_page": limit
        }

        response = requests.get(url, headers=self.headers, params=params)
        if response.status_code != 200:
            logger.error(f"[API] Error {response.status_code} para equipo {team_id}")
            response.raise_for_status()

        partidos = response.json().get("data", [])
        partidos.sort(key=lambda x: x.get("utc_date", ""), reverse=True)

        # Filtrar por condición (local o visitante)
        partidos_filtrados = []
        for p in partidos:
            es_local = p["home_team"]["id"] == team_id
            if (condicion == "home" and es_local) or (condicion == "away" and not es_local):
                partidos_filtrados.append(p)

        return partidos_filtrados[:limit]

    def obtener_detalle_partido(self, match_id):
        """Obtiene detalles completos de un partido.
        
        Args:
            match_id: ID del partido
            
        Returns:
            dict: Score del partido con información de goles HT
            
        Raises:
            requests.HTTPError: Si la respuesta no es 200
        """
        url = f"{self.base_url}/football/matches/{match_id}"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json().get("data", {}).get("score", {})
