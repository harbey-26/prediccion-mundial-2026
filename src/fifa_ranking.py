"""
Carga y normalización del ranking FIFA oficial (scrapeado de Transfermarkt).
Fuente: https://www.transfermarkt.com/statistik/weltrangliste
Última actualización: abril 2026
"""
import json
import os
import urllib.request
from bs4 import BeautifulSoup

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
RANKING_FILE = os.path.join(RAW_DIR, "fifa_ranking_2026.json")

# Mapeo de nombres entre Transfermarkt → nombres usados en nuestro dataset
NAME_MAP = {
    "USA":                    "United States",
    "Turkiye":                "Turkey",
    "Czechia":                "Czech Republic",
    "Bosnia":                 "Bosnia and Herzegovina",
    "DR Congo":               "DR Congo",
    "Ivory Coast":            "Ivory Coast",
    "Cape Verde":             "Cape Verde",
    "Saudi Arabia":           "Saudi Arabia",
    "New Zealand":            "New Zealand",
    "South Korea":            "South Korea",
    "South Africa":           "South Africa",
    "Curaçao":                "Curaçao",
}


def scrape_fifa_rankings() -> list[dict]:
    """Scrapea los rankings FIFA desde Transfermarkt (200 equipos)."""
    http_headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
        "Accept-Language": "en-US,en;q=0.9",
    }
    all_data = []
    for page in range(1, 9):
        url = f"https://www.transfermarkt.com/statistik/weltrangliste?ajax=yw1&page={page}"
        req = urllib.request.Request(url, headers=http_headers)
        with urllib.request.urlopen(req, timeout=15) as r:
            soup = BeautifulSoup(r.read(), "lxml")
            for row in soup.select("table.items tbody tr"):
                cols = [td.get_text(strip=True) for td in row.find_all("td")]
                if len(cols) >= 5:
                    all_data.append({
                        "rank": int(cols[0]),
                        "team": cols[1],
                        "points": float(cols[-1]),
                    })

    os.makedirs(RAW_DIR, exist_ok=True)
    with open(RANKING_FILE, "w") as f:
        json.dump(all_data, f, indent=2)
    return all_data


def load_fifa_rankings(force_scrape: bool = False) -> dict[str, dict]:
    """
    Retorna dict: nombre_equipo → {rank, points}
    Nombres normalizados al estándar de nuestro dataset.
    """
    if force_scrape or not os.path.exists(RANKING_FILE):
        print("Scrapeando ranking FIFA desde Transfermarkt...")
        raw = scrape_fifa_rankings()
        print(f"  {len(raw)} equipos descargados.")
    else:
        with open(RANKING_FILE) as f:
            raw = json.load(f)

    rankings = {}
    for entry in raw:
        name = NAME_MAP.get(entry["team"], entry["team"])
        rankings[name] = {
            "rank": entry["rank"],
            "points": entry["points"],
        }
    return rankings


def get_fifa_points(team: str, rankings: dict[str, dict], default_rank: int = 100) -> float:
    """
    Retorna los puntos FIFA de un equipo.
    Si no está en el ranking, estima por posición media (rank ~100 ≈ 1350 pts).
    """
    if team in rankings:
        return rankings[team]["points"]
    # Estimación para equipos no encontrados (p.ej. Curaçao, Haití)
    estimated = max(900.0, 1700.0 - default_rank * 3.5)
    return estimated


def composite_rating(
    team: str,
    elo_ratings: dict[str, float],
    fifa_rankings: dict[str, dict],
    elo_weight: float = 0.6,
) -> float:
    """
    Combina ELO histórico + puntos FIFA en un rating compuesto.
    Ambas métricas están en escala similar (1400–2000), así que se pueden
    mezclar directamente con pesos.

    elo_weight=0.6 → el historial de resultados pesa más que el ranking oficial.
    """
    elo = elo_ratings.get(team, 1500.0)
    fifa_pts = get_fifa_points(team, fifa_rankings)
    return elo_weight * elo + (1 - elo_weight) * fifa_pts
