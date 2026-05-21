"""
Carga y limpieza del dataset de resultados internacionales de fútbol (Kaggle).
Dataset: martj42/international-football-results-from-1872-to-2017
"""
import os
import zipfile
import subprocess
import pandas as pd

DATASET_SLUG = "martj42/international-football-results-from-1872-to-2017"
RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


def download_dataset():
    """Descarga el dataset de Kaggle a data/raw/."""
    os.makedirs(RAW_DIR, exist_ok=True)
    results_path = os.path.join(RAW_DIR, "results.csv")
    if os.path.exists(results_path):
        print("Dataset ya descargado.")
        return

    print("Descargando dataset de Kaggle...")
    result = subprocess.run(
        ["kaggle", "datasets", "download", "-d", DATASET_SLUG, "-p", RAW_DIR, "--unzip"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Error al descargar:\n{result.stderr}\n\n"
            "Asegurate de tener ~/.kaggle/kaggle.json con tu API key de Kaggle.\n"
            "Descargalo en: https://www.kaggle.com/settings -> API -> Create New Token"
        )
    print("Dataset descargado correctamente.")


def load_results() -> pd.DataFrame:
    """Carga y prepara el CSV de resultados históricos."""
    path = os.path.join(RAW_DIR, "results.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            "No se encontró results.csv. Ejecuta download_dataset() primero."
        )

    df = pd.read_csv(path, parse_dates=["date"])
    df = df.dropna(subset=["home_score", "away_score"])
    df = df.sort_values("date").reset_index(drop=True)

    # Ponderación por importancia del torneo
    tournament_weight = {
        "FIFA World Cup": 2.0,
        "UEFA Euro qualification": 1.5,
        "Copa América": 1.5,
        "African Cup of Nations": 1.5,
        "UEFA Euro": 1.8,
        "CONCACAF Gold Cup": 1.4,
        "AFC Asian Cup": 1.4,
        "FIFA World Cup qualification": 1.3,
        "Friendly": 0.5,
    }

    def get_weight(tournament):
        for key, weight in tournament_weight.items():
            if key.lower() in str(tournament).lower():
                return weight
        return 1.0

    df["weight"] = df["tournament"].apply(get_weight)
    return df


def load_shootouts() -> pd.DataFrame:
    path = os.path.join(RAW_DIR, "shootouts.csv")
    if not os.path.exists(path):
        return pd.DataFrame(columns=["date", "home_team", "away_team", "winner"])
    return pd.read_csv(path, parse_dates=["date"])


def get_processed_path(filename: str) -> str:
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    return os.path.join(PROCESSED_DIR, filename)
