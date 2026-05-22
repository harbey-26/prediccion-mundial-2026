"""
Factor de forma reciente por selección.

Calcula el rendimiento ponderado en los últimos N partidos (default: 10).
Los resultados más recientes pesan más (decaimiento exponencial λ=0.15).

La forma se expresa como un ajuste ELO aditivo en el rango [-80, +80]:
  +80  → equipo en racha excepcional (10 victorias seguidas)
    0  → rendimiento exactamente promedio
  -80  → equipo en crisis total (10 derrotas seguidas)

Este ajuste se suma al rating compuesto en la simulación, con peso form_weight.
"""
import numpy as np
import pandas as pd
from typing import Dict

N_MATCHES = 10
MAX_ELO_ADJ = 80.0   # máximo ajuste ELO en puntos
FORM_WEIGHT = 0.15   # qué fracción del ajuste se aplica al rating efectivo
# Ventana temporal para filtrar equipos con historia real reciente
FORM_CUTOFF_YEARS = 2.0
MIN_RECENT_MATCHES = 6  # mínimo para ser incluido en la normalización


def compute_form(df: pd.DataFrame, n_matches: int = N_MATCHES) -> Dict[str, float]:
    """
    Retorna dict: equipo → ajuste ELO por forma en [-MAX_ELO_ADJ, +MAX_ELO_ADJ].

    La normalización (media/std) se calcula solo sobre equipos con >=MIN_RECENT_MATCHES
    partidos en los últimos FORM_CUTOFF_YEARS años, para evitar que equipos amateurs
    con pocas victorias sesguen la escala de referencia.
    """
    df_sorted = df.sort_values("date", ascending=False)
    cutoff = df["date"].max() - pd.Timedelta(days=int(FORM_CUTOFF_YEARS * 365))
    all_teams = set(df["home_team"]) | set(df["away_team"])

    raw: Dict[str, float] = {}
    recent_match_count: Dict[str, int] = {}

    for team in all_teams:
        mask = (df_sorted["home_team"] == team) | (df_sorted["away_team"] == team)
        recent = df_sorted[mask].head(n_matches)

        if len(recent) == 0:
            raw[team] = 0.0
            recent_match_count[team] = 0
            continue

        # Contar partidos recientes para filtro de normalización
        n_recent = int(((df_sorted[mask])["date"] >= cutoff).sum())
        recent_match_count[team] = n_recent

        total_pts = 0.0
        total_weight = 0.0

        for i, (_, row) in enumerate(recent.iterrows()):
            w = np.exp(-0.15 * i)

            if row["home_team"] == team:
                scored, conceded = row["home_score"], row["away_score"]
            else:
                scored, conceded = row["away_score"], row["home_score"]

            if scored > conceded:
                pts = 3.0
            elif scored == conceded:
                pts = 1.0
            else:
                pts = 0.0

            total_pts += pts * w
            total_weight += w

        raw[team] = total_pts / total_weight if total_weight > 0 else 0.0

    if not raw:
        return {}

    # Normalizar solo con equipos que tienen actividad reciente real
    qualifying = [
        score for team, score in raw.items()
        if recent_match_count.get(team, 0) >= MIN_RECENT_MATCHES
    ]
    if len(qualifying) < 10:
        qualifying = list(raw.values())

    avg = float(np.mean(qualifying))
    std = float(np.std(qualifying)) or 1.0

    form_adj: Dict[str, float] = {}
    for team, score in raw.items():
        z = (score - avg) / std
        adj = float(np.clip(z * MAX_ELO_ADJ / 2.0, -MAX_ELO_ADJ, MAX_ELO_ADJ))
        form_adj[team] = adj

    return form_adj


def compute_form_raw(df: pd.DataFrame, n_matches: int = N_MATCHES) -> Dict[str, float]:
    """
    Retorna raw weighted form scores en escala 0–3 (sin normalizar).
    Usado como feature para el modelo ML — compatible con build_feature_matrix.
    """
    df_sorted = df.sort_values("date", ascending=False)
    all_teams = set(df["home_team"]) | set(df["away_team"])
    raw: Dict[str, float] = {}

    for team in all_teams:
        mask = (df_sorted["home_team"] == team) | (df_sorted["away_team"] == team)
        recent = df_sorted[mask].head(n_matches)

        if len(recent) == 0:
            raw[team] = 1.0
            continue

        total_pts = total_w = 0.0
        for i, (_, row) in enumerate(recent.iterrows()):
            w = np.exp(-0.15 * i)
            if row["home_team"] == team:
                scored, conceded = row["home_score"], row["away_score"]
            else:
                scored, conceded = row["away_score"], row["home_score"]
            pts = 3.0 if scored > conceded else (1.0 if scored == conceded else 0.0)
            total_pts += pts * w
            total_w += w

        raw[team] = total_pts / total_w if total_w > 0 else 1.0

    return raw


def apply_form(
    ratings: Dict[str, float],
    form_adj: Dict[str, float],
    weight: float = FORM_WEIGHT,
) -> Dict[str, float]:
    """
    Retorna un nuevo dict de ratings con el ajuste de forma aplicado.
    rating_efectivo = rating_base + weight * form_adj
    """
    adjusted = {}
    for team, base in ratings.items():
        adj = form_adj.get(team, 0.0)
        adjusted[team] = base + weight * adj
    return adjusted
