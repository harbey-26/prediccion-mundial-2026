"""
Calcula ratings ELO para todas las selecciones a partir del historial de partidos.

ELO Formula:
  E_A = 1 / (1 + 10^((R_B - R_A) / 400))
  R_A_new = R_A + K * weight * (S_A - E_A)

  S_A = 1 (victoria), 0.5 (empate), 0 (derrota)
  K = 20 (base)
"""
import numpy as np
import pandas as pd
from typing import Dict

INITIAL_ELO = 1500
K_BASE = 20


def compute_elo_ratings(df: pd.DataFrame) -> Dict[str, float]:
    """
    Procesa todos los partidos en orden cronológico y retorna el ELO final de cada equipo.
    """
    ratings: Dict[str, float] = {}

    for _, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        home_score = row["home_score"]
        away_score = row["away_score"]
        weight = row.get("weight", 1.0)
        neutral = row.get("neutral", False)

        r_home = ratings.get(home, INITIAL_ELO)
        r_away = ratings.get(away, INITIAL_ELO)

        # Ventaja de local (30 puntos ELO si no es campo neutral)
        home_advantage = 0 if neutral else 30
        r_home_adj = r_home + home_advantage

        # Probabilidades esperadas
        e_home = 1 / (1 + 10 ** ((r_away - r_home_adj) / 400))
        e_away = 1 - e_home

        # Resultado real
        if home_score > away_score:
            s_home, s_away = 1.0, 0.0
        elif home_score < away_score:
            s_home, s_away = 0.0, 1.0
        else:
            s_home, s_away = 0.5, 0.5

        # Actualización ELO
        k = K_BASE * weight
        ratings[home] = r_home + k * (s_home - e_home)
        ratings[away] = r_away + k * (s_away - e_away)

    return ratings


def compute_elo_history(df: pd.DataFrame) -> pd.DataFrame:
    """
    Retorna un DataFrame con el ELO de cada equipo después de cada partido.
    Útil para visualización y análisis.
    """
    ratings: Dict[str, float] = {}
    records = []

    for _, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        home_score = row["home_score"]
        away_score = row["away_score"]
        weight = row.get("weight", 1.0)
        neutral = row.get("neutral", False)

        r_home = ratings.get(home, INITIAL_ELO)
        r_away = ratings.get(away, INITIAL_ELO)

        home_advantage = 0 if neutral else 30
        r_home_adj = r_home + home_advantage

        e_home = 1 / (1 + 10 ** ((r_away - r_home_adj) / 400))
        e_away = 1 - e_home

        if home_score > away_score:
            s_home, s_away = 1.0, 0.0
        elif home_score < away_score:
            s_home, s_away = 0.0, 1.0
        else:
            s_home, s_away = 0.5, 0.5

        k = K_BASE * weight
        ratings[home] = r_home + k * (s_home - e_home)
        ratings[away] = r_away + k * (s_away - e_away)

        records.append({
            "date": row["date"],
            "team": home,
            "elo": ratings[home],
            "opponent": away,
            "tournament": row.get("tournament", ""),
        })
        records.append({
            "date": row["date"],
            "team": away,
            "elo": ratings[away],
            "opponent": home,
            "tournament": row.get("tournament", ""),
        })

    return pd.DataFrame(records)


def win_probability(elo_a: float, elo_b: float, neutral: bool = True) -> tuple[float, float, float]:
    """
    Retorna (prob_A_gana, prob_empate, prob_B_gana) basado en diferencia ELO.
    Modelo de Dixon-Coles simplificado: prob_empate ≈ 25% base, ajustado por diferencia.
    """
    home_adv = 0 if neutral else 30
    diff = (elo_a + home_adv) - elo_b

    # Probabilidad esperada de victoria para A según ELO
    p_win_raw = 1 / (1 + 10 ** (-diff / 400))

    # Distribuir probabilidad entre win/draw/loss
    # La prob de empate decrece cuando la diferencia es grande
    p_draw = max(0.10, 0.28 - 0.001 * abs(diff))
    p_win = p_win_raw * (1 - p_draw)
    p_loss = (1 - p_win_raw) * (1 - p_draw)

    return round(p_win, 4), round(p_draw, 4), round(p_loss, 4)
