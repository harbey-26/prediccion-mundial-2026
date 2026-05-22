"""
Calcula ratings ELO para todas las selecciones a partir del historial de partidos.

ELO Formula:
  E_A = 1 / (1 + 10^((R_B - R_A) / 400))
  R_A_new = R_A + K * weight * goal_mult * time_mult * uncertainty_mult * (S_A - E_A)

Mejoras v2:
  - Goal-difference multiplier (estándar WorldFootballElo)
  - Time decay: partidos recientes pesan más que resultados de hace años
  - K dinámico por incertidumbre: equipos con poca historia aprenden más rápido
  - Probabilidad de empate calibrada empíricamente (P = a*exp(-b*|diff|) + c)
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Union

INITIAL_ELO = 1500
K_BASE = 20
DECAY_LAMBDA = 0.05  # ~45% reducción en 15 años; 0 = sin decay

# Parámetros de probabilidad de empate — reemplazables con calibración empírica.
# P(draw) = _DRAW_A * exp(-_DRAW_B * |ELO_diff|) + _DRAW_C
_DRAW_A: float = 0.18
_DRAW_B: float = 0.005
_DRAW_C: float = 0.07


def set_draw_params(a: float, b: float, c: float) -> None:
    """Inyecta parámetros calibrados empíricamente para la probabilidad de empate."""
    global _DRAW_A, _DRAW_B, _DRAW_C
    _DRAW_A, _DRAW_B, _DRAW_C = a, b, c


def get_draw_params() -> Tuple[float, float, float]:
    return _DRAW_A, _DRAW_B, _DRAW_C


def _goal_multiplier(goal_diff: int) -> float:
    """
    Amplifica el K-factor según la diferencia de goles (estándar WorldFootballElo).
    1 gol → ×1.0  |  2 goles → ×1.5  |  3+ → (11 + diff) / 8
    """
    if goal_diff <= 1:
        return 1.0
    elif goal_diff == 2:
        return 1.5
    else:
        return (11 + goal_diff) / 8


def _uncertainty_multiplier(games_played: int) -> float:
    """K más alto para equipos con poca historia (mayor incertidumbre sobre su nivel real)."""
    if games_played < 30:
        return 1.5
    elif games_played < 80:
        return 1.2
    return 1.0


def compute_elo_ratings(
    df: pd.DataFrame,
    decay_lambda: float = DECAY_LAMBDA,
    return_match_data: bool = False,
) -> Union[Dict[str, float], Tuple[Dict[str, float], pd.DataFrame]]:
    """
    Procesa todos los partidos en orden cronológico y retorna el ELO final de cada equipo.

    Args:
        decay_lambda: tasa de decaimiento temporal (0 = sin decay)
        return_match_data: si True, retorna también un DataFrame por partido para calibración
    """
    ratings: Dict[str, float] = {}
    games_played: Dict[str, int] = {}
    match_records = [] if return_match_data else None

    end_date = df["date"].max()

    for _, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        home_score = int(row["home_score"])
        away_score = int(row["away_score"])
        weight = row.get("weight", 1.0)
        neutral = row.get("neutral", False)
        date = row["date"]

        r_home = ratings.get(home, INITIAL_ELO)
        r_away = ratings.get(away, INITIAL_ELO)

        home_advantage = 0 if neutral else 30
        r_home_adj = r_home + home_advantage

        e_home = 1 / (1 + 10 ** ((r_away - r_home_adj) / 400))
        e_away = 1 - e_home

        if home_score > away_score:
            s_home, s_away, outcome = 1.0, 0.0, "win"
        elif home_score < away_score:
            s_home, s_away, outcome = 0.0, 1.0, "loss"
        else:
            s_home, s_away, outcome = 0.5, 0.5, "draw"

        goal_diff = abs(home_score - away_score)
        goal_mult = _goal_multiplier(goal_diff)

        years_ago = (end_date - date).days / 365.25
        time_mult = np.exp(-decay_lambda * years_ago)

        gp_home = games_played.get(home, 0)
        gp_away = games_played.get(away, 0)

        base_k = K_BASE * weight * goal_mult * time_mult
        k_home = base_k * _uncertainty_multiplier(gp_home)
        k_away = base_k * _uncertainty_multiplier(gp_away)

        if return_match_data:
            elo_diff = r_home_adj - r_away
            match_records.append({
                "date": date,
                "home_team": home,
                "away_team": away,
                "elo_diff": elo_diff,
                "elo_diff_abs": abs(elo_diff),
                "outcome": outcome,
                "tournament": row.get("tournament", ""),
                "neutral": neutral,
            })

        ratings[home] = r_home + k_home * (s_home - e_home)
        ratings[away] = r_away + k_away * (s_away - e_away)
        games_played[home] = gp_home + 1
        games_played[away] = gp_away + 1

    if return_match_data:
        return ratings, pd.DataFrame(match_records)
    return ratings


def compute_elo_history(df: pd.DataFrame, decay_lambda: float = DECAY_LAMBDA) -> pd.DataFrame:
    """Retorna un DataFrame con el ELO de cada equipo después de cada partido."""
    ratings: Dict[str, float] = {}
    games_played: Dict[str, int] = {}
    records = []

    end_date = df["date"].max()

    for _, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        home_score = int(row["home_score"])
        away_score = int(row["away_score"])
        weight = row.get("weight", 1.0)
        neutral = row.get("neutral", False)
        date = row["date"]

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

        goal_diff = abs(home_score - away_score)
        goal_mult = _goal_multiplier(goal_diff)
        years_ago = (end_date - date).days / 365.25
        time_mult = np.exp(-decay_lambda * years_ago)
        gp_home = games_played.get(home, 0)
        gp_away = games_played.get(away, 0)
        base_k = K_BASE * weight * goal_mult * time_mult
        k_home = base_k * _uncertainty_multiplier(gp_home)
        k_away = base_k * _uncertainty_multiplier(gp_away)

        ratings[home] = r_home + k_home * (s_home - e_home)
        ratings[away] = r_away + k_away * (s_away - e_away)
        games_played[home] = gp_home + 1
        games_played[away] = gp_away + 1

        records.append({
            "date": date, "team": home, "elo": ratings[home],
            "opponent": away, "tournament": row.get("tournament", ""),
        })
        records.append({
            "date": date, "team": away, "elo": ratings[away],
            "opponent": home, "tournament": row.get("tournament", ""),
        })

    return pd.DataFrame(records)


def win_probability(elo_a: float, elo_b: float, neutral: bool = True) -> Tuple[float, float, float]:
    """
    Retorna (prob_A_gana, prob_empate, prob_B_gana) basado en diferencia ELO.
    La probabilidad de empate sigue la curva calibrada: a * exp(-b * |diff|) + c.
    """
    home_adv = 0 if neutral else 30
    diff = (elo_a + home_adv) - elo_b

    p_win_raw = 1 / (1 + 10 ** (-diff / 400))

    p_draw = float(_DRAW_A * np.exp(-_DRAW_B * abs(diff)) + _DRAW_C)
    p_draw = max(0.05, min(p_draw, 0.40))

    p_win = p_win_raw * (1 - p_draw)
    p_loss = (1 - p_win_raw) * (1 - p_draw)

    return round(p_win, 4), round(p_draw, 4), round(p_loss, 4)
