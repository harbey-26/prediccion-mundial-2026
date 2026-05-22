"""
Ratings separados de ataque y defensa por selección.
Metodología Dixon-Coles simplificada sobre los últimos N años de partidos.

  attack[team]  = avg_goals_scored   / global_avg   (>1 = goleador fuerte)
  defense[team] = avg_goals_conceded / global_avg   (<1 = defensiva sólida, >1 = débil)

En la simulación de grupos, el lambda Poisson del equipo A contra B es:
  λ_A = GLOBAL_AVG * attack[A] * defense[B]
"""
import numpy as np
import pandas as pd
from typing import Dict

GLOBAL_AVG_GOALS = 1.35  # promedio histórico por equipo por partido
YEARS_WINDOW = 4.0
MIN_MATCHES = 5


def compute_attack_defense(
    df: pd.DataFrame,
    years: float = YEARS_WINDOW,
    min_matches: int = MIN_MATCHES,
) -> Dict[str, Dict[str, float]]:
    """
    Retorna dict: equipo → {"attack": float, "defense": float}

    Usa pesos temporales exponenciales (λ=0.4 / año) para favorecer partidos recientes.
    Solo incluye equipos con al menos `min_matches` partidos en la ventana.
    """
    cutoff = df["date"].max() - pd.Timedelta(days=int(years * 365))
    recent = df[df["date"] >= cutoff].copy()

    if len(recent) == 0:
        return {}

    end_date = recent["date"].max()
    recent["tw"] = np.exp(-0.4 * (end_date - recent["date"]).dt.days / 365.0)

    all_teams = set(recent["home_team"]) | set(recent["away_team"])
    raw: Dict[str, Dict[str, float]] = {}

    for team in all_teams:
        home = recent[recent["home_team"] == team]
        away = recent[recent["away_team"] == team]
        n = len(home) + len(away)

        if n < min_matches:
            continue

        w_total = home["tw"].sum() + away["tw"].sum()
        w_scored = (
            (home["home_score"] * home["tw"]).sum()
            + (away["away_score"] * away["tw"]).sum()
        )
        w_conceded = (
            (home["away_score"] * home["tw"]).sum()
            + (away["home_score"] * away["tw"]).sum()
        )

        raw[team] = {
            "avg_scored": float(w_scored / w_total),
            "avg_conceded": float(w_conceded / w_total),
        }

    if not raw:
        return {}

    global_scored = float(np.mean([s["avg_scored"] for s in raw.values()]))
    global_conceded = float(np.mean([s["avg_conceded"] for s in raw.values()]))

    if global_scored == 0 or global_conceded == 0:
        return {}

    return {
        team: {
            "attack": s["avg_scored"] / global_scored,
            "defense": s["avg_conceded"] / global_conceded,
        }
        for team, s in raw.items()
    }


def expected_goals(
    team_a: str,
    team_b: str,
    attack_defense: Dict[str, Dict[str, float]],
) -> tuple[float, float]:
    """
    Retorna (lambda_a, lambda_b): goles esperados por equipo según Dixon-Coles.
    Fallback a 1.35 si el equipo no tiene datos.
    """
    atk_a = attack_defense.get(team_a, {}).get("attack", 1.0)
    def_b = attack_defense.get(team_b, {}).get("defense", 1.0)
    atk_b = attack_defense.get(team_b, {}).get("attack", 1.0)
    def_a = attack_defense.get(team_a, {}).get("defense", 1.0)

    lambda_a = max(0.25, GLOBAL_AVG_GOALS * atk_a * def_b)
    lambda_b = max(0.25, GLOBAL_AVG_GOALS * atk_b * def_a)
    return lambda_a, lambda_b
