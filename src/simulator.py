"""
Simulación Monte Carlo del Mundial 2026.

Flujo:
  1. Fase de grupos: cada equipo juega 3 partidos → clasifican top-2 + 8 mejores terceros
  2. Fase eliminatoria: ronda de 32 → 16 → QF → SF → Final
  3. Repetir N veces y acumular estadísticas

El rating usado puede ser ELO puro o un rating compuesto (ELO + FIFA ranking).
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

from .elo_calculator import win_probability
from .world_cup_2026 import GROUPS
from .fifa_ranking import load_fifa_rankings, composite_rating


def simulate_match(
    team_a: str,
    team_b: str,
    ratings: Dict[str, float],
    neutral: bool = True,
    allow_draw: bool = True,
) -> str:
    """Simula un partido y retorna el ganador (o 'draw')."""
    elo_a = ratings.get(team_a, 1500)
    elo_b = ratings.get(team_b, 1500)

    p_win, p_draw, p_loss = win_probability(elo_a, elo_b, neutral=neutral)

    if not allow_draw:
        p_win_adj = p_win / (p_win + p_loss)
        p_loss_adj = p_loss / (p_win + p_loss)
        probs = np.array([p_win_adj, 0.0, p_loss_adj], dtype=float)
    else:
        probs = np.array([p_win, p_draw, p_loss], dtype=float)

    probs /= probs.sum()
    outcome = np.random.choice(["win", "draw", "loss"], p=probs)

    if outcome == "win":
        return team_a
    elif outcome == "loss":
        return team_b
    else:
        return "draw"


def simulate_group_stage(ratings: Dict[str, float]) -> Dict[str, List[str]]:
    """
    Simula la fase de grupos. Retorna dict con 'group_X': [1ro, 2do, 3ro, 4to].
    """
    results = {}

    for group_name, teams in GROUPS.items():
        # Tabla: puntos, goles a favor, goles en contra
        table = {t: {"pts": 0, "gf": 0, "gc": 0, "elo": ratings.get(t, 1500)} for t in teams}

        # Round-robin: cada equipo contra todos los demás
        matches = [
            (teams[i], teams[j])
            for i in range(len(teams))
            for j in range(i + 1, len(teams))
        ]

        for team_a, team_b in matches:
            elo_a = ratings.get(team_a, 1500)
            elo_b = ratings.get(team_b, 1500)

            p_win, p_draw, p_loss = win_probability(elo_a, elo_b, neutral=True)
            probs = np.array([p_win, p_draw, p_loss], dtype=float)
            probs /= probs.sum()
            outcome = np.random.choice(["win", "draw", "loss"], p=probs)

            # Simular marcador aproximado
            expected_goals_a = 1.2 + (elo_a - elo_b) / 800
            expected_goals_b = 1.2 + (elo_b - elo_a) / 800
            gf_a = max(0, int(np.random.poisson(max(0.3, expected_goals_a))))
            gf_b = max(0, int(np.random.poisson(max(0.3, expected_goals_b))))

            if outcome == "win":
                gf_a = max(gf_a, gf_b + 1)
                table[team_a]["pts"] += 3
            elif outcome == "loss":
                gf_b = max(gf_b, gf_a + 1)
                table[team_b]["pts"] += 3
            else:
                gf_a = gf_b = max(gf_a, gf_b)
                table[team_a]["pts"] += 1
                table[team_b]["pts"] += 1

            table[team_a]["gf"] += gf_a
            table[team_a]["gc"] += gf_b
            table[team_b]["gf"] += gf_b
            table[team_b]["gc"] += gf_a

        # Ordenar: puntos → diferencia de goles → goles a favor → ELO
        sorted_teams = sorted(
            teams,
            key=lambda t: (
                table[t]["pts"],
                table[t]["gf"] - table[t]["gc"],
                table[t]["gf"],
                table[t]["elo"],
            ),
            reverse=True,
        )
        results[group_name] = sorted_teams

    return results


def get_third_place_qualifiers(group_results: Dict[str, List[str]], ratings: Dict[str, float]) -> List[str]:
    """Selecciona los 8 mejores terceros clasificados."""
    thirds = [(group, teams[2]) for group, teams in group_results.items()]
    thirds_sorted = sorted(thirds, key=lambda x: ratings.get(x[1], 1500), reverse=True)
    return [team for _, team in thirds_sorted[:8]]


def build_knockout_bracket(group_results: Dict[str, List[str]], ratings: Dict[str, float]) -> List[Tuple[str, str]]:
    """Construye los 16 cruces de la ronda de 32."""
    groups = sorted(group_results.keys())

    # Primeros y segundos de cada grupo
    firsts = {g: group_results[g][0] for g in groups}
    seconds = {g: group_results[g][1] for g in groups}
    thirds = get_third_place_qualifiers(group_results, ratings)

    # Cruces estándar: 1A vs 2B, 1B vs 2A, etc. (por pares de grupos)
    matchups = []
    group_pairs = [
        ("A", "B"), ("C", "D"), ("E", "F"), ("G", "H"),
        ("I", "J"), ("K", "L"),
    ]
    for g1, g2 in group_pairs:
        matchups.append((firsts[g1], seconds[g2]))
        matchups.append((firsts[g2], seconds[g1]))

    # Completar con los 8 mejores terceros (vs primeros de los grupos restantes ponderado)
    # Simplificado: se emparejan los 8 terceros contra los 4 mejores primeros del resto
    extra_firsts = [firsts["C"], firsts["D"], firsts["G"], firsts["H"]]
    for i, third in enumerate(thirds):
        if i < len(extra_firsts):
            matchups.append((extra_firsts[i], third))

    return matchups[:16]


def simulate_knockout_round(teams: List[Tuple[str, str]], ratings: Dict[str, float]) -> List[str]:
    """Simula una ronda eliminatoria. Retorna lista de ganadores."""
    winners = []
    for team_a, team_b in teams:
        winner = simulate_match(team_a, team_b, ratings, neutral=True, allow_draw=False)
        winners.append(winner)
    return winners


def simulate_tournament(ratings: Dict[str, float]) -> str:
    """Simula un torneo completo y retorna el campeón."""
    # Fase de grupos
    group_results = simulate_group_stage(ratings)

    # Ronda de 32
    r32_pairs = build_knockout_bracket(group_results, ratings)
    r16_teams = simulate_knockout_round(r32_pairs, ratings)

    # Ronda de 16
    r16_pairs = list(zip(r16_teams[::2], r16_teams[1::2]))
    qf_teams = simulate_knockout_round(r16_pairs, ratings)

    # Cuartos de final
    qf_pairs = list(zip(qf_teams[::2], qf_teams[1::2]))
    sf_teams = simulate_knockout_round(qf_pairs, ratings)

    # Semifinales
    sf_pairs = list(zip(sf_teams[::2], sf_teams[1::2]))
    finalists = simulate_knockout_round(sf_pairs, ratings)

    # Final
    champion = simulate_match(finalists[0], finalists[1], ratings, neutral=True, allow_draw=False)
    return champion


def build_composite_ratings(
    elo_ratings: Dict[str, float],
    elo_weight: float = 0.6,
) -> Dict[str, float]:
    """
    Combina ELO histórico con puntos FIFA actuales en un rating compuesto.
    Carga automáticamente el ranking FIFA desde el archivo local.
    """
    fifa_rankings = load_fifa_rankings()
    composite = {}
    all_teams = set(elo_ratings.keys()) | set(fifa_rankings.keys())
    for team in all_teams:
        composite[team] = composite_rating(team, elo_ratings, fifa_rankings, elo_weight)
    return composite


def run_monte_carlo(
    ratings: Dict[str, float],
    n_simulations: int = 100_000,
    use_composite: bool = False,
    elo_weight: float = 0.6,
) -> pd.DataFrame:
    """
    Ejecuta N simulaciones completas del Mundial 2026.

    Args:
        ratings: dict equipo → ELO rating
        n_simulations: número de torneos a simular
        use_composite: si True, combina ELO con ranking FIFA actual
        elo_weight: peso del ELO en el rating compuesto (0.0–1.0)

    Retorna DataFrame con probabilidades de campeonato por equipo.
    """
    if use_composite:
        print(f"  Usando rating compuesto (ELO {elo_weight:.0%} + FIFA {1-elo_weight:.0%})")
        effective_ratings = build_composite_ratings(ratings, elo_weight)
    else:
        effective_ratings = ratings

    champion_counts: Dict[str, int] = {}

    for _ in tqdm(range(n_simulations), desc="Simulando torneos"):
        champion = simulate_tournament(effective_ratings)
        champion_counts[champion] = champion_counts.get(champion, 0) + 1

    results = pd.DataFrame([
        {"team": team, "champion_count": count, "probability": count / n_simulations}
        for team, count in champion_counts.items()
    ])
    results = results.sort_values("probability", ascending=False).reset_index(drop=True)
    results["rank"] = results.index + 1
    results["probability_pct"] = (results["probability"] * 100).round(2)

    return results
