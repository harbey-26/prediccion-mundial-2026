"""
Simulación Monte Carlo del Mundial 2026.

Flujo:
  1. Fase de grupos: round-robin → clasifican top-2 de cada grupo + 8 mejores terceros
  2. Fase eliminatoria: R32 → R16 → QF → SF → Final
  3. Repetir N veces y acumular estadísticas

Mejoras v2:
  - Poisson lambdas calculados con ratings separados de ataque/defensa (Dixon-Coles)
  - Forma reciente aplicada como ajuste ELO en la probabilidad de resultado
  - Selección de mejores terceros por puntos → GD → GF → ELO (regla FIFA real)
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

from .elo_calculator import win_probability
from . import world_cup_2026  # importar módulo (no atributo) para que backtest pueda reasignar GROUPS
from .fifa_ranking import load_fifa_rankings, composite_rating
from .attack_defense import expected_goals, GLOBAL_AVG_GOALS


def simulate_match(
    team_a: str,
    team_b: str,
    ratings: Dict[str, float],
    neutral: bool = True,
    allow_draw: bool = True,
    form_adj: Optional[Dict[str, float]] = None,
    ml_model=None,
    ml_weight: float = 0.35,
    form_raw: Optional[Dict[str, float]] = None,
    attack_defense: Optional[Dict[str, Dict[str, float]]] = None,
    ml_cache: Optional[Dict] = None,
) -> str:
    """Simula un partido y retorna el ganador (o 'draw').

    ml_cache: dict pre-calculado {(ta, tb): (p_win, p_draw, p_loss)} —
              evita llamar predict_proba en cada simulación (mucho más rápido).
    """
    elo_a = ratings.get(team_a, 1500) + (form_adj.get(team_a, 0.0) if form_adj else 0.0)
    elo_b = ratings.get(team_b, 1500) + (form_adj.get(team_b, 0.0) if form_adj else 0.0)

    p_win, p_draw, p_loss = win_probability(elo_a, elo_b, neutral=neutral)

    # Blend con modelo ML (usa caché pre-calculada si está disponible)
    if ml_cache is not None and (team_a, team_b) in ml_cache:
        ml_win, ml_draw, ml_loss = ml_cache[(team_a, team_b)]
        p_win = (1 - ml_weight) * p_win + ml_weight * ml_win
        p_draw = (1 - ml_weight) * p_draw + ml_weight * ml_draw
        p_loss = (1 - ml_weight) * p_loss + ml_weight * ml_loss
    elif ml_model is not None and form_raw is not None:
        from .ml_model import predict_match_ml
        ml_win, ml_draw, ml_loss = predict_match_ml(
            team_a, team_b, ratings, form_raw, attack_defense, ml_model, neutral
        )
        p_win = (1 - ml_weight) * p_win + ml_weight * ml_win
        p_draw = (1 - ml_weight) * p_draw + ml_weight * ml_draw
        p_loss = (1 - ml_weight) * p_loss + ml_weight * ml_loss

    if not allow_draw:
        total = p_win + p_loss
        probs = np.array([p_win / total, 0.0, p_loss / total])
    else:
        probs = np.array([p_win, p_draw, p_loss])

    probs /= probs.sum()
    outcome = np.random.choice(["win", "draw", "loss"], p=probs)

    if outcome == "win":
        return team_a
    elif outcome == "loss":
        return team_b
    return "draw"


def simulate_group_stage(
    ratings: Dict[str, float],
    attack_defense: Optional[Dict[str, Dict[str, float]]] = None,
    form_adj: Optional[Dict[str, float]] = None,
    ml_model=None,
    ml_weight: float = 0.35,
    form_raw: Optional[Dict[str, float]] = None,
    ml_cache: Optional[Dict] = None,
) -> Tuple[Dict[str, List[str]], Dict[str, Dict]]:
    """
    Simula la fase de grupos.
    Retorna (group_results, group_tables):
      - group_results: {grupo: [1°, 2°, 3°, 4°]}
      - group_tables: {grupo: {equipo: {pts, gf, gc, gd, elo}}}
    """
    results: Dict[str, List[str]] = {}
    tables: Dict[str, Dict] = {}

    for group_name, teams in world_cup_2026.GROUPS.items():
        table = {
            t: {
                "pts": 0, "gf": 0, "gc": 0,
                "elo": ratings.get(t, 1500),
            }
            for t in teams
        }

        matches = [
            (teams[i], teams[j])
            for i in range(len(teams))
            for j in range(i + 1, len(teams))
        ]

        for team_a, team_b in matches:
            # Resultado via simulate_match (incluye ML blend si está activo)
            result = simulate_match(
                team_a, team_b, ratings, neutral=True, allow_draw=True,
                form_adj=form_adj, ml_model=ml_model, ml_weight=ml_weight,
                form_raw=form_raw, attack_defense=attack_defense, ml_cache=ml_cache,
            )
            outcome = "win" if result == team_a else ("draw" if result == "draw" else "loss")

            # Lambdas Poisson: Dixon-Coles si hay datos, fallback a ELO
            if attack_defense:
                lambda_a, lambda_b = expected_goals(team_a, team_b, attack_defense)
            else:
                elo_a = ratings.get(team_a, 1500) + (form_adj.get(team_a, 0.0) if form_adj else 0.0)
                elo_b = ratings.get(team_b, 1500) + (form_adj.get(team_b, 0.0) if form_adj else 0.0)
                lambda_a = max(0.3, 1.2 + (elo_a - elo_b) / 800)
                lambda_b = max(0.3, 1.2 + (elo_b - elo_a) / 800)

            gf_a = int(np.random.poisson(lambda_a))
            gf_b = int(np.random.poisson(lambda_b))

            # Ajustar marcador para que sea consistente con el resultado sorteado
            if outcome == "win":
                if gf_a <= gf_b:
                    gf_a = gf_b + 1
                table[team_a]["pts"] += 3
            elif outcome == "loss":
                if gf_b <= gf_a:
                    gf_b = gf_a + 1
                table[team_b]["pts"] += 3
            else:
                gf_a = gf_b = min(gf_a, gf_b)  # mismo marcador para empate
                table[team_a]["pts"] += 1
                table[team_b]["pts"] += 1

            table[team_a]["gf"] += gf_a
            table[team_a]["gc"] += gf_b
            table[team_b]["gf"] += gf_b
            table[team_b]["gc"] += gf_a

        # Ordenar: puntos → GD → GF → ELO (criterio FIFA oficial)
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
        tables[group_name] = table

    return results, tables


def get_third_place_qualifiers(
    group_results: Dict[str, List[str]],
    group_tables: Dict[str, Dict],
    ratings: Dict[str, float],
    n: int = 8,
) -> List[str]:
    """
    Selecciona los N mejores terceros clasificados.
    Criterio: puntos → GD → GF → ELO (regla FIFA real, no solo ELO).
    """
    thirds = []
    for group, teams in group_results.items():
        team = teams[2]  # tercero del grupo
        tbl = group_tables[group][team]
        thirds.append({
            "team": team,
            "pts": tbl["pts"],
            "gd": tbl["gf"] - tbl["gc"],
            "gf": tbl["gf"],
            "elo": ratings.get(team, 1500),
        })

    thirds_sorted = sorted(
        thirds,
        key=lambda x: (x["pts"], x["gd"], x["gf"], x["elo"]),
        reverse=True,
    )
    return [t["team"] for t in thirds_sorted[:n]]


def build_knockout_bracket(
    group_results: Dict[str, List[str]],
    group_tables: Dict[str, Dict],
    ratings: Dict[str, float],
) -> List[Tuple[str, str]]:
    """
    Construye los cruces de la ronda eliminatoria inicial.
    Formato 8 grupos (32 equipos): genera R16 directamente sin terceros.
    Formato 12 grupos (48 equipos): genera R32 con 8 mejores terceros.
    """
    groups = sorted(group_results.keys())
    firsts = {g: group_results[g][0] for g in groups}
    seconds = {g: group_results[g][1] for g in groups}
    matchups: List[Tuple[str, str]] = []

    if len(groups) == 8:
        # Formato 32 equipos (2018, 2022): 4 pares de grupos → 8 cruces en R16
        group_pairs = [("A", "B"), ("C", "D"), ("E", "F"), ("G", "H")]
        for g1, g2 in group_pairs:
            matchups.append((firsts[g1], seconds[g2]))
            matchups.append((firsts[g2], seconds[g1]))
    else:
        # Formato 48 equipos (2026): 6 pares + 8 mejores terceros → 16 cruces en R32
        thirds = get_third_place_qualifiers(group_results, group_tables, ratings)
        group_pairs = [
            ("A", "B"), ("C", "D"), ("E", "F"), ("G", "H"),
            ("I", "J"), ("K", "L"),
        ]
        for g1, g2 in group_pairs:
            matchups.append((firsts[g1], seconds[g2]))
            matchups.append((firsts[g2], seconds[g1]))
        for i in range(0, len(thirds) - 1, 2):
            matchups.append((thirds[i], thirds[i + 1]))

    return matchups[:16]


def simulate_knockout_round(
    teams: List[Tuple[str, str]],
    ratings: Dict[str, float],
    form_adj: Optional[Dict[str, float]] = None,
    ml_model=None,
    ml_weight: float = 0.35,
    form_raw: Optional[Dict[str, float]] = None,
    attack_defense: Optional[Dict[str, Dict[str, float]]] = None,
    ml_cache: Optional[Dict] = None,
) -> List[str]:
    """Simula una ronda eliminatoria (sin empates). Retorna ganadores."""
    return [
        simulate_match(
            a, b, ratings, neutral=True, allow_draw=False, form_adj=form_adj,
            ml_model=ml_model, ml_weight=ml_weight, form_raw=form_raw,
            attack_defense=attack_defense, ml_cache=ml_cache,
        )
        for a, b in teams
    ]


def simulate_tournament(
    ratings: Dict[str, float],
    attack_defense: Optional[Dict[str, Dict[str, float]]] = None,
    form_adj: Optional[Dict[str, float]] = None,
    ml_model=None,
    ml_weight: float = 0.35,
    form_raw: Optional[Dict[str, float]] = None,
    ml_cache: Optional[Dict] = None,
) -> str:
    """
    Simula un torneo completo y retorna el campeón.
    Soporta formato 32 equipos (8 grupos, 2018/2022) y 48 equipos (12 grupos, 2026).
    """
    # Fase de grupos
    group_results, group_tables = simulate_group_stage(
        ratings, attack_defense, form_adj, ml_model, ml_weight, form_raw, ml_cache
    )

    # Primera ronda eliminatoria (R32 en 2026, R16 en 2018/2022)
    initial_pairs = build_knockout_bracket(group_results, group_tables, ratings)
    next_round = simulate_knockout_round(
        initial_pairs, ratings, form_adj, ml_model, ml_weight, form_raw, attack_defense, ml_cache
    )

    # Rondas siguientes hasta que queden 2 finalistas
    while len(next_round) > 2:
        pairs = list(zip(next_round[::2], next_round[1::2]))
        next_round = simulate_knockout_round(
            pairs, ratings, form_adj, ml_model, ml_weight, form_raw, attack_defense, ml_cache
        )

    # Final
    return simulate_match(
        next_round[0], next_round[1], ratings, neutral=True, allow_draw=False,
        form_adj=form_adj, ml_model=ml_model, ml_weight=ml_weight,
        form_raw=form_raw, attack_defense=attack_defense, ml_cache=ml_cache,
    )


def build_composite_ratings(
    elo_ratings: Dict[str, float],
    elo_weight: float = 0.6,
) -> Dict[str, float]:
    """Combina ELO histórico con puntos FIFA en un rating compuesto."""
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
    attack_defense: Optional[Dict[str, Dict[str, float]]] = None,
    form_adj: Optional[Dict[str, float]] = None,
    ml_model=None,
    ml_weight: float = 0.35,
    form_raw: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Ejecuta N simulaciones completas del Mundial 2026.

    Args:
        ratings: dict equipo → ELO rating
        n_simulations: número de torneos a simular
        use_composite: si True, combina ELO con ranking FIFA actual
        elo_weight: peso del ELO en el rating compuesto (0.0–1.0)
        attack_defense: ratings separados de ataque/defensa (Dixon-Coles)
        form_adj: ajuste ELO por forma reciente por equipo
        ml_model: modelo XGBoost entrenado (opcional)
        ml_weight: peso del modelo ML en el blend (0.0–1.0)
        form_raw: form scores sin normalizar para features ML
    """
    if use_composite:
        print(f"  Rating compuesto (ELO {elo_weight:.0%} + FIFA {1-elo_weight:.0%})")
        effective_ratings = build_composite_ratings(ratings, elo_weight)
    else:
        effective_ratings = ratings

    features = []
    if attack_defense:
        features.append("ataque/defensa")
    if form_adj:
        features.append("forma reciente")
    if ml_model is not None:
        features.append(f"ML ensemble (α={ml_weight:.0%})")
    if features:
        print(f"  Variables adicionales: {', '.join(features)}")

    # Pre-calcular probabilidades ML para todos los pares (evita O(n_sims × n_matches) inferencias)
    ml_cache: Optional[Dict] = None
    if ml_model is not None and form_raw is not None:
        from .ml_model import precompute_ml_probs
        from . import world_cup_2026
        all_teams = [t for teams in world_cup_2026.GROUPS.values() for t in teams]
        print("  Pre-calculando probabilidades ML para los 48 equipos...")
        ml_cache = precompute_ml_probs(
            all_teams, effective_ratings, form_raw, attack_defense, ml_model, neutral=True
        )
        print(f"  Cache ML: {len(ml_cache):,} pares pre-computados.")

    champion_counts: Dict[str, int] = {}

    for _ in tqdm(range(n_simulations), desc="Simulando torneos"):
        champion = simulate_tournament(
            effective_ratings, attack_defense, form_adj,
            ml_model=None, ml_weight=ml_weight, form_raw=None,
            ml_cache=ml_cache,
        )
        champion_counts[champion] = champion_counts.get(champion, 0) + 1

    results = pd.DataFrame([
        {"team": team, "champion_count": count, "probability": count / n_simulations}
        for team, count in champion_counts.items()
    ])
    results = results.sort_values("probability", ascending=False).reset_index(drop=True)
    results["rank"] = results.index + 1
    results["probability_pct"] = (results["probability"] * 100).round(2)

    return results
