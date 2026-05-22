"""
Backtesting del modelo de predicción sobre Mundiales 2018 y 2022.

Para cada torneo:
  1. Filtra el dataset a partidos ANTERIORES a la fecha de inicio del Mundial
  2. Calcula ELO, calibra empate, obtiene attack/defense y forma con esos datos
  3. Simula 50,000 torneos con los grupos y equipos reales de ese Mundial
  4. Compara probabilidades predichas vs resultado real (Brier Score, Log-Loss)

Brier Score: promedio de (prob_predicha - resultado_real)^2 para cada equipo.
  - Rango: 0 (perfecto) a 1 (pésimo). Modelo naive (equiprobable): ~0.021
  - Modelo bueno: < 0.015

Log-Loss:
  - Un modelo perfecto que predice 100% al campeón real: log-loss ≈ 0
  - Un modelo que asigna 2% al campeón real: -log(0.02) ≈ 3.9
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple

# Datos reales de mundiales para backtesting
WORLD_CUP_DATA = {
    2018: {
        "name": "Rusia 2018",
        "start_date": "2018-06-14",
        "champion": "France",
        "groups": {
            "A": ["Russia", "Saudi Arabia", "Egypt", "Uruguay"],
            "B": ["Portugal", "Spain", "Morocco", "Iran"],
            "C": ["France", "Australia", "Peru", "Denmark"],
            "D": ["Argentina", "Iceland", "Croatia", "Nigeria"],
            "E": ["Brazil", "Switzerland", "Costa Rica", "Serbia"],
            "F": ["Germany", "Mexico", "Sweden", "South Korea"],
            "G": ["Belgium", "Panama", "Tunisia", "England"],
            "H": ["Poland", "Senegal", "Colombia", "Japan"],
        },
    },
    2022: {
        "name": "Qatar 2022",
        "start_date": "2022-11-20",
        "champion": "Argentina",
        "groups": {
            "A": ["Qatar", "Ecuador", "Senegal", "Netherlands"],
            "B": ["England", "Iran", "United States", "Wales"],
            "C": ["Argentina", "Saudi Arabia", "Mexico", "Poland"],
            "D": ["France", "Australia", "Denmark", "Tunisia"],
            "E": ["Spain", "Costa Rica", "Germany", "Japan"],
            "F": ["Belgium", "Canada", "Morocco", "Croatia"],
            "G": ["Brazil", "Serbia", "Switzerland", "Cameroon"],
            "H": ["Portugal", "Ghana", "Uruguay", "South Korea"],
        },
    },
}


def brier_score(probabilities: Dict[str, float], champion: str) -> float:
    """
    Calcula el Brier Score: promedio de (p_i - o_i)^2 sobre todos los equipos.
    o_i = 1 si equipo i fue campeón, 0 si no.
    """
    total = 0.0
    n = len(probabilities)
    for team, prob in probabilities.items():
        actual = 1.0 if team == champion else 0.0
        total += (prob - actual) ** 2
    return total / n if n > 0 else 1.0


def log_loss_champion(probabilities: Dict[str, float], champion: str) -> float:
    """
    Log-loss del campeón real: -log(prob_campeón).
    Más bajo es mejor.
    """
    p = probabilities.get(champion, 1e-6)
    return -np.log(max(p, 1e-6))


def run_backtest(
    year: int,
    df_full: pd.DataFrame,
    n_simulations: int = 50_000,
    use_composite: bool = True,
    use_attack_defense: bool = True,
    use_form: bool = True,
) -> Dict:
    """
    Ejecuta el backtesting para un Mundial específico.

    Args:
        year: 2018 o 2022
        df_full: dataset completo de partidos
        n_simulations: simulaciones Monte Carlo
        use_composite: usar ELO + FIFA (no disponible para mundiales pasados, usa solo ELO)
        use_attack_defense: usar ratings separados de ataque/defensa
        use_form: usar factor de forma reciente

    Returns:
        dict con métricas: brier_score, log_loss, champion_rank, champion_prob, probabilities
    """
    from tqdm import tqdm
    from .elo_calculator import compute_elo_ratings, set_draw_params
    from .calibration import calibrate_from_match_data
    from .attack_defense import compute_attack_defense
    from .form import compute_form
    from .simulator import simulate_tournament

    wc = WORLD_CUP_DATA[year]
    cutoff = pd.Timestamp(wc["start_date"])
    champion = wc["champion"]
    groups_backup = None

    # Sobreescribir GROUPS temporalmente con los grupos de ese Mundial
    import src.world_cup_2026 as wc2026
    groups_backup = wc2026.GROUPS.copy()
    wc2026.GROUPS = wc["groups"]

    try:
        # 1. Filtrar datos previos al torneo
        df_past = df_full[df_full["date"] < cutoff].copy()
        print(f"\n  Datos hasta {cutoff.date()}: {len(df_past):,} partidos")

        # 2. Calcular ELO
        print("  Calculando ELO histórico...")
        ratings, match_df = compute_elo_ratings(df_past, return_match_data=True)

        # 3. Calibrar probabilidad de empate con datos de ese momento
        print("  Calibrando probabilidad de empate...")
        a, b, c = calibrate_from_match_data(match_df)
        set_draw_params(a, b, c)

        # 4. Features adicionales (si se activan)
        attack_defense = None
        form_adj = None

        if use_attack_defense:
            attack_defense = compute_attack_defense(df_past)

        if use_form:
            form_adj = compute_form(df_past)

        # 5. Monte Carlo
        all_teams = [t for teams in wc["groups"].values() for t in teams]
        champion_counts: Dict[str, int] = {t: 0 for t in all_teams}

        for _ in tqdm(range(n_simulations), desc=f"  Simulando {year}", leave=False):
            champ = simulate_tournament(ratings, attack_defense, form_adj)
            if champ in champion_counts:
                champion_counts[champ] += 1

        probabilities = {
            team: count / n_simulations
            for team, count in champion_counts.items()
        }

        # 6. Métricas
        bs = brier_score(probabilities, champion)
        ll = log_loss_champion(probabilities, champion)

        sorted_probs = sorted(probabilities.items(), key=lambda x: -x[1])
        champion_rank = next(
            (i + 1 for i, (t, _) in enumerate(sorted_probs) if t == champion), -1
        )
        champion_prob = probabilities.get(champion, 0.0)

        return {
            "year": year,
            "name": wc["name"],
            "champion": champion,
            "brier_score": bs,
            "log_loss": ll,
            "champion_rank": champion_rank,
            "champion_prob_pct": champion_prob * 100,
            "probabilities": probabilities,
        }

    finally:
        # Restaurar grupos del 2026
        wc2026.GROUPS = groups_backup


def print_backtest_results(results: Dict) -> None:
    print(f"\n{'='*60}")
    print(f"  BACKTEST: {results['name']}")
    print(f"{'='*60}")
    print(f"  Campeón real:    {results['champion']}")
    print(f"  Prob. asignada:  {results['champion_prob_pct']:.2f}%")
    print(f"  Ranking del campeón: #{results['champion_rank']}")
    print(f"  Brier Score:     {results['brier_score']:.6f}")
    print(f"  Log-Loss:        {results['log_loss']:.4f}")
    print()
    print(f"  Top 8 predicciones:")
    sorted_probs = sorted(results["probabilities"].items(), key=lambda x: -x[1])
    for i, (team, prob) in enumerate(sorted_probs[:8], 1):
        marker = " ← CAMPEÓN REAL" if team == results["champion"] else ""
        print(f"    {i:>2}. {team:<22} {prob*100:>5.2f}%{marker}")


def run_all_backtests(
    df_full: pd.DataFrame,
    n_simulations: int = 50_000,
    use_attack_defense: bool = True,
    use_form: bool = True,
) -> pd.DataFrame:
    """Ejecuta backtests para todos los mundiales disponibles y retorna tabla resumen."""
    from .elo_calculator import get_draw_params, set_draw_params

    # Guardar params actuales para restaurar después
    original_params = get_draw_params()

    summary = []
    for year in [2018, 2022]:
        print(f"\n{'='*60}")
        print(f"  BACKTESTING {WORLD_CUP_DATA[year]['name'].upper()}")
        print(f"{'='*60}")
        try:
            res = run_backtest(
                year, df_full, n_simulations,
                use_attack_defense=use_attack_defense,
                use_form=use_form,
            )
            print_backtest_results(res)
            summary.append({
                "Mundial": res["name"],
                "Campeón real": res["champion"],
                "Prob. predicha": f"{res['champion_prob_pct']:.2f}%",
                "Ranking campeón": res["champion_rank"],
                "Brier Score": round(res["brier_score"], 6),
                "Log-Loss": round(res["log_loss"], 4),
            })
        except Exception as e:
            print(f"  ERROR en {year}: {e}")
            import traceback
            traceback.print_exc()

    # Restaurar params originales y naïve baseline
    set_draw_params(*original_params)

    df_summary = pd.DataFrame(summary)

    n_teams_avg = 32  # ambos mundiales tenían 32 equipos
    naive_brier = ((1 / n_teams_avg) * (1 - 1 / n_teams_avg) ** 2 +
                   (n_teams_avg - 1) * (1 / n_teams_avg) * (0 - 1 / n_teams_avg) ** 2)
    naive_ll = np.log(n_teams_avg)

    print(f"\n{'='*60}")
    print(f"  RESUMEN BACKTESTING")
    print(f"{'='*60}")
    print(df_summary.to_string(index=False))
    print(f"\n  Referencia naive (equiprobable 1/32): Brier={naive_brier:.6f}, Log-Loss={naive_ll:.4f}")

    return df_summary
