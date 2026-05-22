"""
Pipeline completo de predicción del Mundial 2026.

Uso:
  python main.py                         # ELO puro, 100,000 simulaciones
  python main.py --composite             # ELO + FIFA ranking combinados
  python main.py --compare               # ejecuta ambos modelos y compara
  python main.py --sims 50000            # número personalizado de simulaciones
  python main.py --elo-weight 0.7        # ajusta peso ELO vs FIFA (default 0.6)
  python main.py --match "Brazil" "Argentina"  # predice un partido específico
  python main.py --download              # forzar descarga del dataset
  python main.py --calibrate             # calibrar prob. de empate empíricamente
  python main.py --recalc                # recalcular ELO desde cero (borra caché)
"""
import argparse
import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader import download_dataset, load_results, get_processed_path
from src.elo_calculator import compute_elo_ratings, win_probability, set_draw_params
from src.calibration import load_calibration, run_calibration, print_calibration_summary
from src.fifa_ranking import load_fifa_rankings, composite_rating, get_fifa_points
from src.attack_defense import compute_attack_defense
from src.form import compute_form, compute_form_raw
from src.simulator import run_monte_carlo, simulate_group_stage, build_composite_ratings
from src.visualizer import plot_championship_probabilities, plot_elo_top_teams, plot_group_stage_summary
from src.ml_model import train_model, save_model, load_model


def build_elo_ratings(force: bool = False) -> dict:
    cache_path = get_processed_path("elo_ratings.csv")
    if os.path.exists(cache_path) and not force:
        print("Cargando ratings ELO desde caché (usa --recalc si cambiaste el modelo)...")
        df = pd.read_csv(cache_path)
        return dict(zip(df["team"], df["elo"]))

    print("Calculando ratings ELO históricos (v2: goal multiplier + time decay + K dinámico)...")
    df = load_results()
    ratings = compute_elo_ratings(df)
    pd.DataFrame([{"team": t, "elo": e} for t, e in ratings.items()]).to_csv(cache_path, index=False)
    print(f"ELO calculado para {len(ratings)} selecciones.")
    return ratings


def apply_calibration(args_calibrate: bool) -> None:
    """Carga o ejecuta la calibración de probabilidad de empate y la inyecta en el modelo."""
    print("\n=== CALIBRACIÓN DE PROBABILIDAD DE EMPATE ===")
    if args_calibrate:
        print("  Ejecutando calibración empírica (esto puede tardar ~1 min)...")
        df = load_results()
        a, b, c = run_calibration(df=df, force=True)
        print("  Calibración completada y guardada.")
    else:
        cal = load_calibration()
        if cal:
            a, b, c = cal
            print("  Parámetros cargados desde caché.")
        else:
            print("  Sin calibración guardada — usando parámetros por defecto.")
            print("  Tip: ejecuta con --calibrate para calibrar empíricamente.")
            return

    set_draw_params(a, b, c)
    print_calibration_summary(a, b, c)


def predict_match(team_a: str, team_b: str, elo_ratings: dict, fifa_rankings: dict):
    elo_a = elo_ratings.get(team_a, 1500)
    elo_b = elo_ratings.get(team_b, 1500)
    fifa_a = get_fifa_points(team_a, fifa_rankings)
    fifa_b = get_fifa_points(team_b, fifa_rankings)
    comp_a = composite_rating(team_a, elo_ratings, fifa_rankings)
    comp_b = composite_rating(team_b, elo_ratings, fifa_rankings)

    p_win_elo, p_draw_elo, p_loss_elo = win_probability(elo_a, elo_b)
    p_win_comp, p_draw_comp, p_loss_comp = win_probability(comp_a, comp_b)

    print(f"\n{'='*58}")
    print(f"  {team_a}  vs  {team_b}")
    print(f"{'='*58}")
    print(f"  {'Métrica':<18} {'':>12} {'':>12}")
    print(f"  {'':18} {team_a:>12} {team_b:>12}")
    print(f"  {'-'*44}")
    print(f"  {'ELO Rating':<18} {elo_a:>12.0f} {elo_b:>12.0f}")
    print(f"  {'FIFA Puntos':<18} {fifa_a:>12.0f} {fifa_b:>12.0f}")
    print(f"  {'Rating Compuesto':<18} {comp_a:>12.0f} {comp_b:>12.0f}")
    print(f"  {'='*44}")
    print(f"\n  Modelo ELO puro:")
    print(f"    {team_a} gana:  {p_win_elo*100:.1f}%")
    print(f"    Empate:         {p_draw_elo*100:.1f}%")
    print(f"    {team_b} gana:  {p_loss_elo*100:.1f}%")
    print(f"\n  Modelo ELO + FIFA:")
    print(f"    {team_a} gana:  {p_win_comp*100:.1f}%")
    print(f"    Empate:         {p_draw_comp*100:.1f}%")
    print(f"    {team_b} gana:  {p_loss_comp*100:.1f}%")
    print(f"{'='*58}\n")


def print_top_ratings(elo_ratings: dict, fifa_rankings: dict, n: int = 15):
    from src.world_cup_2026 import ALL_TEAMS
    print(f"\n{'='*72}")
    print(f"  {'#':>3}  {'Equipo':<25} {'ELO':>7} {'FIFA pts':>9} {'Compuesto':>10}")
    print(f"  {'-'*66}")
    wc_ratings = []
    for team in ALL_TEAMS:
        elo = elo_ratings.get(team, 1500)
        fifa_pts = get_fifa_points(team, fifa_rankings)
        comp = composite_rating(team, elo_ratings, fifa_rankings)
        wc_ratings.append((team, elo, fifa_pts, comp))

    wc_ratings.sort(key=lambda x: x[3], reverse=True)
    for i, (team, elo, fifa_pts, comp) in enumerate(wc_ratings[:n], 1):
        print(f"  {i:>3}. {team:<25} {elo:>7.0f} {fifa_pts:>9.0f} {comp:>10.0f}")
    print(f"{'='*72}")


def compare_models(elo_results: pd.DataFrame, comp_results: pd.DataFrame):
    merged = elo_results[["team", "probability_pct"]].rename(
        columns={"probability_pct": "elo_pct"}
    ).merge(
        comp_results[["team", "probability_pct"]].rename(
            columns={"probability_pct": "comp_pct"}
        ),
        on="team", how="outer"
    ).fillna(0).sort_values("comp_pct", ascending=False)

    merged["diff"] = (merged["comp_pct"] - merged["elo_pct"]).round(2)

    print(f"\n{'='*65}")
    print(f"  COMPARACIÓN: ELO puro vs ELO + FIFA (top 15)")
    print(f"  {'Equipo':<25} {'ELO solo':>9} {'ELO+FIFA':>9} {'Δ':>7}")
    print(f"  {'-'*55}")
    for _, row in merged.head(15).iterrows():
        arrow = "↑" if row["diff"] > 0.5 else ("↓" if row["diff"] < -0.5 else " ")
        print(f"  {row['team']:<25} {row['elo_pct']:>8.1f}% {row['comp_pct']:>8.1f}%  {arrow}{abs(row['diff']):.1f}%")
    print(f"{'='*65}")
    return merged


def main():
    parser = argparse.ArgumentParser(description="Predicción Mundial 2026 - Monte Carlo + ELO + FIFA")
    parser.add_argument("--sims", type=int, default=100_000)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--recalc", action="store_true")
    parser.add_argument("--composite", action="store_true", help="Usar ELO + FIFA ranking")
    parser.add_argument("--compare", action="store_true", help="Comparar ELO puro vs ELO+FIFA")
    parser.add_argument("--elo-weight", type=float, default=0.6, dest="elo_weight",
                        help="Peso del ELO en el rating compuesto (0.0–1.0)")
    parser.add_argument("--match", nargs=2, metavar=("EQUIPO_A", "EQUIPO_B"))
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--calibrate", action="store_true",
                        help="Calibrar probabilidad de empate desde datos históricos")
    parser.add_argument("--no-features", action="store_true",
                        help="Desactivar ataque/defensa y forma (modo básico)")
    parser.add_argument("--backtest", action="store_true",
                        help="Ejecutar backtesting sobre Mundiales 2018 y 2022")
    parser.add_argument("--backtest-sims", type=int, default=50_000,
                        help="Simulaciones para backtesting (default: 50,000)")
    parser.add_argument("--train-ml", action="store_true",
                        help="Entrenar modelo XGBoost y guardar en caché")
    parser.add_argument("--ml-weight", type=float, default=0.35, dest="ml_weight",
                        help="Peso del modelo ML en el ensemble (0.0–1.0, default: 0.35)")
    parser.add_argument("--no-ml", action="store_true",
                        help="Desactivar modelo ML aunque exista caché")
    args = parser.parse_args()

    # 1. Datos
    if args.download:
        download_dataset()
    else:
        try:
            download_dataset()
        except Exception as e:
            print(f"Advertencia: {e}")

    # 2. ELO ratings
    elo_ratings = build_elo_ratings(force=args.recalc)

    # 3. Calibración de probabilidad de empate
    apply_calibration(args.calibrate)

    # Backtesting mode (sale después de reportar)
    if args.backtest:
        from src.backtesting import run_all_backtests
        print("\nCargando datos para backtesting...")
        df_full = load_results()
        summary = run_all_backtests(
            df_full,
            n_simulations=args.backtest_sims,
            use_attack_defense=not args.no_features,
            use_form=not args.no_features,
        )
        summary.to_csv("results/csv/backtest_summary.csv", index=False)
        print(f"\nResultados guardados en results/csv/backtest_summary.csv")
        return

    # 4. FIFA rankings
    print("Cargando ranking FIFA...")
    fifa_rankings = load_fifa_rankings()
    print(f"  {len(fifa_rankings)} selecciones con puntos FIFA cargadas.")

    # 5. Features adicionales: ataque/defensa + forma reciente
    attack_defense = None
    form_adj = None
    form_raw = None
    df_full = None
    if not args.no_features:
        print("\nCalculando ratings de ataque/defensa (últimos 4 años)...")
        df_full = load_results()
        attack_defense = compute_attack_defense(df_full)
        print(f"  {len(attack_defense)} equipos con datos de ataque/defensa.")

        print("Calculando factor de forma reciente (últimos 10 partidos)...")
        form_adj = compute_form(df_full)
        form_raw = compute_form_raw(df_full)
        print(f"  {len(form_adj)} equipos con forma calculada.")

    # 6. Modelo ML ensemble
    ml_model = None
    if not args.no_ml:
        if args.train_ml:
            print("\n=== ENTRENANDO MODELO ML (XGBoost) ===")
            if df_full is None:
                df_full = load_results()
            ml_model = train_model(df_full)
            save_model(ml_model)
        else:
            ml_model = load_model()
            if ml_model is not None:
                print(f"Modelo ML cargado desde caché (peso={args.ml_weight:.0%}).")
            else:
                print("Sin modelo ML en caché. Usa --train-ml para entrenar.")

    # 7. Partido específico
    if args.match:
        predict_match(args.match[0], args.match[1], elo_ratings, fifa_rankings)
        return

    # 8. Tabla comparativa de ratings para los 48 equipos
    print_top_ratings(elo_ratings, fifa_rankings, n=20)

    # 9. Fase de grupos (muestra)
    use_composite = args.composite or args.compare
    effective_ratings = (
        build_composite_ratings(elo_ratings, args.elo_weight)
        if use_composite else elo_ratings
    )
    group_sample, _ = simulate_group_stage(effective_ratings, attack_defense, form_adj)
    print("\n=== PREDICCIÓN FASE DE GRUPOS ===")
    for group, teams in group_sample.items():
        elo_str = " | ".join([f"{t} ({effective_ratings.get(t, 1500):.0f})" for t in teams])
        print(f"  Grupo {group}: {elo_str}")

    # 10. Monte Carlo
    os.makedirs("results/csv", exist_ok=True)
    os.makedirs("results/excel", exist_ok=True)

    if args.compare:
        print(f"\n=== MODELO ELO PURO ({args.sims:,} simulaciones) ===")
        elo_results = run_monte_carlo(
            elo_ratings, args.sims, use_composite=False,
            attack_defense=attack_defense, form_adj=form_adj,
        )
        elo_results.to_csv("results/csv/predicciones_elo.csv", index=False)

        label_comp = "ELO + FIFA + Features" + (" + ML" if ml_model else "")
        print(f"\n=== MODELO {label_comp} ({args.sims:,} simulaciones) ===")
        comp_results = run_monte_carlo(
            elo_ratings, args.sims, use_composite=True, elo_weight=args.elo_weight,
            attack_defense=attack_defense, form_adj=form_adj,
            ml_model=ml_model, ml_weight=args.ml_weight, form_raw=form_raw,
        )
        comp_results.to_csv("results/csv/predicciones_compuesto.csv", index=False)

        merged = compare_models(elo_results, comp_results)
        merged.to_csv("results/csv/comparacion_modelos.csv", index=False)
        mc_results = comp_results

    else:
        label = "ELO + FIFA + Features" if args.composite else "ELO puro"
        if ml_model and args.composite:
            label += " + ML"
        print(f"\n=== SIMULACIÓN MONTE CARLO — {label} ({args.sims:,} torneos) ===")
        mc_results = run_monte_carlo(
            elo_ratings, args.sims,
            use_composite=args.composite,
            elo_weight=args.elo_weight,
            attack_defense=attack_defense,
            form_adj=form_adj,
            ml_model=ml_model if args.composite else None,
            ml_weight=args.ml_weight,
            form_raw=form_raw,
        )
        mc_results.to_csv("results/csv/predicciones_campeon.csv", index=False)

    print(f"\nResultados guardados en results/csv/")
    print("\n=== TOP 10 CANDIDATOS AL TÍTULO ===")
    for _, row in mc_results.head(10).iterrows():
        bar = "█" * int(row["probability_pct"] * 2)
        print(f"  {int(row['rank']):>2}. {row['team']:<25} {row['probability_pct']:>5.1f}%  {bar}")

    # 11. Visualizaciones
    if not args.no_plot:
        plot_elo_top_teams(effective_ratings, top_n=25)
        plot_championship_probabilities(mc_results, top_n=20)
        plot_group_stage_summary(group_sample, effective_ratings)


if __name__ == "__main__":
    main()
