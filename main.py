"""
Pipeline completo de predicción del Mundial 2026.

Uso:
  python main.py                    # 100,000 simulaciones (por defecto)
  python main.py --sims 10000       # número personalizado de simulaciones
  python main.py --download         # forzar descarga del dataset
  python main.py --match "Brasil" "Argentina"  # predice un partido específico
"""
import argparse
import sys
import os
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader import download_dataset, load_results, get_processed_path
from src.elo_calculator import compute_elo_ratings, win_probability
from src.simulator import run_monte_carlo, simulate_group_stage
from src.visualizer import plot_championship_probabilities, plot_elo_top_teams, plot_group_stage_summary


def build_elo_ratings(force: bool = False) -> dict:
    """Carga o recalcula los ratings ELO."""
    cache_path = get_processed_path("elo_ratings.csv")

    if os.path.exists(cache_path) and not force:
        print("Cargando ratings ELO desde caché...")
        df = pd.read_csv(cache_path)
        return dict(zip(df["team"], df["elo"]))

    print("Calculando ratings ELO históricos...")
    df = load_results()
    ratings = compute_elo_ratings(df)

    # Guardar caché
    pd.DataFrame([{"team": t, "elo": e} for t, e in ratings.items()]).to_csv(cache_path, index=False)
    print(f"ELO calculado para {len(ratings)} selecciones.")
    return ratings


def predict_match(team_a: str, team_b: str, ratings: dict):
    """Imprime la predicción para un partido específico."""
    elo_a = ratings.get(team_a, 1500)
    elo_b = ratings.get(team_b, 1500)

    p_win, p_draw, p_loss = win_probability(elo_a, elo_b, neutral=True)

    print(f"\n{'='*50}")
    print(f"  {team_a} vs {team_b}")
    print(f"{'='*50}")
    print(f"  ELO {team_a}: {elo_a:.0f}")
    print(f"  ELO {team_b}: {elo_b:.0f}")
    print(f"  Probabilidad {team_a} gana: {p_win*100:.1f}%")
    print(f"  Probabilidad empate:         {p_draw*100:.1f}%")
    print(f"  Probabilidad {team_b} gana: {p_loss*100:.1f}%")
    print(f"{'='*50}\n")


def main():
    parser = argparse.ArgumentParser(description="Predicción Mundial 2026 - Monte Carlo + ELO")
    parser.add_argument("--sims", type=int, default=100_000, help="Número de simulaciones")
    parser.add_argument("--download", action="store_true", help="Forzar descarga del dataset")
    parser.add_argument("--recalc", action="store_true", help="Recalcular ELO desde cero")
    parser.add_argument("--match", nargs=2, metavar=("EQUIPO_A", "EQUIPO_B"), help="Predice un partido específico")
    parser.add_argument("--no-plot", action="store_true", help="No mostrar gráficos")
    args = parser.parse_args()

    # 1. Descargar datos si es necesario
    if args.download:
        download_dataset()
    else:
        try:
            download_dataset()
        except Exception as e:
            print(f"Advertencia: {e}")

    # 2. Calcular ELO ratings
    ratings = build_elo_ratings(force=args.recalc)

    # 3. Predicción de partido específico
    if args.match:
        predict_match(args.match[0], args.match[1], ratings)
        return

    # 4. Mostrar top ELO
    print("\n=== TOP 15 SELECCIONES POR ELO ===")
    sorted_ratings = sorted(ratings.items(), key=lambda x: x[1], reverse=True)
    for i, (team, elo) in enumerate(sorted_ratings[:15], 1):
        print(f"  {i:>2}. {team:<25} ELO: {elo:.0f}")

    # 5. Simular fase de grupos (una vez para referencia)
    print("\n=== PREDICCIÓN FASE DE GRUPOS (muestra única) ===")
    group_sample = simulate_group_stage(ratings)
    for group, teams in group_sample.items():
        elo_str = " | ".join([f"{t} ({ratings.get(t, 1500):.0f})" for t in teams])
        print(f"  Grupo {group}: {elo_str}")

    # 6. Monte Carlo
    print(f"\n=== SIMULACIÓN MONTE CARLO ({args.sims:,} torneos) ===")
    mc_results = run_monte_carlo(ratings, n_simulations=args.sims)

    # 7. Guardar resultados
    os.makedirs("results", exist_ok=True)
    results_path = os.path.join(os.path.dirname(__file__), "results", "predicciones_campeon.csv")
    mc_results.to_csv(results_path, index=False)
    print(f"\nResultados guardados en: {results_path}")

    # 8. Imprimir top 10
    print("\n=== TOP 10 CANDIDATOS AL TÍTULO ===")
    for _, row in mc_results.head(10).iterrows():
        bar = "█" * int(row["probability_pct"] * 2)
        print(f"  {int(row['rank']):>2}. {row['team']:<25} {row['probability_pct']:>5.1f}%  {bar}")

    # 9. Visualizaciones
    if not args.no_plot:
        plot_elo_top_teams(ratings, top_n=25)
        plot_championship_probabilities(mc_results, top_n=20)
        plot_group_stage_summary(group_sample, ratings)


if __name__ == "__main__":
    main()
