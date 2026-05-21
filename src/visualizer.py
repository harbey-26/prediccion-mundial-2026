"""
Visualizaciones para el modelo de predicción del Mundial 2026.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
import os

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def plot_championship_probabilities(results: pd.DataFrame, top_n: int = 20, save: bool = True):
    """Gráfico de barras con las probabilidades de ganar el Mundial."""
    top = results.head(top_n).copy()

    fig, ax = plt.subplots(figsize=(14, 8))
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top)))[::-1]

    bars = ax.barh(top["team"][::-1], top["probability_pct"][::-1], color=colors[::-1])

    for bar, pct in zip(bars, top["probability_pct"][::-1]):
        ax.text(
            bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
            f"{pct:.1f}%", va="center", fontsize=9
        )

    ax.set_xlabel("Probabilidad de ser Campeón (%)", fontsize=12)
    ax.set_title("Predicción: Probabilidad de Ganar el Mundial 2026\n(Simulación Monte Carlo - ELO Rating)", fontsize=14, fontweight="bold")
    ax.set_xlim(0, top["probability_pct"].max() * 1.15)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    if save:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        path = os.path.join(RESULTS_DIR, "championship_probabilities.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Gráfico guardado en: {path}")

    plt.show()


def plot_elo_top_teams(ratings: dict, top_n: int = 30, save: bool = True):
    """Gráfico de barras con los ELO ratings actuales de los mejores equipos."""
    sorted_ratings = sorted(ratings.items(), key=lambda x: x[1], reverse=True)[:top_n]
    teams, elos = zip(*sorted_ratings)

    fig, ax = plt.subplots(figsize=(14, 8))
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(teams)))[::-1]

    bars = ax.barh(list(teams)[::-1], list(elos)[::-1], color=colors[::-1])

    for bar, elo in zip(bars, list(elos)[::-1]):
        ax.text(
            bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
            f"{elo:.0f}", va="center", fontsize=8
        )

    ax.set_xlabel("Rating ELO", fontsize=12)
    ax.set_title(f"Top {top_n} Selecciones por Rating ELO Histórico", fontsize=14, fontweight="bold")
    ax.set_xlim(min(elos) - 100, max(elos) + 100)
    ax.axvline(x=1500, color="red", linestyle="--", alpha=0.5, label="ELO base (1500)")
    ax.legend()
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    if save:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        path = os.path.join(RESULTS_DIR, "elo_ratings.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Gráfico guardado en: {path}")

    plt.show()


def plot_group_stage_summary(group_results: dict, ratings: dict, save: bool = True):
    """Tabla visual con los resultados esperados por grupo."""
    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    axes = axes.flatten()

    for idx, (group_name, teams) in enumerate(group_results.items()):
        ax = axes[idx]
        ax.axis("off")
        ax.set_title(f"Grupo {group_name}", fontsize=12, fontweight="bold", pad=10)

        table_data = []
        for pos, team in enumerate(teams, 1):
            elo = ratings.get(team, 1500)
            medal = ["🥇", "🥈", "🥉", ""][pos - 1] if pos <= 3 else ""
            table_data.append([f"{pos}°", team, f"{elo:.0f}"])

        table = ax.table(
            cellText=table_data,
            colLabels=["Pos", "Equipo", "ELO"],
            loc="center",
            cellLoc="left",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)

        # Color top 2 (clasifican)
        for row in [1, 2]:
            for col in range(3):
                table[row, col].set_facecolor("#d4edda")
        # Color 3ro (posible mejor tercero)
        for col in range(3):
            table[3, col].set_facecolor("#fff3cd")

    green_patch = mpatches.Patch(color="#d4edda", label="Clasifican directamente")
    yellow_patch = mpatches.Patch(color="#fff3cd", label="Posible mejor tercero")
    fig.legend(handles=[green_patch, yellow_patch], loc="lower center", ncol=2, fontsize=10)
    fig.suptitle("Predicción Fase de Grupos - Mundial 2026", fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()

    if save:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        path = os.path.join(RESULTS_DIR, "group_stage_prediction.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Gráfico guardado en: {path}")

    plt.show()
