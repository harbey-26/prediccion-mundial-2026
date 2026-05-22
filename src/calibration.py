"""
Calibración empírica de la probabilidad de empate.

Proceso:
  1. Calcula ELO histórico guardando la diferencia ELO en el momento de cada partido
  2. Ajusta curva P(empate | |ELO_diff|) con scipy.optimize.curve_fit
  3. Guarda parámetros en data/processed/draw_calibration.json

Modelo:  P(draw) = a * exp(-b * |ELO_diff|) + c
  - a: amplitud del componente decreciente (mayor draw rate cuando equipos son iguales)
  - b: tasa de caída (qué tan rápido baja la prob. de empate al aumentar la diferencia)
  - c: piso asintótico (mínima prob. de empate incluso con diferencia enorme)
"""
import json
import os
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from typing import Optional, Tuple

CALIBRATION_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "processed", "draw_calibration.json"
)


def _draw_model(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    return a * np.exp(-b * x) + c


def calibrate_from_match_data(
    match_df: pd.DataFrame,
    min_bin_size: int = 150,
) -> Tuple[float, float, float]:
    """
    Ajusta la curva P(empate | |ELO_diff|) desde el histórico de partidos.

    match_df requiere columnas: elo_diff_abs, outcome ('win'/'draw'/'loss')
    Retorna (a, b, c) para _draw_model.
    """
    bins = np.arange(0, 700, 25)
    bin_labels = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)]

    data = match_df.copy()
    data["bin"] = pd.cut(data["elo_diff_abs"], bins=bins, labels=bin_labels)

    bin_centers = []
    draw_rates = []
    sample_weights = []

    for center, group in data.groupby("bin", observed=True):
        if len(group) >= min_bin_size:
            draw_rate = (group["outcome"] == "draw").mean()
            bin_centers.append(float(center))
            draw_rates.append(draw_rate)
            sample_weights.append(len(group))

    bin_centers = np.array(bin_centers)
    draw_rates = np.array(draw_rates)
    sample_weights = np.array(sample_weights, dtype=float)

    p0 = [0.18, 0.005, 0.07]
    bounds = ([0.0, 0.0001, 0.02], [0.45, 0.05, 0.30])

    popt, _ = curve_fit(
        _draw_model,
        bin_centers,
        draw_rates,
        p0=p0,
        bounds=bounds,
        sigma=1.0 / np.sqrt(sample_weights),
        absolute_sigma=False,
        maxfev=10_000,
    )

    a, b, c = float(popt[0]), float(popt[1]), float(popt[2])
    return a, b, c


def save_calibration(a: float, b: float, c: float) -> None:
    os.makedirs(os.path.dirname(CALIBRATION_PATH), exist_ok=True)
    with open(CALIBRATION_PATH, "w") as f:
        json.dump({"a": a, "b": b, "c": c}, f, indent=2)


def load_calibration() -> Optional[Tuple[float, float, float]]:
    if not os.path.exists(CALIBRATION_PATH):
        return None
    with open(CALIBRATION_PATH) as f:
        data = json.load(f)
    return data["a"], data["b"], data["c"]


def run_calibration(
    df: Optional[pd.DataFrame] = None,
    force: bool = False,
) -> Tuple[float, float, float]:
    """
    Pipeline completo de calibración.
    Carga datos, calcula ELO con historial de partidos, ajusta curva y guarda resultado.

    Args:
        df: DataFrame de resultados. Si None, lo carga desde data_loader.
        force: si True, ignora caché y recalibra.
    """
    if not force:
        cached = load_calibration()
        if cached:
            return cached

    if df is None:
        from .data_loader import load_results
        df = load_results()

    from .elo_calculator import compute_elo_ratings
    print("  Calculando ELO histórico para calibración...")
    _, match_df = compute_elo_ratings(df, return_match_data=True)
    print(f"  {len(match_df):,} partidos procesados.")

    print("  Ajustando curva de probabilidad de empate...")
    a, b, c = calibrate_from_match_data(match_df)

    save_calibration(a, b, c)
    return a, b, c


def print_calibration_summary(a: float, b: float, c: float) -> None:
    """Imprime tabla de probabilidades de empate para distintas diferencias ELO."""
    print(f"\n  Curva calibrada: P(draw) = {a:.4f} × exp(-{b:.5f} × |diff|) + {c:.4f}")
    print(f"  {'|ELO diff|':>10} {'P(empate)':>10}")
    print(f"  {'-'*22}")
    for diff in [0, 50, 100, 150, 200, 300, 400, 500]:
        p = float(a * np.exp(-b * diff) + c)
        p = max(0.05, min(p, 0.40))
        print(f"  {diff:>10}   {p*100:>7.1f}%")
