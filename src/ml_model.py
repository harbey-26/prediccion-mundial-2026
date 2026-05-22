"""
Modelo ML ensemble (XGBoost) para predicción de resultados de partidos.

Entrena un clasificador XGBoost sobre el historial completo de partidos
con features calculadas de forma secuencial (sin data leakage).

Features por partido:
  elo_diff_adj    — ELO local + ventaja local − ELO visitante
  elo_home/away   — ELO absoluto en el momento del partido
  form_home/away  — win-rate ponderado últimos 10 partidos (escala 0–3)
  form_diff       — form_home − form_away
  attack/defense  — ratios vs media global (escala ~1.0)
  goal_ratio      — (atk_H/def_A) / (atk_A/def_H)
  home_advantage  — 0/1
  tournament_weight
  gp_home/away_log — log(1 + partidos_disputados)

Target: 0 = visitante gana, 1 = empate, 2 = local gana
"""
import os
import pickle
import numpy as np
import pandas as pd
from collections import deque
from typing import Dict, List, Optional, Tuple

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "processed", "ml_model.pkl"
)

FEATURE_COLS: List[str] = [
    "elo_diff_adj",
    "elo_home",
    "elo_away",
    "form_home",
    "form_away",
    "form_diff",
    "attack_home",
    "defense_home",
    "attack_away",
    "defense_away",
    "goal_ratio",
    "home_advantage",
    "tournament_weight",
    "gp_home_log",
    "gp_away_log",
]

_INITIAL_ELO = 1500.0
_K_BASE = 20.0
_DECAY_LAMBDA = 0.05
_EMA_ALPHA = 0.15   # velocidad de actualización attack/defense por partido
_GLOBAL_AVG = 1.35  # goles promedio global por equipo por partido


def _goal_mult(gd: int) -> float:
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return (11 + gd) / 8.0


def _uncertainty_mult(gp: int) -> float:
    if gp < 30:
        return 1.5
    if gp < 80:
        return 1.2
    return 1.0


def _raw_form(results: deque) -> float:
    """Weighted average of last N results (3=win, 1=draw, 0=loss)."""
    if not results:
        return 1.0
    total_pts = total_w = 0.0
    for i, pts in enumerate(results):
        w = np.exp(-0.15 * i)
        total_pts += pts * w
        total_w += w
    return float(total_pts / total_w) if total_w > 0 else 1.0


def build_feature_matrix(df: pd.DataFrame, min_year: int = 1980) -> pd.DataFrame:
    """
    Pasa secuencialmente por todos los partidos y construye la matriz de features.
    Solo registra partidos desde min_year (datos anteriores sirven para warm-up).
    """
    df = df.sort_values("date").reset_index(drop=True)
    end_date = df["date"].max()

    ratings: Dict[str, float] = {}
    games_played: Dict[str, int] = {}
    form_deques: Dict[str, deque] = {}
    ema_scored: Dict[str, float] = {}
    ema_conceded: Dict[str, float] = {}

    records = []

    for _, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        hs = int(row["home_score"])
        aws = int(row["away_score"])
        weight = float(row.get("weight", 1.0))
        neutral = bool(row.get("neutral", False))
        date = row["date"]

        for t in (home, away):
            if t not in ratings:
                ratings[t] = _INITIAL_ELO
                games_played[t] = 0
                form_deques[t] = deque(maxlen=10)
                ema_scored[t] = _GLOBAL_AVG
                ema_conceded[t] = _GLOBAL_AVG

        r_h = ratings[home]
        r_a = ratings[away]
        gp_h = games_played[home]
        gp_a = games_played[away]
        home_adv_pts = 0 if neutral else 30
        r_h_adj = r_h + home_adv_pts

        form_h = _raw_form(form_deques[home])
        form_a = _raw_form(form_deques[away])

        atk_h = ema_scored[home] / _GLOBAL_AVG
        def_h = ema_conceded[home] / _GLOBAL_AVG
        atk_a = ema_scored[away] / _GLOBAL_AVG
        def_a = ema_conceded[away] / _GLOBAL_AVG
        expected_h = atk_h / max(def_a, 0.1)
        expected_a = atk_a / max(def_h, 0.1)
        goal_ratio = expected_h / max(expected_a, 0.01)

        if date.year >= min_year:
            records.append({
                "date": date,
                "home_team": home,
                "away_team": away,
                "elo_diff_adj": r_h_adj - r_a,
                "elo_home": r_h,
                "elo_away": r_a,
                "form_home": form_h,
                "form_away": form_a,
                "form_diff": form_h - form_a,
                "attack_home": atk_h,
                "defense_home": def_h,
                "attack_away": atk_a,
                "defense_away": def_a,
                "goal_ratio": float(np.clip(goal_ratio, 0.1, 10.0)),
                "home_advantage": 0 if neutral else 1,
                "tournament_weight": weight,
                "gp_home_log": float(np.log1p(gp_h)),
                "gp_away_log": float(np.log1p(gp_a)),
                "outcome": 2 if hs > aws else (1 if hs == aws else 0),
            })

        # Actualizar estado post-partido
        gd = abs(hs - aws)
        gm = _goal_mult(gd)
        years_ago = (end_date - date).days / 365.25
        tm = np.exp(-_DECAY_LAMBDA * years_ago)
        e_h = 1 / (1 + 10 ** ((r_a - r_h_adj) / 400))
        s_h = 1.0 if hs > aws else (0.5 if hs == aws else 0.0)

        base_k = _K_BASE * weight * gm * tm
        ratings[home] = r_h + base_k * _uncertainty_mult(gp_h) * (s_h - e_h)
        ratings[away] = r_a + base_k * _uncertainty_mult(gp_a) * ((1 - s_h) - (1 - e_h))
        games_played[home] = gp_h + 1
        games_played[away] = gp_a + 1

        pts_h = 3.0 if hs > aws else (1.0 if hs == aws else 0.0)
        pts_a = 3.0 if aws > hs else (1.0 if hs == aws else 0.0)
        form_deques[home].appendleft(pts_h)
        form_deques[away].appendleft(pts_a)

        ema_scored[home] = (1 - _EMA_ALPHA) * ema_scored[home] + _EMA_ALPHA * hs
        ema_conceded[home] = (1 - _EMA_ALPHA) * ema_conceded[home] + _EMA_ALPHA * aws
        ema_scored[away] = (1 - _EMA_ALPHA) * ema_scored[away] + _EMA_ALPHA * aws
        ema_conceded[away] = (1 - _EMA_ALPHA) * ema_conceded[away] + _EMA_ALPHA * hs

    return pd.DataFrame(records)


def train_model(df: pd.DataFrame, val_start: str = "2021-01-01") -> object:
    """
    Entrena XGBClassifier con split temporal:
      - Train: partidos anteriores a val_start
      - Val:   partidos desde val_start (para early stopping y reporte)

    Target: 0=visitante gana, 1=empate, 2=local gana
    """
    try:
        from xgboost import XGBClassifier
    except ImportError:
        raise ImportError("xgboost no está instalado. Ejecuta: pip install xgboost")
    from sklearn.metrics import accuracy_score, log_loss

    print("  Construyendo matriz de features (pase secuencial)...")
    feat_df = build_feature_matrix(df)
    print(f"  {len(feat_df):,} partidos en la matriz (desde 1980).")

    cutoff = pd.Timestamp(val_start)
    train = feat_df[feat_df["date"] < cutoff]
    val = feat_df[feat_df["date"] >= cutoff]

    print(f"  Train: {len(train):,} partidos | Val: {len(val):,} partidos")

    X_train = train[FEATURE_COLS].values.astype(np.float32)
    y_train = train["outcome"].values
    X_val = val[FEATURE_COLS].values.astype(np.float32)
    y_val = val["outcome"].values

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=600,
        max_depth=5,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.75,
        min_child_weight=15,
        gamma=0.05,
        reg_alpha=0.1,
        reg_lambda=1.0,
        n_jobs=-1,
        random_state=42,
        early_stopping_rounds=50,
        eval_metric="mlogloss",
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)

    acc = accuracy_score(y_val, y_pred)
    ll = log_loss(y_val, y_proba)

    # Baseline: distribución uniforme
    uniform_proba = np.full((len(y_val), 3), 1 / 3)
    uniform_ll = log_loss(y_val, uniform_proba)

    print(f"  Accuracy val: {acc:.3f}  |  Log-loss: {ll:.4f} (baseline uniforme: {uniform_ll:.4f})")
    print(f"  Mejora vs uniforme: {(1 - ll / uniform_ll) * 100:.1f}%")

    # Distribución real de outcomes en validación
    from collections import Counter
    dist = Counter(y_val)
    total = len(y_val)
    print(
        f"  Distribución val — Away: {dist[0]/total:.1%}  Draw: {dist[1]/total:.1%}  Home: {dist[2]/total:.1%}"
    )

    return model


def predict_match_ml(
    team_a: str,
    team_b: str,
    elo_ratings: Dict[str, float],
    form_raw: Dict[str, float],
    attack_defense: Optional[Dict[str, Dict[str, float]]],
    model,
    neutral: bool = True,
) -> Tuple[float, float, float]:
    """
    Predice probabilidades para team_a (local) vs team_b (visitante).
    Retorna (p_win_a, p_draw, p_loss_a).
    """
    r_a = elo_ratings.get(team_a, _INITIAL_ELO)
    r_b = elo_ratings.get(team_b, _INITIAL_ELO)
    home_adv = 0 if neutral else 30

    form_a = form_raw.get(team_a, 1.0)
    form_b = form_raw.get(team_b, 1.0)

    if attack_defense:
        atk_a = attack_defense.get(team_a, {}).get("attack", 1.0)
        def_a = attack_defense.get(team_a, {}).get("defense", 1.0)
        atk_b = attack_defense.get(team_b, {}).get("attack", 1.0)
        def_b = attack_defense.get(team_b, {}).get("defense", 1.0)
    else:
        atk_a = def_a = atk_b = def_b = 1.0

    exp_a = atk_a / max(def_b, 0.1)
    exp_b = atk_b / max(def_a, 0.1)
    goal_ratio = float(np.clip(exp_a / max(exp_b, 0.01), 0.1, 10.0))

    features = np.array([[
        r_a + home_adv - r_b,
        r_a,
        r_b,
        form_a,
        form_b,
        form_a - form_b,
        atk_a,
        def_a,
        atk_b,
        def_b,
        goal_ratio,
        0 if neutral else 1,
        2.0,             # World Cup
        float(np.log1p(80)),   # ~80 WC partidos para las selecciones calificadas
        float(np.log1p(80)),
    ]], dtype=np.float32)

    proba = model.predict_proba(features)[0]
    # proba[0]=p_away_win, proba[1]=p_draw, proba[2]=p_home_win
    return float(proba[2]), float(proba[1]), float(proba[0])


def precompute_ml_probs(
    teams: List[str],
    elo_ratings: Dict[str, float],
    form_raw: Dict[str, float],
    attack_defense: Optional[Dict[str, Dict[str, float]]],
    model,
    neutral: bool = True,
) -> Dict[Tuple[str, str], Tuple[float, float, float]]:
    """
    Pre-calcula probabilidades ML para todos los pares posibles de equipos.
    Devuelve {(team_a, team_b): (p_win, p_draw, p_loss)}.

    Reduce la inferencia de O(n_sims × n_matches) a O(n_teams²) llamadas al modelo.
    """
    GLOBAL_AVG = _GLOBAL_AVG
    home_adv = 0 if neutral else 30

    # Build feature rows for all ordered pairs
    pair_list: List[Tuple[str, str]] = []
    rows: List[List[float]] = []

    for team_a in teams:
        for team_b in teams:
            if team_a == team_b:
                continue
            r_a = elo_ratings.get(team_a, _INITIAL_ELO)
            r_b = elo_ratings.get(team_b, _INITIAL_ELO)
            form_a = form_raw.get(team_a, 1.0)
            form_b = form_raw.get(team_b, 1.0)

            if attack_defense:
                atk_a = attack_defense.get(team_a, {}).get("attack", 1.0)
                def_a = attack_defense.get(team_a, {}).get("defense", 1.0)
                atk_b = attack_defense.get(team_b, {}).get("attack", 1.0)
                def_b = attack_defense.get(team_b, {}).get("defense", 1.0)
            else:
                atk_a = def_a = atk_b = def_b = 1.0

            exp_a = atk_a / max(def_b, 0.1)
            exp_b = atk_b / max(def_a, 0.1)
            goal_ratio = float(np.clip(exp_a / max(exp_b, 0.01), 0.1, 10.0))

            rows.append([
                r_a + home_adv - r_b,
                r_a, r_b,
                form_a, form_b, form_a - form_b,
                atk_a, def_a, atk_b, def_b,
                goal_ratio,
                0 if neutral else 1,
                2.0,
                float(np.log1p(80)),
                float(np.log1p(80)),
            ])
            pair_list.append((team_a, team_b))

    X = np.array(rows, dtype=np.float32)
    probas = model.predict_proba(X)  # shape: (n_pairs, 3)

    cache: Dict[Tuple[str, str], Tuple[float, float, float]] = {}
    for (ta, tb), proba in zip(pair_list, probas):
        # proba[0]=away(tb) win, proba[1]=draw, proba[2]=home(ta) win
        cache[(ta, tb)] = (float(proba[2]), float(proba[1]), float(proba[0]))

    return cache


def save_model(model) -> None:
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"  Modelo guardado en {MODEL_PATH}")


def load_model() -> Optional[object]:
    if not os.path.exists(MODEL_PATH):
        return None
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)
