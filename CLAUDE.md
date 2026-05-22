# CLAUDE.md — Contexto del Proyecto

## ¿Qué es este proyecto?

Modelo de predicción del Mundial FIFA 2026 usando **ELO histórico v2 + Ranking FIFA + Ataque/Defensa + Forma Reciente + ML Ensemble (XGBoost)** con **simulación Monte Carlo** (100,000 torneos). Predice el ganador de cada partido, la clasificación por grupos y el campeón.

---

## Estado actual del proyecto

- ✅ Dataset descargado: 49,257 partidos internacionales (1872–2026)
- ✅ ELO v2 calculado para 336 selecciones (cacheado en `data/processed/elo_ratings.csv`)
- ✅ Calibración de empate: `a=0.2389, b=0.00158, c=0.0200` (cacheada en `data/processed/draw_calibration.json`)
- ✅ Ranking FIFA scrapeado: 200 equipos (abril 2026, en `data/raw/fifa_ranking_2026.json`)
- ✅ Ataque/Defensa: 229 equipos con ratings Dixon-Coles (últimos 4 años)
- ✅ Forma reciente: 336 equipos con ajuste ELO ±80 pts (últimos 10 partidos)
- ✅ Modelo ML (XGBoost): entrenado sobre 37,165 partidos, cacheado en `data/processed/ml_model.pkl`
  - Accuracy validación (2021+): **60.6%** | Mejora log-loss vs baseline: **20.6%**
- ✅ Simulaciones v3: 100,000 torneos × 2 modelos (ELO puro + Ensemble completo)
- ✅ Excel generado: `results/excel/Mundial_2026_Predicciones.xlsx` (6 hojas)
- ✅ Google Sheets publicados en Drive del usuario (harbey.26@gmail.com)
- ✅ Repositorio en GitHub: https://github.com/harbey-26/prediccion-mundial-2026

---

## Entorno

- Python 3 (sistema, sin virtualenv)
- Directorio: `/Users/familiaperdomobocachica2/Documents/Proyectos_APP/prediccion-mundial-2026`
- Dependencias: pandas, numpy, scipy, scikit-learn, xgboost, matplotlib, seaborn, openpyxl, beautifulsoup4, lxml, tqdm

---

## Comandos clave

```bash
# Simulación completa recomendada (ELO + FIFA + A/D + Forma + ML)
python3 main.py --composite --compare --sims 100000 --no-plot

# Entrenar/re-entrenar el modelo ML y correr simulaciones
python3 main.py --train-ml --composite --compare --sims 100000 --no-plot

# Solo modelo ELO + FIFA + Features (sin ML, más rápido)
python3 main.py --composite --compare --no-ml --sims 100000 --no-plot

# Predecir un partido específico
python3 main.py --match "Brazil" "Argentina"

# Regenerar el Excel (6 hojas)
python3 generate_excel.py

# Recalcular ELO desde cero (si hay nuevos datos)
python3 main.py --recalc --calibrate --composite --compare --sims 100000
```

---

## Arquitectura del modelo v3

### 1. ELO Histórico v2 (`src/elo_calculator.py`)
- Rating inicial: 1500 para todos
- K-factor dinámico: 20 × peso_torneo × goal_multiplier × time_decay × uncertainty_mult
- Pesos por torneo: Mundial FIFA (2.0), Eurocopa (1.8), Copa América/Africana (1.5), Clasif. (1.3), Amistoso (0.5)
- Goal multiplier: 1-0 → ×1.0 | 2-0 → ×1.5 | 3-0+ → ×(11+gd)/8
- Time decay: exp(-0.05 × años_atrás) — partidos de hace 15 años valen ~45% menos
- Uncertainty multiplier: <30 partidos → ×1.5 | 30–80 → ×1.2 | 80+ → ×1.0
- Ventaja de local: +30 puntos ELO (no aplica en campo neutral)

### 2. Ranking FIFA (`src/fifa_ranking.py`)
- Scrapeado de Transfermarkt (200 equipos, paginación de 8 páginas)
- Puntos FIFA escala: ~1400–2000 (similar a ELO desde reforma 2018)
- Mapeo de nombres: "USA" → "United States", "Turkiye" → "Turkey", etc.

### 3. Rating Compuesto
```python
composite = 0.6 * elo + 0.4 * fifa_points  # ajustable con --elo-weight
```

### 4. Probabilidad de empate (`src/calibration.py`)
Calibrada empíricamente con `scipy.optimize.curve_fit` sobre el historial completo:
```python
P(draw) = 0.2389 * exp(-0.00158 * |elo_diff|) + 0.0200
# Clamped to [0.05, 0.40]
```

### 5. Ataque/Defensa (`src/attack_defense.py`)
- Dixon-Coles simplificado sobre los últimos 4 años (pesos exp λ=0.4/año)
- `attack > 1.0` = ataque fuerte | `defense < 1.0` = defensa sólida
- Goles esperados: `λ_A = 1.35 × attack_A × defense_B`
- Usado en fase de grupos para marcadores Poisson (no afecta probabilidad de resultado)

### 6. Forma Reciente (`src/form.py`)
- Ajuste ELO ±80 pts basado en últimos 10 partidos (pesos exponenciales λ=0.15)
- Normalización solo sobre equipos con ≥6 partidos en los últimos 2 años
- Equipos en mejor forma actual: Alemania, Francia, Argentina (~+80 pts)

### 7. Modelo ML Ensemble (`src/ml_model.py`) ← NUEVO en v3
- XGBClassifier multiclase (0=visitante gana, 1=empate, 2=local gana)
- 15 features: elo_diff_adj, elo_home/away, form_home/away, form_diff, attack/defense x4, goal_ratio, home_advantage, tournament_weight, gp_log x2
- Train: 31,629 partidos (1980–2020) | Val: 5,536 partidos (2021–2026)
- Accuracy val: **60.6%** | Log-loss: 0.8727 (baseline uniforme: 1.0986 → **mejora 20.6%**)
- Pre-calcula 2,256 pares antes de simular → mantiene ~560 torneos/segundo
- Blend: `p_final = 0.65 × p_elo + 0.35 × p_ml`

### 8. Simulación Monte Carlo (`src/simulator.py`)
- Fase de grupos: round-robin, resultado vía ELO+ML blend, goles vía Poisson Dixon-Coles
- Clasifican: top-2 de cada grupo + 8 mejores terceros = 32 equipos
- Criterio terceros: puntos → GD → GF → ELO (regla FIFA oficial)
- Fase eliminatoria: R32 → R16 → QF → SF → Final (sin empate)

---

## Grupos del Mundial 2026 (`src/world_cup_2026.py`)

| Grupo | Equipos |
|-------|---------|
| A | Mexico, South Korea, Czech Republic, South Africa |
| B | Canada, Bosnia and Herzegovina, Qatar, Switzerland |
| C | United States, Paraguay, Australia, Turkey |
| D | Brazil, Morocco, Haiti, Scotland |
| E | Germany, Ivory Coast, Ecuador, Curaçao |
| F | Netherlands, Japan, Sweden, Tunisia |
| G | Belgium, Iran, Egypt, New Zealand |
| H | Spain, Saudi Arabia, Uruguay, Cape Verde |
| I | France, Senegal, Norway, Iraq |
| J | Argentina, Algeria, Austria, Jordan |
| K | Portugal, Colombia, DR Congo, Uzbekistan |
| L | England, Croatia, Ghana, Panama |

---

## Resultados v3 — Modelo Ensemble (100,000 torneos)

### Top 15 — ELO puro vs Ensemble completo (ELO + FIFA + A/D + Forma + ML)

| # | Selección | ELO puro | **Ensemble** | Δ |
|---|-----------|:--------:|:------------:|:-:|
| 1 | España | 22.9% | **23.2%** | ↑0.3% |
| 2 | Argentina | 11.7% | **14.7%** | ↑3.0% |
| 3 | Francia | 11.0% | **14.2%** | ↑3.2% |
| 4 | Marruecos | 8.5% | **8.5%** | →0.0% |
| 5 | Inglaterra | 7.0% | **7.4%** | ↑0.4% |
| 6 | Japón | 7.5% | **4.3%** | ↓3.2% |
| 7 | Países Bajos | 3.1% | **4.3%** | ↑1.2% |
| 8 | Alemania | 3.3% | **3.1%** | ↓0.2% |
| 9 | Brasil | 2.0% | **3.0%** | ↑1.0% |
| 10 | Portugal | 1.9% | **2.4%** | ↑0.5% |
| 11 | Senegal | 2.3% | **1.9%** | ↓0.4% |
| 12 | Croacia | 1.8% | **1.7%** | ↓0.1% |
| 13 | Bélgica | 1.1% | **1.7%** | ↑0.6% |
| 14 | México | 1.1% | **1.5%** | ↑0.4% |
| 15 | Australia | 2.6% | **1.1%** | ↓1.5% |

> El ML penaliza a Japón (-3.2%) y Australia (-1.5%) que tienen buen ELO reciente pero historial débil en eliminatorias de alto nivel. Favorece a Argentina (+3.0%) y Francia (+3.2%) por su rendimiento histórico ajustado por calidad de rival.

### Bracket proyectado (ganador más probable en cada partido)
- **Final:** España vs Argentina → **España**
- **Semifinal 1:** Francia vs España → España
- **Semifinal 2:** Argentina vs Marruecos → Argentina

---

## Archivos de resultados

```
results/
├── csv/
│   ├── predicciones_elo.csv          # Modelo ELO puro (100k sims)
│   ├── predicciones_compuesto.csv    # Modelo Ensemble v3 ← usar este
│   └── comparacion_modelos.csv       # Diferencias entre modelos
└── excel/
    └── Mundial_2026_Predicciones.xlsx  # 6 hojas (ELO + FIFA + ML)

data/processed/
├── elo_ratings.csv       # ELO v2 de 336 selecciones
├── draw_calibration.json # Parámetros P(empate) calibrados
└── ml_model.pkl          # XGBoost entrenado (37k partidos, 60.6% acc)
```

---

## Google Sheets en Drive (harbey.26@gmail.com)

| Archivo | Link |
|---------|------|
| Partidos y Proyecciones (72 partidos) | https://docs.google.com/spreadsheets/d/11QDCdK48WzA5-FFGN6qU6ELMrMH7zcAsWj7CBJCn16g/edit |
| Probabilidades de Campeonato (48 equipos) | https://docs.google.com/spreadsheets/d/1kmZyOijsDKZi9EwnPrBLlNAC0cNhwJhEtVLs-6pXCT0/edit |
| Bracket Eliminatorio Proyectado | https://docs.google.com/spreadsheets/d/1yjlHZ95mK7r107TB1mZyn4Z6SBjbli6A8D9316P61CQ/edit |

> Nota: los Google Sheets reflejan resultados de una versión anterior. El Excel local tiene los resultados v3 actualizados.

---

## Ideas para continuar

- [ ] **Actualizar con resultados reales**: a medida que se jueguen partidos, alimentar los resultados al modelo y recalcular ELO en tiempo real
- [ ] **Agregar ranking FIFA actualizado**: el ranking se actualiza mensualmente, re-scrapear antes de cada fecha
- [ ] **Feature H2H**: estadísticas de enfrentamientos directos entre selecciones
- [ ] **Feature xG**: usar expected goals en lugar de goles reales para el ELO
- [ ] **Dashboard web**: exponer predicciones en app React
- [ ] **Actualizar grupos reales**: si el draw oficial cambia, editar `src/world_cup_2026.py`
- [ ] **Calibrar ml_weight**: optimizar el peso del ML ensemble (actualmente 0.35) vía backtesting

---

## Fuentes de datos

- **Partidos históricos**: https://github.com/martj42/international_results (se auto-descarga)
- **Ranking FIFA**: https://www.transfermarkt.com/statistik/weltrangliste (scraping con BeautifulSoup)
- **Grupos 2026**: extraídos del mismo dataset (el calendario 2026 ya está incluido)

---

## Repositorio

- **GitHub**: https://github.com/harbey-26/prediccion-mundial-2026
- **Rama principal**: `main`
- **Último commit**: feat: agregar modelo ML ensemble XGBoost (Fase 4)
