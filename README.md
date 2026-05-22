# Predicción de Resultados del Mundial 2026

Modelo de predicción basado en **ELO histórico v2** + **Ranking FIFA** + **Ratings de Ataque/Defensa** + **Forma Reciente** con **simulación Monte Carlo** de 100,000 torneos completos.

---

## Resultados en Google Sheets

| Archivo | Descripción |
|---------|-------------|
| [📅 Partidos y Proyecciones](https://docs.google.com/spreadsheets/d/11QDCdK48WzA5-FFGN6qU6ELMrMH7zcAsWj7CBJCn16g/edit) | 72 partidos de fase de grupos con probabilidades y ganador proyectado |
| [🏆 Probabilidades de Campeonato](https://docs.google.com/spreadsheets/d/1kmZyOijsDKZi9EwnPrBLlNAC0cNhwJhEtVLs-6pXCT0/edit) | 48 selecciones — comparativa modelo ELO vs ELO+FIFA |
| [🪜 Bracket Eliminatorio](https://docs.google.com/spreadsheets/d/1yjlHZ95mK7r107TB1mZyn4Z6SBjbli6A8D9316P61CQ/edit) | Ronda de 32 → Final con ganador proyectado en cada duelo |

---

## Resultados del Modelo v3 (100,000 simulaciones)

### Top 10 candidatos al título — Modelo Ensemble (ELO + FIFA + A/D + Forma + ML)

| # | Selección | Prob. ELO puro | Prob. Ensemble | Δ |
|---|-----------|:--------------:|:--------------:|:-:|
| 🥇 | **España** | 22.9% | **23.2%** | ↑0.3% |
| 🥈 | **Argentina** | 11.7% | **14.7%** | ↑3.0% |
| 🥉 | **Francia** | 11.0% | **14.2%** | ↑3.2% |
| 4 | Marruecos | 8.5% | **8.5%** | →0.0% |
| 5 | Inglaterra | 7.0% | **7.4%** | ↑0.4% |
| 6 | Países Bajos | 3.1% | **4.3%** | ↑1.2% |
| 7 | Japón | 7.5% | **4.3%** | ↓3.2% |
| 8 | Alemania | 3.3% | **3.1%** | ↓0.2% |
| 9 | Brasil | 2.0% | **3.0%** | ↑1.0% |
| 10 | Portugal | 1.9% | **2.4%** | ↑0.5% |

> **Lectura:** El modelo ML ensemble (XGBoost, 60.6% accuracy) refuerza el favoritismo de Argentina (+3.0%) y Francia (+3.2%) sobre el ELO puro. Japón baja (-3.2%) porque el modelo ML penaliza su debilidad histórica en eliminatorias de alto nivel. España se mantiene como máxima favorita.

### Validación del modelo — Backtesting Mundiales 2018 y 2022

| Mundial | Campeón real | Prob. asignada | Ranking | Brier Score | Log-Loss |
|---------|-------------|:--------------:|:-------:|:-----------:|:--------:|
| Rusia 2018 | Francia | 7.67% | #4 de 32 | 0.030395 | 2.567 |
| Qatar 2022 | Argentina | 19.37% | #2 de 32 | 0.024207 | 1.641 |
| **Naive (1/32)** | — | 3.13% | — | 0.030273 | 3.466 |

El modelo **supera al baseline equiprobable** en ambos torneos: +20.6% mejor Brier Score en 2022 y +53% mejor Log-Loss. En ambos mundiales el campeón real estuvo entre los dos primeros favoritos.

---

## Descripción del Modelo v3

El modelo combina cinco fuentes de información para cada selección:

| Componente | Peso | Descripción |
|-----------|:----:|-------------|
| **ELO Histórico v2** | 60% | Calculado sobre 49,257 partidos (1872–2026). Incluye **goal-difference multiplier** (3-0 vale 1.75× más que 1-0), **time decay** (partidos recientes pesan más) y **K-factor dinámico** por incertidumbre. |
| **Ranking FIFA** | 40% | Puntos FIFA scrapeados de Transfermarkt (200 selecciones, abril 2026). |
| **Ataque/Defensa** | — | Ratings separados por equipo (metodología Dixon-Coles, últimos 4 años). Determina los goles esperados vía distribución Poisson: `λ = avg_global × ataque_A × defensa_B`. |
| **Forma Reciente** | — | Ajuste ELO de ±80 pts basado en los últimos 10 partidos (pesos exponenciales). Alemania, Francia y Argentina son los equipos en mejor forma actualmente. |
| **ML Ensemble** | 35% | XGBClassifier entrenado sobre 37,165 partidos (1980–2026) con 15 features. Accuracy validación: **60.6%**. Log-loss **20.6% mejor** que baseline uniforme. Se blend con ELO: `p_final = 0.65 × p_elo + 0.35 × p_ml`. |

La **probabilidad de empate** se calibra empíricamente con `scipy.optimize.curve_fit` sobre el historial completo: `P(empate) = 0.239 × exp(-0.00158 × |diff_ELO|) + 0.020`.

### Arquitectura del simulador

```
100,000 torneos completos:
  ├── Fase de grupos (round-robin, 12 grupos)
  │     Resultado: ELO-based  |  Goles: Poisson(λ ataque × defensa)
  │     Clasifican: 1° y 2° de cada grupo + 8 mejores terceros (criterio FIFA)
  ├── R32 → R16 → QF → SF → Final
  │     Sin empate (penalty shootout por probabilidad ELO)
  └── Acumula: P(campeón) por selección
```

---

## Requisitos

- Python 3.10+
- pip / entorno virtual

```bash
python3 -m venv venv
source venv/bin/activate       # Linux / Mac
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

---

## Instalación y primer uso

```bash
git clone https://github.com/harbey-26/prediccion-mundial-2026.git
cd prediccion-mundial-2026

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Los datos históricos se descargan automáticamente desde [martj42/international_results](https://github.com/martj42/international_results) al ejecutar el modelo por primera vez.

---

## Uso

### Simulación completa recomendada

```bash
# Primer uso: recalcular ELO + calibrar empate + correr 100k simulaciones
python main.py --recalc --calibrate --composite --compare --sims 100000

# Ejecuciones posteriores (usa caché de ELO y calibración)
python main.py --composite --compare
```

### Otros comandos

```bash
# Solo modelo ELO + FIFA (sin comparar)
python main.py --composite

# Predecir un partido específico
python main.py --match "Brazil" "Argentina"

# Backtesting sobre Mundiales 2018 y 2022
python main.py --backtest --backtest-sims 50000

# Modo básico sin features adicionales (más rápido)
python main.py --composite --no-features
```

### Generar el archivo Excel

```bash
python generate_excel.py
```

Genera `results/excel/Mundial_2026_Predicciones.xlsx` con **6 hojas** detalladas.

### Todas las opciones

| Argumento | Descripción | Default |
|-----------|-------------|---------|
| `--sims N` | Número de torneos a simular | `100000` |
| `--composite` | Usar rating compuesto ELO + FIFA | `False` |
| `--compare` | Ejecutar ambos modelos y comparar resultados | `False` |
| `--elo-weight X` | Peso del ELO en el rating compuesto (0.0–1.0) | `0.6` |
| `--match A B` | Predecir un partido entre dos selecciones | — |
| `--recalc` | Recalcular ELO desde cero (borra caché) | `False` |
| `--calibrate` | Calibrar probabilidad de empate empíricamente | `False` |
| `--backtest` | Backtesting sobre Mundiales 2018 y 2022 | `False` |
| `--backtest-sims N` | Simulaciones para backtesting | `50000` |
| `--no-features` | Desactivar ataque/defensa y forma (modo básico) | `False` |
| `--train-ml` | Entrenar modelo XGBoost y guardar en caché | `False` |
| `--ml-weight X` | Peso del modelo ML en el blend (0.0–1.0) | `0.35` |
| `--no-ml` | Desactivar modelo ML aunque exista caché | `False` |
| `--download` | Forzar re-descarga del dataset | `False` |
| `--no-plot` | No generar gráficos | `False` |

---

## Estructura del Proyecto

```
prediccion-mundial-2026/
│
├── main.py                  # CLI principal — simulaciones y predicciones
├── generate_excel.py        # Generador del Excel con todas las proyecciones
├── requirements.txt
│
├── src/
│   ├── data_loader.py       # Descarga y limpieza del dataset histórico
│   ├── elo_calculator.py    # ELO v2: goal multiplier, time decay, K dinámico
│   ├── calibration.py       # Calibración empírica de P(empate) con scipy
│   ├── attack_defense.py    # Ratings de ataque/defensa (Dixon-Coles simplificado)
│   ├── form.py              # Factor de forma reciente (últimos 10 partidos)
│   ├── backtesting.py       # Backtesting sobre Mundiales 2018 y 2022
│   ├── fifa_ranking.py      # Scraping del ranking FIFA y rating compuesto
│   ├── simulator.py         # Simulación Monte Carlo — soporta 32 y 48 equipos
│   ├── visualizer.py        # Gráficos matplotlib
│   └── world_cup_2026.py    # Grupos y equipos oficiales del Mundial 2026
│
├── data/
│   ├── raw/                 # Dataset descargado (CSV + JSON ranking FIFA)
│   └── processed/           # ELO ratings y calibración cacheados
│
└── results/
    ├── csv/
    │   ├── predicciones_elo.csv         # Probabilidades modelo ELO puro
    │   ├── predicciones_compuesto.csv   # Probabilidades modelo completo
    │   ├── comparacion_modelos.csv      # Diferencias entre modelos
    │   └── backtest_summary.csv         # Métricas de validación
    └── excel/
        └── Mundial_2026_Predicciones.xlsx  # Reporte completo (6 hojas)
```

---

## Archivo Excel — Contenido (6 hojas)

| Hoja | Contenido |
|------|-----------|
| **Fase de Grupos** | 72 partidos con fecha, probabilidades coloreadas y proyección del ganador |
| **Clasificación Proyectada** | P(1°), P(2°), P(3°), P(4°) para cada equipo en base a 5,000 simulaciones |
| **Fase Eliminatoria** | Bracket completo R32 → Final con ganador proyectado en cada duelo |
| **Prob. Campeonato** | 48 selecciones ordenadas por probabilidad de título — ELO puro vs. modelo completo |
| **Ratings** | ELO v2, puntos FIFA y rating compuesto de los 48 clasificados |
| **Validación Modelo** | Backtesting 2018/2022: Brier Score, Log-Loss y comparativa de modelos |

---

## Grupos del Mundial 2026

| Grupo | Equipos |
|-------|---------|
| A | México, Corea del Sur, Rep. Checa, Sudáfrica |
| B | Canadá, Bosnia y Herz., Suiza, Qatar |
| C | Estados Unidos, Paraguay, Australia, Turquía |
| D | Brasil, Marruecos, Haití, Escocia |
| E | Alemania, Costa de Marfil, Ecuador, Curazao |
| F | Países Bajos, Japón, Suecia, Túnez |
| G | Bélgica, Irán, Egipto, Nueva Zelanda |
| H | España, Arabia Saudita, Uruguay, Cabo Verde |
| I | Francia, Senegal, Noruega, Iraq |
| J | Argentina, Argelia, Austria, Jordania |
| K | Portugal, Colombia, RD Congo, Uzbekistán |
| L | Inglaterra, Croacia, Ghana, Panamá |

---

## Datos

- **Dataset histórico:** [martj42/international_results](https://github.com/martj42/international_results) — 49,257 partidos (1872–2026)
- **Ranking FIFA:** Scrapeado de [Transfermarkt](https://www.transfermarkt.com/statistik/weltrangliste) — actualización abril 2026
- **Grupos del Mundial 2026:** Extraídos del propio dataset (calendario oficial incluido)
