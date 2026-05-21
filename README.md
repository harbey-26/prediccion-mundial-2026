# Predicción de Resultados del Mundial 2026

Modelo de predicción basado en **ELO histórico** + **Ranking FIFA** con **simulación Monte Carlo** para proyectar el ganador de cada partido y el campeón del Mundial FIFA 2026.

---

## Descripción del Modelo

El modelo combina dos fuentes de datos para calcular un **rating compuesto** por selección:

| Fuente | Peso | Descripción |
|--------|:----:|-------------|
| ELO Histórico | 60% | Calculado a partir de 49,257 partidos internacionales (1872–2026). Los torneos más importantes (Mundiales, Eurocopas) tienen mayor ponderación. |
| Ranking FIFA | 40% | Puntos FIFA oficiales scrapeados de Transfermarkt (200 selecciones, actualización abril 2026). |

Con ese rating se calcula la **probabilidad de victoria, empate y derrota** para cada enfrentamiento, y se corre una **simulación Monte Carlo** de 100,000 torneos completos para estimar la probabilidad de que cada selección gane el Mundial.

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

Los datos históricos se descargan automáticamente desde el repositorio público de [martj42/international_results](https://github.com/martj42/international_results) al ejecutar el modelo por primera vez.

---

## Uso

### Simulación completa (modelo compuesto ELO + FIFA)

```bash
python main.py --composite
```

### Comparar modelo ELO puro vs ELO + FIFA

```bash
python main.py --compare
```

### Ajustar número de simulaciones

```bash
python main.py --compare --sims 50000
```

### Predecir un partido específico

```bash
python main.py --match "Brazil" "Argentina"
```

Devuelve las probabilidades de victoria, empate y derrota según ambos modelos.

### Opciones disponibles

| Argumento | Descripción | Default |
|-----------|-------------|---------|
| `--sims N` | Número de torneos a simular | `100000` |
| `--composite` | Usar rating compuesto ELO + FIFA | `False` |
| `--compare` | Ejecutar ambos modelos y comparar resultados | `False` |
| `--elo-weight X` | Peso del ELO en el rating compuesto (0.0–1.0) | `0.6` |
| `--match A B` | Predecir un partido entre dos selecciones | — |
| `--recalc` | Recalcular ELO desde cero (ignora caché) | `False` |
| `--download` | Forzar re-descarga del dataset | `False` |
| `--no-plot` | No generar gráficos | `False` |

### Generar el archivo Excel

```bash
python generate_excel.py
```

Genera `results/excel/Mundial_2026_Predicciones.xlsx` con 5 hojas detalladas.

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
│   ├── elo_calculator.py    # Cálculo de ratings ELO y probabilidades de partido
│   ├── fifa_ranking.py      # Scraping del ranking FIFA y rating compuesto
│   ├── simulator.py         # Simulación Monte Carlo del torneo completo
│   ├── visualizer.py        # Gráficos matplotlib (ELO, probabilidades, grupos)
│   └── world_cup_2026.py    # Grupos y equipos oficiales del Mundial 2026
│
├── data/
│   ├── raw/                 # Dataset descargado (CSV + JSON ranking FIFA)
│   └── processed/           # ELO ratings cacheados
│
└── results/
    ├── csv/
    │   ├── predicciones_elo.csv         # Probabilidades modelo ELO puro
    │   ├── predicciones_compuesto.csv   # Probabilidades modelo ELO + FIFA
    │   ├── predicciones_campeon.csv     # Última simulación ejecutada
    │   └── comparacion_modelos.csv      # Diferencias entre ambos modelos
    └── excel/
        └── Mundial_2026_Predicciones.xlsx  # Reporte completo (5 hojas)
```

---

## Archivo Excel — Contenido

El Excel generado por `generate_excel.py` contiene 5 hojas:

| Hoja | Contenido |
|------|-----------|
| **Fase de Grupos** | 72 partidos con fecha, probabilidades coloreadas y proyección del ganador |
| **Clasificación Proyectada** | P(1°), P(2°), P(3°), P(4°) para cada equipo en base a 5,000 simulaciones de fase de grupos |
| **Fase Eliminatoria** | Bracket completo R32 → Octavos → Cuartos → Semifinales → Final con ganador proyectado |
| **Prob. Campeonato** | Las 48 selecciones ordenadas por probabilidad de título con comparación entre modelos |
| **Ratings** | ELO histórico, puntos FIFA y rating compuesto de los 48 clasificados |

---

## Resultados del Modelo (100,000 simulaciones)

### Top 10 candidatos al título — Modelo ELO + FIFA

| # | Selección | Probabilidad |
|---|-----------|:------------:|
| 🥇 | España | 31.2% |
| 🥈 | Argentina | 11.5% |
| 🥉 | Francia | 10.4% |
| 4 | Brasil | 6.5% |
| 5 | Inglaterra | 6.3% |
| 6 | Marruecos | 5.5% |
| 7 | Países Bajos | 3.5% |
| 8 | Alemania | 3.0% |
| 9 | Portugal | 2.6% |
| 10 | Bélgica | 2.6% |

### Impacto del ranking FIFA sobre el modelo ELO puro

Selecciones que **suben** al incorporar el FIFA ranking: Francia (+1.6%), Marruecos (+1.1%), Bélgica (+1.0%), Brasil (+0.9%), Argentina (+0.9%).

Selecciones que **bajan**: Ecuador (-1.4%), Japón (-1.2%), Australia (-0.9%), Corea del Sur (-0.7%).

---

## Datos

- **Dataset histórico:** [martj42/international_results](https://github.com/martj42/international_results) — 49,257 partidos (1872–2026)
- **Ranking FIFA:** Scrapeado de [Transfermarkt](https://www.transfermarkt.com/statistik/weltrangliste) — actualización abril 2026
- **Grupos del Mundial 2026:** Extraídos del propio dataset (calendario oficial ya incluido)

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
