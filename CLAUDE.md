# CLAUDE.md — Contexto del Proyecto

## ¿Qué es este proyecto?

Modelo de predicción del Mundial FIFA 2026 usando **ELO histórico + Ranking FIFA** con **simulación Monte Carlo** (100,000 torneos). Predice el ganador de cada partido, la clasificación por grupos y el campeón.

---

## Estado actual del proyecto

- ✅ Dataset descargado: 49,257 partidos internacionales (1872–2026)
- ✅ ELO ratings calculados para 336 selecciones (cacheados en `data/processed/elo_ratings.csv`)
- ✅ Ranking FIFA scrapeado: 200 equipos (abril 2026, en `data/raw/fifa_ranking_2026.json`)
- ✅ Simulaciones ejecutadas: 100,000 torneos × 2 modelos (ELO puro y ELO+FIFA)
- ✅ Excel generado: `results/excel/Mundial_2026_Predicciones.xlsx` (5 hojas)
- ✅ Google Sheets publicados en Drive del usuario (harbey.26@gmail.com)
- ✅ Repositorio en GitHub: https://github.com/harbey-26/prediccion-mundial-2026

---

## Entorno

```bash
# Siempre activar el entorno virtual antes de correr cualquier script
cd /home/ubuntu/proyectos/prediccion-mundial-2026
source venv/bin/activate
```

- Python 3.14 con virtualenv en `venv/`
- Dependencias: pandas, numpy, scikit-learn, matplotlib, seaborn, openpyxl, beautifulsoup4, lxml, tqdm

---

## Comandos clave

```bash
# Simulación completa con modelo compuesto (ELO + FIFA) — recomendado
python main.py --composite --sims 100000

# Comparar ambos modelos y guardar diferencias
python main.py --compare --sims 100000

# Predecir un partido específico
python main.py --match "Brazil" "Argentina"

# Regenerar el Excel con las 5 hojas
python generate_excel.py

# Recalcular ELO desde cero (si hay nuevos datos)
python main.py --recalc
```

---

## Arquitectura del modelo

### 1. ELO Histórico (`src/elo_calculator.py`)
- Rating inicial: 1500 para todos
- K-factor: 20 × peso del torneo
- Pesos por torneo: Mundial FIFA (2.0), Eurocopa (1.8), Copa América (1.5), Amistoso (0.5)
- Ventaja de local: +30 puntos ELO (no aplica en campo neutral)

### 2. Ranking FIFA (`src/fifa_ranking.py`)
- Scrapeado de Transfermarkt (200 equipos, paginación de 8 páginas)
- Puntos FIFA escala: ~1400–2000 (similar a ELO desde reforma 2018)
- Mapeo de nombres: "USA" → "United States", "Turkiye" → "Turkey", etc.

### 3. Rating Compuesto
```python
composite = 0.6 * elo + 0.4 * fifa_points  # ajustable con --elo-weight
```

### 4. Probabilidades de partido (`src/elo_calculator.py::win_probability`)
```python
p_draw = max(0.10, 0.28 - 0.001 * abs(elo_diff))
p_win  = p_win_raw * (1 - p_draw)
p_loss = (1 - p_win_raw) * (1 - p_draw)
```

### 5. Simulación Monte Carlo (`src/simulator.py`)
- Fase de grupos: round-robin con marcadores Poisson
- Clasifican: top-2 de cada grupo + 8 mejores terceros = 32 equipos
- Fase eliminatoria: R32 → Octavos → QF → SF → Final
- En eliminatorias: no hay empate (penalty shootout aleatorio por ELO)

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

## Resultados de las simulaciones (100,000 torneos)

### Top 10 — Modelo ELO + FIFA (recomendado)

| # | Selección | ELO solo | ELO+FIFA |
|---|-----------|:--------:|:--------:|
| 1 | España | 31.3% | **31.2%** |
| 2 | Argentina | 10.6% | **11.5%** |
| 3 | Francia | 8.8% | **10.4%** |
| 4 | Brasil | 5.5% | **6.5%** |
| 5 | Inglaterra | 5.5% | **6.3%** |
| 6 | Marruecos | 4.4% | **5.5%** |
| 7 | Países Bajos | 2.9% | **3.5%** |
| 8 | Alemania | 3.1% | **3.0%** |
| 9 | Portugal | 2.4% | **2.6%** |
| 10 | Bélgica | 1.6% | **2.6%** |

### Bracket proyectado (ganador más probable en cada partido)
- **Final:** España vs Argentina → **España** (39% vs 35%)
- **Semifinal 1:** Brasil vs España → España
- **Semifinal 2:** Argentina vs Noruega → Argentina

---

## Archivos de resultados

```
results/
├── csv/
│   ├── predicciones_elo.csv          # Modelo ELO puro
│   ├── predicciones_compuesto.csv    # Modelo ELO + FIFA ← usar este
│   ├── predicciones_campeon.csv      # Última simulación ejecutada
│   └── comparacion_modelos.csv       # Diferencias entre modelos
└── excel/
    └── Mundial_2026_Predicciones.xlsx  # 5 hojas (descargar desde GitHub)
```

---

## Google Sheets en Drive (harbey.26@gmail.com)

| Archivo | Link |
|---------|------|
| Partidos y Proyecciones (72 partidos) | https://docs.google.com/spreadsheets/d/11QDCdK48WzA5-FFGN6qU6ELMrMH7zcAsWj7CBJCn16g/edit |
| Probabilidades de Campeonato (48 equipos) | https://docs.google.com/spreadsheets/d/1kmZyOijsDKZi9EwnPrBLlNAC0cNhwJhEtVLs-6pXCT0/edit |
| Bracket Eliminatorio Proyectado | https://docs.google.com/spreadsheets/d/1a2EkbkNUPmVUjW4ZJdxdjvixUa-ZvdoMj4tERe8jJ6Y/edit |

---

## Ideas para continuar

- [ ] **Actualizar con resultados reales**: a medida que se jueguen partidos, alimentar los resultados al modelo y recalcular ELO en tiempo real
- [ ] **Agregar ranking FIFA actualizado**: el ranking se actualiza mensualmente, re-scrapear antes de cada fecha
- [ ] **Feature: historial H2H**: agregar estadísticas de enfrentamientos directos entre selecciones
- [ ] **Feature: goles esperados (xG)**: usar xG en lugar de goles reales para el ELO
- [ ] **Dashboard web**: exponer las predicciones en una app React (similar al stack de app-control-de-compras)
- [ ] **Actualizar grupos reales**: si el draw oficial cambia, editar `src/world_cup_2026.py`
- [ ] **Predicción de marcadores**: agregar distribución de Poisson para predecir el marcador exacto más probable

---

## Fuentes de datos

- **Partidos históricos**: https://github.com/martj42/international_results (se auto-descarga)
- **Ranking FIFA**: https://www.transfermarkt.com/statistik/weltrangliste (scraping con BeautifulSoup)
- **Grupos 2026**: extraídos del mismo dataset (el calendario 2026 ya está incluido)

---

## Repositorio

- **GitHub**: https://github.com/harbey-26/prediccion-mundial-2026
- **Rama principal**: `main`
- **Último commit**: bracket fix + Google Sheets + README con links
