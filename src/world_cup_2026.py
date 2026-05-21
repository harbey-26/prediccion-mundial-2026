"""
Configuración oficial del Mundial 2026: grupos y equipos clasificados.
Fuente: dataset de resultados internacionales (schedule ya incluido).
48 equipos, 12 grupos de 4. Top 2 + 8 mejores terceros = 32 en fase eliminatoria.
"""

GROUPS: dict[str, list[str]] = {
    "A": ["Mexico", "South Korea", "Czech Republic", "South Africa"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["United States", "Paraguay", "Australia", "Turkey"],
    "D": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "E": ["Germany", "Ivory Coast", "Ecuador", "Curaçao"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Iran", "Egypt", "New Zealand"],
    "H": ["Spain", "Saudi Arabia", "Uruguay", "Cape Verde"],
    "I": ["France", "Senegal", "Norway", "Iraq"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "Colombia", "DR Congo", "Uzbekistan"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}

ALL_TEAMS = [team for teams in GROUPS.values() for team in teams]
