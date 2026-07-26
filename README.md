# Tactical Feature Engineering — SkillCorner Analytics

Extracts 38 match-level tactical features across 7 structural pillars from SkillCorner dynamic events and phases-of-play data.

## Features Engine

`hackathon_features.py` processes raw SkillCorner match data and aggregates:

1. **Half-Space Transitional Kill Chains** (3 features)
2. **Zone 14 Second-Ball Dominance** (3 features)
3. **Pressing Architecture & Collective Disruption** (6 features)
4. **Vertical Progression & Build-Up DNA** (8 features)
5. **Spatial Territorial Control** (6 features)
6. **Chance Quality & Offensive Structure** (9 features)
7. **Width Exploitation & Structural Movement** (3 features)

## Output

Generates `features.csv` (20 rows × 40 columns: `match_id`, `team_id`, + 38 tactical features).

## Usage

```bash
python hackathon_features.py
```
