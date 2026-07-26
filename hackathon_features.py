#!/usr/bin/env python3
"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ELITE TACTICAL FEATURE ENGINEERING
SkillCorner Open Data Hackathon — Kaggle Competition
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Extracts 35 novel match-level tactical features per team from SkillCorner's
dynamic events and phases-of-play data.

Tactical Pillars:
    1. Half-Space Transitional Kill Chains
    2. Zone 14 Second-Ball Dominance
    3. Pressing Architecture & Collective Disruption
    4. Vertical Progression & Build-Up DNA
    5. Spatial Territorial Control
    6. Chance Quality & Offensive Structure
    7. Width Exploitation & Structural Movement

All features are raw aggregated values (counts, sums, durations).
No ratios, percentages, or normalized values.

Output: features.csv  (20 rows = 2 teams × 10 matches, 40 columns)

Usage:
    python hackathon_features.py
"""

import json
import warnings
from pathlib import Path
from typing import Dict, FrozenSet, List, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTS & SPATIAL ZONE DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════

HALF_SPACE_CHANNELS: FrozenSet[str] = frozenset(
    ["half_space_left", "half_space_right"]
)
WIDE_CHANNELS: FrozenSet[str] = frozenset(["wide_left", "wide_right"])

SIDEWAY_DIRECTIONS: FrozenSet[str] = frozenset(["sideway_left", "sideway_right"])

OVERLAP_UNDERLAP_SUBTYPES: FrozenSet[str] = frozenset(["overlap", "underlap"])

PULLING_SUBTYPES: FrozenSet[str] = frozenset(["pulling_wide", "pulling_half_space"])

# Start types indicating the team WON the ball via a defensive action.
# These seed possession chains that originate from disruption.
RECOVERY_START_TYPES: FrozenSet[str] = frozenset([
    "recovery",
    "pass_interception",
    "throw_in_interception",
    "corner_interception",
    "goal_kick_interception",
    "free_kick_interception",
])

# On-ball engagement end types that confirm a successful ball regain.
REGAIN_END_TYPES: FrozenSet[str] = frozenset([
    "direct_regain",
    "indirect_regain",
])

# Start types that indicate unbroken continuation of a possession chain.
CHAIN_CONTINUATION_TYPES: FrozenSet[str] = frozenset([
    "pass_reception",
    "keep_possession",
])

# Maximum seconds after a recovery for a kill-chain pass to count.
KILL_CHAIN_WINDOW_SEC: float = 3.0

# Maximum seconds to search forward when tracing a possession chain.
CHAIN_SEARCH_HORIZON_SEC: float = 120.0


# ═══════════════════════════════════════════════════════════════════════════
# DATA LAYER — discovery, loading, time parsing
# ═══════════════════════════════════════════════════════════════════════════

def discover_match_ids(data_root: Path) -> List[int]:
    """Dynamically discover all match IDs from the directory structure."""
    matches_dir = data_root / "matches"
    return sorted(
        int(d.name)
        for d in matches_dir.iterdir()
        if d.is_dir() and d.name.isdigit()
    )


def load_match_teams(data_root: Path) -> Dict[int, Tuple[int, int]]:
    """Return {match_id: (home_team_id, away_team_id)} from matches.json."""
    with open(data_root / "matches.json") as f:
        matches = json.load(f)
    return {
        m["id"]: (m["home_team"]["id"], m["away_team"]["id"])
        for m in matches
    }


def _time_to_seconds(t) -> float:
    """Convert 'MM:SS.S' time string to total seconds."""
    if pd.isna(t):
        return np.nan
    try:
        parts = str(t).split(":")
        return float(parts[0]) * 60.0 + float(parts[1])
    except (ValueError, IndexError):
        return np.nan


def load_dynamic_events(match_id: int, data_root: Path) -> pd.DataFrame:
    """Load and prepare the dynamic events CSV for one match."""
    path = data_root / "matches" / str(match_id) / f"{match_id}_dynamic_events.csv"
    df = pd.read_csv(path, low_memory=False)
    # Vectorised time parsing — avoids per-row lambda overhead
    df["_t_start"] = df["time_start"].map(_time_to_seconds)
    df["_t_end"] = df["time_end"].map(_time_to_seconds)
    return df


def load_phases_of_play(match_id: int, data_root: Path) -> pd.DataFrame:
    """Load the phases-of-play CSV for one match."""
    path = data_root / "matches" / str(match_id) / f"{match_id}_phases_of_play.csv"
    return pd.read_csv(path)


# ═══════════════════════════════════════════════════════════════════════════
# TACTICAL FEATURE ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class TacticalFeatureEngine:
    """
    Computes 35 elite tactical features for both teams in a single match.

    Design principles:
        • Pre-filters event types on __init__ — avoids redundant masks.
        • Vectorised pandas operations for all simple aggregations.
        • Targeted iteration *only* for chain-sequence metrics where
          event ordering matters (kill chains, second-ball duration).
        • Phase transitions pre-computed once via shift().
    """

    def __init__(
        self,
        events: pd.DataFrame,
        phases: pd.DataFrame,
        match_id: int,
    ) -> None:
        self.match_id = match_id
        self.events = events.sort_values("index").reset_index(drop=True)
        self.phases = phases.sort_values("frame_start").reset_index(drop=True)
        self.team_ids: List[int] = sorted(
            events["team_id"].dropna().unique().astype(int)
        )

        # ── Pre-filter by event type (computed once, reused by every pillar) ──
        self._pp = self.events.loc[
            self.events["event_type"] == "player_possession"
        ]
        self._obe = self.events.loc[
            self.events["event_type"] == "on_ball_engagement"
        ]
        self._obr = self.events.loc[
            self.events["event_type"] == "off_ball_run"
        ]

        # ── Enrich phases with the out-of-possession team id ──
        self._enrich_phases()
        # ── Vectorised next-phase transition columns ──
        self._precompute_phase_transitions()

    # ── internal setup ────────────────────────────────────────────────

    def _enrich_phases(self) -> None:
        """Map the opposing team id onto every phase row."""
        tids = self.phases["team_in_possession_id"].unique()
        if len(tids) == 2:
            swap = {tids[0]: tids[1], tids[1]: tids[0]}
            self.phases["_opp_team_id"] = (
                self.phases["team_in_possession_id"].map(swap)
            )

    def _precompute_phase_transitions(self) -> None:
        """Compute next-phase columns using vectorised shift — O(n)."""
        p = self.phases
        p["_nxt_frame_start"] = p["frame_start"].shift(-1)
        p["_nxt_team"] = p["team_in_possession_id"].shift(-1)
        p["_nxt_phase"] = p["team_in_possession_phase_type"].shift(-1)
        # Two boolean masks re-used across features
        p["_continuous"] = p["frame_end"] == p["_nxt_frame_start"]
        p["_same_team"] = p["team_in_possession_id"] == p["_nxt_team"]

    # ── public API ────────────────────────────────────────────────────

    def compute_all(self) -> pd.DataFrame:
        """Entry point: compute all 38 features for both teams → 2-row DF."""
        rows: List[Dict] = []
        for tid in self.team_ids:
            opp = [t for t in self.team_ids if t != tid][0]

            # Team-specific slices (views, not copies)
            pp = self._pp.loc[self._pp["team_id"] == tid]
            pp_opp = self._pp.loc[self._pp["team_id"] == opp]
            obe = self._obe.loc[self._obe["team_id"] == tid]
            obr = self._obr.loc[self._obr["team_id"] == tid]
            phases_in = self.phases.loc[
                self.phases["team_in_possession_id"] == tid
            ]

            feat: Dict = {"match_id": self.match_id, "team_id": tid}
            feat.update(self._pillar1_halfspace(pp, obe, tid))
            feat.update(self._pillar2_zone14(pp, tid))
            feat.update(self._pillar3_pressing(obe))
            feat.update(self._pillar4_vertical(pp, phases_in, tid))
            feat.update(self._pillar5_spatial(pp, phases_in))
            feat.update(self._pillar6_chance(pp, obr, opp))
            feat.update(self._pillar7_width(pp, obr))
            rows.append(feat)

        return pd.DataFrame(rows)

    # ═══════════════════════════════════════════════════════════════════
    # PILLAR 1 — Half-Space Transitional Kill Chains
    # ═══════════════════════════════════════════════════════════════════

    def _pillar1_halfspace(
        self, pp: pd.DataFrame, obe: pd.DataFrame, tid: int
    ) -> Dict:
        hs = HALF_SPACE_CHANNELS
        in_hs = pp["channel_start"].isin(hs)
        is_recovery = pp["start_type"].isin(RECOVERY_START_TYPES)

        # F1  Kill Chains — recovery in HS → pass to box within 3 s
        kill_chains = self._count_kill_chains(pp, tid)

        # F2  Total defensive recoveries in half-spaces
        hs_pp_rec = pp.loc[in_hs & is_recovery].shape[0]
        hs_obe_reg = obe.loc[
            obe["channel_start"].isin(hs)
            & obe["end_type"].isin(REGAIN_END_TYPES)
        ].shape[0]

        # F3  Passes from half-space that reach the penalty area
        hs_to_box = pp.loc[
            in_hs
            & (pp["player_targeted_penalty_area_reception"] == True)
            & (pp["pass_outcome"] == "successful")
        ].shape[0]

        return {
            "halfspace_kill_chains": kill_chains,
            "halfspace_defensive_recoveries": hs_pp_rec + hs_obe_reg,
            "halfspace_to_box_progressions": hs_to_box,
        }

    def _count_kill_chains(self, pp: pd.DataFrame, tid: int) -> int:
        """
        Sequence: recovery in half-space → successful pass into the
        penalty area within KILL_CHAIN_WINDOW_SEC seconds.
        """
        seeds = pp.loc[
            pp["channel_start"].isin(HALF_SPACE_CHANNELS)
            & pp["start_type"].isin(RECOVERY_START_TYPES)
        ]
        if seeds.empty:
            return 0

        # Pre-filter candidate box-entry passes (vectorised)
        box_passes = pp.loc[
            (pp["team_id"] == tid)
            & (pp["player_targeted_penalty_area_reception"] == True)
            & (pp["pass_outcome"] == "successful")
        ]
        if box_passes.empty:
            return 0

        count = 0
        bp_times = box_passes[["_t_start", "period"]].values
        for _, row in seeds.iterrows():
            t0 = row["_t_start"]
            per = row["period"]
            if pd.isna(t0):
                continue
            # Vectorised window check against all box-passes
            mask = (
                (bp_times[:, 1] == per)
                & (bp_times[:, 0] > t0)
                & (bp_times[:, 0] <= t0 + KILL_CHAIN_WINDOW_SEC)
            )
            if mask.any():
                count += 1
        return count

    # ═══════════════════════════════════════════════════════════════════
    # PILLAR 2 — Zone 14 Second-Ball Dominance
    # ═══════════════════════════════════════════════════════════════════

    def _pillar2_zone14(self, pp: pd.DataFrame, tid: int) -> Dict:
        """
        Zone 14 ≈ attacking third ∩ center channel ∩ NOT penalty area.
        This is the classic 'danger rectangle' just outside the box.
        """
        z14 = (
            (pp["third_start"] == "attacking_third")
            & (pp["channel_start"] == "center")
            & (pp["penalty_area_start"] == False)
        )

        # F4  Second-ball possession duration after Zone 14 recovery
        z14_rec = pp.loc[z14 & pp["start_type"].isin(RECOVERY_START_TYPES)]
        sb_dur = self._chain_duration(z14_rec, tid)

        # F5  Zone 14 recovery count
        z14_count = z14_rec.shape[0]

        # F6  Zone 14 events leading to shot
        z14_shots = pp.loc[z14 & (pp["lead_to_shot"] == True)].shape[0]

        return {
            "zone14_second_ball_duration": round(sb_dur, 2),
            "zone14_recoveries": z14_count,
            "zone14_shot_generation_events": z14_shots,
        }

    def _chain_duration(self, seeds: pd.DataFrame, tid: int) -> float:
        """
        Sum total seconds of continuous possession chains that begin
        from each seed recovery event.  A chain continues as long as
        subsequent team possessions start via pass_reception or
        keep_possession.
        """
        if seeds.empty:
            return 0.0

        pp_team = (
            self._pp.loc[self._pp["team_id"] == tid]
            .sort_values("_t_start")
        )
        pp_times = pp_team["_t_start"].values
        pp_periods = pp_team["period"].values
        pp_start_types = pp_team["start_type"].values
        pp_durations = pp_team["duration"].values

        total = 0.0
        for _, seed in seeds.iterrows():
            t0 = seed["_t_start"]
            per = seed["period"]
            dur = seed.get("duration", 0.0)
            if pd.notna(dur):
                total += dur
            if pd.isna(t0):
                continue

            # Walk forward through same-team possessions via numpy arrays
            idx_start = np.searchsorted(pp_times, t0, side="right")
            for i in range(idx_start, len(pp_times)):
                if pp_periods[i] != per:
                    break
                if pp_times[i] > t0 + CHAIN_SEARCH_HORIZON_SEC:
                    break
                if pp_start_types[i] in CHAIN_CONTINUATION_TYPES:
                    d = pp_durations[i]
                    if not np.isnan(d):
                        total += d
                else:
                    break

        return total

    # ═══════════════════════════════════════════════════════════════════
    # PILLAR 3 — Pressing Architecture & Collective Disruption
    # ═══════════════════════════════════════════════════════════════════

    def _pillar3_pressing(self, obe: pd.DataFrame) -> Dict:
        pressing = obe.loc[obe["pressing_chain"] == True]

        # F7   Total engagements inside pressing chains
        total_eng = pressing.shape[0]

        # F8   Unique chains ending in regain
        if not pressing.empty:
            chain_end = (
                pressing.groupby("pressing_chain_index")
                ["pressing_chain_end_type"]
                .first()
            )
            regain_chains = int((chain_end == "regain").sum())
        else:
            regain_chains = 0

        # F9   High-press engagements (opponent's defensive third)
        #      In SkillCorner labels, thirds are relative to the
        #      IN-POSSESSION team; 'defensive_third' for a defensive
        #      engagement means the presser is deep in the opponent's end.
        deep_press = pressing.loc[
            pressing["third_start"] == "defensive_third"
        ].shape[0]

        # F10  Sum of pressing chain lengths (tactical investment)
        if not pressing.empty:
            lengths = (
                pressing.groupby("pressing_chain_index")
                ["pressing_chain_length"]
                .first()
            )
            chain_len_sum = int(lengths.sum())
        else:
            chain_len_sum = 0

        # F11  Force-backward engagements
        force_back = obe.loc[obe["force_backward"] == True].shape[0]

        # F12  Consecutive on-ball engagements
        consec = obe.loc[
            obe["consecutive_on_ball_engagements"] == True
        ].shape[0]

        return {
            "pressing_chain_total_engagements": total_eng,
            "pressing_chains_leading_to_regain": regain_chains,
            "deep_pressing_chain_engagements": deep_press,
            "pressing_chain_length_sum": chain_len_sum,
            "force_backward_engagements": force_back,
            "consecutive_engagement_events": consec,
        }

    # ═══════════════════════════════════════════════════════════════════
    # PILLAR 4 — Vertical Progression & Build-Up DNA
    # ═══════════════════════════════════════════════════════════════════

    def _pillar4_vertical(
        self, pp: pd.DataFrame, phases_in: pd.DataFrame, tid: int
    ) -> Dict:
        # F13  Full-pitch progression chains (def third → att third)
        d2a = phases_in.loc[
            (phases_in["third_start"] == "defensive_third")
            & (phases_in["third_end"] == "attacking_third")
        ].shape[0]

        # F14  Total line-breaking passes (first + last line breaks)
        lb_first = int((pp["first_line_break"] == True).sum())
        lb_last = int((pp["last_line_break"] == True).sum())

        # F15  Positive opponents bypassed (sum where > 0)
        bypassed = pp["n_opponents_bypassed"].fillna(0)
        total_bypassed = int(bypassed[bypassed > 0].sum())

        # F16  Forward carries
        fwd_carries = pp.loc[
            (pp["carry"] == True)
            & (pp["trajectory_direction"] == "forward")
        ].shape[0]

        # F17  One-touch passes in the attacking third
        one_touch_att = pp.loc[
            (pp["one_touch"] == True)
            & (pp["third_start"] == "attacking_third")
        ].shape[0]

        # F18  Quick passes (tempo indicator)
        quick_passes = int((pp["quick_pass"] == True).sum())

        # F19  High passes (aerial route usage)
        high_passes = int((pp["high_pass"] == True).sum())

        # F20  Build-up → Create phase transitions (vectorised)
        p = self.phases
        bu_to_cr = p.loc[
            p["_continuous"]
            & p["_same_team"]
            & (p["team_in_possession_id"] == tid)
            & (p["team_in_possession_phase_type"] == "build_up")
            & (p["_nxt_phase"] == "create")
        ].shape[0]

        return {
            "defensive_to_attacking_third_chains": d2a,
            "line_breaking_passes_total": lb_first + lb_last,
            "opponents_bypassed_total_positive": total_bypassed,
            "forward_carries_total": fwd_carries,
            "one_touch_passes_in_final_third": one_touch_att,
            "quick_passes_total": quick_passes,
            "high_passes_total": high_passes,
            "build_up_to_create_transitions": bu_to_cr,
        }

    # ═══════════════════════════════════════════════════════════════════
    # PILLAR 5 — Spatial Territorial Control
    # ═══════════════════════════════════════════════════════════════════

    def _pillar5_spatial(
        self, pp: pd.DataFrame, phases_in: pd.DataFrame
    ) -> Dict:
        # F21  Total possession time in attacking third (from phases)
        att_dur = float(
            phases_in.loc[
                phases_in["third_end"] == "attacking_third", "duration"
            ].sum()
        )

        # F22  Penalty-area entries (crossing the box line)
        box_entries = pp.loc[
            (pp["penalty_area_end"] == True)
            & (pp["penalty_area_start"] == False)
        ].shape[0]

        # F23  Wide-channel possessions
        wide = pp.loc[pp["channel_start"].isin(WIDE_CHANNELS)].shape[0]

        # F24  Center-channel possessions
        center = pp.loc[pp["channel_start"] == "center"].shape[0]

        # F25  Total forward distance gained
        fwd_dist = float(
            pp.loc[
                pp["trajectory_direction"] == "forward", "distance_covered"
            ]
            .fillna(0)
            .sum()
        )

        # F26  Total separation gained (metres of space created)
        sep = pp["separation_gain"].fillna(0)
        total_sep = float(sep[sep > 0].sum())

        return {
            "total_possession_duration_attacking_third": round(att_dur, 2),
            "penalty_area_entries": box_entries,
            "wide_channel_possession_events": wide,
            "center_channel_possession_events": center,
            "territorial_gain_distance": round(fwd_dist, 2),
            "total_separation_gained": round(total_sep, 2),
        }

    # ═══════════════════════════════════════════════════════════════════
    # PILLAR 6 — Chance Quality & Offensive Structure
    # ═══════════════════════════════════════════════════════════════════

    def _pillar6_chance(
        self, pp: pd.DataFrame, obr: pd.DataFrame, opp_tid: int
    ) -> Dict:
        # F27  Dangerous (makeable) passing options created
        dang_easy = int(
            pp["n_passing_options_dangerous_not_difficult"].fillna(0).sum()
        )

        # F28  Dangerous + difficult passing options (creative ambition)
        dang_hard = int(
            pp["n_passing_options_dangerous_difficult"].fillna(0).sum()
        )

        # F29  Off-ball runs ending inside the box
        runs_box = obr.loc[obr["penalty_area_end"] == True].shape[0]

        # F30  Runs behind the defensive line
        runs_behind = obr.loc[
            obr["event_subtype"] == "behind"
        ].shape[0]

        # F31  Give-and-go (wall pass) initiations
        gng = int((pp["initiate_give_and_go"] == True).sum())

        # F32  Shot-generating possessions
        shot_poss = int((pp["lead_to_shot"] == True).sum())

        # F33  Goal-generating possessions
        goal_poss = int((pp["lead_to_goal"] == True).sum())

        # F34  Accumulated xThreat from targeted passes
        xthreat = float(pp["player_targeted_xthreat"].fillna(0).sum())

        # F35  Accumulated xShot — offensive shot probability generated.
        #       xshot lives on on_ball_engagement rows where team_id is the
        #       DEFENDER.  To get THIS team's offensive xshot, sum OBE rows
        #       where the opponent is the one defending.
        opp_obe = self._obe.loc[self._obe["team_id"] == opp_tid]
        xshot = float(opp_obe["xshot_player_possession_max"].fillna(0).sum())

        return {
            "dangerous_passing_options_created": dang_easy,
            "difficult_dangerous_options_created": dang_hard,
            "off_ball_runs_into_box": runs_box,
            "off_ball_runs_behind_defense": runs_behind,
            "give_and_go_initiations": gng,
            "shot_generating_possessions": shot_poss,
            "goal_generating_possessions": goal_poss,
            "xthreat_accumulated": round(xthreat, 4),
            "xshot_accumulated": round(xshot, 4),
        }

    # ═══════════════════════════════════════════════════════════════════
    # PILLAR 7 — Width Exploitation & Structural Movement
    # ═══════════════════════════════════════════════════════════════════

    def _pillar7_width(self, pp: pd.DataFrame, obr: pd.DataFrame) -> Dict:
        # F36  Long diagonal switches — long sideway passes that bypass
        #      the midfield press and change the point of attack.
        long_diag = pp.loc[
            (pp["pass_range"] == "long")
            & pp["pass_direction"].isin(SIDEWAY_DIRECTIONS)
        ].shape[0]

        # F37  Overlap/underlap runs received — fullback combinations
        #      where the overlapping or underlapping runner actually
        #      receives the ball, completing the 2v1 advantage.
        ov_ul_recv = obr.loc[
            obr["event_subtype"].isin(OVERLAP_UNDERLAP_SUBTYPES)
            & (obr["received"] == True)
        ].shape[0]

        # F38  Pulling wide / pulling half-space runs — off-ball
        #      movements that drag defenders out of central positions
        #      to create space for teammates through the middle.
        pull_runs = obr.loc[
            obr["event_subtype"].isin(PULLING_SUBTYPES)
        ].shape[0]

        return {
            "long_diagonal_switches": long_diag,
            "overlap_underlap_runs_received": ov_ul_recv,
            "pulling_wide_or_halfspace_runs": pull_runs,
        }


# ═══════════════════════════════════════════════════════════════════════════
# PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    """discover → load → compute → validate → export."""
    script_dir = Path(__file__).resolve().parent
    data_root = script_dir / "data"
    output_path = script_dir / "features.csv"

    banner = "=" * 65
    print(f"\n{banner}")
    print("  ELITE TACTICAL FEATURE ENGINEERING")
    print("  SkillCorner Open Data Hackathon")
    print(f"{banner}")

    # ── Discover matches ──────────────────────────────────────────────
    match_ids = discover_match_ids(data_root)
    print(f"\n📂  Discovered {len(match_ids)} matches: {match_ids}")

    # ── Process ───────────────────────────────────────────────────────
    frames: List[pd.DataFrame] = []
    for mid in match_ids:
        print(f"  ⚙️   Match {mid} …", end=" ", flush=True)
        events = load_dynamic_events(mid, data_root)
        phases = load_phases_of_play(mid, data_root)
        engine = TacticalFeatureEngine(events, phases, mid)
        frames.append(engine.compute_all())
        print("✅")

    df = pd.concat(frames, ignore_index=True)

    # ── Type casting ──────────────────────────────────────────────────
    float_cols = {
        "zone14_second_ball_duration",
        "total_possession_duration_attacking_third",
        "territorial_gain_distance",
        "total_separation_gained",
        "xthreat_accumulated",
        "xshot_accumulated",
    }
    for col in df.columns:
        if col in ("match_id", "team_id"):
            df[col] = df[col].astype(int)
        elif col not in float_cols:
            df[col] = df[col].fillna(0).astype(int)

    # ── Validation ────────────────────────────────────────────────────
    n_rows, n_cols = df.shape
    n_features = n_cols - 2  # minus match_id, team_id
    nulls = int(df.isnull().sum().sum())

    assert n_rows == len(match_ids) * 2, (
        f"Expected {len(match_ids) * 2} rows, got {n_rows}"
    )
    assert n_features == 38, f"Expected 38 features, got {n_features}"
    assert nulls == 0, f"Found {nulls} null values"

    # ── Export ─────────────────────────────────────────────────────────
    df.to_csv(output_path, index=False)

    print(f"\n{banner}")
    print(f"  ✅  Output     : {output_path}")
    print(f"  📊  Shape      : {n_rows} rows × {n_cols} columns")
    print(f"  ⚽  Features   : {n_features}")
    print(f"  🏟   Matches    : {df['match_id'].nunique()}")
    print(f"  🎯  Teams      : {df['team_id'].nunique()}")
    print(f"  ❌  Null cells : {nulls}")
    print(f"{banner}\n")

    # ── Preview ───────────────────────────────────────────────────────
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(df.head(4).to_string(index=False))
    print()


if __name__ == "__main__":
    main()
