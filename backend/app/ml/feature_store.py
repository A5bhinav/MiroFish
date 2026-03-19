"""
Feature Store — all feature engineering for NBA, Soccer, and Kalshi models.

Design principles:
  1. All features have documented sports-analytics rationale.
  2. Rolling windows computed on sorted chronological data only
     (no look-ahead bias).
  3. Feature names are stable — production inference uses the same names.
  4. Missing values handled explicitly (fill strategies documented).

NBA Factor Research:
  Home court: ~3.5 pts advantage (adjusted post-COVID to ~2.5 pts)
  Back-to-back: ~3 pts disadvantage for the team on B2B
  Rest: Each additional rest day worth ~0.5–1 pt up to 3 days
  ELO diff: Strong predictor; 100-pt diff ≈ 60% win probability
  Net rating: Best single-team stat predictor
  Pace mismatch: Fast vs. slow teams creates variance (over/under relevant)

Soccer Factor Research:
  Dixon-Coles model (1997): Poisson with low-score correction best baseline
  xG: Better predictor than actual goals for future performance
  Market odds: Nearly unbeatable for large leagues (Pinnacle < 2% vig)
  Home advantage: ~0.4 goals / match in European leagues
  Form window: Last 5 matches most predictive (diminishing returns beyond 10)
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict


# =============================================================================
# NBA Features
# =============================================================================

class NBAFeatureEngineer:
    """
    Builds per-game feature vectors from raw game-log DataFrames.
    Each row in the output represents one matchup (home vs. away) with
    all features computed from data available *before* that game.
    """

    # Features generated (in order — must match training and inference)
    FEATURE_NAMES = [
        # --- Team A (home) rolling stats ---
        "home_win_pct_L10",       # Win % last 10 games
        "home_win_pct_L5",        # Win % last 5 games
        "home_pts_avg_L10",       # Avg points scored last 10
        "home_pts_against_avg_L10",  # Avg points allowed last 10
        "home_net_rtg_L10",       # Net rating last 10 (pts - pts_against)
        "home_plus_minus_L5",     # Avg +/- last 5 games
        "home_fg_pct_L10",        # FG% last 10
        "home_fg3_pct_L10",       # 3P% last 10
        "home_ft_pct_L10",        # FT% last 10
        "home_reb_avg_L10",       # Rebounds last 10
        "home_ast_avg_L10",       # Assists last 10
        "home_tov_avg_L10",       # Turnovers last 10
        "home_streak",            # Current win (+) or loss (-) streak
        "home_b2b",               # 1 if team is on back-to-back
        "home_rest_days",         # Days since last game (capped at 7)
        # --- Team B (away) rolling stats ---
        "away_win_pct_L10",
        "away_win_pct_L5",
        "away_pts_avg_L10",
        "away_pts_against_avg_L10",
        "away_net_rtg_L10",
        "away_plus_minus_L5",
        "away_fg_pct_L10",
        "away_fg3_pct_L10",
        "away_ft_pct_L10",
        "away_reb_avg_L10",
        "away_ast_avg_L10",
        "away_tov_avg_L10",
        "away_streak",
        "away_b2b",
        "away_rest_days",
        # --- Matchup differentials ---
        "net_rtg_diff",           # home_net_rtg - away_net_rtg (key predictor)
        "win_pct_diff",           # home_win_pct - away_win_pct
        "pts_avg_diff",           # home pts avg - away pts avg
        "rest_advantage",         # home_rest_days - away_rest_days
        # --- ELO (from FiveThirtyEight data) ---
        "elo_diff",               # home_elo - away_elo
        "elo_prob_home",          # FTE pre-game home win probability
        # --- Pace / style matchup ---
        "avg_total_pts_L10",      # (home + away avg pts) — proxy for pace
        # --- Context ---
        "is_playoffs",            # 0/1
        "season_week",            # 1–25 (week of season, normalized)
    ]

    @staticmethod
    def build_team_rolling(game_log: pd.DataFrame) -> pd.DataFrame:
        """
        For each team, compute rolling stats over last 5 and 10 games.
        Expects columns: TEAM_ID, GAME_DATE, WL, PTS, FG_PCT, FG3_PCT,
                         FT_PCT, REB, AST, TOV, PLUS_MINUS, MATCHUP, SEASON.
        Returns enriched DataFrame with rolling columns per team.
        """
        game_log = game_log.sort_values(["TEAM_ID", "GAME_DATE"]).copy()
        game_log["WIN"] = (game_log["WL"] == "W").astype(int)

        g = game_log.groupby("TEAM_ID")

        for window, label in [(10, "L10"), (5, "L5")]:
            game_log[f"win_pct_{label}"] = g["WIN"].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )
            game_log[f"pts_avg_{label}"] = g["PTS"].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )
            game_log[f"plus_minus_{label}"] = g["PLUS_MINUS"].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )

        for col in ["FG_PCT", "FG3_PCT", "FT_PCT", "REB", "AST", "TOV", "PTS"]:
            if col in game_log.columns:
                game_log[f"{col.lower()}_avg_L10"] = g[col].transform(
                    lambda x: x.shift(1).rolling(10, min_periods=1).mean()
                )

        # Opponent points (points against) — need to join with opponent record
        game_log["is_home"] = game_log["MATCHUP"].str.contains("vs.").astype(int)

        # Win streak
        def streak(wins):
            s = []
            cur = 0
            for w in wins:
                if w == 1:
                    cur = max(1, cur + 1)
                else:
                    cur = min(-1, cur - 1)
                s.append(cur)
            return s

        game_log["streak"] = g["WIN"].transform(
            lambda x: pd.Series(streak(x.shift(1).fillna(0).astype(int).tolist()), index=x.index)
        )

        # Rest days
        game_log["prev_game_date"] = g["GAME_DATE"].transform(
            lambda x: x.shift(1)
        )
        game_log["rest_days"] = (
            pd.to_datetime(game_log["GAME_DATE"]) - pd.to_datetime(game_log["prev_game_date"])
        ).dt.days.clip(0, 7).fillna(3)  # Default 3 days for first game

        game_log["b2b"] = (game_log["rest_days"] == 1).astype(int)

        return game_log

    @staticmethod
    def build_matchup_features(game_log: pd.DataFrame,
                                elo_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Pivot team game log into per-matchup rows.
        Each row has home_* and away_* features for one game.
        """
        enriched = NBAFeatureEngineer.build_team_rolling(game_log)

        # Identify game pairs: each game appears twice (home + away team)
        enriched["GAME_DATE"] = pd.to_datetime(enriched["GAME_DATE"])
        home = enriched[enriched["is_home"] == 1].copy()
        away = enriched[enriched["is_home"] == 0].copy()

        # Parse GAME_ID to join
        home = home.rename(columns=lambda c: f"home_{c}" if c not in ["GAME_ID", "GAME_DATE", "SEASON", "is_playoffs"] else c)
        away = away.rename(columns=lambda c: f"away_{c}" if c not in ["GAME_ID", "GAME_DATE", "SEASON", "is_playoffs"] else c)

        matchups = home.merge(away, on=["GAME_ID", "GAME_DATE"], suffixes=("", "_dup"))
        # Drop duplicate columns
        matchups = matchups[[c for c in matchups.columns if not c.endswith("_dup")]]

        # Merge ELO data if available
        if elo_df is not None:
            elo_df = elo_df.copy()
            elo_df["date"] = pd.to_datetime(elo_df["date"])
            matchups = matchups.merge(
                elo_df[["date", "team1", "team2", "elo1_pre", "elo2_pre", "elo_prob1", "playoff"]].rename(
                    columns={"date": "GAME_DATE", "elo1_pre": "elo_home", "elo2_pre": "elo_away",
                             "elo_prob1": "elo_prob_home", "playoff": "is_elo_playoff"}
                ),
                on="GAME_DATE", how="left"
            )
            matchups["elo_diff"] = matchups["elo_home"].fillna(1500) - matchups["elo_away"].fillna(1500)
            matchups["elo_prob_home"] = matchups["elo_prob_home"].fillna(0.5)
        else:
            matchups["elo_diff"] = 0.0
            matchups["elo_prob_home"] = 0.5

        # Differentials
        matchups["net_rtg_diff"] = matchups.get("home_plus_minus_L10", 0) - matchups.get("away_plus_minus_L10", 0)
        matchups["win_pct_diff"] = matchups.get("home_win_pct_L10", 0.5) - matchups.get("away_win_pct_L10", 0.5)
        matchups["pts_avg_diff"] = matchups.get("home_pts_avg_L10", 110) - matchups.get("away_pts_avg_L10", 110)
        matchups["rest_advantage"] = matchups.get("home_rest_days", 2) - matchups.get("away_rest_days", 2)
        matchups["avg_total_pts_L10"] = matchups.get("home_pts_avg_L10", 110) + matchups.get("away_pts_avg_L10", 110)

        # Season week
        matchups["season_start"] = matchups.groupby("SEASON")["GAME_DATE"].transform("min")
        matchups["season_week"] = ((matchups["GAME_DATE"] - matchups["season_start"]).dt.days // 7 + 1).clip(1, 25)

        # Targets
        matchups["home_win"] = (matchups["home_WIN"].fillna(0)).astype(int) if "home_WIN" in matchups.columns else None
        matchups["home_pts"] = matchups.get("home_PTS", np.nan)
        matchups["away_pts"] = matchups.get("away_PTS", np.nan)
        matchups["total_pts"] = matchups["home_pts"].fillna(0) + matchups["away_pts"].fillna(0)
        matchups["point_diff"] = matchups["home_pts"].fillna(0) - matchups["away_pts"].fillna(0)

        return matchups

    @staticmethod
    def build_features_from_elo(elo_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Build feature matrix directly from FiveThirtyEight ELO CSV.
        This is the primary training path (most data, best quality).

        Features: elo_diff, elo_prob_home, recent form proxied by raptor_delta (if available),
                  neutral court flag, playoff flag.
        Target: team1_win (1 if team1 wins).
        """
        df = elo_df.copy()
        df = df.dropna(subset=["score1", "score2", "elo1_pre", "elo2_pre"])
        df["date"] = pd.to_datetime(df["date"])

        # ELO-based features
        df["elo_diff"] = df["elo1_pre"] - df["elo2_pre"]
        df["elo_prob_home"] = df["elo_prob1"].fillna(0.5)

        # RAPTOR features (available in newer ELO file)
        if "raptor1_pre" in df.columns:
            df["raptor_diff"] = df["raptor1_pre"].fillna(0) - df["raptor2_pre"].fillna(0)
        else:
            df["raptor_diff"] = 0.0

        if "carmelo1_pre" in df.columns:
            df["carmelo_diff"] = df["carmelo1_pre"].fillna(0) - df["carmelo2_pre"].fillna(0)
        else:
            df["carmelo_diff"] = 0.0

        df["is_neutral"] = df["neutral"].astype(int) if "neutral" in df.columns else 0
        df["is_playoffs"] = (df["playoff"].notna() & (df["playoff"] != "")).astype(int) if "playoff" in df.columns else 0
        df["season"] = df["season"].astype(int)
        df["season_frac"] = (df["season"] - 2000) / 25.0  # Temporal trend feature

        # Score features (targets)
        df["home_win"] = (df["score1"] > df["score2"]).astype(int)
        df["point_diff"] = df["score1"] - df["score2"]
        df["total_pts"] = df["score1"] + df["score2"]

        feature_cols = [
            "elo_diff", "elo_prob_home", "raptor_diff", "carmelo_diff",
            "is_neutral", "is_playoffs", "season_frac"
        ]

        # Use rolling stats from nba_api data if available (much richer signal)
        rolling_cols = [
            "home_win_pct_L10", "home_win_pct_L5", "home_pts_avg_L10",
            "home_pm_avg_L10", "home_fg_pct_L10",
            "away_win_pct_L10", "away_win_pct_L5", "away_pts_avg_L10",
            "away_pm_avg_L10", "away_fg_pct_L10",
        ]
        available_rolling = [c for c in rolling_cols if c in df.columns]
        if available_rolling:
            feature_cols = feature_cols + available_rolling
            # Derived differentials
            if "home_win_pct_L10" in df.columns and "away_win_pct_L10" in df.columns:
                df["win_pct_diff_L10"] = df["home_win_pct_L10"] - df["away_win_pct_L10"]
                df["pm_diff_L10"] = df.get("home_pm_avg_L10", 0) - df.get("away_pm_avg_L10", 0)
                feature_cols += ["win_pct_diff_L10", "pm_diff_L10"]

        X = df[feature_cols].fillna(0)
        return df, X


# =============================================================================
# Soccer Features
# =============================================================================

class SoccerFeatureEngineer:
    """
    Builds match feature vectors from football-data.co.uk CSVs + SPI data.

    Key insight: The market odds (Pinnacle/Bet365) are already a near-optimal
    predictor. We augment with form, H2H, and xG to try to find edges.
    """

    FEATURE_NAMES = [
        # --- Home team form (last N matches) ---
        "home_pts_per_game_L5",        # (3W+1D+0L)/5 over last 5 matches
        "home_pts_per_game_L10",
        "home_goals_scored_avg_L5",
        "home_goals_conceded_avg_L5",
        "home_goal_diff_L5",
        "home_shots_on_target_avg_L5",
        "home_win_streak",             # Current win/unbeaten streak
        # --- Away team form ---
        "away_pts_per_game_L5",
        "away_pts_per_game_L10",
        "away_goals_scored_avg_L5",
        "away_goals_conceded_avg_L5",
        "away_goal_diff_L5",
        "away_shots_on_target_avg_L5",
        "away_win_streak",
        # --- Differentials ---
        "pts_per_game_diff",           # home_pts - away_pts
        "goals_scored_diff",
        "goals_conceded_diff",
        # --- H2H ---
        "h2h_home_win_rate",           # Home team's win rate vs this opponent (last 5 H2H)
        "h2h_avg_goals",               # Avg total goals in last 5 H2H
        # --- Market odds (best predictor for big leagues) ---
        "b365_home_implied_prob",      # 1 / B365H (normalized to sum to 1)
        "b365_draw_implied_prob",
        "b365_away_implied_prob",
        "market_home_minus_draw_prob", # Market signal: how much is home favored over draw
        # --- SPI ratings (if available) ---
        "spi_diff",                    # home_spi - away_spi
        "spi_prob_home",               # FTE pre-game home win probability
        # --- Context ---
        "is_top6",                     # 1 if both teams are top-6 quality (big games)
        "days_since_last_match_home",
        "days_since_last_match_away",
        "rest_advantage",
    ]

    @staticmethod
    def _team_form(df_matches: pd.DataFrame, team: str, before_date: pd.Timestamp,
                    window: int = 5) -> Dict:
        """
        Compute form stats for `team` in the `window` matches before `before_date`.
        Considers both home and away matches.
        """
        is_home = df_matches["HomeTeam"] == team
        is_away = df_matches["AwayTeam"] == team
        relevant = df_matches[(is_home | is_away) & (df_matches["date"] < before_date)]
        relevant = relevant.sort_values("date").tail(window)

        if len(relevant) == 0:
            return {
                "pts_per_game": 1.2,  # league average
                "goals_scored": 1.3,
                "goals_conceded": 1.3,
                "goal_diff": 0.0,
                "shots_on_target": 5.0,
                "win_streak": 0,
                "last_match_date": None,
            }

        pts, goals_for, goals_against, shots = [], [], [], []
        for _, row in relevant.iterrows():
            if row["HomeTeam"] == team:
                gf, ga = row.get("FTHG", row.get("GF", 0)), row.get("FTAG", row.get("GA", 0))
                st = row.get("HST", row.get("SoT", 5))
            else:
                gf, ga = row.get("FTAG", row.get("GA", 0)), row.get("FTHG", row.get("GF", 0))
                st = row.get("AST", row.get("SoT", 5))

            result_str = row.get("FTR", "D")
            if (row["HomeTeam"] == team and result_str == "H") or \
               (row["AwayTeam"] == team and result_str == "A"):
                pts.append(3)
            elif result_str == "D":
                pts.append(1)
            else:
                pts.append(0)

            goals_for.append(float(gf) if pd.notna(gf) else 1.0)
            goals_against.append(float(ga) if pd.notna(ga) else 1.0)
            shots.append(float(st) if pd.notna(st) else 5.0)

        # Win streak
        streak = 0
        for p in reversed(pts):
            if p == 3:
                streak += 1
            else:
                break

        return {
            "pts_per_game": np.mean(pts),
            "goals_scored": np.mean(goals_for),
            "goals_conceded": np.mean(goals_against),
            "goal_diff": np.mean(goals_for) - np.mean(goals_against),
            "shots_on_target": np.mean(shots),
            "win_streak": streak,
            "last_match_date": relevant["date"].max(),
        }

    @staticmethod
    def _h2h(df_matches: pd.DataFrame, home: str, away: str,
              before_date: pd.Timestamp, window: int = 5) -> Dict:
        """
        Head-to-head record between home and away teams (last `window` meetings).
        """
        h2h = df_matches[
            ((df_matches["HomeTeam"] == home) & (df_matches["AwayTeam"] == away)) |
            ((df_matches["HomeTeam"] == away) & (df_matches["AwayTeam"] == home))
        ]
        h2h = h2h[h2h["date"] < before_date].sort_values("date").tail(window)

        if len(h2h) == 0:
            return {"home_win_rate": 0.4, "avg_goals": 2.5}

        home_wins = 0
        total_goals = []
        for _, row in h2h.iterrows():
            ftr = row.get("FTR", "D")
            if row["HomeTeam"] == home:
                if ftr == "H": home_wins += 1
            else:
                if ftr == "A": home_wins += 1
            g = (row.get("FTHG", 0) or 0) + (row.get("FTAG", 0) or 0)
            total_goals.append(float(g))

        return {
            "home_win_rate": home_wins / len(h2h),
            "avg_goals": np.mean(total_goals),
        }

    @staticmethod
    def _odds_to_prob(h: float, d: float, a: float) -> Tuple[float, float, float]:
        """Convert decimal odds to implied probabilities, normalize to sum to 1."""
        try:
            ph = 1.0 / h if h and h > 1 else 0.33
            pd_ = 1.0 / d if d and d > 1 else 0.33
            pa = 1.0 / a if a and a > 1 else 0.33
            total = ph + pd_ + pa
            return ph / total, pd_ / total, pa / total
        except Exception:
            return 0.45, 0.25, 0.30

    @classmethod
    def build_features(cls, df: pd.DataFrame,
                        spi_df: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Build feature matrix from concatenated football-data.co.uk DataFrame.
        Returns (X, y) where y has columns: result (H/D/A), home_win, over_2_5.
        """
        if "Date" in df.columns and "date" not in df.columns:
            df["date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
        elif "date" not in df.columns:
            df["date"] = pd.NaT
        df = df.dropna(subset=["date", "HomeTeam", "AwayTeam"]).copy()
        df = df.sort_values("date")

        rows = []
        for idx, row in df.iterrows():
            home, away = row["HomeTeam"], row["AwayTeam"]
            before = row["date"]

            hf = cls._team_form(df, home, before, 5)
            af = cls._team_form(df, away, before, 5)
            hf10 = cls._team_form(df, home, before, 10)
            af10 = cls._team_form(df, away, before, 10)
            h2h = cls._h2h(df, home, away, before)

            # Odds
            b365h = row.get("B365H", row.get("BbAvH", 2.0))
            b365d = row.get("B365D", row.get("BbAvD", 3.5))
            b365a = row.get("B365A", row.get("BbAvA", 3.0))
            ph, pd_, pa = cls._odds_to_prob(b365h, b365d, b365a)

            # Rest days
            hl = hf["last_match_date"]
            al = af["last_match_date"]
            h_rest = (before - hl).days if hl else 7
            a_rest = (before - al).days if al else 7
            h_rest = min(h_rest, 14)
            a_rest = min(a_rest, 14)

            feat = {
                # Home form
                "home_pts_per_game_L5": hf["pts_per_game"],
                "home_pts_per_game_L10": hf10["pts_per_game"],
                "home_goals_scored_avg_L5": hf["goals_scored"],
                "home_goals_conceded_avg_L5": hf["goals_conceded"],
                "home_goal_diff_L5": hf["goal_diff"],
                "home_shots_on_target_avg_L5": hf["shots_on_target"],
                "home_win_streak": hf["win_streak"],
                # Away form
                "away_pts_per_game_L5": af["pts_per_game"],
                "away_pts_per_game_L10": af10["pts_per_game"],
                "away_goals_scored_avg_L5": af["goals_scored"],
                "away_goals_conceded_avg_L5": af["goals_conceded"],
                "away_goal_diff_L5": af["goal_diff"],
                "away_shots_on_target_avg_L5": af["shots_on_target"],
                "away_win_streak": af["win_streak"],
                # Differentials
                "pts_per_game_diff": hf["pts_per_game"] - af["pts_per_game"],
                "goals_scored_diff": hf["goals_scored"] - af["goals_scored"],
                "goals_conceded_diff": hf["goals_conceded"] - af["goals_conceded"],
                # H2H
                "h2h_home_win_rate": h2h["home_win_rate"],
                "h2h_avg_goals": h2h["avg_goals"],
                # Odds
                "b365_home_implied_prob": ph,
                "b365_draw_implied_prob": pd_,
                "b365_away_implied_prob": pa,
                "market_home_minus_draw_prob": ph - pd_,
                # Context
                "is_top6": 0,  # Could be enriched with league table position
                "days_since_last_match_home": h_rest,
                "days_since_last_match_away": a_rest,
                "rest_advantage": h_rest - a_rest,
                # Metadata
                "_date": before,
                "_home": home,
                "_away": away,
                "_league": row.get("league", ""),
                "_season": row.get("season", ""),
            }

            # SPI enrichment
            if spi_df is not None:
                spi_match = spi_df[
                    (spi_df["date"] == before) &
                    (spi_df["team1"].str.contains(home[:5], na=False) |
                     spi_df["team2"].str.contains(home[:5], na=False))
                ]
                if len(spi_match) > 0:
                    s = spi_match.iloc[0]
                    feat["spi_diff"] = s.get("spi1", 50) - s.get("spi2", 50)
                    feat["spi_prob_home"] = s.get("prob1", 0.45)
                else:
                    feat["spi_diff"] = 0.0
                    feat["spi_prob_home"] = ph
            else:
                feat["spi_diff"] = 0.0
                feat["spi_prob_home"] = ph

            # Target
            ftr = row.get("FTR")
            fthg = row.get("FTHG", 0) or 0
            ftag = row.get("FTAG", 0) or 0
            feat["_result"] = ftr  # H, D, A
            feat["_home_win"] = 1 if ftr == "H" else 0
            feat["_over_2_5"] = 1 if (fthg + ftag) > 2.5 else 0
            feat["_total_goals"] = fthg + ftag

            rows.append(feat)

        df_out = pd.DataFrame(rows)
        meta_cols = [c for c in df_out.columns if c.startswith("_")]
        feat_cols = [c for c in df_out.columns if not c.startswith("_")]
        return df_out[feat_cols].fillna(0), df_out[meta_cols]


# =============================================================================
# Kalshi Features
# =============================================================================

class KalshiFeatureEngineer:
    """
    Feature engineering for Kalshi binary prediction markets.

    Categories and their key features:
      1. Fed/Monetary policy  → FEDFUNDS level, trend, futures implied rate
      2. Economic threshold   → CPI/PCE level, trend, YoY change, deviation from target
      3. Political            → polling average, time to event, historical base rate
      4. Sports               → ELO/SPI-based probability (use our models)
    """

    @staticmethod
    def build_econ_features(econ_df: pd.DataFrame, date: pd.Timestamp,
                             question_type: str = "cpi") -> Dict:
        """
        Build features for economic prediction questions.
        question_type: 'cpi', 'fedfunds', 'unemployment', 'gdp'
        """
        econ_df = econ_df.copy()
        econ_df = econ_df[econ_df["date"] <= date].tail(24)  # Last 24 months

        if len(econ_df) == 0:
            return {"level": 0, "mom_change": 0, "yoy_change": 0, "trend": 0, "above_target": 0}

        col_map = {
            "cpi": "CPIAUCSL",
            "fedfunds": "FEDFUNDS",
            "unemployment": "UNRATE",
            "pce": "PCEPILFE",
        }
        col = col_map.get(question_type, "CPIAUCSL")
        if col not in econ_df.columns:
            return {"level": 0, "mom_change": 0, "yoy_change": 0, "trend": 0, "above_target": 0}

        series = econ_df[col].dropna()
        if len(series) < 2:
            return {"level": float(series.iloc[-1]) if len(series) > 0 else 0,
                    "mom_change": 0, "yoy_change": 0, "trend": 0, "above_target": 0}

        level = float(series.iloc[-1])
        mom = float(series.pct_change().iloc[-1]) * 100 if len(series) >= 2 else 0.0
        yoy = float(series.pct_change(12).iloc[-1]) * 100 if len(series) >= 13 else 0.0

        # 3-month trend (slope)
        recent = series.tail(3).values
        trend = float(np.polyfit(range(len(recent)), recent, 1)[0]) if len(recent) == 3 else 0.0

        # Above Fed 2% target (for inflation questions)
        above_target = 1 if (question_type in ("cpi", "pce") and yoy > 2.5) else 0

        return {
            "level": level,
            "mom_change": mom,
            "yoy_change": yoy,
            "trend": trend,
            "above_target": above_target,
        }

    @staticmethod
    def build_calibration_features(community_prob: float, time_to_close_days: float,
                                    category: str = "economics") -> Dict:
        """
        Meta-features for calibration model — takes community probability and context.
        """
        cat_map = {"economics": 0, "politics": 1, "sports": 2, "science": 3, "other": 4}
        return {
            "community_prob": community_prob,
            "log_odds": np.log(community_prob / (1 - community_prob + 1e-6) + 1e-6),
            "time_to_close_log": np.log1p(time_to_close_days),
            "category_id": cat_map.get(category, 4),
            "prob_extremity": abs(community_prob - 0.5),  # how far from 50/50
            "is_high_confidence": 1 if community_prob > 0.8 or community_prob < 0.2 else 0,
        }
