"""
Data Pipeline — downloads and caches all free public datasets.

Sources used (all free, no API key required):
  NBA:
    - FiveThirtyEight NBA ELO (2000–present): ~60k rows of game results
      + pre-game ELO ratings + predictions
      URL: https://projects.fivethirtyeight.com/nba-model/nba_elo.csv
    - nba_api (official NBA.com stats, no key): team/player season stats,
      game logs per season (slow but comprehensive)

  Soccer:
    - football-data.co.uk free CSV files: historical match results +
      bookmaker odds for EPL, La Liga, Serie A, Bundesliga, Ligue 1.
      We download last 8 seasons per league (~5k rows per league).
    - FiveThirtyEight Soccer SPI: global club ratings + match predictions
      URL: https://projects.fivethirtyeight.com/soccer-api/club/spi_matches.csv

  Kalshi / Prediction markets:
    - Metaculus community predictions API (free public, no key)
    - FRED economic indicators via requests (key not needed for most series)
      Fallback: bundled fallback CSV of historical Fed decisions + CPI

All data is cached to app/ml/data/ as CSV/JSON.
"""

import os
import time
import json
import logging
import requests
import pandas as pd
from pathlib import Path
from typing import Optional

logger = logging.getLogger("mirofish.ml.data_pipeline")

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# --- URL constants -----------------------------------------------------------
# FiveThirtyEight data moved to GitHub after ABC acquisition
NBA_ELO_URL = "https://raw.githubusercontent.com/fivethirtyeight/data/master/nba-elo/nbaallelo.csv"
NBA_ELO_URL_ALT = "https://raw.githubusercontent.com/fivethirtyeight/data/master/nba-elo/nba_elo_latest.csv"
SOCCER_SPI_URL = "https://raw.githubusercontent.com/fivethirtyeight/data/master/soccer-spi/spi_matches.csv"
SOCCER_DATA_BASE = "https://www.football-data.co.uk/mmz4281"

# football-data.co.uk league codes → human-readable
SOCCER_LEAGUES = {
    "E0": "EPL",           # English Premier League
    "SP1": "LaLiga",       # Spanish La Liga
    "I1": "SerieA",        # Italian Serie A
    "D1": "Bundesliga",    # German Bundesliga
    "F1": "Ligue1",        # French Ligue 1
}

# Seasons to download (most recent 8 seasons)
SOCCER_SEASONS = ["2425", "2324", "2223", "2122", "2021", "1920", "1819", "1718"]

# NBA seasons to pull via nba_api
NBA_SEASONS = [
    "2024-25", "2023-24", "2022-23", "2021-22", "2020-21",
    "2019-20", "2018-19", "2017-18", "2016-17", "2015-16",
]

FRED_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _download(url: str, dest: Path, timeout: int = 30) -> bool:
    """Download URL to dest file. Returns True on success."""
    try:
        headers = {"User-Agent": "MiroFish-ML/1.0 (research; contact@mirofish.ai)"}
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        logger.info(f"Downloaded {url} → {dest.name} ({len(resp.content)//1024} KB)")
        return True
    except Exception as e:
        logger.warning(f"Failed to download {url}: {e}")
        return False


def _cached(path: Path, max_age_hours: int = 24) -> bool:
    """Return True if file exists and is fresh."""
    if not path.exists():
        return False
    age_hours = (time.time() - path.stat().st_mtime) / 3600
    return age_hours < max_age_hours


# ---------------------------------------------------------------------------
# NBA
# ---------------------------------------------------------------------------

def get_nba_elo() -> Optional[pd.DataFrame]:
    """
    NBA game dataset with ELO ratings.
    Primary source: nba_api (free, no key) — 10 seasons of game logs with
    computed ELO ratings and rolling features.
    FiveThirtyEight files are skipped (format changed, columns differ).
    """
    return _build_nba_from_api()


def _build_nba_from_api() -> Optional[pd.DataFrame]:
    """
    Build NBA game dataset via nba_api (free, no key required).
    Pulls last 10 seasons of game logs and formats like the ELO CSV.
    """
    dest = DATA_DIR / "nba_from_api.csv"
    if _cached(dest, max_age_hours=6):
        df = pd.read_csv(dest)
        df["date"] = pd.to_datetime(df["date"])
        logger.info(f"NBA from API (cached): {len(df):,} matchups")
        return df

    logger.info("Building NBA dataset from nba_api...")
    try:
        from nba_api.stats.endpoints import LeagueGameLog
        import time as _time

        frames = []
        for season in NBA_SEASONS:
            try:
                _time.sleep(1.2)  # rate limit
                log = LeagueGameLog(
                    season=season,
                    season_type_all_star="Regular Season",
                    player_or_team_abbreviation="T"
                )
                df = log.get_data_frames()[0]
                df["SEASON"] = season
                frames.append(df)
                logger.info(f"  {season}: {len(df)} team-game records")
            except Exception as e:
                logger.warning(f"  {season}: failed ({e})")

        if not frames:
            return None

        combined = pd.concat(frames, ignore_index=True)
        combined["GAME_DATE"] = pd.to_datetime(combined["GAME_DATE"])

        # Pivot home vs away into matchup rows
        combined["is_home"] = combined["MATCHUP"].str.contains("vs.", na=False)
        home = combined[combined["is_home"]].copy()
        away = combined[~combined["is_home"]].copy()

        home_cols = {"TEAM_ID": "team1_id", "TEAM_NAME": "team1", "WL": "wl1",
                     "PTS": "score1", "PLUS_MINUS": "pm1", "FG_PCT": "fg1",
                     "FG3_PCT": "fg3_1", "REB": "reb1", "AST": "ast1", "TOV": "tov1",
                     "GAME_DATE": "date", "SEASON": "season"}
        away_cols = {"TEAM_ID": "team2_id", "TEAM_NAME": "team2", "WL": "wl2",
                     "PTS": "score2", "PLUS_MINUS": "pm2", "FG_PCT": "fg2",
                     "FG3_PCT": "fg3_2", "REB": "reb2", "AST": "ast2", "TOV": "tov2"}

        h = home.rename(columns=home_cols)[list(home_cols.values()) + ["GAME_ID"]]
        a = away.rename(columns=away_cols)[list(away_cols.values()) + ["GAME_ID"]]

        matchups = h.merge(a, on="GAME_ID", how="inner")
        matchups["neutral"] = 0
        matchups["playoff"] = ""
        matchups["season"] = matchups["season"].str.split("-").str[0].astype(int)
        matchups["season_frac"] = (matchups["season"] - 2000) / 25.0
        matchups = matchups.sort_values("date").reset_index(drop=True)

        # ── Compute real ELO ratings from game results chronologically ──────────
        # K=20 standard, home court +100 ELO equivalent (empirical ~3.5 pts)
        K = 20
        HOME_ADV_ELO = 100
        elos = {}

        elo1_pre_list, elo2_pre_list, elo_prob1_list = [], [], []
        for _, row in matchups.iterrows():
            t1, t2 = row["team1"], row["team2"]
            e1 = elos.get(t1, 1500.0)
            e2 = elos.get(t2, 1500.0)
            # Home court advantage: add 100 to home team's effective ELO
            prob1 = 1.0 / (1.0 + 10.0 ** (-(e1 + HOME_ADV_ELO - e2) / 400.0))
            elo1_pre_list.append(e1)
            elo2_pre_list.append(e2)
            elo_prob1_list.append(prob1)
            # Update ELO (simple margin of victory multiplier: ~0.5 per extra point)
            mov = abs(row.get("score1", 0) - row.get("score2", 0))
            mov_mult = (mov ** 0.8) / (7.5 + 0.006 * abs(e1 - e2))
            outcome1 = 1.0 if row.get("wl1", "L") == "W" else 0.0
            delta = K * mov_mult * (outcome1 - prob1)
            elos[t1] = e1 + delta
            elos[t2] = e2 - delta

        matchups["elo1_pre"] = elo1_pre_list
        matchups["elo2_pre"] = elo2_pre_list
        matchups["elo_prob1"] = elo_prob1_list
        matchups["elo_diff"] = matchups["elo1_pre"] - matchups["elo2_pre"]

        # ── Rolling features: last 10 games win%, avg pts per team ────────────
        # Compute rolling stats on all team-game records (sorted chronologically)
        tg = combined[["GAME_ID", "GAME_DATE", "TEAM_ID", "WL", "PTS",
                        "PLUS_MINUS", "FG_PCT"]].copy()
        tg = tg.sort_values(["TEAM_ID", "GAME_DATE"]).copy()
        tg["win"] = (tg["WL"] == "W").astype(float)
        tg["TEAM_ID"] = tg["TEAM_ID"].astype(str)

        g = tg.groupby("TEAM_ID")
        tg["win_pct_L10"] = g["win"].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
        tg["win_pct_L5"]  = g["win"].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
        tg["pts_avg_L10"] = g["PTS"].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
        tg["pm_avg_L10"]  = g["PLUS_MINUS"].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
        tg["fg_pct_L10"]  = g["FG_PCT"].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())

        # Build home/away lookup: one row per GAME_ID per team
        tg_home = tg[["GAME_ID", "TEAM_ID", "win_pct_L10", "win_pct_L5",
                       "pts_avg_L10", "pm_avg_L10", "fg_pct_L10"]].rename(columns={
            "TEAM_ID": "team1_id",
            "win_pct_L10": "home_win_pct_L10", "win_pct_L5": "home_win_pct_L5",
            "pts_avg_L10": "home_pts_avg_L10", "pm_avg_L10": "home_pm_avg_L10",
            "fg_pct_L10": "home_fg_pct_L10"
        })
        tg_away = tg[["GAME_ID", "TEAM_ID", "win_pct_L10", "win_pct_L5",
                       "pts_avg_L10", "pm_avg_L10", "fg_pct_L10"]].rename(columns={
            "TEAM_ID": "team2_id",
            "win_pct_L10": "away_win_pct_L10", "win_pct_L5": "away_win_pct_L5",
            "pts_avg_L10": "away_pts_avg_L10", "pm_avg_L10": "away_pm_avg_L10",
            "fg_pct_L10": "away_fg_pct_L10"
        })

        # Merge using GAME_ID + team ID (avoids 4x row explosion from join)
        matchups["team1_id"] = matchups["team1_id"].astype(str)
        matchups["team2_id"] = matchups["team2_id"].astype(str)
        matchups = matchups.merge(tg_home, on=["GAME_ID", "team1_id"], how="left")
        matchups = matchups.merge(tg_away, on=["GAME_ID", "team2_id"], how="left")

        # Fill rolling stat NaNs with neutral values
        for col in ["home_win_pct_L10", "home_win_pct_L5", "away_win_pct_L10", "away_win_pct_L5"]:
            matchups[col] = matchups[col].fillna(0.5)
        for col in ["home_pts_avg_L10", "away_pts_avg_L10"]:
            matchups[col] = matchups[col].fillna(110.0)
        for col in ["home_pm_avg_L10", "away_pm_avg_L10"]:
            matchups[col] = matchups[col].fillna(0.0)
        for col in ["home_fg_pct_L10", "away_fg_pct_L10"]:
            matchups[col] = matchups[col].fillna(0.46)

        matchups["raptor_diff"] = 0.0
        matchups["carmelo_diff"] = 0.0

        matchups.to_csv(dest, index=False)
        logger.info(f"NBA from API: {len(matchups):,} matchups saved")
        return matchups

    except Exception as e:
        logger.error(f"nba_api fallback failed: {e}")
        return None


def get_nba_team_stats_from_api(season: str = "2024-25") -> Optional[pd.DataFrame]:
    """
    Pull per-game team stats via nba_api (free, no key).
    Returns all teams' season averages: pts, reb, ast, fg_pct, etc.
    Rate-limited — sleeps between calls.
    """
    try:
        from nba_api.stats.endpoints import LeagueGameLog, TeamEstimatedMetrics
        dest = DATA_DIR / f"nba_team_stats_{season.replace('-', '_')}.csv"
        if _cached(dest, max_age_hours=6):
            return pd.read_csv(dest)

        logger.info(f"Fetching NBA team stats via nba_api for {season}...")
        time.sleep(1)
        metrics = TeamEstimatedMetrics(
            season=season,
            season_type="Regular Season"
        )
        df = metrics.get_data_frames()[0]
        df.to_csv(dest, index=False)
        logger.info(f"NBA team stats {season}: {len(df)} teams")
        return df
    except Exception as e:
        logger.warning(f"nba_api TeamEstimatedMetrics failed: {e}")
        return None


def get_nba_game_log(season: str = "2024-25") -> Optional[pd.DataFrame]:
    """
    Pull game-by-game log for all teams via nba_api.
    Includes: TEAM_ID, GAME_ID, MATCHUP, WL, PTS, FG_PCT, FG3_PCT, FT_PCT,
              REB, AST, STL, BLK, TOV, PLUS_MINUS, etc.
    """
    try:
        from nba_api.stats.endpoints import LeagueGameLog
        dest = DATA_DIR / f"nba_game_log_{season.replace('-', '_')}.csv"
        if _cached(dest, max_age_hours=6):
            return pd.read_csv(dest)

        logger.info(f"Fetching NBA game log via nba_api for {season}...")
        time.sleep(1.5)
        log = LeagueGameLog(
            season=season,
            season_type_all_star="Regular Season",
            player_or_team_abbreviation="T"  # team level
        )
        df = log.get_data_frames()[0]
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
        df.to_csv(dest, index=False)
        logger.info(f"NBA game log {season}: {len(df):,} team-game records")
        return df
    except Exception as e:
        logger.warning(f"nba_api LeagueGameLog failed: {e}")
        return None


def get_nba_all_seasons_game_logs() -> Optional[pd.DataFrame]:
    """
    Load all season game logs and concatenate into one DataFrame.
    Uses cached files if available.
    """
    frames = []
    for season in NBA_SEASONS:
        df = get_nba_game_log(season)
        if df is not None:
            df["SEASON"] = season
            frames.append(df)
        time.sleep(0.6)  # be polite to NBA API
    if frames:
        combined = pd.concat(frames, ignore_index=True)
        dest = DATA_DIR / "nba_all_game_logs.csv"
        combined.to_csv(dest, index=False)
        logger.info(f"NBA combined game logs: {len(combined):,} records")
        return combined
    return None


# ---------------------------------------------------------------------------
# Soccer
# ---------------------------------------------------------------------------

def get_soccer_league_data(league_code: str = "E0", season: str = "2324") -> Optional[pd.DataFrame]:
    """
    Download a single season CSV from football-data.co.uk.
    Includes: HomeTeam, AwayTeam, FTHG, FTAG, FTR (result: H/D/A),
              B365H, B365D, B365A (Bet365 odds), shots, corners, etc.
    """
    url = f"{SOCCER_DATA_BASE}/{season}/{league_code}.csv"
    league_name = SOCCER_LEAGUES.get(league_code, league_code)
    dest = DATA_DIR / f"soccer_{league_name}_{season}.csv"

    if not _cached(dest, max_age_hours=24):
        if not _download(url, dest):
            return None
    try:
        df = pd.read_csv(dest, encoding="latin1")
        df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])
        df["league"] = league_name
        df["season"] = season
        return df
    except Exception as e:
        logger.error(f"Failed to parse {dest}: {e}")
        return None


def get_soccer_all_leagues() -> Optional[pd.DataFrame]:
    """
    Download and concatenate data for all configured leagues and seasons.
    Total: ~5 leagues × 8 seasons = ~40 CSVs → ~20k+ rows.
    """
    dest = DATA_DIR / "soccer_all_leagues.csv"
    if _cached(dest, max_age_hours=24):
        df = pd.read_csv(dest, low_memory=False)
        logger.info(f"Soccer all leagues (cached): {len(df):,} matches")
        return df

    frames = []
    for league_code in SOCCER_LEAGUES:
        for season in SOCCER_SEASONS:
            df = get_soccer_league_data(league_code, season)
            if df is not None and len(df) > 0:
                frames.append(df)
            time.sleep(0.3)

    if not frames:
        logger.error("No soccer data downloaded")
        return None

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(dest, index=False)
    logger.info(f"Soccer all leagues: {len(combined):,} matches from {len(frames)} files")
    return combined


def get_soccer_spi() -> Optional[pd.DataFrame]:
    """
    FiveThirtyEight Soccer SPI — club ratings and match predictions.
    Columns: season, date, league_id, league, team1, team2,
             spi1, spi2, prob1, prob2, probtie, proj_score1, proj_score2,
             score1, score2, xg1, xg2, nsxg1, nsxg2, adj_score1, adj_score2
    """
    dest = DATA_DIR / "soccer_spi_matches.csv"
    if not _cached(dest, max_age_hours=24):
        _download(SOCCER_SPI_URL, dest)
    if not dest.exists():
        return None
    try:
        # Check if downloaded file is HTML or malformed
        with open(dest, "r", errors="ignore") as f:
            first_line = f.readline()
        if first_line.strip().startswith("<") or "<!DOCTYPE" in first_line:
            logger.warning("Soccer SPI file is HTML — skipping SPI (optional dataset)")
            dest.unlink(missing_ok=True)
            return None
        df = pd.read_csv(dest, low_memory=False, on_bad_lines="skip")
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["score1", "score2"])  # completed matches only
        logger.info(f"Soccer SPI: {len(df):,} matches loaded")
        return df
    except Exception as e:
        logger.warning(f"Soccer SPI parse failed (optional, skipping): {e}")
        return None


# ---------------------------------------------------------------------------
# Kalshi / Economic indicators
# ---------------------------------------------------------------------------

def get_fred_series(series_id: str) -> Optional[pd.DataFrame]:
    """
    Download a FRED economic time series (free, no API key).
    Examples:
      FEDFUNDS   - Federal Funds Rate
      CPIAUCSL   - CPI All Urban Consumers
      UNRATE     - Unemployment Rate
      T10YIE     - 10Y Breakeven Inflation
      DGS10      - 10-Year Treasury Rate
      UMCSENT    - University of Michigan Consumer Sentiment
    """
    dest = DATA_DIR / f"fred_{series_id}.csv"
    if _cached(dest, max_age_hours=48):
        return pd.read_csv(dest, parse_dates=["date"])

    url = f"{FRED_BASE}?id={series_id}"
    if _download(url, dest):
        df = pd.read_csv(dest)
        df.columns = ["date", series_id]
        df["date"] = pd.to_datetime(df["date"])
        df = df.dropna()
        df.to_csv(dest, index=False)
        return df
    return None


def get_all_economic_indicators() -> Optional[pd.DataFrame]:
    """
    Download and merge key FRED series for Kalshi economic question modeling.
    Returns a monthly time series DataFrame.
    """
    series_ids = [
        "FEDFUNDS",    # Federal Funds Rate (monthly)
        "CPIAUCSL",    # CPI All Urban (monthly)
        "UNRATE",      # Unemployment Rate (monthly)
        "DGS10",       # 10-Year Treasury yield (daily → resample)
        "T10YIE",      # 10Y Breakeven Inflation (daily → resample)
        "UMCSENT",     # Consumer Sentiment (monthly)
        "GDPC1",       # Real GDP (quarterly)
        "PCEPILFE",    # Core PCE Price Index (monthly)
    ]
    dest = DATA_DIR / "economic_indicators.csv"
    if _cached(dest, max_age_hours=48):
        return pd.read_csv(dest, parse_dates=["date"])

    frames = {}
    for sid in series_ids:
        df = get_fred_series(sid)
        if df is not None:
            df = df.set_index("date")[sid]
            frames[sid] = df
        time.sleep(0.5)

    if not frames:
        return None

    merged = pd.DataFrame(frames)
    merged = merged.resample("ME").last().ffill()
    merged = merged.reset_index().rename(columns={"index": "date"})
    merged.to_csv(dest, index=False)
    logger.info(f"Economic indicators: {len(merged):,} monthly rows")
    return merged


def get_metaculus_predictions(limit: int = 2000) -> Optional[pd.DataFrame]:
    """
    Download resolved binary predictions from Metaculus public API (no key).
    Returns: question, resolution (0/1), community_probability, ...
    Used to calibrate the Kalshi base-rate model.
    """
    dest = DATA_DIR / "metaculus_resolved.json"
    if _cached(dest, max_age_hours=48):
        with open(dest) as f:
            data = json.load(f)
    else:
        url = f"https://www.metaculus.com/api2/questions/?limit={limit}&resolved=true&type=binary&order_by=-resolve_time"
        try:
            resp = requests.get(url, timeout=30, headers={"User-Agent": "MiroFish-ML/1.0"})
            resp.raise_for_status()
            data = resp.json()
            with open(dest, "w") as f:
                json.dump(data, f)
        except Exception as e:
            logger.warning(f"Metaculus fetch failed: {e}")
            return None

    try:
        results = data.get("results", [])
        rows = []
        for q in results:
            comm_pred = q.get("community_prediction", {})
            latest = comm_pred.get("full", {}).get("q2") if isinstance(comm_pred, dict) else None
            resolution = q.get("resolution")
            if latest is not None and resolution in [0, 1]:
                rows.append({
                    "question": q.get("title", ""),
                    "category": q.get("category_tags", [None])[0] if q.get("category_tags") else "unknown",
                    "community_prob": float(latest),
                    "resolution": int(resolution),
                    "resolve_time": q.get("resolve_time"),
                    "close_time": q.get("close_time"),
                })
        df = pd.DataFrame(rows)
        logger.info(f"Metaculus: {len(df):,} resolved binary questions")
        return df
    except Exception as e:
        logger.error(f"Failed to parse Metaculus data: {e}")
        return None


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def download_all(verbose: bool = True) -> dict:
    """
    Download all datasets. Returns dict of {name: row_count or None}.
    Safe to run repeatedly — uses caching.
    """
    results = {}
    print("Downloading NBA ELO data...")
    df = get_nba_elo()
    results["nba_elo"] = len(df) if df is not None else None

    print("Downloading soccer (all leagues, all seasons)...")
    df = get_soccer_all_leagues()
    results["soccer"] = len(df) if df is not None else None

    print("Downloading soccer SPI ratings...")
    df = get_soccer_spi()
    results["soccer_spi"] = len(df) if df is not None else None

    print("Downloading economic indicators...")
    df = get_all_economic_indicators()
    results["economic"] = len(df) if df is not None else None

    print("Downloading Metaculus predictions...")
    df = get_metaculus_predictions()
    results["metaculus"] = len(df) if df is not None else None

    if verbose:
        print("\n=== Download Summary ===")
        for k, v in results.items():
            status = f"{v:,} rows" if v is not None else "FAILED"
            print(f"  {k:20s}: {status}")
    return results
