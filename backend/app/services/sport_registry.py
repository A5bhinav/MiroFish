"""
Sport Registry — catalog of all supported sports with their API mappings.

Every sport entry defines:
  - espn_sport / espn_league  : path for ESPN's unofficial public API (no key)
  - odds_key                  : The Odds API v4 sport key (existing ODDS_API_KEY)
  - data_source               : preferred fetcher ("balldontlie", "api_football",
                                "mlb_api", "espn")
  - api_football_id           : API-Football league ID (soccer only)
  - bet_types                 : markets available for this sport
  - has_draws                 : whether draws are possible (affects probability output)
  - season_format             : "year" (2025) or "year_range" (2024-25)
"""

from typing import Optional, Tuple

# ---------------------------------------------------------------------------
# Master catalog
# ---------------------------------------------------------------------------

SPORT_CATALOG = {
    # -----------------------------------------------------------------------
    # American Football
    # -----------------------------------------------------------------------
    "nfl": {
        "name": "NFL",
        "display": "American Football — NFL",
        "category": "american_football",
        "espn_sport": "football",
        "espn_league": "nfl",
        "odds_key": "americanfootball_nfl",
        "bet_types": ["moneyline", "spread", "total"],
        "season_format": "year",
        "has_draws": False,
        "data_source": "espn",
    },
    "ncaaf": {
        "name": "NCAAF",
        "display": "College Football — NCAA",
        "category": "american_football",
        "espn_sport": "football",
        "espn_league": "college-football",
        "odds_key": "americanfootball_ncaaf",
        "bet_types": ["moneyline", "spread", "total"],
        "season_format": "year",
        "has_draws": False,
        "data_source": "espn",
    },
    # -----------------------------------------------------------------------
    # Basketball
    # -----------------------------------------------------------------------
    "nba": {
        "name": "NBA",
        "display": "Basketball — NBA",
        "category": "basketball",
        "espn_sport": "basketball",
        "espn_league": "nba",
        "odds_key": "basketball_nba",
        "bet_types": ["moneyline", "spread", "total", "props"],
        "season_format": "year_range",
        "has_draws": False,
        "data_source": "balldontlie",   # preferred; ESPN is fallback
    },
    "ncaab": {
        "name": "NCAAB",
        "display": "College Basketball — NCAA",
        "category": "basketball",
        "espn_sport": "basketball",
        "espn_league": "mens-college-basketball",
        "odds_key": "basketball_ncaab",
        "bet_types": ["moneyline", "spread", "total"],
        "season_format": "year_range",
        "has_draws": False,
        "data_source": "espn",
    },
    # -----------------------------------------------------------------------
    # Baseball
    # -----------------------------------------------------------------------
    "mlb": {
        "name": "MLB",
        "display": "Baseball — MLB",
        "category": "baseball",
        "espn_sport": "baseball",
        "espn_league": "mlb",
        "odds_key": "baseball_mlb",
        "bet_types": ["moneyline", "spread", "total"],
        "season_format": "year",
        "has_draws": False,
        "data_source": "mlb_api",       # official MLB Stats API, no key
    },
    # -----------------------------------------------------------------------
    # Ice Hockey
    # -----------------------------------------------------------------------
    "nhl": {
        "name": "NHL",
        "display": "Ice Hockey — NHL",
        "category": "hockey",
        "espn_sport": "hockey",
        "espn_league": "nhl",
        "odds_key": "icehockey_nhl",
        "bet_types": ["moneyline", "spread", "total"],
        "season_format": "year_range",
        "has_draws": False,
        "data_source": "espn",
    },
    # -----------------------------------------------------------------------
    # Soccer / Football
    # -----------------------------------------------------------------------
    "epl": {
        "name": "Premier League",
        "display": "Soccer — English Premier League",
        "category": "soccer",
        "espn_sport": "soccer",
        "espn_league": "eng.1",
        "odds_key": "soccer_epl",
        "api_football_id": 39,
        "bet_types": ["moneyline", "total"],
        "season_format": "year",
        "has_draws": True,
        "data_source": "api_football",
    },
    "la_liga": {
        "name": "La Liga",
        "display": "Soccer — La Liga (Spain)",
        "category": "soccer",
        "espn_sport": "soccer",
        "espn_league": "esp.1",
        "odds_key": "soccer_spain_la_liga",
        "api_football_id": 140,
        "bet_types": ["moneyline", "total"],
        "season_format": "year",
        "has_draws": True,
        "data_source": "api_football",
    },
    "bundesliga": {
        "name": "Bundesliga",
        "display": "Soccer — Bundesliga (Germany)",
        "category": "soccer",
        "espn_sport": "soccer",
        "espn_league": "ger.1",
        "odds_key": "soccer_germany_bundesliga",
        "api_football_id": 78,
        "bet_types": ["moneyline", "total"],
        "season_format": "year",
        "has_draws": True,
        "data_source": "api_football",
    },
    "serie_a": {
        "name": "Serie A",
        "display": "Soccer — Serie A (Italy)",
        "category": "soccer",
        "espn_sport": "soccer",
        "espn_league": "ita.1",
        "odds_key": "soccer_italy_serie_a",
        "api_football_id": 135,
        "bet_types": ["moneyline", "total"],
        "season_format": "year",
        "has_draws": True,
        "data_source": "api_football",
    },
    "ligue_1": {
        "name": "Ligue 1",
        "display": "Soccer — Ligue 1 (France)",
        "category": "soccer",
        "espn_sport": "soccer",
        "espn_league": "fra.1",
        "odds_key": "soccer_france_ligue_1",
        "api_football_id": 61,
        "bet_types": ["moneyline", "total"],
        "season_format": "year",
        "has_draws": True,
        "data_source": "api_football",
    },
    "mls": {
        "name": "MLS",
        "display": "Soccer — MLS (USA)",
        "category": "soccer",
        "espn_sport": "soccer",
        "espn_league": "usa.1",
        "odds_key": "soccer_usa_mls",
        "api_football_id": 253,
        "bet_types": ["moneyline", "total"],
        "season_format": "year",
        "has_draws": True,
        "data_source": "api_football",
    },
    "ucl": {
        "name": "Champions League",
        "display": "Soccer — UEFA Champions League",
        "category": "soccer",
        "espn_sport": "soccer",
        "espn_league": "uefa.champions",
        "odds_key": "soccer_uefa_champs_league",
        "api_football_id": 2,
        "bet_types": ["moneyline", "total"],
        "season_format": "year",
        "has_draws": True,
        "data_source": "api_football",
    },
    "eredivisie": {
        "name": "Eredivisie",
        "display": "Soccer — Eredivisie (Netherlands)",
        "category": "soccer",
        "espn_sport": "soccer",
        "espn_league": "ned.1",
        "odds_key": "soccer_netherlands_eredivisie",
        "api_football_id": 88,
        "bet_types": ["moneyline", "total"],
        "season_format": "year",
        "has_draws": True,
        "data_source": "api_football",
    },
    "liga_mx": {
        "name": "Liga MX",
        "display": "Soccer — Liga MX (Mexico)",
        "category": "soccer",
        "espn_sport": "soccer",
        "espn_league": "mex.1",
        "odds_key": "soccer_mexico_ligamx",
        "api_football_id": 262,
        "bet_types": ["moneyline", "total"],
        "season_format": "year",
        "has_draws": True,
        "data_source": "api_football",
    },
    "brazil_serie_a": {
        "name": "Brasileirão",
        "display": "Soccer — Série A (Brazil)",
        "category": "soccer",
        "espn_sport": "soccer",
        "espn_league": "bra.1",
        "odds_key": "soccer_brazil_campeonato",
        "api_football_id": 71,
        "bet_types": ["moneyline", "total"],
        "season_format": "year",
        "has_draws": True,
        "data_source": "api_football",
    },
    # -----------------------------------------------------------------------
    # Tennis
    # -----------------------------------------------------------------------
    "tennis_atp": {
        "name": "ATP Tennis",
        "display": "Tennis — ATP Tour",
        "category": "tennis",
        "espn_sport": "tennis",
        "espn_league": "atp",
        "odds_key": "tennis_atp",
        "bet_types": ["moneyline"],
        "season_format": "year",
        "has_draws": False,
        "data_source": "espn",
    },
    "tennis_wta": {
        "name": "WTA Tennis",
        "display": "Tennis — WTA Tour",
        "category": "tennis",
        "espn_sport": "tennis",
        "espn_league": "wta",
        "odds_key": "tennis_wta",
        "bet_types": ["moneyline"],
        "season_format": "year",
        "has_draws": False,
        "data_source": "espn",
    },
    # -----------------------------------------------------------------------
    # Golf
    # -----------------------------------------------------------------------
    "golf_pga": {
        "name": "PGA Tour",
        "display": "Golf — PGA Tour",
        "category": "golf",
        "espn_sport": "golf",
        "espn_league": "pga",
        "odds_key": "golf_pga_championship",
        "bet_types": ["moneyline"],
        "season_format": "year",
        "has_draws": False,
        "data_source": "espn",
    },
    # -----------------------------------------------------------------------
    # MMA / Combat Sports
    # -----------------------------------------------------------------------
    "mma": {
        "name": "UFC / MMA",
        "display": "MMA — UFC",
        "category": "mma",
        "espn_sport": "mma",
        "espn_league": "ufc",
        "odds_key": "mma_mixed_martial_arts",
        "bet_types": ["moneyline"],
        "season_format": "year",
        "has_draws": False,
        "data_source": "espn",
    },
    "boxing": {
        "name": "Boxing",
        "display": "Boxing",
        "category": "boxing",
        "espn_sport": "boxing",
        "espn_league": "boxing",
        "odds_key": "boxing_boxing",
        "bet_types": ["moneyline"],
        "season_format": "year",
        "has_draws": False,
        "data_source": "espn",
    },
    # -----------------------------------------------------------------------
    # Rugby
    # -----------------------------------------------------------------------
    "nrl": {
        "name": "NRL",
        "display": "Rugby League — NRL (Australia)",
        "category": "rugby_league",
        "espn_sport": "rugby-league",
        "espn_league": "nrl",
        "odds_key": "rugbyleague_nrl",
        "bet_types": ["moneyline", "spread", "total"],
        "season_format": "year",
        "has_draws": False,
        "data_source": "espn",
    },
    "six_nations": {
        "name": "Six Nations",
        "display": "Rugby Union — Six Nations",
        "category": "rugby_union",
        "espn_sport": "rugby",
        "espn_league": "six.nations",
        "odds_key": "rugbyunion_six_nations",
        "bet_types": ["moneyline", "spread", "total"],
        "season_format": "year",
        "has_draws": False,
        "data_source": "espn",
    },
    # -----------------------------------------------------------------------
    # Australian Rules Football
    # -----------------------------------------------------------------------
    "afl": {
        "name": "AFL",
        "display": "Australian Rules Football — AFL",
        "category": "afl",
        "espn_sport": "australian-football",
        "espn_league": "afl",
        "odds_key": "aussierules_afl",
        "bet_types": ["moneyline", "spread", "total"],
        "season_format": "year",
        "has_draws": False,
        "data_source": "espn",
    },
    # -----------------------------------------------------------------------
    # Cricket
    # -----------------------------------------------------------------------
    "ipl": {
        "name": "IPL",
        "display": "Cricket — IPL (India)",
        "category": "cricket",
        "espn_sport": "cricket",
        "espn_league": "icc-cricket",
        "odds_key": "cricket_ipl",
        "bet_types": ["moneyline"],
        "season_format": "year",
        "has_draws": False,
        "data_source": "espn",
    },
}

# ---------------------------------------------------------------------------
# Aliases — map common/legacy names to catalog keys
# ---------------------------------------------------------------------------

SPORT_ALIASES = {
    # Generic names → most popular league
    "soccer": "epl",
    "football": "nfl",
    "basketball": "nba",
    "baseball": "mlb",
    "hockey": "nhl",
    "tennis": "tennis_atp",
    "golf": "golf_pga",
    "rugby": "nrl",
    "rugby_league": "nrl",
    "rugby_union": "six_nations",
    "aussie_rules": "afl",
    "cricket": "ipl",
    # Alternate spellings
    "american_football": "nfl",
    "ice_hockey": "nhl",
    "college_football": "ncaaf",
    "college_basketball": "ncaab",
    "champions_league": "ucl",
    "premier_league": "epl",
    "serie-a": "serie_a",
    "ufc": "mma",
}


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------

def get_sport_info(sport_key: str) -> Optional[dict]:
    """Return registry entry for a sport key (resolves aliases). None if unknown."""
    key = sport_key.lower().strip()
    if key in SPORT_CATALOG:
        return SPORT_CATALOG[key]
    resolved = SPORT_ALIASES.get(key)
    if resolved:
        return SPORT_CATALOG.get(resolved)
    return None


def resolve_sport_key(sport_key: str) -> Optional[str]:
    """Return the canonical catalog key for a sport string."""
    key = sport_key.lower().strip()
    if key in SPORT_CATALOG:
        return key
    return SPORT_ALIASES.get(key)


def list_sports() -> list:
    """Return all supported sports as a flat list of dicts (for /catalog endpoint)."""
    return [{"key": k, **v} for k, v in SPORT_CATALOG.items()]


def get_odds_key(sport_key: str) -> Optional[str]:
    """Return The Odds API sport key for a given sport."""
    info = get_sport_info(sport_key)
    return info["odds_key"] if info else None


def get_espn_path(sport_key: str) -> Optional[Tuple[str, str]]:
    """Return (espn_sport, espn_league) tuple for ESPN API calls."""
    info = get_sport_info(sport_key)
    if not info:
        return None
    return info["espn_sport"], info["espn_league"]


def is_soccer(sport_key: str) -> bool:
    info = get_sport_info(sport_key)
    return bool(info and info.get("category") == "soccer")


def get_api_football_id(sport_key: str) -> Optional[int]:
    info = get_sport_info(sport_key)
    return info.get("api_football_id") if info else None
