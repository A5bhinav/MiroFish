"""
Sports data fetcher — wraps free-tier APIs across all major sports:

  Paid / key-required (optional):
    - Ball Don't Lie v1  (NBA)        api.balldontlie.io/v1
    - API-Football v3    (Soccer)     v3.football.api-sports.io
    - The Odds API v4    (All odds)   api.the-odds-api.com/v4

  Completely free / no key:
    - ESPN unofficial API  (ALL sports — universal fallback)
    - MLB Stats API        (Official MLB data, statsapi.mlb.com)

Supported sports via SportsDataOrchestrator.fetch_matchup():
  NBA, NFL, NCAAF, MLB, NHL, NCAAB,
  EPL, La Liga, Bundesliga, Serie A, Ligue 1, MLS, UCL,
  Eredivisie, Liga MX, Brasileirão,
  ATP Tennis, WTA Tennis, PGA Golf, UFC/MMA, Boxing,
  NRL Rugby, Six Nations, AFL, IPL Cricket

Top-level SportsDataOrchestrator.fetch_matchup() is the single entry point
consumed by the sports ingestion task.  All HTTP errors are caught and
accumulated non-fatally so a partial dataset is always returned.
"""

import requests
from typing import Any, Dict, List, Optional

from ..config import Config
from ..utils.logger import get_logger
from ..utils.cache import TTLCache, make_key

logger = get_logger("mirofish.sports_fetcher")

_NBA_BASE = "https://api.balldontlie.io/v1"
_FOOTBALL_BASE = "https://v3.football.api-sports.io"
_ODDS_BASE = "https://api.the-odds-api.com/v4"

_TIMEOUT = 15  # seconds

# Module-level cache for all paid-API sports fetchers.
# TTL design — matches the universal_sports_fetcher pattern:
#   players / roster      4 h
#   standings             1 h
#   recent games/fixtures 30 m
#   head-to-head          24 h
#   injuries              10 m  (most time-sensitive!)
#   odds                  10 m  (paid API quota — cache aggressively)
_cache = TTLCache()

_TTL_PLAYERS   = 14_400   # 4 hours
_TTL_STANDINGS =  3_600   # 1 hour
_TTL_GAMES     =  1_800   # 30 minutes
_TTL_H2H       = 86_400   # 24 hours
_TTL_INJURIES  =    600   # 10 minutes
_TTL_ODDS      =    600   # 10 minutes  ← pays per call; cache hard


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get(url: str, headers: Dict, params: Dict = None, ttl: float = 0) -> Any:
    """GET with timeout and optional TTL cache."""
    if ttl > 0:
        key = make_key(url, params or {})
        cached = _cache.get(key)
        if cached is not None:
            return cached

    resp = requests.get(url, headers=headers, params=params or {}, timeout=_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()

    if ttl > 0:
        _cache.set(key, data, ttl)

    return data


# ---------------------------------------------------------------------------
# NBA — Ball Don't Lie v1
# ---------------------------------------------------------------------------

class NBADataFetcher:
    """Wraps the Ball Don't Lie v1 API for NBA data."""

    def __init__(self, api_key: Optional[str] = None):
        key = api_key or Config.BALLDONTLIE_API_KEY
        if not key:
            raise ValueError("BALLDONTLIE_API_KEY is not configured")
        self._headers = {"Authorization": key}

    def get_teams(self) -> List[Dict]:
        """Return all NBA teams (cached 4 h)."""
        data = _get(f"{_NBA_BASE}/teams", self._headers, ttl=_TTL_PLAYERS)
        return data.get("data", [])

    def get_team(self, team_id: int) -> Dict:
        """Return a single team by id (cached 4 h)."""
        return _get(f"{_NBA_BASE}/teams/{team_id}", self._headers, ttl=_TTL_PLAYERS)

    def get_players_for_team(self, team_id: int) -> List[Dict]:
        """Return active players for a team (paginated, cached 4 h)."""
        players = []
        cursor = None
        while True:
            params = {"team_ids[]": team_id, "per_page": 100}
            if cursor:
                params["cursor"] = cursor
            data = _get(f"{_NBA_BASE}/players/active", self._headers, params, ttl=_TTL_PLAYERS)
            players.extend(data.get("data", []))
            cursor = data.get("meta", {}).get("next_cursor")
            if not cursor:
                break
        return players

    def get_recent_games(self, team_id: int, n: int = 10) -> List[Dict]:
        """Return the n most recent completed games for a team (cached 30 min)."""
        params = {"team_ids[]": team_id, "per_page": n, "seasons[]": _current_nba_season()}
        data = _get(f"{_NBA_BASE}/games", self._headers, params, ttl=_TTL_GAMES)
        return data.get("data", [])

    def get_season_averages(self, player_ids: List[int], season: int) -> List[Dict]:
        """Return season averages for the given player ids (cached 4 h)."""
        if not player_ids:
            return []
        params = {"season": season}
        for pid in player_ids[:25]:  # API cap
            params.setdefault("player_ids[]", [])
            if isinstance(params["player_ids[]"], list):
                params["player_ids[]"].append(pid)
            else:
                params["player_ids[]"] = [params["player_ids[]"], pid]
        data = _get(f"{_NBA_BASE}/season_averages", self._headers, params, ttl=_TTL_PLAYERS)
        return data.get("data", [])

    def get_injuries(self) -> List[Dict]:
        """Return current NBA injury report (cached 10 min — most time-sensitive data)."""
        data = _get(f"{_NBA_BASE}/player_injuries", self._headers, ttl=_TTL_INJURIES)
        return data.get("data", [])

    def get_head_to_head(self, team_a_id: int, team_b_id: int, n: int = 10) -> List[Dict]:
        """Return last n head-to-head games between two teams (cached 24 h)."""
        params = {
            "team_ids[]": [team_a_id, team_b_id],
            "per_page": n,
        }
        data = _get(f"{_NBA_BASE}/games", self._headers, params, ttl=_TTL_H2H)
        games = data.get("data", [])
        h2h = [
            g for g in games
            if {g.get("home_team", {}).get("id"), g.get("visitor_team", {}).get("id")} ==
               {team_a_id, team_b_id}
        ]
        return h2h[:n]


def _current_nba_season() -> int:
    """Return the current or most recent NBA season start year."""
    import datetime
    now = datetime.datetime.now()
    # NBA season starts in October; if before October, current season started last year
    return now.year if now.month >= 10 else now.year - 1


# ---------------------------------------------------------------------------
# Soccer — API-Football v3
# ---------------------------------------------------------------------------

class SoccerDataFetcher:
    """Wraps the API-Football v3 API."""

    def __init__(self, api_key: Optional[str] = None):
        key = api_key or Config.API_FOOTBALL_KEY
        if not key:
            raise ValueError("API_FOOTBALL_KEY is not configured")
        self._headers = {"x-apisports-key": key}

    def get_teams_by_league(self, league_id: int, season: int) -> List[Dict]:
        """Return all teams in a league for a season (cached 4 h)."""
        params = {"league": league_id, "season": season}
        data = _get(f"{_FOOTBALL_BASE}/teams", self._headers, params, ttl=_TTL_PLAYERS)
        return data.get("response", [])

    def get_squad(self, team_id: int) -> List[Dict]:
        """Return current squad for a team (cached 4 h)."""
        data = _get(f"{_FOOTBALL_BASE}/players/squads", self._headers, {"team": team_id}, ttl=_TTL_PLAYERS)
        players = []
        for item in data.get("response", []):
            players.extend(item.get("players", []))
        return players

    def get_recent_fixtures(self, team_id: int, league_id: int, season: int, n: int = 10) -> List[Dict]:
        """Return n most recent completed fixtures for a team (cached 30 min)."""
        params = {"team": team_id, "league": league_id, "season": season, "last": n}
        data = _get(f"{_FOOTBALL_BASE}/fixtures", self._headers, params, ttl=_TTL_GAMES)
        return data.get("response", [])

    def get_standings(self, league_id: int, season: int) -> List[Dict]:
        """Return current league standings (cached 1 h)."""
        params = {"league": league_id, "season": season}
        data = _get(f"{_FOOTBALL_BASE}/standings", self._headers, params, ttl=_TTL_STANDINGS)
        standings = []
        for item in data.get("response", []):
            for league_info in item.get("league", {}).get("standings", []):
                standings.extend(league_info)
        return standings

    def get_head_to_head(self, team_a_id: int, team_b_id: int, n: int = 10) -> List[Dict]:
        """Return last n head-to-head fixtures (cached 24 h)."""
        params = {"h2h": f"{team_a_id}-{team_b_id}", "last": n}
        data = _get(f"{_FOOTBALL_BASE}/fixtures/headtohead", self._headers, params, ttl=_TTL_H2H)
        return data.get("response", [])


# ---------------------------------------------------------------------------
# Odds — The Odds API v4
# ---------------------------------------------------------------------------

class OddsDataFetcher:
    """Wraps The Odds API v4."""

    def __init__(self, api_key: Optional[str] = None):
        key = api_key or Config.ODDS_API_KEY
        if not key:
            raise ValueError("ODDS_API_KEY is not configured")
        self._key = key

    def get_odds(
        self,
        sport_key: str,
        regions: str = "us",
        markets: str = "h2h,spreads,totals",
    ) -> List[Dict]:
        """
        Return current odds for all upcoming games in a sport (cached 10 min).

        The Odds API charges per call — caching 10 min saves ~85% of quota
        for typical usage patterns where the same sport is requested repeatedly.

        Args:
            sport_key: e.g. "basketball_nba", "americanfootball_nfl", "soccer_epl"
            regions: comma-separated e.g. "us,eu"
            markets: comma-separated e.g. "h2h,spreads,totals"
        """
        params = {
            "apiKey": self._key,
            "regions": regions,
            "markets": markets,
            "oddsFormat": "american",
        }
        data = _get(f"{_ODDS_BASE}/sports/{sport_key}/odds", {}, params, ttl=_TTL_ODDS)
        return data if isinstance(data, list) else []

    def get_sports(self) -> List[Dict]:
        """Return list of available sports (cached 1 h — rarely changes)."""
        params = {"apiKey": self._key, "all": "false"}
        data = _get(f"{_ODDS_BASE}/sports", {}, params, ttl=_TTL_STANDINGS)
        return data if isinstance(data, list) else []


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class SportsDataOrchestrator:
    """
    Single entry point that calls the right API clients and returns one
    unified dict ready for SportsNarrativeFormatter.

    Dispatch logic:
      nba              → Ball Don't Lie (ESPN fallback if key missing)
      epl/la_liga/...  → API-Football  (ESPN fallback if key missing)
      mlb              → MLB Stats API + ESPN for H2H
      nfl/nhl/ncaaf/
      ncaab/mma/tennis/
      golf/boxing/nrl/
      afl/ipl/...      → ESPN universal fetcher (no key)
    """

    @staticmethod
    def fetch_matchup(sport_config) -> Dict[str, Any]:
        """
        Fetch all data needed for a matchup prediction.

        Args:
            sport_config: SportConfig dataclass instance

        Returns:
            Dict with keys: sport, teams, players, recent_games, head_to_head,
                            standings, odds, errors
            Errors are accumulated without raising so a partial dataset is
            always returned.
        """
        result: Dict[str, Any] = {
            "sport": sport_config.sport,
            "league": sport_config.league,
            "season": sport_config.season,
            "team_a": {"id": sport_config.team_a_id, "name": sport_config.team_a_name},
            "team_b": {"id": sport_config.team_b_id, "name": sport_config.team_b_name},
            "game_date": sport_config.game_date,
            "bet_types": sport_config.bet_types,
            "player_prop_players": sport_config.player_prop_players,
            "players_a": [],
            "players_b": [],
            "recent_games_a": [],
            "recent_games_b": [],
            "head_to_head": [],
            "standings": [],
            "season_averages": [],
            "injuries": [],
            "odds": [],
            "errors": [],
        }

        from .sport_registry import get_sport_info, is_soccer, resolve_sport_key

        sport_key = resolve_sport_key(sport_config.sport) or sport_config.sport.lower()
        sport_info = get_sport_info(sport_key) or {}
        data_source = sport_info.get("data_source", "espn")

        if sport_key == "nba" or data_source == "balldontlie":
            SportsDataOrchestrator._fetch_nba(sport_config, result)
        elif is_soccer(sport_key) or data_source == "api_football":
            SportsDataOrchestrator._fetch_soccer(sport_config, result, sport_info)
        elif sport_key == "mlb" or data_source == "mlb_api":
            SportsDataOrchestrator._fetch_mlb(sport_config, result)
        elif sport_info:
            # Universal ESPN-based fetcher for NFL, NHL, NCAAF, NCAAB,
            # Tennis, Golf, MMA, Boxing, Rugby, AFL, Cricket, etc.
            SportsDataOrchestrator._fetch_espn(sport_config, result, sport_info)
        else:
            result["errors"].append(f"No data fetcher for sport '{sport_config.sport}'")

        # Always try odds regardless of sport
        SportsDataOrchestrator._fetch_odds(sport_config, result)

        return result

    @staticmethod
    def _fetch_nba(sport_config, result: Dict) -> None:
        try:
            fetcher = NBADataFetcher()
        except ValueError as e:
            result["errors"].append(f"NBA API unavailable: {e}")
            # Fall back to ESPN for basic roster/schedule data
            from .sport_registry import get_sport_info
            sport_info = get_sport_info("nba") or {}
            logger.info("NBA API key missing — falling back to ESPN")
            SportsDataOrchestrator._fetch_espn(sport_config, result, sport_info)
            return

        season_year = _current_nba_season()

        for attr, team_id, team_name in [
            ("players_a", sport_config.team_a_id, sport_config.team_a_name),
            ("players_b", sport_config.team_b_id, sport_config.team_b_name),
        ]:
            try:
                result[attr] = fetcher.get_players_for_team(team_id)
                logger.info(f"Fetched {len(result[attr])} players for {team_name}")
            except Exception as e:
                result["errors"].append(f"Players for {team_name}: {e}")

        for attr, team_id, team_name in [
            ("recent_games_a", sport_config.team_a_id, sport_config.team_a_name),
            ("recent_games_b", sport_config.team_b_id, sport_config.team_b_name),
        ]:
            try:
                result[attr] = fetcher.get_recent_games(team_id, n=10)
            except Exception as e:
                result["errors"].append(f"Recent games for {team_name}: {e}")

        try:
            result["head_to_head"] = fetcher.get_head_to_head(
                sport_config.team_a_id, sport_config.team_b_id
            )
        except Exception as e:
            result["errors"].append(f"Head-to-head: {e}")

        # Fetch current injury report and filter to matchup teams
        try:
            all_injuries = fetcher.get_injuries()
            team_ids = {sport_config.team_a_id, sport_config.team_b_id}
            result["injuries"] = [
                i for i in all_injuries
                if i.get("team", {}).get("id") in team_ids
            ]
            logger.info(f"Fetched {len(result['injuries'])} injuries for matchup teams")
        except Exception as e:
            result["errors"].append(f"Injuries: {e}")

        # Fetch season averages for prop players
        if sport_config.player_prop_players and result["players_a"] + result["players_b"]:
            all_players = result["players_a"] + result["players_b"]
            prop_ids = []
            for p in all_players:
                full = f"{p.get('first_name', '')} {p.get('last_name', '')}".strip()
                if any(name.lower() in full.lower() for name in sport_config.player_prop_players):
                    prop_ids.append(p["id"])
            if prop_ids:
                try:
                    result["season_averages"] = fetcher.get_season_averages(prop_ids, season_year)
                except Exception as e:
                    result["errors"].append(f"Season averages: {e}")

    @staticmethod
    def _fetch_soccer(sport_config, result: Dict, sport_info: Dict = None) -> None:
        try:
            fetcher = SoccerDataFetcher()
        except ValueError as e:
            result["errors"].append(f"Soccer API unavailable: {e}")
            # Fall back to ESPN for basic data
            from .sport_registry import get_sport_info
            _info = sport_info or get_sport_info(sport_config.sport) or {}
            if _info:
                logger.info("Soccer API key missing — falling back to ESPN")
                SportsDataOrchestrator._fetch_espn(sport_config, result, _info)
            return

        # Resolve API-Football league_id from registry, then sport_config, then default EPL
        if sport_info and sport_info.get("api_football_id"):
            league_id = sport_info["api_football_id"]
        else:
            from .sport_registry import get_api_football_id
            league_id = get_api_football_id(sport_config.sport) or getattr(sport_config, "league_id", 39)

        season = int(sport_config.season.split("-")[0]) if "-" in sport_config.season else int(sport_config.season)

        for attr, team_id, team_name in [
            ("players_a", sport_config.team_a_id, sport_config.team_a_name),
            ("players_b", sport_config.team_b_id, sport_config.team_b_name),
        ]:
            try:
                result[attr] = fetcher.get_squad(team_id)
                logger.info(f"Fetched {len(result[attr])} players for {team_name}")
            except Exception as e:
                result["errors"].append(f"Squad for {team_name}: {e}")

        for attr, team_id, team_name in [
            ("recent_games_a", sport_config.team_a_id, sport_config.team_a_name),
            ("recent_games_b", sport_config.team_b_id, sport_config.team_b_name),
        ]:
            try:
                result[attr] = fetcher.get_recent_fixtures(team_id, league_id, season, n=10)
            except Exception as e:
                result["errors"].append(f"Recent fixtures for {team_name}: {e}")

        try:
            result["head_to_head"] = fetcher.get_head_to_head(
                sport_config.team_a_id, sport_config.team_b_id
            )
        except Exception as e:
            result["errors"].append(f"Head-to-head: {e}")

        try:
            result["standings"] = fetcher.get_standings(league_id, season)
        except Exception as e:
            result["errors"].append(f"Standings: {e}")

    @staticmethod
    def _fetch_mlb(sport_config, result: Dict) -> None:
        from .universal_sports_fetcher import MLBDataFetcher
        fetcher = MLBDataFetcher()

        season_str = sport_config.season or ""
        try:
            season = int(season_str.split("-")[0]) if "-" in season_str else int(season_str)
        except (ValueError, AttributeError):
            import datetime
            season = datetime.datetime.now().year

        for attr, team_id, team_name in [
            ("players_a", sport_config.team_a_id, sport_config.team_a_name),
            ("players_b", sport_config.team_b_id, sport_config.team_b_name),
        ]:
            try:
                result[attr] = fetcher.get_roster(team_id)
                logger.info(f"MLB: fetched {len(result[attr])} players for {team_name}")
            except Exception as e:
                result["errors"].append(f"MLB roster for {team_name}: {e}")

        for attr, team_id, team_name in [
            ("recent_games_a", sport_config.team_a_id, sport_config.team_a_name),
            ("recent_games_b", sport_config.team_b_id, sport_config.team_b_name),
        ]:
            try:
                result[attr] = fetcher.get_recent_games(team_id, season, n=10)
            except Exception as e:
                result["errors"].append(f"MLB games for {team_name}: {e}")

        try:
            result["standings"] = fetcher.get_standings(season)
        except Exception as e:
            result["errors"].append(f"MLB standings: {e}")

        # Head-to-head via ESPN (MLB official API has no H2H endpoint)
        try:
            from .universal_sports_fetcher import ESPNFetcher
            espn = ESPNFetcher("baseball", "mlb")
            result["head_to_head"] = espn.get_head_to_head(
                sport_config.team_a_id, sport_config.team_b_id
            )
        except Exception as e:
            result["errors"].append(f"MLB H2H (ESPN): {e}")

    @staticmethod
    def _fetch_espn(sport_config, result: Dict, sport_info: Dict) -> None:
        from .universal_sports_fetcher import ESPNFetcher
        espn_sport = sport_info.get("espn_sport", "")
        espn_league = sport_info.get("espn_league", "")
        if not espn_sport or not espn_league:
            result["errors"].append("ESPN path not configured for this sport")
            return

        fetcher = ESPNFetcher(espn_sport, espn_league)

        for attr, team_id, team_name in [
            ("players_a", sport_config.team_a_id, sport_config.team_a_name),
            ("players_b", sport_config.team_b_id, sport_config.team_b_name),
        ]:
            try:
                result[attr] = fetcher.get_roster(team_id)
                logger.info(f"ESPN: fetched {len(result[attr])} players for {team_name}")
            except Exception as e:
                result["errors"].append(f"ESPN roster for {team_name}: {e}")

        for attr, team_id, team_name in [
            ("recent_games_a", sport_config.team_a_id, sport_config.team_a_name),
            ("recent_games_b", sport_config.team_b_id, sport_config.team_b_name),
        ]:
            try:
                result[attr] = fetcher.get_recent_games(team_id, n=10)
            except Exception as e:
                result["errors"].append(f"ESPN recent games for {team_name}: {e}")

        try:
            result["head_to_head"] = fetcher.get_head_to_head(
                sport_config.team_a_id, sport_config.team_b_id
            )
        except Exception as e:
            result["errors"].append(f"ESPN H2H: {e}")

        try:
            result["standings"] = fetcher.get_standings()
        except Exception as e:
            result["errors"].append(f"ESPN standings: {e}")

    @staticmethod
    def _fetch_odds(sport_config, result: Dict) -> None:
        from .sport_registry import get_odds_key, resolve_sport_key
        sport_key = sport_config.odds_sport_key
        if not sport_key:
            # Look up in registry first, then fall back to basketball_nba
            sport_key = get_odds_key(resolve_sport_key(sport_config.sport) or sport_config.sport) or "basketball_nba"

        markets = ",".join(sport_config.bet_types) if sport_config.bet_types else "h2h,spreads,totals"
        # Map bet_type names to Odds API market names
        market_map = {
            "moneyline": "h2h",
            "spread": "spreads",
            "total": "totals",
            "props": "player_props_points",
        }
        api_markets = []
        for bt in (sport_config.bet_types or ["moneyline", "spread", "total"]):
            api_markets.append(market_map.get(bt, bt))
        markets = ",".join(api_markets) if api_markets else "h2h,spreads,totals"

        try:
            fetcher = OddsDataFetcher()
            all_odds = fetcher.get_odds(sport_key, regions="us", markets=markets)
            # Filter to games involving one of our teams
            team_names = {
                sport_config.team_a_name.lower(),
                sport_config.team_b_name.lower(),
            }
            filtered = [
                g for g in all_odds
                if g.get("home_team", "").lower() in team_names
                or g.get("away_team", "").lower() in team_names
            ]
            result["odds"] = filtered or all_odds[:5]  # fallback: first 5 games
            logger.info(f"Fetched {len(result['odds'])} odds records")
        except ValueError as e:
            result["errors"].append(f"Odds API unavailable: {e}")
        except Exception as e:
            result["errors"].append(f"Odds fetch error: {e}")
