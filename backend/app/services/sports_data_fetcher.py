"""
Sports data fetcher — wraps three free-tier APIs:
  - Ball Don't Lie v1  (NBA)        api.balldontlie.io/v1
  - API-Football v3    (Soccer)     v3.football.api-sports.io
  - The Odds API v4    (Live odds)  api.the-odds-api.com/v4

Top-level SportsDataOrchestrator.fetch_matchup() is the single entry point
consumed by the sports ingestion task.  All HTTP errors are caught and
accumulated non-fatally so a partial dataset is always returned.
"""

import requests
from typing import Any, Dict, List, Optional

from ..config import Config
from ..utils.logger import get_logger

logger = get_logger("mirofish.sports_fetcher")

_NBA_BASE = "https://api.balldontlie.io/v1"
_FOOTBALL_BASE = "https://v3.football.api-sports.io"
_ODDS_BASE = "https://api.the-odds-api.com/v4"

_TIMEOUT = 15  # seconds


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get(url: str, headers: Dict, params: Dict = None) -> Any:
    """GET with timeout; raises on HTTP error."""
    resp = requests.get(url, headers=headers, params=params or {}, timeout=_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


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
        """Return all NBA teams."""
        data = _get(f"{_NBA_BASE}/teams", self._headers)
        return data.get("data", [])

    def get_team(self, team_id: int) -> Dict:
        """Return a single team by id."""
        return _get(f"{_NBA_BASE}/teams/{team_id}", self._headers)

    def get_players_for_team(self, team_id: int) -> List[Dict]:
        """Return active players for a team (paginated, up to 100)."""
        players = []
        cursor = None
        while True:
            params = {"team_ids[]": team_id, "per_page": 100}
            if cursor:
                params["cursor"] = cursor
            data = _get(f"{_NBA_BASE}/players/active", self._headers, params)
            players.extend(data.get("data", []))
            cursor = data.get("meta", {}).get("next_cursor")
            if not cursor:
                break
        return players

    def get_recent_games(self, team_id: int, n: int = 10) -> List[Dict]:
        """Return the n most recent completed games for a team."""
        params = {"team_ids[]": team_id, "per_page": n, "seasons[]": _current_nba_season()}
        data = _get(f"{_NBA_BASE}/games", self._headers, params)
        return data.get("data", [])

    def get_season_averages(self, player_ids: List[int], season: int) -> List[Dict]:
        """Return season averages for the given player ids."""
        if not player_ids:
            return []
        params = {"season": season}
        for pid in player_ids[:25]:  # API cap
            params.setdefault("player_ids[]", [])
            if isinstance(params["player_ids[]"], list):
                params["player_ids[]"].append(pid)
            else:
                params["player_ids[]"] = [params["player_ids[]"], pid]
        data = _get(f"{_NBA_BASE}/season_averages", self._headers, params)
        return data.get("data", [])

    def get_head_to_head(self, team_a_id: int, team_b_id: int, n: int = 10) -> List[Dict]:
        """Return last n head-to-head games between two teams."""
        params = {
            "team_ids[]": [team_a_id, team_b_id],
            "per_page": n,
            "seasons[]": _current_nba_season(),
        }
        data = _get(f"{_NBA_BASE}/games", self._headers, params)
        games = data.get("data", [])
        # Filter to only genuine H2H
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
        """Return all teams in a league for a season."""
        params = {"league": league_id, "season": season}
        data = _get(f"{_FOOTBALL_BASE}/teams", self._headers, params)
        return data.get("response", [])

    def get_squad(self, team_id: int) -> List[Dict]:
        """Return current squad for a team."""
        data = _get(f"{_FOOTBALL_BASE}/players/squads", self._headers, {"team": team_id})
        players = []
        for item in data.get("response", []):
            players.extend(item.get("players", []))
        return players

    def get_recent_fixtures(self, team_id: int, league_id: int, season: int, n: int = 10) -> List[Dict]:
        """Return n most recent completed fixtures for a team."""
        params = {"team": team_id, "league": league_id, "season": season, "last": n}
        data = _get(f"{_FOOTBALL_BASE}/fixtures", self._headers, params)
        return data.get("response", [])

    def get_standings(self, league_id: int, season: int) -> List[Dict]:
        """Return current league standings."""
        params = {"league": league_id, "season": season}
        data = _get(f"{_FOOTBALL_BASE}/standings", self._headers, params)
        standings = []
        for item in data.get("response", []):
            for league_info in item.get("league", {}).get("standings", []):
                standings.extend(league_info)
        return standings

    def get_head_to_head(self, team_a_id: int, team_b_id: int, n: int = 10) -> List[Dict]:
        """Return last n head-to-head fixtures."""
        params = {"h2h": f"{team_a_id}-{team_b_id}", "last": n}
        data = _get(f"{_FOOTBALL_BASE}/fixtures/headtohead", self._headers, params)
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
        Return current odds for all upcoming games in a sport.

        Args:
            sport_key: e.g. "basketball_nba", "soccer_epl"
            regions: comma-separated e.g. "us,eu"
            markets: comma-separated e.g. "h2h,spreads,totals"
        """
        params = {
            "apiKey": self._key,
            "regions": regions,
            "markets": markets,
            "oddsFormat": "american",
        }
        data = _get(f"{_ODDS_BASE}/sports/{sport_key}/odds", {}, params)
        return data if isinstance(data, list) else []

    def get_sports(self) -> List[Dict]:
        """Return list of available sports (for discovery)."""
        params = {"apiKey": self._key, "all": "false"}
        data = _get(f"{_ODDS_BASE}/sports", {}, params)
        return data if isinstance(data, list) else []


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class SportsDataOrchestrator:
    """
    Single entry point that calls the right API clients and returns one
    unified dict ready for SportsNarrativeFormatter.
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
            "odds": [],
            "errors": [],
        }

        sport = sport_config.sport.lower()

        if sport == "nba":
            SportsDataOrchestrator._fetch_nba(sport_config, result)
        elif sport in ("soccer", "football"):
            SportsDataOrchestrator._fetch_soccer(sport_config, result)

        # Always try odds regardless of sport
        SportsDataOrchestrator._fetch_odds(sport_config, result)

        return result

    @staticmethod
    def _fetch_nba(sport_config, result: Dict) -> None:
        try:
            fetcher = NBADataFetcher()
        except ValueError as e:
            result["errors"].append(f"NBA API unavailable: {e}")
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
    def _fetch_soccer(sport_config, result: Dict) -> None:
        try:
            fetcher = SoccerDataFetcher()
        except ValueError as e:
            result["errors"].append(f"Soccer API unavailable: {e}")
            return

        # Derive league_id from sport_config (user should pass it as team_a_id convention
        # or we fall back to a common default)
        league_id = getattr(sport_config, "league_id", 39)  # 39 = Premier League
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
    def _fetch_odds(sport_config, result: Dict) -> None:
        sport_key = sport_config.odds_sport_key
        if not sport_key:
            # Map common sport names to Odds API keys
            mapping = {
                "nba": "basketball_nba",
                "soccer": "soccer_epl",
                "football": "soccer_epl",
            }
            sport_key = mapping.get(sport_config.sport.lower(), "basketball_nba")

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
