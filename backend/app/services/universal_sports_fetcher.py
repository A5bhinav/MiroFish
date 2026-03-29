"""
Universal Sports Fetcher

Two data sources, both completely free with no API key:

  1. ESPNFetcher  — ESPN's unofficial public API
     Covers: NFL, NCAAF, NBA, NCAAB, NHL, Tennis, Golf, MMA, Boxing,
             Rugby, AFL, Cricket, and all soccer leagues.

  2. MLBDataFetcher — Official MLB Stats API (statsapi.mlb.com)
     Richer stats than ESPN for baseball; no key required.

These are used by SportsDataOrchestrator as the fallback / primary source
for every sport that does not have a dedicated paid-API fetcher (NBA/Soccer).
"""

import requests
from typing import Any, Dict, List, Optional

from ..utils.logger import get_logger
from ..utils.cache import TTLCache, make_key

logger = get_logger("mirofish.universal_fetcher")

_ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports"
_MLB_BASE = "https://statsapi.mlb.com/api/v1"
_TIMEOUT = 15

# Module-level cache — shared across all ESPNFetcher / MLBDataFetcher instances.
# TTL guide (sports data changes slowly):
#   rosters / teams       4 h  — player signings happen overnight
#   standings             1 h  — updated after each game
#   recent games          30 m — new results trickle in through the day
#   head-to-head          24 h — historical, almost never changes
_cache = TTLCache()

_TTL_ROSTER    = 14_400   # 4 hours
_TTL_STANDINGS =  3_600   # 1 hour
_TTL_GAMES     =  1_800   # 30 minutes
_TTL_H2H       = 86_400   # 24 hours
_TTL_TEAMS     = 14_400   # 4 hours


def _get(url: str, params: Dict = None, ttl: float = 0) -> Any:
    """GET with optional TTL caching."""
    if ttl > 0:
        key = make_key(url, params or {})
        cached = _cache.get(key)
        if cached is not None:
            return cached

    resp = requests.get(url, params=params or {}, timeout=_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()

    if ttl > 0:
        _cache.set(key, data, ttl)

    return data


# ---------------------------------------------------------------------------
# ESPN — universal fetcher (no API key)
# ---------------------------------------------------------------------------

class ESPNFetcher:
    """
    Fetches teams, rosters, schedules, standings, and H2H data from the
    ESPN unofficial public API. No API key required.

    Args:
        espn_sport  : sport slug used by ESPN (e.g. "football", "basketball")
        espn_league : league slug used by ESPN (e.g. "nfl", "nba", "eng.1")
    """

    def __init__(self, espn_sport: str, espn_league: str):
        self._sport = espn_sport
        self._league = espn_league
        self._base = f"{_ESPN_BASE}/{espn_sport}/{espn_league}"

    # ------------------------------------------------------------------
    # Teams
    # ------------------------------------------------------------------

    def get_teams(self) -> List[Dict]:
        """Return all teams / competitors in the league (cached 4 h)."""
        try:
            data = _get(f"{self._base}/teams", {"limit": 200}, ttl=_TTL_TEAMS)
        except Exception as e:
            logger.warning(f"ESPN get_teams failed: {e}")
            return []

        sports = data.get("sports", [])
        teams = []
        for sport in sports:
            for league in sport.get("leagues", []):
                for team_item in league.get("teams", []):
                    team = team_item.get("team", {})
                    teams.append(self._normalize_team(team))
        # Some endpoints return teams directly (e.g. tennis athletes)
        if not teams:
            for item in data.get("items", []):
                teams.append(self._normalize_team(item))
        return teams

    def _normalize_team(self, t: Dict) -> Dict:
        return {
            "id": int(t.get("id", 0)),
            "name": t.get("displayName", t.get("name", "")),
            "abbreviation": t.get("abbreviation", ""),
            "location": t.get("location", ""),
            "logo": (t.get("logos") or [{}])[0].get("href", ""),
        }

    # ------------------------------------------------------------------
    # Roster
    # ------------------------------------------------------------------

    def get_roster(self, team_id: int) -> List[Dict]:
        """Return current roster for a team (cached 4 h)."""
        try:
            data = _get(f"{self._base}/teams/{team_id}/roster", ttl=_TTL_ROSTER)
        except Exception as e:
            logger.warning(f"ESPN get_roster({team_id}) failed: {e}")
            return []

        athletes_raw = data.get("athletes", [])
        players = []
        for group in athletes_raw:
            if isinstance(group, dict) and "items" in group:
                # NFL/NHL style: grouped by offense/defense/special-teams
                for athlete in group.get("items", []):
                    players.append(self._normalize_athlete(athlete))
            elif isinstance(group, dict):
                players.append(self._normalize_athlete(group))
        return players

    def _normalize_athlete(self, a: Dict) -> Dict:
        pos = a.get("position", {})
        pos_abbr = pos.get("abbreviation", "") if isinstance(pos, dict) else str(pos)
        status = a.get("status", {})
        status_name = status.get("name", "Active") if isinstance(status, dict) else "Active"
        return {
            "id": int(a.get("id", 0)),
            "name": a.get("displayName", a.get("fullName", "")),
            "position": pos_abbr,
            "jersey": a.get("jersey", ""),
            "status": status_name,
        }

    # ------------------------------------------------------------------
    # Recent Games
    # ------------------------------------------------------------------

    def get_recent_games(self, team_id: int, n: int = 10) -> List[Dict]:
        """Return last n completed games for a team (cached 30 min)."""
        try:
            data = _get(f"{self._base}/teams/{team_id}/schedule", ttl=_TTL_GAMES)
        except Exception as e:
            logger.warning(f"ESPN get_recent_games({team_id}) failed: {e}")
            return []

        events = data.get("events", [])
        completed = [
            e for e in events
            if e.get("status", {}).get("type", {}).get("completed", False)
        ]
        recent = completed[-n:] if len(completed) > n else completed
        return [self._normalize_event(e) for e in recent]

    # ------------------------------------------------------------------
    # Head-to-Head
    # ------------------------------------------------------------------

    def get_head_to_head(self, team_a_id: int, team_b_id: int, n: int = 10) -> List[Dict]:
        """
        Return last n H2H games (cached 24 h — historical data doesn't change).
        Fetches team A's full schedule and filters for games against team B.
        """
        try:
            data = _get(f"{self._base}/teams/{team_a_id}/schedule", ttl=_TTL_H2H)
        except Exception as e:
            logger.warning(f"ESPN get_h2h failed: {e}")
            return []

        events = data.get("events", [])
        h2h = []
        for e in events:
            if not e.get("status", {}).get("type", {}).get("completed", False):
                continue
            competition = (e.get("competitions") or [{}])[0]
            comp_ids = {
                int(c.get("team", {}).get("id", 0))
                for c in competition.get("competitors", [])
            }
            if team_a_id in comp_ids and team_b_id in comp_ids:
                h2h.append(self._normalize_event(e))

        return h2h[-n:] if len(h2h) > n else h2h

    # ------------------------------------------------------------------
    # Standings
    # ------------------------------------------------------------------

    def get_standings(self) -> List[Dict]:
        """Return league standings (cached 1 h)."""
        try:
            data = _get(f"{self._base}/standings", ttl=_TTL_STANDINGS)
        except Exception as e:
            logger.warning(f"ESPN get_standings failed: {e}")
            return []

        standings = []
        # ESPN wraps standings in "children" (divisions/conferences)
        children = data.get("children", [data])
        for group in children:
            group_name = group.get("name", "")
            for entry in group.get("standings", {}).get("entries", []):
                team = entry.get("team", {})
                stats = {
                    s["name"]: s.get("displayValue", s.get("value", ""))
                    for s in entry.get("stats", [])
                }
                standings.append({
                    "team_id": int(team.get("id", 0)),
                    "team": team.get("displayName", ""),
                    "wins": stats.get("wins", stats.get("gamesWon", "")),
                    "losses": stats.get("losses", stats.get("gamesLost", "")),
                    "pct": stats.get("winPercent", ""),
                    "points": stats.get("points", ""),      # NHL uses points
                    "gb": stats.get("gamesBehind", ""),
                    "group": group_name,
                })
        return standings

    # ------------------------------------------------------------------
    # Scoreboard (current/upcoming games)
    # ------------------------------------------------------------------

    def get_scoreboard(self) -> List[Dict]:
        """Return current scoreboard — no caching (live data)."""
        try:
            data = _get(f"{self._base}/scoreboard")
        except Exception as e:
            logger.warning(f"ESPN get_scoreboard failed: {e}")
            return []
        return [self._normalize_event(e) for e in data.get("events", [])]

    # ------------------------------------------------------------------
    # Event normalizer
    # ------------------------------------------------------------------

    def _normalize_event(self, e: Dict) -> Dict:
        competition = (e.get("competitions") or [{}])[0]
        competitors = competition.get("competitors", [])
        home = next((c for c in competitors if c.get("homeAway") == "home"), {})
        away = next((c for c in competitors if c.get("homeAway") == "away"), {})
        return {
            "id": e.get("id", ""),
            "date": e.get("date", ""),
            "name": e.get("name", ""),
            "home_team": home.get("team", {}).get("displayName", ""),
            "home_team_id": int(home.get("team", {}).get("id", 0)),
            "home_score": home.get("score", ""),
            "away_team": away.get("team", {}).get("displayName", ""),
            "away_team_id": int(away.get("team", {}).get("id", 0)),
            "away_score": away.get("score", ""),
            "status": e.get("status", {}).get("type", {}).get("description", ""),
            "venue": competition.get("venue", {}).get("fullName", ""),
        }


# ---------------------------------------------------------------------------
# MLB — Official MLB Stats API (no API key required)
# ---------------------------------------------------------------------------

class MLBDataFetcher:
    """
    Official MLB Stats API at statsapi.mlb.com — completely free, no key.
    Provides richer stats than ESPN for baseball (ERA, AVG, OPS, etc.).
    """

    def get_teams(self) -> List[Dict]:
        """Return all MLB teams (cached 24 h — never changes mid-season)."""
        try:
            data = _get(f"{_MLB_BASE}/teams", {"sportId": 1}, ttl=_TTL_H2H)
        except Exception as e:
            logger.warning(f"MLB get_teams failed: {e}")
            return []

        teams = []
        for t in data.get("teams", []):
            teams.append({
                "id": t.get("id"),
                "name": t.get("name", ""),
                "abbreviation": t.get("abbreviation", ""),
                "location": t.get("locationName", ""),
                "league": t.get("league", {}).get("name", ""),
                "division": t.get("division", {}).get("name", ""),
            })
        return teams

    def get_roster(self, team_id: int) -> List[Dict]:
        """Return active 40-man roster for a team (cached 4 h)."""
        try:
            data = _get(f"{_MLB_BASE}/teams/{team_id}/roster", {"rosterType": "active"}, ttl=_TTL_ROSTER)
        except Exception as e:
            logger.warning(f"MLB get_roster({team_id}) failed: {e}")
            return []

        players = []
        for p in data.get("roster", []):
            person = p.get("person", {})
            players.append({
                "id": person.get("id"),
                "name": person.get("fullName", ""),
                "position": p.get("position", {}).get("abbreviation", ""),
                "jersey": p.get("jerseyNumber", ""),
                "status": p.get("status", {}).get("description", "Active"),
            })
        return players

    def get_recent_games(self, team_id: int, season: int, n: int = 10) -> List[Dict]:
        """Return last n completed regular-season games for a team (cached 30 min)."""
        try:
            data = _get(f"{_MLB_BASE}/schedule", {
                "sportId": 1,
                "teamId": team_id,
                "season": season,
                "gameType": "R",
            }, ttl=_TTL_GAMES)
        except Exception as e:
            logger.warning(f"MLB get_recent_games({team_id}) failed: {e}")
            return []

        games = []
        for date_obj in data.get("dates", []):
            for g in date_obj.get("games", []):
                if g.get("status", {}).get("detailedState") == "Final":
                    teams = g.get("teams", {})
                    games.append({
                        "id": g.get("gamePk"),
                        "date": g.get("officialDate", ""),
                        "home_team": teams.get("home", {}).get("team", {}).get("name", ""),
                        "home_score": teams.get("home", {}).get("score", ""),
                        "away_team": teams.get("away", {}).get("team", {}).get("name", ""),
                        "away_score": teams.get("away", {}).get("score", ""),
                        "venue": g.get("venue", {}).get("name", ""),
                        "status": "Final",
                    })
        return games[-n:] if len(games) > n else games

    def get_standings(self, season: int) -> List[Dict]:
        """Return MLB standings for both leagues (cached 1 h)."""
        try:
            data = _get(f"{_MLB_BASE}/standings", {
                "leagueId": "103,104",
                "season": season,
            }, ttl=_TTL_STANDINGS)
        except Exception as e:
            logger.warning(f"MLB get_standings failed: {e}")
            return []

        standings = []
        for record in data.get("records", []):
            division = record.get("division", {}).get("name", "")
            for tr in record.get("teamRecords", []):
                team = tr.get("team", {})
                standings.append({
                    "team_id": team.get("id"),
                    "team": team.get("name", ""),
                    "wins": tr.get("wins", 0),
                    "losses": tr.get("losses", 0),
                    "pct": tr.get("winningPercentage", ""),
                    "division": division,
                    "gb": tr.get("gamesBack", ""),
                    "streak": tr.get("streak", {}).get("streakCode", ""),
                })
        return standings
