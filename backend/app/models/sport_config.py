"""
Sport configuration dataclass for sports betting predictions
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class SportConfig:
    """Configuration for a sports matchup prediction.

    The ``sport`` field should be a registry key from sport_registry.py
    (e.g. "nba", "nfl", "epl", "mlb", "nhl", "mma") or a legacy alias.
    See sport_registry.SPORT_ALIASES for supported aliases.
    """
    sport: str                    # registry key: "nba", "nfl", "epl", "mlb", etc.
    league: str                   # human-readable: "NBA", "NFL", "Premier League", etc.
    season: str                   # "2024-25" or "2025"
    team_a_id: int                # home team — source-specific ID (ESPN/BallDontLie/etc.)
    team_a_name: str
    team_b_id: int                # away team
    team_b_name: str
    game_date: Optional[str] = None          # ISO date "2025-01-15"
    bet_types: List[str] = field(default_factory=list)  # ["moneyline", "spread", "total", "props"]
    player_prop_players: List[str] = field(default_factory=list)  # player names for props
    odds_sport_key: str = ""      # override: The Odds API sport key e.g. "basketball_nba"
    league_id: Optional[int] = None  # API-Football league ID (soccer) or external league ID

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sport": self.sport,
            "league": self.league,
            "season": self.season,
            "team_a_id": self.team_a_id,
            "team_a_name": self.team_a_name,
            "team_b_id": self.team_b_id,
            "team_b_name": self.team_b_name,
            "game_date": self.game_date,
            "bet_types": self.bet_types,
            "player_prop_players": self.player_prop_players,
            "odds_sport_key": self.odds_sport_key,
            "league_id": self.league_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SportConfig":
        return cls(
            sport=data["sport"],
            league=data["league"],
            season=data["season"],
            team_a_id=data["team_a_id"],
            team_a_name=data["team_a_name"],
            team_b_id=data["team_b_id"],
            team_b_name=data["team_b_name"],
            game_date=data.get("game_date"),
            bet_types=data.get("bet_types", []),
            player_prop_players=data.get("player_prop_players", []),
            odds_sport_key=data.get("odds_sport_key", ""),
            league_id=data.get("league_id"),
        )
