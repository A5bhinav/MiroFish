"""
Sport configuration dataclass for sports betting predictions
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class SportConfig:
    """Configuration for a sports matchup prediction"""
    sport: str                    # "nba" or "soccer"
    league: str                   # "NBA", "Premier League", etc.
    season: str                   # "2024-25"
    team_a_id: int
    team_a_name: str
    team_b_id: int
    team_b_name: str
    game_date: Optional[str] = None          # ISO date "2025-01-15"
    bet_types: List[str] = field(default_factory=list)  # ["moneyline", "spread", "total", "props"]
    player_prop_players: List[str] = field(default_factory=list)  # player names for props
    odds_sport_key: str = ""      # The Odds API sport key e.g. "basketball_nba"

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
        )
