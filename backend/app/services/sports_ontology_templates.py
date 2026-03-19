"""
Pre-defined ontology templates for sports prediction projects.
These bypass the LLM ontology generation step while producing output in the
exact same shape that OntologyGenerator.generate() returns.

Rules enforced:
- Exactly 10 entity types per template
- Last two entities MUST be Person and Organization (Zep validation requirement)
- 6-10 edge types
- Attribute names must not be: name, uuid, group_id, created_at, summary
"""

from typing import Dict, Any


NBA_ONTOLOGY: Dict[str, Any] = {
    "entity_types": [
        {
            "name": "NBATeam",
            "description": "An NBA franchise, identified by city and nickname.",
            "attributes": [
                {"name": "full_name", "type": "text", "description": "Full franchise name e.g. Boston Celtics"},
                {"name": "abbreviation", "type": "text", "description": "3-letter abbreviation e.g. BOS"},
                {"name": "conference", "type": "text", "description": "Eastern or Western conference"},
            ],
            "examples": ["Boston Celtics", "Golden State Warriors", "Los Angeles Lakers"],
        },
        {
            "name": "NBAPlayer",
            "description": "A current or recent NBA player on an active roster.",
            "attributes": [
                {"name": "full_name", "type": "text", "description": "Player's full name"},
                {"name": "position", "type": "text", "description": "Primary position: G, F, C, G-F, etc."},
                {"name": "injury_status", "type": "text", "description": "Current injury designation or Active"},
            ],
            "examples": ["Jayson Tatum", "Stephen Curry", "LeBron James"],
        },
        {
            "name": "Coach",
            "description": "Head coach or lead assistant coach of an NBA team.",
            "attributes": [
                {"name": "full_name", "type": "text", "description": "Coach's full name"},
                {"name": "role", "type": "text", "description": "Head Coach or Assistant Coach"},
            ],
            "examples": ["Joe Mazzulla", "Steve Kerr", "Doc Rivers"],
        },
        {
            "name": "SportsAnalyst",
            "description": "TV/radio analyst or advanced-stats blogger who covers the NBA.",
            "attributes": [
                {"name": "full_name", "type": "text", "description": "Analyst's name or handle"},
                {"name": "outlet", "type": "text", "description": "Employer or publication"},
            ],
            "examples": ["Zach Lowe", "Kevin Pelton", "Brian Windhorst"],
        },
        {
            "name": "BettingAnalyst",
            "description": "Sports bettor or handicapper specialising in NBA odds and line movement.",
            "attributes": [
                {"name": "full_name", "type": "text", "description": "Name or pseudonym"},
                {"name": "specialty", "type": "text", "description": "Moneyline, spreads, props, etc."},
            ],
            "examples": ["Sharp Money Mike", "Vegas Dave", "Covers.com analyst"],
        },
        {
            "name": "InjuryReporter",
            "description": "Beat writer or reporter whose primary focus is injury updates and availability.",
            "attributes": [
                {"name": "full_name", "type": "text", "description": "Reporter's name"},
                {"name": "outlet", "type": "text", "description": "Publication or news service"},
            ],
            "examples": ["Shams Charania", "Adrian Wojnarowski", "Chris Haynes"],
        },
        {
            "name": "BeatWriter",
            "description": "Journalist assigned to cover a specific NBA team day-to-day.",
            "attributes": [
                {"name": "full_name", "type": "text", "description": "Writer's name"},
                {"name": "team_covered", "type": "text", "description": "Primary team they cover"},
            ],
            "examples": ["Jay King (Celtics)", "Anthony Slater (Warriors)"],
        },
        {
            "name": "Fan",
            "description": "Supporter of a specific NBA team or casual basketball fan on social media.",
            "attributes": [
                {"name": "handle", "type": "text", "description": "Social media username"},
                {"name": "allegiance", "type": "text", "description": "Team they root for"},
            ],
            "examples": ["CelticsNation42", "DubNationFan", "LakeShowForever"],
        },
        {
            "name": "Person",
            "description": "Fallback for any natural person not covered by a more specific type.",
            "attributes": [
                {"name": "full_name", "type": "text", "description": "Person's full name"},
                {"name": "role", "type": "text", "description": "Role or title in context"},
            ],
            "examples": ["League commissioner", "Arena staff"],
        },
        {
            "name": "Organization",
            "description": "Fallback for any organisation not covered by a more specific type.",
            "attributes": [
                {"name": "full_name", "type": "text", "description": "Organisation name"},
                {"name": "description", "type": "text", "description": "Brief description of the org"},
            ],
            "examples": ["NBA League Office", "Sportsbook operator", "Sports media company"],
        },
    ],
    "edge_types": [
        {
            "name": "PLAYS_FOR",
            "description": "Player is on the roster of an NBATeam.",
            "source_targets": [{"source": "NBAPlayer", "target": "NBATeam"}],
            "attributes": [],
        },
        {
            "name": "COACHES",
            "description": "Coach leads or assists an NBATeam.",
            "source_targets": [{"source": "Coach", "target": "NBATeam"}],
            "attributes": [],
        },
        {
            "name": "COMPETES_AGAINST",
            "description": "Two NBATeams face each other in the matchup.",
            "source_targets": [{"source": "NBATeam", "target": "NBATeam"}],
            "attributes": [],
        },
        {
            "name": "COVERS",
            "description": "A journalist or analyst covers a team or player.",
            "source_targets": [
                {"source": "BeatWriter", "target": "NBATeam"},
                {"source": "InjuryReporter", "target": "NBAPlayer"},
                {"source": "SportsAnalyst", "target": "NBATeam"},
            ],
            "attributes": [],
        },
        {
            "name": "REPORTS_ON",
            "description": "Reporter publishes injury or transaction news about a player.",
            "source_targets": [
                {"source": "InjuryReporter", "target": "NBAPlayer"},
                {"source": "BeatWriter", "target": "NBAPlayer"},
            ],
            "attributes": [],
        },
        {
            "name": "SUPPORTS",
            "description": "Fan supports an NBATeam.",
            "source_targets": [
                {"source": "Fan", "target": "NBATeam"},
                {"source": "Person", "target": "NBATeam"},
            ],
            "attributes": [],
        },
        {
            "name": "BETS_ON",
            "description": "BettingAnalyst handicaps or places a bet on an NBATeam matchup.",
            "source_targets": [
                {"source": "BettingAnalyst", "target": "NBATeam"},
                {"source": "BettingAnalyst", "target": "NBAPlayer"},
            ],
            "attributes": [],
        },
    ],
    "analysis_summary": (
        "NBA matchup prediction project. Entities capture teams, players, coaches, "
        "analysts, bettors, injury reporters, beat writers, and fans involved in "
        "shaping narrative and odds around the game."
    ),
}


SOCCER_ONTOLOGY: Dict[str, Any] = {
    "entity_types": [
        {
            "name": "SoccerClub",
            "description": "A professional football/soccer club in a domestic or international league.",
            "attributes": [
                {"name": "full_name", "type": "text", "description": "Club's official name"},
                {"name": "league", "type": "text", "description": "Competition they participate in"},
                {"name": "home_stadium", "type": "text", "description": "Name of their home ground"},
            ],
            "examples": ["Arsenal FC", "Manchester City", "Real Madrid", "PSG"],
        },
        {
            "name": "SoccerPlayer",
            "description": "A current or recent professional footballer on a club or national team squad.",
            "attributes": [
                {"name": "full_name", "type": "text", "description": "Player's full name"},
                {"name": "position", "type": "text", "description": "GK, DEF, MID, FWD"},
                {"name": "injury_status", "type": "text", "description": "Fit, Doubtful, Suspended, etc."},
            ],
            "examples": ["Erling Haaland", "Bukayo Saka", "Vinicius Jr."],
        },
        {
            "name": "Manager",
            "description": "Head manager or first-team coach of a soccer club.",
            "attributes": [
                {"name": "full_name", "type": "text", "description": "Manager's full name"},
                {"name": "tactical_style", "type": "text", "description": "High press, possession, counter etc."},
            ],
            "examples": ["Pep Guardiola", "Mikel Arteta", "Carlo Ancelotti"],
        },
        {
            "name": "SportsAnalyst",
            "description": "TV pundit or tactical analyst who covers soccer.",
            "attributes": [
                {"name": "full_name", "type": "text", "description": "Analyst's name"},
                {"name": "outlet", "type": "text", "description": "Broadcaster or publication"},
            ],
            "examples": ["Gary Neville", "Fabrizio Romano", "Jonathan Wilson"],
        },
        {
            "name": "BettingAnalyst",
            "description": "Handicapper or tipster specialising in soccer odds and markets.",
            "attributes": [
                {"name": "full_name", "type": "text", "description": "Name or handle"},
                {"name": "specialty", "type": "text", "description": "1X2, BTTS, Asian handicap, etc."},
            ],
            "examples": ["FootballTipster", "OddsShark soccer analyst"],
        },
        {
            "name": "InjuryReporter",
            "description": "Journalist focused on squad fitness, injuries, and availability news.",
            "attributes": [
                {"name": "full_name", "type": "text", "description": "Reporter's name"},
                {"name": "outlet", "type": "text", "description": "Publication or service"},
            ],
            "examples": ["David Ornstein", "Fabrizio Romano"],
        },
        {
            "name": "Pundit",
            "description": "Former player or vocal commentator offering match predictions on social/broadcast media.",
            "attributes": [
                {"name": "full_name", "type": "text", "description": "Pundit's name"},
                {"name": "credibility", "type": "text", "description": "Former pro, fan pundit, national media"},
            ],
            "examples": ["Rio Ferdinand", "Jamie Carragher", "Graeme Souness"],
        },
        {
            "name": "Fan",
            "description": "Supporter of a specific club active on social media.",
            "attributes": [
                {"name": "handle", "type": "text", "description": "Username or alias"},
                {"name": "allegiance", "type": "text", "description": "Club they support"},
            ],
            "examples": ["Gooners_forever", "CityTilIDie"],
        },
        {
            "name": "Person",
            "description": "Fallback for any natural person not covered by a more specific type.",
            "attributes": [
                {"name": "full_name", "type": "text", "description": "Person's full name"},
                {"name": "role", "type": "text", "description": "Role in context"},
            ],
            "examples": ["Referee", "League official"],
        },
        {
            "name": "Organization",
            "description": "Fallback for any organisation not covered by a more specific type.",
            "attributes": [
                {"name": "full_name", "type": "text", "description": "Organisation name"},
                {"name": "description", "type": "text", "description": "Brief description"},
            ],
            "examples": ["UEFA", "FA", "Bookmaker"],
        },
    ],
    "edge_types": [
        {
            "name": "PLAYS_FOR",
            "description": "Player is on the squad of a SoccerClub.",
            "source_targets": [{"source": "SoccerPlayer", "target": "SoccerClub"}],
            "attributes": [],
        },
        {
            "name": "MANAGES",
            "description": "Manager leads a SoccerClub.",
            "source_targets": [{"source": "Manager", "target": "SoccerClub"}],
            "attributes": [],
        },
        {
            "name": "COMPETES_AGAINST",
            "description": "Two SoccerClubs are opponents in the fixture.",
            "source_targets": [{"source": "SoccerClub", "target": "SoccerClub"}],
            "attributes": [],
        },
        {
            "name": "COVERS",
            "description": "Journalist or analyst covers a club or player.",
            "source_targets": [
                {"source": "InjuryReporter", "target": "SoccerClub"},
                {"source": "SportsAnalyst", "target": "SoccerClub"},
                {"source": "InjuryReporter", "target": "SoccerPlayer"},
            ],
            "attributes": [],
        },
        {
            "name": "SUPPORTS",
            "description": "Fan supports a SoccerClub.",
            "source_targets": [
                {"source": "Fan", "target": "SoccerClub"},
                {"source": "Person", "target": "SoccerClub"},
            ],
            "attributes": [],
        },
        {
            "name": "BETS_ON",
            "description": "BettingAnalyst handicaps a match or player market.",
            "source_targets": [
                {"source": "BettingAnalyst", "target": "SoccerClub"},
                {"source": "BettingAnalyst", "target": "SoccerPlayer"},
            ],
            "attributes": [],
        },
        {
            "name": "PREDICTS",
            "description": "Pundit publicly predicts the outcome of a match.",
            "source_targets": [
                {"source": "Pundit", "target": "SoccerClub"},
                {"source": "SportsAnalyst", "target": "SoccerClub"},
            ],
            "attributes": [],
        },
    ],
    "analysis_summary": (
        "Soccer matchup prediction project. Entities cover clubs, players, managers, "
        "analysts, bettors, injury reporters, pundits, and fans whose discourse shapes "
        "narrative and market sentiment around the fixture."
    ),
}


def get_sports_ontology(sport: str) -> Dict[str, Any]:
    """
    Return the pre-defined ontology dict for a given sport.

    Args:
        sport: "nba" or "soccer"

    Returns:
        Ontology dict in the same shape as OntologyGenerator.generate() output.

    Raises:
        ValueError: if sport is not recognised.
    """
    sport = sport.lower()
    if sport == "nba":
        return NBA_ONTOLOGY
    if sport in ("soccer", "football"):
        return SOCCER_ONTOLOGY
    raise ValueError(f"Unknown sport '{sport}'. Supported: 'nba', 'soccer'.")
