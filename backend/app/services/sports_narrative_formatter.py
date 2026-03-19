"""
Sports Narrative Formatter

Converts raw API dicts (from SportsDataOrchestrator) into a list of
journalism-style plain-text strings.  No LLM calls, no HTTP.

Each returned string is self-contained and becomes one
  EpisodeData(data=chunk, type="text")
for Zep ingestion.  The formatter purposely writes in a slightly verbose
narrative style so the Zep graph extracts rich entity and relationship nodes.
"""

from typing import Any, Dict, List, Optional


class SportsNarrativeFormatter:
    """Converts raw sports API data into Zep-ingestible text chunks."""

    @staticmethod
    def format(raw_data: Dict[str, Any], sport_config) -> List[str]:
        """
        Convert raw API data into a list of narrative text chunks.

        Args:
            raw_data: Dict returned by SportsDataOrchestrator.fetch_matchup()
            sport_config: SportConfig dataclass instance

        Returns:
            List[str] — each element is one text chunk for Zep ingestion.
        """
        sport = sport_config.sport.lower()
        if sport == "nba":
            return SportsNarrativeFormatter._format_nba(raw_data, sport_config)
        if sport in ("soccer", "football"):
            return SportsNarrativeFormatter._format_soccer(raw_data, sport_config)
        raise ValueError(f"Unknown sport '{sport}'")

    # ------------------------------------------------------------------
    # NBA Formatting
    # ------------------------------------------------------------------

    @staticmethod
    def _format_nba(raw: Dict, sc) -> List[str]:
        chunks: List[str] = []

        team_a = sc.team_a_name
        team_b = sc.team_b_name

        # 1. Matchup overview
        game_date_str = f" on {sc.game_date}" if sc.game_date else ""
        chunks.append(
            f"NBA MATCHUP OVERVIEW\n\n"
            f"The {team_a} will face the {team_b}{game_date_str} in an NBA regular-season "
            f"or playoff game. This prediction analysis covers all available statistical, "
            f"injury, and market intelligence to forecast the outcome across "
            f"{', '.join(sc.bet_types) if sc.bet_types else 'all'} betting markets."
        )

        # 2. Team A recent form
        games_a = raw.get("recent_games_a", [])
        if games_a:
            chunks.append(SportsNarrativeFormatter._nba_team_form(team_a, sc.team_a_id, games_a))

        # 3. Team B recent form
        games_b = raw.get("recent_games_b", [])
        if games_b:
            chunks.append(SportsNarrativeFormatter._nba_team_form(team_b, sc.team_b_id, games_b))

        # 4. Head-to-head
        h2h = raw.get("head_to_head", [])
        if h2h:
            chunks.append(SportsNarrativeFormatter._nba_h2h(team_a, team_b, h2h))

        # 5. Rosters
        players_a = raw.get("players_a", [])
        if players_a:
            chunks.append(SportsNarrativeFormatter._nba_roster(team_a, players_a))

        players_b = raw.get("players_b", [])
        if players_b:
            chunks.append(SportsNarrativeFormatter._nba_roster(team_b, players_b))

        # 6. Season averages (prop players)
        avgs = raw.get("season_averages", [])
        if avgs:
            chunks.append(SportsNarrativeFormatter._nba_season_averages(avgs, players_a + players_b))

        # 7. Odds
        odds = raw.get("odds", [])
        if odds:
            chunks.append(SportsNarrativeFormatter._format_odds(team_a, team_b, odds))

        # 8. Data errors / caveats
        errors = raw.get("errors", [])
        if errors:
            caveats = "\n".join(f"- {e}" for e in errors)
            chunks.append(
                f"DATA AVAILABILITY NOTES\n\n"
                f"The following data sources were unavailable for this analysis:\n{caveats}\n"
                f"Analysts should treat predictions in these areas with additional caution."
            )

        return [c for c in chunks if c.strip()]

    @staticmethod
    def _nba_team_form(team_name: str, team_id: int, games: List[Dict]) -> str:
        lines = [f"RECENT FORM — {team_name.upper()}\n"]
        wins = 0
        losses = 0
        pts_for = []
        pts_against = []

        for g in games[:10]:
            home = g.get("home_team", {})
            visitor = g.get("visitor_team", {})
            home_score = g.get("home_team_score", 0) or 0
            vis_score = g.get("visitor_team_score", 0) or 0

            if home.get("id") == team_id:
                team_pts = home_score
                opp_pts = vis_score
                opp_name = visitor.get("full_name", "Opponent")
                venue = "home"
            else:
                team_pts = vis_score
                opp_pts = home_score
                opp_name = home.get("full_name", "Opponent")
                venue = "away"

            result = "W" if team_pts > opp_pts else "L"
            if result == "W":
                wins += 1
            else:
                losses += 1
            pts_for.append(team_pts)
            pts_against.append(opp_pts)

            date = g.get("date", "")[:10] if g.get("date") else ""
            lines.append(f"  {date} ({venue}) vs {opp_name}: {result} {team_pts}-{opp_pts}")

        avg_for = round(sum(pts_for) / len(pts_for), 1) if pts_for else 0
        avg_against = round(sum(pts_against) / len(pts_against), 1) if pts_against else 0

        lines.append(
            f"\n{team_name} last {len(games)} games record: {wins}W-{losses}L. "
            f"Averaging {avg_for} PPG scored, {avg_against} PPG allowed."
        )
        return "\n".join(lines)

    @staticmethod
    def _nba_h2h(team_a: str, team_b: str, games: List[Dict]) -> str:
        lines = [f"HEAD-TO-HEAD HISTORY — {team_a.upper()} vs {team_b.upper()}\n"]
        if not games:
            lines.append("No head-to-head data available.")
            return "\n".join(lines)

        a_wins = 0
        b_wins = 0
        for g in games[:10]:
            home = g.get("home_team", {})
            vis = g.get("visitor_team", {})
            h_score = g.get("home_team_score", 0) or 0
            v_score = g.get("visitor_team_score", 0) or 0
            home_name = home.get("full_name", "")
            vis_name = vis.get("full_name", "")
            date = g.get("date", "")[:10] if g.get("date") else ""
            winner = home_name if h_score > v_score else vis_name
            if team_a.lower() in winner.lower():
                a_wins += 1
            else:
                b_wins += 1
            lines.append(f"  {date}: {home_name} {h_score} — {v_score} {vis_name}")

        lines.append(
            f"\nIn their last {len(games)} meetings: {team_a} leads {a_wins}-{b_wins} vs {team_b}."
        )
        return "\n".join(lines)

    @staticmethod
    def _nba_roster(team_name: str, players: List[Dict]) -> str:
        lines = [f"ROSTER — {team_name.upper()}\n"]
        for p in players[:20]:
            first = p.get("first_name", "")
            last = p.get("last_name", "")
            pos = p.get("position", "N/A")
            jersey = p.get("jersey_number", "")
            lines.append(f"  #{jersey} {first} {last} ({pos})")
        if len(players) > 20:
            lines.append(f"  ... and {len(players) - 20} more players")
        return "\n".join(lines)

    @staticmethod
    def _nba_season_averages(avgs: List[Dict], all_players: List[Dict]) -> str:
        # Build player id -> name map
        id_map = {}
        for p in all_players:
            id_map[p.get("id")] = f"{p.get('first_name', '')} {p.get('last_name', '')}".strip()

        lines = ["SEASON STATISTICAL AVERAGES — KEY PLAYERS\n"]
        for a in avgs:
            pid = a.get("player_id")
            name = id_map.get(pid, f"Player {pid}")
            pts = a.get("pts", 0)
            reb = a.get("reb", 0)
            ast = a.get("ast", 0)
            stl = a.get("stl", 0)
            blk = a.get("blk", 0)
            fg_pct = round((a.get("fg_pct") or 0) * 100, 1)
            three_pct = round((a.get("fg3_pct") or 0) * 100, 1)
            gp = a.get("games_played", 0)
            lines.append(
                f"  {name}: {pts} PPG / {reb} RPG / {ast} APG / {stl} SPG / {blk} BPG "
                f"| FG {fg_pct}% / 3P {three_pct}% over {gp} games"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Soccer Formatting
    # ------------------------------------------------------------------

    @staticmethod
    def _format_soccer(raw: Dict, sc) -> List[str]:
        chunks: List[str] = []
        team_a = sc.team_a_name
        team_b = sc.team_b_name

        # 1. Matchup overview
        game_date_str = f" on {sc.game_date}" if sc.game_date else ""
        chunks.append(
            f"SOCCER FIXTURE OVERVIEW\n\n"
            f"{team_a} will host (or face) {team_b}{game_date_str} in {sc.league}. "
            f"This prediction analysis covers all available statistical, tactical, "
            f"injury, and market intelligence for "
            f"{', '.join(sc.bet_types) if sc.bet_types else 'all'} markets."
        )

        # 2. Recent form both teams
        for team_name, games_key in [(team_a, "recent_games_a"), (team_b, "recent_games_b")]:
            games = raw.get(games_key, [])
            if games:
                chunks.append(SportsNarrativeFormatter._soccer_team_form(team_name, games))

        # 3. Head-to-head
        h2h = raw.get("head_to_head", [])
        if h2h:
            chunks.append(SportsNarrativeFormatter._soccer_h2h(team_a, team_b, h2h))

        # 4. Squads
        for team_name, pkey in [(team_a, "players_a"), (team_b, "players_b")]:
            players = raw.get(pkey, [])
            if players:
                chunks.append(SportsNarrativeFormatter._soccer_squad(team_name, players))

        # 5. Standings
        standings = raw.get("standings", [])
        if standings:
            chunks.append(SportsNarrativeFormatter._soccer_standings(team_a, team_b, standings, sc.league))

        # 6. Odds
        odds = raw.get("odds", [])
        if odds:
            chunks.append(SportsNarrativeFormatter._format_odds(team_a, team_b, odds))

        # 7. Caveats
        errors = raw.get("errors", [])
        if errors:
            caveats = "\n".join(f"- {e}" for e in errors)
            chunks.append(
                f"DATA AVAILABILITY NOTES\n\n"
                f"The following data sources were unavailable:\n{caveats}"
            )

        return [c for c in chunks if c.strip()]

    @staticmethod
    def _soccer_team_form(team_name: str, fixtures: List[Dict]) -> str:
        lines = [f"RECENT FORM — {team_name.upper()}\n"]
        wins = draws = losses = 0
        goals_for = goals_against = 0

        for f in fixtures[:10]:
            fix = f.get("fixture", {})
            teams = f.get("teams", {})
            goals = f.get("goals", {})
            home_team = teams.get("home", {}).get("name", "")
            away_team = teams.get("away", {}).get("name", "")
            home_goals = goals.get("home") or 0
            away_goals = goals.get("away") or 0
            date = fix.get("date", "")[:10] if fix.get("date") else ""

            is_home = team_name.lower() in home_team.lower()
            team_goals = home_goals if is_home else away_goals
            opp_goals = away_goals if is_home else home_goals
            opp_name = away_team if is_home else home_team
            venue = "home" if is_home else "away"

            goals_for += team_goals
            goals_against += opp_goals

            if team_goals > opp_goals:
                result = "W"
                wins += 1
            elif team_goals == opp_goals:
                result = "D"
                draws += 1
            else:
                result = "L"
                losses += 1

            lines.append(f"  {date} ({venue}) vs {opp_name}: {result} {team_goals}-{opp_goals}")

        n = wins + draws + losses
        avg_for = round(goals_for / n, 2) if n else 0
        avg_against = round(goals_against / n, 2) if n else 0
        lines.append(
            f"\n{team_name} last {n} games: {wins}W {draws}D {losses}L. "
            f"Averaging {avg_for} goals scored, {avg_against} goals conceded."
        )
        return "\n".join(lines)

    @staticmethod
    def _soccer_h2h(team_a: str, team_b: str, fixtures: List[Dict]) -> str:
        lines = [f"HEAD-TO-HEAD — {team_a.upper()} vs {team_b.upper()}\n"]
        a_wins = b_wins = draws = 0

        for f in fixtures[:10]:
            teams = f.get("teams", {})
            goals = f.get("goals", {})
            home = teams.get("home", {}).get("name", "")
            away = teams.get("away", {}).get("name", "")
            hg = goals.get("home") or 0
            ag = goals.get("away") or 0
            date = f.get("fixture", {}).get("date", "")[:10]
            lines.append(f"  {date}: {home} {hg} — {ag} {away}")

            if hg > ag:
                if team_a.lower() in home.lower():
                    a_wins += 1
                else:
                    b_wins += 1
            elif ag > hg:
                if team_b.lower() in away.lower():
                    b_wins += 1
                else:
                    a_wins += 1
            else:
                draws += 1

        n = len(fixtures[:10])
        lines.append(
            f"\nLast {n} meetings: {team_a} {a_wins}W / {draws}D / {b_wins}W {team_b}."
        )
        return "\n".join(lines)

    @staticmethod
    def _soccer_squad(team_name: str, players: List[Dict]) -> str:
        lines = [f"SQUAD — {team_name.upper()}\n"]
        for p in players[:25]:
            name = p.get("name", "Unknown")
            pos = p.get("position", "N/A")
            number = p.get("number", "")
            lines.append(f"  #{number} {name} ({pos})")
        if len(players) > 25:
            lines.append(f"  ... and {len(players) - 25} more")
        return "\n".join(lines)

    @staticmethod
    def _soccer_standings(team_a: str, team_b: str, standings: List[Dict], league: str) -> str:
        lines = [f"LEAGUE STANDINGS — {league.upper()}\n"]
        for entry in standings[:20]:
            team = entry.get("team", {}).get("name", "")
            rank = entry.get("rank", "")
            pts = entry.get("points", 0)
            played = entry.get("all", {}).get("played", 0)
            won = entry.get("all", {}).get("win", 0)
            drawn = entry.get("all", {}).get("draw", 0)
            lost = entry.get("all", {}).get("lose", 0)
            gd = entry.get("goalsDiff", 0)
            highlight = " ◄" if team_a.lower() in team.lower() or team_b.lower() in team.lower() else ""
            lines.append(
                f"  {rank:>2}. {team:<30} {pts} pts  ({won}W {drawn}D {lost}L  GD: {gd}){highlight}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Shared: Odds formatting
    # ------------------------------------------------------------------

    @staticmethod
    def _format_odds(team_a: str, team_b: str, odds_data: List[Dict]) -> str:
        lines = [f"BETTING ODDS — {team_a.upper()} vs {team_b.upper()}\n"]

        for game in odds_data[:3]:  # top 3 bookmakers
            home = game.get("home_team", "")
            away = game.get("away_team", "")
            commence = (game.get("commence_time") or "")[:10]
            lines.append(f"Game: {home} vs {away}  ({commence})")

            for bm in game.get("bookmakers", [])[:3]:
                bm_name = bm.get("title", "Unknown")
                for market in bm.get("markets", []):
                    mkey = market.get("key", "")
                    lines.append(f"  [{bm_name}] {mkey.upper()}:")
                    for outcome in market.get("outcomes", []):
                        name = outcome.get("name", "")
                        price = outcome.get("price", "")
                        point = outcome.get("point", "")
                        point_str = f" ({point:+g})" if point is not None and point != "" else ""
                        lines.append(f"    {name}{point_str}: {price}")

        return "\n".join(lines)
