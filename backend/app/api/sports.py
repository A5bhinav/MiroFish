"""
Sports prediction API blueprint.

Endpoints
---------
GET  /api/sports/teams?sport=nba&league=NBA
GET  /api/sports/players?sport=nba&team_id=2
GET  /api/sports/odds?sport=nba&markets=h2h,spreads,totals
POST /api/sports/ingest
GET  /api/sports/ingest/status/<task_id>
GET  /api/sports/project/<project_id>/config
"""

import traceback
import threading
from flask import request, jsonify

from . import sports_bp
from ..config import Config
from ..models.task import TaskManager, TaskStatus
from ..models.project import ProjectManager, ProjectStatus
from ..models.sport_config import SportConfig
from ..services.sports_data_fetcher import (
    NBADataFetcher,
    OddsDataFetcher,
    SportsDataOrchestrator,
)
from ..services.sports_narrative_formatter import SportsNarrativeFormatter
from ..services.sports_ontology_templates import get_sports_ontology
from ..services.graph_builder import GraphBuilderService
from ..utils.logger import get_logger

logger = get_logger("mirofish.sports_api")


# ---------------------------------------------------------------------------
# GET /api/sports/teams
# ---------------------------------------------------------------------------

@sports_bp.route("/teams", methods=["GET"])
def get_teams():
    """Return team list for a given sport."""
    sport = request.args.get("sport", "nba").lower()
    try:
        if sport == "nba":
            fetcher = NBADataFetcher()
            teams = fetcher.get_teams()
            # Normalise to a consistent shape
            normalised = [
                {
                    "id": t.get("id"),
                    "name": t.get("full_name", f"{t.get('city', '')} {t.get('name', '')}".strip()),
                    "city": t.get("city", ""),
                    "abbreviation": t.get("abbreviation", ""),
                    "conference": t.get("conference", ""),
                    "division": t.get("division", ""),
                }
                for t in teams
            ]
            return jsonify({"success": True, "data": {"teams": normalised}})

        return jsonify({"success": False, "error": f"Teams endpoint not yet implemented for sport '{sport}'"}), 400

    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 503
    except Exception as e:
        logger.error(f"get_teams error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


# ---------------------------------------------------------------------------
# GET /api/sports/players
# ---------------------------------------------------------------------------

@sports_bp.route("/players", methods=["GET"])
def get_players():
    """Return player list for a given team."""
    sport = request.args.get("sport", "nba").lower()
    team_id = request.args.get("team_id")
    if not team_id:
        return jsonify({"success": False, "error": "team_id is required"}), 400

    try:
        team_id = int(team_id)
        if sport == "nba":
            fetcher = NBADataFetcher()
            players = fetcher.get_players_for_team(team_id)
            normalised = [
                {
                    "id": p.get("id"),
                    "name": f"{p.get('first_name', '')} {p.get('last_name', '')}".strip(),
                    "position": p.get("position", ""),
                    "jersey_number": p.get("jersey_number", ""),
                    "status": "Active",
                }
                for p in players
            ]
            return jsonify({"success": True, "data": {"players": normalised}})

        return jsonify({"success": False, "error": f"Players endpoint not yet implemented for sport '{sport}'"}), 400

    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 503
    except Exception as e:
        logger.error(f"get_players error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


# ---------------------------------------------------------------------------
# GET /api/sports/odds
# ---------------------------------------------------------------------------

@sports_bp.route("/odds", methods=["GET"])
def get_odds():
    """Return current odds for a sport."""
    sport = request.args.get("sport", "nba").lower()
    markets = request.args.get("markets", "h2h,spreads,totals")

    sport_key_map = {
        "nba": "basketball_nba",
        "soccer": "soccer_epl",
        "football": "soccer_epl",
    }
    sport_key = sport_key_map.get(sport, sport)

    try:
        fetcher = OddsDataFetcher()
        games = fetcher.get_odds(sport_key, regions="us", markets=markets)
        return jsonify({"success": True, "data": {"games": games}})

    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 503
    except Exception as e:
        logger.error(f"get_odds error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


# ---------------------------------------------------------------------------
# POST /api/sports/ingest
# ---------------------------------------------------------------------------

@sports_bp.route("/ingest", methods=["POST"])
def ingest():
    """
    Start an async sports data ingestion task.

    Body (JSON):
        sport, league, season,
        team_a_id, team_a_name, team_b_id, team_b_name,
        game_date (optional),
        bet_types (list, optional),
        player_prop_players (list, optional),
        simulation_requirement (optional),
        odds_sport_key (optional)
    """
    try:
        body = request.get_json(force=True) or {}

        required = ["sport", "league", "season", "team_a_id", "team_a_name", "team_b_id", "team_b_name"]
        missing = [k for k in required if not body.get(k)]
        if missing:
            return jsonify({"success": False, "error": f"Missing required fields: {missing}"}), 400

        sport_config = SportConfig(
            sport=body["sport"],
            league=body["league"],
            season=body["season"],
            team_a_id=int(body["team_a_id"]),
            team_a_name=body["team_a_name"],
            team_b_id=int(body["team_b_id"]),
            team_b_name=body["team_b_name"],
            game_date=body.get("game_date"),
            bet_types=body.get("bet_types", ["moneyline", "spread", "total"]),
            player_prop_players=body.get("player_prop_players", []),
            odds_sport_key=body.get("odds_sport_key", ""),
        )

        simulation_requirement = body.get(
            "simulation_requirement",
            f"Predict the outcome of {sport_config.team_a_name} vs {sport_config.team_b_name}. "
            f"Focus on {', '.join(sport_config.bet_types)} markets.",
        )

        # Create a project to hold this sports prediction
        project_name = f"{sport_config.team_a_name} vs {sport_config.team_b_name}"
        project = ProjectManager.create_project(name=project_name)
        project.is_sports_project = True
        project.sport_config = sport_config.to_dict()
        project.simulation_requirement = simulation_requirement

        # Use the sports ontology template directly (no LLM call)
        ontology = get_sports_ontology(sport_config.sport)
        project.ontology = ontology
        ProjectManager.save_project(project)

        # Create background task
        task_manager = TaskManager()
        task_id = task_manager.create_task(f"Sports ingest: {project_name}")
        project.graph_build_task_id = task_id
        project.status = ProjectStatus.GRAPH_BUILDING
        ProjectManager.save_project(project)

        project_id = project.project_id

        def ingest_task():
            ingest_logger = get_logger("mirofish.sports_ingest")
            try:
                ingest_logger.info(f"[{task_id}] Starting sports ingest for {project_name}")
                task_manager.update_task(task_id, status=TaskStatus.PROCESSING, message="Fetching sports data...", progress=5)

                # Step 1: fetch API data
                raw_data = SportsDataOrchestrator.fetch_matchup(sport_config)
                task_manager.update_task(task_id, message="Formatting narrative text...", progress=20)

                # Step 2: format into text chunks
                chunks = SportsNarrativeFormatter.format(raw_data, sport_config)
                ingest_logger.info(f"[{task_id}] Produced {len(chunks)} narrative chunks")
                task_manager.update_task(task_id, message=f"Formatted {len(chunks)} narrative chunks", progress=25)

                # Step 3: build Zep graph
                task_manager.update_task(task_id, message="Creating Zep graph...", progress=30)
                builder = GraphBuilderService(api_key=Config.ZEP_API_KEY)
                graph_name = f"sports_{project_id}"
                graph_id = builder.create_graph(name=graph_name)

                # Update project with graph_id
                proj = ProjectManager.get_project(project_id)
                proj.graph_id = graph_id
                ProjectManager.save_project(proj)

                # Step 4: set ontology
                task_manager.update_task(task_id, message="Setting ontology...", progress=35)
                builder.set_ontology(graph_id, ontology)

                # Step 5: ingest chunks into Zep
                def add_progress(msg, ratio):
                    progress = 35 + int(ratio * 40)  # 35–75%
                    task_manager.update_task(task_id, message=msg, progress=progress)

                task_manager.update_task(task_id, message=f"Ingesting {len(chunks)} chunks into Zep...", progress=35)
                episode_uuids = builder.add_text_batches(graph_id, chunks, batch_size=3, progress_callback=add_progress)

                # Step 6: wait for Zep processing
                task_manager.update_task(task_id, message="Waiting for Zep to process data...", progress=75)

                def wait_progress(msg, ratio):
                    progress = 75 + int(ratio * 20)  # 75–95%
                    task_manager.update_task(task_id, message=msg, progress=progress)

                builder._wait_for_episodes(episode_uuids, wait_progress)

                # Step 7: finalise
                task_manager.update_task(task_id, message="Retrieving graph summary...", progress=95)
                graph_data = builder.get_graph_data(graph_id)
                node_count = graph_data.get("node_count", 0)
                edge_count = graph_data.get("edge_count", 0)

                proj = ProjectManager.get_project(project_id)
                proj.status = ProjectStatus.GRAPH_COMPLETED
                ProjectManager.save_project(proj)

                task_manager.update_task(
                    task_id,
                    status=TaskStatus.COMPLETED,
                    message="Sports data ingested successfully",
                    progress=100,
                    result={
                        "project_id": project_id,
                        "graph_id": graph_id,
                        "node_count": node_count,
                        "edge_count": edge_count,
                        "chunk_count": len(chunks),
                        "api_errors": raw_data.get("errors", []),
                    },
                )
                ingest_logger.info(f"[{task_id}] Ingest complete: {node_count} nodes, {edge_count} edges")

            except Exception as e:
                ingest_logger.error(f"[{task_id}] Ingest failed: {e}")
                ingest_logger.debug(traceback.format_exc())

                try:
                    proj = ProjectManager.get_project(project_id)
                    proj.status = ProjectStatus.FAILED
                    proj.error = str(e)
                    ProjectManager.save_project(proj)
                except Exception:
                    pass

                task_manager.update_task(
                    task_id,
                    status=TaskStatus.FAILED,
                    message=f"Ingest failed: {e}",
                    error=traceback.format_exc(),
                )

        thread = threading.Thread(target=ingest_task, daemon=True)
        thread.start()

        return jsonify({
            "success": True,
            "data": {
                "project_id": project_id,
                "task_id": task_id,
                "message": "Sports ingestion task started. Poll /api/sports/ingest/status/<task_id> for progress.",
            },
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e), "traceback": traceback.format_exc()}), 500


# ---------------------------------------------------------------------------
# GET /api/sports/ingest/status/<task_id>
# ---------------------------------------------------------------------------

@sports_bp.route("/ingest/status/<task_id>", methods=["GET"])
def ingest_status(task_id: str):
    """Delegate to TaskManager — same shape as existing task polling."""
    task = TaskManager().get_task(task_id)
    if not task:
        return jsonify({"success": False, "error": f"Task {task_id} not found"}), 404

    return jsonify({
        "success": True,
        "data": {
            "task_id": task.task_id,
            "status": task.status,
            "message": task.message,
            "progress": task.progress,
            "result": task.result,
            "error": task.error,
        },
    })


# ---------------------------------------------------------------------------
# GET /api/sports/project/<project_id>/config
# ---------------------------------------------------------------------------

@sports_bp.route("/project/<project_id>/config", methods=["GET"])
def get_project_config(project_id: str):
    """Return sports-specific config for a project."""
    project = ProjectManager.get_project(project_id)
    if not project:
        return jsonify({"success": False, "error": "Project not found"}), 404

    return jsonify({
        "success": True,
        "data": {
            "project_id": project_id,
            "is_sports_project": project.is_sports_project,
            "sport_config": project.sport_config,
        },
    })


# ---------------------------------------------------------------------------
# POST /api/sports/predict  — direct ML prediction (no Zep/simulation needed)
# ---------------------------------------------------------------------------

@sports_bp.route("/predict", methods=["POST"])
def predict():
    """
    Direct ML prediction endpoint. Does NOT require a Zep graph or simulation.
    Use this for fast, real-time predictions before running the full pipeline.

    Body:
      {
        "sport": "nba" | "soccer",
        "team_a_name": "Boston Celtics",
        "team_b_name": "Miami Heat",
        "league": "NBA",                     # optional
        "home_elo": 1650,                    # NBA only, optional
        "away_elo": 1520,                    # NBA only, optional
        "home_rest_days": 2,                 # optional
        "away_rest_days": 1,                 # optional
        "b365_odds": {"home": 1.9, "draw": 3.4, "away": 3.8},  # soccer only
        "is_playoffs": false                 # NBA only, optional
      }
    """
    try:
        body = request.get_json(force=True) or {}
        sport = body.get("sport", "nba").lower()

        from ..services.ml_prediction_service import predict_nba_game, predict_soccer_game

        if sport == "nba":
            sport_config = {
                "team_a_name": body.get("team_a_name", "Home Team"),
                "team_b_name": body.get("team_b_name", "Away Team"),
                "sport": "nba",
            }
            from ..ml.nba_predictor import NBAPredictor
            predictor = NBAPredictor()
            predictor.load()
            prediction = predictor.predict(
                home_team=sport_config["team_a_name"],
                away_team=sport_config["team_b_name"],
                home_elo=float(body.get("home_elo", 1500)),
                away_elo=float(body.get("away_elo", 1500)),
                home_rest_days=int(body.get("home_rest_days", 2)),
                away_rest_days=int(body.get("away_rest_days", 2)),
                is_playoffs=bool(body.get("is_playoffs", False)),
                season=2025,
            )

        elif sport == "soccer":
            from ..ml.soccer_predictor import SoccerPredictor
            predictor = SoccerPredictor()
            predictor.load()
            prediction = predictor.predict(
                home_team=body.get("team_a_name", "Home Team"),
                away_team=body.get("team_b_name", "Away Team"),
                league=body.get("league", "EPL"),
                b365_odds=body.get("b365_odds"),
                home_rest_days=int(body.get("home_rest_days", 7)),
                away_rest_days=int(body.get("away_rest_days", 7)),
            )

        else:
            return jsonify({"success": False, "error": f"Unknown sport: {sport}"}), 400

        return jsonify({"success": True, "data": prediction})

    except Exception as e:
        logger.error(f"predict error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@sports_bp.route("/predict/kalshi", methods=["POST"])
def predict_kalshi():
    """
    Kalshi market prediction endpoint.

    Body:
      {
        "market_question": "Will the Fed cut rates in June 2026?",
        "current_market_price": 0.35,   # optional, default 0.5
        "time_to_close_days": 90,       # optional, default 30
        "category": "economics"         # optional
      }
    """
    try:
        body = request.get_json(force=True) or {}
        question = body.get("market_question", "")
        if not question:
            return jsonify({"success": False, "error": "market_question is required"}), 400

        from ..ml.kalshi_predictor import KalshiPredictor
        predictor = KalshiPredictor()
        if not predictor.load():
            predictor._build_fallback_calibrator()

        econ_context = predictor.get_economic_context()
        prediction = predictor.predict(
            market_question=question,
            community_prob=float(body.get("current_market_price", 0.5)),
            category=body.get("category", "economics"),
            time_to_close_days=float(body.get("time_to_close_days", 30)),
            economic_context=econ_context,
        )
        return jsonify({"success": True, "data": prediction})

    except Exception as e:
        logger.error(f"predict_kalshi error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@sports_bp.route("/models/status", methods=["GET"])
def model_status():
    """Check which ML models are trained and ready."""
    from ..ml.model_registry import ModelRegistry
    models = ModelRegistry.list_models()
    status = {}
    for key in ["nba", "soccer", "kalshi"]:
        meta = ModelRegistry.get_meta(key)
        status[key] = {
            "trained": key in models,
            "stats": meta if meta else None,
        }
    return jsonify({"success": True, "data": {"models": status}})
