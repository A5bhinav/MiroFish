# MiroFish — Prediction Betting Extension Plan

## Overview

Two extensions to MiroFish that share a common probability output layer:

1. **Kalshi Bets** — Minimal changes to the existing pipeline. Add a `market_question` field, hook a `ProbabilityExtractor` into the report agent, expose a `/probabilities` endpoint. The full document-upload → simulation → report flow is reused as-is.

2. **Sports Bets** — A parallel data path. Instead of uploading documents, the user picks teams and a game. The system fetches live stats from free APIs, converts them to narrative text, and feeds that text into the exact same Zep graph → OASIS simulation → report pipeline. The only sports-specific divergence is: pre-defined ontology (no LLM needed), pre-defined expert personas (no LLM needed), and probability extraction at the end.

**Core principle:** The existing pipeline from `GraphBuilderService` onward is identical for both paths. The sports extension only replaces Step 1 (file upload + ontology generation). Everything from Step 2 (graph build) to Step 5 (interaction) is untouched.

---

## Codebase Quick Reference

Before building, understand these key existing patterns:

### Backend patterns
- **Blueprint registration** — `backend/app/api/__init__.py` defines blueprints; `backend/app/__init__.py` line 66–69 registers them. Follow this exact pattern for `sports_bp`.
- **Background task thread** — `backend/app/api/graph.py` lines 374–508 show the `def build_task()` + `threading.Thread(target=build_task, daemon=True).start()` pattern. Copy this exactly for the sports ingest task.
- **Task progress** — `task_manager.update_task(task_id, status=TaskStatus.PROCESSING, progress=5, message="...")` then `task_manager.complete_task(task_id, result={...})` or `task_manager.fail_task(task_id, error_str)`.
- **API response shape** — Always `{"success": True, "data": {...}}` on success, `{"success": False, "error": "..."}` on failure. Match this exactly.
- **LLMClient** — `LLMClient().chat_json(messages, temperature=0.1, max_tokens=2048)` returns a parsed dict. Use `temperature=0.1` for deterministic probability extraction.
- **Project model** — `Project` dataclass in `backend/app/models/project.py`. Add new fields at line 53 (after `error`). Update both `to_dict()` and `from_dict()`.

### Frontend patterns
- **API client** — `frontend/src/api/graph.js` uses `import service, { requestWithRetry } from './index'`. Every mutating call wraps in `requestWithRetry(() => service({...}))`. Read-only GETs use `service({...})` directly.
- **Axios instance** — `frontend/src/api/index.js` — baseURL is `http://localhost:5001`, timeout 300000ms. Response interceptor unwraps `res.data` automatically — your code receives `res.data` not the raw axios response.

---

## Part 1 — Kalshi Bets

### What changes

Only 5 things change. Everything else is identical to the existing flow.

---

### 1. `backend/app/models/project.py` — add field

In the `Project` dataclass, after line 53 (`error: Optional[str] = None`), add:

```python
# Kalshi prediction market question (optional)
market_question: Optional[str] = None
```

In `to_dict()`, add one line inside the returned dict:
```python
"market_question": self.market_question,
```

In `from_dict()`, add one line inside the `cls(...)` call:
```python
market_question=data.get('market_question'),
```

---

### 2. `backend/app/services/probability_extractor.py` — new file

```python
"""
Probability Extractor
Post-processes a completed report Markdown into structured probability JSON.
Single deterministic LLM call. Used by both Kalshi and Sports paths.
"""

import json
from typing import Optional, Dict, Any, List
from ..utils.llm_client import LLMClient
from ..utils.logger import get_logger

logger = get_logger('mirofish.probability_extractor')


class ProbabilityExtractor:

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or LLMClient()

    # ── Kalshi ──────────────────────────────────────────────────────────────

    def extract_kalshi(self, report_markdown: str, market_question: str) -> Dict[str, Any]:
        """
        Extract a Yes/No probability for a Kalshi market question from report text.

        Returns:
            {
                "yes_probability": 0.72,
                "no_probability": 0.28,
                "confidence": "medium",   # "high" | "medium" | "low" | "insufficient_data"
                "key_factors": ["Fed hawkish tone", "CPI above expectations"],
                "reasoning_summary": "The simulation agents consistently argued..."
            }
        """
        system_prompt = (
            "You are a precise probability estimator. "
            "You will be given a predictive analysis report and a Yes/No market question. "
            "Extract a probability estimate ONLY from evidence explicitly stated in the report. "
            "Do not add outside knowledge. "
            "Return valid JSON only."
        )

        user_prompt = f"""Market Question: {market_question}

Report:
{report_markdown[:12000]}

Based solely on the analysis and agent consensus described in this report, extract:
1. yes_probability (float 0.0–1.0): probability the market resolves YES
2. no_probability (float): must equal 1.0 - yes_probability
3. confidence: "high" if report gives strong clear evidence, "medium" if mixed signals, "low" if weak evidence, "insufficient_data" if report lacks relevant content
4. key_factors: list of 2–5 short strings summarizing what drove the estimate
5. reasoning_summary: 1–2 sentence summary of agent consensus

Return JSON:
{{
  "yes_probability": 0.72,
  "no_probability": 0.28,
  "confidence": "medium",
  "key_factors": ["...", "..."],
  "reasoning_summary": "..."
}}"""

        try:
            result = self.llm.chat_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=1024
            )
            # Enforce sum-to-1
            yes = float(result.get("yes_probability", 0.5))
            result["yes_probability"] = round(yes, 4)
            result["no_probability"] = round(1.0 - yes, 4)
            return result
        except Exception as e:
            logger.error(f"Kalshi probability extraction failed: {e}")
            return {
                "yes_probability": None,
                "no_probability": None,
                "confidence": "insufficient_data",
                "key_factors": [],
                "reasoning_summary": f"Extraction failed: {str(e)}"
            }

    # ── Sports ───────────────────────────────────────────────────────────────

    def extract_sports(self, report_markdown: str, sport_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract structured betting probabilities from a sports prediction report.

        Returns:
            {
                "moneyline": {
                    "team_a": "Boston Celtics", "team_a_probability": 0.65,
                    "team_b": "Miami Heat",     "team_b_probability": 0.35
                },
                "spread": {
                    "line": -3.5, "favorite": "Boston Celtics", "cover_probability": 0.58
                },
                "total": {
                    "line": 220.5, "over_probability": 0.52
                },
                "player_props": [
                    {"player": "Jayson Tatum", "market": "points",
                     "line": 27.5, "over_probability": 0.61}
                ],
                "confidence": "medium",
                "reasoning_summary": "..."
            }
        """
        team_a = sport_config.get("team_a_name", "Team A")
        team_b = sport_config.get("team_b_name", "Team B")
        bet_types = sport_config.get("bet_types", [])
        prop_players = sport_config.get("player_prop_players", [])

        system_prompt = (
            "You are a sports betting probability analyst. "
            "Extract probability estimates ONLY from evidence in the provided report. "
            "Return null for any market the report does not address. "
            "Return valid JSON only."
        )

        user_prompt = f"""Teams: {team_a} vs {team_b}
Requested bet types: {', '.join(bet_types)}
Player prop players: {', '.join(prop_players) if prop_players else 'none'}

Report:
{report_markdown[:12000]}

Extract betting probabilities from the report. Return JSON matching this exact schema:
{{
  "moneyline": {{
    "team_a": "{team_a}", "team_a_probability": 0.65,
    "team_b": "{team_b}", "team_b_probability": 0.35
  }},
  "spread": {{
    "line": -3.5, "favorite": "{team_a}", "cover_probability": 0.58
  }},
  "total": {{
    "line": 220.5, "over_probability": 0.52
  }},
  "player_props": [
    {{"player": "Player Name", "market": "points", "line": 27.5, "over_probability": 0.61}}
  ],
  "confidence": "medium",
  "reasoning_summary": "One to two sentence summary."
}}

Rules:
- moneyline probabilities must sum to 1.0
- Set any field to null if the report has no evidence for it
- confidence: "high" = clear agent consensus, "medium" = mixed signals, "low" = weak evidence"""

        try:
            result = self.llm.chat_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=2048
            )
            # Enforce moneyline sum-to-1
            ml = result.get("moneyline")
            if ml and ml.get("team_a_probability") is not None:
                a = float(ml["team_a_probability"])
                ml["team_a_probability"] = round(a, 4)
                ml["team_b_probability"] = round(1.0 - a, 4)
            return result
        except Exception as e:
            logger.error(f"Sports probability extraction failed: {e}")
            return {
                "moneyline": None, "spread": None, "total": None,
                "player_props": [], "confidence": "insufficient_data",
                "reasoning_summary": f"Extraction failed: {str(e)}"
            }
```

---

### 3. `backend/app/services/report_agent.py` — two surgical changes

**Change A** — `ReportAgent.__init__` (currently at line 883). Add two optional parameters:

```python
def __init__(
    self,
    graph_id: str,
    simulation_id: str,
    simulation_requirement: str,
    llm_client: Optional[LLMClient] = None,
    zep_tools: Optional[ZepToolsService] = None,
    # NEW:
    market_question: Optional[str] = None,
    sport_config: Optional[Dict[str, Any]] = None,
):
```

Inside `__init__`, after the existing `self.zep_tools = ...` assignment, add:
```python
self.market_question = market_question
self.sport_config = sport_config
```

**Change B** — `ReportAgent.generate_report` (at line 1532). After the block that saves `full_report.md` and before the final `return report`, add:

```python
# Probability extraction (Kalshi or Sports)
try:
    from .probability_extractor import ProbabilityExtractor
    extractor = ProbabilityExtractor(llm_client=self.llm_client)
    probabilities = None

    if self.market_question:
        logger.info("Extracting Kalshi probabilities from report...")
        probabilities = extractor.extract_kalshi(full_markdown, self.market_question)

    elif self.sport_config:
        logger.info("Extracting sports betting probabilities from report...")
        probabilities = extractor.extract_sports(full_markdown, self.sport_config)

    if probabilities:
        prob_path = os.path.join(report_dir, 'probabilities.json')
        with open(prob_path, 'w', encoding='utf-8') as f:
            json.dump(probabilities, f, ensure_ascii=False, indent=2)
        logger.info(f"Probabilities saved to {prob_path}")
except Exception as e:
    logger.warning(f"Probability extraction failed (non-fatal): {e}")
```

---

### 4. `backend/app/api/report.py` — two changes

**Change A** — In `generate_report()` route, after line 101 (`simulation_requirement = project.simulation_requirement`), add:

```python
# Read betting config from project (both optional)
market_question = getattr(project, 'market_question', None)
sport_config = getattr(project, 'sport_config', None)
```

In the `run_generate()` inner function, change the `ReportAgent(...)` constructor call to pass the new params:

```python
agent = ReportAgent(
    graph_id=graph_id,
    simulation_id=simulation_id,
    simulation_requirement=simulation_requirement,
    market_question=market_question,   # NEW
    sport_config=sport_config,         # NEW
)
```

**Change B** — Add a new route at the bottom of the file:

```python
@report_bp.route('/<report_id>/probabilities', methods=['GET'])
def get_probabilities(report_id: str):
    """
    Get structured probability estimates for a report.
    Returns 404 if no probabilities.json exists (non-betting report).
    """
    try:
        prob_path = os.path.join(
            Config.UPLOAD_FOLDER, 'reports', report_id, 'probabilities.json'
        )
        if not os.path.exists(prob_path):
            return jsonify({
                "success": False,
                "error": "No probabilities found for this report"
            }), 404

        with open(prob_path, 'r', encoding='utf-8') as f:
            probabilities = json.load(f)

        return jsonify({
            "success": True,
            "data": {
                "report_id": report_id,
                "probabilities": probabilities
            }
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
```

---

### 5. Frontend — Kalshi UI additions

**`frontend/src/views/MainView.vue`** — In Step 1 (the file upload section), add an optional text field below the simulation requirement field:

```html
<div class="form-group" style="margin-top: 12px;">
  <label>Kalshi Market Question (optional)</label>
  <input
    v-model="marketQuestion"
    type="text"
    placeholder="e.g. Will the Fed cut rates in June 2026?"
    class="form-input"
  />
  <p class="form-hint">If set, a Yes/No probability will be extracted from the report.</p>
</div>
```

Add `marketQuestion: ''` to the component's data/reactive state. When calling `generateOntology`, pass it to the backend via the `FormData` object:
```js
formData.append('market_question', this.marketQuestion)
```

The `/api/graph/ontology/generate` endpoint must also be updated to read `market_question` from the form and save it to the project:
```python
# In graph.py generate_ontology(), after setting project.simulation_requirement:
market_question = request.form.get('market_question', '').strip()
if market_question:
    project.market_question = market_question
```

**`frontend/src/components/KalshiProbabilityBadge.vue`** — new file:

```vue
<template>
  <div v-if="probabilities" class="kalshi-badge">
    <h3>Kalshi Market Prediction</h3>
    <p class="market-question">{{ marketQuestion }}</p>
    <div class="probability-circles">
      <div class="circle yes">
        <span class="pct">{{ (probabilities.yes_probability * 100).toFixed(1) }}%</span>
        <span class="label">YES</span>
      </div>
      <div class="circle no">
        <span class="pct">{{ (probabilities.no_probability * 100).toFixed(1) }}%</span>
        <span class="label">NO</span>
      </div>
    </div>
    <div class="confidence" :class="probabilities.confidence">
      Confidence: {{ probabilities.confidence }}
    </div>
    <ul class="key-factors">
      <li v-for="factor in probabilities.key_factors" :key="factor">{{ factor }}</li>
    </ul>
    <p class="reasoning">{{ probabilities.reasoning_summary }}</p>
  </div>
</template>

<script>
export default {
  props: {
    probabilities: Object,   // the full probabilities JSON object
    marketQuestion: String
  }
}
</script>
```

**`frontend/src/components/Step4Report.vue`** — after report generation completes, call `GET /api/report/{reportId}/probabilities` and if it returns data, render `<KalshiProbabilityBadge>`:

```js
import { getProbabilities } from '../api/report'
// After report completes:
try {
  const res = await getProbabilities(this.reportId)
  this.probabilities = res.data.probabilities
} catch (e) {
  // 404 is expected for non-betting reports — ignore silently
}
```

Add `getProbabilities` to `frontend/src/api/report.js`:
```js
export function getProbabilities(reportId) {
  return service({ url: `/api/report/${reportId}/probabilities`, method: 'get' })
}
```

### Kalshi end-to-end flow

```
User fills in Step 1:
  - Uploads documents (PDF/MD/TXT)
  - Fills simulation_requirement
  - Fills market_question: "Will Fed cut rates in June 2026?"

POST /api/graph/ontology/generate
  → project.market_question = "Will Fed cut rates..."
  → project.status = ONTOLOGY_GENERATED

POST /api/graph/build  →  Zep graph built (unchanged)

POST /api/simulation/create + /prepare + /start  →  OASIS simulation (unchanged)

POST /api/report/generate
  → ReportAgent picks up market_question from project
  → ReACT loop runs (unchanged)
  → After full_report.md saved:
     ProbabilityExtractor.extract_kalshi() called
     probabilities.json saved in report dir

GET /api/report/{id}/probabilities
  → returns {yes_probability: 0.72, no_probability: 0.28, confidence: "medium", ...}

Frontend Step4Report shows KalshiProbabilityBadge
```

---

## Part 2 — Sports Bets

### File map — what to create and what to modify

**New backend files (create from scratch):**
- `backend/app/models/sport_config.py`
- `backend/app/services/sports_data_fetcher.py`
- `backend/app/services/sports_narrative_formatter.py`
- `backend/app/services/sports_ontology_templates.py`
- `backend/app/services/sports_persona_library.py`
- `backend/app/api/sports.py`

**Modified backend files:**
- `backend/app/models/project.py` — add `sport_config` and `is_sports_project` fields
- `backend/app/config.py` — add 3 optional API key fields
- `backend/app/api/__init__.py` — add `sports_bp`
- `backend/app/__init__.py` — register `sports_bp`
- `backend/app/services/oasis_profile_generator.py` — add `generate_sports_profiles()`
- `backend/app/services/report_agent.py` — already done in Kalshi section (sport_config param)
- `backend/app/api/report.py` — already done in Kalshi section

**New frontend files (create from scratch):**
- `frontend/src/api/sports.js`
- `frontend/src/components/ProbabilityDashboard.vue`
- `frontend/src/views/SportsView.vue`
- `frontend/src/views/SportsProbabilityView.vue`

**Modified frontend files:**
- `frontend/src/router/index.js` — add 2 routes
- `frontend/src/views/Home.vue` — add Sports button

---

### `backend/app/models/project.py` — add sport fields

After `market_question: Optional[str] = None`, add:

```python
# Sports betting config (optional — only set for sports projects)
is_sports_project: bool = False
sport_config: Optional[Dict[str, Any]] = None
```

In `to_dict()`, add:
```python
"is_sports_project": self.is_sports_project,
"sport_config": self.sport_config,
```

In `from_dict()`, add:
```python
is_sports_project=data.get('is_sports_project', False),
sport_config=data.get('sport_config'),
```

---

### `backend/app/models/sport_config.py` — new file

```python
"""
SportConfig dataclass — describes the sports matchup and bet types for a sports project.
Serialized to/from project.sport_config (Dict).
"""
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any


@dataclass
class SportConfig:
    sport: str                          # "nba" | "soccer"
    league: str                         # "NBA" | "premier_league" | "la_liga" | "serie_a"
    season: str                         # "2024-25"
    team_a_id: int                      # API-native integer team ID
    team_a_name: str
    team_b_id: int
    team_b_name: str
    game_date: Optional[str] = None     # ISO "2026-03-20" — optional (futures don't have one)
    bet_types: List[str] = field(default_factory=list)
    # "moneyline" | "spread" | "total" | "player_props" | "futures"
    player_prop_players: List[str] = field(default_factory=list)  # player names
    odds_sport_key: str = ""            # The Odds API sport key e.g. "basketball_nba"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SportConfig':
        return cls(
            sport=data['sport'],
            league=data['league'],
            season=data['season'],
            team_a_id=data['team_a_id'],
            team_a_name=data['team_a_name'],
            team_b_id=data['team_b_id'],
            team_b_name=data['team_b_name'],
            game_date=data.get('game_date'),
            bet_types=data.get('bet_types', []),
            player_prop_players=data.get('player_prop_players', []),
            odds_sport_key=data.get('odds_sport_key', ''),
        )
```

---

### `backend/app/config.py` — add API keys

Add three lines after the existing LLM config block:

```python
# Sports API keys (all optional — missing keys degrade gracefully)
BALLDONTLIE_API_KEY: str = os.environ.get('BALLDONTLIE_API_KEY', '')
API_FOOTBALL_KEY: str = os.environ.get('API_FOOTBALL_KEY', '')
ODDS_API_KEY: str = os.environ.get('ODDS_API_KEY', '')
```

Add to `.env` / `.env.example`:
```
# Sports APIs (optional — required for sports betting feature)
BALLDONTLIE_API_KEY=your_key_here
API_FOOTBALL_KEY=your_key_here
ODDS_API_KEY=your_key_here
```

---

### `backend/app/services/sports_data_fetcher.py` — new file

```python
"""
Sports Data Fetcher
Fetches raw data from free sports APIs and assembles a unified matchup dict.
No LLM calls. No text formatting. Pure HTTP + JSON.
"""

import requests
from typing import Dict, Any, List, Optional
from ..config import Config
from ..utils.logger import get_logger
from ..models.sport_config import SportConfig

logger = get_logger('mirofish.sports_fetcher')

NBA_BASE = "https://api.balldontlie.io/v1"
SOCCER_BASE = "https://v3.football.api-sports.io"
ODDS_BASE = "https://api.the-odds-api.com/v4"


class NBADataFetcher:
    """Wraps Ball Don't Lie API v1. Free tier, requires API key."""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or Config.BALLDONTLIE_API_KEY
        self.headers = {"Authorization": self.api_key}

    def _get(self, path: str, params: Dict = None) -> Dict:
        url = f"{NBA_BASE}{path}"
        resp = requests.get(url, headers=self.headers, params=params or {}, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def get_teams(self) -> List[Dict]:
        """List all NBA teams. Returns [{id, full_name, abbreviation, city, conference, division}]"""
        data = self._get("/teams", {"per_page": 100})
        return data.get("data", [])

    def get_team(self, team_id: int) -> Dict:
        return self._get(f"/teams/{team_id}").get("data", {})

    def get_players_for_team(self, team_id: int) -> List[Dict]:
        """Current season roster. Returns [{id, first_name, last_name, position, jersey_number}]"""
        data = self._get("/players", {"team_ids[]": team_id, "per_page": 100})
        return data.get("data", [])

    def get_recent_games(self, team_id: int, n: int = 10) -> List[Dict]:
        """Last n completed games for team. Returns [{id, date, home_team, visitor_team, home_team_score, visitor_team_score}]"""
        import datetime
        today = datetime.date.today().isoformat()
        start = (datetime.date.today() - datetime.timedelta(days=90)).isoformat()
        data = self._get("/games", {
            "team_ids[]": team_id,
            "start_date": start,
            "end_date": today,
            "per_page": n,
            "sort": "date",
            "order": "desc"
        })
        return data.get("data", [])

    def get_season_averages(self, player_ids: List[int], season: int) -> List[Dict]:
        """Season averages for a list of players. Returns [{player_id, pts, ast, reb, fg_pct, min}]"""
        params = {"season": season}
        for pid in player_ids:
            params.setdefault("player_ids[]", []).append(pid)
        try:
            data = self._get("/season_averages", params)
            return data.get("data", [])
        except Exception:
            return []

    def get_head_to_head(self, team_a_id: int, team_b_id: int) -> List[Dict]:
        """Games between two teams this season."""
        data = self._get("/games", {
            "team_ids[]": [team_a_id, team_b_id],
            "per_page": 20,
            "sort": "date",
            "order": "desc"
        })
        games = data.get("data", [])
        # Filter to games that actually involved BOTH teams
        return [
            g for g in games
            if g.get("home_team", {}).get("id") in [team_a_id, team_b_id]
            and g.get("visitor_team", {}).get("id") in [team_a_id, team_b_id]
        ]


class SoccerDataFetcher:
    """Wraps API-Football v3. Free tier: 100 req/day."""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or Config.API_FOOTBALL_KEY
        self.headers = {"x-apisports-key": self.api_key}

    def _get(self, path: str, params: Dict = None) -> Dict:
        url = f"{SOCCER_BASE}{path}"
        resp = requests.get(url, headers=self.headers, params=params or {}, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def get_teams_by_league(self, league_id: int, season: int) -> List[Dict]:
        """Returns [{team: {id, name, code}, venue: {...}}]"""
        data = self._get("/teams", {"league": league_id, "season": season})
        return data.get("response", [])

    def get_squad(self, team_id: int) -> List[Dict]:
        """Returns [{player: {id, name, age, nationality}, statistics: [{games, goals, assists}]}]"""
        data = self._get("/players/squads", {"team": team_id})
        resp = data.get("response", [])
        return resp[0].get("players", []) if resp else []

    def get_recent_fixtures(self, team_id: int, last: int = 10) -> List[Dict]:
        """Returns last n fixtures for team."""
        data = self._get("/fixtures", {"team": team_id, "last": last})
        return data.get("response", [])

    def get_standings(self, league_id: int, season: int) -> List[Dict]:
        """Returns standings table."""
        data = self._get("/standings", {"league": league_id, "season": season})
        resp = data.get("response", [])
        if resp:
            return resp[0].get("league", {}).get("standings", [[]])[0]
        return []


class OddsDataFetcher:
    """Wraps The Odds API. Free tier: 500 req/month."""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or Config.ODDS_API_KEY

    def get_odds(self, sport_key: str, regions: str = "us", markets: str = "h2h,spreads,totals") -> List[Dict]:
        """
        Returns current odds for upcoming games.
        sport_key examples: "basketball_nba", "soccer_epl"
        markets: "h2h" = moneyline, "spreads" = spread, "totals" = over/under
        """
        if not self.api_key:
            logger.warning("ODDS_API_KEY not set — skipping odds fetch")
            return []
        url = f"{ODDS_BASE}/sports/{sport_key}/odds"
        resp = requests.get(url, params={
            "apiKey": self.api_key,
            "regions": regions,
            "markets": markets,
            "oddsFormat": "american"
        }, timeout=10)
        resp.raise_for_status()
        return resp.json()


class SportsDataOrchestrator:
    """
    Top-level orchestrator. Fetches all data for a matchup and returns a single unified dict.
    This dict is then passed to SportsNarrativeFormatter.format().
    """

    def fetch_matchup(self, sport_config: SportConfig) -> Dict[str, Any]:
        """
        Returns:
        {
            "sport": "nba",
            "team_a": {team dict},
            "team_b": {team dict},
            "team_a_players": [...],
            "team_b_players": [...],
            "team_a_recent_games": [...],
            "team_b_recent_games": [...],
            "head_to_head": [...],
            "team_a_averages": [...],
            "team_b_averages": [...],
            "odds_lines": [...],
            "errors": []     # non-fatal fetch errors
        }
        """
        result = {"sport": sport_config.sport, "errors": []}

        if sport_config.sport == "nba":
            fetcher = NBADataFetcher()
            try:
                result["team_a"] = fetcher.get_team(sport_config.team_a_id)
                result["team_b"] = fetcher.get_team(sport_config.team_b_id)
            except Exception as e:
                result["errors"].append(f"Team fetch failed: {e}")
                result["team_a"] = {"full_name": sport_config.team_a_name}
                result["team_b"] = {"full_name": sport_config.team_b_name}

            try:
                result["team_a_players"] = fetcher.get_players_for_team(sport_config.team_a_id)
                result["team_b_players"] = fetcher.get_players_for_team(sport_config.team_b_id)
            except Exception as e:
                result["errors"].append(f"Players fetch failed: {e}")
                result["team_a_players"] = []
                result["team_b_players"] = []

            try:
                result["team_a_recent_games"] = fetcher.get_recent_games(sport_config.team_a_id)
                result["team_b_recent_games"] = fetcher.get_recent_games(sport_config.team_b_id)
                result["head_to_head"] = fetcher.get_head_to_head(
                    sport_config.team_a_id, sport_config.team_b_id
                )
            except Exception as e:
                result["errors"].append(f"Games fetch failed: {e}")
                result["team_a_recent_games"] = []
                result["team_b_recent_games"] = []
                result["head_to_head"] = []

        elif sport_config.sport == "soccer":
            fetcher = SoccerDataFetcher()
            try:
                result["team_a_players"] = fetcher.get_squad(sport_config.team_a_id)
                result["team_b_players"] = fetcher.get_squad(sport_config.team_b_id)
                result["team_a_recent_games"] = fetcher.get_recent_fixtures(sport_config.team_a_id)
                result["team_b_recent_games"] = fetcher.get_recent_fixtures(sport_config.team_b_id)
            except Exception as e:
                result["errors"].append(f"Soccer fetch failed: {e}")

        # Odds (optional — skip if no key)
        try:
            odds_fetcher = OddsDataFetcher()
            if sport_config.odds_sport_key:
                result["odds_lines"] = odds_fetcher.get_odds(sport_config.odds_sport_key)
            else:
                result["odds_lines"] = []
        except Exception as e:
            result["errors"].append(f"Odds fetch failed: {e}")
            result["odds_lines"] = []

        logger.info(f"Matchup fetch complete. Errors: {result['errors']}")
        return result
```

---

### `backend/app/services/sports_narrative_formatter.py` — new file

```python
"""
Sports Narrative Formatter
Converts raw API dicts → List[str] of journalism-style plain text.
NO LLM calls. NO HTTP calls. Pure data transformation.

Each string in the output becomes a separate EpisodeData chunk fed to Zep.
Max ~1500 chars per block to stay within Zep episode size limits.
"""

from typing import Dict, Any, List
from ..models.sport_config import SportConfig


class SportsNarrativeFormatter:

    def format(self, raw_data: Dict[str, Any], sport_config: SportConfig) -> List[str]:
        """
        Main entry point. Returns a list of plain text narrative blocks.
        Each block covers one logical topic (team form, players, h2h, odds, injuries).
        """
        blocks = []

        if sport_config.sport == "nba":
            blocks.append(self._format_nba_team_overview(raw_data, sport_config))
            blocks.append(self._format_nba_players(raw_data, sport_config))
            blocks.append(self._format_nba_head_to_head(raw_data, sport_config))
            odds_block = self._format_odds(raw_data, sport_config)
            if odds_block:
                blocks.append(odds_block)

        elif sport_config.sport == "soccer":
            blocks.append(self._format_soccer_overview(raw_data, sport_config))
            blocks.append(self._format_soccer_players(raw_data, sport_config))
            odds_block = self._format_odds(raw_data, sport_config)
            if odds_block:
                blocks.append(odds_block)

        # Remove empty blocks
        return [b for b in blocks if b and len(b.strip()) > 50]

    def _format_nba_team_overview(self, raw: Dict, cfg: SportConfig) -> str:
        a_games = raw.get("team_a_recent_games", [])
        b_games = raw.get("team_b_recent_games", [])

        a_wins = sum(1 for g in a_games if self._nba_team_won(g, cfg.team_a_id))
        b_wins = sum(1 for g in b_games if self._nba_team_won(g, cfg.team_b_id))

        a_pts = [self._nba_team_score(g, cfg.team_a_id) for g in a_games if self._nba_team_score(g, cfg.team_a_id)]
        b_pts = [self._nba_team_score(g, cfg.team_b_id) for g in b_games if self._nba_team_score(g, cfg.team_b_id)]

        a_avg = round(sum(a_pts) / len(a_pts), 1) if a_pts else 0
        b_avg = round(sum(b_pts) / len(b_pts), 1) if b_pts else 0
        n_a = len(a_games)
        n_b = len(b_games)

        lines = [
            f"{cfg.team_a_name} vs {cfg.team_b_name} — {cfg.league} Matchup Preview",
            "",
            f"The {cfg.team_a_name} have gone {a_wins}-{n_a - a_wins} in their last {n_a} games, "
            f"averaging {a_avg} points per game over that stretch.",
            "",
            f"The {cfg.team_b_name} have gone {b_wins}-{n_b - b_wins} in their last {n_b} games, "
            f"averaging {b_avg} points per game.",
        ]
        if cfg.game_date:
            lines.append(f"\nThis matchup is scheduled for {cfg.game_date}.")

        return "\n".join(lines)

    def _format_nba_players(self, raw: Dict, cfg: SportConfig) -> str:
        a_players = raw.get("team_a_players", [])[:8]
        b_players = raw.get("team_b_players", [])[:8]

        def player_line(p):
            name = f"{p.get('first_name', '')} {p.get('last_name', '')}".strip()
            pos = p.get('position', '')
            return f"  - {name} ({pos})" if pos else f"  - {name}"

        lines = [
            f"{cfg.team_a_name} roster (key players):",
            *[player_line(p) for p in a_players],
            "",
            f"{cfg.team_b_name} roster (key players):",
            *[player_line(p) for p in b_players],
        ]
        return "\n".join(lines)

    def _format_nba_head_to_head(self, raw: Dict, cfg: SportConfig) -> str:
        h2h = raw.get("head_to_head", [])
        if not h2h:
            return f"No recent head-to-head data found for {cfg.team_a_name} vs {cfg.team_b_name}."

        a_wins = sum(1 for g in h2h if self._nba_team_won(g, cfg.team_a_id))
        b_wins = len(h2h) - a_wins
        margins = []
        for g in h2h:
            hs = g.get("home_team_score", 0) or 0
            vs = g.get("visitor_team_score", 0) or 0
            margins.append(abs(hs - vs))
        avg_margin = round(sum(margins) / len(margins), 1) if margins else 0

        lines = [
            f"Head-to-head record (last {len(h2h)} meetings):",
            f"{cfg.team_a_name}: {a_wins} wins | {cfg.team_b_name}: {b_wins} wins",
            f"Average margin of victory: {avg_margin} points",
        ]
        return "\n".join(lines)

    def _format_soccer_overview(self, raw: Dict, cfg: SportConfig) -> str:
        a_games = raw.get("team_a_recent_games", [])
        b_games = raw.get("team_b_recent_games", [])

        def count_wins(games, team_id):
            wins = 0
            for g in games:
                teams = g.get("teams", {})
                goals = g.get("goals", {})
                home_id = teams.get("home", {}).get("id")
                away_id = teams.get("away", {}).get("id")
                home_goals = goals.get("home", 0) or 0
                away_goals = goals.get("away", 0) or 0
                if team_id == home_id and home_goals > away_goals:
                    wins += 1
                elif team_id == away_id and away_goals > home_goals:
                    wins += 1
            return wins

        a_wins = count_wins(a_games, cfg.team_a_id)
        b_wins = count_wins(b_games, cfg.team_b_id)

        lines = [
            f"{cfg.team_a_name} vs {cfg.team_b_name} — {cfg.league} Fixture Preview",
            f"{cfg.team_a_name}: {a_wins} wins in last {len(a_games)} matches.",
            f"{cfg.team_b_name}: {b_wins} wins in last {len(b_games)} matches.",
        ]
        return "\n".join(lines)

    def _format_soccer_players(self, raw: Dict, cfg: SportConfig) -> str:
        a_players = raw.get("team_a_players", [])[:8]
        b_players = raw.get("team_b_players", [])[:8]

        def player_line(p):
            return f"  - {p.get('name', 'Unknown')} (age {p.get('age', '?')})"

        lines = [
            f"{cfg.team_a_name} squad:",
            *[player_line(p) for p in a_players],
            "",
            f"{cfg.team_b_name} squad:",
            *[player_line(p) for p in b_players],
        ]
        return "\n".join(lines)

    def _format_odds(self, raw: Dict, cfg: SportConfig) -> str:
        odds_lines = raw.get("odds_lines", [])
        if not odds_lines:
            return ""

        # Find the game matching our teams
        target = None
        for game in odds_lines:
            home = game.get("home_team", "")
            away = game.get("away_team", "")
            if cfg.team_a_name in home or cfg.team_a_name in away or \
               cfg.team_b_name in home or cfg.team_b_name in away:
                target = game
                break

        if not target:
            return ""

        lines = [f"Current betting odds for {cfg.team_a_name} vs {cfg.team_b_name}:"]
        for bookmaker in target.get("bookmakers", [])[:2]:  # top 2 books
            bk_name = bookmaker.get("title", "")
            for market in bookmaker.get("markets", []):
                market_key = market.get("key", "")
                if market_key == "h2h":
                    for outcome in market.get("outcomes", []):
                        lines.append(f"  {bk_name} moneyline — {outcome['name']}: {outcome['price']:+d}")
                elif market_key == "spreads":
                    for outcome in market.get("outcomes", []):
                        lines.append(f"  {bk_name} spread — {outcome['name']}: {outcome.get('point', '')} ({outcome['price']:+d})")
                elif market_key == "totals":
                    for outcome in market.get("outcomes", []):
                        lines.append(f"  {bk_name} total — {outcome['name']} {outcome.get('point', '')}: ({outcome['price']:+d})")
        return "\n".join(lines)

    def _nba_team_won(self, game: Dict, team_id: int) -> bool:
        home_id = game.get("home_team", {}).get("id")
        home_score = game.get("home_team_score", 0) or 0
        visitor_score = game.get("visitor_team_score", 0) or 0
        if team_id == home_id:
            return home_score > visitor_score
        return visitor_score > home_score

    def _nba_team_score(self, game: Dict, team_id: int) -> int:
        home_id = game.get("home_team", {}).get("id")
        if team_id == home_id:
            return game.get("home_team_score", 0) or 0
        return game.get("visitor_team_score", 0) or 0
```

---

### `backend/app/services/sports_ontology_templates.py` — new file

Must comply with Zep's constraint: exactly 10 entity types, last two must be `Person` and `Organization` (enforced at `ontology_generator.py:288`). Each entity must have a `name`, `description`, `attributes` list, and `examples` list. Each edge must have `name`, `description`, and `source_targets` list.

```python
"""
Pre-defined sports ontology templates.
Bypasses OntologyGenerator entirely — no LLM call needed.
Output shape must match OntologyGenerator.generate() exactly.
"""


NBA_ONTOLOGY = {
    "entity_types": [
        {
            "name": "NBATeam",
            "description": "An NBA franchise",
            "attributes": [
                {"name": "conference", "type": "text", "description": "Eastern or Western"},
                {"name": "division", "type": "text", "description": "Atlantic, Central, etc."},
                {"name": "recent_form", "type": "text", "description": "Win-loss in last 10 games"},
            ],
            "examples": ["Boston Celtics", "Miami Heat", "Golden State Warriors"]
        },
        {
            "name": "NBAPlayer",
            "description": "A professional NBA basketball player",
            "attributes": [
                {"name": "position", "type": "text", "description": "PG, SG, SF, PF, or C"},
                {"name": "ppg", "type": "text", "description": "Points per game average"},
                {"name": "status", "type": "text", "description": "Active, Injured, or Questionable"},
            ],
            "examples": ["Jayson Tatum", "Bam Adebayo", "Stephen Curry"]
        },
        {
            "name": "Coach",
            "description": "An NBA head coach or assistant coach",
            "attributes": [
                {"name": "role", "type": "text", "description": "Head coach or assistant"},
            ],
            "examples": ["Joe Mazzulla", "Erik Spoelstra"]
        },
        {
            "name": "SportsAnalyst",
            "description": "A data-driven sports analytics expert",
            "attributes": [
                {"name": "specialty", "type": "text", "description": "Area of analytical focus"},
            ],
            "examples": ["Advanced metrics analyst", "Shot-quality analyst"]
        },
        {
            "name": "BettingAnalyst",
            "description": "A professional sports betting market analyst",
            "attributes": [
                {"name": "strategy", "type": "text", "description": "Sharp money, public fading, etc."},
            ],
            "examples": ["Line movement tracker", "Closing line value analyst"]
        },
        {
            "name": "InjuryReporter",
            "description": "A journalist specializing in player health and availability",
            "attributes": [],
            "examples": ["Beat reporter covering team injuries"]
        },
        {
            "name": "BeatWriter",
            "description": "A team beat reporter covering day-to-day team news",
            "attributes": [
                {"name": "team_covered", "type": "text", "description": "Which team they cover"},
            ],
            "examples": ["Celtics beat writer", "Heat beat reporter"]
        },
        {
            "name": "Fan",
            "description": "A passionate team fan or supporter",
            "attributes": [
                {"name": "team_allegiance", "type": "text", "description": "Which team they support"},
                {"name": "bias_level", "type": "text", "description": "Degree of emotional bias"},
            ],
            "examples": ["Celtics superfan", "Heat season ticket holder"]
        },
        {
            "name": "Person",
            "description": "A generic person not fitting other categories",
            "attributes": [],
            "examples": ["Commissioner", "Referee"]
        },
        {
            "name": "Organization",
            "description": "A generic organization not fitting other categories",
            "attributes": [],
            "examples": ["NBA League Office", "ESPN", "DraftKings"]
        },
    ],
    "edge_types": [
        {
            "name": "PLAYS_FOR",
            "description": "Player plays for a team",
            "source_targets": [{"source": "NBAPlayer", "target": "NBATeam"}],
            "attributes": []
        },
        {
            "name": "COACHES",
            "description": "Coach leads a team",
            "source_targets": [{"source": "Coach", "target": "NBATeam"}],
            "attributes": []
        },
        {
            "name": "COMPETES_AGAINST",
            "description": "Two teams compete in a game",
            "source_targets": [{"source": "NBATeam", "target": "NBATeam"}],
            "attributes": [{"name": "game_date", "type": "text", "description": "Date of the game"}]
        },
        {
            "name": "COVERS",
            "description": "A journalist covers a team",
            "source_targets": [
                {"source": "BeatWriter", "target": "NBATeam"},
                {"source": "InjuryReporter", "target": "NBATeam"}
            ],
            "attributes": []
        },
        {
            "name": "REPORTS_ON",
            "description": "Reporter reports on a player's status",
            "source_targets": [{"source": "InjuryReporter", "target": "NBAPlayer"}],
            "attributes": []
        },
        {
            "name": "SUPPORTS",
            "description": "Fan supports a team",
            "source_targets": [{"source": "Fan", "target": "NBATeam"}],
            "attributes": []
        },
        {
            "name": "BETS_ON",
            "description": "Betting analyst analyzes or bets on a team or game",
            "source_targets": [
                {"source": "BettingAnalyst", "target": "NBATeam"},
                {"source": "BettingAnalyst", "target": "NBAPlayer"}
            ],
            "attributes": [{"name": "bet_type", "type": "text", "description": "moneyline/spread/total/prop"}]
        },
    ]
}


SOCCER_ONTOLOGY = {
    "entity_types": [
        {
            "name": "SoccerTeam",
            "description": "A professional football/soccer club",
            "attributes": [
                {"name": "league", "type": "text", "description": "Which league they play in"},
                {"name": "form", "type": "text", "description": "Last 5 results (W/D/L)"},
            ],
            "examples": ["Arsenal", "Real Madrid", "Bayern Munich"]
        },
        {
            "name": "SoccerPlayer",
            "description": "A professional football player",
            "attributes": [
                {"name": "position", "type": "text", "description": "GK, DEF, MID, FWD"},
                {"name": "goals_this_season", "type": "text", "description": "Goal tally"},
                {"name": "status", "type": "text", "description": "Available, Injured, Suspended"},
            ],
            "examples": ["Erling Haaland", "Bukayo Saka"]
        },
        {
            "name": "Manager",
            "description": "A club manager or head coach",
            "attributes": [],
            "examples": ["Pep Guardiola", "Mikel Arteta"]
        },
        {
            "name": "SportsAnalyst",
            "description": "A football analytics or tactics expert",
            "attributes": [],
            "examples": ["Expected goals analyst", "Pressing metrics expert"]
        },
        {
            "name": "BettingAnalyst",
            "description": "A football betting market specialist",
            "attributes": [],
            "examples": ["Asian handicap specialist"]
        },
        {
            "name": "InjuryReporter",
            "description": "Reporter covering player fitness and availability",
            "attributes": [],
            "examples": ["Club physio reporter"]
        },
        {
            "name": "BeatWriter",
            "description": "Club-specific beat journalist",
            "attributes": [{"name": "club_covered", "type": "text", "description": "Club name"}],
            "examples": ["Arsenal correspondent"]
        },
        {
            "name": "Fan",
            "description": "A passionate club supporter",
            "attributes": [{"name": "club_allegiance", "type": "text", "description": "Supported club"}],
            "examples": ["Arsenal ultras member"]
        },
        {
            "name": "Person",
            "description": "Generic person not fitting other categories",
            "attributes": [],
            "examples": ["Referee", "VAR official"]
        },
        {
            "name": "Organization",
            "description": "Generic organization not fitting other categories",
            "attributes": [],
            "examples": ["FIFA", "UEFA", "Premier League office"]
        },
    ],
    "edge_types": [
        {
            "name": "PLAYS_FOR",
            "description": "Player plays for a club",
            "source_targets": [{"source": "SoccerPlayer", "target": "SoccerTeam"}],
            "attributes": []
        },
        {
            "name": "MANAGES",
            "description": "Manager manages a club",
            "source_targets": [{"source": "Manager", "target": "SoccerTeam"}],
            "attributes": []
        },
        {
            "name": "COMPETES_AGAINST",
            "description": "Two clubs compete in a fixture",
            "source_targets": [{"source": "SoccerTeam", "target": "SoccerTeam"}],
            "attributes": [{"name": "fixture_date", "type": "text", "description": "Date of fixture"}]
        },
        {
            "name": "COVERS",
            "description": "Journalist covers a club",
            "source_targets": [{"source": "BeatWriter", "target": "SoccerTeam"}],
            "attributes": []
        },
        {
            "name": "REPORTS_ON",
            "description": "Injury reporter tracks a player",
            "source_targets": [{"source": "InjuryReporter", "target": "SoccerPlayer"}],
            "attributes": []
        },
        {
            "name": "SUPPORTS",
            "description": "Fan supports a club",
            "source_targets": [{"source": "Fan", "target": "SoccerTeam"}],
            "attributes": []
        },
        {
            "name": "BETS_ON",
            "description": "Betting analyst analyses a club or player",
            "source_targets": [
                {"source": "BettingAnalyst", "target": "SoccerTeam"},
                {"source": "BettingAnalyst", "target": "SoccerPlayer"}
            ],
            "attributes": []
        },
    ]
}


def get_sports_ontology(sport: str) -> Dict:
    if sport == "nba":
        return NBA_ONTOLOGY
    elif sport == "soccer":
        return SOCCER_ONTOLOGY
    else:
        raise ValueError(f"Unknown sport: {sport}. Supported: 'nba', 'soccer'")
```

---

### `backend/app/services/sports_persona_library.py` — new file

These archetypes are used instead of LLM-generated personas. All `OasisAgentProfile` required fields are pre-populated.

```python
"""
Sports Persona Library
Pre-defined expert agent archetypes for sports simulations.
No LLM call needed. Real team/player names are substituted at profile-generation time.
"""

from typing import List, Dict, Any
from .oasis_profile_generator import OasisAgentProfile


def get_archetypes_for_sport(
    sport: str,
    team_a_name: str,
    team_b_name: str,
    start_user_id: int = 1000
) -> List[OasisAgentProfile]:
    """
    Returns 6 pre-defined agent profiles for a sports simulation.
    team_a is the "home" or first-listed team.
    """

    raw_archetypes = [
        {
            "user_name": f"beatwriter_{team_a_name.lower().replace(' ', '_')}",
            "name": f"{team_a_name} Beat Reporter",
            "bio": f"Beat reporter covering {team_a_name} for 8 years. "
                   f"Posts daily practice reports, injury updates, and locker room news.",
            "persona": (
                f"You cover {team_a_name} every day. You have insider access. "
                "You report factually, but lean slightly toward your team. "
                "You focus on injuries, rotations, and coach decisions. "
                "You always cite what you observed at practice or from sources close to the team."
            ),
            "profession": "Sports Journalist",
            "mbti": "ISTJ",
            "age": 38,
            "gender": "male",
            "country": "United States",
            "interested_topics": ["NBA", sport, team_a_name, "injuries", "trade rumors"],
            "follower_count": 45000,
            "friend_count": 800,
            "karma": 12000,
        },
        {
            "user_name": "stats_analyst_nba",
            "name": "Advanced Stats Analyst",
            "bio": "Sports analytics writer. PER, true shooting %, LEBRON, EPM. Data over narrative.",
            "persona": (
                "You are deeply data-driven. You distrust narratives not backed by numbers. "
                "You cite advanced metrics like true shooting percentage, net rating, RAPTOR. "
                "You will push back on conventional wisdom if the numbers contradict it. "
                "You are skeptical of hot streaks and small sample sizes."
            ),
            "profession": "Sports Data Analyst",
            "mbti": "INTJ",
            "age": 31,
            "gender": "non-binary",
            "country": "United States",
            "interested_topics": ["analytics", "statistics", "NBA", sport, "modeling"],
            "follower_count": 28000,
            "friend_count": 500,
            "karma": 8000,
        },
        {
            "user_name": "injury_desk_reporter",
            "name": "Injury & Availability Desk",
            "bio": "Former athletic trainer. Now tracks player health and injury reports for fantasy & betting.",
            "persona": (
                "You obsess over player health status. Any questionable tag matters. "
                "You immediately calculate the impact of an injury on a team's win probability. "
                "You are slightly pessimistic — you assume injured players won't play "
                "until confirmed active. You cite medical reasoning for return timelines."
            ),
            "profession": "Injury Analyst",
            "mbti": "ISFJ",
            "age": 44,
            "gender": "female",
            "country": "United States",
            "interested_topics": ["injuries", "player health", "fantasy sports", "betting"],
            "follower_count": 19000,
            "friend_count": 300,
            "karma": 5000,
        },
        {
            "user_name": "sharp_bettor_lines",
            "name": "Sharp Money Tracker",
            "bio": "Professional bettor. Tracks line movement, reverse line movement, and CLV.",
            "persona": (
                "You focus on market signals: where is the sharp money going? "
                "You discuss closing line value (CLV), reverse line movement, steam moves. "
                "You fade the public when they are overloaded on one side. "
                "You do not have emotional attachment to any team — only edges matter. "
                "You always consider the implied probability of the current market line."
            ),
            "profession": "Sports Betting Analyst",
            "mbti": "ENTP",
            "age": 35,
            "gender": "male",
            "country": "United States",
            "interested_topics": ["sports betting", "line movement", "gambling", "odds"],
            "follower_count": 33000,
            "friend_count": 600,
            "karma": 9500,
        },
        {
            "user_name": f"homer_fan_{team_a_name.lower().replace(' ', '_')}",
            "name": f"{team_a_name} Superfan",
            "bio": f"Die-hard {team_a_name} fan since childhood. Season ticket holder.",
            "persona": (
                f"You are emotionally invested in {team_a_name}. "
                "You overestimate your team's strengths and dismiss the opponent's. "
                "You find silver linings in losses and exaggerate wins. "
                "You get defensive when anyone criticizes your team. "
                "You are convinced your team will win and argue passionately for it."
            ),
            "profession": "Superfan",
            "mbti": "ESFP",
            "age": 29,
            "gender": "male",
            "country": "United States",
            "interested_topics": [team_a_name, sport, "game day", "fan experience"],
            "follower_count": 3000,
            "friend_count": 400,
            "karma": 1500,
        },
        {
            "user_name": "national_pundit_tv",
            "name": "National TV Sports Pundit",
            "bio": "TV analyst. Takes bold contrarian positions. Known for hot takes and strong opinions.",
            "persona": (
                "You gravitate toward bold, narratively compelling takes. "
                "You sometimes pick against the consensus to stand out. "
                "You overweight recent performance and underweight sample size. "
                "You are confident even when wrong. You phrase things as declarations, not questions. "
                "You focus on stars, storylines, and matchups rather than team-level stats."
            ),
            "profession": "Sports Media Personality",
            "mbti": "ESTP",
            "age": 52,
            "gender": "male",
            "country": "United States",
            "interested_topics": [sport, "media", "takes", "storylines", team_b_name],
            "follower_count": 250000,
            "friend_count": 1200,
            "karma": 50000,
        },
    ]

    profiles = []
    for i, arch in enumerate(raw_archetypes):
        profile = OasisAgentProfile(
            user_id=start_user_id + i,
            user_name=arch["user_name"],
            name=arch["name"],
            bio=arch["bio"],
            persona=arch["persona"],
            profession=arch.get("profession"),
            mbti=arch.get("mbti"),
            age=arch.get("age"),
            gender=arch.get("gender"),
            country=arch.get("country"),
            interested_topics=arch.get("interested_topics", []),
            follower_count=arch.get("follower_count", 150),
            friend_count=arch.get("friend_count", 100),
            karma=arch.get("karma", 1000),
        )
        profiles.append(profile)

    return profiles
```

---

### `backend/app/services/oasis_profile_generator.py` — add one method

Add `generate_sports_profiles()` as a new classmethod or standalone function at the bottom of the file, after the existing class definition. Do NOT modify any existing methods.

```python
def generate_sports_profiles(
    team_a_name: str,
    team_b_name: str,
    sport: str,
    start_user_id: int = 1000
) -> list:
    """
    Generates agent profiles for a sports simulation using pre-defined archetypes.
    Returns a list of OasisAgentProfile objects.
    Does NOT call the LLM — uses sports_persona_library directly.
    """
    from .sports_persona_library import get_archetypes_for_sport
    return get_archetypes_for_sport(
        sport=sport,
        team_a_name=team_a_name,
        team_b_name=team_b_name,
        start_user_id=start_user_id
    )
```

---

### `backend/app/api/__init__.py` — add sports blueprint

```python
"""
API路由模块
"""

from flask import Blueprint

graph_bp = Blueprint('graph', __name__)
simulation_bp = Blueprint('simulation', __name__)
report_bp = Blueprint('report', __name__)
sports_bp = Blueprint('sports', __name__)          # NEW

from . import graph    # noqa: E402, F401
from . import simulation  # noqa: E402, F401
from . import report   # noqa: E402, F401
from . import sports   # noqa: E402, F401          # NEW
```

---

### `backend/app/__init__.py` — register sports blueprint

Change line 66:
```python
from .api import graph_bp, simulation_bp, report_bp
```
To:
```python
from .api import graph_bp, simulation_bp, report_bp, sports_bp
```

After line 69, add:
```python
app.register_blueprint(sports_bp, url_prefix='/api/sports')
```

---

### `backend/app/api/sports.py` — new file (full implementation)

```python
"""
Sports Betting API
Handles sports data ingestion and probability extraction for NBA/Soccer predictions.
"""

import traceback
import threading
import uuid
from flask import request, jsonify

from . import sports_bp
from ..config import Config
from ..models.project import ProjectManager, ProjectStatus
from ..models.sport_config import SportConfig
from ..models.task import TaskManager, TaskStatus
from ..services.sports_data_fetcher import SportsDataOrchestrator, NBADataFetcher, SoccerDataFetcher
from ..services.sports_narrative_formatter import SportsNarrativeFormatter
from ..services.sports_ontology_templates import get_sports_ontology
from ..services.text_processor import TextProcessor
from ..services.graph_builder import GraphBuilderService
from ..utils.logger import get_logger

logger = get_logger('mirofish.api.sports')


@sports_bp.route('/teams', methods=['GET'])
def get_teams():
    """
    GET /api/sports/teams?sport=nba
    Returns list of teams for the given sport.
    """
    try:
        sport = request.args.get('sport', 'nba').lower()

        if sport == 'nba':
            fetcher = NBADataFetcher()
            if not Config.BALLDONTLIE_API_KEY:
                return jsonify({"success": False, "error": "BALLDONTLIE_API_KEY not configured"}), 400
            teams = fetcher.get_teams()
            formatted = [
                {
                    "id": t.get("id"),
                    "name": t.get("full_name"),
                    "abbreviation": t.get("abbreviation"),
                    "city": t.get("city"),
                    "conference": t.get("conference"),
                    "division": t.get("division"),
                }
                for t in teams
            ]
        else:
            return jsonify({"success": False, "error": f"Sport '{sport}' not yet supported for team listing"}), 400

        return jsonify({"success": True, "data": {"sport": sport, "teams": formatted}})
    except Exception as e:
        return jsonify({"success": False, "error": str(e), "traceback": traceback.format_exc()}), 500


@sports_bp.route('/players', methods=['GET'])
def get_players():
    """
    GET /api/sports/players?sport=nba&team_id=2
    Returns players for the given team.
    """
    try:
        sport = request.args.get('sport', 'nba').lower()
        team_id = request.args.get('team_id', type=int)

        if not team_id:
            return jsonify({"success": False, "error": "team_id is required"}), 400

        if sport == 'nba':
            fetcher = NBADataFetcher()
            players = fetcher.get_players_for_team(team_id)
            formatted = [
                {
                    "id": p.get("id"),
                    "name": f"{p.get('first_name', '')} {p.get('last_name', '')}".strip(),
                    "position": p.get("position", ""),
                    "jersey_number": p.get("jersey_number"),
                }
                for p in players
            ]
        else:
            return jsonify({"success": False, "error": f"Sport '{sport}' not supported"}), 400

        return jsonify({"success": True, "data": {"team_id": team_id, "players": formatted}})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@sports_bp.route('/odds', methods=['GET'])
def get_odds():
    """
    GET /api/sports/odds?sport=nba&markets=h2h,spreads,totals
    Returns current betting odds from The Odds API.
    """
    try:
        sport = request.args.get('sport', 'nba').lower()
        markets = request.args.get('markets', 'h2h,spreads,totals')

        sport_key_map = {
            "nba": "basketball_nba",
            "soccer_epl": "soccer_epl",
            "soccer_laliga": "soccer_spain_la_liga",
        }
        odds_sport_key = sport_key_map.get(sport, f"basketball_{sport}")

        from ..services.sports_data_fetcher import OddsDataFetcher
        fetcher = OddsDataFetcher()
        if not Config.ODDS_API_KEY:
            return jsonify({"success": False, "error": "ODDS_API_KEY not configured"}), 400

        games = fetcher.get_odds(sport_key=odds_sport_key, markets=markets)
        return jsonify({"success": True, "data": {"sport_key": odds_sport_key, "games": games}})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@sports_bp.route('/ingest', methods=['POST'])
def ingest_sports_data():
    """
    POST /api/sports/ingest
    Fetches sports data, builds narrative text, and creates a Zep knowledge graph.
    This replaces the document-upload + ontology-generate flow for sports projects.

    Body:
    {
        "project_name": "Celtics vs Heat",
        "sport": "nba",
        "league": "NBA",
        "season": "2024-25",
        "team_a_id": 2,
        "team_a_name": "Boston Celtics",
        "team_b_id": 14,
        "team_b_name": "Miami Heat",
        "game_date": "2026-03-20",
        "bet_types": ["moneyline", "spread", "total", "player_props"],
        "player_prop_players": ["Jayson Tatum", "Bam Adebayo"],
        "simulation_requirement": "Predict the outcome of this game"
    }
    """
    try:
        data = request.get_json() or {}

        # Validate required fields
        required = ["sport", "team_a_id", "team_a_name", "team_b_id", "team_b_name"]
        missing = [f for f in required if not data.get(f)]
        if missing:
            return jsonify({"success": False, "error": f"Missing required fields: {missing}"}), 400

        sport = data["sport"].lower()
        if sport not in ["nba", "soccer"]:
            return jsonify({"success": False, "error": "sport must be 'nba' or 'soccer'"}), 400

        # Build SportConfig
        sport_config = SportConfig(
            sport=sport,
            league=data.get("league", sport.upper()),
            season=data.get("season", "2024-25"),
            team_a_id=int(data["team_a_id"]),
            team_a_name=data["team_a_name"],
            team_b_id=int(data["team_b_id"]),
            team_b_name=data["team_b_name"],
            game_date=data.get("game_date"),
            bet_types=data.get("bet_types", ["moneyline"]),
            player_prop_players=data.get("player_prop_players", []),
            odds_sport_key=data.get("odds_sport_key", f"basketball_{sport}" if sport == "nba" else ""),
        )

        # Create project
        project_name = data.get("project_name", f"{sport_config.team_a_name} vs {sport_config.team_b_name}")
        project = ProjectManager.create_project(name=project_name)
        project.simulation_requirement = data.get(
            "simulation_requirement",
            f"Predict the outcome of the {sport_config.team_a_name} vs {sport_config.team_b_name} game"
        )
        project.is_sports_project = True
        project.sport_config = sport_config.to_dict()
        ProjectManager.save_project(project)

        # Create async task
        task_manager = TaskManager()
        task_id = task_manager.create_task(f"Sports ingest: {project_name}")
        logger.info(f"Created sports ingest task: {task_id} for project {project.project_id}")

        def ingest_task():
            ingest_logger = get_logger('mirofish.sports_ingest')
            try:
                # Step 1 — Fetch sports data
                task_manager.update_task(task_id, status=TaskStatus.PROCESSING, progress=5,
                                         message="Fetching sports data from APIs...")
                orchestrator = SportsDataOrchestrator()
                raw_data = orchestrator.fetch_matchup(sport_config)
                ingest_logger.info(f"Fetched matchup data. Errors: {raw_data.get('errors', [])}")

                # Step 2 — Format as narrative text
                task_manager.update_task(task_id, progress=20, message="Formatting narrative text...")
                formatter = SportsNarrativeFormatter()
                document_texts = formatter.format(raw_data, sport_config)

                if not document_texts:
                    raise ValueError("No narrative text generated from sports data — check API keys and team IDs")

                all_text = "\n\n".join(document_texts)
                project.total_text_length = len(all_text)
                ProjectManager.save_extracted_text(project.project_id, all_text)

                # Step 3 — Set pre-defined ontology (no LLM)
                task_manager.update_task(task_id, progress=30, message="Loading sports ontology template...")
                ontology = get_sports_ontology(sport)
                project.ontology = {
                    "entity_types": ontology["entity_types"],
                    "edge_types": ontology["edge_types"],
                }
                project.status = ProjectStatus.ONTOLOGY_GENERATED
                ProjectManager.save_project(project)

                # Step 4 — Build Zep graph (same as /api/graph/build logic)
                task_manager.update_task(task_id, progress=35, message="Creating Zep knowledge graph...")
                builder = GraphBuilderService(api_key=Config.ZEP_API_KEY)

                chunks = TextProcessor.split_text(all_text, chunk_size=project.chunk_size,
                                                  overlap=project.chunk_overlap)

                graph_name = project_name
                graph_id = builder.create_graph(name=graph_name)
                project.graph_id = graph_id
                project.status = ProjectStatus.GRAPH_BUILDING
                ProjectManager.save_project(project)

                task_manager.update_task(task_id, progress=40, message="Setting ontology in Zep...")
                builder.set_ontology(graph_id, project.ontology)

                def graph_progress(msg, ratio):
                    progress = 40 + int(ratio * 45)  # 40% to 85%
                    task_manager.update_task(task_id, progress=progress, message=msg)

                task_manager.update_task(task_id, progress=42, message="Adding text chunks to Zep...")
                builder.add_text_batches(graph_id, chunks, progress_callback=graph_progress)

                task_manager.update_task(task_id, progress=86, message="Waiting for Zep to process episodes...")
                builder.wait_for_episodes(graph_id)

                # Step 5 — Finalize
                project.status = ProjectStatus.GRAPH_COMPLETED
                ProjectManager.save_project(project)

                task_manager.complete_task(task_id, result={
                    "project_id": project.project_id,
                    "graph_id": graph_id,
                    "sport": sport,
                    "team_a": sport_config.team_a_name,
                    "team_b": sport_config.team_b_name,
                    "text_length": project.total_text_length,
                    "narrative_blocks": len(document_texts),
                })
                ingest_logger.info(f"Sports ingest complete for project {project.project_id}")

            except Exception as e:
                ingest_logger.error(f"Sports ingest failed: {e}\n{traceback.format_exc()}")
                project.status = ProjectStatus.FAILED
                project.error = str(e)
                ProjectManager.save_project(project)
                task_manager.fail_task(task_id, str(e))

        thread = threading.Thread(target=ingest_task, daemon=True)
        thread.start()

        return jsonify({
            "success": True,
            "data": {
                "project_id": project.project_id,
                "task_id": task_id,
                "sport": sport,
                "team_a": sport_config.team_a_name,
                "team_b": sport_config.team_b_name,
                "message": "Sports data ingestion started. Poll /api/sports/ingest/status/<task_id> for progress."
            }
        })

    except Exception as e:
        logger.error(f"Failed to start sports ingest: {e}")
        return jsonify({"success": False, "error": str(e), "traceback": traceback.format_exc()}), 500


@sports_bp.route('/ingest/status/<task_id>', methods=['GET'])
def get_ingest_status(task_id: str):
    """
    GET /api/sports/ingest/status/<task_id>
    Delegates to TaskManager. Same response shape as /api/graph/task/<task_id>.
    """
    try:
        task_manager = TaskManager()
        task = task_manager.get_task(task_id)
        if not task:
            return jsonify({"success": False, "error": f"Task not found: {task_id}"}), 404
        return jsonify({"success": True, "data": task.to_dict()})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@sports_bp.route('/project/<project_id>/config', methods=['GET'])
def get_sports_project_config(project_id: str):
    """
    GET /api/sports/project/<project_id>/config
    Returns sports config for a sports project.
    """
    try:
        project = ProjectManager.get_project(project_id)
        if not project:
            return jsonify({"success": False, "error": f"Project not found: {project_id}"}), 404
        return jsonify({
            "success": True,
            "data": {
                "project_id": project_id,
                "is_sports_project": getattr(project, 'is_sports_project', False),
                "sport_config": getattr(project, 'sport_config', None),
                "status": project.status.value if hasattr(project.status, 'value') else project.status,
            }
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
```

---

### `backend/app/api/graph.py` — add sports prepare support

In the `generate_ontology()` route, after setting `project.simulation_requirement`, add:

```python
# Kalshi market question (optional)
market_question = request.form.get('market_question', '').strip()
if market_question:
    project.market_question = market_question
```

---

### Frontend — `frontend/src/api/sports.js` — new file

```js
import service, { requestWithRetry } from './index'

/**
 * Get team list for a sport
 * @param {string} sport - "nba" | "soccer"
 * @param {string} league - optional league filter
 */
export function getTeams(sport, league = '') {
  return service({
    url: '/api/sports/teams',
    method: 'get',
    params: { sport, league }
  })
}

/**
 * Get players for a team
 * @param {string} sport
 * @param {number} teamId
 */
export function getPlayers(sport, teamId) {
  return service({
    url: '/api/sports/players',
    method: 'get',
    params: { sport, team_id: teamId }
  })
}

/**
 * Get current betting odds
 * @param {string} sport
 * @param {string} markets - comma separated: "h2h,spreads,totals"
 */
export function getOdds(sport, markets = 'h2h,spreads,totals') {
  return service({
    url: '/api/sports/odds',
    method: 'get',
    params: { sport, markets }
  })
}

/**
 * Ingest sports data and build knowledge graph (async)
 * @param {Object} data - SportConfig fields + project_name + simulation_requirement
 */
export function ingestSportsData(data) {
  return requestWithRetry(() =>
    service({
      url: '/api/sports/ingest',
      method: 'post',
      data
    })
  )
}

/**
 * Poll sports ingest task status
 * @param {string} taskId
 */
export function getIngestStatus(taskId) {
  return service({
    url: `/api/sports/ingest/status/${taskId}`,
    method: 'get'
  })
}

/**
 * Get sports config for a project
 * @param {string} projectId
 */
export function getSportsProjectConfig(projectId) {
  return service({
    url: `/api/sports/project/${projectId}/config`,
    method: 'get'
  })
}
```

Also add `getProbabilities` to `frontend/src/api/report.js`:

```js
export function getProbabilities(reportId) {
  return service({
    url: `/api/report/${reportId}/probabilities`,
    method: 'get'
  })
}
```

---

### Frontend — `frontend/src/components/ProbabilityDashboard.vue` — new file

```vue
<template>
  <div class="probability-dashboard" v-if="probabilities">
    <h3 class="dashboard-title">Betting Probability Estimates</h3>

    <!-- Moneyline -->
    <div class="bet-section" v-if="probabilities.moneyline">
      <h4>Moneyline (Win Probability)</h4>
      <div class="prob-bars">
        <div class="team-bar">
          <span class="team-name">{{ probabilities.moneyline.team_a }}</span>
          <div class="bar-bg">
            <div class="bar-fill team-a"
                 :style="{ width: (probabilities.moneyline.team_a_probability * 100) + '%' }"/>
          </div>
          <span class="pct">{{ (probabilities.moneyline.team_a_probability * 100).toFixed(1) }}%</span>
        </div>
        <div class="team-bar">
          <span class="team-name">{{ probabilities.moneyline.team_b }}</span>
          <div class="bar-bg">
            <div class="bar-fill team-b"
                 :style="{ width: (probabilities.moneyline.team_b_probability * 100) + '%' }"/>
          </div>
          <span class="pct">{{ (probabilities.moneyline.team_b_probability * 100).toFixed(1) }}%</span>
        </div>
      </div>
    </div>

    <!-- Spread -->
    <div class="bet-section" v-if="probabilities.spread && probabilities.spread.cover_probability">
      <h4>Spread</h4>
      <p>
        {{ probabilities.spread.favorite }} {{ probabilities.spread.line > 0 ? '+' : '' }}{{ probabilities.spread.line }}
        — Cover probability: <strong>{{ (probabilities.spread.cover_probability * 100).toFixed(1) }}%</strong>
      </p>
    </div>

    <!-- Total -->
    <div class="bet-section" v-if="probabilities.total && probabilities.total.over_probability">
      <h4>Total (Over/Under)</h4>
      <p>
        Line: {{ probabilities.total.line }} —
        Over: <strong>{{ (probabilities.total.over_probability * 100).toFixed(1) }}%</strong> /
        Under: <strong>{{ ((1 - probabilities.total.over_probability) * 100).toFixed(1) }}%</strong>
      </p>
    </div>

    <!-- Player Props -->
    <div class="bet-section" v-if="probabilities.player_props && probabilities.player_props.length">
      <h4>Player Props</h4>
      <table class="props-table">
        <thead>
          <tr><th>Player</th><th>Market</th><th>Line</th><th>Over %</th></tr>
        </thead>
        <tbody>
          <tr v-for="prop in probabilities.player_props" :key="prop.player + prop.market">
            <td>{{ prop.player }}</td>
            <td>{{ prop.market }}</td>
            <td>{{ prop.line }}</td>
            <td>{{ (prop.over_probability * 100).toFixed(1) }}%</td>
          </tr>
        </tbody>
      </table>
    </div>

    <!-- Confidence + Summary -->
    <div class="confidence-block" :class="'confidence-' + probabilities.confidence">
      Confidence: <strong>{{ probabilities.confidence }}</strong>
    </div>
    <p class="reasoning-summary" v-if="probabilities.reasoning_summary">
      {{ probabilities.reasoning_summary }}
    </p>
  </div>
</template>

<script>
export default {
  name: 'ProbabilityDashboard',
  props: {
    probabilities: {
      type: Object,
      default: null
    }
  }
}
</script>
```

---

### Frontend — `frontend/src/views/SportsView.vue` — new file (structure)

This view mirrors `MainView.vue`'s 5-step structure. Step 1 is replaced with a sports form. Steps 2–5 import and reuse the existing step components unchanged.

```vue
<template>
  <div class="sports-view">
    <h1>Sports Prediction</h1>

    <!-- Step indicator (same pattern as MainView) -->
    <div class="step-indicator">
      <span v-for="n in 5" :key="n" :class="{ active: currentStep === n, done: currentStep > n }">
        {{ n }}
      </span>
    </div>

    <!-- STEP 1: Sports Form (replaces file upload) -->
    <div v-if="currentStep === 1" class="step-content">
      <h2>Step 1 — Select Matchup</h2>

      <div class="form-row">
        <label>Sport</label>
        <select v-model="sport" @change="onSportChange">
          <option value="nba">NBA Basketball</option>
          <option value="soccer">Soccer</option>
        </select>
      </div>

      <div class="form-row">
        <label>Team A</label>
        <select v-model="teamAId" @change="onTeamAChange">
          <option v-for="t in teams" :key="t.id" :value="t.id">{{ t.name }}</option>
        </select>
      </div>

      <div class="form-row">
        <label>Team B</label>
        <select v-model="teamBId">
          <option v-for="t in teams.filter(t => t.id !== teamAId)" :key="t.id" :value="t.id">
            {{ t.name }}
          </option>
        </select>
      </div>

      <div class="form-row">
        <label>Game Date (optional)</label>
        <input type="date" v-model="gameDate" />
      </div>

      <div class="form-row">
        <label>Bet Types</label>
        <label v-for="bt in betTypeOptions" :key="bt.value">
          <input type="checkbox" :value="bt.value" v-model="betTypes" />
          {{ bt.label }}
        </label>
      </div>

      <div class="form-row" v-if="betTypes.includes('player_props') && teamAPlayers.length">
        <label>Player Prop Players</label>
        <select multiple v-model="propPlayers">
          <option v-for="p in [...teamAPlayers, ...teamBPlayers]" :key="p.id" :value="p.name">
            {{ p.name }}
          </option>
        </select>
      </div>

      <div class="form-row">
        <label>Simulation Requirement</label>
        <textarea v-model="simulationRequirement" rows="3" />
      </div>

      <button @click="submitSportsIngest" :disabled="ingestLoading">
        {{ ingestLoading ? 'Fetching data...' : 'Start Prediction' }}
      </button>
    </div>

    <!-- STEPS 2–5: Reuse existing components -->
    <Step2EnvSetup v-if="currentStep === 2" :projectId="projectId" :graphId="graphId"
                   @next="onStep2Complete" />
    <Step3Simulation v-if="currentStep === 3" :projectId="projectId"
                     @next="onStep3Complete" />
    <Step4Report v-if="currentStep === 4" :simulationId="simulationId"
                 @next="onStep4Complete" />
    <Step5Interaction v-if="currentStep === 5" :simulationId="simulationId"
                      :reportId="reportId" />

    <!-- Probability dashboard after step 5 -->
    <div v-if="currentStep === 5 && reportId">
      <button @click="loadProbabilities">View Betting Probabilities</button>
      <ProbabilityDashboard v-if="probabilities" :probabilities="probabilities" />
    </div>
  </div>
</template>

<script>
import { getTeams, getPlayers, ingestSportsData, getIngestStatus } from '../api/sports'
import { getProbabilities } from '../api/report'
import Step2EnvSetup from '../components/Step2EnvSetup.vue'
import Step3Simulation from '../components/Step3Simulation.vue'
import Step4Report from '../components/Step4Report.vue'
import Step5Interaction from '../components/Step5Interaction.vue'
import ProbabilityDashboard from '../components/ProbabilityDashboard.vue'

export default {
  name: 'SportsView',
  components: { Step2EnvSetup, Step3Simulation, Step4Report, Step5Interaction, ProbabilityDashboard },
  data() {
    return {
      currentStep: 1,
      sport: 'nba',
      teams: [],
      teamAId: null,
      teamBId: null,
      teamAPlayers: [],
      teamBPlayers: [],
      gameDate: '',
      betTypes: ['moneyline'],
      betTypeOptions: [
        { value: 'moneyline', label: 'Moneyline (Win)' },
        { value: 'spread', label: 'Spread / ATS' },
        { value: 'total', label: 'Over/Under Total' },
        { value: 'player_props', label: 'Player Props' },
        { value: 'futures', label: 'Futures' },
      ],
      propPlayers: [],
      simulationRequirement: '',
      ingestLoading: false,
      projectId: null,
      graphId: null,
      simulationId: null,
      reportId: null,
      probabilities: null,
    }
  },
  async mounted() {
    await this.loadTeams()
  },
  methods: {
    async loadTeams() {
      const res = await getTeams(this.sport)
      this.teams = res.data.teams
    },
    async onSportChange() {
      this.teams = []
      this.teamAId = null
      this.teamBId = null
      await this.loadTeams()
    },
    async onTeamAChange() {
      if (this.teamAId) {
        const res = await getPlayers(this.sport, this.teamAId)
        this.teamAPlayers = res.data.players
      }
    },
    async submitSportsIngest() {
      if (!this.teamAId || !this.teamBId) return alert('Select both teams')
      const teamA = this.teams.find(t => t.id === this.teamAId)
      const teamB = this.teams.find(t => t.id === this.teamBId)

      this.ingestLoading = true
      try {
        const res = await ingestSportsData({
          sport: this.sport,
          league: this.sport.toUpperCase(),
          season: '2024-25',
          team_a_id: this.teamAId,
          team_a_name: teamA.name,
          team_b_id: this.teamBId,
          team_b_name: teamB.name,
          game_date: this.gameDate || null,
          bet_types: this.betTypes,
          player_prop_players: this.propPlayers,
          simulation_requirement: this.simulationRequirement ||
            `Predict the outcome of the ${teamA.name} vs ${teamB.name} game including ${this.betTypes.join(', ')} markets.`,
          project_name: `${teamA.name} vs ${teamB.name}`,
        })

        const { project_id, task_id } = res.data
        this.projectId = project_id

        // Poll until graph is built
        await this.pollTask(task_id)
        this.currentStep = 2

      } catch (e) {
        alert(`Ingestion failed: ${e.message}`)
      } finally {
        this.ingestLoading = false
      }
    },
    async pollTask(taskId) {
      return new Promise((resolve, reject) => {
        const interval = setInterval(async () => {
          try {
            const res = await getIngestStatus(taskId)
            const task = res.data
            if (task.status === 'completed') {
              clearInterval(interval)
              this.graphId = task.result?.graph_id
              resolve()
            } else if (task.status === 'failed') {
              clearInterval(interval)
              reject(new Error(task.error || 'Task failed'))
            }
          } catch (e) {
            clearInterval(interval)
            reject(e)
          }
        }, 3000)
      })
    },
    onStep2Complete({ simulationId }) { this.simulationId = simulationId; this.currentStep = 3 },
    onStep3Complete() { this.currentStep = 4 },
    onStep4Complete({ reportId }) { this.reportId = reportId; this.currentStep = 5 },
    async loadProbabilities() {
      try {
        const res = await getProbabilities(this.reportId)
        this.probabilities = res.data.probabilities
      } catch (e) {
        alert('No probability estimates available for this report.')
      }
    },
  }
}
</script>
```

---

### Frontend — `frontend/src/views/SportsProbabilityView.vue` — new file

Standalone shareable view for a completed prediction.

```vue
<template>
  <div class="sports-prediction-view">
    <h1>Sports Prediction Result</h1>
    <div v-if="loading">Loading...</div>
    <div v-else-if="error" class="error">{{ error }}</div>
    <div v-else>
      <ProbabilityDashboard :probabilities="probabilities" />
      <router-link to="/sports">Run Another Prediction</router-link>
    </div>
  </div>
</template>

<script>
import { getProbabilities } from '../api/report'
import ProbabilityDashboard from '../components/ProbabilityDashboard.vue'

export default {
  name: 'SportsProbabilityView',
  components: { ProbabilityDashboard },
  props: { reportId: String },
  data() { return { loading: true, probabilities: null, error: null } },
  async mounted() {
    try {
      const res = await getProbabilities(this.reportId)
      this.probabilities = res.data.probabilities
    } catch (e) {
      this.error = 'Probabilities not found for this report.'
    } finally {
      this.loading = false
    }
  }
}
</script>
```

---

### Frontend — `frontend/src/router/index.js` — add routes

Add after the existing routes array:

```js
import SportsView from '../views/SportsView.vue'
import SportsProbabilityView from '../views/SportsProbabilityView.vue'

// Inside routes array:
{
  path: '/sports',
  name: 'Sports',
  component: SportsView
},
{
  path: '/sports/prediction/:reportId',
  name: 'SportsProbability',
  component: SportsProbabilityView,
  props: true
},
```

### Frontend — `frontend/src/views/Home.vue` — add button

Add alongside the existing "Start" / "Upload Documents" button:

```html
<router-link to="/sports">
  <button class="btn-sports">Sports Prediction</button>
</router-link>
```

---

## Implementation Order

Work strictly in this order. Each step leaves the system in a buildable, runnable state.

| # | Action | File(s) | Verifiable by |
|---|--------|---------|---------------|
| 1 | Add `market_question`, `is_sports_project`, `sport_config` to Project | `models/project.py` | Load/save existing project JSON — no change in output |
| 2 | Create `SportConfig` dataclass | `models/sport_config.py` | `python -c "from app.models.sport_config import SportConfig; print(SportConfig('nba','NBA','2024-25',2,'Celtics',14,'Heat').to_dict())"` |
| 3 | Add API keys to Config | `config.py`, `.env` | App boots, no error |
| 4 | Create sports ontology templates | `services/sports_ontology_templates.py` | `get_sports_ontology("nba")` returns dict with 10 entity types |
| 5 | Create sports persona library | `services/sports_persona_library.py` | `get_archetypes_for_sport("nba","Celtics","Heat")` returns 6 profiles |
| 6 | Create ProbabilityExtractor | `services/probability_extractor.py` | Pass sample markdown → returns JSON with probabilities summing to 1.0 |
| 7 | Hook ProbabilityExtractor into ReportAgent | `services/report_agent.py` | Existing report generation unaffected; sports project gets `probabilities.json` |
| 8 | Add `/probabilities` route to report API | `api/report.py` | `GET /api/report/fake_id/probabilities` returns 404 (expected) |
| 9 | Add `market_question` reading to graph API | `api/graph.py` | Form with `market_question` sets field on project |
| 10 | Create `sports_data_fetcher.py` | new file | `NBADataFetcher().get_teams()` returns team list (requires API key) |
| 11 | Create `sports_narrative_formatter.py` | new file | Pass mock raw_data dict → returns list of non-empty strings |
| 12 | Add `generate_sports_profiles()` to profile generator | `services/oasis_profile_generator.py` | Returns 6 OasisAgentProfile objects |
| 13 | Add `sports_bp` to `api/__init__.py` | `api/__init__.py` | File importable without error |
| 14 | Register `sports_bp` in `app/__init__.py` | `app/__init__.py` | `GET /api/sports/teams?sport=nba` returns team list |
| 15 | Create `api/sports.py` | new file | All 5 endpoints respond correctly |
| 16 | End-to-end backend test | — | See verification checklist |
| 17 | Create `frontend/src/api/sports.js` | new file | Functions importable |
| 18 | Create `ProbabilityDashboard.vue` | new component | Renders with mock probabilities prop |
| 19 | Create `KalshiProbabilityBadge.vue` | new component | Renders with mock props |
| 20 | Add Kalshi field to `MainView.vue` + `Step4Report.vue` | existing files | Field appears in Step 1 UI |
| 21 | Create `SportsView.vue` | new view | `/sports` route loads, team dropdown populates |
| 22 | Create `SportsProbabilityView.vue` | new view | `/sports/prediction/test_id` loads without error |
| 23 | Update `router/index.js` | existing file | Both routes work |
| 24 | Add button to `Home.vue` | existing file | Button navigates to `/sports` |

---

## Verification Checklist

### Kalshi path
- [ ] `POST /api/graph/ontology/generate` with `market_question` field → project JSON contains `market_question`
- [ ] After report generation: `uploads/reports/{id}/probabilities.json` exists
- [ ] `GET /api/report/{id}/probabilities` returns `{yes_probability, no_probability, confidence, key_factors}`
- [ ] `yes_probability + no_probability = 1.0` (±0.001)
- [ ] Existing projects without `market_question` → no `probabilities.json` created (regression-free)

### Sports path
- [ ] `GET /api/sports/teams?sport=nba` returns list of 30 teams
- [ ] `GET /api/sports/players?sport=nba&team_id=2` returns player list
- [ ] `get_sports_ontology("nba")` has exactly 10 entity types; last two are `Person`, `Organization`
- [ ] `POST /api/sports/ingest` returns `{project_id, task_id}`
- [ ] Polling `/api/sports/ingest/status/{task_id}` reaches `progress: 100, status: completed`
- [ ] `GET /api/graph/project/{project_id}` shows `status: graph_completed`, `is_sports_project: true`
- [ ] `GET /api/graph/data/{graph_id}` returns `node_count > 0`; node types include `NBATeam` or `NBAPlayer`
- [ ] Simulation prepare generates profiles; `twitter_profiles.csv` has a row with `profession: Sports Betting Analyst`
- [ ] After report generation: `probabilities.json` exists in report dir
- [ ] `moneyline.team_a_probability + moneyline.team_b_probability = 1.0`
- [ ] `spread.cover_probability` is between 0.0 and 1.0

### Regression (existing flow)
- [ ] Document upload → ontology → graph build → simulate → report still works end-to-end
- [ ] Existing project JSON has `sport_config: null`, `is_sports_project: false`, `market_question: null`
- [ ] Existing report API endpoints return unchanged response shapes
- [ ] No new required fields on any existing request body
