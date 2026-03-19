import service, { requestWithRetry } from './index'

/**
 * Get team list for a sport
 * @param {String} sport - "nba" or "soccer"
 * @param {String} league - optional league name filter
 */
export function getTeams(sport = 'nba', league = '') {
  return service({
    url: '/api/sports/teams',
    method: 'get',
    params: { sport, league }
  })
}

/**
 * Get players for a team
 * @param {String} sport
 * @param {Number} teamId
 */
export function getPlayers(sport, teamId) {
  return service({
    url: '/api/sports/players',
    method: 'get',
    params: { sport, team_id: teamId }
  })
}

/**
 * Get current odds
 * @param {String} sport
 * @param {String} markets - comma-separated e.g. "h2h,spreads,totals"
 */
export function getOdds(sport = 'nba', markets = 'h2h,spreads,totals') {
  return service({
    url: '/api/sports/odds',
    method: 'get',
    params: { sport, markets }
  })
}

/**
 * Start a sports data ingestion task (async)
 * @param {Object} data - sport config + simulation_requirement
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
 * Poll ingest task status
 * @param {String} taskId
 */
export function getIngestStatus(taskId) {
  return service({
    url: `/api/sports/ingest/status/${taskId}`,
    method: 'get'
  })
}

/**
 * Get sports project config
 * @param {String} projectId
 */
export function getSportsProjectConfig(projectId) {
  return service({
    url: `/api/sports/project/${projectId}/config`,
    method: 'get'
  })
}

/**
 * Get probability extraction results for a report
 * @param {String} reportId
 */
export function getProbabilities(reportId) {
  return service({
    url: `/api/report/${reportId}/probabilities`,
    method: 'get'
  })
}
