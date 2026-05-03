<template>
  <div class="sports-view">
    <nav class="navbar">
      <div class="nav-brand" @click="$router.push('/')">MIROFISH</div>
      <span class="nav-sub">Sports Prediction</span>
    </nav>

    <div class="content">
      <!-- Step: Form -->
      <div v-if="phase === 'form'" class="form-panel">
        <div class="form-header">
          <div class="step-tag">01 / Sports Setup</div>
          <h2 class="form-title">Configure Matchup</h2>
          <p class="form-desc">Select the teams, markets, and players you want to analyse.</p>
        </div>

        <!-- Sport selector -->
        <div class="field-group">
          <label class="field-label">Sport</label>
          <div class="toggle-row">
            <button
              v-for="s in sports"
              :key="s.value"
              class="toggle-btn"
              :class="{ active: form.sport === s.value }"
              @click="form.sport = s.value; form.teamAId = null; form.teamBId = null; loadTeams()"
            >{{ s.label }}</button>
          </div>
        </div>

        <!-- League / Season -->
        <div class="field-row">
          <div class="field-group">
            <label class="field-label">League</label>
            <input v-model="form.league" class="text-input" placeholder="e.g. NBA" />
          </div>
          <div class="field-group">
            <label class="field-label">Season</label>
            <input v-model="form.season" class="text-input" placeholder="e.g. 2024-25" />
          </div>
        </div>

        <!-- Team pickers -->
        <div class="field-row">
          <div class="field-group">
            <label class="field-label">Team A (Home)</label>
            <div v-if="teamsLoading" class="loading-hint">Loading teams...</div>
            <select v-else v-model="form.teamAId" class="select-input" @change="onTeamAChange">
              <option :value="null" disabled>Select a team</option>
              <option v-for="t in teams" :key="t.id" :value="t.id">{{ t.name }}</option>
            </select>
          </div>
          <div class="field-group">
            <label class="field-label">Team B (Away)</label>
            <select v-model="form.teamBId" class="select-input" @change="onTeamBChange">
              <option :value="null" disabled>Select a team</option>
              <option v-for="t in teams" :key="t.id" :value="t.id">{{ t.name }}</option>
            </select>
          </div>
        </div>

        <!-- Game Date -->
        <div class="field-group">
          <label class="field-label">Game Date (optional)</label>
          <input v-model="form.gameDate" type="date" class="text-input" />
        </div>

        <!-- Bet Types -->
        <div class="field-group">
          <label class="field-label">Bet Markets</label>
          <div class="checkbox-row">
            <label v-for="bt in betTypeOptions" :key="bt.value" class="checkbox-label">
              <input type="checkbox" :value="bt.value" v-model="form.betTypes" class="checkbox-input" />
              {{ bt.label }}
            </label>
          </div>
        </div>

        <!-- Player Props -->
        <div class="field-group">
          <label class="field-label">Player Props — Enter player names (one per line)</label>
          <textarea
            v-model="propPlayersText"
            class="text-input textarea"
            placeholder="e.g.&#10;Jayson Tatum&#10;Stephen Curry"
            rows="3"
          ></textarea>
        </div>

        <!-- Simulation Requirement -->
        <div class="field-group">
          <label class="field-label">Simulation Focus (optional)</label>
          <textarea
            v-model="form.simulationRequirement"
            class="text-input textarea"
            placeholder="Describe what you want the AI to focus on, e.g. 'Focus on injury impact and line value for moneyline and first-half spread.'"
            rows="3"
          ></textarea>
        </div>

        <!-- Error -->
        <div v-if="formError" class="form-error">{{ formError }}</div>

        <!-- Submit -->
        <button class="submit-btn" :disabled="!canSubmit" @click="startIngest">
          <span>Analyse Matchup</span>
          <span class="btn-arrow">→</span>
        </button>
      </div>

      <!-- Step: Ingesting -->
      <div v-else-if="phase === 'ingesting'" class="progress-panel">
        <div class="progress-header">
          <div class="step-tag">01 / Ingesting Data</div>
          <h2 class="progress-title">{{ teamAName }} vs {{ teamBName }}</h2>
        </div>

        <div class="progress-bar-wrap">
          <div class="progress-bar" :style="{ width: progress + '%' }"></div>
        </div>
        <div class="progress-pct">{{ progress }}%</div>
        <div class="progress-msg">{{ progressMsg }}</div>

        <div v-if="apiErrors.length" class="api-warnings">
          <div class="warnings-title">Data availability notes:</div>
          <div v-for="(e, i) in apiErrors" :key="i" class="warning-item">⚠ {{ e }}</div>
        </div>
      </div>

      <!-- Step: Done -->
      <div v-else-if="phase === 'done'" class="done-panel">
        <div class="done-icon">✓</div>
        <h2 class="done-title">Data Ingested</h2>
        <p class="done-desc">
          {{ teamAName }} vs {{ teamBName }} — {{ nodeCount }} entities extracted.
          Continue to run the agent simulation.
        </p>
        <div v-if="apiErrors.length" class="api-warnings">
          <div class="warnings-title">Some data sources were unavailable:</div>
          <div v-for="(e, i) in apiErrors" :key="i" class="warning-item">⚠ {{ e }}</div>
        </div>
        <button class="submit-btn" @click="continueToProcess">Continue to Simulation →</button>
      </div>

      <!-- Step: Error -->
      <div v-else-if="phase === 'error'" class="error-panel">
        <div class="error-icon">✗</div>
        <h2 class="error-title">Ingest Failed</h2>
        <p class="error-desc">{{ errorMsg }}</p>
        <button class="submit-btn" @click="phase = 'form'">Try Again</button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { useRouter } from 'vue-router'
import { getTeams, ingestSportsData, getIngestStatus } from '../api/sports'

const router = useRouter()

// Form state
const form = ref({
  sport: 'nba',
  league: 'NBA',
  season: '2024-25',
  teamAId: null,
  teamAName: '',
  teamBId: null,
  teamBName: '',
  gameDate: '',
  betTypes: ['moneyline', 'spread', 'total'],
  simulationRequirement: ''
})

const propPlayersText = ref('')
const sports = [
  { value: 'nba',          label: 'NBA' },
  { value: 'ncaab',        label: 'NCAA Basketball' },
  { value: 'nfl',          label: 'NFL' },
  { value: 'ncaaf',        label: 'NCAA Football' },
  { value: 'mlb',          label: 'MLB' },
  { value: 'nhl',          label: 'NHL' },
  { value: 'epl',          label: 'Premier League' },
  { value: 'la_liga',      label: 'La Liga' },
  { value: 'bundesliga',   label: 'Bundesliga' },
  { value: 'serie_a',      label: 'Serie A' },
  { value: 'ligue_1',      label: 'Ligue 1' },
  { value: 'mls',          label: 'MLS' },
  { value: 'ucl',          label: 'Champions League' },
  { value: 'tennis_atp',   label: 'ATP Tennis' },
  { value: 'tennis_wta',   label: 'WTA Tennis' },
  { value: 'golf_pga',     label: 'PGA Tour' },
  { value: 'mma',          label: 'UFC / MMA' },
  { value: 'boxing',       label: 'Boxing' },
  { value: 'nrl',          label: 'NRL Rugby' },
  { value: 'afl',          label: 'AFL' },
  { value: 'ipl',          label: 'IPL Cricket' },
]
const betTypeOptions = [
  { value: 'moneyline', label: 'Moneyline' },
  { value: 'spread', label: 'Spread' },
  { value: 'total', label: 'Over/Under' },
  { value: 'props', label: 'Player Props' }
]

const teams = ref([])
const teamsLoading = ref(false)
const formError = ref('')

// Ingest state
const phase = ref('form')
const progress = ref(0)
const progressMsg = ref('')
const apiErrors = ref([])
const nodeCount = ref(0)
const teamAName = ref('')
const teamBName = ref('')
const projectId = ref('')
const errorMsg = ref('')

let pollInterval = null

const canSubmit = computed(() =>
  form.value.teamAId && form.value.teamBId && form.value.teamAId !== form.value.teamBId
)

const loadTeams = async () => {
  teamsLoading.value = true
  formError.value = ''
  try {
    const res = await getTeams(form.value.sport, form.value.league)
    teams.value = res.data?.teams || []
  } catch (e) {
    formError.value = `Failed to load teams: ${e.message}`
  } finally {
    teamsLoading.value = false
  }
}

const onTeamAChange = () => {
  const t = teams.value.find(t => t.id === form.value.teamAId)
  form.value.teamAName = t?.name || ''
}

const onTeamBChange = () => {
  const t = teams.value.find(t => t.id === form.value.teamBId)
  form.value.teamBName = t?.name || ''
}

const startIngest = async () => {
  formError.value = ''
  if (!canSubmit.value) return

  const playerProps = propPlayersText.value
    .split('\n')
    .map(s => s.trim())
    .filter(Boolean)

  const payload = {
    sport: form.value.sport,
    league: form.value.league,
    season: form.value.season,
    team_a_id: form.value.teamAId,
    team_a_name: form.value.teamAName,
    team_b_id: form.value.teamBId,
    team_b_name: form.value.teamBName,
    game_date: form.value.gameDate || undefined,
    bet_types: form.value.betTypes,
    player_prop_players: playerProps,
    simulation_requirement: form.value.simulationRequirement || undefined,
  }

  teamAName.value = form.value.teamAName
  teamBName.value = form.value.teamBName
  phase.value = 'ingesting'
  progress.value = 0
  progressMsg.value = 'Starting...'

  try {
    const res = await ingestSportsData(payload)
    if (!res.success) {
      throw new Error(res.error || 'Ingest request failed')
    }
    projectId.value = res.data.project_id
    const taskId = res.data.task_id
    pollInterval = setInterval(() => pollStatus(taskId), 3000)
  } catch (e) {
    phase.value = 'error'
    errorMsg.value = e.message
  }
}

const pollStatus = async (taskId) => {
  try {
    const res = await getIngestStatus(taskId)
    if (!res.success) return

    const task = res.data
    progress.value = task.progress || 0
    progressMsg.value = task.message || ''

    if (task.status === 'completed') {
      clearInterval(pollInterval)
      nodeCount.value = task.result?.node_count || 0
      apiErrors.value = task.result?.api_errors || []
      phase.value = 'done'
    } else if (task.status === 'failed') {
      clearInterval(pollInterval)
      phase.value = 'error'
      errorMsg.value = task.message || 'Task failed'
    }
  } catch (e) {
    // non-fatal polling error
    console.warn('Poll error:', e.message)
  }
}

const continueToProcess = () => {
  router.push({ name: 'Process', params: { projectId: projectId.value } })
}

// Load teams on mount
loadTeams()
</script>

<style scoped>
.sports-view {
  min-height: 100vh;
  background: #fff;
  font-family: 'JetBrains Mono', monospace;
}

.navbar {
  height: 60px;
  background: #000;
  color: #fff;
  display: flex;
  align-items: center;
  padding: 0 40px;
  gap: 20px;
}

.nav-brand {
  font-weight: 800;
  letter-spacing: 1px;
  cursor: pointer;
}

.nav-sub {
  font-size: 0.75rem;
  color: #999;
}

.content {
  max-width: 760px;
  margin: 0 auto;
  padding: 60px 40px;
}

/* Form */
.form-header {
  margin-bottom: 40px;
}

.step-tag {
  font-size: 0.65rem;
  font-weight: 700;
  letter-spacing: 2px;
  color: #999;
  margin-bottom: 12px;
}

.form-title {
  font-size: 2rem;
  font-weight: 700;
  margin: 0 0 8px;
  letter-spacing: -1px;
}

.form-desc {
  font-size: 0.85rem;
  color: #666;
  margin: 0;
}

.field-group {
  margin-bottom: 24px;
}

.field-row {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 20px;
  margin-bottom: 24px;
}

.field-label {
  display: block;
  font-size: 0.65rem;
  font-weight: 700;
  letter-spacing: 1px;
  color: #666;
  margin-bottom: 8px;
}

.text-input, .select-input {
  width: 100%;
  border: 1px solid #e5e5e5;
  background: #fafafa;
  padding: 12px 16px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.85rem;
  color: #000;
  outline: none;
  box-sizing: border-box;
  transition: border-color 0.2s;
}

.text-input:focus, .select-input:focus {
  border-color: #000;
  background: #fff;
}

.textarea {
  resize: vertical;
  min-height: 80px;
}

.toggle-row {
  display: flex;
  gap: 8px;
}

.toggle-btn {
  padding: 10px 20px;
  border: 1px solid #e5e5e5;
  background: transparent;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.8rem;
  font-weight: 700;
  cursor: pointer;
  transition: all 0.2s;
}

.toggle-btn.active {
  background: #000;
  color: #fff;
  border-color: #000;
}

.checkbox-row {
  display: flex;
  gap: 16px;
  flex-wrap: wrap;
}

.checkbox-label {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 0.8rem;
  cursor: pointer;
}

.checkbox-input {
  width: 14px;
  height: 14px;
  accent-color: #000;
}

.form-error {
  color: #ef4444;
  font-size: 0.8rem;
  margin-bottom: 16px;
  padding: 10px 14px;
  border: 1px solid #ef4444;
  background: #fff5f5;
}

.loading-hint {
  font-size: 0.8rem;
  color: #999;
  padding: 12px 0;
}

.submit-btn {
  width: 100%;
  background: #000;
  color: #fff;
  border: none;
  padding: 20px;
  font-family: 'JetBrains Mono', monospace;
  font-weight: 700;
  font-size: 1rem;
  display: flex;
  justify-content: space-between;
  align-items: center;
  cursor: pointer;
  letter-spacing: 1px;
  transition: background 0.2s;
}

.submit-btn:hover:not(:disabled) {
  background: #FF4500;
}

.submit-btn:disabled {
  background: #e5e5e5;
  color: #999;
  cursor: not-allowed;
}

.btn-arrow { font-size: 1.2rem; }

/* Progress */
.progress-panel, .done-panel, .error-panel {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 20px;
  padding: 80px 0;
  text-align: center;
}

.progress-header {
  text-align: center;
}

.progress-title, .done-title, .error-title {
  font-size: 1.8rem;
  font-weight: 700;
  margin: 0;
  letter-spacing: -1px;
}

.progress-bar-wrap {
  width: 100%;
  max-width: 500px;
  height: 6px;
  background: #f0f0f0;
}

.progress-bar {
  height: 100%;
  background: #000;
  transition: width 0.5s ease;
}

.progress-pct {
  font-size: 2rem;
  font-weight: 700;
}

.progress-msg {
  font-size: 0.8rem;
  color: #666;
}

.done-icon {
  font-size: 3rem;
  color: #22c55e;
}

.done-desc, .error-desc {
  font-size: 0.9rem;
  color: #666;
  max-width: 500px;
}

.error-icon {
  font-size: 3rem;
  color: #ef4444;
}

.api-warnings {
  border: 1px solid #f59e0b;
  padding: 16px;
  text-align: left;
  width: 100%;
  max-width: 500px;
  background: #fffbeb;
}

.warnings-title {
  font-size: 0.7rem;
  font-weight: 700;
  color: #92400e;
  margin-bottom: 8px;
  letter-spacing: 1px;
}

.warning-item {
  font-size: 0.8rem;
  color: #78350f;
  margin-bottom: 4px;
}
</style>
