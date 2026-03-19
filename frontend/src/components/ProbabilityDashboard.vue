<template>
  <div class="probability-dashboard">
    <!-- Header -->
    <div class="dashboard-header">
      <span class="header-label">PREDICTION PROBABILITIES</span>
      <span class="confidence-badge" :class="confidenceClass">
        {{ confidenceLabel }}
      </span>
    </div>

    <!-- Moneyline -->
    <div v-if="probabilities.moneyline" class="market-card">
      <div class="market-title">MONEYLINE</div>
      <div class="ml-row">
        <div class="ml-team">
          <div class="team-name">{{ probabilities.moneyline.team_a }}</div>
          <div class="prob-bar-wrap">
            <div class="prob-bar team-a-bar" :style="{ width: pct(probabilities.moneyline.team_a_probability) }"></div>
          </div>
          <div class="prob-value">{{ pct(probabilities.moneyline.team_a_probability) }}</div>
        </div>
        <div class="ml-vs">VS</div>
        <div class="ml-team right">
          <div class="team-name">{{ teamBName }}</div>
          <div class="prob-bar-wrap">
            <div class="prob-bar team-b-bar" :style="{ width: pct(probabilities.moneyline.team_b_probability) }"></div>
          </div>
          <div class="prob-value">{{ pct(probabilities.moneyline.team_b_probability) }}</div>
        </div>
      </div>
    </div>

    <!-- Spread -->
    <div v-if="probabilities.spread" class="market-card">
      <div class="market-title">SPREAD</div>
      <div class="spread-row">
        <div class="spread-info">
          <span class="favorite-name">{{ probabilities.spread.favorite }}</span>
          <span class="spread-line">{{ spreadDisplay }}</span>
        </div>
        <div class="cover-prob">
          <div class="cover-bar-wrap">
            <div class="cover-bar" :style="{ width: pct(probabilities.spread.cover_probability) }"></div>
          </div>
          <div class="cover-label">Cover probability: {{ pct(probabilities.spread.cover_probability) }}</div>
        </div>
      </div>
    </div>

    <!-- Total -->
    <div v-if="probabilities.total" class="market-card">
      <div class="market-title">OVER / UNDER</div>
      <div class="total-row">
        <div class="total-info">Line: <span class="total-line">{{ probabilities.total.line }}</span></div>
        <div class="total-bars">
          <div class="ou-item">
            <span class="ou-label">OVER</span>
            <div class="ou-bar-wrap">
              <div class="ou-bar over-bar" :style="{ width: pct(probabilities.total.over_probability) }"></div>
            </div>
            <span class="ou-prob">{{ pct(probabilities.total.over_probability) }}</span>
          </div>
          <div class="ou-item">
            <span class="ou-label">UNDER</span>
            <div class="ou-bar-wrap">
              <div class="ou-bar under-bar" :style="{ width: pct(underProbability) }"></div>
            </div>
            <span class="ou-prob">{{ pct(underProbability) }}</span>
          </div>
        </div>
      </div>
    </div>

    <!-- Player Props -->
    <div v-if="probabilities.player_props && probabilities.player_props.length" class="market-card">
      <div class="market-title">PLAYER PROPS</div>
      <table class="props-table">
        <thead>
          <tr>
            <th>Player</th>
            <th>Market</th>
            <th>Line</th>
            <th>Over %</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="(prop, i) in probabilities.player_props" :key="i">
            <td>{{ prop.player }}</td>
            <td class="market-cell">{{ prop.market }}</td>
            <td>{{ prop.line }}</td>
            <td>
              <span class="prop-prob" :class="{ 'prob-high': prop.over_probability >= 0.55 }">
                {{ pct(prop.over_probability) }}
              </span>
            </td>
          </tr>
        </tbody>
      </table>
    </div>

    <!-- Reasoning Summary -->
    <div v-if="probabilities.reasoning_summary" class="reasoning-card">
      <div class="reasoning-label">ANALYST REASONING</div>
      <p class="reasoning-text">{{ probabilities.reasoning_summary }}</p>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  probabilities: {
    type: Object,
    required: true
  }
})

const pct = (v) => v != null ? `${Math.round(v * 100)}%` : '—'

const confidenceClass = computed(() => ({
  'conf-high': props.probabilities.confidence === 'high',
  'conf-medium': props.probabilities.confidence === 'medium',
  'conf-low': props.probabilities.confidence === 'low',
}))

const confidenceLabel = computed(() => {
  const map = { high: 'HIGH CONFIDENCE', medium: 'MEDIUM CONFIDENCE', low: 'LOW CONFIDENCE' }
  return map[props.probabilities.confidence] || 'CONFIDENCE UNKNOWN'
})

const teamBName = computed(() => {
  const ml = props.probabilities.moneyline
  if (!ml) return 'Team B'
  // team_b name isn't stored explicitly — derive from reasoning or fall back
  return 'Opponent'
})

const spreadDisplay = computed(() => {
  const s = props.probabilities.spread
  if (!s) return ''
  const sign = s.line > 0 ? '+' : ''
  return `${sign}${s.line}`
})

const underProbability = computed(() => {
  const total = props.probabilities.total
  if (!total) return 0.5
  return Math.max(0, 1 - (total.over_probability || 0.5))
})
</script>

<style scoped>
.probability-dashboard {
  display: flex;
  flex-direction: column;
  gap: 16px;
  font-family: 'JetBrains Mono', monospace;
}

.dashboard-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 16px;
  background: #000;
  color: #fff;
}

.header-label {
  font-size: 0.75rem;
  font-weight: 700;
  letter-spacing: 2px;
}

.confidence-badge {
  font-size: 0.65rem;
  font-weight: 700;
  padding: 4px 10px;
  letter-spacing: 1px;
}

.conf-high { background: #22c55e; color: #000; }
.conf-medium { background: #f59e0b; color: #000; }
.conf-low { background: #ef4444; color: #fff; }

.market-card {
  border: 1px solid #e5e5e5;
  padding: 20px;
}

.market-title {
  font-size: 0.65rem;
  font-weight: 700;
  letter-spacing: 2px;
  color: #999;
  margin-bottom: 16px;
}

/* Moneyline */
.ml-row {
  display: flex;
  align-items: center;
  gap: 20px;
}

.ml-team {
  flex: 1;
}

.ml-team.right {
  text-align: right;
}

.team-name {
  font-size: 0.9rem;
  font-weight: 700;
  margin-bottom: 8px;
}

.prob-bar-wrap {
  height: 6px;
  background: #f0f0f0;
  width: 100%;
  margin-bottom: 6px;
}

.prob-bar {
  height: 100%;
  transition: width 0.6s ease;
}

.team-a-bar { background: #000; }
.team-b-bar { background: #FF4500; }

.prob-value {
  font-size: 1.4rem;
  font-weight: 700;
}

.ml-vs {
  font-size: 0.7rem;
  color: #999;
  font-weight: 700;
}

/* Spread */
.spread-row {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.spread-info {
  display: flex;
  align-items: baseline;
  gap: 12px;
}

.favorite-name {
  font-weight: 700;
}

.spread-line {
  font-size: 1.2rem;
  font-weight: 700;
  color: #FF4500;
}

.cover-bar-wrap {
  height: 6px;
  background: #f0f0f0;
  margin-bottom: 6px;
}

.cover-bar {
  height: 100%;
  background: #000;
  transition: width 0.6s ease;
}

.cover-label {
  font-size: 0.8rem;
  color: #666;
}

/* Total */
.total-row {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.total-info {
  font-size: 0.85rem;
  color: #666;
}

.total-line {
  font-weight: 700;
  color: #000;
}

.total-bars {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.ou-item {
  display: flex;
  align-items: center;
  gap: 12px;
}

.ou-label {
  font-size: 0.7rem;
  font-weight: 700;
  letter-spacing: 1px;
  width: 50px;
  color: #666;
}

.ou-bar-wrap {
  flex: 1;
  height: 6px;
  background: #f0f0f0;
}

.ou-bar {
  height: 100%;
  transition: width 0.6s ease;
}

.over-bar { background: #22c55e; }
.under-bar { background: #ef4444; }

.ou-prob {
  font-size: 0.85rem;
  font-weight: 700;
  width: 40px;
  text-align: right;
}

/* Player Props */
.props-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.85rem;
}

.props-table th {
  text-align: left;
  font-size: 0.65rem;
  letter-spacing: 1px;
  color: #999;
  padding: 0 0 10px 0;
  border-bottom: 1px solid #e5e5e5;
}

.props-table td {
  padding: 10px 0;
  border-bottom: 1px solid #f5f5f5;
}

.market-cell {
  text-transform: capitalize;
  color: #666;
}

.prop-prob {
  font-weight: 700;
}

.prob-high {
  color: #22c55e;
}

/* Reasoning */
.reasoning-card {
  border: 1px solid #e5e5e5;
  padding: 20px;
  background: #fafafa;
}

.reasoning-label {
  font-size: 0.65rem;
  font-weight: 700;
  letter-spacing: 2px;
  color: #999;
  margin-bottom: 10px;
}

.reasoning-text {
  font-size: 0.85rem;
  line-height: 1.7;
  color: #444;
  margin: 0;
}
</style>
