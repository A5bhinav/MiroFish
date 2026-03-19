<template>
  <div class="kalshi-badge">
    <div class="badge-header">
      <span class="badge-label">KALSHI PREDICTION</span>
      <span class="confidence-tag" :class="confidenceClass">{{ confidenceLabel }}</span>
    </div>

    <div v-if="marketQuestion" class="market-question">
      {{ marketQuestion }}
    </div>

    <div class="circles-row">
      <div class="circle-wrap">
        <svg class="circle-svg" viewBox="0 0 80 80">
          <circle cx="40" cy="40" r="34" fill="none" stroke="#f0f0f0" stroke-width="8" />
          <circle
            cx="40" cy="40" r="34"
            fill="none"
            stroke="#22c55e"
            stroke-width="8"
            stroke-dasharray="213.6"
            :stroke-dashoffset="213.6 * (1 - yesProbability)"
            stroke-linecap="round"
            transform="rotate(-90 40 40)"
          />
        </svg>
        <div class="circle-label">
          <div class="circle-pct">{{ pct(yesProbability) }}</div>
          <div class="circle-name">YES</div>
        </div>
      </div>

      <div class="circle-divider">VS</div>

      <div class="circle-wrap">
        <svg class="circle-svg" viewBox="0 0 80 80">
          <circle cx="40" cy="40" r="34" fill="none" stroke="#f0f0f0" stroke-width="8" />
          <circle
            cx="40" cy="40" r="34"
            fill="none"
            stroke="#ef4444"
            stroke-width="8"
            stroke-dasharray="213.6"
            :stroke-dashoffset="213.6 * (1 - noProbability)"
            stroke-linecap="round"
            transform="rotate(-90 40 40)"
          />
        </svg>
        <div class="circle-label">
          <div class="circle-pct no-pct">{{ pct(noProbability) }}</div>
          <div class="circle-name">NO</div>
        </div>
      </div>
    </div>

    <div v-if="probabilities.key_factors && probabilities.key_factors.length" class="key-factors">
      <div class="factors-label">KEY FACTORS</div>
      <ul class="factors-list">
        <li v-for="(f, i) in probabilities.key_factors" :key="i">{{ f }}</li>
      </ul>
    </div>

    <div v-if="probabilities.reasoning_summary" class="summary">
      {{ probabilities.reasoning_summary }}
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  probabilities: { type: Object, required: true },
  marketQuestion: { type: String, default: '' }
})

const pct = (v) => v != null ? `${Math.round(v * 100)}%` : '—'

const yesProbability = computed(() => props.probabilities.yes_probability ?? 0.5)
const noProbability = computed(() => props.probabilities.no_probability ?? 0.5)

const confidenceClass = computed(() => ({
  'conf-high': props.probabilities.confidence === 'high',
  'conf-medium': props.probabilities.confidence === 'medium',
  'conf-low': props.probabilities.confidence === 'low',
}))

const confidenceLabel = computed(() => {
  const map = { high: 'HIGH', medium: 'MEDIUM', low: 'LOW' }
  return map[props.probabilities.confidence] || '—'
})
</script>

<style scoped>
.kalshi-badge {
  border: 1px solid #e5e5e5;
  padding: 24px;
  font-family: 'JetBrains Mono', monospace;
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.badge-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.badge-label {
  font-size: 0.65rem;
  font-weight: 700;
  letter-spacing: 2px;
  color: #999;
}

.confidence-tag {
  font-size: 0.65rem;
  font-weight: 700;
  padding: 3px 8px;
  letter-spacing: 1px;
}

.conf-high { background: #22c55e; color: #000; }
.conf-medium { background: #f59e0b; color: #000; }
.conf-low { background: #ef4444; color: #fff; }

.market-question {
  font-size: 0.95rem;
  font-weight: 600;
  line-height: 1.5;
  color: #000;
  border-left: 3px solid #FF4500;
  padding-left: 12px;
}

.circles-row {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 24px;
}

.circle-wrap {
  position: relative;
  width: 80px;
  height: 80px;
}

.circle-svg {
  width: 80px;
  height: 80px;
}

.circle-label {
  position: absolute;
  inset: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
}

.circle-pct {
  font-size: 1.1rem;
  font-weight: 700;
  color: #22c55e;
}

.no-pct {
  color: #ef4444;
}

.circle-name {
  font-size: 0.6rem;
  font-weight: 700;
  letter-spacing: 1px;
  color: #999;
}

.circle-divider {
  font-size: 0.7rem;
  font-weight: 700;
  color: #ccc;
}

.key-factors {
  border-top: 1px solid #f0f0f0;
  padding-top: 12px;
}

.factors-label {
  font-size: 0.6rem;
  font-weight: 700;
  letter-spacing: 2px;
  color: #999;
  margin-bottom: 8px;
}

.factors-list {
  list-style: none;
  padding: 0;
  margin: 0;
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.factors-list li {
  font-size: 0.8rem;
  color: #444;
  padding-left: 12px;
  position: relative;
}

.factors-list li::before {
  content: '›';
  position: absolute;
  left: 0;
  color: #FF4500;
  font-weight: 700;
}

.summary {
  font-size: 0.8rem;
  line-height: 1.6;
  color: #666;
  border-top: 1px solid #f0f0f0;
  padding-top: 12px;
}
</style>
