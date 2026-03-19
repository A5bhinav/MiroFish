<template>
  <div class="sports-prob-view">
    <nav class="navbar">
      <div class="nav-brand" @click="$router.push('/')">MIROFISH</div>
      <span class="nav-sub">Sports Prediction</span>
    </nav>

    <div class="content">
      <div v-if="loading" class="loading-state">
        <div class="spinner"></div>
        <p>Loading prediction results...</p>
      </div>

      <div v-else-if="error" class="error-state">
        <div class="error-icon">✗</div>
        <p>{{ error }}</p>
        <button class="back-btn" @click="$router.push('/')">Back to Home</button>
      </div>

      <div v-else>
        <div class="page-header">
          <div class="report-meta">Report: {{ reportId }}</div>
          <h1 class="page-title">Prediction Dashboard</h1>
        </div>

        <ProbabilityDashboard :probabilities="probabilities" />

        <div class="actions">
          <button class="action-btn" @click="$router.push('/')">New Prediction</button>
          <button class="action-btn secondary" @click="$router.push(`/report/${reportId}`)">View Full Report</button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { useRoute } from 'vue-router'
import ProbabilityDashboard from '../components/ProbabilityDashboard.vue'
import { getProbabilities } from '../api/sports'

const route = useRoute()
const reportId = route.params.reportId

const probabilities = ref(null)
const loading = ref(true)
const error = ref('')

onMounted(async () => {
  try {
    const res = await getProbabilities(reportId)
    probabilities.value = res.data
  } catch (e) {
    error.value = e.message || 'Failed to load probabilities'
  } finally {
    loading.value = false
  }
})
</script>

<style scoped>
.sports-prob-view {
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
  max-width: 900px;
  margin: 0 auto;
  padding: 60px 40px;
}

.loading-state, .error-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 16px;
  padding: 80px 0;
  color: #666;
}

.spinner {
  width: 36px;
  height: 36px;
  border: 3px solid #e5e5e5;
  border-top-color: #000;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

@keyframes spin { to { transform: rotate(360deg); } }

.error-icon {
  font-size: 2rem;
  color: #ef4444;
}

.page-header {
  margin-bottom: 32px;
}

.report-meta {
  font-size: 0.7rem;
  color: #999;
  letter-spacing: 1px;
  margin-bottom: 8px;
}

.page-title {
  font-size: 2rem;
  font-weight: 700;
  margin: 0;
  letter-spacing: -1px;
}

.actions {
  display: flex;
  gap: 12px;
  margin-top: 32px;
}

.action-btn {
  padding: 14px 28px;
  font-family: 'JetBrains Mono', monospace;
  font-weight: 700;
  font-size: 0.85rem;
  letter-spacing: 1px;
  cursor: pointer;
  border: none;
  background: #000;
  color: #fff;
  transition: background 0.2s;
}

.action-btn:hover {
  background: #FF4500;
}

.action-btn.secondary {
  background: transparent;
  color: #000;
  border: 1px solid #000;
}

.action-btn.secondary:hover {
  background: #f5f5f5;
}

.back-btn {
  padding: 10px 24px;
  font-family: 'JetBrains Mono', monospace;
  font-weight: 700;
  font-size: 0.8rem;
  cursor: pointer;
  background: #000;
  color: #fff;
  border: none;
}
</style>
