<template>
  <div class="kalshi-view">
    <nav class="navbar">
      <div class="nav-brand" @click="$router.push('/')">MIROFISH</div>
      <span class="nav-sub">Kalshi Markets</span>
      <div class="nav-status" :class="statusClass">
        <span class="status-dot"></span>
        {{ statusLabel }}
      </div>
    </nav>

    <div class="layout">
      <!-- LEFT: Market Browser -->
      <aside class="sidebar">
        <div class="sidebar-header">
          <div class="section-tag">MARKETS</div>
          <div class="search-row">
            <input
              v-model="searchQuery"
              class="search-input"
              placeholder="Search Kalshi markets..."
              @keyup.enter="doSearch"
            />
            <button class="icon-btn" @click="doSearch">→</button>
          </div>
          <div class="tab-row">
            <button class="tab-btn" :class="{ active: tab === 'catalog' }" @click="switchTab('catalog')">Catalog</button>
            <button class="tab-btn" :class="{ active: tab === 'scan' }" @click="switchTab('scan')">Opportunities</button>
            <button class="tab-btn" :class="{ active: tab === 'portfolio' }" @click="switchTab('portfolio')">Portfolio</button>
          </div>
        </div>

        <!-- Catalog tab -->
        <div v-if="tab === 'catalog'" class="market-list">
          <div v-if="marketsLoading" class="list-loading">Loading markets...</div>
          <div v-else-if="!markets.length" class="list-empty">No markets found.</div>
          <div
            v-for="m in markets"
            :key="m.ticker"
            class="market-item"
            :class="{ selected: selectedMarket && selectedMarket.ticker === m.ticker }"
            @click="selectMarket(m)"
          >
            <div class="market-item-question">{{ m.question || m.ticker }}</div>
            <div class="market-item-meta">
              <span class="meta-price yes-price">YES {{ pct(m.yes_price) }}</span>
              <span class="meta-cat">{{ m.category }}</span>
            </div>
          </div>
          <button v-if="markets.length >= limit" class="load-more-btn" @click="loadMore">Load more</button>
        </div>

        <!-- Opportunities tab -->
        <div v-if="tab === 'scan'" class="market-list">
          <div v-if="scanLoading" class="list-loading">Scanning for edge...</div>
          <div v-else-if="!opportunities.length" class="list-empty">No opportunities found.</div>
          <div
            v-for="o in opportunities"
            :key="o.ticker"
            class="market-item opp-item"
            :class="{ selected: selectedMarket && selectedMarket.ticker === o.ticker }"
            @click="selectOpportunity(o)"
          >
            <div class="market-item-question">{{ o.question || o.ticker }}</div>
            <div class="market-item-meta">
              <span class="meta-edge" :class="edgeClass(o.edge_signal)">{{ o.edge_signal }}</span>
              <span class="meta-edge-val">edge {{ pctSigned(o.edge) }}</span>
            </div>
          </div>
        </div>

        <!-- Portfolio tab -->
        <div v-if="tab === 'portfolio'" class="portfolio-panel">
          <div v-if="portfolioLoading" class="list-loading">Loading portfolio...</div>
          <div v-else>
            <div class="portfolio-section">
              <div class="portfolio-label">BALANCE</div>
              <div class="portfolio-balance">
                {{ portfolio.balance ? '$' + portfolio.balance.balance_usd : 'N/A' }}
              </div>
            </div>
            <div class="portfolio-section">
              <div class="portfolio-label">OPEN ORDERS ({{ portfolio.orders.length }})</div>
              <div v-for="o in portfolio.orders" :key="o.order_id" class="portfolio-row">
                <span class="portfolio-ticker">{{ o.ticker }}</span>
                <span class="portfolio-side" :class="sideClass(o.side)">{{ o.side }}</span>
                <span class="portfolio-price">{{ o.price_cents }}¢</span>
                <span class="portfolio-count">{{ o.count }} cts</span>
                <span class="portfolio-status" :class="{ 'status-dry': o.dry_run }">
                  {{ o.dry_run ? 'DRY RUN' : o.status }}
                </span>
              </div>
              <div v-if="!portfolio.orders.length" class="list-empty">No open orders.</div>
            </div>
            <div class="portfolio-section">
              <div class="portfolio-label">POSITIONS ({{ portfolio.positions.length }})</div>
              <div v-for="p in portfolio.positions" :key="p.ticker" class="portfolio-row">
                <span class="portfolio-ticker">{{ p.ticker }}</span>
                <span>{{ p.position }}</span>
              </div>
              <div v-if="!portfolio.positions.length" class="list-empty">No open positions.</div>
            </div>
          </div>
        </div>
      </aside>

      <!-- RIGHT: Detail + Trade -->
      <main class="detail-panel">
        <div v-if="!selectedMarket" class="empty-state">
          <div class="empty-icon">◈</div>
          <div class="empty-title">Select a market</div>
          <div class="empty-desc">Browse the catalog or scan for opportunities, then click a market to analyse and trade.</div>
        </div>

        <div v-else>
          <!-- Market header -->
          <div class="detail-header">
            <div class="detail-ticker">{{ selectedMarket.ticker }}</div>
            <div class="detail-question">{{ selectedMarket.question }}</div>
            <div class="detail-meta-row">
              <span class="detail-cat">{{ selectedMarket.category }}</span>
              <span class="detail-close">{{ selectedMarket.days_to_close != null ? selectedMarket.days_to_close + 'd to close' : '' }}</span>
              <span class="detail-vol" v-if="selectedMarket.volume">Vol: {{ fmtVol(selectedMarket.volume) }}</span>
            </div>
          </div>

          <!-- Live prices -->
          <div class="price-bar">
            <div class="price-block yes-block">
              <div class="price-label">YES</div>
              <div class="price-val">{{ pct(selectedMarket.yes_price) }}</div>
              <div class="price-cents">{{ selectedMarket.yes_price != null ? Math.round(selectedMarket.yes_price * 100) + '¢' : '' }}</div>
            </div>
            <div class="price-divider">VS</div>
            <div class="price-block no-block">
              <div class="price-label">NO</div>
              <div class="price-val no-val">{{ pct(selectedMarket.no_price) }}</div>
              <div class="price-cents">{{ selectedMarket.no_price != null ? Math.round(selectedMarket.no_price * 100) + '¢' : '' }}</div>
            </div>
          </div>

          <!-- Prediction -->
          <div class="prediction-section">
            <div class="section-tag">ML PREDICTION</div>
            <button
              class="predict-btn"
              :disabled="predicting"
              @click="runPrediction"
            >
              {{ predicting ? 'Running...' : 'Run Prediction' }}
            </button>

            <div v-if="prediction" class="prediction-result">
              <div class="pred-row">
                <div class="pred-block">
                  <div class="pred-label">ML YES</div>
                  <div class="pred-val yes-val">{{ pct(prediction.yes_probability) }}</div>
                </div>
                <div class="pred-block">
                  <div class="pred-label">EDGE</div>
                  <div class="pred-val edge-val" :class="edgeClass(prediction.edge_signal)">
                    {{ pctSigned(prediction.edge) }}
                  </div>
                </div>
                <div class="pred-block">
                  <div class="pred-label">SIGNAL</div>
                  <div class="pred-signal" :class="edgeClass(prediction.edge_signal)">
                    {{ prediction.edge_signal || '—' }}
                  </div>
                </div>
                <div class="pred-block">
                  <div class="pred-label">CONF</div>
                  <div class="pred-conf" :class="'conf-' + prediction.confidence">
                    {{ prediction.confidence?.toUpperCase() || '—' }}
                  </div>
                </div>
              </div>
              <div v-if="prediction.reasoning_summary" class="pred-reasoning">
                {{ prediction.reasoning_summary }}
              </div>
              <div v-if="prediction.key_factors?.length" class="pred-factors">
                <div class="factors-label">KEY FACTORS</div>
                <ul>
                  <li v-for="(f, i) in prediction.key_factors" :key="i">{{ f }}</li>
                </ul>
              </div>
              <div class="pred-kelly" v-if="prediction.kelly_fraction">
                Kelly fraction: <strong>{{ (prediction.kelly_fraction * 100).toFixed(1) }}%</strong>
                &nbsp;|&nbsp; Suggested bet: <strong>${{ prediction.suggested_bet_size?.toFixed(2) || '—' }}</strong>
              </div>
            </div>
          </div>

          <!-- Trade form -->
          <div class="trade-section">
            <div class="section-tag">
              PLACE TRADE
              <span class="dry-badge" v-if="dryRunMode">DRY RUN</span>
            </div>

            <div class="trade-form">
              <div class="form-row">
                <div class="form-field">
                  <label class="form-label">SIDE</label>
                  <div class="side-btns">
                    <button
                      v-for="s in ['BUY_YES', 'BUY_NO', 'SELL_YES', 'SELL_NO']"
                      :key="s"
                      class="side-btn"
                      :class="{ active: tradeForm.side === s, 'buy-yes': s === 'BUY_YES', 'buy-no': s === 'BUY_NO', 'sell-yes': s === 'SELL_YES', 'sell-no': s === 'SELL_NO' }"
                      @click="tradeForm.side = s"
                    >{{ s.replace('_', ' ') }}</button>
                  </div>
                </div>
              </div>

              <div class="form-row">
                <div class="form-field">
                  <label class="form-label">AMOUNT (USD)</label>
                  <input v-model.number="tradeForm.amount" type="number" min="1" class="form-input" placeholder="100" />
                </div>
                <div class="form-field">
                  <label class="form-label">PRICE (0–1)</label>
                  <input v-model.number="tradeForm.price" type="number" min="0.01" max="0.99" step="0.01" class="form-input" :placeholder="(selectedMarket.yes_price || 0.5).toFixed(2)" />
                </div>
                <div class="form-field">
                  <label class="form-label">KELLY</label>
                  <input v-model.number="tradeForm.kelly" type="number" min="0.05" max="1" step="0.05" class="form-input" placeholder="0.25" />
                </div>
              </div>

              <div class="form-summary" v-if="tradeForm.amount && tradeForm.price">
                Sized: <strong>${{ (tradeForm.amount * (tradeForm.kelly || 0.25)).toFixed(2) }}</strong>
                &nbsp;→&nbsp;
                <strong>{{ calcContracts() }} contracts</strong> @ {{ Math.round((tradeForm.price || selectedMarket.yes_price || 0.5) * 100) }}¢
              </div>

              <button
                class="trade-btn"
                :disabled="trading || !tradeForm.side || !tradeForm.amount || !tradeForm.price"
                @click="placeTrade"
              >
                {{ trading ? 'Placing...' : (dryRunMode ? 'Simulate Trade' : 'Place Trade') }}
              </button>
            </div>

            <div v-if="tradeResult" class="trade-result" :class="{ 'result-ok': !tradeResult.error, 'result-err': tradeResult.error }">
              <div class="result-status">{{ tradeResult.error ? 'FAILED' : (tradeResult.dry_run ? 'DRY RUN ACCEPTED' : 'ORDER PLACED') }}</div>
              <div class="result-id" v-if="tradeResult.order_id">ID: {{ tradeResult.order_id }}</div>
              <div class="result-detail" v-if="tradeResult.count">
                {{ tradeResult.count }} contracts · {{ tradeResult.price_cents }}¢ · ${{ tradeResult.amount_usd?.toFixed(2) }}
              </div>
              <div class="result-error" v-if="tradeResult.error">{{ tradeResult.error }}</div>
            </div>
          </div>
        </div>
      </main>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import {
  getKalshiCatalog, searchKalshiMarkets, getKalshiMarket,
  predictKalshi, scanKalshiOpportunities, placeKalshiTrade,
  getKalshiOrders, getKalshiPositions, getKalshiBalance, getKalshiStatus
} from '../api/kalshi'

const tab = ref('catalog')
const searchQuery = ref('')
const markets = ref([])
const opportunities = ref([])
const marketsLoading = ref(false)
const scanLoading = ref(false)
const portfolioLoading = ref(false)
const limit = ref(50)
const cursor = ref('')

const selectedMarket = ref(null)
const prediction = ref(null)
const predicting = ref(false)

const trading = ref(false)
const tradeResult = ref(null)
const tradeForm = ref({ side: 'BUY_YES', amount: 100, price: null, kelly: 0.25 })

const dryRunMode = ref(true)
const executorStatus = ref(null)

const portfolio = ref({ balance: null, orders: [], positions: [] })

const statusClass = computed(() => {
  if (!executorStatus.value) return 'status-unknown'
  if (executorStatus.value.executor?.connected) return 'status-ok'
  return executorStatus.value.executor?.has_api_key ? 'status-warn' : 'status-off'
})

const statusLabel = computed(() => {
  if (!executorStatus.value) return 'Checking...'
  const e = executorStatus.value.executor
  if (!e) return 'Unknown'
  if (e.connected) return e.dry_run ? 'DRY RUN' : e.mode
  return e.has_api_key ? 'Disconnected' : 'No Credentials'
})

const pct = (v) => v != null ? `${Math.round(v * 100)}%` : '—'
const pctSigned = (v) => {
  if (v == null) return '—'
  const p = Math.round(v * 100)
  return p > 0 ? `+${p}%` : `${p}%`
}
const fmtVol = (v) => v >= 1_000_000 ? `$${(v / 1_000_000).toFixed(1)}M` : v >= 1000 ? `$${(v / 1000).toFixed(0)}K` : `$${v}`

const edgeClass = (signal) => ({
  'edge-buy': signal === 'BUY YES',
  'edge-sell': signal === 'BUY NO',
  'edge-neutral': signal === 'NEUTRAL',
})

const sideClass = (side) => ({
  'side-buy-yes': side === 'BUY_YES',
  'side-buy-no': side === 'BUY_NO',
  'side-sell': side?.startsWith('SELL'),
})

function calcContracts() {
  const price = tradeForm.value.price || selectedMarket.value?.yes_price || 0.5
  const amount = tradeForm.value.amount * (tradeForm.value.kelly || 0.25)
  const side = tradeForm.value.side || 'BUY_YES'
  const isBuyYes = side === 'BUY_YES'
  const costPer = isBuyYes ? price : (1 - price)
  return Math.max(1, Math.floor(amount / costPer))
}

async function loadCatalog(reset = false) {
  if (reset) { markets.value = []; cursor.value = '' }
  marketsLoading.value = true
  try {
    const res = await getKalshiCatalog(limit.value, cursor.value)
    markets.value = [...markets.value, ...(res.data?.markets || [])]
  } catch (e) {
    console.error(e)
  } finally {
    marketsLoading.value = false
  }
}

async function loadMore() {
  cursor.value = markets.value[markets.value.length - 1]?.ticker || ''
  await loadCatalog()
}

async function doSearch() {
  if (!searchQuery.value.trim()) { loadCatalog(true); return }
  marketsLoading.value = true
  try {
    const res = await searchKalshiMarkets(searchQuery.value)
    markets.value = res.data?.markets || []
  } catch (e) {
    console.error(e)
  } finally {
    marketsLoading.value = false
  }
}

async function scanOpportunities() {
  scanLoading.value = true
  opportunities.value = []
  try {
    const res = await scanKalshiOpportunities()
    opportunities.value = res.data?.opportunities || []
  } catch (e) {
    console.error(e)
  } finally {
    scanLoading.value = false
  }
}

async function loadPortfolio() {
  portfolioLoading.value = true
  try {
    const [bal, ord, pos] = await Promise.allSettled([
      getKalshiBalance(), getKalshiOrders(), getKalshiPositions()
    ])
    portfolio.value.balance = bal.status === 'fulfilled' ? bal.value?.data : null
    portfolio.value.orders = ord.status === 'fulfilled' ? (ord.value?.data?.orders || []) : []
    portfolio.value.positions = pos.status === 'fulfilled' ? (pos.value?.data?.positions || []) : []
  } finally {
    portfolioLoading.value = false
  }
}

function switchTab(t) {
  tab.value = t
  if (t === 'catalog' && !markets.value.length) loadCatalog(true)
  if (t === 'scan' && !opportunities.value.length) scanOpportunities()
  if (t === 'portfolio') loadPortfolio()
}

async function selectMarket(m) {
  prediction.value = null
  tradeResult.value = null
  tradeForm.value.price = m.yes_price ? parseFloat(m.yes_price.toFixed(2)) : null
  try {
    const res = await getKalshiMarket(m.ticker)
    selectedMarket.value = res.data || m
  } catch {
    selectedMarket.value = m
  }
}

function selectOpportunity(o) {
  prediction.value = {
    yes_probability: o.predicted_prob,
    edge: o.edge,
    edge_signal: o.edge_signal,
    confidence: o.confidence,
    kelly_fraction: o.kelly_fraction,
    suggested_bet_size: o.suggested_bet_size,
    reasoning_summary: o.reasoning_summary,
    key_factors: o.key_factors,
  }
  selectMarket(o)
}

async function runPrediction() {
  if (!selectedMarket.value) return
  predicting.value = true
  try {
    const res = await predictKalshi({ ticker: selectedMarket.value.ticker })
    prediction.value = res.data
    if (!tradeForm.value.price && selectedMarket.value.yes_price) {
      tradeForm.value.price = parseFloat(selectedMarket.value.yes_price.toFixed(2))
    }
    if (prediction.value?.kelly_fraction) {
      tradeForm.value.kelly = parseFloat(prediction.value.kelly_fraction.toFixed(2))
    }
  } catch (e) {
    console.error(e)
  } finally {
    predicting.value = false
  }
}

async function placeTrade() {
  if (!selectedMarket.value) return
  trading.value = true
  tradeResult.value = null
  try {
    const body = {
      ticker: selectedMarket.value.ticker,
      side: tradeForm.value.side,
      amount: tradeForm.value.amount,
      price: tradeForm.value.price || selectedMarket.value.yes_price,
      kelly_fraction: tradeForm.value.kelly || 0.25,
    }
    const res = await placeKalshiTrade(body)
    tradeResult.value = res.data
  } catch (e) {
    tradeResult.value = { error: e.message || String(e) }
  } finally {
    trading.value = false
  }
}

async function checkStatus() {
  try {
    const res = await getKalshiStatus()
    executorStatus.value = res.data
    dryRunMode.value = res.data?.executor?.dry_run ?? true
  } catch (e) {
    console.error(e)
  }
}

onMounted(() => {
  checkStatus()
  loadCatalog(true)
})
</script>

<style scoped>
.kalshi-view {
  min-height: 100vh;
  background: #fff;
  font-family: 'JetBrains Mono', monospace;
  display: flex;
  flex-direction: column;
}

.navbar {
  display: flex;
  align-items: center;
  padding: 0 24px;
  height: 52px;
  border-bottom: 1px solid #e5e5e5;
  gap: 16px;
}

.nav-brand {
  font-size: 0.85rem;
  font-weight: 700;
  letter-spacing: 2px;
  cursor: pointer;
}

.nav-sub {
  font-size: 0.75rem;
  color: #999;
  letter-spacing: 1px;
}

.nav-status {
  margin-left: auto;
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 0.65rem;
  font-weight: 700;
  letter-spacing: 1px;
}

.status-dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: #ccc;
}

.status-ok .status-dot { background: #22c55e; }
.status-ok { color: #22c55e; }
.status-warn .status-dot { background: #f59e0b; }
.status-warn { color: #f59e0b; }
.status-off .status-dot { background: #ef4444; }
.status-off { color: #ef4444; }
.status-unknown { color: #999; }

.layout {
  display: flex;
  flex: 1;
  overflow: hidden;
}

.sidebar {
  width: 320px;
  min-width: 280px;
  border-right: 1px solid #e5e5e5;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.sidebar-header {
  padding: 16px;
  border-bottom: 1px solid #e5e5e5;
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.section-tag {
  font-size: 0.6rem;
  font-weight: 700;
  letter-spacing: 2px;
  color: #999;
}

.search-row {
  display: flex;
  gap: 6px;
}

.search-input {
  flex: 1;
  border: 1px solid #e5e5e5;
  padding: 7px 10px;
  font-family: inherit;
  font-size: 0.8rem;
  outline: none;
}

.search-input:focus { border-color: #000; }

.icon-btn {
  border: 1px solid #000;
  background: #000;
  color: #fff;
  padding: 0 12px;
  cursor: pointer;
  font-size: 1rem;
}

.tab-row {
  display: flex;
  gap: 4px;
}

.tab-btn {
  flex: 1;
  border: 1px solid #e5e5e5;
  background: none;
  padding: 5px 0;
  font-family: inherit;
  font-size: 0.65rem;
  font-weight: 700;
  letter-spacing: 1px;
  cursor: pointer;
  color: #999;
}

.tab-btn.active {
  background: #000;
  color: #fff;
  border-color: #000;
}

.market-list {
  flex: 1;
  overflow-y: auto;
  padding: 8px 0;
}

.list-loading, .list-empty {
  padding: 24px 16px;
  font-size: 0.8rem;
  color: #999;
}

.market-item {
  padding: 12px 16px;
  cursor: pointer;
  border-bottom: 1px solid #f5f5f5;
  transition: background 0.15s;
}

.market-item:hover { background: #fafafa; }
.market-item.selected { background: #f0f0f0; border-left: 3px solid #000; }

.market-item-question {
  font-size: 0.78rem;
  font-weight: 600;
  line-height: 1.4;
  margin-bottom: 5px;
  color: #000;
}

.market-item-meta {
  display: flex;
  gap: 10px;
  align-items: center;
}

.meta-price {
  font-size: 0.65rem;
  font-weight: 700;
  padding: 2px 6px;
}

.yes-price { background: #dcfce7; color: #166534; }
.meta-cat { font-size: 0.65rem; color: #999; text-transform: uppercase; letter-spacing: 1px; }

.meta-edge {
  font-size: 0.65rem;
  font-weight: 700;
  padding: 2px 6px;
}

.edge-buy { background: #dcfce7; color: #166534; }
.edge-sell { background: #fee2e2; color: #991b1b; }
.edge-neutral { background: #f3f4f6; color: #6b7280; }

.meta-edge-val { font-size: 0.7rem; color: #666; }

.load-more-btn {
  width: 100%;
  padding: 10px;
  border: none;
  border-top: 1px solid #e5e5e5;
  background: none;
  font-family: inherit;
  font-size: 0.7rem;
  color: #666;
  cursor: pointer;
}

.load-more-btn:hover { background: #fafafa; }

.portfolio-panel {
  flex: 1;
  overflow-y: auto;
  padding: 12px 16px;
}

.portfolio-section { margin-bottom: 20px; }

.portfolio-label {
  font-size: 0.6rem;
  font-weight: 700;
  letter-spacing: 2px;
  color: #999;
  margin-bottom: 8px;
}

.portfolio-balance {
  font-size: 1.4rem;
  font-weight: 700;
}

.portfolio-row {
  display: flex;
  gap: 8px;
  align-items: center;
  padding: 6px 0;
  border-bottom: 1px solid #f5f5f5;
  font-size: 0.75rem;
  flex-wrap: wrap;
}

.portfolio-ticker { font-weight: 700; flex: 1; min-width: 80px; }
.portfolio-side { padding: 1px 5px; font-size: 0.6rem; font-weight: 700; }
.side-buy-yes { background: #dcfce7; color: #166534; }
.side-buy-no { background: #fee2e2; color: #991b1b; }
.side-sell { background: #fef3c7; color: #92400e; }
.portfolio-price { color: #666; }
.portfolio-count { color: #666; }
.portfolio-status { font-size: 0.6rem; font-weight: 700; letter-spacing: 1px; margin-left: auto; }
.status-dry { color: #f59e0b; }

/* Detail Panel */
.detail-panel {
  flex: 1;
  overflow-y: auto;
  padding: 24px;
  display: flex;
  flex-direction: column;
  gap: 24px;
}

.empty-state {
  margin: auto;
  text-align: center;
  color: #ccc;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 12px;
}

.empty-icon { font-size: 2.5rem; }
.empty-title { font-size: 0.9rem; font-weight: 700; color: #999; letter-spacing: 2px; }
.empty-desc { font-size: 0.8rem; color: #bbb; max-width: 300px; line-height: 1.6; }

.detail-header {
  border-bottom: 2px solid #000;
  padding-bottom: 16px;
}

.detail-ticker {
  font-size: 0.7rem;
  font-weight: 700;
  letter-spacing: 2px;
  color: #999;
  margin-bottom: 6px;
}

.detail-question {
  font-size: 1.1rem;
  font-weight: 700;
  line-height: 1.4;
  margin-bottom: 10px;
}

.detail-meta-row {
  display: flex;
  gap: 16px;
  font-size: 0.75rem;
  color: #666;
}

.detail-cat { text-transform: uppercase; letter-spacing: 1px; font-weight: 700; }

.price-bar {
  display: flex;
  align-items: center;
  gap: 20px;
  padding: 20px;
  border: 1px solid #e5e5e5;
}

.price-block {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 4px;
}

.price-label {
  font-size: 0.6rem;
  font-weight: 700;
  letter-spacing: 2px;
  color: #999;
}

.price-val {
  font-size: 2rem;
  font-weight: 700;
  color: #22c55e;
}

.no-val { color: #ef4444; }

.price-cents {
  font-size: 0.75rem;
  color: #999;
}

.price-divider {
  font-size: 0.7rem;
  font-weight: 700;
  color: #ccc;
}

.prediction-section {
  border: 1px solid #e5e5e5;
  padding: 20px;
  display: flex;
  flex-direction: column;
  gap: 14px;
}

.predict-btn {
  align-self: flex-start;
  border: 1px solid #000;
  background: #000;
  color: #fff;
  padding: 8px 20px;
  font-family: inherit;
  font-size: 0.75rem;
  font-weight: 700;
  letter-spacing: 1px;
  cursor: pointer;
}

.predict-btn:disabled { opacity: 0.4; cursor: default; }

.prediction-result { display: flex; flex-direction: column; gap: 12px; }

.pred-row {
  display: flex;
  gap: 24px;
  align-items: flex-end;
}

.pred-block { display: flex; flex-direction: column; gap: 4px; }
.pred-label { font-size: 0.6rem; font-weight: 700; letter-spacing: 1px; color: #999; }

.pred-val {
  font-size: 1.4rem;
  font-weight: 700;
}

.yes-val { color: #22c55e; }

.edge-val { }
.edge-buy.edge-val { color: #22c55e; }
.edge-sell.edge-val { color: #ef4444; }

.pred-signal {
  font-size: 0.75rem;
  font-weight: 700;
  padding: 3px 8px;
  letter-spacing: 1px;
}

.pred-signal.edge-buy { background: #dcfce7; color: #166534; }
.pred-signal.edge-sell { background: #fee2e2; color: #991b1b; }
.pred-signal.edge-neutral { background: #f3f4f6; color: #6b7280; }

.pred-conf {
  font-size: 0.65rem;
  font-weight: 700;
  padding: 3px 8px;
  letter-spacing: 1px;
}

.conf-high { background: #22c55e; color: #000; }
.conf-medium { background: #f59e0b; color: #000; }
.conf-low { background: #ef4444; color: #fff; }

.pred-reasoning {
  font-size: 0.8rem;
  line-height: 1.6;
  color: #444;
  border-left: 2px solid #FF4500;
  padding-left: 10px;
}

.pred-factors { }
.factors-label { font-size: 0.6rem; font-weight: 700; letter-spacing: 1px; color: #999; margin-bottom: 6px; }
.pred-factors ul { list-style: none; padding: 0; margin: 0; display: flex; flex-direction: column; gap: 3px; }
.pred-factors li { font-size: 0.78rem; color: #444; padding-left: 10px; position: relative; }
.pred-factors li::before { content: '›'; position: absolute; left: 0; color: #FF4500; font-weight: 700; }

.pred-kelly { font-size: 0.78rem; color: #666; }

.trade-section {
  border: 1px solid #e5e5e5;
  padding: 20px;
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.section-tag {
  display: flex;
  align-items: center;
  gap: 10px;
}

.dry-badge {
  font-size: 0.6rem;
  font-weight: 700;
  padding: 2px 8px;
  background: #fef3c7;
  color: #92400e;
  letter-spacing: 1px;
}

.trade-form { display: flex; flex-direction: column; gap: 14px; }
.form-row { display: flex; gap: 16px; flex-wrap: wrap; }
.form-field { display: flex; flex-direction: column; gap: 6px; }
.form-label { font-size: 0.6rem; font-weight: 700; letter-spacing: 1px; color: #999; }

.form-input {
  border: 1px solid #e5e5e5;
  padding: 7px 10px;
  font-family: inherit;
  font-size: 0.85rem;
  width: 110px;
  outline: none;
}

.form-input:focus { border-color: #000; }

.side-btns { display: flex; gap: 4px; flex-wrap: wrap; }

.side-btn {
  border: 1px solid #e5e5e5;
  background: none;
  padding: 6px 10px;
  font-family: inherit;
  font-size: 0.65rem;
  font-weight: 700;
  letter-spacing: 1px;
  cursor: pointer;
  color: #666;
}

.side-btn.active.buy-yes { background: #22c55e; color: #000; border-color: #22c55e; }
.side-btn.active.buy-no { background: #ef4444; color: #fff; border-color: #ef4444; }
.side-btn.active.sell-yes { background: #f59e0b; color: #000; border-color: #f59e0b; }
.side-btn.active.sell-no { background: #6b7280; color: #fff; border-color: #6b7280; }

.form-summary { font-size: 0.78rem; color: #666; }

.trade-btn {
  align-self: flex-start;
  border: 2px solid #000;
  background: #000;
  color: #fff;
  padding: 10px 28px;
  font-family: inherit;
  font-size: 0.75rem;
  font-weight: 700;
  letter-spacing: 1px;
  cursor: pointer;
}

.trade-btn:disabled { opacity: 0.4; cursor: default; }

.trade-result {
  padding: 14px;
  border-left: 3px solid #22c55e;
  background: #f0fdf4;
}

.trade-result.result-err {
  border-left-color: #ef4444;
  background: #fef2f2;
}

.result-status { font-size: 0.65rem; font-weight: 700; letter-spacing: 2px; margin-bottom: 4px; }
.result-id { font-size: 0.75rem; color: #666; margin-bottom: 4px; }
.result-detail { font-size: 0.8rem; font-weight: 600; }
.result-error { font-size: 0.78rem; color: #ef4444; margin-top: 4px; }
</style>
