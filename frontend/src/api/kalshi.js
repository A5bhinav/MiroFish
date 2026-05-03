import service from './index'

export function getKalshiCatalog(limit = 50, cursor = '') {
  return service({ url: '/api/kalshi/catalog', method: 'get', params: { limit, cursor } })
}

export function searchKalshiMarkets(q, limit = 20) {
  return service({ url: '/api/kalshi/search', method: 'get', params: { q, limit } })
}

export function getKalshiMarket(ticker) {
  return service({ url: `/api/kalshi/markets/${ticker}`, method: 'get' })
}

export function predictKalshi(body) {
  return service({ url: '/api/kalshi/predict', method: 'post', data: body })
}

export function scanKalshiOpportunities() {
  return service({ url: '/api/kalshi/scan-opportunities', method: 'get' })
}

export function placeKalshiTrade(body) {
  return service({ url: '/api/kalshi/trade', method: 'post', data: body })
}

export function getKalshiOrders() {
  return service({ url: '/api/kalshi/orders', method: 'get' })
}

export function getKalshiPositions(ticker) {
  return service({ url: '/api/kalshi/positions', method: 'get', params: ticker ? { ticker } : {} })
}

export function getKalshiBalance() {
  return service({ url: '/api/kalshi/balance', method: 'get' })
}

export function getKalshiStatus() {
  return service({ url: '/api/kalshi/status', method: 'get' })
}
