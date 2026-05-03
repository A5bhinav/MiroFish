import service from './index'

export function getPolymarketCatalog(limit = 50, offset = 0) {
  return service({ url: '/api/polymarket/catalog', method: 'get', params: { limit, offset } })
}

export function searchPolymarketMarkets(q, limit = 20) {
  return service({ url: '/api/polymarket/search', method: 'get', params: { q, limit } })
}

export function getPolymarketMarket(marketId) {
  return service({ url: `/api/polymarket/markets/${marketId}`, method: 'get' })
}

export function predictPolymarket(body) {
  return service({ url: '/api/polymarket/predict', method: 'post', data: body })
}

export function scanPolymarketOpportunities() {
  return service({ url: '/api/polymarket/scan-opportunities', method: 'get' })
}

export function placePolymarketTrade(body) {
  return service({ url: '/api/polymarket/trade', method: 'post', data: body })
}

export function getPolymarketOrders() {
  return service({ url: '/api/polymarket/orders', method: 'get' })
}

export function getPolymarketPositions() {
  return service({ url: '/api/polymarket/positions', method: 'get' })
}

export function getPolymarketStatus() {
  return service({ url: '/api/polymarket/status', method: 'get' })
}
