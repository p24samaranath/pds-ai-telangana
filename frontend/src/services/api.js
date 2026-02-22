import axios from 'axios';

const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: BASE_URL,
  timeout: 60000,
  headers: { 'Content-Type': 'application/json' },
});

// ── Dashboard ──────────────────────────────────────────────────────────────

export const getDashboardMetrics = () =>
  api.get('/api/v1/dashboard/metrics').then(r => r.data);

export const getExecutiveSummary = () =>
  api.get('/api/v1/dashboard/summary').then(r => r.data);

// ── Fraud Detection ────────────────────────────────────────────────────────

export const getFraudAlerts = (params = {}) =>
  api.get('/api/v1/fraud/alerts', { params }).then(r => r.data);

export const getCriticalAlerts = () =>
  api.get('/api/v1/fraud/alerts/critical').then(r => r.data);

export const getFraudSummary = () =>
  api.get('/api/v1/fraud/summary').then(r => r.data);

// ── Demand Forecasts ───────────────────────────────────────────────────────

export const getForecasts = (body) =>
  api.post('/api/v1/forecasts/', body).then(r => r.data);

export const getRiskFlags = (params = {}) =>
  api.get('/api/v1/forecasts/risk-flags', { params }).then(r => r.data);

// ── Geospatial ─────────────────────────────────────────────────────────────

export const getGeospatialAnalysis = (params = {}) =>
  api.get('/api/v1/geo/analysis', { params }).then(r => r.data);

export const getShopsGeoJSON = (params = {}) =>
  api.get('/api/v1/geo/shops', { params }).then(r => r.data);

export const getUnderservedZones = (params = {}) =>
  api.get('/api/v1/geo/underserved-zones', { params }).then(r => r.data);

export const getNewShopRecommendations = (n = 5) =>
  api.get('/api/v1/geo/recommendations', { params: { n } }).then(r => r.data);

export const getAccessibilityScores = () =>
  api.get('/api/v1/geo/accessibility-scores').then(r => r.data);

// ── Agents & NL Query ──────────────────────────────────────────────────────

export const getOrchestratorStatus = () =>
  api.get('/api/v1/agents/status').then(r => r.data);

export const runFullPipeline = () =>
  api.post('/api/v1/agents/run/full').then(r => r.data);

// Legacy single-turn query (kept for backwards compatibility)
export const submitNLQuery = (query, sessionId = 'default') =>
  api.post('/api/v1/agents/query', { query, session_id: sessionId }).then(r => r.data);

// Multi-turn RAG chat — use this for the chat interface
export const sendChatMessage = (query, sessionId = 'default') =>
  api.post('/api/v1/agents/chat', { query, session_id: sessionId }).then(r => r.data);

export const listChatSessions = () =>
  api.get('/api/v1/agents/chat/sessions').then(r => r.data);

export const clearChatSession = (sessionId) =>
  api.delete(`/api/v1/agents/chat/sessions/${sessionId}`).then(r => r.data);

export const getRagStats = () =>
  api.get('/api/v1/agents/chat/rag/stats').then(r => r.data);

// ── Data Management ────────────────────────────────────────────────────────

export const getDataStatus = () =>
  api.get('/api/v1/data/status').then(r => r.data);

export const getDataManifest = () =>
  api.get('/api/v1/data/manifest').then(r => r.data);

export const fetchLatestData = () =>
  api.post('/api/v1/data/fetch/latest').then(r => r.data);

export const fetchDataRange = (body) =>
  api.post('/api/v1/data/fetch/range', body).then(r => r.data);

export const getDataJobStatus = () =>
  api.get('/api/v1/data/job-status').then(r => r.data);

export const getVisualizationData = () =>
  api.get('/api/v1/data/visualize').then(r => r.data);

// ── Health & System ────────────────────────────────────────────────────────

export const getHealth = () =>
  api.get('/api/v1/health').then(r => r.data);

export const triggerShutdown = () =>
  api.post('/api/v1/system/shutdown').then(r => r.data);

export default api;
