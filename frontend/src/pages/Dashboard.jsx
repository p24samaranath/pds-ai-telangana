import React, { useState, useEffect } from 'react';
import { getDashboardMetrics, getOrchestratorStatus } from '../services/api';

const METRIC_CONFIG = [
  { key: 'total_fps_shops',           label: 'Total FPS Shops',     icon: '🏪', type: 'info'    },
  { key: 'active_fps_shops',          label: 'Active Shops',        icon: '✅', type: 'success' },
  { key: 'total_beneficiaries',       label: 'Beneficiaries',       icon: '👥', type: 'info'    },
  { key: 'transactions_this_month',   label: 'Transactions (Month)',icon: '📋', type: 'info'    },
  { key: 'fraud_alerts_open',         label: 'Fraud Alerts',        icon: '⚠️', type: 'warning' },
  { key: 'fraud_alerts_critical',     label: 'Critical Alerts',     icon: '🚨', type: 'danger'  },
  { key: 'beneficiaries_within_3km_pct', label: 'Within 3km (%)',   icon: '📍', type: 'success', pct: true },
  { key: 'avg_forecast_accuracy',     label: 'Forecast Accuracy',   icon: '🎯', type: 'success', pct: true },
];

export default function Dashboard() {
  const [metrics, setMetrics] = useState(null);
  const [summary, setSummary] = useState('');
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchData();
  }, []);

  async function fetchData() {
    setLoading(true);
    setError(null);
    try {
      const [dashData, statusData] = await Promise.all([
        getDashboardMetrics().catch(() => null),
        getOrchestratorStatus().catch(() => null),
      ]);

      if (dashData?.metrics) setMetrics(dashData.metrics);
      if (dashData?.executive_summary) setSummary(dashData.executive_summary);
      if (statusData) setStatus(statusData);
    } catch (e) {
      setError('Failed to load dashboard. Make sure the backend is running on port 8000.');
    } finally {
      setLoading(false);
    }
  }

  if (loading) return (
    <div className="loading"><div className="spinner" /><span>Loading PDS dashboard…</span></div>
  );

  return (
    <div>
      <div className="page-header">
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <div>
            <div className="page-title">PDS AI Dashboard — Telangana</div>
            <div className="page-subtitle">Real-time overview of Fair Price Shop network</div>
          </div>
          <button className="btn btn-primary" onClick={fetchData}>↻ Refresh</button>
        </div>
      </div>

      {error && (
        <div className="card" style={{ borderColor: '#ef4444', background: 'rgba(239,68,68,0.08)' }}>
          <p style={{ color: '#ef4444' }}>⚠️ {error}</p>
          <p style={{ color: '#94a3b8', marginTop: 8, fontSize: 13 }}>
            Start the backend: <code style={{ background: '#0f172a', padding: '2px 6px', borderRadius: 4 }}>
              cd backend && python -m uvicorn app.main:app --reload
            </code>
          </p>
        </div>
      )}

      {/* Metric Cards */}
      {metrics && (
        <div className="metrics-grid">
          {METRIC_CONFIG.map(cfg => (
            <div key={cfg.key} className={`metric-card ${cfg.type}`}>
              <div className="metric-icon">{cfg.icon}</div>
              <div className="metric-value">
                {cfg.pct
                  ? `${(Number(metrics[cfg.key] || 0) * (cfg.key.includes('accuracy') ? 100 : 1)).toFixed(1)}%`
                  : (metrics[cfg.key] ?? '—').toLocaleString()
                }
              </div>
              <div className="metric-label">{cfg.label}</div>
            </div>
          ))}
        </div>
      )}

      <div className="grid-2">
        {/* Fraud by District */}
        {metrics?.top_fraud_districts?.length > 0 && (
          <div className="card">
            <div className="card-header">
              <div>
                <div className="card-title">🚨 Top Fraud Districts</div>
                <div className="card-subtitle">Highest alert count this period</div>
              </div>
            </div>
            {metrics.top_fraud_districts.map((d, i) => (
              <div key={i} style={{ marginBottom: 12 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                  <span style={{ fontSize: 14, color: '#e2e8f0' }}>{d.district}</span>
                  <span style={{ fontSize: 14, fontWeight: 700, color: '#f59e0b' }}>{d.alert_count} alerts</span>
                </div>
                <div className="progress-bar">
                  <div
                    className="progress-fill red"
                    style={{ width: `${Math.min(100, (d.alert_count / 20) * 100)}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Commodity Distribution */}
        {metrics?.monthly_distribution_kg && Object.keys(metrics.monthly_distribution_kg).length > 0 && (
          <div className="card">
            <div className="card-header">
              <div>
                <div className="card-title">🌾 Monthly Distribution</div>
                <div className="card-subtitle">Commodity volumes this month (kg)</div>
              </div>
            </div>
            {Object.entries(metrics.monthly_distribution_kg).map(([commodity, kg]) => (
              <div key={commodity} style={{ marginBottom: 12 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                  <span style={{ fontSize: 14, color: '#e2e8f0', textTransform: 'capitalize' }}>{commodity.replace('_', ' ')}</span>
                  <span style={{ fontSize: 14, fontWeight: 700, color: '#10b981' }}>{Number(kg).toLocaleString()} kg</span>
                </div>
                <div className="progress-bar">
                  <div className="progress-fill green" style={{ width: '75%' }} />
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Executive Summary */}
      {summary && (
        <div className="card">
          <div className="card-header">
            <div>
              <div className="card-title">🤖 AI Executive Summary</div>
              <div className="card-subtitle">Generated by Claude — {new Date().toLocaleDateString('en-IN', { month: 'long', year: 'numeric' })}</div>
            </div>
          </div>
          <p style={{ fontSize: 14, lineHeight: 1.8, color: '#94a3b8', whiteSpace: 'pre-wrap' }}>{summary}</p>
        </div>
      )}

      {/* Orchestrator Status */}
      {status && (
        <div className="card">
          <div className="card-header">
            <div className="card-title">⚙️ Orchestrator Status</div>
            <span className="badge badge-medium">{status.status}</span>
          </div>
          <div style={{ display: 'flex', gap: 32, flexWrap: 'wrap', fontSize: 13, color: '#94a3b8' }}>
            <div><strong>Last Fraud Check:</strong> {status.last_fraud_check ? new Date(status.last_fraud_check).toLocaleString('en-IN') : 'Never'}</div>
            <div><strong>Last Forecast:</strong> {status.last_forecast_run ? new Date(status.last_forecast_run).toLocaleString('en-IN') : 'Never'}</div>
            <div><strong>Pending Alerts:</strong> {status.pending_alerts}</div>
            <div><strong>System Health:</strong> <span style={{ color: '#10b981' }}>✅ {status.system_health}</span></div>
          </div>
        </div>
      )}
    </div>
  );
}
