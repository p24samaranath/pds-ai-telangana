import React, { useState, useEffect } from 'react';
import { getFraudAlerts, getFraudSummary } from '../services/api';

const SEVERITIES = ['All', 'Critical', 'High', 'Medium', 'Low'];

export default function FraudAlertsPage() {
  const [alerts, setAlerts] = useState([]);
  const [summary, setSummary] = useState(null);
  const [severity, setSeverity] = useState('All');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchAlerts();
  }, [severity]);

  async function fetchAlerts() {
    setLoading(true);
    try {
      const params = severity !== 'All' ? { severity } : {};
      const [alertData, sumData] = await Promise.all([
        getFraudAlerts(params),
        getFraudSummary(),
      ]);
      setAlerts(alertData.alerts || []);
      setSummary(sumData.summary || null);
    } catch {
      setAlerts([]);
    } finally {
      setLoading(false);
    }
  }

  const getSeverityClass = (s) => ({
    Critical: 'critical', High: 'high', Medium: 'medium', Low: 'low'
  }[s] || 'low');

  return (
    <div>
      <div className="page-header">
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <div>
            <div className="page-title">🚨 Fraud Detection Alerts</div>
            <div className="page-subtitle">AI-powered anomaly detection across all FPS transactions</div>
          </div>
          <button className="btn btn-primary" onClick={fetchAlerts}>↻ Refresh</button>
        </div>
      </div>

      {/* Summary Cards */}
      {summary && (
        <div className="metrics-grid" style={{ gridTemplateColumns: 'repeat(5, 1fr)' }}>
          {[
            { label: 'Total Alerts',    value: summary.total_alerts,          type: 'warning' },
            { label: 'Critical',        value: summary.critical,              type: 'danger'  },
            { label: 'High',            value: summary.high,                  type: 'warning' },
            { label: 'Medium',          value: summary.medium,                type: 'info'    },
            { label: 'Txns Analysed',   value: summary.transactions_analysed, type: 'success' },
          ].map((m, i) => (
            <div key={i} className={`metric-card ${m.type}`}>
              <div className="metric-value">{(m.value || 0).toLocaleString()}</div>
              <div className="metric-label">{m.label}</div>
            </div>
          ))}
        </div>
      )}

      {/* Severity Filter */}
      <div className="tab-row">
        {SEVERITIES.map(s => (
          <button
            key={s}
            className={`tab-btn ${severity === s ? 'active' : ''}`}
            onClick={() => setSeverity(s)}
          >
            {s}
          </button>
        ))}
      </div>

      {/* Alerts List */}
      {loading ? (
        <div className="loading"><div className="spinner" /><span>Running fraud detection…</span></div>
      ) : alerts.length === 0 ? (
        <div className="card" style={{ textAlign: 'center', color: '#64748b', padding: 48 }}>
          ✅ No fraud alerts for the selected filter.
        </div>
      ) : (
        <div>
          {alerts.map((alert, i) => (
            <div key={i} className={`alert-item ${getSeverityClass(alert.severity)}`}>
              <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 12 }}>
                <div style={{ flex: 1 }}>
                  <div className="alert-title">
                    <span className={`badge badge-${getSeverityClass(alert.severity)}`} style={{ marginRight: 8 }}>
                      {alert.severity}
                    </span>
                    {alert.fraud_pattern?.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}
                  </div>
                  <div className="alert-desc">{alert.description || alert.explanation}</div>
                  <div className="alert-meta">
                    {alert.beneficiary_card_id && <span>Card: <strong>{alert.beneficiary_card_id}</strong></span>}
                    {alert.fps_shop_id && <span>Shop: <strong>{alert.fps_shop_id}</strong></span>}
                    <span>Score: <strong>{(alert.anomaly_score * 100).toFixed(0)}%</strong></span>
                    {alert.detected_at && <span>{new Date(alert.detected_at).toLocaleString('en-IN')}</span>}
                  </div>
                </div>
                <div style={{ textAlign: 'right', flexShrink: 0 }}>
                  <div style={{ fontSize: 12, color: '#64748b', marginBottom: 4 }}>Action</div>
                  <div style={{ fontSize: 13, color: '#f59e0b', maxWidth: 160 }}>{alert.recommended_action}</div>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
