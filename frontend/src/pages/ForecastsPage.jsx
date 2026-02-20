import React, { useState } from 'react';
import { getForecasts } from '../services/api';

const DISTRICTS = [
  'Hyderabad', 'Nizamabad', 'Warangal', 'Khammam',
  'Nalgonda', 'Karimnagar', 'Mahbubnagar', 'Adilabad', 'Rangareddy', 'Medak',
];
const COMMODITIES = ['rice', 'wheat', 'kerosene', 'sugar'];

export default function ForecastsPage() {
  const [district, setDistrict] = useState('');
  const [commodity, setCommodity] = useState('rice');
  const [monthsAhead, setMonthsAhead] = useState(3);
  const [forecasts, setForecasts] = useState([]);
  const [riskFlags, setRiskFlags] = useState([]);
  const [loading, setLoading] = useState(false);
  const [ran, setRan] = useState(false);

  async function runForecast() {
    setLoading(true);
    try {
      const body = { commodity, months_ahead: monthsAhead };
      if (district) body.district = district;
      const data = await getForecasts(body);
      setForecasts(data.forecasts || []);
      setRiskFlags((data.forecasts || []).filter(f => f.risk_flag));
    } catch {
      setForecasts([]);
    } finally {
      setLoading(false);
      setRan(true);
    }
  }

  const riskColor = (flag) => flag === 'understock_risk' ? '#ef4444' : '#f59e0b';

  return (
    <div>
      <div className="page-header">
        <div className="page-title">📈 Demand Forecasts</div>
        <div className="page-subtitle">LSTM + Prophet ensemble — 90-day lookahead per FPS shop</div>
      </div>

      {/* Filters */}
      <div className="card">
        <div className="card-title" style={{ marginBottom: 16 }}>Configure Forecast</div>
        <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', alignItems: 'flex-end' }}>
          <div>
            <label style={{ fontSize: 12, color: '#64748b', display: 'block', marginBottom: 4 }}>District</label>
            <select
              value={district}
              onChange={e => setDistrict(e.target.value)}
              style={{ background: '#0f172a', border: '1px solid #334155', color: '#e2e8f0', padding: '8px 12px', borderRadius: 8 }}
            >
              <option value="">All Districts</option>
              {DISTRICTS.map(d => <option key={d} value={d}>{d}</option>)}
            </select>
          </div>
          <div>
            <label style={{ fontSize: 12, color: '#64748b', display: 'block', marginBottom: 4 }}>Commodity</label>
            <select
              value={commodity}
              onChange={e => setCommodity(e.target.value)}
              style={{ background: '#0f172a', border: '1px solid #334155', color: '#e2e8f0', padding: '8px 12px', borderRadius: 8 }}
            >
              {COMMODITIES.map(c => <option key={c} value={c}>{c.charAt(0).toUpperCase() + c.slice(1)}</option>)}
            </select>
          </div>
          <div>
            <label style={{ fontSize: 12, color: '#64748b', display: 'block', marginBottom: 4 }}>Months Ahead</label>
            <select
              value={monthsAhead}
              onChange={e => setMonthsAhead(Number(e.target.value))}
              style={{ background: '#0f172a', border: '1px solid #334155', color: '#e2e8f0', padding: '8px 12px', borderRadius: 8 }}
            >
              {[1,2,3,6].map(m => <option key={m} value={m}>{m} month{m > 1 ? 's' : ''}</option>)}
            </select>
          </div>
          <button className="btn btn-primary" onClick={runForecast} disabled={loading}>
            {loading ? 'Running…' : '▶ Run Forecast'}
          </button>
        </div>
      </div>

      {/* Risk Flags */}
      {riskFlags.length > 0 && (
        <div className="card" style={{ borderColor: '#f59e0b' }}>
          <div className="card-title" style={{ marginBottom: 12 }}>⚠️ Stock Risk Flags ({riskFlags.length})</div>
          {riskFlags.slice(0, 10).map((f, i) => (
            <div key={i} style={{ display: 'flex', justifyContent: 'space-between', padding: '8px 0', borderBottom: '1px solid #1e293b', fontSize: 13 }}>
              <span>{f.shop_name} — {f.district}</span>
              <span style={{ color: riskColor(f.risk_flag) }}>{f.risk_flag?.replace(/_/g, ' ')}</span>
              <span style={{ color: '#64748b' }}>{f.forecast_month}</span>
              <span style={{ color: '#10b981' }}>{f.predicted_quantity_kg} kg</span>
            </div>
          ))}
        </div>
      )}

      {/* Results Table */}
      {ran && (
        <div className="card">
          <div className="card-title" style={{ marginBottom: 16 }}>
            Forecast Results — {forecasts.length} entries
          </div>
          {loading ? (
            <div className="loading"><div className="spinner" /></div>
          ) : forecasts.length === 0 ? (
            <p style={{ color: '#64748b', textAlign: 'center', padding: 32 }}>No forecasts generated. Try different parameters.</p>
          ) : (
            <div className="table-container">
              <table>
                <thead>
                  <tr>
                    <th>Shop</th>
                    <th>District</th>
                    <th>Month</th>
                    <th>Predicted (kg)</th>
                    <th>CI Lower</th>
                    <th>CI Upper</th>
                    <th>Model</th>
                    <th>Risk</th>
                  </tr>
                </thead>
                <tbody>
                  {forecasts.slice(0, 50).map((f, i) => (
                    <tr key={i}>
                      <td style={{ maxWidth: 160, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{f.shop_name}</td>
                      <td>{f.district}</td>
                      <td>{f.forecast_month}</td>
                      <td style={{ color: '#10b981', fontWeight: 700 }}>{f.predicted_quantity_kg}</td>
                      <td style={{ color: '#64748b' }}>{f.confidence_lower}</td>
                      <td style={{ color: '#64748b' }}>{f.confidence_upper}</td>
                      <td><span className="badge badge-medium">{f.model_used}</span></td>
                      <td>
                        {f.risk_flag
                          ? <span style={{ color: riskColor(f.risk_flag), fontSize: 12 }}>{f.risk_flag.replace(/_/g, ' ')}</span>
                          : <span style={{ color: '#10b981', fontSize: 12 }}>✓ OK</span>
                        }
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
