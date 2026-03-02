/**
 * SimulationPage.jsx — Agent-Based SCM Simulation Dashboard
 *
 * Implements the full research paper framework:
 *   State:     S_t = (I_t, D̂_t, p̂_t, ĉ)
 *   Cost:      C_t = α·C^trans + β·C^stock + γ·C^leak + δ·C^ineq
 *   Policies:  Proportional | LP-Optimised | Equity-First | Risk-Averse (CVaR)
 *
 * Charts (react-chartjs-2):
 *   - Stacked-bar cost breakdown per period
 *   - Service level vs fraud probability (dual-axis line)
 *   - Inventory vs demand (area + line)
 *   - Policy comparison table + radar summary
 *   - Per-district final-state table with heatmap columns
 */

import React, { useState, useCallback } from 'react';
import {
  Chart as ChartJS,
  CategoryScale, LinearScale,
  BarElement, LineElement, PointElement,
  ArcElement,
  Title, Tooltip, Legend, Filler,
} from 'chart.js';
import { Bar, Line } from 'react-chartjs-2';
import {
  runSimulation,
  getSimulationPresets,
  compareSimulationPolicies,
} from '../services/api';

ChartJS.register(
  CategoryScale, LinearScale,
  BarElement, LineElement, PointElement, ArcElement,
  Title, Tooltip, Legend, Filler,
);

// ── Colour palette ─────────────────────────────────────────────────────────
const C = {
  transport: '#3B82F6',
  stockout:  '#EF4444',
  leakage:   '#F59E0B',
  equity:    '#8B5CF6',
  service:   '#10B981',
  fraud:     '#F87171',
  inventory: '#60A5FA',
  demand:    '#34D399',
  total:     '#E2E8F0',
};

const CHART_OPTS = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    legend: { labels: { color: '#94A3B8', font: { size: 11 } } },
    tooltip: { mode: 'index', intersect: false },
  },
  scales: {
    x: { ticks: { color: '#64748B', maxRotation: 0 }, grid: { color: '#1E293B' } },
    y: { ticks: { color: '#64748B' },                 grid: { color: '#1E293B' } },
  },
};

// ── Default config ─────────────────────────────────────────────────────────
const DEFAULT_CFG = {
  n_periods:            24,
  discount_factor:      0.95,
  alpha:                0.25,
  beta:                 0.35,
  gamma:                0.25,
  delta:                0.15,
  policy:               'optimized',
  inspection_effectiveness: 0.40,
  fraud_shock_scale:    0.03,
  supply_fraction:      0.91,
  inspection_cost:      5000,
  budget_per_period:    50000000,
  stockout_penalty:     50,
  min_service_level:    0.70,
  max_allocation_ratio: 2.0,
  inspection_threshold: 0.65,
  cvar_confidence:      0.95,
  seed:                 42,
};

const POLICY_LABELS = {
  proportional:  'Proportional',
  optimized:     'LP Optimised',
  equity_first:  'Equity-First',
  risk_averse:   'Risk-Averse (CVaR)',
};

const POLICY_COLORS = {
  proportional: '#64748B',
  optimized:    '#3B82F6',
  equity_first: '#10B981',
  risk_averse:  '#F59E0B',
};

// ── Helpers ────────────────────────────────────────────────────────────────
const fmt = (n, dec = 0) =>
  n == null ? '—' : Number(n).toLocaleString('en-IN', { maximumFractionDigits: dec });

const pct = (n, dec = 1) => (n == null ? '—' : `${(n * 100).toFixed(dec)}%`);

function Slider({ label, value, min, max, step = 0.01, onChange, tooltip }) {
  return (
    <div style={{ marginBottom: 14 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
        <label style={{ fontSize: 12, color: '#94A3B8' }} title={tooltip}>{label}</label>
        <span style={{ fontSize: 12, color: '#E2E8F0', fontWeight: 600 }}>
          {Number(value).toFixed(step < 1 ? 2 : 0)}
        </span>
      </div>
      <input
        type="range" min={min} max={max} step={step} value={value}
        onChange={e => onChange(Number(e.target.value))}
        style={{ width: '100%', accentColor: '#3B82F6' }}
      />
    </div>
  );
}

function KpiCard({ label, value, sub, color = '#3B82F6', icon }) {
  return (
    <div style={{
      background: '#1E293B', borderRadius: 10, padding: '14px 18px',
      borderLeft: `3px solid ${color}`, flex: 1, minWidth: 140,
    }}>
      <div style={{ fontSize: 20, marginBottom: 4 }}>{icon}</div>
      <div style={{ fontSize: 20, fontWeight: 700, color: '#F1F5F9' }}>{value}</div>
      <div style={{ fontSize: 11, color: '#64748B', marginTop: 2 }}>{label}</div>
      {sub && <div style={{ fontSize: 10, color: color, marginTop: 4 }}>{sub}</div>}
    </div>
  );
}

// ── Main Component ─────────────────────────────────────────────────────────
export default function SimulationPage() {
  const [cfg, setCfg]             = useState({ ...DEFAULT_CFG });
  const [result, setResult]       = useState(null);
  const [comparison, setComparison] = useState(null);
  const [loading, setLoading]     = useState(false);
  const [compLoading, setCompLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('costs');
  const [error, setError]         = useState(null);

  const set = (key) => (val) => setCfg(c => ({ ...c, [key]: val }));

  // Period labels (Month abbreviations)
  const periodLabels = result
    ? Array.from({ length: result.n_periods }, (_, i) => {
        const months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
        return months[(4 + i) % 12] + (i >= 8 ? " '"+ (26 + Math.floor((i+4)/12)) : '');
      })
    : [];

  const handleRun = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await runSimulation(cfg);
      setResult(res);
      setActiveTab('costs');
    } catch (e) {
      setError(e?.response?.data?.detail || e.message || 'Simulation failed');
    } finally {
      setLoading(false);
    }
  }, [cfg]);

  const handleCompare = useCallback(async () => {
    setCompLoading(true);
    setError(null);
    try {
      const res = await compareSimulationPolicies(cfg);
      setComparison(res);
    } catch (e) {
      setError(e?.response?.data?.detail || e.message || 'Comparison failed');
    } finally {
      setCompLoading(false);
    }
  }, [cfg]);

  const loadPreset = async (preset) => {
    setCfg(c => ({ ...c, ...preset }));
  };

  const handlePresetsLoad = async () => {
    try {
      const data = await getSimulationPresets();
      return data.presets;
    } catch { return []; }
  };

  // ── Chart data builders ──────────────────────────────────────────────────

  const costChartData = result ? {
    labels: periodLabels,
    datasets: [
      { label: 'Transport (α)',  data: result.cost_transport_series, backgroundColor: C.transport + 'CC', stack: 'cost' },
      { label: 'Stockout (β)',   data: result.cost_stockout_series,  backgroundColor: C.stockout  + 'CC', stack: 'cost' },
      { label: 'Leakage (γ)',    data: result.cost_leakage_series,   backgroundColor: C.leakage   + 'CC', stack: 'cost' },
      { label: 'Equity (δ)',     data: result.cost_equity_series,    backgroundColor: C.equity    + 'CC', stack: 'cost' },
    ],
  } : null;

  const serviceFraudChartData = result ? {
    labels: periodLabels,
    datasets: [
      {
        label: 'Avg Service Level',
        data: result.service_level_series.map(v => (v * 100).toFixed(2)),
        borderColor: C.service, backgroundColor: C.service + '22',
        yAxisID: 'y', tension: 0.4, fill: true,
      },
      {
        label: 'Avg Fraud Probability ×100',
        data: result.fraud_prob_series.map(v => (v * 100).toFixed(2)),
        borderColor: C.fraud, backgroundColor: 'transparent',
        yAxisID: 'y2', tension: 0.4, borderDash: [4, 2],
      },
    ],
  } : null;

  const inventoryChartData = result ? {
    labels: periodLabels,
    datasets: [
      {
        label: 'Total Inventory (kg)',
        data: result.inventory_total_series,
        borderColor: C.inventory, backgroundColor: C.inventory + '33',
        fill: true, tension: 0.4,
      },
      {
        label: 'Allocation (kg)',
        data: result.allocation_total_series,
        borderColor: C.demand, backgroundColor: 'transparent',
        tension: 0.4, borderDash: [5, 3],
      },
    ],
  } : null;

  const dualAxisOpts = {
    ...CHART_OPTS,
    scales: {
      ...CHART_OPTS.scales,
      y:  { ...CHART_OPTS.scales.y, position: 'left',  title: { display: true, text: 'Service Level (%)', color: '#64748B' } },
      y2: { ...CHART_OPTS.scales.y, position: 'right', title: { display: true, text: 'Fraud Prob ×100', color: '#64748B' }, grid: { drawOnChartArea: false } },
    },
  };

  // ── Comparison chart ─────────────────────────────────────────────────────
  const compChartData = comparison ? {
    labels: periodLabels,
    datasets: Object.entries(comparison.comparison).map(([policy, data]) => ({
      label: POLICY_LABELS[policy] || policy,
      data: data.cost_series,
      borderColor: POLICY_COLORS[policy],
      backgroundColor: 'transparent',
      tension: 0.4,
      borderWidth: 2,
    })),
  } : null;

  const compServiceData = comparison ? {
    labels: periodLabels,
    datasets: Object.entries(comparison.comparison).map(([policy, data]) => ({
      label: POLICY_LABELS[policy] || policy,
      data: data.service_level_series.map(v => (v * 100).toFixed(2)),
      borderColor: POLICY_COLORS[policy],
      backgroundColor: 'transparent',
      tension: 0.4,
      borderWidth: 2,
    })),
  } : null;

  // ── Per-district heatmap colour ──────────────────────────────────────────
  const heatColor = (val, min, max, good = 'high') => {
    if (max === min) return '#1E293B';
    const t = (val - min) / (max - min);
    const r = good === 'high' ? t : 1 - t;
    const g = good === 'high' ? 1 - t : t;
    return `rgba(${Math.round((1 - r) * 59 + r * 16)}, ${Math.round((1 - g) * 130 + g * 27)}, 60, 0.85)`;
  };

  const districtRows = result?.district_final_state || [];
  const svcVals  = districtRows.map(d => d.avg_service_ratio);
  const fraudVals = districtRows.map(d => d.final_fraud_prob);
  const leakVals  = districtRows.map(d => d.total_leakage_kg);
  const svcMin = Math.min(...svcVals),  svcMax  = Math.max(...svcVals);
  const frMin  = Math.min(...fraudVals), frMax   = Math.max(...fraudVals);
  const lkMin  = Math.min(...leakVals),  lkMax   = Math.max(...leakVals);

  // ── Render ───────────────────────────────────────────────────────────────
  return (
    <div>
      {/* Header */}
      <div className="page-header">
        <div className="page-title">⚙️ Agent-Based SCM Simulation</div>
        <div className="page-subtitle">
          Multi-period stochastic optimisation — S_t = (I_t, D̂_t, p̂_t, ĉ) · N=33 districts · T up to 60 periods
        </div>
      </div>

      {error && (
        <div className="card" style={{ borderColor: '#EF4444', background: 'rgba(239,68,68,0.08)', marginBottom: 16 }}>
          <p style={{ color: '#EF4444', margin: 0 }}>⚠ {error}</p>
        </div>
      )}

      <div style={{ display: 'flex', gap: 16, alignItems: 'flex-start' }}>

        {/* ── Left Panel: Configuration ──────────────────────────────────── */}
        <div style={{ width: 280, flexShrink: 0 }}>
          <div className="card" style={{ padding: '16px 18px' }}>
            <div className="card-title" style={{ marginBottom: 16, fontSize: 13 }}>
              Simulation Configuration
            </div>

            {/* Period & Discount */}
            <Slider label={`Horizon T = ${cfg.n_periods} months`}
              value={cfg.n_periods} min={6} max={60} step={1}
              onChange={set('n_periods')}
              tooltip="Number of simulation periods (months)" />
            <Slider label={`Discount λ`}
              value={cfg.discount_factor} min={0.7} max={1.0} step={0.01}
              onChange={set('discount_factor')}
              tooltip="Future cost discount factor" />

            {/* Divider */}
            <div style={{ borderTop: '1px solid #334155', margin: '10px 0' }} />
            <div style={{ fontSize: 11, color: '#64748B', marginBottom: 10 }}>
              COST WEIGHTS (α+β+γ+δ must ≈ 1)
            </div>

            <Slider label={`α Transport`}
              value={cfg.alpha} min={0} max={1} step={0.05}
              onChange={set('alpha')}
              tooltip="Weight on transport cost C^trans" />
            <Slider label={`β Stockout`}
              value={cfg.beta} min={0} max={1} step={0.05}
              onChange={set('beta')}
              tooltip="Weight on stockout penalty C^stock" />
            <Slider label={`γ Leakage`}
              value={cfg.gamma} min={0} max={1} step={0.05}
              onChange={set('gamma')}
              tooltip="Weight on leakage cost C^leak" />
            <Slider label={`δ Equity`}
              value={cfg.delta} min={0} max={1} step={0.05}
              onChange={set('delta')}
              tooltip="Weight on service equity C^ineq" />

            {/* Divider */}
            <div style={{ borderTop: '1px solid #334155', margin: '10px 0' }} />
            <div style={{ fontSize: 11, color: '#64748B', marginBottom: 10 }}>FRAUD DYNAMICS</div>

            <Slider label={`η Inspection Effectiveness`}
              value={cfg.inspection_effectiveness} min={0.05} max={0.95} step={0.05}
              onChange={set('inspection_effectiveness')}
              tooltip="Fraction fraud prob drops per inspection" />
            <Slider label={`ε Fraud Shock Scale`}
              value={cfg.fraud_shock_scale} min={0.0} max={0.15} step={0.01}
              onChange={set('fraud_shock_scale')}
              tooltip="Exogenous fraud probability shocks" />
            <Slider label={`Inspect Threshold`}
              value={cfg.inspection_threshold} min={0.2} max={0.95} step={0.05}
              onChange={set('inspection_threshold')}
              tooltip="Always inspect if fraud prob exceeds this" />

            {/* Divider */}
            <div style={{ borderTop: '1px solid #334155', margin: '10px 0' }} />
            <div style={{ fontSize: 11, color: '#64748B', marginBottom: 10 }}>SUPPLY & RISK</div>

            <Slider label={`Supply Fraction`}
              value={cfg.supply_fraction} min={0.5} max={1.0} step={0.01}
              onChange={set('supply_fraction')}
              tooltip="Fraction of total demand available as supply" />
            <Slider label={`CVaR Confidence`}
              value={cfg.cvar_confidence} min={0.5} max={0.999} step={0.01}
              onChange={set('cvar_confidence')}
              tooltip="Confidence level for CVaR risk measure" />

            {/* Policy selector */}
            <div style={{ borderTop: '1px solid #334155', margin: '10px 0' }} />
            <div style={{ fontSize: 11, color: '#64748B', marginBottom: 8 }}>ALLOCATION POLICY</div>
            {Object.entries(POLICY_LABELS).map(([key, label]) => (
              <label key={key} style={{
                display: 'flex', alignItems: 'center', gap: 8,
                marginBottom: 7, cursor: 'pointer',
                color: cfg.policy === key ? '#F1F5F9' : '#64748B',
                fontSize: 12,
              }}>
                <input type="radio" name="policy" value={key}
                  checked={cfg.policy === key}
                  onChange={() => set('policy')(key)}
                  style={{ accentColor: POLICY_COLORS[key] }} />
                <span style={{
                  padding: '2px 8px', borderRadius: 10,
                  background: cfg.policy === key ? POLICY_COLORS[key] + '33' : 'transparent',
                  border: `1px solid ${cfg.policy === key ? POLICY_COLORS[key] : '#334155'}`,
                }}>{label}</span>
              </label>
            ))}

            {/* Action Buttons */}
            <div style={{ marginTop: 16, display: 'flex', flexDirection: 'column', gap: 8 }}>
              <button className="btn btn-primary" onClick={handleRun} disabled={loading}
                style={{ width: '100%', fontSize: 13, fontWeight: 700 }}>
                {loading ? '⏳ Running…' : '▶ Run Simulation'}
              </button>
              <button
                onClick={handleCompare} disabled={compLoading}
                style={{
                  width: '100%', padding: '8px 12px', fontSize: 12,
                  background: '#1E293B', color: '#94A3B8',
                  border: '1px solid #334155', borderRadius: 8, cursor: 'pointer',
                }}>
                {compLoading ? '⏳ Comparing…' : '⚖ Compare All Policies'}
              </button>
            </div>

            {/* Quick Presets */}
            <div style={{ borderTop: '1px solid #334155', margin: '14px 0 10px' }} />
            <div style={{ fontSize: 11, color: '#64748B', marginBottom: 8 }}>QUICK PRESETS</div>
            {[
              { label: 'Balanced',      cfg: { alpha: 0.25, beta: 0.35, gamma: 0.25, delta: 0.15, policy: 'optimized' } },
              { label: 'Equity-First',  cfg: { alpha: 0.10, beta: 0.30, gamma: 0.15, delta: 0.45, policy: 'equity_first' } },
              { label: 'Fraud Control', cfg: { alpha: 0.15, beta: 0.25, gamma: 0.50, delta: 0.10, policy: 'risk_averse', inspection_threshold: 0.40 } },
              { label: 'Cost Optimal',  cfg: { alpha: 0.50, beta: 0.30, gamma: 0.15, delta: 0.05, policy: 'optimized' } },
            ].map(p => (
              <button key={p.label}
                onClick={() => loadPreset(p.cfg)}
                style={{
                  display: 'block', width: '100%', padding: '5px 10px',
                  marginBottom: 5, fontSize: 11, background: '#0F172A',
                  color: '#94A3B8', border: '1px solid #334155',
                  borderRadius: 6, cursor: 'pointer', textAlign: 'left',
                }}>
                {p.label}
              </button>
            ))}
          </div>
        </div>

        {/* ── Right Panel: Results ───────────────────────────────────────── */}
        <div style={{ flex: 1, minWidth: 0 }}>

          {!result && !comparison && (
            <div className="card" style={{ textAlign: 'center', padding: '64px 32px' }}>
              <div style={{ fontSize: 48, marginBottom: 16 }}>⚙️</div>
              <div style={{ color: '#94A3B8', fontSize: 15, marginBottom: 8 }}>
                Configure parameters and click <strong style={{ color: '#3B82F6' }}>▶ Run Simulation</strong>
              </div>
              <div style={{ color: '#475569', fontSize: 13, maxWidth: 500, margin: '0 auto' }}>
                The simulation models 33 Telangana districts over up to 60 months using the
                Agent-Based SCM framework: demand forecasting, fraud detection, geospatial
                cost optimisation, and multi-objective allocation with CVaR risk measure.
              </div>
              <div style={{ marginTop: 24, display: 'flex', gap: 16, justifyContent: 'center', flexWrap: 'wrap' }}>
                {[
                  { icon: '📦', label: 'Inventory Dynamics',  desc: 'I_{t+1} = I_t + x_t − D_t − L_t' },
                  { icon: '🔍', label: 'Fraud Update',        desc: 'p_{t+1} = p_t(1−η·y_t) + ε_t' },
                  { icon: '💰', label: 'Multi-Obj Cost',      desc: 'C_t = α·trans + β·stock + γ·leak + δ·ineq' },
                  { icon: '📊', label: 'CVaR Risk Measure',   desc: 'CVaR_α(Σ λ^t · C_t)' },
                ].map(item => (
                  <div key={item.label} style={{
                    background: '#0F172A', borderRadius: 8, padding: '12px 16px',
                    border: '1px solid #1E293B', textAlign: 'left', width: 200,
                  }}>
                    <div style={{ fontSize: 22, marginBottom: 6 }}>{item.icon}</div>
                    <div style={{ color: '#CBD5E1', fontSize: 12, fontWeight: 600, marginBottom: 4 }}>{item.label}</div>
                    <div style={{ color: '#475569', fontSize: 10, fontFamily: 'monospace' }}>{item.desc}</div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* ── KPI Summary Cards ─────────────────────────────────────────── */}
          {result && (
            <>
              <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 16 }}>
                <KpiCard icon="💰" label="Total Discounted Cost"
                  value={`₹${fmt(result.total_discounted_cost / 1e6, 1)}M`}
                  sub={`CVaR: ₹${fmt(result.cvar_cost / 1e6, 1)}M`}
                  color={C.total} />
                <KpiCard icon="📦" label="Avg Service Level"
                  value={pct(result.avg_service_level)}
                  sub={`${fmt(result.total_stockout_kg / 1e3, 0)}T stockout`}
                  color={C.service} />
                <KpiCard icon="🔍" label="Avg Fraud Probability"
                  value={pct(result.avg_fraud_prob)}
                  sub={`${fmt(result.total_leakage_kg / 1e3, 0)}T leaked`}
                  color={C.leakage} />
                <KpiCard icon="⚙️" label="Policy Used"
                  value={POLICY_LABELS[result.policy] || result.policy}
                  sub={`${result.n_periods} periods · ${result.n_districts} districts`}
                  color={POLICY_COLORS[result.policy] || '#3B82F6'} />
                <KpiCard icon="⏱️" label="Runtime"
                  value={`${result.runtime_seconds}s`}
                  sub="full simulation time"
                  color="#6366F1" />
              </div>

              {/* ── Chart Tabs ───────────────────────────────────────────── */}
              <div className="card">
                <div style={{ display: 'flex', gap: 8, marginBottom: 16, flexWrap: 'wrap' }}>
                  {[
                    ['costs',     '💰 Cost Breakdown'],
                    ['service',   '📈 Service & Fraud'],
                    ['inventory', '📦 Inventory'],
                    ['districts', '🗺️ Districts'],
                    ['log',       '🤖 Agent Log'],
                  ].map(([tab, label]) => (
                    <button key={tab} onClick={() => setActiveTab(tab)}
                      style={{
                        padding: '6px 14px', borderRadius: 6, fontSize: 12,
                        background: activeTab === tab ? '#3B82F6' : '#0F172A',
                        color: activeTab === tab ? '#FFF' : '#94A3B8',
                        border: `1px solid ${activeTab === tab ? '#3B82F6' : '#334155'}`,
                        cursor: 'pointer',
                      }}>
                      {label}
                    </button>
                  ))}
                </div>

                {/* Cost Breakdown */}
                {activeTab === 'costs' && costChartData && (
                  <div>
                    <div style={{ fontSize: 12, color: '#64748B', marginBottom: 12 }}>
                      C_t = α·C^trans + β·C^stock + γ·C^leak + δ·C^ineq per period
                    </div>
                    <div style={{ height: 320 }}>
                      <Bar data={costChartData} options={{ ...CHART_OPTS, plugins: { ...CHART_OPTS.plugins, legend: { labels: { color: '#94A3B8' } } } }} />
                    </div>
                    <div style={{ marginTop: 16, display: 'flex', gap: 24, fontSize: 12, color: '#94A3B8', flexWrap: 'wrap' }}>
                      {[
                        ['Transport', result.cost_transport_series.reduce((a, b) => a + b, 0), C.transport],
                        ['Stockout',  result.cost_stockout_series.reduce((a, b) => a + b, 0),  C.stockout],
                        ['Leakage',   result.cost_leakage_series.reduce((a, b) => a + b, 0),   C.leakage],
                        ['Equity',    result.cost_equity_series.reduce((a, b) => a + b, 0),    C.equity],
                      ].map(([name, total, col]) => (
                        <div key={name} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                          <span style={{ width: 10, height: 10, borderRadius: 2, background: col, display: 'inline-block' }} />
                          <span>{name}: <strong style={{ color: '#E2E8F0' }}>₹{fmt(total / 1e6, 1)}M total</strong></span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Service & Fraud */}
                {activeTab === 'service' && serviceFraudChartData && (
                  <div>
                    <div style={{ fontSize: 12, color: '#64748B', marginBottom: 12 }}>
                      Service Level (%) and Fraud Probability ×100 over simulation horizon
                    </div>
                    <div style={{ height: 320 }}>
                      <Line data={serviceFraudChartData} options={dualAxisOpts} />
                    </div>
                    <div style={{ marginTop: 16, display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12 }}>
                      {[
                        { label: 'Min Service',  value: pct(Math.min(...result.service_level_series)), color: C.stockout },
                        { label: 'Avg Service',  value: pct(result.avg_service_level),                 color: C.service  },
                        { label: 'Max Service',  value: pct(Math.max(...result.service_level_series)), color: C.demand   },
                        { label: 'Min Fraud',    value: pct(Math.min(...result.fraud_prob_series)),    color: C.service  },
                        { label: 'Avg Fraud',    value: pct(result.avg_fraud_prob),                   color: C.leakage  },
                        { label: 'Max Fraud',    value: pct(Math.max(...result.fraud_prob_series)),    color: C.stockout },
                      ].map(item => (
                        <div key={item.label} style={{
                          background: '#0F172A', borderRadius: 6, padding: '10px 12px',
                          border: '1px solid #1E293B',
                        }}>
                          <div style={{ fontSize: 11, color: '#64748B' }}>{item.label}</div>
                          <div style={{ fontSize: 18, fontWeight: 700, color: item.color }}>{item.value}</div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Inventory */}
                {activeTab === 'inventory' && inventoryChartData && (
                  <div>
                    <div style={{ fontSize: 12, color: '#64748B', marginBottom: 12 }}>
                      Total inventory I_t (kg) and allocation x_t (kg) across all 33 districts
                    </div>
                    <div style={{ height: 320 }}>
                      <Line data={inventoryChartData} options={CHART_OPTS} />
                    </div>
                    <div style={{ marginTop: 16, display: 'flex', gap: 24, fontSize: 12, color: '#94A3B8', flexWrap: 'wrap' }}>
                      <span>Total leakage: <strong style={{ color: C.leakage }}>
                        {fmt(result.total_leakage_kg / 1e6, 2)}M kg</strong></span>
                      <span>Total stockout: <strong style={{ color: C.stockout }}>
                        {fmt(result.total_stockout_kg / 1e6, 2)}M kg</strong></span>
                      <span>Avg allocation/period: <strong style={{ color: C.service }}>
                        {fmt(result.allocation_total_series.reduce((a, b) => a + b, 0) / result.n_periods / 1e6, 2)}M kg</strong></span>
                    </div>
                  </div>
                )}

                {/* District Table */}
                {activeTab === 'districts' && districtRows.length > 0 && (
                  <div>
                    <div style={{ fontSize: 12, color: '#64748B', marginBottom: 12 }}>
                      Per-district final state — colour-coded heatmap (green=better)
                    </div>
                    <div className="table-container" style={{ maxHeight: 480, overflowY: 'auto' }}>
                      <table>
                        <thead>
                          <tr>
                            <th>District</th>
                            <th>Shops</th>
                            <th>Beneficiaries</th>
                            <th>Avg Service</th>
                            <th>Final Fraud Prob</th>
                            <th>Total Leakage (kg)</th>
                            <th>Inspected Periods</th>
                            <th>Final Inventory (kg)</th>
                          </tr>
                        </thead>
                        <tbody>
                          {districtRows.map((d, i) => (
                            <tr key={d.id}>
                              <td style={{ fontWeight: 600, color: '#CBD5E1' }}>{d.name}</td>
                              <td style={{ color: '#64748B' }}>{fmt(d.shops)}</td>
                              <td style={{ color: '#64748B' }}>{fmt(d.beneficiaries)}</td>
                              <td style={{
                                fontWeight: 700,
                                background: heatColor(d.avg_service_ratio, svcMin, svcMax, 'high') + '44',
                                color: d.avg_service_ratio > 0.85 ? '#10B981' : d.avg_service_ratio > 0.70 ? '#F59E0B' : '#EF4444',
                              }}>{pct(d.avg_service_ratio)}</td>
                              <td style={{
                                background: heatColor(d.final_fraud_prob, frMin, frMax, 'low') + '44',
                                color: d.final_fraud_prob > 0.2 ? '#EF4444' : d.final_fraud_prob > 0.1 ? '#F59E0B' : '#10B981',
                              }}>{pct(d.final_fraud_prob, 2)}</td>
                              <td style={{ color: '#F59E0B' }}>{fmt(d.total_leakage_kg)}</td>
                              <td style={{ color: '#8B5CF6' }}>{d.inspected_periods}</td>
                              <td style={{ color: '#60A5FA' }}>{fmt(d.final_inventory_kg)}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}

                {/* Agent Log */}
                {activeTab === 'log' && result.periods.length > 0 && (
                  <div>
                    <div style={{ fontSize: 12, color: '#64748B', marginBottom: 12 }}>
                      Agent decision log — what each agent did each period
                    </div>
                    <div style={{ maxHeight: 500, overflowY: 'auto' }}>
                      {result.periods.map((pr, i) => {
                        const acts = pr.agent_actions || {};
                        return (
                          <div key={i} style={{
                            marginBottom: 10, padding: '10px 14px',
                            background: '#0F172A', borderRadius: 8,
                            border: '1px solid #1E293B',
                          }}>
                            <div style={{ display: 'flex', gap: 12, alignItems: 'center', marginBottom: 8 }}>
                              <span style={{
                                background: '#1E3A5F', color: '#60A5FA',
                                padding: '2px 8px', borderRadius: 10, fontSize: 11, fontWeight: 700,
                              }}>Period {i + 1}</span>
                              <span style={{ fontSize: 11, color: '#64748B' }}>
                                Cost: ₹{fmt(pr.cost_total / 1e3, 0)}K · Service: {pct(pr.avg_service_ratio)} · Fraud: {pct(pr.avg_fraud_prob)}
                              </span>
                              <span style={{ fontSize: 11, color: '#8B5CF6' }}>
                                {pr.n_inspections} inspections · {pr.n_stockouts} stockout districts
                              </span>
                            </div>
                            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 8 }}>
                              {[
                                { icon: '📊', name: 'Demand Agent',      data: acts.demand_agent },
                                { icon: '🔍', name: 'Fraud Agent',       data: acts.fraud_agent },
                                { icon: '🗺️', name: 'Geo Agent',         data: acts.geo_agent },
                                { icon: '⚖️', name: 'Allocation Agent',  data: acts.allocation_agent },
                                { icon: '🏛️', name: 'Governance Agent',  data: acts.governance_agent },
                              ].map(({ icon, name, data }) => data && (
                                <div key={name} style={{
                                  background: '#1E293B', borderRadius: 6, padding: '6px 10px',
                                  fontSize: 10, color: '#94A3B8',
                                }}>
                                  <div style={{ color: '#CBD5E1', fontWeight: 600, marginBottom: 4 }}>
                                    {icon} {name}
                                  </div>
                                  {Object.entries(data).filter(([k]) => k !== 'action').map(([k, v]) => (
                                    <div key={k}>
                                      <span style={{ color: '#475569' }}>{k.replace(/_/g, ' ')}: </span>
                                      <span style={{ color: '#94A3B8' }}>
                                        {typeof v === 'number' ? (v > 1e5 ? `${(v / 1e6).toFixed(2)}M` : v.toFixed(2)) : String(v)}
                                      </span>
                                    </div>
                                  ))}
                                </div>
                              ))}
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}
              </div>
            </>
          )}

          {/* ── Policy Comparison ─────────────────────────────────────────── */}
          {comparison && (
            <div className="card" style={{ marginTop: 16 }}>
              <div className="card-title" style={{ marginBottom: 16 }}>
                ⚖️ Policy Comparison — All 4 Allocation Strategies
              </div>

              {/* Summary Table */}
              <div className="table-container" style={{ marginBottom: 20 }}>
                <table>
                  <thead>
                    <tr>
                      <th>Policy</th>
                      <th>Discounted Cost</th>
                      <th>CVaR (95%)</th>
                      <th>Avg Service Level</th>
                      <th>Avg Fraud Prob</th>
                      <th>Total Leakage (kg)</th>
                      <th>Total Stockout (kg)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(comparison.comparison).map(([policy, data]) => {
                      const bestCost = Math.min(...Object.values(comparison.comparison).map(d => d.total_discounted_cost));
                      const bestSvc  = Math.max(...Object.values(comparison.comparison).map(d => d.avg_service_level));
                      const isBestCost = data.total_discounted_cost === bestCost;
                      const isBestSvc  = data.avg_service_level === bestSvc;
                      return (
                        <tr key={policy}>
                          <td>
                            <span style={{
                              padding: '2px 8px', borderRadius: 10,
                              background: POLICY_COLORS[policy] + '33',
                              color: POLICY_COLORS[policy], fontSize: 12, fontWeight: 600,
                            }}>
                              {POLICY_LABELS[policy]}
                            </span>
                          </td>
                          <td style={{ color: isBestCost ? '#10B981' : '#E2E8F0', fontWeight: isBestCost ? 700 : 400 }}>
                            ₹{fmt(data.total_discounted_cost / 1e6, 1)}M {isBestCost && '✓ Best'}
                          </td>
                          <td style={{ color: '#F59E0B' }}>
                            ₹{fmt(data.cvar_cost / 1e6, 1)}M
                          </td>
                          <td style={{ color: isBestSvc ? '#10B981' : '#E2E8F0', fontWeight: isBestSvc ? 700 : 400 }}>
                            {pct(data.avg_service_level)} {isBestSvc && '✓ Best'}
                          </td>
                          <td style={{ color: '#F87171' }}>{pct(data.avg_fraud_prob)}</td>
                          <td style={{ color: '#F59E0B' }}>{fmt(data.total_leakage_kg / 1e3, 0)}T</td>
                          <td style={{ color: '#EF4444' }}>{fmt(data.total_stockout_kg / 1e3, 0)}T</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>

              {/* Comparison Charts */}
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
                <div>
                  <div style={{ fontSize: 12, color: '#64748B', marginBottom: 8 }}>Period Cost by Policy</div>
                  <div style={{ height: 240 }}>
                    {compChartData && <Line data={compChartData} options={CHART_OPTS} />}
                  </div>
                </div>
                <div>
                  <div style={{ fontSize: 12, color: '#64748B', marginBottom: 8 }}>Service Level by Policy (%)</div>
                  <div style={{ height: 240 }}>
                    {compServiceData && <Line data={compServiceData} options={CHART_OPTS} />}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* ── Paper Reference ───────────────────────────────────────────── */}
          <div className="card" style={{ marginTop: 16, padding: '14px 18px' }}>
            <div style={{ display: 'flex', gap: 24, flexWrap: 'wrap', fontSize: 11, color: '#475569' }}>
              <div>
                <strong style={{ color: '#64748B' }}>State Space:</strong>{' '}
                S_t = (I_t, D̂_t, p̂_t, ĉ)
              </div>
              <div>
                <strong style={{ color: '#64748B' }}>Inventory:</strong>{' '}
                I_&#123;t+1&#125; = max(0, I_t + x_t − D_t − p_t·x_t)
              </div>
              <div>
                <strong style={{ color: '#64748B' }}>Fraud:</strong>{' '}
                p_&#123;t+1&#125; = p_t·(1−η·y_t) + ε_t
              </div>
              <div>
                <strong style={{ color: '#64748B' }}>Cost:</strong>{' '}
                C_t = α·C^trans + β·C^stock + γ·C^leak + δ·Var(S_i)
              </div>
              <div>
                <strong style={{ color: '#64748B' }}>Objective:</strong>{' '}
                min CVaR_α(Σ λ^t · C_t)
              </div>
              <div>
                <strong style={{ color: '#64748B' }}>N:</strong> 33 districts ·{' '}
                <strong style={{ color: '#64748B' }}>T:</strong> up to 60 periods
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
