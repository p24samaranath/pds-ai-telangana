/**
 * DataVizPage — visualises real Telangana PDS data fetched from the backend.
 * Charts: district volumes (bar), commodity breakdown (pie/doughnut),
 *         monthly trend (line), shop health (doughnut), card types (pie),
 *         top districts by beneficiaries (horizontal bar).
 */
import React, { useEffect, useState, useCallback } from 'react';
import {
  Chart as ChartJS,
  CategoryScale, LinearScale, BarElement, LineElement,
  PointElement, ArcElement, Title, Tooltip, Legend, Filler,
} from 'chart.js';
import { Bar, Line, Doughnut, Pie } from 'react-chartjs-2';
import { getVisualizationData, fetchLatestData, getDataStatus } from '../services/api';

ChartJS.register(
  CategoryScale, LinearScale, BarElement, LineElement,
  PointElement, ArcElement, Title, Tooltip, Legend, Filler
);

// ── Colour palettes ──────────────────────────────────────────────────────────
const PALETTE = [
  '#3B82F6','#10B981','#F59E0B','#EF4444','#8B5CF6',
  '#06B6D4','#F97316','#84CC16','#EC4899','#14B8A6',
  '#6366F1','#A3E635','#FB923C','#38BDF8','#E879F9',
];

const COMMODITY_COLORS = {
  rice:          '#10B981',
  wheat:         '#F59E0B',
  sugar:         '#3B82F6',
  kerosene:      '#EF4444',
  red_gram_dal:  '#8B5CF6',
  salt:          '#06B6D4',
};

// ── Helpers ──────────────────────────────────────────────────────────────────
const fmt = (n) => (n >= 1_000_000 ? `${(n/1_000_000).toFixed(1)}M` : n >= 1_000 ? `${(n/1_000).toFixed(1)}K` : String(n));

const KpiCard = ({ label, value, sub, color = '#3B82F6', icon }) => (
  <div style={{
    background: '#1E293B', borderRadius: 12, padding: '20px 24px',
    borderLeft: `4px solid ${color}`, display: 'flex', flexDirection: 'column', gap: 4,
  }}>
    <div style={{ fontSize: 13, color: '#94A3B8', display: 'flex', alignItems: 'center', gap: 6 }}>
      {icon && <span>{icon}</span>}{label}
    </div>
    <div style={{ fontSize: 28, fontWeight: 700, color: '#F1F5F9' }}>{value}</div>
    {sub && <div style={{ fontSize: 12, color: '#64748B' }}>{sub}</div>}
  </div>
);

const Section = ({ title, children }) => (
  <div style={{ background: '#1E293B', borderRadius: 12, padding: 24, marginBottom: 24 }}>
    <h3 style={{ margin: '0 0 18px', color: '#F1F5F9', fontSize: 15, fontWeight: 600, letterSpacing: 0.5 }}>
      {title}
    </h3>
    {children}
  </div>
);

const chartDefaults = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    legend: { labels: { color: '#CBD5E1', font: { size: 12 } } },
    tooltip: { bodyColor: '#F1F5F9', titleColor: '#94A3B8', backgroundColor: '#0F172A' },
  },
  scales: {
    x: { ticks: { color: '#94A3B8', font: { size: 11 } }, grid: { color: '#1E3A5F22' } },
    y: { ticks: { color: '#94A3B8', font: { size: 11 } }, grid: { color: '#1E3A5F44' } },
  },
};

const noAxes = { ...chartDefaults, scales: undefined };

// ── Main component ───────────────────────────────────────────────────────────
export default function DataVizPage() {
  const [data, setData]           = useState(null);
  const [loading, setLoading]     = useState(true);
  const [fetching, setFetching]   = useState(false);
  const [error, setError]         = useState(null);
  const [dataStatus, setStatus]   = useState(null);
  const [lastRefresh, setLast]    = useState(null);

  const load = useCallback(async () => {
    setLoading(true); setError(null);
    try {
      const [viz, status] = await Promise.all([getVisualizationData(), getDataStatus()]);
      if (viz.error) throw new Error(viz.error);
      setData(viz);
      setStatus(status);
      setLast(new Date().toLocaleTimeString());
    } catch (e) {
      setError(e.message || 'Failed to load visualisation data');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  const handleFetchLatest = async () => {
    setFetching(true);
    try {
      await fetchLatestData();
      await load();
    } catch (e) {
      setError('Failed to fetch latest data: ' + (e.message || ''));
    } finally {
      setFetching(false);
    }
  };

  // ── Loading / error states ───────────────────────────────────────────────
  if (loading) return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '60vh', gap: 16, color: '#94A3B8' }}>
      <div style={{ fontSize: 40 }}>📊</div>
      <div style={{ fontSize: 16 }}>Loading live data visualisation…</div>
      <div style={{ width: 200, height: 4, background: '#1E293B', borderRadius: 2, overflow: 'hidden' }}>
        <div style={{ height: '100%', background: '#3B82F6', animation: 'pulse 1.5s infinite', width: '60%' }} />
      </div>
    </div>
  );

  if (error) return (
    <div style={{ padding: 32, color: '#EF4444', background: '#1E293B', borderRadius: 12, margin: 24 }}>
      <strong>⚠ Error:</strong> {error}
      <button onClick={load} style={{ marginLeft: 16, padding: '6px 16px', background: '#3B82F6', color: '#fff', border: 'none', borderRadius: 6, cursor: 'pointer' }}>Retry</button>
    </div>
  );

  const kpis  = data?.kpis || {};
  const sh    = data?.shop_health || {};

  // ── Chart datasets ───────────────────────────────────────────────────────

  // 1. District volumes — bar
  const distVols = data?.district_volumes || [];
  const barDistData = {
    labels: distVols.map(d => d.district || d.distName || '?'),
    datasets: [{
      label: 'Total Distributed (kg)',
      data: distVols.map(d => d.total_kg),
      backgroundColor: PALETTE.slice(0, distVols.length),
      borderRadius: 4,
    }],
  };

  // 2. Commodity breakdown — doughnut
  const commBreak = data?.commodity_breakdown || [];
  const doughnutCommData = {
    labels: commBreak.map(c => c.commodity),
    datasets: [{
      data: commBreak.map(c => c.total_kg),
      backgroundColor: commBreak.map(c => COMMODITY_COLORS[c.commodity] || PALETTE[commBreak.indexOf(c) % PALETTE.length]),
      borderWidth: 2,
      borderColor: '#0F172A',
    }],
  };

  // 3. Monthly trend — line
  const trend = data?.monthly_trend || [];
  const lineData = {
    labels: trend.map(t => t.ym),
    datasets: [
      {
        label: 'Transactions',
        data: trend.map(t => t.transactions),
        borderColor: '#3B82F6',
        backgroundColor: '#3B82F620',
        tension: 0.4,
        fill: true,
        yAxisID: 'y',
      },
      {
        label: 'Volume (kg)',
        data: trend.map(t => t.total_kg),
        borderColor: '#10B981',
        backgroundColor: '#10B98120',
        tension: 0.4,
        fill: true,
        yAxisID: 'y1',
      },
    ],
  };
  const lineOptions = {
    ...chartDefaults,
    plugins: { ...chartDefaults.plugins, legend: { ...chartDefaults.plugins.legend, position: 'top' } },
    scales: {
      x: { ...chartDefaults.scales.x },
      y:  { ...chartDefaults.scales.y, position: 'left',  title: { display: true, text: 'Transactions', color: '#94A3B8' } },
      y1: { ...chartDefaults.scales.y, position: 'right', title: { display: true, text: 'Volume (kg)',   color: '#94A3B8' }, grid: { drawOnChartArea: false } },
    },
  };

  // 4. Shop health — doughnut
  const shopHealthData = {
    labels: ['Active', 'Inactive'],
    datasets: [{
      data: [sh.active || 0, sh.inactive || 0],
      backgroundColor: ['#10B981', '#EF4444'],
      borderWidth: 2,
      borderColor: '#0F172A',
    }],
  };

  // 5. Card types — pie
  const cardTypes = data?.card_type_distribution || [];
  const pieCardData = {
    labels: cardTypes.map(c => c.card_type),
    datasets: [{
      data: cardTypes.map(c => c.count),
      backgroundColor: ['#3B82F6','#10B981','#F59E0B','#EF4444','#8B5CF6'],
      borderWidth: 2,
      borderColor: '#0F172A',
    }],
  };

  // 6. Top districts by beneficiaries — horizontal bar
  const topBene = data?.top_districts_beneficiaries || [];
  const barBeneData = {
    labels: topBene.map(d => d.district),
    datasets: [{
      label: 'Beneficiaries',
      data: topBene.map(d => d.beneficiary_count),
      backgroundColor: '#8B5CF6',
      borderRadius: 4,
    }],
  };
  const horizOptions = {
    ...chartDefaults,
    indexAxis: 'y',
    plugins: { ...chartDefaults.plugins, legend: { display: false } },
  };

  // ── Render ───────────────────────────────────────────────────────────────
  return (
    <div style={{ padding: '24px 28px', color: '#F1F5F9' }}>

      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', marginBottom: 28 }}>
        <div>
          <h1 style={{ margin: 0, fontSize: 24, fontWeight: 700 }}>📊 Data Visualisation</h1>
          <p style={{ margin: '6px 0 0', color: '#64748B', fontSize: 13 }}>
            Live data from Open Data Telangana · Last refreshed: {lastRefresh || '—'}
          </p>
          {dataStatus && (
            <div style={{ marginTop: 8, display: 'flex', gap: 12, flexWrap: 'wrap' }}>
              {Object.entries(dataStatus).filter(([k]) => k !== 'data_dir' && k !== 'checked_at').map(([key, val]) => (
                <span key={key} style={{
                  fontSize: 11, padding: '2px 10px', borderRadius: 20,
                  background: val.source === 'real' ? '#064E3B' : '#1E293B',
                  color: val.source === 'real' ? '#6EE7B7' : '#94A3B8',
                  border: '1px solid', borderColor: val.source === 'real' ? '#10B98133' : '#33415533',
                }}>
                  {key.replace('_', ' ')}: {val.source} · {(val.rows||0).toLocaleString()} rows
                </span>
              ))}
            </div>
          )}
        </div>
        <div style={{ display: 'flex', gap: 10 }}>
          <button onClick={load} disabled={loading} style={{
            padding: '8px 18px', background: '#1E293B', color: '#CBD5E1',
            border: '1px solid #334155', borderRadius: 8, cursor: 'pointer', fontSize: 13,
          }}>
            🔄 Refresh
          </button>
          <button onClick={handleFetchLatest} disabled={fetching || loading} style={{
            padding: '8px 18px', background: fetching ? '#1D4ED8' : '#2563EB',
            color: '#fff', border: 'none', borderRadius: 8, cursor: 'pointer', fontSize: 13,
          }}>
            {fetching ? '⏳ Downloading…' : '⬇️ Fetch Latest Data'}
          </button>
        </div>
      </div>

      {/* KPI row */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: 16, marginBottom: 28 }}>
        <KpiCard icon="🏪" label="Total FPS Shops"        value={fmt(kpis.total_shops || 0)}        color="#3B82F6" />
        <KpiCard icon="✅" label="Active Shops"            value={fmt(kpis.active_shops || 0)}        color="#10B981" sub={`${sh.active_pct || 0}% active`} />
        <KpiCard icon="👥" label="Beneficiaries"           value={fmt(kpis.total_beneficiaries || 0)} color="#8B5CF6" />
        <KpiCard icon="🔄" label="Transaction Records"     value={fmt(kpis.total_transactions || 0)}  color="#F59E0B" />
        <KpiCard icon="⚖️" label="Total Distributed"       value={fmt(kpis.total_volume_kg || 0) + ' kg'} color="#06B6D4" />
      </div>

      {/* Row 1: District volumes + Commodity doughnut */}
      <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: 20, marginBottom: 20 }}>
        <Section title="District-wise Distribution Volume (kg)">
          <div style={{ height: 280 }}>
            <Bar data={barDistData} options={{ ...chartDefaults, plugins: { ...chartDefaults.plugins, legend: { display: false } } }} />
          </div>
        </Section>
        <Section title="Commodity Breakdown">
          <div style={{ height: 280 }}>
            <Doughnut data={doughnutCommData} options={{ ...noAxes, cutout: '60%' }} />
          </div>
        </Section>
      </div>

      {/* Row 2: Monthly trend — full width */}
      {trend.length > 0 && (
        <Section title="Monthly Transaction & Volume Trend">
          <div style={{ height: 260 }}>
            <Line data={lineData} options={lineOptions} />
          </div>
        </Section>
      )}

      {/* Row 3: Shop health + Card types + Top districts */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 2fr', gap: 20, marginBottom: 20 }}>
        <Section title="Shop Network Health">
          <div style={{ height: 220, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <Doughnut data={shopHealthData} options={{ ...noAxes, cutout: '55%' }} />
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-around', marginTop: 12, fontSize: 13, color: '#94A3B8' }}>
            <span>✅ Active: <strong style={{ color: '#10B981' }}>{(sh.active || 0).toLocaleString()}</strong></span>
            <span>❌ Inactive: <strong style={{ color: '#EF4444' }}>{(sh.inactive || 0).toLocaleString()}</strong></span>
          </div>
        </Section>

        <Section title="Card Type Distribution">
          {cardTypes.length > 0 ? (
            <div style={{ height: 220, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <Pie data={pieCardData} options={noAxes} />
            </div>
          ) : (
            <div style={{ color: '#64748B', fontSize: 13, paddingTop: 80, textAlign: 'center' }}>No card type data</div>
          )}
        </Section>

        <Section title="Top 10 Districts by Beneficiaries">
          {topBene.length > 0 ? (
            <div style={{ height: 260 }}>
              <Bar data={barBeneData} options={horizOptions} />
            </div>
          ) : (
            <div style={{ color: '#64748B', fontSize: 13, paddingTop: 100, textAlign: 'center' }}>No beneficiary district data</div>
          )}
        </Section>
      </div>

      {/* Raw data status table */}
      {dataStatus && (
        <Section title="Data Source Summary">
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ borderBottom: '1px solid #334155' }}>
                {['Dataset', 'Source', 'Rows', 'Date Range'].map(h => (
                  <th key={h} style={{ padding: '8px 12px', color: '#94A3B8', textAlign: 'left', fontWeight: 600 }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {Object.entries(dataStatus).filter(([k]) => k !== 'data_dir' && k !== 'checked_at').map(([key, val]) => (
                <tr key={key} style={{ borderBottom: '1px solid #1E293B' }}>
                  <td style={{ padding: '10px 12px', color: '#CBD5E1' }}>{key.replace(/_/g, ' ')}</td>
                  <td style={{ padding: '10px 12px' }}>
                    <span style={{
                      padding: '2px 10px', borderRadius: 20, fontSize: 11,
                      background: val.source === 'real' ? '#064E3B' : '#334155',
                      color: val.source === 'real' ? '#6EE7B7' : '#94A3B8',
                    }}>
                      {val.source}
                    </span>
                  </td>
                  <td style={{ padding: '10px 12px', color: '#F1F5F9', fontFamily: 'monospace' }}>{(val.rows || 0).toLocaleString()}</td>
                  <td style={{ padding: '10px 12px', color: '#94A3B8', fontSize: 12 }}>{val.date_range || '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Section>
      )}
    </div>
  );
}
