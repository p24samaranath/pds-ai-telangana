/**
 * App.jsx — PDS AI Optimization System
 * Features:
 *  - Startup health polling until backend is ready
 *  - Auto-navigate to Data Viz tab on first load
 *  - Exit button calls /api/v1/system/shutdown
 *  - 6 navigation tabs including Data Visualisation
 */
import React, { useState, useEffect, useCallback, useRef } from 'react';
import { BrowserRouter, Routes, Route, NavLink, useNavigate, useLocation } from 'react-router-dom';
import Dashboard      from './pages/Dashboard';
import FraudAlertsPage from './pages/FraudAlertsPage';
import ForecastsPage  from './pages/ForecastsPage';
import GeoMapPage     from './pages/GeoMapPage';
import NLQueryPage    from './pages/NLQueryPage';
import DataVizPage    from './pages/DataVizPage';
import { getHealth, triggerShutdown } from './services/api';

const NAV_ITEMS = [
  { path: '/viz',      label: 'Data Overview',  icon: '📊' },
  { path: '/',         label: 'Dashboard',      icon: '🏠' },
  { path: '/fraud',    label: 'Fraud Alerts',   icon: '🚨' },
  { path: '/forecasts',label: 'Forecasts',      icon: '📈' },
  { path: '/map',      label: 'Geo Map',        icon: '🗺️'  },
  { path: '/query',    label: 'AI Query',       icon: '🤖' },
];

// ── Boot screen shown while backend is starting ────────────────────────────
function BootScreen({ status, attempt, maxAttempts }) {
  const pct = Math.round((attempt / maxAttempts) * 100);
  return (
    <div style={{
      height: '100vh', background: '#0F172A',
      display: 'flex', flexDirection: 'column',
      alignItems: 'center', justifyContent: 'center', gap: 20,
    }}>
      <div style={{ fontSize: 56 }}>🌾</div>
      <div style={{ color: '#F1F5F9', fontSize: 22, fontWeight: 700 }}>PDS AI System</div>
      <div style={{ color: '#94A3B8', fontSize: 14 }}>Telangana Fair Price Shop Optimizer</div>
      <div style={{ width: 280, height: 6, background: '#1E293B', borderRadius: 3, overflow: 'hidden', marginTop: 12 }}>
        <div style={{
          height: '100%', borderRadius: 3,
          background: status === 'error' ? '#EF4444' : '#3B82F6',
          width: `${pct}%`,
          transition: 'width 0.4s ease',
        }} />
      </div>
      <div style={{ color: '#64748B', fontSize: 12 }}>
        {status === 'connecting' && `Connecting to backend… (${attempt}/${maxAttempts})`}
        {status === 'ready'      && '✅ Connected! Starting application…'}
        {status === 'error'      && '⚠ Backend not responding — check start.sh logs'}
      </div>
      {status === 'error' && (
        <div style={{ color: '#94A3B8', fontSize: 12, maxWidth: 320, textAlign: 'center' }}>
          Run <code style={{ background: '#1E293B', padding: '2px 6px', borderRadius: 4 }}>./start.sh</code> from the project root to start all services.
        </div>
      )}
    </div>
  );
}

// ── Exit confirmation dialog ───────────────────────────────────────────────
function ExitDialog({ onConfirm, onCancel, shuttingDown }) {
  return (
    <div style={{
      position: 'fixed', inset: 0, background: '#0008',
      display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 9999,
    }}>
      <div style={{
        background: '#1E293B', borderRadius: 14, padding: 32,
        width: 380, boxShadow: '0 20px 60px #00000088',
      }}>
        <div style={{ fontSize: 36, marginBottom: 12 }}>👋</div>
        <h3 style={{ margin: '0 0 10px', color: '#F1F5F9', fontSize: 18 }}>Exit PDS AI System?</h3>
        <p style={{ color: '#94A3B8', fontSize: 14, margin: '0 0 24px', lineHeight: 1.6 }}>
          This will stop the backend API and frontend server. All running processes will be terminated.
        </p>
        {shuttingDown ? (
          <div style={{ color: '#F59E0B', fontSize: 14, textAlign: 'center' }}>
            ⏳ Shutting down…
          </div>
        ) : (
          <div style={{ display: 'flex', gap: 12, justifyContent: 'flex-end' }}>
            <button onClick={onCancel} style={{
              padding: '8px 20px', background: '#334155', color: '#CBD5E1',
              border: 'none', borderRadius: 8, cursor: 'pointer', fontSize: 14,
            }}>Cancel</button>
            <button onClick={onConfirm} style={{
              padding: '8px 20px', background: '#EF4444', color: '#fff',
              border: 'none', borderRadius: 8, cursor: 'pointer', fontSize: 14, fontWeight: 600,
            }}>🛑 Stop System</button>
          </div>
        )}
      </div>
    </div>
  );
}

// ── Inner shell (needs Router context) ────────────────────────────────────
function Shell() {
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [showExitDlg, setShowExitDlg] = useState(false);
  const [shuttingDown, setShuttingDown] = useState(false);
  const navigate = useNavigate();
  const location = useLocation();

  // Navigate to Data Viz only on root path (first launch), not when deep-linking
  useEffect(() => {
    if (location.pathname === '/') {
      navigate('/viz', { replace: true });
    }
  }, []); // eslint-disable-line

  const handleExit = async () => {
    setShuttingDown(true);
    try {
      await triggerShutdown();
    } catch (_) { /* backend will close so request may not complete */ }
    setTimeout(() => window.close(), 1500);
  };

  return (
    <div className="app-layout">
      {/* Sidebar */}
      <aside className={`sidebar ${sidebarOpen ? 'open' : 'collapsed'}`}>
        <div className="sidebar-header">
          <span className="logo">🌾</span>
          {sidebarOpen && (
            <div className="logo-text">
              <div className="logo-title">PDS AI System</div>
              <div className="logo-sub">Telangana</div>
            </div>
          )}
          <button className="toggle-btn" onClick={() => setSidebarOpen(!sidebarOpen)}>
            {sidebarOpen ? '◀' : '▶'}
          </button>
        </div>

        <nav className="sidebar-nav">
          {NAV_ITEMS.map(item => (
            <NavLink
              key={item.path}
              to={item.path}
              end={item.path === '/'}
              className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}
            >
              <span className="nav-icon">{item.icon}</span>
              {sidebarOpen && <span className="nav-label">{item.label}</span>}
            </NavLink>
          ))}
        </nav>

        {/* Footer with Exit button */}
        <div className="sidebar-footer">
          {sidebarOpen && <div className="version-badge">v1.0.0 — Phase 1</div>}
          <button
            onClick={() => setShowExitDlg(true)}
            title="Stop all services and exit"
            style={{
              marginTop: sidebarOpen ? 10 : 0,
              width: sidebarOpen ? '100%' : 40,
              padding: sidebarOpen ? '8px 12px' : '8px',
              background: '#7F1D1D',
              color: '#FCA5A5',
              border: '1px solid #991B1B',
              borderRadius: 8,
              cursor: 'pointer',
              fontSize: 13,
              fontWeight: 600,
              display: 'flex',
              alignItems: 'center',
              justifyContent: sidebarOpen ? 'flex-start' : 'center',
              gap: 8,
            }}
          >
            <span>🛑</span>
            {sidebarOpen && <span>Exit System</span>}
          </button>
        </div>
      </aside>

      {/* Main content */}
      <main className="main-content">
        <Routes>
          <Route path="/viz"       element={<DataVizPage />} />
          <Route path="/"          element={<Dashboard />} />
          <Route path="/fraud"     element={<FraudAlertsPage />} />
          <Route path="/forecasts" element={<ForecastsPage />} />
          <Route path="/map"       element={<GeoMapPage />} />
          <Route path="/query"     element={<NLQueryPage />} />
        </Routes>
      </main>

      {/* Exit dialog */}
      {showExitDlg && (
        <ExitDialog
          shuttingDown={shuttingDown}
          onCancel={() => setShowExitDlg(false)}
          onConfirm={handleExit}
        />
      )}
    </div>
  );
}

// ── Root App — handles boot / health polling ───────────────────────────────
export default function App() {
  const MAX_ATTEMPTS = 30;           // 30 × 2 s = 60 s window
  const POLL_INTERVAL = 2000;

  const [bootStatus, setBootStatus] = useState('connecting');  // connecting | ready | error
  const [attempt, setAttempt]       = useState(0);
  const [ready, setReady]           = useState(false);
  const intervalRef = useRef(null);

  const poll = useCallback(async (att) => {
    try {
      await getHealth();
      clearInterval(intervalRef.current);
      setBootStatus('ready');
      setTimeout(() => setReady(true), 600);
    } catch (_) {
      if (att >= MAX_ATTEMPTS) {
        clearInterval(intervalRef.current);
        setBootStatus('error');
      }
    }
  }, []);

  useEffect(() => {
    let att = 0;
    poll(att);
    intervalRef.current = setInterval(() => {
      att += 1;
      setAttempt(att);
      poll(att);
    }, POLL_INTERVAL);
    return () => clearInterval(intervalRef.current);
  }, [poll]);

  if (!ready) {
    return <BootScreen status={bootStatus} attempt={attempt} maxAttempts={MAX_ATTEMPTS} />;
  }

  return (
    <BrowserRouter>
      <Shell />
    </BrowserRouter>
  );
}
