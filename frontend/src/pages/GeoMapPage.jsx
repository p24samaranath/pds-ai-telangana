import React, { useState, useEffect } from 'react';
import { getShopsGeoJSON, getUnderservedZones, getNewShopRecommendations, getAccessibilityScores } from '../services/api';

// Leaflet loaded via CDN in index.html
const L = typeof window !== 'undefined' ? window.L : null;

export default function GeoMapPage() {
  const [shops, setShops] = useState([]);
  const [underserved, setUnderserved] = useState([]);
  const [recommendations, setRecommendations] = useState([]);
  const [accessScores, setAccessScores] = useState({});
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('map');
  const [mapReady, setMapReady] = useState(false);

  useEffect(() => {
    loadData();
  }, []);

  useEffect(() => {
    if (activeTab === 'map' && shops.length > 0 && !mapReady) {
      setTimeout(initMap, 200);
    }
  }, [activeTab, shops]);

  async function loadData() {
    setLoading(true);
    try {
      const [shopData, underData, recData, scoreData] = await Promise.all([
        getShopsGeoJSON().catch(() => ({ features: [] })),
        getUnderservedZones().catch(() => ({ underserved_zones: [] })),
        getNewShopRecommendations(5).catch(() => ({ recommendations: [] })),
        getAccessibilityScores().catch(() => ({ district_scores: {} })),
      ]);
      setShops(shopData.features || []);
      setUnderserved(underData.underserved_zones || []);
      setRecommendations(recData.recommendations || []);
      setAccessScores(scoreData.district_scores || {});
    } catch {
      // silent
    } finally {
      setLoading(false);
    }
  }

  function initMap() {
    if (!L || mapReady) return;
    const mapEl = document.getElementById('leaflet-map');
    if (!mapEl) return;

    const map = L.map('leaflet-map').setView([17.385, 78.487], 7);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '© OpenStreetMap contributors',
    }).addTo(map);

    // FPS shops
    shops.forEach(feature => {
      const [lon, lat] = feature.geometry.coordinates;
      const p = feature.properties;
      const color = p.is_active ? '#10b981' : '#ef4444';
      L.circleMarker([lat, lon], { radius: 6, color, fillColor: color, fillOpacity: 0.8, weight: 1 })
        .bindPopup(`<b>${p.shop_name}</b><br>${p.district}<br>Cards: ${p.total_cards}<br>Active: ${p.is_active}`)
        .addTo(map);
    });

    // Underserved zones
    underserved.slice(0, 50).forEach(zone => {
      if (!zone.latitude || !zone.longitude) return;
      L.circleMarker([zone.latitude, zone.longitude], { radius: 5, color: '#f59e0b', fillColor: '#f59e0b', fillOpacity: 0.6, weight: 1 })
        .bindPopup(`<b>Underserved Zone</b><br>${zone.district}<br>Distance: ${zone.nearest_fps_distance_km} km<br>Affected: ${zone.affected_beneficiaries}`)
        .addTo(map);
    });

    // Recommendations
    recommendations.forEach((rec, i) => {
      L.marker([rec.recommended_lat, rec.recommended_lon], {
        icon: L.divIcon({
          html: `<div style="background:#1d4ed8;color:white;border-radius:50%;width:28px;height:28px;display:flex;align-items:center;justify-content:center;font-weight:bold;font-size:12px;border:2px solid white">${i + 1}</div>`,
          iconSize: [28, 28],
          iconAnchor: [14, 14],
        }),
      })
        .bindPopup(`<b>Recommended New FPS #${rec.rank}</b><br>${rec.district}<br>Coverage: ${rec.projected_coverage} beneficiaries<br>Distance reduction: ${rec.distance_reduction_km} km`)
        .addTo(map);
    });

    setMapReady(true);
  }

  const tabs = [
    { id: 'map',      label: '🗺️ Map View' },
    { id: 'zones',    label: '⚠️ Underserved Zones' },
    { id: 'recs',     label: '📍 New Shop Recommendations' },
    { id: 'scores',   label: '📊 Accessibility Scores' },
  ];

  return (
    <div>
      <div className="page-header">
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <div>
            <div className="page-title">🗺️ Geospatial Optimization</div>
            <div className="page-subtitle">FPS shop coverage analysis — Telangana</div>
          </div>
          <button className="btn btn-primary" onClick={() => { setMapReady(false); loadData(); }}>↻ Refresh</button>
        </div>
      </div>

      {loading ? (
        <div className="loading"><div className="spinner" /><span>Analysing geospatial data…</span></div>
      ) : (
        <>
          {/* Summary */}
          <div className="metrics-grid" style={{ gridTemplateColumns: 'repeat(4, 1fr)' }}>
            <div className="metric-card info">
              <div className="metric-value">{shops.filter(s => s.properties?.is_active).length}</div>
              <div className="metric-label">Active FPS Shops</div>
            </div>
            <div className="metric-card warning">
              <div className="metric-value">{underserved.length}</div>
              <div className="metric-label">Underserved Zones</div>
            </div>
            <div className="metric-card success">
              <div className="metric-value">{recommendations.length}</div>
              <div className="metric-label">New Shop Recommendations</div>
            </div>
            <div className="metric-card info">
              <div className="metric-value">{Object.keys(accessScores).length}</div>
              <div className="metric-label">Districts Analysed</div>
            </div>
          </div>

          <div className="tab-row">
            {tabs.map(t => <button key={t.id} className={`tab-btn ${activeTab === t.id ? 'active' : ''}`} onClick={() => setActiveTab(t.id)}>{t.label}</button>)}
          </div>

          {/* Map */}
          {activeTab === 'map' && (
            <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
              <div id="leaflet-map" style={{ height: 550 }} />
              <div style={{ padding: 12, display: 'flex', gap: 24, fontSize: 12, color: '#64748b' }}>
                <span><span style={{ color: '#10b981' }}>●</span> Active FPS</span>
                <span><span style={{ color: '#ef4444' }}>●</span> Inactive FPS</span>
                <span><span style={{ color: '#f59e0b' }}>●</span> Underserved Zone</span>
                <span><span style={{ color: '#1d4ed8' }}>①</span> New Location Recommendation</span>
              </div>
            </div>
          )}

          {/* Underserved Zones */}
          {activeTab === 'zones' && (
            <div className="card">
              <div className="card-title" style={{ marginBottom: 16 }}>{underserved.length} Underserved Zones</div>
              <div className="table-container">
                <table>
                  <thead><tr><th>District</th><th>Village</th><th>Distance (km)</th><th>Affected</th><th>Priority</th></tr></thead>
                  <tbody>
                    {underserved.slice(0, 30).map((z, i) => (
                      <tr key={i}>
                        <td>{z.district}</td>
                        <td>{z.village || '—'}</td>
                        <td style={{ color: z.nearest_fps_distance_km > 10 ? '#ef4444' : '#f59e0b' }}>{z.nearest_fps_distance_km} km</td>
                        <td>{z.affected_beneficiaries}</td>
                        <td><span className="score-pill" style={{ background: '#1d4ed8', color: '#fff' }}>{z.priority_score?.toFixed(1)}</span></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Recommendations */}
          {activeTab === 'recs' && (
            <div>
              {recommendations.map((rec, i) => (
                <div key={i} className="card" style={{ borderColor: '#1d4ed8' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                    <div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 8 }}>
                        <span style={{ background: '#1d4ed8', color: '#fff', borderRadius: '50%', width: 32, height: 32, display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 700, fontSize: 14 }}>#{rec.rank}</span>
                        <div>
                          <div style={{ fontSize: 16, fontWeight: 600, color: '#f1f5f9' }}>{rec.district}</div>
                          <div style={{ fontSize: 12, color: '#64748b' }}>{rec.recommended_lat.toFixed(4)}, {rec.recommended_lon.toFixed(4)}</div>
                        </div>
                      </div>
                      <div style={{ fontSize: 13, color: '#94a3b8', lineHeight: 1.6 }}>{rec.justification}</div>
                    </div>
                    <div style={{ textAlign: 'right', minWidth: 160 }}>
                      <div style={{ fontSize: 24, fontWeight: 700, color: '#10b981' }}>{rec.projected_coverage.toLocaleString()}</div>
                      <div style={{ fontSize: 12, color: '#64748b' }}>beneficiaries covered</div>
                      <div style={{ marginTop: 8, fontSize: 13, color: '#f59e0b' }}>↓ {rec.distance_reduction_km} km avg travel</div>
                    </div>
                  </div>
                </div>
              ))}
              {recommendations.length === 0 && <div style={{ color: '#64748b', textAlign: 'center', padding: 48 }}>No recommendations available.</div>}
            </div>
          )}

          {/* Accessibility Scores */}
          {activeTab === 'scores' && (
            <div className="card">
              <div className="card-title" style={{ marginBottom: 16 }}>District Accessibility Scores</div>
              {Object.entries(accessScores).sort(([, a], [, b]) => a - b).map(([district, score]) => (
                <div key={district} style={{ marginBottom: 14 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                    <span style={{ fontSize: 14, color: '#e2e8f0' }}>{district}</span>
                    <span style={{ fontSize: 14, fontWeight: 700, color: score > 0.8 ? '#10b981' : score > 0.5 ? '#f59e0b' : '#ef4444' }}>
                      {(score * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="progress-bar">
                    <div
                      className={`progress-fill ${score > 0.8 ? 'green' : score > 0.5 ? 'yellow' : 'red'}`}
                      style={{ width: `${score * 100}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}
