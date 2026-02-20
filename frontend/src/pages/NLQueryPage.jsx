import React, { useState, useRef, useEffect } from 'react';
import { submitNLQuery } from '../services/api';

const EXAMPLE_QUERIES = [
  "Which FPS shops in Nizamabad had fraudulent transactions last month?",
  "Show me the top 5 underserved areas in Warangal district",
  "Which commodities are at risk of going out of stock next month?",
  "How many beneficiaries are more than 5km from their nearest FPS shop?",
  "Which dealers have the lowest biometric verification rates?",
  "What is the overall fraud detection status for Hyderabad?",
];

export default function NLQueryPage() {
  const [query, setQuery] = useState('');
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  async function sendQuery() {
    if (!query.trim() || loading) return;
    const userQuery = query.trim();
    setQuery('');
    setMessages(prev => [...prev, { role: 'user', text: userQuery }]);
    setLoading(true);
    try {
      const data = await submitNLQuery(userQuery);
      setMessages(prev => [...prev, {
        role: 'assistant',
        text: data.answer,
        meta: { agent: data.agent_used, time: data.generated_at },
      }]);
    } catch (e) {
      setMessages(prev => [...prev, {
        role: 'assistant',
        text: '⚠️ Backend unavailable. Please start the server with:\n\ncd backend && python -m uvicorn app.main:app --reload',
        meta: { agent: 'system' },
      }]);
    } finally {
      setLoading(false);
    }
  }

  function handleKey(e) {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendQuery(); }
  }

  return (
    <div>
      <div className="page-header">
        <div className="page-title">🤖 AI Query Interface</div>
        <div className="page-subtitle">Ask natural language questions — powered by Claude</div>
      </div>

      {/* Example queries */}
      {messages.length === 0 && (
        <div className="card">
          <div className="card-title" style={{ marginBottom: 16 }}>Example Questions</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {EXAMPLE_QUERIES.map((q, i) => (
              <button
                key={i}
                onClick={() => setQuery(q)}
                style={{
                  background: '#0f172a', border: '1px solid #334155', borderRadius: 8,
                  padding: '10px 14px', color: '#94a3b8', cursor: 'pointer',
                  textAlign: 'left', fontSize: 14, transition: 'all 0.2s',
                }}
                onMouseEnter={e => e.currentTarget.style.borderColor = '#1d4ed8'}
                onMouseLeave={e => e.currentTarget.style.borderColor = '#334155'}
              >
                💬 {q}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Messages */}
      <div className="chat-container">
        {messages.map((msg, i) => (
          <div key={i} className={`chat-message ${msg.role}`}>
            <div className="chat-role">{msg.role === 'user' ? '👤 You' : '🤖 PDS AI Assistant'}</div>
            <div className="chat-text">{msg.text}</div>
            {msg.meta && (
              <div style={{ marginTop: 8, fontSize: 12, color: '#475569' }}>
                Agent: {msg.meta.agent} {msg.meta.time && `• ${new Date(msg.meta.time).toLocaleTimeString('en-IN')}`}
              </div>
            )}
          </div>
        ))}

        {loading && (
          <div className="chat-message assistant">
            <div className="chat-role">🤖 PDS AI Assistant</div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 12, color: '#64748b' }}>
              <div className="spinner" style={{ width: 20, height: 20, borderWidth: 2 }} />
              <span>Analysing PDS data…</span>
            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="card" style={{ position: 'sticky', bottom: 0, marginTop: 16 }}>
        <div className="chat-input-row">
          <textarea
            className="chat-input"
            value={query}
            onChange={e => setQuery(e.target.value)}
            onKeyDown={handleKey}
            placeholder="Ask a question about PDS shops, fraud, forecasts, or coverage…"
            rows={2}
            style={{ resize: 'none' }}
          />
          <button
            className="btn btn-primary"
            onClick={sendQuery}
            disabled={loading || !query.trim()}
            style={{ alignSelf: 'flex-end', padding: '10px 20px' }}
          >
            {loading ? '…' : '➤ Send'}
          </button>
        </div>
        <div style={{ fontSize: 12, color: '#475569', marginTop: 8 }}>
          Press Enter to send · Shift+Enter for new line
        </div>
      </div>
    </div>
  );
}
