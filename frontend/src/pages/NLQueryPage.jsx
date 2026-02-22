import React, { useState, useRef, useEffect, useCallback } from 'react';
import { sendChatMessage, clearChatSession } from '../services/api';

const EXAMPLE_QUERIES = [
  "Which FPS shops in Nizamabad had fraudulent transactions last month?",
  "Show me the top 5 underserved areas in Warangal district",
  "Which commodities are at risk of going out of stock next month?",
  "How many beneficiaries are more than 5km from their nearest FPS shop?",
  "Which dealers have the lowest biometric verification rates?",
  "What is the overall fraud detection status for Hyderabad?",
];

// Stable session ID per browser tab — persists for the page lifetime
const SESSION_ID = `session-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;

export default function NLQueryPage() {
  const [query, setQuery] = useState('');
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [ragSources, setRagSources] = useState([]);
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Clear backend session history when component unmounts
  useEffect(() => {
    return () => { clearChatSession(SESSION_ID).catch(() => {}); };
  }, []);

  const sendQuery = useCallback(async () => {
    if (!query.trim() || loading) return;
    const userQuery = query.trim();
    setQuery('');
    setMessages(prev => [...prev, { role: 'user', text: userQuery }]);
    setLoading(true);
    setRagSources([]);

    try {
      const data = await sendChatMessage(userQuery, SESSION_ID);
      const sources = data.rag_sources || [];
      setRagSources(sources);
      setMessages(prev => [...prev, {
        role: 'assistant',
        text: data.answer || 'No answer returned.',
        meta: {
          agent: data.agent_used || 'reporting_rag',
          time: data.generated_at,
          turn: data.conversation_turn,
          sources,
        },
      }]);
    } catch (e) {
      const errorMsg = e?.response?.data?.detail
        || e?.message
        || 'Unknown error';
      const isNetworkError = !e?.response;
      setMessages(prev => [...prev, {
        role: 'assistant',
        text: isNetworkError
          ? `⚠️ Cannot reach backend.\n\nStart the server:\n\ncd backend && python -m uvicorn app.main:app --reload --port 8000`
          : `⚠️ Error: ${errorMsg}`,
        meta: { agent: 'system' },
      }]);
    } finally {
      setLoading(false);
    }
  }, [query, loading]);

  function handleKey(e) {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendQuery(); }
  }

  function handleClearChat() {
    setMessages([]);
    setRagSources([]);
    clearChatSession(SESSION_ID).catch(() => {});
  }

  return (
    <div>
      <div className="page-header">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
          <div>
            <div className="page-title">🤖 AI Query Interface</div>
            <div className="page-subtitle">
              RAG-powered multi-turn chat — grounded in live PDS data
            </div>
          </div>
          {messages.length > 0 && (
            <button
              onClick={handleClearChat}
              style={{
                background: 'transparent', border: '1px solid #475569',
                borderRadius: 6, padding: '6px 12px', color: '#94a3b8',
                cursor: 'pointer', fontSize: 13,
              }}
            >
              🗑 Clear chat
            </button>
          )}
        </div>
      </div>

      {/* RAG source pills — shown after last response */}
      {ragSources.length > 0 && (
        <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginBottom: 12 }}>
          <span style={{ fontSize: 12, color: '#64748b', alignSelf: 'center' }}>
            Retrieved from:
          </span>
          {ragSources.map(src => (
            <span
              key={src}
              style={{
                fontSize: 11, padding: '2px 8px', borderRadius: 12,
                background: '#1e3a5f', color: '#93c5fd', border: '1px solid #1d4ed8',
              }}
            >
              {src.replace(/_/g, ' ')}
            </span>
          ))}
        </div>
      )}

      {/* Example queries (shown when chat is empty) */}
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

      {/* Message thread */}
      <div className="chat-container">
        {messages.map((msg, i) => (
          <div key={i} className={`chat-message ${msg.role}`}>
            <div className="chat-role">{msg.role === 'user' ? '👤 You' : '🤖 PDS AI Assistant'}</div>
            <div className="chat-text" style={{ whiteSpace: 'pre-wrap' }}>{msg.text}</div>
            {msg.meta && (
              <div style={{ marginTop: 8, fontSize: 12, color: '#475569', display: 'flex', gap: 12, flexWrap: 'wrap' }}>
                <span>Agent: {msg.meta.agent}</span>
                {msg.meta.turn != null && <span>Turn {msg.meta.turn}</span>}
                {msg.meta.time && (
                  <span>{new Date(msg.meta.time).toLocaleTimeString('en-IN')}</span>
                )}
                {msg.meta.sources?.length > 0 && (
                  <span>Sources: {msg.meta.sources.join(', ')}</span>
                )}
              </div>
            )}
          </div>
        ))}

        {loading && (
          <div className="chat-message assistant">
            <div className="chat-role">🤖 PDS AI Assistant</div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 12, color: '#64748b' }}>
              <div className="spinner" style={{ width: 20, height: 20, borderWidth: 2 }} />
              <span>Retrieving context and generating answer…</span>
            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {/* Input bar */}
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
          Press Enter to send · Shift+Enter for new line · Session: {SESSION_ID.slice(-8)}
        </div>
      </div>
    </div>
  );
}
