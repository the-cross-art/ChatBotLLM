import { useState, useRef, useEffect } from 'react'
import { useWebSocket } from './services/useWebSocket'

export default function App() {
    const { connected, initialized, messages, sendMessage, error, initSession, userId, threadId, reconnect } = useWebSocket()
    const [input, setInput] = useState('')
    const [uidInput, setUidInput] = useState('')
    const [tidInput, setTidInput] = useState('')
    const listRef = useRef<HTMLDivElement | null>(null)

    useEffect(() => {
        listRef.current?.scrollTo({ top: listRef.current.scrollHeight, behavior: 'smooth' })
    }, [messages.length])

    const onSubmit = (e: React.FormEvent) => {
        e.preventDefault()
        if (!input.trim() || !initialized) return
        sendMessage(input.trim())
        setInput('')
    }

    const onInit = (e: React.FormEvent) => {
        e.preventDefault()
        const u = uidInput.trim()
        if (!u) return
        const t = tidInput.trim() || `${u}_${Date.now()}_${Math.random().toString(36).slice(2)}`
        initSession(u, t)
    }

    return (
        <div style={{ maxWidth: 800, margin: '0 auto', padding: 24, fontFamily: 'system-ui, sans-serif' }}>
            <h1>ChatBot Frontend</h1>
            <p>
                Connection: <strong>{connected ? 'Connected' : 'Disconnected'}</strong>
                {' '}
                Session: <strong>{initialized ? 'Ready' : 'Not initialized'}</strong>
                {!connected && (
                    <button onClick={reconnect} style={{ marginLeft: 8, padding: '4px 8px', fontSize: '12px' }}>
                        Reconnect
                    </button>
                )}
                {error ? <span style={{ color: 'red', marginLeft: 8 }}>({error})</span> : null}
            </p>

            {!initialized && (
                <form onSubmit={onInit} style={{ marginBottom: 16, display: 'flex', flexDirection: 'column', gap: 8 }}>
                    <input
                        type="text"
                        placeholder="Enter user_id (e.g., john_doe)"
                        value={uidInput}
                        onChange={(e) => setUidInput(e.target.value)}
                        style={{ padding: 8, borderRadius: 8, border: '1px solid #ddd' }}
                    />
                    <input
                        type="text"
                        placeholder="Enter thread_id (optional - will auto-generate if empty)"
                        value={tidInput}
                        onChange={(e) => setTidInput(e.target.value)}
                        style={{ padding: 8, borderRadius: 8, border: '1px solid #ddd' }}
                    />
                    <button type="submit" disabled={!connected} style={{ padding: '8px 12px' }}>
                        Initialize Session
                    </button>
                </form>
            )}

            {initialized && (
                <p style={{ color: '#555' }}>User: <strong>{userId}</strong> · Thread: <strong>{threadId}</strong></p>
            )}

            <div ref={listRef} style={{ border: '1px solid #ddd', borderRadius: 8, padding: 16, height: 360, overflowY: 'auto', background: '#fafafa' }}>
                {messages.length === 0 && <p style={{ color: '#888' }}>{initialized ? 'Send a message to start!' : 'Enter user_id to initialize.'}</p>}
                {messages.map((m, i) => {
                    const isUser = m.role === 'user'
                    return (
                        <div
                            key={i}
                            style={{
                                display: 'flex',
                                justifyContent: isUser ? 'flex-end' : 'flex-start',
                                marginBottom: 12,
                            }}
                        >
                            <div
                                style={{
                                    maxWidth: '70%',
                                    padding: '8px 12px',
                                    borderRadius: 12,
                                    background: isUser ? '#e0f0ff' : '#fff',
                                    border: '1px solid #ddd',
                                    boxShadow: '0 1px 2px rgba(0,0,0,0.04)',
                                }}
                            >
                                <div style={{ whiteSpace: 'pre-wrap' }}>{m.content}</div>
                                {!isUser && m.streaming && (
                                    <span style={{ fontSize: 12, color: '#888' }}>typing…</span>
                                )}
                                {!isUser && m.citations && m.citations.length > 0 && (
                                    <details style={{ marginTop: 8 }}>
                                        <summary>Citations</summary>
                                        <ul>
                                            {m.citations.map((c, j) => (
                                                <li key={j}>
                                                    {c.source || c.title || c.doc_id} {c.section ? `(${c.section})` : ''}
                                                </li>
                                            ))}
                                        </ul>
                                    </details>
                                )}
                            </div>
                        </div>
                    )
                })}
            </div>
            <form onSubmit={onSubmit} style={{ marginTop: 16 }}>
                <textarea
                    rows={3}
                    style={{ width: '100%', padding: 8, borderRadius: 8, border: '1px solid #ddd' }}
                    placeholder={initialized ? 'Type message...' : 'Initialize session to start chatting'}
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={(e) => {
                        if (e.key === 'Enter' && !e.shiftKey) {
                            e.preventDefault()
                            onSubmit(e)
                        }
                    }}
                    disabled={!initialized}
                />
                <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: 8 }}>
                    <button type="submit" style={{ padding: '8px 12px' }} disabled={!initialized}>
                        Send
                    </button>
                </div>
            </form>
        </div>
    )
}
