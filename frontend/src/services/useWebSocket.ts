import { useEffect, useRef, useState, useCallback } from 'react'

export type ChatMessage = {
    role: 'user' | 'assistant'
    content: string
    streaming?: boolean
    citations?: Array<{ text: string; source?: string; doc_id?: string; title?: string; section?: string }>
}

function randomId(): string {
    // Prefer crypto.randomUUID when available; fallback otherwise
    const cryptoObj =
        typeof globalThis !== 'undefined' && 'crypto' in globalThis
            ? (globalThis as unknown as { crypto?: Crypto }).crypto
            : undefined
    if (cryptoObj && typeof cryptoObj.randomUUID === 'function') {
        return cryptoObj.randomUUID()
    }
    return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`
}

export function useWebSocket(options?: { url?: string }) {
    const url = options?.url ?? 'ws://localhost:8000/ws'

    const wsRef = useRef<WebSocket | null>(null)
    const [connected, setConnected] = useState(false)
    const [initialized, setInitialized] = useState(false)
    const [messages, setMessages] = useState<ChatMessage[]>([])
    const [error, setError] = useState<string | null>(null)
    const [userId, setUserId] = useState<string | null>(null)
    const [threadId, setThreadId] = useState<string | null>(null)

    const connect = useCallback(() => {
        const ws = new WebSocket(url)
        wsRef.current = ws

        ws.onopen = () => {
            setConnected(true)
            setError(null)
            // Wait for explicit initSession(userId)
        }

        ws.onmessage = (evt) => {
            try {
                const data = JSON.parse(evt.data)
                if (data.ok) {
                    setInitialized(true)
                    return
                }
                // Prefer repo-style streaming event keys; fall back to previous tokens
                const chunk: string | undefined = data.on_chat_model_stream
                const end: boolean | undefined = data.on_chat_model_end

                if (typeof chunk === 'string' && chunk.length > 0) {
                    // Append chunk to the last assistant streaming message or start one
                    setMessages((prev) => {
                        const next = [...prev]
                        const last = next[next.length - 1]
                        if (last && last.role === 'assistant' && last.streaming) {
                            last.content = `${last.content}${chunk}`
                        } else {
                            next.push({ role: 'assistant', content: chunk, streaming: true })
                        }
                        return next
                    })
                    return
                }

                if (end === true) {
                    // Mark current assistant streaming bubble as ended; content/citations finalized later
                    setMessages((prev) => {
                        const next = [...prev]
                        for (let i = next.length - 1; i >= 0; i--) {
                            if (next[i].role === 'assistant' && next[i].streaming) {
                                next[i] = { ...next[i], streaming: false }
                                break
                            }
                        }
                        return next
                    })
                    return
                }

                if (data.type === 'assistant_message') {
                    // Finalize assistant message
                    setMessages((prev) => {
                        const next = [...prev]
                        // Find the most recent assistant streaming bubble to finalize
                        let idx = next.length - 1
                        while (idx >= 0) {
                            if (next[idx].role === 'assistant' && next[idx].streaming) break
                            idx--
                        }
                        if (idx >= 0) {
                            next[idx] = {
                                role: 'assistant',
                                content: data.content,
                                streaming: false,
                                citations: data.citations,
                            }
                        } else {
                            // No streaming bubble found; create a single finalized assistant message
                            next.push({ role: 'assistant', content: data.content, streaming: false, citations: data.citations })
                        }
                        return next
                    })
                    return
                }
            } catch (e) {
                console.error('WS message parse error', e)
            }
        }

        ws.onclose = (event) => {
            setConnected(false)
            setInitialized(false)
            console.log('WebSocket connection closed:', event.code, event.reason)
            // No auto-reconnect - let user manually reconnect if needed
        }

        ws.onerror = () => {
            setError('WebSocket error')
        }
    }, [url])

    const initSession = useCallback((uid: string, tid?: string) => {
        const ws = wsRef.current
        if (!ws || ws.readyState !== WebSocket.OPEN) return
        const threadId = tid || `${uid}_${randomId()}`
        setUserId(uid)
        setThreadId(threadId)
        ws.send(JSON.stringify({ user_id: uid, thread_id: threadId, init: true }))
    }, [])

    const sendMessage = useCallback((text: string) => {
        const ws = wsRef.current
        if (!ws || ws.readyState !== WebSocket.OPEN) return
        if (!initialized) return
        // Push user message locally for chat view; assistant placeholder will be created on first token
        setMessages((prev) => [...prev, { role: 'user', content: text }])
        ws.send(JSON.stringify({ message: text }))
    }, [initialized])

    const reconnect = useCallback(() => {
        wsRef.current?.close()
        connect()
    }, [connect])

    useEffect(() => {
        connect()
        return () => {
            wsRef.current?.close()
        }
    }, [connect])

    return { connected, initialized, messages, sendMessage, error, initSession, userId, threadId, reconnect }
}
