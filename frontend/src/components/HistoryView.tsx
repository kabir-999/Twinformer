import { useEffect, useState } from 'react'
import { api } from '../api/client'
import type { Chat, ChatSummary } from '../types'
import { message } from './format'

export function HistoryView({ onOpen, refreshKey }: { onOpen: (chat: Chat) => void; refreshKey?: number }) {
  const [chats, setChats] = useState<ChatSummary[]>()
  const [error, setError] = useState('')
  const load = () => api.chats().then(setChats).catch(e => setError(message(e)))
  useEffect(() => { load() }, [refreshKey])
  const open = async (id: string) => { try { onOpen(await api.getChat(id)) } catch (e) { setError(message(e)) } }
  const remove = async (id: string) => { if (!window.confirm('Delete this chat?')) return; try { await api.deleteChat(id); load() } catch (e) { setError(message(e)) } }
  return (
    <div className="history-view-container">
      <div className="twin-header-row">
        <div>
          <h2 style={{ margin: '0 0 4px', fontSize: '1.15rem' }}>Chat History</h2>
          <p className="muted small" style={{ margin: 0 }}>Review or resume your earlier conversations</p>
        </div>
      </div>
      {error && <p className="error small">{error}</p>}
      {chats && !chats.length && <p className="muted small" style={{ marginTop: 12 }}>No chats yet. Ask a question to begin.</p>}
      <div className="history-list">
        {chats?.map(c => (
          <div className="hist-card" key={c.id}>
            <div className="hist-meta">
              <strong>{c.title}</strong>
              <div className="small muted">
                {new Date(c.updated_at).toLocaleDateString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })} · {c.turn_count} question{c.turn_count === 1 ? '' : 's'} · {c.document_count ? `${c.document_count} file${c.document_count === 1 ? '' : 's'}` : 'all files'}
              </div>
            </div>
            <div className="hist-actions">
              <button className="small primary" onClick={() => open(c.id)}>Open</button>
              <button className="small" onClick={() => remove(c.id)} title="Delete conversation">Delete</button>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
