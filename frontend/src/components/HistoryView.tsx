import { useEffect, useState } from 'react'
import { api } from '../api/client'
import type { Chat, ChatSummary } from '../types'
import { message } from './format'

export function HistoryView({ onOpen }: { onOpen: (chat: Chat) => void }) {
  const [chats, setChats] = useState<ChatSummary[]>()
  const [error, setError] = useState('')
  const load = () => api.chats().then(setChats).catch(e => setError(message(e)))
  useEffect(() => { load() }, [])
  const open = async (id: string) => { try { onOpen(await api.getChat(id)) } catch (e) { setError(message(e)) } }
  const remove = async (id: string) => { if (!window.confirm('Delete this chat?')) return; try { await api.deleteChat(id); load() } catch (e) { setError(message(e)) } }
  return (
    <>
      <h1>History</h1>
      <p className="muted">Your earlier chats. Open one to read it or keep asking follow-up questions.</p>
      {error && <p className="error">{error}</p>}
      {chats && !chats.length && <p className="muted">No chats yet. Ask a question on the Chat tab.</p>}
      {chats?.map(c => (
        <div className="hist" key={c.id}>
          <div><strong>{c.title}</strong><div className="small muted">{new Date(c.updated_at).toLocaleString()} · {c.turn_count} question{c.turn_count === 1 ? '' : 's'} · {c.document_count ? `${c.document_count} file${c.document_count === 1 ? '' : 's'}` : 'all files'}</div></div>
          <div className="row" style={{ margin: 0 }}><button onClick={() => open(c.id)}>Open</button><button onClick={() => remove(c.id)}>Delete</button></div>
        </div>
      ))}
    </>
  )
}
