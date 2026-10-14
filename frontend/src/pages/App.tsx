import { useEffect, useState } from 'react'
import { ChatView } from '../components/ChatView'
import { TwinView } from '../components/TwinView'
import { HistoryView } from '../components/HistoryView'
import type { Chat } from '../types'

const tabs = { chat: 'Chat', history: 'History', twin: 'Twin' } as const
type Tab = keyof typeof tabs
const fromHash = (): Tab => (window.location.hash.slice(1) in tabs ? (window.location.hash.slice(1) as Tab) : 'chat')

export function App() {
  const [tab, setTab] = useState<Tab>(fromHash)
  const [chat, setChat] = useState<Chat | null>(null)
  useEffect(() => { const onHash = () => setTab(fromHash()); window.addEventListener('hashchange', onHash); return () => window.removeEventListener('hashchange', onHash) }, [])
  const startNew = () => { setChat(null); window.location.hash = 'chat'; window.scrollTo(0, 0) }
  return (
    <>
      <header className="topbar">
        <div className="inner">
          <a href="#chat" className="brand" onClick={startNew} title="Home: start a new chat">TransformerTwin</a>
          <nav>
            {(Object.keys(tabs) as Tab[]).map(key => <a key={key} href={`#${key}`} className={key === tab ? 'active' : ''}>{tabs[key]}</a>)}
            <button className="newchat" onClick={startNew}>+ New chat</button>
          </nav>
        </div>
      </header>
      <main>
        <div hidden={tab !== 'chat'}><ChatView chat={chat} onChat={setChat} /></div>
        {tab === 'twin' && <TwinView />}
        {tab === 'history' && <HistoryView onOpen={c => { setChat(c); window.location.hash = 'chat' }} />}
      </main>
    </>
  )
}
