import { useEffect, useState } from 'react'
import { ChatView } from '../components/ChatView'
import { TwinView } from '../components/TwinView'
import { HistoryView } from '../components/HistoryView'
import { api } from '../api/client'
import type { Chat, TwinStatus } from '../types'

export function App() {
  const [chat, setChat] = useState<Chat | null>(null)
  const [sideTab, setSideTab] = useState<'twin' | 'history'>('twin')
  const [panelOpen, setPanelOpen] = useState(true)
  const [refreshKey, setRefreshKey] = useState(0)
  const [twinInfo, setTwinInfo] = useState<TwinStatus>()

  const handleTurnComplete = () => {
    setRefreshKey(k => k + 1)
  }

  useEffect(() => {
    api.twinStatus().then(setTwinInfo).catch(() => undefined)
  }, [refreshKey])

  const startNew = () => {
    setChat(null)
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  const handleOpenChat = (c: Chat) => {
    setChat(c)
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  const runsCount = twinInfo?.points.length ?? 0
  const isTrained = twinInfo?.calibration?.status === 'trained'

  return (
    <div className="app-root">
      <header className="topbar">
        <div className="inner">
          <div className="brand-group">
            <span className="brand" onClick={startNew} role="button" tabIndex={0} title="Start a new chat">
              TransformerTwin
            </span>
            <button
              className="twin-pill-btn"
              onClick={() => { setSideTab('twin'); setPanelOpen(true) }}
              title="Click to view digital twin telemetry"
            >
              <span className="live-indicator" />
              <span>Digital Twin: <strong>{runsCount}</strong> runs {isTrained ? 'calibrated' : 'learning'}</span>
            </button>
          </div>

          <nav>
            <button
              className={`panel-toggle-btn ${panelOpen && sideTab === 'twin' ? 'active' : ''}`}
              onClick={() => {
                if (panelOpen && sideTab === 'twin') setPanelOpen(false)
                else { setSideTab('twin'); setPanelOpen(true) }
              }}
            >
              Digital Twin
            </button>
            <button
              className={`panel-toggle-btn ${panelOpen && sideTab === 'history' ? 'active' : ''}`}
              onClick={() => {
                if (panelOpen && sideTab === 'history') setPanelOpen(false)
                else { setSideTab('history'); setPanelOpen(true) }
              }}
            >
              History
            </button>
            <button className="newchat primary" onClick={startNew}>+ New chat</button>
          </nav>
        </div>
      </header>

      <main className="workspace-main">
        <div className={`workspace-layout ${panelOpen ? 'with-panel' : 'full-chat'}`}>
          <section className="chat-column">
            <ChatView chat={chat} onChat={setChat} onTurnComplete={handleTurnComplete} />
          </section>

          {panelOpen && (
            <aside className="telemetry-column">
              <div className="telemetry-panel-tabs">
                <button
                  className={`telemetry-tab ${sideTab === 'twin' ? 'active' : ''}`}
                  onClick={() => setSideTab('twin')}
                >
                  Digital Twin Telemetry
                </button>
                <button
                  className={`telemetry-tab ${sideTab === 'history' ? 'active' : ''}`}
                  onClick={() => setSideTab('history')}
                >
                  Chat History
                </button>
                <button className="panel-close-btn" onClick={() => setPanelOpen(false)} title="Hide panel">
                  &times;
                </button>
              </div>

              <div className="telemetry-panel-body">
                {sideTab === 'twin' ? (
                  <TwinView refreshKey={refreshKey} />
                ) : (
                  <HistoryView onOpen={handleOpenChat} refreshKey={refreshKey} />
                )}
              </div>
            </aside>
          )}
        </div>
      </main>
    </div>
  )
}

