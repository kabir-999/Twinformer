import { useEffect, useRef, useState } from 'react'
import { api } from '../api/client'
import type { Architecture, ArchAnswer, Chat, ChatTurn, RagIndexStatus, TwinEstimate } from '../types'
import { FilePicker } from './FilePicker'
import { message } from './format'

const names: Record<Architecture, string> = { encoder_only: 'Encoder-only', decoder_only: 'Decoder-only', encoder_decoder: 'Encoder-decoder' }
const colors: Record<Architecture, string> = { encoder_only: '#2b5fd9', decoder_only: '#c8691c', encoder_decoder: '#2f8f5b' }
const short = (id: string) => id.split('/').pop() ?? id
const fileName = (d: string) => d.split('/').pop() ?? d

function AnswerAccordion({ a, open, best }: { a: ArchAnswer; open: boolean; best: boolean }) {
  const text = a.error ?? a.answer ?? 'No answer.'
  const accuracy = a.predicted_seconds != null && a.seconds != null && a.seconds > 0
    ? Math.max(0, Math.min(100, Math.round((1 - Math.abs(a.seconds - a.predicted_seconds) / Math.max(a.seconds, a.predicted_seconds)) * 100)))
    : null

  return (
    <details className="acc" open={open}>
      <summary>
        <span><span className="dot" style={{ background: colors[a.architecture] }} /><strong>{names[a.architecture]}</strong></span>
        <span className="muted small">
          {short(a.model_id)}
          {a.seconds != null ? ` · ${a.seconds}s` : ''}
          {a.predicted_seconds != null ? ` (twin predicted ${a.predicted_seconds}s)` : ''}
          {a.cold_start ? ' · cold start' : ''}
        </span>
        {best && <span className="best">Best</span>}
        <span className="preview">{text.replace(/\s+/g, ' ')}</span>
      </summary>
      <div className="body">
        {a.error ? <p className="error small">{a.error}</p> : <div className="answer-text">{a.answer}</div>}

        {/* Digital Twin Telemetry Card */}
        {a.seconds != null && (
          <div className="twin-telemetry-card">
            <div className="twin-telemetry-head">
              <span className="twin-telemetry-title">Digital Twin Telemetry</span>
              {accuracy != null && <span className="twin-accuracy-tag">{accuracy}% Accuracy</span>}
            </div>
            <div className="twin-telemetry-stats">
              <div className="twin-stat-col">
                <span className="twin-stat-label">Twin Predicted</span>
                <span className="twin-stat-val">{a.predicted_seconds != null ? `${a.predicted_seconds}s` : '—'}</span>
                <span className="twin-stat-hint">Physics + Ridge</span>
              </div>
              <div className="twin-stat-col actual">
                <span className="twin-stat-label">Actual Measured</span>
                <span className="twin-stat-val highlight">{a.seconds}s</span>
                <span className="twin-stat-hint">Hardware Timer</span>
              </div>
              <div className="twin-stat-col">
                <span className="twin-stat-label">Delta</span>
                <span className="twin-stat-val">
                  {a.predicted_seconds != null
                    ? `${(a.seconds - a.predicted_seconds) > 0 ? '+' : ''}${(a.seconds - a.predicted_seconds).toFixed(2)}s`
                    : '—'}
                </span>
                <span className="twin-stat-hint">
                  {a.predicted_seconds != null && Math.abs(a.seconds - a.predicted_seconds) < 0.25 ? 'High precision' : 'Calibration target'}
                </span>
              </div>
            </div>

            {a.predicted_seconds != null && a.seconds > 0 && (
              <div className="twin-bar-container">
                <div className="twin-bar-row">
                  <span className="bar-tag">Predicted ({a.predicted_seconds}s)</span>
                  <div className="bar-track">
                    <div className="bar-fill predicted" style={{ width: `${Math.min(100, (a.predicted_seconds / Math.max(a.seconds, a.predicted_seconds, 0.01)) * 100)}%` }} />
                  </div>
                </div>
                <div className="twin-bar-row">
                  <span className="bar-tag">Actual ({a.seconds}s)</span>
                  <div className="bar-track">
                    <div className="bar-fill actual" style={{ width: `${Math.min(100, (a.seconds / Math.max(a.seconds, a.predicted_seconds, 0.01)) * 100)}%` }} />
                  </div>
                </div>
              </div>
            )}

            <div className="twin-telemetry-footer">
              {a.cold_start ? (
                <span className="muted">Cold start (model weight loading).</span>
              ) : (
                <span className="ok">Hardware timing calibrated into digital twin.</span>
              )}
            </div>
          </div>
        )}

        {a.metrics && <div className="small muted" style={{ marginTop: 8 }}>
          {a.metrics.grounding != null && <>Matches the sources {Math.round(a.metrics.grounding * 100)}%</>}{a.metrics.relevance != null && <> · relevance {a.metrics.relevance.toFixed(2)}</>}
        </div>}
      </div>
    </details>
  )
}

function TurnView({ turn }: { turn: ChatTurn }) {
  const [openSources, setOpenSources] = useState<Set<number>>(new Set())
  const { result, route } = turn
  const verdict = result.verdicts[0]
  const winner = verdict?.winner ?? null
  const single = result.answers.length === 1
  const shown = (a: ArchAnswer) => single || (winner ? a.architecture === winner : a.architecture === 'decoder_only')
  return (
    <>
      <div className="bubble">{turn.question}</div>
      {route && <p className="small" style={{ margin: '4px 0 8px' }}><strong>Auto-picked {route.architectures.map(a => names[a]).join(' + ')}.</strong></p>}
      {result.answers.map(a => <AnswerAccordion key={a.architecture} a={a} open={shown(a)} best={winner === a.architecture && !single} />)}

      <details className="acc">
        <summary><strong>Sources</strong><span className="muted small">{result.sources.length} passages from {new Set(result.sources.map(s => s.document)).size} file(s)</span></summary>
        <div className="body">
          {result.sources.map(s => (
            <div className="src-item" key={s.rank}>
              <div className="src-head"><strong>[{s.rank}] {fileName(s.document)}</strong>{s.location && <span className="pill">{s.location}</span>}</div>
              <p className={openSources.has(s.rank) ? 'small' : 'small clamp'} style={{ margin: '4px 0', whiteSpace: 'pre-wrap' }}>{s.text}</p>
              <button className="link" onClick={() => setOpenSources(prev => { const n = new Set(prev); n.has(s.rank) ? n.delete(s.rank) : n.add(s.rank); return n })}>{openSources.has(s.rank) ? 'Show less' : 'Show more'}</button>
            </div>
          ))}
        </div>
      </details>

      {verdict && winner && (
        <div className="final">
          <div><strong>Best model: <span style={{ color: colors[winner] }}>{names[winner]}</span></strong></div>
          <div className="small" style={{ margin: '4px 0' }}>{verdict.summary}</div>
        </div>
      )}
    </>
  )
}

export function ChatView({ chat, onChat, onTurnComplete }: { chat: Chat | null; onChat: (c: Chat | null) => void; onTurnComplete?: () => void }) {
  const [docs, setDocs] = useState<string[]>([])
  const [selected, setSelected] = useState<Set<string>>(() => { try { return new Set(JSON.parse(localStorage.getItem('selectedFiles') ?? '[]')) } catch { return new Set() } })
  const [showFiles, setShowFiles] = useState(false)
  const [mode, setMode] = useState<'all' | 'auto'>('auto')
  const [allowDownload, setAllowDownload] = useState(false)
  const [question, setQuestion] = useState('')
  const [status, setStatus] = useState<RagIndexStatus>()
  const [groq, setGroq] = useState<boolean>()
  const [estimate, setEstimate] = useState<TwinEstimate[]>([])
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const end = useRef<HTMLDivElement>(null)

  const refresh = () => api.ragStatus().then(s => { setStatus(s); if (s.indexed) api.documents().then(setDocs).catch(() => undefined) }).catch(e => setError(message(e)))
  useEffect(() => { api.twinEstimate(4).then(setEstimate).catch(() => undefined) }, [chat?.turns.length])
  useEffect(() => { refresh(); api.judgeStatus().then(s => setGroq(s.groq_configured)).catch(() => undefined) }, [])
  useEffect(() => { if (status?.state !== 'building') return; const t = setInterval(refresh, 1500); return () => clearInterval(t) }, [status?.state])
  useEffect(() => { try { localStorage.setItem('selectedFiles', JSON.stringify([...selected])) } catch { /* private mode */ } }, [selected])
  useEffect(() => { if (chat) setSelected(new Set(chat.documents.filter(d => docs.length === 0 || docs.includes(d)))) }, [chat?.id])
  useEffect(() => { if (!showFiles) return; const onKey = (e: KeyboardEvent) => e.key === 'Escape' && setShowFiles(false); window.addEventListener('keydown', onKey); return () => window.removeEventListener('keydown', onKey) }, [showFiles])
  useEffect(() => { end.current?.scrollIntoView({ behavior: 'smooth', block: 'end' }) }, [chat?.turns.length, busy])

  const files = [...selected].filter(f => docs.includes(f))
  const send = async () => {
    const q = question.trim()
    if (!q || busy) return
    setBusy(true); setError('')
    try {
      const updated = await api.chat({ chat_id: chat?.id ?? null, question: q, documents: files, mode, top_k: 4, allow_model_download: allowDownload })
      onChat(updated); setQuestion('')
      onTurnComplete?.()
    } catch (e) { setError(message(e)) } finally { setBusy(false) }
  }
  const ready = status?.indexed
  const building = status?.state === 'building'

  return (
    <>
      {!ready && (
        <div className="card">
          {building ? <><p className="small" style={{ marginTop: 0 }}>{status?.stage}{status?.total ? ` (${status.done} of ${status.total})` : ''}</p><progress value={status?.done} max={status?.total || 1} /></>
            : <><p style={{ marginTop: 0 }}>Your documents are not indexed yet. This reads every file once and can take a few minutes.</p><button className="primary" onClick={() => api.ragIndex().then(setStatus).catch(e => setError(message(e)))}>Index documents</button></>}
        </div>
      )}

      {chat?.turns.length ? chat.turns.map(t => <TurnView key={t.id} turn={t} />) : (
        <div style={{ margin: '40px 0 24px' }}>
          <h1>Ask your documents</h1>
          <p className="muted">Pick the files to search and ask a question.</p>
        </div>
      )}
      {busy && <p className="muted small">Searching files and running models…</p>}
      {error && <p className="error">{error}</p>}
      <div ref={end} />

      <div className="composer">
        {estimate.length > 0 && (
          <div className="composer-twin-forecast">
            <span className="forecast-title">Twin Estimated Latency:</span>
            <span className="forecast-items">
              {estimate.map(e => (
                <span key={e.architecture} className="forecast-pill">
                  <strong>{names[e.architecture]}</strong> ≈ {e.seconds < 0.1 ? '<0.1' : e.seconds}s
                </span>
              ))}
            </span>
          </div>
        )}

        <div className="row" style={{ marginBottom: 8 }}>
          <button onClick={() => setShowFiles(!showFiles)} disabled={!ready}>Choose files · {files.length ? `${files.length} of ${docs.length} selected` : `all ${docs.length || ''} files`}</button>

          <label className="inline small" title="Fetch a model from Hugging Face once if it is not on this computer (for example a translation model for a new language)"><input type="checkbox" checked={allowDownload} onChange={e => setAllowDownload(e.target.checked)} />Allow downloads</label>
          {chat && <button onClick={() => onChat(null)}>New chat</button>}
        </div>
        {showFiles && (
          <div className="overlay" onClick={() => setShowFiles(false)} role="presentation">
            <div className="modal" role="dialog" aria-label="Choose files" onClick={e => e.stopPropagation()}>
              <div className="row" style={{ margin: '0 0 4px', justifyContent: 'space-between' }}>
                <strong>Choose files to search</strong>
                <button onClick={() => setShowFiles(false)} aria-label="Close">&times;</button>
              </div>
              <FilePicker docs={docs} selected={selected} onChange={setSelected} />
              <div className="row" style={{ margin: '10px 0 0' }}>
                <button className="primary" onClick={() => setShowFiles(false)}>Done</button>
                <span className="small muted">{files.length ? `${files.length} of ${docs.length} files selected` : `All ${docs.length} files searched`}</span>
              </div>
            </div>
          </div>
        )}
        <form className="row" onSubmit={e => { e.preventDefault(); send() }}>
          <label className="search" style={{ flex: 1 }}><span className="muted small">{chat ? 'Ask a follow-up' : 'Your question'}</span>
            <input value={question} onChange={e => setQuestion(e.target.value)} placeholder={chat ? 'e.g. Can you explain that more simply?' : 'e.g. What is knowledge distillation?'} disabled={!ready} /></label>
          <button className="primary" disabled={!ready || busy || !question.trim()}>{busy ? 'Working…' : 'Send'}</button>
        </form>
      </div>
    </>
  )
}
