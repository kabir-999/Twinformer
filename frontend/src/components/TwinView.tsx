import { useEffect, useState } from 'react'
import { api } from '../api/client'
import type { Architecture, TwinStatus } from '../types'
import { message } from './format'

const names: Record<Architecture, string> = { encoder_only: 'Encoder-only', decoder_only: 'Decoder-only', encoder_decoder: 'Encoder-decoder' }
const colors: Record<Architecture, string> = { encoder_only: '#2b5fd9', decoder_only: '#c8691c', encoder_decoder: '#2f8f5b' }
const sec = (ms: number) => (ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${Math.round(ms)} ms`)

function Scatter({ points }: { points: TwinStatus['points'] }) {
  const W = 420, H = 300, pad = 44
  const max = Math.max(1, ...points.flatMap(p => [p.actual_ms, p.predicted_ms])) / 1000 * 1.08
  const x = (v: number) => pad + (v / max) * (W - pad - 12), y = (v: number) => H - pad - (v / max) * (H - pad - 12)
  const ticks = [0, max / 2, max]
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Predicted versus actual time" style={{ width: '100%', maxWidth: 480 }}>
      {ticks.map(t => <g key={t}><line x1={pad} x2={W - 12} y1={y(t)} y2={y(t)} stroke="var(--line)" /><text x={pad - 6} y={y(t) + 4} textAnchor="end" fontSize="10" fill="var(--muted)">{t.toFixed(1)}</text><text x={x(t)} y={H - pad + 14} textAnchor="middle" fontSize="10" fill="var(--muted)">{t.toFixed(1)}</text></g>)}
      <line x1={x(0)} y1={y(0)} x2={x(max)} y2={y(max)} stroke="var(--muted)" strokeDasharray="4 4" />
      {points.map((p, i) => <circle key={i} cx={x(p.actual_ms / 1000)} cy={y(p.predicted_ms / 1000)} r="4" fill={colors[p.architecture]} fillOpacity=".75"><title>{`${names[p.architecture]}: actual ${sec(p.actual_ms)}, twin ${sec(p.predicted_ms)}`}</title></circle>)}
      <text x={W / 2} y={H - 6} textAnchor="middle" fontSize="11" fill="var(--muted)">actual time (seconds)</text>
      <text transform={`translate(12 ${H / 2}) rotate(-90)`} textAnchor="middle" fontSize="11" fill="var(--muted)">twin’s prediction (seconds)</text>
    </svg>
  )
}

export function TwinView() {
  const [status, setStatus] = useState<TwinStatus>()
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const load = () => api.twinStatus().then(setStatus).catch(e => setError(message(e)))
  useEffect(() => { load() }, [])
  const relearn = async () => { setBusy(true); setError(''); try { await api.train(); await load() } catch (e) { setError(message(e)) } finally { setBusy(false) } }
  const cal = status?.calibration
  const trained = cal?.status === 'trained' && cal.baseline_latency && cal.calibrated_latency
  return (
    <>
      <h1>Digital twin</h1>
      <p className="muted">A digital twin is a model of a real system that you keep in step with measurements. This one models how long the language models take to run on <em>this</em> computer. It starts from a physics-based guess, then learns from every answer you generate in Chat.</p>
      {error && <p className="error">{error}</p>}
      {status && (
        <>
          <div className="card">
            <p style={{ margin: 0 }}><strong>{status.chat_records}</strong> timed runs from your chats · <strong>{status.measured_total}</strong> real measurements in total · <strong>{status.lengths_seen}</strong> different input lengths</p>
            {trained ? (
              <p style={{ margin: '8px 0 0' }} className="ok">On longer inputs it had never seen, the typical error fell from {sec(cal!.baseline_latency!.mae)} (first-principles guess) to {sec(cal!.calibrated_latency!.mae)} (learned from your runs). {cal!.held_out_record_count < 10 ? 'That test is small, so treat it as a rough guide.' : ''}</p>
            ) : (
              <p style={{ margin: '8px 0 0' }} className="muted">Not learned yet. It needs at least 8 measured runs covering 3 or more different input lengths. Ask a few more questions in Chat and it will start learning automatically.</p>
            )}
            <div className="row" style={{ marginBottom: 0 }}><button onClick={relearn} disabled={busy}>{busy ? 'Learning…' : 'Re-learn now'}</button></div>
          </div>

          {status.corrections.length > 0 && (
            <>
              <h2>What it learned about this computer</h2>
              <p className="small muted">The first guess assumes a typical laptop. These factors show how far off that was: 1.0 means the assumption was right, 2.0 means the real machine is twice as slow for that step.</p>
              <table>
                <thead><tr><th>Model type</th><th className="num">Reading the prompt</th><th className="num">Writing each token</th></tr></thead>
                <tbody>{status.corrections.map(c => <tr key={c.architecture}><td>{names[c.architecture]}</td><td className="num">×{c.prefill_factor}</td><td className="num">×{c.decode_factor}</td></tr>)}</tbody>
              </table>
            </>
          )}

          {status.points.length > 0 && (
            <>
              <h2>Prediction vs. reality</h2>
              <p className="small muted">Each dot is one answer. A dot on the dashed line means the twin predicted the time exactly. These runs were also used for learning, so this shows how well it fits; the held-out result above is the honest accuracy.</p>
              <div className="card"><Scatter points={status.points} />
                <p className="small" style={{ margin: '4px 0 0' }}>{(Object.keys(colors) as Architecture[]).filter(a => status.points.some(p => p.architecture === a)).map(a => <span key={a} style={{ marginRight: 14 }}><span className="dot" style={{ background: colors[a] }} />{names[a]}</span>)}</p></div>
              <h2>Recent runs</h2>
              <table>
                <thead><tr><th>Model</th><th className="num">Tokens in / out</th><th className="num">Actual</th><th className="num">Twin</th></tr></thead>
                <tbody>{status.points.slice(-8).reverse().map((p, i) => <tr key={i}><td>{p.model_id.split('/').pop()}</td><td className="num">{p.input_tokens} / {p.output_tokens}</td><td className="num">{sec(p.actual_ms)}</td><td className="num">{sec(p.predicted_ms)}</td></tr>)}</tbody>
              </table>
            </>
          )}
          {!status.points.length && <p className="muted">No timed runs yet. Ask a question in Chat (compare-all-three mode) and the decoder-only and encoder-decoder timings will appear here.</p>}
          <details><summary>How it works</summary>
            <p className="small muted">The first guess assumes prefill is compute-bound (about 800 GFLOP/s) and each generated token streams the model weights from memory (about 50 GB/s), using each model’s real parameter count. Calibration is a ridge regression on those two physics terms plus lengths and model identity, trained only on measured runs, and tested on input lengths longer than any it trained on. The assumed speeds were picked from rough early timings on this machine, so the first guess is better than a blind one and the learning gain here is understated. Cold runs (when a model had to be loaded) are excluded, as are encoder-only answers, which are not generation. It is specific to this machine, these models and batch size 1.</p>
          </details>
        </>
      )}
    </>
  )
}
