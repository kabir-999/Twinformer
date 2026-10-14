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

export function TwinView({ refreshKey }: { refreshKey?: number }) {
  const [status, setStatus] = useState<TwinStatus>()
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const load = () => api.twinStatus().then(setStatus).catch(e => setError(message(e)))
  useEffect(() => { load() }, [refreshKey])
  const relearn = async () => { setBusy(true); setError(''); try { await api.train(); await load() } catch (e) { setError(message(e)) } finally { setBusy(false) } }
  const cal = status?.calibration
  const trained = cal?.status === 'trained' && cal.baseline_latency && cal.calibrated_latency
  return (
    <div className="twin-view-container">
      <div className="twin-header-row">
        <div>
          <h2 style={{ margin: '0 0 4px', fontSize: '1.15rem' }}>Digital Twin Telemetry</h2>
          <p className="muted small" style={{ margin: 0 }}>Real-time hardware performance model of this machine</p>
        </div>
        <button className="small" onClick={relearn} disabled={busy || !status?.points?.length} title="Re-run ridge regression on all recorded chat runs">
          {busy ? 'Calibrating…' : 'Re-calibrate'}
        </button>
      </div>

      {error && <p className="error small">{error}</p>}

      {status && (
        <>
          <div className="twin-stat-grid">
            <div className="twin-stat-card">
              <span className="stat-label">Chat Runs Timed</span>
              <span className="stat-number">{status.chat_records}</span>
              <span className="stat-sub">{status.measured_total} total measurements</span>
            </div>
            <div className="twin-stat-card">
              <span className="stat-label">Calibration State</span>
              <span className={`stat-badge ${trained ? 'badge-ok' : 'badge-pending'}`}>
                {trained ? 'Calibrated' : 'Gathering Data'}
              </span>
              <span className="stat-sub">{status.lengths_seen} token lengths tested</span>
            </div>
          </div>

          <div className="card" style={{ padding: '10px 12px', margin: '8px 0' }}>
            {trained ? (
              <div className="small">
                <strong className="ok">Machine Profile Active:</strong> On unseen token lengths, prediction error dropped from{' '}
                <strong>{sec(cal!.baseline_latency!.mae)}</strong> (physics prior) down to{' '}
                <strong>{sec(cal!.calibrated_latency!.mae)}</strong> (calibrated to your CPU).
              </div>
            ) : (
              <div className="small muted">
                <strong>Learning in progress:</strong> The twin automatically fits to your hardware after 8 measured chat runs. Every question you ask trains the model.
              </div>
            )}
          </div>

          {status.points.length > 0 && (
            <div className="twin-section">
              <h3 className="small-title">Prediction vs. Reality (Seconds)</h3>
              <p className="small muted" style={{ margin: '2px 0 6px' }}>Points near the dashed diagonal mean exact timing prediction.</p>
              <div className="card" style={{ padding: '8px', textAlign: 'center' }}>
                <Scatter points={status.points} />
                <div className="small" style={{ margin: '6px 0 0', display: 'flex', justifyContent: 'center', gap: 12 }}>
                  {(Object.keys(colors) as Architecture[]).filter(a => status.points.some(p => p.architecture === a)).map(a => (
                    <span key={a} style={{ display: 'inline-flex', alignItems: 'center' }}>
                      <span className="dot" style={{ background: colors[a] }} />{names[a]}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          )}

          {status.corrections.length > 0 && (
            <div className="twin-section">
              <h3 className="small-title">Learned Hardware Factors</h3>
              <p className="small muted" style={{ margin: '2px 0 6px' }}>1.0× = standard baseline; &gt;1.0× = actual CPU/RAM execution cost.</p>
              <table className="compact-table">
                <thead><tr><th>Model</th><th className="num">Prompt Prefill</th><th className="num">Token Decode</th></tr></thead>
                <tbody>
                  {status.corrections.map(c => (
                    <tr key={c.architecture}>
                      <td><span className="dot" style={{ background: colors[c.architecture] }} />{names[c.architecture]}</td>
                      <td className="num">×{c.prefill_factor.toFixed(2)}</td>
                      <td className="num">×{c.decode_factor.toFixed(2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {status.points.length > 0 && (
            <div className="twin-section">
              <h3 className="small-title">Recent Run Accuracy</h3>
              <table className="compact-table">
                <thead><tr><th>Model</th><th className="num">Actual</th><th className="num">Twin Predicted</th><th className="num">Error</th></tr></thead>
                <tbody>
                  {status.points.slice(-5).reverse().map((p, i) => {
                    const delta = (p.actual_ms - p.predicted_ms) / 1000
                    return (
                      <tr key={i}>
                        <td>{p.model_id.split('/').pop()}</td>
                        <td className="num">{sec(p.actual_ms)}</td>
                        <td className="num">{sec(p.predicted_ms)}</td>
                        <td className={`num ${Math.abs(delta) < 0.25 ? 'ok' : 'muted'}`}>
                          {delta > 0 ? '+' : ''}{delta.toFixed(2)}s
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          )}

        </>
      )}
    </div>
  )
}

