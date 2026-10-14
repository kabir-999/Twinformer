import type {
  BenchmarkRecord, TwinEstimate, TwinStatus, Chat, ChatSummary, ThreeWayAnswer, CompareModels, CompareResult, CacheComparison, CalibrationReport, LocalModel, MultiTurnStudy, Overview, RagAnswer, RagEvalReport,
  RagIndexStatus, ScalingStudy, TwinPrediction, Workload,
} from '../types'

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`/api/v1${path}`, init)
  if (!response.ok) {
    const body = await response.json().catch(() => ({}))
    const detail = body.detail
    throw new Error(typeof detail === 'string' ? detail : detail ? 'The request was not valid. Check the values and try again.' : response.statusText)
  }
  return response.json()
}

const post = <T,>(path: string, body?: object) =>
  request<T>(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: body ? JSON.stringify(body) : undefined })

export const api = {
  twinEstimate: (topK = 4) => request<TwinEstimate[]>(`/twin/estimate?top_k=${topK}`),
  twinStatus: () => request<TwinStatus>('/twin/status'),
  chat: (body: object) => post<Chat>('/chat', body),
  chats: () => request<ChatSummary[]>('/chats'),
  getChat: (id: string) => request<Chat>(`/chats/${id}`),
  deleteChat: async (id: string) => { const r = await fetch(`/api/v1/chats/${id}`, { method: 'DELETE' }); if (!r.ok) throw new Error('Could not delete that chat.') },
  documents: () => request<string[]>('/rag/documents'),
  ragAnswers: (body: object) => post<ThreeWayAnswer>('/rag/answers', body),
  judgeStatus: () => request<{ groq_configured: boolean }>('/judge/status'),
  compareModels: () => request<CompareModels>('/compare/models'),
  compare: (body: object) => post<CompareResult>('/compare', body),
  overview: () => request<Overview>('/overview'),
  records: () => request<BenchmarkRecord[]>('/records'),
  localModels: () => request<LocalModel[]>('/models/local'),
  benchmark: (workload: Workload) => post<BenchmarkRecord>('/benchmarks', workload),
  scalingStudy: (body: object) => post<ScalingStudy>('/studies/scaling', body),
  cacheStudy: (body: object) => post<CacheComparison>('/studies/cache', body),
  multiTurnStudy: (body: object) => post<MultiTurnStudy>('/studies/multi-turn', body),
  predict: (workload: Workload) => post<TwinPrediction>('/twin/predict', workload),
  train: () => post<CalibrationReport>('/calibration/train'),
  ragStatus: () => request<RagIndexStatus>('/rag/index'),
  ragIndex: () => post<RagIndexStatus>('/rag/index', {}),
  ragQuery: (question: string, generate: boolean, modelId?: string) => post<RagAnswer>('/rag/query', { question, top_k: 4, generate, model_id: modelId }),
  ragEvaluate: async () => post<RagEvalReport>('/rag/evaluate', { cases: await request<object[]>('/rag/eval-cases') }),
  report: () => post<{ path: string; markdown: string }>('/research/report'),
}
