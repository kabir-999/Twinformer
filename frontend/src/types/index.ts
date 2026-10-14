export type Architecture = 'encoder_only' | 'decoder_only' | 'encoder_decoder'
export type WorkloadType = 'encoder_inference' | 'autoregressive_generation' | 'encoder_decoder_generation' | 'multi_turn_chat'

export interface Workload {
  architecture: Architecture; model_id?: string; workload_type: WorkloadType
  input_tokens: number; requested_output_tokens: number; batch_size: number; concurrent_requests: number
  history_turns: number; cache_enabled: boolean; precision: 'fp32' | 'fp16' | 'bf16' | 'int8'
  warmup_runs?: number; repetitions?: number; device?: 'auto' | 'cpu' | 'cuda'; allow_model_download?: boolean
}

export interface BenchmarkRecord {
  id: string; created_at: string; source: 'synthetic' | 'measured'; benchmark_version: string; workload: Workload
  end_to_end_latency_ms: number | null; tokens_per_second: number | null; peak_memory_mb: number | null
  variability_ms: number | null; notes?: string
}
export interface Overview { record_count: number; synthetic_record_count: number; measured_record_count: number; calibration_status: string }
export interface LocalModel { id: string; kind: Architecture }

export interface TwinPrediction {
  prediction_source: 'analytical' | 'calibrated'; predicted_end_to_end_latency_ms: number; predicted_peak_memory_mb: number
  latency_interval_ms: { lower: number; upper: number }; calibration_sample_count: number
}
interface Errors { mae: number; rmse: number; mape_percent: number | null; sample_count: number }
export interface CalibrationReport {
  status: 'untrained' | 'trained' | 'insufficient_data'; measured_record_count: number; training_record_count: number
  held_out_record_count: number; split_description: string; baseline_latency?: Errors | null; calibrated_latency?: Errors | null
}

export interface ScalingStudy { axis: string; points: { value: number; end_to_end_latency_ms: number | null }[]; latency_scaling_exponent: number | null; notes: string[] }
export interface CacheComparison { cache_on_latency_ms: number | null; cache_off_latency_ms: number | null; speedup: number | null }
export interface MultiTurnStudy {
  model_id: string; total_speedup: number | null
  modes: { mode: 'recompute_history' | 'reuse_kv_cache'; total_ms: number; turns: { turn: number; prefill_ms: number }[] }[]
}

export interface RagSource { label: string; path: string; file_count: number }
export interface RagIndexStatus {
  indexed: boolean; state: 'idle' | 'building' | 'ready' | 'error'; stage: string | null; done: number; total: number; error: string | null
  skipped: string[]; documents_changed: boolean; sources: RagSource[]; document_count: number; chunk_count: number; built_at: string | null
}
export interface Citation { rank: number; document: string; chunk_id: string; location: string; score: number; text: string }
export interface RagAnswer { question: string; citations: Citation[]; answer: string | null; answer_model: string | null; answer_seconds: number | null; notes: string[] }
export interface RagEvalReport { case_count: number; hit_rate_at_k: Record<string, number>; mrr: number; per_case: { question: string; first_relevant_rank: number | null }[]; retrieval_latency_ms_median: number | null }

export interface Timings { tokenize_ms: number; encode_ms: number | null; decode_ms: number | null; total_ms: number }
export interface Heatmap { kind: 'self_bidirectional' | 'self_causal' | 'cross'; rows: string[]; cols: string[]; row_index: number[]; col_index: number[]; values: number[][]; layer: number; truncated: boolean }
export interface TokenStep { step: number; chosen: string; chosen_prob: number; alternatives: { token: string; prob: number }[] }
export interface ModelRun {
  architecture: Architecture; status: 'ok' | 'unavailable'; error: string | null; model_id: string; params_millions: number | null; weights_mb: number | null
  load_ms: number | null; prompt: string | null; input_tokens: string[]; input_token_count: number; input_truncated: boolean
  output_text: string | null; output_token_count: number; tokens_per_second: number | null; timings: Timings | null
  embedding_dim: number | null; embedding_preview: number[]; question_similarity: number | null; attention: Heatmap | null; token_steps: TokenStep[]
  related_passages: { document: string; location: string; text: string; score: number }[]
}
export interface CompareResult { id: string; runs: ModelRun[]; notes: string[]; request: { text: string; task: string } }
export interface CompareModels { local: Record<Architecture, string[]>; defaults: Record<Architecture, string>; missing_defaults: Partial<Record<Architecture, string>> }

export interface SourcePassage { rank: number; document: string; location: string; text: string; score: number }
export interface AnswerMetrics { grounding: number | null; relevance: number | null; repetition: number | null; words: number }
export interface ArchAnswer { architecture: Architecture; model_id: string; method: string; answer: string | null; seconds: number | null; error: string | null; metrics: AnswerMetrics | null; cold_start?: boolean; predicted_seconds?: number | null }
export interface Verdict { source: 'heuristic' | 'groq'; judge_model: string | null; winner: Architecture | null; summary: string; per_answer: { architecture: Architecture; score: number; comment: string }[]; caveat: string }
export interface ThreeWayAnswer { question: string; sources: SourcePassage[]; answers: ArchAnswer[]; verdicts: Verdict[]; notes: string[] }

export interface Route { intent: string; architectures: Architecture[]; reason: string; sentiment: boolean }
export interface ChatTurn { id: string; created_at: string; question: string; standalone_question: string; mode: 'all' | 'auto'; route: Route | null; result: ThreeWayAnswer }
export interface Chat { id: string; title: string; created_at: string; updated_at: string; documents: string[]; turns: ChatTurn[] }
export interface ChatSummary { id: string; title: string; updated_at: string; turn_count: number; document_count: number }

export interface TwinEstimate { architecture: Architecture; model_id: string; seconds: number; low: number; high: number; source: 'analytical' | 'calibrated'; samples: number }
export interface TwinPoint { created_at: string; architecture: Architecture; model_id: string; input_tokens: number | null; output_tokens: number | null; actual_ms: number; predicted_ms: number }
export interface TwinStatus { corrections: { architecture: Architecture; prefill_factor: number; decode_factor: number }[]; measured_total: number; chat_records: number; lengths_seen: number; points: TwinPoint[]; calibration: CalibrationReport }
