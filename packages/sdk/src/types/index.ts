/**
 * InsightSerenity SDK — Public Types
 * ====================================
 * All types mirror the API Gateway's request/response schemas.
 * Naming follows the OpenAI SDK convention for drop-in compatibility.
 */

// ─────────────────────────────────────────────────────────────────────────────
// Client configuration
// ─────────────────────────────────────────────────────────────────────────────

export interface InsightSerenityClientOptions {
  /** Your INSIGHTSERENITY_API_KEY (begins with is_sk_) */
  apiKey:   string;
  /** Base URL of the API Gateway. Default: https://api.insightserenity.com */
  baseURL?: string;
  /** Request timeout in milliseconds. Default: 60000 */
  timeout?: number;
  /** Max retries on 429 or 5xx responses. Default: 2 */
  maxRetries?: number;
}

// ─────────────────────────────────────────────────────────────────────────────
// Completions
// ─────────────────────────────────────────────────────────────────────────────

export interface CompletionCreateParams {
  prompt:       string | string[];
  model?:       string;
  max_tokens?:  number;
  temperature?: number;
  top_p?:       number;
  top_k?:       number;
  stop?:        string[];
  stream?:      boolean;
  echo?:        boolean;
}

export interface CompletionChoice {
  text:          string;
  index:         number;
  finish_reason: 'stop' | 'length' | null;
}

export interface CompletionUsage {
  prompt_tokens:     number;
  completion_tokens: number;
  total_tokens:      number;
}

export interface Completion {
  id:      string;
  object:  'text_completion';
  created: number;
  model:   string;
  choices: CompletionChoice[];
  usage:   CompletionUsage;
}

// ─────────────────────────────────────────────────────────────────────────────
// Chat
// ─────────────────────────────────────────────────────────────────────────────

export interface ChatMessage {
  role:    'system' | 'user' | 'assistant';
  content: string;
  name?:   string;
}

export interface ChatCompletionCreateParams {
  messages:     ChatMessage[];
  model?:       string;
  max_tokens?:  number;
  temperature?: number;
  top_p?:       number;
  top_k?:       number;
  stop?:        string[];
  stream?:      boolean;
}

export interface ChatChoice {
  index:         number;
  message:       ChatMessage;
  finish_reason: 'stop' | 'length' | null;
}

export interface ChatCompletion {
  id:      string;
  object:  'chat.completion';
  created: number;
  model:   string;
  choices: ChatChoice[];
  usage:   CompletionUsage;
}

// Streaming chunk types
export interface ChatCompletionChunkDelta {
  role?:    string;
  content?: string;
}

export interface ChatCompletionChunk {
  id:      string;
  object:  'chat.completion.chunk';
  created: number;
  model:   string;
  choices: Array<{ index: number; delta: ChatCompletionChunkDelta; finish_reason: string | null }>;
}

// ─────────────────────────────────────────────────────────────────────────────
// Embeddings
// ─────────────────────────────────────────────────────────────────────────────

export interface EmbeddingCreateParams {
  input:            string | string[];
  model?:           string;
  encoding_format?: 'float' | 'base64';
  dimensions?:      number;
}

export interface EmbeddingObject {
  object:    'embedding';
  embedding: number[];
  index:     number;
}

export interface Embeddings {
  object: 'list';
  data:   EmbeddingObject[];
  model:  string;
  usage:  { prompt_tokens: number; total_tokens: number };
}

// ─────────────────────────────────────────────────────────────────────────────
// Agents
// ─────────────────────────────────────────────────────────────────────────────

export interface AgentRunParams {
  task:          string;
  model?:        string;
  max_steps?:    number;
  stream?:       boolean;
  tools?:        string[];
  is_reflection?: boolean;
  context?:       string;
}

export interface AgentStep {
  step_num:     number;
  thought:      string;
  action?:      string;
  action_input?: string;
  observation?:  string;
  is_final:     boolean;
}

export interface AgentRun {
  id:           string;
  task:         string;
  answer?:      string;
  success:      boolean;
  steps:        AgentStep[];
  total_steps:  number;
  elapsed_secs: number;
  model:        string;
}

// ─────────────────────────────────────────────────────────────────────────────
// Errors
// ─────────────────────────────────────────────────────────────────────────────

export interface APIErrorBody {
  success: false;
  error:   { code: string; message: string; details?: unknown };
}
