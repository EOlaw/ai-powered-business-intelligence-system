/**
 * InsightSerenity SDK — Completions Resource
 * POST /v1/completions
 */

import { HttpClient }    from '../client.js';
import type {
  CompletionCreateParams,
  Completion,
  ChatCompletionChunk,
} from '../types/index.js';

export class Completions {
  constructor(private readonly http: HttpClient) {}

  /** Create a text completion. Set stream: true to get an async iterator. */
  create(params: CompletionCreateParams & { stream: true }):  AsyncGenerator<ChatCompletionChunk>;
  create(params: CompletionCreateParams & { stream?: false }): Promise<Completion>;
  create(params: CompletionCreateParams): Promise<Completion> | AsyncGenerator<ChatCompletionChunk> {
    if (params.stream) {
      return this.http.stream<ChatCompletionChunk>('POST', '/v1/completions', params);
    }
    return this.http.request<Completion>('POST', '/v1/completions', params);
  }
}
