/**
 * InsightSerenity SDK — Chat Resource
 * POST /v1/chat/completions
 */

import { HttpClient }    from '../client.js';
import type {
  ChatCompletionCreateParams,
  ChatCompletion,
  ChatCompletionChunk,
} from '../types/index.js';

export class ChatCompletions {
  constructor(private readonly http: HttpClient) {}

  create(params: ChatCompletionCreateParams & { stream: true }):  AsyncGenerator<ChatCompletionChunk>;
  create(params: ChatCompletionCreateParams & { stream?: false }): Promise<ChatCompletion>;
  create(params: ChatCompletionCreateParams): Promise<ChatCompletion> | AsyncGenerator<ChatCompletionChunk> {
    if (params.stream) {
      return this.http.stream<ChatCompletionChunk>('POST', '/v1/chat/completions', params);
    }
    return this.http.request<ChatCompletion>('POST', '/v1/chat/completions', params);
  }
}

export class Chat {
  readonly completions: ChatCompletions;

  constructor(http: HttpClient) {
    this.completions = new ChatCompletions(http);
  }
}
