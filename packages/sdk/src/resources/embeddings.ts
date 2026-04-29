/**
 * InsightSerenity SDK — Embeddings Resource
 * POST /v1/embeddings
 */

import { HttpClient }            from '../client.js';
import type { EmbeddingCreateParams, Embeddings } from '../types/index.js';

export class EmbeddingsResource {
  constructor(private readonly http: HttpClient) {}

  create(params: EmbeddingCreateParams): Promise<Embeddings> {
    return this.http.request<Embeddings>('POST', '/v1/embeddings', params);
  }
}
