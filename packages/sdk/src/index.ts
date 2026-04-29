/**
 * InsightSerenity SDK
 * ====================
 * Official TypeScript/JavaScript SDK for the InsightSerenity AI platform.
 *
 * Usage:
 *   import InsightSerenity from '@insightserenity/sdk';
 *
 *   const client = new InsightSerenity({ apiKey: 'is_sk_...' });
 *
 *   // Text completion
 *   const completion = await client.completions.create({
 *     prompt:      'Explain quantum computing in simple terms:',
 *     max_tokens:  256,
 *   });
 *   console.log(completion.choices[0].text);
 *
 *   // Chat completion (streaming)
 *   const stream = client.chat.completions.create({
 *     messages: [{ role: 'user', content: 'Write a haiku about AI.' }],
 *     stream:   true,
 *   });
 *   for await (const chunk of stream) {
 *     process.stdout.write(chunk.choices[0].delta.content ?? '');
 *   }
 *
 *   // Embeddings
 *   const { data } = await client.embeddings.create({
 *     input: ['Hello world', 'Goodbye world'],
 *   });
 *   console.log(data[0].embedding.length);  // e.g. 768
 *
 *   // Agents
 *   const result = await client.agents.run({
 *     task: 'What is the capital of France and what is its population?',
 *   });
 *   console.log(result.answer);
 */

import { HttpClient }        from './client.js';
import { Completions }       from './resources/completions.js';
import { Chat }              from './resources/chat.js';
import { EmbeddingsResource } from './resources/embeddings.js';
import { Agents }            from './resources/agents.js';

export type { InsightSerenityClientOptions } from './types/index.js';
export * from './types/index.js';
export { InsightSerenityError } from './client.js';

import type { InsightSerenityClientOptions } from './types/index.js';

// ─────────────────────────────────────────────────────────────────────────────
// Main client
// ─────────────────────────────────────────────────────────────────────────────

export class InsightSerenity extends HttpClient {
  readonly completions: Completions;
  readonly chat:        Chat;
  readonly embeddings:  EmbeddingsResource;
  readonly agents:      Agents;

  constructor(options: InsightSerenityClientOptions) {
    if (!options.apiKey) {
      throw new Error('InsightSerenity SDK: apiKey is required');
    }
    super(options);
    this.completions = new Completions(this);
    this.chat        = new Chat(this);
    this.embeddings  = new EmbeddingsResource(this);
    this.agents      = new Agents(this);
  }
}

export default InsightSerenity;
