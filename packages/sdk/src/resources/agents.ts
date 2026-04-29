/**
 * InsightSerenity SDK — Agents Resource
 * POST /v1/agents/run
 */

import { HttpClient }         from '../client.js';
import type { AgentRunParams, AgentRun } from '../types/index.js';

interface AgentStreamEvent {
  event:        string;
  step_num?:    number;
  thought?:     string;
  action?:      string;
  action_input?: string;
  observation?: string;
  is_final?:    boolean;
  answer?:      string;
  success?:     boolean;
  total_steps?: number;
}

export class Agents {
  constructor(private readonly http: HttpClient) {}

  run(params: AgentRunParams & { stream: true }):  AsyncGenerator<AgentStreamEvent>;
  run(params: AgentRunParams & { stream?: false }): Promise<AgentRun>;
  run(params: AgentRunParams): Promise<AgentRun> | AsyncGenerator<AgentStreamEvent> {
    if (params.stream) {
      return this.http.stream<AgentStreamEvent>('POST', '/v1/agents/run', params);
    }
    return this.http.request<AgentRun>('POST', '/v1/agents/run', params);
  }
}
