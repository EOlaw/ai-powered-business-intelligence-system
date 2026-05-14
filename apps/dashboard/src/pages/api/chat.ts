import type { NextApiRequest, NextApiResponse } from 'next';

type ChatMessage = {
  role: 'system' | 'user' | 'assistant';
  content: string;
};

type ChatRequestBody = {
  messages?: ChatMessage[];
  maxTokens?: number;
  temperature?: number;
};

type ChatResponseBody = {
  answer?: string;
  model?: string;
  usage?: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
  error?: string;
};

const AI_ENGINE_URL = process.env['AI_ENGINE_URL'] ?? 'http://localhost:8001';
const INTERNAL_SECRET =
  process.env['SERVING_INTERNAL_API_SECRET'] ??
  process.env['AI_ENGINE_INTERNAL_SECRET'] ??
  'change-me-in-production';

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse<ChatResponseBody>,
) {
  if (req.method !== 'POST') {
    res.setHeader('Allow', 'POST');
    return res.status(405).json({ error: 'Method not allowed' });
  }

  const body = req.body as ChatRequestBody;
  const messages = Array.isArray(body.messages) ? body.messages : [];

  const cleanMessages = messages
    .filter(message => message && typeof message.content === 'string')
    .map(message => ({
      role: message.role,
      content: message.content.trim(),
    }))
    .filter(message => message.content.length > 0);

  if (cleanMessages.length === 0) {
    return res.status(400).json({ error: 'Message is required' });
  }

  try {
    const upstream = await fetch(`${AI_ENGINE_URL}/v1/chat/completions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${INTERNAL_SECRET}`,
      },
      body: JSON.stringify({
        model: 'insightserenity-1',
        messages: cleanMessages,
        max_tokens: body.maxTokens ?? 80,
        temperature: body.temperature ?? 0.7,
        stream: false,
      }),
    });

    const payload = await upstream.json().catch(() => null);

    if (!upstream.ok) {
      const detail =
        typeof payload?.detail === 'string'
          ? payload.detail
          : payload?.error?.message ?? `AI engine returned ${upstream.status}`;
      return res.status(upstream.status).json({ error: detail });
    }

    const answer = payload?.choices?.[0]?.message?.content;
    if (typeof answer !== 'string') {
      return res.status(502).json({ error: 'AI engine returned an unexpected response' });
    }

    return res.status(200).json({
      answer,
      model: payload.model,
      usage: payload.usage,
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown error';
    return res.status(502).json({
      error: `Could not reach the AI engine at ${AI_ENGINE_URL}. ${message}`,
    });
  }
}
