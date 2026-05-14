import Head from 'next/head';
import { FormEvent, useEffect, useMemo, useRef, useState } from 'react';

type Message = {
  role: 'user' | 'assistant';
  content: string;
};

type Conversation = {
  id: string;
  title: string;
  messages: Message[];
  createdAt: string;
};

type ApiChatResponse = {
  answer?: string;
  error?: string;
  model?: string;
  usage?: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
};

const STARTER_MESSAGE: Message = {
  role: 'assistant',
  content: 'Ask a question and I will answer through your promoted InsightSerenity model.',
};

const PROMPT_LIBRARY = [
  {
    title: 'Business summary',
    prompt: 'Summarize the InsightSerenity business intelligence platform for a potential client.',
  },
  {
    title: 'Operations ideas',
    prompt: 'Give me five ways AI can improve operational reporting for a small business.',
  },
  {
    title: 'Model status',
    prompt: 'What should I know about this local model before relying on its answers?',
  },
  {
    title: 'Executive brief',
    prompt: 'Write a short executive brief about automating data analysis and reporting.',
  },
];

const DEFAULT_CONVERSATIONS: Conversation[] = [
  {
    id: 'first-chat',
    title: 'New client question',
    createdAt: 'Today',
    messages: [STARTER_MESSAGE],
  },
];

function conversationTitle(input: string) {
  const cleaned = input.replace(/\s+/g, ' ').trim();
  if (!cleaned) return 'New client question';
  return cleaned.length > 42 ? `${cleaned.slice(0, 42)}...` : cleaned;
}

export default function AskPage() {
  const [conversations, setConversations] = useState<Conversation[]>(DEFAULT_CONVERSATIONS);
  const [activeId, setActiveId] = useState(DEFAULT_CONVERSATIONS[0]?.id ?? 'first-chat');
  const [input, setInput] = useState('');
  const [isSending, setIsSending] = useState(false);
  const [error, setError] = useState('');
  const [usage, setUsage] = useState<ApiChatResponse['usage']>();
  const [modelName, setModelName] = useState('insightserenity-1');
  const [maxTokens, setMaxTokens] = useState(120);
  const [temperature, setTemperature] = useState(0.7);

  const inputRef = useRef<HTMLTextAreaElement | null>(null);
  const scrollRef = useRef<HTMLDivElement | null>(null);

  const activeConversation =
    conversations.find(conversation => conversation.id === activeId) ?? conversations[0];
  const messages = activeConversation?.messages ?? [STARTER_MESSAGE];
  const hasUserMessages = messages.some(message => message.role === 'user');

  const chatPayload = useMemo(
    () => messages.map(message => ({ role: message.role, content: message.content })),
    [messages],
  );

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' });
  }, [messages, isSending]);

  function updateActiveConversation(nextMessages: Message[], nextTitle?: string) {
    setConversations(current =>
      current.map(conversation =>
        conversation.id === activeId
          ? {
              ...conversation,
              title: nextTitle ?? conversation.title,
              messages: nextMessages,
            }
          : conversation,
      ),
    );
  }

  function startNewChat() {
    const id = `chat-${Date.now()}`;
    const next: Conversation = {
      id,
      title: 'New client question',
      createdAt: 'Now',
      messages: [STARTER_MESSAGE],
    };

    setConversations(current => [next, ...current]);
    setActiveId(id);
    setInput('');
    setError('');
    setUsage(undefined);
    window.setTimeout(() => inputRef.current?.focus(), 0);
  }

  async function sendMessage(text: string) {
    const question = text.trim();
    if (!question || isSending) return;

    const nextMessages: Message[] = [...messages, { role: 'user', content: question }];
    const nextTitle = hasUserMessages ? undefined : conversationTitle(question);
    updateActiveConversation(nextMessages, nextTitle);
    setInput('');
    setError('');
    setIsSending(true);

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: [...chatPayload, { role: 'user', content: question }],
          maxTokens,
          temperature,
        }),
      });

      const payload = (await res.json()) as ApiChatResponse;
      if (!res.ok || !payload.answer) {
        throw new Error(payload.error ?? 'The model did not return an answer.');
      }

      updateActiveConversation([...nextMessages, { role: 'assistant', content: payload.answer }], nextTitle);
      setUsage(payload.usage);
      if (payload.model) setModelName(payload.model);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Something went wrong.';
      setError(message);
      updateActiveConversation(
        [...nextMessages, { role: 'assistant', content: 'I could not reach the model server.' }],
        nextTitle,
      );
    } finally {
      setIsSending(false);
      window.setTimeout(() => inputRef.current?.focus(), 0);
    }
  }

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    void sendMessage(input);
  }

  return (
    <>
      <Head>
        <title>Ask InsightSerenity</title>
      </Head>

      <main className="min-h-screen bg-[#0a0d12] text-gray-100">
        <div className="grid min-h-screen lg:grid-cols-[280px_minmax(0,1fr)]">
          <aside className="hidden border-r border-gray-800 bg-[#10141d] lg:flex lg:flex-col">
            <div className="border-b border-gray-800 px-5 py-5">
              <div className="flex items-center justify-between gap-3">
                <div>
                  <p className="text-xs font-semibold uppercase tracking-[0.18em] text-cyan-300">InsightSerenity</p>
                  <h1 className="mt-2 text-lg font-semibold text-white">Client AI</h1>
                </div>
                <span className="rounded-md border border-emerald-700 bg-emerald-950 px-2 py-1 text-xs text-emerald-200">
                  Live
                </span>
              </div>
              <button
                type="button"
                onClick={startNewChat}
                className="mt-5 w-full rounded-md border border-gray-700 bg-gray-950 px-3 py-2.5 text-left text-sm font-medium text-gray-100 transition hover:border-cyan-600 hover:bg-gray-900"
              >
                + New conversation
              </button>
            </div>

            <div className="flex-1 overflow-y-auto px-3 py-4">
              <p className="px-2 text-xs font-medium uppercase tracking-[0.14em] text-gray-500">Recent</p>
              <div className="mt-3 space-y-1">
                {conversations.map(conversation => (
                  <button
                    key={conversation.id}
                    type="button"
                    onClick={() => setActiveId(conversation.id)}
                    className={`w-full rounded-md px-3 py-3 text-left transition ${
                      conversation.id === activeId
                        ? 'bg-cyan-950 text-white'
                        : 'text-gray-400 hover:bg-gray-900 hover:text-gray-100'
                    }`}
                  >
                    <span className="block truncate text-sm font-medium">{conversation.title}</span>
                    <span className="mt-1 block text-xs text-gray-500">{conversation.createdAt}</span>
                  </button>
                ))}
              </div>
            </div>

            <div className="border-t border-gray-800 p-4">
              <div className="rounded-md border border-gray-800 bg-gray-950 p-3">
                <p className="text-xs text-gray-500">Active model</p>
                <p className="mt-1 truncate text-sm font-medium text-gray-200">{modelName}</p>
              </div>
            </div>
          </aside>

          <section className="flex min-w-0 flex-col">
            <header className="flex flex-col gap-4 border-b border-gray-800 bg-[#0d1118]/95 px-4 py-4 backdrop-blur md:flex-row md:items-center md:justify-between lg:px-8">
              <div>
                <p className="text-xs font-semibold uppercase tracking-[0.18em] text-cyan-300">Ask anything</p>
                <h2 className="mt-1 text-xl font-semibold text-white md:text-2xl">Client Question Workspace</h2>
              </div>
              <div className="flex flex-wrap items-center gap-2 text-xs text-gray-400">
                <span className="rounded-md border border-gray-700 bg-gray-950 px-3 py-2">Engine: localhost:8001</span>
                <span className="rounded-md border border-gray-700 bg-gray-950 px-3 py-2">UI: localhost:3001</span>
              </div>
            </header>

            <div className="grid flex-1 min-h-0 xl:grid-cols-[minmax(0,1fr)_320px]">
              <div className="flex min-h-0 flex-col">
                <div ref={scrollRef} className="flex-1 overflow-y-auto px-4 py-6 lg:px-8">
                  {!hasUserMessages && (
                    <div className="mx-auto max-w-4xl py-8">
                      <div className="max-w-2xl">
                        <p className="text-sm font-medium text-cyan-300">Ready for client questions</p>
                        <h3 className="mt-3 text-3xl font-semibold tracking-tight text-white md:text-5xl">
                          Ask your local AI model.
                        </h3>
                        <p className="mt-4 max-w-xl text-base leading-7 text-gray-400">
                          Use this workspace to test answers, draft insights, and preview how clients will interact with the promoted model.
                        </p>
                      </div>

                      <div className="mt-8 grid gap-3 md:grid-cols-2">
                        {PROMPT_LIBRARY.map(item => (
                          <button
                            key={item.title}
                            type="button"
                            onClick={() => void sendMessage(item.prompt)}
                            disabled={isSending}
                            className="rounded-lg border border-gray-800 bg-[#10141d] p-4 text-left transition hover:border-cyan-700 hover:bg-[#151b26] disabled:opacity-50"
                          >
                            <span className="text-sm font-semibold text-white">{item.title}</span>
                            <span className="mt-2 block text-sm leading-6 text-gray-400">{item.prompt}</span>
                          </button>
                        ))}
                      </div>
                    </div>
                  )}

                  <div className="mx-auto max-w-4xl space-y-5">
                    {messages.map((message, index) => (
                      <article key={`${message.role}-${index}`} className="grid gap-3 md:grid-cols-[96px_minmax(0,1fr)]">
                        <div className="text-xs font-semibold uppercase tracking-[0.14em] text-gray-500">
                          {message.role === 'user' ? 'You' : 'AI'}
                        </div>
                        <div
                          className={`rounded-lg px-4 py-4 text-sm leading-7 shadow-sm ${
                            message.role === 'user'
                              ? 'border border-cyan-800 bg-cyan-950/70 text-cyan-50'
                              : 'border border-gray-800 bg-[#10141d] text-gray-100'
                          }`}
                        >
                          <p className="whitespace-pre-wrap break-words">{message.content}</p>
                        </div>
                      </article>
                    ))}

                    {isSending && (
                      <article className="grid gap-3 md:grid-cols-[96px_minmax(0,1fr)]">
                        <div className="text-xs font-semibold uppercase tracking-[0.14em] text-gray-500">AI</div>
                        <div className="rounded-lg border border-gray-800 bg-[#10141d] px-4 py-4 text-sm text-gray-400">
                          Thinking...
                        </div>
                      </article>
                    )}
                  </div>
                </div>

                {error && (
                  <div className="border-t border-red-900/70 bg-red-950/40 px-4 py-3 text-sm text-red-200 lg:px-8">
                    {error}
                  </div>
                )}

                <form onSubmit={handleSubmit} className="border-t border-gray-800 bg-[#0d1118] px-4 py-4 lg:px-8">
                  <div className="mx-auto max-w-4xl">
                    <div className="rounded-lg border border-gray-700 bg-gray-950 p-2 shadow-2xl shadow-black/20 focus-within:border-cyan-600">
                      <textarea
                        ref={inputRef}
                        value={input}
                        onChange={event => setInput(event.target.value)}
                        onKeyDown={event => {
                          if (event.key === 'Enter' && !event.shiftKey) {
                            event.preventDefault();
                            void sendMessage(input);
                          }
                        }}
                        rows={3}
                        className="max-h-40 min-h-20 w-full resize-none bg-transparent px-3 py-3 text-sm leading-6 text-white outline-none placeholder:text-gray-500"
                        placeholder="Ask a business question, request a summary, or test the model..."
                      />
                      <div className="flex flex-col gap-3 border-t border-gray-800 px-2 py-2 sm:flex-row sm:items-center sm:justify-between">
                        <div className="flex flex-wrap items-center gap-2 text-xs text-gray-500">
                          <span>Enter to send</span>
                          <span>Shift+Enter for newline</span>
                        </div>
                        <button
                          type="submit"
                          disabled={isSending || !input.trim()}
                          className="rounded-md bg-cyan-600 px-5 py-2.5 text-sm font-semibold text-white transition hover:bg-cyan-500 disabled:cursor-not-allowed disabled:opacity-50"
                        >
                          Send
                        </button>
                      </div>
                    </div>
                  </div>
                </form>
              </div>

              <aside className="hidden border-l border-gray-800 bg-[#10141d] xl:block">
                <div className="space-y-5 p-5">
                  <section className="rounded-lg border border-gray-800 bg-gray-950 p-4">
                    <h3 className="text-sm font-semibold text-white">Generation</h3>
                    <label className="mt-4 block text-xs font-medium text-gray-500" htmlFor="temperature">
                      Temperature: {temperature.toFixed(1)}
                    </label>
                    <input
                      id="temperature"
                      type="range"
                      min="0"
                      max="2"
                      step="0.1"
                      value={temperature}
                      onChange={event => setTemperature(Number(event.target.value))}
                      className="mt-2 w-full accent-cyan-500"
                    />
                    <label className="mt-4 block text-xs font-medium text-gray-500" htmlFor="maxTokens">
                      Max tokens: {maxTokens}
                    </label>
                    <input
                      id="maxTokens"
                      type="range"
                      min="16"
                      max="256"
                      step="8"
                      value={maxTokens}
                      onChange={event => setMaxTokens(Number(event.target.value))}
                      className="mt-2 w-full accent-cyan-500"
                    />
                  </section>

                  <section className="rounded-lg border border-gray-800 bg-gray-950 p-4">
                    <h3 className="text-sm font-semibold text-white">Session</h3>
                    <dl className="mt-4 space-y-3 text-sm">
                      <div className="flex justify-between gap-3">
                        <dt className="text-gray-500">Messages</dt>
                        <dd className="font-medium text-gray-200">{messages.length}</dd>
                      </div>
                      <div className="flex justify-between gap-3">
                        <dt className="text-gray-500">Prompt tokens</dt>
                        <dd className="font-medium text-gray-200">{usage?.prompt_tokens ?? 0}</dd>
                      </div>
                      <div className="flex justify-between gap-3">
                        <dt className="text-gray-500">Output tokens</dt>
                        <dd className="font-medium text-gray-200">{usage?.completion_tokens ?? 0}</dd>
                      </div>
                      <div className="flex justify-between gap-3">
                        <dt className="text-gray-500">Status</dt>
                        <dd className={isSending ? 'text-cyan-300' : 'text-emerald-300'}>
                          {isSending ? 'Running' : 'Ready'}
                        </dd>
                      </div>
                    </dl>
                  </section>

                  <section className="rounded-lg border border-amber-900/70 bg-amber-950/30 p-4">
                    <h3 className="text-sm font-semibold text-amber-100">Model note</h3>
                    <p className="mt-2 text-sm leading-6 text-amber-100/80">
                      This interface is ready for clients, but answer quality depends on how much the local model has been trained.
                    </p>
                  </section>
                </div>
              </aside>
            </div>
          </section>
        </div>
      </main>
    </>
  );
}
