"use client";

import { SendHorizontal } from "lucide-react";
import { FormEvent, useEffect, useRef, useState } from "react";
import { cn } from "../../lib/utils";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";

type ChatSource = {
  doc_id?: string | null;
  chunk_id: string;
  page?: number | null;
  text: string;
  score: number;
};

type ChatMessage = {
  id: number;
  role: "user" | "assistant";
  content: string;
  sources?: ChatSource[];
};

type DocumentItem = {
  doc_id: string;
  filename: string;
};

type ChatPanelProps = {
  className?: string;
  placeholder?: string;
  introText?: string;
  docId?: string;
  projectId?: string;
};

export default function ChatPanel({
  className,
  placeholder = "Ask about this project or document…",
  introText = "Ask a question and I will answer from indexed chunks.",
  docId,
  projectId,
}: ChatPanelProps) {
  const messageIdRef = useRef(0);
  const nextMessageId = () => {
    messageIdRef.current += 1;
    return messageIdRef.current;
  };
  const [messages, setMessages] = useState<ChatMessage[]>([{ id: 0, role: "assistant", content: introText }]);
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [selectedDocId, setSelectedDocId] = useState(docId ?? "");
  const [text, setText] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (docId) {
      setSelectedDocId(docId);
      return;
    }
    if (projectId) {
      setSelectedDocId("");
      return;
    }

    let cancelled = false;

    const loadDocuments = async () => {
      try {
        const response = await fetch(`${API_BASE}/v1/documents`);
        if (!response.ok || cancelled) {
          return;
        }
        const payload = (await response.json()) as DocumentItem[];
        if (cancelled) {
          return;
        }
        setDocuments(payload);
        setSelectedDocId((prev) => prev || payload[0]?.doc_id || "");
      } catch {
        if (!cancelled) {
          setDocuments([]);
        }
      }
    };

    void loadDocuments();
    return () => {
      cancelled = true;
    };
  }, [docId, projectId]);

  const canSend = text.trim().length > 0 && !!(projectId || selectedDocId) && !loading;

  const formatError = async (response: Response): Promise<string> => {
    const raw = await response.text().catch(() => "");
    let detail = raw.trim();
    try {
      const payload = JSON.parse(raw) as { detail?: string };
      if (payload.detail) {
        detail = payload.detail;
      }
    } catch {
      // Keep response text fallback.
    }
    const suffix = detail ? ` - ${detail}` : "";
    return `${response.status} ${response.statusText || "Request failed"}${suffix}`;
  };

  const onSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const value = text.trim();
    if (!value) {
      return;
    }

    if (!projectId && !selectedDocId) {
      setMessages((prev) => [
        ...prev,
        { id: nextMessageId(), role: "assistant", content: "Select a document before sending a message." },
      ]);
      return;
    }

    setMessages((prev) => [...prev, { id: nextMessageId(), role: "user", content: value }]);
    setText("");
    setLoading(true);

    try {
      const endpoint = projectId ? `${API_BASE}/v1/chat/project` : `${API_BASE}/v1/chat`;
      const body = projectId
        ? { project_id: projectId, message: value, k_docs: 10, k_per_doc: 3, k_total: 8 }
        : { doc_id: selectedDocId, message: value, k: 5 };

      const response = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });

      if (!response.ok) {
        throw new Error(await formatError(response));
      }

      const payload = (await response.json()) as { answer: string; sources?: ChatSource[] };
      setMessages((prev) => [
        ...prev,
        {
          id: nextMessageId(),
          role: "assistant",
          content: payload.answer || "No answer returned.",
          sources: payload.sources || [],
        },
      ]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          id: nextMessageId(),
          role: "assistant",
          content: err instanceof Error ? err.message : "Chat request failed.",
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <section
      className={cn(
        "flex h-full min-h-[21rem] flex-col overflow-hidden rounded-2xl border border-slate-700/70 bg-slate-900/70",
        className
      )}
    >
      {!docId && !projectId ? (
        <div className="border-b border-slate-700/70 px-4 py-3">
          <label htmlFor="chat-doc" className="mb-1 block text-xs font-medium text-slate-400">
            Chat document
          </label>
          <select
            id="chat-doc"
            value={selectedDocId}
            onChange={(event) => setSelectedDocId(event.target.value)}
            className="h-9 w-full rounded-lg border border-slate-600 bg-slate-800 px-3 text-sm text-slate-100 focus:border-blue-400 focus:outline-none"
          >
            {documents.length === 0 ? <option value="">No documents available</option> : null}
            {documents.map((doc) => (
              <option key={doc.doc_id} value={doc.doc_id}>
                {doc.filename}
              </option>
            ))}
          </select>
        </div>
      ) : null}

      <div className="min-h-0 flex-1 space-y-3 overflow-y-auto px-4 py-4">
        {messages.map((message) => (
          <div key={message.id} className={cn("flex", message.role === "user" ? "justify-end" : "justify-start")}>
            <div
              className={cn(
                "max-w-[85%] rounded-2xl px-4 py-2 text-sm leading-relaxed",
                message.role === "user"
                  ? "rounded-br-md bg-blue-500 text-white"
                  : "rounded-bl-md border border-slate-700 bg-slate-800 text-slate-100"
              )}
            >
              <p>{message.content}</p>
              {message.role === "assistant" && message.sources && message.sources.length > 0 ? (
                <div className="mt-3 border-t border-slate-600/60 pt-2 text-xs">
                  <p className="mb-1 font-semibold text-slate-300">Sources</p>
                  <ul className="space-y-1">
                    {message.sources.map((source) => (
                      <li key={`${message.id}-${source.chunk_id}`} className="rounded-md bg-slate-900/60 px-2 py-1">
                        {source.doc_id ? <p className="text-slate-400">{source.doc_id}</p> : null}
                        <p className="font-medium text-blue-300">{source.chunk_id}</p>
                        <p className="text-slate-300">
                          {(source.text || "").length > 140 ? `${source.text.slice(0, 140)}...` : source.text}
                        </p>
                      </li>
                    ))}
                  </ul>
                </div>
              ) : null}
            </div>
          </div>
        ))}
      </div>

      <div className="shrink-0 border-t border-slate-700 bg-slate-900/95 p-3 sm:p-4">
        <form onSubmit={onSubmit} className="flex items-center gap-2">
          <input
            value={text}
            onChange={(event) => setText(event.target.value)}
            placeholder={placeholder}
            className="h-11 w-full rounded-xl border border-slate-600 bg-slate-800 px-4 text-sm text-slate-100 placeholder:text-slate-400 focus:border-blue-400 focus:outline-none"
          />
          <button
            type="submit"
            disabled={!canSend}
            className="inline-flex h-11 items-center gap-2 rounded-xl bg-blue-500 px-4 text-sm font-semibold text-white transition hover:bg-blue-400 disabled:cursor-not-allowed disabled:opacity-60"
          >
            <SendHorizontal className="h-4 w-4" aria-hidden="true" />
            {loading ? "Sending..." : "Send"}
          </button>
        </form>
      </div>
    </section>
  );
}
