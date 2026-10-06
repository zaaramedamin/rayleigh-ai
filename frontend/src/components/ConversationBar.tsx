import { useCallback, useEffect, useState } from "react";
import type { FormEvent } from "react";
import type { Api, ConversationInfo } from "../api/types";
import { events } from "../state/events";
import type { Assistant } from "./ChatBubble";

function when(iso: string): string {
  const date = new Date(iso);
  return Number.isNaN(date.getTime()) ? "" : date.toLocaleString([], { dateStyle: "medium", timeStyle: "short" });
}

function messageOf(error: unknown): string {
  return error instanceof Error ? error.message : "Something went wrong.";
}

/**
 * The saved conversations: start a new one, reopen, rename or delete one. Deleting removes the
 * conversation and everything in it from this computer, for good.
 */
export function ConversationBar({ api, assistant }: { api: Api; assistant: Assistant }) {
  const [open, setOpen] = useState(false);
  const [items, setItems] = useState<ConversationInfo[]>([]);
  const [retention, setRetention] = useState(0);
  const [error, setError] = useState("");
  const [renaming, setRenaming] = useState<number | null>(null);
  const [draft, setDraft] = useState("");
  const [confirm, setConfirm] = useState<number | "all" | null>(null);
  const [busy, setBusy] = useState(false);

  const reload = useCallback(async () => {
    try {
      const list = await api.conversations();
      setItems(list.conversations);
      setRetention(list.retention_days);
      setError("");
    } catch (err) {
      setError(messageOf(err));
    }
  }, [api]);

  // On opening the page, after something was saved, and whenever the list is opened.
  useEffect(() => {
    void reload();
  }, [reload, assistant.savedVersion, open]);

  const act = async (work: () => Promise<void>) => {
    setBusy(true);
    try {
      await work();
      await reload();
    } catch (err) {
      events.notify("error", "THAT DID NOT WORK", { detail: messageOf(err) });
    } finally {
      setBusy(false);
    }
  };

  const reopen = (id: number) =>
    act(async () => {
      await assistant.open(id);
      setOpen(false);
    });

  const remove = (id: number) =>
    act(async () => {
      await api.deleteConversation(id);
      if (assistant.conversationId === id) assistant.clear();
      setConfirm(null);
      events.notify("info", "CONVERSATION DELETED", { detail: "It was removed from this computer.", sound: "remove" });
    });

  const removeAll = () =>
    act(async () => {
      const count = await api.deleteConversations();
      assistant.clear();
      setConfirm(null);
      events.notify("info", "CONVERSATIONS DELETED", { detail: `${count} removed from this computer.`, sound: "remove" });
    });

  const rename = (event: FormEvent, id: number) => {
    event.preventDefault();
    if (!draft.trim()) return;
    void act(async () => {
      await api.renameConversation(id, draft.trim());
      setRenaming(null);
    });
  };

  const current = items.find((c) => c.id === assistant.conversationId);
  const title = !assistant.saveChats
    ? "NOT SAVED // saving is off"
    : assistant.conversationId === null
      ? "NEW CONVERSATION"
      : (current?.title ?? "...");

  return (
    <>
      <div className="convbar">
        <button className="btn btn--ghost" aria-expanded={open} onClick={() => setOpen(!open)}>
          HISTORY ({items.length})
        </button>
        <button className="btn btn--ghost" onClick={assistant.clear} disabled={assistant.messages.length === 0}>
          NEW
        </button>
        <span className="convbar-title muted ellipsis" title={title}>
          {title}
        </span>
        <label className="toggle toggle--small" title="Keep conversations on this computer so they can be reopened">
          <input type="checkbox" checked={assistant.saveChats} onChange={(e) => assistant.setSaveChats(e.target.checked)} />
          <span>SAVE</span>
        </label>
      </div>

      {open && (
        <div className="convlist-wrap">
          {error && (
            <p className="bad" role="alert">
              {error}
            </p>
          )}
          {!error && items.length === 0 && <p className="muted">No saved conversations yet.</p>}
          <ul className="rows convlist">
            {items.map((item) => (
              <li key={item.id} className={`row-item${item.id === assistant.conversationId ? " row-item--open" : ""}`}>
                <div className="row-main">
                  {renaming === item.id ? (
                    <form onSubmit={(e) => rename(e, item.id)}>
                      <input
                        className="field"
                        value={draft}
                        onChange={(e) => setDraft(e.target.value)}
                        maxLength={200}
                        aria-label="New title"
                        autoFocus
                      />
                    </form>
                  ) : (
                    <button className="title ellipsis" onClick={() => void reopen(item.id)} disabled={busy} title="Open this conversation">
                      {item.title || "(untitled)"}
                    </button>
                  )}
                  <span className="row-meta">
                    <span>{when(item.updated_at)}</span>
                    <span>{item.message_count} messages</span>
                  </span>
                </div>
                {confirm === item.id ? (
                  <div className="confirm">
                    <span className="muted">Delete it and everything in it?</span>
                    <button className="btn btn--danger" disabled={busy} onClick={() => void remove(item.id)}>
                      DELETE
                    </button>
                    <button className="link" onClick={() => setConfirm(null)}>
                      cancel
                    </button>
                  </div>
                ) : renaming === item.id ? (
                  <div className="row-actions">
                    <button className="btn" disabled={busy || !draft.trim()} onClick={(e) => rename(e, item.id)}>
                      SAVE
                    </button>
                    <button className="link" onClick={() => setRenaming(null)}>
                      cancel
                    </button>
                  </div>
                ) : (
                  <div className="row-actions">
                    <button
                      className="btn btn--ghost"
                      onClick={() => {
                        setRenaming(item.id);
                        setDraft(item.title);
                      }}
                    >
                      RENAME
                    </button>
                    <button className="btn btn--ghost" onClick={() => setConfirm(item.id)} aria-label={`Delete ${item.title}`}>
                      DELETE
                    </button>
                  </div>
                )}
              </li>
            ))}
          </ul>
          {items.length > 0 &&
            (confirm === "all" ? (
              <div className="confirm">
                <span className="muted">Delete all {items.length} conversations, for good?</span>
                <button className="btn btn--danger" disabled={busy} onClick={() => void removeAll()}>
                  DELETE ALL
                </button>
                <button className="link" onClick={() => setConfirm(null)}>
                  cancel
                </button>
              </div>
            ) : (
              <button className="link" onClick={() => setConfirm("all")}>
                delete all conversations
              </button>
            ))}
          <p className="command-hint">
            Conversations are kept on this computer only{retention > 0 ? `, and deleted after ${retention} days without use` : ""}. A
            conversation that quoted a note loses that answer if the note is removed from the library.
          </p>
        </div>
      )}
    </>
  );
}
