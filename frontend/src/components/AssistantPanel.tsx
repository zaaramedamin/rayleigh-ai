import { useCallback, useEffect, useState } from "react";
import type { FormEvent } from "react";
import type { Api, AssistantInfo, Memory } from "../api/types";
import { events } from "../state/events";
import { HudFrame } from "./HudFrame";

const MAX_NAME = 40;
const MAX_ROLE = 2000;
const MAX_MEMORY = 400;

function messageOf(error: unknown): string {
  return error instanceof Error ? error.message : "Something went wrong.";
}

interface Props {
  api: Api;
  /** The name or form of address changed: the rest of the interface should use the new one. */
  onChanged: () => void;
  /** Changes whenever the assistant may have saved a memory, so the list reloads. */
  memoryVersion: number;
}

/** Who the assistant is, what role you gave it, and everything it remembers about you. */
export function AssistantPanel({ api, onChanged, memoryVersion }: Props) {
  const [info, setInfo] = useState<AssistantInfo | null>(null);
  const [draft, setDraft] = useState<AssistantInfo | null>(null);
  const [memories, setMemories] = useState<Memory[]>([]);
  const [limit, setLimit] = useState(200);
  const [newMemory, setNewMemory] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [confirmForget, setConfirmForget] = useState(false);

  const loadMemories = useCallback(() => {
    api.memories().then(
      (list) => {
        setMemories(list.memories);
        setLimit(list.limit);
      },
      (err) => setError(messageOf(err)),
    );
  }, [api]);

  useEffect(() => {
    api.assistant().then(
      (loaded) => {
        setInfo(loaded);
        setDraft(loaded);
      },
      (err) => setError(messageOf(err)),
    );
  }, [api]);
  useEffect(loadMemories, [loadMemories, memoryVersion]);

  if (!info || !draft) {
    return (
      <HudFrame title="YOUR ASSISTANT" tag="IDENTITY">
        {error ? (
          <p className="bad" role="alert">
            {error}
          </p>
        ) : (
          <p className="muted">Loading...</p>
        )}
      </HudFrame>
    );
  }

  const dirty = (Object.keys(draft) as Array<keyof AssistantInfo>).some((key) => draft[key] !== info[key]);

  const save = async (event: FormEvent) => {
    event.preventDefault();
    setBusy(true);
    setError("");
    try {
      const saved = await api.saveAssistant(draft);
      setInfo(saved);
      setDraft(saved);
      onChanged();
      events.notify("success", "ASSISTANT SAVED", { detail: "It uses this from the next message." });
    } catch (err) {
      setError(messageOf(err));
      events.notify("error", "ASSISTANT NOT SAVED", { detail: messageOf(err) });
    } finally {
      setBusy(false);
    }
  };

  const add = async (event: FormEvent) => {
    event.preventDefault();
    if (!newMemory.trim()) return;
    setError("");
    try {
      await api.addMemory(newMemory);
      setNewMemory("");
      loadMemories();
    } catch (err) {
      setError(messageOf(err));
    }
  };

  const forget = async (id: number) => {
    try {
      await api.deleteMemory(id);
      events.notify("info", "MEMORY DELETED", { sound: "remove", toast: false });
      loadMemories();
    } catch (err) {
      setError(messageOf(err));
    }
  };

  const forgetAll = async () => {
    try {
      const count = await api.clearMemories();
      setConfirmForget(false);
      events.notify("info", "MEMORY CLEARED", { detail: `${count} forgotten.`, sound: "remove" });
      loadMemories();
    } catch (err) {
      setError(messageOf(err));
    }
  };

  return (
    <>
      <HudFrame title="YOUR ASSISTANT" tag="IDENTITY">
        <form className="profile-form" onSubmit={save}>
          <label className="profile-field">
            <span>
              Its name<small>What it calls itself, for example Jarvis</small>
            </span>
            <input
              maxLength={MAX_NAME}
              value={draft.name}
              onChange={(e) => setDraft({ ...draft, name: e.target.value })}
            />
          </label>
          <label className="profile-field">
            <span>
              How it addresses you<small>sir, ma'am, your first name... or empty</small>
            </span>
            <input
              maxLength={MAX_NAME}
              value={draft.address}
              onChange={(e) => setDraft({ ...draft, address: e.target.value })}
            />
          </label>
          <label className="profile-field">
            <span>
              Its role<small>What you want it to be and do for you. Empty restores the default.</small>
            </span>
            <textarea
              rows={5}
              maxLength={MAX_ROLE}
              value={draft.role}
              onChange={(e) => setDraft({ ...draft, role: e.target.value })}
            />
          </label>
          <label className="toggle">
            <input
              type="checkbox"
              checked={draft.use_profile}
              onChange={(e) => setDraft({ ...draft, use_profile: e.target.checked })}
            />
            <span>Let it read the "About you" form above in every conversation</span>
          </label>
          <label className="toggle">
            <input
              type="checkbox"
              checked={draft.use_memory}
              onChange={(e) => setDraft({ ...draft, use_memory: e.target.checked })}
            />
            <span>Let it use its memories, and save new ones when you ask it to remember something</span>
          </label>
          {error && (
            <p className="bad" role="alert">
              {error}
            </p>
          )}
          <div className="profile-actions">
            <button className="btn" disabled={busy || !dirty}>
              {busy ? "SAVING..." : "SAVE ASSISTANT"}
            </button>
            {draft.role !== info.default_role && (
              <button
                type="button"
                className="btn btn--ghost"
                onClick={() => setDraft({ ...draft, role: info.default_role })}
              >
                USE DEFAULT ROLE
              </button>
            )}
          </div>
        </form>
      </HudFrame>

      <HudFrame title="MEMORY" tag={`${memories.length} / ${limit}`}>
        {memories.length === 0 ? (
          <p className="muted">
            Nothing yet. Say or type "remember that..." in the chat, or add something below. Memories last between
            sessions, and you can delete any of them.
          </p>
        ) : (
          <ul className="memory-list">
            {memories.map((memory) => (
              <li key={memory.id} className="memory-item">
                <span>
                  {memory.text}
                  <small>
                    {memory.origin === "assistant" ? "saved by the assistant" : "added by you"} ·{" "}
                    {new Date(memory.created_at).toLocaleDateString()}
                  </small>
                </span>
                <button className="link" onClick={() => void forget(memory.id)} aria-label="Delete this memory">
                  delete
                </button>
              </li>
            ))}
          </ul>
        )}
        <form className="memory-add" onSubmit={add}>
          <input
            maxLength={MAX_MEMORY}
            value={newMemory}
            onChange={(e) => setNewMemory(e.target.value)}
            placeholder="Something it should always know about you..."
            aria-label="New memory"
          />
          <button className="btn" disabled={!newMemory.trim()}>
            REMEMBER
          </button>
        </form>
        {memories.length > 0 &&
          (confirmForget ? (
            <span className="confirm">
              <span className="muted">Forget everything?</span>
              <button className="btn btn--danger" onClick={() => void forgetAll()}>
                FORGET ALL
              </button>
              <button className="link" onClick={() => setConfirmForget(false)}>
                cancel
              </button>
            </span>
          ) : (
            <p>
              <button className="link" onClick={() => setConfirmForget(true)}>
                forget everything
              </button>
            </p>
          ))}
      </HudFrame>
    </>
  );
}
