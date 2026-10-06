import { useEffect, useState } from "react";
import type { FormEvent } from "react";
import type { Api, Profile, ProfileKey, ProfileValues } from "../api/types";
import { AssistantPanel } from "../components/AssistantPanel";
import { HudFrame } from "../components/HudFrame";
import { events } from "../state/events";

const MAX_CHARS = 2000;
const MULTILINE: ProfileKey[] = ["about", "preferences", "interests"];

function messageOf(error: unknown): string {
  return error instanceof Error ? error.message : "Something went wrong.";
}

/** What the assistant should know about you. Kept as an ordinary note you can read and delete. */
interface Props {
  api: Api;
  onChanged: () => void;
  /** The assistant's name or form of address changed. */
  onAssistantChanged: () => void;
  /** Changes when the assistant may have saved a memory, so the list reloads. */
  memoryVersion: number;
}

export function ProfileView({ api, onChanged, onAssistantChanged, memoryVersion }: Props) {
  const [profile, setProfile] = useState<Profile | null>(null);
  const [values, setValues] = useState<ProfileValues | null>(null);
  const [saved, setSaved] = useState<ProfileValues | null>(null);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [confirmClear, setConfirmClear] = useState(false);

  useEffect(() => {
    api
      .profile()
      .then((p) => {
        setProfile(p);
        setValues(p.values);
        setSaved(p.values);
      })
      .catch((err) => setError(messageOf(err)));
  }, [api]);

  if (!profile || !values || !saved) {
    return (
      <HudFrame title="PROFILE" tag="ABOUT YOU">
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

  const dirty = profile.fields.some((f) => values[f.key] !== saved[f.key]);
  const empty = profile.fields.every((f) => !saved[f.key].trim());

  const apply = (next: Profile) => {
    setProfile(next);
    setValues(next.values);
    setSaved(next.values);
  };

  const save = async (event: FormEvent) => {
    event.preventDefault();
    setBusy(true);
    setError("");
    setMessage("");
    try {
      const next = await api.saveProfile(values);
      apply(next);
      onChanged();
      const nothing = profile.fields.every((f) => !next.values[f.key].trim());
      if (nothing) events.notify("info", "PROFILE CLEARED", { sound: "remove" });
      else if (next.searchable) events.notify("success", "PROFILE SAVED", { detail: "The assistant can use it now." });
      else events.notify("warning", "PROFILE SAVED, NOT SEARCHABLE YET", { detail: "Update the library from the Knowledge page." });
      setMessage(
        nothing
          ? "Profile cleared."
          : next.searchable
            ? "Saved. The assistant can now use this when your questions are about you."
            : "Saved, but not searchable yet. Run an update from the Knowledge page once the embedding model is ready.",
      );
    } catch (err) {
      setError(messageOf(err));
      events.notify("error", "PROFILE NOT SAVED", { detail: messageOf(err) });
    } finally {
      setBusy(false);
    }
  };

  const clear = async () => {
    setBusy(true);
    setError("");
    try {
      await api.clearProfile();
      const next = await api.profile();
      apply(next);
      onChanged();
      setMessage("Profile deleted.");
      events.notify("info", "PROFILE DELETED", { sound: "remove" });
      setConfirmClear(false);
    } catch (err) {
      setError(messageOf(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="stack">
      <div className="profile">
      <HudFrame title="ABOUT YOU" tag={profile.updated_at ? "SAVED" : "EMPTY"}>
        <form className="profile-form" onSubmit={save}>
          {profile.fields.map((field) => {
            const multiline = MULTILINE.includes(field.key);
            return (
              <label key={field.key} className="profile-field">
                <span>
                  {field.label}
                  <small>{field.hint}</small>
                </span>
                {multiline ? (
                  <textarea
                    rows={3}
                    maxLength={MAX_CHARS}
                    value={values[field.key]}
                    onChange={(e) => setValues({ ...values, [field.key]: e.target.value })}
                  />
                ) : (
                  <input
                    maxLength={MAX_CHARS}
                    value={values[field.key]}
                    onChange={(e) => setValues({ ...values, [field.key]: e.target.value })}
                  />
                )}
              </label>
            );
          })}

          {error && (
            <p className="bad" role="alert">
              {error}
            </p>
          )}
          {message && !error && <p className="ok">{message}</p>}

          <div className="profile-actions">
            <button className="btn" disabled={busy || !dirty}>
              {busy ? "SAVING..." : "SAVE PROFILE"}
            </button>
            {!empty &&
              (confirmClear ? (
                <span className="confirm">
                  <span className="muted">Delete everything you wrote here?</span>
                  <button type="button" className="btn btn--danger" disabled={busy} onClick={() => void clear()}>
                    DELETE
                  </button>
                  <button type="button" className="link" onClick={() => setConfirmClear(false)}>
                    cancel
                  </button>
                </span>
              ) : (
                <button type="button" className="btn btn--ghost" onClick={() => setConfirmClear(true)}>
                  DELETE PROFILE
                </button>
              ))}
          </div>
        </form>
      </HudFrame>

      <HudFrame title="HOW THIS WORKS" tag="PRIVACY">
        <ul className="plain">
          <li>
            Your answers are saved as a normal note called <b>{profile.saved_as}</b> in your library, on this computer
            only.
          </li>
          <li>
            Like any note, it is used <b>only when a question matches it</b> (for example "where do I live?"), and the
            answer cites it, so you always see when your profile was used.
          </li>
          <li>
            You can read, change or delete it at any time here, or remove it from the Knowledge page. Fill in only what
            you are comfortable keeping in your notes.
          </li>
          <li>
            Status:{" "}
            {profile.updated_at === null ? (
              <span className="muted">nothing saved yet</span>
            ) : profile.searchable ? (
              <span className="ok">searchable</span>
            ) : (
              <span className="warn">saved, not searchable yet</span>
            )}
          </li>
        </ul>
      </HudFrame>
      </div>
      <div className="profile">
        <AssistantPanel api={api} onChanged={onAssistantChanged} memoryVersion={memoryVersion} />
      </div>
    </div>
  );
}
