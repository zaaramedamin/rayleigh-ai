import { HudFrame } from "../components/HudFrame";

// These restate docs/security.md. They describe how the backend is built, not live measurements.
const RULES: Array<[string, string]> = [
  ["Access password", "The app and its API need your password. Only a salted hash is stored. It guards access to the application; it does not by itself encrypt your files on disk."],
  ["Local only", "Embeddings and answers are produced on this machine. No note or question is sent to a cloud API."],
  ["Loopback network", "The API listens on 127.0.0.1 and the model server address must be on this machine."],
  ["Allow-listed folders", "Only folders you chose are indexed. The app never widens that list on its own."],
  ["Notes are data", "Retrieved text is never treated as instructions, and the model has no tools to act with."],
  ["Honest sources", "Citations are built by the app from its database. Invented source numbers are discarded."],
  ["General chat", "With MY NOTES off, the local model answers from what it knows and your notes are not read. Those replies are labelled GENERAL, have no sources and can be wrong."],
  ["No content in logs", "Questions, notes and answers are never written to logs."],
];

export function PrivacyView() {
  return (
    <div className="stack">
      <HudFrame title="PRIVACY CORE" tag="POLICY">
        <div className="grid">
          {RULES.map(([title, body]) => (
            <article key={title} className="module module--live">
              <header>
                <h3>{title}</h3>
              </header>
              <p>{body}</p>
            </article>
          ))}
        </div>
      </HudFrame>
      <HudFrame title="VERIFY IT YOURSELF" tag="TERMINAL">
        <p className="muted">Do not take the interface's word for it. Prove it with the network blocked:</p>
        <pre className="code">python -m app offline-check</pre>
        <p className="muted">
          This page also makes no external requests: fonts and scripts are bundled, and the page's security policy only
          allows connections to its own origin.
        </p>
      </HudFrame>
    </div>
  );
}
