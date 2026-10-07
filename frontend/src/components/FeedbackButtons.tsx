import type { ChatMode, FeedbackKind } from "../api/types";

const LABELS: Record<FeedbackKind, string> = {
  helpful: "HELPFUL",
  not_helpful: "NOT HELPFUL",
  wrong_source: "WRONG SOURCE",
  missing_info: "MISSING INFO",
};

const HINTS: Record<FeedbackKind, string> = {
  helpful: "This answer helped",
  not_helpful: "This answer did not help",
  wrong_source: "The answer used the wrong note",
  missing_info: "My notes should have held the answer",
};

/** The marks to offer: after a thumbs-down, an answer from the notes can say what went wrong. */
export function kindsToOffer(mark: FeedbackKind | undefined, mode: ChatMode): FeedbackKind[] {
  const failed = mark !== undefined && mark !== "helpful";
  return failed && mode === "notes"
    ? ["helpful", "not_helpful", "wrong_source", "missing_info"]
    : ["helpful", "not_helpful"];
}

interface Props {
  /** What is marked now, if anything. */
  mark?: FeedbackKind;
  mode: ChatMode;
  /** Pressing the mark that is on takes it back. */
  onRate: (kind: FeedbackKind) => void;
}

/** Mark an answer. A mark is kept on this computer; a failure can later become an evaluation question. */
export function FeedbackButtons({ mark, mode, onRate }: Props) {
  return (
    <div className="feedback" role="group" aria-label="Rate this answer">
      {kindsToOffer(mark, mode).map((kind) => (
        <button
          key={kind}
          type="button"
          className={`feedback-btn${mark === kind ? " feedback-btn--on" : ""}`}
          aria-pressed={mark === kind}
          title={HINTS[kind]}
          onClick={() => onRate(kind)}
        >
          {LABELS[kind]}
        </button>
      ))}
    </div>
  );
}
