import { useSpeaking } from "../audio/speech";
import { useMicLevel } from "../state/useVoice";
import type { Voice } from "../state/useVoice";
import { Icon } from "./Icon";

interface Props {
  voice: Voice;
  /** The assistant is working on a reply, so a new order has to wait. */
  busy: boolean;
  /** A smaller button, for the corner chat. */
  compact?: boolean;
}

/** The loudness of the microphone, as a bar that fills the button while it listens. */
function Meter() {
  const level = useMicLevel();
  return <span className="mic-meter" style={{ transform: `scaleX(${Math.max(0.04, level)})` }} aria-hidden />;
}

/**
 * Speak an order. One click starts listening; it stops by itself when you pause. While the
 * assistant is talking, the same button makes it stop.
 */
export function MicButton({ voice, busy, compact = false }: Props) {
  const speaking = useSpeaking();
  const listening = voice.state === "listening";
  const transcribing = voice.state === "transcribing";
  const missing = voice.status?.state === "not_downloaded";

  const label = listening ? "LISTENING" : transcribing ? "HEARD" : speaking ? "STOP" : "VOICE";
  const title = !voice.supported
    ? "This browser cannot record from the microphone."
    : listening
      ? "Listening. It stops when you pause. Click to cancel."
      : speaking
        ? "Stop talking"
        : missing
          ? (voice.status?.hint ?? "The speech model is not downloaded.")
          : "Speak an order or a question. Your voice is turned into text on this computer.";

  return (
    <button
      type="button"
      className={`btn btn--ghost mic${listening ? " mic--on" : ""}${compact ? " mic--compact" : ""}`}
      disabled={!voice.supported || transcribing || (busy && !speaking && !listening)}
      aria-pressed={listening}
      aria-label={listening ? "Stop listening" : speaking ? "Stop talking" : "Speak an order"}
      title={title}
      onClick={() => (listening || speaking ? voice.stop() : voice.start())}
    >
      {listening && <Meter />}
      <Icon name={speaking && !listening ? "mute" : "mic"} size={14} />
      {!compact && <span>{label}</span>}
    </button>
  );
}
