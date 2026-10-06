import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createApi } from "./api/client";
import { createMockApi } from "./api/mock";
import type { Command } from "./lib/commands";
import { sound } from "./audio/sound";
import { useSpeaking } from "./audio/speech";
import { BootSequence } from "./components/BootSequence";
import { Clock } from "./components/Clock";
import { CommandPalette } from "./components/CommandPalette";
import { Icon } from "./components/Icon";
import { LockScreen } from "./components/LockScreen";
import { MiniChat } from "./components/MiniChat";
import { NavRail } from "./components/NavRail";
import { VIEW_TITLES } from "./components/NavRail";
import type { ViewId } from "./components/NavRail";
import { StatusPanel } from "./components/StatusPanel";
import type { AppControls } from "./state/actions";
import { events } from "./state/events";
import { greetingFor, statusReport } from "./state/greeting";
import { useClickPing } from "./state/useClickPing";
import { usePanelGlow } from "./state/usePanelGlow";
import { useSettings } from "./state/settings";
import { useAssistant } from "./state/useAssistant";
import { useHealth } from "./state/useHealth";
import { useIdentity } from "./state/useIdentity";
import { useSystemStatus } from "./state/useSystemStatus";
import { useVoice } from "./state/useVoice";
import { ConsoleView } from "./views/ConsoleView";
import { DashboardView } from "./views/DashboardView";
import { KnowledgeView } from "./views/KnowledgeView";
import { ModulesView } from "./views/ModulesView";
import { PrivacyView } from "./views/PrivacyView";
import { ProfileView } from "./views/ProfileView";
import { SettingsView } from "./views/SettingsView";

// lock: the password screen. boot: start-up screen. opening: the reactor flies right while the
// interface slides in. ready: the interface alone.
type Phase = "lock" | "boot" | "opening" | "ready";

const BOOT_FADE_OUT_MS = 500;

export function App() {
  const { settings, update } = useSettings();
  usePanelGlow();
  useClickPing(!settings.reducedMotion);

  // The sign-in token lives in memory only: reloading the page asks for the password again.
  const token = useRef<string | null>(null);
  const signedIn = useRef(false);
  const lockRef = useRef<() => void>(() => undefined);

  const api = useMemo(
    () =>
      settings.demo
        ? createMockApi()
        : createApi("/api/v1", {
            getToken: () => token.current,
            onUnauthorized: () => {
              if (!signedIn.current) return;
              events.notify("warning", "SESSION ENDED", { detail: "Sign in again to continue.", key: "session" });
              lockRef.current();
            },
          }),
    [settings.demo],
  );
  const { link, recheck, history } = useHealth(api);
  const [phase, setPhase] = useState<Phase>(settings.demo ? "boot" : "lock");
  const [landed, setLanded] = useState(false);
  const [view, setView] = useState<ViewId>("home");
  const [statusKey, setStatusKey] = useState(0);
  const refreshStatus = useCallback(() => setStatusKey((k) => k + 1), []);
  const system = useSystemStatus(api, link.state === "online" && phase !== "lock", statusKey);

  // The assistant: who it is, what it may operate, how it listens and how it is greeted.
  const inside = phase !== "lock";
  const { identity, reload: reloadIdentity } = useIdentity(api, inside && link.state === "online");
  const controls = useRef<AppControls | null>(null);
  const assistant = useAssistant(api, controls, identity);
  const voice = useVoice(api, assistant.ask, landed && inside);
  const speaking = useSpeaking();

  const reducedMotion = useRef(settings.reducedMotion);
  reducedMotion.current = settings.reducedMotion;

  const lock = useCallback(() => {
    token.current = null;
    signedIn.current = false;
    voice.stop();
    assistant.clear();
    setLanded(false);
    setView("home");
    setPhase("lock");
  }, [assistant, voice]);
  lockRef.current = lock;

  controls.current = {
    navigate: setView,
    update,
    lock,
    statusReport: () => statusReport(system.state === "ready" ? system.status : null, link.state === "online"),
    changed: () => {
      refreshStatus();
      reloadIdentity();
    },
  };

  // Say hello once the application has opened, as soon as the system status is known.
  const greeted = useRef(false);
  useEffect(() => {
    if (phase === "lock") greeted.current = false;
  }, [phase]);
  useEffect(() => {
    if (!landed || greeted.current || system.state === "loading") return;
    greeted.current = true;
    if (!settings.greeting) return;
    const status = system.state === "ready" ? system.status : null;
    assistant.note(greetingFor(new Date(), identity.address, status, link.state === "online"));
  }, [landed, system, settings.greeting, identity.address, link.state, assistant]);

  // Escape stops the voice: listening and talking.
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") voice.stop();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [voice]);

  const signOut = () => {
    void api.logout().catch(() => undefined);
    sound.play("lock");
    lock();
  };

  // Ctrl+K (or Cmd+K) opens the command palette once the interface is up.
  const [paletteOpen, setPaletteOpen] = useState(false);
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        if (phase === "ready") setPaletteOpen((open) => !open);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [phase]);

  // Say so when the link to the backend is lost or comes back (not during start-up).
  const lastLink = useRef(link.state);
  useEffect(() => {
    const previous = lastLink.current;
    lastLink.current = link.state;
    if (!landed || settings.demo) return;
    if (link.state === "offline" && previous === "online") {
      events.notify("alert", "LINK LOST", {
        detail: "The backend stopped answering. Start it again to continue.",
        key: "link",
      });
    } else if (link.state === "online" && previous === "offline") {
      events.notify("success", "LINK RESTORED", { detail: "The backend is answering again." });
    }
  }, [link.state, landed, settings.demo]);

  // Warn once when the local model or the embedding model is not usable.
  const warned = useRef({ llm: "", embedding: false });
  useEffect(() => {
    if (!landed || system.state !== "ready") return;
    const { llm, embedding } = system.status;
    if (llm.state === "ready") warned.current.llm = "";
    else if (warned.current.llm !== llm.state) {
      warned.current.llm = llm.state;
      const title =
        llm.state === "not_running"
          ? "LOCAL MODEL NOT RUNNING"
          : llm.state === "model_missing"
            ? "LOCAL MODEL NOT INSTALLED"
            : "LOCAL MODEL PROBLEM";
      events.notify("warning", title, { detail: llm.hint ?? undefined });
    }
    if (!embedding.downloaded && !warned.current.embedding) {
      warned.current.embedding = true;
      events.notify("warning", "EMBEDDING MODEL MISSING", { detail: "Run python -m app download-model" });
    }
  }, [landed, system]);

  // Browsers allow sound only after a click or key press; the password button is the first one.
  useEffect(() => {
    const unlock = () => sound.unlock();
    window.addEventListener("pointerdown", unlock);
    window.addEventListener("keydown", unlock);
    return () => {
      window.removeEventListener("pointerdown", unlock);
      window.removeEventListener("keydown", unlock);
    };
  }, []);

  // Leaving demo mode means real data again, which needs the password.
  useEffect(() => {
    if (!settings.demo && !signedIn.current && phase !== "lock") lockRef.current();
  }, [settings.demo, phase]);

  const unlocked = useCallback((newToken: string | null) => {
    token.current = newToken;
    signedIn.current = true;
    setPhase("boot");
  }, []);

  const startDemo = useCallback(() => {
    update({ demo: true });
    signedIn.current = false;
    setPhase("boot");
  }, [update]);

  const enter = useCallback(() => {
    if (reducedMotion.current) {
      setLanded(true);
      setPhase("ready");
    } else {
      sound.play("whoosh");
      setPhase("opening");
    }
  }, []);

  const onLanded = useCallback(() => {
    sound.play("landed");
    setLanded(true);
  }, []);

  // Once the reactor has landed, let the start-up layer finish fading, then remove it.
  useEffect(() => {
    if (phase !== "opening" || !landed) return;
    const timer = setTimeout(() => setPhase("ready"), BOOT_FADE_OUT_MS);
    return () => clearTimeout(timer);
  }, [phase, landed]);

  const backdrop = <div className="bg" aria-hidden />;
  if (phase === "lock") {
    return (
      <>
        {backdrop}
        <LockScreen api={api} onUnlocked={unlocked} onDemo={startDemo} />
      </>
    );
  }

  const commands: Command[] = [
    ...(Object.keys(VIEW_TITLES) as ViewId[]).map((id) => ({
      id: `go-${id}`,
      label: `Go to ${VIEW_TITLES[id]}`,
      group: "NAVIGATE",
      keywords: id === "chat" ? "ask question talk message" : id === "knowledge" ? "documents folders search library" : "",
      run: () => setView(id),
    })),
    {
      id: "update-library",
      label: "Update the library",
      group: "LIBRARY",
      keywords: "read folders index make searchable scan",
      run: () => {
        api
          .startSync()
          .then(() => {
            events.notify("info", "UPDATE STARTED", { detail: "Follow it on the Knowledge page.", sound: null });
            setView("knowledge");
          })
          .catch((error: Error) => events.notify("error", "COULD NOT START THE UPDATE", { detail: error.message }));
      },
    },
    { id: "recheck", label: "Recheck the backend", group: "SYSTEM", keywords: "link connection ping", run: recheck },
    {
      id: "theme",
      label: `Switch to the ${settings.theme === "reactor" ? "Mark III (gold)" : "Arc Reactor (cyan)"} theme`,
      group: "SYSTEM",
      keywords: "colour color look",
      run: () => update({ theme: settings.theme === "reactor" ? "mark3" : "reactor" }),
    },
    {
      id: "sound",
      label: settings.sound ? "Turn interface sounds off" : "Turn interface sounds on",
      group: "SYSTEM",
      keywords: "audio mute volume",
      run: () => update({ sound: !settings.sound }),
    },
    {
      id: "motion",
      label: settings.reducedMotion ? "Turn animations on" : "Reduce animations",
      group: "SYSTEM",
      keywords: "motion accessibility",
      run: () => update({ reducedMotion: !settings.reducedMotion }),
    },
    {
      id: "alert",
      label: "Preview a system alert",
      group: "TEST",
      keywords: "siren sound warning",
      run: () => events.notify("alert", "TEST ALERT", { detail: "A system-level problem would look like this." }),
    },
    { id: "lock", label: "Lock the application", group: "SYSTEM", keywords: "password sign out leave", run: signOut },
  ];

  const showShell = phase !== "boot";
  // Until the reactor has landed, the top bar reads OFFLINE like the rest of the interface.
  const linkState = landed ? link.state : "offline";

  return (
    <>
      {backdrop}
      {showShell && (
        <div className={`shell${phase === "opening" ? " shell--enter" : ""}`}>
          <header className="topbar">
            <div className="topbar-left">
              <div className="brand glitch" data-text="REYLEIGHT">
                REYLEIGHT
              </div>
              <span className="topbar-sep" aria-hidden />
              <h1 className="crumb">{VIEW_TITLES[view]}</h1>
            </div>
            <div className="topbar-center">
              {settings.demo && <div className="banner">DEMO MODE // SIMULATED DATA</div>}
              {!settings.demo && landed && link.state === "offline" && (
                <div className="banner banner--bad">
                  BACKEND OFFLINE
                  <button className="link" onClick={recheck}>
                    retry
                  </button>
                  <button className="link" onClick={() => update({ demo: true })}>
                    demo mode
                  </button>
                </div>
              )}
            </div>
            <div className="topbar-right">
              <button className="kbd-hint" onClick={() => setPaletteOpen(true)} aria-label="Open the command palette">
                <kbd>CTRL</kbd>
                <kbd>K</kbd>
              </button>
              <Clock />
              <span className="topbar-sep" aria-hidden />
              <span className={`status-pill status-pill--${linkState}`}>
                <span className={`dot dot--${linkState}`} aria-hidden />
                {linkState === "online" ? "LOCAL LINK" : linkState === "checking" ? "SCANNING" : "OFFLINE"}
              </span>
              <button className="lock-btn" onClick={signOut} aria-label="Lock the application">
                <Icon name="lock" size={15} />
                <span>LOCK</span>
              </button>
            </div>
          </header>

          <NavRail view={view} onChange={setView} />

          <main className="main">
            <div key={view} className="page">
              {view === "home" && (
                <DashboardView
                  assistant={assistant}
                  link={link}
                  history={history}
                  system={system}
                  animate={!settings.reducedMotion}
                  onNavigate={setView}
                  onRecheck={recheck}
                />
              )}
              {view === "chat" && <ConsoleView api={api} assistant={assistant} voice={voice} />}
              {view === "knowledge" && <KnowledgeView api={api} onChanged={refreshStatus} />}
              {view === "profile" && (
                <ProfileView
                  api={api}
                  onChanged={refreshStatus}
                  onAssistantChanged={reloadIdentity}
                  memoryVersion={statusKey}
                />
              )}
              {view === "modules" && <ModulesView />}
              {view === "privacy" && <PrivacyView />}
              {view === "settings" && (
                <SettingsView
                  api={api}
                  onPasswordChanged={(newToken) => {
                    token.current = newToken;
                  }}
                />
              )}
            </div>
          </main>

          <aside className="side">
            <StatusPanel
              link={link}
              thinking={assistant.status === "thinking"}
              listening={voice.state === "listening"}
              speaking={speaking}
              demo={settings.demo}
              messages={assistant.messages}
              system={system}
              reactorHidden={!landed}
            />
          </aside>

          {view !== "chat" && landed && (
            <MiniChat assistant={assistant} voice={voice} onShowSources={() => setView("home")} />
          )}
        </div>
      )}
      {phase === "ready" && <CommandPalette open={paletteOpen} commands={commands} onClose={() => setPaletteOpen(false)} />}
      {phase !== "ready" && (
        <BootSequence
          link={link}
          demo={settings.demo}
          leaving={phase === "opening"}
          onEnter={enter}
          onRetry={recheck}
          onDemo={startDemo}
          onLanded={onLanded}
        />
      )}
    </>
  );
}
