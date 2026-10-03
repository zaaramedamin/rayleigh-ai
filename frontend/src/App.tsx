import { useCallback, useMemo, useState } from "react";
import { createApi } from "./api/client";
import { createMockApi } from "./api/mock";
import { BootSequence } from "./components/BootSequence";
import { NavRail } from "./components/NavRail";
import type { ViewId } from "./components/NavRail";
import { StatusPanel } from "./components/StatusPanel";
import { useSettings } from "./state/settings";
import { useAssistant } from "./state/useAssistant";
import { useHealth } from "./state/useHealth";
import { ConsoleView } from "./views/ConsoleView";
import { KnowledgeView } from "./views/KnowledgeView";
import { ModulesView } from "./views/ModulesView";
import { PrivacyView } from "./views/PrivacyView";
import { SettingsView } from "./views/SettingsView";

export function App() {
  const { settings, update } = useSettings();
  const api = useMemo(() => (settings.demo ? createMockApi() : createApi()), [settings.demo]);
  const { link, recheck } = useHealth(api);
  const assistant = useAssistant(api);
  const [booted, setBooted] = useState(false);
  const [view, setView] = useState<ViewId>("console");

  const enter = useCallback(() => setBooted(true), []);

  if (!booted) {
    return (
      <>
        <div className="bg" aria-hidden />
        <BootSequence
          link={link}
          demo={settings.demo}
          onEnter={enter}
          onRetry={recheck}
          onDemo={() => update({ demo: true })}
        />
      </>
    );
  }

  return (
    <>
      <div className="bg" aria-hidden />
      <div className="scan" aria-hidden />
      <div className="shell">
        <header className="topbar">
          <div className="brand glitch" data-text="REYLEIGHT">
            REYLEIGHT
          </div>
          {settings.demo && <div className="banner">DEMO MODE // SIMULATED DATA</div>}
          {!settings.demo && link.state === "offline" && (
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
          <div className="topbar-right">
            <span className={`dot dot--${link.state}`} aria-hidden />
            <span className="muted">{link.state === "online" ? "LOCAL LINK" : link.state === "checking" ? "SCANNING" : "NO LINK"}</span>
          </div>
        </header>

        <NavRail view={view} onChange={setView} />

        <main className="main">
          {view === "console" && <ConsoleView assistant={assistant} />}
          {view === "knowledge" && <KnowledgeView api={api} />}
          {view === "modules" && <ModulesView />}
          {view === "privacy" && <PrivacyView />}
          {view === "settings" && <SettingsView />}
        </main>

        <aside className="side">
          <StatusPanel link={link} thinking={assistant.status === "thinking"} demo={settings.demo} messages={assistant.messages} />
        </aside>
      </div>
    </>
  );
}
