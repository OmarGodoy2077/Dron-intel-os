import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { DroneMap } from "./components/DroneMap";
import { LiveStats } from "./components/LiveStats";
import { RuleTerminal, RuleLogEntry } from "./components/RuleTerminal";
import { ControlPanel } from "./components/ControlPanel";
import { SidebarLeft } from "./components/SidebarLeft";
import { useSocket } from "./hooks/useSocket";
import {
  DroneState,
  DynamicsState,
  EpisodePoint,
  TrainingState,
  SystemHealth,
  StepUpdateMsg,
  EpisodeCompleteMsg,
  TrainingCompleteMsg,
  DroneStateResponse,
  TrainingStartResponse,
  TrainingStatus,
  RawLogEntry,
} from "./types";

// ─── Constants ────────────────────────────────────────────────────────────────

const WS_URL = "ws://localhost:8000/ws";
const API = "http://localhost:8000";
const GRID_SIZE = 50;
const MAX_EPISODE_HISTORY = 200;
const MAX_LOG_ENTRIES = 500;

// ─── Color palette (Neon / Cyberpunk) ────────────────────────────────────────

export const P = {
  drone:  "#22d3ee",
  warn:   "#fbbf24",
  crit:   "#ef4444",
  ok:     "#10b981",
  storm:  "#a855f7",
  bg:     "#07090d",
  panel:  "#0a0e14",
  border: "rgba(125,211,252,0.08)",
  text:   "#e2e8f0",
  muted:  "rgba(148,163,184,0.7)",
  mono:   "'JetBrains Mono', 'Fira Code', Consolas, monospace",
} as const;

// ─── Helpers ──────────────────────────────────────────────────────────────────

function normalizeLevel(raw: string): RuleLogEntry["level"] {
  const MAP: Record<string, RuleLogEntry["level"]> = {
    MASK:        "MASK",
    ALERT:       "ALERT",
    AVISO:       "ALERT",
    CRITICAL:    "CRITICAL",
    REWARD:      "REWARD",
    PRIO:        "PRIO",
    RUTA:        "RUTA",
    CLIMA:       "CLIMA",
    NEGOC:       "NEGOTIATION",
    NEGOTIATION: "NEGOTIATION",
    INFO:        "INFO",
  };
  return MAP[raw.toUpperCase()] ?? "INFO";
}

function parseSymLog(entries: RawLogEntry[]): RuleLogEntry[] {
  return entries.map((e) => {
    const agentM = e.message.match(/drone_(\d+)/);
    const ruleM  = e.message.match(/^(R\d+)\s/);
    return {
      timestamp: e.timestamp,
      level:     normalizeLevel(e.level),
      agent:     agentM ? `drone_${agentM[1]}` : "sys",
      rule:      ruleM ? ruleM[1] : undefined,
      message:   e.message,
    };
  });
}

function useUptime(): string {
  const start = useRef(Date.now());
  const [uptime, setUptime] = useState("00:00:00");
  useEffect(() => {
    const id = setInterval(() => {
      const s  = Math.floor((Date.now() - start.current) / 1000);
      const hh = String(Math.floor(s / 3600)).padStart(2, "0");
      const mm = String(Math.floor((s % 3600) / 60)).padStart(2, "0");
      const ss = String(s % 60).padStart(2, "0");
      setUptime(`${hh}:${mm}:${ss}`);
    }, 1000);
    return () => clearInterval(id);
  }, []);
  return uptime;
}

// ─── App ──────────────────────────────────────────────────────────────────────

export default function App() {
  const { isConnected, lastMessage } = useSocket(WS_URL);
  const uptime = useUptime();

  const [droneState, setDroneState] = useState<DroneState>({
    positions: [], batteries: [], alive: [], rewards: [],
  });
  const [dynamics, setDynamics] = useState<DynamicsState>({ storms: 0, winds: 0, nfzs: 0 });
  const [training, setTraining] = useState<TrainingState>({
    isTraining: false, system: "neuro_dqn",
    currentEpisode: 0, currentStep: 0, maxEpisodes: 200,
  });
  const [epHistory, setEpHistory] = useState<EpisodePoint[]>([]);
  const [symLogs, setSymLogs]     = useState<RuleLogEntry[]>([]);
  const [health, setHealth]       = useState<SystemHealth | null>(null);
  const [demandZones, setDemandZones] = useState<[number, number][]>([]);
  const [activeTab, setActiveTab] = useState("operaciones");
  const [compareData, setCompareData] = useState<Record<string, unknown> | null>(null);
  const [trainingStatus, setTrainingStatus] = useState<TrainingStatus | null>(null);

  // Estado de conocimiento previo (checkpoints + episodios registrados)
  const refreshStatus = useCallback(() => {
    fetch(`${API}/training/status`)
      .then<TrainingStatus>((r) => r.json())
      .then(setTrainingStatus)
      .catch(() => {});
  }, []);

  // ── Initial load ────────────────────────────────────────────────────────────
  useEffect(() => {
    refreshStatus();
    fetch(`${API}/health`)
      .then<SystemHealth>((r) => r.json())
      .then(setHealth)
      .catch(() => {});

    fetch(`${API}/drone-state`)
      .then<DroneStateResponse>((r) => r.json())
      .then((data) => {
        if (Array.isArray(data.positions)) {
          setDroneState({
            positions: data.positions,
            batteries: data.batteries,
            alive:     data.alive,
            rewards:   new Array(data.positions.length).fill(0),
          });
        }
        if (Array.isArray(data.demand_zones)) {
          setDemandZones(data.demand_zones as [number, number][]);
        }
      })
      .catch(() => {});
  }, []);

  // ── WebSocket messages ───────────────────────────────────────────────────────
  useEffect(() => {
    if (!lastMessage) return;
    const msg = lastMessage as unknown;

    const t = (msg as { type?: string }).type;

    if (t === "step_update") {
      const m = msg as StepUpdateMsg;
      const len = (m.positions ?? []).length;
      setDroneState({
        positions: m.positions  ?? [],
        batteries: m.batteries  ?? [],
        alive:     m.alive ?? new Array(len).fill(true),
        rewards:   m.rewards    ?? [],
      });
      if (m.dynamics) setDynamics(m.dynamics);
      setTraining((prev) => ({
        ...prev,
        isTraining:     true,
        system:         m.system as TrainingState["system"],
        currentEpisode: m.episode,
        currentStep:    m.step,
      }));
    } else if (t === "episode_complete") {
      const m = msg as EpisodeCompleteMsg;
      const rec = m.record;
      setEpHistory((prev) => [
        ...prev.slice(-(MAX_EPISODE_HISTORY - 1)),
        {
          episode:               m.episode,
          reward:                rec.total_reward,
          successRate:           rec.success_rate,
          ruleViolations:        rec.rule_violations,
          symbolicInterventions: rec.symbolic_ops,
          avgBattery:            rec.avg_battery_remaining,
          deliveries:            rec.deliveries,
          epsilon:               rec.epsilon,
          avgLoss:               rec.avg_loss,
        },
      ]);
      if (Array.isArray(m.symbolic_log) && m.symbolic_log.length > 0) {
        const parsed = parseSymLog(m.symbolic_log);
        setSymLogs((prev) => [...prev, ...parsed].slice(-MAX_LOG_ENTRIES));
      }
      if (Array.isArray(m.demand_zones)) {
        setDemandZones(m.demand_zones as [number, number][]);
      }
      setTraining((prev) => ({
        ...prev,
        system:         m.system as TrainingState["system"],
        currentEpisode: m.episode,
      }));
    } else if (t === "training_complete") {
      const m = msg as TrainingCompleteMsg;
      setTraining((prev) => ({ ...prev, isTraining: false, system: m.system as TrainingState["system"] }));
      refreshStatus();
    }
  }, [lastMessage, refreshStatus]);

  // ── Control handlers ─────────────────────────────────────────────────────────
  const handleStart = useCallback((
    system: TrainingState["system"],
    episodes: number,
    mode: "resume" | "scratch",
  ) => {
    fetch(`${API}/training/start?system=${system}&episodes=${episodes}&mode=${mode}`, { method: "POST" })
      .then<TrainingStartResponse>((r) => r.json())
      .then((data) => {
        if (!data.error) {
          setTraining((prev) => ({
            ...prev, isTraining: true, system, maxEpisodes: episodes,
            currentEpisode: 0, currentStep: 0,
          }));
          // En 'resume' conservamos el histórico en pantalla; en 'scratch' lo limpiamos.
          if (mode === "scratch") {
            setEpHistory([]);
            setSymLogs([]);
          }
        }
      })
      .catch(() => {});
  }, []);

  const handleStop = useCallback(() => {
    fetch(`${API}/training/stop`, { method: "POST" })
      .then(() => setTimeout(refreshStatus, 500))
      .catch(() => {});
  }, [refreshStatus]);

  const handleDeleteData = useCallback((system?: string) => {
    const url = system
      ? `${API}/training/delete-data?system=${system}`
      : `${API}/training/delete-data`;
    fetch(url, { method: "POST" })
      .then((r) => r.json())
      .then(() => {
        setEpHistory([]);
        setSymLogs([]);
        refreshStatus();
      })
      .catch(() => {});
  }, [refreshStatus]);

  const handleTabChange = useCallback((tab: string) => {
    setActiveTab(tab);
    if (tab === "historico" && !compareData) {
      fetch(`${API}/metrics/comparison`)
        .then<Record<string, unknown>>((r) => r.json())
        .then(setCompareData)
        .catch(() => {});
    }
  }, [compareData]);

  // ── Derived KPIs ─────────────────────────────────────────────────────────────
  const latest   = epHistory[epHistory.length - 1];
  const avgRwd   = epHistory.length > 0
    ? (epHistory.slice(-20).reduce((a, b) => a + b.reward, 0) / Math.min(20, epHistory.length)).toFixed(1)
    : "—";
  const successPct = latest ? (latest.successRate * 100).toFixed(1) + "%" : "—";
  const violations = latest ? String(latest.ruleViolations) : "—";
  const epsilonKpi = latest?.epsilon != null ? latest.epsilon.toFixed(3) : "—";

  // ── Layout ───────────────────────────────────────────────────────────────────
  return (
    <div style={{
      width: "100vw", height: "100vh",
      background: P.bg, color: P.text,
      fontFamily: "Inter, system-ui, sans-serif",
      display: "grid",
      gridTemplateColumns: "220px 1fr 360px",
      gridTemplateRows: "52px 1fr 220px",
      gap: 1,
      overflow: "hidden",
    }}>

      {/* ── TOP BAR ──────────────────────────────────────── */}
      <TopBar
        isConnected={isConnected}
        training={training}
        health={health}
        uptime={uptime}
      />

      {/* ── LEFT SIDEBAR ─────────────────────────────────── */}
      <SidebarLeft
        droneState={droneState}
        dynamics={dynamics}
        activeTab={activeTab}
        onTabChange={handleTabChange}
      />

      {/* ── MAIN CONTENT (switches by activeTab) ─────────── */}
      <main style={{
        background: P.bg,
        display: "flex", flexDirection: "column",
        overflow: "hidden", minHeight: 0,
        borderBottom: `1px solid ${P.border}`,
      }}>
        {activeTab === "reglas" ? (
          <PrologRulesView />
        ) : activeTab === "historico" ? (
          <HistoricoView compareData={compareData} epHistory={epHistory} />
        ) : activeTab === "flota" ? (
          <FlotaView droneState={droneState} />
        ) : activeTab === "entrenamiento" ? (
          <EntrenamientoView
            epHistory={epHistory}
            training={training}
            health={health}
          />
        ) : activeTab === "config" ? (
          <ConfigView health={health} training={training} />
        ) : (
          <>
            {/* Map toolbar */}
            <div style={{
              display: "flex", alignItems: "center",
              padding: "8px 16px", gap: 10,
              borderBottom: `1px solid ${P.border}`,
              fontSize: 11, color: P.muted, letterSpacing: "0.08em",
            }}>
              <span style={{ color: P.drone, fontWeight: 500 }}>◈</span>
              MAPA OPERACIONAL · GRID {GRID_SIZE}×{GRID_SIZE}
              {demandZones.length > 0 && (
                <span style={{
                  color: P.drone, fontFamily: P.mono, fontSize: 10,
                  background: `${P.drone}12`, border: `1px solid ${P.drone}33`,
                  borderRadius: 4, padding: "2px 8px",
                }}>
                  ▢ {demandZones.length} zonas demanda ML
                </span>
              )}
              {(dynamics.storms > 0 || dynamics.winds > 0 || dynamics.nfzs > 0) && (
                <span style={{
                  marginLeft: "auto", color: P.storm,
                  fontFamily: P.mono, fontSize: 10,
                  background: `${P.storm}15`, border: `1px solid ${P.storm}44`,
                  borderRadius: 4, padding: "2px 8px",
                }}>
                  {dynamics.storms > 0 && `⚡ ${dynamics.storms} storm${dynamics.storms > 1 ? "s" : ""}`}
                  {dynamics.winds  > 0 && `  ·  💨 ${dynamics.winds}`}
                  {dynamics.nfzs   > 0 && `  ·  🚫 ${dynamics.nfzs} NFZ`}
                </span>
              )}
            </div>
            {/* Map container */}
            <div style={{
              flex: 1, overflow: "auto",
              display: "flex", alignItems: "flex-start",
              justifyContent: "flex-start",
              padding: 12,
            }}>
              <DroneMap
                gridSize={GRID_SIZE}
                dronePositions={droneState.positions}
                droneBatteries={droneState.batteries}
                droneAlive={droneState.alive}
                noFlyZones={[]}
                stormRegions={[]}
                demandZones={demandZones}
              />
            </div>
          </>
        )}
      </main>

      {/* ── RIGHT RAIL ───────────────────────────────────── */}
      <aside style={{
        background: P.panel,
        borderLeft: `1px solid ${P.border}`,
        borderBottom: `1px solid ${P.border}`,
        padding: 14,
        display: "flex", flexDirection: "column", gap: 12,
        overflowY: "auto", overflowX: "hidden",
      }}>
        {/* KPI row */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
          <Kpi label="Tasa éxito"  value={successPct} accent={P.ok}    sub="último ep." />
          <Kpi label="Reward avg"  value={avgRwd}     accent={P.drone} sub="ventana 20ep" />
          <Kpi label="Violaciones" value={violations} sub="último ep." />
          <Kpi label="Epsilon ε"   value={epsilonKpi} accent={P.warn}  sub="exploración" />
        </div>

        {/* Live charts */}
        <LiveStats
          episodeData={epHistory}
          systemName={training.system}
          currentEpisode={training.currentEpisode}
          isTraining={training.isTraining}
        />

        {/* Training controls */}
        <ControlPanel
          training={training}
          health={health}
          status={trainingStatus}
          onStart={handleStart}
          onStop={handleStop}
          onDeleteData={handleDeleteData}
        />
      </aside>

      {/* ── BOTTOM LEFT: system stats ─────────────────────── */}
      <div style={{
        background: P.panel,
        borderTop: `1px solid ${P.border}`,
        borderRight: `1px solid ${P.border}`,
        padding: "10px 14px",
        display: "flex", flexDirection: "column", gap: 5,
        overflow: "hidden",
      }}>
        <StatBlock training={training} dynamics={dynamics} droneState={droneState} health={health} />
      </div>

      {/* ── BOTTOM: Prolog terminal ───────────────────────── */}
      <div style={{
        gridColumn: "2 / 4",
        background: P.panel,
        borderTop: `1px solid ${P.border}`,
        overflow: "hidden",
      }}>
        <RuleTerminal logs={symLogs} maxLines={200} />
      </div>
    </div>
  );
}

// ─── Tab Views ────────────────────────────────────────────────────────────────

const PROLOG_RULES = [
  { id: "R1",  name: "No-Fly Zone",          type: "Mask",       weight: -100, desc: "Bloquea movimiento a celdas NFZ estáticas y dinámicas" },
  { id: "R2",  name: "Batería crítica <15%",  type: "Mask+Shape", weight: -50,  desc: "Fuerza aterrizaje/carga; penaliza −50/paso si no cumple" },
  { id: "R3",  name: "Colisión inminente",    type: "Mask",       weight: -200, desc: "Bloquea movimiento si otra celda ya ocupada por dron" },
  { id: "R4",  name: "Conflicto de celda",    type: "Shape",      weight: -30,  desc: "Penaliza −30·(N−1) por cada dron adicional en la celda" },
  { id: "R5",  name: "Estación ocupada",      type: "Mask",       weight: -20,  desc: "Bloquea 'cargar' si la estación ya tiene otro dron" },
  { id: "R6",  name: "Entrega médica",        type: "Shape",      weight: +150, desc: "Bonus +150 por entregar paquete de prioridad médica" },
  { id: "R7",  name: "Tormenta activa",       type: "Mask",       weight: -80,  desc: "Bloquea vuelo en celdas de tormenta activa" },
  { id: "R8",  name: "Viento fuerte",         type: "Shape",      weight: -15,  desc: "Penaliza −15 por moverse contra viento de alta intensidad" },
  { id: "R9",  name: "Zona congestionada",    type: "Shape",      weight: -40,  desc: "Penaliza −40 si >2 drones en radio de 3 celdas" },
  { id: "R10", name: "Ruta eficiente",        type: "Shape",      weight: +20,  desc: "Bonus +20 si la acción reduce distancia Manhattan al objetivo" },
  { id: "R11", name: "Negociación de paso",   type: "Shape",      weight: +25,  desc: "+25 al ceder el paso; −10 al recibir prioridad" },
  { id: "R12", name: "Predicción fallo bat.", type: "Mask",       weight: -500, desc: "Bloquea misión si ML predice fallo de batería antes de retorno" },
];

function PrologRulesView() {
  return (
    <div style={{ flex: 1, overflow: "auto", padding: "20px 24px" }}>
      <div style={{ fontSize: 11, letterSpacing: "0.1em", color: P.muted, marginBottom: 16 }}>
        MOTOR SIMBÓLICO · 12 REGLAS PROLOG
      </div>
      <table style={{ width: "100%", borderCollapse: "collapse", fontFamily: P.mono, fontSize: 11 }}>
        <thead>
          <tr style={{ borderBottom: `1px solid ${P.border}` }}>
            {["Regla", "Nombre", "Tipo", "Peso", "Descripción"].map((h) => (
              <th key={h} style={{ textAlign: "left", padding: "6px 10px", color: P.muted, fontWeight: 500, fontSize: 10, letterSpacing: "0.08em" }}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {PROLOG_RULES.map((r, i) => {
            const wColor = r.weight > 0 ? P.ok : r.weight < -100 ? P.crit : P.warn;
            return (
              <tr key={r.id} style={{ borderBottom: `1px solid rgba(125,211,252,0.04)`, background: i % 2 === 0 ? "rgba(255,255,255,0.01)" : "transparent" }}>
                <td style={{ padding: "8px 10px", color: P.drone, fontWeight: 600 }}>{r.id}</td>
                <td style={{ padding: "8px 10px", color: P.text }}>{r.name}</td>
                <td style={{ padding: "8px 10px" }}>
                  <span style={{ fontSize: 9.5, padding: "2px 6px", borderRadius: 3, background: r.type === "Mask" ? `${P.crit}18` : r.type === "Shape" ? `${P.ok}18` : `${P.warn}18`, color: r.type === "Mask" ? P.crit : r.type === "Shape" ? P.ok : P.warn }}>
                    {r.type}
                  </span>
                </td>
                <td style={{ padding: "8px 10px", color: wColor, fontWeight: 600 }}>{r.weight > 0 ? `+${r.weight}` : r.weight}</td>
                <td style={{ padding: "8px 10px", color: "rgba(148,163,184,0.7)", fontSize: 10.5 }}>{r.desc}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function HistoricoView({ compareData, epHistory }: { compareData: Record<string, unknown> | null; epHistory: EpisodePoint[] }) {
  const systems = ["astar", "dqn", "neuro_dqn"] as const;
  const totalEpisodes = epHistory.length;

  const compRows = useMemo(() => {
    if (!compareData) return [];
    const keys = Object.keys(compareData);
    if (keys.length === 0) return [];
    const firstKey = keys[0];
    const firstCol = compareData[firstKey] as Record<string, unknown>;
    return Object.keys(firstCol).map((idx) => {
      const row: Record<string, unknown> = { idx };
      keys.forEach((k) => { row[k] = (compareData[k] as Record<string, unknown>)[idx]; });
      return row;
    });
  }, [compareData]);

  const compCols = compareData ? Object.keys(compareData) : [];

  return (
    <div style={{ flex: 1, overflow: "auto", padding: "20px 24px" }}>
      <div style={{ fontSize: 11, letterSpacing: "0.1em", color: P.muted, marginBottom: 16 }}>
        HISTÓRICO DE ENTRENAMIENTO
      </div>

      {/* Session summary */}
      <div style={{ display: "flex", gap: 12, marginBottom: 24 }}>
        {systems.map((sys) => {
          const label = sys === "neuro_dqn" ? "Neuro-DQN" : sys === "dqn" ? "DQN Puro" : "A* Baseline";
          const maxDel = epHistory.length > 0 ? Math.max(...epHistory.map(e => e.deliveries ?? 0)) : 0;
          const avgSucc = epHistory.length > 0 ? (epHistory.reduce((a, b) => a + b.successRate, 0) / epHistory.length * 100).toFixed(1) : "—";
          const latestEps = epHistory[epHistory.length - 1]?.epsilon;
          return (
            <div key={sys} style={{ flex: 1, padding: "12px 14px", borderRadius: 8, background: "rgba(255,255,255,0.015)", border: `1px solid ${P.border}` }}>
              <div style={{ fontSize: 10, color: P.drone, fontFamily: P.mono, marginBottom: 6 }}>{label}</div>
              <div style={{ fontSize: 13, color: P.text, fontFamily: P.mono }}>{totalEpisodes} ep</div>
              <div style={{ fontSize: 10, color: P.muted }}>
                Éxito avg: <span style={{ color: P.ok }}>{avgSucc}%</span>
                {" · "}Mejor entrega: <span style={{ color: P.drone }}>{maxDel}/10</span>
              </div>
              {latestEps != null && (
                <div style={{ fontSize: 9, color: P.muted, marginTop: 3 }}>
                  ε actual: <span style={{ color: P.warn }}>{latestEps.toFixed(3)}</span>
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Comparison table from API */}
      {compRows.length > 0 ? (
        <>
          <div style={{ fontSize: 10, color: P.muted, marginBottom: 8, letterSpacing: "0.08em" }}>TABLA COMPARATIVA — /metrics/comparison</div>
          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontFamily: P.mono, fontSize: 10.5 }}>
              <thead>
                <tr style={{ borderBottom: `1px solid ${P.border}` }}>
                  {compCols.map((c) => (
                    <th key={c} style={{ textAlign: "left", padding: "6px 10px", color: P.muted, fontWeight: 500, fontSize: 10, letterSpacing: "0.06em" }}>
                      {c.replace(/_/g, " ").toUpperCase()}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {compRows.slice(0, 50).map((row, i) => (
                  <tr key={i} style={{ borderBottom: `1px solid rgba(125,211,252,0.04)`, background: i % 2 === 0 ? "rgba(255,255,255,0.01)" : "transparent" }}>
                    {compCols.map((c) => (
                      <td key={c} style={{ padding: "6px 10px", color: P.text }}>
                        {typeof row[c] === "number" ? (row[c] as number).toFixed(3) : String(row[c] ?? "—")}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      ) : (
        <div style={{ fontSize: 11, color: "rgba(148,163,184,0.4)", fontFamily: P.mono, padding: "20px 0" }}>
          {compareData === null ? "Cargando datos históricos…" : "Sin datos de comparación disponibles. Ejecuta entrenamiento con múltiples sistemas."}
        </div>
      )}

      {/* Recent episodes this session */}
      {epHistory.length > 0 && (
        <>
          <div style={{ fontSize: 10, color: P.muted, marginBottom: 8, marginTop: 24, letterSpacing: "0.08em" }}>EPISODIOS SESIÓN ACTUAL (últimos 50)</div>
          <table style={{ width: "100%", borderCollapse: "collapse", fontFamily: P.mono, fontSize: 10.5 }}>
            <thead>
              <tr style={{ borderBottom: `1px solid ${P.border}` }}>
                {["EP", "Reward", "Éxito %", "Entregas", "Violaciones", "Batería avg", "ε"].map((h) => (
                  <th key={h} style={{ textAlign: "left", padding: "6px 10px", color: P.muted, fontWeight: 500, fontSize: 10 }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {[...epHistory].reverse().slice(0, 50).map((ep, i) => (
                <tr key={i} style={{ borderBottom: `1px solid rgba(125,211,252,0.04)` }}>
                  <td style={{ padding: "6px 10px", color: P.drone }}>{ep.episode}</td>
                  <td style={{ padding: "6px 10px", color: ep.reward >= 0 ? P.ok : P.crit }}>{ep.reward.toFixed(0)}</td>
                  <td style={{ padding: "6px 10px", color: ep.successRate >= 0.9 ? P.ok : ep.successRate >= 0.5 ? P.warn : P.crit }}>{(ep.successRate * 100).toFixed(1)}%</td>
                  <td style={{ padding: "6px 10px", color: (ep.deliveries ?? 0) > 0 ? P.ok : P.muted }}>{ep.deliveries ?? 0}/10</td>
                  <td style={{ padding: "6px 10px", color: ep.ruleViolations > 0 ? P.crit : P.text }}>{ep.ruleViolations}</td>
                  <td style={{ padding: "6px 10px", color: P.muted }}>{ep.avgBattery != null ? ep.avgBattery.toFixed(1) + "%" : "—"}</td>
                  <td style={{ padding: "6px 10px", color: (ep.epsilon ?? 1) < 0.5 ? P.ok : P.warn }}>{ep.epsilon?.toFixed(3) ?? "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </div>
  );
}

function FlotaView({ droneState }: { droneState: DroneState }) {
  const { positions, batteries, alive, rewards } = droneState;
  return (
    <div style={{ flex: 1, overflow: "auto", padding: "20px 24px" }}>
      <div style={{ fontSize: 11, letterSpacing: "0.1em", color: P.muted, marginBottom: 16 }}>
        GESTIÓN DE FLOTA · {positions.length} DRONES
      </div>
      {positions.length === 0 ? (
        <div style={{ fontSize: 11, color: "rgba(148,163,184,0.4)", fontFamily: P.mono }}>Sin telemetría — inicia el entrenamiento.</div>
      ) : (
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(260px, 1fr))", gap: 12 }}>
          {positions.map((pos, i) => {
            const bat   = batteries[i] ?? 0;
            const isAlive = alive.length === 0 || alive[i];
            const rwd   = rewards[i] ?? 0;
            const x = pos[0] ?? 0;
            const y = pos[1] ?? 0;
            const z = pos[2] ?? 0;
            const batColor  = bat > 50 ? P.ok : bat > 25 ? P.warn : P.crit;
            const statusLabel = !isAlive ? "OFFLINE" : bat < 15 ? "CRÍTICO" : bat < 30 ? "BAJO" : "ACTIVO";
            return (
              <div key={i} style={{
                padding: "14px 16px", borderRadius: 8,
                background: "rgba(125,211,252,0.025)",
                border: `1px solid ${isAlive ? P.border : "rgba(239,68,68,0.2)"}`,
                opacity: isAlive ? 1 : 0.5,
              }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
                  <span style={{ fontFamily: P.mono, fontWeight: 700, color: P.drone }}>D-{String(i + 1).padStart(2, "0")}</span>
                  <span style={{ fontSize: 10, padding: "2px 7px", borderRadius: 4, background: `${batColor}18`, color: batColor, fontFamily: P.mono }}>{statusLabel}</span>
                </div>
                <div style={{ height: 4, background: "rgba(255,255,255,0.06)", borderRadius: 2, overflow: "hidden", marginBottom: 10 }}>
                  <div style={{ width: `${bat}%`, height: "100%", background: batColor, transition: "width 0.5s, background 0.5s" }} />
                </div>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "6px 12px", fontFamily: P.mono, fontSize: 10.5 }}>
                  <FlotaStat l="Batería"  v={`${bat.toFixed(1)}%`}  c={batColor} />
                  <FlotaStat l="Reward"   v={rwd.toFixed(1)}         c={rwd >= 0 ? P.ok : P.crit} />
                  <FlotaStat l="Pos X/Y"  v={`${x.toFixed(0)},${y.toFixed(0)}`} />
                  <FlotaStat l="Altitud"  v={`${z.toFixed(1)} u`} />
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

function FlotaStat({ l, v, c }: { l: string; v: string; c?: string }) {
  return (
    <div>
      <div style={{ fontSize: 9, color: "rgba(148,163,184,0.4)", letterSpacing: "0.08em", marginBottom: 2 }}>{l.toUpperCase()}</div>
      <div style={{ color: c ?? P.text, fontWeight: 500 }}>{v}</div>
    </div>
  );
}

// ─── EntrenamientoView ────────────────────────────────────────────────────────

interface EntrenamientoViewProps {
  epHistory: EpisodePoint[];
  training: TrainingState;
  health: SystemHealth | null;
}

function EntrenamientoView({ epHistory, training }: EntrenamientoViewProps) {
  const latest    = epHistory[epHistory.length - 1];
  const eps       = latest?.epsilon ?? 1.0;
  const exploitPct = Math.round((1 - eps) * 100);
  const avgReward20 = epHistory.length > 0
    ? epHistory.slice(-20).reduce((a, b) => a + b.reward, 0) / Math.min(20, epHistory.length)
    : null;
  const maxDeliveries = epHistory.length > 0
    ? Math.max(...epHistory.map(e => e.deliveries ?? 0))
    : 0;
  const avgLoss   = latest?.avgLoss ?? 0;

  // Trend: compare last 10 vs prev 10
  const trend = useMemo(() => {
    if (epHistory.length < 20) return null;
    const recent = epHistory.slice(-10).reduce((a, b) => a + b.reward, 0) / 10;
    const older  = epHistory.slice(-20, -10).reduce((a, b) => a + b.reward, 0) / 10;
    return recent - older;
  }, [epHistory]);

  // Phase label
  const phase = eps > 0.7 ? "EXPLORACIÓN" : eps > 0.3 ? "TRANSICIÓN" : "EXPLOTACIÓN";
  const phaseColor = eps > 0.7 ? P.warn : eps > 0.3 ? P.drone : P.ok;

  return (
    <div style={{ flex: 1, overflow: "auto", padding: "20px 24px" }}>
      <div style={{ fontSize: 11, letterSpacing: "0.1em", color: P.muted, marginBottom: 20 }}>
        PANEL DE ENTRENAMIENTO · {training.system.toUpperCase()}
      </div>

      {/* Phase / KPI cards */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 12, marginBottom: 24 }}>
        {/* Exploration phase */}
        <div style={{ padding: "14px 16px", borderRadius: 8, background: `${phaseColor}10`, border: `1px solid ${phaseColor}30` }}>
          <div style={{ fontSize: 9, color: P.muted, letterSpacing: "0.1em", marginBottom: 6 }}>FASE ACTUAL</div>
          <div style={{ fontSize: 14, fontFamily: P.mono, fontWeight: 700, color: phaseColor }}>{phase}</div>
          <div style={{ fontSize: 10, color: P.muted, marginTop: 4 }}>ε = {eps.toFixed(3)}</div>
        </div>

        {/* Explotación progress */}
        <div style={{ padding: "14px 16px", borderRadius: 8, background: "rgba(255,255,255,0.015)", border: `1px solid ${P.border}` }}>
          <div style={{ fontSize: 9, color: P.muted, letterSpacing: "0.1em", marginBottom: 6 }}>EXPLOTACIÓN</div>
          <div style={{ fontSize: 14, fontFamily: P.mono, fontWeight: 700, color: P.ok }}>{exploitPct}%</div>
          <div style={{ height: 3, background: "rgba(255,255,255,0.06)", borderRadius: 2, marginTop: 8, overflow: "hidden" }}>
            <div style={{ width: `${exploitPct}%`, height: "100%", background: P.ok }} />
          </div>
        </div>

        {/* Avg reward */}
        <div style={{ padding: "14px 16px", borderRadius: 8, background: "rgba(255,255,255,0.015)", border: `1px solid ${P.border}` }}>
          <div style={{ fontSize: 9, color: P.muted, letterSpacing: "0.1em", marginBottom: 6 }}>REWARD MEDIO (20ep)</div>
          <div style={{ fontSize: 14, fontFamily: P.mono, fontWeight: 700, color: avgReward20 != null && avgReward20 >= 0 ? P.ok : P.crit }}>
            {avgReward20 != null ? avgReward20.toFixed(0) : "—"}
          </div>
          {trend != null && (
            <div style={{ fontSize: 10, color: trend >= 0 ? P.ok : P.crit, marginTop: 4 }}>
              {trend >= 0 ? "▲" : "▼"} {Math.abs(trend).toFixed(0)} vs 10ep ant.
            </div>
          )}
        </div>

        {/* Best deliveries */}
        <div style={{ padding: "14px 16px", borderRadius: 8, background: "rgba(255,255,255,0.015)", border: `1px solid ${P.border}` }}>
          <div style={{ fontSize: 9, color: P.muted, letterSpacing: "0.1em", marginBottom: 6 }}>MEJOR ENTREGA</div>
          <div style={{ fontSize: 14, fontFamily: P.mono, fontWeight: 700, color: P.drone }}>
            {maxDeliveries}/10
          </div>
          <div style={{ fontSize: 10, color: P.muted, marginTop: 4 }}>
            loss avg: {avgLoss > 0 ? avgLoss.toFixed(4) : "—"}
          </div>
        </div>
      </div>

      {/* Epsilon decay progress bar */}
      <div style={{ marginBottom: 24, padding: "14px 16px", borderRadius: 8, background: "rgba(255,255,255,0.015)", border: `1px solid ${P.border}` }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
          <div style={{ fontSize: 10, color: P.muted, letterSpacing: "0.08em" }}>CURVA ε-GREEDY · EXPLORACIÓN → EXPLOTACIÓN</div>
          <div style={{ fontSize: 10, fontFamily: P.mono, color: P.drone }}>
            Ep {training.currentEpisode}/{training.maxEpisodes}
          </div>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 6 }}>
          <span style={{ fontSize: 9, color: P.muted, fontFamily: P.mono, minWidth: 55 }}>ε=1.0 RAND</span>
          <div style={{ flex: 1, height: 10, background: "rgba(255,255,255,0.06)", borderRadius: 5, overflow: "hidden", position: "relative" }}>
            {/* Random zone */}
            <div style={{ position: "absolute", left: 0, top: 0, height: "100%", width: `${Math.min(100, (1 - eps) * 100)}%`,
              background: `linear-gradient(90deg, ${P.warn}88, ${P.drone}88, ${P.ok}88)` }} />
            {/* Current position */}
            <div style={{
              position: "absolute", top: 0, height: "100%",
              left: `calc(${(1 - eps) * 100}% - 2px)`, width: 3,
              background: P.text, borderRadius: 2,
            }} />
          </div>
          <span style={{ fontSize: 9, color: P.muted, fontFamily: P.mono, minWidth: 55, textAlign: "right" }}>ε=0.05 EXPL</span>
        </div>
        <div style={{ display: "flex", justifyContent: "space-around", fontSize: 9, color: "rgba(148,163,184,0.4)", fontFamily: P.mono }}>
          <span style={{ color: P.warn }}>► ep 0 (exploración)</span>
          <span style={{ color: P.drone }}>► ep ~140 (ε=0.5)</span>
          <span style={{ color: P.ok }}>► ep ~500 (ε_min)</span>
        </div>
      </div>

      {/* Learning curve — episode table with trend */}
      {epHistory.length > 0 ? (
        <>
          <div style={{ fontSize: 10, color: P.muted, marginBottom: 8, letterSpacing: "0.08em" }}>
            HISTORIAL DE APRENDIZAJE (últimos {Math.min(50, epHistory.length)} episodios)
          </div>
          <table style={{ width: "100%", borderCollapse: "collapse", fontFamily: P.mono, fontSize: 10.5 }}>
            <thead>
              <tr style={{ borderBottom: `1px solid ${P.border}` }}>
                {["EP", "Reward", "Δ", "Éxito", "Entregas", "Batería avg", "ε", "Loss"].map((h) => (
                  <th key={h} style={{ textAlign: "left", padding: "6px 10px", color: P.muted, fontWeight: 500, fontSize: 10 }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {[...epHistory].reverse().slice(0, 50).map((ep, i, arr) => {
                const prev = arr[i + 1];
                const delta = prev ? ep.reward - prev.reward : 0;
                const improving = delta > 0;
                return (
                  <tr key={ep.episode} style={{ borderBottom: `1px solid rgba(125,211,252,0.04)`, background: i % 2 === 0 ? "rgba(255,255,255,0.01)" : "transparent" }}>
                    <td style={{ padding: "6px 10px", color: P.drone }}>{ep.episode}</td>
                    <td style={{ padding: "6px 10px", color: ep.reward >= 0 ? P.ok : P.crit }}>{ep.reward.toFixed(0)}</td>
                    <td style={{ padding: "6px 10px", color: delta > 0 ? P.ok : delta < 0 ? P.crit : P.muted, fontWeight: 600 }}>
                      {delta !== 0 ? (improving ? "▲" : "▼") + Math.abs(delta).toFixed(0) : "—"}
                    </td>
                    <td style={{ padding: "6px 10px", color: ep.successRate >= 0.5 ? P.ok : ep.successRate > 0 ? P.warn : P.muted }}>
                      {(ep.successRate * 100).toFixed(0)}%
                    </td>
                    <td style={{ padding: "6px 10px", color: (ep.deliveries ?? 0) > 0 ? P.ok : P.muted }}>
                      {ep.deliveries ?? 0}/10
                    </td>
                    <td style={{ padding: "6px 10px", color: P.muted }}>{ep.avgBattery != null ? ep.avgBattery.toFixed(1) + "%" : "—"}</td>
                    <td style={{ padding: "6px 10px", color: (ep.epsilon ?? 1) < 0.5 ? P.ok : P.warn }}>{(ep.epsilon ?? 1).toFixed(3)}</td>
                    <td style={{ padding: "6px 10px", color: P.muted }}>{ep.avgLoss != null && ep.avgLoss > 0 ? ep.avgLoss.toFixed(4) : "—"}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </>
      ) : (
        <div style={{ padding: "40px 0", textAlign: "center", color: "rgba(148,163,184,0.35)", fontFamily: P.mono, fontSize: 12 }}>
          Sin datos de entrenamiento — inicia una sesión con los controles de la derecha.
        </div>
      )}
    </div>
  );
}

// ─── ConfigView ───────────────────────────────────────────────────────────────

interface ConfigViewProps {
  health: SystemHealth | null;
  training: TrainingState;
}

function ConfigView({ health, training }: ConfigViewProps) {
  const hyperparams = [
    { k: "Algoritmo base",   v: "Double-DQN (DDQN)" },
    { k: "Dimensión estado", v: "11 (x,y,z,β,κ,ω,η,tdx,tdy,cdx,cdy)" },
    { k: "Dimensión acción", v: "8 (despegar, aterrizar, N/S/E/O, esperar, cargar)" },
    { k: "Hidden dim",       v: "256 × 2 capas + LayerNorm" },
    { k: "Gamma (γ)",        v: "0.99" },
    { k: "Learning rate",    v: "1e-3 (Adam)" },
    { k: "Batch size",       v: "64" },
    { k: "Buffer capacity",  v: "100 000 transiciones" },
    { k: "Tau (τ soft)",     v: "0.001 (Polyak avg)" },
    { k: "Epsilon ε₀",       v: "1.0 → 0.05 (d=0.995/ep)" },
    { k: "Learn freq.",      v: "cada 4 pasos" },
    { k: "Prolog bridge",    v: health?.symbolic_ok ? "✓ Activo" : "✗ Offline" },
  ];

  const envParams = [
    { k: "Grid",             v: `${health?.grid_size ?? 50} × ${health?.grid_size ?? 50}` },
    { k: "Drones",           v: String(health?.num_drones ?? 5) },
    { k: "Paquetes/ep",      v: "10 (2 médicos + 8 estándar)" },
    { k: "Estaciones carga", v: "4" },
    { k: "NFZ estáticas",    v: "5 zonas × radio 2" },
    { k: "Batería costo",    v: "0.75%/mov · 0.25%/espera" },
    { k: "Batería recarga",  v: "+25%/step en estación" },
    { k: "Max steps/ep",     v: "500" },
    { k: "Predictor demanda", v: health?.ml_ok ? "✓ GBR activo (sesga destinos)" : "✗ Offline" },
    { k: "Suite de tests",   v: "113 tests · pytest /tests" },
    { k: "Sistema activo",   v: training.system.toUpperCase() },
  ];

  return (
    <div style={{ flex: 1, overflow: "auto", padding: "20px 24px" }}>
      <div style={{ fontSize: 11, letterSpacing: "0.1em", color: P.muted, marginBottom: 20 }}>
        CONFIGURACIÓN DEL SISTEMA
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24 }}>
        {/* Hyperparameters */}
        <div>
          <div style={{ fontSize: 10, color: P.drone, letterSpacing: "0.1em", marginBottom: 12, fontFamily: P.mono }}>
            HIPERPARÁMETROS DQN
          </div>
          <table style={{ width: "100%", borderCollapse: "collapse", fontFamily: P.mono, fontSize: 10.5 }}>
            <tbody>
              {hyperparams.map(({ k, v }) => (
                <tr key={k} style={{ borderBottom: `1px solid rgba(125,211,252,0.04)` }}>
                  <td style={{ padding: "7px 10px", color: "rgba(148,163,184,0.65)" }}>{k}</td>
                  <td style={{ padding: "7px 10px", color: P.text }}>{v}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Environment parameters */}
        <div>
          <div style={{ fontSize: 10, color: P.drone, letterSpacing: "0.1em", marginBottom: 12, fontFamily: P.mono }}>
            PARÁMETROS DE ENTORNO
          </div>
          <table style={{ width: "100%", borderCollapse: "collapse", fontFamily: P.mono, fontSize: 10.5 }}>
            <tbody>
              {envParams.map(({ k, v }) => (
                <tr key={k} style={{ borderBottom: `1px solid rgba(125,211,252,0.04)` }}>
                  <td style={{ padding: "7px 10px", color: "rgba(148,163,184,0.65)" }}>{k}</td>
                  <td style={{ padding: "7px 10px", color: P.text }}>{v}</td>
                </tr>
              ))}
            </tbody>
          </table>

          {/* Formal model reference */}
          <div style={{ marginTop: 20, padding: "12px 14px", borderRadius: 6, background: `${P.drone}08`, border: `1px solid ${P.drone}20` }}>
            <div style={{ fontSize: 9, color: P.drone, letterSpacing: "0.1em", marginBottom: 8 }}>DEC-POMDP · FORMAL_MODELING.MD</div>
            <div style={{ fontSize: 10, color: P.muted, lineHeight: 1.6 }}>
              ℳ = ⟨𝕀, 𝒮, 𝒜, 𝒯, ℛ, Ω, 𝒪, γ⟩<br />
              𝒜 = 8 acciones · |𝒮ⁱ| = ℝ¹¹<br />
              π_NS(a|s) = argmax_(a:M[a]=1) Q_θ(s,a)
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// ─── TopBar ───────────────────────────────────────────────────────────────────

interface TopBarProps {
  isConnected: boolean;
  training: TrainingState;
  health: SystemHealth | null;
  uptime: string;
}

function TopBar({ isConnected, training, health, uptime }: TopBarProps) {
  const connColor = isConnected ? P.ok : P.crit;
  const trainDot  = training.isTraining ? P.ok : "rgba(148,163,184,0.4)";

  return (
    <header style={{
      gridColumn: "1 / -1",
      background: P.panel,
      borderBottom: `1px solid ${P.border}`,
      display: "flex", alignItems: "center",
      padding: "0 18px", gap: 18,
    }}>
      {/* Logo */}
      <div style={{ display: "flex", alignItems: "center", gap: 10, flexShrink: 0 }}>
        <svg width="22" height="22" viewBox="0 0 22 22">
          <polygon points="11,2 20,11 11,20 2,11" fill="none" stroke={P.drone} strokeWidth="1.4" />
          <circle cx="11" cy="11" r="3" fill={P.drone} />
        </svg>
        <span style={{ fontWeight: 600, letterSpacing: "0.05em", fontSize: 13 }}>
          SMART<span style={{ color: P.drone }}>·</span>SWARM
        </span>
        <span style={{ fontSize: 10, color: "rgba(148,163,184,0.5)", fontFamily: P.mono }}>
          NEURO·SYMBOLIC v3.2
        </span>
      </div>

      {/* Connection badge */}
      <div style={{
        display: "flex", alignItems: "center", gap: 6,
        fontSize: 10, fontFamily: P.mono, color: P.muted,
        background: `${connColor}15`, border: `1px solid ${connColor}44`,
        borderRadius: 4, padding: "3px 8px",
      }}>
        <span style={{ width: 6, height: 6, borderRadius: "50%", background: connColor, display: "inline-block" }} />
        {isConnected ? "WS CONECTADO" : "WS DESCONECTADO"}
      </div>

      {/* Symbolic engine status */}
      {health && (
        <div style={{
          display: "flex", alignItems: "center", gap: 6,
          fontSize: 10, fontFamily: P.mono, color: P.muted,
          background: health.symbolic_ok ? `${P.ok}15` : `${P.warn}15`,
          border: `1px solid ${health.symbolic_ok ? P.ok : P.warn}44`,
          borderRadius: 4, padding: "3px 8px",
        }}>
          <span style={{
            width: 6, height: 6, borderRadius: "50%",
            background: health.symbolic_ok ? P.ok : P.warn,
            display: "inline-block",
          }} />
          PROLOG {health.symbolic_ok ? "OK" : "OFFLINE"}
        </div>
      )}

      {/* ML predictor status */}
      {health && (
        <div style={{
          display: "flex", alignItems: "center", gap: 6,
          fontSize: 10, fontFamily: P.mono, color: P.muted,
          background: health.ml_ok ? `${P.drone}15` : `${P.warn}15`,
          border: `1px solid ${health.ml_ok ? P.drone : P.warn}44`,
          borderRadius: 4, padding: "3px 8px",
        }}>
          <span style={{
            width: 6, height: 6, borderRadius: "50%",
            background: health.ml_ok ? P.drone : P.warn,
            display: "inline-block",
          }} />
          ML {health.ml_ok ? "OK" : "OFFLINE"}
        </div>
      )}

      {/* Right cluster */}
      <div style={{
        marginLeft: "auto", display: "flex",
        gap: 18, fontSize: 11, fontFamily: P.mono, color: P.muted,
        alignItems: "center",
      }}>
        <span style={{ display: "flex", alignItems: "center", gap: 5 }}>
          <span style={{ width: 7, height: 7, borderRadius: "50%", background: trainDot, display: "inline-block",
            boxShadow: training.isTraining ? `0 0 6px ${P.ok}` : "none" }} />
          {training.isTraining ? (
            <span>EP <span style={{ color: P.text }}>{training.currentEpisode}</span>
              <span style={{ color: "rgba(148,163,184,0.4)" }}>/{training.maxEpisodes}</span>
            </span>
          ) : "IDLE"}
        </span>
        <span>SYS <span style={{ color: P.text }}>{training.system.toUpperCase()}</span></span>
        <span>UPTIME <span style={{ color: P.text }}>{uptime}</span></span>
        <span>OPERADOR <span style={{ color: P.text }}>m.castillo</span></span>
      </div>
    </header>
  );
}

// ─── Kpi card ─────────────────────────────────────────────────────────────────

interface KpiProps { label: string; value: string; accent?: string; sub?: string; }

function Kpi({ label, value, accent, sub }: KpiProps) {
  return (
    <div style={{
      padding: "10px 12px", borderRadius: 8,
      background: "rgba(255,255,255,0.015)",
      border: `1px solid ${P.border}`,
      display: "flex", flexDirection: "column", gap: 3,
    }}>
      <div style={{ fontSize: 9.5, letterSpacing: "0.1em", textTransform: "uppercase", color: "rgba(148,163,184,0.6)" }}>
        {label}
      </div>
      <div style={{
        fontFamily: P.mono, fontSize: 17,
        color: accent ?? P.text,
        fontWeight: 500, lineHeight: 1,
        fontVariantNumeric: "tabular-nums",
      }}>
        {value}
      </div>
      {sub && <div style={{ fontSize: 10, color: "rgba(148,163,184,0.45)", fontFamily: P.mono }}>{sub}</div>}
    </div>
  );
}

// ─── StatBlock ────────────────────────────────────────────────────────────────

interface StatBlockProps {
  training: TrainingState;
  dynamics: DynamicsState;
  droneState: DroneState;
  health: SystemHealth | null;
}

function StatBlock({ training, dynamics, droneState, health }: StatBlockProps) {
  const droneCount = droneState.positions.length;
  const avgBat = droneState.batteries.length > 0
    ? (droneState.batteries.reduce((a, b) => a + b, 0) / droneState.batteries.length).toFixed(0) + "%"
    : "—";

  return (
    <>
      <div style={{ fontSize: 9, letterSpacing: "0.14em", color: "rgba(148,163,184,0.4)", marginBottom: 4 }}>SISTEMA</div>
      <StatRow l="Política"    v={training.system.toUpperCase()} />
      <StatRow l="Step"        v={String(training.currentStep)} />
      <StatRow l="Drones act." v={String(droneCount)} />
      <StatRow l="Bat. media"  v={avgBat} />
      <StatRow l="Tormentas"   v={String(dynamics.storms)} c={dynamics.storms > 0 ? P.storm : undefined} />
      <StatRow l="Prolog"      v={health?.symbolic_ok ? "OK" : "OFFLINE"}
               c={health?.symbolic_ok ? P.ok : P.warn} />
      <StatRow l="Grid"        v={`${health?.grid_size ?? GRID_SIZE}²`} />
    </>
  );
}

function StatRow({ l, v, c }: { l: string; v: string; c?: string }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", fontFamily: P.mono, fontSize: 10.5 }}>
      <span style={{ color: "rgba(148,163,184,0.65)" }}>{l}</span>
      <span style={{ color: c ?? P.text }}>{v}</span>
    </div>
  );
}
