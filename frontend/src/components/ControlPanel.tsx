import React, { useState } from "react";
import { TrainingState, SystemHealth, TrainingStatus } from "../types";
import { P } from "../App";

interface ControlPanelProps {
  training: TrainingState;
  health: SystemHealth | null;
  status: TrainingStatus | null;
  onStart: (system: TrainingState["system"], episodes: number, mode: "resume" | "scratch", seed?: number) => void;
  onStop: () => void;
  onDeleteData: (system?: string) => void;
}

const SYSTEMS: { value: TrainingState["system"]; label: string; desc: string }[] = [
  { value: "neuro_dqn", label: "Neuro-DQN", desc: "DQN + Prolog masking" },
  { value: "dqn",       label: "DQN Puro",  desc: "Baseline neuronal"   },
  { value: "astar",     label: "A* Baseline", desc: "Planificador clásico" },
];

export function ControlPanel({ training, health, status, onStart, onStop, onDeleteData }: ControlPanelProps) {
  const [selectedSystem, setSelectedSystem] = useState<TrainingState["system"]>("neuro_dqn");
  const [episodes, setEpisodes] = useState(200);
  const [mode, setMode] = useState<"resume" | "scratch">("scratch");
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [reproducible, setReproducible] = useState(false);
  const [seed, setSeed] = useState(42);

  const canStart = !training.isTraining;
  const progress = training.maxEpisodes > 0
    ? (training.currentEpisode / training.maxEpisodes) * 100
    : 0;

  // Estado de conocimiento previo para el sistema seleccionado
  const sysStatus   = status?.systems?.[selectedSystem];
  const hasCkpts    = !!sysStatus?.has_checkpoints;
  const epsRecorded = sysStatus?.episodes_recorded ?? 0;
  const isAstar     = selectedSystem === "astar";
  // 'resume' solo tiene sentido si hay checkpoints y el sistema aprende
  const effectiveMode: "resume" | "scratch" =
    mode === "resume" && hasCkpts && !isAstar ? "resume" : "scratch";

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>

      {/* Section header */}
      <SectionLabel>Control de Entrenamiento</SectionLabel>

      {/* System selector */}
      <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
        {SYSTEMS.map((s) => {
          const active = selectedSystem === s.value;
          return (
            <button
              key={s.value}
              onClick={() => canStart && setSelectedSystem(s.value)}
              disabled={!canStart}
              style={{
                display: "flex", alignItems: "center", justifyContent: "space-between",
                padding: "8px 10px", borderRadius: 5,
                background: active ? `${P.drone}12` : "rgba(255,255,255,0.02)",
                border: `1px solid ${active ? P.drone + "55" : "rgba(125,211,252,0.1)"}`,
                color: active ? P.drone : "rgba(203,213,225,0.6)",
                fontSize: 11, fontFamily: P.mono,
                cursor: canStart ? "pointer" : "not-allowed",
                opacity: canStart ? 1 : 0.6,
                transition: "all 0.15s",
              }}
            >
              <span style={{ display: "flex", alignItems: "center", gap: 6 }}>
                <span style={{
                  width: 6, height: 6, borderRadius: "50%",
                  background: active ? P.drone : "rgba(148,163,184,0.3)",
                  boxShadow: active ? `0 0 6px ${P.drone}` : "none",
                  display: "inline-block", flexShrink: 0,
                }} />
                {s.label}
              </span>
              <span style={{ fontSize: 9, color: "rgba(148,163,184,0.4)" }}>{s.desc}</span>
            </button>
          );
        })}
      </div>

      {/* Episodes slider */}
      <div>
        <div style={{
          display: "flex", justifyContent: "space-between",
          fontSize: 10, color: P.muted, marginBottom: 5,
        }}>
          <span>Episodios</span>
          <span style={{ fontFamily: P.mono, color: P.drone }}>{episodes}</span>
        </div>
        <input
          type="range"
          min={10} max={500} step={10}
          value={episodes}
          disabled={!canStart}
          onChange={(e) => setEpisodes(Number(e.target.value))}
          style={{
            width: "100%", height: 4,
            accentColor: P.drone,
            opacity: canStart ? 1 : 0.5,
          }}
        />
        <div style={{
          display: "flex", justifyContent: "space-between",
          fontSize: 9, color: "rgba(148,163,184,0.35)",
          marginTop: 3, fontFamily: P.mono,
        }}>
          <span>10</span><span>250</span><span>500</span>
        </div>
      </div>

      {/* Modo de inicio: continuar (resume) vs desde cero (scratch) */}
      {!isAstar && (
        <div style={{ display: "flex", flexDirection: "column", gap: 5 }}>
          <div style={{ fontSize: 10, color: P.muted, display: "flex", justifyContent: "space-between" }}>
            <span>Modo de inicio</span>
            <span style={{ fontFamily: P.mono, color: hasCkpts ? P.ok : "rgba(148,163,184,0.4)" }}>
              {hasCkpts ? `${epsRecorded} ep guardados` : "sin datos previos"}
            </span>
          </div>
          <div style={{ display: "flex", gap: 6 }}>
            {([
              { v: "scratch", label: "Desde cero", desc: "pesos nuevos" },
              { v: "resume",  label: "Continuar",  desc: "conocimiento previo" },
            ] as const).map((opt) => {
              const active   = effectiveMode === opt.v;
              const disabled = !canStart || (opt.v === "resume" && !hasCkpts);
              return (
                <button
                  key={opt.v}
                  onClick={() => canStart && !disabled && setMode(opt.v)}
                  disabled={disabled}
                  title={opt.v === "resume" && !hasCkpts ? "No hay checkpoints guardados todavía" : opt.desc}
                  style={{
                    flex: 1, padding: "7px 8px", borderRadius: 5,
                    background: active ? `${P.drone}15` : "rgba(255,255,255,0.02)",
                    border: `1px solid ${active ? P.drone + "66" : "rgba(125,211,252,0.1)"}`,
                    color: active ? P.drone : "rgba(203,213,225,0.55)",
                    fontSize: 10.5, fontFamily: P.mono,
                    cursor: disabled ? "not-allowed" : "pointer",
                    opacity: disabled ? 0.45 : 1,
                    display: "flex", flexDirection: "column", gap: 2, alignItems: "center",
                    transition: "all 0.15s",
                  }}
                >
                  <span style={{ fontWeight: 600 }}>{opt.label}</span>
                  <span style={{ fontSize: 8.5, color: "rgba(148,163,184,0.5)" }}>{opt.desc}</span>
                </button>
              );
            })}
          </div>
        </div>
      )}

      {/* Reproducibilidad: semilla fija (comparación justa entre sistemas) */}
      <div style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 10, color: P.muted }}>
        <label style={{ display: "flex", alignItems: "center", gap: 6, cursor: canStart ? "pointer" : "not-allowed" }}>
          <input
            type="checkbox"
            checked={reproducible}
            disabled={!canStart}
            onChange={(e) => setReproducible(e.target.checked)}
            style={{ accentColor: P.drone }}
          />
          Semilla fija
        </label>
        {reproducible && (
          <input
            type="number"
            value={seed}
            disabled={!canStart}
            onChange={(e) => setSeed(Number(e.target.value))}
            style={{
              width: 70, padding: "3px 6px", borderRadius: 4,
              background: "rgba(255,255,255,0.03)", border: `1px solid ${P.border}`,
              color: P.text, fontFamily: P.mono, fontSize: 10.5,
            }}
          />
        )}
        <span style={{ fontSize: 9, color: "rgba(148,163,184,0.4)", marginLeft: "auto" }}>
          {reproducible ? "reproducible" : "estocástico"}
        </span>
      </div>

      {/* Progress bar (visible while training) */}
      {training.isTraining && (
        <div>
          <div style={{
            display: "flex", justifyContent: "space-between",
            fontSize: 10, color: "rgba(148,163,184,0.5)", marginBottom: 4,
          }}>
            <span>Progreso</span>
            <span style={{ fontFamily: P.mono }}>
              {training.currentEpisode}
              <span style={{ color: "rgba(148,163,184,0.35)" }}>/{training.maxEpisodes}</span>
            </span>
          </div>
          <div style={{ height: 3, background: "rgba(255,255,255,0.05)", borderRadius: 2, overflow: "hidden" }}>
            <div style={{
              height: "100%",
              width: `${Math.min(100, progress)}%`,
              background: `linear-gradient(90deg, ${P.ok}, ${P.drone})`,
              borderRadius: 2,
              transition: "width 0.4s ease",
            }} />
          </div>
          <div style={{
            marginTop: 4, fontSize: 9, fontFamily: P.mono,
            color: "rgba(148,163,184,0.35)", textAlign: "right",
          }}>
            step {training.currentStep}
          </div>
        </div>
      )}

      {/* Start / Stop buttons */}
      {canStart ? (
        <button
          onClick={() => onStart(selectedSystem, episodes, effectiveMode, reproducible ? seed : undefined)}
          style={btnStyle(P.ok)}
        >
          <span style={dotStyle(P.ok)} />
          {effectiveMode === "resume" ? "Continuar Entrenamiento" : "Iniciar Entrenamiento"}
        </button>
      ) : (
        <button onClick={onStop} style={btnStyle(P.crit)}>
          <span style={dotStyle(P.crit)} />
          Detener
        </button>
      )}

      {/* Eliminar datos de entrenamiento (checkpoints + métricas) */}
      {canStart && (status?.total_episodes_recorded ?? 0) > 0 && (
        confirmDelete ? (
          <div style={{ display: "flex", gap: 6 }}>
            <button
              onClick={() => { onDeleteData(); setConfirmDelete(false); }}
              style={{ ...btnStyle(P.crit), flex: 1, padding: "8px 10px", fontSize: 11 }}
            >
              Confirmar borrado
            </button>
            <button
              onClick={() => setConfirmDelete(false)}
              style={{
                flex: 1, padding: "8px 10px", borderRadius: 6,
                background: "rgba(255,255,255,0.03)", border: "1px solid rgba(125,211,252,0.12)",
                color: P.muted, fontSize: 11, cursor: "pointer",
              }}
            >
              Cancelar
            </button>
          </div>
        ) : (
          <button
            onClick={() => setConfirmDelete(true)}
            style={{
              width: "100%", padding: "7px 10px", borderRadius: 6,
              background: "transparent", border: `1px solid ${P.crit}33`,
              color: `${P.crit}cc`, fontSize: 10.5, fontFamily: P.mono,
              cursor: "pointer", display: "flex", alignItems: "center",
              justifyContent: "center", gap: 6,
            }}
          >
            ⌫ Eliminar datos de entrenamiento ({status?.total_episodes_recorded} ep)
          </button>
        )
      )}

      {/* Prolog status note */}
      {health && !health.symbolic_ok && selectedSystem === "neuro_dqn" && (
        <div style={{
          fontSize: 10, color: P.warn, fontFamily: P.mono,
          background: `${P.warn}10`, border: `1px solid ${P.warn}33`,
          borderRadius: 4, padding: "6px 8px", lineHeight: 1.5,
        }}>
          ⚠ Motor Prolog no disponible. Neuro-DQN correrá sin masking simbólico.
        </div>
      )}
    </div>
  );
}

// ─── Micro helpers ────────────────────────────────────────────────────────────

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <div style={{
      fontSize: 9.5, letterSpacing: "0.12em", textTransform: "uppercase",
      color: "rgba(148,163,184,0.5)", display: "flex", alignItems: "center", gap: 6,
    }}>
      <span style={{ width: 4, height: 4, background: P.drone, borderRadius: "50%", display: "inline-block" }} />
      {children}
    </div>
  );
}

function btnStyle(accent: string): React.CSSProperties {
  return {
    width: "100%", padding: "10px 14px", borderRadius: 6,
    background: `${accent}18`, border: `1px solid ${accent}66`,
    color: accent, fontSize: 12, fontWeight: 500,
    cursor: "pointer", display: "flex", alignItems: "center",
    justifyContent: "center", gap: 8,
    transition: "all 0.15s",
  };
}

function dotStyle(accent: string): React.CSSProperties {
  return {
    width: 8, height: 8, borderRadius: "50%",
    background: accent, boxShadow: `0 0 8px ${accent}`,
    display: "inline-block", flexShrink: 0,
  };
}
