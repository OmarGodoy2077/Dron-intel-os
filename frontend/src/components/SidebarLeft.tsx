import React from "react";
import { DroneState, DynamicsState } from "../types";
import { P } from "../App";

interface SidebarLeftProps {
  droneState: DroneState;
  dynamics: DynamicsState;
  activeTab: string;
  onTabChange: (tab: string) => void;
}

const NAV_ITEMS: [string, string, string][] = [
  ["◈", "Operaciones",   "operaciones"],
  ["⬡", "Entrenamiento", "entrenamiento"],
  ["⊞", "Reglas Prolog", "reglas"],
  ["▦", "Flota",         "flota"],
  ["⋮", "Histórico",     "historico"],
  ["⚙", "Configuración", "config"],
];

function batteryGradient(pct: number): string {
  if (pct > 50) return P.ok;
  if (pct > 25) return P.warn;
  return P.crit;
}

function stateLabel(battery: number, alive: boolean): string {
  if (!alive)     return "OFF";
  if (battery < 15) return "CRÍT";
  if (battery < 30) return "BAJO";
  return "ACT";
}

export function SidebarLeft({ droneState, dynamics, activeTab, onTabChange }: SidebarLeftProps) {
  const { positions, batteries, alive } = droneState;

  return (
    <aside style={{
      background: P.panel,
      borderRight: `1px solid ${P.border}`,
      padding: "14px 12px",
      display: "flex", flexDirection: "column", gap: 16,
      overflow: "hidden",
      borderBottom: `1px solid ${P.border}`,
    }}>

      {/* Navigation */}
      <nav>
        <div style={sectionLabel}>NAVEGACIÓN</div>
        {NAV_ITEMS.map(([icon, label, tab]) => {
          const active = activeTab === tab;
          return (
            <div
              key={tab}
              onClick={() => onTabChange(tab)}
              style={{
                display: "flex", alignItems: "center", gap: 8,
                padding: "6px 8px", borderRadius: 5,
                marginBottom: 2,
                background: active ? "rgba(125,211,252,0.07)" : "transparent",
                borderLeft: `2px solid ${active ? P.drone : "transparent"}`,
                color: active ? P.drone : "rgba(203,213,225,0.65)",
                fontSize: 12, cursor: "pointer",
                transition: "background 0.12s",
              }}
            >
              <span style={{ width: 14, textAlign: "center", opacity: 0.75, fontSize: 11 }}>{icon}</span>
              {label}
            </div>
          );
        })}
      </nav>

      {/* Fleet list */}
      <div style={{ flex: 1, overflow: "hidden", display: "flex", flexDirection: "column", gap: 0 }}>
        <div style={sectionLabel}>FLOTA · {positions.length} DRONES</div>
        <div style={{ overflowY: "auto", display: "flex", flexDirection: "column", gap: 5, paddingRight: 2 }}>
          {positions.length === 0 ? (
            <div style={{ fontSize: 10, color: "rgba(148,163,184,0.35)", fontFamily: P.mono, padding: "8px 0" }}>
              Esperando telemetría...
            </div>
          ) : (
            positions.map((pos, i) => {
              const bat    = batteries[i] ?? 0;
              const isAlive = alive.length === 0 || alive[i];
              const col    = batteryGradient(bat);
              const status = stateLabel(bat, isAlive);
              const [x, y, z] = pos.length >= 3 ? pos : [pos[0] ?? 0, pos[1] ?? 0, 0];

              return (
                <DroneCard
                  key={i}
                  id={i}
                  battery={bat}
                  alive={isAlive}
                  color={col}
                  status={status}
                  x={x} y={y} z={z}
                />
              );
            })
          )}
        </div>
      </div>

      {/* Dynamic events */}
      <div>
        <div style={sectionLabel}>DINÁMICA ACTIVA</div>
        <DynRow icon="⚡" label="Tormentas" value={dynamics.storms} accent={P.storm} />
        <DynRow icon="💨" label="Vientos"   value={dynamics.winds}  accent={P.warn}  />
        <DynRow icon="🚫" label="NFZ din."  value={dynamics.nfzs}   accent={P.crit}  />
      </div>

      {/* Footer */}
      <div style={{ fontSize: 9.5, color: "rgba(148,163,184,0.35)", fontFamily: P.mono, lineHeight: 1.6 }}>
        GRID 50×50 · {positions.length} AGENTES<br />
        DRL · PROLOG · MARL<br />
        región: cyber·city·west
      </div>
    </aside>
  );
}

// ─── Sub-components ───────────────────────────────────────────────────────────

interface DroneCardProps {
  id: number;
  battery: number;
  alive: boolean;
  color: string;
  status: string;
  x: number; y: number; z: number;
}

function DroneCard({ id, battery, alive, color, status, x, y }: DroneCardProps) {
  return (
    <div style={{
      display: "grid", gridTemplateColumns: "8px 1fr auto",
      gap: 8, alignItems: "center",
      padding: "7px 8px", borderRadius: 5,
      background: "rgba(125,211,252,0.025)",
      border: `1px solid rgba(125,211,252,0.07)`,
      opacity: alive ? 1 : 0.45,
    }}>
      {/* Status dot */}
      <span style={{
        width: 7, height: 7, borderRadius: "50%",
        background: color,
        boxShadow: alive ? `0 0 6px ${color}88` : "none",
        display: "inline-block",
        transition: "background 0.4s",
      }} />

      {/* Info */}
      <div style={{ display: "flex", flexDirection: "column", gap: 3, minWidth: 0 }}>
        <div style={{
          display: "flex", alignItems: "center", gap: 6,
          fontFamily: P.mono, fontSize: 10.5, color: P.text,
        }}>
          <span style={{ fontWeight: 600 }}>D-{String(id + 1).padStart(2, "0")}</span>
          <StatusBadge status={status} color={color} />
          <span style={{ color: "rgba(148,163,184,0.5)", fontSize: 9 }}>
            ({x.toFixed(0)},{y.toFixed(0)})
          </span>
        </div>
        {/* Battery bar */}
        <div style={{ height: 3, background: "rgba(255,255,255,0.05)", borderRadius: 2, overflow: "hidden" }}>
          <div style={{
            width: `${battery}%`, height: "100%",
            background: color,
            transition: "width 0.5s, background 0.5s",
          }} />
        </div>
      </div>

      {/* Battery % */}
      <div style={{ fontFamily: P.mono, fontSize: 10, color: color, textAlign: "right" }}>
        {battery.toFixed(0)}%
      </div>
    </div>
  );
}

function StatusBadge({ status, color }: { status: string; color: string }) {
  return (
    <span style={{
      fontSize: 8, padding: "1px 4px", borderRadius: 3,
      background: `${color}1a`, color,
      letterSpacing: "0.05em", fontWeight: 600,
    }}>
      {status}
    </span>
  );
}

function DynRow({ icon, label, value, accent }: { icon: string; label: string; value: number; accent: string }) {
  return (
    <div style={{
      display: "flex", alignItems: "center", justifyContent: "space-between",
      fontSize: 10.5, fontFamily: P.mono,
      padding: "3px 0",
      color: value > 0 ? accent : "rgba(148,163,184,0.35)",
    }}>
      <span style={{ display: "flex", alignItems: "center", gap: 5 }}>
        <span style={{ fontSize: 11 }}>{icon}</span>
        {label}
      </span>
      <span style={{ fontWeight: value > 0 ? 600 : 400 }}>{value}</span>
    </div>
  );
}

// ─── Shared style ─────────────────────────────────────────────────────────────

const sectionLabel: React.CSSProperties = {
  fontSize: 9, letterSpacing: "0.14em", textTransform: "uppercase",
  color: "rgba(148,163,184,0.4)", marginBottom: 8,
};
