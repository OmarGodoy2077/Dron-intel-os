import React, { useEffect, useRef } from "react";

export interface RuleLogEntry {
  timestamp: string;
  level: "INFO" | "MASK" | "REWARD" | "ALERT" | "CRITICAL" | "NEGOTIATION" | "PRIO" | "RUTA" | "CLIMA";
  agent: string;
  rule?: string;
  message: string;
}

interface RuleTerminalProps {
  logs: RuleLogEntry[];
  maxLines?: number;
  title?: string;
}

const COLORS: Record<string, string> = {
  INFO:        "#64748b",
  MASK:        "#f59e0b",
  REWARD:      "#22c55e",
  ALERT:       "#fb923c",
  CRITICAL:    "#ef4444",
  NEGOTIATION: "#a78bfa",
  PRIO:        "#f472b6",
  RUTA:        "#34d399",
  CLIMA:       "#60a5fa",
};

const BADGES: Record<string, string> = {
  INFO:        "INF",
  MASK:        "MSK",
  REWARD:      "RWD",
  ALERT:       "ALT",
  CRITICAL:    "CRT",
  NEGOTIATION: "NEG",
  PRIO:        "PRI",
  RUTA:        "RTA",
  CLIMA:       "CLM",
};

export const RuleTerminal: React.FC<RuleTerminalProps> = ({
  logs,
  maxLines = 250,
  title = "SYMBOLIC ENGINE — PROLOG DECISION LOG",
}) => {
  const bottomRef = useRef<HTMLDivElement>(null);
  const display = logs.slice(-maxLines);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs.length]);

  return (
    <div
      style={{
        background: "#020617",
        borderRadius: 10,
        border: "1px solid #1e293b",
        padding: "10px 14px",
        fontFamily: "'JetBrains Mono', 'Fira Code', 'Consolas', monospace",
        fontSize: 11,
        height: 380,
        overflowY: "auto",
      }}
    >
      {/* Header bar */}
      <div style={{
        color: "#334155",
        marginBottom: 10,
        fontSize: 10,
        letterSpacing: 1.2,
        borderBottom: "1px solid #1e293b",
        paddingBottom: 6,
        display: "flex",
        justifyContent: "space-between",
      }}>
        <span>{title}</span>
        <span style={{ color: "#1e3a5f" }}>{display.length} entries</span>
      </div>

      {display.length === 0 && (
        <div style={{ color: "#1e3a5f", fontStyle: "italic", marginTop: 12 }}>
          Awaiting symbolic engine output...
        </div>
      )}

      {display.map((entry, i) => {
        const col = COLORS[entry.level] ?? "#64748b";
        const badge = BADGES[entry.level] ?? entry.level;
        return (
          <div
            key={i}
            style={{ marginBottom: 3, lineHeight: 1.5, display: "flex", gap: 8 }}
          >
            {/* Timestamp */}
            <span style={{ color: "#334155", flexShrink: 0, width: 68 }}>
              {entry.timestamp}
            </span>

            {/* Level badge */}
            <span
              style={{
                color: col,
                background: col + "18",
                borderRadius: 3,
                padding: "0 4px",
                fontSize: 9,
                fontWeight: "bold",
                flexShrink: 0,
                display: "flex",
                alignItems: "center",
                letterSpacing: 0.5,
              }}
            >
              {badge}
            </span>

            {/* Rule tag */}
            {entry.rule && (
              <span style={{ color: "#1e3a5f", flexShrink: 0 }}>
                [{entry.rule}]
              </span>
            )}

            {/* Agent */}
            <span style={{ color: "#7dd3fc", flexShrink: 0 }}>
              {entry.agent}
            </span>

            {/* Message */}
            <span style={{ color: "#cbd5e1" }}>{entry.message}</span>
          </div>
        );
      })}

      <div ref={bottomRef} />
    </div>
  );
};
