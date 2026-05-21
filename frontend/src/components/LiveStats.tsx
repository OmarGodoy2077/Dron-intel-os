import React from "react";
import {
  AreaChart,
  Area,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";

interface EpisodePoint {
  episode: number;
  reward?: number;
  successRate?: number;
  ruleViolations?: number;
  symbolicInterventions?: number;
  avgBattery?: number;
}

interface LiveStatsProps {
  episodeData: EpisodePoint[];
  systemName: string;
  currentEpisode: number;
  isTraining: boolean;
}

const CHART_STYLE = {
  background: "#030712",
  border: "none",
  borderRadius: 6,
  fontSize: 11,
  color: "#94a3b8",
};

export const LiveStats: React.FC<LiveStatsProps> = ({
  episodeData,
  systemName,
  currentEpisode,
  isTraining,
}) => {
  const data = episodeData.map((d) => ({
    ep:      d.episode,
    Reward:  d.reward ? +d.reward.toFixed(1) : 0,
    "Rate%": d.successRate ? +(d.successRate * 100).toFixed(1) : 0,
    Violations: d.ruleViolations ?? 0,
    "Symbolic Ops": d.symbolicInterventions ?? 0,
    Battery: d.avgBattery ? +d.avgBattery.toFixed(1) : 0,
  }));

  const statusDot = isTraining ? "#22c55e" : "#ef4444";

  return (
    <div style={{
      background: "#0f172a",
      borderRadius: 10,
      padding: 16,
      color: "#e2e8f0",
      border: "1px solid #1e293b",
    }}>
      {/* Header */}
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 16, alignItems: "center" }}>
        <h3 style={{ margin: 0, fontSize: 13, color: "#7dd3fc", letterSpacing: 0.5 }}>
          Live Training — <span style={{ color: "#a78bfa" }}>{systemName}</span>
        </h3>
        <div style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 11, color: "#94a3b8" }}>
          <span style={{
            width: 8, height: 8, borderRadius: "50%",
            background: statusDot, display: "inline-block",
          }} />
          {isTraining ? "Training" : "Idle"} · Ep {currentEpisode}
        </div>
      </div>

      {/* Reward curve */}
      <section style={{ marginBottom: 20 }}>
        <p style={{ margin: "0 0 6px", fontSize: 11, color: "#64748b" }}>Cumulative Reward</p>
        <ResponsiveContainer width="100%" height={140}>
          <AreaChart data={data} margin={{ left: -10, right: 8 }}>
            <defs>
              <linearGradient id="rGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%"  stopColor="#38bdf8" stopOpacity={0.35} />
                <stop offset="95%" stopColor="#38bdf8" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
            <XAxis dataKey="ep" stroke="#334155" tick={{ fontSize: 9 }} />
            <YAxis stroke="#334155" tick={{ fontSize: 9 }} />
            <Tooltip contentStyle={CHART_STYLE} />
            <Area
              type="monotone" dataKey="Reward"
              stroke="#38bdf8" fill="url(#rGrad)" strokeWidth={2} dot={false}
            />
          </AreaChart>
        </ResponsiveContainer>
      </section>

      {/* Success rate */}
      <section style={{ marginBottom: 20 }}>
        <p style={{ margin: "0 0 6px", fontSize: 11, color: "#64748b" }}>Delivery Success Rate (%)</p>
        <ResponsiveContainer width="100%" height={110}>
          <LineChart data={data} margin={{ left: -10, right: 8 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
            <XAxis dataKey="ep" stroke="#334155" tick={{ fontSize: 9 }} />
            <YAxis stroke="#334155" tick={{ fontSize: 9 }} domain={[0, 100]} />
            <Tooltip contentStyle={CHART_STYLE} />
            <ReferenceLine y={90} stroke="#22c55e" strokeDasharray="4 4" opacity={0.5} />
            <Line
              type="monotone" dataKey="Rate%"
              stroke="#22c55e" strokeWidth={2} dot={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </section>

      {/* Symbolic engine activity */}
      <section style={{ marginBottom: 20 }}>
        <p style={{ margin: "0 0 6px", fontSize: 11, color: "#64748b" }}>Symbolic Engine Activity</p>
        <ResponsiveContainer width="100%" height={110}>
          <LineChart data={data} margin={{ left: -10, right: 8 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
            <XAxis dataKey="ep" stroke="#334155" tick={{ fontSize: 9 }} />
            <YAxis stroke="#334155" tick={{ fontSize: 9 }} />
            <Tooltip contentStyle={CHART_STYLE} />
            <Legend wrapperStyle={{ fontSize: 10, color: "#94a3b8" }} />
            <Line
              type="monotone" dataKey="Violations"
              stroke="#ef4444" strokeWidth={1.5} dot={false}
            />
            <Line
              type="monotone" dataKey="Symbolic Ops"
              stroke="#a78bfa" strokeWidth={1.5} dot={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </section>

      {/* Battery trend */}
      <section>
        <p style={{ margin: "0 0 6px", fontSize: 11, color: "#64748b" }}>Avg Battery Remaining (%)</p>
        <ResponsiveContainer width="100%" height={100}>
          <AreaChart data={data} margin={{ left: -10, right: 8 }}>
            <defs>
              <linearGradient id="bGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%"  stopColor="#f59e0b" stopOpacity={0.35} />
                <stop offset="95%" stopColor="#f59e0b" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
            <XAxis dataKey="ep" stroke="#334155" tick={{ fontSize: 9 }} />
            <YAxis stroke="#334155" tick={{ fontSize: 9 }} domain={[0, 100]} />
            <Tooltip contentStyle={CHART_STYLE} />
            <Area
              type="monotone" dataKey="Battery"
              stroke="#f59e0b" fill="url(#bGrad)" strokeWidth={2} dot={false}
            />
          </AreaChart>
        </ResponsiveContainer>
      </section>
    </div>
  );
};
