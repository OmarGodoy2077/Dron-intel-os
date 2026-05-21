import React, { useMemo } from "react";

interface DroneMapProps {
  gridSize: number;
  dronePositions: number[][];   // [[x,y,z], ...]
  droneBatteries: number[];
  droneAlive?: boolean[];
  noFlyZones: [number, number][];
  stormRegions?: {
    x_range: [number, number];
    y_range: [number, number];
  }[];
  packagePositions?: number[][];
  chargingStations?: number[][];
  demandZones?: [number, number][];   // zonas de alta demanda (predicción ML)
  onDroneClick?: (idx: number) => void;
}

const CELL = 11;   // px per grid cell

function batteryColor(pct: number): string {
  if (pct > 50) return "#22c55e";
  if (pct > 25) return "#f59e0b";
  return "#ef4444";
}

export const DroneMap: React.FC<DroneMapProps> = ({
  gridSize,
  dronePositions,
  droneBatteries,
  droneAlive = [],
  noFlyZones,
  stormRegions = [],
  packagePositions = [],
  chargingStations = [],
  demandZones = [],
  onDroneClick,
}) => {
  const SIZE = gridSize * CELL;

  const nfzSet = useMemo(
    () => new Set(noFlyZones.map(([x, y]) => `${x},${y}`)),
    [noFlyZones]
  );

  const stormSet = useMemo(() => {
    const s = new Set<string>();
    stormRegions.forEach(({ x_range, y_range }) => {
      for (let x = x_range[0]; x <= x_range[1]; x++)
        for (let y = y_range[0]; y <= y_range[1]; y++)
          s.add(`${x},${y}`);
    });
    return s;
  }, [stormRegions]);

  return (
    <div
      style={{
        background: "#0a0f1e",
        borderRadius: 10,
        padding: 10,
        overflow: "auto",
        border: "1px solid #1e293b",
      }}
    >
      <svg
        width={SIZE}
        height={SIZE}
        style={{ display: "block", imageRendering: "pixelated" }}
      >
        {/* Background cells */}
        {Array.from({ length: gridSize }, (_, y) =>
          Array.from({ length: gridSize }, (_, x) => {
            const k = `${x},${y}`;
            const isNFZ   = nfzSet.has(k);
            const isStorm = stormSet.has(k);
            const fill = isNFZ ? "#5a0a0a" : isStorm ? "#0c2a4a" : "#111827";
            return (
              <rect
                key={k}
                x={x * CELL}
                y={(gridSize - 1 - y) * CELL}
                width={CELL - 1}
                height={CELL - 1}
                fill={fill}
              />
            );
          })
        )}

        {/* High-demand zones (predicción ML) — anillo cian translúcido */}
        {demandZones.map(([zx, zy], i) => (
          <rect
            key={`dz-${i}`}
            x={zx * CELL - CELL}
            y={(gridSize - 1 - zy) * CELL - CELL}
            width={CELL * 3}
            height={CELL * 3}
            fill="#06b6d4"
            opacity={0.1}
            stroke="#22d3ee"
            strokeWidth={1}
            strokeDasharray="3,2"
            rx={3}
          />
        ))}

        {/* Charging stations */}
        {chargingStations.map(([cx, cy], i) => (
          <rect
            key={`cs-${i}`}
            x={cx * CELL + 1}
            y={(gridSize - 1 - cy) * CELL + 1}
            width={CELL - 3}
            height={CELL - 3}
            fill="none"
            stroke="#facc15"
            strokeWidth={1.5}
            strokeDasharray="2,2"
          />
        ))}

        {/* Package markers */}
        {packagePositions.map(([px, py], i) => (
          <circle
            key={`pkg-${i}`}
            cx={px * CELL + CELL / 2}
            cy={(gridSize - 1 - py) * CELL + CELL / 2}
            r={2.5}
            fill="#a78bfa"
            opacity={0.9}
          />
        ))}

        {/* Drones */}
        {dronePositions.map((pos, i) => {
          if (!pos || pos.length < 2) return null;
          const alive = droneAlive.length === 0 || droneAlive[i];
          const [dx, dy] = pos;
          const bat = droneBatteries[i] ?? 100;
          const cx = dx * CELL + CELL / 2;
          const cy = (gridSize - 1 - dy) * CELL + CELL / 2;
          return (
            <g
              key={`d-${i}`}
              style={{ cursor: onDroneClick ? "pointer" : "default" }}
              onClick={() => onDroneClick?.(i)}
              opacity={alive ? 1 : 0.3}
            >
              <circle
                cx={cx} cy={cy} r={4.5}
                fill={batteryColor(bat)}
                stroke="#fff"
                strokeWidth={1}
              />
              <text
                x={cx} y={cy + 0.5}
                textAnchor="middle"
                dominantBaseline="middle"
                fontSize={6}
                fontWeight="bold"
                fill="#fff"
              >
                {i}
              </text>
            </g>
          );
        })}
      </svg>

      {/* Legend */}
      <div style={{
        display: "flex", gap: 14, marginTop: 8,
        fontSize: 10, color: "#64748b", flexWrap: "wrap",
      }}>
        {[
          ["#5a0a0a", "No-Fly Zone"],
          ["#0c2a4a", "Storm"],
          ["#22c55e", "Drone OK"],
          ["#ef4444", "Drone Critical"],
          ["#a78bfa", "Package"],
          ["#facc15", "Charge Station"],
          ["#22d3ee", "Demanda ML"],
        ].map(([color, label]) => (
          <span key={label}>
            <span style={{ color }}>{color.includes("0a0a") || color.includes("2a4a") ? "■" : color === "#22d3ee" ? "▢" : "●"} </span>
            {label}
          </span>
        ))}
      </div>
    </div>
  );
};
