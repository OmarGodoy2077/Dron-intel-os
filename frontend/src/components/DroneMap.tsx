import React, { useEffect, useRef, useState } from "react";

interface DeliveryTarget {
  x: number;
  y: number;
  type: "medical" | "standard";
}

interface DroneCargo {
  pkg: number;
  type: "medical" | "standard";
  dest: [number, number];
}

interface DeliveryBurst {
  id: number;
  x: number;
  y: number;
  type: "medical" | "standard";
}

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
  deliveryTargets?: DeliveryTarget[];  // destinos de entrega pendientes
  cargos?: (DroneCargo | null)[];      // carga actual por dron (para la línea guía)
  deliveryBursts?: DeliveryBurst[];    // estallidos de entrega en curso (animados)
  onDroneClick?: (idx: number) => void;
}

const MEDICAL = "#f472b6";   // color para entregas/destinos médicos
const STANDARD = "#34d399";  // color para entregas/destinos estándar

const CELL = 13;            // px por celda del grid (un poco mayor para legibilidad)
const TRAIL_LEN = 8;        // nº de posiciones recientes que conserva la estela
const DRONE_COLORS = [      // paleta distinguible por dron (hasta 8)
  "#22d3ee", "#a78bfa", "#34d399", "#f472b6", "#fbbf24", "#60a5fa", "#fb7185", "#4ade80",
];

function batteryColor(pct: number): string {
  if (pct > 50) return "#22c55e";
  if (pct > 25) return "#f59e0b";
  return "#ef4444";
}

function droneColor(i: number): string {
  return DRONE_COLORS[i % DRONE_COLORS.length];
}

interface Trail {
  pts: { x: number; y: number }[];   // en coordenadas de grid (y arriba)
}

/**
 * DroneMap — visualización del grid de simulación.
 *
 * Mejoras visuales (presentacionales, sin tocar la lógica de datos):
 *  - Fondo con rejilla sutil + viñeta radial para dar profundidad.
 *  - NFZ con patrón de rayas y tormentas con celdas pulsantes.
 *  - Zonas de demanda ML como blobs difusos (heatmap suave).
 *  - Drones con color propio, anillo de batería, halo pulsante, indicador de
 *    rumbo (hacia dónde se mueve) y altitud; estela de las últimas posiciones.
 *  - Transición suave entre celdas (CSS) y tooltip al pasar el ratón.
 */
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
  deliveryTargets = [],
  cargos = [],
  deliveryBursts = [],
  onDroneClick,
}) => {
  const SIZE = gridSize * CELL;
  const [hover, setHover] = useState<number | null>(null);

  // Conversión grid→pixel (eje Y invertido: el norte queda arriba)
  const px = (gx: number) => gx * CELL + CELL / 2;
  const py = (gy: number) => (gridSize - 1 - gy) * CELL + CELL / 2;

  // ── Estelas y rumbo: guardamos posiciones recientes por dron en un ref ───────
  const trailsRef = useRef<Map<number, Trail>>(new Map());
  const [, force] = useState(0);
  useEffect(() => {
    const trails = trailsRef.current;
    dronePositions.forEach((pos, i) => {
      if (!pos || pos.length < 2) return;
      const cur = { x: pos[0], y: pos[1] };
      const t = trails.get(i) ?? { pts: [] };
      const last = t.pts[t.pts.length - 1];
      if (!last) {
        t.pts = [cur];
      } else if (last.x !== cur.x || last.y !== cur.y) {
        // Salto grande (reset de episodio / teletransporte) → reiniciar la estela
        // para no dibujar una línea atravesando todo el mapa.
        const jump = Math.abs(last.x - cur.x) + Math.abs(last.y - cur.y);
        t.pts = jump > 3 ? [cur] : [...t.pts, cur].slice(-TRAIL_LEN);
      }
      trails.set(i, t);
    });
    force((n) => n + 1);   // re-render para reflejar la estela
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dronePositions]);

  return (
    <div
      style={{
        background: "radial-gradient(circle at 50% 35%, #0d1730 0%, #060912 80%)",
        borderRadius: 12,
        padding: 12,
        overflow: "auto",
        border: "1px solid #1e293b",
        boxShadow: "inset 0 0 60px rgba(0,0,0,0.5)",
      }}
    >
      <svg
        width={SIZE}
        height={SIZE}
        style={{ display: "block", borderRadius: 6, overflow: "visible" }}
      >
        <defs>
          {/* Patrón de rayas para NFZ */}
          <pattern id="nfzHatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
            <rect width="6" height="6" fill="#3b0a0a" />
            <line x1="0" y1="0" x2="0" y2="6" stroke="#ef4444" strokeWidth="1.4" opacity="0.55" />
          </pattern>
          {/* Glow para tormentas */}
          <radialGradient id="stormGlow" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stopColor="#3b82f6" stopOpacity="0.45" />
            <stop offset="100%" stopColor="#1e3a8a" stopOpacity="0.05" />
          </radialGradient>
          {/* Glow para zonas de demanda ML */}
          <radialGradient id="demandGlow" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stopColor="#22d3ee" stopOpacity="0.5" />
            <stop offset="60%" stopColor="#06b6d4" stopOpacity="0.18" />
            <stop offset="100%" stopColor="#06b6d4" stopOpacity="0" />
          </radialGradient>
          <filter id="softGlow" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="2.2" result="b" />
            <feMerge>
              <feMergeNode in="b" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>

        {/* ── Rejilla de fondo (líneas sutiles cada celda) ─────────────────── */}
        <rect x={0} y={0} width={SIZE} height={SIZE} fill="#0b1322" rx={6} />
        <g opacity={0.5}>
          {Array.from({ length: gridSize + 1 }, (_, i) => (
            <React.Fragment key={`gl-${i}`}>
              <line x1={i * CELL} y1={0} x2={i * CELL} y2={SIZE} stroke="#16203a" strokeWidth={0.5} />
              <line x1={0} y1={i * CELL} x2={SIZE} y2={i * CELL} stroke="#16203a" strokeWidth={0.5} />
            </React.Fragment>
          ))}
        </g>

        {/* ── Zonas de demanda ML (heatmap difuso pulsante) ────────────────── */}
        {demandZones.map(([zx, zy], i) => (
          <circle
            key={`dz-${i}`}
            cx={px(zx)} cy={py(zy)} r={CELL * 2.4}
            fill="url(#demandGlow)"
          >
            <animate attributeName="opacity" values="0.55;0.9;0.55" dur="3s" repeatCount="indefinite" begin={`${i * 0.4}s`} />
          </circle>
        ))}

        {/* ── Tormentas (regiones rectangulares con glow pulsante) ─────────── */}
        {stormRegions.map(({ x_range, y_range }, i) => {
          const x0 = Math.min(x_range[0], x_range[1]);
          const x1 = Math.max(x_range[0], x_range[1]);
          const y0 = Math.min(y_range[0], y_range[1]);
          const y1 = Math.max(y_range[0], y_range[1]);
          const left = x0 * CELL;
          const top = (gridSize - 1 - y1) * CELL;
          const w = (x1 - x0 + 1) * CELL;
          const h = (y1 - y0 + 1) * CELL;
          return (
            <g key={`st-${i}`}>
              <rect x={left} y={top} width={w} height={h} fill="url(#stormGlow)" rx={4} />
              <rect x={left} y={top} width={w} height={h} fill="none" stroke="#60a5fa" strokeWidth={1} strokeDasharray="4,3" rx={4} opacity={0.7}>
                <animate attributeName="opacity" values="0.3;0.8;0.3" dur="2s" repeatCount="indefinite" />
              </rect>
              <text x={left + 4} y={top + 12} fontSize={9} fill="#93c5fd" fontFamily="monospace">⛈ tormenta</text>
            </g>
          );
        })}

        {/* ── No-Fly Zones (celdas con patrón de rayas) ────────────────────── */}
        {noFlyZones.map(([x, y], i) => (
          <rect
            key={`nfz-${i}`}
            x={x * CELL} y={(gridSize - 1 - y) * CELL}
            width={CELL} height={CELL}
            fill="url(#nfzHatch)"
          />
        ))}

        {/* ── Estaciones de carga (ícono de rayo) ──────────────────────────── */}
        {chargingStations.map(([cx, cy], i) => (
          <g key={`cs-${i}`}>
            <rect
              x={cx * CELL + 1.5} y={(gridSize - 1 - cy) * CELL + 1.5}
              width={CELL - 3} height={CELL - 3}
              fill="#facc1522" stroke="#facc15" strokeWidth={1.3} rx={2}
            />
            <text x={px(cx)} y={py(cy) + 0.5} textAnchor="middle" dominantBaseline="middle" fontSize={8} fill="#facc15">⚡</text>
          </g>
        ))}

        {/* ── Destinos de entrega pendientes (dianas) ──────────────────────── */}
        {deliveryTargets.map((d, i) => {
          const col = d.type === "medical" ? MEDICAL : STANDARD;
          const cx = px(d.x), cy = py(d.y);
          return (
            <g key={`dt-${i}`} opacity={0.85}>
              {/* Diana: dos anillos concéntricos + cruz */}
              <circle cx={cx} cy={cy} r={CELL * 0.5} fill="none" stroke={col} strokeWidth={1.2} opacity={0.5} strokeDasharray="2,2" />
              <circle cx={cx} cy={cy} r={CELL * 0.28} fill="none" stroke={col} strokeWidth={1.3} />
              <circle cx={cx} cy={cy} r={1.6} fill={col} />
              {d.type === "medical" && (
                <text x={cx} y={cy - CELL * 0.62} textAnchor="middle" fontSize={7} fill={MEDICAL}>✚</text>
              )}
            </g>
          );
        })}

        {/* ── Líneas guía: dron con carga → su destino de entrega ──────────── */}
        {cargos.map((c, i) => {
          if (!c) return null;
          const pos = dronePositions[i];
          if (!pos || pos.length < 2) return null;
          const alive = droneAlive.length === 0 || droneAlive[i];
          if (!alive) return null;
          const col = c.type === "medical" ? MEDICAL : STANDARD;
          return (
            <line
              key={`cl-${i}`}
              x1={px(pos[0])} y1={py(pos[1])}
              x2={px(c.dest[0])} y2={py(c.dest[1])}
              stroke={col} strokeWidth={1.1} strokeDasharray="3,4" opacity={0.4}
            >
              <animate attributeName="stroke-dashoffset" values="0;-7" dur="0.8s" repeatCount="indefinite" />
            </line>
          );
        })}

        {/* ── Paquetes (cajas) ─────────────────────────────────────────────── */}
        {packagePositions.map(([pkx, pky], i) => (
          <text key={`pkg-${i}`} x={px(pkx)} y={py(pky) + 0.5} textAnchor="middle" dominantBaseline="middle" fontSize={9}>📦</text>
        ))}

        {/* ── Estelas de movimiento de cada dron ───────────────────────────── */}
        {dronePositions.map((_pos, i) => {
          const t = trailsRef.current.get(i);
          if (!t || t.pts.length < 2) return null;
          const alive = droneAlive.length === 0 || droneAlive[i];
          if (!alive) return null;
          const col = droneColor(i);
          return (
            <polyline
              key={`tr-${i}`}
              points={t.pts.map((p) => `${px(p.x)},${py(p.y)}`).join(" ")}
              fill="none"
              stroke={col}
              strokeWidth={2}
              strokeLinejoin="round"
              strokeLinecap="round"
              opacity={0.25}
            />
          );
        })}

        {/* ── Drones ───────────────────────────────────────────────────────── */}
        {dronePositions.map((pos, i) => {
          if (!pos || pos.length < 2) return null;
          const alive = droneAlive.length === 0 || droneAlive[i];
          const [dx, dy, dz = 0] = pos;
          const bat = droneBatteries[i] ?? 100;
          const cx = px(dx);
          const cy = py(dy);
          const col = droneColor(i);
          const batCol = batteryColor(bat);
          const isHover = hover === i;

          // Rumbo: vector desde la penúltima a la última posición de la estela
          const trail = trailsRef.current.get(i);
          let heading: { x: number; y: number } | null = null;
          if (trail && trail.pts.length >= 2) {
            const a = trail.pts[trail.pts.length - 2];
            const b = trail.pts[trail.pts.length - 1];
            const vx = b.x - a.x, vy = b.y - a.y;
            if (vx !== 0 || vy !== 0) {
              const n = Math.hypot(vx, vy);
              heading = { x: vx / n, y: -vy / n }; // -vy porque el SVG invierte Y
            }
          }

          // Circunferencia para el anillo de batería (arco proporcional)
          const R = 7;
          const circ = 2 * Math.PI * R;
          const batDash = `${(bat / 100) * circ} ${circ}`;

          return (
            <g
              key={`d-${i}`}
              style={{
                cursor: onDroneClick ? "pointer" : "default",
                transform: `translate(${cx}px, ${cy}px)`,
                transition: "transform 0.45s cubic-bezier(0.4,0,0.2,1)",
              }}
              onClick={() => onDroneClick?.(i)}
              onMouseEnter={() => setHover(i)}
              onMouseLeave={() => setHover(null)}
              opacity={alive ? 1 : 0.28}
            >
              {/* Halo pulsante (solo vivo) */}
              {alive && (
                <circle cx={0} cy={0} r={R + 3} fill={col} opacity={0.18} filter="url(#softGlow)">
                  <animate attributeName="r" values={`${R + 2};${R + 6};${R + 2}`} dur="2.4s" repeatCount="indefinite" begin={`${i * 0.3}s`} />
                  <animate attributeName="opacity" values="0.22;0.06;0.22" dur="2.4s" repeatCount="indefinite" begin={`${i * 0.3}s`} />
                </circle>
              )}

              {/* Anillo de batería */}
              <circle cx={0} cy={0} r={R} fill="none" stroke="#1e293b" strokeWidth={2.4} />
              <circle
                cx={0} cy={0} r={R} fill="none"
                stroke={batCol} strokeWidth={2.4}
                strokeDasharray={batDash} strokeLinecap="round"
                transform="rotate(-90)"
                style={{ transition: "stroke-dasharray 0.45s ease, stroke 0.3s" }}
              />

              {/* Cuerpo del dron */}
              <circle cx={0} cy={0} r={R - 2.5} fill={col} stroke="#0b1322" strokeWidth={1} />
              <text x={0} y={0.5} textAnchor="middle" dominantBaseline="middle" fontSize={7} fontWeight="bold" fill="#0b1322">
                {i}
              </text>

              {/* Indicador de rumbo (flecha en la dirección de movimiento) */}
              {alive && heading && (
                <polygon
                  points="0,-3 2.4,2.4 -2.4,2.4"
                  fill="#fff"
                  opacity={0.95}
                  transform={`translate(${heading.x * (R + 3)}, ${heading.y * (R + 3)}) rotate(${(Math.atan2(heading.y, heading.x) * 180) / Math.PI + 90})`}
                />
              )}

              {/* Marca de altitud (puntos = nivel z) */}
              {alive && dz > 0 && (
                <text x={0} y={-R - 4} textAnchor="middle" fontSize={6} fill="#94a3b8" fontFamily="monospace">
                  ▲{dz}
                </text>
              )}

              {/* Tooltip al pasar el ratón */}
              {isHover && (
                <g transform={`translate(${R + 6}, ${-R - 6})`}>
                  <rect x={0} y={0} width={104} height={46} rx={5} fill="#0b1322" stroke={col} strokeWidth={1} opacity={0.96} />
                  <text x={8} y={15} fontSize={9} fill={col} fontFamily="monospace" fontWeight="bold">DRON {i}</text>
                  <text x={8} y={28} fontSize={8.5} fill="#cbd5e1" fontFamily="monospace">Pos {dx},{dy} · Alt {dz}</text>
                  <text x={8} y={40} fontSize={8.5} fill={batCol} fontFamily="monospace">
                    Batería {bat.toFixed(0)}% {alive ? "" : "· OFFLINE"}
                  </text>
                </g>
              )}
            </g>
          );
        })}

        {/* ── Estallidos de entrega (animación al completar una entrega) ───── */}
        {deliveryBursts.map((b) => {
          const cx = px(b.x), cy = py(b.y);
          const col = b.type === "medical" ? MEDICAL : STANDARD;
          return (
            <g key={`burst-${b.id}`} style={{ pointerEvents: "none" }}>
              {/* Onda expansiva */}
              <circle cx={cx} cy={cy} r={3} fill="none" stroke={col} strokeWidth={2.5}>
                <animate attributeName="r" from="3" to={`${CELL * 2.2}`} dur="1.1s" fill="freeze" />
                <animate attributeName="opacity" from="0.95" to="0" dur="1.1s" fill="freeze" />
                <animate attributeName="stroke-width" from="2.5" to="0.4" dur="1.1s" fill="freeze" />
              </circle>
              {/* Segunda onda (desfasada) */}
              <circle cx={cx} cy={cy} r={3} fill="none" stroke={col} strokeWidth={1.5}>
                <animate attributeName="r" from="3" to={`${CELL * 1.5}`} dur="1.1s" begin="0.15s" fill="freeze" />
                <animate attributeName="opacity" from="0.7" to="0" dur="1.1s" begin="0.15s" fill="freeze" />
              </circle>
              {/* Destello central */}
              <circle cx={cx} cy={cy} r={CELL * 0.45} fill={col} filter="url(#softGlow)">
                <animate attributeName="opacity" from="0.9" to="0" dur="0.6s" fill="freeze" />
                <animate attributeName="r" from={`${CELL * 0.45}`} to={`${CELL * 0.15}`} dur="0.6s" fill="freeze" />
              </circle>
              {/* Etiqueta "+1 entrega" que asciende y se desvanece */}
              <text x={cx} y={cy} textAnchor="middle" fontSize={9} fontWeight="bold" fontFamily="monospace" fill={col}>
                {b.type === "medical" ? "✚ +1" : "+1"}
                <animate attributeName="y" from={`${cy - 4}`} to={`${cy - CELL * 1.8}`} dur="1.1s" fill="freeze" />
                <animate attributeName="opacity" from="1" to="0" dur="1.1s" fill="freeze" />
              </text>
            </g>
          );
        })}
      </svg>

      {/* Leyenda */}
      <div style={{
        display: "flex", gap: 16, marginTop: 12,
        fontSize: 10.5, color: "#94a3b8", flexWrap: "wrap", alignItems: "center",
      }}>
        <LegendItem icon="◉" color="#22d3ee" label="Dron (color propio)" />
        <LegendItem icon="◌" color="#22c55e" label="Anillo = batería" />
        <LegendItem icon="▦" color="#ef4444" label="No-Fly Zone" />
        <LegendItem icon="⛈" color="#60a5fa" label="Tormenta" />
        <LegendItem icon="◍" color="#22d3ee" label="Demanda ML" />
        <LegendItem icon="⚡" color="#facc15" label="Estación carga" />
        <LegendItem icon="◎" color={STANDARD} label="Destino entrega" />
        <LegendItem icon="✚" color={MEDICAL} label="Destino médico" />
      </div>
    </div>
  );
};

function LegendItem({ icon, color, label }: { icon: string; color: string; label: string }) {
  return (
    <span style={{ display: "inline-flex", alignItems: "center", gap: 5 }}>
      <span style={{ color, fontSize: 12 }}>{icon}</span>
      <span>{label}</span>
    </span>
  );
}
