# Frontend Status — Smart-Swarm Neuro-Simbólico

> Último update: 2026-05-08
> Estado global: ✅ **Implementación completa y funcional** — lista para ejecutar con `npm run dev`

---

## 1. Resumen de Implementación

El frontend es una **Single Page Application** React 18 + TypeScript que replica visualmente el prototipo `variant-a.jsx` (estética "Cyberpunk / Centro de Control") y conecta en tiempo real al backend FastAPI mediante **WebSocket** y **REST**.

### Stack técnico usado

| Tecnología | Versión | Rol |
|---|---|---|
| React | 18.3.1 | UI framework |
| TypeScript | 5.5.3 | Tipado estático |
| Vite | 5.3.4 | Build tool + dev server |
| Recharts | 2.12.7 | Gráficas de aprendizaje |
| Google Fonts | Inter + JetBrains Mono | Tipografía |

> **Sin Tailwind**: el diseño usa **inline styles** idiomáticos de React para máxima compatibilidad con los componentes pre-existentes (`DroneMap`, `LiveStats`, `RuleTerminal`).

---

## 2. Estructura de Archivos

```
frontend/
├── index.html                    # Entry HTML (Google Fonts, reset CSS)
├── vite.config.ts                # Vite + proxy hacia http://localhost:8000
├── tsconfig.json                 # TypeScript strict mode
├── tsconfig.node.json            # TS para archivos de configuración
├── package.json                  # Dependencias (React, Recharts, Vite)
└── src/
    ├── main.tsx                  # ReactDOM.createRoot — punto de entrada
    ├── App.tsx                   # Layout principal + orquestación de datos
    ├── types/
    │   └── index.ts              # Todas las interfaces TypeScript
    ├── hooks/
    │   └── useSocket.ts          # Hook WebSocket con reconexión automática
    └── components/
        ├── DroneMap.tsx          # Mapa SVG interactivo (preexistente)
        ├── LiveStats.tsx         # 4 gráficas Recharts (preexistente)
        ├── RuleTerminal.tsx      # Terminal Prolog con badges (preexistente)
        ├── ControlPanel.tsx      # Panel Start/Stop + selector de sistema
        └── SidebarLeft.tsx       # Barra lateral con navegación, flota y eventos
```

---

## 3. Layout — CSS Grid

El layout replica `variant-a.jsx` usando un **grid 3×3** de pantalla completa:

```
┌────────────────────────────────────────────────────────────────┐
│  TopBar (52px) — logo · WS status · Prolog · episode · uptime │
│                          [col 1-3]                             │
├─────────────────┬──────────────────────────┬───────────────────┤
│  SidebarLeft    │   Área Central (variable) │   Right Rail      │
│  (220px)        │                           │   (360px)         │
│  · Navegación   │   • DroneMap (default)    │   · 4 KPIs        │
│  · Flota (cards)│   • PrologRulesView       │   · LiveStats     │
│  · Dinámica act │   • HistoricoView         │   · ControlPanel  │
│                 │   • FlotaView             │                   │
├─────────────────┼──────────────────────────┴───────────────────┤
│  StatBlock      │   RuleTerminal (Prolog terminal)              │
│  sistema info   │   [col 2-3]                                   │
└─────────────────┴─────────────────────────────────────────────┘
```

**Dimensiones:** `gridTemplateColumns: "220px 1fr 360px"` · `gridTemplateRows: "52px 1fr 220px"`

---

## 4. Sistema de Navegación

### 4.1 Estado Activo (`activeTab`)

`App.tsx` mantiene `activeTab: string` (`useState("operaciones")`). El callback `handleTabChange(tab)` actualiza este estado y dispara side-effects por tab:

| Tab key | Vista central renderizada | Side-effect |
|---|---|---|
| `operaciones` | `DroneMap` + `RuleTerminal` | — |
| `entrenamiento` | `DroneMap` + `RuleTerminal` | — (foco visual en gráficas) |
| `reglas` | `PrologRulesView` | — |
| `historico` | `HistoricoView` | fetch `GET /metrics/comparison` (lazy) |
| `flota` | `FlotaView` | — |

### 4.2 `SidebarLeft.tsx` — Integración de Navegación

`SidebarLeft` recibe `activeTab: string` y `onTabChange: (tab: string) => void` como props. Cada ítem de `NAV_ITEMS` (array `[label, icon, tabKey]`) llama `onTabChange(tabKey)` al hacer clic. El ítem activo se destaca visualmente por comparación `activeTab === tabKey`.

---

## 5. Vistas de Navegación (Inline en App.tsx)

### `PrologRulesView` — Reglas Simbólicas

Tabla con las 12 reglas Prolog del sistema neuro-simbólico:

| Columna | Contenido |
|---|---|
| ID | `R1`–`R12` |
| Nombre | Nombre semántico de la regla |
| Tipo | Badge coloreado: `MASK` (ámbar), `REWARD` (verde), `NEGOC` (violeta) |
| Peso | Importancia relativa |
| Descripción | Qué hace la regla (lenguaje natural) |

### `HistoricoView` — Histórico de Entrenamiento

- Tabla con los últimos 50 episodios del sistema activo: episodio, reward, tasa de éxito, colisiones, batería promedio restante.
- Sección de comparación entre sistemas: tabla con métricas A* / DQN / Neuro-DQN obtenidas del endpoint `GET /metrics/comparison`.

### `FlotaView` — Estado Detallado de Flota

- Grid de tarjetas, una por dron activo.
- Cada tarjeta muestra: ID, posición (X, Y), batería con barra de progreso coloreada (verde/ámbar/rojo), reward acumulado del episodio, altitud y estado operativo.
- Sub-componente `FlotaStat` para cada fila de estadística.

---

## 6. Integración de Datos

### 6.1 WebSocket (`/ws`)

El hook `useSocket("ws://localhost:8000/ws")` gestiona la conexión con reconexión automática cada 3 s.

#### Mensajes manejados

| `type` | Trigger backend | Acción frontend |
|---|---|---|
| `step_update` | Cada 25 steps | Actualiza `droneState` (posiciones, baterías, alive[]) + `dynamics` |
| `episode_complete` | Cada 10 episodios | Appends a `epHistory` (con avgBattery) → gráficas + `RuleTerminal` |
| `training_complete` | Fin del loop | Pone `isTraining=false` |

#### Estructura `step_update` (actualizada)
```json
{
  "type": "step_update",
  "episode": 42,
  "step": 125,
  "system": "neuro_dqn",
  "positions": [[12.5, 8.3, 1.2], ...],
  "batteries": [78.4, 55.1, 23.6, 91.0, 44.8],
  "rewards": [120.5, 88.2, 5.0, 210.3, 67.1],
  "alive": [true, true, false, true, true],
  "dynamics": { "storms": 1, "winds": 0, "nfzs": 2 },
  "symbolic_mask": [1,0,1,1,0,1,1,0]
}
```

> `alive[]` — nuevo campo: indica qué drones están operativos. Si el backend no lo envía, se usa `true` para todos (fail-safe).

#### Estructura `episode_complete` (actualizada)
```json
{
  "type": "episode_complete",
  "episode": 10,
  "system": "neuro_dqn",
  "record": {
    "total_reward": 450.2,
    "success_rate": 0.7,
    "rule_violations": 3,
    "symbolic_ops": 28,
    "collisions": 0,
    "avg_battery_remaining": 62.4
  },
  "symbolic_log": [
    { "timestamp": "12:34:56.789", "level": "MASK", "message": "R1 NFZ: drone_0→mover_n bloqueado" }
  ]
}
```

> `avg_battery_remaining` — nuevo campo: batería promedio de la flota al final del episodio.

### 6.2 REST API

| Endpoint | Método | Cuándo se llama |
|---|---|---|
| `/health` | GET | Al montar `App` (estado inicial del sistema) |
| `/drone-state` | GET | Al montar `App` (posiciones iniciales de drones) |
| `/training/start?system=X&episodes=Y` | POST | Al pulsar "Iniciar Entrenamiento" |
| `/training/stop` | POST | Al pulsar "Detener" |
| `/metrics/comparison` | GET | Al entrar en tab "Histórico" (lazy, una vez) |

### 6.3 Proxy Vite

```ts
proxy: {
  "/health":       { target: "http://localhost:8000" },
  "/training":     { target: "http://localhost:8000" },
  "/drone-state":  { target: "http://localhost:8000" },
  "/metrics":      { target: "http://localhost:8000" },
  "/ws":           { target: "ws://localhost:8000", ws: true },
}
```

---

## 7. Tipos TypeScript (`types/index.ts`)

### Cambios respecto a la versión anterior

| Tipo | Campo nuevo | Descripción |
|---|---|---|
| `EpisodePoint` | `avgBattery?: number` | Batería promedio de flota al cierre del episodio |
| `StepUpdateMsg` | `alive?: boolean[]` | Array de estado operativo por dron |
| `EpisodeCompleteMsg.record` | `avg_battery_remaining?: number` | Campo de raw del backend |

---

## 8. Componentes — Descripción Detallada

### `App.tsx`
- Fuente de verdad de todo el estado de la aplicación
- Gestiona 6 áreas de estado: `droneState`, `dynamics`, `training`, `epHistory`, `symLogs`, `activeTab`, `compareData`
- Parsea mensajes WebSocket y los distribuye a los componentes hijos
- `handleTabChange`: callback que cambia de tab y dispara fetches lazy (ej. `/metrics/comparison`)
- Componentes inline: `TopBar`, `Kpi`, `StatBlock`, `PrologRulesView`, `HistoricoView`, `FlotaView`, `FlotaStat`

### `SidebarLeft.tsx`
- **Navegación funcional**: recibe `activeTab` + `onTabChange`; cada ítem activa su vista
- Cards de drones: estado (ACT/BAJO/CRÍT/OFF), batería animada, posición XY
- Sección de dinámica activa: tormentas, vientos, NFZ dinámicas con contadores
- Footer con info de grid y configuración

### `ControlPanel.tsx`
- Selector de sistema: `neuro_dqn` / `dqn` / `astar`
- Slider de episodios: 10–500 (step 10)
- Barra de progreso animada durante entrenamiento
- Warning si Prolog está offline y se selecciona `neuro_dqn`

### `DroneMap.tsx`
- SVG 550×550px para grid 50×50 (11px/celda)
- Drones renderizados como círculos coloreados por batería
- **Bug fix**: propiedad CSS `overflowAuto` corregida a `overflow: "auto"`

### `LiveStats.tsx`
- 4 gráficas Recharts apiladas: Reward (AreaChart), Success Rate (LineChart con 90% ref), Symbolic Activity (LineChart 2 series), Battery (AreaChart)
- Indicador de estado (verde = entrenando, rojo = idle)

### `RuleTerminal.tsx`
- Terminal monospace con auto-scroll
- 9 niveles de badge: INFO, MASK, REWARD, ALERT, CRITICAL, NEGOTIATION, PRIO, RUTA, CLIMA

---

## 9. Mapeo de Niveles Prolog → RuleTerminal

| Bridge (Python) | → | RuleTerminal (React) | Color |
|---|---|---|---|
| MASK | → | MASK | `#f59e0b` ámbar |
| ALERT / AVISO | → | ALERT | `#fb923c` naranja |
| CRITICAL | → | CRITICAL | `#ef4444` rojo |
| REWARD | → | REWARD | `#22c55e` verde |
| PRIO | → | PRIO | `#f472b6` rosa |
| RUTA | → | RUTA | `#34d399` esmeralda |
| CLIMA | → | CLIMA | `#60a5fa` azul |
| NEGOC | → | NEGOTIATION | `#a78bfa` violeta |
| INFO | → | INFO | `#64748b` gris |

---

## 10. Paleta de Colores

```ts
const P = {
  drone:  "#22d3ee",  // cyan  — drones operativos
  warn:   "#fbbf24",  // ámbar — advertencias, Prolog offline
  crit:   "#ef4444",  // rojo  — crítico, batería baja
  ok:     "#10b981",  // verde — OK, éxito, conectado
  storm:  "#a855f7",  // violeta — tormentas activas
  bg:     "#07090d",  // fondo oscuro principal
  panel:  "#0a0e14",  // fondo de paneles
  border: "rgba(125,211,252,0.08)",
}
```

---

## 11. Inicio Rápido

```bash
# Terminal 1 — Backend
cd backend
python main.py

# Terminal 2 — Frontend
cd frontend
npm install
npm run dev
# → http://localhost:5173
```

---

## 12. Correcciones Aplicadas (sesión 2026-05-07/08)

| Área | Problema | Solución |
|---|---|---|
| `DroneMap.tsx` | CSS `overflowAuto: "auto"` no existe en CSSProperties | Cambiado a `overflow: "auto"` |
| `App.tsx` | Import `React` declarado pero no usado | Eliminado (react-jsx transform activo) |
| Navegación | Tabs sin funcionalidad real | `activeTab` state + `handleTabChange` + 3 vistas inline |
| `SidebarLeft` | Ítems de nav no tenían tab key | Prop `onTabChange` + `activeTab` añadidos |
| `types/index.ts` | Campos `alive` y `avgBattery` faltantes | Añadidos como opcionales con fail-safe |
| Backend | Tasa de éxito = 0% siempre | Estado expandido a 11D: delta a objetivo + delta a estación de carga |
| Backend | Reward inflado (R6 duplicado) | R6 eliminado del bridge; solo el env lo emite en delivery |
| Backend | Entrenamiento muy lento | `fast_action_mask` en Python; throttling Prolog (mask/5, reward/10, learn/4) |

---

## 13. Limitaciones Conocidas

| Limitación | Causa | Workaround |
|---|---|---|
| NFZ estáticas sin polígono visual en mapa | WS solo envía contadores | Indicador textual en barra lateral |
| Zonas de tormenta sin polígono visual | Ídem | Contador en dinámica activa |
| Sin autenticación | Prototipo académico | CORS abierto en backend |
| Un solo operador simultáneo | Estado global en backend | Por diseño del experimento |
| `FlotaView` no muestra carga de paquetes | Campo no enviado por WS | Requiere ampliar `step_update` |

---

## 14. Próximos Pasos Sugeridos

### Prioridad Alta

- [ ] **Endpoint `GET /state/environment`**: exponer coords de NFZ + charging stations → renderizar en `DroneMap` como polígonos/iconos
- [ ] **Precargar histórico**: `GET /metrics/live/{system}` al iniciar para poblar gráficas con sesiones anteriores
- [ ] **Persistencia de checkpoints en UI**: botón "Cargar checkpoint" en `ControlPanel` usando `POST /training/load`
- [ ] **Mostrar drones "muertos"** (`alive=false`): atenuarlos o marcarlos con X en el mapa

### Prioridad Media

- [ ] **Comparación visual A* vs DQN vs Neuro-DQN**: gráfica de barras en `HistoricoView` con las métricas del endpoint `/metrics/comparison`
- [ ] **Mapa con paquetes**: añadir posiciones de paquetes pendientes/entregados al payload de `step_update`
- [ ] **Filtrado de logs en `RuleTerminal`**: checkboxes por nivel de badge (MASK, REWARD, ALERT…)
- [ ] **Export de datos**: botón para descargar `epHistory` como CSV

### Prioridad Baja

- [ ] **Tests con Vitest + React Testing Library** para los hooks y componentes clave
- [ ] **Docker Compose** con frontend + backend + SWI-Prolog
- [ ] **Modo oscuro/claro** (actualmente solo oscuro)
- [ ] **Zustand** para reemplazar prop-drilling si el estado crece más
