"""
main.py — FastAPI backend + WebSocket para el Smart-Swarm Neuro-Simbólico.

Endpoints REST:
  GET  /health                 → estado del servicio
  GET  /metrics/summary        → estadísticas agregadas
  GET  /metrics/comparison     → tabla comparativa A*/DQN/Neuro-DQN
  GET  /metrics/curve/{system} → curva de aprendizaje (incluye reward_smooth)
  GET  /metrics/report         → reporte experimental con estadística (media±std, IC95%)
  GET  /metrics/live/{system}  → últimos N episodios (feed WebSocket)
  POST /training/start         → lanza el loop de entrenamiento (async; mode=resume|scratch)
  POST /training/stop          → detiene el entrenamiento
  GET  /training/status        → checkpoints + episodios registrados por sistema
  POST /training/delete-data   → elimina checkpoints y métricas (todo o por sistema)
  GET  /drone-state            → snapshot actual de los drones
  GET  /ml/demand              → predicción de demanda del modelo ML (zonas + heatmap)
  GET  /symbolic-log           → últimas decisiones del motor Prolog

WebSocket:
  WS /ws  — canal bidireccional ping/pong + broadcast de entrenamiento
             broadcast cada 25 steps: posiciones, baterías, recompensas, dinámica
             broadcast cada 10 episodios: métricas, log simbólico

Notas de despliegue:
  cd backend && python main.py
  o bien: uvicorn backend.main:app --host 0.0.0.0 --port 8000 --workers 1
"""

import asyncio
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Set

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Asegurar que el directorio backend esté en sys.path
# (permite tanto "python main.py" como "uvicorn backend.main:app")
_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from agents.astar_agent import AStarAgent
from agents.dqn_agent import DQNAgent
from analysis.metrics import EpisodeRecord, MetricsCollector
from environment.city_env import ACTIONS as ENV_ACTIONS
from environment.city_env import CyberCityEnv
from environment.dynamics import DynamicsEngine
from logic.neuro_symbolic_bridge import NeuroSymbolicBridge
from ml_models.demand_predictor import DemandPredictor

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("dron-intel-os")

# ─────────────────────────────────────────────────────────────────────────────
# App FastAPI
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Dron-Intel-OS API",
    version="1.0.0",
    description="Smart-Swarm Neuro-Simbólico — backend de entrenamiento y telemetría",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# Constantes del experimento (experimental_protocol.md §2.1-2.2)
# ─────────────────────────────────────────────────────────────────────────────

GRID_SIZE    = 50
NUM_DRONES   = 5
STATE_DIM    = 11
ACTION_DIM   = 8
RULES_FILE   = os.path.join(_BACKEND_DIR, "logic", "rules.pl")

# Throttling de las llamadas al motor Prolog (carísimas) durante entrenamiento.
# Reward shaping simbólico se evalúa cada N steps; entremedio se usa el shaping del env.
SYMBOLIC_REWARD_EVERY = 10
# Action masking: el mask vectorizado del env es default; el bridge solo se consulta
# para reglas avanzadas (R11/R12) cada N steps.
SYMBOLIC_MASK_EVERY = 5

# ─────────────────────────────────────────────────────────────────────────────
# Estado global de la simulación
# ─────────────────────────────────────────────────────────────────────────────

env:              Optional[CyberCityEnv]       = None
agents:           List[Any]                    = []   # DQNAgent | AStarAgent
bridge:           Optional[NeuroSymbolicBridge] = None
_DATA_DIR = os.path.join(os.path.dirname(_BACKEND_DIR), "data")
# Directorio de checkpoints DQN — un .pt por dron y por sistema.
# Permite continuar el entrenamiento entre ejecuciones (conocimiento previo).
_CKPT_DIR = os.path.join(_DATA_DIR, "checkpoints")
metrics:          MetricsCollector             = MetricsCollector(
    log_path=os.path.join(_DATA_DIR, "training_logs.csv")
)
dynamics:         Optional[DynamicsEngine]     = None
predictor:        Optional[DemandPredictor]    = None   # ML: predictor de demanda
training_active:  bool                         = False
current_system:   str                          = "neuro_dqn"

# Top-K zonas de alta demanda que el predictor ML inyecta en el entorno cada episodio.
DEMAND_TOP_K = 5

# ─────────────────────────────────────────────────────────────────────────────
# Connection Manager WebSocket
# ─────────────────────────────────────────────────────────────────────────────

class ConnectionManager:
    """Gestiona múltiples conexiones WebSocket con tolerancia a desconexiones."""

    def __init__(self) -> None:
        self.connections: Set[WebSocket] = set()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self.connections.add(ws)
        logger.debug("WS conectado — total: %d", len(self.connections))

    def disconnect(self, ws: WebSocket) -> None:
        self.connections.discard(ws)
        logger.debug("WS desconectado — total: %d", len(self.connections))

    async def broadcast(self, payload: Dict) -> None:
        text  = json.dumps(payload, default=str)
        stale: Set[WebSocket] = set()
        for ws in list(self.connections):
            try:
                await ws.send_text(text)
            except Exception:
                stale.add(ws)
        self.connections -= stale


manager = ConnectionManager()

# ─────────────────────────────────────────────────────────────────────────────
# Startup
# ─────────────────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup() -> None:
    global env, agents, bridge, dynamics, predictor
    env      = CyberCityEnv(grid_size=GRID_SIZE, num_drones=NUM_DRONES)
    dynamics = DynamicsEngine(grid_size=GRID_SIZE)
    agents   = [
        DQNAgent(agent_id=f"drone_{i}", state_dim=STATE_DIM, action_dim=ACTION_DIM)
        for i in range(NUM_DRONES)
    ]
    try:
        bridge = NeuroSymbolicBridge(RULES_FILE)
        logger.info("Motor Prolog activo: %s", RULES_FILE)
    except Exception as exc:
        logger.warning("Bridge simbólico no disponible: %s", exc)
        bridge = None

    # Modelo ML predictivo del entorno: se entrena una vez al arrancar con datos
    # sintéticos realistas. Sus predicciones de demanda sesgan dónde aparecen los
    # destinos de entrega cada episodio (uso activo de ML antes de que el DQN actúe).
    try:
        predictor = DemandPredictor(grid_size=GRID_SIZE)
        train_stats = predictor.train()
        logger.info(
            "DemandPredictor entrenado: R²=%.3f sobre %d muestras",
            train_stats["r2_score"], train_stats["n_samples"],
        )
    except Exception as exc:
        logger.warning("DemandPredictor no disponible: %s", exc)
        predictor = None


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints REST
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> Dict:
    return {
        "status":      "ok",
        "training":    training_active,
        "system":      current_system,
        "symbolic_ok": bridge is not None,
        "ml_ok":       predictor is not None and predictor.is_trained,
        "num_drones":  NUM_DRONES,
        "grid_size":   GRID_SIZE,
    }


@app.get("/metrics/summary")
async def get_summary(system: Optional[str] = None, last_n: int = 100) -> Dict:
    return metrics.get_summary(system=system, last_n=last_n)


@app.get("/metrics/comparison")
async def get_comparison() -> JSONResponse:
    df = metrics.get_comparison_table()
    return JSONResponse(content=df.to_dict() if not df.empty else {})


@app.get("/metrics/curve/{system}")
async def get_curve(system: str, metric: str = "success_rate") -> Dict:
    df = metrics.get_learning_curve(system, metric)
    return df.to_dict(orient="list")


@app.get("/metrics/live/{system}")
async def get_live(system: str, last_n: int = 50) -> Dict:
    return metrics.to_live_json(system, last_n)


@app.get("/metrics/report")
async def get_experimental_report() -> Dict:
    """Reporte experimental con estadística (media±std, IC95%) por sistema."""
    return metrics.get_experimental_report()


@app.post("/training/start")
@app.post("/start-training")
async def start_training(
    system: str = "neuro_dqn",
    episodes: int = 200,
    mode: str = "scratch",
    seed: Optional[int] = None,
) -> Dict:
    """Inicia el entrenamiento.

    Args:
        system  : 'astar' | 'dqn' | 'neuro_dqn'.
        episodes: número de episodios a correr.
        mode    : 'resume' carga checkpoints previos (conocimiento acumulado);
                  'scratch' entrena desde cero (pesos aleatorios, ε=1.0).
        seed    : semilla base opcional. Si se indica, cada episodio usa
                  `seed + episodio` para entorno y dinámica → corridas
                  reproducibles y condiciones IDÉNTICAS entre A*/DQN/Neuro-DQN
                  (mismo layout de paquetes y misma secuencia de eventos).
    """
    global training_active, current_system, agents
    if training_active:
        return {"error": "Entrenamiento ya activo", "system": current_system}

    # Reinicializar agentes según el sistema elegido
    agents = _build_agents(system)
    resumed = 0
    if mode == "resume":
        resumed = _load_checkpoints(agents, system)

    training_active = True
    current_system  = system
    asyncio.create_task(
        _training_loop(system=system, max_episodes=episodes, base_seed=seed)
    )
    logger.info(
        "Entrenamiento iniciado: system=%s, episodes=%d, mode=%s, seed=%s (checkpoints cargados: %d)",
        system, episodes, mode, seed, resumed,
    )
    return {
        "status":           "started",
        "system":           system,
        "max_episodes":     episodes,
        "mode":             mode,
        "seed":             seed,
        "checkpoints_loaded": resumed,
    }


@app.post("/training/stop")
@app.post("/stop-training")
async def stop_training() -> Dict:
    global training_active
    training_active = False
    logger.info("Señal de parada de entrenamiento enviada")
    return {"status": "stopping"}


@app.get("/training/status")
async def training_status() -> Dict:
    """Reporta disponibilidad de conocimiento previo (checkpoints + métricas).

    El frontend lo usa para ofrecer 'continuar' vs 'desde cero' y mostrar
    cuántos episodios hay registrados por sistema.
    """
    return {
        "training_active": training_active,
        "current_system":  current_system,
        "systems": {
            sys_name: {
                "has_checkpoints":  _has_checkpoints(sys_name),
                "episodes_recorded": metrics.count(system=sys_name),
            }
            for sys_name in ("dqn", "neuro_dqn", "astar")
        },
        "total_episodes_recorded": metrics.count(),
    }


@app.post("/training/delete-data")
async def delete_training_data(system: Optional[str] = None) -> Dict:
    """Elimina checkpoints y registros de métricas (todo o por sistema).

    No permite borrar mientras hay un entrenamiento activo (evita corrupción).
    """
    if training_active:
        return {"error": "No se puede eliminar datos durante un entrenamiento activo"}
    result = _delete_training_data(system=system)
    return {"status": "deleted", "system": system or "ALL", **result}


@app.get("/drone-state")
@app.get("/state/drones")
async def get_drone_state() -> Dict:
    if env is None:
        return {"error": "Entorno no inicializado"}
    return {
        "positions":  env.drone_positions.tolist(),
        "batteries":  env.drone_batteries.round(1).tolist(),
        "alive":      env.drone_alive.tolist(),
        "cargos":     env.drone_cargos,
        "deliveries": int(env.package_delivered.sum()),
        "step":       env.current_step,
        "demand_zones": [list(z) for z in env.demand_zones],
    }


@app.get("/ml/demand")
async def get_ml_demand(
    hour: Optional[int] = None,
    weekday: Optional[int] = None,
    top_k: int = DEMAND_TOP_K,
) -> Dict:
    """Predicción de demanda del modelo ML para un contexto temporal dado.

    Sin parámetros usa la hora/día actuales del servidor. Retorna las top-K
    zonas de alta demanda y un mapa de calor submuestreado para visualización.
    """
    if predictor is None or not predictor.is_trained:
        return {"available": False, "zones": [], "heatmap": []}

    from datetime import datetime as _dt
    now = _dt.now()
    context = {
        "hour":        now.hour if hour is None else int(hour),
        "weekday":     now.weekday() if weekday is None else int(weekday),
        "temperature": 20.0,
        "rain":        0,
    }
    zones = predictor.get_high_demand_zones(context, top_k=top_k)
    heatmap = predictor.predict(context)
    # Submuestrear el heatmap (cada 5 celdas) para un payload ligero
    step = 5
    coarse = heatmap[::step, ::step]
    return {
        "available": True,
        "context":   context,
        "zones":     [list(z) for z in zones],
        "heatmap":   np.round(coarse, 2).tolist(),
        "heatmap_step": step,
    }


@app.get("/symbolic-log")
@app.get("/log/symbolic")
async def get_symbolic_log() -> Dict:
    if bridge is None:
        return {"entries": [], "available": False}
    return {
        "entries":    bridge.get_log_entries(last_n=50),
        "available":  True,
        "interventions": bridge.intervention_count,
    }


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket endpoint
# ─────────────────────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    await manager.connect(ws)
    try:
        while True:
            data = await ws.receive_text()
            try:
                msg = json.loads(data)
                if msg.get("type") == "ping":
                    await ws.send_text(json.dumps({"type": "pong"}))
            except json.JSONDecodeError:
                pass
    except WebSocketDisconnect:
        manager.disconnect(ws)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers para el training loop
# ─────────────────────────────────────────────────────────────────────────────

def _build_agents(system: str) -> List[Any]:
    """Construye la lista de agentes según el sistema de comparación."""
    if system == "astar":
        return [
            AStarAgent(
                agent_id=f"drone_{i}",
                state_dim=STATE_DIM,
                action_dim=ACTION_DIM,
                grid_size=GRID_SIZE,
            )
            for i in range(NUM_DRONES)
        ]
    # DQN puro o Neuro-DQN (misma arquitectura, distinto masking)
    return [
        DQNAgent(
            agent_id=f"drone_{i}",
            state_dim=STATE_DIM,
            action_dim=ACTION_DIM,
        )
        for i in range(NUM_DRONES)
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Persistencia de checkpoints DQN (continuar vs desde cero)
# ─────────────────────────────────────────────────────────────────────────────

def _ckpt_path(system: str, drone_idx: int) -> str:
    """Ruta del checkpoint de un dron para un sistema dado."""
    return os.path.join(_CKPT_DIR, f"{system}_drone_{drone_idx}.pt")


def _has_checkpoints(system: str) -> bool:
    """True si existe al menos un checkpoint guardado para el sistema."""
    if system == "astar":
        return False  # A* no aprende → no persiste pesos
    return any(os.path.exists(_ckpt_path(system, i)) for i in range(NUM_DRONES))


def _load_checkpoints(agent_list: List[Any], system: str) -> int:
    """Carga checkpoints en los agentes si existen. Retorna cuántos cargó."""
    loaded = 0
    for i, agent in enumerate(agent_list):
        path = _ckpt_path(system, i)
        if hasattr(agent, "load_checkpoint") and os.path.exists(path):
            try:
                agent.load_checkpoint(path)
                loaded += 1
            except Exception as exc:
                logger.warning("No se pudo cargar checkpoint %s: %s", path, exc)
    if loaded:
        logger.info("Checkpoints cargados para %s: %d/%d drones", system, loaded, NUM_DRONES)
    return loaded


def _save_checkpoints(agent_list: List[Any], system: str) -> int:
    """Guarda checkpoints de todos los agentes que sepan persistir. Retorna cuántos guardó."""
    if system == "astar":
        return 0  # A* no tiene estado entrenable
    os.makedirs(_CKPT_DIR, exist_ok=True)
    saved = 0
    for i, agent in enumerate(agent_list):
        if hasattr(agent, "save_checkpoint"):
            try:
                agent.save_checkpoint(_ckpt_path(system, i))
                saved += 1
            except Exception as exc:
                logger.warning("No se pudo guardar checkpoint de drone_%d: %s", i, exc)
    if saved:
        logger.info("Checkpoints guardados para %s: %d/%d drones", system, saved, NUM_DRONES)
    return saved


def _delete_training_data(system: Optional[str] = None) -> Dict[str, int]:
    """Elimina checkpoints y registros de métricas.

    Args:
        system: si se indica, borra solo los datos de ese sistema; None = todo.

    Returns:
        Dict con conteos de archivos/filas eliminados.
    """
    removed_ckpts = 0
    systems = [system] if system else ["dqn", "neuro_dqn"]
    for sys_name in systems:
        for i in range(NUM_DRONES):
            path = _ckpt_path(sys_name, i)
            if os.path.exists(path):
                try:
                    os.remove(path)
                    removed_ckpts += 1
                except OSError as exc:
                    logger.warning("No se pudo eliminar %s: %s", path, exc)
    removed_rows = metrics.clear(system=system)
    logger.info(
        "Datos de entrenamiento eliminados (system=%s): %d checkpoints, %d filas de métricas",
        system or "ALL", removed_ckpts, removed_rows,
    )
    return {"checkpoints_removed": removed_ckpts, "metric_rows_removed": removed_rows}


def _episode_demand_context(episode: int) -> Dict[str, Any]:
    """Construye el contexto temporal del episodio para el predictor ML.

    Mapea el índice de episodio a una hora-del-día y día-de-semana cíclicos,
    de modo que distintos episodios "viven" en franjas horarias distintas y el
    predictor produce mapas de demanda variados (picos de mañana/almuerzo/tarde).
    """
    hour    = episode % 24
    weekday = (episode // 24) % 7
    return {"hour": hour, "weekday": weekday, "temperature": 20.0, "rain": 0}


def _predicted_demand_zones(episode: int) -> List[tuple]:
    """Devuelve las top-K zonas de alta demanda predichas por el modelo ML.

    Si el predictor no está disponible, retorna [] → el entorno usa spawn
    uniforme (comportamiento previo intacto).
    """
    if predictor is None:
        return []
    try:
        context = _episode_demand_context(episode)
        return predictor.get_high_demand_zones(context, top_k=DEMAND_TOP_K)
    except Exception as exc:
        logger.warning("Predicción de demanda falló (ep %d): %s", episode, exc)
        return []


def _get_astar_target(
    env_ref: CyberCityEnv,
    drone_idx: int,
) -> Optional[tuple]:
    """Retorna el objetivo del dron para el AStarAgent.

    Si lleva carga → destino de entrega.
    Si no lleva carga → paquete no entregado más cercano.
    """
    pi = env_ref.drone_cargos[drone_idx]
    if pi is not None:
        dest = env_ref.package_destinations[pi]
        return (int(dest[0]), int(dest[1]))

    # Buscar el paquete no-entregado más cercano
    dx, dy, _ = env_ref.drone_positions[drone_idx]
    best_dist = float("inf")
    best_pos  = None
    for i in range(env_ref.num_packages):
        if not env_ref.package_delivered[i] and not env_ref.package_picked[i]:
            px = int(env_ref.package_positions[i, 0])
            py = int(env_ref.package_positions[i, 1])
            d  = abs(px - dx) + abs(py - dy)
            if d < best_dist:
                best_dist = d
                best_pos  = (px, py)
    return best_pos


# ─────────────────────────────────────────────────────────────────────────────
# Loop de entrenamiento asíncrono
# ─────────────────────────────────────────────────────────────────────────────

async def _training_loop(
    system: str, max_episodes: int, base_seed: Optional[int] = None
) -> None:
    """Loop principal de entrenamiento — ejecuta max_episodes episodios.

    Args:
        base_seed: si se indica, el episodio `e` usa la semilla `base_seed + e`
                   para entorno y dinámica (reproducibilidad y comparación justa
                   entre sistemas). None = estocástico (comportamiento previo).

    Broadcast:
      - Cada 25 steps: step_update con posiciones, baterías, dinámica.
      - Cada 10 episodios: episode_complete con métricas y log simbólico.
    """
    global training_active, env, agents, bridge, dynamics

    logger.info("Training loop iniciado: system=%s, max_ep=%d", system, max_episodes)

    for episode in range(max_episodes):
        if not training_active:
            break

        assert env is not None and dynamics is not None

        # ── Semilla reproducible por episodio (si se configuró base_seed) ─────
        # Misma semilla → mismo layout de paquetes y misma dinámica para los 3
        # sistemas, permitiendo una comparación estadística justa (protocolo §2).
        ep_seed = (base_seed + episode) if base_seed is not None else None

        # ── ML: predecir zonas de alta demanda e inyectarlas en el entorno ────
        # El predictor decide dónde aparecerán los destinos de entrega ANTES de
        # que los agentes (DQN/A*) planifiquen sus acciones para este episodio.
        demand_zones = _predicted_demand_zones(episode)
        obs, _  = env.reset(seed=ep_seed, options={"demand_zones": demand_zones})
        dynamics.reset(seed=ep_seed)
        if bridge:
            bridge.reset_intervention_count()

        ep_reward           = np.zeros(NUM_DRONES, dtype=np.float64)
        ep_collisions       = 0
        ep_violations       = 0
        ep_battery_failures = 0
        ep_symbolic_ops     = 0
        ep_losses:    List[float] = []
        last_masks_json: List = []

        for step in range(env.max_steps):
            if not training_active:
                break

            # Avanzar dinámica del entorno
            dyn = dynamics.step()
            env.apply_dynamics(dyn)

            actions: List[int]    = []
            masks_json: List[Any] = []
            step_deliveries: List[Dict[str, Any]] = []  # entregas ocurridas en este step

            # Consulta al bridge solo cuando epsilon es bajo (explotación) y en el throttle.
            # Durante exploración alta (ε > 0.5) las acciones son aleatorias de todas formas,
            # por lo que el overhead de Prolog es puro desperdicio.
            agent_epsilon = getattr(agents[0], "epsilon", 0.0) if agents else 0.0
            consult_bridge_mask = (
                system == "neuro_dqn" and bridge is not None
                and step % SYMBOLIC_MASK_EVERY == 0
                and agent_epsilon <= 0.5
            )

            # ── Selección de acción por cada agente ──────────────────────────
            for i, agent in enumerate(agents):
                if not env.drone_alive[i]:
                    actions.append(6)  # esperar
                    continue

                state_vec = obs[f"drone_{i}"]
                mask: Optional[np.ndarray] = None

                if system == "astar":
                    # AStarAgent necesita objetivo y obstáculos actualizados
                    astar_agent: AStarAgent = agent  # type: ignore[assignment]
                    astar_agent.set_target(_get_astar_target(env, i))
                    astar_agent.set_obstacles(
                        env.no_fly_zones,
                        storm_cells=env.get_blocked_cells(),
                    )

                elif system in ("dqn", "neuro_dqn"):
                    # Mask rápido vectorizado (R1, R3, R5, R7, R2) — sin Prolog
                    # DQN puro también necesita masking para evitar violaciones NFZ/cargar
                    mask = env.fast_action_mask(i)
                    # Refinar con Prolog ocasionalmente (R11, R12) — solo neuro_dqn
                    if system == "neuro_dqn" and consult_bridge_mask:
                        state_dict = env.get_state_dict(i)
                        prolog_mask = bridge.get_action_mask(f"drone_{i}", state_dict)  # type: ignore[union-attr]
                        mask = mask * prolog_mask  # AND lógico
                        if float(mask.sum()) == 0.0:
                            mask = env.fast_action_mask(i)  # fail-safe
                    if float(mask.sum()) < ACTION_DIM:
                        ep_symbolic_ops += 1
                    masks_json.append(mask.tolist())

                action = agent.select_action(state_vec, mask)
                actions.append(action)

            next_obs, rewards, dones, truncated, infos = env.step(np.array(actions))

            # ── Reward shaping + aprendizaje ─────────────────────────────────
            # Reward shaping simbólico solo cada SYMBOLIC_REWARD_EVERY steps
            # y solo en fase de explotación (ε ≤ 0.5); en exploración es irrelevante.
            apply_symbolic_reward = (
                system == "neuro_dqn" and bridge is not None
                and step % SYMBOLIC_REWARD_EVERY == 0
                and agent_epsilon <= 0.5
            )

            for i, agent in enumerate(agents):
                if not env.drone_alive[i]:
                    continue

                r = float(rewards[i])

                # Reward shaping simbólico (throttled)
                if apply_symbolic_reward:
                    state_dict = env.get_state_dict(i)
                    r += bridge.get_reward_modifier(  # type: ignore[union-attr]
                        agent_id=f"drone_{i}",
                        state=state_dict,
                        action=ENV_ACTIONS[actions[i]],
                    )

                # Calcular máscara para el next_state y almacenarla en la transición
                try:
                    next_mask = env.fast_action_mask(i)
                    if consult_bridge_mask:
                        state_dict_next = env.get_state_dict(i)
                        prolog_mask_next = bridge.get_action_mask(f"drone_{i}", state_dict_next)  # type: ignore[union-attr]
                        next_mask = next_mask * prolog_mask_next
                except Exception:
                    next_mask = None

                agent.remember(
                    obs[f"drone_{i}"], actions[i], r,
                    next_obs[f"drone_{i}"], bool(dones[i]),
                    next_mask=next_mask,
                )
                # Throttle learn() — cada 4 steps en lugar de cada step.
                # Reduce 75% el costo de gradiente sin afectar convergencia.
                if step % 4 == 0:
                    learn_info = agent.learn()
                    if isinstance(learn_info, dict):
                        loss_val = learn_info.get("loss", 0.0)
                        if loss_val and loss_val > 0.0:
                            ep_losses.append(float(loss_val))
                ep_reward[i] += r

                info = infos[f"drone_{i}"]
                ep_collisions       += info.get("collisions", 0)
                ep_violations       += info.get("rule_violations", 0)
                ep_battery_failures += 1 if bool(dones[i]) else 0

                # Capturar eventos de entrega para animarlos en el frontend.
                # La entrega ocurre en la celda actual del dron (= destino del paquete).
                if info.get("delivery"):
                    dpos = env.drone_positions[i]
                    step_deliveries.append({
                        "drone":     i,
                        "x":         int(dpos[0]),
                        "y":         int(dpos[1]),
                        "pkg_type":  info.get("pkg_type", "standard"),
                    })

            obs = next_obs
            last_masks_json = masks_json

            # ── Broadcast inmediato de entregas (no esperar al throttle de 25) ──
            if step_deliveries:
                await manager.broadcast({
                    "type":       "delivery",
                    "episode":    episode,
                    "step":       step,
                    "system":     system,
                    "deliveries": step_deliveries,
                    "total":      int(env.package_delivered.sum()),
                })
                step_deliveries = []

            # ── Broadcast cada 25 steps ──────────────────────────────────────
            if step % 25 == 0:
                await manager.broadcast({
                    "type":     "step_update",
                    "episode":  episode,
                    "step":     step,
                    "system":   system,
                    "positions": env.drone_positions.tolist(),
                    "batteries": env.drone_batteries.round(1).tolist(),
                    "rewards":   ep_reward.round(2).tolist(),
                    "alive":    env.drone_alive.tolist(),
                    "dynamics": {
                        "storms": dyn["num_active_storms"],
                        "winds":  dyn["num_active_winds"],
                        "nfzs":   dyn["num_dynamic_nfzs"],
                    },
                    # Celdas/regiones reales para que el mapa pinte NFZ y tormentas en vivo.
                    # NFZ: lista de [x,y] (capada para no inflar el payload).
                    # Tormentas: rectángulos {x_range,y_range} que DroneMap renderea.
                    "no_fly_zones": [list(c) for c in env.no_fly_zones[:400]],
                    "storm_regions": [
                        {"x_range": list(z["x_range"]), "y_range": list(z["y_range"])}
                        for z in env.climate_zones.values()
                        if z.get("type") == "storm"
                    ],
                    # Destinos de entrega pendientes (objetivo de cada paquete sin entregar)
                    # con su tipo, para que el mapa muestre adónde van las entregas.
                    "delivery_targets": [
                        {
                            "x":    int(env.package_destinations[p, 0]),
                            "y":    int(env.package_destinations[p, 1]),
                            "type": env.package_types[p],
                        }
                        for p in range(env.num_packages)
                        if not env.package_delivered[p]
                    ],
                    # Carga actual por dron: índice del paquete y su destino (para la línea guía).
                    "cargos": [
                        None if env.drone_cargos[i] is None else {
                            "pkg":  int(env.drone_cargos[i]),
                            "type": env.package_types[env.drone_cargos[i]],
                            "dest": [
                                int(env.package_destinations[env.drone_cargos[i], 0]),
                                int(env.package_destinations[env.drone_cargos[i], 1]),
                            ],
                        }
                        for i in range(env.num_drones)
                    ],
                    "deliveries_done": int(env.package_delivered.sum()),
                    "symbolic_mask": last_masks_json[0] if last_masks_json else None,
                })
                await asyncio.sleep(0)  # ceder el event loop

            if dones.all() or truncated.all():
                break

        # ── Decay epsilon una vez por episodio (formal_modeling.md §7) ─────────
        # El decay per-learn-step causaba ε → 0.05 en solo 5 episodios;
        # el decay per-episode garantiza ~500 episodios de exploración.
        if system in ("dqn", "neuro_dqn"):
            for agent in agents:
                if hasattr(agent, "update_epsilon"):
                    agent.update_epsilon()

        # ── Registrar episodio ────────────────────────────────────────────────
        record = EpisodeRecord(
            episode              = episode,
            system               = system,
            total_reward         = float(ep_reward.sum()),
            deliveries_completed = int(env.package_delivered.sum()),
            deliveries_total     = env.num_packages,
            rule_violations      = ep_violations,
            collisions           = ep_collisions,
            battery_failures     = ep_battery_failures,
            steps                = env.current_step,
            avg_battery_remaining= float(env.drone_batteries.mean()),
            symbolic_interventions= ep_symbolic_ops,
        )
        metrics.record_episode(record)

        # ── Broadcast cada episodio (o cada 2 para reducir overhead) ──────────
        if episode % 2 == 0:
            metrics.save()
            symbolic_log_entries = (
                bridge.get_log_entries(last_n=10) if bridge else []
            )
            avg_epsilon = float(np.mean([
                getattr(a, "epsilon", 1.0) for a in agents
                if hasattr(a, "epsilon")
            ])) if agents else 1.0
            avg_loss = float(np.mean(ep_losses)) if ep_losses else 0.0
            await manager.broadcast({
                "type":    "episode_complete",
                "episode": episode,
                "system":  system,
                "record": {
                    "total_reward":          record.total_reward,
                    "success_rate":          record.success_rate,
                    "rule_violations":       record.rule_violations,
                    "symbolic_ops":          record.symbolic_interventions,
                    "collisions":            record.collisions,
                    "avg_battery_remaining": record.avg_battery_remaining,
                    "deliveries":            record.deliveries_completed,
                    "epsilon":               avg_epsilon,
                    "avg_loss":              avg_loss,
                },
                "summary":      metrics.get_summary(system=system, last_n=50),
                "symbolic_log": symbolic_log_entries,
                "demand_zones": [list(z) for z in demand_zones],
                "ml_active":    predictor is not None,
            })
            logger.info(
                "Ep %d/%d [%s] reward=%.1f | success=%.1f%% | deliveries=%d | eps=%.3f | loss=%.4f",
                episode, max_episodes, system,
                record.total_reward, record.success_rate * 100,
                record.deliveries_completed, avg_epsilon, avg_loss,
            )

        await asyncio.sleep(0)  # yield para no bloquear el event loop

    # ── Fin del entrenamiento ─────────────────────────────────────────────────
    training_active = False
    metrics.save()
    # Persistir pesos para poder continuar en una próxima ejecución (resume).
    saved_ckpts = _save_checkpoints(agents, system)
    logger.info(
        "Entrenamiento completado: system=%s, episodios=%d, checkpoints=%d",
        system, max_episodes, saved_ckpts,
    )
    await manager.broadcast({
        "type":    "training_complete",
        "system":  system,
        "episodes": max_episodes,
        "summary":  metrics.get_summary(system=system),
        "checkpoints_saved": saved_ckpts,
    })


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        workers=1,   # 1 worker obligatorio (estado global compartido)
    )
