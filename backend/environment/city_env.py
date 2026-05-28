"""
city_env.py — CyberCityEnv: entorno Dec-POMDP para el Smart-Swarm.

Implementa formal_modeling.md §1-5:
  - Grid 50×50, 5 drones, 10 paquetes (2 médicos + 8 estándar), 4 estaciones
  - Observation space: Dict {drone_i: Box(7)} — s^i_t = (x,y,z,β,κ,ω,η)
  - Action space: MultiDiscrete([8]*5)
  - R_total = R_entrega + R_eficiencia − C_movimiento − P_colisión − P_batería − P_simbólica

Novedades vs versión anterior:
  - _calculate_rewards(): implementa la fórmula R_total de formal_modeling.md §5
  - _apply_symbolic_masking(): integra NeuroSymbolicBridge en el paso de entorno
  - Soporte de viento en get_state_dict() para sync_state() del bridge
  - get_blocked_cells(): celdas inaccesibles para el AStarAgent
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constantes globales
# ─────────────────────────────────────────────────────────────────────────────

ACTIONS: List[str] = [
    "despegar", "aterrizar",
    "mover_n",  "mover_s", "mover_e", "mover_o",
    "esperar",  "cargar",
]
N_ACTIONS = len(ACTIONS)

CLIMATE_ENC: Dict[str, int] = {"clear": 0, "storm": 1, "wind": 2}

# Costo de batería por acción (% por step).
# Calibrado para que 100% de carga dure ~250 movimientos: suficiente para
# cruzar el grid 50×50 (≤100 celdas Manhattan), recoger y entregar varios
# paquetes antes de necesitar recargar. Con el costo previo (0.75) la batería
# se agotaba en ~133 movimientos y los drones morían antes de poder entregar,
# dominando la recompensa con la penalización de batería (causa raíz de la
# no-convergencia diagnosticada en data/training_logs.csv).
BATTERY_COST: Dict[str, float] = {
    "despegar": 0.5, "aterrizar": 0.2,
    "mover_n":  0.4, "mover_s": 0.4, "mover_e": 0.4, "mover_o": 0.4,
    "esperar":  0.2, "cargar":   0.0,
}
BATTERY_RECHARGE = 40.0   # % ganado por step de carga (recarga rápida en 3 steps)


class CyberCityEnv(gym.Env):
    """Entorno Dec-POMDP para entrega de paquetes con drones autónomos.

    Observation space (por dron):
        Box([x, y, z, battery, cargo, climate, neighbor_count])
        s^i_t ∈ ℝ^7 según formal_modeling.md §2

    Action space:
        MultiDiscrete([8, 8, 8, 8, 8])  — 8 acciones × 5 drones

    Reward (por dron):
        R_total = R_entrega + R_eficiencia − C_movimiento − P_colisión
                  − P_batería − P_simbólica (añadida en el training loop)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(
        self,
        grid_size:             int = 50,
        num_drones:            int = 5,
        num_packages:          int = 10,
        num_charging_stations: int = 4,
        max_steps:             int = 500,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.grid_size             = grid_size
        self.num_drones            = num_drones
        self.num_packages          = num_packages
        self.num_charging_stations = num_charging_stations
        self.max_steps             = max_steps
        self.render_mode           = render_mode

        # ── Spaces Gymnasium-compliant ───────────────────────────────────────
        # s^i_t ∈ ℝ^11 = (x, y, z, β, κ, ω, η, tdx, tdy, cdx, cdy)
        # tdx/tdy = delta normalizado al objetivo (paquete o destino)
        # cdx/cdy = delta normalizado a la estación de carga más cercana
        # Todos los features normalizados a [0,1] excepto los deltas de dirección [-1,1]
        obs_low  = np.array([0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1], dtype=np.float32)
        obs_high = np.array([1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1], dtype=np.float32)
        self.observation_space = spaces.Dict({
            f"drone_{i}": spaces.Box(obs_low, obs_high, dtype=np.float32)
            for i in range(num_drones)
        })
        self.action_space = spaces.MultiDiscrete([N_ACTIONS] * num_drones)

        # ── Arrays de estado runtime (inicializados en reset()) ─────────────
        self.drone_positions:  np.ndarray = np.zeros((num_drones, 3), dtype=np.int32)
        self.drone_batteries:  np.ndarray = np.full(num_drones, 100.0)
        self.drone_cargos:     List[Optional[int]] = [None] * num_drones
        self.drone_alive:      np.ndarray = np.ones(num_drones, dtype=bool)

        self.package_positions:    np.ndarray = np.zeros((num_packages, 2), dtype=np.int32)
        self.package_destinations: np.ndarray = np.zeros((num_packages, 2), dtype=np.int32)
        self.package_delivered:    np.ndarray = np.zeros(num_packages, dtype=bool)
        self.package_picked:       np.ndarray = np.zeros(num_packages, dtype=bool)
        self.package_types:        List[str]  = ["standard"] * num_packages

        self.charging_stations: np.ndarray = np.zeros((num_charging_stations, 2), dtype=np.int32)
        self.charging_occupied: np.ndarray = np.zeros(num_charging_stations, dtype=bool)

        self.no_fly_zones:   List[Tuple[int, int]] = []
        self.climate_zones:  Dict[str, Dict]       = {}
        self.current_wind:   Optional[Tuple[str, float]] = None
        self.current_step:   int                   = 0

        # Caches precalculados para evitar reconstruir sets en hot paths
        self._nfz_set:     set = set()   # updated in reset() and apply_dynamics()
        self._storm_cells: set = set()   # updated in apply_dynamics()

        # Bridge simbólico inyectable (opcional)
        self._bridge: Optional[Any] = None
        self._last_interventions: int = 0

        # Tracker de distancia al objetivo (para proximity reward shaping)
        self._prev_target_dist: np.ndarray = np.full(num_drones, -1.0, dtype=np.float32)

        # Cache de la lista de NFZ serializada para Prolog — se invalida en apply_dynamics/reset
        self._nfz_list_cache: Optional[List[Tuple[int, int]]] = None

        # Zonas de alta demanda predichas por el modelo ML (DemandPredictor).
        # Cuando se proveen en reset(), una fracción de los paquetes se genera
        # cerca de ellas → la predicción de demanda da forma a la misión que
        # los agentes deben planificar (uso activo de ML, criterio 5 de la rúbrica).
        self._demand_zones: List[Tuple[int, int]] = []

    # ──────────────────────────────────────────────────────────────────────────
    # Interfaz Gymnasium
    # ──────────────────────────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict]:
        super().reset(seed=seed)
        rng = self.np_random
        g   = self.grid_size
        self.current_step = 0
        self.current_wind = None

        # Zonas de alta demanda (predicción ML) inyectadas vía options.
        # Si están presentes, sesgan el spawn de paquetes hacia esas celdas.
        if options is not None and "demand_zones" in options:
            self._demand_zones = [
                (int(zx), int(zy)) for zx, zy in options["demand_zones"]
                if 0 <= int(zx) < g and 0 <= int(zy) < g
            ]
        else:
            self._demand_zones = []

        # Posiciones iniciales de los drones (suelo, z=0)
        xy = rng.integers(0, g, size=(self.num_drones, 2))
        self.drone_positions = np.column_stack(
            [xy, np.zeros(self.num_drones, dtype=np.int32)]
        )
        self.drone_batteries = np.full(self.num_drones, 100.0)
        self.drone_cargos    = [None] * self.num_drones
        self.drone_alive     = np.ones(self.num_drones, dtype=bool)

        # Paquetes: 2 médicos + 8 estándar (experimental_protocol §2.1)
        # Los DESTINOS se sesgan hacia las zonas de alta demanda predichas por
        # el modelo ML; los orígenes (recogida) permanecen uniformes.
        self.package_positions    = rng.integers(0, g, size=(self.num_packages, 2))
        self.package_destinations = self._spawn_package_destinations(rng)
        self.package_delivered    = np.zeros(self.num_packages, dtype=bool)
        self.package_picked       = np.zeros(self.num_packages, dtype=bool)
        n_medical = 2  # exactamente 2 médicos según el protocolo
        self.package_types = (
            ["medical"] * n_medical
            + ["standard"] * (self.num_packages - n_medical)
        )

        # Infraestructura
        self.charging_stations = rng.integers(5, g - 5, size=(self.num_charging_stations, 2))
        self.charging_occupied = np.zeros(self.num_charging_stations, dtype=bool)
        self.no_fly_zones      = self._spawn_no_fly_zones(rng)
        self.climate_zones     = {}
        self._nfz_set          = set(self.no_fly_zones)
        self._storm_cells      = set()
        self._nfz_list_cache   = None

        if self._bridge is not None:
            self._bridge.reset_intervention_count()

        self._prev_target_dist = np.full(self.num_drones, -1.0, dtype=np.float32)
        return self._get_obs(), {}

    def step(
        self,
        actions: np.ndarray,
    ) -> Tuple[Dict, np.ndarray, np.ndarray, np.ndarray, Dict]:
        """Avanza un step para todos los drones.

        Si hay un bridge simbólico inyectado, aplica masking automáticamente.
        """
        rewards   = np.zeros(self.num_drones, dtype=np.float32)
        dones     = ~self.drone_alive.copy()
        infos: Dict[str, Any] = {f"drone_{i}": {} for i in range(self.num_drones)}

        # Aplicar masking simbólico si hay bridge inyectado
        if self._bridge is not None:
            actions, _, self._last_interventions = self._apply_symbolic_masking(
                self._bridge, actions
            )

        for i in range(self.num_drones):
            if not self.drone_alive[i]:
                continue
            action = ACTIONS[int(actions[i])]
            r, done, info = self._apply_action(i, action)
            rewards[i] = r
            if done:
                dones[i]           = True
                self.drone_alive[i] = False
            infos[f"drone_{i}"] = info

        self.current_step += 1
        truncated = np.full(self.num_drones, self.current_step >= self.max_steps)

        # Terminación anticipada cuando todos los paquetes fueron entregados
        if self.package_delivered.all():
            truncated = np.ones(self.num_drones, dtype=bool)

        # Resetear ocupación de estaciones al final del step para que
        # fast_action_mask del siguiente step vea las estaciones libres.
        # La ocupación DENTRO del step evita que dos drones carguen simultáneamente.
        self.charging_occupied[:] = False

        return self._get_obs(), rewards, dones, truncated, infos

    # ──────────────────────────────────────────────────────────────────────────
    # Enmascaramiento simbólico
    # ──────────────────────────────────────────────────────────────────────────

    def _apply_symbolic_masking(
        self,
        bridge: Any,
        actions: np.ndarray,
    ) -> Tuple[np.ndarray, List[np.ndarray], int]:
        """Aplica máscaras Prolog a las acciones propuestas de todos los drones.

        Args:
            bridge : NeuroSymbolicBridge activo.
            actions: Array de índices de acciones propuestos (shape: num_drones).

        Returns:
            (safe_actions, masks, intervention_count)
            safe_actions      : acciones post-masking
            masks             : lista de máscaras por dron (para logging)
            intervention_count: drones donde la máscara cambió la acción
        """
        safe_actions = actions.copy()
        masks: List[np.ndarray] = []
        interventions = 0

        for i in range(self.num_drones):
            if not self.drone_alive[i]:
                masks.append(np.ones(N_ACTIONS, dtype=np.float32))
                continue

            state_dict = self.get_state_dict(i)
            mask       = bridge.get_action_mask(f"drone_{i}", state_dict)
            masks.append(mask)

            original = int(safe_actions[i])
            if float(mask[original]) == 0.0:
                valid = np.where(mask == 1.0)[0]
                safe_actions[i] = int(self.np_random.choice(valid)) if len(valid) > 0 else original
                interventions  += 1

        return safe_actions, masks, interventions

    # ──────────────────────────────────────────────────────────────────────────
    # Cálculo de recompensas (fórmula formal_modeling.md §5)
    # ──────────────────────────────────────────────────────────────────────────

    def _calculate_rewards(
        self,
        action:          str,
        battery_after:   float,
        delivered:       bool,
        pkg_type:        Optional[str],
        picked_up:       bool,
        collision_count: int,
        nfz_violation:   bool,
        dist_delta:      float = 0.0,
        charge_delta:    float = 0.0,
    ) -> float:
        """Implementa R_total = R_entrega + R_eficiencia − C_movimiento − P_colisión − P_batería − P_simbólica.

        NOTA: P_simbólica (R4, R8, R9, R10, R11) se añade en el training loop
              mediante NeuroSymbolicBridge.get_reward_modifier().

        Args:
            action         : Nombre de la acción ejecutada.
            battery_after  : Nivel de batería resultante.
            delivered      : True si se completó una entrega.
            pkg_type       : 'medical' | 'standard' | None.
            picked_up      : True si se recogió un paquete.
            collision_count: Número de colisiones detectadas.
            nfz_violation  : True si el movimiento entró en una NFZ.
            dist_delta     : dist_antes − dist_después al objetivo (positivo = se acercó).
            charge_delta   : dist_antes − dist_después a la estación de carga más cercana.

        Returns:
            Recompensa escalar para este dron en este step.
        """
        reward = 0.0

        # ── R_entrega: recompensa terminal grande (señal de objetivo) ───────────
        #   +200 médico, +100 estándar, +10 recogida.
        #   Escalada respecto a la versión previa para que una entrega supere con
        #   margen el grind acumulado de un episodio (≈ −50 con el shaping nuevo),
        #   garantizando que entregar sea netamente positivo.
        if delivered and pkg_type is not None:
            reward += 200.0 if pkg_type == "medical" else 100.0
        if picked_up:
            reward += 10.0

        # ── C_movimiento: grind mínimo por step (incentiva rapidez sin dominar) ─
        reward -= 0.05

        # ── Proximity shaping hacia el objetivo activo (paquete o destino) ──────
        #   Shaping basado en potencial (Ng et al. 1999): la recompensa por
        #   acercarse y la penalización por alejarse son SIMÉTRICAS (±0.5·Δdist),
        #   de modo que un dron que deambula sin progresar neto recibe ≈0 — no
        #   acumula deriva negativa por explorar. Solo el progreso real hacia el
        #   objetivo (y las entregas) mueve la recompensa. Esto preserva la
        #   política óptima y da gradiente denso para navegar.
        reward += 0.5 * dist_delta

        # ── Charging proximity shaping cuando la batería baja ───────────────────
        #   Igual que arriba pero hacia la estación de carga, solo activo con
        #   batería <35%: guía al dron a recargar antes de morir (la penalización
        #   por muerte llega demasiado tarde para enseñar por sí sola).
        if battery_after < 35.0:
            reward += 0.4 * charge_delta

        # ── P_colisión: penalización moderada (la máscara R3 ya las previene) ───
        reward -= 30.0 * collision_count

        # ── P_simbólica (parcial): violación de NFZ gestionada localmente ───────
        if nfz_violation:
            reward -= 15.0

        # ── P_batería: penalización proporcionada, no dominante ─────────────────
        #   Muerte por batería: −80 (antes −200; con el costo de batería previo el
        #   agente moría casi siempre y esta penalización aplastaba toda la señal).
        if battery_after <= 0.0:
            reward -= 80.0
        elif battery_after < 15.0:
            reward -= 5.0

        return reward

    # ──────────────────────────────────────────────────────────────────────────
    # Ejecución de acción individual
    # ──────────────────────────────────────────────────────────────────────────

    def _apply_action(
        self,
        drone_idx: int,
        action:    str,
    ) -> Tuple[float, bool, Dict[str, Any]]:
        """Aplica una acción para un dron y retorna (reward, done, info)."""
        x, y, z  = self.drone_positions[drone_idx]
        battery  = self.drone_batteries[drone_idx]
        g        = self.grid_size - 1
        done     = False
        info: Dict[str, Any] = {
            "action": action, "collisions": 0,
            "rule_violations": 0, "delivery": False, "pickup": False,
        }

        # Distancias previas: objetivo de paquete y estación de carga más cercana
        target_before = self._current_target(drone_idx)
        dist_before = (
            abs(target_before[0] - x) + abs(target_before[1] - y)
            if target_before is not None else 0
        )
        charge_near = self._nearest_charging_station(int(x), int(y))
        charge_dist_before = (
            abs(charge_near[0] - x) + abs(charge_near[1] - y)
            if charge_near is not None else 0
        )

        # ── Calcular nueva posición ──────────────────────────────────────────
        nx, ny, nz = x, y, z
        if   action == "mover_n":   ny = min(y + 1, g)
        elif action == "mover_s":   ny = max(y - 1, 0)
        elif action == "mover_e":   nx = min(x + 1, g)
        elif action == "mover_o":   nx = max(x - 1, 0)
        elif action == "despegar":  nz = min(z + 1, 10)
        elif action == "aterrizar": nz = max(z - 1, 0)

        # ── No-fly zone ──────────────────────────────────────────────────────
        nfz_violation = False
        if (nx, ny) in self._nfz_set:
            nfz_violation = True
            info["rule_violations"] += 1
            nx, ny = x, y   # revertir movimiento

        # ── Colisiones con otros drones ──────────────────────────────────────
        collision_count = 0
        for j in range(self.num_drones):
            if j != drone_idx and self.drone_alive[j]:
                jp = self.drone_positions[j]
                if jp[0] == nx and jp[1] == ny and jp[2] == nz:
                    collision_count     += 1
                    info["collisions"]  += 1

        # ── Batería ──────────────────────────────────────────────────────────
        battery -= BATTERY_COST.get(action, 0.0)
        if action == "cargar":
            cs_idx = self._charging_station_at(nx, ny)
            if cs_idx is not None and not self.charging_occupied[cs_idx]:
                battery = min(100.0, battery + BATTERY_RECHARGE)
                self.charging_occupied[cs_idx] = True
            else:
                # Intento fallido de carga (estación ocupada o no existe)
                info["rule_violations"] += 1

        if battery <= 0.0:
            battery = 0.0
            done    = True

        # ── Recogida de paquete ──────────────────────────────────────────────
        picked_up = False
        if self.drone_cargos[drone_idx] is None:
            for pi in range(self.num_packages):
                if (
                    not self.package_delivered[pi]
                    and not self.package_picked[pi]
                    and int(self.package_positions[pi, 0]) == nx
                    and int(self.package_positions[pi, 1]) == ny
                ):
                    self.drone_cargos[drone_idx] = pi
                    self.package_picked[pi]      = True
                    picked_up                    = True
                    info["pickup"]               = True
                    break

        # ── Entrega ──────────────────────────────────────────────────────────
        delivered  = False
        pkg_type   = None
        pi = self.drone_cargos[drone_idx]
        if pi is not None:
            dest = self.package_destinations[pi]
            if nx == int(dest[0]) and ny == int(dest[1]):
                pkg_type                      = self.package_types[pi]
                delivered                     = True
                self.package_delivered[pi]    = True
                self.drone_cargos[drone_idx]  = None
                info["delivery"]              = True
                info["pkg_type"]              = pkg_type

        # ── Aplicar cambios de estado ────────────────────────────────────────
        self.drone_positions[drone_idx] = [nx, ny, nz]
        self.drone_batteries[drone_idx] = battery

        # ── Proximity shaping: comparar distancias antes/después ─────────────
        # Objetivo de paquete: solo aplicar si no cambió en este step
        # (pickup/delivery cambia el target y daría señal espuria).
        target_after = self._current_target(drone_idx)
        dist_delta = 0.0
        if (
            target_before is not None and target_after is not None
            and target_before == target_after
        ):
            dist_after = abs(target_after[0] - nx) + abs(target_after[1] - ny)
            dist_delta = float(dist_before - dist_after)

        # Charging proximity: distancia a estación de carga antes/después
        charge_dist_after = (
            abs(charge_near[0] - nx) + abs(charge_near[1] - ny)
            if charge_near is not None else 0
        )
        charge_delta = float(charge_dist_before - charge_dist_after)

        # ── Calcular recompensa con la fórmula formal ────────────────────────
        reward = self._calculate_rewards(
            action          = action,
            battery_after   = battery,
            delivered       = delivered,
            pkg_type        = pkg_type,
            picked_up       = picked_up,
            collision_count = collision_count,
            nfz_violation   = nfz_violation,
            dist_delta      = dist_delta,
            charge_delta    = charge_delta,
        )

        return reward, done, info

    # ──────────────────────────────────────────────────────────────────────────
    # Dinámica externa e integración
    # ──────────────────────────────────────────────────────────────────────────

    def apply_dynamics(self, dynamics_state: Dict) -> None:
        """Actualiza el entorno con el estado del DynamicsEngine.

        Llamado cada step por el training loop antes de env.step().
        """
        self.climate_zones = dynamics_state.get("storm_regions", {})
        self.current_wind  = dynamics_state.get("wind")

        # Añadir NFZ dinámicas al conjunto existente
        extra_nfz = [tuple(c) for c in dynamics_state.get("dynamic_nfz_cells", [])]
        self._static_nfzs = getattr(self, "_static_nfzs", list(self.no_fly_zones))
        self.no_fly_zones  = list(set(self._static_nfzs) | set(extra_nfz))  # type: ignore[arg-type]

        # Actualizar caches (invalidar también la lista serializada para Prolog)
        self._nfz_set = set(self.no_fly_zones)
        self._nfz_list_cache = None
        storm_cells: set = set()
        for zone in self.climate_zones.values():
            if zone.get("type") == "storm":
                xmin, xmax = zone["x_range"]
                ymin, ymax = zone["y_range"]
                for sx in range(int(xmin), int(xmax) + 1):
                    for sy in range(int(ymin), int(ymax) + 1):
                        storm_cells.add((sx, sy))
        self._storm_cells = storm_cells

    def set_symbolic_bridge(self, bridge: Any) -> None:
        """Inyecta el NeuroSymbolicBridge para masking automático en step()."""
        self._bridge = bridge
        logger.info("Bridge simbólico inyectado en CyberCityEnv")

    # ──────────────────────────────────────────────────────────────────────────
    # Serialización de estado para el bridge Prolog
    # ──────────────────────────────────────────────────────────────────────────

    def get_state_dict(self, drone_idx: int) -> Dict:
        """Produce el dict consumido por NeuroSymbolicBridge.sync_state().

        Incluye: posición, batería, carga, destino, NFZ, estaciones,
                 otros agentes, regiones de tormenta y viento dominante.
        """
        x, y, z = self.drone_positions[drone_idx]
        pi = self.drone_cargos[drone_idx]

        return {
            "agent_id":  f"drone_{drone_idx}",
            "position":  (int(x), int(y), int(z)),
            "battery":   float(self.drone_batteries[drone_idx]),
            "cargo":     f"pkg_{pi}" if pi is not None else None,
            "cargo_type": self.package_types[pi] if pi is not None else None,
            "destination": (
                (int(self.package_destinations[pi, 0]),
                 int(self.package_destinations[pi, 1]))
                if pi is not None else None
            ),
            "no_fly_zones": self._get_nfz_list(),
            "charging_stations": {
                f"station_{j}": ("ocupada" if self.charging_occupied[j] else "libre")
                for j in range(self.num_charging_stations)
            },
            "other_agents": {
                f"drone_{j}": (int(self.drone_positions[j, 0]),
                               int(self.drone_positions[j, 1]))
                for j in range(self.num_drones)
                if j != drone_idx and self.drone_alive[j]
            },
            "other_batteries": {
                f"drone_{j}": float(self.drone_batteries[j])
                for j in range(self.num_drones)
                if j != drone_idx
            },
            "storm_regions": {
                name: (
                    data["x_range"][0], data["x_range"][1],
                    data["y_range"][0], data["y_range"][1],
                )
                for name, data in self.climate_zones.items()
                if data.get("type") == "storm"
            },
            "wind": self.current_wind,  # (direction, intensity_kmh) o None
        }

    @property
    def demand_zones(self) -> List[Tuple[int, int]]:
        """Zonas de alta demanda (predicción ML) activas en el episodio actual."""
        return list(self._demand_zones)

    def get_blocked_cells(self) -> List[Tuple[int, int]]:
        """Retorna todas las celdas actualmente bloqueadas (NFZ + tormentas).

        Usado por AStarAgent.set_obstacles() para replanning con obstáculos dinámicos.
        Usa los caches precalculados en lugar de recalcular desde climate_zones.
        """
        return list(self._nfz_set | self._storm_cells)

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers internos
    # ──────────────────────────────────────────────────────────────────────────

    def _get_obs(self) -> Dict[str, np.ndarray]:
        obs = {}
        g = float(self.grid_size)
        for i in range(self.num_drones):
            x, y, z = self.drone_positions[i]
            target = self._current_target(i)
            charge = self._nearest_charging_station(int(x), int(y))
            tdx = (target[0] - x) / g if target is not None else 0.0
            tdy = (target[1] - y) / g if target is not None else 0.0
            cdx = (charge[0] - x) / g if charge is not None else 0.0
            cdy = (charge[1] - y) / g if charge is not None else 0.0
            obs[f"drone_{i}"] = np.array(
                [x / g, y / g, z / 10.0,
                 self.drone_batteries[i] / 100.0,
                 1.0 if self.drone_cargos[i] is not None else 0.0,
                 self._climate_at(x, y) / 2.0,
                 float(self._neighbor_count(i)) / self.num_drones,
                 tdx, tdy, cdx, cdy],
                dtype=np.float32,
            )
        return obs

    def _current_target(self, drone_idx: int) -> Optional[Tuple[int, int]]:
        """Objetivo activo: destino si lleva carga; paquete exclusivo asignado si no.

        Asignación por prioridad de índice: un paquete se asigna al dron sin-carga
        más cercano. Si otro dron con índice menor está a igual o menor distancia,
        este dron lo salta. Un fallback garantiza que ningún dron quede sin objetivo.
        Esto evita que los 5 drones persigan el mismo paquete y se bloqueen entre sí.
        """
        pi = self.drone_cargos[drone_idx]
        if pi is not None:
            return (int(self.package_destinations[pi, 0]), int(self.package_destinations[pi, 1]))

        x, y, _ = self.drone_positions[drone_idx]

        # Drones sin carga con índice menor (tienen prioridad de asignación)
        higher_priority = [
            (int(self.drone_positions[j, 0]), int(self.drone_positions[j, 1]))
            for j in range(drone_idx)
            if self.drone_alive[j] and self.drone_cargos[j] is None
        ]

        best, best_d = None, float("inf")
        for i in range(self.num_packages):
            if self.package_delivered[i] or self.package_picked[i]:
                continue
            px, py = int(self.package_positions[i, 0]), int(self.package_positions[i, 1])
            d = abs(px - x) + abs(py - y)
            # Saltar si un dron de mayor prioridad está igual o más cerca a este paquete
            if any(abs(px - jx) + abs(py - jy) <= d for jx, jy in higher_priority):
                continue
            if d < best_d:
                best, best_d = (px, py), d

        # Fallback: si todos los paquetes disponibles están "reclamados", tomar el más cercano
        if best is None:
            for i in range(self.num_packages):
                if self.package_delivered[i] or self.package_picked[i]:
                    continue
                px, py = int(self.package_positions[i, 0]), int(self.package_positions[i, 1])
                d = abs(px - x) + abs(py - y)
                if d < best_d:
                    best, best_d = (px, py), d

        return best

    def _nearest_charging_station(self, x: int, y: int) -> Optional[Tuple[int, int]]:
        if self.num_charging_stations == 0:
            return None
        deltas = self.charging_stations - np.array([x, y])
        dists = np.abs(deltas).sum(axis=1)
        idx = int(np.argmin(dists))
        return (int(self.charging_stations[idx, 0]), int(self.charging_stations[idx, 1]))

    def _climate_at(self, x: int, y: int) -> float:
        for zone in self.climate_zones.values():
            xmin, xmax = zone["x_range"]
            ymin, ymax = zone["y_range"]
            if xmin <= x <= xmax and ymin <= y <= ymax:
                return float(CLIMATE_ENC.get(zone.get("type", "clear"), 0))
        return 0.0

    def _neighbor_count(self, drone_idx: int) -> int:
        x = int(self.drone_positions[drone_idx, 0])
        y = int(self.drone_positions[drone_idx, 1])
        count = 0
        for j in range(self.num_drones):
            if j != drone_idx and self.drone_alive[j]:
                dx = int(self.drone_positions[j, 0]) - x
                dy = int(self.drone_positions[j, 1]) - y
                if dx * dx + dy * dy <= 4:  # Euclidean ≤ 2, squared — avoids sqrt
                    count += 1
        return count

    def _charging_station_at(self, x: int, y: int) -> Optional[int]:
        for i, cs in enumerate(self.charging_stations):
            if int(cs[0]) == x and int(cs[1]) == y:
                return i
        return None

    def _get_nfz_list(self) -> List[Tuple[int, int]]:
        """NFZ list cached entre llamadas Prolog del mismo step."""
        if self._nfz_list_cache is None:
            self._nfz_list_cache = [(int(nx), int(ny)) for nx, ny in self.no_fly_zones]
        return self._nfz_list_cache

    def _spawn_package_destinations(
        self,
        rng: np.random.Generator,
        demand_fraction: float = 0.6,
        jitter: int = 3,
    ) -> np.ndarray:
        """Genera los destinos de entrega sesgados hacia zonas de alta demanda.

        Si el modelo ML proveyó `self._demand_zones`, una fracción
        `demand_fraction` de los paquetes se entrega cerca (±jitter) de una
        zona de alta demanda elegida al azar; el resto se distribuye de forma
        uniforme. Sin zonas predichas, el comportamiento es uniforme (idéntico
        a la versión previa: spawn aleatorio en todo el grid).

        Returns:
            np.ndarray de shape (num_packages, 2) con coordenadas (x, y).
        """
        g = self.grid_size
        if not self._demand_zones:
            return rng.integers(0, g, size=(self.num_packages, 2))

        dests = np.zeros((self.num_packages, 2), dtype=np.int64)
        n_demand = int(round(self.num_packages * demand_fraction))
        for i in range(self.num_packages):
            if i < n_demand:
                zx, zy = self._demand_zones[int(rng.integers(0, len(self._demand_zones)))]
                x = int(np.clip(zx + rng.integers(-jitter, jitter + 1), 0, g - 1))
                y = int(np.clip(zy + rng.integers(-jitter, jitter + 1), 0, g - 1))
                dests[i] = (x, y)
            else:
                dests[i] = rng.integers(0, g, size=2)
        return dests

    def _spawn_no_fly_zones(
        self,
        rng: np.random.Generator,
        n_zones: int = 5,
        radius:  int = 2,
    ) -> List[Tuple[int, int]]:
        cells: List[Tuple[int, int]] = []
        g = self.grid_size
        for _ in range(n_zones):
            cx = int(rng.integers(radius + 1, g - radius - 1))
            cy = int(rng.integers(radius + 1, g - radius - 1))
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < g and 0 <= ny < g:
                        cells.append((nx, ny))
        return cells

    # ──────────────────────────────────────────────────────────────────────────
    # Action masking vectorizado (sin Prolog) — reglas R1, R3, R5, R7
    # ──────────────────────────────────────────────────────────────────────────

    def fast_action_mask(self, drone_idx: int) -> np.ndarray:
        """Calcula la máscara de acciones en Python puro (~100x más rápido que Prolog).

        Reglas evaluadas localmente:
            R1 (NFZ), R3 (colisión), R5 (estación ocupada), R7 (tormenta).
            R2 (batería crítica): permite aterrizar/cargar/esperar.

        Las reglas que requieren razonamiento simbólico complejo (R11 negociación,
        R12 predicción ML) deben evaluarse con el bridge cuando esté disponible.
        """
        mask = np.ones(N_ACTIONS, dtype=np.float32)
        if not self.drone_alive[drone_idx]:
            return mask

        x, y, _ = self.drone_positions[drone_idx]
        battery = float(self.drone_batteries[drone_idx])

        # Usar caches precalculados (actualizados en apply_dynamics/reset)
        nfz_set     = self._nfz_set
        storm_cells = self._storm_cells
        other_cells = {
            (int(self.drone_positions[j, 0]), int(self.drone_positions[j, 1]))
            for j in range(self.num_drones)
            if j != drone_idx and self.drone_alive[j]
        }

        g = self.grid_size - 1
        # Mapeo acción → desplazamiento XY
        deltas: Dict[str, Tuple[int, int]] = {
            "mover_n": (0,  1), "mover_s": (0, -1),
            "mover_e": (1,  0), "mover_o": (-1, 0),
        }
        for i, name in enumerate(ACTIONS):
            if name in deltas:
                dx, dy = deltas[name]
                nx = max(0, min(g, x + dx))
                ny = max(0, min(g, y + dy))
                # R1 — NFZ
                if (nx, ny) in nfz_set:
                    mask[i] = 0.0
                    continue
                # R7 — Tormenta
                if (nx, ny) in storm_cells:
                    mask[i] = 0.0
                    continue
                # R3 — Colisión
                if (nx, ny) in other_cells:
                    mask[i] = 0.0
                    continue
            elif name == "cargar":
                # R5 — Estación ocupada o no presente en celda actual
                cs_idx = self._charging_station_at(int(x), int(y))
                if cs_idx is None or self.charging_occupied[cs_idx]:
                    mask[i] = 0.0

        # R2 — Batería crítica (<15%): permitir también movimiento hacia estación de carga.
        # Sin este fix el dron queda atrapado: cargar requiere ESTAR en la estación,
        # pero los movimientos están bloqueados, así que el dron solo puede esperar
        # hasta que la batería llega a 0 (penalidad -500).
        # IMPORTANTE: "cargar" solo se permite si el dron está físicamente en una estación
        # (R5 lo valida arriba), nunca se fuerza aquí para evitar violaciones por cargar fuera.
        if battery < 15.0:
            cs_at_pos = self._charging_station_at(int(x), int(y))
            allowed = {"aterrizar", "esperar"}
            if cs_at_pos is not None and not self.charging_occupied[cs_at_pos]:
                allowed.add("cargar")
            charge_pos = self._nearest_charging_station(int(x), int(y))
            if charge_pos is not None:
                cx, cy = charge_pos
                if cx > x: allowed.add("mover_e")
                elif cx < x: allowed.add("mover_o")
                if cy > y: allowed.add("mover_n")
                elif cy < y: allowed.add("mover_s")
            for i, name in enumerate(ACTIONS):
                if name not in allowed:
                    mask[i] = 0.0

        # Fail-safe: nunca devolver máscara todo-cero
        if mask.sum() == 0.0:
            mask = np.zeros(N_ACTIONS, dtype=np.float32)
            mask[ACTIONS.index("esperar")] = 1.0

        return mask

    # ──────────────────────────────────────────────────────────────────────────
    # Render
    # ──────────────────────────────────────────────────────────────────────────

    def render(self) -> Optional[np.ndarray]:
        if self.render_mode == "human":
            self._render_terminal()
            return None
        if self.render_mode == "rgb_array":
            return self._render_rgb()
        return None

    def _render_terminal(self) -> None:
        g    = self.grid_size
        grid = [["·"] * g for _ in range(g)]
        for nx, ny in self.no_fly_zones:
            if 0 <= nx < g and 0 <= ny < g:
                grid[ny][nx] = "X"
        for i, pos in enumerate(self.drone_positions):
            if self.drone_alive[i]:
                grid[pos[1]][pos[0]] = str(i)
        print("\n".join(" ".join(row) for row in reversed(grid)))
        print(f"  Step {self.current_step} | Baterías: {self.drone_batteries.round(1)}")
        print("-" * (g * 2))

    def _render_rgb(self) -> np.ndarray:
        img = np.full((self.grid_size, self.grid_size, 3), 200, dtype=np.uint8)
        for nx, ny in self.no_fly_zones:
            img[ny, nx] = [200, 30, 30]
        for i, pos in enumerate(self.drone_positions):
            if self.drone_alive[i]:
                img[pos[1], pos[0]] = [30, 120, 255]
        return img
