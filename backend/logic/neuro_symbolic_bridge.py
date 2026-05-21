"""
neuro_symbolic_bridge.py — Puente bidireccional Python ↔ SWI-Prolog.

Arquitectura de influencia (logic_rules.md):
    Estado (Python) → sync_state() → hechos Prolog
    DQN propone acción → validate_action() → (is_valid, penalty)
    DQN ejecuta acción → get_reward_modifier() → reward shaped

API pública principal:
    sync_state(env_state)                       → None
    validate_action(agent_id, action, state)    → (bool, float)
    get_action_mask(agent_id, state)            → np.ndarray[8]
    get_reward_modifier(agent_id, state, action)→ float
    negotiate_passage(a1, a2, state)            → int  (0=a1, 1=a2)
    log_decision(level, message)                → None
"""

import logging
import os
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constantes de espacio de acciones
# ─────────────────────────────────────────────────────────────────────────────

ACTIONS: List[str] = [
    "despegar", "aterrizar",
    "mover_n",  "mover_s", "mover_e", "mover_o",
    "esperar",  "cargar",
]
N_ACTIONS = len(ACTIONS)

# ─────────────────────────────────────────────────────────────────────────────
# Pesos de reglas (formal_modeling.md Tabla §5.3)
# ─────────────────────────────────────────────────────────────────────────────

# Action-Masking rules — bloquean ANTES de ejecutar
_AM_PENALTIES: Dict[str, float] = {
    "R1_nfz":              -100.0,   # Zona prohibida
    "R2_battery":           -50.0,   # Batería crítica por paso
    "R3_collision":        -200.0,   # Colisión inminente
    "R5_station":           -20.0,   # Estación ocupada
    "R7_storm":             -80.0,   # Vuelo en tormenta
    "R12_battery_route":    -50.0,   # Riesgo batería en ruta (escala; full -500 en env)
}

# Reward-Shaping rules — modifican DESPUÉS del paso
_RS_VALUES: Dict[str, float] = {
    "R4_cell_conflict":    -30.0,    # por drone extra (×(N-1))
    "R6_medical":          150.0,    # entrega médica
    "R6_standard":          50.0,    # entrega estándar
    "R8_wind":             -15.0,    # movimiento contra viento
    "R9_congested":        -40.0,    # zona con ≥3 drones
    "R10_route":            20.0,    # bonus ruta eficiente
    "R11_yield":            25.0,    # agente que cede paso
    "R11_interrupt":       -10.0,    # agente que interrumpe
}


# ─────────────────────────────────────────────────────────────────────────────
# Bridge principal
# ─────────────────────────────────────────────────────────────────────────────

class NeuroSymbolicBridge:
    """Interfaz Python ↔ SWI-Prolog para masking y reward shaping simbólico.

    Inicialización:
        bridge = NeuroSymbolicBridge("backend/logic/rules.pl")

    Dependencia externa: SWI-Prolog debe estar instalado en el sistema.
        Windows : winget install SWI-Prolog.SWI-Prolog
        Linux   : apt install swi-prolog
        macOS   : brew install swi-prolog
    """

    def __init__(self, rules_file: str) -> None:
        """Carga el motor Prolog y consulta el archivo de reglas.

        Args:
            rules_file: Ruta al archivo rules.pl con las 12 reglas.

        Raises:
            FileNotFoundError: Si rules_file no existe.
            ImportError: Si pyswip no está instalado.
            RuntimeError: Si SWI-Prolog falla al inicializar.
        """
        if not os.path.exists(rules_file):
            raise FileNotFoundError(
                f"Archivo de reglas Prolog no encontrado: {rules_file}\n"
                "Asegúrate de que rules.pl existe en backend/logic/"
            )
        self._rules_file      = rules_file
        self._log: List[str]  = []
        self._intervention_count: int = 0

        try:
            from pyswip import Prolog as _Prolog  # type: ignore[import-untyped]  # runtime dep
            self.prolog = _Prolog()
            self._load_rules()
            logger.info("Motor Prolog inicializado desde %s", rules_file)
        except ImportError as exc:
            raise ImportError(
                "pyswip no está instalado. Instala SWI-Prolog y luego: pip install pyswip"
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"Error al inicializar el motor Prolog: {exc}"
            ) from exc

    # ──────────────────────────────────────────────────────────────────────────
    # Gestión del motor
    # ──────────────────────────────────────────────────────────────────────────

    def _load_rules(self) -> None:
        safe = self._rules_file.replace("\\", "/")
        self.prolog.consult(safe)
        logger.debug("Reglas Prolog cargadas desde %s", safe)

    def _query(self, goal: str) -> List[Dict]:
        """Ejecuta una consulta Prolog y retorna la lista de soluciones."""
        try:
            return list(self.prolog.query(goal))
        except Exception as exc:
            logger.warning("Consulta Prolog falló [%.70s...]: %s", goal, exc)
            return []

    def _assert(self, fact: str) -> None:
        self._query(f"assertz({fact})")

    def _retract_all(self, pattern: str) -> None:
        self._query(f"retractall({pattern})")

    # ──────────────────────────────────────────────────────────────────────────
    # Sincronización de estado (PÚBLICA — spec-compliant)
    # ──────────────────────────────────────────────────────────────────────────

    def sync_state(self, env_state: Dict) -> None:
        """Limpia todos los hechos dinámicos y los reaserta desde el estado Python.

        Llamado automáticamente por validate_action, get_action_mask y
        get_reward_modifier. También disponible para uso externo.

        Args:
            env_state: Dict producido por CyberCityEnv.get_state_dict(drone_idx).
                       Campos: agent_id, position, battery, cargo, cargo_type,
                               destination, no_fly_zones, charging_stations,
                               other_agents, other_batteries, storm_regions, wind.
        """
        # Limpiar todos los hechos dinámicos
        for pattern in (
            "agente_en_celda(_, _)",
            "bateria_agente(_, _)",
            "tormenta(_)",
            "viento(_, _)",
            "zona_congestionada(_)",
            "agente_lleva_paquete(_, _)",
            "agente_destino(_, _)",
            "estacion_carga(_, _)",
            "no_fly_zone(_)",
            "region_limites(_, _, _, _, _)",
            "paquete(_, _, _)",
        ):
            self._retract_all(pattern)

        # Agente principal
        agent_id = env_state["agent_id"]
        x, y, _  = env_state["position"]
        battery  = max(0, int(env_state["battery"]))
        self._assert(f"agente_en_celda({agent_id}, celda({x},{y}))")
        self._assert(f"bateria_agente({agent_id}, {battery})")

        # Carga del dron
        if env_state.get("cargo"):
            pkg      = env_state["cargo"]
            pkg_type = env_state.get("cargo_type", "standard")
            self._assert(f"agente_lleva_paquete({agent_id}, {pkg})")
            self._assert(f"paquete({pkg}, {pkg_type}, normal)")

        # Destino actual
        if env_state.get("destination"):
            dx, dy = env_state["destination"]
            self._assert(f"agente_destino({agent_id}, celda({dx},{dy}))")

        # No-fly zones (estáticas + dinámicas)
        for nfz in env_state.get("no_fly_zones", []):
            self._assert(f"no_fly_zone(celda({int(nfz[0])},{int(nfz[1])}))")

        # Regiones de tormenta → region_limites + tormenta
        for region_id, limits in env_state.get("storm_regions", {}).items():
            xmin, xmax, ymin, ymax = (int(v) for v in limits)
            self._assert(f"region_limites({region_id},{xmin},{xmax},{ymin},{ymax})")
            self._assert(f"tormenta({region_id})")

        # Viento dominante
        if env_state.get("wind"):
            direction, intensity = env_state["wind"]
            self._assert(f"viento({direction},{int(intensity)})")

        # Estaciones de carga
        for sid, status in env_state.get("charging_stations", {}).items():
            self._assert(f"estacion_carga({sid},{status})")

        # Otros agentes (posición + batería)
        for other_id, pos in env_state.get("other_agents", {}).items():
            ox, oy = int(pos[0]), int(pos[1])
            self._assert(f"agente_en_celda({other_id}, celda({ox},{oy}))")
        for other_id, bat in env_state.get("other_batteries", {}).items():
            self._assert(f"bateria_agente({other_id}, {max(0, int(bat))})")

    # ──────────────────────────────────────────────────────────────────────────
    # Validación de acciones (Action Masking — reglas R1, R2, R3, R5, R7, R12)
    # ──────────────────────────────────────────────────────────────────────────

    def validate_action(
        self,
        agent_id: str,
        action:   str,
        state:    Dict,
    ) -> Tuple[bool, float]:
        """Valida una acción propuesta contra las reglas de Action Masking.

        Reglas evaluadas: R1 (NFZ), R2 (batería), R3 (colisión),
                          R5 (estación), R7 (tormenta), R12 (ruta batería).

        Args:
            agent_id: ID del dron (e.g. 'drone_0').
            action  : Nombre de la acción (e.g. 'mover_n').
            state   : Dict de CyberCityEnv.get_state_dict().

        Returns:
            (is_valid, cumulative_penalty)
            is_valid=False → la acción debe ser enmascarada en select_action().
        """
        self.sync_state(state)
        return self._check_action_rules(agent_id, action)

    def _check_action_rules(
        self,
        agent_id: str,
        action:   str,
    ) -> Tuple[bool, float]:
        """Evalúa las reglas AM sin re-sincronizar el estado (uso interno)."""
        is_valid = True
        penalty  = 0.0

        # R1 — Zona de vuelo prohibida
        if self._query(f"accion_invalida({agent_id}, {action})"):
            is_valid = False
            penalty += _AM_PENALTIES["R1_nfz"]
            self.log_decision("MASK", f"R1 NFZ: {agent_id}→{action} bloqueado")
            self._intervention_count += 1

        # R2 — Batería crítica: solo cargar/aterrizar
        if self._query(f"solo_acciones_emergencia({agent_id})"):
            if action not in ("cargar", "aterrizar"):
                is_valid = False
                penalty += _AM_PENALTIES["R2_battery"]
                self.log_decision("ALERT", f"R2 Batería crítica: {agent_id}→{action} bloqueado")
                self._intervention_count += 1

        # R3 — Colisión inminente
        if self._query(f"accion_causa_colision({agent_id}, {action})"):
            is_valid = False
            penalty += _AM_PENALTIES["R3_collision"]
            self.log_decision("CRITICAL", f"R3 Colisión: {agent_id}→{action} bloqueado")
            self._intervention_count += 1

        # R5 — Estación de carga ocupada
        if action == "cargar" and self._query(f"accion_carga_invalida({agent_id})"):
            is_valid = False
            penalty += _AM_PENALTIES["R5_station"]
            self.log_decision("MASK", f"R5 Estación ocupada: {agent_id}")
            self._intervention_count += 1

        # R7 — Tormenta activa en celda destino
        if self._query(f"accion_invalida_tormenta({agent_id}, {action})"):
            is_valid = False
            penalty += _AM_PENALTIES["R7_storm"]
            self.log_decision("MASK", f"R7 Tormenta: {agent_id}→{action} bloqueado")
            self._intervention_count += 1

        # R12 — Predicción de batería insuficiente para la ruta
        if self._query(f"accion_riesgo_bateria({agent_id}, {action})"):
            is_valid = False
            penalty += _AM_PENALTIES["R12_battery_route"]
            self.log_decision("CRITICAL", f"R12 Riesgo batería: {agent_id}→{action} bloqueado")
            self._intervention_count += 1

        return is_valid, penalty

    # ──────────────────────────────────────────────────────────────────────────
    # Action mask vectorial
    # ──────────────────────────────────────────────────────────────────────────

    def get_action_mask(
        self,
        agent_id: str,
        state:    Dict,
    ) -> np.ndarray:
        """Retorna máscara binaria float32 de shape (8,): 0=prohibida, 1=válida.

        Optimización: sync_state() se llama UNA VEZ para las 8 consultas.
        Fail-safe: nunca retorna una máscara todo-cero.

        Args:
            agent_id: ID del dron.
            state   : Dict de CyberCityEnv.get_state_dict().

        Returns:
            np.ndarray(8,) con valores 0.0 o 1.0.
        """
        self.sync_state(state)  # sincronizar UNA vez para todas las acciones

        mask = np.ones(N_ACTIONS, dtype=np.float32)
        for i, action in enumerate(ACTIONS):
            is_valid, _ = self._check_action_rules(agent_id, action)
            if not is_valid:
                mask[i] = 0.0

        # Fail-safe: garantizar al menos una acción disponible
        if mask.sum() == 0.0:
            logger.warning(
                "Máscara todo-cero para %s — reseteando a todo-válido (fail-safe)",
                agent_id,
            )
            mask = np.ones(N_ACTIONS, dtype=np.float32)

        return mask

    # ──────────────────────────────────────────────────────────────────────────
    # Reward Shaping (reglas R4, R6, R8, R9, R10, R11)
    # ──────────────────────────────────────────────────────────────────────────

    def get_reward_modifier(
        self,
        agent_id: str,
        state:    Dict,
        action:   str,
    ) -> float:
        """Calcula el modificador de recompensa acumulado post-step.

        Suma de pesos de reglas RS activas: R4 + R6 + R8 + R9 + R10 + R11.
        Llamado por el training loop DESPUÉS de env.step() para reward shaping.

        Args:
            agent_id: ID del dron.
            state   : Dict de get_state_dict() DESPUÉS del paso.
            action  : Acción ejecutada (nombre, e.g. 'mover_n').

        Returns:
            float: modificador acumulado (puede ser positivo o negativo).
        """
        self.sync_state(state)
        x, y, _ = state["position"]
        modifier = 0.0

        # R4 — Conflicto de celda: -30 × (N-1) drones en la misma celda
        results = self._query(f"penalizacion_conflicto(celda({x},{y}), P)")
        for r in results:
            val = float(r.get("P", 0))
            modifier += val
            if val != 0.0:
                self.log_decision("REWARD", f"R4 Conflicto celda ({x},{y}): {val:.0f}")

        # R6 — Bonus por entrega: NO aplicar aquí (lo maneja el env en _calculate_rewards
        # exactamente en el momento de la entrega). Aplicarlo cada step inflaba el reward
        # mientras el dron solo "lleva" carga sin entregar.

        # R8 — Penalización por movimiento contra viento fuerte (>60 km/h)
        results = self._query(f"penalizacion_viento({action}, P)")
        for r in results:
            val = float(r.get("P", 0))
            modifier += val
            if val != 0.0:
                self.log_decision("CLIMA", f"R8 Viento: {action}: {val:.0f}")

        # R9 — Zona congestionada (≥3 drones)
        if self._query(f"desvio_necesario({agent_id}, celda({x},{y}))"):
            modifier += _RS_VALUES["R9_congested"]
            self.log_decision("AVISO", f"R9 Zona congestionada ({x},{y}): {_RS_VALUES['R9_congested']:.0f}")

        # R10 — Bonus por seguir ruta eficiente pre-calculada
        results = self._query(f"bonus_ruta_optima({agent_id}, B)")
        for r in results:
            val = float(r.get("B", 0))
            modifier += val
            if val > 0.0:
                self.log_decision("RUTA", f"R10 Ruta eficiente {agent_id}: +{val:.0f}")

        # R11 — Negociación de paso solo con drones REALMENTE adyacentes (radio 1)
        # Antes consultaba todos los drones vivos cada step → 5 queries × 5 drones × 500 steps.
        ax, ay, _ = state["position"]
        for other_id, other_pos in state.get("other_agents", {}).items():
            ox, oy = int(other_pos[0]), int(other_pos[1])
            if abs(ox - ax) + abs(oy - ay) > 1:
                continue  # solo vecinos inmediatos
            res = self._query(f"prioridad_paso({agent_id}, {other_id}, Ganador)")
            if res:
                winner = str(res[0].get("Ganador", ""))
                if winner == agent_id:
                    modifier += _RS_VALUES["R11_yield"]
                    self.log_decision("NEGOC", f"R11 {agent_id} tiene paso: +{_RS_VALUES['R11_yield']:.0f}")
                else:
                    modifier += _RS_VALUES["R11_interrupt"]
                    self.log_decision("NEGOC", f"R11 {agent_id} debe ceder: {_RS_VALUES['R11_interrupt']:.0f}")

        return modifier

    # ──────────────────────────────────────────────────────────────────────────
    # Negociación de paso (R11)
    # ──────────────────────────────────────────────────────────────────────────

    def negotiate_passage(
        self,
        a1:    str,
        a2:    str,
        state: Dict,
    ) -> int:
        """Determina derecho de paso por jerarquía R11: médico > batería baja > solicitud.

        Args:
            a1, a2 : IDs de los dos drones en conflicto.
            state  : Estado actual (debe contener ambos agentes).

        Returns:
            0 si a1 tiene prioridad, 1 si a2 tiene prioridad.
        """
        self.sync_state(state)
        results = self._query(f"prioridad_paso({a1}, {a2}, Ganador)")
        if results:
            winner = str(results[0].get("Ganador", a1))
            priority = 0 if (winner == a1) else 1
            loser = a2 if priority == 0 else a1
            self.log_decision("NEGOC", f"R11 {winner} tiene paso; {loser} debe ceder")
            return priority
        return 0  # default: primer solicitante tiene prioridad

    # ──────────────────────────────────────────────────────────────────────────
    # Log de decisiones simbólicas (PÚBLICA — spec-compliant)
    # ──────────────────────────────────────────────────────────────────────────

    def log_decision(self, level: str, message: str) -> None:
        """Registra una decisión simbólica en el log interno.

        Args:
            level  : Nivel semántico (MASK | ALERT | CRITICAL | REWARD |
                     PRIO | RUTA | CLIMA | AVISO | NEGOC).
            message: Descripción de la decisión.
        """
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        entry = f"{ts}|{level}|{message}"
        self._log.append(entry)
        # Mantener buffer acotado (1000 entradas → ~60KB)
        if len(self._log) > 1000:
            self._log = self._log[-1000:]

    def get_log_entries(self, last_n: int = 100) -> List[Dict]:
        """Retorna las últimas last_n entradas del log como lista de dicts.

        Format compatible con el componente RuleTerminal del frontend.
        """
        entries = []
        for raw in self._log[-last_n:]:
            parts = raw.split("|", 2)
            if len(parts) == 3:
                ts, level, message = parts
                entries.append({
                    "timestamp": ts,
                    "level":     level,
                    "message":   message,
                })
        return entries

    def clear_log(self) -> None:
        self._log.clear()

    # ──────────────────────────────────────────────────────────────────────────
    # Métricas
    # ──────────────────────────────────────────────────────────────────────────

    @property
    def intervention_count(self) -> int:
        """Total de intervenciones simbólicas (acciones bloqueadas) desde el inicio."""
        return self._intervention_count

    def reset_intervention_count(self) -> None:
        """Reiniciar contador (llamar al inicio de cada episodio)."""
        self._intervention_count = 0
