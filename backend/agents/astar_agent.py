"""
astar_agent.py — Agente Baseline A* con heurística Manhattan.

Sistema A (experimental_protocol.md §3.1):
  - Búsqueda A* con distancia Manhattan al destino.
  - Sin aprendizaje; replanning completo ante cambios de obstáculos.
  - Respeta la máscara simbólica: si la acción planificada está bloqueada,
    replanifica evitando esa celda o espera.
  - Limitación documentada: no considera batería ni colisiones multi-agente.
"""

import heapq
import logging
import os
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)

# Orden canónico — debe coincidir con CyberCityEnv.ACTIONS
ACTIONS: List[str] = [
    "despegar", "aterrizar",
    "mover_n",  "mover_s", "mover_e", "mover_o",
    "esperar",  "cargar",
]
N_ACTIONS = len(ACTIONS)

# Delta (dx, dy) → nombre de acción de movimiento
_DELTA_ACTION: Dict[Tuple[int, int], str] = {
    (0,  1): "mover_n",
    (0, -1): "mover_s",
    (1,  0): "mover_e",
    (-1, 0): "mover_o",
}


class AStarAgent(BaseAgent):
    """Planificador A* para entorno CyberCity.

    Ciclo de operación por step:
      1. set_target(...)    — actualiza objetivo (dispara replanning si cambió)
      2. set_obstacles(...) — actualiza NFZ + storm cells (dispara replanning si cambió)
      3. select_action(state, mask) — retorna siguiente acción del path calculado

    Si la máscara simbólica bloquea la acción planificada, el agente añade
    la celda bloqueada como obstáculo temporal, replanifica y reintenta.
    Si no hay ruta, devuelve 'esperar'.
    """

    def __init__(
        self,
        agent_id:   str,
        state_dim:  int = 11,
        action_dim: int = 8,
        grid_size:  int = 50,
    ) -> None:
        super().__init__(agent_id, state_dim, action_dim)
        self.grid_size = grid_size

        self.target:    Optional[Tuple[int, int]] = None
        self.obstacles: Set[Tuple[int, int]]      = set()
        self._path:     List[Tuple[int, int]]     = []
        self._last_pos: Optional[Tuple[int, int]] = None
        self._replan_count: int = 0

    # ──────────────────────────────────────────────────────────────────────────
    # Configuración runtime (llamar antes de select_action en cada step)
    # ──────────────────────────────────────────────────────────────────────────

    def set_target(self, target: Optional[Tuple[int, int]]) -> None:
        """Establece el destino de navegación. Fuerza replanning si cambió."""
        if target != self.target:
            self.target = target
            self._path = []
            self._logger.debug("Nuevo objetivo: %s", target)

    def set_obstacles(
        self,
        no_fly_zones: List[Tuple[int, int]],
        storm_cells:  Optional[List[Tuple[int, int]]] = None,
    ) -> None:
        """Actualiza el conjunto de obstáculos (NFZ estáticas + dinámicas + tormentas).

        Fuerza replanning si el conjunto cambió respecto al anterior.
        """
        new_obs: Set[Tuple[int, int]] = set(map(tuple, no_fly_zones))  # type: ignore[arg-type]
        if storm_cells:
            new_obs |= set(map(tuple, storm_cells))  # type: ignore[arg-type]
        if new_obs != self.obstacles:
            self.obstacles = new_obs
            self._path = []  # path anterior puede atravesar nuevos obstáculos

    # ──────────────────────────────────────────────────────────────────────────
    # Interfaz BaseAgent
    # ──────────────────────────────────────────────────────────────────────────

    def select_action(
        self,
        state: np.ndarray,
        symbolic_mask: Optional[np.ndarray] = None,
    ) -> int:
        """Retorna el siguiente paso del plan A*.

        Si la acción planificada está enmascarada por Prolog, replanifica
        añadiendo la celda bloqueada como obstáculo temporal.
        """
        x, y = int(state[0]), int(state[1])
        current_pos = (x, y)

        # Objetivo no definido o ya alcanzado
        if self.target is None or self.target == current_pos:
            return self._safe_idx("esperar", symbolic_mask)

        # Consumir el waypoint si el agente llegó a él
        if self._last_pos != current_pos and self._path:
            if self._path[0] == current_pos:
                self._path.pop(0)

        self._last_pos = current_pos

        # Replanificar si el path está vacío
        if not self._path:
            self._path = self._astar(current_pos, self.target)
            if not self._path:
                self._logger.warning(
                    "Sin ruta de %s a %s (obstáculos=%d)",
                    current_pos, self.target, len(self.obstacles),
                )
                return self._safe_idx("esperar", symbolic_mask)

        next_wp = self._path[0]
        action_name = self._wp_to_action(current_pos, next_wp)
        action_idx  = ACTIONS.index(action_name)

        # Si la máscara Prolog bloquea la acción planificada → replanificar
        if symbolic_mask is not None and symbolic_mask[action_idx] == 0.0:
            self._logger.debug(
                "Acción '%s' bloqueada por máscara, replanificando desde %s",
                action_name, current_pos,
            )
            self.obstacles.add(next_wp)
            alt_path = self._astar(current_pos, self.target)
            self.obstacles.discard(next_wp)
            self._replan_count += 1

            if alt_path:
                self._path = alt_path
                next_wp     = self._path[0]
                action_name = self._wp_to_action(current_pos, next_wp)
                action_idx  = ACTIONS.index(action_name)
            else:
                return self._safe_idx("esperar", symbolic_mask)

        return action_idx

    def learn(
        self,
        batch: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """No-op: A* no tiene parámetros entrenables."""
        return {"loss": 0.0}

    def remember(
        self,
        state:      np.ndarray,
        action:     int,
        reward:     float,
        next_state: np.ndarray,
        done:       bool,
    ) -> None:
        """Actualiza contadores de episodio (no almacena transiciones)."""
        self.total_reward  += reward
        self.episode_steps += 1

    def save_checkpoint(self, path: str) -> None:
        """No-op: el planificador es sin estado persistente."""
        pass

    def load_checkpoint(self, path: str) -> None:
        """No-op."""
        pass

    def reset_episode(self) -> None:
        super().reset_episode()
        self.target      = None
        self._path       = []
        self._last_pos   = None

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "target":        list(self.target) if self.target else None,
            "path_length":   len(self._path),
            "replan_count":  self._replan_count,
            "obstacles":     len(self.obstacles),
        })
        return base

    # ──────────────────────────────────────────────────────────────────────────
    # Algoritmo A*
    # ──────────────────────────────────────────────────────────────────────────

    def _astar(
        self,
        start: Tuple[int, int],
        goal:  Tuple[int, int],
    ) -> List[Tuple[int, int]]:
        """Retorna lista de waypoints (x, y) desde start+1 hasta goal (inclusive).

        Heurística: distancia Manhattan h(n) = |n.x - goal.x| + |n.y - goal.y|
        Costo g(n): número de celdas recorridas (cada paso = 1).
        Celdas en self.obstacles son inaccesibles.
        Retorna [] si no existe ruta.
        """
        if start == goal:
            return []

        g = self.grid_size

        def h(pos: Tuple[int, int]) -> int:
            return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

        # (f_score, g_cost, celda)
        open_heap: List[Tuple[int, int, Tuple[int, int]]] = []
        heapq.heappush(open_heap, (h(start), 0, start))

        came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
        g_cost:    Dict[Tuple[int, int], int]                        = {start: 0}

        while open_heap:
            _, cost, current = heapq.heappop(open_heap)

            if current == goal:
                return self._reconstruct(came_from, goal)

            for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                nx, ny = current[0] + dx, current[1] + dy
                neighbor: Tuple[int, int] = (nx, ny)

                if not (0 <= nx < g and 0 <= ny < g):
                    continue
                if neighbor in self.obstacles:
                    continue

                new_cost = cost + 1
                if neighbor not in g_cost or new_cost < g_cost[neighbor]:
                    g_cost[neighbor] = new_cost
                    heapq.heappush(open_heap, (new_cost + h(neighbor), new_cost, neighbor))
                    came_from[neighbor] = current

        return []  # sin ruta

    def _reconstruct(
        self,
        came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]],
        goal:      Tuple[int, int],
    ) -> List[Tuple[int, int]]:
        """Reconstruye el path desde goal hasta start y lo invierte."""
        path: List[Tuple[int, int]] = []
        node: Optional[Tuple[int, int]] = goal
        while node is not None:
            path.append(node)
            node = came_from.get(node)
        path.reverse()
        return path[1:]  # excluir la posición inicial

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _wp_to_action(
        self,
        current: Tuple[int, int],
        waypoint: Tuple[int, int],
    ) -> str:
        dx = waypoint[0] - current[0]
        dy = waypoint[1] - current[1]
        return _DELTA_ACTION.get((dx, dy), "esperar")

    def _safe_idx(
        self,
        action_name: str,
        mask: Optional[np.ndarray],
    ) -> int:
        """Retorna el índice de action_name; si la máscara lo bloquea, retorna el primer válido."""
        idx = ACTIONS.index(action_name)
        if mask is not None and float(mask[idx]) == 0.0:
            valid = [i for i, m in enumerate(mask) if float(m) > 0.0]
            return valid[0] if valid else idx
        return idx
