"""
base_agent.py — Clase abstracta BaseAgent + AgentState dataclass.

Todos los agentes del Smart-Swarm heredan de BaseAgent:
  - DQNAgent  (aprendizaje por refuerzo)
  - AStarAgent (planificador baseline)
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Estado tipado de un dron individual
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AgentState:
    """Snapshot tipado del estado de un dron — s^i_t ∈ ℝ^7.

    Campos según formal_modeling.md Sección 2:
        position    : (x, y, z)  coordenadas en el grid 50×50, altitud 0-10
        battery     : [0, 100]   porcentaje de carga restante
        cargo       : package_id o None
        climate     : 'clear' | 'storm' | 'wind'
        neighbor_occupancy : {agent_id: True} para drones en radio ≤ 2
    """

    agent_id: str
    position: Tuple[int, int, int]
    battery: float
    cargo: Optional[str]
    climate: str
    neighbor_occupancy: Dict[str, bool]

    # Codificación de clima para el vector de observación
    _CLIMATE_ENC: Dict[str, int] = None  # type: ignore

    def __post_init__(self) -> None:
        object.__setattr__(self, "_CLIMATE_ENC", {"clear": 0, "storm": 1, "wind": 2})

    def to_obs_vector(self) -> np.ndarray:
        """Retorna el vector de observación 7-dim s^i_t = (x, y, z, β, κ, ω, η)."""
        x, y, z = self.position
        climate_enc = self._CLIMATE_ENC.get(self.climate, 0)  # type: ignore[union-attr]
        neighbor_count = sum(1 for v in self.neighbor_occupancy.values() if v)
        return np.array(
            [x, y, z, self.battery,
             1.0 if self.cargo is not None else 0.0,
             float(climate_enc),
             float(neighbor_count)],
            dtype=np.float32,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id":  self.agent_id,
            "position":  list(self.position),
            "battery":   round(self.battery, 2),
            "cargo":     self.cargo,
            "climate":   self.climate,
            "neighbors": len(self.neighbor_occupancy),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Interfaz abstracta
# ─────────────────────────────────────────────────────────────────────────────

class BaseAgent(ABC):
    """Interfaz común para todos los agentes del Smart-Swarm.

    Las subclases deben implementar los cinco métodos abstractos:
        select_action, learn, remember, save_checkpoint, load_checkpoint.
    """

    #: Orden canónico de acciones — coincide con CyberCityEnv.ACTIONS
    ACTIONS: List[str] = [
        "despegar", "aterrizar",
        "mover_n",  "mover_s", "mover_e", "mover_o",
        "esperar",  "cargar",
    ]

    def __init__(
        self,
        agent_id: str,
        state_dim: int,
        action_dim: int,
    ) -> None:
        self.agent_id   = agent_id
        self.state_dim  = state_dim
        self.action_dim = action_dim

        self._state: Optional[AgentState] = None
        self.total_reward:  float = 0.0
        self.episode_steps: int   = 0

        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}.{agent_id}")
        self._logger.debug("Agent %s (%s) initialized", agent_id, self.__class__.__name__)

    # ──────────────────────────────────────────────────────────────────────────
    # Métodos abstractos — contrato obligatorio para subclases
    # ──────────────────────────────────────────────────────────────────────────

    @abstractmethod
    def select_action(
        self,
        state: np.ndarray,
        symbolic_mask: Optional[np.ndarray] = None,
    ) -> int:
        """Retorna un índice de acción en [0, action_dim).

        Args:
            state         : Vector de observación s^i_t de dimensión state_dim.
            symbolic_mask : Máscara binaria (float32) de dimensión action_dim.
                            0.0 = acción prohibida por el motor Prolog.
                            None = sin restricciones simbólicas.
        Returns:
            Índice entero de la acción seleccionada.
        """

    @abstractmethod
    def learn(
        self,
        batch: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """Ejecuta un paso de aprendizaje/actualización.

        Returns:
            Dict con al menos {"loss": float}. Los agentes no-learners
            devuelven {"loss": 0.0}.
        """

    @abstractmethod
    def remember(
        self,
        state:      np.ndarray,
        action:     int,
        reward:     float,
        next_state: np.ndarray,
        done:       bool,
        next_mask:  Optional[np.ndarray] = None,
    ) -> None:
        """Almacena una transición (s, a, r, s', done) en el buffer de experiencia.

        Args:
            next_mask: máscara de acciones válidas en next_state. Los agentes con
                masking (DQN/Neuro-DQN) la usan para el target Q; los planificadores
                sin buffer (A*) la ignoran. Mantener el parámetro en la firma base
                permite que el training loop invoque a todos los agentes igual.
        """

    @abstractmethod
    def save_checkpoint(self, path: str) -> None:
        """Persiste los pesos del modelo y el estado de entrenamiento en disco.

        Args:
            path: Ruta completa del archivo (.pt, .pkl, etc.).
        """

    @abstractmethod
    def load_checkpoint(self, path: str) -> None:
        """Restaura pesos y estado de entrenamiento desde un checkpoint.

        Args:
            path: Ruta del archivo creado por save_checkpoint().

        Raises:
            FileNotFoundError: Si el archivo no existe.
        """

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers compartidos (no abstractos)
    # ──────────────────────────────────────────────────────────────────────────

    def update_state(self, state: AgentState) -> None:
        """Actualiza el snapshot interno del estado del agente."""
        self._state = state

    def get_state(self) -> Optional[AgentState]:
        """Retorna el último snapshot AgentState, o None si no se ha inicializado."""
        return self._state

    def reset_episode(self) -> None:
        """Reinicia los acumuladores de episodio (llamar al inicio de cada episodio)."""
        self.total_reward  = 0.0
        self.episode_steps = 0

    def to_dict(self) -> Dict[str, Any]:
        """Serializa la telemetría del agente para broadcast WebSocket."""
        return {
            "agent_id":      self.agent_id,
            "type":          self.__class__.__name__,
            "total_reward":  round(self.total_reward, 4),
            "episode_steps": self.episode_steps,
            "state": self._state.to_dict() if self._state is not None else None,
        }

    def action_name(self, idx: int) -> str:
        """Convierte índice de acción a nombre legible."""
        if 0 <= idx < len(self.ACTIONS):
            return self.ACTIONS[idx]
        return f"unknown({idx})"
