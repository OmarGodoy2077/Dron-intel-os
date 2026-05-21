"""conftest.py — Configuración compartida de pytest para la suite de Dron-Intel-OS.

Añade `backend/` a sys.path para que los tests usen los mismos imports que el
código de producción (`from agents... import`, `from environment... import`),
y expone fixtures reutilizables (entornos pequeños, agentes, predictor entrenado).

Ejecutar desde la raíz del proyecto:
    pytest                # toda la suite
    pytest -m "not prolog"  # omite los tests que requieren SWI-Prolog
    pytest --cov=backend  # con cobertura
"""

import os
import sys

import numpy as np
import pytest

# ── Hacer importable el paquete backend (mismos imports que producción) ────────
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.join(os.path.dirname(_PROJECT_ROOT), "backend")
# conftest está en tests/, el backend es ../backend respecto a la raíz del proyecto
_REPO_ROOT = os.path.dirname(_PROJECT_ROOT)
_BACKEND_DIR = os.path.join(_REPO_ROOT, "backend")
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)


# ── Detección de disponibilidad de Prolog (pyswip + SWI-Prolog) ────────────────
def _prolog_available() -> bool:
    """True si pyswip importa y SWI-Prolog está instalado/inicializable."""
    try:
        from pyswip import Prolog  # noqa: F401
        Prolog()  # fuerza la inicialización del motor nativo
        return True
    except Exception:
        return False


PROLOG_AVAILABLE = _prolog_available()


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def small_env():
    """Entorno pequeño y determinista para tests rápidos (grid 20, 3 drones)."""
    from environment.city_env import CyberCityEnv

    env = CyberCityEnv(
        grid_size=20,
        num_drones=3,
        num_packages=4,
        num_charging_stations=2,
        max_steps=50,
    )
    env.reset(seed=123)
    return env


@pytest.fixture
def dqn_agent():
    """Agente DQN recién inicializado (state_dim=11, action_dim=8)."""
    from agents.dqn_agent import DQNAgent

    return DQNAgent(agent_id="drone_0", state_dim=11, action_dim=8)


@pytest.fixture
def trained_predictor():
    """DemandPredictor entrenado con datos sintéticos (grid 20)."""
    from ml_models.demand_predictor import DemandPredictor

    p = DemandPredictor(grid_size=20)
    p.train()
    return p


@pytest.fixture
def rng():
    """Generador NumPy con semilla fija para reproducibilidad."""
    return np.random.default_rng(42)
