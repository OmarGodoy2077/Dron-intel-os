"""test_astar_agent.py — Tests del baseline A* (Sistema A del protocolo experimental).

Verifica:
  - Búsqueda A* con heurística Manhattan: encuentra ruta óptima en grid libre.
  - Evitación de obstáculos y caso sin ruta.
  - Replanning al cambiar objetivo u obstáculos.
  - Respeto de la máscara simbólica (si bloquea la acción planificada, replanifica/espera).
  - Conversión waypoint→acción y no-ops de aprendizaje.
"""

import numpy as np
import pytest

from agents.astar_agent import ACTIONS, AStarAgent


@pytest.fixture
def agent():
    return AStarAgent(agent_id="drone_0", grid_size=10)


# ── Algoritmo A* ────────────────────────────────────────────────────────────────

class TestAStarSearch:
    def test_finds_optimal_path_in_free_grid(self, agent):
        path = agent._astar((0, 0), (3, 0))
        # Distancia Manhattan = 3 → ruta de 3 pasos
        assert len(path) == 3
        assert path[-1] == (3, 0)

    def test_path_excludes_start_includes_goal(self, agent):
        path = agent._astar((0, 0), (2, 2))
        assert (0, 0) not in path
        assert path[-1] == (2, 2)
        assert len(path) == 4  # Manhattan(2,2) = 4

    def test_same_start_goal_empty_path(self, agent):
        assert agent._astar((5, 5), (5, 5)) == []

    def test_avoids_obstacles(self, agent):
        # Muro vertical en x=1 salvo un hueco en (1,3)
        agent.obstacles = {(1, 0), (1, 1), (1, 2), (1, 4)}
        path = agent._astar((0, 0), (2, 0))
        assert path  # existe ruta rodeando
        assert all(cell not in agent.obstacles for cell in path)

    def test_no_path_returns_empty(self, agent):
        # Encerrar el objetivo por completo
        agent.obstacles = {(4, 5), (6, 5), (5, 4), (5, 6)}
        path = agent._astar((0, 0), (5, 5))
        assert path == []


# ── Replanning ────────────────────────────────────────────────────────────────

class TestReplanning:
    def test_set_target_change_clears_path(self, agent):
        agent.set_target((5, 5))
        agent._path = [(1, 1), (2, 2)]
        agent.set_target((7, 7))  # cambio de objetivo
        assert agent._path == []

    def test_set_obstacles_change_clears_path(self, agent):
        agent.set_target((5, 5))
        agent._path = [(1, 1)]
        agent.set_obstacles([(3, 3)])  # nuevos obstáculos
        assert agent._path == []

    def test_set_obstacles_merges_storm_cells(self, agent):
        agent.set_obstacles([(1, 1)], storm_cells=[(2, 2)])
        assert (1, 1) in agent.obstacles
        assert (2, 2) in agent.obstacles


# ── select_action ──────────────────────────────────────────────────────────────

class TestSelectAction:
    def test_moves_toward_target(self, agent):
        agent.set_target((5, 0))
        agent.set_obstacles([])
        state = np.array([0, 0, 0] + [0] * 8, dtype=np.float32)  # x=0,y=0
        action = agent.select_action(state)
        assert ACTIONS[action] == "mover_e"  # debe avanzar en +x

    def test_waits_when_no_target(self, agent):
        agent.set_target(None)
        state = np.zeros(11, dtype=np.float32)
        assert ACTIONS[agent.select_action(state)] == "esperar"

    def test_waits_when_at_target(self, agent):
        agent.set_target((0, 0))
        state = np.zeros(11, dtype=np.float32)
        assert ACTIONS[agent.select_action(state)] == "esperar"

    def test_respects_symbolic_mask(self, agent):
        """Si la máscara bloquea la acción planificada, no debe devolverla."""
        agent.set_target((5, 0))
        agent.set_obstacles([])
        state = np.array([0, 0, 0] + [0] * 8, dtype=np.float32)
        mask = np.ones(8, dtype=np.float32)
        mask[ACTIONS.index("mover_e")] = 0.0  # bloquear el movimiento óptimo
        action = agent.select_action(state, mask)
        assert mask[action] == 1.0  # la acción devuelta es válida


# ── Contrato BaseAgent (no-ops de aprendizaje) ────────────────────────────────

class TestLearningNoOps:
    def test_learn_is_noop(self, agent):
        assert agent.learn()["loss"] == 0.0

    def test_save_load_are_noops(self, agent, tmp_path):
        # No deben lanzar excepción aunque A* no persista nada
        agent.save_checkpoint(str(tmp_path / "x.pt"))
        agent.load_checkpoint(str(tmp_path / "x.pt"))

    def test_reset_episode_clears_navigation(self, agent):
        agent.set_target((3, 3))
        agent._path = [(1, 1)]
        agent.reset_episode()
        assert agent.target is None
        assert agent._path == []
