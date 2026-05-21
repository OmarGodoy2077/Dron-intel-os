"""test_dynamics.py — Tests del motor de dinámica estocástica (entorno realista).

Verifica:
  - Spawn estocástico de tormentas, viento y NFZ dinámicas según probabilidades.
  - Expiración por duración (los eventos terminan).
  - Estructura del estado retornado.
  - Reset limpia todos los eventos activos.
  - Las celdas de NFZ dinámica respetan el radio y los límites del grid.
"""

import numpy as np
import pytest

from environment.dynamics import (
    DynamicNoFlyZone,
    DynamicsEngine,
    Storm,
    WindCondition,
)


# ── Dataclasses de eventos ──────────────────────────────────────────────────────

class TestEventDataclasses:
    def test_storm_lifecycle(self):
        s = Storm("storm_0", (0, 5), (0, 5), intensity=0.5, duration=3)
        assert s.is_active()
        for _ in range(3):
            s.tick()
        assert not s.is_active()

    def test_wind_lifecycle(self):
        w = WindCondition("norte", 80.0, duration=2)
        assert w.is_active()
        w.tick(); w.tick()
        assert not w.is_active()

    def test_dynamic_nfz_cells_within_radius_and_grid(self):
        nfz = DynamicNoFlyZone("nfz_0", center=(5, 5), radius=2, start_step=0, end_step=10)
        cells = nfz.get_cells(grid_size=20)
        assert (5, 5) in cells
        for (x, y) in cells:
            assert 0 <= x < 20 and 0 <= y < 20
            assert (x - 5) ** 2 + (y - 5) ** 2 <= 2 ** 2

    def test_dynamic_nfz_clipped_to_grid_edges(self):
        nfz = DynamicNoFlyZone("nfz_edge", center=(0, 0), radius=3, start_step=0, end_step=5)
        cells = nfz.get_cells(grid_size=10)
        assert all(0 <= x < 10 and 0 <= y < 10 for x, y in cells)

    def test_dynamic_nfz_active_window(self):
        nfz = DynamicNoFlyZone("nfz_w", center=(5, 5), radius=1, start_step=3, end_step=6)
        assert not nfz.is_active(2)
        assert nfz.is_active(3)
        assert nfz.is_active(5)
        assert not nfz.is_active(6)


# ── Motor de dinámica ────────────────────────────────────────────────────────

class TestDynamicsEngine:
    def test_get_state_structure(self):
        eng = DynamicsEngine(grid_size=50)
        st = eng.step()
        for key in ("storm_regions", "wind", "dynamic_nfz_cells",
                    "num_active_storms", "num_active_winds", "num_dynamic_nfzs", "step"):
            assert key in st

    def test_reset_clears_events(self):
        eng = DynamicsEngine(grid_size=50, storm_prob=1.0, wind_prob=1.0, nfz_prob=1.0,
                             rng=np.random.default_rng(0))
        for _ in range(5):
            eng.step()
        assert eng.active_storms or eng.active_winds or eng.dynamic_nfzs
        eng.reset()
        assert not eng.active_storms
        assert not eng.active_winds
        assert not eng.dynamic_nfzs
        assert eng.current_step == 0

    def test_high_probability_spawns_events(self):
        eng = DynamicsEngine(grid_size=50, storm_prob=1.0, wind_prob=1.0, nfz_prob=1.0,
                             rng=np.random.default_rng(1))
        eng.step()
        assert len(eng.active_storms) >= 1
        assert len(eng.active_winds) >= 1
        assert len(eng.dynamic_nfzs) >= 1

    def test_zero_probability_spawns_nothing(self):
        eng = DynamicsEngine(grid_size=50, storm_prob=0.0, wind_prob=0.0, nfz_prob=0.0,
                             rng=np.random.default_rng(2))
        for _ in range(20):
            eng.step()
        assert not eng.active_storms
        assert not eng.active_winds
        assert not eng.dynamic_nfzs

    def test_events_expire_over_time(self):
        eng = DynamicsEngine(grid_size=50, storm_prob=0.0, wind_prob=0.0, nfz_prob=0.0,
                             rng=np.random.default_rng(3))
        # Inyectar una tormenta corta manualmente y avanzar hasta que expire
        eng.active_storms.append(Storm("s", (0, 5), (0, 5), 0.5, duration=2))
        eng.step()
        eng.step()
        eng.step()
        assert not eng.active_storms

    def test_dominant_wind_is_highest_intensity(self):
        eng = DynamicsEngine(grid_size=50, storm_prob=0.0, wind_prob=0.0, nfz_prob=0.0,
                             rng=np.random.default_rng(4))
        eng.active_winds.append(WindCondition("norte", 40.0, duration=10))
        eng.active_winds.append(WindCondition("sur", 120.0, duration=10))
        st = eng.get_state()
        assert st["wind"][0] == "sur"  # el más intenso domina
