"""test_city_env.py — Tests del entorno Dec-POMDP CyberCityEnv.

Verifica:
  - Espacios de observación/acción conformes a formal_modeling.md (ℝ¹¹, 8 acciones×N).
  - Determinismo por semilla.
  - Fórmula de recompensa R_total: entregas positivas, penalizaciones correctas.
  - Action masking vectorizado (R1 NFZ, R3 colisión, R5 estación, R7 tormenta, R2 batería).
  - Economía de batería rebalanceada (no agota en pocos pasos).
  - Inyección de zonas de demanda del modelo ML (sesgo de destinos + retrocompatibilidad).
"""

import numpy as np
import pytest

from environment.city_env import (
    ACTIONS,
    BATTERY_COST,
    BATTERY_RECHARGE,
    N_ACTIONS,
    CyberCityEnv,
)


# ── Espacios y reset ────────────────────────────────────────────────────────────

class TestSpaces:
    def test_action_constants(self):
        assert N_ACTIONS == 8
        assert ACTIONS == [
            "despegar", "aterrizar",
            "mover_n", "mover_s", "mover_e", "mover_o",
            "esperar", "cargar",
        ]

    def test_observation_is_11_dim_per_drone(self, small_env):
        obs, _ = small_env.reset(seed=1)
        assert set(obs.keys()) == {f"drone_{i}" for i in range(small_env.num_drones)}
        for v in obs.values():
            assert v.shape == (11,)
            assert v.dtype == np.float32

    def test_observation_within_declared_bounds(self, small_env):
        obs, _ = small_env.reset(seed=7)
        for v in obs.values():
            # 7 features normalizados [0,1] + 4 deltas [-1,1]
            assert v.min() >= -1.0 - 1e-6
            assert v.max() <= 1.0 + 1e-6

    def test_action_space_multidiscrete(self, small_env):
        assert list(small_env.action_space.nvec) == [N_ACTIONS] * small_env.num_drones

    def test_exactly_two_medical_packages(self, small_env):
        small_env.reset(seed=1)
        assert small_env.package_types.count("medical") == 2

    def test_reset_is_deterministic_with_seed(self):
        a = CyberCityEnv(grid_size=20, num_drones=3, num_packages=4, max_steps=50)
        b = CyberCityEnv(grid_size=20, num_drones=3, num_packages=4, max_steps=50)
        oa, _ = a.reset(seed=99)
        ob, _ = b.reset(seed=99)
        for k in oa:
            np.testing.assert_array_equal(oa[k], ob[k])

    def test_same_seed_gives_identical_package_layout(self):
        """Comparación justa entre sistemas: misma semilla → mismo layout de paquetes."""
        a = CyberCityEnv(grid_size=30, num_drones=3, num_packages=6, max_steps=50)
        b = CyberCityEnv(grid_size=30, num_drones=3, num_packages=6, max_steps=50)
        a.reset(seed=7)
        b.reset(seed=7)
        np.testing.assert_array_equal(a.package_positions, b.package_positions)
        np.testing.assert_array_equal(a.package_destinations, b.package_destinations)


# ── Dinámica de pasos ─────────────────────────────────────────────────────────

class TestStep:
    def test_step_returns_gym_tuple(self, small_env):
        actions = np.zeros(small_env.num_drones, dtype=np.int64)
        obs, rewards, dones, truncated, infos = small_env.step(actions)
        assert rewards.shape == (small_env.num_drones,)
        assert dones.shape == (small_env.num_drones,)
        assert truncated.shape == (small_env.num_drones,)
        assert set(infos.keys()) == {f"drone_{i}" for i in range(small_env.num_drones)}

    def test_truncates_at_max_steps(self):
        env = CyberCityEnv(grid_size=20, num_drones=2, num_packages=2, max_steps=5)
        env.reset(seed=1)
        truncated = np.zeros(2, dtype=bool)
        for _ in range(5):
            _, _, _, truncated, _ = env.step(np.array([6, 6]))  # esperar
        assert truncated.all()

    def test_step_advances_counter(self, small_env):
        before = small_env.current_step
        small_env.step(np.array([6] * small_env.num_drones))
        assert small_env.current_step == before + 1

    def test_delivery_info_reports_type_and_location(self):
        """Al entregar, info expone delivery=True y pkg_type (para animar en el frontend)."""
        env = CyberCityEnv(grid_size=20, num_drones=1, num_packages=1, max_steps=50)
        env.reset(seed=1)
        # Forzar: el dron lleva el paquete 0 y está sobre su destino, luego 'esperar'
        env.drone_cargos[0] = 0
        env.package_picked[0] = True
        env.package_types[0] = "medical"
        dest = env.package_destinations[0]
        env.drone_positions[0] = [int(dest[0]), int(dest[1]), 0]
        _, _, _, _, infos = env.step(np.array([6]))  # esperar (no mueve, entrega in situ)
        info = infos["drone_0"]
        assert info.get("delivery") is True
        assert info.get("pkg_type") == "medical"
        assert env.package_delivered[0]


# ── Fórmula de recompensa R_total ─────────────────────────────────────────────

class TestReward:
    def test_medical_delivery_outranks_standard(self, small_env):
        med = small_env._calculate_rewards(
            action="esperar", battery_after=80, delivered=True,
            pkg_type="medical", picked_up=False, collision_count=0, nfz_violation=False,
        )
        std = small_env._calculate_rewards(
            action="esperar", battery_after=80, delivered=True,
            pkg_type="standard", picked_up=False, collision_count=0, nfz_violation=False,
        )
        assert med > std > 0

    def test_collision_is_penalized(self, small_env):
        r = small_env._calculate_rewards(
            action="mover_n", battery_after=80, delivered=False, pkg_type=None,
            picked_up=False, collision_count=1, nfz_violation=False,
        )
        assert r < 0

    def test_nfz_violation_is_penalized(self, small_env):
        clean = small_env._calculate_rewards(
            action="mover_n", battery_after=80, delivered=False, pkg_type=None,
            picked_up=False, collision_count=0, nfz_violation=False,
        )
        viol = small_env._calculate_rewards(
            action="mover_n", battery_after=80, delivered=False, pkg_type=None,
            picked_up=False, collision_count=0, nfz_violation=True,
        )
        assert viol < clean

    def test_battery_death_is_penalized(self, small_env):
        r = small_env._calculate_rewards(
            action="esperar", battery_after=0.0, delivered=False, pkg_type=None,
            picked_up=False, collision_count=0, nfz_violation=False,
        )
        assert r < 0

    def test_potential_shaping_is_symmetric(self, small_env):
        """El shaping basado en potencial: acercarse +k·Δ y alejarse −k·Δ son simétricos."""
        approach = small_env._calculate_rewards(
            action="mover_n", battery_after=80, delivered=False, pkg_type=None,
            picked_up=False, collision_count=0, nfz_violation=False, dist_delta=+2.0,
        )
        recede = small_env._calculate_rewards(
            action="mover_n", battery_after=80, delivered=False, pkg_type=None,
            picked_up=False, collision_count=0, nfz_violation=False, dist_delta=-2.0,
        )
        # La diferencia debe ser ~2 * (0.5 * 2) = 2.0 (simétrico alrededor del grind)
        assert approach > recede
        assert approach + recede == pytest.approx(2 * (-0.05), abs=1e-6)

    def test_delivery_outweighs_episode_grind(self, small_env):
        """Una entrega estándar (+100) debe superar el grind de un episodio entero."""
        grind_per_step = abs(-0.05)
        worst_case_grind = grind_per_step * small_env.max_steps
        assert 100.0 > worst_case_grind


# ── Action masking vectorizado (reglas simbólicas en Python) ──────────────────

class TestActionMask:
    def test_mask_shape_and_binary(self, small_env):
        m = small_env.fast_action_mask(0)
        assert m.shape == (N_ACTIONS,)
        assert set(np.unique(m)).issubset({0.0, 1.0})

    def test_mask_never_all_zero(self, small_env):
        for i in range(small_env.num_drones):
            assert small_env.fast_action_mask(i).sum() > 0

    def test_nfz_blocks_movement_into_zone(self):
        env = CyberCityEnv(grid_size=20, num_drones=1, num_packages=1, max_steps=50)
        env.reset(seed=1)
        # Colocar el dron en (5,5) y una NFZ al norte (5,6)
        env.drone_positions[0] = [5, 5, 0]
        env.no_fly_zones = [(5, 6)]
        env._nfz_set = {(5, 6)}
        env._storm_cells = set()
        m = env.fast_action_mask(0)
        assert m[ACTIONS.index("mover_n")] == 0.0  # R1: bloqueado
        assert m[ACTIONS.index("mover_s")] == 1.0  # libre

    def test_collision_blocks_occupied_cell(self):
        env = CyberCityEnv(grid_size=20, num_drones=2, num_packages=1, max_steps=50)
        env.reset(seed=1)
        env.drone_positions[0] = [5, 5, 0]
        env.drone_positions[1] = [5, 6, 0]  # otro dron justo al norte
        env._nfz_set = set()
        env._storm_cells = set()
        m = env.fast_action_mask(0)
        assert m[ACTIONS.index("mover_n")] == 0.0  # R3: celda ocupada

    def test_critical_battery_restricts_actions(self):
        env = CyberCityEnv(grid_size=20, num_drones=1, num_packages=1, max_steps=50)
        env.reset(seed=1)
        env.drone_positions[0] = [5, 5, 0]
        env.drone_batteries[0] = 10.0  # < 15% crítico
        env._nfz_set = set()
        env._storm_cells = set()
        m = env.fast_action_mask(0)
        # despegar nunca debe permitirse en batería crítica
        assert m[ACTIONS.index("despegar")] == 0.0


# ── Economía de batería (fix de convergencia) ─────────────────────────────────

class TestBatteryEconomy:
    def test_battery_lasts_enough_to_cross_grid(self):
        """Con el costo rebalanceado, 100% debe durar >200 movimientos."""
        autonomy = 100.0 / BATTERY_COST["mover_n"]
        assert autonomy >= 200

    def test_recharge_refills_in_few_steps(self):
        assert BATTERY_RECHARGE >= 30.0  # recarga en ≤3 steps

    def test_charging_increases_battery_on_station(self):
        env = CyberCityEnv(grid_size=20, num_drones=1, num_packages=1,
                           num_charging_stations=1, max_steps=50)
        env.reset(seed=1)
        cs = env.charging_stations[0]
        env.drone_positions[0] = [int(cs[0]), int(cs[1]), 0]
        env.drone_batteries[0] = 40.0
        env.charging_occupied[:] = False
        env.step(np.array([ACTIONS.index("cargar")]))
        assert env.drone_batteries[0] > 40.0


# ── Integración con el modelo ML (zonas de demanda) ───────────────────────────

class TestDemandZoneInjection:
    def test_demand_zones_bias_destinations(self):
        env = CyberCityEnv(grid_size=50, num_drones=3, num_packages=10, max_steps=50)
        zones = [(25, 25)]
        env.reset(seed=1, options={"demand_zones": zones})
        assert env.demand_zones == zones
        # Con demand_fraction=0.6, ≥5 de 10 destinos deben caer cerca de la zona
        near = sum(
            abs(int(d[0]) - 25) <= 4 and abs(int(d[1]) - 25) <= 4
            for d in env.package_destinations
        )
        assert near >= 5

    def test_no_demand_zones_keeps_uniform_spawn(self):
        """Retrocompatibilidad: sin zonas, el spawn es uniforme (lógica previa intacta)."""
        env = CyberCityEnv(grid_size=50, num_drones=3, num_packages=10, max_steps=50)
        env.reset(seed=1)
        assert env.demand_zones == []
        assert env.package_destinations.shape == (10, 2)

    def test_demand_zones_clipped_to_grid(self):
        env = CyberCityEnv(grid_size=20, num_drones=2, num_packages=4, max_steps=50)
        env.reset(seed=1, options={"demand_zones": [(999, 999), (5, 5)]})
        # La zona fuera del grid se descarta; solo (5,5) sobrevive
        assert (5, 5) in env.demand_zones
        assert (999, 999) not in env.demand_zones


# ── Serialización para el bridge Prolog ───────────────────────────────────────

class TestStateDict:
    def test_state_dict_has_required_keys(self, small_env):
        sd = small_env.get_state_dict(0)
        required = {
            "agent_id", "position", "battery", "cargo", "destination",
            "no_fly_zones", "charging_stations", "other_agents", "storm_regions", "wind",
        }
        assert required.issubset(sd.keys())

    def test_blocked_cells_includes_nfz(self, small_env):
        small_env.no_fly_zones = [(1, 1), (2, 2)]
        small_env._nfz_set = set(small_env.no_fly_zones)  # get_blocked_cells lee el cache
        blocked = small_env.get_blocked_cells()
        assert (1, 1) in blocked and (2, 2) in blocked
