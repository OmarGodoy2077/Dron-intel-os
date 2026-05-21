"""test_neuro_symbolic_bridge.py — Tests del puente Neuro-Simbólico (Python ↔ Prolog).

Estructura en dos niveles:
  1. Tests SIN Prolog: pesos de reglas, manejo de archivo inexistente, log de decisiones.
     (El log y las constantes no requieren el motor nativo.)
  2. Tests CON Prolog (marca `prolog`): se omiten automáticamente si SWI-Prolog/pyswip
     no están instalados. Verifican que las reglas simbólicas inyectan el comportamiento
     esperado — Paso 1 de la verificación automatizada de la rúbrica.

Ejecutar solo los que no requieren Prolog:
    pytest tests/test_neuro_symbolic_bridge.py -m "not prolog"
"""

import os

import numpy as np
import pytest

from logic.neuro_symbolic_bridge import (
    ACTIONS,
    N_ACTIONS,
    _AM_PENALTIES,
    _RS_VALUES,
    NeuroSymbolicBridge,
)
from conftest import PROLOG_AVAILABLE

_RULES_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "backend", "logic", "rules.pl",
)

requires_prolog = pytest.mark.skipif(
    not PROLOG_AVAILABLE,
    reason="SWI-Prolog + pyswip no están instalados (test omitido, no fallido)",
)


# ── Nivel 1: sin motor Prolog ─────────────────────────────────────────────────

class TestConstants:
    def test_action_set_matches_env(self):
        assert N_ACTIONS == 8
        assert ACTIONS[0] == "despegar" and ACTIONS[-1] == "cargar"

    def test_masking_penalties_negative(self):
        # Las reglas de Action Masking penalizan (peso < 0)
        for rule, w in _AM_PENALTIES.items():
            assert w < 0, f"{rule} debería penalizar"

    def test_reward_shaping_signs(self):
        # Entregas y bonus positivos; conflictos/viento negativos
        assert _RS_VALUES["R6_medical"] > 0
        assert _RS_VALUES["R6_standard"] > 0
        assert _RS_VALUES["R10_route"] > 0
        assert _RS_VALUES["R4_cell_conflict"] < 0
        assert _RS_VALUES["R8_wind"] < 0

    def test_medical_bonus_exceeds_standard(self):
        assert _RS_VALUES["R6_medical"] > _RS_VALUES["R6_standard"]

    def test_twelve_rules_have_weights(self):
        """Las 12 reglas deben estar representadas entre masking + shaping."""
        rule_ids = set()
        for key in list(_AM_PENALTIES) + list(_RS_VALUES):
            rule_ids.add(key.split("_")[0])  # 'R1', 'R2', ...
        # R1..R12 (R6 y R11 aparecen con sufijos múltiples)
        for n in range(1, 13):
            assert f"R{n}" in rule_ids, f"Falta peso para R{n}"


class TestFileHandling:
    def test_missing_rules_file_raises(self):
        with pytest.raises(FileNotFoundError):
            NeuroSymbolicBridge("ruta/que/no/existe/rules.pl")


# ── Nivel 2: con motor Prolog (omitidos si no está disponible) ────────────────

@requires_prolog
class TestPrologRules:
    @pytest.fixture
    def bridge(self):
        return NeuroSymbolicBridge(_RULES_FILE)

    def _base_state(self, **overrides):
        state = {
            "agent_id": "drone_0",
            "position": (10, 10, 1),
            "battery": 80.0,
            "cargo": None,
            "cargo_type": None,
            "destination": None,
            "no_fly_zones": [],
            "charging_stations": {"station_0": "libre"},
            "other_agents": {},
            "other_batteries": {},
            "storm_regions": {},
            "wind": None,
        }
        state.update(overrides)
        return state

    def test_mask_shape_and_failsafe(self, bridge):
        mask = bridge.get_action_mask("drone_0", self._base_state())
        assert mask.shape == (N_ACTIONS,)
        assert mask.sum() > 0  # nunca todo-cero

    def test_r1_nfz_blocks_movement(self, bridge):
        # NFZ justo al norte del dron en (10,10)
        state = self._base_state(no_fly_zones=[(10, 11)])
        valid, penalty = bridge.validate_action("drone_0", "mover_n", state)
        assert valid is False
        assert penalty <= _AM_PENALTIES["R1_nfz"]

    def test_r3_collision_blocks_into_occupied(self, bridge):
        state = self._base_state(other_agents={"drone_1": (10, 11)})
        valid, _ = bridge.validate_action("drone_0", "mover_n", state)
        assert valid is False

    def test_r2_critical_battery_restricts(self, bridge):
        state = self._base_state(battery=8.0)  # < 15% crítico
        valid_move, _ = bridge.validate_action("drone_0", "mover_n", state)
        valid_land, _ = bridge.validate_action("drone_0", "aterrizar", state)
        assert valid_move is False   # movimiento bloqueado
        assert valid_land is True    # aterrizar permitido en emergencia

    def test_safe_action_is_valid(self, bridge):
        state = self._base_state()
        valid, penalty = bridge.validate_action("drone_0", "esperar", state)
        assert valid is True
        assert penalty == 0.0

    def test_intervention_counter_increments(self, bridge):
        bridge.reset_intervention_count()
        state = self._base_state(no_fly_zones=[(10, 11)])
        bridge.validate_action("drone_0", "mover_n", state)
        assert bridge.intervention_count >= 1

    def test_medical_priority_in_passage_negotiation(self, bridge):
        """R11: un dron con carga médica tiene prioridad de paso sobre uno sin carga."""
        state = self._base_state(
            cargo="pkg_0", cargo_type="medical",
            other_agents={"drone_1": (11, 10)},
            other_batteries={"drone_1": 90.0},
        )
        # drone_0 lleva médico → debe ganar la prioridad (retorna 0)
        winner = bridge.negotiate_passage("drone_0", "drone_1", state)
        assert winner == 0


# ── Log de decisiones (no requiere Prolog si construimos sin consultar) ───────

@requires_prolog
class TestDecisionLog:
    def test_log_decision_records_entries(self):
        bridge = NeuroSymbolicBridge(_RULES_FILE)
        bridge.clear_log()
        bridge.log_decision("MASK", "test message")
        entries = bridge.get_log_entries(last_n=10)
        assert len(entries) == 1
        assert entries[0]["level"] == "MASK"
        assert "message" in entries[0]
