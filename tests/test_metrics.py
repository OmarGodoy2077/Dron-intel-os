"""test_metrics.py — Tests del MetricsCollector (análisis Pandas + persistencia).

Verifica:
  - EpisodeRecord.success_rate.
  - Registro en DataFrame y persistencia/recarga CSV.
  - Análisis: get_summary, ventanas rodantes en get_learning_curve, tabla comparativa.
  - clear()/count() para la gestión de datos (borrado total y por sistema).
"""

import numpy as np
import pytest

from analysis.metrics import COLUMNS, EpisodeRecord, MetricsCollector


def _record(ep, system="dqn", delivered=5, total=10, reward=100.0):
    return EpisodeRecord(
        episode=ep, system=system, total_reward=reward,
        deliveries_completed=delivered, deliveries_total=total,
        rule_violations=0, collisions=0, battery_failures=0,
        steps=300, avg_battery_remaining=50.0, symbolic_interventions=3,
    )


# ── EpisodeRecord ────────────────────────────────────────────────────────────

class TestEpisodeRecord:
    def test_success_rate(self):
        assert _record(0, delivered=5, total=10).success_rate == 0.5

    def test_success_rate_zero_total_safe(self):
        assert _record(0, delivered=0, total=0).success_rate == 0.0


# ── Persistencia ────────────────────────────────────────────────────────────

class TestPersistence:
    def test_record_and_count(self, tmp_path):
        mc = MetricsCollector(log_path=str(tmp_path / "logs.csv"))
        for ep in range(5):
            mc.record_episode(_record(ep))
        assert mc.count() == 5
        assert mc.count(system="dqn") == 5
        assert mc.count(system="neuro_dqn") == 0

    def test_save_and_reload(self, tmp_path):
        path = str(tmp_path / "logs.csv")
        mc = MetricsCollector(log_path=path)
        mc.record_episode(_record(0))
        mc.save()
        # Nueva instancia carga del CSV
        mc2 = MetricsCollector(log_path=path)
        assert mc2.count() == 1

    def test_columns_present(self, tmp_path):
        mc = MetricsCollector(log_path=str(tmp_path / "logs.csv"))
        mc.record_episode(_record(0))
        assert set(COLUMNS).issubset(set(mc._df.columns))


# ── Análisis ────────────────────────────────────────────────────────────────

class TestAnalytics:
    def test_summary_aggregates(self, tmp_path):
        mc = MetricsCollector(log_path=str(tmp_path / "logs.csv"))
        for ep in range(10):
            mc.record_episode(_record(ep, reward=float(ep * 10), delivered=ep, total=10))
        s = mc.get_summary(system="dqn")
        assert s["episodes_recorded"] == 10
        assert "mean_reward" in s and "mean_success_rate" in s

    def test_learning_curve_has_rolling_average(self, tmp_path):
        mc = MetricsCollector(log_path=str(tmp_path / "logs.csv"))
        for ep in range(30):
            mc.record_episode(_record(ep, delivered=ep % 10, total=10))
        df = mc.get_learning_curve("dqn", metric="success_rate")
        assert "rolling_avg" in df.columns
        assert len(df) == 30

    def test_comparison_table_per_system(self, tmp_path):
        mc = MetricsCollector(log_path=str(tmp_path / "logs.csv"))
        for ep in range(5):
            mc.record_episode(_record(ep, system="dqn"))
            mc.record_episode(_record(ep, system="neuro_dqn"))
        table = mc.get_comparison_table()
        assert "dqn" in table.index
        assert "neuro_dqn" in table.index

    def test_live_json_compact_lists(self, tmp_path):
        mc = MetricsCollector(log_path=str(tmp_path / "logs.csv"))
        for ep in range(3):
            mc.record_episode(_record(ep))
        live = mc.to_live_json("dqn")
        assert isinstance(live["episodes"], list)
        assert len(live["rewards"]) == 3


# ── Gestión de datos (clear/count) ────────────────────────────────────────────

class TestDataManagement:
    def test_clear_all(self, tmp_path):
        mc = MetricsCollector(log_path=str(tmp_path / "logs.csv"))
        for ep in range(8):
            mc.record_episode(_record(ep))
        removed = mc.clear()
        assert removed == 8
        assert mc.count() == 0

    def test_clear_by_system_keeps_others(self, tmp_path):
        mc = MetricsCollector(log_path=str(tmp_path / "logs.csv"))
        for ep in range(4):
            mc.record_episode(_record(ep, system="dqn"))
            mc.record_episode(_record(ep, system="astar"))
        removed = mc.clear(system="dqn")
        assert removed == 4
        assert mc.count(system="dqn") == 0
        assert mc.count(system="astar") == 4

    def test_clear_persists_to_disk(self, tmp_path):
        path = str(tmp_path / "logs.csv")
        mc = MetricsCollector(log_path=path)
        mc.record_episode(_record(0))
        mc.clear()
        # Recargar desde disco confirma el borrado persistido
        mc2 = MetricsCollector(log_path=path)
        assert mc2.count() == 0
