"""test_demand_predictor.py — Tests del modelo ML predictivo del entorno.

Verifica:
  - Generación de datos sintéticos con estructura temporal + espacial.
  - Entrenamiento (R² razonable) y bandera is_trained.
  - predict() retorna un heatmap del tamaño del grid con valores no negativos.
  - get_high_demand_zones() retorna coordenadas válidas dentro del grid.
  - Las zonas son geográficamente significativas (cercanas a los hotspots, no uniformes).
  - Fallback heurístico cuando el modelo no está entrenado.
"""

import numpy as np
import pytest

from ml_models.demand_predictor import DemandPredictor


# ── Datos sintéticos ─────────────────────────────────────────────────────────

class TestSyntheticData:
    def test_generates_expected_columns(self):
        p = DemandPredictor(grid_size=20)
        df = p.generate_synthetic_data(n_samples=200)
        assert {"hour", "weekday", "x_region", "y_region",
                "temperature", "rain", "demand"}.issubset(df.columns)
        assert len(df) == 200

    def test_demand_is_non_negative(self):
        p = DemandPredictor(grid_size=20)
        df = p.generate_synthetic_data(n_samples=500)
        assert (df["demand"] >= 0).all()

    def test_demand_has_spatial_structure(self):
        """La demanda cerca de un hotspot debe superar la de la periferia."""
        p = DemandPredictor(grid_size=50)
        hotspots = p._demand_hotspots()
        hx, hy, _ = hotspots[0]
        center_factor = p._spatial_factor(hx, hy)
        corner_factor = p._spatial_factor(0, 49)  # lejos de todos los hotspots
        assert center_factor > corner_factor


# ── Entrenamiento ────────────────────────────────────────────────────────────

class TestTraining:
    def test_train_sets_flag_and_reports_r2(self, trained_predictor):
        assert trained_predictor.is_trained
        stats = trained_predictor.train()
        assert "r2_score" in stats and "n_samples" in stats
        assert stats["r2_score"] > 0.3  # el modelo captura señal real

    def test_untrained_predict_uses_heuristic_fallback(self):
        p = DemandPredictor(grid_size=20)
        assert not p.is_trained
        heatmap = p.predict({"hour": 12})
        assert heatmap.shape == (20, 20)


# ── Inferencia ────────────────────────────────────────────────────────────────

class TestInference:
    def test_predict_shape_and_non_negative(self, trained_predictor):
        heatmap = trained_predictor.predict({"hour": 9, "weekday": 1})
        assert heatmap.shape == (20, 20)
        assert (heatmap >= 0).all()

    def test_high_demand_zones_valid_coords(self, trained_predictor):
        zones = trained_predictor.get_high_demand_zones({"hour": 19, "weekday": 2}, top_k=5)
        assert len(zones) == 5
        for x, y in zones:
            assert 0 <= x < 20 and 0 <= y < 20

    def test_zones_cluster_near_a_hotspot(self):
        """Las zonas de alta demanda deben caer cerca de algún hotspot, no en una esquina aleatoria."""
        p = DemandPredictor(grid_size=50)
        p.train()
        zones = p.get_high_demand_zones({"hour": 13, "weekday": 2}, top_k=5)
        hotspots = [(hx, hy) for hx, hy, _ in p._demand_hotspots()]
        # Al menos una zona top debe estar a ≤12 celdas de un hotspot
        def min_dist(z):
            return min(abs(z[0] - hx) + abs(z[1] - hy) for hx, hy in hotspots)
        assert min(min_dist(z) for z in zones) <= 12

    def test_different_hours_give_different_zones(self):
        """El predictor responde al contexto temporal (no devuelve siempre lo mismo)."""
        p = DemandPredictor(grid_size=50)
        p.train()
        z_morning = set(p.get_high_demand_zones({"hour": 9, "weekday": 1}, top_k=5))
        z_evening = set(p.get_high_demand_zones({"hour": 19, "weekday": 1}, top_k=5))
        # No exigimos disjuntos, pero sí que el modelo no colapse a un único punto fijo
        assert len(z_morning | z_evening) > 5
