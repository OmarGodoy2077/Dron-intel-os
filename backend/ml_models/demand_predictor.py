from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler


class DemandPredictor:
    """
    Predicts delivery-demand heatmaps for the Cyber-City Grid.

    Features engineered from temporal context (hour, weekday) encoded
    as cyclical sin/cos pairs, plus spatial coordinates, temperature,
    and rain indicator.

    Used by the training loop to bias drone dispatch toward high-demand
    zones, reducing median delivery wait time.
    """

    def __init__(self, grid_size: int = 50, time_horizon: int = 10) -> None:
        self.grid_size = grid_size
        self.time_horizon = time_horizon
        self.model = GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42
        )
        self.scaler = StandardScaler()
        self.is_trained: bool = False
        self._history: List[Dict] = []

    # ------------------------------------------------------------------ #
    #  Feature engineering                                                 #
    # ------------------------------------------------------------------ #

    def _features(self, record: Dict) -> np.ndarray:
        hour        = float(record.get("hour", 12))
        weekday     = float(record.get("weekday", 0))
        temperature = float(record.get("temperature", 20.0))
        rain        = float(record.get("rain", 0))
        x_norm      = float(record.get("x_region", 0)) / self.grid_size
        y_norm      = float(record.get("y_region", 0)) / self.grid_size

        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        day_sin  = np.sin(2 * np.pi * weekday / 7)
        day_cos  = np.cos(2 * np.pi * weekday / 7)

        return np.array(
            [hour_sin, hour_cos, day_sin, day_cos,
             temperature / 40.0, rain, x_norm, y_norm],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------ #
    #  Data ingestion                                                      #
    # ------------------------------------------------------------------ #

    def add_observation(self, record: Dict) -> None:
        self._history.append(record)

    # Centros comerciales/residenciales fijos del Cyber-City Grid: la demanda
    # se concentra alrededor de ellos (componente espacial gaussiano), de modo
    # que las predicciones de zonas de alta demanda son geográficamente
    # significativas y no uniformes.
    def _demand_hotspots(self) -> List[Tuple[float, float, float]]:
        g = self.grid_size
        return [
            (0.25 * g, 0.30 * g, 1.6),   # distrito comercial SO
            (0.70 * g, 0.65 * g, 1.4),   # zona residencial NE
            (0.50 * g, 0.85 * g, 1.2),   # corredor norte
            (0.80 * g, 0.20 * g, 1.0),   # polígono industrial SE
        ]

    def _spatial_factor(self, x: float, y: float) -> float:
        """Multiplicador de demanda según cercanía a los hotspots (suma de gaussianas)."""
        sigma = 0.15 * self.grid_size
        factor = 0.4  # demanda base de fondo en zonas alejadas
        for hx, hy, weight in self._demand_hotspots():
            d2 = (x - hx) ** 2 + (y - hy) ** 2
            factor += weight * np.exp(-d2 / (2.0 * sigma * sigma))
        return float(factor)

    def generate_synthetic_data(self, n_samples: int = 2000) -> pd.DataFrame:
        """Create realistic synthetic demand data for initial training.

        La demanda combina una componente temporal (picos horarios) y una
        componente espacial (hotspots comerciales/residenciales fijos), de modo
        que el regresor aprende un mapa de demanda con estructura geográfica.
        """
        rng = np.random.default_rng(0)
        records = []
        for _ in range(n_samples):
            hour    = int(rng.integers(0, 24))
            weekday = int(rng.integers(0, 7))
            x = int(rng.integers(0, self.grid_size))
            y = int(rng.integers(0, self.grid_size))
            temp = float(rng.normal(20, 8))
            rain = int(rng.choice([0, 1], p=[0.8, 0.2]))

            # Business-hours peak + lunch + evening surge
            if 8 <= hour <= 10:    base = 22
            elif 12 <= hour <= 14: base = 17
            elif 18 <= hour <= 20: base = 28
            elif 0 <= hour <= 5:   base = 2
            else:                  base = 8

            # Modulación espacial: la misma franja horaria genera más demanda
            # cerca de los hotspots que en la periferia.
            base *= self._spatial_factor(x, y)

            demand = max(0.0, float(rng.poisson(base)) - rain * 4
                         + float(rng.normal(0, 1.5)))
            records.append({
                "hour": hour, "weekday": weekday,
                "x_region": x, "y_region": y,
                "temperature": temp, "rain": rain, "demand": demand,
            })
        return pd.DataFrame(records)

    # ------------------------------------------------------------------ #
    #  Training                                                            #
    # ------------------------------------------------------------------ #

    def train(self, df: Optional[pd.DataFrame] = None) -> Dict:
        if df is None:
            if len(self._history) < 200:
                df = self.generate_synthetic_data()
            else:
                df = pd.DataFrame(self._history)

        X = np.array([self._features(row) for _, row in df.iterrows()])
        y = df["demand"].values

        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_trained = True

        return {"r2_score": float(self.model.score(X_scaled, y)), "n_samples": len(y)}

    # ------------------------------------------------------------------ #
    #  Inference                                                           #
    # ------------------------------------------------------------------ #

    def predict(self, context: Dict) -> np.ndarray:
        """Return demand heatmap (grid_size × grid_size) for the given context."""
        if not self.is_trained:
            return self._heuristic_demand(context)

        step = 4
        features_batch, coords = [], []
        for x in range(0, self.grid_size, step):
            for y in range(0, self.grid_size, step):
                features_batch.append(self._features({**context, "x_region": x, "y_region": y}))
                coords.append((x, y))

        X = self.scaler.transform(np.array(features_batch))
        preds = self.model.predict(X)

        heatmap = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        for (x, y), val in zip(coords, preds):
            heatmap[y : y + step, x : x + step] = max(0.0, float(val))
        return heatmap

    def _heuristic_demand(self, context: Dict) -> np.ndarray:
        rng = np.random.default_rng()
        heatmap = rng.exponential(scale=2.0, size=(self.grid_size, self.grid_size)).astype(np.float32)
        hour = context.get("hour", 12)
        if 8 <= hour <= 10 or 18 <= hour <= 20:
            heatmap *= 2.5
        elif 12 <= hour <= 14:
            heatmap *= 1.5
        elif 0 <= hour <= 5:
            heatmap *= 0.2
        return heatmap

    def get_high_demand_zones(
        self, context: Dict, top_k: int = 5
    ) -> List[Tuple[int, int]]:
        heatmap = self.predict(context)
        flat = heatmap.flatten()
        top_indices = np.argsort(flat)[-top_k:][::-1]
        return [(int(i % self.grid_size), int(i // self.grid_size)) for i in top_indices]
