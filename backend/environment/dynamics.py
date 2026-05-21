import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Storm:
    region_id: str
    x_range: Tuple[int, int]
    y_range: Tuple[int, int]
    intensity: float        # 0–1 severity
    duration: int           # steps until dissipation
    elapsed: int = 0

    def is_active(self) -> bool:
        return self.elapsed < self.duration

    def tick(self) -> None:
        self.elapsed += 1

    def to_dict(self) -> Dict:
        return {
            "x_range": self.x_range,
            "y_range": self.y_range,
            "intensity": self.intensity,
            "type": "storm",
        }


@dataclass
class WindCondition:
    direction: str          # norte | sur | este | oeste
    intensity: float        # km/h
    duration: int
    elapsed: int = 0

    def is_active(self) -> bool:
        return self.elapsed < self.duration

    def tick(self) -> None:
        self.elapsed += 1


@dataclass
class DynamicNoFlyZone:
    zone_id: str
    center: Tuple[int, int]
    radius: int
    start_step: int
    end_step: int
    reason: str = "temporary_restriction"

    def is_active(self, current_step: int) -> bool:
        return self.start_step <= current_step < self.end_step

    def get_cells(self, grid_size: int) -> List[Tuple[int, int]]:
        cx, cy = self.center
        cells = []
        for dx in range(-self.radius, self.radius + 1):
            for dy in range(-self.radius, self.radius + 1):
                if dx * dx + dy * dy <= self.radius * self.radius:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < grid_size and 0 <= ny < grid_size:
                        cells.append((nx, ny))
        return cells


class DynamicsEngine:
    """
    Stochastic dynamics for Cyber-City Grid.
    Spawns random storms, wind conditions and temporary no-fly zones
    each step according to configurable probabilities.
    """

    WIND_DIRECTIONS = ["norte", "sur", "este", "oeste"]

    def __init__(
        self,
        grid_size: int,
        storm_prob: float = 0.02,
        wind_prob: float = 0.03,
        nfz_prob: float = 0.01,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.grid_size = grid_size
        self.storm_prob = storm_prob
        self.wind_prob = wind_prob
        self.nfz_prob = nfz_prob
        self.rng = rng or np.random.default_rng()

        self.active_storms: List[Storm] = []
        self.active_winds: List[WindCondition] = []
        self.dynamic_nfzs: List[DynamicNoFlyZone] = []
        self.current_step: int = 0

    # ------------------------------------------------------------------ #
    #  Public interface                                                    #
    # ------------------------------------------------------------------ #

    def step(self) -> Dict:
        """Advance one simulation step and return current dynamics state."""
        self._tick_and_expire()
        self._spawn_events()
        self.current_step += 1
        return self.get_state()

    def reset(self) -> None:
        self.active_storms.clear()
        self.active_winds.clear()
        self.dynamic_nfzs.clear()
        self.current_step = 0

    def get_state(self) -> Dict:
        storm_regions = {s.region_id: s.to_dict() for s in self.active_storms}

        wind_state = None
        if self.active_winds:
            dominant = max(self.active_winds, key=lambda w: w.intensity)
            wind_state = (dominant.direction, dominant.intensity)

        nfz_cells: List[Tuple[int, int]] = []
        for nfz in self.dynamic_nfzs:
            nfz_cells.extend(nfz.get_cells(self.grid_size))

        return {
            "storm_regions": storm_regions,
            "wind": wind_state,
            "dynamic_nfz_cells": nfz_cells,
            "num_active_storms": len(self.active_storms),
            "num_active_winds": len(self.active_winds),
            "num_dynamic_nfzs": len(self.dynamic_nfzs),
            "step": self.current_step,
        }

    # ------------------------------------------------------------------ #
    #  Internal mechanics                                                  #
    # ------------------------------------------------------------------ #

    def _tick_and_expire(self) -> None:
        for s in self.active_storms:
            s.tick()
        for w in self.active_winds:
            w.tick()
        self.active_storms = [s for s in self.active_storms if s.is_active()]
        self.active_winds  = [w for w in self.active_winds  if w.is_active()]
        self.dynamic_nfzs  = [z for z in self.dynamic_nfzs  if z.is_active(self.current_step)]

    def _spawn_events(self) -> None:
        if self.rng.random() < self.storm_prob:
            self._spawn_storm()
        if self.rng.random() < self.wind_prob:
            self._spawn_wind()
        if self.rng.random() < self.nfz_prob:
            self._spawn_dynamic_nfz()

    def _spawn_storm(self) -> None:
        g = self.grid_size
        x0 = int(self.rng.integers(0, g - 10))
        y0 = int(self.rng.integers(0, g - 10))
        w  = int(self.rng.integers(5, 16))
        h  = int(self.rng.integers(5, 16))
        self.active_storms.append(Storm(
            region_id=f"storm_{self.current_step}",
            x_range=(x0, min(x0 + w, g - 1)),
            y_range=(y0, min(y0 + h, g - 1)),
            intensity=float(self.rng.uniform(0.3, 1.0)),
            duration=int(self.rng.integers(20, 80)),
        ))

    def _spawn_wind(self) -> None:
        self.active_winds.append(WindCondition(
            direction=random.choice(self.WIND_DIRECTIONS),
            intensity=float(self.rng.uniform(30.0, 130.0)),
            duration=int(self.rng.integers(10, 40)),
        ))

    def _spawn_dynamic_nfz(self) -> None:
        g = self.grid_size
        cx = int(self.rng.integers(5, g - 5))
        cy = int(self.rng.integers(5, g - 5))
        r  = int(self.rng.integers(2, 6))
        dur = int(self.rng.integers(30, 100))
        self.dynamic_nfzs.append(DynamicNoFlyZone(
            zone_id=f"nfz_{self.current_step}",
            center=(cx, cy),
            radius=r,
            start_step=self.current_step,
            end_step=self.current_step + dur,
        ))
