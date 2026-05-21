import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class EpisodeRecord:
    episode: int
    system: str                    # 'astar' | 'dqn' | 'neuro_dqn'
    total_reward: float
    deliveries_completed: int
    deliveries_total: int
    rule_violations: int
    collisions: int
    battery_failures: int
    steps: int
    avg_battery_remaining: float
    symbolic_interventions: int    # Prolog mask activations this episode
    timestamp: str = field(
        default_factory=lambda: datetime.now().isoformat(timespec="seconds")
    )

    @property
    def success_rate(self) -> float:
        return (
            self.deliveries_completed / self.deliveries_total
            if self.deliveries_total > 0 else 0.0
        )


COLUMNS = [
    "episode", "system", "total_reward",
    "deliveries_completed", "deliveries_total", "success_rate",
    "rule_violations", "collisions", "battery_failures",
    "steps", "avg_battery_remaining", "symbolic_interventions",
    "timestamp",
]


class MetricsCollector:
    """
    Collects per-episode training telemetry and exposes analytics via Pandas.

    Persists data to CSV so the training can be resumed across sessions.
    Provides learning-curve data and system-comparison tables for the
    LiveStats React component.
    """

    def __init__(self, log_path: str = "data/training_logs.csv") -> None:
        self.log_path = log_path
        self._df: pd.DataFrame = pd.DataFrame(columns=COLUMNS)
        self._load()

    # ------------------------------------------------------------------ #
    #  Ingestion                                                           #
    # ------------------------------------------------------------------ #

    def record_episode(self, record: EpisodeRecord) -> None:
        row = {
            "episode": record.episode,
            "system": record.system,
            "total_reward": record.total_reward,
            "deliveries_completed": record.deliveries_completed,
            "deliveries_total": record.deliveries_total,
            "success_rate": record.success_rate,
            "rule_violations": record.rule_violations,
            "collisions": record.collisions,
            "battery_failures": record.battery_failures,
            "steps": record.steps,
            "avg_battery_remaining": record.avg_battery_remaining,
            "symbolic_interventions": record.symbolic_interventions,
            "timestamp": record.timestamp,
        }
        self._df = pd.concat(
            [self._df, pd.DataFrame([row])], ignore_index=True
        )

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
        self._df.to_csv(self.log_path, index=False)

    def _load(self) -> None:
        if os.path.exists(self.log_path):
            try:
                self._df = pd.read_csv(self.log_path)
            except Exception:
                self._df = pd.DataFrame(columns=COLUMNS)

    def clear(self, system: Optional[str] = None) -> int:
        """Elimina registros de entrenamiento y reescribe el CSV.

        Args:
            system: si se indica, borra solo las filas de ese sistema;
                    si es None, borra todo el histórico.

        Returns:
            Número de filas eliminadas.
        """
        before = len(self._df)
        if system is None:
            self._df = pd.DataFrame(columns=COLUMNS)
        else:
            self._df = self._df[self._df["system"] != system].reset_index(drop=True)
        removed = before - len(self._df)
        self.save()
        return removed

    def count(self, system: Optional[str] = None) -> int:
        """Número de episodios registrados (opcionalmente por sistema)."""
        if system is None:
            return len(self._df)
        return int((self._df["system"] == system).sum())

    # ------------------------------------------------------------------ #
    #  Analytics                                                           #
    # ------------------------------------------------------------------ #

    def get_summary(
        self, system: Optional[str] = None, last_n: int = 100
    ) -> Dict:
        df = self._df if system is None else self._df[self._df["system"] == system]
        df = df.tail(last_n)
        if df.empty:
            return {}
        return {
            "mean_reward":              float(df["total_reward"].mean()),
            "std_reward":               float(df["total_reward"].std()),
            "mean_success_rate":        float(df["success_rate"].mean()),
            "total_rule_violations":    int(df["rule_violations"].sum()),
            "total_collisions":         int(df["collisions"].sum()),
            "mean_battery_remaining":   float(df["avg_battery_remaining"].mean()),
            "total_symbolic_ops":       int(df["symbolic_interventions"].sum()),
            "episodes_recorded":        len(df),
            "convergence_episode":      self._convergence_episode(df),
        }

    def _convergence_episode(
        self, df: pd.DataFrame, threshold: float = 0.9, window: int = 10
    ) -> Optional[int]:
        rolling = df["success_rate"].rolling(window=window, min_periods=window).mean()
        hits = rolling[rolling >= threshold]
        if not hits.empty:
            return int(df.iloc[hits.index[0]]["episode"])
        return None

    def get_comparison_table(self) -> pd.DataFrame:
        rows = []
        for system in self._df["system"].unique():
            summary = self.get_summary(system=system, last_n=200)
            summary["system"] = system
            rows.append(summary)
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).set_index("system")

    def get_learning_curve(
        self, system: str, metric: str = "success_rate"
    ) -> pd.DataFrame:
        df = (
            self._df[self._df["system"] == system][["episode", metric]]
            .copy()
            .reset_index(drop=True)
        )
        df["rolling_avg"] = df[metric].rolling(window=20, min_periods=1).mean()
        return df

    def to_live_json(self, system: str, last_n: int = 50) -> Dict:
        """Compact format consumed by the LiveStats WebSocket broadcast."""
        df = self._df[self._df["system"] == system].tail(last_n)
        return {
            "episodes":              df["episode"].tolist(),
            "rewards":               df["total_reward"].round(2).tolist(),
            "success_rates":         df["success_rate"].round(4).tolist(),
            "rule_violations":       df["rule_violations"].tolist(),
            "symbolic_interventions": df["symbolic_interventions"].tolist(),
            "avg_battery":           df["avg_battery_remaining"].round(1).tolist(),
        }

    def print_report(self, system: Optional[str] = None) -> None:
        s = self.get_summary(system=system)
        if not s:
            print("No data recorded yet.")
            return
        label = system or "ALL SYSTEMS"
        print(f"\n{'='*50}")
        print(f"  METRICS REPORT — {label}")
        print(f"{'='*50}")
        print(f"  Episodes recorded   : {s['episodes_recorded']}")
        print(f"  Mean reward         : {s['mean_reward']:.2f} ± {s['std_reward']:.2f}")
        print(f"  Mean success rate   : {s['mean_success_rate']*100:.1f}%")
        print(f"  Total violations    : {s['total_rule_violations']}")
        print(f"  Total collisions    : {s['total_collisions']}")
        print(f"  Mean battery left   : {s['mean_battery_remaining']:.1f}%")
        print(f"  Symbolic ops        : {s['total_symbolic_ops']}")
        print(f"  Convergence episode : {s.get('convergence_episode', 'N/A')}")
        print(f"{'='*50}\n")
