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
        # Pending rows accumulated between saves — flushed to _df on save() or
        # when an analytics query forces a flush. Using a list avoids the O(n²)
        # cost of pd.concat on every episode.
        self._pending: List[Dict] = []
        self._load()

    def _flush_pending(self) -> None:
        if not self._pending:
            return
        new_rows = pd.DataFrame(self._pending, columns=COLUMNS)
        self._df = pd.concat([self._df, new_rows], ignore_index=True)
        self._pending.clear()

    # ------------------------------------------------------------------ #
    #  Ingestion                                                           #
    # ------------------------------------------------------------------ #

    def record_episode(self, record: EpisodeRecord) -> None:
        self._pending.append({
            "episode":               record.episode,
            "system":                record.system,
            "total_reward":          record.total_reward,
            "deliveries_completed":  record.deliveries_completed,
            "deliveries_total":      record.deliveries_total,
            "success_rate":          record.success_rate,
            "rule_violations":       record.rule_violations,
            "collisions":            record.collisions,
            "battery_failures":      record.battery_failures,
            "steps":                 record.steps,
            "avg_battery_remaining": record.avg_battery_remaining,
            "symbolic_interventions": record.symbolic_interventions,
            "timestamp":             record.timestamp,
        })

    def save(self) -> None:
        self._flush_pending()
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
        self._flush_pending()
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
            return len(self._df) + len(self._pending)
        return int((self._df["system"] == system).sum()) + sum(
            1 for r in self._pending if r["system"] == system
        )

    # ------------------------------------------------------------------ #
    #  Analytics                                                           #
    # ------------------------------------------------------------------ #

    def get_summary(
        self, system: Optional[str] = None, last_n: int = 100
    ) -> Dict:
        self._flush_pending()
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
            # hits.index[0] es la ETIQUETA de fila del primer cruce del umbral.
            # Usar .loc (label-based), no .iloc (posicional): cuando df es un slice
            # filtrado por sistema, las etiquetas no son contiguas desde 0 y .iloc
            # lanzaría IndexError out-of-bounds.
            return int(df.loc[hits.index[0], "episode"])
        return None

    def get_comparison_table(self) -> pd.DataFrame:
        self._flush_pending()
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
        self._flush_pending()
        df = (
            self._df[self._df["system"] == system][["episode", metric]]
            .copy()
            .reset_index(drop=True)
        )
        df["rolling_avg"] = df[metric].rolling(window=20, min_periods=1).mean()
        # reward_smooth: media móvil de recompensa para evidenciar la tendencia
        # asintótica de convergencia (Paso 2 de la verificación automatizada).
        reward_col = self._df[self._df["system"] == system]["total_reward"].reset_index(drop=True)
        df["reward_smooth"] = reward_col.rolling(window=20, min_periods=1).mean()
        return df

    def get_experimental_report(self) -> Dict:
        """Reporte experimental con significancia estadística por sistema.

        Para cada sistema calcula, sobre todos sus episodios registrados:
          n, media ± desviación estándar e intervalo de confianza al 95 % de la
          recompensa y la tasa de éxito, mejor episodio, episodio de convergencia,
          y totales de violaciones/colisiones. Apto para la sección de resultados
          del reporte técnico (Criterio 10).
        """
        self._flush_pending()
        report: Dict = {"systems": {}, "generated_episodes": int(len(self._df))}
        for system in sorted(self._df["system"].dropna().unique()):
            d = self._df[self._df["system"] == system]
            n = len(d)
            if n == 0:
                continue
            reward = d["total_reward"].astype(float)
            success = d["success_rate"].astype(float)
            # IC 95 % aproximado (normal): media ± 1.96 · sd/√n
            def ci95(series):
                sd = float(series.std(ddof=1)) if n > 1 else 0.0
                half = 1.96 * sd / (n ** 0.5) if n > 0 else 0.0
                return [round(float(series.mean()) - half, 3),
                        round(float(series.mean()) + half, 3)]
            report["systems"][system] = {
                "n_episodes":          int(n),
                "reward_mean":         round(float(reward.mean()), 2),
                "reward_std":          round(float(reward.std(ddof=1)) if n > 1 else 0.0, 2),
                "reward_ci95":         ci95(reward),
                "success_rate_mean":   round(float(success.mean()), 4),
                "success_rate_std":    round(float(success.std(ddof=1)) if n > 1 else 0.0, 4),
                "success_rate_ci95":   ci95(success),
                "best_deliveries":     int(d["deliveries_completed"].max()),
                "total_rule_violations": int(d["rule_violations"].sum()),
                "total_collisions":    int(d["collisions"].sum()),
                "convergence_episode": self._convergence_episode(d.reset_index(drop=True)),
            }
        return report

    def to_live_json(self, system: str, last_n: int = 50) -> Dict:
        """Compact format consumed by the LiveStats WebSocket broadcast."""
        self._flush_pending()
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
