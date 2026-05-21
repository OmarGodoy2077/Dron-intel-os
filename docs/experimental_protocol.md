# Protocolo Experimental — A* vs DQN vs Neuro-DQN

## 1. Hipótesis

| ID | Hipótesis | Sistema |
|---|---|---|
| H1 | A* produce rutas óptimas en entorno estático pero degrada con dinamismo | A* |
| H2 | DQN puro aprende pero comete violaciones críticas de seguridad | DQN |
| H3 | Neuro-DQN converge más rápido y mantiene 0 violaciones de seguridad | Neuro-DQN |
| H4 | Las intervenciones simbólicas disminuyen con el entrenamiento (el agente internaliza las reglas) | Neuro-DQN |

---

## 2. Configuración Experimental

### 2.1 Entorno

| Parámetro | Valor |
|---|---|
| Grid size | 50 × 50 celdas |
| Número de drones | 5 |
| Número de paquetes | 10 (2 médicos, 8 estándar) |
| Estaciones de carga | 4 |
| Zonas no-fly estáticas | 5 (radio ~2 celdas) |
| Probabilidad tormenta/step | 2% |
| Probabilidad viento/step | 3% |
| Pasos máximos/episodio | 500 |
| Episodios de evaluación | 200 entrenamiento + 50 evaluación |

### 2.2 Hiperparámetros DQN

| Parámetro | Valor |
|---|---|
| Arquitectura red | Linear(7→256) → LayerNorm → ReLU × 2 → Linear(→8) |
| Learning rate | $1 \times 10^{-3}$ |
| Discount factor $\gamma$ | $0.99$ |
| $\varepsilon$ inicial | $1.0$ |
| $\varepsilon$ mínimo | $0.05$ |
| Decay por episodio | $0.995$ |
| Replay buffer | $100{,}000$ transiciones |
| Batch size | $64$ |
| Target update freq | $10$ pasos |
| Optimizador | Adam + grad clip (max_norm=1.0) |

### 2.3 Configuración A*

- Heurística: distancia Manhattan al destino.
- Replanning: cada vez que una NFZ o tormenta bloquea la ruta actual.
- Costo: número de celdas recorridas (no considera batería).

---

## 3. Sistemas a Comparar

### Sistema A: A* Clásico
- Búsqueda heurística A* con distancia Manhattan.
- Sin aprendizaje; replanning completo ante cambios.
- **Limitación:** no considera batería, colisiones multi-agente ni coordinación.

### Sistema B: DQN Puro
- Red Double-DQN (arquitectura idéntica a Neuro-DQN).
- Sin motor simbólico; solo penalizaciones en reward.
- **Permite** acciones a NFZ, zonas de tormenta, etc. (el agente aprende a evitarlas lentamente).

### Sistema C: Neuro-DQN (Propuesta)
- DQN + Prolog: 12 reglas activas.
- Action masking: R1, R2, R3, R5, R7, R12.
- Reward shaping: R4, R6, R8, R9, R10, R11.

---

## 4. Métricas de Evaluación

### 4.1 Métricas primarias (Pandas DataFrames)

```python
# Estructura de training_logs.csv
columns = [
    "episode",                 # int: número de episodio
    "system",                  # str: 'astar' | 'dqn' | 'neuro_dqn'
    "total_reward",            # float: recompensa acumulada
    "deliveries_completed",    # int: paquetes entregados
    "deliveries_total",        # int: paquetes totales
    "success_rate",            # float: deliveries_completed / deliveries_total
    "rule_violations",         # int: entradas a NFZ, colisiones, vuelos en tormenta
    "collisions",              # int: encuentros entre drones
    "battery_failures",        # int: drones con batería = 0
    "steps",                   # int: pasos hasta terminar el episodio
    "avg_battery_remaining",   # float: % batería promedio al final
    "symbolic_interventions",  # int: activaciones del motor Prolog
    "timestamp",               # str: ISO 8601
]
```

### 4.2 Análisis por métrica

| Métrica | Cálculo | Objetivo |
|---|---|---|
| **Tasa de éxito** | `deliveries_completed / deliveries_total` | Neuro-DQN ≥ 90% en ep 200 |
| **Violaciones de reglas** | `rule_violations.sum()` | Neuro-DQN = 0 (por diseño) |
| **Convergencia** | Primer episodio con rolling(20).mean ≥ 0.9 | Neuro-DQN < DQN |
| **Colisiones** | `collisions.sum()` | Neuro-DQN = 0 |
| **Energía usada** | `100 - avg_battery_remaining` | Minimizar |
| **Intervenciones simbólicas** | `symbolic_interventions` serie temporal | Debe decrecer (aprendizaje) |

---

## 5. Procedimiento de Ejecución

```python
# Bloque de ejecución comparativa
from backend.environment.city_env import CyberCityEnv
from backend.agents.dqn_agent import DQNAgent
from backend.logic.neuro_symbolic_bridge import NeuroSymbolicBridge
from backend.analysis.metrics import MetricsCollector

env = CyberCityEnv(grid_size=50, num_drones=5)

for system in ["astar", "dqn", "neuro_dqn"]:
    metrics = MetricsCollector(f"data/logs_{system}.csv")
    run_training(system=system, episodes=200, env=env, metrics=metrics)
    metrics.print_report(system=system)

# Tabla comparativa final
collector = MetricsCollector("data/training_logs.csv")
print(collector.get_comparison_table().to_markdown())
```

---

## 6. Análisis Estadístico

Para comparación rigurosa entre sistemas:

- **Mann-Whitney U test** (no paramétrico) sobre distribuciones de success_rate.
- **Effect size (Cohen's d)** para reward total.
- **Curvas de aprendizaje** con intervalo de confianza al 95% (bootstrapping sobre 5 seeds).

```python
from scipy import stats

dqn_rates    = df[df.system == "dqn"]["success_rate"]
neuro_rates  = df[df.system == "neuro_dqn"]["success_rate"]

stat, p_val = stats.mannwhitneyu(neuro_rates, dqn_rates, alternative="greater")
print(f"H3 p-value: {p_val:.4f} ({'ACCEPTED' if p_val < 0.05 else 'REJECTED'})")
```

---

## 7. Resultados Esperados

| Sistema | Éxito (ep200) | Violaciones | Convergencia | Colisiones |
|---|---|---|---|---|
| A* | ~65% | ~5% (replanning lag) | N/A | ~2/ep |
| DQN | ~75% ± 12% | ~15% | ep ~120 | ~3/ep |
| **Neuro-DQN** | **~92% ± 5%** | **0%** | **ep ~80** | **0/ep** |

Las intervenciones simbólicas deben mostrar tendencia decreciente, indicando que el DQN internaliza las restricciones en su función Q, requiriendo cada vez menos supervisión del motor Prolog.
