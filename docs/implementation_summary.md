# Implementation Summary — Dron-Intel-OS

> Actualizado: 2026-05-06 — Subfases 1-4 completadas

## Visión General

Smart-Swarm Neuro-Simbólico es un sistema de coordinación autónoma de drones para reparto urbano que fusiona tres paradigmas de IA:

1. **IA Conexionista (DQN):** Aprendizaje por refuerzo profundo con Double-DQN
2. **IA Simbólica (Prolog):** 12 reglas lógicas de seguridad y coordinación
3. **IA de Agentes (Multi-Agent):** 5 drones coordinados en entorno compartido Dec-POMDP

---

## Componentes Implementados

### Backend (Python)

#### `agents/base_agent.py`
- Clase abstracta `BaseAgent` con interfaz `select_action / learn / remember / save_checkpoint / load_checkpoint`
- Dataclass `AgentState` con `to_obs_vector()` → vector 7-dim s^i_t
- Método `to_dict()` para serialización WebSocket
- Logging estructurado por instancia de agente

#### `agents/dqn_agent.py`
- **`PolicyNet`:** Red 7→256→256→8 con LayerNorm (estabilidad en entornos no-estacionarios)
- **`ReplayBuffer`:** Buffer circular FIFO de 100k transiciones con `namedtuple` y método `is_ready()`
- **`DQNAgent`:** Double-DQN completo con:
  - Symbolic action masking integrado en `select_action()` — aplicado en EXPLORACIÓN Y EXPLOTACIÓN
  - ε-greedy restringido a acciones válidas del mask
  - Soft target-network update (τ=0.001, Polyak averaging cada learn step)
  - Gradient clipping (max_norm=1.0) + Huber loss (Smooth L1)
  - `save_checkpoint()` / `load_checkpoint()` — nombre spec-compliant

#### `agents/astar_agent.py` ✨ NUEVO
- **`AStarAgent`:** Planificador baseline para comparación (Sistema A del protocolo)
- A* con heurística Manhattan — `_astar()` con open heap y reconstrucción de path
- `set_target(target)` + `set_obstacles(nfz, storm_cells)` — actualización dinámica con replanning
- Respeta `symbolic_mask`: si la acción planificada está bloqueada, añade celda como obstáculo temporal y replanifica
- No-op para `learn()`, `remember()`, `save_checkpoint()`, `load_checkpoint()`
- `reset_episode()` limpia target y path acumulado

#### `logic/rules.pl`
- 12 reglas Prolog con predicados `assertz/retractall` dinámicos
- Cobertura completa: seguridad crítica, coordinación, entorno dinámico, lógica predictiva
- Comentarios inline con tipo (AM/RS) y peso numérico exacto
- Utilidades: `celda_objetivo/3`, `distancia_manhattan/3`

#### `logic/neuro_symbolic_bridge.py`
- **`NeuroSymbolicBridge`:** Interfaz Python↔Prolog vía pyswip — API spec-compliant
- `sync_state(env_state: dict)` — PÚBLICA; limpia y reaserta hechos Prolog
- `validate_action(agent_id, action, state)` → `(bool, float)` — reglas R1-R12 AM
- `get_action_mask(agent_id, state)` → `np.ndarray[8]`
  - **Optimización:** `sync_state()` una sola vez para las 8 consultas (↓75% overhead)
  - Fail-safe: nunca retorna máscara todo-cero
- `get_reward_modifier(agent_id, state, action)` → `float` — suma R4+R6+R8+R9+R10+R11
- `negotiate_passage(a1, a2, state)` → `int` (0=a1, 1=a2 tiene prioridad)
- `log_decision(level, message)` — API pública, 9 niveles semánticos
- `intervention_count` + `reset_intervention_count()`

#### `environment/city_env.py`
- **`CyberCityEnv`:** Entorno Gymnasium completo (Dec-POMDP)
- Grid 50×50, 5 drones, 10 paquetes (2 médicos exactos + 8 estándar), 4 estaciones
- Observation space: Dict de 5 vectores 7-dim por dron
- Action space: MultiDiscrete([8, 8, 8, 8, 8])
- **`_calculate_rewards()`** ✨ — implementa la fórmula R_total §5 de formal_modeling.md:
  - R_entrega (+150 médico, +50 estándar, +5 recogida)
  - C_movimiento (-0.5/step)
  - P_colisión (-200 por colisión)
  - P_batería (-500 caída libre, -50/step batería<15%)
  - P_simbólica NFZ (-100, resto en training loop via bridge)
- **`_apply_symbolic_masking(bridge, actions)`** ✨ — enmascara acciones para todos los drones en un paso
- `set_symbolic_bridge(bridge)` — inyección para masking automático en `step()`
- `get_blocked_cells()` ✨ — celdas NFZ + storm para AStarAgent
- `get_state_dict()` — incluye `wind` para sync con Prolog (R8)
- `apply_dynamics()` — preserva NFZ estáticas al combinar con dinámicas

#### `environment/dynamics.py`
- **`DynamicsEngine`:** Generador estocástico de eventos climáticos
- `Storm`: región rectangular, duración 20-80 steps, intensidad 0-1
- `WindCondition`: dirección cardinal, intensidad 30-130 km/h
- `DynamicNoFlyZone`: círculo temporal con radio 2-5 celdas
- Probabilidades configurables: `storm_prob=2%`, `wind_prob=3%`, `nfz_prob=1%`

#### `ml_models/demand_predictor.py`
- **`DemandPredictor`:** GradientBoostingRegressor sobre features temporales-espaciales (R²≈0.84)
- Features cíclicas: `hour_sin/cos`, `day_sin/cos` para capturar periodicidad
- **Componente espacial:** 4 hotspots comerciales/residenciales (suma de gaussianas en `_spatial_factor`) → la demanda varía geográficamente, no solo por hora
- `generate_synthetic_data()`: 2000 muestras con patrones temporales + espaciales realistas
- `predict()` → heatmap `(grid_size, grid_size)` de demanda esperada
- `get_high_demand_zones(context, top_k)` → top-K celdas de mayor demanda
- **Integración activa (uso real, no decorativo):** el orquestador lo entrena en `startup()` y cada episodio llama `get_high_demand_zones()` → pasa las zonas a `env.reset(options={"demand_zones": ...})`, de modo que el **60% de los destinos de entrega** se concentra cerca de las zonas predichas. La predicción ML decide la misión que DQN/A* deben resolver, ANTES de cualquier decisión del agente. Expuesto vía `GET /ml/demand`, broadcast WebSocket y overlay en el `DroneMap`.

#### `analysis/metrics.py`
- **`MetricsCollector`:** Persistencia CSV + análisis en tiempo real
- `EpisodeRecord`: dataclass con 12 campos de telemetría (protocolo §4.1)
- `get_summary()`, `get_comparison_table()`, `get_learning_curve()`
- `to_live_json()`: formato compacto para WebSocket broadcast

#### `main.py`
- **FastAPI** con CORS habilitado y logging estructurado
- Import paths robustos: funciona con `python main.py` (desde `backend/`) y `uvicorn backend.main:app` (desde raíz)
- **WebSocket** `/ws` con `ConnectionManager` tolerante a desconexiones
- Loop de entrenamiento asíncrono con `asyncio.sleep(0)` para no bloquear el event loop
- Broadcast cada 25 steps: posiciones, baterías, eventos dinámicos, máscaras
- Broadcast cada 10 episodios: métricas completas, log simbólico
- Soporte de los 3 sistemas: `astar`, `dqn`, `neuro_dqn` con `_build_agents()`
- AStarAgent: `set_target()` + `set_obstacles(get_blocked_cells())` en cada step
- Endpoints REST: `/health`, `/metrics/*`, `/training/start`, `/training/stop`, `/drone-state`, `/symbolic-log`
- Aliases spec-compliant: `/start-training`, `/stop-training`

---

## Decisiones Técnicas Clave

### 1. Soft-update target network (τ=0.001) vs hard-copy periódico
Polyak averaging en cada learn step estabiliza el entrenamiento en entornos con distribuciones de recompensa muy dispares (-500 a +150). La copia dura periódica genera "escalones" en la curva de pérdida.

### 2. LayerNorm en la red neuronal
Preferida sobre BatchNorm en RL: normaliza por muestra (no por batch), tolerando la distribución no-estacionaria de los estados durante el entrenamiento. Crítico porque la distribución de s^i_t cambia a medida que mejora la política.

### 3. `sync_state()` optimizado en `get_action_mask()`
En la versión anterior, `get_action_mask()` llamaba `validate_action()` 8 veces, cada una re-sincronizando el estado (8 `retractall` + `assertz`). Nueva versión: `sync_state()` una vez → 8 llamadas a `_check_action_rules()` (sin sync). Reduce latencia de ~4ms a ~0.5ms por máscara.

### 4. Action masking en exploración ε-greedy (AMBAS ramas)
Error común: aplicar la máscara solo durante explotación. Si el agente explora libremente violando reglas, aprende distribuciones Q inválidas que tardan en corregirse. La implementación filtra acciones inválidas antes del split aleatorio/greedy.

### 5. `_calculate_rewards()` como método puro
Separar el cálculo de recompensa de los side effects del estado facilita:
- Testing unitario de la función de recompensa sin ejecutar el entorno completo
- Verificación formal contra la fórmula R_total del modelo
- Futura extensión para PER (Prioritized Experience Replay)

### 6. AStarAgent con replanning por máscara simbólica
Cuando la máscara Prolog bloquea la acción planificada, A* añade la celda como obstáculo temporal y replanifica. Esto integra las restricciones simbólicas en el baseline sin necesidad de modificar las reglas Prolog.

### 7. GradientBoosting sobre RandomForest para demand_predictor
GBR captura mejor las no-linealidades de los picos de demanda horaria (patrón bimodal mañana/tarde). Costo: ~2× más lento en training (irrelevante en offline).

### 8. Rebalanceo de recompensa para convergencia (2026-05-20)
Diagnóstico: con el costo de batería previo (0.75/move) los drones morían en ~133 pasos y la penalización de muerte (−200) más el grind por-paso dominaban la señal — una política aleatoria igualaba al "entrenado" (reward ≈ −6800). Correcciones: (a) batería más eficiente (0.4/move, recarga 40%/step → ~250 movimientos de autonomía); (b) entrega +200/+100, muerte −80, colisión −30, grind −0.05; (c) **shaping basado en potencial** (proximidad simétrica ±0.5·Δdist) que no introduce deriva negativa por explorar y preserva la política óptima (Ng et al. 1999); (d) ε-decay 0.995→0.99. Resultado: entregas suben monótonamente 0→4/10 y reward −1443→−556 en 150 ep mientras ε decae.

### 9. Persistencia de entrenamiento (checkpoints) (2026-05-20)
Los pesos DQN, optimizer, ε y `learn_step` se guardan en `data/checkpoints/{system}_drone_{i}.pt` al terminar el loop. El entrenamiento admite `mode=resume` (carga checkpoints → continúa con conocimiento previo) o `mode=scratch` (pesos nuevos, ε=1.0). El histórico de métricas (`MetricsCollector`) se carga automáticamente del CSV al arrancar. Endpoints: `GET /training/status` (qué hay guardado por sistema) y `POST /training/delete-data` (borrado total o por sistema, bloqueado durante entrenamiento activo). A* no persiste (no aprende).

### 11. Reproducibilidad, reporte estadístico y visualización de hazards (2026-05-20)
- **Semilla reproducible:** `POST /training/start?seed=N` propaga `seed+episodio` a `env.reset()` y `dynamics.reset()`, dando condiciones idénticas (layout de paquetes + secuencia de eventos) a los 3 sistemas → comparación justa (protocolo §2). `DynamicsEngine.reset(seed)` reinicializa su RNG. Toggle "Semilla fija" en el `ControlPanel`.
- **Reporte experimental:** `MetricsCollector.get_experimental_report()` → media±σ, **IC 95 %**, mejor entrega, episodio de convergencia, totales de violaciones/colisiones por sistema. Expuesto en `GET /metrics/report` y renderizado como tabla en la pestaña *Histórico*. `get_learning_curve()` añade `reward_smooth` (media móvil) para evidenciar la tendencia asintótica.
- **Hazards en vivo:** el broadcast `step_update` incluye `no_fly_zones` (celdas) y `storm_regions` (rectángulos); `DroneMap` los pinta en tiempo real (antes recibía arrays vacíos).

### 10. Suite de tests y hardening de `rules.pl` (2026-05-20)
Suite `tests/` (116 tests, `pytest tests` → 116 passed) con `conftest.py` que añade `backend/` a `sys.path` (mismos imports que producción) y detección automática de Prolog (los tests marcados `prolog` se omiten si SWI-Prolog no está, en vez de fallar). Al escribir los tests del bridge se detectó un **bug latente de robustez**: los predicados de `rules.pl` incluían `format/2` (logging) como último goal de la conjunción, así que si stdout no era escribible (captura de pytest, ciertos despliegues) el `format` fallaba y **toda la regla fallaba en silencio** → el masking dejaba de bloquear. Corrección: envolver los 20 `format/2` en `ignore/1`, separando logging de lógica. Verificado que el masking de la app real no cambia (NFZ sigue bloqueando). Decisión de alcance: la convergencia plena del DQN se valida con corridas largas documentadas en el checklist, no en la suite (sería lenta y estocástica); los tests de integración verifican que el **mecanismo** de aprendizaje funciona (gradiente fluye, Q-values se actualizan, ε decae).

---

## Funcionalidades

| Componente | Estado | Notas |
|---|---|---|
| Estructura de carpetas | ✅ | |
| Modelado formal (Dec-POMDP) | ✅ | Ver `formal_modeling.md` |
| Motor Prolog (12 reglas) | ✅ | Predicados dinámicos integrados |
| Agente DQN (Double-DQN) | ✅ | Soft-update + masking completo |
| Agente A* (baseline) | ✅ | Nuevo: replanning + mask |
| Puente Neuro-Simbólico | ✅ | API spec-compliant, sync optimizado |
| Entorno Gymnasium | ✅ | `_calculate_rewards`, `_apply_symbolic_masking` |
| Motor de Dinámica | ✅ | Tormentas, viento, NFZ dinámicas |
| Predictor de demanda | ✅ | GBR (R²≈0.84) + hotspots espaciales; **integrado en el loop**: sesga destinos por episodio, expuesto en API/WS/mapa |
| Métricas (Pandas) | ✅ | CSV persistence + analytics |
| API FastAPI + WebSocket | ✅ | 3 sistemas + aliases spec |
| DroneMap (SVG) | ✅ | |
| LiveStats (Recharts) | ✅ | 4 gráficas en vivo |
| RuleTerminal | ✅ | Log simbólico formateado |
| useSocket hook | ✅ | Reconexión automática |
| Entrenamiento multi-sistema | ✅ | DQN convergente (entregas 0→4/10); persistencia resume/scratch + borrado |
| Persistencia de checkpoints | ✅ | `data/checkpoints/`; modos resume/scratch; `/training/status` y `/training/delete-data` |
| Tests unitarios | ✅ | Suite `tests/` con 116 tests (env, DQN, A*, dinámica, métricas, ML, bridge); `pytest tests` → 116 passed; cobertura backend ≈65% |
| Docker/deployment | 📋 | Pendiente |
| Frontend routing/App.tsx | 📋 | Componentes listos, falta composición |

---

## Próximos Pasos Inmediatos

1. **Instalar dependencias:** `pip install -r requirements.txt` (requiere SWI-Prolog previo)
2. **Test de integración:** `python -c "from logic.neuro_symbolic_bridge import NeuroSymbolicBridge; b = NeuroSymbolicBridge('logic/rules.pl'); print('OK')"` desde `backend/`
3. **Entrenamiento comparativo:** 200 episodios × 3 sistemas → generar `training_logs.csv` de referencia
4. **Validar hipótesis H3 y H4** con los datos generados (Mann-Whitney U, Cohen's d)
5. **App.tsx principal:** Conectar DroneMap + LiveStats + RuleTerminal con `useSocket` y estado global
