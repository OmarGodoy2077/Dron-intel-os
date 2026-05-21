# Suite de Tests — Dron-Intel-OS

Tests unitarios e integrados del backend Neuro-Simbólico. Verifican que cada
componente cumple su contrato y que las reglas simbólicas inyectan el
comportamiento esperado (Paso 1 de la verificación automatizada de la rúbrica).

## Ejecutar

Desde la **raíz del proyecto** (no desde `backend/`):

```bash
pytest                      # toda la suite
pytest -m "not slow"        # omite tests de aprendizaje (más rápido)
pytest -m "not prolog"      # omite tests que requieren SWI-Prolog
pytest --cov=backend        # con reporte de cobertura
pytest tests/test_city_env.py -v   # un módulo concreto, verboso
```

> Los imports usan las mismas rutas que producción (`from agents... import`).
> `conftest.py` añade `backend/` a `sys.path` automáticamente.

## Cobertura por módulo

| Archivo | Qué verifica |
|---|---|
| `test_city_env.py` | Espacios ℝ¹¹/8-acciones, determinismo por semilla, fórmula R_total, action masking (R1/R3/R5/R2), economía de batería, inyección de zonas de demanda ML |
| `test_dqn_agent.py` | ReplayBuffer circular, PolicyNet, ε-greedy enmascarado, learn(), soft-update, decay de ε, checkpoints (round-trip) |
| `test_astar_agent.py` | A* (ruta óptima, evitación de obstáculos, sin-ruta), replanning, respeto de máscara |
| `test_dynamics.py` | Spawn/expiración estocástica de tormentas/viento/NFZ, estado, reset, viento dominante |
| `test_metrics.py` | EpisodeRecord, persistencia CSV, análisis Pandas (rolling, comparación), clear/count |
| `test_demand_predictor.py` | Datos sintéticos con estructura espacial, entrenamiento (R²), zonas geográficamente significativas, fallback |
| `test_neuro_symbolic_bridge.py` | Pesos de las 12 reglas; con Prolog: R1 NFZ, R2 batería, R3 colisión, R11 negociación, contador de intervenciones |
| `test_integration.py` | Mini-loop env+DQN+masking+learning, propiedad de seguridad (nunca entra a NFZ), pipeline ML→entorno, baseline A*, flujo de gradiente |

## Marcadores (`pytest.ini`)

- `prolog` — requiere SWI-Prolog + pyswip. Se **omiten automáticamente** si no
  están instalados (no fallan). Verifican el motor lógico real.
- `slow` — ejecutan pasos de aprendizaje (gradiente). Excluibles con `-m "not slow"`.
- `integration` — integración entre múltiples módulos.

## Nota de robustez

Los predicados de `rules.pl` envuelven sus `format/2` (logging) en `ignore/1`,
de modo que la verdad lógica de una regla nunca depende de que la salida estándar
sea escribible. Esto evita que el masking falle silenciosamente cuando stdout está
redirigido (p. ej. bajo captura de pytest o en ciertos entornos de despliegue).
