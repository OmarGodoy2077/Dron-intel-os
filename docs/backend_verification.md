# Backend Verification Guide — Dron-Intel-OS

> Checklist secuencial para verificar que el backend está 100 % funcional  
> antes de proceder con la implementación del frontend.

---

## Requisitos previos del sistema

Antes de instalar paquetes Python, estos tres componentes deben estar presentes:

| Componente | Versión mínima | Verificación |
|---|---|---|
| Python | 3.10 | `python --version` |
| SWI-Prolog | 9.x | `swipl --version` |
| Node.js | 18 LTS | `node --version` (para el frontend, instalar ya) |

### Instalar SWI-Prolog (si no está)

```powershell
# Windows (winget)
winget install SWI-Prolog.SWI-Prolog

# Verificar que swipl está en PATH después de instalarlo
swipl --version
```

> **Verificado en este entorno:** SWI-Prolog 10.0.2 instalado en
> `C:\Program Files\swipl\bin\swipl.exe`. El instalador de winget NO agrega
> el PATH automáticamente en todas las versiones de Windows. Si `swipl --version`
> falla, agregar manualmente en las variables de entorno del sistema:
>
> ```powershell
> # Solo para la sesión actual (temporal):
> $env:PATH = "C:\Program Files\swipl\bin;" + $env:PATH
>
> # Permanente (requiere reiniciar terminal):
> [System.Environment]::SetEnvironmentVariable(
>     "PATH",
>     "C:\Program Files\swipl\bin;" + [System.Environment]::GetEnvironmentVariable("PATH","Machine"),
>     "Machine"
> )
> ```

---

## Paso 1 — Instalar dependencias Python

```powershell
cd "w:\Proyectos FullStack\Dron-intel-os\backend"

# Crear entorno virtual (recomendado)
python -m venv .venv
.venv\Scripts\Activate.ps1

# Instalar todas las dependencias
pip install -r requirements.txt
```

**Verificar que la instalación fue exitosa:**

```powershell
pip show torch gymnasium fastapi pyswip pandas numpy scikit-learn scipy uvicorn
```

Todos los paquetes deben mostrar una versión. Si alguno falla, instalarlo individualmente:

```powershell
pip install pyswip>=0.2.10   # Requiere SWI-Prolog instalado primero
```

---

## Paso 2 — Verificación de imports por módulo

Ejecutar cada bloque desde `backend/` con el entorno virtual activo.

### 2.1 Agentes

```powershell
cd "w:\Proyectos FullStack\Dron-intel-os\backend"

python -c "
from agents.base_agent import BaseAgent, AgentState
import numpy as np
s = AgentState(
    agent_id='drone_0', position=(5, 10, 1),
    battery=80.0, cargo=None, climate='clear', neighbor_occupancy={}
)
vec = s.to_obs_vector()
assert vec.shape == (7,), f'Esperado (7,), obtenido {vec.shape}'
print('BaseAgent + AgentState OK -- vector:', vec)
"
```

```powershell
python -c "
from agents.dqn_agent import DQNAgent
import numpy as np
agent = DQNAgent(agent_id='drone_0', state_dim=7, action_dim=8)
state = np.random.rand(7).astype('float32')
mask  = np.ones(8, dtype='float32')
action = agent.select_action(state, symbolic_mask=mask)
assert 0 <= action < 8, f'Acción fuera de rango: {action}'
print('DQNAgent OK — acción seleccionada:', action)
"
```

```powershell
python -c "
from agents.astar_agent import AStarAgent
import numpy as np
agent = AStarAgent(agent_id='drone_0', grid_size=50)
agent.set_target((25, 25))
agent.set_obstacles([(10, 10), (10, 11)])
# select_action espera np.ndarray 7-dim, NO un dict
state = np.array([0.0, 0.0, 1.0, 80.0, 0.0, 0.0, 0.0], dtype='float32')
mask  = np.ones(8, dtype='float32')
action = agent.select_action(state, symbolic_mask=mask)
assert 0 <= action < 8, f'Accion fuera de rango: {action}'
print('AStarAgent OK -- accion planificada:', action)
"
```

### 2.2 Motor Simbólico (Prolog)

```powershell
python -c "
from logic.neuro_symbolic_bridge import NeuroSymbolicBridge
bridge = NeuroSymbolicBridge('logic/rules.pl')
print('NeuroSymbolicBridge instanciado OK')
print('  intervention_count:', bridge.intervention_count)
"
```

Si aparece `ImportError: libswipl.dll not found`, SWI-Prolog no está en PATH.

```powershell
# Diagnóstico de PATH en Windows
python -c "
import os
path_dirs = os.environ['PATH'].split(';')
swi_dirs  = [d for d in path_dirs if 'swipl' in d.lower() or 'prolog' in d.lower()]
print('Directorios SWI en PATH:', swi_dirs or 'NINGUNO — agregar manualmente')
"
```

### 2.3 Entorno de simulación

```powershell
python -c "
from environment.city_env import CyberCityEnv
env = CyberCityEnv(grid_size=50, num_drones=5)
obs, info = env.reset()
assert len(obs) == 5, f'Esperadas 5 obs, obtenidas {len(obs)}'
for k, v in obs.items():
    assert v.shape == (7,), f'{k}: shape {v.shape}'
# get_state_dict requiere drone_idx (0..4), NO se llama sin argumentos
blocked  = env.get_blocked_cells()
state_d  = env.get_state_dict(0)
print('CyberCityEnv OK -- keys obs:', list(obs.keys()))
print('  blocked_cells:', len(blocked), '  state_dict keys:', list(state_d.keys()))
"
```

### 2.4 Dinámica climática

```powershell
python -c "
from environment.dynamics import DynamicsEngine
dyn = DynamicsEngine(grid_size=50)
from environment.city_env import CyberCityEnv
env = CyberCityEnv()
nfzs = dyn.step()
print('DynamicsEngine.step() OK — NFZs activas:', len(nfzs))
"
```

### 2.5 Métricas

```powershell
python -c "
from analysis.metrics import MetricsCollector, EpisodeRecord
mc = MetricsCollector()
# Firma real de EpisodeRecord (11 campos obligatorios):
rec = EpisodeRecord(
    episode=1, system='test', total_reward=100.0,
    deliveries_completed=4, deliveries_total=10, rule_violations=0,
    collisions=0, battery_failures=0, steps=120,
    avg_battery_remaining=70.0, symbolic_interventions=5
)
mc.record_episode(rec)   # metodo correcto: record_episode(), NO add()
summary = mc.get_summary()
print('MetricsCollector OK -- summary keys:', list(summary.keys()))
"
```

### 2.6 Predictor de demanda

```powershell
python -c "
from ml_models.demand_predictor import DemandPredictor
dp = DemandPredictor(grid_size=50)
dp.train()
heatmap = dp.predict(hour=9, day_of_week=1)
assert heatmap.shape == (50, 50), f'Shape esperado (50,50), obtenido {heatmap.shape}'
print('DemandPredictor OK — heatmap shape:', heatmap.shape)
"
```

---

## Paso 3 — Test de integración Neuro-Simbólica

Este test verifica el pipeline completo: entorno → bridge → máscaras → recompensas.

```powershell
python -c "
import numpy as np
from environment.city_env import CyberCityEnv
from logic.neuro_symbolic_bridge import NeuroSymbolicBridge

env    = CyberCityEnv(grid_size=50, num_drones=5)
bridge = NeuroSymbolicBridge('logic/rules.pl')

obs, _ = env.reset()

# sync_state requiere el dict de un drone especifico (get_state_dict(drone_idx))
state_dict = env.get_state_dict(0)
bridge.sync_state(state_dict)
print('sync_state() OK')

# get_action_mask — pasar state_dict individual por dron
for i in range(5):
    sd   = env.get_state_dict(i)
    mask = bridge.get_action_mask(f'drone_{i}', sd)
    assert mask.shape == (8,),                     f'Mask shape incorrecto: {mask.shape}'
    assert mask.sum() > 0,                          f'drone_{i}: mascara todo-cero'
    assert set(mask.tolist()).issubset({0.0, 1.0}), f'Valores inesperados: {set(mask.tolist())}'
    print(f'  drone_{i} mask OK -- acciones validas: {int(mask.sum())}/8')

# validate_action
valid, penalty = bridge.validate_action('drone_0', 'mover_n', state_dict)
print(f'validate_action() OK -- valido: {valid}, penalidad: {penalty}')

# get_reward_modifier
modifier = bridge.get_reward_modifier('drone_0', state_dict, 'mover_n')
print(f'get_reward_modifier() OK -- modificador: {modifier:.2f}')

# negotiate_passage
winner = bridge.negotiate_passage('drone_0', 'drone_1', state_dict)
assert winner in (0, 1), f'Resultado invalido: {winner}'
print(f'negotiate_passage() OK -- gana: drone_{winner}')

print()
print('=== INTEGRACION NEURO-SIMBOLICA: PASS ===')
"
```

---

## Paso 4 — Test del loop de entrenamiento (sin API)

Verifica que un episodio completo se ejecuta sin errores antes de levantar el servidor.

> **Nota de implementación:** `env.step()` retorna `rewards`, `terminated` y `truncated`
> como `np.ndarray`, NO como dicts. Pasar `actions` como `np.ndarray` de int64.

```powershell
python -c "
import numpy as np
from environment.city_env import CyberCityEnv
from agents.dqn_agent import DQNAgent
from logic.neuro_symbolic_bridge import NeuroSymbolicBridge

GRID     = 50
N_DRONES = 5
STEPS    = 50

env     = CyberCityEnv(grid_size=GRID, num_drones=N_DRONES)
bridge  = NeuroSymbolicBridge('logic/rules.pl')
agents  = [DQNAgent(f'drone_{i}', state_dim=7, action_dim=8) for i in range(N_DRONES)]
env.set_symbolic_bridge(bridge)

obs, _     = env.reset()
total_rew  = 0.0
step_count = 0

for step in range(STEPS):
    bridge.sync_state(env.get_state_dict(0))

    action_list = []
    for i, agent in enumerate(agents):
        state = obs[f'drone_{i}']
        mask  = bridge.get_action_mask(f'drone_{i}', env.get_state_dict(i))
        action_list.append(agent.select_action(state, symbolic_mask=mask))

    # step() espera np.ndarray, retorna rewards/terminated/truncated como np.ndarray
    actions_arr = np.array(action_list, dtype=np.int64)
    obs, rewards, terminated, truncated, info = env.step(actions_arr)
    total_rew  += float(rewards.sum())
    step_count += 1

    if terminated.all() or truncated.all():
        break

print(f'Episodio de prueba OK -- {step_count} steps, recompensa total: {total_rew:.1f}')
print('=== LOOP DE ENTRENAMIENTO: PASS ===')
"
```

---

## Paso 5 — Levantar el servidor FastAPI

```powershell
cd "w:\Proyectos FullStack\Dron-intel-os\backend"
python main.py
```

El log debe mostrar (en los primeros segundos):

```
INFO  dron-intel-os — Motor Prolog activo: ...rules.pl
INFO  uvicorn.error — Application startup complete.
INFO  uvicorn.error — Uvicorn running on http://0.0.0.0:8000
```

Si aparece `Bridge simbólico no disponible` en lugar de "Motor Prolog activo", revisar el Paso 2.2.

---

## Paso 6 — Verificación de endpoints REST

Con el servidor corriendo en otra terminal, ejecutar cada comprobación:

### 6.1 Health check

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/health" | ConvertTo-Json
```

Respuesta esperada:

```json
{
  "status": "ok",
  "training": false,
  "system": "neuro_dqn",
  "symbolic_ok": true,
  "num_drones": 5,
  "grid_size": 50
}
```

> `"symbolic_ok": true` confirma que SWI-Prolog está operativo.

### 6.2 Estado de drones

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/drone-state" | ConvertTo-Json -Depth 5
```

### 6.3 Log simbólico (vacío antes de entrenar)

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/symbolic-log" | ConvertTo-Json
```

### 6.4 Métricas vacías (sin episodios aún)

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/metrics/summary" | ConvertTo-Json
```

### 6.5 Iniciar entrenamiento corto (5 episodios)

```powershell
Invoke-RestMethod -Method Post `
  -Uri "http://localhost:8000/start-training?system=neuro_dqn&episodes=5"
```

Respuesta esperada:

```json
{ "status": "started", "system": "neuro_dqn", "episodes": 5 }
```

### 6.6 Detener entrenamiento

```powershell
Invoke-RestMethod -Method Post -Uri "http://localhost:8000/stop-training"
```

### 6.7 Métricas post-entrenamiento

```powershell
# Resumen general
Invoke-RestMethod -Uri "http://localhost:8000/metrics/summary" | ConvertTo-Json -Depth 4

# Curva de aprendizaje
Invoke-RestMethod -Uri "http://localhost:8000/metrics/curve/neuro_dqn" | ConvertTo-Json
```

---

## Paso 7 — Verificación del WebSocket

Guardar este script como `ws_test.py` en cualquier carpeta y ejecutarlo:

```python
# ws_test.py — Ejecutar con: python ws_test.py
import asyncio
import json
import websockets

async def test_ws():
    uri = "ws://localhost:8000/ws"
    try:
        async with websockets.connect(uri) as ws:
            print("Conexión WebSocket establecida")

            # Enviar ping
            await ws.send(json.dumps({"type": "ping"}))
            response = await asyncio.wait_for(ws.recv(), timeout=3.0)
            data = json.loads(response)
            print("Respuesta ping:", data)
            assert data.get("type") == "pong", f"Esperado pong, recibido: {data}"
            print("=== WebSocket: PASS ===")
    except Exception as e:
        print(f"ERROR WebSocket: {e}")

asyncio.run(test_ws())
```

```powershell
python ws_test.py
```

---

## Paso 8 — Verificar escritura de CSV de métricas

```powershell
# Lanzar 3 episodios de astar (el más rápido, sin aprendizaje)
Invoke-RestMethod -Method Post `
  -Uri "http://localhost:8000/start-training?system=astar&episodes=3"

# Esperar ~30 segundos y verificar que el CSV se generó
Start-Sleep -Seconds 35
Test-Path "w:\Proyectos FullStack\Dron-intel-os\data\training_logs.csv"

# Si existe, ver las primeras filas
Get-Content "w:\Proyectos FullStack\Dron-intel-os\data\training_logs.csv" | Select-Object -First 5
```

---

## Resumen del estado de verificación

Completa la siguiente tabla después de cada paso:

| # | Componente | Comando clave | Estado |
|---|---|---|---|
| 1 | SWI-Prolog en PATH | `swipl --version` (v10.0.2 en `C:\Program Files\swipl\bin`) | ✅ |
| 2 | Dependencias pip | `pip show pyswip torch fastapi` | ✅ |
| 3a | BaseAgent + AgentState | import + `to_obs_vector()` | ✅ |
| 3b | DQNAgent | `select_action()` con mask | ✅ |
| 3c | AStarAgent | `select_action()` con target | ✅ |
| 3d | NeuroSymbolicBridge | instanciar `NeuroSymbolicBridge(rules.pl)` | ✅ |
| 3e | CyberCityEnv | `reset()` + shape (7,) | ✅ |
| 4 | Integración Neuro-Simbólica | sync → mask → validate → modifier | ✅ |
| 5 | Loop de entrenamiento | 50 steps sin errores, masking R1 activo | ✅ |
| 6 | Servidor FastAPI | `python main.py` → `symbolic_ok: true` | ✅ |
| 7a | `/health` | `symbolic_ok: true` | ✅ |
| 7b | `/start-training` | `status: started` + 3 episodios completados | ✅ |
| 7c | `/metrics/summary` | datos post-entrenamiento con 3 episodios | ✅ |
| 8 | WebSocket `/ws` | ping → pong | ✅ |
| 9 | CSV de métricas | archivo `training_logs.csv` generado | ✅ |

---

## Contratos de API que consume el frontend

Una vez que todos los pasos anteriores estén en verde, el frontend puede consumir estos
contratos sin cambios en el backend:

### REST

| Método | Endpoint | Uso en frontend |
|---|---|---|
| `GET` | `/health` | Badge de estado y modo `symbolic_ok` |
| `POST` | `/start-training?system=&episodes=` | Botón "Iniciar" del panel |
| `POST` | `/stop-training` | Botón "Detener" del panel |
| `GET` | `/drone-state` | Snapshot para DroneMap sin WS |
| `GET` | `/symbolic-log` | Feed inicial del RuleTerminal |
| `GET` | `/metrics/summary?system=&last_n=` | LiveStats — resumen |
| `GET` | `/metrics/comparison` | Tabla comparativa A*/DQN/Neuro-DQN |
| `GET` | `/metrics/curve/{system}?metric=` | LiveStats — curva de aprendizaje |

### WebSocket `ws://localhost:8000/ws`

El frontend ya tiene el hook `useSocket.ts` implementado. Los mensajes que llegan del
backend tienen esta estructura:

```typescript
// Broadcast cada 25 steps
{
  type: "step_update",
  episode: number,
  step: number,
  drones: Array<{ id: string, x: number, y: number, battery: number, carrying: number }>,
  rewards: number[],
  masks: number[][],         // [N_DRONES][8] — máscaras simbólicas
  dynamics: {
    storms: any[],
    wind: { direction: string, speed: number } | null,
    dynamic_nfzs: any[]
  }
}

// Broadcast cada 10 episodios
{
  type: "episode_update",
  metrics: { ... },          // EpisodeRecord serializado
  symbolic_log: string[]     // Últimas decisiones Prolog
}
```

---

## Solución de problemas comunes

### `ImportError: DLL load failed — libswipl.dll`
SWI-Prolog no está en PATH. Agregar `C:\Program Files\swipl\bin` a las variables de entorno del sistema y reiniciar la terminal.

### `ModuleNotFoundError: No module named 'agents'`
Ejecutar desde la carpeta `backend/`, no desde la raíz del proyecto.

### `RuntimeError: CUDA out of memory`
Agregar `device="cpu"` en la instanciación de `DQNAgent` durante pruebas locales. El entrenamiento en CPU es suficiente para 200 episodios.

### WebSocket conecta pero no recibe datos
El entrenamiento debe estar activo (`/start-training`) para que haya broadcasts. `/health` + `training: true` confirma que está corriendo.

### `training_logs.csv` vacío o no existe
La ruta `data/` se resuelve relativa al directorio de trabajo. Ejecutar el servidor siempre desde `backend/` o verificar que `data/` existe en la raíz del proyecto.

---

> Cuando todos los ítems de la tabla estén marcados y los contratos de API respondan
> correctamente, el backend está listo para la integración con el frontend.
