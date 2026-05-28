# Dron-Intel-OS — Sistema Operativo Neuro-Simbólico para Enjambres de Drones

> Un sistema de inteligencia artificial autónomo para la coordinación estratégica y el reparto urbano eficiente mediante una flota de drones.

## 1. Resumen Ejecutivo

**Dron-Intel-OS** es un sistema de IA avanzado que fusiona dos paradigmas para resolver el complejo problema de la logística urbana con drones:

1.  **Inteligencia Adaptativa (Instinto):** Un modelo de **Aprendizaje por Refuerzo Profundo (Double Deep Q-Network)** permite que cada dron aprenda de forma autónoma las rutas más eficientes a través de la experiencia, optimizando entregas y consumo de energía por ensayo y error.
2.  **Conocimiento Experto (Conciencia):** Un **motor de lógica simbólica (Prolog)** actúa como un supervisor, aplicando un conjunto de 12 reglas estrictas que garantizan la seguridad y el cumplimiento de normativas. Este motor previene colisiones, evita zonas prohibidas y gestiona emergencias como la batería baja.

La sinergia de estos dos pilares crea un "enjambre inteligente" (Smart-Swarm) que no solo aprende a entregar, sino que lo hace bajo restricciones de seguridad explícitas. El motor Prolog aplica **action masking** que bloquea las acciones peligrosas (zonas prohibidas, colisiones inminentes, vuelo en tormenta, batería crítica) *antes* de que se ejecuten, y **reward shaping** que guía la coordinación multi-agente.

> **Nota sobre resultados:** En el experimento real de 200 episodios (ver [entregables/](entregables/)), Neuro-DQN y DQN puro alcanzan una tasa de éxito final **estadísticamente equivalente** (~39%). La ventaja del Neuro-DQN es **cualitativa**: arranque de aprendizaje más rápido, garantías de seguridad por masking, coordinación explícita y trazabilidad de cada decisión. Las cifras de este README provienen directamente de `data/training_logs.csv` y la API `/metrics/comparison`.

---

## 2. Stack Tecnológico

-   **Backend (Python):** PyTorch (DQN), `pyswip` para integración con SWI-Prolog, Gymnasium (para el entorno de simulación), FastAPI (API REST y WebSockets).
-   **Frontend (React):** TypeScript, Vite, Recharts (para visualización de datos en tiempo real), Mapas SVG interactivos.
-   **IA y Lógica:**
    -   **Modelo de Aprendizaje:** Double Deep Q-Network (DQN).
    -   **Motor Lógico:** SWI-Prolog, con 12 reglas definidas en `logic/rules.pl`.
    -   **Modelo de Entorno:** Formalizado como un Problema de Decisión Parcialmente Observable Descentralizado (Dec-POMDP).

---

## 3. Inicio Rápido

### Requisitos Previos
- **Python 3.10+** (probado en 3.12)
- **Node.js 18+** (probado en 22)
- **SWI-Prolog** — necesario **solo** para el modo Neuro-DQN. El backend arranca igualmente sin él (`/health` devuelve `symbolic_ok: false`) y los modos A* y DQN puro funcionan; el modo Neuro-DQN queda deshabilitado.
  - Windows: `winget install SWI-Prolog.SWI-Prolog`
  - Linux: `sudo apt install swi-prolog`
  - macOS: `brew install swi-prolog`

> El repositorio incluye **checkpoints pre-entrenados** en `data/checkpoints/` y los logs del experimento en `data/training_logs.csv`, de modo que las gráficas y métricas se ven sin necesidad de re-entrenar.

### Pasos para la Ejecución

1.  **Iniciar el Backend:**
    Abre una terminal, navega a la carpeta `backend` y ejecuta:
    ```bash
    # Instalar dependencias
    pip install -r requirements.txt

    # Iniciar el servidor
    python main.py
    ```
    La API estará disponible en `http://localhost:8000` y el WebSocket en `ws://localhost:8000/ws`.

2.  **Iniciar el Frontend:**
    Abre una segunda terminal, navega a la carpeta `frontend` y ejecuta:
    ```bash
    # Instalar dependencias
    npm install

    # Iniciar la aplicación de desarrollo
    npm run dev
    ```
    La interfaz de usuario se abrirá en tu navegador en `http://localhost:5173`.

3.  **Lanzar un Entrenamiento:**
    Una vez que la interfaz esté activa, puedes iniciar el proceso de entrenamiento desde el **Panel de Control** en la UI o enviando una petición directa a la API:
    ```bash
    # Ejemplo: entrenar el sistema Neuro-DQN por 200 episodios
    curl -X POST "http://localhost:8000/training/start?system=neuro_dqn&episodes=200"
    ```

### Ejecutar las pruebas

La suite (117 tests) se ejecuta desde la **raíz del proyecto**:

```bash
pip install -r backend/requirements.txt   # incluye pytest
pytest -q
```

---

## 4. Arquitectura del Sistema

El proyecto está organizado en módulos cohesivos que representan las diferentes capas de la arquitectura.

```
dron-intel-os/
├── backend/
│   ├── agents/          # Agentes de IA (DQN, A*, etc.)
│   ├── logic/           # Lógica simbólica (reglas Prolog y puente neuro-simbólico)
│   ├── environment/     # Entorno de simulación (CyberCityEnv basado en Gymnasium)
│   ├── ml_models/       # Modelos auxiliares (ej. predictor de demanda)
│   ├── analysis/        # Herramientas para recolección y análisis de métricas
│   ├── main.py          # Servidor FastAPI: API REST y WebSockets para telemetría
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/  # Componentes React (Mapa, Gráficas, Terminal, etc.)
│   │   ├── hooks/       # Hooks personalizados (ej. para gestionar WebSockets)
│   │   └── App.tsx      # Componente principal que orquesta la UI
│   └── vite.config.ts   # Configuración del servidor de desarrollo (incluye proxy al backend)
├── docs/                # Documentación detallada sobre modelos, protocolos y reglas
├── entregables/         # Informes académicos: modelado matemático, resultados y comparación
├── tests/               # Suite de pruebas (pytest, 117 tests)
└── data/                # Checkpoints pre-entrenados + logs/métricas de simulación
```

### ¿Cómo funciona?

1.  **El Entorno (`CyberCityEnv`):** Simula una ciudad con drones, paquetes, estaciones de carga, zonas prohibidas y condiciones climáticas dinámicas. Define las "leyes de la física" del mundo.
2.  **El Agente (`DQNAgent`):** Cada dron es controlado por un agente. Observa su estado (posición, batería, etc.) y decide qué acción tomar (moverse, cargar, etc.). Su cerebro es la red neuronal `PolicyNet`, que aprende a estimar qué acción es más valiosa en cada situación.
3.  **El Oráculo (`NeuroSymbolicBridge` y `rules.pl`):** Antes de que el agente ejecute una acción, el puente neuro-simbólico consulta al motor Prolog. Las reglas lógicas actúan como un filtro:
    - **Action Masking:** Si una acción es peligrosa (ej. entrar en una zona prohibida o chocar), Prolog la bloquea. El agente DQN ni siquiera la considera como una opción válida.
    - **Reward Shaping:** Si una acción no es peligrosa pero sí indeseable (ej. acercarse demasiado a otro dron), Prolog modifica la recompensa que recibe el agente, enseñándole sutilmente a evitar esos comportamientos.
4.  **El Entrenamiento:** El ciclo se repite miles de veces por episodio. El agente explora el entorno, toma decisiones, recibe recompensas (o castigos) y almacena estas experiencias en un "buffer de memoria". Periódicamente, usa esta memoria para re-entrenar su red neuronal, volviéndose progresivamente más inteligente.
5.  **La Interfaz (`Frontend`):** Se conecta al backend vía WebSockets para recibir datos en tiempo real sobre la posición de los drones, el estado del entrenamiento y las decisiones lógicas. Visualiza toda esta información para que el usuario pueda monitorizar el sistema.

---

## 5. La Curva de Aprendizaje: De la Exploración a la Explotación

Una pregunta común es: **¿Por qué se necesitan tantos episodios para que el sistema sea eficiente?**

La respuesta reside en el dilema fundamental del Aprendizaje por Refuerzo: **Exploración vs. Explotación**.

1.  **Fase de Exploración (Episodios Iniciales):**
    - Al principio, el agente no sabe nada sobre el mundo. Su red neuronal está inicializada con pesos aleatorios. Para aprender, debe **explorar**.
    - El sistema utiliza una estrategia llamada **"epsilon-greedy"**: la mayor parte del tiempo, el agente tomará acciones aleatorias (un valor alto de "epsilon"). Esto es crucial. Aunque parezca caótico y resulte en una tasa de éxito muy baja, es la única forma de que el agente descubra qué acciones conducen a buenas recompensas (entregar un paquete) y cuáles a castigos (quedarse sin batería).
    - Durante esta fase, el agente visita muchas situaciones diferentes, comete errores y construye un "mapa" mental del entorno en su memoria de repetición. El motor simbólico es vital aquí, ya que actúa como una red de seguridad, permitiendo la exploración sin consecuencias catastróficas.

2.  **Fase de Transición (Episodios Intermedios):**
    - A medida que el agente acumula experiencias, el valor de "epsilon" disminuye gradualmente. Esto significa que el agente comienza a confiar más en lo que ha aprendido.
    - En lugar de tomar acciones puramente aleatorias, empieza a **explotar** el conocimiento adquirido, eligiendo las acciones que su red neuronal predice que serán las mejores.
    - La tasa de éxito comienza a aumentar notablemente, y el comportamiento de los drones se vuelve visiblemente más intencionado y menos errático.

3.  **Fase de Explotación (Episodios Avanzados):**
    - En esta etapa, el valor de "epsilon" es muy bajo. El agente actúa de forma casi determinista, siguiendo la política óptima que ha aprendido.
    - Las acciones son precisas y eficientes. Los drones se dirigen directamente a sus objetivos, gestionan su batería de forma proactiva y cooperan para evitar congestiones. La tasa de éxito se estabiliza en un valor alto.

En resumen, el largo proceso de entrenamiento es una inversión necesaria. Los errores y el caos inicial son la base sobre la cual se construye la inteligencia y la eficiencia del sistema final. Sin una exploración exhaustiva, el agente podría quedarse atascado en estrategias subóptimas y nunca descubrir su verdadero potencial.

---

## 6. Modelo Formal (Dec-POMDP)

El problema se modela como un **Problema de Decisión Parcialmente Observable Descentralizado (Dec-POMDP)**, donde cada agente `i` tiene una observación parcial del estado global.

- **Estado del Agente (`s^i_t`):** Un vector que incluye su posición, batería, estado de carga, y la percepción del entorno.
  $$s^i_t = (x, y, z, \beta, \kappa, \omega, \eta, \dots)$$

- **Función de Recompensa (`R_total`):** La recompensa total es una suma ponderada de varios factores, diseñada para incentivar el comportamiento deseado.
  $$R_{\text{total}} = R_{\text{entrega}} + R_{\text{eficiencia}} - C_{\text{mov}} - P_{\text{colisión}} - P_{\text{batería}} - P_{\text{simbólica}}$$
  Donde `P_simbólica` es la penalización o bonus añadido por el motor Prolog.

$$\pi_{\text{NS}}(a|s) = \arg\max_{a:\mathcal{M}[a]=1} Q_\theta(s, a)$$

Ver [docs/formal_modeling.md](docs/formal_modeling.md) para la derivación completa.

---

## API Endpoints

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/health` | Estado del servicio |
| POST | `/training/start` | Iniciar entrenamiento |
| POST | `/training/stop` | Detener entrenamiento |
| GET | `/metrics/summary` | Estadísticas agregadas |
| GET | `/metrics/comparison` | Tabla A* vs DQN vs Neuro-DQN |
| GET | `/metrics/curve/{system}` | Curva de aprendizaje |
| GET | `/state/drones` | Snapshot actual de la flota |
| GET | `/log/symbolic` | Últimas decisiones Prolog |
| WS | `/ws` | Canal tiempo real |

---

## Experimento Comparativo

Protocolo en [docs/experimental_protocol.md](docs/experimental_protocol.md). Resultados reales (200 episodios/modelo, `data/training_logs.csv`) detallados en [entregables/04_comparacion_modelos.md](entregables/04_comparacion_modelos.md).

| Sistema | Tasa de éxito | Recompensa media | Violaciones/ep. | Colisiones/ep. |
|---|---|---|---|---|
| DQN puro | 39.3% | −1,571.3 | 42.6 | 0.68 |
| **Neuro-DQN** | **39.2%** | **−1,443.6** | 51.8 \* | 0.74 |

\* El Neuro-DQN reporta más "violaciones" porque el motor Prolog **detecta y registra** más categorías de intervención (R4, R8, R9, R11 = reward shaping), no porque sea menos seguro. Su ventaja real está en el *masking preventivo* y el arranque de aprendizaje más rápido (ver entregable 04, §2.2 y §4).

---

## Reglas Simbólicas (resumen)

| # | Regla | Tipo | Peso |
|---|---|---|---|
| R1 | No-Fly Zone | Mask | −100 |
| R2 | Batería crítica <15% | Mask+Shape | −50/paso |
| R3 | Colisión inminente | Mask | −200 |
| R4 | Conflicto de celda | Shape | −30(N−1) |
| R5 | Estación ocupada | Mask | −20 |
| R6 | Entrega médica | Shape | +150 |
| R7 | Tormenta activa | Mask | −80 |
| R8 | Viento fuerte | Shape | −15 |
| R9 | Zona congestionada | Shape | −40 |
| R10 | Ruta eficiente | Shape | +20 |
| R11 | Negociación de paso | Shape | +25/−10 |
| R12 | Predicción fallo batería | Mask | −500 |

---

## Autor
Selvin Godoy
Proyecto académico — Inteligencia Artificial Avanzada  
Universidad: MIUMG | Fecha: Mayo 2026
