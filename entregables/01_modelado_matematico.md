# Modelado Matemático — Dron-Intel-OS
## Sistema Autónomo Multi-Agente con Aprendizaje por Refuerzo Profundo y Razonamiento Neuro-Simbólico

---

## 1. Formulación como Proceso de Decisión de Markov Descentralizado (Dec-POMDP)

El sistema se formaliza como un **Dec-POMDP** (Decentralized Partially Observable Markov Decision Process) de 6-tupla:

$$\mathcal{M} = \langle \mathcal{I}, \mathcal{S}, \mathcal{A}, \mathcal{T}, \mathcal{R}, \Omega, \mathcal{O}, \gamma \rangle$$

| Símbolo | Descripción |
|---------|-------------|
| $\mathcal{I} = \{1,\ldots,5\}$ | Conjunto de 5 agentes (drones) |
| $\mathcal{S}$ | Espacio de estados global del entorno |
| $\mathcal{A} = \prod_{i} A^i$ | Espacio de acciones conjuntas |
| $\mathcal{T}: \mathcal{S} \times \mathcal{A} \to \Delta(\mathcal{S})$ | Función de transición estocástica |
| $\mathcal{R}: \mathcal{S} \times \mathcal{A} \to \mathbb{R}$ | Función de recompensa |
| $\Omega, \mathcal{O}$ | Espacio y función de observaciones parciales |
| $\gamma = 0.99$ | Factor de descuento |

---

## 2. Espacio de Estados $\mathcal{S}$

### 2.1 Observación individual por agente

Cada agente $i$ percibe una observación local $s^i_t \in \mathbb{R}^{11}$:

$$s^i_t = \bigl(x^i, y^i, z^i, \beta^i, \kappa^i, \omega^i, \eta^i, \Delta x^{T}, \Delta y^{T}, \Delta x^{C}, \Delta y^{C}\bigr)$$

| Componente | Dominio | Descripción |
|-----------|---------|-------------|
| $x^i, y^i$ | $[0, 1]$ | Posición normalizada en el grid 50×50 |
| $z^i$ | $\{0, 1\}$ | Altitud (tierra=0, vuelo=1) |
| $\beta^i$ | $[0, 1]$ | Nivel de batería normalizado (0%–100%) |
| $\kappa^i$ | $\{0, 1\}$ | Indicador de carga (0=vacío, 1=cargado) |
| $\omega^i$ | $[0, 1]$ | Condición climática codificada (clear=0, storm=1, wind=0.5) |
| $\eta^i$ | $[0, 1]$ | Densidad de vecinos normalizadaa |
| $\Delta x^T, \Delta y^T$ | $[-1, 1]$ | Delta normalizado hacia objetivo activo (paquete/destino) |
| $\Delta x^C, \Delta y^C$ | $[-1, 1]$ | Delta normalizado hacia estación de carga más cercana |

El espacio de observación completo es:

$$\mathcal{S}^i = \mathcal{B}^{11} = \{s \in \mathbb{R}^{11} : \mathbf{l} \leq s \leq \mathbf{u}\}$$

donde $\mathbf{l} = (0,0,0,0,0,0,0,-1,-1,-1,-1)^\top$ y $\mathbf{u} = (1,1,1,1,1,1,1,1,1,1,1)^\top$.

---

## 3. Espacio de Acciones $\mathcal{A}$

Cada agente $i$ selecciona $a^i_t \in A^i = \{0, 1, \ldots, 7\}$ (8 acciones discretas):

$$A^i = \{\text{despegar, aterrizar, mover\_n, mover\_s, mover\_e, mover\_o, esperar, cargar}\}$$

El espacio de acciones **conjuntas** es:

$$\mathcal{A} = \text{MultiDiscrete}([8]^5) \quad \Rightarrow \quad |\mathcal{A}| = 8^5 = 32{,}768$$

### 3.1 Costo energético por acción

| Acción | Costo de batería (% / step) |
|--------|-----------------------------|
| `mover_*` (N/S/E/O) | 0.4 |
| `despegar` | 0.5 |
| `aterrizar` | 0.2 |
| `esperar` | 0.2 |
| `cargar` | 0.0 (recarga +40%/step) |

Calibrado para que carga completa (100%) dure $\approx 250$ movimientos — suficiente para cruzar el grid (dist. máx. 100 celdas Manhattan) y completar múltiples entregas.

---

## 4. Función de Recompensa $\mathcal{R}$

La recompensa total por paso se compone de seis términos:

$$R_{\text{total}} = R_{\text{entrega}} + R_{\text{eficiencia}} - C_{\text{movimiento}} - P_{\text{colisión}} - P_{\text{batería}} - P_{\text{simbólica}}$$

### 4.1 Desglose de cada término

| Término | Valor | Condición de activación |
|---------|-------|-------------------------|
| $R_{\text{entrega}}$ (médico) | **+200** | Entrega de paquete médico completada |
| $R_{\text{entrega}}$ (estándar) | **+100** | Entrega de paquete estándar completada |
| $R_{\text{recogida}}$ | **+10** | Recogida de paquete del suelo |
| $R_{\text{eficiencia}}$ (shaping proximal) | $+0.5 \cdot \Delta d^T$ | Acercamiento al objetivo activo |
| $R_{\text{eficiencia}}$ (shaping carga) | $+0.4 \cdot \Delta d^C$ | Acercamiento a estación, si $\beta < 35\%$ |
| $C_{\text{movimiento}}$ | **−0.05** | Cada step (incentiva rapidez) |
| $P_{\text{colisión}}$ | $-30 \times N_c$ | Por cada colisión $N_c$ detectada |
| $P_{\text{batería}}$ (muerte) | **−80** | Batería agotada ($\beta = 0$) |
| $P_{\text{batería}}$ (crítica) | **−5** | Batería en zona crítica ($\beta < 15\%$) |
| $P_{\text{NFZ}}$ | **−15** | Entrada a zona de vuelo prohibido |
| $P_{\text{simbólica}}$ (R4, R8, R9, R11) | Variable | Modulada por el motor Prolog |

### 4.2 Shaping basado en potencial (Ng et al., 1999)

El shaping proximal implementa una **función de potencial** $\Phi(s)$ que preserva la política óptima:

$$F(s_t, s_{t+1}) = \gamma \cdot \Phi(s_{t+1}) - \Phi(s_t)$$

Con $\Phi(s^i_t) = 0.5 \cdot d(s^i_t, \text{objetivo})$, el agente que deambula sin progresar recibe $F \approx 0$ (sin deriva negativa que inhiba la exploración). Solo el progreso real hacia el objetivo acumula recompensa positiva.

---

## 5. Política $\pi$ — ε-greedy enmascarada

La política del agente $i$ combina exploración ε-greedy con **acción masking simbólico**:

$$\pi^i_{\theta}(a | s^i_t, \mathbf{M}^i_t) = \begin{cases} \text{Uniforme}\bigl(\{a : M^i_t[a] = 1\}\bigr) & \text{con prob. } \varepsilon_t \\ \arg\max_{a: M^i_t[a]=1} Q_{\theta}(s^i_t, a) & \text{con prob. } 1 - \varepsilon_t \end{cases}$$

donde:
- $\mathbf{M}^i_t \in \{0,1\}^8$ es la máscara binaria generada por el motor Prolog en el step $t$
- $\varepsilon_t = \max(\varepsilon_{\min}, \varepsilon_0 \cdot \delta^\varepsilon_t)$ con $\varepsilon_0 = 1.0$, $\delta = 0.99$, $\varepsilon_{\min} = 0.05$
- A los 150 episodios: $\varepsilon_{150} \approx 0.22$; convergencia de explotación a $\varepsilon \approx 0.05$ alrededor del episodio $\approx$ 580

---

## 6. Arquitectura de la Red Q — PolicyNet

$$Q_{\theta}: \mathbb{R}^{11} \rightarrow \mathbb{R}^8$$

```
Input(11) → Linear(256) → LayerNorm(256) → ReLU
          → Linear(256) → LayerNorm(256) → ReLU
          → Linear(8)   → Q-values
```

- **LayerNorm** preferida sobre BatchNorm: normaliza por muestra, no por batch, tolerando la distribución no-estacionaria de los estados en RL.
- **Double-DQN**: desacopla la selección de acción (red online $\theta$) y la evaluación del valor (red target $\theta^-$):

$$y_t = r_t + \gamma \cdot Q_{\theta^-}\!\left(s_{t+1},\; \arg\max_a Q_\theta(s_{t+1}, a)\right)$$

- **Soft target update** (Polyak averaging): $\theta^- \leftarrow \tau\,\theta + (1-\tau)\,\theta^-$, con $\tau = 0.001$
- **Huber loss** + gradient clipping (max-norm = 1.0) para estabilidad numérica

---

## 7. Replay Buffer

Buffer circular FIFO de capacidad $N = 100{,}000$ transiciones. Cada transición almacenada:

$$\mathcal{D} = \{(s_t, a_t, r_t, s_{t+1}, d_t, \mathbf{M}_{t+1}) \mid t = 0,1,\ldots\}$$

donde $d_t \in \{0,1\}$ es el indicador de fin de episodio y $\mathbf{M}_{t+1}$ la máscara del siguiente estado (necesaria para el masking en la estimación de $y_t$).

El muestreo aleatorio rompe la correlación temporal y permite la actualización estocástica de los pesos.

---

## 8. Motor Neuro-Simbólico — 12 Reglas Prolog

El puente neuro-simbólico implementa $\pi_{NS}(a|s) = \arg\max_{a:\mathbf{M}[a]=1} Q_\theta(s,a)$ mediante dos mecanismos:

- **Action Masking**: bloquea acciones *antes* de que el DQN decida ($\mathbf{M}^i_t$)
- **Reward Shaping**: modifica la recompensa *después* de cada paso ($P_{\text{simbólica}}$)

### 8.1 Tabla de reglas

| ID | Nombre | Tipo | Efecto cuantitativo |
|----|--------|------|---------------------|
| R1 | Zona de vuelo prohibida (NFZ) | Action Masking | Bloqueo + penalización −100 |
| R2 | Batería crítica — Modo emergencia | Action Masking estricto | Solo {cargar, aterrizar}; −50/step |
| R3 | Colisión inminente entre agentes | Action Masking preventivo | Bloqueo de celda ocupada; −200 |
| R4 | Conflicto de celda — múltiples drones | Reward Shaping negativo | $-30 \times (N-1)$ por dron extra |
| R5 | Estación de carga ocupada | Action Masking + Reward Shaping | Bloquea carga; −20 |
| R6 | Prioridad de entrega médica | Reward Shaping positivo | +150 médica, +50 estándar, +30 ruta |
| R7 | Tormenta activa en región | Action Masking + Reward Shaping | Bloquea vuelo; −80 |
| R8 | Viento fuerte — afecta movimiento | Reward Shaping negativo | −15 si $v > 60$ km/h contra viento |
| R9 | Zona congestionada — tráfico denso | Action Masking suave + Reward Shaping | −40 con ≥3 drones en región |
| R10 | Ruta más eficiente pre-calculada | Reward Shaping positivo | +20 por ruta eficiente conocida |
| R11 | Negociación de derecho de paso | Reward Shaping coordinativo | +25 al cedente; −10 al que interrumpe |
| R12 | Predicción de fallo de batería en ruta | Action Masking crítico | Bloquea ruta insuficiente; −500 caída |

### 8.2 Jerarquía de prioridad de paso (R11)

$$\text{prioridad}(A_1, A_2) = \begin{cases} A_1 & \text{si } A_1 \text{ lleva carga médica} \wedge A_2 \text{ no} \\ A_2 & \text{si } A_2 \text{ lleva carga médica} \wedge A_1 \text{ no} \\ \arg\min(\beta_{A_1}, \beta_{A_2}) & \text{si ninguno es médico (batería más baja prioriza)} \\ A_1 & \text{desempate por índice} \end{cases}$$

---

## 9. Modelo ML Complementario — DemandPredictor

Modelo **GradientBoostingRegressor** ($R^2 \approx 0.84$) que anticipa zonas de alta demanda antes de cada episodio:

**Features de entrada:**
$$\mathbf{x}_{\text{ML}} = \bigl(\sin(2\pi h/24),\; \cos(2\pi h/24),\; \sin(2\pi d/7),\; \cos(2\pi d/7),\; x_{\text{grid}},\; y_{\text{grid}}\bigr)$$

donde $h$ es la hora del día y $d$ el día de la semana. Las features cíclicas capturan periodicidad sin discontinuidades.

**Salida:** Las top-5 zonas de alta demanda predichas sesgan el 60% de los destinos de entrega al inicio de cada episodio, determinando la misión que DQN/A\* deben planificar.

---

## 10. Algoritmo de Referencia — A* con Replanning

**Heurística**: distancia Manhattan $h(n) = |x_n - x_g| + |y_n - y_g|$

**Función de coste**: $f(n) = g(n) + h(n)$

**Replanning dinámico**: cuando una NFZ o tormenta bloquea la ruta calculada, el agente recalcula inmediatamente desde la posición actual, incorporando los nuevos obstáculos al grafo de búsqueda. No requiere aprendizaje pero tampoco mejora con la experiencia.

---

## 11. Entorno Dinámico — DynamicsEngine

El entorno estocástico implementa:

- **Zonas de vuelo prohibido (NFZ)**: $n_{NFZ} \sim \text{Poisson}(\lambda=2)$ por episodio, con duración aleatoria
- **Tormentas**: rectangulares, spawn probabilístico, modelan coeficiente extra de consumo (3.5%/celda vs 1.5% normal)  
- **Viento dominante**: dirección y velocidad $v \sim \mathcal{U}(0, 100)$ km/h
- **Recursos finitos**: 4 estaciones de carga con ocupación exclusiva
- **Fidelidad estocástica**: el mismo `seed` en `env.reset()` y `dynamics.reset()` produce condiciones idénticas para comparación A\* vs DQN vs Neuro-DQN

---

*Documento generado: 2026-05-22 — Basado en código fuente real: [city_env.py](../backend/environment/city_env.py), [dqn_agent.py](../backend/agents/dqn_agent.py), [rules.pl](../backend/logic/rules.pl)*
