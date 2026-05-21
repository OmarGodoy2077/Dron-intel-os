# Modelado Formal — Smart-Swarm Neuro-Simbólico

## 1. Formulación como Dec-POMDP

El sistema se modela como un **Decentralised Partially Observable Markov Decision Process (Dec-POMDP)** de $n$ agentes:

$$\mathcal{M} = \langle \mathcal{I}, \mathcal{S}, \mathcal{A}, \mathcal{T}, \mathcal{R}, \Omega, \mathcal{O}, \gamma \rangle$$

donde $\mathcal{I} = \{1, \dots, n\}$ es el conjunto de drones.

---

## 2. Espacio de Estados $\mathcal{S}$

Cada dron $i$ observa un vector de estado local $s^i_t \in \mathbb{R}^{11}$ (implementación en `CyberCityEnv._get_obs()`).

$$s^i_t = \bigl(x,\; y,\; z,\; \beta \in [0,100],\; \kappa,\; \omega,\; \eta \bigr)$$

| Componente | Símbolo | Dominio | Descripción |
|---|---|---|---|
| Posición horizontal | $x, y$ | $\{0,\ldots,G-1\}^2$ | Coordenadas en el grid $G \times G$ |
| Altitud | $z$ | $\{0,\ldots,10\}$ | Nivel de vuelo |
| Batería | $\beta$ | $[0,100]$ | Porcentaje de carga restante |
| Carga | $\kappa$ | $\{0, 1\}$ | 1 = transportando paquete |
| Clima | $\omega$ | $\{0,1,2\}$ | 0=claro, 1=tormenta, 2=viento |
| Vecinos | $\eta$ | $\{0,\ldots,n-1\}$ | Drones en radio de 2 celdas |
| Delta a objetivo (x) | $tdx$ | $[-1,1]$ | Delta normalizado en x hacia objetivo (destino/paquete) |
| Delta a objetivo (y) | $tdy$ | $[-1,1]$ | Delta normalizado en y hacia objetivo (destino/paquete) |
| Delta a estación (x) | $cdx$ | $[-1,1]$ | Delta normalizado en x hacia estación de carga más cercana |
| Delta a estación (y) | $cdy$ | $[-1,1]$ | Delta normalizado en y hacia estación de carga más cercana |

El estado global es el producto $\mathcal{S} = \bigtimes_{i=1}^{n} \mathcal{S}^i$, pero cada agente solo observa $s^i_t$ (observabilidad parcial).

---

## 3. Espacio de Acciones $\mathcal{A}$

Acción conjunta: $\mathbf{a}_t = (a^1_t, \ldots, a^n_t) \in \mathcal{A}^n$

$$\mathcal{A} = \{\texttt{Despegar},\; \texttt{Aterrizar},\; \texttt{Mover}_N,\; \texttt{Mover}_S,\; \texttt{Mover}_E,\; \texttt{Mover}_O,\; \texttt{Esperar},\; \texttt{Cargar}\}$$

$|\mathcal{A}| = 8$, por lo que el espacio de acciones conjuntas crece como $8^n$.

### Enmascaramiento simbólico (Action Masking)

El motor Prolog impone una máscara binaria antes de cada decisión:

$$\mathcal{M}_t^i = \bigl(m_0, m_1, \ldots, m_7\bigr) \in \{0,1\}^8, \quad m_j = \llbracket a_j \text{ es válida en } s^i_t \rrbracket$$

La acción efectiva satisface $a^i_t \in \{a_j : m_j = 1\}$.

---

## 4. Función de Transición $\mathcal{T}$

$$\mathcal{T}(s' \mid s, \mathbf{a}) = P(\mathbf{s}_{t+1} = s' \mid \mathbf{s}_t = s,\; \mathbf{a}_t = \mathbf{a})$$

La transición incluye:
- Dinámica determinista del movimiento (sujeta a límites del grid)
- Eventos estocásticos del motor de dinámica: tormentas, viento, zonas no-fly temporales
- Colisiones detectadas y penalizadas instantáneamente

---

## 5. Función de Recompensa $\mathcal{R}$

$$\boxed{R_{\text{total}} = R_{\text{entrega}} + R_{\text{eficiencia}} - C_{\text{movimiento}} - P_{\text{colisión}} - P_{\text{batería}} - P_{\text{simbólica}}}$$

### 5.1 Componentes positivos

$$R_{\text{entrega}} = \begin{cases} +150 & \text{si paquete médico entregado} \\ +50 & \text{si paquete estándar entregado} \\ +5 & \text{por recogida de paquete} \end{cases}$$

$$R_{\text{eficiencia}} = +20 \cdot \mathbb{1}[\text{sigue ruta eficiente (R10)}]$$

### 5.2 Costos y penalizaciones

$$C_{\text{movimiento}} = 0.5 \quad \forall \text{ paso de tiempo}$$

$$P_{\text{colisión}} = 200 \cdot \mathbb{1}[\text{dos drones en misma celda}]$$

$$P_{\text{batería}} = \begin{cases} 500 & \text{si } \beta = 0 \text{ (caída libre)} \\ 50 & \text{por paso con } \beta < 15\% \end{cases}$$

$$P_{\text{simbólica}} = \sum_{r \in \mathcal{R}_{\text{violadas}}} w_r$$

donde $\mathcal{R}$ es el conjunto de reglas Prolog y $w_r$ su peso de penalización.

### 5.3 Tabla de pesos simbólicos $w_r$

| Regla | Descripción | $w_r$ | Tipo |
|---|---|---|---|
| R1 | Zona prohibida | $-100$ | Action Mask |
| R2 | Batería crítica (por paso) | $-50$ | Mask + Shaping |
| R3 | Colisión inminente | $-200$ | Action Mask |
| R4 | Conflicto de celda | $-30(N-1)$ | Reward Shaping |
| R5 | Estación ocupada | $-20$ | Action Mask |
| R6 | Entrega médica | $+150$ | Reward Shaping |
| R7 | Vuelo en tormenta | $-80$ | Action Mask |
| R8 | Viento fuerte | $-15$ | Reward Shaping |
| R9 | Zona congestionada | $-40$ | Reward Shaping |
| R10 | Ruta eficiente | $+20$ | Reward Shaping |
| R11 | Negociación de paso | $+25/-10$ | Reward Shaping |
| R12 | Predicción fallo batería | $-500$ | Action Mask |

---

## 6. Política $\pi_\theta$

Cada agente aprende una política $\pi_\theta^i$ parametrizada por la red neuronal DQN:

$$\pi_\theta^i(a \mid s^i) = \arg\max_{a \in \mathcal{A}_{\text{válidas}}(s^i)} Q_\theta(s^i, a)$$

### 6.1 Función Q — Actualización Double-DQN

$$Q_\theta(s, a) \leftarrow r + \gamma \cdot Q_{\theta^-}\!\Bigl(s',\; \arg\max_{a'} Q_\theta(s', a')\Bigr)$$

donde $\theta^-$ son los pesos de la red **target** (copiados cada $K$ pasos).

### 6.2 Loss Huber (Smooth L1)

$$\mathcal{L}(\theta) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}}\Bigl[\ell_1\bigl(\delta_t\bigr)\Bigr], \quad \delta_t = r + \gamma \max_{a'} Q_{\theta^-}(s', a') - Q_\theta(s, a)$$

### 6.3 Política integrada Neuro-Simbólica

$$\pi_{\text{NS}}^i(a \mid s^i) = \arg\max_{a \,:\, \mathcal{M}^i[a]=1} Q_\theta(s^i, a)$$

La clave: **el espacio de búsqueda de $\arg\max$ está restringido** por la máscara simbólica $\mathcal{M}^i$, garantizando seguridad en tiempo de inferencia sin reentrenar.

---

## 7. Exploración ε-greedy con restricción simbólica

$$a_t^i = \begin{cases} \text{Uniforme}(\{a : \mathcal{M}^i[a]=1\}) & \text{con prob. } \varepsilon_t \\ \arg\max_{a:\mathcal{M}^i[a]=1} Q_\theta(s^i_t, a) & \text{con prob. } 1 - \varepsilon_t \end{cases}$$

$$\varepsilon_t = \max(\varepsilon_{\min},\; \varepsilon_0 \cdot d^t), \quad d = 0.995,\; \varepsilon_{\min} = 0.05$$

---

## 8. Convergencia y garantías teóricas

**Teorema (convergencia DQN con masking):** Si $|\mathcal{A}_{\text{válidas}}(s)| \geq 1$ para todo $s$, el masking no interrumpe la convergencia de Q-learning, ya que el subconjunto restringido sigue siendo un MDP bien definido.

**Propiedad de seguridad (hardcoded):** Bajo la política $\pi_{\text{NS}}$, las violaciones de R1, R2, R3, R7 y R12 son **cero por construcción** (el agente físicamente no puede tomar esas acciones). Esto es verificable formalmente a diferencia de un reward penalty puro.

---

## 9. Complejidad computacional

| Componente | Complejidad temporal |
|---|---|
| Inferencia DQN | $O(d_{\text{hidden}}^2)$ por agente |
| Consulta Prolog (1 regla) | $O(|\text{hechos dinámicos}|)$ |
| Máscaras completas (8 acciones) | $O(8 \cdot |\mathcal{R}|)$ |
| Paso de entorno (n agentes) | $O(n^2)$ por comprobación de colisiones |

El cuello de botella es Pyswip por overhead de I/O entre Python y Prolog; en producción puede paralelizarse con un proceso Prolog dedicado por agente.
