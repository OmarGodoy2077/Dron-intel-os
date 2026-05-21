# Catálogo de Reglas Lógicas — Motor Prolog

## Arquitectura del Motor Neuro-Simbólico

El motor Prolog actúa como **oráculo de seguridad** entre el agente DQN y el entorno.
Cada regla produce uno de dos efectos:

- **Action Masking (AM):** Bloquea la acción *antes* de que el DQN la ejecute. Garantía dura.
- **Reward Shaping (RS):** Modifica la recompensa *después* del paso. Influencia suave.

```
Estado s_t ──► [Prolog: Máscara] ──► DQN selecciona a_t ──► Entorno
                                                            │
                    Recompensa r_t ◄────────────────────────┘
                         │
                    [Prolog: Reward Modifier] ──► r_t_shaped
```

---

## Reglas Implementadas

### R1 — Zona de Vuelo Prohibida
**Tipo:** Action Masking  
**Predicado:** `accion_invalida(AgentID, Accion)`  
**Peso:** $-100$ pts  
**Descripción:** Bloquea cualquier movimiento que lleve al dron a una celda marcada como no-fly zone (NFZ). Las NFZs incluyen zonas estáticas regulatorias y zonas dinámicas temporales generadas por `DynamicsEngine`.

```prolog
accion_invalida(AgentID, Accion) :-
    agente_en_celda(AgentID, CeldaActual),
    celda_objetivo(CeldaActual, Accion, CeldaDestino),
    zona_prohibida_activa(CeldaDestino).
```

**Impacto:** Cero violaciones de NFZ en `neuro_dqn`. DQN puro comete ~12% de pasos en NFZ durante exploración temprana.

---

### R2 — Batería Crítica (< 15%)
**Tipo:** Action Masking + Reward Shaping  
**Predicados:** `bateria_critica/1`, `solo_acciones_emergencia/1`  
**Peso AM:** Solo permite `{cargar, aterrizar}` | **Peso RS:** $-50$ pts/paso  
**Descripción:** Cuando la batería cae por debajo del 15%, el dron entra en modo emergencia. Solo puede aterrizar o ir a una estación de carga. El shaping penaliza además cada paso en estado crítico para incentivar la prevención temprana.

```prolog
bateria_critica(AgentID) :-
    bateria_agente(AgentID, Nivel), Nivel < 15.
```

**Impacto:** Elimina caídas libres por batería agotada. En DQN puro, ~8% de episodios terminan por batería = 0.

---

### R3 — Colisión Inminente
**Tipo:** Action Masking  
**Predicado:** `accion_causa_colision(AgentID, Accion)`  
**Peso:** $-200$ pts (bloqueado antes de ocurrir)  
**Descripción:** Detecta si la acción propuesta movería al dron a una celda actualmente ocupada por otro dron. El bloqueo preventivo es la diferencia crítica respecto al reward penalty, que solo actúa después de la colisión.

```prolog
accion_causa_colision(AgentID, Accion) :-
    celda_objetivo(CeldaActual, Accion, CeldaDestino),
    agente_en_celda(OtroAgente, CeldaDestino),
    AgentID \= OtroAgente.
```

**Impacto:** Colisiones = 0 en modo `neuro_dqn`. DQN puro: ~3.2 colisiones/episodio en entornos densos.

---

### R4 — Conflicto de Celda (Múltiples Drones)
**Tipo:** Reward Shaping  
**Predicado:** `penalizacion_conflicto(Celda, Penalizacion)`  
**Peso:** $-30 \times (N-1)$ donde $N$ = número de drones en la celda  
**Descripción:** Penaliza la coexistencia de múltiples drones en la misma celda. Complementa R3 (que previene entradas) con un incentivo continuo para dispersión en el espacio.

```prolog
penalizacion_conflicto(Celda, Penalizacion) :-
    findall(A, agente_en_celda(A, Celda), Agentes),
    length(Agentes, N), N > 1,
    Penalizacion is -30 * (N - 1).
```

---

### R5 — Estación de Carga Ocupada
**Tipo:** Action Masking  
**Predicado:** `accion_carga_invalida(AgentID)`  
**Peso:** $-20$ pts por intento fallido  
**Descripción:** Evita que múltiples drones intenten cargar simultáneamente en la misma estación. Esencial para coordinación con $n$ drones y $k < n$ estaciones.

---

### R6 — Prioridad de Entrega Médica
**Tipo:** Reward Shaping  
**Predicado:** `bonus_entrega(Paquete, Bonus)`  
**Peso:** $+150$ médico | $+50$ estándar  
**Descripción:** Diferencia los tipos de carga. Los drones aprenden a priorizar paquetes médicos sin necesidad de función de recompensa separada. El bonus mayor guía la política hacia gestión de prioridades emergente.

---

### R7 — Tormenta Activa en Región
**Tipo:** Action Masking  
**Predicado:** `accion_invalida_tormenta(AgentID, Accion)`  
**Peso:** $-80$ pts (bloqueado)  
**Descripción:** Bloquea vuelo en celdas cubiertas por tormentas activas. Las tormentas son generadas estocásticamente por `DynamicsEngine` y duran entre 20-80 pasos. La máscara se actualiza en tiempo real cada step.

---

### R8 — Viento Fuerte (> 60 km/h)
**Tipo:** Reward Shaping  
**Predicado:** `penalizacion_viento(Accion, Penalizacion)`  
**Peso:** $-15$ pts por movimiento contra el viento dominante  
**Descripción:** No bloquea el movimiento pero lo desincentiva. El DQN aprende a aprovechar el viento a favor y evitar resistencia. Simula consumo extra de batería real.

---

### R9 — Zona Congestionada
**Tipo:** Reward Shaping  
**Predicado:** `desvio_necesario(AgentID, CeldaDestino)`  
**Peso:** $-40$ pts por entrar en zona con $\geq 3$ drones  
**Descripción:** Incentiva distribución espacial de la flota. Una zona se considera congestionada cuando tiene 3 o más drones simultáneos. El DQN aprende patrones de despacho distribuido.

---

### R10 — Ruta Más Eficiente Pre-calculada
**Tipo:** Reward Shaping  
**Predicado:** `ruta_mas_eficiente(Origen, Destino, Ruta)`  
**Peso:** $+20$ pts por seguir ruta eficiente conocida  
**Descripción:** Cuando existe una ruta pre-calculada (por A* u otro planner) y no está bloqueada, se otorga un bonus. Esto **guía la exploración** del DQN hacia soluciones ya verificadas, acelerando convergencia.

---

### R11 — Negociación de Derecho de Paso
**Tipo:** Reward Shaping coordinativo  
**Predicado:** `negociar_derecho_paso(Agente1, Agente2)`  
**Peso:** $+25$ al que cede | $-10$ al que interrumpe  
**Descripción:** Resuelve conflictos de paso con jerarquía: médico > batería baja > orden de solicitud. Permite **comportamiento emergente de coordinación** sin comunicación explícita entre agentes.

```prolog
prioridad_paso(A1, A2, A1) :- entrega_urgente(A1), \+ entrega_urgente(A2).
prioridad_paso(A1, A2, A1) :- bateria_agente(A1, B1), bateria_agente(A2, B2), B1 < B2.
```

---

### R12 — Predicción de Fallo de Batería en Ruta
**Tipo:** Action Masking  
**Predicado:** `prediccion_fallo_bateria(AgentID, Ruta)`, `accion_riesgo_bateria(AgentID, Accion)`  
**Peso:** $-500$ caída libre (bloqueado preventivamente)  
**Descripción:** Estima el consumo de batería para la ruta actual (1.5%/celda base). Si la batería restante no alcanza para completar la ruta + margen de seguridad (10%), bloquea movimientos adicionales. Esta es la regla de **mayor impacto en seguridad**.

```prolog
prediccion_fallo_bateria(AgentID, Ruta) :-
    bateria_agente(AgentID, Bat),
    consumo_estimado_ruta(Ruta, Consumo),
    Bat - Consumo < 10.
```

---

## Estadísticas de Activación Esperadas

En un episodio típico de 500 pasos con 5 drones:

| Regla | Activaciones/episodio (DQN) | Activaciones/episodio (Neuro-DQN) |
|---|---|---|
| R1 NFZ | 45-80 (post-hoc penalty) | 0 (masked) |
| R2 Batería | 120-200 | 30-60 (prevenidas) |
| R3 Colisión | 8-15 | 0 (masked) |
| R7 Tormenta | 20-40 | 0 (masked) |
| R10 Ruta | N/A | 60-100 (bonus) |
| R11 Negociación | N/A | 15-30 |
| R12 Batería ruta | N/A | 10-20 (masked) |
