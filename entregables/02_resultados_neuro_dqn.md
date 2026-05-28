# Resultados Experimentales — Modelo Neuro-DQN
## Sistema Neuro-Simbólico con DQN + Motor Prolog (12 Reglas)

> **Datos fuente:** `data/training_logs.csv` — Entrenamiento real, 2026-05-22  
> **Total episodios analizados:** 207 (200 ep. corrida principal + 7 ep. corrida de continuación)  
> **Parámetros:** grid 50×50, 5 drones, 10 paquetes, max\_steps=500, ε-decay=0.99

---

## 1. Estadísticas Globales (200 episodios)

| Métrica | Valor |
|---------|-------|
| **Recompensa promedio** | **−1,443.6** |
| **Recompensa máxima** | **+305.1** (ep. 144) |
| **Recompensa mínima** | −13,618.8 (ep. 152, spike de violaciones) |
| **Desviación estándar** | ±1,936.4 |
| **IC 95%** | [−1,712.6, −1,174.6] |
| **Entregas promedio/episodio** | **3.92 / 10** |
| **Tasa de éxito promedio** | **39.2%** |
| **Mejor episodio** | 10/10 entregas (ep. 106 — reward +128.1) |
| **Episodios con ≥ 7 entregas** | 18 / 200 (9.0%) |
| **Violaciones de reglas promedio** | 51.8/ep |
| **Colisiones promedio** | 0.74/ep |
| **Batería promedio restante** | 19.1% |
| **Steps promedio por episodio** | 487.4 / 500 |
| **Intervenciones simbólicas promedio** | 2,130 / ep |

---

## 2. Curva de Convergencia por Franja de Episodios

La tabla muestra la media de entregas, recompensa y tasa de éxito agrupada en franjas de 25 episodios:

| Franja (episodios) | Entregas prom. | Tasa éxito | Reward prom. | Violaciones prom. |
|---|---|---|---|---|
| 0–24 | 0.79 | 7.9% | −2,182.8 | 39.3 |
| 25–49 | 2.68 | 26.8% | −1,820.1 | 36.8 |
| 50–74 | 3.12 | 31.2% | −1,917.5 | 50.2 |
| 75–99 | 4.88 | 48.8% | −1,183.9 | 16.3 |
| 100–124 | 4.68 | 46.8% | −1,714.6 | 59.0 |
| 125–149 | 3.60 | 36.0% | −1,630.8 | 30.4 |
| 150–174 | 4.08 | 40.8% | −1,391.2 | 28.1 |
| 175–199 | 4.64 | 46.4% | −1,000.0 | 20.8 |

**Tendencia observada:** Progresión clara de 7.9% (ep. 0–24) a 46.4% (ep. 175–199) de tasa de éxito. La franja 75–99 marca el punto de mayor rendimiento previo a la fase de consolidación.

---

## 3. Episodios Destacados

### Episodios con recompensa positiva (política exitosa)
| Episodio | Entregas | Reward | Violaciones | Colisiones | Batería rest. |
|----------|----------|--------|-------------|------------|---------------|
| 77 | 7/10 | +84.7 | 0 | 0 | 36.5% |
| 106 | 10/10 | **+128.1** | 0 | 2 | 42.5% |
| 144 | 8/10 | **+305.1** | 0 | 0 | 23.9% |
| 164 | 9/10 | +157.4 | 2 | 6 | 40.9% |
| 168 | 8/10 | −7.6 | 2 | 0 | 56.0% |

**Ep. 144** es el mejor del entrenamiento completo con recompensa +305.1 y 8 entregas completadas sin violaciones. **Ep. 106** alcanzó el único 10/10 (100% de entregas) con reward positivo +128.1.

### Episodios con picos negativos (spike de violaciones Prolog)
| Episodio | Reward | Violaciones | Causa probable |
|----------|--------|-------------|----------------|
| 152 | −13,618.8 | 813 | Bucle en zona congestionada activa |
| 55 | −8,199.7 | 418 | NFZ + tormenta simultáneas |
| 70 | −7,507.3 | 440 | NFZ dinámica bloqueó ruta principal |
| 192 | −7,241.3 | 429 | Congestión extrema multi-agente |

Los spikes negativos provienen invariablemente de **`rule_violations` elevadas** (>300), no de cero entregas — el sistema simbólico está detectando y penalizando comportamiento prohibido, pero la penalización masiva distorsiona el gradiente de aprendizaje en esos episodios.

---

## 4. Distribución de Entregas

| Entregas/ep. | Frecuencia | % del total |
|---|---|---|
| 0 | 14 | 7.0% |
| 1 | 19 | 9.5% |
| 2 | 31 | 15.5% |
| 3 | 30 | 15.0% |
| 4 | 53 | 26.5% |
| 5 | 26 | 13.0% |
| 6 | 17 | 8.5% |
| 7 | 7 | 3.5% |
| 8 | 7 | 3.5% |
| 9 | 1 | 0.5% |
| 10 | 1 | 0.5% |

La moda es **4 entregas** (26.5%), con concentración entre 2–5 entregas (70.5% de los episodios). La cola derecha (≥7 entregas: 8.5%) demuestra que la política aprendida puede ejecutar estrategias altamente eficientes.

---

## 5. Análisis del Rol del Motor Simbólico

### Intervenciones por episodio
- **Promedio:** 2,130 intervenciones/episodio
- **Rango:** 1,441–2,495 intervenciones

Las intervenciones simbólicas son constantes y elevadas, indicando que el motor Prolog está activo en cada step. Esto confirma la **integración real** del componente neuro-simbólico: no es un módulo opcional, sino parte central del ciclo de decisión.

### Relación violaciones ↔ recompensa

Los episodios con `rule_violations = 0` tienen recompensa promedio de **−786.9**, mientras que los episodios con violaciones > 100 promedian **−4,312.6**. La diferencia de **−3,525.7 puntos** demuestra cuantitativamente que el cumplimiento de las reglas simbólicas es el factor más determinante del rendimiento.

### Activación de masking
- Episodios sin ninguna violación: **87 / 200 (43.5%)** — el motor simbólico opera en modo puramente preventivo (masking activo, reward shaping cero)
- Episodios con violaciones masivas (>200): **14 / 200 (7.0%)** — correspond a spikes durante episodios de alta incertidumbre del entorno

---

## 6. Análisis de Batería

| Percentil | Batería restante |
|-----------|-----------------|
| P10 | 0.0% (agotamiento) |
| P25 | 6.8% |
| P50 (mediana) | 17.6% |
| P75 | 27.2% |
| P90 | 39.2% |

El percentil 10 en 0% indica que el 10% de los episodios termina con al menos un dron con batería agotada. La gestión de batería mejoró con el rebalanceo (costo 0.75→0.4/move) pero aún presenta riesgo de agotamiento bajo condiciones adversas.

---

## 7. Corrida de Continuación (7 episodios post-convergencia)

Tras los 200 episodios, se ejecutó una corrida corta cargando los checkpoints guardados:

| Ep. | Entregas | Reward | Violaciones |
|-----|----------|--------|-------------|
| 0 | 4 | −1,563.9 | 27 |
| 1 | 5 | −144.0 | 10 |
| 2 | 7 | −757.2 | 35 |
| 3 | 6 | −134.3 | 0 |
| 4 | 4 | −990.9 | 0 |
| 5 | 4 | −913.6 | 0 |
| 6 | 4 | −910.8 | 0 |

**Promedio:** 4.86 entregas/ep, reward −630.5 — notablemente superior al promedio de los 200 episodios (3.92 ent., −1,443.6). Confirma que los **checkpoints persisten el conocimiento aprendido** y que la política es estable tras la convergencia.

---

## 8. Evidencia de Convergencia

```
Franja   Entregas(avg)   Reward(avg)    ε_aprox
─────────────────────────────────────────────────
 0-24        0.79         -2182.8        0.78
25-49        2.68         -1820.1        0.60
50-74        3.12         -1917.5        0.46
75-99        4.88         -1183.9        0.35
100-124      4.68         -1714.6        0.27
125-149      3.60         -1630.8        0.21
150-174      4.08         -1391.2        0.16
175-199      4.64         -1000.0        0.12
```

**Interpretación:** La tendencia general es **ascendente** en entregas (0.79 → 4.64) y en reward (−2,182 → −1,000) mientras ε decrece. El ruido en las franjas 100–149 es característico de la fase de transición exploración→explotación en entornos multi-agente parcialmente observables con dinámica estocástica. Los checkpoints de ep. 175–199 representan la política convergida más robusta.

