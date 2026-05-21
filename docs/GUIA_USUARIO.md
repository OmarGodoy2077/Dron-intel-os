# Guía de Usuario — Smart-Swarm Centro de Control

> Para quien solo interactúa con el frontend. No se requiere conocimiento de Python, Prolog ni IA.

---

## ¿Qué es esto?

**Smart-Swarm** es una plataforma de simulación donde una flota de drones autónomos aprende a entregar paquetes en una ciudad virtual. Puedes ver cómo los drones navegan, evitan zonas restringidas y mejoran su rendimiento episodio a episodio, todo en tiempo real desde este panel de control.

Hay tres tipos de cerebros disponibles para los drones:

| Sistema | Qué hace |
|---|---|
| **A\*** | Sigue el camino más corto calculado matemáticamente. No aprende, pero es predecible. |
| **DQN** | Red neuronal que aprende por ensayo y error. Mejora con la experiencia. |
| **Neuro-DQN** | DQN + reglas lógicas (Prolog) que impiden acciones peligrosas. El más inteligente. |

---

## Antes de empezar

1. **Arranca el backend** (alguien del equipo técnico lo hará, o tú mismo si tienes acceso):
   ```
   cd backend
   python main.py
   ```

2. **Arranca el frontend**:
   ```
   cd frontend
   npm run dev
   ```

3. Abre tu navegador en **`http://localhost:5173`**.

Verás el panel cargarse. En la barra superior aparecerá:
- 🟢 **WS Conectado** — comunicación en tiempo real activa.
- 🔵 **Prolog Online** — motor de reglas lógicas disponible (necesario para Neuro-DQN).

Si alguno aparece en rojo o ámbar, el sistema puede funcionar parcialmente, pero no todas las funciones estarán disponibles.

---

## La pantalla principal

```
┌──────────────────────────────────────────────────────────────┐
│  LOGO  │  WS Conectado  │  Prolog Online  │  Ep: 0  │  Uptime │
├──────────┬─────────────────────────────────┬─────────────────┤
│          │                                 │  KPIs           │
│  MENÚ    │        MAPA DE DRONES           │  (4 métricas)   │
│  LATERAL │       (cuadrícula 50×50)        │                 │
│          │                                 │  GRÁFICAS       │
│          │                                 │                 │
│          │                                 │  CONTROL        │
├──────────┼─────────────────────────────────┴─────────────────┤
│  Info    │          TERMINAL PROLOG                          │
└──────────┴────────────────────────────────────────────────────┘
```

### Barra superior
Muestra en tiempo real: el episodio actual, cuánto tiempo lleva corriendo el sistema y el estado de las conexiones.

### Mapa central
El mapa de 50×50 celdas muestra los drones como círculos de colores:
- **Cyan brillante**: dron con batería alta, operativo.
- **Ámbar**: batería media, funcionando.
- **Rojo**: batería crítica, necesita carga pronto.
- **Gris/apagado**: dron inactivo o fuera de servicio.

Los drones se mueven en tiempo real mientras el entrenamiento está activo.

### Panel derecho — KPIs y gráficas
Los cuatro números grandes son:
- **Reward**: recompensa acumulada del episodio (mayor = mejor comportamiento).
- **Éxito**: porcentaje de entregas completadas exitosamente.
- **Infracc.**: número de veces que un dron intentó una acción prohibida.
- **Colisiones**: número de colisiones entre drones.

Debajo, las gráficas muestran la evolución de estas métricas a lo largo de los episodios.

### Terminal inferior (Prolog)
Registro en tiempo real de las decisiones del motor lógico. Cada línea muestra qué regla se activó, qué dron se vio afectado y qué acción fue bloqueada o modificada.

---

## Cómo lanzar un entrenamiento

1. En el **panel de control** (columna derecha, parte inferior), selecciona el sistema:
   - Haz clic en **A\***, **DQN** o **Neuro-DQN**.
   - Si Prolog está offline y seleccionas Neuro-DQN, aparecerá un aviso ámbar.

2. Ajusta el **número de episodios** con el slider (10 – 500). Un episodio es una sesión completa de vuelo hasta que todos los paquetes se entregan o se acaba el tiempo.

3. Elige el **modo de inicio** (solo para DQN / Neuro-DQN):
   - **Desde cero** — los drones empiezan con cerebro nuevo (pesos aleatorios). Útil para demostrar el aprendizaje desde el principio.
   - **Continuar** — los drones retoman el conocimiento de entrenamientos anteriores (se cargan los pesos guardados). Solo está disponible si ya hay datos guardados; verás cuántos episodios hay almacenados junto al selector.

4. Pulsa **Iniciar / Continuar Entrenamiento** (botón verde).

5. La barra de progreso irá avanzando. El mapa mostrará los drones moviéndose y las gráficas se irán actualizando cada 10 episodios.

6. Para detener antes de tiempo, pulsa **Detener** (botón rojo). Al terminar (o detenerse), el conocimiento se **guarda automáticamente** para poder continuar después.

> **Consejo**: Para ver resultados significativos con DQN o Neuro-DQN, usa al menos **100 episodios**. Con A* los resultados son inmediatos porque no aprende.

### Borrar datos de entrenamiento

Si quieres empezar completamente limpio (por ejemplo, para una demostración desde cero), usa el botón **⌫ Eliminar datos de entrenamiento** bajo los controles. Pide confirmación y elimina tanto los pesos guardados (checkpoints) como el historial de métricas. Esta acción no se puede deshacer.

---

## Navegación por secciones

El **menú lateral izquierdo** tiene cinco secciones. Haz clic en cualquiera para cambiar la vista central:

### Operaciones (vista por defecto)
El mapa de drones en tiempo real. Es la vista principal durante el entrenamiento activo.

### Entrenamiento
Igual que Operaciones pero pensado para seguir la evolución de las gráficas mientras el entrenamiento corre.

### Reglas Prolog
Tabla con las 12 reglas lógicas que controlan el comportamiento de Neuro-DQN:

| Tipo | Qué significa |
|---|---|
| **MASK** (ámbar) | La regla **bloquea** una acción peligrosa antes de que ocurra. |
| **REWARD** (verde) | La regla **ajusta la recompensa** para guiar el aprendizaje. |
| **NEGOC** (violeta) | La regla **coordina** entre drones cuando están cerca. |

Estas reglas son la diferencia entre DQN puro y Neuro-DQN.

### Histórico
Tabla con los últimos 50 episodios del entrenamiento activo. Muestra:
- Número de episodio
- Reward total
- Tasa de éxito
- Colisiones
- Batería promedio restante

También incluye una comparación entre los tres sistemas (A*, DQN, Neuro-DQN) con sus métricas promedio.

### Flota
Vista detallada de cada dron individual:
- **Posición** actual en la cuadrícula (X, Y)
- **Batería** con barra de color
- **Reward** acumulado en el episodio
- **Altitud** y estado operativo

---

## Interpretar los resultados

### ¿Qué es un buen resultado?

| Métrica | Señal positiva |
|---|---|
| Reward promedio | Sube episodio a episodio |
| Tasa de éxito | Se acerca al 80–90% tras muchos episodios |
| Colisiones | Se mantiene en 0 o cerca de 0 |
| Infracciones | Bajan con el tiempo (el agente aprende las reglas) |
| Batería promedio | Se mantiene alta (los drones aprenden a cargar a tiempo) |

### ¿Por qué el reward puede ser negativo al principio?
En las primeras decenas de episodios, el agente DQN explora al azar y comete muchos errores. Es normal. La curva de reward debería empezar a subir después de 30–50 episodios.

### ¿Por qué Neuro-DQN es diferente?
La línea de referencia en la gráfica de tasa de éxito marca el 90%. Neuro-DQN, al tener las reglas Prolog como guía, suele alcanzar ese umbral más rápido que DQN puro, y con menos colisiones.

### El terminal Prolog dice "R1 NFZ bloqueado" — ¿es malo?
No, es exactamente lo que debe pasar. Significa que la regla R1 detectó que un dron iba a entrar en una zona de vuelo restringida y lo bloqueó antes de que ocurriera. Es el sistema funcionando correctamente.

---

## Glosario

| Término | Significado |
|---|---|
| **Episodio** | Una ronda completa de simulación. Los drones salen, hacen entregas y el ciclo termina. |
| **Step** | Un paso de tiempo dentro del episodio. Los drones ejecutan una acción cada step. |
| **Reward** | Puntuación del comportamiento. Positiva por entregar paquetes, negativa por colisiones o zonas prohibidas. |
| **NFZ** | No-Fly Zone. Zona de vuelo restringida. Los drones no pueden entrar. |
| **Prolog** | Motor de reglas lógicas. Revisa cada acción antes de que el dron la ejecute. |
| **Máscara simbólica** | Lista de acciones permitidas/prohibidas calculada por Prolog para cada dron en cada step. |
| **WS** | WebSocket. Canal de comunicación en tiempo real entre backend y frontend. |
| **DQN** | Deep Q-Network. Algoritmo de aprendizaje por refuerzo basado en redes neuronales. |
| **Checkpoint** | Punto de guardado del estado del agente entrenado. Permite reanudar el aprendizaje. |
| **A*** | Algoritmo de búsqueda de caminos. Siempre encuentra el camino más corto, pero no aprende. |

---

## Preguntas frecuentes

**¿Puedo cerrar el navegador mientras entrena?**
Sí. El entrenamiento ocurre en el backend. Al volver a abrir el frontend, la conexión WebSocket se reconecta automáticamente en 3 segundos y reanudarás la vista en tiempo real.

**¿Se pierde el entrenamiento si cierro el backend?**
El estado del agente no se guarda automáticamente en la interfaz. Habla con el equipo técnico para usar la función de checkpoint si necesitas guardar el progreso.

**Las gráficas solo muestran unos pocos puntos, ¿es normal?**
Sí. Las gráficas se actualizan cada 10 episodios. Con pocos episodios completados, habrá pocos puntos. Son más informativas a partir de 50–100 episodios.

**El mapa no muestra movimiento, ¿qué pasó?**
Comprueba que el indicador "WS Conectado" en la barra superior esté verde. Si está rojo, espera unos segundos; el frontend intenta reconectarse solo. Si persiste, recarga la página.

**Neuro-DQN es más lento que DQN, ¿por qué?**
Neuro-DQN consulta el motor Prolog para calcular la máscara de acciones. Esto añade tiempo de procesamiento. El sistema está optimizado para consultar Prolog solo cada 5 steps y usar un cálculo rápido en los pasos intermedios.
