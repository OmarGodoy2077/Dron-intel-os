% ============================================================
% SMART-SWARM NEURO-SYMBOLIC ENGINE — MOTOR LÓGICO PROLOG
% Proyecto: Dron-Intel-OS  |  Motor: SWI-Prolog via pyswip
% ============================================================
%
% ARQUITECTURA DE INFLUENCIA:
%   Action Masking  → bloquea acciones ANTES de que el DQN decida
%   Reward Shaping  → modifica la recompensa DESPUÉS de cada paso
%   Ambas afectan: π_θ(a|s) = argmax_a Q_θ(s,a) ∩ MaskProlog(s)
% ============================================================

% ----- HECHOS DINÁMICOS (actualizados desde Python en cada step) -----
:- dynamic no_fly_zone/1.           % no_fly_zone(celda(X,Y))
:- dynamic estacion_carga/2.        % estacion_carga(ID, libre|ocupada)
:- dynamic tormenta/1.              % tormenta(region_id)
:- dynamic viento/2.                % viento(Direccion, Intensidad_kmh)
:- dynamic agente_en_celda/2.       % agente_en_celda(AgentID, celda(X,Y))
:- dynamic paquete/3.               % paquete(ID, Tipo, Prioridad)
:- dynamic bateria_agente/2.        % bateria_agente(AgentID, Nivel_0_100)
:- dynamic zona_congestionada/1.    % zona_congestionada(region_id)
:- dynamic ruta_segura/3.           % ruta_segura(Origen, Destino, [Celdas])
:- dynamic region_limites/5.        % region_limites(ID, Xmin, Xmax, Ymin, Ymax)
:- dynamic agente_lleva_paquete/2.  % agente_lleva_paquete(AgentID, PaqueteID)
:- dynamic agente_destino/2.        % agente_destino(AgentID, celda(X,Y))


% ============================================================
% REGLA 1: ZONA DE VUELO PROHIBIDA  (No-Fly Zone)
% Tipo: Action Masking
% Efecto: Penalización -100 en reward + bloqueo de la acción
% ============================================================
zona_prohibida_activa(Celda) :-
    no_fly_zone(Celda).

%% accion_invalida(+AgentID, +Accion)
%  Verdadero si la acción lleva al agente a una zona prohibida.
accion_invalida(AgentID, Accion) :-
    agente_en_celda(AgentID, CeldaActual),
    celda_objetivo(CeldaActual, Accion, CeldaDestino),
    zona_prohibida_activa(CeldaDestino),
    ignore(format("[MASK-R1] ~w: acción ~w bloqueada → zona prohibida ~w~n",
           [AgentID, Accion, CeldaDestino])).


% ============================================================
% REGLA 2: BATERÍA CRÍTICA — MODO EMERGENCIA
% Tipo: Action Masking estricto
% Efecto: Solo permite {cargar, aterrizar}; penaliza -50/step
% ============================================================

%% bateria_critica(+AgentID)
%  Umbral de seguridad: < 15 % de carga restante.
bateria_critica(AgentID) :-
    bateria_agente(AgentID, Nivel),
    Nivel < 15,
    ignore(format("[ALERT-R2] ~w: batería crítica ~w%%~n", [AgentID, Nivel])).

%% solo_acciones_emergencia(+AgentID)
%  El agente debe resolver la emergencia antes de cualquier misión.
solo_acciones_emergencia(AgentID) :-
    bateria_critica(AgentID),
    ignore(format("[MASK-R2] ~w: solo cargar/aterrizar disponibles~n", [AgentID])).

%% accion_emergencia_valida(+AgentID, +Accion)
accion_emergencia_valida(AgentID, Accion) :-
    solo_acciones_emergencia(AgentID),
    member(Accion, [cargar, aterrizar]).


% ============================================================
% REGLA 3: COLISIÓN INMINENTE ENTRE AGENTES
% Tipo: Action Masking preventivo
% Efecto: Penalización -200 si dos agentes comparten celda
% ============================================================

%% colision_inminente(+Agente1, +Agente2)
%  Detecta cuando dos drones ya están en la misma celda.
colision_inminente(Agente1, Agente2) :-
    agente_en_celda(Agente1, Celda),
    agente_en_celda(Agente2, Celda),
    Agente1 \= Agente2,
    ignore(format("[PELIGRO-R3] Colisión: ~w ↔ ~w en ~w~n", [Agente1, Agente2, Celda])).

%% accion_causa_colision(+AgentID, +Accion)
%  Verdadero si la acción movería al agente a una celda ocupada.
accion_causa_colision(AgentID, Accion) :-
    agente_en_celda(AgentID, CeldaActual),
    celda_objetivo(CeldaActual, Accion, CeldaDestino),
    agente_en_celda(OtroAgente, CeldaDestino),
    AgentID \= OtroAgente,
    ignore(format("[MASK-R3] ~w: movimiento a ~w bloqueado (ocupa ~w)~n",
           [AgentID, CeldaDestino, OtroAgente])).


% ============================================================
% REGLA 4: CONFLICTO DE CELDA — MÚLTIPLES DRONES
% Tipo: Reward Shaping negativo
% Efecto: -30 × (N-1) por cada dron extra en la misma celda
% ============================================================

%% conflicto_celda(+Celda, -Agentes)
%  Lista todos los agentes en Celda si hay más de uno.
conflicto_celda(Celda, Agentes) :-
    findall(A, agente_en_celda(A, Celda), Agentes),
    length(Agentes, N),
    N > 1.

%% penalizacion_conflicto(+Celda, -Penalizacion)
%  Calcula penalización proporcional al número de conflictos.
penalizacion_conflicto(Celda, Penalizacion) :-
    conflicto_celda(Celda, Agentes),
    length(Agentes, N),
    Penalizacion is -30 * (N - 1),
    ignore(format("[REWARD-R4] Conflicto en ~w (N=~w): ~w pts~n",
           [Celda, N, Penalizacion])).


% ============================================================
% REGLA 5: ESTACIÓN DE CARGA OCUPADA
% Tipo: Action Masking + Reward Shaping
% Efecto: Bloquea acción cargar; -20 si intenta cargar en ocupada
% ============================================================

%% estacion_carga_ocupada(+Estacion)
estacion_carga_ocupada(Estacion) :-
    estacion_carga(Estacion, ocupada).

%% puede_cargar(+AgentID)
%  El agente está sobre una estación libre.
puede_cargar(AgentID) :-
    agente_en_celda(AgentID, Celda),
    estacion_carga(Celda, libre),
    \+ estacion_carga_ocupada(Celda).

%% accion_carga_invalida(+AgentID)
%  La estación bajo el dron está ocupada → bloqueo.
accion_carga_invalida(AgentID) :-
    agente_en_celda(AgentID, Celda),
    estacion_carga(Celda, _),
    estacion_carga_ocupada(Celda),
    ignore(format("[MASK-R5] ~w: estación ~w ocupada, carga bloqueada~n",
           [AgentID, Celda])).


% ============================================================
% REGLA 6: PRIORIDAD DE ENTREGA MÉDICA
% Tipo: Reward Shaping positivo
% Efecto: +150 entrega médica, +50 estándar, +30 por rutas médicas
% ============================================================

%% prioridad_entrega(+Paquete, -Tipo)
prioridad_entrega(Paquete, medico) :-
    paquete(Paquete, medico, _),
    ignore(format("[PRIO-R6] Entrega médica urgente: ~w~n", [Paquete])).

%% bonus_entrega(+Paquete, -Bonus)
bonus_entrega(Paquete, Bonus) :-
    paquete(Paquete, Tipo, _),
    (Tipo = medico -> Bonus = 150 ; Bonus = 50),
    ignore(format("[REWARD-R6] Bonus entrega ~w: +~w pts~n", [Tipo, Bonus])).

%% entrega_urgente(+AgentID)
%  El agente actualmente transporta carga médica.
entrega_urgente(AgentID) :-
    agente_lleva_paquete(AgentID, Paquete),
    prioridad_entrega(Paquete, medico).


% ============================================================
% REGLA 7: TORMENTA ACTIVA EN REGIÓN
% Tipo: Action Masking + Reward Shaping
% Efecto: Bloquea vuelo en zona de tormenta; penaliza -80
% ============================================================

%% tormenta_activa(+Region)
tormenta_activa(Region) :-
    tormenta(Region),
    ignore(format("[CLIMA-R7] Tormenta activa: ~w~n", [Region])).

%% celda_en_region(+Celda, +Region)
celda_en_region(celda(X, Y), Region) :-
    region_limites(Region, Xmin, Xmax, Ymin, Ymax),
    X >= Xmin, X =< Xmax,
    Y >= Ymin, Y =< Ymax.

%% vuelo_en_tormenta(+AgentID, +CeldaDestino)
%  Verdadero si el destino está dentro de una tormenta activa.
vuelo_en_tormenta(AgentID, CeldaDestino) :-
    celda_en_region(CeldaDestino, Region),
    tormenta_activa(Region),
    ignore(format("[MASK-R7] ~w: vuelo bloqueado en tormenta ~w~n",
           [AgentID, Region])).

%% accion_invalida_tormenta(+AgentID, +Accion)
accion_invalida_tormenta(AgentID, Accion) :-
    agente_en_celda(AgentID, CeldaActual),
    celda_objetivo(CeldaActual, Accion, CeldaDestino),
    vuelo_en_tormenta(AgentID, CeldaDestino).


% ============================================================
% REGLA 8: VIENTO FUERTE — AFECTA MOVIMIENTO
% Tipo: Reward Shaping negativo
% Efecto: -15 por moverse contra viento > 60 km/h
% ============================================================

%% viento_fuerte(+Direccion)
%  Intensidad supera umbral operativo de los drones.
viento_fuerte(Direccion) :-
    viento(Direccion, Intensidad),
    Intensidad > 60,
    ignore(format("[CLIMA-R8] Viento fuerte ~w km/h desde ~w~n",
           [Intensidad, Direccion])).

%% movimiento_contra_viento(+Accion)
movimiento_contra_viento(mover_n) :- viento_fuerte(norte).
movimiento_contra_viento(mover_s) :- viento_fuerte(sur).
movimiento_contra_viento(mover_e) :- viento_fuerte(este).
movimiento_contra_viento(mover_o) :- viento_fuerte(oeste).

%% penalizacion_viento(+Accion, -Penalizacion)
penalizacion_viento(Accion, -15) :-
    movimiento_contra_viento(Accion),
    ignore(format("[REWARD-R8] Penalización: movimiento contra viento fuerte~n")).


% ============================================================
% REGLA 9: ZONA CONGESTIONADA — TRÁFICO AÉREO DENSO
% Tipo: Action Masking suave + Reward Shaping
% Efecto: -40 por entrar en zona con ≥ 3 drones; recomienda desvío
% ============================================================

%% zona_congestionada_activa(+Region)
%  Se activa cuando hay 3 o más drones en la misma región.
zona_congestionada_activa(Region) :-
    zona_congestionada(Region),
    findall(A, (agente_en_celda(A, C), celda_en_region(C, Region)), Agentes),
    length(Agentes, N),
    N >= 3.

%% desvio_necesario(+AgentID, +CeldaDestino)
%  Señaliza necesidad de ruta alternativa.
desvio_necesario(AgentID, CeldaDestino) :-
    celda_en_region(CeldaDestino, Region),
    zona_congestionada_activa(Region),
    ignore(format("[AVISO-R9] ~w: zona ~w congestionada, desvío recomendado~n",
           [AgentID, Region])).


% ============================================================
% REGLA 10: RUTA MÁS EFICIENTE PRE-CALCULADA
% Tipo: Reward Shaping positivo (guía exploración)
% Efecto: +20 por seguir una ruta eficiente conocida
% ============================================================

%% ruta_mas_eficiente(+Origen, +Destino, -Ruta)
%  Recupera ruta pre-calculada si no está bloqueada por eventos.
ruta_mas_eficiente(Origen, Destino, Ruta) :-
    ruta_segura(Origen, Destino, Ruta),
    \+ ruta_bloqueada(Ruta),
    ignore(format("[RUTA-R10] Ruta eficiente: ~w → ~w~n", [Origen, Destino])).

%% ruta_bloqueada(+Ruta)
ruta_bloqueada(Ruta) :-
    member(Celda, Ruta),
    (zona_prohibida_activa(Celda) ; tormenta_activa_en_celda(Celda)).

%% tormenta_activa_en_celda(+Celda)
tormenta_activa_en_celda(Celda) :-
    celda_en_region(Celda, Region),
    tormenta_activa(Region).

%% bonus_ruta_optima(+AgentID, -Bonus)
bonus_ruta_optima(AgentID, 20) :-
    agente_en_celda(AgentID, Origen),
    agente_destino(AgentID, Destino),
    ruta_mas_eficiente(Origen, Destino, _),
    ignore(format("[REWARD-R10] +20 pts por ruta eficiente para ~w~n", [AgentID])).


% ============================================================
% REGLA 11: NEGOCIACIÓN DE DERECHO DE PASO
% Tipo: Reward Shaping coordinativo
% Efecto: +25 al agente que cede; -10 al que interrumpe
% ============================================================

%% prioridad_paso(+Agente1, +Agente2, -Ganador)
%  Desempate jerárquico: médico > batería baja > ninguno.
prioridad_paso(A1, A2, A1) :-
    entrega_urgente(A1),
    \+ entrega_urgente(A2), !.
prioridad_paso(A1, A2, A2) :-
    entrega_urgente(A2),
    \+ entrega_urgente(A1), !.
prioridad_paso(A1, A2, A1) :-
    bateria_agente(A1, B1),
    bateria_agente(A2, B2),
    B1 < B2, !,
    ignore(format("[NEGOC-R11] ~w prioridad por batería baja (~w%%)~n", [A1, B1])).
prioridad_paso(A1, _A2, A1).   % Desempate: el primero en solicitar.

%% negociar_derecho_paso(+Agente1, +Agente2)
%  Determina quién tiene el derecho de paso y emite la decisión.
negociar_derecho_paso(Agente1, Agente2) :-
    prioridad_paso(Agente1, Agente2, Ganador),
    (Ganador = Agente1 -> Cedente = Agente2 ; Cedente = Agente1),
    ignore(format("[NEGOC-R11] ~w tiene paso; ~w debe ceder (+25/-10 pts)~n",
           [Ganador, Cedente])).


% ============================================================
% REGLA 12: PREDICCIÓN DE FALLO DE BATERÍA EN RUTA
% Tipo: Action Masking crítico
% Efecto: Bloquea rutas que agotarían batería; penaliza -500 caída
% ============================================================

consumo_por_celda(1.5).   % % de batería por celda en vuelo normal
consumo_en_tormenta(3.5). % % adicional por celda en tormenta

%% consumo_estimado_ruta(+Ruta, -Consumo)
consumo_estimado_ruta(Ruta, Consumo) :-
    length(Ruta, N),
    consumo_por_celda(Base),
    Consumo is N * Base.

%% prediccion_fallo_bateria(+AgentID, +Ruta)
%  Verdadero si la batería será insuficiente para completar la ruta.
prediccion_fallo_bateria(AgentID, Ruta) :-
    bateria_agente(AgentID, BateriaActual),
    consumo_estimado_ruta(Ruta, Consumo),
    Restante is BateriaActual - Consumo,
    Restante < 10,
    ignore(format("[CRITICO-R12] ~w: batería insuficiente. ~w%% → consumo estimado ~w%%~n",
           [AgentID, BateriaActual, Consumo])).

%% ruta_segura_bateria(+AgentID, +Ruta)
ruta_segura_bateria(AgentID, Ruta) :-
    \+ prediccion_fallo_bateria(AgentID, Ruta).

%% accion_riesgo_bateria(+AgentID, +Accion)
%  Bloquea movimiento si no hay suficiente batería para retornar.
accion_riesgo_bateria(AgentID, Accion) :-
    member(Accion, [mover_n, mover_s, mover_e, mover_o, despegar]),
    bateria_agente(AgentID, Nivel),
    Nivel < 20,
    \+ puede_cargar(AgentID),
    ignore(format("[MASK-R12] ~w: batería ~w%% insuficiente para movimiento seguro~n",
           [AgentID, Nivel])).


% ============================================================
% UTILIDADES: Geometría de celdas y navegación
% ============================================================

%% celda_objetivo(+CeldaActual, +Accion, -CeldaDestino)
%  Calcula celda de destino para cada acción de movimiento.
celda_objetivo(celda(X, Y), mover_n,  celda(X,  Y1)) :- Y1 is Y + 1.
celda_objetivo(celda(X, Y), mover_s,  celda(X,  Y1)) :- Y1 is Y - 1.
celda_objetivo(celda(X, Y), mover_e,  celda(X1, Y))  :- X1 is X + 1.
celda_objetivo(celda(X, Y), mover_o,  celda(X1, Y))  :- X1 is X - 1.
celda_objetivo(celda(X, Y), despegar, celda(X,  Y)).
celda_objetivo(celda(X, Y), aterrizar,celda(X,  Y)).
celda_objetivo(celda(X, Y), esperar,  celda(X,  Y)).
celda_objetivo(celda(X, Y), cargar,   celda(X,  Y)).

%% distancia_manhattan(+celda(X1,Y1), +celda(X2,Y2), -D)
distancia_manhattan(celda(X1, Y1), celda(X2, Y2), D) :-
    D is abs(X2 - X1) + abs(Y2 - Y1).
