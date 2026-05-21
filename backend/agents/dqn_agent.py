"""
dqn_agent.py — Double-DQN con Action Masking Simbólico.

Implementa π_NS(a|s) = argmax_{a: M[a]=1} Q_θ(s, a)
donde M es la máscara binaria del motor Prolog.

Características clave:
  - Soft target-network update (τ = 0.001 cada learn step)
  - ε-greedy restringido al mask en EXPLORACIÓN y EXPLOTACIÓN
  - Huber loss + Adam + gradient clipping (max_norm=1.0)
  - Checkpoints completos (pesos + optimizer + ε + step)
"""

import logging
import os
import random
from collections import namedtuple
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)

# Coeficiente de actualización suave para la red target
TAU: float = 0.001

Transition = namedtuple(
    "Transition",
    ("state", "action", "reward", "next_state", "done", "next_mask"),
)


# ─────────────────────────────────────────────────────────────────────────────
# Red neuronal Q
# ─────────────────────────────────────────────────────────────────────────────

class PolicyNet(nn.Module):
    """Q-network: estado 7-dim → Q-values para las 8 acciones.

    Arquitectura (según experimental_protocol.md §2.2):
        Linear(7→256) → LayerNorm(256) → ReLU
        Linear(256→256) → LayerNorm(256) → ReLU
        Linear(256→8)

    LayerNorm preferida sobre BatchNorm en RL: normaliza por muestra,
    no por batch, tolerando la distribución no-estacionaria de los estados.
    """

    def __init__(
        self,
        state_dim:  int = 11,
        action_dim: int = 8,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# Buffer de replay
# ─────────────────────────────────────────────────────────────────────────────

class ReplayBuffer:
    """Buffer circular FIFO de 100k transiciones (configurable).

    Implementado como lista fija con índice de escritura circular.
    random.sample(list, k) es O(k); random.sample(deque, k) es O(n) — 3.7x más lento.
    """

    def __init__(self, capacity: int = 100_000) -> None:
        self._capacity = capacity
        self._buffer: List[Optional[Transition]] = [None] * capacity
        self._pos:  int = 0
        self._size: int = 0

    def push(self, *args: Any) -> None:
        self._buffer[self._pos] = Transition(*args)
        self._pos = (self._pos + 1) % self._capacity
        if self._size < self._capacity:
            self._size += 1

    def sample(self, batch_size: int) -> List[Transition]:
        indices = random.sample(range(self._size), batch_size)
        return [self._buffer[i] for i in indices]  # type: ignore[return-value]

    def is_ready(self, batch_size: int) -> bool:
        return self._size >= batch_size

    def __len__(self) -> int:
        return self._size


# ─────────────────────────────────────────────────────────────────────────────
# Agente Double-DQN
# ─────────────────────────────────────────────────────────────────────────────

class DQNAgent(BaseAgent):
    """Agente Double-DQN con integración nativa de Action Masking Simbólico.

    Política neuro-simbólica (formal_modeling.md §6.3):
        π_NS(a|s) = argmax_{a: M[a]=1} Q_θ(s, a)

    Exploración ε-greedy restringida (§7):
        a_t = Uniforme({a: M[a]=1})   con prob ε_t
              argmax_{a: M[a]=1} Q     con prob 1-ε_t
        ε_t = max(ε_min, ε_0 · d^t),  d=0.995, ε_min=0.05
    """

    def __init__(
        self,
        agent_id:       str,
        state_dim:      int   = 11,
        action_dim:     int   = 8,
        hidden_dim:     int   = 256,
        lr:             float = 1e-3,
        gamma:          float = 0.99,
        epsilon:        float = 1.0,
        epsilon_min:    float = 0.05,
        epsilon_decay:  float = 0.99,
        batch_size:     int   = 64,
        buffer_capacity: int  = 100_000,
        tau:            float = TAU,
    ) -> None:
        super().__init__(agent_id, state_dim, action_dim)

        self.device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size    = batch_size
        self.tau           = tau
        self._learn_step: int = 0

        # Redes policy y target
        self.policy_net = PolicyNet(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net = PolicyNet(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer    = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_capacity)

        logger.info(
            "DQNAgent '%s' | device=%s | hidden=%d | τ=%.4f | ε₀=%.2f",
            agent_id, self.device, hidden_dim, tau, epsilon,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Selección de acción
    # ──────────────────────────────────────────────────────────────────────────

    def select_action(
        self,
        state: np.ndarray,
        symbolic_mask: Optional[np.ndarray] = None,
    ) -> int:
        """ε-greedy con máscara simbólica aplicada ANTES de elegir (explorar o explotar).

        El filtrado previo a la exploración evita que el agente aprenda distribuciones
        Q para acciones inválidas, acelerando la convergencia (§4 formal_modeling.md).
        """
        valid_actions = self._valid_from_mask(symbolic_mask)

        # Exploración: uniforme sobre acciones válidas
        if random.random() < self.epsilon:
            return random.choice(valid_actions)

        # Explotación: argmax Q sobre acciones válidas
        # Inferencia en CPU para batch=1 (evita overhead de transferencia GPU↔CPU).
        with torch.no_grad():
            state_t = torch.from_numpy(np.asarray(state, dtype=np.float32)).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_t).squeeze(0).detach().cpu().numpy()

        # Enmascarar acciones inválidas con -∞ para que nunca sean el argmax
        if symbolic_mask is not None:
            masked_q = np.where(symbolic_mask.astype(bool), q_values, -np.inf)
        else:
            masked_q = q_values

        return int(np.argmax(masked_q))

    def _valid_from_mask(self, mask: Optional[np.ndarray]) -> List[int]:
        """Extrae índices de acciones permitidas desde la máscara binaria."""
        if mask is None:
            return list(range(self.action_dim))
        valid = [i for i, m in enumerate(mask) if float(m) > 0.0]
        # Fail-safe: si la máscara bloquea todo, se permite todo
        return valid if valid else list(range(self.action_dim))

    # ──────────────────────────────────────────────────────────────────────────
    # Memoria
    # ──────────────────────────────────────────────────────────────────────────

    def remember(
        self,
        state:      np.ndarray,
        action:     int,
        reward:     float,
        next_state: np.ndarray,
        done:       bool,
        next_mask:  Optional[np.ndarray] = None,
    ) -> None:
        # Store next_mask (fail-safe: all-ones if None)
        if next_mask is None:
            nm = np.ones(self.action_dim, dtype=np.float32)
        else:
            nm = np.asarray(next_mask, dtype=np.float32)
        self.replay_buffer.push(state, action, reward, next_state, done, nm)
        self.total_reward  += reward
        self.episode_steps += 1

    # ──────────────────────────────────────────────────────────────────────────
    # Paso de aprendizaje Double-DQN
    # ──────────────────────────────────────────────────────────────────────────

    def learn(
        self,
        batch: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """Un paso de gradiente Double-DQN con soft-update de la red target.

        Target Double-DQN (formal_modeling.md §6.1):
            y = r + γ · Q_θ⁻(s', argmax_{a'} Q_θ(s', a'))

        Soft update cada step (§2.2 experimental_protocol):
            θ⁻ ← τ·θ + (1−τ)·θ⁻    (τ = 0.001)
        """
        if not self.replay_buffer.is_ready(self.batch_size):
            return {"loss": 0.0, "epsilon": self.epsilon}

        transitions = self.replay_buffer.sample(self.batch_size)
        b = Transition(*zip(*transitions))

        states      = torch.from_numpy(np.array(b.state,      dtype=np.float32)).to(self.device)
        actions     = torch.from_numpy(np.array(b.action,     dtype=np.int64)).unsqueeze(1).to(self.device)
        rewards     = torch.from_numpy(np.array(b.reward,     dtype=np.float32)).to(self.device)
        next_states = torch.from_numpy(np.array(b.next_state, dtype=np.float32)).to(self.device)
        dones       = torch.from_numpy(np.array(b.done,       dtype=np.bool_)).to(self.device)

        # Q-values actuales: Q_θ(s, a)
        current_q = self.policy_net(states).gather(1, actions).squeeze(1)

        # Double-DQN: policy_net elige, target_net evalúa
        # Handle action masking for next states when computing Double-DQN target
        with torch.no_grad():
            # policy values for next states (used to select best actions)
            policy_next_q = self.policy_net(next_states)

            # reconstruct next_masks from batch
            next_masks = np.array(b.next_mask, dtype=np.bool_)
            mask_tensor = torch.BoolTensor(next_masks).to(self.device)

            # Fail-safe: if any mask row is all False, set it to all True
            invalid_rows = ~mask_tensor.any(dim=1)
            if invalid_rows.any():
                mask_tensor[invalid_rows, :] = True

            # Mask invalid actions with -inf so they are never selected
            policy_next_q_masked = policy_next_q.masked_fill(~mask_tensor, float("-inf"))

            best_next_actions = policy_next_q_masked.argmax(1, keepdim=True)

            next_q = self.target_net(next_states).gather(1, best_next_actions).squeeze(1)
            next_q[dones] = 0.0
            target_q = rewards + self.gamma * next_q

        # Huber loss (Smooth L1) — robusta a outliers con penalizaciones grandes
        loss = F.smooth_l1_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Soft update: θ⁻ ← τ·θ + (1−τ)·θ⁻  (cada step, no cada K steps)
        self._soft_update_target()

        self._learn_step += 1

        loss_val = loss.item()
        if self._learn_step % 1000 == 0:
            logger.debug(
                "DQNAgent '%s' | learn_step=%d | loss=%.5f | ε=%.4f | buffer=%d",
                self.agent_id, self._learn_step, loss_val,
                self.epsilon, len(self.replay_buffer),
            )

        return {"loss": loss_val, "epsilon": self.epsilon}

    def update_epsilon(self) -> None:
        """Decrementa epsilon una vez por episodio (formal_modeling.md §7: ε_t = max(ε_min, ε_0·d^t), d=0.995)."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def _soft_update_target(self) -> None:
        """Polyak averaging: θ⁻ ← τ·θ + (1−τ)·θ⁻"""
        for t_p, p_p in zip(
            self.target_net.parameters(),
            self.policy_net.parameters(),
        ):
            t_p.data.copy_(self.tau * p_p.data + (1.0 - self.tau) * t_p.data)

    # ──────────────────────────────────────────────────────────────────────────
    # Checkpoints
    # ──────────────────────────────────────────────────────────────────────────

    def save_checkpoint(self, path: str) -> None:
        """Persiste pesos, optimizer, ε y step counter en un archivo .pt."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(
            {
                "agent_id":   self.agent_id,
                "policy_net": self.policy_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer":  self.optimizer.state_dict(),
                "epsilon":    self.epsilon,
                "learn_step": self._learn_step,
            },
            path,
        )
        logger.info("Checkpoint guardado: %s (step=%d)", path, self._learn_step)

    def load_checkpoint(self, path: str) -> None:
        """Restaura el estado completo de entrenamiento desde un checkpoint."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint no encontrado: {path}")
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.policy_net.load_state_dict(ckpt["policy_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon     = ckpt.get("epsilon", self.epsilon)
        self._learn_step = ckpt.get("learn_step", 0)
        logger.info(
            "Checkpoint cargado: %s | ε=%.4f | step=%d",
            path, self.epsilon, self._learn_step,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Serialización
    # ──────────────────────────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "epsilon":     round(self.epsilon, 4),
            "learn_step":  self._learn_step,
            "buffer_size": len(self.replay_buffer),
            "device":      str(self.device),
        })
        return base
