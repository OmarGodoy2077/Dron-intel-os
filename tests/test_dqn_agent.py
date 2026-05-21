"""test_dqn_agent.py — Tests del agente Double-DQN y sus componentes.

Verifica:
  - ReplayBuffer circular (capacidad, sobrescritura FIFO, is_ready).
  - PolicyNet: dimensiones de entrada/salida.
  - select_action: respeto estricto de la máscara simbólica en exploración y explotación.
  - learn(): paso de gradiente válido (loss finito, no-op con buffer insuficiente).
  - Soft update de la red target (Polyak).
  - update_epsilon: decay por episodio acotado a epsilon_min.
  - Checkpoints: round-trip de pesos + ε + step.
"""

import numpy as np
import pytest
import torch

from agents.dqn_agent import DQNAgent, PolicyNet, ReplayBuffer, TAU


# ── ReplayBuffer ────────────────────────────────────────────────────────────────

class TestReplayBuffer:
    def _push(self, rb, n):
        for k in range(n):
            rb.push(
                np.zeros(11, dtype=np.float32), k % 8, float(k),
                np.zeros(11, dtype=np.float32), False, np.ones(8, dtype=np.float32),
            )

    def test_starts_empty(self):
        rb = ReplayBuffer(10)
        assert len(rb) == 0
        assert not rb.is_ready(1)

    def test_is_ready_threshold(self):
        rb = ReplayBuffer(100)
        self._push(rb, 64)
        assert rb.is_ready(64)
        assert not rb.is_ready(65)

    def test_circular_overwrite_caps_at_capacity(self):
        rb = ReplayBuffer(10)
        self._push(rb, 25)  # más que la capacidad
        assert len(rb) == 10  # no crece más allá de capacity

    def test_sample_returns_requested_size(self):
        rb = ReplayBuffer(100)
        self._push(rb, 50)
        batch = rb.sample(16)
        assert len(batch) == 16


# ── PolicyNet ──────────────────────────────────────────────────────────────────

class TestPolicyNet:
    def test_output_shape(self):
        net = PolicyNet(state_dim=11, action_dim=8)
        out = net(torch.zeros(4, 11))  # batch de 4
        assert out.shape == (4, 8)

    def test_single_sample_inference(self):
        net = PolicyNet(state_dim=11, action_dim=8)
        out = net(torch.zeros(1, 11))
        assert out.shape == (1, 8)


# ── select_action con máscara simbólica ──────────────────────────────────────

class TestSelectAction:
    def test_returns_valid_index_range(self, dqn_agent):
        a = dqn_agent.select_action(np.zeros(11, dtype=np.float32))
        assert 0 <= a < 8

    def test_mask_respected_in_exploration(self, dqn_agent):
        """Con ε=1 (toda exploración), solo debe elegir acciones permitidas por la máscara."""
        dqn_agent.epsilon = 1.0
        mask = np.zeros(8, dtype=np.float32)
        mask[3] = 1.0  # solo 'mover_s'
        chosen = {dqn_agent.select_action(np.zeros(11, dtype=np.float32), mask) for _ in range(50)}
        assert chosen == {3}

    def test_mask_respected_in_exploitation(self, dqn_agent):
        """Con ε=0 (explotación), el argmax nunca debe caer en una acción enmascarada."""
        dqn_agent.epsilon = 0.0
        mask = np.zeros(8, dtype=np.float32)
        mask[[2, 5]] = 1.0
        for _ in range(20):
            assert dqn_agent.select_action(np.random.randn(11).astype(np.float32), mask) in (2, 5)

    def test_all_zero_mask_failsafe(self, dqn_agent):
        """Máscara todo-cero no debe romper: se permite cualquier acción (fail-safe)."""
        dqn_agent.epsilon = 1.0
        a = dqn_agent.select_action(np.zeros(11, dtype=np.float32), np.zeros(8, dtype=np.float32))
        assert 0 <= a < 8


# ── learn() y soft update ────────────────────────────────────────────────────

class TestLearn:
    def test_noop_when_buffer_insufficient(self, dqn_agent):
        info = dqn_agent.learn()
        assert info["loss"] == 0.0

    @pytest.mark.slow
    def test_learn_produces_finite_loss(self, dqn_agent):
        for k in range(128):
            dqn_agent.remember(
                np.random.randn(11).astype(np.float32), k % 8, float(np.random.randn()),
                np.random.randn(11).astype(np.float32), bool(k % 10 == 0),
                next_mask=np.ones(8, dtype=np.float32),
            )
        info = dqn_agent.learn()
        assert info["loss"] >= 0.0
        assert np.isfinite(info["loss"])

    @pytest.mark.slow
    def test_soft_update_moves_target_toward_policy(self, dqn_agent):
        """Tras un soft update, los pesos target deben acercarse a los policy (τ pequeño)."""
        # Perturbar la policy net para que difiera de la target
        with torch.no_grad():
            for p in dqn_agent.policy_net.parameters():
                p.add_(torch.randn_like(p))
        before = [tp.clone() for tp in dqn_agent.target_net.parameters()]
        dqn_agent._soft_update_target()
        for tp_before, tp_after, pp in zip(
            before, dqn_agent.target_net.parameters(), dqn_agent.policy_net.parameters()
        ):
            # target_after = τ·policy + (1-τ)·target_before → debe moverse hacia policy
            expected = TAU * pp.data + (1 - TAU) * tp_before
            assert torch.allclose(tp_after.data, expected, atol=1e-6)


# ── Epsilon ────────────────────────────────────────────────────────────────────

class TestEpsilon:
    def test_decay_reduces_epsilon(self, dqn_agent):
        dqn_agent.epsilon = 1.0
        dqn_agent.update_epsilon()
        assert dqn_agent.epsilon == pytest.approx(1.0 * dqn_agent.epsilon_decay)

    def test_decay_floored_at_minimum(self, dqn_agent):
        dqn_agent.epsilon = dqn_agent.epsilon_min
        for _ in range(100):
            dqn_agent.update_epsilon()
        assert dqn_agent.epsilon >= dqn_agent.epsilon_min - 1e-9

    def test_decay_per_episode_not_collapsing_fast(self, dqn_agent):
        """El decay 0.99/episodio debe mantener exploración significativa decenas de episodios."""
        dqn_agent.epsilon = 1.0
        for _ in range(50):
            dqn_agent.update_epsilon()
        assert dqn_agent.epsilon > 0.3  # aún explora a los 50 episodios


# ── Checkpoints ──────────────────────────────────────────────────────────────

class TestCheckpoint:
    def test_save_and_load_roundtrip(self, dqn_agent, tmp_path):
        # Entrenar un poco para que ε y step difieran de los valores iniciales
        for k in range(128):
            dqn_agent.remember(
                np.random.randn(11).astype(np.float32), k % 8, 1.0,
                np.random.randn(11).astype(np.float32), False,
                next_mask=np.ones(8, dtype=np.float32),
            )
        dqn_agent.learn()
        dqn_agent.update_epsilon()

        path = str(tmp_path / "ckpt.pt")
        dqn_agent.save_checkpoint(path)

        fresh = DQNAgent(agent_id="drone_0", state_dim=11, action_dim=8)
        fresh.load_checkpoint(path)

        assert fresh.epsilon == pytest.approx(dqn_agent.epsilon)
        assert fresh._learn_step == dqn_agent._learn_step
        # Los pesos cargados deben coincidir con los guardados
        for p_src, p_dst in zip(
            dqn_agent.policy_net.parameters(), fresh.policy_net.parameters()
        ):
            assert torch.allclose(p_src.data, p_dst.data)

    def test_load_missing_file_raises(self, dqn_agent, tmp_path):
        with pytest.raises(FileNotFoundError):
            dqn_agent.load_checkpoint(str(tmp_path / "no_existe.pt"))
