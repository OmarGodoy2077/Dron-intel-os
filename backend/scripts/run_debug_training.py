"""run_debug_training.py

Ejecuta un entrenamiento corto (configurable) sin levantar FastAPI.
Guarda un CSV con métricas por episodio y un CSV con losses por paso.

Uso (desde carpeta backend):
    python scripts/run_debug_training.py
"""
import asyncio
import csv
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np

from environment.city_env import CyberCityEnv
from environment.dynamics import DynamicsEngine
from logic.neuro_symbolic_bridge import NeuroSymbolicBridge
from agents.dqn_agent import DQNAgent

# Parámetros
GRID = 50
N_DRONES = 5
EPISODES = 200
MAX_STEPS = 200
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run():
    env = CyberCityEnv(grid_size=GRID, num_drones=N_DRONES, max_steps=MAX_STEPS)
    dyn = DynamicsEngine(grid_size=GRID)
    try:
        bridge = NeuroSymbolicBridge(os.path.join(os.path.dirname(__file__), '..', 'logic', 'rules.pl'))
    except Exception as e:
        print('Bridge Prolog no disponible:', e)
        bridge = None

    agents = [DQNAgent(agent_id=f'drone_{i}', state_dim=env.observation_space[f'drone_{i}'].shape[0], action_dim=8) for i in range(N_DRONES)]

    episodes_summary: List[Dict[str, Any]] = []
    losses_records: List[Dict[str, Any]] = []

    for ep in range(EPISODES):
        obs, _ = env.reset()
        dyn.reset()
        ep_reward = np.zeros(N_DRONES, dtype=np.float32)
        ep_losses = []

        for step in range(MAX_STEPS):
            dstate = dyn.step()
            env.apply_dynamics(dstate)

            actions = []
            for i, agent in enumerate(agents):
                if not env.drone_alive[i]:
                    actions.append(6)
                    continue
                state = obs[f'drone_{i}']
                mask = env.fast_action_mask(i)
                if bridge is not None and step % 5 == 0:
                    try:
                        pmask = bridge.get_action_mask(f'drone_{i}', env.get_state_dict(i))
                        mask = mask * pmask
                    except Exception:
                        pass
                a = agent.select_action(state, mask)
                actions.append(a)

            next_obs, rewards, dones, truncated, infos = env.step(np.array(actions))

            # compute next_mask and store transitions
            for i, agent in enumerate(agents):
                if not env.drone_alive[i]:
                    continue
                r = float(rewards[i])
                # next_mask (refine with prolog occasionally)
                try:
                    next_mask = env.fast_action_mask(i)
                    if bridge is not None and step % 5 == 0:
                        next_mask = next_mask * bridge.get_action_mask(f'drone_{i}', env.get_state_dict(i))
                except Exception:
                    next_mask = None

                agent.remember(obs[f'drone_{i}'], actions[i], r, next_obs[f'drone_{i}'], bool(dones[i]), next_mask=next_mask)

                if step % 4 == 0:
                    info = agent.learn()
                    if isinstance(info, dict) and 'loss' in info:
                        losses_records.append({'episode': ep, 'step': step, 'agent': agent.agent_id, 'loss': float(info['loss']), 'epsilon': float(info.get('epsilon', -1.0))})

                ep_reward[i] += r

            obs = next_obs
            if dones.all() or truncated.all():
                break

        episodes_summary.append({
            'episode': ep,
            'total_reward': float(ep_reward.sum()),
            'deliveries': int(env.package_delivered.sum()),
            'steps': int(env.current_step)
        })
        print(f'Ep {ep} | reward={ep_reward.sum():.1f} | deliveries={env.package_delivered.sum()} | steps={env.current_step}')

    # Guardar CSVs
    ep_file = os.path.join(OUTPUT_DIR, 'debug_episodes.csv')
    with open(ep_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['episode','total_reward','deliveries','steps'])
        writer.writeheader()
        writer.writerows(episodes_summary)

    loss_file = os.path.join(OUTPUT_DIR, 'debug_losses.csv')
    with open(loss_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['episode','step','agent','loss','epsilon'])
        writer.writeheader()
        writer.writerows(losses_records)

    print('Resultados guardados en:', ep_file, loss_file)


if __name__ == '__main__':
    run()
