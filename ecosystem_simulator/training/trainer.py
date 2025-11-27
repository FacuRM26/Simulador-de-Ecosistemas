"""
Módulo principal para el entrenamiento del modelo RL.
"""
import os
from unittest import result
os.environ.pop("AIR_VERBOSITY", None)
import logging
import re
import random
import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
import torch
import ray
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.env import PettingZooEnv
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.policy.policy import PolicySpec

from ..environment.multi_agent_ecosystem import MultiAgentEcosystem
from .callbacks import PerAgentAndReasonMetrics

# Para importar el visualizer
try:
    from ..utils.pygame_visualizer import EcosystemVisualizer
    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False
    print("Pygame no disponible - visualización desactivada")

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def main(enable_visualization: bool = False):
    """Función principal de entrenamiento con visualización opcional"""
    ray.init(ignore_reinit_error=True)

    # --- Configuración del entorno ---
    ENV_CFG = {
        "n_agents": 8,
        "veg_density": 15,
        "water_density": 10,
        "map_width": 800,
        "map_height": 500,
        "max_steps": 350,
        "n_predators": 2,
    }

    # Parámetros de entrenamiento
    NUM_RUNNERS = 4
    FRAG = 350 #ENV_CFG["max_steps"]
    TOTAL_BATCH = NUM_RUNNERS * FRAG

    register_env(
        "multi_eco",
        lambda cfg: ParallelPettingZooEnv(MultiAgentEcosystem(**cfg))
    )

    config = (
        PPOConfig()
        .environment(env="multi_eco", env_config=ENV_CFG)
        .callbacks(PerAgentAndReasonMetrics)
        .framework("torch")
        .multi_agent(
            policies={
                "pred": PolicySpec(),
                "herb": PolicySpec(),
            },
            policy_mapping_fn=lambda agent_id, *a, **k: (
                "pred" if int(agent_id.split("_")[1]) < ENV_CFG["n_predators"] else "herb"
            ),
        )
    )
    config = config.api_stack(
    enable_rl_module_and_learner=False,
    enable_env_runner_and_connector_v2=False
    )
    config = config.resources(num_gpus=0)

    # Configuración de rollout y entrenamiento
    try:
        config = config.rollouts(batch_mode="truncate_episodes")
    except Exception:
        pass

    config = config.env_runners(
        num_env_runners=NUM_RUNNERS,
        rollout_fragment_length=FRAG,
        sample_timeout_s=300
    )

    # Hiperparámetros
    try:
        config = config.training(
            train_batch_size=TOTAL_BATCH,
            minibatch_size=200,
            num_epochs=2,
            lr=3e-4,
            gamma=0.99, lambda_=0.95,
            clip_param=0.2, vf_clip_param=10.0,
            grad_clip=0.5, entropy_coeff=0.01
        )
    except TypeError:
        config = config.training(
            train_batch_size=TOTAL_BATCH,
            sgd_minibatch_size=200,
            num_sgd_iter=2,
            lr=3e-4,
            gamma=0.99, lambda_=0.95,
            clip_param=0.2, vf_clip_param=10.0,
            grad_clip=0.5, entropy_coeff=0.01
        )

    # Inicializar visualizador si está habilitado
    visualizer = None
    if enable_visualization and HAS_PYGAME:
        visualizer = EcosystemVisualizer(
            map_width=ENV_CFG["map_width"],
            map_height=ENV_CFG["map_height"]
        )
        print("Visualización en tiempo real activada")

    # Importar torch aquí para evitar problemas de importación circular
    try:
        import torch
        HAS_TORCH = True
    except ImportError:
        HAS_TORCH = False
        print("PyTorch no disponible")

    # Entrenamiento y monitoreo
    with open("monitor.csv", "w", newline="") as f:
        n_agents = ENV_CFG["n_agents"]
        writer = csv.writer(f)
        header = (
            ["iter", "r_mean", "l_mean",
            "reason_timeout_pct", "reason_starvation_pct", "reason_dehydration_pct",
            "reason_predation_pct", "attacks_attempted", "attacks_hit", "attack_hit_rate"]  # <-- nueva columna
            + [f"r_agent_{i}" for i in range(n_agents)]
            + [f"l_agent_{i}" for i in range(n_agents)]
            #+ ["gpu_mem_used_mb", "gpu_mem_total_mb", "gpu_load_pct"]
        )
        writer.writerow(header)

        trainer = config.build()

        for i in range(10):
            result = trainer.train()

            ev = result.get("env_runners", {}) or {}
            cm = (ev.get("custom_metrics", {}) or
                result.get("custom_metrics", {}) or {})

            #print("env_runners.custom_metrics keys:", cm.keys())
            #print("env_runners.custom_metrics:", cm)

            def m(key, default=0.0):
                """
                key: nombre "lógico" sin _mean, por ejemplo:
                    'reason_timeout_pct', 'agent_0/episode_return', etc.
                """
                # En RLlib las métricas agregadas suelen llamarse key + '_mean'
                return cm.get(f"{key}_mean", default)

            r_mean = ev.get("episode_return_mean")
            l_mean = ev.get("episode_len_mean")

            timeout_pct     = m("reason_timeout_pct")
            starvation_pct  = m("reason_starvation_pct")
            dehydration_pct = m("reason_dehydration_pct")
            predation_pct   = m("reason_predation_pct")
            att_attempt     = m("attacks_attempted")
            att_hit         = m("attacks_hit")
            att_rate        = m("attack_hit_rate")

            row = [i, r_mean, l_mean, timeout_pct, starvation_pct, dehydration_pct, predation_pct]

            for a in range(n_agents):
                row.append(m(f"agent_{a}/episode_return"))

            for a in range(n_agents):
                row.append(m(f"agent_{a}/episode_len"))

            #row += [mem_used, mem_total, load_pct]
            writer.writerow(row)

            print(f"Iter {i}: ep_return_mean={r_mean}, ep_len_mean={l_mean}")

            # VISUALIZACIÓN EN TIEMPO REAL - VERSIÓN SIMPLIFICADA
            if visualizer and i % 5 == 0:
                try:
                    print(f"Mostrando visualización de la iteración {i}")
                    
                    # Crear un entorno separado para visualización
                    viz_env = MultiAgentEcosystem(**ENV_CFG)
                    obs, _ = viz_env.reset()
                    
                    # Ejecutar algunos pasos con comportamiento aleatorio para demo
                    for viz_step in range(ENV_CFG["max_steps"]):#range(50):
                        # Acciones aleatorias (solo para demo visual)
                        actions = {agent: viz_env.action_space(agent).sample() 
                                  for agent in viz_env.agents}
                        
                        # Step en el entorno
                        obs, rewards, terminations, truncations, infos = viz_env.step(actions)
                        
                        # Calcular métricas
                        alive_count = sum(1 for terminated in terminations.values() if not terminated)
                        metrics = {
                            'mean_reward': np.mean(list(rewards.values())) if rewards else 0,
                            'total_reward': sum(rewards.values()) if rewards else 0,
                            'alive_agents': alive_count,
                            'total_agents': n_agents,
                            'iteration': i
                        }
                        
                        # Renderizar
                        if not visualizer.render(viz_env, viz_env.species, viz_step, i, rewards, metrics):
                            print("Visualización cerrada por el usuario")
                            break
                        
                        # Condición de término
                        if all(terminations.values()) or all(truncations.values()):
                            break
                            
                    print("Visualización completada")
                    
                except Exception as e:
                    print(f"Error en visualización: {e}")

        # Cerrar visualizador al finalizar
        if visualizer:
            visualizer.close()

if __name__ == "__main__":
    main()