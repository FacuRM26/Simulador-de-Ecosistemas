"""
Módulo principal para el entrenamiento del modelo RL.
"""
import os
os.environ.pop("AIR_VERBOSITY", None)
import logging
import re
import random
import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
import GPUtil

import ray
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.env import PettingZooEnv
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.policy.policy import PolicySpec

from ..environment.multi_agent_ecosystem import MultiAgentEcosystem
from .callbacks import PerAgentAndReasonMetrics

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def main():
    """Función principal de entrenamiento."""
    ray.init(ignore_reinit_error=True)

    # --- Configuración del entorno ---
    ENV_CFG = {
        "n_agents": 8,
        "veg_density": 25,
        "water_density": 20,
        "map_width": 1200,
        "map_height": 800,
        "max_steps": 350,
    }

    # Parámetros de entrenamiento
    NUM_RUNNERS = 4
    FRAG = ENV_CFG["max_steps"]               # 350
    TOTAL_BATCH = NUM_RUNNERS * FRAG          # 1400

    register_env(
        "multi_eco",
        lambda cfg: ParallelPettingZooEnv(MultiAgentEcosystem(**cfg))
    )

    config = (
        PPOConfig()
        .environment(env="multi_eco", env_config=ENV_CFG)
        .callbacks(callbacks_class=PerAgentAndReasonMetrics)
        .framework("torch")
        .multi_agent(
            policies={"shared_policy": PolicySpec()},
            policy_mapping_fn=lambda *a, **k: "shared_policy"
        )
        .env_runners(num_env_runners=NUM_RUNNERS, rollout_fragment_length=50, sample_timeout_s=300)
    )

    # Configuración de rollout y entrenamiento
    try:
        config = config.rollouts(batch_mode="complete_episodes")
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
        # Fallback para APIs antiguas
        config = config.training(
            train_batch_size=TOTAL_BATCH,
            sgd_minibatch_size=200,
            num_sgd_iter=2,
            lr=3e-4,
            gamma=0.99, lambda_=0.95,
            clip_param=0.2, vf_clip_param=10.0,
            grad_clip=0.5, entropy_coeff=0.01
        )

    # Entrenamiento y monitoreo
    with open("monitor.csv", "w", newline="") as f:
        n_agents = 8
        writer = csv.writer(f)
        header = (
            ["iter", "r_mean", "l_mean",
            "reason_timeout_pct", "reason_starvation_pct", "reason_dehydration_pct"] +
            [f"r_agent_{i}" for i in range(n_agents)] +
            [f"l_agent_{i}" for i in range(n_agents)] +
            ["gpu_mem_used_mb", "gpu_mem_total_mb", "gpu_load_pct"]
        )
        writer.writerow(header)

        trainer = config.build()

        for i in range(50):
            result = trainer.train()

            # Log de tiempos de ejecución
            t = result.get("timers", {}) or {}
            print("sampling_s=", t.get("env_runner_sampling_timer"),
                "learner_s=", t.get("learner_update_timer"))

            # Extracción de métricas
            ev = result.get("env_runners", {}) or {}

            def m(key):
                return ev.get(key) or (ev.get("custom_metrics", {}) or {}).get(key)

            r_mean = ev.get("episode_return_mean")
            l_mean = ev.get("episode_len_mean")

            timeout_pct     = m("reason_timeout_pct")
            starvation_pct  = m("reason_starvation_pct")
            dehydration_pct = m("reason_dehydration_pct")

            row = [i, r_mean, l_mean, timeout_pct, starvation_pct, dehydration_pct]

            # Métricas por agente
            for a in range(n_agents):
                row.append(m(f"agent_{a}/episode_return"))
            for a in range(n_agents):
                row.append(m(f"agent_{a}/episode_len"))

            # Métricas GPU
            mem_used = mem_total = load_pct = float("nan")
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    g = gpus[0]
                    mem_used  = getattr(g, "memoryUsed", float("nan"))
                    mem_total = getattr(g, "memoryTotal", float("nan"))
                    load_pct  = getattr(g, "load", 0.0) * 100.0
            except Exception:
                pass  # si no hay GPU o no está GPUtil, deja NaN

            row += [mem_used, mem_total, load_pct]
            writer.writerow(row)

            print(f"Iter {i}: ep_return_mean={r_mean}, ep_len_mean={l_mean}")

if __name__ == "__main__":
    main()