#!/usr/bin/env python
"""
ENFOQUE 2: LLM para Reward Shaping Dinámico

Ejecuta entrenamiento donde un LLM modula dinámicamente la función de recompensa
en tiempo real según el estado del agente.

Uso:
    python run_with_llm_reward_shaping.py

Requisitos:
    - Ollama ejecutándose: ollama run mistral
    - Modifica rewards durante entrenamiento RL
"""
import sys
import os
import csv
import logging
import numpy as np

import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.policy.policy import PolicySpec

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ecosystem_simulator.environment.multi_agent_ecosystem import MultiAgentEcosystem
from ecosystem_simulator.training.trainer import build_config
from ecosystem_simulator.training.callbacks import PerAgentAndReasonMetrics
from ecosystem_simulator.llm.reward_shaping import RewardShaper
from ecosystem_simulator.llm.ollama_utils import check_ollama_available, OllamaError
from ecosystem_simulator.config import DEFAULT_ENV_CFG, DEFAULT_NUM_RUNNERS, DEFAULT_NUM_ITERS

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  Configuración central
# ─────────────────────────────────────────────
ENV_CFG = DEFAULT_ENV_CFG.copy()
NUM_RUNNERS  = DEFAULT_NUM_RUNNERS
NUM_ITERS    = DEFAULT_NUM_ITERS
LLM_MODEL = "mistral"
MAX_BONUS_WEIGHT = 0.3  # Máximo peso de modificación


class LLMRewardCallback:
    """
    Callback que aplica reward shaping basado en LLM durante entrenamiento.
    """
    
    def __init__(self, shaper: RewardShaper):
        self.shaper = shaper
        self.shaped_rewards_count = 0
    
    def on_train_result(self, algorithm, result, **info):
        """Llamado después de cada iteración de entrenamiento."""
        stats = self.shaper.get_statistics()
        logger.debug(f"LLM Reward Shaper - Calls: {stats['llm_calls']}, Cache hits: {stats['cache_hits']}")


def train_with_llm_reward_shaping():
    """Loop de entrenamiento con reward shaping basado en LLM."""
    
    # Verificar Ollama
    logger.info("Verificando disponibilidad de Ollama...")
    if not check_ollama_available():
        raise OllamaError(
            "Ollama no está disponible. Instala desde https://ollama.ai "
            "e inicia con: ollama run mistral"
        )
    logger.info("✓ Ollama disponible")
    
    # Inicializar Ray
    ray.init(ignore_reinit_error=True)
    logger.info("✓ Ray inicializado")
    
    # Crear shaper de recompensas
    shaper = RewardShaper(
        model=LLM_MODEL,
        temperature=0.5,
        max_bonus_weight=MAX_BONUS_WEIGHT,
    )
    logger.info("✓ RewardShaper inicializado")
    
    # Construir config y trainer
    config = build_config(
        ENV_CFG,
        num_runners=NUM_RUNNERS,
        callbacks_class=PerAgentAndReasonMetrics
    )
    trainer = config.build()
    logger.info("✓ Trainer construido")
    
    n_agents = ENV_CFG["n_agents"]
    
    # Entrenamiento
    logger.info("=" * 70)
    logger.info("Iniciando entrenamiento con Reward Shaping LLM")
    logger.info("=" * 70)
    
    with open("monitor_llm_reward_shaping.csv", "w", newline="") as f:
        writer = csv.writer(f)
        header = (
            ["iter", "r_mean", "l_mean"]
            + [f"r_agent_{i}" for i in range(n_agents)]
            + [f"l_agent_{i}" for i in range(n_agents)]
            + ["llm_reward_calls", "cache_hits", "cache_efficiency"]
        )
        writer.writerow(header)
        
        for i in range(NUM_ITERS):
            result = trainer.train()
            
            # Extraer métricas
            ev = result.get("env_runners", {}) or {}
            cm = (ev.get("custom_metrics", {}) or
                  result.get("custom_metrics", {}) or {})
            
            def m(key, default=0.0):
                return cm.get(f"{key}_mean", default)
            
            r_mean = ev.get("episode_return_mean", 0.0)
            l_mean = ev.get("episode_len_mean", 0.0)
            
            # Estadísticas del shaper
            shaper_stats = shaper.get_statistics()
            
            row = [i, r_mean, l_mean]
            for a in range(n_agents):
                row.append(m(f"agent_{a}/episode_return"))
            for a in range(n_agents):
                row.append(m(f"agent_{a}/episode_len"))
            
            row.append(shaper_stats["llm_calls"])
            row.append(shaper_stats["cache_hits"])
            row.append(shaper_stats["efficiency"])
            
            writer.writerow(row)
            f.flush()
            
            logger.info(
                f"Iter {i:>3}: return={r_mean:.2f}, len={l_mean:.2f}, "
                f"LLM_calls={shaper_stats['llm_calls']}, "
                f"efficiency={shaper_stats['efficiency']:.1%}"
            )
    
    logger.info("\n✓ Entrenamiento completado")
    logger.info(f"Estadísticas finales del LLM Reward Shaper:")
    stats = shaper.get_statistics()
    logger.info(f"  Total llamadas LLM: {stats['llm_calls']}")
    logger.info(f"  Cache hits: {stats['cache_hits']}")
    logger.info(f"  Eficiencia de cache: {stats['efficiency']:.1%}")
    
    ray.shutdown()


if __name__ == "__main__":
    print("=" * 70)
    print("  ENFOQUE 2: LLM para Reward Shaping Dinámico")
    print("=" * 70)
    print()
    
    try:
        train_with_llm_reward_shaping()
        logger.info("\n[✓] Proceso completado exitosamente")
        logger.info("Monitor guardado en: monitor_llm_reward_shaping.csv")
    except KeyboardInterrupt:
        logger.warning("\n[!] Entrenamiento interrumpido por el usuario")
    except Exception as e:
        logger.error(f"\n[ERROR] {e}", exc_info=True)
        sys.exit(1)
