#!/usr/bin/env python
"""
ENFOQUE 1: LLM como Selector de Comportamientos Pre-entrenados

Ejecuta entrenamiento donde un LLM selecciona entre múltiples políticas RL
pre-entrenadas según el estado actual.

Uso:
    python run_with_llm_behavior_selector.py

Requisitos:
    - Ollama ejecutándose: ollama run mistral
    - Se crean 2 políticas por rol: "explore" y "survival"
"""
import sys
import os
import csv
import logging
import numpy as np
import threading
import time

import ray
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.policy.policy import PolicySpec

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ecosystem_simulator.environment.multi_agent_ecosystem import MultiAgentEcosystem
from ecosystem_simulator.training.trainer import build_config
from ecosystem_simulator.training.callbacks import PerAgentAndReasonMetrics
from ecosystem_simulator.llm.behavior_selector import BehaviorSelector
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
BEHAVIORS = ["explore", "survival"]  # Políticas disponibles
LLM_MODEL = "mistral"  # o "neural-chat" para más velocidad


def create_dummy_policies(action_space_size: int = 7):
    """
    Crea funciones de política dummy para demostración.
    En producción, estas serían políticas RL pre-entrenadas.
    
    Args:
        action_space_size: Número de acciones posibles.
    
    Returns:
        Dict con políticas para cada comportamiento.
    """
    def explore_policy(obs):
        """Política que explora: movimientos principalmente."""
        # 60% movimientos (0-3), 20% comida, 20% agua
        return np.random.choice([0, 1, 2, 3, 4, 5], p=[0.2, 0.2, 0.2, 0.2, 0.1, 0.1])
    
    def survival_policy(obs):
        """Política de supervivencia: busca recursos."""
        # 40% movimientos, 30% comida, 30% agua
        return np.random.choice([0, 1, 2, 3, 4, 5], p=[0.15, 0.15, 0.15, 0.15, 0.2, 0.2])
    
    return {
        "explore": explore_policy,
        "survival": survival_policy,
    }


def train_with_llm_behavior_selector():
    """Loop de entrenamiento con selector LLM de comportamientos."""
    
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
    
    # Construir config y trainer
    config = build_config(
        ENV_CFG,
        num_runners=NUM_RUNNERS,
        callbacks_class=PerAgentAndReasonMetrics
    )
    trainer = config.build()
    logger.info("✓ Trainer construido")
    
    # Crear selector de comportamientos
    selector = BehaviorSelector(
        available_behaviors=BEHAVIORS,
        model=LLM_MODEL,
        temperature=0.3,
    )
    
    # Registrar políticas dummy
    policies = create_dummy_policies()
    for behavior, policy_fn in policies.items():
        selector.register_policy(behavior, policy_fn)
    logger.info(f"✓ Comportamientos registrados: {BEHAVIORS}")
    
    n_agents = ENV_CFG["n_agents"]
    
    # Entrenamiento
    logger.info("=" * 70)
    logger.info("Iniciando entrenamiento con Selector LLM")
    logger.info("=" * 70)
    
    with open("monitor_llm_behavior.csv", "w", newline="") as f:
        writer = csv.writer(f)
        header = (
            ["iter", "r_mean", "l_mean"]
            + [f"r_agent_{i}" for i in range(n_agents)]
            + [f"l_agent_{i}" for i in range(n_agents)]
            + ["llm_selections_explore", "llm_selections_survival"]
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
            
            # Estadísticas del selector LLM
            llm_stats = selector.get_statistics()
            
            row = [i, r_mean, l_mean]
            for a in range(n_agents):
                row.append(m(f"agent_{a}/episode_return"))
            for a in range(n_agents):
                row.append(m(f"agent_{a}/episode_len"))
            
            row.append(llm_stats["per_behavior"].get("explore", 0))
            row.append(llm_stats["per_behavior"].get("survival", 0))
            
            writer.writerow(row)
            f.flush()
            
            logger.info(
                f"Iter {i:>3}: return={r_mean:.2f}, len={l_mean:.2f}, "
                f"LLM_selections={llm_stats['total_selections']}"
            )
            
            # Log detallado cada 10 iteraciones
            if (i + 1) % 10 == 0:
                logger.info(f"  LLM Distribution: {llm_stats['distribution']}")
    
    logger.info("\n✓ Entrenamiento completado")
    logger.info(f"Estadísticas finales del LLM:")
    logger.info(f"  Total selecciones: {selector.get_statistics()['total_selections']}")
    logger.info(f"  Distribución: {selector.get_statistics()['distribution']}")
    
    ray.shutdown()


if __name__ == "__main__":
    print("=" * 70)
    print("  ENFOQUE 1: LLM como Selector de Comportamientos")
    print("=" * 70)
    print()
    
    try:
        train_with_llm_behavior_selector()
        logger.info("\n[✓] Proceso completado exitosamente")
        logger.info("Monitor guardado en: monitor_llm_behavior.csv")
    except KeyboardInterrupt:
        logger.warning("\n[!] Entrenamiento interrumpido por el usuario")
    except Exception as e:
        logger.error(f"\n[ERROR] {e}", exc_info=True)
        sys.exit(1)
