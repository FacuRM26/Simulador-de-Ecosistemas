#!/usr/bin/env python
"""
ENFOQUE 3: LLM con Reflexion (Aprendizaje entre Episodios)

Ejecuta entrenamiento donde un LLM analiza episodios completos al finalizar
y genera lecciones aprendidas que se inyectan en el contexto de futuros episodios.

Inspirado en: "Reflexion: Language Agents with Verbal Reinforcement Learning"

Uso:
    python run_with_llm_reflexion.py

Requisitos:
    - Ollama ejecutándose: ollama run mistral
    - El LLM genera reflexiones post-episodio
    - Acumula aprendizaje a nivel episódico
"""
import sys
import os
import csv
import logging
import numpy as np
import json

import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.policy.policy import PolicySpec

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ecosystem_simulator.environment.multi_agent_ecosystem import MultiAgentEcosystem
from ecosystem_simulator.training.trainer import build_config
from ecosystem_simulator.training.callbacks import PerAgentAndReasonMetrics
from ecosystem_simulator.llm.reflexion import ReflectionAgent
from ecosystem_simulator.llm.ollama_utils import check_ollama_available, OllamaError

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Configuración central
ENV_CFG = {
    "n_agents": 8,
    "veg_density": 15,
    "water_density": 10,
    "map_width": 800,
    "map_height": 600,
    "max_steps": 350,
    "n_predators": 2,
}

NUM_RUNNERS = 4
NUM_ITERS = 100  # Reducido para pruebas
LLM_MODEL = "mistral"
REFLECTION_INTERVAL = 5  # Generar reflexión cada N episodios


class SimpleEpisodeRecorder:
    """
    Registra episodios simples para reflexión.
    En producción, esto vendría de los runners de Ray.
    """
    
    def __init__(self):
        self.episodes = []
        self.episode_count = 0
    
    def record_episode(self, trajectory: list, success: bool, reward: float):
        """Registra un episodio."""
        self.episodes.append({
            "trajectory": trajectory,
            "success": success,
            "reward": reward,
        })
        self.episode_count += 1
    
    def get_recent_episodes(self, n: int = 1):
        """Retorna los últimos N episodios."""
        return self.episodes[-n:] if self.episodes else []


def train_with_llm_reflexion():
    """Loop de entrenamiento con Reflexion."""
    
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
    
    # Crear agente reflexivo
    reflection_agent = ReflectionAgent(
        model=LLM_MODEL,
        temperature=0.6,
    )
    logger.info("✓ ReflectionAgent inicializado")
    
    # Recorder para episodios
    episode_recorder = SimpleEpisodeRecorder()
    
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
    logger.info("Iniciando entrenamiento con Reflexion LLM")
    logger.info("=" * 70)
    
    with open("monitor_llm_reflexion.csv", "w", newline="") as f:
        writer = csv.writer(f)
        header = (
            ["iter", "r_mean", "l_mean"]
            + [f"r_agent_{i}" for i in range(n_agents)]
            + [f"l_agent_{i}" for i in range(n_agents)]
            + ["total_episodes", "reflections_generated", "success_rate"]
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
            
            # Simular episodio para reflexión
            if (i + 1) % REFLECTION_INTERVAL == 0:
                # En producción, esto vendría de Ray runners
                trajectory = [
                    {"action": np.random.randint(0, 7), "reward": np.random.randn()}
                    for _ in range(10)
                ]
                success = r_mean > 0
                
                # Procesar episodio (genera reflexión LLM)
                reflection_agent.process_episode(
                    trajectory=trajectory,
                    success=success,
                    total_reward=r_mean
                )
                
                logger.info(f"  Reflexión generada para episodio {reflection_agent.episode_count}")
            
            # Estadísticas de reflexión
            reflexion_stats = reflection_agent.get_statistics()
            
            row = [i, r_mean, l_mean]
            for a in range(n_agents):
                row.append(m(f"agent_{a}/episode_return"))
            for a in range(n_agents):
                row.append(m(f"agent_{a}/episode_len"))
            
            row.append(reflexion_stats.get("total_episodes", 0))
            row.append(reflexion_stats.get("reflections_generated", 0))
            row.append(reflexion_stats.get("success_rate", 0))
            
            writer.writerow(row)
            f.flush()
            
            logger.info(
                f"Iter {i:>3}: return={r_mean:.2f}, len={l_mean:.2f}, "
                f"reflexiones={reflexion_stats.get('reflections_generated', 0)}"
            )
    
    logger.info("\n✓ Entrenamiento completado")
    logger.info(f"Estadísticas finales de Reflexion:")
    stats = reflection_agent.get_statistics()
    logger.info(f"  Total episodios procesados: {stats.get('total_episodes', 0)}")
    logger.info(f"  Reflexiones generadas: {stats.get('reflections_generated', 0)}")
    logger.info(f"  Tasa de éxito: {stats.get('success_rate', 0):.1%}")
    
    # Mostrar últimas lecciones aprendidas
    logger.info("\nÚltimas lecciones aprendidas:")
    context = reflection_agent.get_context_prompt()
    for line in context.split("\n"):
        if line.strip():
            logger.info(f"  {line}")
    
    ray.shutdown()


if __name__ == "__main__":
    print("=" * 70)
    print("  ENFOQUE 3: LLM con Reflexion (Aprendizaje entre Episodios)")
    print("=" * 70)
    print()
    
    try:
        train_with_llm_reflexion()
        logger.info("\n[✓] Proceso completado exitosamente")
        logger.info("Monitor guardado en: monitor_llm_reflexion.csv")
    except KeyboardInterrupt:
        logger.warning("\n[!] Entrenamiento interrumpido por el usuario")
    except Exception as e:
        logger.error(f"\n[ERROR] {e}", exc_info=True)
        sys.exit(1)
