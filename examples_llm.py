#!/usr/bin/env python
"""
Ejemplos simples para probar los módulos LLM sin Ray.

Estos ejemplos muestran cómo usar cada módulo de forma independiente,
útil para debugging y entendimiento.

Requisitos:
    - Ollama ejecutándose: ollama run mistral
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ecosystem_simulator.llm.behavior_selector import BehaviorSelector
from ecosystem_simulator.llm.reward_shaping import RewardShaper
from ecosystem_simulator.llm.reflexion import ReflectionAgent
from ecosystem_simulator.llm.ollama_utils import check_ollama_available


def example_1_behavior_selector():
    """
    Ejemplo 1: Selector de Comportamientos
    
    El LLM decide qué comportamiento activar según el estado.
    """
    print("\n" + "="*70)
    print("EJEMPLO 1: LLM Behavior Selector")
    print("="*70)
    
    # Crear selector
    selector = BehaviorSelector(
        available_behaviors=["hunt", "explore", "survive"],
        temperature=0.3
    )
    
    # Registrar políticas dummy
    selector.register_policy("hunt", lambda obs: 6)      # action: atacar
    selector.register_policy("explore", lambda obs: np.random.randint(0, 4))  # movimiento
    selector.register_policy("survive", lambda obs: 4)   # action: comer
    
    # Simular algunos estados
    test_cases = [
        (np.array([100, 200, 0.9, 0, 500]), "agent_0", "Mucha energía, solo"),
        (np.array([100, 200, 0.1, 0, 50]),  "agent_1", "Sin energía, presa cerca"),
        (np.array([100, 200, 0.5, 3, 200]), "agent_2", "Energía media, con grupo"),
    ]
    
    print("\nProbando selector LLM con observaciones:")
    for obs, agent_id, description in test_cases:
        print(f"\n  Estado: {description}")
        print(f"  Obs: {obs}")
        
        behavior = selector.select_behavior(obs, agent_id)
        print(f"  ✓ Comportamiento seleccionado: {behavior}")
    
    print("\n\nEstadísticas de selecciones:")
    stats = selector.get_statistics()
    for behavior, count in stats["per_behavior"].items():
        print(f"  {behavior}: {count} veces ({stats['distribution'][behavior]:.1%})")


def example_2_reward_shaping():
    """
    Ejemplo 2: Reward Shaping Dinámico
    
    El LLM modula la recompensa según el contexto.
    """
    print("\n" + "="*70)
    print("EJEMPLO 2: LLM Reward Shaping")
    print("="*70)
    
    shaper = RewardShaper(temperature=0.5)
    
    # Simular algunos pasos
    print("\nAplicando reward shaping a diferentes estados:")
    
    test_steps = [
        (np.array([100, 200, 0.1, 0, 50]),  4, 1.0, "agent_0", "Sin energía, comiendo"),
        (np.array([100, 200, 0.9, 0, 500]), 2, 0.5, "agent_1", "Mucha energía, moviéndose"),
        (np.array([100, 200, 0.3, 5, 100]), 0, 0.2, "agent_2", "Rodeado, intentando escapar"),
    ]
    
    for obs, action, reward, agent_id, description in test_steps:
        print(f"\n  {description}")
        print(f"    Recompensa base: {reward:.4f}")
        
        shaped = shaper.shape_reward(obs, action, reward, agent_id, use_llm=True)
        bonus = shaped - reward
        
        print(f"    Recompensa ajustada: {shaped:.4f}")
        print(f"    Bonus/penalización: {bonus:+.4f}")
    
    print("\n\nEstadísticas de reward shaping:")
    stats = shaper.get_statistics()
    print(f"  Llamadas LLM: {stats['llm_calls']}")
    print(f"  Cache hits: {stats['cache_hits']}")
    print(f"  Eficiencia: {stats['efficiency']:.1%}")


def example_3_reflexion():
    """
    Ejemplo 3: Reflexion
    
    El LLM genera lecciones aprendidas de episodios.
    """
    print("\n" + "="*70)
    print("EJEMPLO 3: LLM Reflexion")
    print("="*70)
    
    agent = ReflectionAgent(temperature=0.6)
    
    # Simular algunos episodios
    print("\nProcesando episodios y generando reflexiones:")
    
    episodes = [
        {
            "trajectory": [
                {"action": 4, "reward": 2.0},   # comer
                {"action": 4, "reward": 2.5},   # comer más
                {"action": 5, "reward": 1.5},   # beber
                {"action": 0, "reward": 0.0},   # movimiento
            ],
            "success": True,
            "total_reward": 6.0,
            "description": "Episodio exitoso: comió bien"
        },
        {
            "trajectory": [
                {"action": 6, "reward": -0.5},  # atacar (falló)
                {"action": 2, "reward": 0.5},   # escapar
                {"action": 3, "reward": 0.3},   # escapar
            ],
            "success": False,
            "total_reward": 0.3,
            "description": "Episodio fallido: atacó sin éxito"
        },
    ]
    
    for ep in episodes:
        print(f"\n  {ep['description']}")
        print(f"    Recompensa total: {ep['total_reward']:.2f}")
        
        agent.process_episode(
            trajectory=ep["trajectory"],
            success=ep["success"],
            total_reward=ep["total_reward"]
        )
    
    print("\n\nLecciones aprendidas:")
    context = agent.get_context_prompt()
    for line in context.split("\n"):
        if line.strip() and not line.startswith("Experiencia"):
            print(f"  {line}")
    
    print("\n\nEstadísticas de reflexion:")
    stats = agent.get_statistics()
    print(f"  Total episodios procesados: {stats.get('total_episodes', 0)}")
    print(f"  Lecciones generadas: {stats.get('reflections_generated', 0)}")
    print(f"  Tasa de éxito: {stats.get('success_rate', 0):.1%}")


def main():
    """Ejecuta todos los ejemplos."""
    print("\n" + "="*70)
    print("EJEMPLOS DE MÓDULOS LLM PARA ENTRENAMIENTO RL")
    print("="*70)
    
    # Verificar Ollama
    print("\nVerificando disponibilidad de Ollama...")
    if not check_ollama_available():
        print("✗ ERROR: Ollama no está disponible")
        print("  Instala desde https://ollama.ai e inicia con: ollama run mistral")
        sys.exit(1)
    print("✓ Ollama disponible en http://localhost:11434")
    
    # Ejecutar ejemplos
    try:
        example_1_behavior_selector()
        example_2_reward_shaping()
        example_3_reflexion()
        
        print("\n" + "="*70)
        print("✓ Todos los ejemplos completados exitosamente")
        print("="*70 + "\n")
        
    except KeyboardInterrupt:
        print("\n\n[!] Ejemplos interrumpidos por el usuario")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
