"""
Demo de visualización del ecosistema sin entrenamiento.

Este script ejecuta una demostración del ecosistema con agentes que se mueven
aleatoriamente (sin RL). Es útil para:
- Verificar que la visualización funciona correctamente
- Entender cómo se ve el ecosistema
- Probar el rendimiento de la visualización
- Depurar problemas sin la complejidad del entrenamiento

Uso:
    python demo_visualization.py
    
Controles:
    ESC - Salir de la demostración
"""
import sys
import os
import numpy as np

# Agregar el directorio actual al path de Python
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ecosystem_simulator.environment.multi_agent_ecosystem import MultiAgentEcosystem
from ecosystem_simulator.utils.pygame_visualizer import EcosystemVisualizer

def demo_visualization():
    """
    Ejecuta una demostración visual del ecosistema con agentes aleatorios.
    
    Los agentes se mueven aleatoriamente sin aprendizaje, permitiendo
    observar la dinámica básica del ecosistema: búsqueda de recursos,
    consumo, regeneración, y eventual muerte por inanición/deshidratación.
    """
    # === CONFIGURACIÓN DEL ENTORNO ===
    env_config = {
        "n_agents": 8,           # 8 agentes en el ecosistema
        "veg_density": 25,       # 25 parches de vegetación
        "water_density": 20,     # 20 fuentes de agua
        "map_width": 1200,       # Mapa de 1200x800 píxeles
        "map_height": 800,
        "max_steps": 350,        # Máximo 350 pasos por episodio
    }
    
    # === CREAR ENTORNO Y VISUALIZADOR ===
    env = MultiAgentEcosystem(**env_config)
    visualizer = EcosystemVisualizer(
        map_width=env_config["map_width"],
        map_height=env_config["map_height"]
    )
    
    # === INICIAR EPISODIO ===
    obs, _ = env.reset()
    
    print("Demo de visualización iniciada")
    print("Presiona ESC para salir")
    
    # === LOOP DE SIMULACIÓN ===
    running = True  # Flag de control del loop
    step = 0        # Contador de pasos
    
    while running:
        # Generar acciones aleatorias para todos los agentes activos
        # (sin aprendizaje, solo para demostración)
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        
        # Ejecutar un paso en el entorno
        obs, rewards, terminations, truncations, infos = env.step(actions)
        
        # === CALCULAR MÉTRICAS PARA VISUALIZACIÓN ===
        metrics = {
            'mean_reward': np.mean(list(rewards.values())),  # Recompensa promedio
            'total_reward': sum(rewards.values()),           # Recompensa total
            'alive_agents': len(env.agents),                 # Agentes vivos
            'total_agents': env_config["n_agents"]           # Total de agentes
        }
        
        # === RENDERIZAR FRAME ===
        # Retorna False si el usuario cierra la ventana
        running = visualizer.render(env, env.species, step, 1, rewards, metrics)
        
        step += 1
        
        # === CONDICIONES DE SALIDA ===
        # Terminar si todos los agentes murieron o se alcanzó el límite
        if all(terminations.values()) or all(truncations.values()) or step >= env_config["max_steps"]:
            print("Demo completada")
            break
    
    # === LIMPIAR RECURSOS ===
    visualizer.close()

if __name__ == "__main__":
    demo_visualization()