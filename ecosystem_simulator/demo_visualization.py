"""
Demo de visualización del ecosistema
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ecosystem_simulator.environment.multi_agent_ecosystem import MultiAgentEcosystem
from ecosystem_simulator.utils.pygame_visualizer import EcosystemVisualizer

def demo_visualization():
    """Demo de la visualización sin entrenamiento"""
    # Configuración
    env_config = {
        "n_agents": 8,
        "veg_density": 25,
        "water_density": 20,
        "map_width": 1200,
        "map_height": 800,
        "max_steps": 1000,
    }
    
    # Crear entorno
    env = MultiAgentEcosystem(**env_config)
    visualizer = EcosystemVisualizer(
        map_width=env_config["map_width"],
        map_height=env_config["map_height"]
    )
    
    # Resetear entorno
    obs, _ = env.reset()
    
    print("Demo de visualización iniciada")
    print("Presiona ESC para salir")
    
    # Loop de demo
    running = True
    step = 0
    
    while running:
        # Acciones aleatorias para demo
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        
        # Step en el entorno
        obs, rewards, terminations, truncations, infos = env.step(actions)
        
        # Métricas para visualización
        metrics = {
            'mean_reward': np.mean(list(rewards.values())),
            'total_reward': sum(rewards.values()),
            'alive_agents': len(env.agents),
            'total_agents': env_config["n_agents"]
        }
        
        # Renderizar
        running = visualizer.render(env, env.species, step, 1, rewards, metrics)
        
        step += 1
        
        # Condición de salida
        if all(terminations.values()) or all(truncations.values()) or step >= env_config["max_steps"]:
            print("Demo completada")
            break
    
    visualizer.close()

if __name__ == "__main__":
    demo_visualization()