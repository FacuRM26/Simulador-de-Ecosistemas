"""
Script para entrenar con visualización en tiempo real en Godot.
Actualiza el estado en CADA PASO del ecosistema.
"""
import sys
import os
import threading
import time
import csv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ecosystem_simulator.server.api_server import run_server, update_ecosystem_state, serialize_ecosystem_state
from ecosystem_simulator.utils.visualization import analyze_training_results

import numpy as np
import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import RolloutWorker
from ray.rllib.policy import Policy
from typing import Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ray.rllib.evaluation.episode_v2 import EpisodeV2

from ecosystem_simulator.environment.multi_agent_ecosystem import MultiAgentEcosystem
from ecosystem_simulator.training.callbacks import PerAgentAndReasonMetrics


class GodotVisualizationCallback(DefaultCallbacks):
    """
    Callback que actualiza Godot en CADA PASO del entrenamiento.
    """
    
    def __init__(self):
        super().__init__()
        self.step_count = 0
        self.episode_count = 0
        self.update_every = 5  # Actualizar cada N pasos (1 = cada paso)
    
    def on_episode_step(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Optional[Dict[str, Policy]] = None,
        episode = None,  # Sin tipo específico para compatibilidad
        **kwargs
    ) -> None:
        """Se llama en CADA PASO del episodio."""
        self.step_count += 1
        
        # Actualizar solo cada N pasos para reducir carga
        if self.step_count % self.update_every != 0:
            return
        
        try:
            # Intentar obtener el entorno de varias formas
            env = None
            
            # Método 1: Desde base_env
            if base_env is not None:
                try:
                    env = base_env.get_sub_environments()[0]
                except:
                    pass
            
            # Método 2: Desde worker
            if env is None and worker is not None:
                try:
                    env = worker.env
                except:
                    pass
            
            # Método 3: Desde episode
            if env is None and episode is not None:
                try:
                    env = episode.env
                except:
                    pass
            
            if env is None:
                # No hay entorno disponible
                return
            
            # Si está envuelto en ParallelPettingZooEnv, extraer el env real
            if hasattr(env, 'env'):
                env = env.env
            
            # Verificar que tiene los atributos necesarios
            if not hasattr(env, 'species') or not hasattr(env, 'map_width'):
                return
            
            # Serializar estado
            state = serialize_ecosystem_state(
                env,
                env.species,
                episode=self.episode_count,
                step=self.step_count
            )
            
            # Actualizar servidor para Godot
            update_ecosystem_state(state)
            
        except Exception as e:
            # No detener el entrenamiento por errores de visualización
            if self.step_count % 100 == 0:  # Solo mostrar error cada 100 pasos
                print(f"[Callback] Error actualizando Godot: {e}")
    
    def on_episode_start(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode = None,
        **kwargs
    ) -> None:
        """Al inicio de cada episodio."""
        self.episode_count += 1
        self.step_count = 0
        print(f"[Callback] Episodio {self.episode_count} iniciado")
    
    def on_episode_end(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode = None,
        **kwargs
    ) -> None:
        """Al final de cada episodio."""
        print(f"[Callback] Episodio {self.episode_count} terminado (pasos: {self.step_count})")


class CombinedCallbacks(PerAgentAndReasonMetrics):
    """Combina métricas + visualización de Godot."""
    
    def __init__(self):
        super().__init__()
        self.godot_callback = GodotVisualizationCallback()
    
    def on_episode_step(self, *, worker=None, base_env=None, policies=None, episode=None, **kwargs):
        # Llamar ambos callbacks con argumentos explícitos
        super().on_episode_step(worker=worker, base_env=base_env, policies=policies, episode=episode, **kwargs)
        self.godot_callback.on_episode_step(worker=worker, base_env=base_env, policies=policies, episode=episode, **kwargs)
    
    def on_episode_start(self, *, worker=None, base_env=None, policies=None, episode=None, **kwargs):
        super().on_episode_start(worker=worker, base_env=base_env, policies=policies, episode=episode, **kwargs)
        self.godot_callback.on_episode_start(worker=worker, base_env=base_env, policies=policies, episode=episode, **kwargs)
    
    def on_episode_end(self, *, worker=None, base_env=None, policies=None, episode=None, **kwargs):
        super().on_episode_end(worker=worker, base_env=base_env, policies=policies, episode=episode, **kwargs)
        self.godot_callback.on_episode_end(worker=worker, base_env=base_env, policies=policies, episode=episode, **kwargs)


def train_with_realtime_godot():
    """Entrenamiento con actualización en tiempo real para Godot."""
    
    ray.init(ignore_reinit_error=True)

    # Configuración del entorno
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
    FRAG = ENV_CFG["max_steps"]
    TOTAL_BATCH = NUM_RUNNERS * FRAG

    register_env(
        "multi_eco",
        lambda cfg: ParallelPettingZooEnv(MultiAgentEcosystem(**cfg))
    )

    config = (
        PPOConfig()
        .environment(env="multi_eco", env_config=ENV_CFG)
        .callbacks(callbacks_class=CombinedCallbacks)  # Solo métricas # PerAgentAndReasonMetrics
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
        #.env_runners(num_env_runners=NUM_RUNNERS, rollout_fragment_length=50, sample_timeout_s=300)
    )

    config = config.api_stack(
        enable_rl_module_and_learner=False,
        enable_env_runner_and_connector_v2=False
    )
    config = config.resources(num_gpus=0)

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
    
    # Entrenamiento y monitoreo
    with open("monitor.csv", "w", newline="") as f:
        n_agents = ENV_CFG["n_agents"]
        writer = csv.writer(f)
        header = (
            ["iter", "r_mean", "l_mean",
             "reason_timeout_pct", "reason_starvation_pct",
             "reason_dehydration_pct", "reason_predation_pct",
             "attacks_attempted", "attacks_hit", "attack_hit_rate"]
            + [f"r_agent_{i}" for i in range(n_agents)]
            + [f"l_agent_{i}" for i in range(n_agents)]
        )
        writer.writerow(header)

        trainer = config.build()

        for i in range(100): # Número de iteraciones de entrenamiento
            result = trainer.train()

            # Extracción de métricas
            ev = result.get("env_runners", {}) or {}
            cm = (ev.get("custom_metrics", {}) or
                  result.get("custom_metrics", {}) or {})
            
            #def m(key):
            #    return ev.get(key) or (ev.get("custom_metrics", {}) or {}).get(key)
            def m(key, default=0.0):
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

            row = [i, r_mean, l_mean, 
                   timeout_pct, starvation_pct, 
                   dehydration_pct, predation_pct, 
                   att_attempt, att_hit, att_rate]

            for a in range(n_agents):
                row.append(m(f"agent_{a}/episode_return"))
            for a in range(n_agents):
                row.append(m(f"agent_{a}/episode_len"))

            writer.writerow(row)

            print(f"Iter {i}: ep_return_mean={r_mean:.2f}, ep_len_mean={l_mean:.2f}")

            # ACTUALIZAR VISUALIZACIÓN para Godot
            try:
                if i < 80: 
                    continue # Esperar algunas iteraciones antes de visualizar
                # Crear un entorno separado para visualización
                viz_env = MultiAgentEcosystem(**ENV_CFG)
                obs, _ = viz_env.reset()
                
                # Ejecutar algunos pasos con el modelo entrenado
                for viz_step in range(ENV_CFG["max_steps"]):
                    if not obs: # Por si ya no hay agentes vivos
                        break

                    # Obtener acciones usando el modelo entrenado
                    actions = {}
                    for agent in viz_env.agents:
                        if agent in obs:
                            try:
                                idx = int(agent.split("_")[1])
                                pol_id = (
                                    "pred"
                                    if idx < ENV_CFG["n_predators"]
                                    else "herb"
                                )
                                out = trainer.get_policy(pol_id).compute_single_action(
                                    obs[agent],
                                    explore=False,
                                )
                                action = out[0] if isinstance(out, tuple) else out
                                actions[agent] = action
                            except:
                                # Si falla, usar acción aleatoria como fallback
                                actions[agent] = viz_env.action_space(agent).sample()
                    
                    # Ejecutar paso
                    obs, rewards, terminations, truncations, infos = viz_env.step(actions)
                    viz_step += 1
                    
                    # Calcular métricas
                    alive_count = sum(1 for terminated in terminations.values() if not terminated)
                    metrics = {
                        'mean_reward': np.mean(list(rewards.values())) if rewards else 0,
                        'total_reward': sum(rewards.values()) if rewards else 0,
                        'alive_agents': alive_count,
                        'total_agents': n_agents,
                        'iteration': i
                    }

                    # Actualizar estado para Godot
                    state = serialize_ecosystem_state(
                        viz_env,
                        viz_env.species,
                        episode=i,
                        step=viz_step
                    )
                    update_ecosystem_state(state)

                    time.sleep(0.07) # Pequeña pausa para Godot y que no se ejecute todo de golpe
                    
                    # Si el episodio terminó, resetear
                    if all(terminations.values()) or all(truncations.values()):
                        break
                
            except Exception as e:
                print(f"[Godot] Error actualizando visualización: {e}")

        print("\n[+] Entrenamiento completado")


if __name__ == "__main__":
    print("=" * 70)
    print(" ENTRENAMIENTO CON VISUALIZACIÓN TIEMPO REAL EN GODOT ")
    print("=" * 70)
    print()
    print("Servidor: http://localhost:5000/state")
    print()
    print("=" * 70)
    print()
    
    # Iniciar servidor HTTP en thread separado
    print("\n[+] Iniciando servidor HTTP...")
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    
    #time.sleep(2)
    print("[+] Servidor HTTP activo en http://localhost:5000")
    print("[+] Iniciando entrenamiento...\n")
    
    try:
        train_with_realtime_godot()
        
        print("\n[+] Analizando resultados...")
        analyze_training_results("monitor.csv")
        
    except KeyboardInterrupt:
        print("\n\n[!] Entrenamiento interrumpido")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()