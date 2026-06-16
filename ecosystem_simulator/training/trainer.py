"""
Módulo principal para el entrenamiento del modelo RL.
"""
import os
os.environ.pop("AIR_VERBOSITY", None)

import csv
from pathlib import Path

import numpy as np
import ray

from ..environment.multi_agent_ecosystem import MultiAgentEcosystem
from .callbacks import PerAgentAndReasonMetrics

from ecosystem_simulator.config import ENV_CFG, NUM_RUNNERS, NUM_ITERS

#NUM_ITERS       = DEFAULT_NUM_ITERS         
VIS_START_FRAC  = 0.7         # empezar al 50% del entrenamiento
VIS_INTERVAL    = 10 

try:
    from ..utils.pygame_visualizer import EcosystemVisualizer
    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False
    print("Pygame no disponible - visualización desactivada")


def build_config(
    env_cfg: dict,
    num_runners: int = 4,
    callbacks_class=None,
    use_lstm: bool = True,
):
    """
    Construye y retorna la configuración PPO lista para usar.

    use_lstm=True activa una política recurrente con LSTM.
    No cambia el entorno ni las observaciones.
    """
    from ray.tune.registry import register_env
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
    from ray.rllib.policy.policy import PolicySpec

    if callbacks_class is None:
        callbacks_class = PerAgentAndReasonMetrics

    frag = env_cfg["max_steps"]

    # Con LSTM conviene usar un batch más grande para que aprenda secuencias.
    # Si lo sientes muy lento, puedes bajar 2048 a 1024.
    if use_lstm:
        # Con tu configuración actual:
        # num_runners=4 y max_steps=350
        # total_batch = 4 * 350 = 1400
        total_batch = num_runners * frag

        # Mejor que el minibatch sea compatible con el tamaño del episodio.
        # 350 funciona bien porque coincide con max_steps.
        minibatch = frag

        num_epochs = 2
    else:
        total_batch = num_runners * frag
        minibatch = 200
        num_epochs = 2

    register_env(
        "multi_eco",
        lambda cfg: ParallelPettingZooEnv(MultiAgentEcosystem(**cfg))
    )

    config = (
        PPOConfig()
        .environment(env="multi_eco", env_config=env_cfg)
        .callbacks(callbacks_class)
        .framework("torch")
        .multi_agent(
            policies={
                "pred": PolicySpec(),
                "herb": PolicySpec(),
            },
            policy_mapping_fn=lambda agent_id, *a, **k: (
                "pred"
                if int(agent_id.split("_")[1]) < env_cfg["n_predators"]
                else "herb"
            ),
        )
    )

    config = config.api_stack(
        enable_rl_module_and_learner=False,
        enable_env_runner_and_connector_v2=False
    )

    config = config.resources(num_gpus=1)

    try:
        config = config.rollouts(batch_mode="truncate_episodes")
    except Exception:
        pass

    config = config.env_runners(
        num_env_runners=num_runners,
        rollout_fragment_length=frag,
        sample_timeout_s=300
    )

    # Configuración del modelo.
    # La LSTM aprende dependencias temporales dentro del episodio.
    model_cfg = {
        "fcnet_hiddens": [64, 64],
        "fcnet_activation": "tanh",

        "use_lstm": use_lstm,
        "lstm_cell_size": 64,
        "max_seq_len": 16,

        "lstm_use_prev_action": False,
        "lstm_use_prev_reward": False,
        "vf_share_layers": False,
    }

    try:
        config = config.training(
            train_batch_size=total_batch,
            minibatch_size=minibatch,
            num_epochs=num_epochs,
            lr=3e-4,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            vf_clip_param=10.0,
            grad_clip=0.5,
            entropy_coeff=0.01,
            model=model_cfg,
        )
    except TypeError:
        config = config.training(
            train_batch_size=total_batch,
            sgd_minibatch_size=minibatch,
            num_sgd_iter=num_epochs,
            lr=3e-4,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            vf_clip_param=10.0,
            grad_clip=0.5,
            entropy_coeff=0.01,
            model=model_cfg,
        )

    return config
def get_policy_id_for_agent(agent_id: str, env_cfg: dict) -> str:
    """
    Devuelve la política correspondiente al agente.
    """
    idx = int(agent_id.split("_")[1])
    return "pred" if idx < env_cfg["n_predators"] else "herb"


def init_agent_lstm_states(trainer, agents, env_cfg: dict) -> dict:
    """
    Crea el estado inicial de LSTM para cada agente.

    Aunque varios agentes compartan la misma política,
    cada agente necesita su propio estado recurrente.
    """
    states = {}

    for agent_id in agents:
        policy_id = get_policy_id_for_agent(agent_id, env_cfg)
        policy = trainer.get_policy(policy_id)
        states[agent_id] = list(policy.get_initial_state())

    return states


def compute_action_with_lstm_state(
    trainer,
    agent_id: str,
    agent_obs,
    env_cfg: dict,
    agent_states: dict,
    explore: bool = False,
):
    """
    Calcula una acción manteniendo el estado recurrente de cada agente.

    Funciona tanto si la política tiene LSTM como si no.
    """
    policy_id = get_policy_id_for_agent(agent_id, env_cfg)
    policy = trainer.get_policy(policy_id)

    state_in = agent_states.get(agent_id)

    if state_in is None:
        state_in = list(policy.get_initial_state())

    out = policy.compute_single_action(
        agent_obs,
        state=state_in,
        explore=explore,
    )

    if isinstance(out, tuple):
        action, state_out, _ = out
        agent_states[agent_id] = state_out
    else:
        action = out
        agent_states[agent_id] = state_in

    return action

def get_monitor_header(n_agents: int):
    return (
        [
            "iter", "r_mean", "l_mean",
            "attacks_attempted", "attacks_hit", "attacks_kill",
            "attack_hit_rate", "attack_kill_rate",

            "herbivore_episode_return", "predator_episode_return",
            "herbivore_episode_len", "predator_episode_len",

            "herbivore_survival_pct", "predator_survival_pct",

            "herbivore_reason_timeout_pct",
            "herbivore_reason_starvation_pct",
            "herbivore_reason_dehydration_pct",
            "herbivore_reason_predation_pct",

            "predator_reason_timeout_pct",
            "predator_reason_starvation_pct",
            "predator_reason_dehydration_pct",
            "predator_reason_predation_pct",

            "herbivore_eat_count", "herbivore_drink_count",
            "predator_drink_count",

            "predator_attack_attempt_count",
            "predator_attack_hit_count",
            "predator_attack_kill_count",

            "herbivore_avg_food", "herbivore_avg_water",
            "predator_avg_food", "predator_avg_water",

            "herbivore_critical_ratio", "predator_critical_ratio",
        ]
        + [f"r_agent_{i}" for i in range(n_agents)]
        + [f"l_agent_{i}" for i in range(n_agents)]
    )


def build_monitor_row(result: dict, iteration: int, n_agents: int):
    ev = result.get("env_runners", {}) or {}
    cm = (ev.get("custom_metrics", {}) or result.get("custom_metrics", {}) or {})

    def m(key, default=0.0):
        return cm.get(f"{key}_mean", cm.get(key, default))

    r_mean = ev.get("episode_return_mean", 0.0)
    l_mean = ev.get("episode_len_mean", 0.0)

    row = [
        iteration, r_mean, l_mean,
        m("attacks_attempted"),
        m("attacks_hit"),
        m("attacks_kill"),
        m("attack_hit_rate"),
        m("attack_kill_rate"),

        m("herbivore_episode_return"),
        m("predator_episode_return"),
        m("herbivore_episode_len"),
        m("predator_episode_len"),

        m("herbivore_survival_pct"),
        m("predator_survival_pct"),

        m("herbivore_reason_timeout_pct"),
        m("herbivore_reason_starvation_pct"),
        m("herbivore_reason_dehydration_pct"),
        m("herbivore_reason_predation_pct"),

        m("predator_reason_timeout_pct"),
        m("predator_reason_starvation_pct"),
        m("predator_reason_dehydration_pct"),
        m("predator_reason_predation_pct"),

        m("herbivore_eat_count"),
        m("herbivore_drink_count"),
        m("predator_drink_count"),

        m("predator_attack_attempt_count"),
        m("predator_attack_hit_count"),
        m("predator_attack_kill_count"),

        m("herbivore_avg_food"),
        m("herbivore_avg_water"),
        m("predator_avg_food"),
        m("predator_avg_water"),

        m("herbivore_critical_ratio"),
        m("predator_critical_ratio"),
    ]

    row.extend(m(f"agent_{a}/episode_return") for a in range(n_agents))
    row.extend(m(f"agent_{a}/episode_len") for a in range(n_agents))

    return row, r_mean, l_mean


def run_policy_episode(trainer, env_cfg: dict, step_callback=None):
    """
    Ejecuta un episodio usando las políticas entrenadas.
    Compatible con políticas normales y políticas con LSTM.
    """
    env = MultiAgentEcosystem(**env_cfg)
    obs, _ = env.reset()

    agent_states = init_agent_lstm_states(
        trainer=trainer,
        agents=env.possible_agents,
        env_cfg=env_cfg,
    )

    for step in range(env_cfg["max_steps"]):
        if not obs:
            break

        actions = {}

        for agent_id, agent_obs in obs.items():
            try:
                actions[agent_id] = compute_action_with_lstm_state(
                    trainer=trainer,
                    agent_id=agent_id,
                    agent_obs=agent_obs,
                    env_cfg=env_cfg,
                    agent_states=agent_states,
                    explore=False,
                )
            except Exception:
                actions[agent_id] = env.action_space(agent_id).sample()

        obs, rewards, terminations, truncations, infos = env.step(actions)

        if step_callback is not None:
            should_continue = step_callback(
                env=env,
                step=step,
                rewards=rewards,
                terminations=terminations,
                truncations=truncations,
                infos=infos,
            )

            if should_continue is False:
                break

        if all(terminations.values()) or all(truncations.values()):
            break

def main(enable_visualization: bool = False):
    ray.init(ignore_reinit_error=True)

    # --- Configuración del entorno ---
    ENV_CFG = DEFAULT_ENV_CFG.copy()
    n_agents = ENV_CFG["n_predators"] + ENV_CFG["n_herbivores"]

    # Parámetros de entrenamiento
    NUM_RUNNERS  = DEFAULT_NUM_RUNNERS
    FRAG = ENV_CFG["max_steps"]
    TOTAL_BATCH = NUM_RUNNERS * FRAG

    config = build_config(
        env_cfg=ENV_CFG,
        num_runners=NUM_RUNNERS,
        callbacks_class=PerAgentAndReasonMetrics,
    )

    trainer = config.build()
    start_vis_iter = int(NUM_ITERS * VIS_START_FRAC)

    visualizer = None
    if enable_visualization and HAS_PYGAME:
        visualizer = EcosystemVisualizer(
            map_width=ENV_CFG["map_width"],
            map_height=ENV_CFG["map_height"]
        )
        print("Visualización en tiempo real activada")

    output_csv = Path(__file__).resolve().parents[2] / "monitor.csv"
    print("Guardando monitor en:", output_csv)

    try:
        with open(output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(get_monitor_header(n_agents))

            for i in range(NUM_ITERS):
                result = trainer.train()
                row, r_mean, l_mean = build_monitor_row(result, i, n_agents)

                writer.writerow(row)
                f.flush()

                print(f"Iter {i}: ep_return_mean={r_mean}, ep_len_mean={l_mean}")

                if visualizer and i >= start_vis_iter and (i - start_vis_iter) % VIS_INTERVAL == 0:
                    try:
                        print(f"Mostrando visualización de la iteración {i}")

                        def render_step(env, step, rewards, terminations, truncations, infos):
                            alive_count = len(env.agents)
                            metrics = {
                                "mean_reward": np.mean(list(rewards.values())) if rewards else 0,
                                "total_reward": sum(rewards.values()) if rewards else 0,
                                "alive_agents": alive_count,
                                "total_agents": n_agents,
                                "iteration": i,
                            }

                            return visualizer.render(
                                env, env.species, step, i, rewards, metrics
                            )

                        run_policy_episode(trainer, ENV_CFG, step_callback=render_step)
                        print("Visualización completada")

                    except Exception as e:
                        print(f"Error en visualización: {e}")
    finally:
        if visualizer:
            visualizer.close()
        ray.shutdown()


if __name__ == "__main__":
    main()