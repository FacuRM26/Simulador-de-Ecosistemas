"""
Configuración de PPO y helpers reutilizables de entrenamiento.

Este módulo NO ejecuta el loop de entrenamiento (eso vive en
`orchestrator.py`). Solo expone:
  - build_config: la configuración PPO lista para usar.
  - helpers de política/estado LSTM.
  - get_monitor_header / build_monitor_row: filas del monitor.csv.
  - run_policy_episode: correr un episodio con las políticas entrenadas.
"""
import os
os.environ.pop("AIR_VERBOSITY", None)

from ..environment.multi_agent_ecosystem import MultiAgentEcosystem
from .callbacks import PerAgentAndReasonMetrics


def build_config(
    env_cfg: dict,
    num_runners: int = 4,
    callbacks_class=None,
    use_lstm: bool = False,
    num_gpus: int = 0,
    seed: int | None = None,
):
    """
    Construye y retorna la configuración PPO lista para usar.

    use_lstm=False por defecto: la observación YA incluye features de memoria
    (índices 16-33), así que una MLP es suficiente y aprende más rápido/estable
    que una política recurrente. Puedes activar la LSTM para experimentar.
    """
    from ray.tune.registry import register_env
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
    from ray.rllib.policy.policy import PolicySpec

    if callbacks_class is None:
        callbacks_class = PerAgentAndReasonMetrics

    frag = env_cfg["max_steps"]

    # Batch total = num_runners * frag (p.ej. 4 * 350 = 1400).
    total_batch = num_runners * frag

    if use_lstm:
        # Con LSTM el minibatch debe respetar secuencias completas.
        minibatch = frag
        num_epochs = 8
    else:
        # MLP: minibatch más pequeño => más pasos de SGD por iteración.
        minibatch = 256
        num_epochs = 10

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

    # Old API stack
    config.api_stack(
        enable_rl_module_and_learner=False,
        enable_env_runner_and_connector_v2=False,
    )

    # Forzar uso de CPU
    config.resources(
        num_gpus=num_gpus,
    )

    # Configuración de los runners.
    # batch_mode se configura aquí, sin usar rollouts().
    config.env_runners(
        num_env_runners=num_runners,
        rollout_fragment_length=frag,
        sample_timeout_s=300,
        batch_mode="truncate_episodes",
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

    # NOTA: estás en el old API stack (enable_rl_module_and_learner=False),
    # cuyos nombres canónicos son num_sgd_iter / sgd_minibatch_size.
    # vf_clip_param=10 era el gran problema: con retornos de episodio de decenas
    # (o cientos), la función de valor quedaba recortada y no podía ajustar la
    # línea base => las ventajas (GAE) salían basura => la política no aprendía.
    train_kwargs = dict(
        train_batch_size=total_batch,
        lr=3e-4,
        gamma=0.99,
        lambda_=0.95,
        clip_param=0.2,
        vf_clip_param=50.0,   # acorde a la escala de recompensa (antes 10)
        grad_clip=1.0,        # menos agresivo que 0.5
        entropy_coeff=0.01,
        model=model_cfg,
    )

    try:
        config.training(
            num_sgd_iter=num_epochs,
            sgd_minibatch_size=minibatch,
            **train_kwargs,
        )
    except TypeError:
        config.training(
            num_epochs=num_epochs,
            minibatch_size=minibatch,
            **train_kwargs,
        )

        if seed is not None:
            config.debugging(seed=seed)

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