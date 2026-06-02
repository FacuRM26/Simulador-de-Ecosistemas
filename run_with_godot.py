"""
Script para entrenar con visualización en tiempo real en Godot.

Usa los mismos hiperparámetros y configuración que run_training.py,
añadiendo un servidor HTTP que Godot puede consultar en cada paso
para visualizar el estado del ecosistema.

Servidor disponible en: http://localhost:5000/state
"""
import sys
import os
import threading
import time
import csv

import ray

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# --- Módulos del ecosistema ---
from ecosystem_simulator.environment.multi_agent_ecosystem import MultiAgentEcosystem
from ecosystem_simulator.training.callbacks import PerAgentAndReasonMetrics
from ecosystem_simulator.training.trainer import (
    build_config,
    get_monitor_header,
    build_monitor_row,
    init_agent_lstm_states,
    compute_action_with_lstm_state,
)
from ecosystem_simulator.server.api_server import (
    run_server,
    update_ecosystem_state,
    serialize_ecosystem_state,
)
from ecosystem_simulator.utils.visualization import analyze_training_results


# ─────────────────────────────────────────────
#  Configuración central (igual que trainer.py)
# ─────────────────────────────────────────────
ENV_CFG = {
    "n_agents": 8,
    "veg_density": 15,
    "water_density": 10,
    "map_width": 800,
    "map_height": 600,
    "max_steps": 350,
    "n_predators": 2,
}
NUM_RUNNERS = 6
NUM_ITERS = 2000
VIS_START_IT = 2000


# ─────────────────────────────────────────────
#  Función de visualización hacia Godot
# ─────────────────────────────────────────────
def _run_godot_episode(trainer, env_cfg: dict, iteration: int) -> None:
    """
    Ejecuta un episodio completo usando las políticas entrenadas y
    envía cada paso al servidor HTTP para que Godot lo visualice.

    Compatible con políticas normales y políticas con LSTM.
    """
    viz_env = MultiAgentEcosystem(**env_cfg)
    obs, _ = viz_env.reset()

    agent_states = init_agent_lstm_states(
        trainer=trainer,
        agents=viz_env.possible_agents,
        env_cfg=env_cfg,
    )

    for viz_step in range(env_cfg["max_steps"]):
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
                actions[agent_id] = viz_env.action_space(agent_id).sample()

        obs, rewards, terminations, truncations, _ = viz_env.step(actions)

        state = serialize_ecosystem_state(
            viz_env,
            viz_env.species,
            episode=iteration,
            step=viz_step,
        )

        update_ecosystem_state(state)

        time.sleep(0.06)

        if all(terminations.values()) or all(truncations.values()):
            break

# ─────────────────────────────────────────────
#  Entrenamiento principal
# ─────────────────────────────────────────────
def train_with_godot() -> None:
    """
    Loop de entrenamiento PPO con envío de estado a Godot.
    Idéntico a trainer.main() salvo por la visualización Godot.
    """
    ray.init(ignore_reinit_error=True)

    config = build_config(
        ENV_CFG,
        num_runners=NUM_RUNNERS,
        callbacks_class=PerAgentAndReasonMetrics,
    )
    trainer = config.build()

    n_agents = ENV_CFG["n_agents"]

    with open("monitor.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(get_monitor_header(n_agents))

        for i in range(NUM_ITERS):
            start_time = time.time()

            result = trainer.train()

            iter_seconds = time.time() - start_time

            row, r_mean, l_mean = build_monitor_row(result, i, n_agents)

            writer.writerow(row)
            f.flush()

            print(
                f"Iter {i:>3}: "
                f"ep_return_mean={r_mean:.2f}, "
                f"ep_len_mean={l_mean:.2f}, "
                f"time={iter_seconds:.2f}s"
            )

            if i >= VIS_START_IT:
                try:
                    _run_godot_episode(trainer, ENV_CFG, iteration=i)
                except Exception as e:
                    print(f"[Godot] Error en episodio de visualización (iter {i}): {e}")

    print("\n[+] Entrenamiento completado")
    ray.shutdown()


# ─────────────────────────────────────────────
#  Entry-point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 70)
    print("  ENTRENAMIENTO CON VISUALIZACIÓN EN TIEMPO REAL → GODOT")
    print("=" * 70)
    print("  Servidor: http://localhost:5000/state")
    print("=" * 70)
    print()

    print("[+] Iniciando servidor HTTP...")
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    print("[+] Servidor HTTP activo en http://localhost:5000/state")

    print("[+] Iniciando entrenamiento...\n")

    try:
        train_with_godot()

        print("\n[+] Analizando resultados...")
        analyze_training_results("monitor.csv")

    except KeyboardInterrupt:
        print("\n\n[!] Entrenamiento interrumpido por el usuario")
    except Exception as e:
        import traceback
        print(f"\n[ERROR] {e}")
        traceback.print_exc()