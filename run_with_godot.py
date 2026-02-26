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

import numpy as np
import ray

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# --- Módulos del ecosistema ---
from ecosystem_simulator.environment.multi_agent_ecosystem import MultiAgentEcosystem
from ecosystem_simulator.training.trainer import build_config
from ecosystem_simulator.training.callbacks import PerAgentAndReasonMetrics
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
    "map_height": 600,   # igual que trainer.py
    "max_steps": 350,
    "n_predators": 2,
}
NUM_RUNNERS  = 4
NUM_ITERS    = 100
VIS_START_IT = 5   # Iteración a partir de la cual se envía estado a Godot


# ─────────────────────────────────────────────
#  Función de visualización hacia Godot
# ─────────────────────────────────────────────
def _run_godot_episode(trainer, env_cfg: dict, iteration: int) -> None:
    """
    Ejecuta un episodio completo usando las políticas entrenadas y
    envía cada paso al servidor HTTP para que Godot lo visualice.

    Args:
        trainer: Algoritmo PPO ya entrenado.
        env_cfg: Configuración del entorno.
        iteration: Número de iteración actual (para el campo 'episode').
    """
    viz_env = MultiAgentEcosystem(**env_cfg)
    obs, _ = viz_env.reset()

    for viz_step in range(env_cfg["max_steps"]):
        if not obs:
            break

        # Calcular acciones con las políticas entrenadas
        actions = {}
        for agent_id, agent_obs in obs.items():
            try:
                idx    = int(agent_id.split("_")[1])
                pol_id = "pred" if idx < env_cfg["n_predators"] else "herb"
                out    = trainer.get_policy(pol_id).compute_single_action(
                    agent_obs, explore=False
                )
                actions[agent_id] = out[0] if isinstance(out, tuple) else out
            except Exception:
                actions[agent_id] = viz_env.action_space(agent_id).sample()

        obs, rewards, terminations, truncations, _ = viz_env.step(actions)

        # Serializar y enviar estado a Godot
        state = serialize_ecosystem_state(
            viz_env,
            viz_env.species,
            episode=iteration,
            step=viz_step,
        )
        update_ecosystem_state(state)

        time.sleep(0.06)  # Pausa para que Godot pueda consumir el estado

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

    # Construir config reutilizando el módulo de trainer
    config  = build_config(ENV_CFG, num_runners=NUM_RUNNERS,
                           callbacks_class=PerAgentAndReasonMetrics)
    trainer = config.build()

    n_agents = ENV_CFG["n_agents"]

    with open("monitor.csv", "w", newline="") as f:
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

        for i in range(NUM_ITERS):
            result = trainer.train()

            # ── Extraer métricas ──────────────────────────────────────────
            ev = result.get("env_runners", {}) or {}
            cm = (ev.get("custom_metrics", {}) or
                  result.get("custom_metrics", {}) or {})

            def m(key, default=0.0):
                return cm.get(f"{key}_mean", default)

            r_mean = ev.get("episode_return_mean", 0.0)
            l_mean = ev.get("episode_len_mean",    0.0)

            row = [
                i, r_mean, l_mean,
                m("reason_timeout_pct"),
                m("reason_starvation_pct"),
                m("reason_dehydration_pct"),
                m("reason_predation_pct"),
                m("attacks_attempted"),
                m("attacks_hit"),
                m("attack_hit_rate"),
            ]
            for a in range(n_agents):
                row.append(m(f"agent_{a}/episode_return"))
            for a in range(n_agents):
                row.append(m(f"agent_{a}/episode_len"))

            writer.writerow(row)
            f.flush()  # Asegurar escritura inmediata al disco

            print(f"Iter {i:>3}: ep_return_mean={r_mean:.2f}, ep_len_mean={l_mean:.2f}")

            # ── Enviar estado a Godot (solo desde VIS_START_IT) ───────────
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

    # Servidor HTTP en hilo daemon (se cierra al terminar el proceso)
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