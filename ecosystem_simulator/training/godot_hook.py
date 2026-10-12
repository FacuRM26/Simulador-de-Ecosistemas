"""
Hook de visualización en Godot.

Arranca el servidor HTTP y, cuando el orquestador se lo pide, corre un
episodio de visualización REAL con las políticas entrenadas actuales (el
mismo entorno y las mismas políticas del entrenamiento; nada simulado) y
transmite cada paso al servidor para que Godot lo consuma en tiempo real.

Se corre como episodio aparte solo porque los episodios del entrenamiento
ocurren en paralelo dentro de los workers de Ray y no son observables en
vivo; el comportamiento es exactamente el de la política entrenada.

Es un módulo autocontenido: el orquestador solo hace `start()` una vez y
luego `stream_episode(...)` al final de cada iteración que quiera visualizar.
"""
import threading
import time

from ..environment.multi_agent_ecosystem import MultiAgentEcosystem
from ..server.api_server import (
    run_server,
    update_ecosystem_state,
    serialize_ecosystem_state,
)
from .trainer import init_agent_lstm_states, compute_action_with_lstm_state


class GodotStreamer:
    """Servidor HTTP + streaming de episodios de visualización reales hacia Godot."""

    def __init__(self, host: str = "0.0.0.0", port: int = 5000,
                 step_delay: float = 0.06, explore: bool = True):
        """
        Args:
            host / port: dónde escucha el servidor HTTP (Godot hace polling a /state).
            step_delay: pausa entre pasos para que Godot alcance a consumir cada frame.
            explore: si True, la visualización usa la política ESTOCÁSTICA (muestrea acciones),
                que es el comportamiento realmente entrenado. Con explore=False
                (argmax greedy) los agentes se ven degenerados: los herbívoros casi
                nunca comen y los depredadores casi nunca beben, aunque en el
                entrenamiento sí lo hacen. Por eso el default es True.
        """
        self.host = host
        self.port = port
        self.step_delay = step_delay
        self.explore = explore
        self._server_thread = threading.Thread(
            target=run_server,
            kwargs={"host": host, "port": port},
            daemon=True,
        )

    def start(self) -> None:
        """Arranca el servidor HTTP en un hilo daemon."""
        self._server_thread.start()
        print(f"[Godot] Servidor HTTP activo en http://localhost:{self.port}/state")

    def stream_episode(self, trainer, env_cfg: dict, iteration: int) -> None:
        """
        Corre un episodio completo con las políticas entrenadas y envía cada
        paso al servidor. Compatible con políticas normales y con LSTM.
        """
        viz_env = MultiAgentEcosystem(**env_cfg)
        obs, _ = viz_env.reset()

        agent_states = init_agent_lstm_states(
            trainer=trainer,
            agents=viz_env.possible_agents,
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
                        explore=self.explore,
                    )
                except Exception as e:
                    # Si esto se dispara, la política entrenada NO se está usando y
                    # el agente actúa al azar. Lo mostramos en vez de ocultarlo.
                    print(f"[Godot] fallo policy {agent_id}: {type(e).__name__}: {e}")
                    actions[agent_id] = viz_env.action_space(agent_id).sample()

            obs, rewards, terminations, truncations, _ = viz_env.step(actions)

            state = serialize_ecosystem_state(
                viz_env,
                viz_env.species,
                episode=iteration,
                step=step,
            )
            update_ecosystem_state(state)

            time.sleep(self.step_delay)  # tiempo entre fotogramas

            if all(terminations.values()) or all(truncations.values()):
                break
