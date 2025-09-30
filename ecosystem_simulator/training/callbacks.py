"""
Módulo que define callbacks personalizados para el entrenamiento RL.
"""
from ray.rllib.callbacks.callbacks import RLlibCallback

class PerAgentAndReasonMetrics(RLlibCallback):
    """Callback para registrar métricas por agente y razones de terminación."""
    
    def __init__(self):
        super().__init__()
        self._ep_state = {}  # episode_id -> {agent_id: {"ret": float, "len": int, "reason": str|None}}

    def _eid(self, episode):
        # Compatible con distintas versiones de RLlib
        return getattr(episode, "id_", None) or getattr(episode, "episode_id", None) or id(episode)

    def on_episode_start(self, *, episode, **kwargs):
        """Inicializa el estado del episodio."""
        self._ep_state[self._eid(episode)] = {}

    def on_episode_step(self, *, episode, **kwargs):
        """Actualiza el estado del episodio en cada paso."""
        eid = self._eid(episode)
        per_agent = self._ep_state.setdefault(eid, {})
        step_infos = episode.get_infos(-1) or {}  # dict: agent_id -> info

        for agent_id, info in step_infos.items():
            if not info:
                continue
            rec = per_agent.setdefault(agent_id, {"ret": 0.0, "len": 0, "reason": None})
            if "ep_return" in info:
                rec["ret"] = float(info["ep_return"])
            if "ep_len" in info:
                rec["len"] = int(info["ep_len"])
            r = info.get("reason")
            if r:
                rec["reason"] = r

    def on_episode_end(self, *, episode, metrics_logger, **kwargs):
        """Registra métricas al final del episodio."""
        eid = self._eid(episode)
        per_agent = self._ep_state.pop(eid, {})

        # Métricas por agente
        for agent_id, rec in per_agent.items():
            metrics_logger.log_value(f"{agent_id}/episode_return", rec.get("ret", 0.0),
                                     reduce="mean", window=50)
            metrics_logger.log_value(f"{agent_id}/episode_len", rec.get("len", 0),
                                     reduce="mean", window=50)

        # Razones de terminación (siempre loguear, aunque sea 0)
        reasons = ["timeout", "starvation", "dehydration"]
        counts  = {k: 0 for k in reasons}
        total   = max(1, len(per_agent))  # evita división por 0

        for rec in per_agent.values():
            r = (rec.get("reason") or "").strip()
            if r in counts:
                counts[r] += 1

        for k in reasons:
            pct = 100.0 * counts[k] / total
            metrics_logger.log_value(f"reason_{k}_pct", pct, reduce="mean", window=50)