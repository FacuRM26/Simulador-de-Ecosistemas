# callbacks.py
from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from collections import Counter

class PerAgentAndReasonMetrics(DefaultCallbacks):
    def __init__(self):
        super().__init__()
        self._ep_state = {}

    def _eid(self, episode):
        return getattr(episode, "id_", None) or getattr(episode, "episode_id", None) or id(episode)

    def on_episode_start(self, *, episode, **kwargs):
        self._ep_state[self._eid(episode)] = {"per_agent": {}, "att": 0, "hit": 0, "roles": {}}

    def on_episode_step(self, *, episode, **kwargs):
        eid = self._eid(episode)
        st = self._ep_state.setdefault(eid, {"per_agent": {}, "att": 0, "hit": 0, "roles": {}})
        per_agent = st["per_agent"]
        roles = st.setdefault("roles", {})

        # RLlib v2: EpisodeV2
        try:
            agent_ids = episode.get_agents()
            step_infos = {aid: (episode.last_info_for(aid) or {}) for aid in agent_ids}
            roles = st["roles"]
            for agent_id, info in step_infos.items():
                if info and "role" in info and agent_id not in roles:
                    roles[agent_id] = info["role"]

        except AttributeError:
            # Muy legacy; si no existe, no hacemos nada
            step_infos = {}

        for agent_id, info in step_infos.items():
            if not info:
                continue
            rec = per_agent.setdefault(agent_id, {"ret": 0.0, "len": 0, "reason": None})
            if "ep_return" in info: rec["ret"] = float(info["ep_return"])
            if "ep_len" in info:    rec["len"]  = int(info["ep_len"])
            if "reason" in info and info["reason"]:
                rec["reason"] = info["reason"]
            st["att"] += int(info.get("attack_attempt", 0))
            st["hit"] += int(info.get("attack_hit", 0))

    def on_episode_end(self, *, episode, metrics_logger=None, **kwargs):
        """Soporta tanto el stack nuevo (metrics_logger) como el legacy (episode.custom_metrics)."""
        eid = self._eid(episode)
        st = self._ep_state.pop(eid, {"per_agent": {}, "att": 0, "hit": 0})
        per_agent, att, hit = st["per_agent"], st["att"], st["hit"]

        # Helpers de salida
        def log_metric(key, value):
            if metrics_logger is not None:
                metrics_logger.log_value(key, value, reduce="mean", window=50)
            else:
                episode.custom_metrics[key] = value
        for agent_id, rec in per_agent.items():
            log_metric(f"{agent_id}/episode_return", float(rec.get("ret", 0.0)))
            log_metric(f"{agent_id}/episode_len",    int(rec.get("len", 0)))
        # Por-agente
        for agent_id, role in st["roles"].items():
            log_metric(f"{agent_id}/role_is_predator", 1.0 if role == "PREDATOR" else 0.0)

        # Razones (porcentaje sobre agentes que reportaron razón)
        reasons = ["timeout", "starvation", "dehydration", "predation"]
        counts = Counter((rec.get("reason") or "").strip() for rec in per_agent.values())
        total = max(1, sum(1 for r in per_agent.values() if r.get("reason")))
        for k in reasons:
            pct = 100.0 * counts.get(k, 0) / total
            log_metric(f"reason_{k}_pct", pct)

        # Ataques
        log_metric("attacks_attempted", float(att))
        log_metric("attacks_hit", float(hit))
        rate = (float(hit) / float(att)) if att > 0 else 0.0
        log_metric("attack_hit_rate", rate)
