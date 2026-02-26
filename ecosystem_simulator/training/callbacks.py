from collections import Counter
from ray.rllib.algorithms.callbacks import DefaultCallbacks

class PerAgentAndReasonMetrics(DefaultCallbacks):
    def __init__(self):
        super().__init__()
        self._ep_state = {}

    def _eid(self, episode):
        return getattr(episode, "id_", None) or getattr(episode, "episode_id", None) or id(episode)

    def on_episode_start(self, *, episode, **kwargs):
        self._ep_state[self._eid(episode)] = {
            "per_agent": {},
            "att": 0,
            "hit": 0,
            "kill": 0,
            "outcomes": Counter(),
            "roles": {},
        }

    def on_episode_step(self, *, episode, **kwargs):
        eid = self._eid(episode)
        st = self._ep_state.setdefault(eid, {
            "per_agent": {},
            "att": 0,
            "hit": 0,
            "kill": 0,
            "outcomes": Counter(),
            "roles": {},
        })

        per_agent = st["per_agent"]
        roles = st["roles"]

        try:
            agent_ids = episode.get_agents()
            step_infos = {aid: (episode.last_info_for(aid) or {}) for aid in agent_ids}
        except AttributeError:
            step_infos = episode.get_infos(-1) or {} if hasattr(episode, "get_infos") else {}

        for agent_id, info in step_infos.items():
            if not info:
                continue

            rec = per_agent.setdefault(agent_id, {"ret": 0.0, "len": 0, "reason": None})

            if "ep_return" in info:
                rec["ret"] = float(info["ep_return"])
            if "ep_len" in info:
                rec["len"] = int(info["ep_len"])

            if info.get("reason"):
                rec["reason"] = info["reason"]

            if "role" in info and agent_id not in roles:
                roles[agent_id] = info["role"]

            # Ataques (una sola vez)
            attempt = int(info.get("attack_attempt", 0))
            if attempt:
                st["att"] += 1
                outcome = (info.get("attack_outcome") or "").strip()

                if outcome in ("hit", "kill"):
                    st["hit"] += 1
                if outcome == "kill":
                    st["kill"] += 1

                st["outcomes"][outcome or "unknown"] += 1

    def on_episode_end(self, *, episode, metrics_logger=None, **kwargs):
        eid = self._eid(episode)
        st = self._ep_state.pop(eid, {
            "per_agent": {},
            "att": 0,
            "hit": 0,
            "kill": 0,
            "outcomes": Counter(),
            "roles": {},
        })

        per_agent = st["per_agent"]
        att = st["att"]
        hit = st["hit"]
        kill = st["kill"]
        outcomes = st["outcomes"]
        roles = st["roles"]

        def log_metric(key, value):
            if metrics_logger is not None:
                metrics_logger.log_value(key, value, reduce="mean", window=50)
            else:
                episode.custom_metrics[key] = value

        for agent_id, rec in per_agent.items():
            log_metric(f"{agent_id}/episode_return", float(rec.get("ret", 0.0)))
            log_metric(f"{agent_id}/episode_len", int(rec.get("len", 0)))

        for agent_id, role in roles.items():
            log_metric(f"{agent_id}/role_is_predator", 1.0 if role == "PREDATOR" else 0.0)

        reasons = ["timeout", "starvation", "dehydration", "predation"]
        counts = Counter((rec.get("reason") or "").strip() for rec in per_agent.values())
        total = max(1, sum(1 for r in per_agent.values() if r.get("reason")))
        for k in reasons:
            log_metric(f"reason_{k}_pct", 100.0 * counts.get(k, 0) / total)

        log_metric("attacks_attempted", float(att))
        log_metric("attacks_hit", float(hit))
        log_metric("attacks_kill", float(kill))
        log_metric("attack_hit_rate", float(hit) / att if att else 0.0)
        log_metric("attack_kill_rate", float(kill) / att if att else 0.0)

        for k, v in outcomes.items():
            log_metric(f"attack_outcome_{k}", float(v))