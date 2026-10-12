from collections import Counter, defaultdict
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
        }

    def on_episode_step(self, *, episode, **kwargs):
        eid = self._eid(episode)
        st = self._ep_state.setdefault(eid, {
            "per_agent": {},
            "att": 0,
            "hit": 0,
            "kill": 0,
            "outcomes": Counter(),
        })

        per_agent = st["per_agent"]

        step_infos = {}
        if hasattr(episode, "get_infos"):
            try:
                step_infos = episode.get_infos(-1) or {}
            except Exception:
                step_infos = {}

        if not step_infos:
            try:
                agent_ids = episode.get_agents()
                step_infos = {aid: (episode.last_info_for(aid) or {}) for aid in agent_ids}
            except Exception:
                step_infos = {}

        for agent_id, info in step_infos.items():
            if not info:
                continue

            rec = per_agent.setdefault(agent_id, {
                "ret": 0.0,
                "len": 0,
                "reason": "",
                "role": "",
                "eat": 0,
                "drink": 0,
                "attack_attempt": 0,
                "attack_hit": 0,
                "attack_kill": 0,
                "avg_food": 0.0,
                "avg_water": 0.0,
                "critical_steps": 0,
                "critical_ratio": 0.0,
                "avg_nn_dist": 0.0,
                "avg_pred_dist": 0.0,
            })

            rec["ret"] = float(info.get("ep_return", rec["ret"]))
            rec["len"] = int(info.get("ep_len", rec["len"]))
            rec["reason"] = info.get("reason", rec["reason"])
            rec["role"] = info.get("role", rec["role"])

            rec["eat"] = int(info.get("ep_eat", rec["eat"]))
            rec["drink"] = int(info.get("ep_drink", rec["drink"]))

            rec["attack_attempt"] = int(info.get("ep_attack_attempt", rec["attack_attempt"]))
            rec["attack_hit"] = int(info.get("ep_attack_hit", rec["attack_hit"]))
            rec["attack_kill"] = int(info.get("ep_attack_kill", rec["attack_kill"]))

            rec["avg_food"] = float(info.get("ep_avg_food", rec["avg_food"]))
            rec["avg_water"] = float(info.get("ep_avg_water", rec["avg_water"]))

            rec["critical_steps"] = int(info.get("ep_critical_steps", rec["critical_steps"]))
            rec["critical_ratio"] = float(info.get("ep_critical_ratio", rec["critical_ratio"]))

            # Comportamiento emergente
            rec["avg_nn_dist"] = float(info.get("ep_avg_nn_dist", rec["avg_nn_dist"]))
            rec["avg_pred_dist"] = float(info.get("ep_avg_pred_dist", rec["avg_pred_dist"]))

            # Ataques del step actual
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
        })

        per_agent = st["per_agent"]
        att = st["att"]
        hit = st["hit"]
        kill = st["kill"]
        outcomes = st["outcomes"]

        def log_metric(key, value):
            if metrics_logger is not None:
                metrics_logger.log_value(key, value, reduce="mean", window=50)
            else:
                episode.custom_metrics[key] = value

        def mean_val(records, key):
            if not records:
                return 0.0
            return sum(float(r.get(key, 0.0)) for _, r in records) / len(records)

        # ========= métricas por agente =========
        for agent_id, rec in per_agent.items():
            log_metric(f"{agent_id}/episode_return", float(rec.get("ret", 0.0)))
            log_metric(f"{agent_id}/episode_len", int(rec.get("len", 0)))

            log_metric(f"{agent_id}/eat_count", int(rec.get("eat", 0)))
            log_metric(f"{agent_id}/drink_count", int(rec.get("drink", 0)))

            log_metric(f"{agent_id}/attack_attempt_count", int(rec.get("attack_attempt", 0)))
            log_metric(f"{agent_id}/attack_hit_count", int(rec.get("attack_hit", 0)))
            log_metric(f"{agent_id}/attack_kill_count", int(rec.get("attack_kill", 0)))

            log_metric(f"{agent_id}/avg_food", float(rec.get("avg_food", 0.0)))
            log_metric(f"{agent_id}/avg_water", float(rec.get("avg_water", 0.0)))

            log_metric(f"{agent_id}/critical_steps", int(rec.get("critical_steps", 0)))
            log_metric(f"{agent_id}/critical_ratio", float(rec.get("critical_ratio", 0.0)))

            role = rec.get("role", "")
            log_metric(f"{agent_id}/role_is_predator", 1.0 if role == "PREDATOR" else 0.0)

        # ========= agrupar por rol =========
        by_role = defaultdict(list)
        for agent_id, rec in per_agent.items():
            role = rec.get("role", "")
            if role:
                by_role[role].append((agent_id, rec))

        role_name_map = {
            "HERBIVORE": "herbivore",
            "PREDATOR": "predator",
        }

        reasons = ["timeout", "starvation", "dehydration", "predation"]

        for role_raw, records in by_role.items():
            role_key = role_name_map.get(role_raw, role_raw.lower())
            total = max(1, len(records))

            log_metric(f"{role_key}_episode_return", mean_val(records, "ret"))
            log_metric(f"{role_key}_episode_len", mean_val(records, "len"))

            log_metric(f"{role_key}_eat_count", mean_val(records, "eat"))
            log_metric(f"{role_key}_drink_count", mean_val(records, "drink"))

            log_metric(f"{role_key}_attack_attempt_count", mean_val(records, "attack_attempt"))
            log_metric(f"{role_key}_attack_hit_count", mean_val(records, "attack_hit"))
            log_metric(f"{role_key}_attack_kill_count", mean_val(records, "attack_kill"))

            log_metric(f"{role_key}_avg_food", mean_val(records, "avg_food"))
            log_metric(f"{role_key}_avg_water", mean_val(records, "avg_water"))
            log_metric(f"{role_key}_critical_ratio", mean_val(records, "critical_ratio"))
            log_metric(f"{role_key}_critical_steps", mean_val(records, "critical_steps"))
            log_metric(f"{role_key}_avg_nn_dist", mean_val(records, "avg_nn_dist"))
            log_metric(f"{role_key}_avg_pred_dist", mean_val(records, "avg_pred_dist"))

            counts = Counter((r.get("reason") or "").strip() for _, r in records)
            for reason in reasons:
                log_metric(
                    f"{role_key}_reason_{reason}_pct",
                    100.0 * counts.get(reason, 0) / total
                )

            # Supervivencia = llegar a timeout
            log_metric(
                f"{role_key}_survival_pct",
                100.0 * counts.get("timeout", 0) / total
            )

        # ========= métricas globales de ataque =========
        log_metric("attacks_attempted", float(att))
        log_metric("attacks_hit", float(hit))
        log_metric("attacks_kill", float(kill))
        log_metric("attack_hit_rate", float(hit) / att if att else 0.0)
        log_metric("attack_kill_rate", float(kill) / att if att else 0.0)

        for k, v in outcomes.items():
            log_metric(f"attack_outcome_{k}", float(v))