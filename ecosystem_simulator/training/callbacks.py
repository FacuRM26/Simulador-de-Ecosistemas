"""
Módulo que define callbacks personalizados para el entrenamiento RL.

Los callbacks permiten interceptar eventos durante el entrenamiento para
registrar métricas personalizadas, especialmente:
- Retornos y longitudes por agente individual
- Razones de terminación de episodios (timeout, inanición, deshidratación, depredación)
- Métricas de ataques (intentos, aciertos)
"""

from collections import Counter
from ray.rllib.algorithms.callbacks import DefaultCallbacks


class PerAgentAndReasonMetrics(DefaultCallbacks):
    def __init__(self):
        """
        Inicializa el callback.

        Crea un diccionario de estado para rastrear información de episodios
        en curso, indexado por ID de episodio.
        """
        super().__init__()
        # episode_id -> {
        #   "per_agent": {agent_id: {"ret": float, "len": int, "reason": str|None}},
        #   "att": int,
        #   "hit": int,
        #   "roles": {agent_id: "PREDATOR"|"HERBIVORE"}
        # }
        self._ep_state = {}

    # ------------------------------------------------------------------ #
    # Utilidad: obtener ID de episodio compatible con varias versiones
    # ------------------------------------------------------------------ #
    def _eid(self, episode):
        """
        Obtiene el ID del episodio de forma compatible con diferentes versiones de RLlib.
        """
        return (
            getattr(episode, "id_", None)
            or getattr(episode, "episode_id", None)
            or id(episode)
        )

    # ------------------------------------------------------------------ #
    # Inicio de episodio
    # ------------------------------------------------------------------ #
    def on_episode_start(self, *, episode, **kwargs):
        """
        Se llama al inicio de cada episodio.

        Inicializa el estado de rastreo para este nuevo episodio,
        creando la estructura donde se guardarán métricas por agente
        y contadores de ataques.
        """
        self._ep_state[self._eid(episode)] = {
            "per_agent": {},
            "att": 0,
            "hit": 0,
            "roles": {},
        }

    # ------------------------------------------------------------------ #
    # Paso de episodio
    # ------------------------------------------------------------------ #
    def on_episode_step(self, *, episode, **kwargs):
        """
        Se llama después de cada paso del episodio.

        Actualiza el estado de rastreo con información del último paso,
        incluyendo:
        - retorno acumulado por agente
        - longitud del episodio por agente
        - razón de terminación (si existe)
        - intentos y aciertos de ataques
        - rol de cada agente (PREDATOR / HERBIVORE)
        """
        eid = self._eid(episode)
        st = self._ep_state.setdefault(
            eid, {"per_agent": {}, "att": 0, "hit": 0, "roles": {}}
        )
        per_agent = st["per_agent"]
        roles = st["roles"]

        # Caso normal (RLlib v2, EpisodeV2)
        try:
            agent_ids = episode.get_agents()
            step_infos = {aid: (episode.last_info_for(aid) or {}) for aid in agent_ids}
        except AttributeError:
            # Fallback para APIs más viejas
            if hasattr(episode, "get_infos"):
                step_infos = episode.get_infos(-1) or {}
            else:
                step_infos = {}

        # Actualizar estado de cada agente
        for agent_id, info in step_infos.items():
            if not info:
                continue

            # Registro base por agente
            rec = per_agent.setdefault(
                agent_id, {"ret": 0.0, "len": 0, "reason": None}
            )

            # Retorno acumulado y longitud
            if "ep_return" in info:
                rec["ret"] = float(info["ep_return"])
            if "ep_len" in info:
                rec["len"] = int(info["ep_len"])

            # Razón de terminación (puede llegar sólo al final)
            if "reason" in info and info["reason"]:
                rec["reason"] = info["reason"]

            # Rol (se guarda la primera vez que aparezca)
            if "role" in info and agent_id not in roles:
                roles[agent_id] = info["role"]

            # Métricas de ataque
            st["att"] += int(info.get("attack_attempt", 0))
            st["hit"] += int(info.get("attack_hit", 0))

    # ------------------------------------------------------------------ #
    # Fin de episodio
    # ------------------------------------------------------------------ #
    def on_episode_end(self, *, episode, metrics_logger=None, **kwargs):
        """
        Se llama al final de cada episodio.

        Registra todas las métricas acumuladas durante el episodio:
        - Retorno y longitud por cada agente individual
        - Porcentajes de cada tipo de razón de terminación
        - Métricas de ataque agregadas
        """
        eid = self._eid(episode)
        st = self._ep_state.pop(
            eid, {"per_agent": {}, "att": 0, "hit": 0, "roles": {}}
        )
        per_agent, att, hit, roles = (
            st["per_agent"],
            st["att"],
            st["hit"],
            st["roles"],
        )

        # Helper para soportar metrics_logger nuevo o episode.custom_metrics viejo
        def log_metric(key, value):
            if metrics_logger is not None:
                metrics_logger.log_value(key, value, reduce="mean", window=50)
            else:
                episode.custom_metrics[key] = value

        # --- Métricas por agente ---
        for agent_id, rec in per_agent.items():
            log_metric(f"{agent_id}/episode_return", float(rec.get("ret", 0.0)))
            log_metric(f"{agent_id}/episode_len", int(rec.get("len", 0)))

        # Rol por agente (útil cuando mezcles depredadores/herbívoros)
        for agent_id, role in roles.items():
            log_metric(
                f"{agent_id}/role_is_predator", 1.0 if role == "PREDATOR" else 0.0
            )

        # --- Razones de terminación ---
        reasons = ["timeout", "starvation", "dehydration", "predation"]
        counts = Counter(
            (rec.get("reason") or "").strip() for rec in per_agent.values()
        )
        total = max(1, sum(1 for r in per_agent.values() if r.get("reason")))
        for k in reasons:
            pct = 100.0 * counts.get(k, 0) / total
            log_metric(f"reason_{k}_pct", pct)

        # --- Métricas de ataque ---
        log_metric("attacks_attempted", float(att))
        log_metric("attacks_hit", float(hit))
        rate = float(hit) / float(att) if att > 0 else 0.0
        log_metric("attack_hit_rate", rate)
