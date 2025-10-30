"""
Módulo que define callbacks personalizados para el entrenamiento RL.

Los callbacks permiten interceptar eventos durante el entrenamiento para
registrar métricas personalizadas, especialmente:
- Retornos y longitudes por agente individual
- Razones de terminación de episodios (timeout, inanición, deshidratación)
"""
from ray.rllib.callbacks.callbacks import RLlibCallback

class PerAgentAndReasonMetrics(RLlibCallback):
    """
    Callback para registrar métricas detalladas por agente y razones de terminación.
    
    Este callback rastrea:
    - Retorno acumulado por cada agente individual
    - Longitud del episodio por cada agente individual
    - Razones de terminación (timeout, starvation, dehydration)
    - Porcentajes de cada tipo de terminación
    
    Las métricas se registran en RLlib y son visibles en TensorBoard y logs.
    """
    
    def __init__(self):
        """
        Inicializa el callback.
        
        Crea un diccionario de estado para rastrear información de episodios
        en curso, indexado por ID de episodio.
        """
        super().__init__()
        # Diccionario: episode_id -> {agent_id: {"ret": float, "len": int, "reason": str|None}}
        self._ep_state = {}

    def _eid(self, episode):
        """
        Obtiene el ID del episodio de forma compatible con diferentes versiones de RLlib.
        
        RLlib ha cambiado el nombre del atributo del ID de episodio en diferentes
        versiones. Este método intenta obtenerlo de múltiples formas para
        garantizar compatibilidad.
        
        Args:
            episode: Objeto episode de RLlib
            
        Returns:
            ID del episodio (puede ser string, int, o id de objeto Python)
        """
        # Intentar obtener con diferentes nombres de atributo
        return getattr(episode, "id_", None) or getattr(episode, "episode_id", None) or id(episode)

    def on_episode_start(self, *, episode, **kwargs):
        """
        Se llama al inicio de cada episodio.
        
        Inicializa el estado de rastreo para este nuevo episodio,
        creando un diccionario vacío para almacenar métricas por agente.
        
        Args:
            episode: Objeto episode de RLlib
            **kwargs: Argumentos adicionales (ignorados)
        """
        # Inicializar diccionario vacío para este episodio
        self._ep_state[self._eid(episode)] = {}

    def on_episode_step(self, *, episode, **kwargs):
        """
        Se llama después de cada paso del episodio.
        
        Actualiza el estado de rastreo con información del último paso,
        incluyendo retornos acumulados, longitudes y razones de terminación
        de cada agente.
        
        Args:
            episode: Objeto episode de RLlib con información del paso
            **kwargs: Argumentos adicionales (ignorados)
        """
        eid = self._eid(episode)
        # Obtener o crear diccionario de agentes para este episodio
        per_agent = self._ep_state.setdefault(eid, {})
        
        # Obtener información del último paso: dict {agent_id: info_dict}
        step_infos = episode.get_infos(-1) or {}

        # Actualizar estado de cada agente
        for agent_id, info in step_infos.items():
            if not info:
                continue  # Saltar si info está vacío
            
            # Obtener o crear registro para este agente
            rec = per_agent.setdefault(agent_id, {"ret": 0.0, "len": 0, "reason": None})
            
            # Actualizar retorno acumulado si está disponible
            if "ep_return" in info:
                rec["ret"] = float(info["ep_return"])
            
            # Actualizar longitud del episodio si está disponible
            if "ep_len" in info:
                rec["len"] = int(info["ep_len"])
            
            # Actualizar razón de terminación si está disponible
            r = info.get("reason")
            if r:
                rec["reason"] = r

    def on_episode_end(self, *, episode, metrics_logger, **kwargs):
        """
        Se llama al final de cada episodio.
        
        Registra todas las métricas acumuladas durante el episodio:
        - Retorno y longitud por cada agente individual
        - Porcentajes de cada tipo de razón de terminación
        
        Las métricas se registran usando el metrics_logger de RLlib con
        reducción "mean" y ventana de 50 episodios para suavizado.
        
        Args:
            episode: Objeto episode de RLlib
            metrics_logger: Logger de métricas de RLlib
            **kwargs: Argumentos adicionales (ignorados)
        """
        eid = self._eid(episode)
        # Obtener y eliminar el estado de este episodio
        per_agent = self._ep_state.pop(eid, {})

        # === REGISTRAR MÉTRICAS POR AGENTE ===
        for agent_id, rec in per_agent.items():
            # Registrar retorno del episodio para este agente
            metrics_logger.log_value(f"{agent_id}/episode_return", rec.get("ret", 0.0),
                                     reduce="mean", window=50)
            # Registrar longitud del episodio para este agente
            metrics_logger.log_value(f"{agent_id}/episode_len", rec.get("len", 0),
                                     reduce="mean", window=50)

        # === CALCULAR Y REGISTRAR RAZONES DE TERMINACIÓN ===
        # Razones posibles de terminación
        reasons = ["timeout", "starvation", "dehydration"]
        # Inicializar contadores
        counts  = {k: 0 for k in reasons}
        # Total de agentes (evitar división por 0)
        total   = max(1, len(per_agent))

        # Contar ocurrencias de cada razón
        for rec in per_agent.values():
            r = (rec.get("reason") or "").strip()
            if r in counts:
                counts[r] += 1

        # Registrar porcentajes de cada razón
        for k in reasons:
            pct = 100.0 * counts[k] / total  # Convertir a porcentaje
            metrics_logger.log_value(f"reason_{k}_pct", pct, reduce="mean", window=50)