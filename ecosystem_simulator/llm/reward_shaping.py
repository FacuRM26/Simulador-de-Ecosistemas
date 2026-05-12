"""
ENFOQUE 2: LLM para Reward Shaping Dinámico

El LLM NO decide acciones. En cambio:
1. Analiza el estado del agente en cada paso
2. Genera insights sobre qué es "bueno" en ese contexto
3. Ajusta dinámicamente la función de recompensa

Ejemplo:
- Si el LLM detecta "hay depredadores cercanos", agrega bonificación por alejarse
- Si detecta "está solo", agrega bonificación por acercarse a otros
- Si ve "mucha hambre", agrega penalización si no busca comida

Ventajas:
- El RL sigue siendo el motor principal
- El LLM solo modula (no controla)
- Permite comportamientos emergentes con objetivos dinámicos
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from .ollama_utils import call_ollama_with_retry, extract_first_line, OllamaError
import logging

logger = logging.getLogger(__name__)


class RewardShaper:
    """
    Modula dinámicamente la función de recompensa usando LLM.
    
    El LLM analiza el estado y genera "insights" que modifican
    los bonos/penalizaciones aplicados a cada acción.
    """

    def __init__(
        self,
        model: str = "mistral",
        temperature: float = 0.5,
        max_bonus_weight: float = 0.5,
        enable_logging: bool = True,
    ):
        """
        Args:
            model: Modelo de Ollama a usar.
            temperature: Control de creatividad del LLM.
            max_bonus_weight: Peso máximo de modificación de recompensa.
            enable_logging: Si registrar decisiones del LLM.
        """
        self.model = model
        self.temperature = temperature
        self.max_bonus_weight = max_bonus_weight
        self.enable_logging = enable_logging
        
        # Cache de insights para evitar llamadas repetidas
        self.insight_cache: Dict[str, Tuple[str, float]] = {}
        self.cache_size = 100
        self.llm_calls = 0
        self.cache_hits = 0

    def state_to_text(self, obs: np.ndarray, agent_id: str) -> str:
        """
        Convierte observación a descripción textual.
        """
        if len(obs) >= 5:
            x, y, energy, n_nearby, dist_to_prey = obs[:5]
            
            energy_level = "critica" if energy < 0.2 else "baja" if energy < 0.4 else "media" if energy < 0.7 else "alta"
            nearby_status = "rodeado" if n_nearby > 3 else "con compania" if n_nearby > 0 else "solo"
            prey_status = "presa cercana" if dist_to_prey < 100 else "presa lejana" if dist_to_prey < 300 else "sin presas visibles"
            
            return (
                f"Estado: energía {energy_level}, {nearby_status}, {prey_status}. "
                f"Posición ({x:.0f}, {y:.0f}). Agentes cercanos: {int(n_nearby)}."
            )
        return f"Agente {agent_id}, estado general: {np.mean(obs):.2f}"

    def get_llm_insight(self, obs: np.ndarray, agent_id: str) -> str:
        """
        Consulta al LLM sobre qué objetivo priorizar.
        
        Args:
            obs: Observación numérica.
            agent_id: ID del agente.
        
        Returns:
            Insight del LLM (ej: "buscar comida", "acercarse a grupo", etc).
        """
        state_desc = self.state_to_text(obs, agent_id)
        
        # Cache simple
        cache_key = f"{agent_id}:{state_desc[:50]}"
        if cache_key in self.insight_cache:
            self.cache_hits += 1
            return self.insight_cache[cache_key][0]
        
        prompt = f"""Eres un consejero para un agente en un ecosistema.
Estado actual: {state_desc}

¿Cuál es el objetivo MÁS IMPORTANTE en este momento?
Opciones: buscar_comida, buscar_agua, evitar_predador, acercarse_grupo, 
alejarse_grupo, explorar, congelarse.

Responde SOLO con UNA palabra (la opción), sin explicación."""

        try:
            response = call_ollama_with_retry(
                prompt,
                model=self.model,
                temperature=self.temperature,
                max_retries=1,
            )
            
            insight = extract_first_line(response).strip().lower()
            self.llm_calls += 1
            
            # Guardar en cache
            if len(self.insight_cache) < self.cache_size:
                self.insight_cache[cache_key] = (insight, self.llm_calls)
            
            if self.enable_logging:
                logger.debug(f"[RewardShaper] {agent_id}: {insight}")
            
            return insight
            
        except OllamaError as e:
            logger.warning(f"[RewardShaper] Error consultando LLM: {e}")
            return "explorar"  # comportamiento neutro por defecto

    def calculate_bonus(
        self,
        insight: str,
        action: int,
        obs: np.ndarray,
        base_reward: float,
    ) -> float:
        """
        Calcula bonificación de recompensa según insight del LLM.
        
        Args:
            insight: Insight del LLM (ej: "buscar_comida").
            action: Acción numérica (0-6 típicamente).
            obs: Observación del estado.
            base_reward: Recompensa base del entorno.
        
        Returns:
            Bonificación a agregar a la recompensa.
        """
        bonus = 0.0
        
        # Mapping simplificado de acciones:
        # 0-3: movimientos (north, south, east, west)
        # 4: comer
        # 5: beber
        # 6: atacar
        
        if insight == "buscar_comida":
            if action == 4:  # comer
                bonus = self.max_bonus_weight * 0.5
            elif action in [0, 1, 2, 3]:  # moverse (exploración)
                bonus = self.max_bonus_weight * 0.1
        
        elif insight == "buscar_agua":
            if action == 5:  # beber
                bonus = self.max_bonus_weight * 0.5
            elif action in [0, 1, 2, 3]:
                bonus = self.max_bonus_weight * 0.1
        
        elif insight == "evitar_predador":
            if action in [0, 1, 2, 3]:  # moverse
                # Preferir alejamiento
                if len(obs) >= 5:
                    bonus = self.max_bonus_weight * 0.3
            if action == 6:  # atacar (no es ideal al huir)
                bonus = -self.max_bonus_weight * 0.2
        
        elif insight == "acercarse_grupo":
            if action in [0, 1, 2, 3]:  # moverse hacia otros
                bonus = self.max_bonus_weight * 0.2
        
        elif insight == "alejarse_grupo":
            if action in [0, 1, 2, 3]:  # moverse
                bonus = self.max_bonus_weight * 0.1
        
        elif insight == "congelarse":
            if action not in [0, 1, 2, 3, 6]:  # no moverse ni atacar
                bonus = self.max_bonus_weight * 0.2
        
        return bonus

    def shape_reward(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        agent_id: str,
        use_llm: bool = True,
    ) -> float:
        """
        Aplica reward shaping basado en LLM.
        
        Args:
            obs: Observación actual.
            action: Acción tomada.
            reward: Recompensa base del entorno.
            agent_id: ID del agente.
            use_llm: Si usar LLM o simplemente retornar reward base.
        
        Returns:
            Recompensa ajustada.
        """
        if not use_llm:
            return reward
        
        insight = self.get_llm_insight(obs, agent_id)
        bonus = self.calculate_bonus(insight, action, obs, reward)
        
        shaped_reward = reward + bonus
        
        if self.enable_logging and bonus != 0:
            logger.debug(
                f"[RewardShaper] {agent_id}: insight='{insight}', "
                f"bonus={bonus:.4f}, reward: {reward:.4f} → {shaped_reward:.4f}"
            )
        
        return shaped_reward

    def get_statistics(self) -> Dict[str, Any]:
        """Retorna estadísticas de llamadas al LLM."""
        return {
            "llm_calls": self.llm_calls,
            "cache_hits": self.cache_hits,
            "cache_size": len(self.insight_cache),
            "efficiency": (
                self.cache_hits / (self.llm_calls + self.cache_hits)
                if (self.llm_calls + self.cache_hits) > 0
                else 0
            )
        }

    def reset_statistics(self) -> None:
        """Reinicia estadísticas."""
        self.llm_calls = 0
        self.cache_hits = 0
        self.insight_cache.clear()
