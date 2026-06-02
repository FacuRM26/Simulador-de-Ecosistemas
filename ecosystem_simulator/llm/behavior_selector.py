"""
ENFOQUE 1: LLM como Selector de Comportamientos Pre-entrenados

El LLM NO decide acciones numéricas. En cambio:
1. Se entrenan múltiples políticas RL para comportamientos específicos
2. El LLM analiza el estado y selecciona cuál política activar
3. La política seleccionada genera la acción

Ventajas:
- No modifica el entrenamiento RL existente
- El LLM solo hace clasificación (tarea simple y confiable)
- Las políticas son explícitas y interpretables
"""

import numpy as np
from typing import Dict, Callable, Any, Optional, List
from .ollama_utils import call_ollama_with_retry, extract_first_line, OllamaError
import logging

logger = logging.getLogger(__name__)


class BehaviorSelector:
    """
    Selector de comportamientos basado en LLM.
    
    Mantiene un conjunto de políticas pre-entrenadas y usa LLM para
    decidir cuál activar según el estado actual.
    """

    def __init__(
        self,
        available_behaviors: List[str],
        model: str = "mistral",
        temperature: float = 0.3,
        enable_logging: bool = True,
    ):
        """
        Args:
            available_behaviors: Lista de nombres de comportamientos disponibles.
            model: Modelo de Ollama a usar.
            temperature: Control de creatividad del LLM.
            enable_logging: Si registrar decisiones del LLM.
        """
        self.available_behaviors = available_behaviors
        self.model = model
        self.temperature = temperature
        self.enable_logging = enable_logging
        
        # Políticas almacenadas: {nombre: función}
        self.policies: Dict[str, Callable] = {}
        
        # Estadísticas
        self.selection_stats: Dict[str, int] = {b: 0 for b in available_behaviors}
        self.total_selections = 0

    def register_policy(
        self,
        behavior_name: str,
        policy_fn: Callable[[np.ndarray], int],
    ) -> None:
        """
        Registra una política para un comportamiento.
        
        Args:
            behavior_name: Nombre del comportamiento.
            policy_fn: Función que toma observación y devuelve acción.
        """
        if behavior_name not in self.available_behaviors:
            raise ValueError(f"Comportamiento '{behavior_name}' no en lista disponible")
        self.policies[behavior_name] = policy_fn

    def state_to_text(self, obs: np.ndarray, agent_id: str) -> str:
        """
        Convierte observación numérica a descripción textual.
        
        Esto es simplificado - en producción podría ser más detallado.
        
        Args:
            obs: Vector de observación numérica.
            agent_id: ID del agente.
        
        Returns:
            Descripción textual del estado.
        """
        # Asumir que obs = [x, y, energía, n_cercanos, distancia_presa, ...]
        if len(obs) >= 5:
            x, y, energy, n_nearby, dist_to_prey = obs[:5]
            
            energy_level = "baja" if energy < 0.3 else "media" if energy < 0.7 else "alta"
            nearby_level = "muchos" if n_nearby > 3 else "algunos" if n_nearby > 0 else "ninguno"
            
            return (
                f"Agente en ({x:.0f}, {y:.0f}) con energía {energy_level}. "
                f"Hay {nearby_level} agentes cercanos. "
                f"Distancia a presa: {dist_to_prey:.1f}."
            )
        return f"Agente {agent_id} con energía normalizada: {np.mean(obs):.2f}"

    def select_behavior(self, obs: np.ndarray, agent_id: str) -> str:
        """
        Usa LLM para seleccionar el mejor comportamiento.
        
        Args:
            obs: Observación numérica del agente.
            agent_id: ID del agente.
        
        Returns:
            Nombre del comportamiento seleccionado.
        """
        state_description = self.state_to_text(obs, agent_id)
        
        prompt = f"""Eres un agente inteligente en un ecosistema. Tu estado actual:

{state_description}

Comportamientos disponibles:
{chr(10).join(f"- {b}" for b in self.available_behaviors)}

Basado en tu estado, ¿cuál comportamiento es el más adecuado?
Responde SOLO con el nombre del comportamiento, sin explicación."""

        try:
            response = call_ollama_with_retry(
                prompt,
                model=self.model,
                temperature=self.temperature,
                max_retries=1,
            )
            
            # Extraer primera línea
            selected = extract_first_line(response).strip().lower()
            
            # Buscar coincidencia en comportamientos disponibles
            for behavior in self.available_behaviors:
                if behavior.lower() in selected or selected in behavior.lower():
                    if self.enable_logging:
                        logger.info(f"[LLM] {agent_id} seleccionó: {behavior}")
                    self.selection_stats[behavior] += 1
                    self.total_selections += 1
                    return behavior
            
            # Si no hay coincidencia, usar el primero por defecto
            logger.warning(
                f"[LLM] Respuesta '{selected}' no reconocida. "
                f"Usando '{self.available_behaviors[0]}'."
            )
            return self.available_behaviors[0]
            
        except OllamaError as e:
            logger.error(f"[LLM] Error en selector: {e}. Comportamiento por defecto.")
            return self.available_behaviors[0]

    def get_action(
        self,
        obs: np.ndarray,
        agent_id: str,
        use_llm: bool = True,
    ) -> int:
        """
        Obtiene la acción para un agente.
        
        Args:
            obs: Observación numérica.
            agent_id: ID del agente.
            use_llm: Si usar LLM para seleccionar comportamiento (False = uso directo).
        
        Returns:
            Acción (entero).
        """
        if not self.policies:
            raise RuntimeError("No hay políticas registradas")
        
        if use_llm:
            behavior = self.select_behavior(obs, agent_id)
        else:
            behavior = self.available_behaviors[0]
        
        policy = self.policies[behavior]
        return policy(obs)

    def get_statistics(self) -> Dict[str, Any]:
        """Retorna estadísticas de selecciones del LLM."""
        return {
            "total_selections": self.total_selections,
            "per_behavior": self.selection_stats.copy(),
            "distribution": {
                b: (self.selection_stats[b] / self.total_selections 
                    if self.total_selections > 0 else 0)
                for b in self.available_behaviors
            }
        }

    def reset_statistics(self) -> None:
        """Reinicia las estadísticas."""
        self.selection_stats = {b: 0 for b in self.available_behaviors}
        self.total_selections = 0
