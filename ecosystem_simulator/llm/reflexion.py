"""
ENFOQUE 3: LLM con Reflexion (Aprendizaje entre Episodios)

Inspirado en el paper "Reflexion: Language Agents with Verbal Reinforcement Learning"

El LLM:
1. Analiza trayectorias completas de episodios (al finalizar)
2. Genera "lecciones aprendidas" en lenguaje natural
3. Almacena estas lecciones en memoria a largo plazo
4. Inyecta las lecciones en prompts futuros para mejorar decisiones

Ventajas:
- Aprendizaje acumulativo a nivel episódico
- El agente "recuerda" y aprende de sus errores
- El LLM genera explicaciones interpretables
- Puede reducir el número de iteraciones necesarias
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import deque
from datetime import datetime
from .ollama_utils import call_ollama_with_retry, extract_first_line, OllamaError
import logging

logger = logging.getLogger(__name__)


class EpisodeMemory:
    """
    Almacena información de episodios y lecciones aprendidas.
    """

    def __init__(self, max_history: int = 50):
        """
        Args:
            max_history: Máximo número de episodios a recordar.
        """
        self.max_history = max_history
        self.episodes: deque = deque(maxlen=max_history)
        self.lessons: deque = deque(maxlen=5)  # Últimas 5 lecciones principales

    def add_episode(
        self,
        episode_number: int,
        trajectory: List[Dict],
        success: bool,
        total_reward: float,
        duration: int,
    ) -> None:
        """
        Agrega un episodio al historial.
        
        Args:
            episode_number: Número del episodio.
            trajectory: Lista de pasos (cada uno: {obs, action, reward, done}).
            success: Si el episodio fue exitoso.
            total_reward: Recompensa total del episodio.
            duration: Duración del episodio en pasos.
        """
        self.episodes.append({
            "number": episode_number,
            "timestamp": datetime.now().isoformat(),
            "success": success,
            "total_reward": total_reward,
            "duration": duration,
            "trajectory": trajectory,
        })

    def add_lesson(self, lesson: str) -> None:
        """Agrega una lección aprendida."""
        self.lessons.append({
            "timestamp": datetime.now().isoformat(),
            "text": lesson,
        })

    def get_recent_lessons(self, n: int = 3) -> str:
        """Retorna las últimas N lecciones como texto."""
        if not self.lessons:
            return "No hay lecciones previas aún."
        
        recent = list(self.lessons)[-n:]
        lessons_text = "\n".join(
            f"- {lesson['text']}"
            for lesson in recent
        )
        return lessons_text

    def get_episode_summary(self, episode_idx: int) -> str:
        """Genera un resumen textual de un episodio."""
        if episode_idx >= len(self.episodes):
            return ""
        
        ep = list(self.episodes)[episode_idx]
        traj = ep["trajectory"][:5]  # Primeros 5 pasos
        
        summary = f"Episodio {ep['number']} (éxito: {ep['success']}, recompensa: {ep['total_reward']:.2f}):\n"
        for i, step in enumerate(traj):
            summary += f"  Paso {i}: acción={step.get('action', '?')}, recompensa={step.get('reward', 0):.2f}\n"
        
        return summary

    def get_statistics(self) -> Dict[str, Any]:
        """Retorna estadísticas del historial."""
        if not self.episodes:
            return {"episodes": 0}
        
        episodes_list = list(self.episodes)
        successful = sum(1 for ep in episodes_list if ep["success"])
        avg_reward = np.mean([ep["total_reward"] for ep in episodes_list])
        avg_duration = np.mean([ep["duration"] for ep in episodes_list])
        
        return {
            "total_episodes": len(episodes_list),
            "successful": successful,
            "success_rate": successful / len(episodes_list) if episodes_list else 0,
            "avg_reward": avg_reward,
            "avg_duration": avg_duration,
            "lessons_learned": len(self.lessons),
        }


class ReflectionAgent:
    """
    Agente que usa Reflexion para aprender de episodios completos.
    """

    def __init__(
        self,
        model: str = "mistral",
        temperature: float = 0.6,
        enable_logging: bool = True,
    ):
        """
        Args:
            model: Modelo de Ollama a usar.
            temperature: Control de creatividad del LLM.
            enable_logging: Si registrar reflexiones.
        """
        self.model = model
        self.temperature = temperature
        self.enable_logging = enable_logging
        
        # Memoria del agente
        self.memory = EpisodeMemory()
        
        # Contador de episodios
        self.episode_count = 0
        self.reflection_count = 0

    def trajectory_to_text(self, trajectory: List[Dict], max_steps: int = 10) -> str:
        """
        Convierte una trayectoria a descripción textual.
        
        Args:
            trajectory: Lista de pasos del episodio.
            max_steps: Máximo número de pasos a incluir.
        
        Returns:
            Descripción textual de la trayectoria.
        """
        text = f"Trayectoria ({len(trajectory)} pasos total):\n"
        
        for i, step in enumerate(trajectory[:max_steps]):
            action = step.get("action", "?")
            reward = step.get("reward", 0)
            obs = step.get("obs", [])
            
            # Convertir acción numérica a nombre
            action_names = {
                0: "norte", 1: "sur", 2: "este", 3: "oeste",
                4: "comer", 5: "beber", 6: "atacar"
            }
            action_name = action_names.get(action, f"accion_{action}")
            
            text += f"  Paso {i}: {action_name} → recompensa {reward:.2f}\n"
        
        if len(trajectory) > max_steps:
            text += f"  ... ({len(trajectory) - max_steps} pasos más)\n"
        
        return text

    def reflect_on_episode(
        self,
        trajectory: List[Dict],
        success: bool,
        total_reward: float,
    ) -> str:
        """
        Usa LLM para reflexionar sobre un episodio completo.
        
        Args:
            trajectory: Trayectoria del episodio.
            success: Si fue exitoso.
            total_reward: Recompensa total.
        
        Returns:
            Lección aprendida generada por el LLM.
        """
        traj_text = self.trajectory_to_text(trajectory)
        recent_lessons = self.memory.get_recent_lessons(2)
        
        prompt = f"""Eres un agente reflexivo en un ecosistema que aprende de sus experiencias.

Lecciones previas:
{recent_lessons}

Episodio reciente (éxito: {success}, recompensa total: {total_reward:.2f}):
{traj_text}

Analiza por qué el episodio {'tuvo éxito' if success else 'falló'}.
Escribe 1-2 oraciones sobre la lección aprendida en primera persona.
Sé específico y accionable."""

        try:
            response = call_ollama_with_retry(
                prompt,
                model=self.model,
                temperature=self.temperature,
                max_retries=1,
            )
            
            lesson = response.strip()
            
            if self.enable_logging:
                logger.info(f"[Reflexion] Lección: {lesson[:80]}...")
            
            self.memory.add_lesson(lesson)
            self.reflection_count += 1
            
            return lesson
            
        except OllamaError as e:
            logger.warning(f"[Reflexion] Error al reflexionar: {e}")
            return f"Episodio con resultado: {'éxito' if success else 'fracaso'}"

    def process_episode(
        self,
        trajectory: List[Dict],
        success: bool,
        total_reward: float,
    ) -> None:
        """
        Procesa un episodio completo y genera reflexión.
        
        Args:
            trajectory: Trayectoria del episodio.
            success: Si fue exitoso.
            total_reward: Recompensa total.
        """
        self.episode_count += 1
        duration = len(trajectory)
        
        # Generar reflexión
        lesson = self.reflect_on_episode(trajectory, success, total_reward)
        
        # Guardar episodio
        self.memory.add_episode(
            episode_number=self.episode_count,
            trajectory=trajectory,
            success=success,
            total_reward=total_reward,
            duration=duration,
        )

    def get_context_prompt(self) -> str:
        """
        Genera un prompt de contexto con lecciones aprendidas.
        
        Esto se puede usar para inyectar en prompts futuros.
        
        Returns:
            String con contexto de lecciones aprendidas.
        """
        stats = self.memory.get_statistics()
        
        if stats["episodes"] == 0:
            return "Sin experiencia previa aún."
        
        lessons = self.memory.get_recent_lessons(3)
        success_rate = stats.get("success_rate", 0)
        
        context = f"""Experiencia acumulada:
- Total episodios: {stats['episodes']}
- Tasa de éxito: {success_rate:.1%}
- Recompensa promedio: {stats['avg_reward']:.2f}

Lecciones aprendidas:
{lessons}

Usa estas lecciones para mejorar tu desempeño."""

        return context

    def get_reflection_guide(self, agent_state_desc: str) -> str:
        """
        Genera una guía de reflexión basada en experiencia pasada.
        
        Args:
            agent_state_desc: Descripción textual del estado actual del agente.
        
        Returns:
            Guía de reflexión con recomendaciones.
        """
        context = self.get_context_prompt()
        
        prompt = f"""{context}

Estado actual del agente:
{agent_state_desc}

Basado en tus lecciones aprendidas, ¿qué deberías hacer ahora?
Sé breve y específico."""

        try:
            response = call_ollama_with_retry(
                prompt,
                model=self.model,
                temperature=0.5,
                max_retries=1,
            )
            return response.strip()
        except OllamaError:
            return "Sin recomendación disponible."

    def get_statistics(self) -> Dict[str, Any]:
        """Retorna estadísticas completas."""
        memory_stats = self.memory.get_statistics()
        return {
            **memory_stats,
            "reflections_generated": self.reflection_count,
        }

    def reset(self) -> None:
        """Reinicia la memoria del agente."""
        self.memory = EpisodeMemory()
        self.episode_count = 0
        self.reflection_count = 0
