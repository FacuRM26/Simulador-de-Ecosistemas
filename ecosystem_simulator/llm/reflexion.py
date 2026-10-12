"""
ENFOQUE 3 (real): Reflexion por LLM a nivel de ITERACIÓN.

Tras cada iteración (todos sus episodios), el LLM analiza las métricas agregadas
del ecosistema y escribe una LECCIÓN en lenguaje natural: cuál es el desequilibrio
más importante ahora y qué estrategia debería corregirlo. Las lecciones se
ACUMULAN y se inyectan como contexto en los prompts del behavior_selector y el
reward_shaper en las siguientes iteraciones.

Por sí sola, la reflexión NO cambia el entrenamiento (solo produce análisis
interpretable). Su valor aparece COMBINADA: guía las decisiones de los otros dos
módulos, dándoles memoria de lo que ya pasó.
"""
import logging
from collections import deque
from typing import Dict, Optional

from .ollama_utils import call_ollama_with_retry, OllamaError

logger = logging.getLogger(__name__)


class ReflectionAgent:
    """Genera y acumula lecciones en lenguaje natural a partir de las métricas."""

    def __init__(self, model: str = "mistral", temperature: float = 0.6, max_lessons: int = 3):
        self.model = model
        self.temperature = temperature
        self.lessons = deque(maxlen=max_lessons)
        self.llm_calls = 0
        self._prev: Optional[Dict[str, float]] = None  # métricas de la reflexión previa

    def _summary(self, metrics: Dict[str, float], prev: Optional[Dict[str, float]]) -> str:
        def g(m, k):
            return float(m.get(k, 0.0))

        def delta(k):
            if not prev:
                return ""
            d = g(metrics, k) - g(prev, k)
            return f" ({'+' if d >= 0 else ''}{d:.0f})"

        return (
            "HERBÍVOROS: sobreviven={:.0f}%{}, cazados={:.0f}%{}, mueren_sed={:.0f}%, mueren_hambre={:.0f}%\n"
            "DEPREDADORES: kills/ep={:.2f}, sobreviven={:.0f}%{}, mueren_hambre={:.0f}%"
        ).format(
            g(metrics, "herbivore_reason_timeout_pct"), delta("herbivore_reason_timeout_pct"),
            g(metrics, "herbivore_reason_predation_pct"), delta("herbivore_reason_predation_pct"),
            g(metrics, "herbivore_reason_dehydration_pct"), g(metrics, "herbivore_reason_starvation_pct"),
            g(metrics, "attacks_kill"),
            g(metrics, "predator_reason_timeout_pct"), delta("predator_reason_timeout_pct"),
            g(metrics, "predator_reason_starvation_pct"),
        )

    def reflect(self, metrics: Dict[str, float]) -> Optional[str]:
        """Analiza las métricas y genera/acumula una lección. Devuelve la lección."""
        summary = self._summary(metrics, self._prev)
        prev_lessons = self.context() or "(ninguna aún)"

        prompt = f"""Eres un analista de un ecosistema con aprendizaje por refuerzo (herbívoros y
depredadores). Meta: que AMBOS roles sean viables (ni exterminio de las presas,
ni inanición masiva de ningún rol).

Lecciones previas:
{prev_lessons}

Estado actual (entre paréntesis, el cambio desde la última revisión):
{summary}

Escribe UNA lección concisa (una sola oración) sobre el desequilibrio MÁS
importante ahora y qué estrategia debería corregirlo en las próximas iteraciones.
Sé concreto y accionable: menciona el rol y si conviene cazar/huir/comer/beber
más o menos. Responde solo con la oración, sin preámbulo."""

        try:
            response = call_ollama_with_retry(
                prompt, model=self.model, temperature=self.temperature, max_retries=1
            )
            self.llm_calls += 1
        except OllamaError as e:
            logger.warning(f"[Reflexion] Error consultando LLM: {e}.")
            self._prev = dict(metrics)
            return None

        lesson = " ".join(response.strip().split())[:300]
        if lesson:
            self.lessons.append(lesson)
        self._prev = dict(metrics)
        return lesson or None

    def context(self) -> str:
        """Lecciones recientes como bloque de texto para inyectar en otros prompts."""
        return "\n".join(f"- {l}" for l in self.lessons)

    def get_statistics(self) -> Dict:
        return {"llm_calls": self.llm_calls, "lessons": list(self.lessons)}
