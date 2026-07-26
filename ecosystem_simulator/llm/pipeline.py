"""
Pipeline LLM: orquesta los módulos (behavior selector, reward shaping, reflexion)
en el orden fijo acordado, compartiendo el contexto de reflexión.

Orden por iteración:
  1. behavior_selector  (Fase 3) -> fija un preset de pesos según el estado.
  2. reward_shaping     (Fase 2) -> ajusta finos sobre ese preset.
     -> los pesos resultantes se empujan al entorno ANTES de train().
  3. reflexion          (Fase 4) -> tras la iteración, analiza resultados y
     genera lecciones que alimentan a (1) y (2) en la siguiente iteración.

En la Fase 2 solo `reward_shaping` está cableado; los otros dos son huecos que
se llenan en sus fases. `pre_train` devuelve los pesos a aplicar (o None).
"""
import logging
from typing import Dict, Optional

from .ollama_utils import check_ollama_available, OllamaError
from .reward_shaping import RewardShaper
from .behavior_selector import BehaviorSelector
from .reflexion import ReflectionAgent

logger = logging.getLogger(__name__)


class LLMPipeline:
    """Coordina los módulos LLM activos y produce los pesos de recompensa."""

    def __init__(self, modes, model: str = "mistral"):
        self.modes = set(modes)
        self.model = model

        # Fase 3: behavior selector (fija el preset base, ANTES del shaping).
        self.selector = (
            BehaviorSelector(model=model) if "behavior_selector" in self.modes else None
        )
        # Fase 2: reward shaping (ajusta fino, sobre el preset si existe).
        self.shaper = RewardShaper(model=model) if "reward_shaping" in self.modes else None
        # Fase 4: reflexion (analiza tras la iteración, alimenta a los otros dos).
        self.reflexion = ReflectionAgent(model=model) if "reflexion" in self.modes else None

        # Contexto que la reflexión (Fase 4) inyecta en selector/shaper.
        self._reflexion_context = ""

    # ── Requisito de Ollama (sin fallback) ─────────────────────────────────
    def require_ollama(self) -> None:
        """Lanza si algún módulo LLM está activo y Ollama no está disponible."""
        if not self.active:
            return
        if not check_ollama_available():
            raise OllamaError(
                "Ollama no está disponible en http://localhost:11434.\n"
                "Los modos LLM lo requieren. Instálalo (https://ollama.ai) e inicia "
                f"el modelo con:  ollama run {self.model}"
            )

    @property
    def active(self) -> bool:
        return any([self.shaper, self.selector, self.reflexion])

    # ── Hook PRE-entrenamiento ─────────────────────────────────────────────
    def pre_train(self, metrics: Dict[str, float]) -> Optional[Dict[str, float]]:
        """
        Produce los pesos de recompensa para la próxima iteración a partir de
        las métricas de la iteración anterior. Devuelve None si no hay nada que
        aplicar (ningún módulo que afecte pesos activo).
        """
        weights: Optional[Dict[str, float]] = None

        # Fase 3: behavior_selector fija el preset base de la iteración.
        if self.selector is not None:
            weights = self.selector.select_presets(metrics, self._reflexion_context)

        # Fase 2: reward_shaping ajusta fino (sobre el preset si existe).
        if self.shaper is not None:
            weights = self.shaper.propose_weights(
                metrics, self._reflexion_context, base_weights=weights
            )

        return weights

    # ── Hook POST-entrenamiento ────────────────────────────────────────────
    def post_train(self, metrics: Dict[str, float]) -> Optional[str]:
        """
        Fase 4: reflexion analiza la iteración, genera una lección y actualiza el
        contexto que usarán selector y shaper en las siguientes iteraciones.
        Devuelve la lección (o None) para que el orquestador la muestre/loguee.
        """
        if self.reflexion is None:
            return None
        lesson = self.reflexion.reflect(metrics)
        self._reflexion_context = self.reflexion.context()
        return lesson

    def statistics(self) -> Dict:
        stats = {}
        if self.selector is not None:
            stats["behavior_selector"] = self.selector.get_statistics()
        if self.shaper is not None:
            stats["reward_shaping"] = self.shaper.get_statistics()
        if self.reflexion is not None:
            stats["reflexion"] = self.reflexion.get_statistics()
        return stats
