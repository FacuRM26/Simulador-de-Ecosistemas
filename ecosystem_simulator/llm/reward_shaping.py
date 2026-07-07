"""
ENFOQUE 2 (real): Reward Shaping por LLM a nivel de ITERACIÓN.

El LLM no se llama por paso (inviable con Ollama: miles de pasos por iteración
× segundos por llamada). En cambio, ENTRE iteraciones el LLM observa métricas
agregadas del ecosistema y ajusta los PESOS de recompensa que el entorno aplica
a TODOS los pasos de la siguiente iteración. Así modula de verdad el
entrenamiento con ~1 llamada por iteración (o cada N).

Los pesos multiplican términos de la recompensa (ver DEFAULT_REWARD_WEIGHTS en
el entorno). 1.0 = neutro. El entorno acota cada peso a un rango seguro.
"""
import logging
from typing import Dict

from .ollama_utils import (
    call_ollama_with_retry,
    extract_json_from_response,
    OllamaError,
)

logger = logging.getLogger(__name__)

# Claves de peso que el LLM puede ajustar (deben coincidir con el entorno).
# Separadas por rol: así se puede ayudar a un rol sin perjudicar al otro.
WEIGHT_KEYS = [
    "eat", "escape", "drink_herb", "homeo_herb", "explore_herb",
    "hunt", "drink_pred", "homeo_pred", "explore_pred",
]

_WEIGHT_MEANING = """HERBÍVORO:
- eat: incentivo a comer vegetación
- escape: incentivo a huir de los depredadores
- drink_herb: incentivo del herbívoro a beber
- homeo_herb: mantener recursos altos del herbívoro (supervivencia)
- explore_herb: que el herbívoro se acerque a recursos con propósito
DEPREDADOR:
- hunt: incentivo a golpear/matar presas
- drink_pred: incentivo del depredador a beber
- homeo_pred: mantener recursos altos del depredador (supervivencia)
- explore_pred: que el depredador se acerque a presa/agua con propósito"""


class RewardShaper:
    """Propone ajustes de pesos de recompensa consultando a un LLM (Ollama)."""

    def __init__(
        self,
        model: str = "mistral",
        temperature: float = 0.4,
        weight_min: float = 0.5,
        weight_max: float = 2.0,
    ):
        self.model = model
        self.temperature = temperature
        self.weight_min = weight_min
        self.weight_max = weight_max
        self.llm_calls = 0
        # Estado: los últimos pesos propuestos (arranca neutro).
        self.last_weights: Dict[str, float] = {k: 1.0 for k in WEIGHT_KEYS}

    def _metrics_summary(self, metrics: Dict[str, float]) -> str:
        def g(k):
            return float(metrics.get(k, 0.0))
        return (
            "HERBÍVOROS: comer/ep={:.1f}, beber/ep={:.1f}, sobreviven={:.0f}%, "
            "mueren_sed={:.0f}%, mueren_hambre={:.0f}%, cazados={:.0f}%\n"
            "DEPREDADORES: kills/ep={:.2f}, sobreviven={:.0f}%, "
            "mueren_hambre={:.0f}%, mueren_sed={:.0f}%"
        ).format(
            g("herbivore_eat_count"), g("herbivore_drink_count"),
            g("herbivore_reason_timeout_pct"), g("herbivore_reason_dehydration_pct"),
            g("herbivore_reason_starvation_pct"), g("herbivore_reason_predation_pct"),
            g("attacks_kill"), g("predator_reason_timeout_pct"),
            g("predator_reason_starvation_pct"), g("predator_reason_dehydration_pct"),
        )

    def _clamp(self, value, fallback: float) -> float:
        try:
            return float(max(self.weight_min, min(self.weight_max, float(value))))
        except (TypeError, ValueError):
            return fallback

    def propose_weights(
        self,
        metrics: Dict[str, float],
        reflexion_context: str = "",
        base_weights: Dict[str, float] | None = None,
    ) -> Dict[str, float]:
        """
        Consulta al LLM y devuelve los nuevos pesos. Si el LLM falla o no
        devuelve JSON válido, mantiene los pesos base (no rompe el training).

        base_weights: si viene (p.ej. el preset del Behavior Selector), el shaper
        AJUSTA sobre ese base en vez de sobre sus últimos pesos. Así el orden
        selector -> shaping se respeta: primero el preset, luego el ajuste fino.
        """
        start = dict(base_weights) if base_weights is not None else dict(self.last_weights)
        summary = self._metrics_summary(metrics)
        current = ", ".join(f"{k}={v:.2f}" for k, v in start.items())
        ctx_block = (
            f"\nLecciones del analista (reflexión de iteraciones previas):\n{reflexion_context}\n"
            if reflexion_context else ""
        )

        prompt = f"""Eres un experto en aprendizaje por refuerzo ajustando la función de recompensa
de un ecosistema con herbívoros y depredadores.

OBJETIVO CENTRAL: que AMBOS roles sean VIABLES. La meta es que herbívoros Y depredadores sobrevivan a una
tasa razonable (idealmente 25-50% cada uno). Un rol con supervivencia < 20% o
> 75% está desequilibrado y hay que corregirlo.

REGLAS IMPORTANTES:
- Los pesos son POR ROL. Ayuda al rol que está peor SIN perjudicar al otro: por
  ejemplo, si el herbívoro muere de sed sube 'drink_herb' (NO 'drink_pred').
- Si un rol pasa hambre, súbele su capacidad de conseguir comida (herbívoro:
  'eat'/'explore_herb'; depredador: 'hunt'/'explore_pred'). NO bajes 'hunt' si
  los depredadores ya pasan hambre.
- Evita llevar un peso a los extremos ({self.weight_min} o {self.weight_max}) salvo que sea claramente necesario.
- Cambios GRADUALES: no más de ~0.2 por iteración.

Pesos actuales (multiplican cada término; rango {self.weight_min}-{self.weight_max}; 1.0 = neutro):
{current}

Significado de cada peso:
{_WEIGHT_MEANING}

Estado del ecosistema en la última iteración:
{summary}
{ctx_block}
Responde SOLO con un objeto JSON con las 9 claves y valores numéricos, sin texto.
Ejemplo: {{"eat": 1.1, "escape": 1.2, "drink_herb": 1.3, "homeo_herb": 1.1, "explore_herb": 1.1, "hunt": 1.2, "drink_pred": 1.0, "homeo_pred": 1.0, "explore_pred": 1.1}}"""

        try:
            response = call_ollama_with_retry(
                prompt, model=self.model, temperature=self.temperature, max_retries=1
            )
            self.llm_calls += 1
        except OllamaError as e:
            logger.warning(f"[RewardShaper] Error consultando LLM: {e}. Mantengo pesos.")
            return dict(start)

        data = extract_json_from_response(response)
        if not data:
            logger.warning("[RewardShaper] El LLM no devolvió JSON válido. Mantengo pesos.")
            return dict(start)

        new_weights = {
            k: self._clamp(data.get(k, start[k]), start[k])
            for k in WEIGHT_KEYS
        }
        self.last_weights = new_weights
        return dict(new_weights)

    def get_statistics(self) -> Dict:
        return {"llm_calls": self.llm_calls, "weights": dict(self.last_weights)}
