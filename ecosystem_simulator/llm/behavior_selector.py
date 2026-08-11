"""
ENFOQUE 1 (real): Behavior Selector por LLM a nivel de ITERACIÓN.

El LLM elige, para cada rol, un PRESET de comportamiento de alto nivel (una
estrategia: "sobrevivir", "evadir", "cazar", "conservar"...). Cada preset es un
perfil de pesos de recompensa. El preset fija los pesos BASE de la iteración;
si además está activo el reward_shaping, éste ajusta finamente sobre ese base.

A diferencia del reward_shaping (que mueve pesos individuales de forma continua),
el selector toma una decisión DISCRETA e interpretable: "esta iteración los
herbívoros priorizan evadir y los depredadores conservan energía".
"""
import logging
import unicodedata
from typing import Dict

from .ollama_utils import (
    call_ollama_with_retry,
    extract_json_from_response,
    OllamaError,
)

logger = logging.getLogger(__name__)

def _norm_preset(value: str) -> str:
    """Normaliza la respuesta del LLM: minúsculas, sin tildes, sin comillas ni ruido."""
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.strip().strip('"\'').lower()
    # Si viene con ruido ("estrategia: evasion"), quedarse con la última palabra.
    return text.split()[-1] if text.split() else ""

# Presets por rol. Cada uno es un perfil de pesos (los no listados quedan en 1.0).
# Los nombres son la "estrategia" que el LLM elige según el estado del ecosistema.
HERBIVORE_PRESETS: Dict[str, Dict[str, float]] = {
    "balanceado": {},
    "supervivencia": {"homeo_herb": 1.4, "drink_herb": 1.3, "eat": 1.3, "explore_herb": 1.2},
    "evasion":       {"escape": 1.6, "explore_herb": 1.3, "homeo_herb": 1.1},
    "forrajeo":      {"eat": 1.4, "drink_herb": 1.4, "explore_herb": 1.4},
}
PREDATOR_PRESETS: Dict[str, Dict[str, float]] = {
    "balanceado": {},
    "caza":        {"hunt": 1.4, "explore_pred": 1.4},
    "conservar":   {"hunt": 0.7, "homeo_pred": 1.3, "drink_pred": 1.2},
    "acecho":      {"explore_pred": 1.5, "hunt": 1.1},
}


class BehaviorSelector:
    """El LLM elige un preset de comportamiento por rol cada iteración."""

    def __init__(self, model: str = "mistral", temperature: float = 0.3):
        self.model = model
        self.temperature = temperature
        self.llm_calls = 0
        # Últimos presets elegidos (para estadística/log).
        self.last_choice = {"herbivore": "balanceado", "predator": "balanceado"}
        # Conteo de cuántas veces se eligió cada preset.
        self.stats = {
            "herbivore": {k: 0 for k in HERBIVORE_PRESETS},
            "predator": {k: 0 for k in PREDATOR_PRESETS},
        }

    def _metrics_summary(self, metrics: Dict[str, float]) -> str:
        def g(k):
            return float(metrics.get(k, 0.0))
        return (
            "HERBÍVOROS: sobreviven={:.0f}%, mueren_sed={:.0f}%, mueren_hambre={:.0f}%, "
            "cazados={:.0f}%\nDEPREDADORES: kills/ep={:.2f}, sobreviven={:.0f}%, "
            "mueren_hambre={:.0f}%, mueren_sed={:.0f}%"
        ).format(
            g("herbivore_reason_timeout_pct"), g("herbivore_reason_dehydration_pct"),
            g("herbivore_reason_starvation_pct"), g("herbivore_reason_predation_pct"),
            g("attacks_kill"), g("predator_reason_timeout_pct"),
            g("predator_reason_starvation_pct"), g("predator_reason_dehydration_pct"),
        )

    def _preset_weights(self, herb_preset: str, pred_preset: str) -> Dict[str, float]:
        """Combina los dos presets elegidos en un dict de pesos completo (base 1.0)."""
        from ..environment.multi_agent_ecosystem import DEFAULT_REWARD_WEIGHTS
        weights = dict(DEFAULT_REWARD_WEIGHTS)  # todo 1.0
        weights.update(HERBIVORE_PRESETS.get(herb_preset, {}))
        weights.update(PREDATOR_PRESETS.get(pred_preset, {}))
        return weights

    def select_presets(
        self,
        metrics: Dict[str, float],
        reflexion_context: str = "",
    ) -> Dict[str, float]:
        """
        Consulta al LLM qué preset usar por rol y devuelve los pesos base
        resultantes. Si el LLM falla, mantiene la última elección.
        """
        summary = self._metrics_summary(metrics)
        herb_opts = ", ".join(HERBIVORE_PRESETS)
        pred_opts = ", ".join(PREDATOR_PRESETS)
        ctx_block = (
            f"\nLecciones del analista (reflexión):\n{reflexion_context}\n"
            if reflexion_context else ""
        )

        prompt = f"""Eres un estratega de un ecosistema con herbívoros y depredadores. Cada
iteración elegís una ESTRATEGIA (preset) para cada rol, buscando que AMBOS sean
viables (que sobrevivan a una tasa razonable, ni exterminio ni inanición).

Estrategias de HERBÍVORO ({herb_opts}):
- balanceado: sin énfasis.
- supervivencia: prioriza comer, beber y mantener recursos.
- evasion: prioriza huir de depredadores y moverse.
- forrajeo: máximo esfuerzo por conseguir comida y agua.

Estrategias de DEPREDADOR ({pred_opts}):
- balanceado: sin énfasis.
- caza: prioriza cazar y perseguir presas.
- conservar: caza menos, ahorra energía (cuando ya está bien alimentado o exterminando).
- acecho: prioriza posicionarse cerca de la presa.

Estado del ecosistema:
{summary}
{ctx_block}
Elige la estrategia que corrija los desequilibrios (ej.: si cazan demasiado a los
herbívoros, el depredador debería 'conservar' y el herbívoro 'evasion'; si el
herbívoro pasa hambre, 'forrajeo').
Responde SOLO con un JSON: {{"herbivore": "<estrategia>", "predator": "<estrategia>"}}"""

        try:
            response = call_ollama_with_retry(
                prompt, model=self.model, temperature=self.temperature, max_retries=1
            )
            self.llm_calls += 1
        except OllamaError as e:
            logger.warning(f"[BehaviorSelector] Error consultando LLM: {e}. Mantengo elección.")
            return self._preset_weights(self.last_choice["herbivore"], self.last_choice["predator"])

        data = extract_json_from_response(response) or {}
        herb = _norm_preset(data.get("herbivore", ""))
        pred = _norm_preset(data.get("predator", ""))
        if herb not in HERBIVORE_PRESETS:
            logger.warning(f"[BehaviorSelector] preset herbívoro inválido: {data.get('herbivore')!r}")
            herb = self.last_choice["herbivore"]
        if pred not in PREDATOR_PRESETS:
            logger.warning(f"[BehaviorSelector] preset depredador inválido: {data.get('predator')!r}")
            pred = self.last_choice["predator"]

        self.last_choice = {"herbivore": herb, "predator": pred}
        self.stats["herbivore"][herb] += 1
        self.stats["predator"][pred] += 1
        logger.info(f"[BehaviorSelector] herbívoro='{herb}', depredador='{pred}'")
        return self._preset_weights(herb, pred)

    def get_statistics(self) -> Dict:
        return {
            "llm_calls": self.llm_calls,
            "last_choice": dict(self.last_choice),
            "distribution": {k: dict(v) for k, v in self.stats.items()},
        }
