"""
Orquestador de entrenamiento COMBINABLE.

Un único loop de entrenamiento PPO sobre el que se APILAN modos opcionales.
A diferencia del diseño anterior (varios scripts que corrían entrenamientos
separados por subprocess), acá todo comparte el mismo trainer y el mismo
monitor.csv, de modo que los modos se pueden combinar y comparar.

Modos disponibles:
  - base           : entrenamiento PPO (siempre activo).
  - godot          : transmite un episodio de visualización real a Godot.
  - reward_shaping : el LLM ajusta los pesos de recompensa entre iteraciones.

Orden por iteración (los modos LLM se enchufan en `_pre_train`/`_post_train`):
  behavior_selector (Fase 3) -> reward_shaping (Fase 2) -> train() -> godot ->
  reflexion (Fase 4).
"""
import csv
import time
from pathlib import Path

import ray

from ..config import ENV_CFG, NUM_RUNNERS, NUM_ITERS, VIS_START_IT
from .callbacks import PerAgentAndReasonMetrics
from .trainer import build_config, get_monitor_header, build_monitor_row
from .godot_hook import GodotStreamer

MONITOR_PATH = Path(__file__).resolve().parents[2] / "monitor.csv"
SHAPING_LOG_PATH = Path(__file__).resolve().parents[2] / "monitor_shaping.csv"
REFLEXION_LOG_PATH = Path(__file__).resolve().parents[2] / "monitor_reflexion.log"

# Modos que requieren el pipeline LLM (y por tanto Ollama).
_LLM_MODES = {"reward_shaping", "behavior_selector", "reflexion"}

# Métricas agregadas que el pipeline LLM necesita de cada iteración.
_METRIC_KEYS = [
    "herbivore_eat_count", "herbivore_drink_count",
    "herbivore_reason_timeout_pct", "herbivore_reason_dehydration_pct",
    "herbivore_reason_starvation_pct", "herbivore_reason_predation_pct",
    "attacks_kill", "predator_reason_timeout_pct",
    "predator_reason_starvation_pct", "predator_reason_dehydration_pct",
]


def _force_cpu_resources(config):
    """Fuerza 0 GPUs de forma tolerante a distintas versiones de RLlib."""
    try:
        return config.resources(
            num_gpus=0,
            num_gpus_per_learner=0,
            num_gpus_per_env_runner=0,
        )
    except TypeError:
        try:
            return config.resources(num_gpus=0)
        except TypeError:
            return config


def _extract_metrics(result: dict) -> dict:
    """Saca del resultado de train() las métricas que consume el pipeline LLM."""
    ev = result.get("env_runners", {}) or {}
    cm = ev.get("custom_metrics", {}) or result.get("custom_metrics", {}) or {}

    def m(key):
        return float(cm.get(f"{key}_mean", cm.get(key, 0.0)))

    return {key: m(key) for key in _METRIC_KEYS}


class TrainingOrchestrator:
    """Corre un entrenamiento PPO con los modos pedidos apilados encima."""

    def __init__(
        self,
        modes,
        env_cfg: dict | None = None,
        num_runners: int = NUM_RUNNERS,
        num_iters: int = NUM_ITERS,
        vis_start_it: int = VIS_START_IT,
        plot: bool = False,
        llm_model: str = "mistral",
        shape_interval: int = 5,
    ):
        self.modes = set(modes)
        # Copia local para no mutar el ENV_CFG global importado.
        self.env_cfg = dict(env_cfg if env_cfg is not None else ENV_CFG)
        self.num_runners = num_runners
        self.num_iters = num_iters
        self.vis_start_it = vis_start_it
        self.plot = plot
        # Cada cuántas iteraciones consulta el LLM (para no llamarlo por iteración).
        self.shape_interval = max(1, shape_interval)

        self.godot = GodotStreamer() if "godot" in self.modes else None

        # Pipeline LLM (Fase 2+): solo se crea si hay algún modo LLM activo.
        self.pipeline = None
        if self.modes & _LLM_MODES:
            from ..llm.pipeline import LLMPipeline
            self.pipeline = LLMPipeline(self.modes, model=llm_model)

        self._trainer = None
        self._shaping_log = None    # writer del csv de trazas de pesos
        self._reflexion_log = None  # file de lecciones de reflexión

    # ── Hooks de extensión ─────────────────────────────────────────────────
    def _pre_train(self, iteration: int, prev_result: dict | None) -> None:
        """
        Antes de train(): el pipeline LLM (behavior selector + reward shaping)
        propone pesos de recompensa a partir de las métricas de la iteración
        anterior y se empujan a todos los env runners. Solo cada `shape_interval`
        iteraciones (los pesos persisten entre llamadas).
        """
        if self.pipeline is None or prev_result is None:
            return
        if iteration % self.shape_interval != 0:
            return

        metrics = _extract_metrics(prev_result)
        weights = self.pipeline.pre_train(metrics)
        if weights:
            self._push_weights(weights)
            self._log_weights(iteration, weights)

    def _post_train(self, iteration: int, result: dict) -> None:
        """
        Después de train() y del envío a Godot: Fase 4 (reflexion). Genera una
        lección de la iteración y actualiza el contexto que usarán selector y
        shaper en las siguientes. Se llama a la misma cadencia que el shaping.
        """
        if self.pipeline is None or self.pipeline.reflexion is None:
            return
        if iteration % self.shape_interval != 0:
            return
        lesson = self.pipeline.post_train(_extract_metrics(result))
        if lesson:
            print(f"    [reflexión] iter {iteration}: {lesson}")
            if self._reflexion_log is not None:
                self._reflexion_log.write(f"{iteration}\t{lesson}\n")
                self._reflexion_log.flush()

    def _push_weights(self, weights: dict) -> None:
        """Empuja los pesos a la instancia real del entorno en cada env runner."""
        self._trainer.env_runner_group.foreach_env(
            lambda env: env.par_env.set_reward_weights(weights)
        )

    def _log_weights(self, iteration: int, weights: dict) -> None:
        # Si el behavior selector está activo, mostrar el preset elegido.
        sel = getattr(self.pipeline, "selector", None)
        if sel is not None:
            c = sel.last_choice
            print(f"    [preset] iter {iteration}: herbívoro='{c['herbivore']}', depredador='{c['predator']}'")
        keys = sorted(weights)
        line = "  ".join(f"{k}={weights[k]:.2f}" for k in keys)
        print(f"    [pesos]  iter {iteration}: {line}")
        if self._shaping_log is not None:
            self._shaping_log.writerow([iteration] + [weights[k] for k in keys])
            self._shaping_log_file.flush()

    # ── Loop principal ─────────────────────────────────────────────────────
    def run(self) -> None:
        active = ", ".join(sorted(self.modes))
        print(f"[Orquestador] Modos activos: base{', ' + active if active else ''}")

        # Ollama es requerido si hay modos LLM (falla temprano y claro).
        if self.pipeline is not None:
            self.pipeline.require_ollama()

        ray.init(ignore_reinit_error=True)
        if self.godot:
            self.godot.start()

        config = build_config(
            self.env_cfg,
            num_runners=self.num_runners,
            callbacks_class=PerAgentAndReasonMetrics,
        )
        config = _force_cpu_resources(config)
        trainer = config.build()
        self._trainer = trainer

        n_agents = self.env_cfg["n_agents"]
        prev_result = None

        # Log de pesos: se crea si algún módulo que PRODUCE pesos está activo
        # (behavior selector o reward shaping).
        produces_weights = self.pipeline is not None and (
            self.pipeline.selector is not None or self.pipeline.shaper is not None
        )
        shaping_file = None
        if produces_weights:
            shaping_file = open(SHAPING_LOG_PATH, "w", newline="")
            self._shaping_log_file = shaping_file
            self._shaping_log = csv.writer(shaping_file)
            from ..llm.reward_shaping import WEIGHT_KEYS
            self._shaping_log.writerow(["iter"] + sorted(WEIGHT_KEYS))

        # Log de lecciones de reflexión (texto, una por línea: iter<TAB>lección).
        if self.pipeline is not None and self.pipeline.reflexion is not None:
            self._reflexion_log = open(REFLEXION_LOG_PATH, "w", encoding="utf-8")

        try:
            with open(MONITOR_PATH, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(get_monitor_header(n_agents))

                for i in range(self.num_iters):
                    t0 = time.time()

                    self._pre_train(i, prev_result)

                    result = trainer.train()

                    row, r_mean, l_mean = build_monitor_row(result, i, n_agents)
                    writer.writerow(row)
                    f.flush()

                    print(
                        f"Iter {i:>3}: return={r_mean:.2f}, len={l_mean:.2f}, "
                        f"time={time.time() - t0:.1f}s"
                    )

                    if self.godot and i >= self.vis_start_it:
                        self.godot.stream_episode(trainer, self.env_cfg, i)

                    self._post_train(i, result)
                    prev_result = result
        finally:
            if shaping_file is not None:
                shaping_file.close()
            if self._reflexion_log is not None:
                self._reflexion_log.close()
            ray.shutdown()

        print("\n[+] Entrenamiento completado. Monitor en:", MONITOR_PATH)

        if self.plot:
            from ..utils.visualization import analyze_training_results
            analyze_training_results(str(MONITOR_PATH))


def run_training(modes, **kwargs) -> None:
    """Atajo funcional: construye el orquestador y lo corre."""
    TrainingOrchestrator(modes, **kwargs).run()
