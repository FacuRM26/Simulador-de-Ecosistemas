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
import json
import statistics
from collections import defaultdict
from datetime import datetime
from ..config import ENV_CFG, NUM_RUNNERS, NUM_ITERS, VIS_START_IT
from .callbacks import PerAgentAndReasonMetrics
from .trainer import build_config, get_monitor_header, build_monitor_row
from .godot_hook import GodotStreamer

PROJECT_ROOT = Path(__file__).resolve().parents[2]

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
        output_dir: str | Path | None = None,
        seed: int | None = None,
        use_lstm: bool = False,
    ):
        self.modes = set(modes)
        # Copia local para no mutar el ENV_CFG global importado.
        self.env_cfg = dict(env_cfg if env_cfg is not None else ENV_CFG)
        self.num_runners = num_runners
        self.num_iters = num_iters
        self.vis_start_it = vis_start_it
        self.plot = plot
        self.seed = seed
        self.use_lstm = use_lstm

        self.output_dir = (
            Path(output_dir)
            if output_dir is not None
            else PROJECT_ROOT
        )

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.monitor_path = self.output_dir / "monitor.csv"
        self.shaping_log_path = self.output_dir / "monitor_shaping.csv"
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
    def run(self) -> Path:
        active = ", ".join(sorted(self.modes))
        print(
            f"[Orquestador] Modos activos: "
            f"base{', ' + active if active else ''}"
        )
        print(f"[Orquestador] Semilla: {self.seed}")
        print(f"[Orquestador] Salida: {self.output_dir}")

        if self.pipeline is not None:
            self.pipeline.require_ollama()

        trainer = None
        shaping_file = None

        # Log de lecciones de reflexión (texto, una por línea: iter<TAB>lección).
        if self.pipeline is not None and self.pipeline.reflexion is not None:
            self._reflexion_log = open(REFLEXION_LOG_PATH, "w", encoding="utf-8")

        try:
            ray.init(ignore_reinit_error=True)

            if self.godot:
                self.godot.start()

            config = build_config(
                self.env_cfg,
                num_runners=self.num_runners,
                callbacks_class=PerAgentAndReasonMetrics,
                use_lstm=self.use_lstm,
                seed=self.seed,
            )

            config = _force_cpu_resources(config)

            trainer = config.build()
            self._trainer = trainer

            n_agents = self.env_cfg["n_agents"]
            prev_result = None

            produces_weights = self.pipeline is not None and (
                self.pipeline.selector is not None
                or self.pipeline.shaper is not None
            )

            if produces_weights:
                shaping_file = open(
                    self.shaping_log_path,
                    "w",
                    newline="",
                    encoding="utf-8",
                )

                self._shaping_log_file = shaping_file
                self._shaping_log = csv.writer(shaping_file)

                from ..llm.reward_shaping import WEIGHT_KEYS

                self._shaping_log.writerow(
                    ["iter"] + sorted(WEIGHT_KEYS)
                )

            with open(
                self.monitor_path,
                "w",
                newline="",
                encoding="utf-8",
            ) as monitor_file:

                writer = csv.writer(monitor_file)
                writer.writerow(get_monitor_header(n_agents))

                for i in range(self.num_iters):
                    t0 = time.time()

                    self._pre_train(i, prev_result)

                    result = trainer.train()

                    row, r_mean, l_mean = build_monitor_row(
                        result=result,
                        iteration=i,
                        n_agents=n_agents,
                    )

                    writer.writerow(row)
                    monitor_file.flush()

                    print(
                        f"Iter {i:>3}: "
                        f"return={r_mean:.2f}, "
                        f"len={l_mean:.2f}, "
                        f"time={time.time() - t0:.1f}s"
                    )

                    if self.godot and i >= self.vis_start_it:
                        self.godot.stream_episode(
                            trainer,
                            self.env_cfg,
                            i,
                        )

                    self._post_train(i, result)
                    prev_result = result

        finally:
            if shaping_file is not None:
                shaping_file.close()

            if self._reflexion_log is not None:
                self._reflexion_log.close()

            if trainer is not None:
                trainer.stop()

            self._trainer = None

            if self.godot is not None and hasattr(self.godot, "stop"):
                self.godot.stop()

            ray.shutdown()

        print(
            "\n[+] Entrenamiento completado. Monitor en:",
            self.monitor_path,
        )

        if self.plot:
            from ..utils.visualization import analyze_training_results

            analyze_training_results(str(self.monitor_path))

        return self.monitor_path


def run_training(modes, **kwargs) -> Path:
    return TrainingOrchestrator(modes, **kwargs).run()
def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _read_monitor(monitor_path: Path) -> list[dict]:
    with open(
        monitor_path,
        "r",
        newline="",
        encoding="utf-8",
    ) as monitor_file:
        return list(csv.DictReader(monitor_file))


def _summarize_monitor(
    monitor_path: Path,
    tail_iters: int = 1,
) -> dict[str, float]:
    """
    Resume una ejecución.

    tail_iters=1:
        usa la última fila del monitor.

    tail_iters=10:
        promedia las últimas diez iteraciones.
    """
    rows = _read_monitor(monitor_path)

    if not rows:
        raise ValueError(
            f"El monitor no contiene resultados: {monitor_path}"
        )

    selected_rows = rows[-max(1, tail_iters):]
    summary = {}

    for column in rows[0]:
        if column == "iter":
            continue

        values = []

        for row in selected_rows:
            number = _to_float(row.get(column))

            if number is not None:
                values.append(number)

        if values:
            summary[column] = statistics.mean(values)

    return summary


def _write_runs_summary(
    experiment_dir: Path,
    completed_runs: list[dict],
) -> None:
    metric_names = sorted({
        metric
        for run in completed_runs
        for metric in run["metrics"]
    })

    output_path = experiment_dir / "runs_summary.csv"

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as output_file:

        fieldnames = [
            "run",
            "seed",
            *metric_names,
        ]

        writer = csv.DictWriter(
            output_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for run in completed_runs:
            row = {
                "run": run["run"],
                "seed": run["seed"],
            }

            row.update(run["metrics"])
            writer.writerow(row)


def _write_final_summary(
    experiment_dir: Path,
    completed_runs: list[dict],
) -> None:
    """
    Calcula promedio y desviación estándar entre ejecuciones.
    """
    metric_names = sorted({
        metric
        for run in completed_runs
        for metric in run["metrics"]
    })

    output_path = experiment_dir / "final_summary.csv"

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as output_file:

        fieldnames = [
            "metric",
            "mean",
            "standard_deviation",
            "minimum",
            "maximum",
            "number_of_runs",
        ]

        writer = csv.DictWriter(
            output_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for metric in metric_names:
            values = [
                run["metrics"][metric]
                for run in completed_runs
                if metric in run["metrics"]
            ]

            if not values:
                continue

            writer.writerow({
                "metric": metric,
                "mean": statistics.mean(values),
                "standard_deviation": (
                    statistics.stdev(values)
                    if len(values) > 1
                    else 0.0
                ),
                "minimum": min(values),
                "maximum": max(values),
                "number_of_runs": len(values),
            })


def _write_learning_curve_summary(
    experiment_dir: Path,
    completed_runs: list[dict],
) -> None:
    """
    Genera el promedio por iteración entre todas las ejecuciones.

    Esto permite crear una gráfica:
        iteración vs promedio
    junto con:
        promedio ± desviación estándar
    """
    grouped_values = defaultdict(list)

    for run in completed_runs:
        rows = _read_monitor(run["monitor_path"])

        for row in rows:
            iteration_value = _to_float(row.get("iter"))

            if iteration_value is None:
                continue

            iteration = int(iteration_value)

            for metric, value in row.items():
                if metric == "iter":
                    continue

                number = _to_float(value)

                if number is not None:
                    grouped_values[(iteration, metric)].append(number)

    output_path = experiment_dir / "learning_curve_summary.csv"

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as output_file:

        fieldnames = [
            "iter",
            "metric",
            "mean",
            "standard_deviation",
            "minimum",
            "maximum",
            "number_of_runs",
        ]

        writer = csv.DictWriter(
            output_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for iteration, metric in sorted(grouped_values):
            values = grouped_values[(iteration, metric)]

            writer.writerow({
                "iter": iteration,
                "metric": metric,
                "mean": statistics.mean(values),
                "standard_deviation": (
                    statistics.stdev(values)
                    if len(values) > 1
                    else 0.0
                ),
                "minimum": min(values),
                "maximum": max(values),
                "number_of_runs": len(values),
            })


def run_experiments(
    modes,
    num_runs: int = 10,
    base_seed: int = 1001,
    tail_iters: int = 1,
    experiment_name: str | None = None,
    **orchestrator_kwargs,
) -> Path:
    """
    Ejecuta múltiples entrenamientos completos de manera secuencial.

    Cada entrenamiento:
      - crea un trainer nuevo;
      - utiliza una semilla diferente;
      - guarda su propio monitor.csv;
      - libera Ray antes de iniciar la siguiente prueba.

    Al final genera:
      - runs_summary.csv;
      - final_summary.csv;
      - learning_curve_summary.csv.
    """
    if num_runs < 1:
        raise ValueError("num_runs debe ser al menos 1.")

    modes = set(modes)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    folder_name = (
        experiment_name
        if experiment_name
        else f"experiment_{timestamp}"
    )

    experiment_dir = (
        PROJECT_ROOT
        / "experiment_results"
        / folder_name
    )

    experiment_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    clean_kwargs = dict(orchestrator_kwargs)

    # Estos valores son controlados por esta función.
    clean_kwargs.pop("output_dir", None)
    clean_kwargs.pop("seed", None)
    clean_kwargs.pop("plot", None)

    experiment_metadata = {
        "modes": sorted(modes),
        "num_runs": num_runs,
        "base_seed": base_seed,
        "seeds": [
            base_seed + index
            for index in range(num_runs)
        ],
        "tail_iters": tail_iters,
        "orchestrator_arguments": clean_kwargs,
    }

    with open(
        experiment_dir / "experiment_config.json",
        "w",
        encoding="utf-8",
    ) as config_file:
        json.dump(
            experiment_metadata,
            config_file,
            indent=4,
            ensure_ascii=False,
            default=str,
        )

    completed_runs = []

    for run_index in range(1, num_runs + 1):
        seed = base_seed + run_index - 1

        run_dir = (
            experiment_dir
            / f"run_{run_index:02d}_seed_{seed}"
        )

        run_dir.mkdir(
            parents=True,
            exist_ok=False,
        )

        print(
            "\n"
            + "=" * 70
            + f"\nPRUEBA {run_index}/{num_runs}"
            + f"\nSemilla: {seed}"
            + f"\nCarpeta: {run_dir}"
            + "\n"
            + "=" * 70
        )

        with open(
            run_dir / "run_config.json",
            "w",
            encoding="utf-8",
        ) as run_config_file:
            json.dump(
                {
                    "run": run_index,
                    "seed": seed,
                    "modes": sorted(modes),
                    "arguments": clean_kwargs,
                },
                run_config_file,
                indent=4,
                ensure_ascii=False,
                default=str,
            )

        orchestrator = TrainingOrchestrator(
            modes=modes,
            output_dir=run_dir,
            seed=seed,
            plot=False,
            **clean_kwargs,
        )

        monitor_path = orchestrator.run()

        metrics = _summarize_monitor(
            monitor_path=monitor_path,
            tail_iters=tail_iters,
        )

        completed_runs.append({
            "run": run_index,
            "seed": seed,
            "monitor_path": monitor_path,
            "metrics": metrics,
        })

        # Se actualizan después de cada ejecución.
        # Si una prueba posterior falla, las anteriores no se pierden.
        _write_runs_summary(
            experiment_dir,
            completed_runs,
        )

        _write_final_summary(
            experiment_dir,
            completed_runs,
        )

        _write_learning_curve_summary(
            experiment_dir,
            completed_runs,
        )

    print("\n" + "=" * 70)
    print("[+] Todas las pruebas finalizaron.")
    print("[+] Resultados:", experiment_dir)
    print("[+] Resumen por ejecución: runs_summary.csv")
    print("[+] Resumen final: final_summary.csv")
    print("[+] Curva promedio: learning_curve_summary.csv")
    print("=" * 70)

    return experiment_dir