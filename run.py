#!/usr/bin/env python
"""
Lanzador unificado del simulador.

Ejemplos:
    python run.py
    python run.py --godot
    python run.py --iters 50

    # Múltiples entrenamientos independientes:
    python run.py --runs 10

    # Diez entrenamientos de 50 iteraciones:
    python run.py --runs 10 --iters 50

    # Experimentos con reward shaping:
    python run.py --reward-shaping --runs 10 --iters 50

    # Todos los modos:
    python run.py --all
"""
from __future__ import annotations

import argparse

from ecosystem_simulator.config import (
    NUM_ITERS,
    NUM_RUNS,
    BASE_SEED,
    TAIL_ITERS,
)

from ecosystem_simulator.training.orchestrator import (
    run_experiments,
    run_training,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Lanza uno o varios entrenamientos del "
            "simulador de ecosistemas."
        )
    )

    # ── Modos del entrenamiento ───────────────────────────────
    parser.add_argument(
        "--godot",
        action="store_true",
        help="Visualización en tiempo real en Godot.",
    )

    parser.add_argument(
        "--reward-shaping",
        action="store_true",
        help=(
            "Reward shaping dinámico por LLM "
            "(requiere Ollama)."
        ),
    )

    parser.add_argument(
        "--behavior-selector",
        action="store_true",
        help=(
            "Selector LLM de comportamientos por rol "
            "(requiere Ollama)."
        ),
    )

    parser.add_argument(
        "--reflexion",
        action="store_true",
        help=(
            "Reflexión LLM que alimenta a los otros módulos "
            "(requiere Ollama)."
        ),
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help=(
            "Activa Godot, reward shaping, "
            "behavior selector y reflexion."
        ),
    )

    parser.add_argument(
        "--plot",
        action="store_true",
        help="Muestra los gráficos de análisis al finalizar.",
    )

    # ── Configuración del entrenamiento ───────────────────────
    parser.add_argument(
        "--iters",
        type=int,
        default=NUM_ITERS,
        help=(
            "Cantidad de iteraciones PPO por entrenamiento "
            f"(config.py: {NUM_ITERS})."
        ),
    )

    parser.add_argument(
        "--llm-model",
        default="mistral",
        help=(
            "Modelo de Ollama para los modos LLM "
            "(default: mistral)."
        ),
    )

    parser.add_argument(
        "--shape-interval",
        type=int,
        default=5,
        help=(
            "Cada cuántas iteraciones se consulta el LLM "
            "(default: 5)."
        ),
    )

    # ── Configuración de múltiples ejecuciones ────────────────
    parser.add_argument(
        "--runs",
        type=int,
        default=NUM_RUNS,
        help=(
            "Cantidad de entrenamientos completos "
            f"(config.py: {NUM_RUNS})."
        ),
    )

    parser.add_argument(
        "--base-seed",
        type=int,
        default=BASE_SEED,
        help=(
            "Semilla de la primera ejecución "
            f"(config.py: {BASE_SEED})."
        ),
    )

    parser.add_argument(
        "--tail-iters",
        type=int,
        default=TAIL_ITERS,
        help=(
            "Cantidad de iteraciones finales que se promedian "
            f"por ejecución (config.py: {TAIL_ITERS})."
        ),
    )

    parser.add_argument(
        "--experiment-name",
        default=None,
        help="Nombre opcional para la carpeta del experimento.",
    )

    return parser


def main() -> int:
    args = build_parser().parse_args()

    # ── Validaciones ──────────────────────────────────────────
    if args.runs < 1:
        print("[!] --runs debe ser igual o mayor que 1.")
        return 1

    if args.iters < 1:
        print("[!] --iters debe ser igual o mayor que 1.")
        return 1

    if args.tail_iters < 1:
        print("[!] --tail-iters debe ser igual o mayor que 1.")
        return 1

    if args.shape_interval < 1:
        print("[!] --shape-interval debe ser igual o mayor que 1.")
        return 1

    # ── Construcción de modos ─────────────────────────────────
    modes = set()

    if args.all:
        modes.update({
            "godot",
            "behavior_selector",
            "reward_shaping",
            "reflexion",
        })
    else:
        if args.godot:
            modes.add("godot")

        if args.behavior_selector:
            modes.add("behavior_selector")

        if args.reward_shaping:
            modes.add("reward_shaping")

        if args.reflexion:
            modes.add("reflexion")

    common_kwargs = {
        "num_iters": args.iters,
        "llm_model": args.llm_model,
        "shape_interval": args.shape_interval,
    }

    # ── Una ejecución ─────────────────────────────────────────
    if args.runs == 1:
        run_training(
            modes=modes,
            plot=args.plot,
            seed=args.base_seed,
            **common_kwargs,
        )

        return 0

    # ── Múltiples ejecuciones ─────────────────────────────────
    if args.plot:
        print(
            "[Aviso] --plot no se ejecutará individualmente "
            "durante las múltiples pruebas."
        )

    if "godot" in modes:
        print(
            "[Aviso] Godot se utilizará durante cada una de "
            f"las {args.runs} ejecuciones."
        )

    run_experiments(
        modes=modes,
        num_runs=args.runs,
        base_seed=args.base_seed,
        tail_iters=args.tail_iters,
        experiment_name=args.experiment_name,
        **common_kwargs,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())