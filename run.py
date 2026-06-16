#!/usr/bin/env python
"""
Lanzador unificado del proyecto.

Este archivo no cambia la logica interna de los modos existentes: solo
orquesta los entrypoints ya creados.

Ejemplos:
    python run.py
    python run.py --godot
    python run.py --reward-shaping --reflexion
    python run.py --all
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent

MODES = {
    "base": [sys.executable, "-m", "ecosystem_simulator.training.trainer"],
    "godot": [sys.executable, str(ROOT / "run_with_godot.py")],
    "behavior-selector": [sys.executable, str(ROOT / "run_with_llm_behavior_selector.py")],
    "reward-shaping": [sys.executable, str(ROOT / "run_with_llm_reward_shaping.py")],
    "reflexion": [sys.executable, str(ROOT / "run_with_llm_reflexion.py")],
    "examples-llm": [sys.executable, str(ROOT / "examples_llm.py")],
}

DEFAULT_ORDER = ["base"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Lanza uno o varios modos del simulador sin modificar su logica interna."
    )
    parser.add_argument("--base", action="store_true", help="Ejecuta el entrenamiento base sin LLM.")
    parser.add_argument("--godot", action="store_true", help="Ejecuta el modo con visualizacion en Godot.")
    parser.add_argument(
        "--behavior-selector",
        action="store_true",
        help="Ejecuta el modo LLM de selector de comportamientos.",
    )
    parser.add_argument(
        "--reward-shaping",
        action="store_true",
        help="Ejecuta el modo LLM de reward shaping.",
    )
    parser.add_argument(
        "--reflexion",
        action="store_true",
        help="Ejecuta el modo LLM de reflexion.",
    )
    parser.add_argument(
        "--examples-llm",
        action="store_true",
        help="Ejecuta los ejemplos de los modulos LLM.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Ejecuta todos los modos en el orden definido por el lanzador.",
    )
    parser.add_argument("--list", action="store_true", help="Muestra los modos disponibles y termina.")
    return parser


def selected_modes(args: argparse.Namespace) -> list[str]:
    if args.all:
        return list(MODES.keys())

    chosen: list[str] = []
    for mode_name in MODES:
        attr_name = mode_name.replace("-", "_")
        if getattr(args, attr_name, False):
            chosen.append(mode_name)

    if chosen:
        return chosen

    return DEFAULT_ORDER


def run_mode(mode_name: str) -> None:
    command = MODES[mode_name]
    print(f"\n=== Ejecutando {mode_name} ===")
    subprocess.run(command, check=True)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.list:
        print("Modos disponibles:")
        for mode_name in MODES:
            print(f"  - {mode_name}")
        return 0

    modes = selected_modes(args)
    print("Modos seleccionados:")
    for mode_name in modes:
        print(f"  - {mode_name}")

    try:
        for mode_name in modes:
            run_mode(mode_name)
    except subprocess.CalledProcessError as exc:
        print(f"\n[ERROR] Fallo el modo {mode_name} con codigo {exc.returncode}")
        return exc.returncode
    except KeyboardInterrupt:
        print("\n[!] Ejecucion interrumpida por el usuario")
        return 130

    return 0


if __name__ == "__main__":
    raise SystemExit(main())