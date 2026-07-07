#!/usr/bin/env python
"""
Lanzador unificado y COMBINABLE del simulador.

A diferencia de la versión anterior (que lanzaba scripts separados por
subprocess sin combinar nada), esta versión corre UN solo entrenamiento
sobre el que se APILAN los modos que elijas. Todos comparten el mismo
trainer y el mismo monitor.csv, así se pueden comparar.

Ejemplos:
    python run.py                       # entrenamiento base (PPO)
    python run.py --godot               # base + visualización en Godot
    python run.py --godot --plot        # + gráficos al finalizar
    python run.py --iters 50            # override rápido de iteraciones

Los modos LLM requieren Ollama corriendo (p.ej. `ollama run mistral`).
--reflexion se construye en la fase siguiente.
"""
from __future__ import annotations

import argparse

from ecosystem_simulator.training.orchestrator import run_training

# Modos LLM aún no cableados (se construyen en fases siguientes).
_PENDING = {
    "reflexion": "Fase 4",
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Lanza un entrenamiento combinable del simulador de ecosistemas."
    )
    p.add_argument("--godot", action="store_true",
                   help="Visualización en tiempo real en Godot.")
    p.add_argument("--reward-shaping", action="store_true",
                   help="Reward shaping dinámico por LLM (requiere Ollama).")
    p.add_argument("--behavior-selector", action="store_true",
                   help="Selector LLM de comportamientos por rol (requiere Ollama).")
    p.add_argument("--reflexion", action="store_true",
                   help="[Fase 4] Reflexión LLM entre iteraciones.")
    p.add_argument("--plot", action="store_true",
                   help="Muestra los gráficos de análisis al finalizar.")
    p.add_argument("--iters", type=int, default=None,
                   help="Sobrescribe el número de iteraciones de config.py.")
    p.add_argument("--llm-model", default="mistral",
                   help="Modelo de Ollama para los modos LLM (default: mistral).")
    p.add_argument("--shape-interval", type=int, default=5,
                   help="Cada cuántas iteraciones consulta el LLM de shaping (default: 5).")
    return p


def main() -> int:
    args = build_parser().parse_args()

    # Avisar sobre modos LLM todavía no implementados.
    pending = [name for name in _PENDING if getattr(args, name)]
    if pending:
        for name in pending:
            flag = "--" + name.replace("_", "-")
            print(f"[!] {flag} aún no está implementado (se construye en {_PENDING[name]}).")
        return 1

    modes = set()
    if args.godot:
        modes.add("godot")
    if args.behavior_selector:
        modes.add("behavior_selector")
    if args.reward_shaping:
        modes.add("reward_shaping")

    kwargs = {
        "plot": args.plot,
        "llm_model": args.llm_model,
        "shape_interval": args.shape_interval,
    }
    if args.iters is not None:
        kwargs["num_iters"] = args.iters

    run_training(modes, **kwargs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
