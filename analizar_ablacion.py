"""
Analisis de ablacion: compara todas las condiciones LLM contra la linea base.

Como todas las condiciones usan LAS MISMAS semillas (1005-1007), la comparacion
es PAREADA: para cada semilla se calcula la diferencia (condicion - base) y luego
se promedia. Esto elimina la varianza entre semillas y es mucho mas sensible que
comparar promedios sueltos.

Uso:
    python analizar_ablacion.py                # tail por defecto (500)
    python analizar_ablacion.py --tail=500
"""
import csv
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "experiment_results"

# etiqueta corta -> carpeta
CONDITIONS = {
    "base":              "base_2000",
    "shaping":           "reward-shaping_2000",
    "selector":          "behavior_selector_2000",
    "shaping+selector":  "reward-shaping_and_behavior-selector_2000",
    "shaping+reflex":    "reward-shaping_and_reflexion_2000",
    "selector+reflex":   "behavior-selector_and_reflexion_2000",
    "todos":             "all_2000",
}

# metrica -> (etiqueta, unidad, direccion buena)
METRICS = [
    ("herbivore_avg_nn_dist",      "Agrupacion presas (dist. vecino)", "px", "down"),
    ("herbivore_avg_pred_dist",    "Evasion (dist. depredador)",       "px", "up"),
    ("herbivore_survival_pct",     "Supervivencia presas",             "%",  "up"),
    ("herbivore_eat_count",        "Ingestas presas/ep",               "",   "up"),
    ("herbivore_drink_count",      "Tomas agua presas/ep",             "",   "up"),
    ("predator_attack_kill_count", "Cazas exitosas/ep",                "",   "up"),
    ("predator_survival_pct",      "Supervivencia depredadores",       "%",  "up"),
]


def tail_mean(monitor: Path, tail: int) -> dict:
    rows = list(csv.DictReader(open(monitor, encoding="utf-8-sig")))
    rows = rows[-tail:]
    out = {}
    for k in rows[0]:
        if k == "iter":
            continue
        vals = []
        for r in rows:
            try:
                vals.append(float(r[k]))
            except (TypeError, ValueError):
                pass
        if vals:
            out[k] = statistics.mean(vals)
    return out


def load_condition(folder: str, tail: int) -> dict:
    """Devuelve {seed: {metrica: valor}}."""
    out = {}
    for monitor in sorted((ROOT / folder).glob("run_*/monitor.csv")):
        seed = monitor.parent.name.split("seed_")[-1]
        out[seed] = tail_mean(monitor, tail)
    return out


def main():
    tail = 500
    for a in sys.argv[1:]:
        if a.startswith("--tail="):
            tail = int(a.split("=", 1)[1])

    data = {name: load_condition(folder, tail)
            for name, folder in CONDITIONS.items()}

    base = data["base"]
    seeds = sorted(base)

    print(f"\nABLACION LLM  |  cola = ultimas {tail} de 2000 iteraciones  |  semillas: {', '.join(seeds)}")
    print("Comparacion PAREADA por semilla (condicion - base).")
    print("(*) = la direccion NO fue consistente en las 3 semillas -> no concluyente.\n")

    for key, label, unit, better in METRICS:
        base_vals = [base[s][key] for s in seeds]
        base_mean = statistics.mean(base_vals)
        print("=" * 92)
        print(f"{label}   [base = {base_mean:.2f}{unit}]   (mejor: {'menor' if better=='down' else 'mayor'})")
        print("-" * 92)
        print(f"{'Condicion':<20}{'valor':>12}{'delta':>12}{'delta %':>11}{'semillas a favor':>20}")

        rows = []
        for name in CONDITIONS:
            if name == "base":
                continue
            cond = data[name]
            deltas = [cond[s][key] - base[s][key] for s in seeds]
            mean_delta = statistics.mean(deltas)
            cond_mean = statistics.mean([cond[s][key] for s in seeds])

            # mejora en la direccion correcta
            signed = -mean_delta if better == "down" else mean_delta
            pct = 100.0 * signed / abs(base_mean) if base_mean else 0.0

            # consistencia: cuantas semillas mejoraron
            good = sum(1 for d in deltas
                       if (-d if better == "down" else d) > 0)
            consistent = good == 3 or good == 0
            rows.append((name, cond_mean, mean_delta, pct, good, consistent))

        rows.sort(key=lambda r: -r[3])
        for name, cond_mean, mean_delta, pct, good, consistent in rows:
            flag = "" if consistent else "  (*)"
            print(f"{name:<20}{cond_mean:>11.2f}{unit:<1}{mean_delta:>+12.2f}{pct:>+10.1f}%{good:>13}/3{flag}")
        print()


if __name__ == "__main__":
    main()
