"""
Compara dos EXPERIMENTOS (carpetas con varias corridas) y arma la tabla
"comportamiento / linea base / con modelo / mejora %" para la diapositiva.

A diferencia de comparar.py (que compara dos monitor.csv sueltos), este script
promedia TODAS las corridas de cada experimento. Eso es lo correcto: cada corrida
usa una semilla distinta, asi que elegir una sola seria cherry-picking.

Uso:
    python comparar_experimentos.py experiment_results/base_200 experiment_results/modelo_200
    python comparar_experimentos.py <base> <modelo> --tail=50

El PRIMER argumento es siempre la linea base.

--tail=N : cuantas iteraciones FINALES de cada corrida se promedian (default 50).
           Se promedia la cola porque el inicio del entrenamiento es ruido.

COMO SE MIDE EL RUIDO:
    Cada corrida aporta un valor (su cola promediada). Con esos valores se calcula
    media y desviacion estandar ENTRE CORRIDAS. Esa desviacion es la estimacion
    honesta del azar. Si la diferencia entre condiciones es menor que ese ruido,
    se marca con (*) y NO se debe reportar como mejora.
"""
import csv
import statistics
import sys
from pathlib import Path


# better: "up" = mas alto es mejor | "down" = mas bajo es mejor
BEHAVIORS = [
    {
        "label": "Agrupacion de presas (dist. al vecino mas cercano)",
        "metric": "herbivore_avg_nn_dist",
        "unit": "px",
        "better": "down",
    },
    {
        "label": "Evasion (dist. al depredador mas cercano)",
        "metric": "herbivore_avg_pred_dist",
        "unit": "px",
        "better": "up",
    },
    {
        "label": "Supervivencia de presas",
        "metric": "herbivore_survival_pct",
        "unit": "%",
        "better": "up",
    },
    {
        "label": "Busqueda de alimento (ingestas/episodio)",
        "metric": "herbivore_eat_count",
        "unit": "",
        "better": "up",
    },
    {
        "label": "Busqueda de agua (tomas/episodio)",
        "metric": "herbivore_drink_count",
        "unit": "",
        "better": "up",
    },
    {
        "label": "Exito de caza (presas cazadas/episodio)",
        "metric": "predator_attack_kill_count",
        "unit": "",
        "better": "up",
    },
    {
        "label": "Supervivencia de depredadores",
        "metric": "predator_survival_pct",
        "unit": "%",
        "better": "up",
    },
]


def tail_mean_of_run(monitor_path: Path, tail: int) -> dict:
    """Promedia las ultimas `tail` iteraciones de UNA corrida."""
    with open(monitor_path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return {}

    rows = rows[-tail:]
    out = {}

    for key in rows[0]:
        if key == "iter":
            continue

        vals = []
        for r in rows:
            try:
                vals.append(float(r[key]))
            except (TypeError, ValueError):
                pass

        if vals:
            out[key] = statistics.mean(vals)

    return out


def load_experiment(exp_dir: Path, tail: int) -> tuple[dict, int]:
    """
    Lee todas las corridas de un experimento.

    Devuelve {metrica: [valor_por_corrida, ...]} y la cantidad de corridas.
    """
    if not exp_dir.exists():
        sys.exit(f"ERROR: no existe la carpeta {exp_dir}")

    monitors = sorted(exp_dir.glob("run_*/monitor.csv"))

    if not monitors:
        sys.exit(
            f"ERROR: no se encontraron corridas (run_*/monitor.csv) en {exp_dir}"
        )

    per_metric = {}
    used = 0

    for monitor in monitors:
        run_values = tail_mean_of_run(monitor, tail)

        if not run_values:
            print(f"[!] Corrida vacia, se omite: {monitor}")
            continue

        used += 1
        for metric, value in run_values.items():
            per_metric.setdefault(metric, []).append(value)

    if used == 0:
        sys.exit(f"ERROR: todas las corridas de {exp_dir} estan vacias")

    return per_metric, used


def mean_std(values: list[float]) -> tuple[float, float]:
    return (
        statistics.mean(values),
        statistics.stdev(values) if len(values) > 1 else 0.0,
    )


def pct_change(base: float, treat: float, better: str):
    if base == 0:
        return None
    raw = 100.0 * (treat - base) / abs(base)
    # Si lo bueno es BAJAR, una reduccion cuenta como mejora positiva.
    return -raw if better == "down" else raw


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]

    tail = 50
    for a in sys.argv[1:]:
        if a.startswith("--tail="):
            try:
                tail = int(a.split("=", 1)[1])
            except ValueError:
                pass

    if len(args) != 2:
        sys.exit(__doc__)

    base_dir, treat_dir = Path(args[0]), Path(args[1])
    base, n_base = load_experiment(base_dir, tail)
    treat, n_treat = load_experiment(treat_dir, tail)

    print()
    print(f"LINEA BASE : {base_dir}  ({n_base} corridas, cola de {tail} iteraciones)")
    print(f"CON MODELO : {treat_dir}  ({n_treat} corridas, cola de {tail} iteraciones)")
    print()

    header = (
        f"{'Comportamiento':<52} {'Linea base':>18} {'Con modelo':>18} {'Mejora':>9}"
    )
    print(header)
    print("-" * len(header))

    rows_out = []
    faltantes = []

    for b in BEHAVIORS:
        m = b["metric"]

        if m not in base or m not in treat:
            faltantes.append(m)
            continue

        b_mean, b_std = mean_std(base[m])
        t_mean, t_std = mean_std(treat[m])
        change = pct_change(b_mean, t_mean, b["better"])

        u = b["unit"]
        b_txt = f"{b_mean:.2f}{u} +-{b_std:.2f}"
        t_txt = f"{t_mean:.2f}{u} +-{t_std:.2f}"
        c_txt = "n/d" if change is None else f"{change:+.1f}%"

        # Ruido = dispersion ENTRE corridas (la estimacion honesta del azar).
        noise = max(b_std, t_std)
        weak = change is not None and abs(t_mean - b_mean) < noise
        flag = "  (*)" if weak else ""

        print(f"{b['label']:<52} {b_txt:>18} {t_txt:>18} {c_txt:>9}{flag}")
        rows_out.append(
            (b["label"], b_txt, t_txt, c_txt, "si" if weak else "")
        )

    print("-" * len(header))
    print("(*) La diferencia es menor que la variacion entre corridas:")
    print("    NO es concluyente. No reportar como mejora en la diapositiva.")

    if n_base < 3 or n_treat < 3:
        print()
        print(
            f"[!] Pocas corridas (base={n_base}, modelo={n_treat}). "
            "Con menos de 3 la estimacion de ruido es debil."
        )

    if faltantes:
        print()
        print("[!] Metricas no encontradas en los CSV: " + ", ".join(faltantes))
        print("    (si falta 'herbivore_avg_nn_dist', ese experimento se genero")
        print("     antes de aplicar el parche de metricas de comportamiento)")

    out = treat_dir / "tabla_diapositiva.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            ["Comportamiento", "Linea base", "Con modelo", "Mejora %", "No concluyente"]
        )
        w.writerows(rows_out)

    print(f"\n[+] Tabla guardada en: {out}")


if __name__ == "__main__":
    main()
