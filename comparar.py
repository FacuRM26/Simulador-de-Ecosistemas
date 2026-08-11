"""
Compara dos monitor.csv (linea base vs con modelo) y arma la tabla
"comportamiento / linea base / con modelo / mejora %" para la diapositiva.

Uso:
    python comparar.py monitor_base.csv monitor_modelo.csv
    python comparar.py monitor_base.csv monitor_modelo.csv --tail=50

El PRIMER archivo es siempre la linea base.

--tail=N : cuantas iteraciones FINALES se promedian (por defecto 50).
           Se promedia la cola porque el inicio del entrenamiento es ruido.

NOTA SOBRE UNA SOLA CORRIDA POR CONDICION:
    Con una corrida no hay forma de medir cuanto varia el resultado por puro
    azar entre corridas. Como aproximacion, este script usa la variabilidad
    ENTRE ITERACIONES de la cola como estimacion del ruido, y marca con (*)
    las diferencias que caen dentro de ese ruido. Esas NO se deben reportar
    como mejora.
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


def load_tail(path: Path, tail: int) -> dict:
    """Lee un monitor.csv y devuelve {metrica: (media_cola, desviacion_cola)}."""
    if not path.exists():
        sys.exit(f"ERROR: no existe el archivo {path}")

    with open(path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        sys.exit(f"ERROR: {path} esta vacio")

    total = len(rows)
    rows = rows[-tail:]

    out = {}
    for key in rows[0]:
        vals = []
        for r in rows:
            try:
                vals.append(float(r[key]))
            except (TypeError, ValueError):
                pass
        if vals:
            out[key] = (
                statistics.mean(vals),
                statistics.stdev(vals) if len(vals) > 1 else 0.0,
            )

    out["__n_iters__"] = (float(total), float(len(rows)))
    return out


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

    base_path, treat_path = Path(args[0]), Path(args[1])
    base = load_tail(base_path, tail)
    treat = load_tail(treat_path, tail)

    b_total, b_used = base.pop("__n_iters__")
    t_total, t_used = treat.pop("__n_iters__")

    print()
    print(f"LINEA BASE : {base_path}  ({int(b_total)} iteraciones, se usan las ultimas {int(b_used)})")
    print(f"CON MODELO : {treat_path}  ({int(t_total)} iteraciones, se usan las ultimas {int(t_used)})")
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

        b_mean, b_std = base[m]
        t_mean, t_std = treat[m]
        change = pct_change(b_mean, t_mean, b["better"])

        u = b["unit"]
        b_txt = f"{b_mean:.2f}{u} +-{b_std:.2f}"
        t_txt = f"{t_mean:.2f}{u} +-{t_std:.2f}"
        c_txt = "n/d" if change is None else f"{change:+.1f}%"

        # Ruido = variabilidad entre iteraciones de la cola.
        noise = max(b_std, t_std)
        weak = change is not None and abs(t_mean - b_mean) < noise
        flag = "  (*)" if weak else ""

        print(f"{b['label']:<52} {b_txt:>18} {t_txt:>18} {c_txt:>9}{flag}")
        rows_out.append((b["label"], b_txt, t_txt, c_txt, "si" if weak else ""))

    print("-" * len(header))
    print("(*) La diferencia es menor que la fluctuacion entre iteraciones:")
    print("    NO es concluyente. No reportar como mejora en la diapositiva.")

    if faltantes:
        print()
        print("[!] Metricas no encontradas en los CSV: " + ", ".join(faltantes))
        print("    (si falta 'herbivore_avg_nn_dist', el monitor.csv se genero")
        print("     antes de aplicar el parche de metricas de comportamiento)")

    out = treat_path.parent / "tabla_diapositiva.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            ["Comportamiento", "Linea base", "Con modelo", "Mejora %", "No concluyente"]
        )
        w.writerows(rows_out)

    print(f"\n[+] Tabla guardada en: {out}")


if __name__ == "__main__":
    main()