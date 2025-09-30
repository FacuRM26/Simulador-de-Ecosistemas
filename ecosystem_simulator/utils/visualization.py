"""
Módulo para visualizar los resultados del entrenamiento.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

def analyze_training_results(csv_path="monitor.csv"):
    """Analiza y visualiza los resultados del entrenamiento."""
    try:
        df = pd.read_csv(csv_path)

        # Conversión a numérico
        for c in df.columns:
            if c != "iter":
                df[c] = pd.to_numeric(df[c], errors="coerce")

        # Verificación de columnas requeridas
        if not {"r_mean", "l_mean"}.issubset(df.columns):
            raise KeyError("El CSV no tiene r_mean/l_mean.")

        iters = df["iter"].values if "iter" in df.columns else np.arange(len(df))
        r_mean = df["r_mean"]
        l_mean = df["l_mean"]

        # Detección de columnas de razones
        reason_cols = ["reason_timeout_pct", "reason_starvation_pct", "reason_dehydration_pct"]
        has_reasons = all(col in df.columns for col in reason_cols)
        
        # Detección de columnas por agente
        r_agent_cols = sorted([c for c in df.columns if c.startswith("r_agent_")],
                            key=lambda x: int(re.search(r"(\d+)$", x).group(1)))
        l_agent_cols = sorted([c for c in df.columns if c.startswith("l_agent_")],
                            key=lambda x: int(re.search(r"(\d+)$", x).group(1)))

        # Suavizado dinámico
        SMOOTH_WINDOW = max(5, min(20, max(1, len(df)//3)))
        mov_avg_r = r_mean.rolling(SMOOTH_WINDOW, min_periods=1).mean()
        mov_avg_l = l_mean.rolling(SMOOTH_WINDOW, min_periods=1).mean()
        cumavg_r = r_mean.expanding().mean()
        cumavg_l = l_mean.expanding().mean()

        # Visualización
        _create_plots(iters, r_mean, l_mean, mov_avg_r, mov_avg_l, cumavg_r, cumavg_l, 
                     r_agent_cols, l_agent_cols, df, SMOOTH_WINDOW, reason_cols, has_reasons)

        # Resumen numérico
        print("\nResumen:")
        print(f"r_mean: mean={np.nanmean(r_mean):.3f}, std={np.nanstd(r_mean):.3f}, "
              f"median={np.nanmedian(r_mean):.3f}")
        print(f"l_mean: mean={np.nanmean(l_mean):.3f}, std={np.nanstd(l_mean):.3f}, "
              f"median={np.nanmedian(l_mean):.3f}")

    except FileNotFoundError:
        print(f"No se encontró el archivo '{csv_path}'. Genera antes el CSV de monitorización.")
    except Exception as e:
        print("Error en análisis:", repr(e))

def _create_plots(iters, r_mean, l_mean, mov_avg_r, mov_avg_l, cumavg_r, cumavg_l,
                 r_agent_cols, l_agent_cols, df, SMOOTH_WINDOW, reason_cols, has_reasons):
    """Crea las visualizaciones de los resultados."""
    # Figura 1: Métricas agregadas
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(iters, r_mean, alpha=0.3, label="r_mean (raw)")
    plt.plot(iters, mov_avg_r, label=f"r_mean mov.avg (w={SMOOTH_WINDOW})")
    plt.title("Recompensa media por iteración")
    plt.xlabel("Iteración")
    plt.ylabel("Return")
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(iters, l_mean, alpha=0.3, label="l_mean (raw)")
    plt.plot(iters, mov_avg_l, label=f"l_mean mov.avg (w={SMOOTH_WINDOW})")
    plt.title("Longitud media por iteración")
    plt.xlabel("Iteración")
    plt.ylabel("Timesteps")
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(iters, cumavg_r)
    plt.title("Return acumulado (promedio)")
    plt.xlabel("Iteración")
    plt.ylabel("Return")

    plt.subplot(2, 2, 4)
    plt.plot(iters, cumavg_l)
    plt.title("Longitud acumulada (promedio)")
    plt.xlabel("Iteración")
    plt.ylabel("Timesteps")

    plt.tight_layout()
    plt.show()

    # Figura 2: Retorno por agente
    if r_agent_cols:
        plt.figure(figsize=(12, 6))
        for c in r_agent_cols:
            series = df[c].rolling(SMOOTH_WINDOW, min_periods=1).mean()
            plt.plot(iters, series, label=c)
        plt.title("Return por agente (media móvil)")
        plt.xlabel("Iteración")
        plt.ylabel("Return")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # Figura 3: Longitud por agente
    if l_agent_cols:
        plt.figure(figsize=(12, 6))
        for c in l_agent_cols:
            series = df[c].rolling(SMOOTH_WINDOW, min_periods=1).mean()
            plt.plot(iters, series, label=c)
        plt.title("Longitud por agente (media móvil)")
        plt.xlabel("Iteración")
        plt.ylabel("Timesteps")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # Figura 4: Razones de terminación
    if has_reasons:
        timeout_pct, starvation_pct, dehydration_pct = [df[c] for c in reason_cols]
        plt.figure(figsize=(10,4))
        plt.plot(iters, timeout_pct,     label="timeout %")
        plt.plot(iters, starvation_pct,  label="starvation %")
        plt.plot(iters, dehydration_pct, label="dehydration %")
        plt.title("Razones de terminación de episodios")
        plt.xlabel("Iteración")
        plt.ylabel("Porcentaje")
        plt.legend()
        plt.tight_layout()
        plt.show()