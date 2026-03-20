"""
Módulo para visualizar los resultados del entrenamiento.

Proporciona funciones para analizar y graficar las métricas de entrenamiento
guardadas en archivos CSV, incluyendo:
- Recompensas y longitudes de episodios
- Métricas por agente individual
- Métricas por rol (herbívoros / depredadores)
- Razones de terminación por rol
- Métricas de ataque
- Promedios móviles y acumulados
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

N_PREDATORS = 2

from pathlib import Path

def analyze_training_results(csv_path=None):
    """
    Analiza y visualiza los resultados del entrenamiento desde un archivo CSV.
    """
    try:
        if csv_path is None:
            csv_path = Path(__file__).resolve().parents[2] / "monitor.csv"
        else:
            csv_path = Path(csv_path).resolve()
        print("Leyendo monitor desde:", csv_path)
        df = pd.read_csv(csv_path)

        # Convertir columnas numéricas
        for c in df.columns:
            if c != "iter":
                df[c] = pd.to_numeric(df[c], errors="coerce")

        if not {"r_mean", "l_mean"}.issubset(df.columns):
            raise KeyError("El CSV no tiene r_mean/l_mean.")

        iters = df["iter"].values if "iter" in df.columns else np.arange(len(df))
        r_mean = df["r_mean"]
        l_mean = df["l_mean"]

        # Columnas por rol
        herb_reason_cols = [
            "herbivore_reason_timeout_pct",
            "herbivore_reason_starvation_pct",
            "herbivore_reason_dehydration_pct",
            "herbivore_reason_predation_pct",
        ]
        pred_reason_cols = [
            "predator_reason_timeout_pct",
            "predator_reason_starvation_pct",
            "predator_reason_dehydration_pct",
            "predator_reason_predation_pct",
        ]
        has_herb_reasons = all(col in df.columns for col in herb_reason_cols)
        has_pred_reasons = all(col in df.columns for col in pred_reason_cols)

        attack_cols = [
            "attacks_attempted",
            "attacks_hit",
            "attacks_kill",
            "attack_hit_rate",
            "attack_kill_rate",
        ]
        has_attacks = all(c in df.columns for c in attack_cols)

        role_cols = [
            "herbivore_episode_return",
            "predator_episode_return",
            "herbivore_episode_len",
            "predator_episode_len",
            "herbivore_survival_pct",
            "predator_survival_pct",
            "herbivore_avg_food",
            "herbivore_avg_water",
            "predator_avg_food",
            "predator_avg_water",
            "herbivore_critical_ratio",
            "predator_critical_ratio",
        ]
        has_role_metrics = any(c in df.columns for c in role_cols)

        # Detectar columnas por agente
        r_agent_cols = sorted(
            [c for c in df.columns if c.startswith("r_agent_")],
            key=lambda x: int(re.search(r"(\d+)$", x).group(1))
        )
        l_agent_cols = sorted(
            [c for c in df.columns if c.startswith("l_agent_")],
            key=lambda x: int(re.search(r"(\d+)$", x).group(1))
        )

        # Suavizado
        SMOOTH_WINDOW = max(5, min(20, max(1, len(df) // 3)))
        mov_avg_r = r_mean.rolling(SMOOTH_WINDOW, min_periods=1).mean()
        mov_avg_l = l_mean.rolling(SMOOTH_WINDOW, min_periods=1).mean()
        cumavg_r = r_mean.expanding().mean()
        cumavg_l = l_mean.expanding().mean()
        print("\nColumnas del CSV:")
        print(df.columns.tolist())

        print("\nFlags detectados:")
        print("has_role_metrics =", has_role_metrics)
        print("has_herb_reasons =", has_herb_reasons)
        print("has_pred_reasons =", has_pred_reasons)
        print("has_attacks =", has_attacks)
        _create_plots(
            iters=iters,
            r_mean=r_mean,
            l_mean=l_mean,
            mov_avg_r=mov_avg_r,
            mov_avg_l=mov_avg_l,
            cumavg_r=cumavg_r,
            cumavg_l=cumavg_l,
            r_agent_cols=r_agent_cols,
            l_agent_cols=l_agent_cols,
            df=df,
            smooth_window=SMOOTH_WINDOW,
            herb_reason_cols=herb_reason_cols,
            pred_reason_cols=pred_reason_cols,
            has_herb_reasons=has_herb_reasons,
            has_pred_reasons=has_pred_reasons,
            attack_cols=attack_cols,
            has_attacks=has_attacks,
            has_role_metrics=has_role_metrics,
        )

        print("\nResumen:")
        print(f"r_mean: mean={np.nanmean(r_mean):.3f}, std={np.nanstd(r_mean):.3f}, median={np.nanmedian(r_mean):.3f}")
        print(f"l_mean: mean={np.nanmean(l_mean):.3f}, std={np.nanstd(l_mean):.3f}, median={np.nanmedian(l_mean):.3f}")

        if "herbivore_survival_pct" in df.columns:
            print(f"Herbivore survival pct (último): {df['herbivore_survival_pct'].iloc[-1]:.2f}")
        if "predator_survival_pct" in df.columns:
            print(f"Predator survival pct (último): {df['predator_survival_pct'].iloc[-1]:.2f}")
        if "attack_hit_rate" in df.columns:
            print(f"Attack hit rate (último): {df['attack_hit_rate'].iloc[-1]:.3f}")
        if "attack_kill_rate" in df.columns:
            print(f"Attack kill rate (último): {df['attack_kill_rate'].iloc[-1]:.3f}")

    except FileNotFoundError:
        print(f"No se encontró el archivo '{csv_path}'. Genera antes el CSV de monitorización.")
    except Exception as e:
        print("Error en análisis:", repr(e))


def _create_plots(
    iters,
    r_mean,
    l_mean,
    mov_avg_r,
    mov_avg_l,
    cumavg_r,
    cumavg_l,
    r_agent_cols,
    l_agent_cols,
    df,
    smooth_window,
    herb_reason_cols,
    pred_reason_cols,
    has_herb_reasons,
    has_pred_reasons,
    attack_cols,
    has_attacks,
    has_role_metrics,
):
    """
    Crea todas las visualizaciones.
    """

    # =========================
    # FIGURA 1: MÉTRICAS AGREGADAS
    # =========================
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(iters, r_mean, alpha=0.3, label="r_mean (raw)")
    plt.plot(iters, mov_avg_r, label=f"r_mean mov.avg (w={smooth_window})")
    plt.title("Recompensa media por iteración")
    plt.xlabel("Iteración")
    plt.ylabel("Return")
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(iters, l_mean, alpha=0.3, label="l_mean (raw)")
    plt.plot(iters, mov_avg_l, label=f"l_mean mov.avg (w={smooth_window})")
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

    # =========================
    # FIGURA 2: RETURN POR AGENTE
    # =========================
    if r_agent_cols:
        plt.figure(figsize=(12, 6))
        for idx, c in enumerate(r_agent_cols):
            series = df[c].rolling(smooth_window, min_periods=1).mean()
            is_pred = idx < N_PREDATORS
            label = f"P{idx}" if is_pred else f"H{idx - N_PREDATORS}"
            plt.plot(iters, series, label=label, linestyle="-" if is_pred else "--")
        plt.title("Return por agente (media móvil)")
        plt.xlabel("Iteración")
        plt.ylabel("Return")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # =========================
    # FIGURA 3: LONGITUD POR AGENTE
    # =========================
    if l_agent_cols:
        plt.figure(figsize=(12, 6))
        for idx, c in enumerate(l_agent_cols):
            series = df[c].rolling(smooth_window, min_periods=1).mean()
            is_pred = idx < N_PREDATORS
            label = f"P{idx}" if is_pred else f"H{idx - N_PREDATORS}"
            plt.plot(iters, series, label=label, linestyle="-" if is_pred else "--")
        plt.title("Longitud por agente (media móvil)")
        plt.xlabel("Iteración")
        plt.ylabel("Timesteps")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # =========================
    # FIGURA 4: MÉTRICAS POR ROL
    # =========================
    if has_role_metrics:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Return por rol
        if {"herbivore_episode_return", "predator_episode_return"}.issubset(df.columns):
            axes[0, 0].plot(
                iters,
                df["herbivore_episode_return"].rolling(smooth_window, min_periods=1).mean(),
                label="Herbivore",
            )
            axes[0, 0].plot(
                iters,
                df["predator_episode_return"].rolling(smooth_window, min_periods=1).mean(),
                label="Predator",
            )
            axes[0, 0].set_title("Return por rol")
            axes[0, 0].set_xlabel("Iteración")
            axes[0, 0].set_ylabel("Return")
            axes[0, 0].legend()

        # Longitud por rol
        if {"herbivore_episode_len", "predator_episode_len"}.issubset(df.columns):
            axes[0, 1].plot(
                iters,
                df["herbivore_episode_len"].rolling(smooth_window, min_periods=1).mean(),
                label="Herbivore",
            )
            axes[0, 1].plot(
                iters,
                df["predator_episode_len"].rolling(smooth_window, min_periods=1).mean(),
                label="Predator",
            )
            axes[0, 1].set_title("Longitud por rol")
            axes[0, 1].set_xlabel("Iteración")
            axes[0, 1].set_ylabel("Timesteps")
            axes[0, 1].legend()

        # Survival por rol
        if {"herbivore_survival_pct", "predator_survival_pct"}.issubset(df.columns):
            axes[1, 0].plot(
                iters,
                df["herbivore_survival_pct"].rolling(smooth_window, min_periods=1).mean(),
                label="Herbivore",
            )
            axes[1, 0].plot(
                iters,
                df["predator_survival_pct"].rolling(smooth_window, min_periods=1).mean(),
                label="Predator",
            )
            axes[1, 0].set_title("Supervivencia por rol")
            axes[1, 0].set_xlabel("Iteración")
            axes[1, 0].set_ylabel("%")
            axes[1, 0].legend()

        # Critical ratio por rol
        if {"herbivore_critical_ratio", "predator_critical_ratio"}.issubset(df.columns):
            axes[1, 1].plot(
                iters,
                df["herbivore_critical_ratio"].rolling(smooth_window, min_periods=1).mean(),
                label="Herbivore",
            )
            axes[1, 1].plot(
                iters,
                df["predator_critical_ratio"].rolling(smooth_window, min_periods=1).mean(),
                label="Predator",
            )
            axes[1, 1].set_title("Critical ratio por rol")
            axes[1, 1].set_xlabel("Iteración")
            axes[1, 1].set_ylabel("Ratio")
            axes[1, 1].legend()

        plt.tight_layout()
        plt.show()

    # =========================
    # FIGURA 5: RAZONES DE TERMINACIÓN POR ROL
    # =========================
    if has_herb_reasons or has_pred_reasons:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

        if has_herb_reasons:
            for c in herb_reason_cols:
                serie = df[c].rolling(smooth_window, min_periods=1).mean()
                label = c.replace("herbivore_reason_", "").replace("_pct", " %")
                axes[0].plot(iters, serie, label=label)
            axes[0].set_title("Razones de terminación - Herbívoros")
            axes[0].set_xlabel("Iteración")
            axes[0].set_ylabel("Porcentaje")
            axes[0].legend()

        if has_pred_reasons:
            for c in pred_reason_cols:
                serie = df[c].rolling(smooth_window, min_periods=1).mean()
                label = c.replace("predator_reason_", "").replace("_pct", " %")
                axes[1].plot(iters, serie, label=label)
            axes[1].set_title("Razones de terminación - Depredadores")
            axes[1].set_xlabel("Iteración")
            axes[1].legend()

        plt.tight_layout()
        plt.show()

    # =========================
    # FIGURA 6: MÉTRICAS DE RECURSOS POR ROL
    # =========================
    resource_cols = [
        "herbivore_avg_food", "herbivore_avg_water",
        "predator_avg_food", "predator_avg_water",
    ]
    if all(c in df.columns for c in resource_cols):
        plt.figure(figsize=(12, 5))
        for c in resource_cols:
            serie = df[c].rolling(smooth_window, min_periods=1).mean()
            plt.plot(iters, serie, label=c)
        plt.title("Promedio de recursos por rol")
        plt.xlabel("Iteración")
        plt.ylabel("Promedio normalizado")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # =========================
    # FIGURA 7: MÉTRICAS DE ATAQUE
    # =========================
    if has_attacks:
        plt.figure(figsize=(12, 5))
        for c in ["attacks_attempted", "attacks_hit", "attacks_kill"]:
            serie = df[c].rolling(smooth_window, min_periods=1).mean()
            plt.plot(iters, serie, label=c)

        plt.title("Ataques (conteos por iteración - media móvil)")
        plt.xlabel("Iteración")
        plt.ylabel("Conteo")
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(12, 4))
        for c in ["attack_hit_rate", "attack_kill_rate"]:
            serie = df[c].rolling(smooth_window, min_periods=1).mean()
            plt.plot(iters, serie, label=c)

        plt.title("Tasas de ataque (media móvil)")
        plt.xlabel("Iteración")
        plt.ylabel("Rate")
        plt.ylim(0, 1)
        plt.legend()
        plt.tight_layout()
        plt.show()