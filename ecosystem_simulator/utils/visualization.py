"""
Módulo para visualizar los resultados del entrenamiento.

Proporciona funciones para analizar y graficar las métricas de entrenamiento
guardadas en archivos CSV, incluyendo:
- Recompensas y longitudes de episodios
- Métricas por agente individual
- Razones de terminación
- Promedios móviles y acumulados
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

def analyze_training_results(csv_path="monitor.csv"):
    """
    Analiza y visualiza los resultados del entrenamiento desde un archivo CSV.
    
    Lee el archivo CSV de monitoreo, procesa las métricas y genera múltiples
    gráficos para analizar el progreso del entrenamiento:
    
    1. Métricas agregadas (recompensa y longitud media)
    2. Retorno por agente individual
    3. Longitud por agente individual  
    4. Razones de terminación
    
    También imprime un resumen estadístico en consola.
    
    Args:
        csv_path: Ruta al archivo CSV de monitoreo (por defecto "monitor.csv")
    """
    try:
        # Leer archivo CSV
        df = pd.read_csv(csv_path)

        # Convertir todas las columnas numéricas (excepto 'iter')
        for c in df.columns:
            if c != "iter":
                df[c] = pd.to_numeric(df[c], errors="coerce")

        # Verificar que el CSV tenga las columnas esenciales
        if not {"r_mean", "l_mean"}.issubset(df.columns):
            raise KeyError("El CSV no tiene r_mean/l_mean.")

        # Extraer columnas principales
        iters = df["iter"].values if "iter" in df.columns else np.arange(len(df))
        r_mean = df["r_mean"]  # Recompensa media
        l_mean = df["l_mean"]  # Longitud media

        # === DETECCIÓN DE COLUMNAS OPCIONALES ===
        # Verificar si hay columnas de razones de terminación
        reason_cols = ["reason_timeout_pct", "reason_starvation_pct", "reason_dehydration_pct"]
        has_reasons = all(col in df.columns for col in reason_cols)
        
        # Detectar columnas por agente (retorno)
        r_agent_cols = sorted([c for c in df.columns if c.startswith("r_agent_")],
                            key=lambda x: int(re.search(r"(\d+)$", x).group(1)))
        # Detectar columnas por agente (longitud)
        l_agent_cols = sorted([c for c in df.columns if c.startswith("l_agent_")],
                            key=lambda x: int(re.search(r"(\d+)$", x).group(1)))

        # === CALCULAR SUAVIZADOS ===
        # Ventana de suavizado dinámica: entre 5 y 20, o 1/3 de los datos
        SMOOTH_WINDOW = max(5, min(20, max(1, len(df)//3)))
        
        # Media móvil (suaviza oscilaciones)
        mov_avg_r = r_mean.rolling(SMOOTH_WINDOW, min_periods=1).mean()
        mov_avg_l = l_mean.rolling(SMOOTH_WINDOW, min_periods=1).mean()
        
        # Media acumulativa (tendencia general)
        cumavg_r = r_mean.expanding().mean()
        cumavg_l = l_mean.expanding().mean()

        # === GENERAR VISUALIZACIONES ===
        _create_plots(iters, r_mean, l_mean, mov_avg_r, mov_avg_l, cumavg_r, cumavg_l, 
                     r_agent_cols, l_agent_cols, df, SMOOTH_WINDOW, reason_cols, has_reasons)

        # === IMPRIMIR RESUMEN ESTADÍSTICO ===
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
    """
    Crea todas las visualizaciones de los resultados del entrenamiento.
    
    Genera 4 tipos de figuras:
    1. Métricas agregadas: recompensa y longitud con suavizados
    2. Retorno por agente individual
    3. Longitud por agente individual
    4. Porcentajes de razones de terminación
    
    Args:
        iters: Array de iteraciones
        r_mean, l_mean: Series de recompensa y longitud media
        mov_avg_r, mov_avg_l: Medias móviles
        cumavg_r, cumavg_l: Medias acumuladas
        r_agent_cols, l_agent_cols: Listas de nombres de columnas por agente
        df: DataFrame con todos los datos
        SMOOTH_WINDOW: Tamaño de ventana para suavizado
        reason_cols: Lista de nombres de columnas de razones
        has_reasons: Flag indicando si hay datos de razones
    """
    # === FIGURA 1: MÉTRICAS AGREGADAS ===
    plt.figure(figsize=(12, 8))

    # Subplot 1: Recompensa media
    plt.subplot(2, 2, 1)
    plt.plot(iters, r_mean, alpha=0.3, label="r_mean (raw)")  # Datos crudos translúcidos
    plt.plot(iters, mov_avg_r, label=f"r_mean mov.avg (w={SMOOTH_WINDOW})")  # Media móvil
    plt.title("Recompensa media por iteración")
    plt.xlabel("Iteración")
    plt.ylabel("Return")
    plt.legend()

    # Subplot 2: Longitud media
    plt.subplot(2, 2, 3)
    plt.plot(iters, l_mean, alpha=0.3, label="l_mean (raw)")
    plt.plot(iters, mov_avg_l, label=f"l_mean mov.avg (w={SMOOTH_WINDOW})")
    plt.title("Longitud media por iteración")
    plt.xlabel("Iteración")
    plt.ylabel("Timesteps")
    plt.legend()

    # Subplot 3: Retorno acumulado
    plt.subplot(2, 2, 2)
    plt.plot(iters, cumavg_r)
    plt.title("Return acumulado (promedio)")
    plt.xlabel("Iteración")
    plt.ylabel("Return")

    # Subplot 4: Longitud acumulada
    plt.subplot(2, 2, 4)
    plt.plot(iters, cumavg_l)
    plt.title("Longitud acumulada (promedio)")
    plt.xlabel("Iteración")
    plt.ylabel("Timesteps")

    plt.tight_layout()
    plt.show()

    # === FIGURA 2: RETORNO POR AGENTE ===
    if r_agent_cols:
        plt.figure(figsize=(12, 6))
        # Graficar cada agente con media móvil
        for c in r_agent_cols:
            series = df[c].rolling(SMOOTH_WINDOW, min_periods=1).mean()
            plt.plot(iters, series, label=c)
        plt.title("Return por agente (media móvil)")
        plt.xlabel("Iteración")
        plt.ylabel("Return")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # === FIGURA 3: LONGITUD POR AGENTE ===
    if l_agent_cols:
        plt.figure(figsize=(12, 6))
        # Graficar cada agente con media móvil
        for c in l_agent_cols:
            series = df[c].rolling(SMOOTH_WINDOW, min_periods=1).mean()
            plt.plot(iters, series, label=c)
        plt.title("Longitud por agente (media móvil)")
        plt.xlabel("Iteración")
        plt.ylabel("Timesteps")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # === FIGURA 4: RAZONES DE TERMINACIÓN ===
    if has_reasons:
        # Extraer series de razones
        timeout_pct, starvation_pct, dehydration_pct = [df[c] for c in reason_cols]
        
        plt.figure(figsize=(10,4))
        plt.plot(iters, timeout_pct,     label="timeout %")      # Azul: timeout
        plt.plot(iters, starvation_pct,  label="starvation %")   # Naranja: inanición
        plt.plot(iters, dehydration_pct, label="dehydration %")  # Verde: deshidratación
        plt.title("Razones de terminación de episodios")
        plt.xlabel("Iteración")
        plt.ylabel("Porcentaje")
        plt.legend()
        plt.tight_layout()
        plt.show()