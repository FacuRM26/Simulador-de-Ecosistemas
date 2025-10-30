"""
Script principal para ejecutar el entrenamiento y visualización de resultados.

Este script:
1. Pregunta al usuario si desea habilitar visualización en tiempo real
2. Ejecuta el entrenamiento del modelo RL multi-agente
3. Analiza y grafica los resultados del entrenamiento

Uso:
    python run_training.py
    
El script generará un archivo 'monitor.csv' con las métricas de entrenamiento
y mostrará gráficos al finalizar.
"""
import sys
import os

# Agregar el directorio actual al path de Python para importaciones
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ecosystem_simulator.training.trainer import main
from ecosystem_simulator.utils.visualization import analyze_training_results

if __name__ == "__main__":
    # === CONFIGURACIÓN DE VISUALIZACIÓN ===
    # Preguntar al usuario si desea ver el entrenamiento en tiempo real
    enable_viz = input("¿Habilitar visualización en tiempo real? (s/n): ").lower().strip() == 's'

    # === EJECUTAR ENTRENAMIENTO ===
    # Entrenar el modelo RL con o sin visualización según la elección del usuario
    main(enable_visualization=enable_viz)
    
    # === ANALIZAR Y GRAFICAR RESULTADOS ===
    # Leer el archivo CSV generado y crear gráficos de análisis
    analyze_training_results("monitor.csv")