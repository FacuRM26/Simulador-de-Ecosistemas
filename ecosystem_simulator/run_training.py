"""
Script principal para ejecutar el entrenamiento y visualización de resultados.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ecosystem_simulator.training.trainer import main
from ecosystem_simulator.utils.visualization import analyze_training_results

if __name__ == "__main__":
    # Ejecutar entrenamiento
    main()
    
    # Analizar resultados
    analyze_training_results("monitor.csv")