ENV_CFG = {
    "n_agents": 8,
    "veg_density": 20,
    "water_density": 14,
    "map_width": 800,
    "map_height": 600,
    "max_steps": 350,
    "n_predators": 2,
}

# Configuración de cada entrenamiento
NUM_RUNNERS = 8
NUM_ITERS = 2000

# Configuración de múltiples experimentos
NUM_RUNS = 3
BASE_SEED = 1005

# Cantidad de iteraciones finales que se promedian
# dentro de cada ejecución.
TAIL_ITERS = 100

# Iteración desde la que se visualiza en Godot.
VIS_START_IT = 2001
