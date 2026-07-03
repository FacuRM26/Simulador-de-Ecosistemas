ENV_CFG = {
    "n_agents": 8,
    "veg_density": 15,
    "water_density": 10,
    "map_width": 800,
    "map_height": 600,
    "max_steps": 350,
    "n_predators": 2,
}
NUM_RUNNERS = 4
NUM_ITERS = 500
# Iteración a partir de la cual se envía estado a Godot.
# Ponlo bajo (p.ej. 30) para ir viendo cómo aprenden desde temprano;
# súbelo (p.ej. 400) solo para una demo final ya entrenada.
VIS_START_IT = 450