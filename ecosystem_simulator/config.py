ENV_CFG = {
    "n_agents": 8,
    # Más densidad de recursos (veg 15->20, agua 10->14): con la densidad anterior
    # los parches quedaban muy separados y los herbívoros casi no encontraban
    # comida/agua. Un ambiente de pastoreo más sostenible y fácil de encontrar.
    "veg_density": 20,
    "water_density": 14,
    "map_width": 800,
    "map_height": 600,
    "max_steps": 350,
    "n_predators": 2,
}
NUM_RUNNERS = 4
NUM_ITERS = 1000

# Iteración a partir de la cual se envía estado a Godot.
VIS_START_IT = 980