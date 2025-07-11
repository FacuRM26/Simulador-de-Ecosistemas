import os
os.environ.pop("AIR_VERBOSITY", None)
import random
import math
import logging

import numpy as np
from scipy.spatial import KDTree

import gymnasium as gym
from gymnasium.spaces import Discrete, Box

from pettingzoo.utils import ParallelEnv

import ray
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.env import PettingZooEnv
from ray.rllib.algorithms.ppo import PPOConfig

from typing import List, Dict, Tuple
import csv


class Specie:
    """
    Representa una especie en el simulador de ecosistemas con aprendizaje por refuerzo (RL).

    Cada instancia mantiene su nivel de alimento y agua, su posición en el mapa
    y métodos para metabolizar y moverse, respetando los límites del entorno.

    Atributos:
        food (float): Cantidad actual de alimento.
        water (float): Cantidad actual de agua.
        x (float): Coordenada X de la posición.
        y (float): Coordenada Y de la posición.
        max_food (float): Capacidad máxima de alimento.
        max_water (float): Capacidad máxima de agua.
        map_width (int): Ancho del entorno.
        map_height (int): Alto del entorno.

    Constantes de clase:
        AGENT_SIZE (int): Tamaño del agente en píxeles (para el clamp de posición).
        _DIR_VECTORS (dict): Vectores unitarios para cada dirección cardinal.
    """
    __slots__ = (
        "food", "water",
        "x", "y",
        "max_food", "max_water",
        "map_width", "map_height"
    )

    # Tamaño del agente en el entorno (usa para evitar salirse del mapa)
    AGENT_SIZE = 20

    # Vectores de dirección para movimiento
    _DIR_VECTORS = {
        "north": (0, -1),
        "south": (0,  1),
        "east":  (1,  0),
        "west":  (-1, 0),
    }

    def __init__(
        self,
        food: float,
        water: float,
        x: float = 0,
        y: float = 0,
        max_food: float = 100,
        max_water: float = 100,
        map_width: int = 800,
        map_height: int = 600,
    ):
        """
        Inicializa una nueva especie con recursos y posición dados.

        Args:
            food (float): Cantidad inicial de alimento (se recorta a max_food).
            water (float): Cantidad inicial de agua (se recorta a max_water).
            x (float): Posición inicial en X.
            y (float): Posición inicial en Y.
            max_food (float): Límite superior de alimento.
            max_water (float): Límite superior de agua.
            map_width (int): Ancho del mapa.
            map_height (int): Alto del mapa.
        """
        # Parámetros estáticos del entorno
        self.max_food   = max_food
        self.max_water  = max_water
        self.map_width  = map_width
        self.map_height = map_height

        # Estado inicial (se asegura que food & water no superen sus máximos)
        self.food  = min(food,  max_food)
        self.water = min(water, max_water)

        # Posición inicial
        self.x = x
        self.y = y

    @property
    def total_energy(self) -> float:
        """
        Calcula la energía total de la especie como promedio de alimento y agua.

        Returns:
            float: Energía media [(food + water) / 2].
        """
        return 0.5 * (self.food + self.water)

    def metabolize(self, rate: float = 0.3) -> None:
        """
        Aplica el gasto metabólico basal reduciendo recursos.

        Args:
            rate (float): Tasa de consumo de alimento y agua por paso.
                          Por defecto 0.3 unidades.
        """
        # Se asegura que no baje de 0.0
        self.food  = max(0.0, self.food  - rate)
        self.water = max(0.0, self.water - rate)

    def move(self, velocity: float, direction: str) -> None:
        """
        Desplaza la especie en la dirección indicada con una velocidad dada,
        y aplica clamp para no salirse del mapa.

        Args:
            velocity (float): Distancia a recorrer en este paso.
            direction (str): Una de "north", "south", "east" o "west".
        """
        # Obtiene el vector de dirección; si no existe, no se mueve.
        dx, dy = self._DIR_VECTORS.get(direction, (0, 0))
        self.x += dx * velocity
        self.y += dy * velocity

        # Máximos permitidos en X e Y (considerando el tamaño del agente)
        max_x = self.map_width  - self.AGENT_SIZE
        max_y = self.map_height - self.AGENT_SIZE

        # Clamp de coordenadas entre [0, max]
        self.x = min(max(self.x, 0.0), max_x)
        self.y = min(max(self.y, 0.0), max_y)

    def walk(self, velocity: float, direction: str) -> None:
        """
        Realiza un paso completo de la especie en el entorno:
          1. metabolize() → disminuye recursos.
          2. move()      → actualiza posición y aplica clamp.

        Args:
            velocity (float): Velocidad de movimiento.
            direction (str): Dirección de desplazamiento.
        """
        # Gasto metabólico
        self.metabolize()
        # Movimiento y restricción de posición
        self.move(velocity, direction)


import numpy as np
from scipy.spatial import KDTree

class Ecosystem:
    """
    Simulador de ecosistema con recursos de vegetación y agua.

    Los recursos se almacenan en arrays de NumPy y se indexan mediante KD-Trees
    para consultas eficientes de vecino más cercano.

    Atributos:
        map_width (int): Ancho del entorno en píxeles.
        map_height (int): Alto del entorno en píxeles.
        vegetation (dict): Datos de vegetación con campos "x", "y", "w", "h", "centers".
        water_sources (dict): Datos de fuentes de agua con estructura similar.
        _veg_tree (KDTree): KD-Tree construido sobre los centros de vegetación.
        _wat_tree (KDTree): KD-Tree construido sobre los centros de agua.
    """

    def __init__(
        self,
        veg_density: int,
        water_density: int,
        map_width: int = 800,
        map_height: int = 600
    ):
        """
        Inicializa el ecosistema generando los recursos y construyendo los KD-Trees.

        Args:
            veg_density (int): Número de parches de vegetación a generar.
            water_density (int): Número de fuentes de agua a generar.
            map_width (int): Ancho del mapa. Default: 800.
            map_height (int): Alto del mapa. Default: 600.
        """
        self.map_width  = map_width
        self.map_height = map_height

        # Generación vectorizada de recursos
        self.vegetation    = self._gen_resources_array(veg_density, size_min=20, size_max=20)
        self.water_sources = self._gen_resources_array(water_density, size_min=20, size_max=60)

        # Construcción de KD-Trees para búsquedas de vecino más cercano
        self._veg_tree = KDTree(self.vegetation["centers"])
        self._wat_tree = KDTree(self.water_sources["centers"])

    def _gen_resources_array(self, n: int, size_min: int, size_max: int) -> dict:
        """
        Genera un diccionario con arrays que describen n recursos aleatorios.

        Cada recurso tiene posición (x, y), tamaño (w, h) y centro calculado.

        Args:
            n (int): Número de recursos a generar.
            size_min (int): Tamaño mínimo (en píxeles) de ancho y alto.
            size_max (int): Tamaño máximo (en píxeles) de ancho y alto.

        Returns:
            dict: Contiene arrays NumPy para:
                - "x", "y": coordenadas de esquina superior izquierda.
                - "w", "h": ancho y alto de cada recurso.
                - "centers": array de forma (n, 2) con los puntos medios.
        """
        # Anchos y altos aleatorios dentro del rango
        w = np.random.randint(size_min, size_max + 1, size=n)
        h = np.random.randint(size_min, size_max + 1, size=n)

        # Posiciones aleatorias, evitando salirse del mapa
        x = np.random.randint(0, self.map_width  - size_max + 1, size=n)
        y = np.random.randint(0, self.map_height - size_max + 1, size=n)

        # Cálculo de centros para cada recurso
        centers = np.stack([x + w / 2, y + h / 2], axis=1)  # shape (n, 2)

        return {
            "x":       x,
            "y":       y,
            "w":       w,
            "h":       h,
            "centers": centers
        }

    def closest_vegetation(self, position: Tuple[float, float], k: int = 1):
        """
        Encuentra la(s) parcela(s) de vegetación más cercana(s) a una posición dada.

        Args:
            position (tuple): Coordenadas (x, y) de referencia.
            k (int): Número de vecinos más cercanos a consultar. Default: 1.

        Returns:
            tuple: (índices, distancias) de longitud k.
        """
        dist, idx = self._veg_tree.query(position, k=k)
        return idx, dist

    def closest_water(self, position: Tuple[float, float], k: int = 1):
        """
        Igual que closest_vegetation, pero para fuentes de agua.

        Args:
            position (tuple): Coordenadas (x, y) de referencia.
            k (int): Número de vecinos más cercanos. Default: 1.

        Returns:
            tuple: (índices, distancias) de longitud k.
        """
        dist, idx = self._wat_tree.query(position, k=k)
        return idx, dist

    def remove_vegetation(self, idx: int) -> None:
        """
        Elimina la parcela de vegetación en el índice dado y actualiza el KD-Tree.

        Args:
            idx (int): Índice del recurso a remover.
        """
        # Máscara booleana para mantener todos excepto idx
        mask = np.arange(self.vegetation["x"].shape[0]) != idx
        for key in ("x", "y", "w", "h", "centers"):
            self.vegetation[key] = self.vegetation[key][mask]
        # Reconstrucción del árbol
        self._veg_tree = KDTree(self.vegetation["centers"])

    def remove_water(self, idx: int) -> None:
        """
        Elimina la fuente de agua en el índice dado y actualiza el KD-Tree.

        Args:
            idx (int): Índice del recurso a remover.
        """
        mask = np.arange(self.water_sources["x"].shape[0]) != idx
        for key in ("x", "y", "w", "h", "centers"):
            self.water_sources[key] = self.water_sources[key][mask]
        self._wat_tree = KDTree(self.water_sources["centers"])

    def collide_resources(self, position: Tuple[float, float], size: float, resources: dict) -> np.ndarray:
        """
        Detecta colisiones entre un agente cuadrado y un conjunto de recursos.

        Se usa un chequeo AABB (Axis-Aligned Bounding Box) entre el agente centrado
        en `position` de lado `size` y cada recurso listado en el dict `resources`.

        Args:
            position (tuple): Coordenadas (x, y) del centro del agente.
            size (float): Longitud del lado del agente.
            resources (dict): Estructura de recursos (igual a vegetation o water_sources).

        Returns:
            np.ndarray: Índices de los recursos que colisionan.
        """
        cx, cy = position
        half = size / 2

        # Desplazamientos absolutos en X e Y entre centros
        dx = np.abs(resources["centers"][:, 0] - cx)
        dy = np.abs(resources["centers"][:, 1] - cy)
        half_w = resources["w"] / 2
        half_h = resources["h"] / 2

        # Colisión AABB: distancia menor que suma de semianchos/altos
        hits = (dx <= half_w + half) & (dy <= half_h + half)
        return np.nonzero(hits)[0]



logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # o INFO para menos verbosidad
# multi_agent_ecosystem.py


import ray
from ray import tune
from ray.tune.registry import register_env
# … tus otros imports de RLlib …
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv

DIRECTIONS = ["north", "south", "east", "west"]

class MultiAgentEcosystem(ParallelEnv, Ecosystem):
    metadata = {"render_modes": [], "name": "multi_agent_eco"}
    
    def __init__(
        self,
        n_agents: int = 2,
        veg_density: int = 15,
        water_density: int = 10,
        map_width: int = 800,
        map_height: int = 600,
        max_steps: int = 350,
        gamma: float = 0.995,
    ):
        ParallelEnv.__init__(self)
        Ecosystem.__init__(self, veg_density, water_density, map_width, map_height)

        # --- Parámetros multi-agente ---
        self.n_agents  = n_agents
        self.agents    = [f"agent_{i}" for i in range(n_agents)]
        self.possible_agents = list(self.agents)
        # mapa rápido de agente → índice en self.species
        self._agent_idx = {a: i for i, a in enumerate(self.agents)}

        self.max_steps = max_steps
        self.gamma     = gamma

        # espacios
        low  = np.array([0.0, 0.0, -1.0, -1.0, -1.0, -1.0], dtype=np.float32)
        high = np.array([1.0, 1.0,  1.0,  1.0,  1.0,  1.0], dtype=np.float32)
        self.action_spaces      = {a: Discrete(4)                    for a in self.agents}
        self.observation_spaces = {a: Box(low, high, dtype=np.float32) for a in self.agents}

        self._step_count = 0
        self._reset_species()
    
    def _reset_species(self):
        """ (re)crea la lista de Specie """
        self.species: List[Specie] = [
            Specie(food=100, water=100,
                   x=self.map_width/2, y=self.map_height/2,
                   map_width=self.map_width, map_height=self.map_height)
            for _ in range(self.n_agents)
        ]

    def observation_space(self, agent: str) -> gym.Space:
        """PettingZoo/rllib llamarán a esto para cada agente."""
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> gym.Space:
        """PettingZoo/rllib llamarán a esto para cada agente."""
        return self.action_spaces[agent]

    def reset(self, *, seed: int = None, options: dict = None):
            # 1) Si quieres reproducibilidad, fija tus semillas
            if seed is not None:
                import random, numpy as np
                random.seed(seed)
                np.random.seed(seed)

            # 2) Regenera recursos y especies
            Ecosystem.__init__(
                self,
                veg_density=len(self.vegetation["x"]),
                water_density=len(self.water_sources["x"]),
                map_width=self.map_width,
                map_height=self.map_height
            )
            self._reset_species()
            self._step_count = 0

            # 3) Construye el dict de observaciones
            observations = {
                agent: self._get_obs(self.species[i])
                for i, agent in enumerate(self.agents)
            }

            # 4) Prepara un dict vacío de infos (o rellénalo si lo necesitas)
            infos = {agent: {} for agent in self.agents}

            # 5) Devuelve ambos valores
            return observations, infos

    def step(
        self,
        actions: Dict[str, int]
    ) -> Tuple[
        Dict[str, np.ndarray],  # obs
        Dict[str, float],       # rewards
        Dict[str, bool],        # terminations
        Dict[str, bool],        # truncations
        Dict[str, dict]         # infos
    ]:
        obs, rewards = {}, {}
        terminations, truncations, infos = {}, {}, {}
        self._step_count += 1

        # Recorremos SOLO los agentes que realmente recibieron acción.
        for agent, action in actions.items():
            i  = self._agent_idx[agent]
            sp = self.species[i]
            prev_energy = sp.total_energy

            # Movimiento + metabolizar
            sp.walk(velocity=2.0, direction=DIRECTIONS[action])

            # Recompensas por consumos
            reward = -0.05
            idxs = self.collide_resources((sp.x, sp.y), Specie.AGENT_SIZE, self.water_sources)
            if idxs.size:
                sp.water = min(sp.water + 10, sp.max_water)
                self.remove_water(idxs[0])
                reward += 3

            idxs = self.collide_resources((sp.x, sp.y), Specie.AGENT_SIZE, self.vegetation)
            if idxs.size:
                sp.food = min(sp.food + 10, sp.max_food)
                self.remove_vegetation(idxs[0])
                reward += 3

            # Shaping de energía
            new_energy = sp.total_energy
            reward += max(0, self.gamma * new_energy - prev_energy)

            # Flags de terminación
            is_dead      = sp.food <= 0 or sp.water <= 0
            is_truncated = self._step_count >= self.max_steps
            if is_dead:
                reward -= 10

            # Guardar salidas
            obs[agent]           = self._get_obs(sp)
            rewards[agent]       = float(np.clip(reward, -10, 60))
            terminations[agent]  = is_dead
            truncations[agent]   = is_truncated
            if sp.food <= 0:
                reason = "starvation"
            elif sp.water <= 0:
                reason = "dehydration"
            elif self._step_count >= self.max_steps:
                reason = "timeout"
            else:
                reason = ""
            infos[agent] = {"reason": reason}


        # "__all__" para ambos
        terminations["__all__"] = any(terminations.values())
        truncations["__all__"]  = any(truncations.values())

        return obs, rewards, terminations, truncations, infos


    def _get_obs(self, sp: Specie) -> np.ndarray:
        """
        Observación para un agente concreto `sp`.
        [food_norm, water_norm, dx_w, dy_w, dx_v, dy_v]
        dx/dy = 0 si no hay recurso.
        """
        # Recursos de agua
        if self.water_sources["centers"].shape[0] > 0:
            _, idx_w = self._wat_tree.query([sp.x, sp.y], k=1)
            wx, wy = self.water_sources["centers"][idx_w]
            dx_w = (wx - sp.x) / self.map_width
            dy_w = (wy - sp.y) / self.map_height
        else:
            dx_w, dy_w = 0.0, 0.0

        # Recursos de vegetación
        if self.vegetation["centers"].shape[0] > 0:
            _, idx_v = self._veg_tree.query([sp.x, sp.y], k=1)
            vx, vy = self.vegetation["centers"][idx_v]
            dx_v = (vx - sp.x) / self.map_width
            dy_v = (vy - sp.y) / self.map_height
        else:
            dx_v, dy_v = 0.0, 0.0

        # Normalización de recursos
        f_norm = sp.food / sp.max_food
        w_norm = sp.water / sp.max_water

        return np.array([f_norm, w_norm, dx_w, dy_w, dx_v, dy_v],
                        dtype=np.float32)



    from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv

if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)
    rewards_history = []
    register_env(
        "multi_eco",
        lambda cfg: ParallelPettingZooEnv(MultiAgentEcosystem(**cfg))
    )

    # 2) define los espacios aquí
    obs_space = Box(0.0, 1.0, shape=(6,), dtype=np.float32)
    act_space = Discrete(4)

    # 3) configura PPO
    config = (
        PPOConfig()
        .environment(
            env="multi_eco",
            env_config={
                "n_agents": 4,
                "veg_density": 15,
                "water_density": 10,
                "map_width": 800,
                "map_height": 600,
            }
        )
        .framework("torch")
        .multi_agent(
        policies={
            "shared_policy": (None, obs_space, act_space, {})
        },
        # Ahora aceptamos `agent_id` y `episode` (y cualquier otro kwarg)
        policy_mapping_fn=lambda agent_id, episode, **kwargs: "shared_policy"
        )
        .env_runners(
        num_env_runners=2,
        rollout_fragment_length=20,
        sample_timeout_s=60
    )
)
    trainer = config.build()
    # — Abrimos el CSV una sola vez para TODO el entrenamiento —
    with open("monitor.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iter", "r", "l", "reason"])

        for i in range(10):
            result = trainer.train()
            r = result.get("episode_reward_mean", 0.0)
            l = result.get("episode_len_mean", 0.0)
            reason = "timeout" if result.get("episodes_this_iter", 0) == 0 else ""
            writer.writerow([i, r, l, reason])

            # imprimir por pantalla
            avg = result.get("episode_reward_mean",
                             result.get("metrics", {}).get("episode_reward_mean"))
            print(f"Iter {i}:\t episode_reward_mean = {avg}")


    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    # Parámetros de análisis
    MONITOR_CSV = "monitor.csv"
    SMOOTH_WINDOW = 50  # ventana de media móvil
    START_EP = SMOOTH_WINDOW  # episodio inicial para graficar media móvil

    try:
        # 1) Cargar log de entrenamiento
        df = pd.read_csv(MONITOR_CSV, comment="#")
        
        # 2) Métricas globales
        metrics = {
            "metric": ["reward_mean", "reward_std", "reward_median", "reward_q1", "reward_q3",
                    "length_mean", "length_std", "length_median", "length_q1", "length_q3",
                    "timeout_pct", "starvation_pct", "dehydration_pct"],
            "value": [
                df["r"].mean(), df["r"].std(), df["r"].median(),
                np.percentile(df["r"], 25), np.percentile(df["r"], 75),
                df["l"].mean(), df["l"].std(), df["l"].median(),
                np.percentile(df["l"], 25), np.percentile(df["l"], 75),
                (df["reason"] == "timeout").mean() * 100,
                (df["reason"] == "starvation").mean() * 100,
                (df["reason"] == "dehydration").mean() * 100
            ]
        }
        summary_df = pd.DataFrame(metrics)
        # Mostrar las métricas globales al usuario

        # 3) Suavizados y acumulados
        episodes = df.index.values
        rewards = df["r"].values
        lengths = df["l"].values

        mov_avg_r = pd.Series(rewards).rolling(SMOOTH_WINDOW, min_periods=1).mean().values
        mov_avg_l = pd.Series(lengths).rolling(SMOOTH_WINDOW, min_periods=1).mean().values
        cumavg_r = np.cumsum(rewards) / (np.arange(len(rewards)) + 1)
        cumavg_l = np.cumsum(lengths) / (np.arange(len(lengths)) + 1)

        # 4) Graficar
        plt.figure(figsize=(12, 8))

        # Recompensa por episodio y media móvil
        plt.subplot(2, 2, 1)
        plt.plot(episodes, rewards, alpha=0.2, label="Raw reward")
        plt.plot(episodes[START_EP:], mov_avg_r[START_EP:], label=f"Mov. avg (w={SMOOTH_WINDOW})")
        plt.title("Recompensa por episodio")
        plt.xlabel("Episodio")
        plt.ylabel("Recompensa")
        plt.legend()

        # Promedio acumulado de recompensa
        plt.subplot(2, 2, 2)
        plt.plot(episodes, cumavg_r)
        plt.title("Promedio acumulado de recompensa")
        plt.xlabel("Episodio")
        plt.ylabel("Mean Reward")

        # Duración por episodio y media móvil
        plt.subplot(2, 2, 3)
        plt.plot(episodes, lengths, alpha=0.2, label="Raw length")
        plt.plot(episodes[START_EP:], mov_avg_l[START_EP:], label=f"Mov. avg (w={SMOOTH_WINDOW})")
        plt.title("Duración por episodio")
        plt.xlabel("Episodio")
        plt.ylabel("Timesteps")
        plt.legend()

        # Promedio acumulado de duración
        plt.subplot(2, 2, 4)
        plt.plot(episodes, cumavg_l)
        plt.title("Promedio acumulado de duración")
        plt.xlabel("Episodio")
        plt.ylabel("Mean Length")

        plt.tight_layout()
        plt.show()

    except FileNotFoundError:
        print(f"No se encontró el archivo '{MONITOR_CSV}'. Genera antes el CSV de monitorización.")
