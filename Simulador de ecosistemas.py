import os
os.environ.pop("AIR_VERBOSITY", None)
import random
import math
import logging
import re
import random, numpy as np
from scipy.spatial import KDTree
import GPUtil
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
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.policy.policy import PolicySpec

import pandas as pd
import matplotlib.pyplot as plt

from ray.rllib.callbacks.callbacks import RLlibCallback

from ray.rllib.callbacks.callbacks import RLlibCallback

class Specie:
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
        "stay":  (0,  0),
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
        return 0.5 * (self.food + self.water)

    def metabolize(self, rate: float = 0.3) -> None:
        # Se asegura que no baje de 0.0
        self.food  = max(0.0, self.food  - rate)
        self.water = max(0.0, self.water - rate)

    def move(self, velocity: float, direction: str) -> None:
        dx, dy = self._DIR_VECTORS.get(direction, (0, 0))
        self.x += dx * velocity
        self.y += dy * velocity
        half = self.AGENT_SIZE / 2
        self.x = min(max(self.x, half), self.map_width  - half)
        self.y = min(max(self.y, half), self.map_height - half)

    def walk(self, velocity: float, direction: str) -> None:
        # Gasto metabólico
        self.metabolize()
        # Movimiento y restricción de posición
        self.move(velocity, direction)



class Ecosystem:

    def __init__(
        self,
        veg_density: int,
        water_density: int,
        map_width: int = 800,
        map_height: int = 600
    ):
        self.map_width  = map_width
        self.map_height = map_height

        # Generación vectorizada de recursos
        self.vegetation = self._gen_resources_array(veg_density, size_min=25, size_max=45)
        self.water_sources = self._gen_resources_array(water_density, size_min=20, size_max=60)

        # Construcción de KD-Trees para búsquedas de vecino más cercano
        self._veg_tree = KDTree(self.vegetation["centers"])
        self._wat_tree = KDTree(self.water_sources["centers"])
        self._init_veg = veg_density
        self._init_wat = water_density

    def _gen_resources_array(self, n: int, size_min: int, size_max: int) -> dict:
        # Anchos y altos aleatorios dentro del rango
        w = np.random.randint(size_min, size_max + 1, size=n)
        h = np.random.randint(size_min, size_max + 1, size=n)

        # Posiciones aleatorias, evitando salirse del mapa
        x = np.random.randint(0, self.map_width  - size_max + 1, size=n)
        y = np.random.randint(0, self.map_height - size_max + 1, size=n)

        # Cálculo de centros para cada recurso
        centers = np.stack([x + w / 2, y + h / 2], axis=1)  # shape (n, 2)
        charges = np.full(n, 3, dtype=np.int32)  # p.ej., 3 bocados
        return {"x": x, "y": y, "w": w, "h": h, "centers": centers, "charges": charges}

    def closest_vegetation(self, position: Tuple[float, float], k: int = 1):

        dist, idx = self._veg_tree.query(position, k=k)
        return idx, dist

    def closest_water(self, position: Tuple[float, float], k: int = 1):

        dist, idx = self._wat_tree.query(position, k=k)
        return idx, dist

    def consume_water(self, idx: int) -> bool:
        # bounds check por seguridad
        n = self.water_sources["charges"].shape[0]
        if idx < 0 or idx >= n:
            return False
        self.water_sources["charges"][idx] -= 1
        return self.water_sources["charges"][idx] <= 0  # True si quedó sin carga

    def consume_vegetation(self, idx: int) -> bool:
        n = self.vegetation["charges"].shape[0]
        if idx < 0 or idx >= n:
            return False
        self.vegetation["charges"][idx] -= 1
        return self.vegetation["charges"][idx] <= 0

    def collide_resources(self, position: Tuple[float, float], size: float, resources: dict) -> np.ndarray:
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
    def spawn_vegetation(self, n: int = 1, size_min: int = 25, size_max: int = 45) -> None:
        w = np.random.randint(size_min, size_max + 1, size=n)
        h = np.random.randint(size_min, size_max + 1, size=n)
        x = np.random.randint(0, self.map_width  - size_max + 1, size=n)
        y = np.random.randint(0, self.map_height - size_max + 1, size=n)
        centers = np.stack([x + w / 2, y + h / 2], axis=1)
        charges = np.full(n, 3, dtype=np.int32)                 

        for k, arr in (("x", x), ("y", y), ("w", w), ("h", h)):
            self.vegetation[k] = np.concatenate([self.vegetation[k], arr])
        self.vegetation["centers"] = np.vstack([self.vegetation["centers"], centers])
        self.vegetation["charges"] = np.concatenate([self.vegetation["charges"], charges])
        self._veg_tree = KDTree(self.vegetation["centers"])

    def spawn_water(self, n: int = 1, size_min: int = 20, size_max: int = 60):
        w = np.random.randint(size_min, size_max + 1, size=n)
        h = np.random.randint(size_min, size_max + 1, size=n)
        x = np.random.randint(0, self.map_width  - size_max + 1, size=n)
        y = np.random.randint(0, self.map_height - size_max + 1, size=n)
        centers = np.stack([x + w/2, y + h/2], axis=1)
        charges = np.full(n, 3, dtype=np.int32)              
        for k, arr in (("x", x), ("y", y), ("w", w), ("h", h)):
            self.water_sources[k] = np.concatenate([self.water_sources[k], arr])
        self.water_sources["centers"] = np.vstack([self.water_sources["centers"], centers])
        self.water_sources["charges"] = np.concatenate([self.water_sources["charges"], charges])  
        self._wat_tree = KDTree(self.water_sources["centers"])



logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # o INFO para menos verbosidad
# multi_agent_ecosystem.py




DIRECTIONS = ["north", "south", "east", "west", "stay"]

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
        base_w, base_h = 800, 600
        area_scale = (map_width * map_height) / (base_w * base_h)
        veg_density_scaled   = max(1, int(round(veg_density   * area_scale)))
        water_density_scaled = max(1, int(round(water_density * area_scale)))
        Ecosystem.__init__(self, veg_density_scaled, water_density_scaled, map_width, map_height)

        # --- Parámetros multi-agente ---
        self.n_agents  = n_agents
        self.agents    = [f"agent_{i}" for i in range(n_agents)]
        self.possible_agents = list(self.agents)
        # mapa rápido de agente → índice en self.species
        self._agent_idx = {a: i for i, a in enumerate(self.agents)}
        self.max_steps = max_steps
        self.gamma     = gamma
    
        # espacios
        SQRT2 = np.sqrt(2.0)
        low  = np.array([0,0,-1,-1,-1,-1, 0,   0,    0,0], dtype=np.float32)
        high = np.array([1,1,  1,  1,  1,  1, SQRT2, SQRT2, 1,1], dtype=np.float32)
        self.action_spaces = {a: Discrete(len(DIRECTIONS)) for a in self.agents}
        self.observation_spaces = {a: Box(low, high, dtype=np.float32) for a in self.agents}
        self._step_count = 0
        self._reset_species()
        self.success_thr = 0.90     # 90% de reservas
        self.success_hold = 25      # mantenerlo 25 steps seguidos
    
    def _reset_species(self):
        spawn_xy = self._sample_spawn_positions(self.n_agents, min_dist=80.0, avoid_resources=True)

        # arranque entre 60% y 80% del máximo
        def init_level(max_val): 
            return float(np.random.uniform(0.60, 0.80) * max_val)

        self.species = [
            Specie(
                food=init_level(100), water=init_level(100),
                x=spawn_xy[i][0], y=spawn_xy[i][1],
                map_width=self.map_width, map_height=self.map_height
            )
            for i in range(self.n_agents)
        ]
        

    def observation_space(self, agent: str) -> gym.Space:
        """PettingZoo/rllib llamarán a esto para cada agente."""
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> gym.Space:
        """PettingZoo/rllib llamarán a esto para cada agente."""
        return self.action_spaces[agent]

    def reset(self, *, seed: int = None, options: dict = None):
        if seed is not None:
            random.seed(seed); np.random.seed(seed)
        Ecosystem.__init__(self, self._init_veg, self._init_wat, self.map_width, self.map_height)
        self._reset_species()
        self._step_count = 0

        # REPUEBLA los agentes vivos del nuevo episodio:
        self._ep_return = {a: 0.0 for a in self.possible_agents}
        self._ep_len    = {a: 0   for a in self.possible_agents}
        self.agents = list(self.possible_agents)

        observations = {agent: self._get_obs(self.species[i]) for i, agent in enumerate(self.agents)}
        infos = {agent: {} for agent in self.agents}
        #print("reset -> agents:", self.agents)
        self._satiated = {a: 0 for a in self.agents} 
        return observations, infos
    
    def _norm_dist_to(self, sp: Specie, kind: str) -> float:
        tree = self._wat_tree if kind == "water" else self._veg_tree
        src  = self.water_sources if kind == "water" else self.vegetation
        if tree is None or src["centers"].shape[0] == 0:
            return 1.0
        _, idx = tree.query([sp.x, sp.y], k=1)
        cx, cy = src["centers"][idx]
        d = ((cx - sp.x)**2 + (cy - sp.y)**2)**0.5
        return d / ((self.map_width**2 + self.map_height**2)**0.5)
    
    def _prune_depleted_resources(self):
        # --- Agua ---
        if self.water_sources["charges"].size:
            mask_w = self.water_sources["charges"] > 0
            for k, arr in self.water_sources.items():
                # centers suele ser (N,2); el resto (N,)
                self.water_sources[k] = arr[mask_w] if arr.ndim == 1 else arr[mask_w, :]
            self._wat_tree = (
                KDTree(self.water_sources["centers"])
                if self.water_sources["centers"].size
                else None
            )

        # --- Vegetación ---
        if self.vegetation["charges"].size:
            mask_v = self.vegetation["charges"] > 0
            for k, arr in self.vegetation.items():
                self.vegetation[k] = arr[mask_v] if arr.ndim == 1 else arr[mask_v, :]
            self._veg_tree = (
                KDTree(self.vegetation["centers"])
                if self.vegetation["centers"].size
                else None
            )

    
    def step(self, actions):
        obs, rewards, terminations, truncations, infos = {}, {}, {}, {}, {}
        self._step_count += 1

        prev_agents   = list(self.agents)
        alive         = []
        episode_trunc = (self._step_count >= self.max_steps)

        # Acumuladores por step
        veg_claims, wat_claims = {}, {}
        step_bonus = {a: 0.0 for a in prev_agents}
        energy_after_move = {}

        # Procesar agentes en orden aleatorio
        proc_order = random.sample(prev_agents, len(prev_agents))

        for agent in proc_order:
            i  = self._agent_idx[agent]
            sp = self.species[i]

            reward = 0.0
            done_term  = False
            done_trunc = False
            reason     = ""

            if agent in actions:
                prev_energy = sp.total_energy

                # Necesidad dominante antes de mover
                f_def   = 1.0 - (sp.food  / sp.max_food)
                w_def   = 1.0 - (sp.water / sp.max_water)
                need    = "water" if w_def > f_def else "veg"
                need_def = max(f_def, w_def)
                d_prev  = self._norm_dist_to(sp, need)

                act_dir_name = DIRECTIONS[actions[agent]]
                BASE_COST, MOVE_EXTRA = 0.30, 0.02  # stay y move comparten base

                if act_dir_name == "stay":
                    sp.metabolize(BASE_COST)
                    sp.move(0.0, "stay")
                    reward -= 0.01 + 0.05 * need_def  # impuesto por inacción
                else:
                    sp.metabolize(BASE_COST + MOVE_EXTRA)
                    sp.move(2.0, act_dir_name)

                # Progreso hacia recurso necesario
                d_now = self._norm_dist_to(sp, need)
                reward += 2.0 * (d_prev - d_now)

                # Shaping por energía tras movimiento
                energy_after_move[agent] = sp.total_energy
                reward += 0.1 * (energy_after_move[agent] - prev_energy)

                # Reclamos
                hit = self.collide_resources((sp.x, sp.y), Specie.AGENT_SIZE, self.water_sources)
                if hit.size:
                    wat_claims.setdefault(int(hit[0]), []).append(agent)
                hit = self.collide_resources((sp.x, sp.y), Specie.AGENT_SIZE, self.vegetation)
                if hit.size:
                    veg_claims.setdefault(int(hit[0]), []).append(agent)

                # ¿Murió?
                if (sp.food <= 0) or (sp.water <= 0):
                    reward   -= 10.0
                    done_term = True
                    reason    = "starvation" if sp.food <= 0 else "dehydration"

                # Bonus de “homeostasis”, PERO SIN TERMINAR EL AGENTE
                if not done_term:
                    if sp.food >= self.success_thr * sp.max_food and sp.water >= self.success_thr * sp.max_water:
                        self._satiated[agent] += 1
                        if self._satiated[agent] >= self.success_hold:
                            reward += 30.0
                            # resetea la racha para no dar el bonus cada step
                            self._satiated[agent] = 0
                    else:
                        self._satiated[agent] = 0

                # Truncation individual por timeout (solo marca; el cierre global lo hacemos abajo)
                if (not done_term) and episode_trunc:
                    done_trunc = True
                    reason = "timeout"

            # Shaping de salud (si sigue vivo)
            if not done_term:
                f_norm = sp.food / sp.max_food
                w_norm = sp.water / sp.max_water
                min_norm = min(f_norm, w_norm)
                reward += 0.02 * min_norm
                if min_norm < 0.15: reward -= 0.5
                if min_norm < 0.08: reward -= 1.0

            rewards[agent]      = float(reward)
            terminations[agent] = done_term
            truncations[agent]  = done_trunc
            infos[agent]        = {"reason": reason}

            if (not done_term) and (not done_trunc):
                alive.append(agent)

        # Resolver reclamos por NECESIDAD
        removed_w = 0
        for idx, claimers in wat_claims.items():
            vivos = [a for a in claimers if not terminations.get(a, False)]
            if not vivos:
                continue
            winner = min(vivos, key=lambda a: self.species[self._agent_idx[a]].water /
                                        self.species[self._agent_idx[a]].max_water)
            sp_w = self.species[self._agent_idx[winner]]
            sp_w.water = min(sp_w.water + 10, sp_w.max_water)
            step_bonus[winner] += 5.0
            # ↓ solo decrementa charges; NO elimines aquí
            if self.consume_water(idx):
                removed_w += 1

        removed_v = 0
        for idx, claimers in veg_claims.items():
            vivos = [a for a in claimers if not terminations.get(a, False)]
            if not vivos:
                continue
            winner = min(vivos, key=lambda a: self.species[self._agent_idx[a]].food /
                                        self.species[self._agent_idx[a]].max_food)
            sp_w = self.species[self._agent_idx[winner]]
            sp_w.food = min(sp_w.food + 10, sp_w.max_food)
            step_bonus[winner] += 8.0
            if self.consume_vegetation(idx):
                removed_v += 1

        # Añade el bonus del consumo y shaping post-consumo
        for a in prev_agents:
            rewards[a] = float(rewards.get(a, 0.0) + step_bonus.get(a, 0.0))
        for a in prev_agents:
            if (a in energy_after_move) and (not terminations[a]):
                post = self.species[self._agent_idx[a]].total_energy
                rewards[a] += 0.1 * (post - energy_after_move[a])
        # Reposición (si no es timeout global)
        if not episode_trunc:
            target_veg = self._init_veg
            target_wat = self._init_wat
            cur_veg = self.vegetation["x"].shape[0]
            cur_wat = self.water_sources["x"].shape[0]
            spawn_veg = max(0, min(max(1, removed_v // 2), target_veg - cur_veg))
            spawn_wat = max(0, min(max(1, removed_w // 2), target_wat - cur_wat))
            if spawn_veg:
                self.spawn_vegetation(n=spawn_veg)
            if spawn_wat:
                self.spawn_water(n=spawn_wat)

        # ✅ Eliminar en lote los agotados y reconstruir árboles
        self._prune_depleted_resources()

        # Observaciones + métricas acumuladas
        for agent in prev_agents:
            obs[agent] = self._get_obs(self.species[self._agent_idx[agent]])
            infos.setdefault(agent, {})
            self._ep_return[agent] = self._ep_return.get(agent, 0.0) + rewards.get(agent, 0.0)
            self._ep_len[agent]    = self._ep_len.get(agent, 0) + 1
            infos[agent].setdefault("ep_return", self._ep_return[agent])
            infos[agent].setdefault("ep_len",    self._ep_len[agent])

        # --- Terminación global (mixta):
        # A) timeout: todos los vivos quedan truncated y cerramos episodio
        if episode_trunc and len(alive) > 0:
            for a in alive:
                truncations[a] = True
                infos[a]["reason"] = "timeout"
            alive = []

        # B) extinción: si no queda nadie vivo, episodio termina
        # (esto ya ocurre al dejar alive=[], pero lo dejamos explícito)
        self.agents = alive
        return obs, rewards, terminations, truncations, infos


    def _get_obs(self, sp: Specie) -> np.ndarray:
        w_avail = float(self._wat_tree is not None and self.water_sources["centers"].size > 0)
        v_avail = float(self._veg_tree is not None and self.vegetation["centers"].size > 0)

        if w_avail:
            _, idx_w = self._wat_tree.query([sp.x, sp.y], k=1)
            wx, wy = self.water_sources["centers"][idx_w]
            dx_w, dy_w = (wx - sp.x)/self.map_width, (wy - sp.y)/self.map_height
            dist_w = (dx_w**2 + dy_w**2)**0.5
        else:
            dx_w = dy_w = 0.0
            dist_w = 1.0

        if v_avail:
            _, idx_v = self._veg_tree.query([sp.x, sp.y], k=1)
            vx, vy = self.vegetation["centers"][idx_v]
            dx_v, dy_v = (vx - sp.x)/self.map_width, (vy - sp.y)/self.map_height
            dist_v = (dx_v**2 + dy_v**2)**0.5
        else:
            dx_v = dy_v = 0.0
            dist_v = 1.0

        f_norm = sp.food / sp.max_food
        w_norm = sp.water / sp.max_water

        return np.array([f_norm, w_norm, dx_w, dy_w, dx_v, dy_v, dist_w, dist_v, w_avail, v_avail],
                        dtype=np.float32)
    
    def _sample_spawn_positions(
        self,
        n: int,
        min_dist: float = 80.0,        # distancia mínima entre agentes
        avoid_resources: bool = True,   # evitar spawnear sobre agua/vegetación
        max_tries: int = 5000,
    ):
        half = Specie.AGENT_SIZE / 2
        positions = []

        for k in range(n):
            ok = False
            tries = 0
            while (not ok) and tries < max_tries:
                tries += 1
                x = np.random.uniform(half, self.map_width  - half)
                y = np.random.uniform(half, self.map_height - half)

                # 1) Evitar recursos (opcional)
                if avoid_resources:
                    if self._veg_tree is not None:
                        if self.collide_resources((x, y), Specie.AGENT_SIZE, self.vegetation).size:
                            continue
                    if self._wat_tree is not None:
                        if self.collide_resources((x, y), Specie.AGENT_SIZE, self.water_sources).size:
                            continue

                # 2) Separación mínima entre agentes
                too_close = any(
                    ( (x - px)**2 + (y - py)**2 )**0.5 < min_dist
                    for (px, py) in positions
                )
                if too_close:
                    continue

                positions.append((x, y))
                ok = True

            # Si no encontró lugar (muy raro), relajá restricciones:
            if not ok:
                positions.append((self.map_width/2 + k*half, self.map_height/2))

        return positions



class PerAgentAndReasonMetrics(RLlibCallback):
    def __init__(self):
        super().__init__()
        self._ep_state = {}  # episode_id -> {agent_id: {"ret": float, "len": int, "reason": str|None}}

    def _eid(self, episode):
        # Compatible con distintas versiones
        return getattr(episode, "id_", None) or getattr(episode, "episode_id", None) or id(episode)

    def on_episode_start(self, *, episode, **kwargs):
        self._ep_state[self._eid(episode)] = {}

    def on_episode_step(self, *, episode, **kwargs):
        eid = self._eid(episode)
        per_agent = self._ep_state.setdefault(eid, {})
        step_infos = episode.get_infos(-1) or {}  # dict: agent_id -> info (tu env ya escribe ep_return/ep_len)

        for agent_id, info in step_infos.items():
            if not info:
                continue
            rec = per_agent.setdefault(agent_id, {"ret": 0.0, "len": 0, "reason": None})
            if "ep_return" in info:
                rec["ret"] = float(info["ep_return"])
            if "ep_len" in info:
                rec["len"] = int(info["ep_len"])
            r = info.get("reason")
            if r:
                rec["reason"] = r

    def on_episode_end(self, *, episode, metrics_logger, **kwargs):
        eid = self._eid(episode)
        per_agent = self._ep_state.pop(eid, {})

        # Métricas por agente
        for agent_id, rec in per_agent.items():
            metrics_logger.log_value(f"{agent_id}/episode_return", rec.get("ret", 0.0),
                                     reduce="mean", window=50)
            metrics_logger.log_value(f"{agent_id}/episode_len", rec.get("len", 0),
                                     reduce="mean", window=50)

        # ---- Razones (siempre loguear, aunque sea 0) ----
        reasons = ["timeout", "starvation", "dehydration"]
        counts  = {k: 0 for k in reasons}
        total   = max(1, len(per_agent))  # evita división por 0

        for rec in per_agent.values():
            r = (rec.get("reason") or "").strip()
            if r in counts:
                counts[r] += 1

        for k in reasons:
            pct = 100.0 * counts[k] / total
            metrics_logger.log_value(f"reason_{k}_pct", pct, reduce="mean", window=50)

if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)

    # --- Config del entorno (reutilizable) ---
    ENV_CFG = {
        "n_agents": 8,
        "veg_density": 25,
        "water_density": 20,
        "map_width": 1200,
        "map_height": 800,
        "max_steps": 350,   # lo pasamos al env por env_config
    }

    # Parámetros de muestreo y tamaño de batch derivados
    NUM_RUNNERS = 4
    FRAG = ENV_CFG["max_steps"]               # 350
    TOTAL_BATCH = NUM_RUNNERS * FRAG          # 1400

    register_env(
        "multi_eco",
        lambda cfg: ParallelPettingZooEnv(MultiAgentEcosystem(**cfg))
    )

    config = (
        PPOConfig()
        .environment(env="multi_eco", env_config=ENV_CFG)
        .callbacks(callbacks_class=PerAgentAndReasonMetrics)
        .framework("torch")
        .multi_agent(
            policies={"shared_policy": PolicySpec()},
            policy_mapping_fn=lambda *a, **k: "shared_policy"
        )
        # valor inicial (se sobreescribe abajo)
        .env_runners(num_env_runners=NUM_RUNNERS, rollout_fragment_length=50, sample_timeout_s=300)
    )

    # ✅ Recolectar episodios completos (evita trocear episodios)
    try:
        config = config.rollouts(batch_mode="complete_episodes")
    except Exception:
        pass

    # ✅ Alinear el fragmento con la duración de episodio
    config = config.env_runners(
        num_env_runners=NUM_RUNNERS,
        rollout_fragment_length=FRAG,   # 350
        sample_timeout_s=300
    )

    # --- Hiperparámetros (ajustados al TOTAL_BATCH=1400) ---
    try:
        config = config.training(
            train_batch_size=TOTAL_BATCH,   # 1400 (= 4 * 350)
            minibatch_size=200,             # 1400/200 = 7 minibatches
            num_epochs=2,
            lr=3e-4,
            gamma=0.99, lambda_=0.95,
            clip_param=0.2, vf_clip_param=10.0,
            grad_clip=0.5, entropy_coeff=0.01
        )
    except TypeError:
        # Fallback para APIs antiguas
        config = config.training(
            train_batch_size=TOTAL_BATCH,
            sgd_minibatch_size=200,
            num_sgd_iter=2,
            lr=3e-4,
            gamma=0.99, lambda_=0.95,
            clip_param=0.2, vf_clip_param=10.0,
            grad_clip=0.5, entropy_coeff=0.01
        )

    # — Abrimos el CSV una sola vez para TODO el entrenamiento —
    with open("monitor.csv", "w", newline="") as f:
        n_agents = 8  # si cambias en env_config, actualiza este valor
        writer = csv.writer(f)
        header = (
            ["iter", "r_mean", "l_mean",
            "reason_timeout_pct", "reason_starvation_pct", "reason_dehydration_pct"] +
            [f"r_agent_{i}" for i in range(n_agents)] +
            [f"l_agent_{i}" for i in range(n_agents)] +
            ["gpu_mem_used_mb", "gpu_mem_total_mb", "gpu_load_pct"]
        )
        writer.writerow(header)

        trainer = config.build()

        for i in range(50):
            result = trainer.train()

            # (opcional) tiempos para entender “pausas” de muestreo vs. aprendizaje
            t = result.get("timers", {}) or {}
            print("sampling_s=", t.get("env_runner_sampling_timer"),
                "learner_s=", t.get("learner_update_timer"))

            # métricas del new API stack
            ev = result.get("env_runners", {}) or {}

            # helper: busca primero en env_runners y, si acaso, en custom_metrics
            def m(key):
                return ev.get(key) or (ev.get("custom_metrics", {}) or {}).get(key)

            r_mean = ev.get("episode_return_mean")
            l_mean = ev.get("episode_len_mean")

            timeout_pct     = m("reason_timeout_pct")
            starvation_pct  = m("reason_starvation_pct")
            dehydration_pct = m("reason_dehydration_pct")

            row = [i, r_mean, l_mean, timeout_pct, starvation_pct, dehydration_pct]

            # métricas por agente (publicadas por el callback)
            for a in range(n_agents):
                row.append(m(f"agent_{a}/episode_return"))
            for a in range(n_agents):
                row.append(m(f"agent_{a}/episode_len"))

            # métricas GPU (opcionales)
            mem_used = mem_total = load_pct = float("nan")
            try:

                gpus = GPUtil.getGPUs()
                if gpus:
                    g = gpus[0]
                    mem_used  = getattr(g, "memoryUsed", float("nan"))
                    mem_total = getattr(g, "memoryTotal", float("nan"))
                    load_pct  = getattr(g, "load", 0.0) * 100.0
            except Exception:
                pass  # si no hay GPU o no está GPUtil, deja NaN

            row += [mem_used, mem_total, load_pct]
            writer.writerow(row)

            print(f"Iter {i}: ep_return_mean={r_mean}, ep_len_mean={l_mean}")



    MONITOR_CSV = "monitor.csv"

    try:
        df = pd.read_csv(MONITOR_CSV)

        # Conviertes a numérico lo que se pueda (deja strings como NaN si no aplica)
        for c in df.columns:
            if c != "iter":
                df[c] = pd.to_numeric(df[c], errors="coerce")

        # Columnas agregadas nuevas
        if not {"r_mean", "l_mean"}.issubset(df.columns):
            raise KeyError("El CSV no tiene r_mean/l_mean. Revisa el header escrito en el loop de entrenamiento.")

        iters = df["iter"].values if "iter" in df.columns else np.arange(len(df))

        r_mean = df["r_mean"]
        l_mean = df["l_mean"]

        # Razones (si el callback las publicó)
        reason_cols = ["reason_timeout_pct", "reason_starvation_pct", "reason_dehydration_pct", "reason_success_pct"]
        has_reasons = all(col in df.columns for col in reason_cols)
        timeout_pct = df[reason_cols[0]] if has_reasons else None
        starvation_pct = df[reason_cols[1]] if has_reasons else None
        dehydration_pct = df[reason_cols[2]] if has_reasons else None
        
        # Detecta columnas por agente
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

        # --- Fig 1: agregados ---
        plt.figure(figsize=(12, 8))

        # Recompensa media con suavizado
        plt.subplot(2, 2, 1)
        plt.plot(iters, r_mean, alpha=0.3, label="r_mean (raw)")
        plt.plot(iters, mov_avg_r, label=f"r_mean mov.avg (w={SMOOTH_WINDOW})")
        plt.title("Recompensa media por iteración")
        plt.xlabel("Iteración")
        plt.ylabel("Return")
        plt.legend()

        # Duración media con suavizado
        plt.subplot(2, 2, 3)
        plt.plot(iters, l_mean, alpha=0.3, label="l_mean (raw)")
        plt.plot(iters, mov_avg_l, label=f"l_mean mov.avg (w={SMOOTH_WINDOW})")
        plt.title("Longitud media por iteración")
        plt.xlabel("Iteración")
        plt.ylabel("Timesteps")
        plt.legend()

        # Promedios acumulados
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

        # --- Fig 2: retorno por agente ---
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

        # --- Fig 3: longitud por agente ---
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

        # --- Fig 4: razones de terminación (si están) ---
        if has_reasons:
            timeout_pct, starvation_pct, dehydration_pct, success_pct = [df[c] for c in reason_cols]
            plt.figure(figsize=(10,4))
            plt.plot(iters, timeout_pct,     label="timeout %")
            plt.plot(iters, starvation_pct,  label="starvation %")
            plt.plot(iters, dehydration_pct, label="dehydration %")
            plt.plot(iters, success_pct,     label="success %")

        # Resumen numérico rápido en consola
        print("\nResumen:")
        print(f"r_mean: mean={np.nanmean(r_mean):.3f}, std={np.nanstd(r_mean):.3f}, "
            f"median={np.nanmedian(r_mean):.3f}")
        print(f"l_mean: mean={np.nanmean(l_mean):.3f}, std={np.nanstd(l_mean):.3f}, "
            f"median={np.nanmedian(l_mean):.3f}")

    except FileNotFoundError:
        print(f"No se encontró el archivo '{MONITOR_CSV}'. Genera antes el CSV de monitorización.")
    except Exception as e:
        print("Error en análisis:", repr(e))
