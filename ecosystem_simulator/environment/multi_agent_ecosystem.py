"""
Módulo que define el entorno multi-agente principal, combinando especies y ecosistema
en un entorno compatible con PettingZoo y RLlib.
"""
import random
import numpy as np
import gymnasium as gym
from pettingzoo.utils import ParallelEnv
from typing import Tuple
from scipy.spatial import KDTree
from gymnasium.spaces import Discrete, Box, Dict, MultiBinary
from ..entities.specie import Specie, Role
from .ecosystem import Ecosystem

# Direcciones posibles para el movimiento
DIRECTIONS = ["north", "south", "east", "west", "stay"]  # 0..4
ACT_EAT   = 5
ACT_DRINK = 6
ACT_ATTACK= 7
BASE_STEP = 4.0 
BASE_COST = 0.50 
MOVE_EXTRA = 0.05 
class MultiAgentEcosystem(ParallelEnv, Ecosystem):
    """Entorno multi-agente que simula un ecosistema con especies que necesitan recursos."""
    
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
        n_predators: int = 1
    ):
        ParallelEnv.__init__(self)
        # Escalar densidad de recursos según el área del mapa
        base_w, base_h = 800, 600
        area_scale = (map_width * map_height) / (base_w * base_h)
        veg_density_scaled   = max(1, int(round(veg_density   * area_scale)))
        water_density_scaled = max(1, int(round(water_density * area_scale)))
        Ecosystem.__init__(self, veg_density_scaled, water_density_scaled, map_width, map_height)

        # --- Parámetros multi-agente ---
        self.n_agents  = n_agents
        self.n_predators = max(0, min(n_predators, n_agents))
        self.agents    = [f"agent_{i}" for i in range(n_agents)]
        self.possible_agents = list(self.agents)
        # mapa rápido de agente → índice en self.species
        self._agent_idx = {a: i for i, a in enumerate(self.agents)}
        self.max_steps = max_steps
        self.gamma     = gamma
        self.action_spaces = {a: Discrete(8) for a in self.agents}
        # espacios de observación y acción
        SQRT2 = np.sqrt(2.0)
        low  = np.array([0,0,-1,-1,-1,-1, 0,    0,    0,0,  0,0, -1, -1, 0, 0], dtype=np.float32)
        high = np.array([1,1,  1,  1,  1,  1, SQRT2, SQRT2, 1,1,  1,1,  1,  1, SQRT2,1], dtype=np.float32)

        self.action_spaces = {a: Discrete(8) for a in self.agents}
        self.observation_spaces = {a: Box(low, high, dtype=np.float32) for a in self.agents}
        self._step_count = 0
        self._reset_species()
        self.success_thr = 0.90     # 90% de reservas
        self.success_hold = 25      # mantenerlo 25 steps seguidos
    
    def _reset_species(self):
        spawn_xy = self._sample_spawn_positions(self.n_agents, min_dist=80.0, avoid_resources=True)

        def init_level(max_val):
            return float(np.random.uniform(0.60, 0.80) * max_val)

        self.species = []
        for i in range(self.n_agents):
            role = Role.PREDATOR if i < self.n_predators else Role.HERBIVORE
            # puedes ajustar speed/range por rol si quieres
            speed = 2.0 if role is Role.HERBIVORE else 2.2
            attack_range = 30.0 if role is Role.PREDATOR else 20.0
            s = Specie(
                food=init_level(100), water=init_level(100),
                x=spawn_xy[i][0], y=spawn_xy[i][1],
                map_width=self.map_width, map_height=self.map_height,
                role=role, speed=speed, attack_range=attack_range
            )
            self.species.append(s)
        
    def observation_space(self, agent: str) -> gym.Space:
        """Retorna el espacio de observación para un agente específico."""
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> gym.Space:
        """Retorna el espacio de acción para un agente específico."""
        return self.action_spaces[agent]

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """Reinicia el entorno a un estado inicial (Parallel API)."""
        # Si te pasan semilla, úsala en ambos RNGs
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Re-crear recursos/árboles kd desde Ecosystem
        # (asegúrate de que _init_veg/_init_wat existan; si no, usa valores por defecto)
        init_veg = getattr(self, "_init_veg",  self.vegetation["x"].shape[0] if hasattr(self, "vegetation") else 10)
        init_wat = getattr(self, "_init_wat",  self.water_sources["x"].shape[0] if hasattr(self, "water_sources") else 8)
        Ecosystem.__init__(self, self._init_veg, self._init_wat, self.map_width, self.map_height)

        # Re-crear especies (roles, stats, posiciones)
        self._reset_species()
        self._step_count = 0

        # Repoblar lista de agentes activos en este episodio
        self.agents = list(self.possible_agents)

        # Contadores por episodio
        self._ep_return = {a: 0.0 for a in self.possible_agents}
        self._ep_len    = {a: 0   for a in self.possible_agents}
        self._satiated  = {a: 0   for a in self.possible_agents}

        # Observaciones iniciales
        observations = {
            agent: self._get_obs(self.species[self._agent_idx[agent]])
            for agent in self.agents
        }

        # Infos iniciales útiles (rol + action_mask para que callbacks lo vean desde step 0)
        infos = {
            agent: {
                "role": "PREDATOR" if self.species[self._agent_idx[agent]].role is Role.PREDATOR else "HERBIVORE",
                "action_mask": self._action_mask(self.species[self._agent_idx[agent]]),
                "ep_return": 0.0,
                "ep_len": 0,
            }
            for agent in self.agents
        }

        return observations, infos

    
    def _norm_dist_to(self, sp: Specie, kind: str) -> float:
        """Normaliza la distancia al recurso más cercano (agua o vegetación) respecto al tamaño del mapa."""
        tree = self._wat_tree if kind == "water" else self._veg_tree
        src  = self.water_sources if kind == "water" else self.vegetation
        if tree is None or src["centers"].shape[0] == 0:
            return 1.0
        _, idx = tree.query([sp.x, sp.y], k=1)
        cx, cy = src["centers"][idx]
        d = ((cx - sp.x)**2 + (cy - sp.y)**2)**0.5
        return d / ((self.map_width**2 + self.map_height**2)**0.5)
    
    def _norm_dist_to_prey(self, sp: Specie) -> float:
        if sp.role is not Role.PREDATOR:
            return 1.0
        preys = [a for a in self.species if a.alive and a.role is Role.HERBIVORE]
        if not preys:
            return 1.0
        if len(preys) >= 2:
            coords = np.array([(p.x, p.y) for p in preys], dtype=np.float32)
            idx = np.argmin(((coords[:,0]-sp.x)**2 + (coords[:,1]-sp.y)**2))
            px, py = coords[idx]
        else:
            px, py = preys[0].x, preys[0].y
        d = ((px - sp.x)**2 + (py - sp.y)**2)**0.5
        return d / ((self.map_width**2 + self.map_height**2)**0.5)

    def _action_mask(self, sp: Specie) -> np.ndarray:
        # 8 acciones: mover(0..4), comer(5), beber(6), atacar(7)
        mask = np.ones(8, dtype=np.int8)
        if sp.role is Role.HERBIVORE:
            mask[ACT_ATTACK] = 0

        return mask
    def step(self, actions):
        obs, rewards, terminations, truncations, infos = {}, {}, {}, {}, {}
        self._step_count += 1
        prev_agents   = list(self.agents)
        alive         = []
        episode_trunc = (self._step_count >= self.max_steps)

        veg_claims, wat_claims = {}, {}
        step_bonus = {a: 0.0 for a in prev_agents}
        energy_after_move = {}

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

                # Necesidad dominante
                f_def = 1.0 - (sp.food  / sp.max_food)
                w_def = 1.0 - (sp.water / sp.max_water)
                need  = "water" if w_def > f_def else "veg"
                need_def = max(f_def, w_def)
                d_prev  = self._norm_dist_to(sp, need)
                prey_prev = self._norm_dist_to_prey(sp) 

                a = int(actions[agent])

                # === 0..4: movimiento ===
                if 0 <= a <= 4:
                    act_dir_name = DIRECTIONS[a]
                    if act_dir_name == "stay":
                        sp.metabolize(BASE_COST)
                        sp.move(0.0, "stay")
                        reward -= 0.01 + 0.05 * need_def
                    else:
                        sp.metabolize(BASE_COST + MOVE_EXTRA)
                        sp.move(BASE_STEP, act_dir_name)

                    d_now = self._norm_dist_to(sp, need)
                    reward += 2.0 * (d_prev - d_now)
                    energy_after_move[agent] = sp.total_energy
                    reward += 0.1 * (energy_after_move[agent] - prev_energy)

                    # detectar colisión para “claims” (solo si no hizo eat/drink explícito)
                    hit = self.collide_resources((sp.x, sp.y), Specie.AGENT_SIZE, self.water_sources)
                    if hit.size:
                        wat_claims.setdefault(int(hit[0]), []).append(agent)
                    hit = self.collide_resources((sp.x, sp.y), Specie.AGENT_SIZE, self.vegetation)
                    if hit.size:
                        veg_claims.setdefault(int(hit[0]), []).append(agent)

                # === 5: comer vegetación explícito ===
                
                elif a == ACT_EAT:
                    sp.metabolize(BASE_COST)
                    prey_now = self._norm_dist_to_prey(sp)
                    if sp.role is Role.PREDATOR:
                        # el depredador no obtiene beneficio de la vegetación
                        reward += 1.0 * (prey_prev - prey_now) 
                    else:
                        ate = self.try_eat_vegetation(sp, bite_gain=20.0)
                        reward += 1.0 if ate else -0.1

                # === 6: beber agua explícito ===
                elif a == ACT_DRINK:
                    sp.metabolize(BASE_COST)
                    drank = self.try_drink(sp, sip_gain=15.0)
                    if drank:
                        reward += 0.5
                    else:
                        reward -= 0.1  # intentó beber sin estar sobre agua

                # === 7: atacar (solo depredador) ===
                elif a == ACT_ATTACK:
                    sp.metabolize(BASE_COST)
                    if sp.role is Role.PREDATOR:
                        hit = self.try_attack(sp, self.species, dmg=50.0)
                        reward += 3.0 if hit else -0.1
                        # métricas por paso
                        infos.setdefault(agent, {})
                        infos[agent]["attack_attempt"] = 1
                        infos[agent]["attack_hit"] = 1 if hit else 0
                    else:
                        reward -= 0.1

                # Muertes por recursos
                if (sp.food <= 0) or (sp.water <= 0):
                    reward   -= 10.0
                    done_term = True
                    reason    = "starvation" if sp.food <= 0 else "dehydration"

                # Bonus por homeostasis
                if not done_term:
                    if sp.food >= self.success_thr * sp.max_food and sp.water >= self.success_thr * sp.max_water:
                        self._satiated[agent] += 1
                        if self._satiated[agent] >= self.success_hold:
                            reward += 30.0
                            self._satiated[agent] = 0
                    else:
                        self._satiated[agent] = 0

                # Timeout
                if (not done_term) and episode_trunc:
                    done_trunc = True
                    reason = "timeout"

            # Shaping por estado
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
            # Etiqueta el rol para que el callback pueda agregar métricas por rol
            role_str = "PREDATOR" if sp.role is Role.PREDATOR else "HERBIVORE"
            base_info = {"reason": reason, "role": role_str, "action_mask": self._action_mask(sp)}
            infos[agent] = {**infos.get(agent, {}), **base_info}

            prev_info = infos.get(agent, {})
            # Mantén lo que ya estaba (attack_*), y actualiza reason/role/action_mask
            infos[agent] = {**prev_info, **base_info}

            if (not done_term) and (not done_trunc):
                alive.append(agent)

        # Resolver conflictos por recursos (solo para quienes no hicieron eat/drink explícito)
        removed_w = 0
        for idx, claimers in wat_claims.items():
            vivos = [a for a in claimers if not terminations.get(a, False)]
            if not vivos:
                continue
            winner = min(vivos, key=lambda a: self.species[self._agent_idx[a]].water /
                                        self.species[self._agent_idx[a]].max_water)
            sp_w = self.species[self._agent_idx[winner]]
            sp_w.water = min(sp_w.water + 10, sp_w.max_water)
            step_bonus[winner] += 0.5  # coherente con beber explícito
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
            step_bonus[winner] += 1.0   # coherente con comer explícito
            if self.consume_vegetation(idx):
                removed_v += 1

        # Bonos acumulados
        for a in prev_agents:
            rewards[a] = float(rewards.get(a, 0.0) + step_bonus.get(a, 0.0))
        for a in prev_agents:
            if (a in energy_after_move) and (not terminations[a]):
                post = self.species[self._agent_idx[a]].total_energy
                rewards[a] += 0.1 * (post - energy_after_move[a])

        # Reponer recursos
        if not episode_trunc:
            target_veg = self._init_veg
            target_wat = self._init_wat
            cur_veg = self.vegetation["x"].shape[0]
            cur_wat = self.water_sources["x"].shape[0]
            spawn_veg = max(0, min(max(1, removed_v // 2), target_veg - cur_veg))
            spawn_wat = max(0, min(max(1, removed_w // 2), target_wat - cur_wat))
            if spawn_veg: self.spawn_vegetation(n=spawn_veg)
            if spawn_wat: self.spawn_water(n=spawn_wat)

        self._prune_depleted_resources()

        # === NUEVO: terminar por depredación (agentes que ahora están muertos) ===
        for agent in prev_agents:
            i = self._agent_idx[agent]
            sp = self.species[i]
            if (not terminations.get(agent, False)) and sp.alive is False:
                terminations[agent] = True
                truncations[agent]  = False
                infos[agent] = {**infos.get(agent, {}), "reason": "predation"}
                if agent in alive:
                    alive.remove(agent)



        # Observaciones y métricas
        for agent in prev_agents:
            obs[agent] = self._get_obs(self.species[self._agent_idx[agent]])
            infos.setdefault(agent, {})
            self._ep_return[agent] = self._ep_return.get(agent, 0.0) + rewards.get(agent, 0.0)
            self._ep_len[agent]    = self._ep_len.get(agent, 0) + 1
            infos[agent].setdefault("ep_return", self._ep_return[agent])
            infos[agent].setdefault("ep_len",    self._ep_len[agent])

        if episode_trunc and len(alive) > 0:
            for a in list(alive):
                truncations[a] = True
                infos[a]["reason"] = "timeout"
            alive = []

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

        # Rol (one-hot)
        role_h = 1.0 if sp.role is Role.HERBIVORE else 0.0
        role_p = 1.0 if sp.role is Role.PREDATOR else 0.0

        # Vector a la presa más cercana (solo útil para depredador)
        prey_dx = prey_dy = 0.0
        prey_dist = 1.0
        prey_avail = 0.0
        if sp.role is Role.PREDATOR:
            preys = [a for a in self.species if a.alive and a.role is Role.HERBIVORE]
            if preys:
                prey_avail = 1.0
                if len(preys) >= 2:
                    coords = np.array([(p.x, p.y) for p in preys], dtype=np.float32)
                    tree = KDTree(coords)
                    _, idx_p = tree.query([sp.x, sp.y], k=1)
                    px, py = coords[int(idx_p)]
                else:
                    px, py = preys[0].x, preys[0].y
                prey_dx, prey_dy = (px - sp.x)/self.map_width, (py - sp.y)/self.map_height
                prey_dist = (prey_dx**2 + prey_dy**2)**0.5

        f_norm = sp.food / sp.max_food
        w_norm = sp.water / sp.max_water

        return np.array([
            f_norm, w_norm,
            dx_w, dy_w, dx_v, dy_v,
            dist_w, dist_v,
            w_avail, v_avail,
            role_h, role_p,
            prey_dx, prey_dy, prey_dist, prey_avail
        ], dtype=np.float32)

    

    
    def _sample_spawn_positions(
        self,
        n: int,
        min_dist: float = 80.0,        # distancia mínima entre agentes
        avoid_resources: bool = True,   # evitar spawnear sobre agua/vegetación
        max_tries: int = 5000,
    ):
        """Muestrea posiciones de spawn para los agentes con restricciones."""
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

            # Si no encontró lugar (muy raro), relajar restricciones
            if not ok:
                positions.append((self.map_width/2 + k*half, self.map_height/2))

        return positions
