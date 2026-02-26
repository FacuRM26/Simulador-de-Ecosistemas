"""
Módulo que define el entorno multi-agente principal, combinando especies y ecosistema
en un entorno compatible con PettingZoo y RLlib.

Este es el entorno principal para el aprendizaje por refuerzo multi-agente.
Implementa la interfaz de PettingZoo ParallelEnv y extiende la clase Ecosystem
para crear un simulador completo donde múltiples agentes aprenden a sobrevivir.
"""
import random
import numpy as np
import gymnasium as gym
from pettingzoo.utils import ParallelEnv
from typing import Dict, List
from gymnasium.spaces import Discrete, Box
from scipy.spatial import KDTree

from ..entities.specie import Specie, Role
from .ecosystem import Ecosystem
from dataclasses import dataclass, field


@dataclass
class StepContext:
    prev_agents: List[str]
    alive: List[str] = field(default_factory=list)
    obs: Dict[str, np.ndarray] = field(default_factory=dict)
    rewards: Dict[str, float] = field(default_factory=dict)
    terminations: Dict[str, bool] = field(default_factory=dict)
    truncations: Dict[str, bool] = field(default_factory=dict)
    infos: Dict[str, dict] = field(default_factory=dict)
    veg_claims: Dict[int, List[str]] = field(default_factory=dict)
    wat_claims: Dict[int, List[str]] = field(default_factory=dict)
    step_bonus: Dict[str, float] = field(default_factory=dict)
    energy_after_move: Dict[str, float] = field(default_factory=dict)
    removed_v: int = 0
    removed_w: int = 0
    episode_trunc: bool = False

# Direcciones posibles para el movimiento
DIRECTIONS = ["north", "south", "east", "west"]
ACT_EAT    = 4
ACT_DRINK  = 5
ACT_ATTACK = 6
BASE_STEP = 2.0 
BASE_COST = 0.50 
MOVE_EXTRA = 0.05 
PRED_ATTACK_CD = 2  # pasos de cooldown después de atacar
class MultiAgentEcosystem(ParallelEnv, Ecosystem):
    """
    Entorno multi-agente que simula un ecosistema con especies que necesitan recursos.
    
    Este entorno combina múltiples agentes (especies) que deben aprender a:
    - Buscar y consumir recursos (vegetación y agua)
    - Gestionar sus niveles de energía
    - Sobrevivir el mayor tiempo posible
    
    Hereda de:
    - ParallelEnv (PettingZoo): Para soporte de RL multi-agente
    - Ecosystem: Para la gestión de recursos del ecosistema
    
    Atributos:
        n_agents (int): Número de agentes en el entorno
        agents (list): Lista de IDs de agentes activos
        possible_agents (list): Lista de todos los agentes posibles
        species (list): Lista de objetos Specie (estado de cada agente)
        max_steps (int): Máximo número de pasos por episodio
        gamma (float): Factor de descuento para RL
    """
    
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
        """
        Inicializa el entorno multi-agente con agentes y recursos.
        
        Args:
            n_agents: Número de agentes en el entorno (por defecto 2)
            veg_density: Densidad de vegetación (se escala por área del mapa)
            water_density: Densidad de agua (se escala por área del mapa)
            map_width: Ancho del mapa (por defecto 800)
            map_height: Alto del mapa (por defecto 600)
            max_steps: Máximo número de pasos por episodio (por defecto 350)
            gamma: Factor de descuento para RL (por defecto 0.995)
        """
        # Inicializar clase padre ParallelEnv
        ParallelEnv.__init__(self)
        
        # Escalar densidad de recursos según el área del mapa
        # Esto mantiene la densidad relativa constante para diferentes tamaños de mapa
        base_w, base_h = 800, 600
        area_scale = (map_width * map_height) / (base_w * base_h)
        veg_density_scaled   = max(1, int(round(veg_density   * area_scale)))
        water_density_scaled = max(1, int(round(water_density * area_scale)))
        
        # Inicializar clase padre Ecosystem con densidades escaladas
        Ecosystem.__init__(self, veg_density_scaled, water_density_scaled, map_width, map_height)

        # --- Configuración multi-agente ---
        self.n_agents  = n_agents
        self.n_predators = max(0, min(n_predators, n_agents))
        # Crear IDs de agentes: "agent_0", "agent_1", etc.
        self.agents    = [f"agent_{i}" for i in range(n_agents)]
        # Lista de todos los agentes posibles (para reset)
        self.possible_agents = list(self.agents)
        # Diccionario para mapeo rápido agente: índice en self.species
        self._agent_idx = {a: i for i, a in enumerate(self.agents)}
        self._init_veg = veg_density_scaled
        self._init_wat = water_density_scaled
        self.max_steps = max_steps
        self.gamma     = gamma
        self.action_spaces = {a: Discrete(7) for a in self.agents}
        # --- Definición de espacios de observación y acción ---
        # Raíz cuadrada de 2 para normalizar distancias diagonales máximas
        SQRT2 = np.sqrt(2.0)

        # Límites inferiores de la observación
        low  = np.array(
            [0, 0, -1, -1, -1, -1,
             0, 0,
             0, 0,
             0, 0,
             -1, -1, 0, 0],
            dtype=np.float32
        )

        # Límites superiores de la observación
        high = np.array(
            [1, 1,  1,  1,  1,  1,
             SQRT2, SQRT2,
             1, 1,
             1, 1,
             1, 1, SQRT2, 1],
            dtype=np.float32
        )

        # Espacio de observación: continuo con 16 características normalizadas
        self.observation_spaces = {a: Box(low, high, dtype=np.float32) for a in self.agents}
        
        # Contador de pasos en el episodio actual
        self._step_count = 0
        
        # Inicializar especies
        self._reset_species()
        
        # Umbrales para bonificación por homeostasis (mantener recursos altos)
        self.success_thr = 0.90     # 90% de las reservas máximas
        self.success_hold = 25      # Mantenerlo durante 25 pasos seguidos
    
    def _reset_species(self):
        spawn_xy = self._sample_spawn_positions(
            self.n_agents, min_dist=80.0, avoid_resources=True
        )

        def init_level(max_val):
            return float(np.random.uniform(0.60, 0.80) * max_val)

        self.species = []
        for i in range(self.n_agents):
            role = Role.PREDATOR if i < self.n_predators else Role.HERBIVORE

            # stats por rol (ejemplo razonable)
            if role is Role.HERBIVORE:
                speed = 2.1
                hp = 100.0
                attack_range = 0.0
                attack_cost = 0.0
                max_food, max_water = 100.0, 100.0
            else:
                speed = 3.0           # un poco más rápido que la presa
                hp = 120.0
                attack_range = 45.0
                attack_cost = 0.5
                max_food, max_water = 100.0, 100.0

            x, y = spawn_xy[i]

            s = Specie(
                food=init_level(max_food),
                water=init_level(max_water),
                x=float(x),
                y=float(y),
                max_food=max_food,
                max_water=max_water,
                map_width=self.map_width,
                map_height=self.map_height,
                role=role,
                hp=hp,
                speed=speed,
                attack_range=attack_range,
                attack_cost=attack_cost,
            )
            s.attack_cd = 0  # arranca sin cooldown
            self.species.append(s)
                    
    def observation_space(self, agent: str) -> gym.Space:
        """
        Retorna el espacio de observación para un agente específico.
        
        Args:
            agent: ID del agente (ej. "agent_0")
            
        Returns:
            gym.Space: Espacio de observación (Box de 16 dimensiones)
        """
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> gym.Space:
        """
        Retorna el espacio de acción para un agente específico.
        
        Args:
            agent: ID del agente (ej. "agent_0")
            
        Returns:
            gym.Space: Espacio de acción (Discrete con 7 opciones)
        """
        return self.action_spaces[agent]
        # Re-crear especies (roles, stats, posiciones)
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """
        Reinicia el entorno a un estado inicial para un nuevo episodio (Parallel API).

        Reinicializa:
        - El ecosistema (recursos de vegetación y agua)
        - Las especies (agentes) con sus roles y estadísticas
        - Contadores y métricas por episodio

        Args:
            seed: Semilla para la generación aleatoria (opcional).
            options: Opciones adicionales (no usadas actualmente).

        Returns:
            tuple:
                observations (dict): observación inicial por agente activo.
                infos        (dict): info inicial por agente (rol, action_mask, etc.).
        """
        # Establecer semilla si se proporciona
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Re-crear recursos/árboles KD desde Ecosystem.
        # Si _init_veg/_init_wat no existen aún, usamos el tamaño actual o un valor por defecto.
        init_veg = getattr(
            self,
            "_init_veg",
            self.vegetation["x"].shape[0] if hasattr(self, "vegetation") else 10,
        )
        init_wat = getattr(
            self,
            "_init_wat",
            self.water_sources["x"].shape[0] if hasattr(self, "water_sources") else 8,
        )
        Ecosystem.__init__(self, init_veg, init_wat, self.map_width, self.map_height)

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
                "role": (
                    "PREDATOR"
                    if self.species[self._agent_idx[agent]].role is Role.PREDATOR
                    else "HERBIVORE"
                ),
                "action_mask": self._action_mask(self.species[self._agent_idx[agent]]),
                "ep_return": 0.0,
                "ep_len": 0,
            }
            for agent in self.agents
        }

        return observations, infos

    
    def _norm_dist_to(self, sp: Specie, kind: str) -> float:
        """
        Calcula y normaliza la distancia al recurso más cercano.
        
        Normaliza la distancia dividiendo por la diagonal del mapa (máxima
        distancia posible), resultando en un valor entre 0 y 1.
        
        Args:
            sp: Objeto Specie del agente
            kind: Tipo de recurso ("water" o "veg")
            
        Returns:
            float: Distancia normalizada al recurso más cercano (0-1)
                   Retorna 1.0 si no hay recursos disponibles
        """
        # Seleccionar el árbol KD y recursos según el tipo
        tree = self._wat_tree if kind == "water" else self._veg_tree
        src  = self.water_sources if kind == "water" else self.vegetation
        
        # Si no hay recursos disponibles, retornar distancia máxima
        if tree is None or src["centers"].shape[0] == 0:
            return 1.0
        
        # Buscar el recurso más cercano usando el KD-Tree
        _, idx = tree.query([sp.x, sp.y], k=1)
        
        # Obtener coordenadas del centro del recurso más cercano
        cx, cy = src["centers"][idx]
        
        # Calcular distancia euclidiana
        d = ((cx - sp.x)**2 + (cy - sp.y)**2)**0.5
        
        # Normalizar por la diagonal del mapa (distancia máxima posible)
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
        # 7 acciones: mover(0..3), comer(4), beber(5), atacar(6)
        mask = np.ones(7, dtype=np.int8)
        if sp.role is Role.HERBIVORE:
            mask[ACT_ATTACK] = 0  # ahora ACT_ATTACK = 6, índice válido
        return mask
    
    def _resolve_water_claims(self, ctx: StepContext):
        for idx, claimers in ctx.wat_claims.items():
            vivos = [a for a in claimers if not ctx.terminations.get(a, False)]
            if not vivos:
                continue

            winner = min(
                vivos,
                key=lambda a: self.species[self._agent_idx[a]].water /
                            self.species[self._agent_idx[a]].max_water
            )

            sp_w = self.species[self._agent_idx[winner]]
            sp_w.water = min(sp_w.water + 10, sp_w.max_water)

            ctx.step_bonus[winner] += 5.0

            if self.consume_water(idx):
                ctx.removed_w += 1

    def _resolve_veg_claims(self, ctx: StepContext):
        for idx, claimers in ctx.veg_claims.items():
            vivos = [
                a for a in claimers
                if (not ctx.terminations.get(a, False))
                and self.species[self._agent_idx[a]].alive
                and self.species[self._agent_idx[a]].role is Role.HERBIVORE
            ]
            if not vivos:
                continue

            winner = min(
                vivos,
                key=lambda a: self.species[self._agent_idx[a]].food /
                            self.species[self._agent_idx[a]].max_food
            )

            sp_w = self.species[self._agent_idx[winner]]
            sp_w.food = min(sp_w.food + 10, sp_w.max_food)

            ctx.step_bonus[winner] += 8.0

            if self.consume_vegetation(idx):
                ctx.removed_v += 1
    def _apply_step_bonus(self, ctx: StepContext):
        for a in ctx.prev_agents:
            ctx.rewards[a] = float(ctx.rewards.get(a, 0.0) + ctx.step_bonus.get(a, 0.0))
    def _init_step_context(self) -> StepContext:
        self._step_count += 1
        prev_agents = list(self.agents)
        ctx = StepContext(prev_agents=prev_agents)
        ctx.episode_trunc = (self._step_count >= self.max_steps)

        # Inicializar bonus por defecto
        ctx.step_bonus = {a: 0.0 for a in prev_agents}
        return ctx


    def _process_agents(self, actions: dict, ctx: StepContext):
        proc_order = sorted(ctx.prev_agents, key=lambda a: 0 if self.species[self._agent_idx[a]].role is Role.PREDATOR else 1)

        for agent in proc_order:
            i = self._agent_idx[agent]
            sp = self.species[i]

            reward = 0.0
            done_term = False
            done_trunc = False
            reason = ""

            # Cooldown tick aquí (más simple que hacerlo al final)
            if sp.attack_cd > 0:
                sp.attack_cd -= 1

            if agent in actions:
                prev_energy = sp.total_energy

                # déficit de recursos
                f_def = 1.0 - (sp.food / sp.max_food)
                w_def = 1.0 - (sp.water / sp.max_water)

                if sp.role is Role.PREDATOR:
                    # Para depredador: "food" = presa
                    need = "water" if w_def > f_def else "prey"
                    d_prev = self._norm_dist_to_prey(sp) if need == "prey" else self._norm_dist_to(sp, "water")
                else:
                    need = "water" if w_def > f_def else "veg"
                    d_prev = self._norm_dist_to(sp, need)

                prey_prev = self._norm_dist_to_prey(sp)

                a = int(actions[agent])

                # --- movimiento 0..3 ---
                if 0 <= a <= 3:
                    act_dir_name = DIRECTIONS[a]
                    prev_x, prev_y = sp.x, sp.y

                    sp.metabolize(BASE_COST + MOVE_EXTRA)
                    sp.move(BASE_STEP, act_dir_name)

                    moved_dist = abs(sp.x - prev_x) + abs(sp.y - prev_y)
                    if moved_dist < 1e-3:
                        reward -= 0.3

                    # progreso hacia objetivo
                    d_now = self._norm_dist_to_prey(sp) if need == "prey" else self._norm_dist_to(sp, need)

                    reward += 6.0 * (d_prev - d_now)
                    if need == "prey":
                        diag = (self.map_width**2 + self.map_height**2) ** 0.5
                        in_range = 1.0 if d_now <= (sp.attack_range / diag) else 0.0
                        reward += 0.5 * in_range   # pequeño bonus por llegar a rango
                    else:
                        reward += 1.0 * (1.0 - d_now)
                    ctx.energy_after_move[agent] = sp.total_energy
                    reward += 0.02 * (ctx.energy_after_move[agent] - prev_energy)

                    # claims de agua
                    hit = self.collide_resources((sp.x, sp.y), Specie.AGENT_SIZE, self.water_sources)
                    if hit.size:
                        ctx.wat_claims.setdefault(int(hit[0]), []).append(agent)

                    # claims de vegetación (solo herbívoro)
                    hit = self.collide_resources((sp.x, sp.y), Specie.AGENT_SIZE, self.vegetation)
                    if hit.size and sp.role is Role.HERBIVORE:
                        ctx.veg_claims.setdefault(int(hit[0]), []).append(agent)

                # --- comer ---
                elif a == ACT_EAT:
                    sp.metabolize(BASE_COST)

                    if sp.role is Role.PREDATOR:
                        # depredador no come veg: shaping hacia presa
                        prey_now = self._norm_dist_to_prey(sp)
                        reward += 1.0 * (prey_prev - prey_now)
                    else:
                        ate = self.try_eat_vegetation(sp, bite_gain=20.0)
                        reward += 1.0 if ate else -0.1

                # --- beber ---
                elif a == ACT_DRINK:
                    sp.metabolize(BASE_COST)
                    drank = self.try_drink(sp, sip_gain=15.0)
                    reward += 0.5 if drank else -0.1

                # --- atacar ---
                elif a == ACT_ATTACK:
                    sp.metabolize(BASE_COST)

                    if sp.role is Role.PREDATOR:
                        if sp.attack_cd > 0:
                            reward -= 0.05
                            outcome = "cooldown"
                        else:
                            outcome = self.try_attack(sp, self.species, dmg=50.0)

                            # Recompensas por ataque (esto NO lo habías puesto en el refactor)
                            if outcome == "miss":
                                reward -= 0.1
                            elif outcome == "hit":
                                reward += 10.0
                                sp.attack_cd = PRED_ATTACK_CD
                            elif outcome == "kill":
                                reward += 25.0
                                sp.food = min(sp.max_food, sp.food + 60.0)  # <- clave
                                sp.attack_cd = PRED_ATTACK_CD

                        ctx.infos.setdefault(agent, {})
                        ctx.infos[agent]["attack_hit"] = 1 if outcome in ("hit", "kill") else 0
                        ctx.infos[agent]["attack_kill"] = 1 if outcome == "kill" else 0
                    else:
                        reward -= 0.1

                # muerte por recursos
                if (sp.food <= 0) or (sp.water <= 0):
                    reward -= 10.0
                    done_term = True
                    reason = "starvation" if sp.food <= 0 else "dehydration"
                    sp.alive = False

                # homeostasis
                if not done_term:
                    if sp.food >= self.success_thr * sp.max_food and sp.water >= self.success_thr * sp.max_water:
                        self._satiated[agent] += 1
                        if self._satiated[agent] >= self.success_hold:
                            reward += 15.0
                            self._satiated[agent] = 0
                    else:
                        self._satiated[agent] = 0

                # truncation por timeout
                if (not done_term) and ctx.episode_trunc:
                    done_trunc = True
                    reason = "timeout"

            # recompensas extra por “salud”
            if not done_term:
                f_norm = sp.food / sp.max_food
                w_norm = sp.water / sp.max_water
                min_norm = min(f_norm, w_norm)

                reward += 0.02 * min_norm
                if min_norm < 0.15:
                    reward -= 0.5
                if min_norm < 0.08:
                    reward -= 1.0

                # penalización por bordes
                dist_left   = sp.x / self.map_width
                dist_right  = (self.map_width - sp.x) / self.map_width
                dist_top    = sp.y / self.map_height
                dist_bottom = (self.map_height - sp.y) / self.map_height
                dist_to_edge = min(dist_left, dist_right, dist_top, dist_bottom)

                edge_margin = 0.15
                if dist_to_edge < edge_margin:
                    reward -= 0.5 * (edge_margin - dist_to_edge) / edge_margin

            # guardar outputs por agente
            ctx.rewards[agent] = float(reward)
            ctx.terminations[agent] = done_term
            ctx.truncations[agent] = done_trunc

            role_str = "PREDATOR" if sp.role is Role.PREDATOR else "HERBIVORE"
            base_info = {
                "reason": reason,
                "role": role_str,
                "action_mask": self._action_mask(sp),
            }
            ctx.infos[agent] = {**ctx.infos.get(agent, {}), **base_info}

            if (not done_term) and (not done_trunc):
                ctx.alive.append(agent)


    def _apply_post_move_energy_shaping(self, ctx: StepContext):
        for a in ctx.prev_agents:
            if (a in ctx.energy_after_move) and (not ctx.terminations.get(a, False)):
                post = self.species[self._agent_idx[a]].total_energy
                ctx.rewards[a] = float(ctx.rewards.get(a, 0.0) + 0.02 * (post - ctx.energy_after_move[a]))


    def _respawn_resources_if_needed(self, ctx: StepContext):
        if ctx.episode_trunc:
            return

        target_veg = self._init_veg
        target_wat = self._init_wat

        cur_veg = self.vegetation["x"].shape[0]
        cur_wat = self.water_sources["x"].shape[0]

        spawn_veg = max(0, min(max(1, ctx.removed_v // 2), target_veg - cur_veg))
        spawn_wat = max(0, min(max(1, ctx.removed_w // 2), target_wat - cur_wat))

        if spawn_veg:
            self.spawn_vegetation(n=spawn_veg)
        if spawn_wat:
            self.spawn_water(n=spawn_wat)


    def _finalize_predation_and_cooldowns(self, ctx: StepContext):
        # marcar terminación por depredación (presas que quedaron hp<=0)
        for agent in ctx.prev_agents:
            i = self._agent_idx[agent]
            sp = self.species[i]

            if (not ctx.terminations.get(agent, False)) and (sp.alive is False):
                ctx.terminations[agent] = True
                ctx.truncations[agent] = False
                ctx.infos[agent] = {**ctx.infos.get(agent, {}), "reason": "predation"}
                if agent in ctx.alive:
                    ctx.alive.remove(agent)


    def _build_obs_and_episode_metrics(self, ctx: StepContext):
        for agent in ctx.prev_agents:
            ctx.obs[agent] = self._get_obs(self.species[self._agent_idx[agent]])

            ctx.infos.setdefault(agent, {})
            self._ep_return[agent] = self._ep_return.get(agent, 0.0) + ctx.rewards.get(agent, 0.0)
            self._ep_len[agent] = self._ep_len.get(agent, 0) + 1

            ctx.infos[agent].setdefault("ep_return", self._ep_return[agent])
            ctx.infos[agent].setdefault("ep_len", self._ep_len[agent])


    def _apply_timeout_if_needed(self, ctx: StepContext):
        if ctx.episode_trunc and len(ctx.alive) > 0:
            for a in list(ctx.alive):
                ctx.truncations[a] = True
                ctx.infos[a]["reason"] = "timeout"
            ctx.alive = []
    def step(self, actions):
        ctx = self._init_step_context()
        self._process_agents(actions, ctx)
        self._resolve_water_claims(ctx)
        self._resolve_veg_claims(ctx)
        self._apply_step_bonus(ctx)
        self._apply_post_move_energy_shaping(ctx)
        self._respawn_resources_if_needed(ctx)
        self._prune_depleted_resources()
        self._finalize_predation_and_cooldowns(ctx)
        self._build_obs_and_episode_metrics(ctx)
        self._apply_timeout_if_needed(ctx)
        self.agents = ctx.alive
        return ctx.obs, ctx.rewards, ctx.terminations, ctx.truncations, ctx.infos
    
    def _get_obs(self, sp: Specie) -> np.ndarray:
        """
        Construye la observación para un agente específico.
        
        La observación contiene 10 características normalizadas:
        0-1: Niveles de recursos del agente (comida, agua) [0-1]
        2-3: Vector dirección al agua más cercana (dx, dy) [-1 a 1]
        4-5: Vector dirección a la vegetación más cercana (dx, dy) [-1 a 1]
        6: Distancia al agua más cercana [0 a sqrt(2)]
        7: Distancia a la vegetación más cercana [0 a sqrt(2)]
        8: Flag de disponibilidad de agua (0 o 1)
        9: Flag de disponibilidad de vegetación (0 o 1)
        
        Args:
            sp: Objeto Specie del cual generar la observación
            
        Returns:
            np.ndarray: Array de 10 elementos con la observación normalizada
        """
        # Verificar disponibilidad de recursos
        w_avail = float(self._wat_tree is not None and self.water_sources["centers"].size > 0)
        v_avail = float(self._veg_tree is not None and self.vegetation["centers"].size > 0)

        # === CALCULAR INFORMACIÓN SOBRE AGUA ===
        if w_avail:
            # Encontrar agua más cercana
            _, idx_w = self._wat_tree.query([sp.x, sp.y], k=1)
            wx, wy = self.water_sources["centers"][idx_w]
            
            # Vector dirección normalizado (respecto al tamaño del mapa)
            dx_w, dy_w = (wx - sp.x)/self.map_width, (wy - sp.y)/self.map_height
            
            # Distancia euclidiana normalizada
            dist_w = (dx_w**2 + dy_w**2)**0.5
        else:
            # No hay agua disponible
            dx_w = dy_w = 0.0
            dist_w = 1.0  # Distancia máxima

        # === CALCULAR INFORMACIÓN SOBRE VEGETACIÓN ===
        if v_avail:
            # Encontrar vegetación más cercana
            _, idx_v = self._veg_tree.query([sp.x, sp.y], k=1)
            vx, vy = self.vegetation["centers"][idx_v]
            
            # Vector dirección normalizado
            dx_v, dy_v = (vx - sp.x)/self.map_width, (vy - sp.y)/self.map_height
            
            # Distancia euclidiana normalizada
            dist_v = (dx_v**2 + dy_v**2)**0.5
        else:
            # No hay vegetación disponible
            dx_v = dy_v = 0.0
            dist_v = 1.0  # Distancia máxima

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
        min_dist: float = 80.0,        # Distancia mínima entre agentes
        avoid_resources: bool = True,   # Evitar spawnear sobre recursos
        max_tries: int = 5000,          # Intentos máximos por posición
    ):
        """
        Genera posiciones de spawn válidas para los agentes con restricciones.
        
        Intenta generar posiciones que:
        1. No colisionen con recursos existentes (opcional)
        2. Mantengan una distancia mínima entre agentes
        3. Estén dentro de los límites del mapa
        
        Si no puede encontrar una posición válida después de max_tries,
        coloca el agente en el centro del mapa con un pequeño offset.
        
        Args:
            n: Número de posiciones a generar
            min_dist: Distancia mínima entre agentes (por defecto 80.0)
            avoid_resources: Si True, evita spawnear sobre recursos (por defecto True)
            max_tries: Máximo número de intentos por posición (por defecto 5000)
            
        Returns:
            list: Lista de tuplas (x, y) con las posiciones de spawn
        """
        # Calcular mitad del tamaño del agente para los límites
        half = Specie.AGENT_SIZE / 2
        positions = []  # Lista de posiciones generadas

        # Generar una posición para cada agente
        for k in range(n):
            ok = False      # Flag de posición válida encontrada
            tries = 0       # Contador de intentos
            
            while (not ok) and tries < max_tries:
                tries += 1
                
                # Generar posición aleatoria dentro de los límites del mapa
                x = np.random.uniform(half, self.map_width  - half)
                y = np.random.uniform(half, self.map_height - half)

                # 1) Verificar colisión con recursos (si está habilitado)
                if avoid_resources:
                    # Verificar colisión con vegetación
                    if self._veg_tree is not None:
                        if self.collide_resources((x, y), Specie.AGENT_SIZE, self.vegetation).size:
                            continue  # Colisiona, intentar otra posición
                    
                    # Verificar colisión con agua
                    if self._wat_tree is not None:
                        if self.collide_resources((x, y), Specie.AGENT_SIZE, self.water_sources).size:
                            continue  # Colisiona, intentar otra posición

                # 2) Verificar separación mínima con otros agentes ya colocados
                too_close = any(
                    ( (x - px)**2 + (y - py)**2 )**0.5 < min_dist
                    for (px, py) in positions
                )
                if too_close:
                    continue  # Demasiado cerca, intentar otra posición

                # Posición válida encontrada
                positions.append((x, y))
                ok = True

            # Si no se encontró posición válida (muy raro), usar fallback
            # Colocar en el centro con un offset basado en el índice del agente
            if not ok:
                positions.append((self.map_width/2 + k*half, self.map_height/2))

        return positions
