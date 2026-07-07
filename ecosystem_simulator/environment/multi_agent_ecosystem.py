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

@dataclass
class AgentMemory:
    # Última dirección conocida hacia agua
    last_water_dx: float = 0.0
    last_water_dy: float = 0.0
    last_water_dist: float = 1.0
    water_seen: float = 0.0
    water_age: float = 1.0   # normalizado 0..1, 0 = recién visto, 1 = muy viejo

    # Última dirección conocida hacia vegetación
    last_veg_dx: float = 0.0
    last_veg_dy: float = 0.0
    last_veg_dist: float = 1.0
    veg_seen: float = 0.0
    veg_age: float = 1.0

    # Memoria de interacción / rendimiento reciente
    last_reward: float = 0.0
    reward_ema: float = 0.0          # promedio suavizado reciente
    failed_action_streak: int = 0
    danger_steps_recent: int = 0     # pasos recientes con depredador cerca o riesgo

# Direcciones posibles para el movimiento
DIRECTIONS = ["north", "south", "east", "west"]
ACT_EAT    = 4
ACT_DRINK  = 5
ACT_ATTACK = 6
BASE_STEP = 2.0
# BASE_COST drena food Y water en cada acción. Con 0.50 los agentes morían de
# hambre/sed ~paso 130 de 350 antes de aprender a comer de forma sostenida.
# 0.35 les da margen para que la política aprenda a consumir a tiempo.
BASE_COST = 0.35
MOVE_EXTRA = 0.05
PRED_ATTACK_CD = 2  # pasos de cooldown después de atacar

# Pesos de recompensa ajustables por el LLM (Fase 2: reward shaping).
# Cada peso MULTIPLICA un término de la recompensa. El default 1.0 reproduce
# EXACTAMENTE el comportamiento base, así que sin shaping nada cambia. El LLM
# los sube/baja (acotado) entre iteraciones para reorientar el aprendizaje.
#
# POR ROL: los términos compartidos (beber, homeostasis, explorar) tienen versión
# herbívoro y depredador separadas, para que el LLM pueda equilibrar ambos roles
# de forma independiente (antes un solo 'drink' global forzaba a los depredadores
# a sobre-beber cuando subía el del herbívoro, y morían de hambre).
DEFAULT_REWARD_WEIGHTS = {
    # Herbívoro
    "eat":          1.0,  # comer vegetación (solo herbívoro)
    "escape":       1.0,  # huir de depredadores (solo herbívoro)
    "drink_herb":   1.0,  # beber (herbívoro)
    "homeo_herb":   1.0,  # homeostasis / supervivencia (herbívoro)
    "explore_herb": 1.0,  # acercarse a recursos con propósito (herbívoro)
    # Depredador
    "hunt":         1.0,  # golpear/matar presa (solo depredador)
    "drink_pred":   1.0,  # beber (depredador)
    "homeo_pred":   1.0,  # homeostasis / supervivencia (depredador)
    "explore_pred": 1.0,  # acercarse a presa/agua con propósito (depredador)
}
# Rango seguro: el LLM no puede poner pesos absurdos que destruyan el balance.
REWARD_WEIGHT_MIN = 0.5
REWARD_WEIGHT_MAX = 2.0


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
        n_predators: int = 1,
        herbivore_vision_radius: float | None = 220.0,
        # Visión de recursos subida de 140 a 240: con 140 la comida/agua solía
        # quedar FUERA de la vista (parches separados ~180px) y el herbívoro
        # deambulaba a ciegas sin señal de hacia dónde ir => casi no comía/bebía
        # y moría de hambre/sed. Un grazer debe ver su comida al menos tan lejos
        # como ve a los depredadores.
        herbivore_resource_vision_radius: float | None = 240.0,
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
        self.herbivore_vision_radius = herbivore_vision_radius
        self.herbivore_resource_vision_radius = herbivore_resource_vision_radius
        self.action_spaces = {a: Discrete(7) for a in self.agents}
        self.memory_horizon = 25  # pasos antes de considerar una memoria "vieja"

        # Memoria corta por agente (se reinicia cada episodio)
        self._agent_memory = {
            a: AgentMemory() for a in self.agents
        }

        # Memoria larga simple entre episodios
        # [food_risk, water_risk, danger, exploration_need]
        self._ltm_bias = {
            a: np.zeros(4, dtype=np.float32) for a in self.agents
        }
        # --- Definición de espacios de observación y acción ---
        # Raíz cuadrada de 2 para normalizar distancias diagonales máximas
        SQRT2 = np.sqrt(2.0)

        # 34 features:
        # 0-15  : observación base actual
        # 16-20 : memoria agua   (dx, dy, dist, seen, age)
        # 21-25 : memoria veg    (dx, dy, dist, seen, age)
        # 26-29 : memoria reciente (last_reward, reward_ema, fail_streak, danger_recent)
        # 30-33 : memoria larga  (food_risk, water_risk, danger, exploration_need)

        low = np.array(
            [
                # Base 16
                0, 0, -1, -1, -1, -1,
                0, 0,
                0, 0,
                0, 0,
                -1, -1, 0, 0,

                # Water memory
                -1, -1, 0, 0, 0,

                # Veg memory
                -1, -1, 0, 0, 0,

                # Recent performance
                -1, -1, 0, 0,

                # Long-term bias
                0, 0, 0, 0
            ],
            dtype=np.float32
        )

        high = np.array(
            [
                # Base 16
                1, 1, 1, 1, 1, 1,
                SQRT2, SQRT2,
                1, 1,
                1, 1,
                1, 1, SQRT2, 1,

                # Water memory
                1, 1, SQRT2, 1, 1,

                # Veg memory
                1, 1, SQRT2, 1, 1,

                # Recent performance
                1, 1, 1, 1,

                # Long-term bias
                1, 1, 1, 1
            ],
            dtype=np.float32
        )

        self.observation_spaces = {
            a: Box(low, high, dtype=np.float32) for a in self.agents
        }
        
        # Contador de pasos en el episodio actual
        self._step_count = 0
        
        # Inicializar especies
        self._reset_species()
        
        # Umbrales para bonificación por homeostasis (mantener recursos altos)
        self.success_thr = 0.90     # 90% de las reservas máximas
        self.success_hold = 25      # Mantenerlo durante 25 pasos seguidos

        # Pesos de recompensa (Fase 2: los ajusta el LLM entre iteraciones).
        # Persisten entre episodios; reset() NO los toca.
        self.reward_weights = dict(DEFAULT_REWARD_WEIGHTS)

    def _rw_role(self, base: str, sp: Specie) -> float:
        """Peso por rol para términos compartidos (base = 'drink'|'homeo'|'explore')."""
        suffix = "pred" if sp.role is Role.PREDATOR else "herb"
        return self.reward_weights[f"{base}_{suffix}"]

    def set_reward_weights(self, weights: dict) -> None:
        """
        Actualiza los pesos de recompensa (llamado desde los env runners vía
        foreach_env). Ignora claves desconocidas y acota cada valor al rango
        seguro para que el LLM no pueda destruir el balance.
        """
        if not weights:
            return
        for k, v in weights.items():
            if k in self.reward_weights:
                try:
                    self.reward_weights[k] = float(
                        max(REWARD_WEIGHT_MIN, min(REWARD_WEIGHT_MAX, v))
                    )
                except (TypeError, ValueError):
                    pass

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
                # Ventaja de velocidad PEQUEÑA para la presa (3.1 vs 2.9). A igual
                # velocidad, una presa que huye solo mantenía la distancia => el
                # reward de escape (8*Δdist) era ~0 y nunca aprendía a huir. Con esta
                # ventaja, huir abre distancia de a poco y le permite salir de la zona
                # de peligro, así que huir SÍ rinde. El depredador igual la alcanza
                # cuando la presa se detiene a comer/beber (no se mueve ese paso).
                speed = 3.1
                hp = 100.0
                attack_range = 0.0
                attack_cost = 0.0
                max_food, max_water = 100.0, 100.0
                # Los vegetales se digieren rápido: desgaste de comida normal.
                food_metab_factor = 1.0
            else:
                speed = 2.9           # un pelín más lento que la presa (ver nota arriba)
                hp = 120.0
                # Alcance 52: 55 hacía la caza demasiado fácil (exterminio, 81%);
                # 40 la volvía imposible (morían de hambre). 48 balanceaba, pero al
                # mejorar la visión de los herbívoros (más evasivos y sanos) la caza
                # se volvió a poner difícil, así que subimos a 52 para compensar. El
                # exterminio no vuelve porque el reward de matar es puro-hambre (un
                # depredador lleno casi no gana cazando).
                attack_range = 52.0
                attack_cost = 0.5
                max_food, max_water = 100.0, 100.0
                # La CARNE dura MUCHO más que los vegetales (una comida grande llena
                # por largo rato): el depredador gasta su comida a 0.5x, así se
                # mantiene lleno más tiempo, su hambre (f_def) baja lento y NO tiene
                # urgencia de cazar de más. Antes 0.7 los dejaba sobrealimentados
                # (avg_food 0.74) pero igual sobre-cazaban (5.4 kills) y exterminaban.
                food_metab_factor = 0.5

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
                food_metab_factor=food_metab_factor,
            )
            s.attack_cd = 0  # arranca sin cooldown
            self.species.append(s)
                    
    def observation_space(self, agent: str) -> gym.Space:
        """
        Retorna el espacio de observación para un agente específico.
        
        Args:
            agent: ID del agente (ej. "agent_0")
            
        Returns:
            gym.Space: Espacio de observación (Box de 34 dimensiones)
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
        # Reiniciar memoria corta por episodio
        self._agent_memory = {
            a: AgentMemory() for a in self.possible_agents
        }
        # Asegurar que exista memoria larga para todos los agentes
        for a in self.possible_agents:
            if a not in self._ltm_bias:
                self._ltm_bias[a] = np.zeros(4, dtype=np.float32)
        # Contadores por episodio
        self._ep_return = {a: 0.0 for a in self.possible_agents}
        self._ep_len    = {a: 0   for a in self.possible_agents}
        self._satiated  = {a: 0   for a in self.possible_agents}
        self._ep_eat = {a: 0 for a in self.possible_agents}
        self._ep_drink = {a: 0 for a in self.possible_agents}

        self._ep_attack_attempt = {a: 0 for a in self.possible_agents}
        self._ep_attack_hit = {a: 0 for a in self.possible_agents}
        self._ep_attack_kill = {a: 0 for a in self.possible_agents}

        self._ep_food_sum = {a: 0.0 for a in self.possible_agents}
        self._ep_water_sum = {a: 0.0 for a in self.possible_agents}
        self._ep_critical_steps = {a: 0 for a in self.possible_agents}
        # Observaciones iniciales
        observations = {
            agent: self._get_obs(self.species[self._agent_idx[agent]], agent)
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
    
    def _norm_dist_to_predator(self, sp: Specie) -> float:
        if sp.role is not Role.HERBIVORE:
            return 1.0

        preds = [a for a in self.species if a.alive and a.role is Role.PREDATOR]
        if not preds:
            return 1.0

        coords = np.array([(p.x, p.y) for p in preds], dtype=np.float32)
        idx = np.argmin(((coords[:, 0] - sp.x) ** 2 + (coords[:, 1] - sp.y) ** 2))
        px, py = coords[idx]

        d = ((px - sp.x) ** 2 + (py - sp.y) ** 2) ** 0.5
        return d / ((self.map_width ** 2 + self.map_height ** 2) ** 0.5)
    
    def _age_memories(self):
        for agent in self.agents:
            mem = self._agent_memory[agent]

            if mem.water_seen > 0.0:
                mem.water_age = min(1.0, mem.water_age + 1.0 / self.memory_horizon)
                if mem.water_age >= 1.0:
                    mem.water_seen = 0.0

            if mem.veg_seen > 0.0:
                mem.veg_age = min(1.0, mem.veg_age + 1.0 / self.memory_horizon)
                if mem.veg_age >= 1.0:
                    mem.veg_seen = 0.0

            if mem.danger_steps_recent > 0:
                mem.danger_steps_recent = max(0, mem.danger_steps_recent - 1)

    def _update_agent_memory_pre_action(self, agent: str, sp: Specie):
        """
        Actualiza memoria corta antes de ejecutar la acción:
        - última info de agua visible
        - última info de vegetación visible
        - peligro reciente (para herbívoros)
        """
        mem = self._agent_memory[agent]

        # Solo restringimos recursos para herbívoros
        resource_radius = (
            self.herbivore_resource_vision_radius
            if sp.role is Role.HERBIVORE
            else None
        )

        # --- Agua visible ---
        dx_w, dy_w, dist_w, w_avail = self._nearest_resource_features(
            sp,
            kind="water",
            max_radius=resource_radius,
        )

        if w_avail > 0.0:
            mem.last_water_dx = float(dx_w)
            mem.last_water_dy = float(dy_w)
            mem.last_water_dist = float(dist_w)
            mem.water_seen = 1.0
            mem.water_age = 0.0

        # --- Vegetación visible ---
        dx_v, dy_v, dist_v, v_avail = self._nearest_resource_features(
            sp,
            kind="veg",
            max_radius=resource_radius,
        )

        if v_avail > 0.0:
            mem.last_veg_dx = float(dx_v)
            mem.last_veg_dy = float(dy_v)
            mem.last_veg_dist = float(dist_v)
            mem.veg_seen = 1.0
            mem.veg_age = 0.0

        # --- Peligro reciente ---
        if sp.role is Role.HERBIVORE:
            _, _, pred_dist, pred_avail = self._nearest_agent_features(
                sp,
                target_role=Role.PREDATOR,
                max_radius=self.herbivore_vision_radius,
            )

            if pred_avail > 0.0 and pred_dist < 0.25:
                mem.danger_steps_recent = min(
                    self.memory_horizon,
                    mem.danger_steps_recent + 3
                )


    def _update_agent_memory_post_action(
        self,
        agent: str,
        action: int | None,
        reward: float,
        info: dict,
    ):
        """
        Actualiza memoria corta después de la acción:
        - reward reciente
        - promedio suavizado de reward
        - racha de acciones fallidas
        """
        mem = self._agent_memory[agent]

        mem.last_reward = float(reward)
        mem.reward_ema = float(0.9 * mem.reward_ema + 0.1 * reward)

        failed = False

        if action is None:
            failed = False
        elif action == ACT_EAT:
            failed = info.get("eat_success", 0) == 0
        elif action == ACT_DRINK:
            failed = info.get("drink_success", 0) == 0
        elif action == ACT_ATTACK:
            failed = info.get("attack_outcome", "") in ("", "miss", "cooldown")
        elif 0 <= action <= 3:
            failed = reward < -0.05

        if failed:
            mem.failed_action_streak = min(10, mem.failed_action_streak + 1)
        else:
            mem.failed_action_streak = 0


    def _update_long_term_bias(self, agent: str, terminal_reason: str):
        """
        Construye una memoria larga simple entre episodios:
        [food_risk, water_risk, danger, exploration_need]
        """
        ep_len = max(1, self._ep_len[agent])

        avg_food = self._ep_food_sum[agent] / ep_len
        avg_water = self._ep_water_sum[agent] / ep_len
        critical_ratio = self._ep_critical_steps[agent] / ep_len

        # Riesgo por recursos
        food_risk = float(1.0 - avg_food)
        water_risk = float(1.0 - avg_water)

        # Riesgo / peligro
        danger = 1.0 if terminal_reason == "predation" else min(
            1.0, self._agent_memory[agent].danger_steps_recent / 5.0
        )

        # Necesidad de exploración:
        # si casi no encontró comida/agua en un episodio relativamente largo
        total_intake = self._ep_eat[agent] + self._ep_drink[agent]
        if ep_len > 40 and total_intake <= 1:
            exploration_need = 1.0
        else:
            exploration_need = min(1.0, critical_ratio * 1.5)

        new_bias = np.array(
            [food_risk, water_risk, danger, exploration_need],
            dtype=np.float32
        )

        # EMA para que la memoria larga no cambie de golpe
        self._ltm_bias[agent] = 0.7 * self._ltm_bias[agent] + 0.3 * new_bias

    def _action_mask(self, sp: Specie) -> np.ndarray:
        # 7 acciones: mover(0..3), comer(4), beber(5), atacar(6)
        mask = np.ones(7, dtype=np.int8)
        if sp.role is Role.HERBIVORE:
            mask[ACT_ATTACK] = 0  # ahora ACT_ATTACK = 6, índice válido
        return mask
    
    def _resolve_water_claims(self, ctx: StepContext):
        return

    def _resolve_veg_claims(self, ctx: StepContext):
        return

    def _apply_step_bonus(self, ctx: StepContext):
        for a in ctx.prev_agents:
            ctx.rewards[a] = float(ctx.rewards.get(a, 0.0) + ctx.step_bonus.get(a, 0.0))
    def _init_step_context(self) -> StepContext:
        self._step_count += 1
        self._age_memories()
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
            a = None

            # Actualizar memoria antes de decidir
            self._update_agent_memory_pre_action(agent, sp)
            ctx.infos.setdefault(agent, {})
            ctx.infos[agent].update({
                "eat_success": 0,
                "drink_success": 0,
                "attack_attempt": 0,
                "attack_outcome": "",
                "attack_hit": 0,
                "attack_kill": 0,
            })
            # Cooldown tick aquí (más simple que hacerlo al final)
            if sp.attack_cd > 0:
                sp.attack_cd -= 1

            # Si la presa ya murió este mismo paso (un depredador se procesa antes
            # y la mató), no dejamos que "actúe" estando muerta.
            if not sp.alive:
                ctx.terminations[agent] = True
                ctx.truncations[agent] = False
                ctx.rewards[agent] = float(reward)
                role_str = "PREDATOR" if sp.role is Role.PREDATOR else "HERBIVORE"
                ctx.infos[agent] = {
                    **ctx.infos.get(agent, {}),
                    "reason": "predation",
                    "role": role_str,
                    "action_mask": self._action_mask(sp),
                }
                continue

            if agent in actions:
                prev_energy = sp.total_energy

                # déficit de recursos
                f_def = 1.0 - (sp.food / sp.max_food)
                w_def = 1.0 - (sp.water / sp.max_water)

                if sp.role is Role.PREDATOR:
                    need = "water" if w_def > f_def else "prey"
                    d_prev = (
                        self._norm_dist_to_prey(sp)
                        if need == "prey"
                        else self._norm_dist_to(sp, "water")
                    )
                else:
                    pred_dx, pred_dy, pred_dist, pred_avail = self._nearest_agent_features(
                        sp,
                        target_role=Role.PREDATOR,
                        max_radius=self.herbivore_vision_radius,
                    )

                    # Si hay amenaza visible y está bastante cerca, prioriza huir
                    if pred_avail > 0.0 and pred_dist < 0.25:
                        need = "escape"
                        d_prev = pred_dist
                        resource_visible_prev = 0.0
                    else:
                        need = "water" if w_def > f_def else "veg"

                        _, _, d_prev_visible, resource_visible_prev = self._nearest_resource_features(
                            sp,
                            kind=need,
                            max_radius=self.herbivore_resource_vision_radius,
                        )

                        # Si no ve recurso, usamos distancia "máxima" para indicar ausencia visual
                        d_prev = d_prev_visible if resource_visible_prev > 0.0 else 1.0

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
                        reward -= 0.1

                    # progreso hacia objetivo
                    if need == "prey":
                        d_now = self._norm_dist_to_prey(sp)
                        reward += self._rw_role("explore", sp) * 6.0 * (d_prev - d_now)

                        diag = (self.map_width**2 + self.map_height**2) ** 0.5
                        in_range = 1.0 if d_now <= (sp.attack_range / diag) else 0.0
                        # Recompensa por posicionarse a distancia de ataque, escalada
                        # por hambre: un depredador hambriento gana más por ubicarse
                        # para cazar (antes 0.5 fijo => cazar apenas competía con beber).
                        reward += (1.0 + 2.0 * f_def) * in_range

                    elif need == "escape":
                        d_now = self._norm_dist_to_predator(sp)

                        # Recompensar aumentar distancia al depredador
                        reward += self.reward_weights["escape"] * 8.0 * (d_now - d_prev)

                        # Pequeña penalización por quedarse muy cerca
                        if d_now < 0.12:
                            reward -= 1.0
                        elif d_now < 0.20:
                            reward -= 0.3

                    else:
                        # Herbívoro: solo reward denso por recurso si el recurso está visible
                        if sp.role is Role.HERBIVORE:
                            _, _, d_now_visible, resource_visible_now = self._nearest_resource_features(
                                sp,
                                kind=need,
                                max_radius=self.herbivore_resource_vision_radius,
                            )

                            if resource_visible_now > 0.0:
                                d_now = d_now_visible
                                # Progreso (potential-based). Coeficiente bajado de
                                # 6.0 a 3.0: con 6.0 el agente farmeaba el reward de
                                # ACERCARSE sin llegar a comer. Ahora acercarse es un
                                # incentivo suave y COMER (abajo) domina claramente.
                                reward += self._rw_role("explore", sp) * 3.0 * (d_prev - d_now)
                            else:
                                # Si no ve recurso, no damos reward privilegiado.
                                # Pequeño incentivo a explorar.
                                reward += 0.02

                        else:
                            # Depredador se queda como estaba
                            d_now = self._norm_dist_to(sp, need)
                            reward += self._rw_role("explore", sp) * 6.0 * (d_prev - d_now)
                    ctx.energy_after_move[agent] = sp.total_energy
                    reward += 0.02 * (ctx.energy_after_move[agent] - prev_energy)

                # --- comer ---
                elif a == ACT_EAT:
                    sp.metabolize(BASE_COST)

                    if sp.role is Role.PREDATOR:
                        # depredador no come vegetación
                        prey_now = self._norm_dist_to_prey(sp)
                        reward += 1.0 * (prey_prev - prey_now)
                    else:
                        ate = self.try_eat_vegetation(sp, bite_gain=20.0)
                        ctx.infos[agent]["eat_success"] = int(ate)
                        if ate:
                            self._ep_eat[agent] += 1
                            # SIN componente fijo: puro déficit (f_def). El fijo era
                            # lo que permitía farmear (comer lleno seguía dando +2).
                            # Ahora comer con la comida llena da ~0, así que el agente
                            # solo come cuando de verdad tiene hambre.
                            reward += self.reward_weights["eat"] * 9.0 * f_def
                        else:
                            reward -= 0.2  # intentar comer sin recurso cuesta un poco

                # --- beber ---
                elif a == ACT_DRINK:
                    sp.metabolize(BASE_COST)
                    drank = self.try_drink(sp, sip_gain=15.0)
                    ctx.infos[agent]["drink_success"] = int(drank)
                    if drank:
                        self._ep_drink[agent] += 1
                        # SIN componente fijo: puro déficit (w_def). Beber con el agua
                        # llena da ~0, así que el depredador ya no puede farmear agua
                        # (era lo que lo hacía llegar a 0.89 de agua y morir de hambre).
                        reward += self._rw_role("drink", sp) * 9.0 * w_def
                    else:
                        reward -= 0.2

                # --- atacar ---
                elif a == ACT_ATTACK:
                    sp.metabolize(BASE_COST)

                    if sp.role is Role.PREDATOR:
                        ctx.infos[agent]["attack_attempt"] = 1
                        self._ep_attack_attempt[agent] += 1

                        # Compuerta por sed: cuando el depredador tiene poca agua,
                        # cazar rinde menos para que priorice beber. Sin esto,
                        # aprendía a cazar sin parar y moría deshidratado (~58%).
                        # Con agua >= 0.4 no hay penalización (caza libre); por
                        # debajo, el premio de caza se reduce proporcionalmente.
                        w_norm_pred = sp.water / sp.max_water
                        thirst_gate = min(1.0, w_norm_pred / 0.4)

                        if sp.attack_cd > 0:
                            reward -= 0.05
                            outcome = "cooldown"
                        else:
                            outcome = self.try_attack(sp, self.species, dmg=50.0)

                            if outcome == "miss":
                                reward -= 0.1
                            elif outcome == "hit":
                                # SIN componente fijo (puro hambre f_def), igual que
                                # comer/beber. El fijo (antes 10+10) dejaba farmear
                                # kills estando lleno => sobre-caza y exterminio.
                                reward += self.reward_weights["hunt"] * (20.0 * f_def) * thirst_gate
                                sp.attack_cd = PRED_ATTACK_CD
                            elif outcome == "kill":
                                # Puro hambre: un depredador lleno (f_def bajo) casi no
                                # gana por matar, así que caza solo cuando lo necesita.
                                reward += self.reward_weights["hunt"] * (50.0 * f_def) * thirst_gate
                                sp.attack_cd = PRED_ATTACK_CD

                        ctx.infos[agent]["attack_outcome"] = outcome
                        ctx.infos[agent]["attack_hit"] = 1 if outcome in ("hit", "kill") else 0
                        ctx.infos[agent]["attack_kill"] = 1 if outcome == "kill" else 0

                        if outcome in ("hit", "kill"):
                            self._ep_attack_hit[agent] += 1
                        if outcome == "kill":
                            self._ep_attack_kill[agent] += 1

                    else:
                        # Herbívoro atacando: acción inútil (no puede) que lo deja
                        # quieto ese paso. La máscara de acción no se aplica en este
                        # stack de RLlib. Penalizamos IGUAL que un comer/beber fallido
                        # (-0.2): así ninguna acción quieta es un "refugio" más barato
                        # que otra. Como comer/beber SÍ pueden tener éxito y atacar
                        # nunca, la política prefiere las útiles sin spamear una sola.
                        # (Con -0.5 la masa se iba toda a COMER-en-el-aire.)
                        reward -= 0.2

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

            # recompensa homeostática (el motor principal de supervivencia)
            if not done_term:
                f_norm = sp.food / sp.max_food
                w_norm = sp.water / sp.max_water
                min_norm = min(f_norm, w_norm)

                # DOMINANTE (subido de 0.02 a 0.4): premia el ESTADO de mantener el
                # recurso más BAJO alto. No es farmeable (min está acotado a 1.0) y,
                # clave: llenar el recurso que YA está alto no sube el min => no da
                # reward. Solo subir el cuello de botella (el recurso bajo) rinde.
                # Esto obliga a equilibrar comida Y agua = sobrevivir, en vez de
                # farmear un solo recurso.
                reward += self._rw_role("homeo", sp) * 0.4 * min_norm
                # Penalizaciones por estar en zona crítica (marcan peligro real).
                if min_norm < 0.15:
                    reward -= 0.1
                if min_norm < 0.08:
                    reward -= 0.2

                # penalización por bordes
                dist_left   = sp.x / self.map_width
                dist_right  = (self.map_width - sp.x) / self.map_width
                dist_top    = sp.y / self.map_height
                dist_bottom = (self.map_height - sp.y) / self.map_height
                dist_to_edge = min(dist_left, dist_right, dist_top, dist_bottom)

                edge_margin = 0.15
                if dist_to_edge < edge_margin:
                    reward -= 0.15 * (edge_margin - dist_to_edge) / edge_margin

            # Actualizar memoria post acción SIEMPRE
            self._update_agent_memory_post_action(
                agent=agent,
                action=a,
                reward=reward,
                info=ctx.infos[agent],
            )

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

        missing_veg = max(0, target_veg - cur_veg)
        missing_wat = max(0, target_wat - cur_wat)

        if missing_veg > 0:
            spawn_veg = max(1, missing_veg // 2)
            self.spawn_vegetation(n=spawn_veg)

        if missing_wat > 0:
            spawn_wat = max(1, missing_wat // 2)
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
            sp = self.species[self._agent_idx[agent]]
            ctx.obs[agent] = self._get_obs(sp, agent)

            ctx.infos.setdefault(agent, {})

            # Métricas básicas por episodio
            self._ep_return[agent] = self._ep_return.get(agent, 0.0) + ctx.rewards.get(agent, 0.0)
            self._ep_len[agent] = self._ep_len.get(agent, 0) + 1

            # Estado promedio de recursos
            f_norm = sp.food / sp.max_food
            w_norm = sp.water / sp.max_water
            self._ep_food_sum[agent] += f_norm
            self._ep_water_sum[agent] += w_norm

            # Estado crítico
            if min(f_norm, w_norm) < 0.15:
                self._ep_critical_steps[agent] += 1

            ep_len = max(1, self._ep_len[agent])

            # Guardar resumen acumulado en info
            ctx.infos[agent]["ep_return"] = float(self._ep_return[agent])
            ctx.infos[agent]["ep_len"] = int(self._ep_len[agent])

            ctx.infos[agent]["ep_eat"] = int(self._ep_eat[agent])
            ctx.infos[agent]["ep_drink"] = int(self._ep_drink[agent])

            ctx.infos[agent]["ep_attack_attempt"] = int(self._ep_attack_attempt[agent])
            ctx.infos[agent]["ep_attack_hit"] = int(self._ep_attack_hit[agent])
            ctx.infos[agent]["ep_attack_kill"] = int(self._ep_attack_kill[agent])

            ctx.infos[agent]["ep_avg_food"] = float(self._ep_food_sum[agent] / ep_len)
            ctx.infos[agent]["ep_avg_water"] = float(self._ep_water_sum[agent] / ep_len)
            ctx.infos[agent]["ep_critical_steps"] = int(self._ep_critical_steps[agent])
            ctx.infos[agent]["ep_critical_ratio"] = float(self._ep_critical_steps[agent] / ep_len)

            if ctx.terminations.get(agent, False) or ctx.truncations.get(agent, False):
                terminal_reason = ctx.infos[agent].get("reason", "")
                self._update_long_term_bias(agent, terminal_reason)


    def _apply_timeout_if_needed(self, ctx: StepContext):
        if ctx.episode_trunc and len(ctx.alive) > 0:
            for a in list(ctx.alive):
                ctx.truncations[a] = True
                ctx.infos[a]["reason"] = "timeout"
                self._update_long_term_bias(a, "timeout")
            ctx.alive = []
    def step(self, actions):
        ctx = self._init_step_context()
        self._process_agents(actions, ctx)
        self._apply_step_bonus(ctx)
        self._apply_post_move_energy_shaping(ctx)
        self._respawn_resources_if_needed(ctx)
        self._prune_depleted_resources()
        self._finalize_predation_and_cooldowns(ctx)
        self._build_obs_and_episode_metrics(ctx)
        self._apply_timeout_if_needed(ctx)
        self.agents = ctx.alive
        return ctx.obs, ctx.rewards, ctx.terminations, ctx.truncations, ctx.infos
    def _nearest_agent_features(
        self,
        sp: Specie,
        target_role: Role,
        max_radius: float | None = None,
    ):
        """
        Retorna (dx, dy, dist, avail) del agente vivo más cercano con el rol target_role.

        dx, dy se normalizan por ancho/alto del mapa.
        dist se normaliza y queda en [0, sqrt(2)] aproximadamente.
        avail = 1.0 si existe objetivo visible, 0.0 si no.
        """

        targets = [
            a for a in self.species
            if a.alive and a.role is target_role and a is not sp
        ]

        if not targets:
            return 0.0, 0.0, 1.0, 0.0

        coords = np.array([(t.x, t.y) for t in targets], dtype=np.float32)
        d2 = (coords[:, 0] - sp.x) ** 2 + (coords[:, 1] - sp.y) ** 2
        idx = int(np.argmin(d2))

        tx, ty = coords[idx]
        real_dist = float(np.sqrt(d2[idx]))

        # Si hay radio de visión y está fuera, no se detecta
        if max_radius is not None and real_dist > max_radius:
            return 0.0, 0.0, 1.0, 0.0

        dx = (tx - sp.x) / self.map_width
        dy = (ty - sp.y) / self.map_height
        dist = float(np.sqrt(dx**2 + dy**2))

        return dx, dy, dist, 1.0
    
    def _nearest_resource_features(
        self,
        sp: Specie,
        kind: str,
        max_radius: float | None = None,
    ):
        """
        Retorna (dx, dy, dist, avail) del recurso más cercano.

        kind: "water" o "veg"
        dx, dy normalizados por ancho/alto del mapa.
        dist normalizada.
        avail = 1.0 si existe recurso visible dentro del radio, 0.0 si no.
        """
        if kind == "water":
            tree = self._wat_tree
            src = self.water_sources
        else:
            tree = self._veg_tree
            src = self.vegetation

        if tree is None or src["centers"].size == 0:
            return 0.0, 0.0, 1.0, 0.0

        _, idx = tree.query([sp.x, sp.y], k=1)
        rx, ry = src["centers"][idx]

        real_dx = float(rx - sp.x)
        real_dy = float(ry - sp.y)
        real_dist = float(np.sqrt(real_dx**2 + real_dy**2))

        # Si hay radio de visión y el recurso está fuera, no se detecta
        if max_radius is not None and real_dist > max_radius:
            return 0.0, 0.0, 1.0, 0.0

        # dx, dy se dejan por eje (para dar dirección a la red), pero la
        # distancia usada en el shaping se normaliza por la DIAGONAL del mapa
        # para que sea isotrópica (moverse la misma distancia física vale igual
        # en cualquier dirección).
        diag = float(np.sqrt(self.map_width**2 + self.map_height**2))
        dx = real_dx / self.map_width
        dy = real_dy / self.map_height
        dist = real_dist / diag

        return dx, dy, dist, 1.0

    def _get_obs(self, sp: Specie, agent: str) -> np.ndarray:
        """
        Observación de 34 dimensiones:
        0-15  : observación base
        16-20 : memoria de agua
        21-25 : memoria de vegetación
        26-29 : memoria reciente
        30-33 : memoria larga
        """
        # Para herbívoros, recursos solo visibles dentro de un radio
        resource_radius = (
            self.herbivore_resource_vision_radius
            if sp.role is Role.HERBIVORE
            else None
        )

        dx_w, dy_w, dist_w, w_avail = self._nearest_resource_features(
            sp,
            kind="water",
            max_radius=resource_radius,
        )

        dx_v, dy_v, dist_v, v_avail = self._nearest_resource_features(
            sp,
            kind="veg",
            max_radius=resource_radius,
        )

        role_h = 1.0 if sp.role is Role.HERBIVORE else 0.0
        role_p = 1.0 if sp.role is Role.PREDATOR else 0.0

        # Animal relevante
        other_dx = other_dy = 0.0
        other_dist = 1.0
        other_avail = 0.0

        if sp.role is Role.PREDATOR:
            other_dx, other_dy, other_dist, other_avail = self._nearest_agent_features(
                sp,
                target_role=Role.HERBIVORE,
                max_radius=None,
            )
        else:
            other_dx, other_dy, other_dist, other_avail = self._nearest_agent_features(
                sp,
                target_role=Role.PREDATOR,
                max_radius=self.herbivore_vision_radius,
            )

        f_norm = sp.food / sp.max_food
        w_norm = sp.water / sp.max_water

        # --- Memoria ---
        mem = self._agent_memory[agent]
        ltm = self._ltm_bias[agent]

        # Normalizaciones para que entren bien a la red
        last_reward_norm = float(np.tanh(mem.last_reward / 5.0))
        reward_ema_norm = float(np.tanh(mem.reward_ema / 5.0))
        fail_streak_norm = float(min(1.0, mem.failed_action_streak / 5.0))
        danger_recent_norm = float(min(1.0, mem.danger_steps_recent / 5.0))

        return np.array([
            # Base 16
            f_norm, w_norm,
            dx_w, dy_w, dx_v, dy_v,
            dist_w, dist_v,
            w_avail, v_avail,
            role_h, role_p,
            other_dx, other_dy, other_dist, other_avail,

            # Water memory 5
            mem.last_water_dx,
            mem.last_water_dy,
            mem.last_water_dist,
            mem.water_seen,
            mem.water_age,

            # Veg memory 5
            mem.last_veg_dx,
            mem.last_veg_dy,
            mem.last_veg_dist,
            mem.veg_seen,
            mem.veg_age,

            # Recent performance 4
            last_reward_norm,
            reward_ema_norm,
            fail_streak_norm,
            danger_recent_norm,

            # Long-term memory 4
            ltm[0], ltm[1], ltm[2], ltm[3],
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
