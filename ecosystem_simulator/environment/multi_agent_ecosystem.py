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
from gymnasium.spaces import Discrete, Box
from pettingzoo.utils import ParallelEnv
from typing import Tuple

from ..entities.specie import Specie
from .ecosystem import Ecosystem

# Acciones posibles: 5 direcciones de movimiento incluyendo quedarse quieto
DIRECTIONS = ["north", "south", "east", "west", "stay"]

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
        # Crear IDs de agentes: "agent_0", "agent_1", etc.
        self.agents    = [f"agent_{i}" for i in range(n_agents)]
        # Lista de todos los agentes posibles (para reset)
        self.possible_agents = list(self.agents)
        # Diccionario para mapeo rápido agente: índice en self.species
        self._agent_idx = {a: i for i, a in enumerate(self.agents)}
        
        self.max_steps = max_steps
        self.gamma     = gamma
    
        # --- Definición de espacios de observación y acción ---
        # Raíz cuadrada de 2 para normalizar distancias diagonales máximas
        SQRT2 = np.sqrt(2.0)
        # Límites inferiores de la observación
        low  = np.array([0,0,-1,-1,-1,-1, 0,   0,    0,0], dtype=np.float32)
        # Límites superiores de la observación
        high = np.array([1,1,  1,  1,  1,  1, SQRT2, SQRT2, 1,1], dtype=np.float32)
        
        # Espacio de acción: discreto con 5 opciones (norte, sur, este, oeste, quedarse)
        self.action_spaces = {a: Discrete(len(DIRECTIONS)) for a in self.agents}
        # Espacio de observación: continuo con 10 características normalizadas
        self.observation_spaces = {a: Box(low, high, dtype=np.float32) for a in self.agents}
        
        # Contador de pasos en el episodio actual
        self._step_count = 0
        
        # Inicializar especies
        self._reset_species()
        
        # Umbrales para bonificación por homeostasis (mantener recursos altos)
        self.success_thr = 0.90     # 90% de las reservas máximas
        self.success_hold = 25      # Mantenerlo durante 25 pasos seguidos
    
    def _reset_species(self):
        """
        Inicializa o reinicia todas las especies (agentes) en el mapa.
        
        Genera posiciones de spawn aleatorias que evitan:
        - Estar demasiado cerca de otros agentes
        - Aparecer sobre recursos existentes
        
        Cada agente comienza con niveles de recursos entre 60% y 80% del máximo.
        """
        # Generar posiciones de spawn con restricciones espaciales
        spawn_xy = self._sample_spawn_positions(self.n_agents, min_dist=80.0, avoid_resources=True)

        # Función auxiliar para generar nivel inicial aleatorio entre 60% y 80%
        def init_level(max_val): 
            return float(np.random.uniform(0.60, 0.80) * max_val)

        # Crear lista de especies con posiciones y recursos iniciales
        self.species = [
            Specie(
                food=init_level(100),   # Comida inicial aleatoria
                water=init_level(100),  # Agua inicial aleatoria
                x=spawn_xy[i][0],       # Posición X del spawn
                y=spawn_xy[i][1],       # Posición Y del spawn
                map_width=self.map_width,
                map_height=self.map_height
            )
            for i in range(self.n_agents)
        ]
        
    def observation_space(self, agent: str) -> gym.Space:
        """
        Retorna el espacio de observación para un agente específico.
        
        Args:
            agent: ID del agente (ej. "agent_0")
            
        Returns:
            gym.Space: Espacio de observación (Box de 10 dimensiones)
        """
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> gym.Space:
        """
        Retorna el espacio de acción para un agente específico.
        
        Args:
            agent: ID del agente (ej. "agent_0")
            
        Returns:
            gym.Space: Espacio de acción (Discrete con 5 opciones)
        """
        return self.action_spaces[agent]

    def reset(self, *, seed: int = None, options: dict = None):
        """
        Reinicia el entorno a un estado inicial para un nuevo episodio.
        
        Reinicializa:
        - El ecosistema (recursos)
        - Las especies (agentes)
        - Contadores y métricas
        
        Args:
            seed: Semilla para generación aleatoria (opcional)
            options: Opciones adicionales (no usado actualmente)
            
        Returns:
            tuple: (observaciones, infos) para cada agente activo
        """
        # Establecer semilla si se proporciona
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Reiniciar el ecosistema con las densidades iniciales
        Ecosystem.__init__(self, self._init_veg, self._init_wat, self.map_width, self.map_height)
        
        # Reiniciar especies con nuevas posiciones
        self._reset_species()
        
        # Resetear contador de pasos
        self._step_count = 0

        # Repoblar la lista de agentes vivos para el nuevo episodio
        self._ep_return = {a: 0.0 for a in self.possible_agents}  # Retorno acumulado por agente
        self._ep_len    = {a: 0   for a in self.possible_agents}  # Longitud del episodio por agente
        self.agents = list(self.possible_agents)  # Todos los agentes comienzan vivos

        # Generar observaciones iniciales para cada agente
        observations = {agent: self._get_obs(self.species[i]) for i, agent in enumerate(self.agents)}
        infos = {agent: {} for agent in self.agents}
        
        # Inicializar contador de pasos con recursos altos (para bonificación homeostasis)
        self._satiated = {a: 0 for a in self.agents}
        
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
    
    def step(self, actions):
        """
        Ejecuta un paso de simulación para todos los agentes.
        
        Este es el método principal que procesa las acciones de todos los agentes,
        actualiza el estado del mundo, calcula recompensas, y determina
        terminaciones. El flujo es:
        
        1. Procesar acciones de agentes (movimiento, metabolismo)
        2. Detectar colisiones con recursos
        3. Resolver conflictos (múltiples agentes queriendo el mismo recurso)
        4. Calcular recompensas basadas en múltiples criterios
        5. Verificar condiciones de terminación
        6. Regenerar recursos consumidos
        7. Actualizar métricas
        
        Args:
            actions: Diccionario {agent_id: action} con acciones de cada agente
            
        Returns:
            tuple: (obs, rewards, terminations, truncations, infos)
                - obs: Observaciones para cada agente
                - rewards: Recompensas para cada agente
                - terminations: Flags de terminación por muerte
                - truncations: Flags de terminación por timeout
                - infos: Información adicional por agente
        """
        # Inicializar diccionarios de retorno
        obs, rewards, terminations, truncations, infos = {}, {}, {}, {}, {}
        
        # Incrementar contador de pasos del episodio
        self._step_count += 1

        # Guardar lista de agentes del paso anterior
        prev_agents   = list(self.agents)
        alive         = []  # Lista para agentes que sobreviven este paso
        
        # Verificar si el episodio alcanzó el máximo de pasos
        episode_trunc = (self._step_count >= self.max_steps)

        # Diccionarios para rastrear reclamos de recursos (múltiples agentes pueden reclamar el mismo)
        veg_claims, wat_claims = {}, {}  # {idx_recurso: [agente1, agente2, ...]}
        
        # Diccionario para bonificaciones por consumir recursos
        step_bonus = {a: 0.0 for a in prev_agents}
        
        # Diccionario para guardar energía después del movimiento (para reward shaping)
        energy_after_move = {}

        # Procesar agentes en orden aleatorio para evitar sesgos de orden
        proc_order = random.sample(prev_agents, len(prev_agents))

        # === LOOP PRINCIPAL: Procesar cada agente ===
        for agent in proc_order:
            # Obtener índice y objeto Specie del agente
            i  = self._agent_idx[agent]
            sp = self.species[i]

            # Inicializar variables de este paso para el agente
            reward = 0.0       # Recompensa acumulada
            done_term  = False # Terminación por muerte
            done_trunc = False # Terminación por timeout
            reason     = ""    # Razón de terminación

            # Solo procesar si el agente tiene una acción
            if agent in actions:
                # Guardar energía previa para reward shaping
                prev_energy = sp.total_energy

                # === DETERMINAR NECESIDAD DOMINANTE ===
                # Calcular déficit de cada recurso (0 = lleno, 1 = vacío)
                f_def   = 1.0 - (sp.food  / sp.max_food)   # Déficit de comida
                w_def   = 1.0 - (sp.water / sp.max_water)  # Déficit de agua
                
                # Determinar cuál recurso necesita más urgentemente
                need    = "water" if w_def > f_def else "veg"
                need_def = max(f_def, w_def)  # Magnitud del déficit máximo
                
                # Calcular distancia normalizada al recurso necesitado (antes del movimiento)
                d_prev  = self._norm_dist_to(sp, need)

                # === PROCESAR ACCIÓN DEL AGENTE ===
                act_dir_name = DIRECTIONS[actions[agent]]  # Convertir índice a nombre de dirección
                
                # Costos metabólicos: base compartido, movimiento tiene costo extra
                BASE_COST, MOVE_EXTRA = 0.30, 0.02

                if act_dir_name == "stay":
                    # Acción de quedarse quieto
                    sp.metabolize(BASE_COST)  # Solo costo metabólico base
                    sp.move(0.0, "stay")      # Sin movimiento
                    # Penalización por inacción (proporcional a la necesidad)
                    reward -= 0.01 + 0.05 * need_def
                else:
                    # Acción de movimiento
                    sp.metabolize(BASE_COST + MOVE_EXTRA)  # Costo base + extra por moverse
                    sp.move(2.0, act_dir_name)             # Mover con velocidad 2.0

                # === RECOMPENSA POR PROGRESO HACIA RECURSO ===
                # Calcular nueva distancia al recurso necesitado
                d_now = self._norm_dist_to(sp, need)
                # Recompensar si se acercó, penalizar si se alejó
                reward += 2.0 * (d_prev - d_now)

                # === REWARD SHAPING: Cambio de energía después del movimiento ===
                energy_after_move[agent] = sp.total_energy
                reward += 0.1 * (energy_after_move[agent] - prev_energy)

                # === DETECCIÓN DE COLISIONES CON RECURSOS ===
                # Verificar colisión con agua
                hit = self.collide_resources((sp.x, sp.y), Specie.AGENT_SIZE, self.water_sources)
                if hit.size:
                    # Registrar reclamo del agente sobre el primer recurso de agua colisionado
                    wat_claims.setdefault(int(hit[0]), []).append(agent)
                
                # Verificar colisión con vegetación
                hit = self.collide_resources((sp.x, sp.y), Specie.AGENT_SIZE, self.vegetation)
                if hit.size:
                    # Registrar reclamo del agente sobre el primer recurso de vegetación colisionado
                    veg_claims.setdefault(int(hit[0]), []).append(agent)

                # === VERIFICAR CONDICIONES DE MUERTE ===
                if (sp.food <= 0) or (sp.water <= 0):
                    # El agente murió por inanición o deshidratación
                    reward   -= 10.0  # Penalización fuerte por morir
                    done_term = True
                    reason    = "starvation" if sp.food <= 0 else "dehydration"

                # === BONIFICACIÓN POR HOMEOSTASIS (mantener recursos altos) ===
                if not done_term:
                    # Solo si el agente sigue vivo
                    if sp.food >= self.success_thr * sp.max_food and sp.water >= self.success_thr * sp.max_water:
                        # El agente mantiene ambos recursos por encima del 90%
                        self._satiated[agent] += 1
                        if self._satiated[agent] >= self.success_hold:
                            # Ha mantenido niveles altos por 25 pasos consecutivos
                            reward += 30.0  # Gran bonificación por supervivencia exitosa
                            # Resetear contador para que pueda ganar el bonus nuevamente
                            self._satiated[agent] = 0
                    else:
                        # Si no cumple la condición, resetear el contador
                        self._satiated[agent] = 0

                # === VERIFICAR TRUNCATION POR TIMEOUT ===
                if (not done_term) and episode_trunc:
                    done_trunc = True
                    reason = "timeout"

            # === RECOMPENSAS ADICIONALES BASADAS EN ESTADO DE SALUD ===
            if not done_term:
                # Solo aplicar estas recompensas si el agente sigue vivo
                # Normalizar niveles de recursos a rango [0, 1]
                f_norm = sp.food / sp.max_food
                w_norm = sp.water / sp.max_water
                min_norm = min(f_norm, w_norm)  # Recurso más crítico
                
                # Recompensa pequeña por mantener recursos
                reward += 0.02 * min_norm
                
                # Penalizaciones crecientes por niveles críticos
                if min_norm < 0.15: reward -= 0.5   # Nivel crítico
                if min_norm < 0.08: reward -= 1.0   # Nivel muy crítico

            # === GUARDAR RESULTADOS DEL AGENTE ===
            rewards[agent]      = float(reward)
            terminations[agent] = done_term
            truncations[agent]  = done_trunc
            infos[agent]        = {"reason": reason}

            # Agregar a la lista de vivos si no terminó ni se truncó
            if (not done_term) and (not done_trunc):
                alive.append(agent)

        # === RESOLVER CONFLICTOS POR RECURSOS DE AGUA ===
        # Cuando múltiples agentes reclaman el mismo recurso, gana el más necesitado
        removed_w = 0  # Contador de recursos de agua agotados
        for idx, claimers in wat_claims.items():
            # Filtrar solo los agentes que siguen vivos
            vivos = [a for a in claimers if not terminations.get(a, False)]
            if not vivos:
                continue  # Si nadie vivo reclama, pasar al siguiente
            
            # El ganador es el agente con menor ratio agua/max_agua (más sediento)
            winner = min(vivos, key=lambda a: self.species[self._agent_idx[a]].water /
                                        self.species[self._agent_idx[a]].max_water)
            
            # El ganador obtiene agua
            sp_w = self.species[self._agent_idx[winner]]
            sp_w.water = min(sp_w.water + 10, sp_w.max_water)  # +10 agua, sin exceder máximo
            
            # Bonificación por conseguir recurso
            step_bonus[winner] += 5.0
            
            # Consumir el recurso y verificar si se agotó
            if self.consume_water(idx):
                removed_w += 1

        # === RESOLVER CONFLICTOS POR RECURSOS DE VEGETACIÓN ===
        removed_v = 0  # Contador de recursos de vegetación agotados
        for idx, claimers in veg_claims.items():
            # Filtrar solo los agentes que siguen vivos
            vivos = [a for a in claimers if not terminations.get(a, False)]
            if not vivos:
                continue
            
            # El ganador es el agente con menor ratio comida/max_comida (más hambriento)
            winner = min(vivos, key=lambda a: self.species[self._agent_idx[a]].food /
                                        self.species[self._agent_idx[a]].max_food)
            
            # El ganador obtiene comida
            sp_w = self.species[self._agent_idx[winner]]
            sp_w.food = min(sp_w.food + 10, sp_w.max_food)  # +10 comida, sin exceder máximo
            
            # Bonificación por conseguir recurso (más que agua por ser más escaso)
            step_bonus[winner] += 8.0
            
            # Consumir el recurso y verificar si se agotó
            if self.consume_vegetation(idx):
                removed_v += 1

        # === APLICAR BONIFICACIONES POR CONSUMO DE RECURSOS ===
        for a in prev_agents:
            rewards[a] = float(rewards.get(a, 0.0) + step_bonus.get(a, 0.0))
        
        # === REWARD SHAPING ADICIONAL: Cambio de energía tras consumo ===
        # Aplicar reward shaping por cambio de energía después de consumir recursos
        for a in prev_agents:
            if (a in energy_after_move) and (not terminations[a]):
                # Comparar energía actual con energía después del movimiento
                post = self.species[self._agent_idx[a]].total_energy
                rewards[a] += 0.1 * (post - energy_after_move[a])
                
        # === REPONER RECURSOS CONSUMIDOS ===
        # Solo si el episodio no terminó por timeout global
        if not episode_trunc:
            # Calcular cuántos recursos generar para mantener densidades objetivo
            target_veg = self._init_veg  # Densidad objetivo de vegetación
            target_wat = self._init_wat  # Densidad objetivo de agua
            
            # Contar recursos actuales
            cur_veg = self.vegetation["x"].shape[0]
            cur_wat = self.water_sources["x"].shape[0]
            
            # Calcular cuántos generar (mitad de los removidos, sin exceder objetivo)
            spawn_veg = max(0, min(max(1, removed_v // 2), target_veg - cur_veg))
            spawn_wat = max(0, min(max(1, removed_w // 2), target_wat - cur_wat))
            
            # Generar nuevos recursos si es necesario
            if spawn_veg:
                self.spawn_vegetation(n=spawn_veg)
            if spawn_wat:
                self.spawn_water(n=spawn_wat)

        # === ELIMINAR RECURSOS COMPLETAMENTE AGOTADOS ===
        self._prune_depleted_resources()

        # === ACTUALIZAR OBSERVACIONES Y MÉTRICAS ===
        for agent in prev_agents:
            # Generar observación actualizada para cada agente
            obs[agent] = self._get_obs(self.species[self._agent_idx[agent]])
            
            # Asegurar que infos existe para este agente
            infos.setdefault(agent, {})
            
            # Actualizar métricas acumuladas del episodio
            self._ep_return[agent] = self._ep_return.get(agent, 0.0) + rewards.get(agent, 0.0)
            self._ep_len[agent]    = self._ep_len.get(agent, 0) + 1
            
            # Agregar métricas a infos para callbacks de RLlib
            infos[agent].setdefault("ep_return", self._ep_return[agent])
            infos[agent].setdefault("ep_len",    self._ep_len[agent])

        # === TERMINACIÓN GLOBAL POR TIMEOUT ===
        # Si el episodio alcanzó el máximo de pasos, truncar todos los agentes vivos
        if episode_trunc and len(alive) > 0:
            for a in alive:
                truncations[a] = True
                infos[a]["reason"] = "timeout"
            alive = []  # Ningún agente queda vivo tras el timeout

        # Actualizar lista de agentes activos para el próximo paso
        self.agents = alive
        
        return obs, rewards, terminations, truncations, infos

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

        # === NORMALIZAR NIVELES DE RECURSOS DEL AGENTE ===
        f_norm = sp.food / sp.max_food    # Comida [0-1]
        w_norm = sp.water / sp.max_water  # Agua [0-1]

        # Construir y retornar observación completa
        return np.array([f_norm, w_norm, dx_w, dy_w, dx_v, dy_v, dist_w, dist_v, w_avail, v_avail],
                        dtype=np.float32)
    
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