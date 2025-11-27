"""
Módulo que define la clase Ecosystem, que gestiona los recursos (vegetación y agua)
y sus interacciones con las especies.
"""
import numpy as np
from scipy.spatial import KDTree
from typing import Tuple
from math import hypot
from ..entities.specie import Role 


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
        """Genera un array de recursos (vegetación o agua) con posiciones y cargas."""
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

    def closest_vegetation(self, position, k=1):
        if self._veg_tree is None: return (None, np.inf)
        dist, idx = self._veg_tree.query(position, k=k)
        return idx, dist

    def closest_water(self, position, k=1):
        if self._wat_tree is None: return (None, np.inf)
        dist, idx = self._wat_tree.query(position, k=k)
        return idx, dist

    def consume_water(self, idx: int) -> bool:
        """Consume una unidad de agua en el índice dado. Retorna True si se agotó."""
        n = self.water_sources["charges"].shape[0]
        if idx < 0 or idx >= n:
            return False
        self.water_sources["charges"][idx] -= 1
        return self.water_sources["charges"][idx] <= 0  # True si quedó sin carga

    def consume_vegetation(self, idx: int) -> bool:
        """Consume una unidad de vegetación en el índice dado. Retorna True si se agotó."""
        n = self.vegetation["charges"].shape[0]
        if idx < 0 or idx >= n:
            return False
        self.vegetation["charges"][idx] -= 1
        return self.vegetation["charges"][idx] <= 0

    def collide_resources(self, position: Tuple[float, float], size: float, resources: dict) -> np.ndarray:
        """Detecta colisiones entre la posición (con tamaño) y los recursos."""
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
        """Genera nueva vegetación y actualiza el KD-Tree."""
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
        """Genera nuevas fuentes de agua y actualiza el KD-Tree."""
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

    def _prune_depleted_resources(self):
        """Elimina recursos agotados y reconstruye los KD-Trees."""
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

    def try_attack(self, predator, agents, dmg: float = 50.0) -> bool:
        # Filtra presas válidas (herbívoros vivos)
        preys = [a for a in agents if getattr(a, "alive", True) and getattr(a, "role", None) is Role.HERBIVORE]
        if not preys:
            return False

        # Construye KDTree o usa búsqueda lineal si hay muy pocos
        if len(preys) >= 2:
            coords = np.array([(p.x, p.y) for p in preys], dtype=np.float32)
            tree = KDTree(coords)
            dist, idx = tree.query((predator.x, predator.y))
            prey = preys[int(idx)]
        else:
            prey = preys[0]
            dist = hypot(prey.x - predator.x, prey.y - predator.y)

        if dist <= predator.attack_range and prey.alive:
            prey.take_damage(dmg)
            predator.food  = min(predator.max_food,  predator.food  + 40.0)
            predator.water = min(predator.max_water, predator.water + 5.0)
            predator.food  = max(0.0, predator.food - predator.attack_cost)
            return True
        return False
    
        # === Nuevos: helpers de consumo con clamp ===
    def try_eat_vegetation(self, agent, bite_gain: float = 20.0) -> bool:
        hits = self.collide_resources(agent.pos, agent.AGENT_SIZE, self.vegetation)
        if hits.size == 0:
            return False
        idx = int(hits[0])
        depleted = self.consume_vegetation(idx)
        agent.food = max(0.0, min(agent.max_food, agent.food + bite_gain))
        if depleted:
            self._prune_depleted_resources()
        return True

    def try_drink(self, agent, sip_gain: float = 15.0) -> bool:
        hits = self.collide_resources(agent.pos, agent.AGENT_SIZE, self.water_sources)
        if hits.size == 0:
            return False
        idx = int(hits[0])
        depleted = self.consume_water(idx)
        agent.water = max(0.0, min(agent.max_water, agent.water + sip_gain))
        if depleted:
            self._prune_depleted_resources()
        return True