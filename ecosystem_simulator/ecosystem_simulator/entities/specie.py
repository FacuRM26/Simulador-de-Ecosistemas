"""
Módulo que define la clase Specie, que representa un agente individual en el ecosistema.
Cada especie tiene necesidades básicas (comida, agua) y capacidad de movimiento.
"""
import numpy as np

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
        """Calcula la energía total como el promedio de comida y agua."""
        return 0.5 * (self.food + self.water)

    def metabolize(self, rate: float = 0.3) -> None:
        """Reduce la comida y agua por metabolismo, asegurando que no sean negativas."""
        self.food  = max(0.0, self.food  - rate)
        self.water = max(0.0, self.water - rate)

    def move(self, velocity: float, direction: str) -> None:
        """Mueve al agente en la dirección dada, restringiendo la posición dentro del mapa."""
        dx, dy = self._DIR_VECTORS.get(direction, (0, 0))
        self.x += dx * velocity
        self.y += dy * velocity
        half = self.AGENT_SIZE / 2
        self.x = min(max(self.x, half), self.map_width  - half)
        self.y = min(max(self.y, half), self.map_height - half)

    def walk(self, velocity: float, direction: str) -> None:
        """Realiza el movimiento y el metabolismo en un solo paso."""
        # Gasto metabólico
        self.metabolize()
        # Movimiento y restricción de posición
        self.move(velocity, direction)