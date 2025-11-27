"""
Módulo que define la clase Specie, que representa un agente individual en el ecosistema.
Cada especie tiene necesidades básicas (comida, agua) y capacidad de movimiento.
"""
import numpy as np
from enum import Enum
class Role(Enum): HERBIVORE=0; PREDATOR=1



class Specie:
    __slots__ = (
        "food", "water",
        "x", "y",
        "max_food", "max_water",
        "map_width", "map_height", "role", "alive", "hp", "speed","attack_range","attack_cost"
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

    def __init__(self, food, water, x=0, y=0, max_food=100, max_water=100,
                 map_width=800, map_height=600, role=Role.HERBIVORE,
                 hp=100, speed=1.0, attack_range=25.0, attack_cost=2.0):
        # Parámetros estáticos del entorno
        self.max_food   = max_food
        self.max_water  = max_water
        self.map_width  = map_width
        self.map_height = map_height
        self.role = role
        # Estado inicial (se asegura que food & water no superen sus máximos)
        self.food  = min(food,  max_food)
        self.water = min(water, max_water)
        self.alive = True
        self.hp = hp
        self.speed = speed
        self.attack_range = attack_range
        self.attack_cost = attack_cost
        # Posición inicial
        self.x = x
        self.y = y

    def take_damage(self, dmg: float):
        self.hp = max(0.0, self.hp - dmg)
        if self.hp <= 0: self.alive = False

    @property
    def total_energy(self) -> float:
        """Calcula la energía total como el promedio de comida y agua."""
        return 0.5 * (self.food + self.water)
    
    @property
    def pos(self): 
        return (self.x, self.y)

    def is_alive(self): 
        return bool(self.alive)

    def metabolize(self, rate: float = 0.3) -> None:
        """Reduce la comida y agua por metabolismo, asegurando que no sean negativas."""
        self.food  = max(0.0, self.food  - rate)
        self.water = max(0.0, self.water - rate)

    def move(self, distance: float = 1.0, direction: str = "stay") -> None:
        # distancia efectiva = distancia base * speed del agente
        step = float(distance) * float(self.speed)
        dx, dy = self._DIR_VECTORS.get(direction, (0, 0))
        self.x += dx * step
        self.y += dy * step
        half = self.AGENT_SIZE / 2.0
        self.x = min(max(self.x, half), self.map_width  - half)
        self.y = min(max(self.y, half), self.map_height - half)


    def walk(self, velocity: float, direction: str) -> None:
        self.metabolize()
        self.move(velocity, direction)