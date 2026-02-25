"""
Módulo que define la clase Specie, que representa un agente individual en el ecosistema.
Cada especie tiene necesidades básicas (comida, agua) y capacidad de movimiento.
"""
import numpy as np
from enum import Enum
class Role(Enum): HERBIVORE=0; PREDATOR=1



class Specie:
    """
    Clase que representa un agente individual (especie) en el ecosistema.
    
    Esta clase maneja los recursos vitales del agente (comida y agua), su posición
    en el mapa, y su capacidad de moverse en diferentes direcciones. Utiliza __slots__
    para optimización de memoria ya que pueden existir múltiples instancias.
    
    Atributos:
        food (float): Nivel actual de comida del agente
        water (float): Nivel actual de agua del agente
        x (float): Posición X del agente en el mapa
        y (float): Posición Y del agente en el mapa
        max_food (float): Capacidad máxima de comida
        max_water (float): Capacidad máxima de agua
        map_width (int): Ancho del mapa del ecosistema
        map_height (int): Alto del mapa del ecosistema
    """
    __slots__ = (
        "food", "water",
        "x", "y",
        "max_food", "max_water",
        "map_width", "map_height",
        "role", "alive",
        "hp", "speed",
        "attack_range", "attack_cost", "attack_cd"
    )

    # Tamaño del agente en píxeles (usado para colisiones y límites del mapa)
    AGENT_SIZE = 20

    # Diccionario de vectores de dirección para el movimiento
    # Cada dirección mapea a un vector (dx, dy) que indica el cambio en x e y
    _DIR_VECTORS = {
        "north": (0, -1),   # Arriba: sin cambio en x, -1 en y
        "south": (0,  1),   # Abajo: sin cambio en x, +1 en y
        "east":  (1,  0),   # Derecha: +1 en x, sin cambio en y
        "west":  (-1, 0),   # Izquierda: -1 en x, sin cambio en y
        "stay":  (0,  0),   # Quedarse: sin cambio en posición
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
        self.food = max(0.0, min(food, max_food))
        self.water = max(0.0, min(water, max_water))
        self.alive = True
        self.hp = hp
        self.speed = speed
        self.attack_range = attack_range
        self.attack_cost = attack_cost
        # Posición inicial
        self.x = x
        self.y = y
        self.attack_cd = 0

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

    def metabolize(self, rate: float = 0.01) -> None:
        """
        Reduce los recursos del agente debido al metabolismo natural.
        
        Args:
            rate: Tasa de consumo metabólico (cantidad reducida por paso)
            Por defecto 0.3 unidades por recurso
        """
        # Reducir comida, asegurando que no sea negativa
        self.food  = max(0.0, self.food - rate)
        # Reducir agua, asegurando que no sea negativa
        self.water = max(0.0, self.water - rate)

    def move(self, distance: float = 1.0, direction: str = "stay") -> None:
        # distancia efectiva = distancia base * speed del agente
        step = float(distance) * float(self.speed)
        dx, dy = self._DIR_VECTORS.get(direction, (0, 0))
        self.x += dx * step
        self.y += dy * step
        half = self.AGENT_SIZE / 2.0
        self.x = min(max(self.x, half), self.map_width  - half)
        # Restringir Y dentro de los límites del mapa
        self.y = min(max(self.y, half), self.map_height - half)