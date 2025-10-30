"""
Módulo que define la clase Specie, que representa un agente individual en el ecosistema.
Cada especie tiene necesidades básicas (comida, agua) y capacidad de movimiento.
"""
import numpy as np

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
        "food", "water",  # Recursos vitales del agente
        "x", "y",  # Posición en el mapa
        "max_food", "max_water",  # Capacidades máximas de recursos
        "map_width", "map_height"  # Dimensiones del mapa para restricciones
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
        Inicializa una nueva especie (agente) con recursos y posición.
        
        Args:
            food: Nivel inicial de comida (se limita a max_food)
            water: Nivel inicial de agua (se limita a max_water)
            x: Posición inicial en el eje X (por defecto 0)
            y: Posición inicial en el eje Y (por defecto 0)
            max_food: Capacidad máxima de comida (por defecto 100)
            max_water: Capacidad máxima de agua (por defecto 100)
            map_width: Ancho del mapa (por defecto 800)
            map_height: Alto del mapa (por defecto 600)
        """
        # Guardar los parámetros estáticos del entorno
        self.max_food   = max_food
        self.max_water  = max_water
        self.map_width  = map_width
        self.map_height = map_height

        # Establecer estado inicial, asegurando que no excedan los máximos
        self.food  = min(food,  max_food)
        self.water = min(water, max_water)

        # Establecer posición inicial
        self.x = x
        self.y = y

    @property
    def total_energy(self) -> float:
        """Calcula la energía total como el promedio de comida y agua."""
        return 0.5 * (self.food + self.water)

    def metabolize(self, rate: float = 0.3) -> None:
        """
        Reduce los recursos del agente debido al metabolismo natural.
        
        Args:
            rate: Tasa de consumo metabólico (cantidad reducida por paso)
            Por defecto 0.3 unidades por recurso
        """
        # Reducir comida, asegurando que no sea negativa
        self.food  = max(0.0, self.food  - rate)
        # Reducir agua, asegurando que no sea negativa
        self.water = max(0.0, self.water - rate)

    def move(self, velocity: float, direction: str) -> None:
        """
        Mueve al agente en una dirección específica con velocidad dada.
        
        Args:
            velocity: Velocidad de movimiento (multiplicador del vector dirección)
            direction: Dirección de movimiento ("north", "south", "east", "west", "stay")
        """
        # Obtener el vector de dirección (dx, dy) o (0, 0) si no existe
        dx, dy = self._DIR_VECTORS.get(direction, (0, 0))
        
        # Actualizar posición aplicando velocidad y dirección
        self.x += dx * velocity
        self.y += dy * velocity
        
        # Calcular mitad del tamaño del agente para los límites
        half = self.AGENT_SIZE / 2
        
        # Restringir X dentro de los límites del mapa
        self.x = min(max(self.x, half), self.map_width  - half)
        # Restringir Y dentro de los límites del mapa
        self.y = min(max(self.y, half), self.map_height - half)

    def walk(self, velocity: float, direction: str) -> None:
        """
        Ejecuta un ciclo completo de movimiento: metabolismo + desplazamiento.
        
        Args:
            velocity: Velocidad de movimiento
            direction: Dirección de movimiento
        """
        # Primero: consumir energía por metabolismo
        self.metabolize()
        # Segundo: realizar el movimiento y restringir posición
        self.move(velocity, direction)