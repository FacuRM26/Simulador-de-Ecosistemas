"""
Módulo que define la clase Ecosystem, que gestiona los recursos (vegetación y agua)
y sus interacciones con las especies.

Este módulo implementa el sistema de recursos del ecosistema usando estructuras
de datos vectorizadas (NumPy) y árboles KD para búsquedas espaciales eficientes.
"""
import numpy as np
from scipy.spatial import KDTree
from typing import Tuple

class Ecosystem:
    """
    Clase que gestiona los recursos del ecosistema y sus interacciones.
    
    El ecosistema contiene dos tipos de recursos: vegetación (comida) y fuentes de agua.
    Utiliza KD-Trees para búsquedas espaciales eficientes de recursos cercanos.
    Cada recurso tiene una posición, tamaño y carga (número de veces que puede ser consumido).
    
    Atributos:
        map_width (int): Ancho del mapa del ecosistema
        map_height (int): Alto del mapa del ecosistema
        vegetation (dict): Diccionario con arrays de vegetación (x, y, w, h, centers, charges)
        water_sources (dict): Diccionario con arrays de fuentes de agua
        _veg_tree (KDTree): Árbol KD para búsquedas rápidas de vegetación
        _wat_tree (KDTree): Árbol KD para búsquedas rápidas de agua
    """
    def __init__(
        self,
        veg_density: int,
        water_density: int,
        map_width: int = 800,
        map_height: int = 600
    ):
        """
        Inicializa el ecosistema con recursos distribuidos aleatoriamente.
        
        Args:
            veg_density: Número de parches de vegetación a generar
            water_density: Número de fuentes de agua a generar
            map_width: Ancho del mapa (por defecto 800)
            map_height: Alto del mapa (por defecto 600)
        """
        self.map_width  = map_width
        self.map_height = map_height

        # Generar recursos de forma vectorizada (más eficiente que loops)
        self.vegetation = self._gen_resources_array(veg_density, size_min=25, size_max=45)
        self.water_sources = self._gen_resources_array(water_density, size_min=20, size_max=60)

        # Construir KD-Trees para búsquedas espaciales O(log n) en lugar de O(n)
        self._veg_tree = KDTree(self.vegetation["centers"])
        self._wat_tree = KDTree(self.water_sources["centers"])
        
        # Guardar densidades iniciales para regeneración de recursos
        self._init_veg = veg_density
        self._init_wat = water_density

    def _gen_resources_array(self, n: int, size_min: int, size_max: int) -> dict:
        """
        Genera un array de recursos con posiciones y tamaños aleatorios.
        
        Crea n recursos rectangulares distribuidos aleatoriamente en el mapa.
        Cada recurso tiene: posición (x,y), dimensiones (w,h), centro calculado,
        y carga inicial (número de veces que puede ser consumido).
        
        Args:
            n: Número de recursos a generar
            size_min: Tamaño mínimo del recurso (ancho/alto)
            size_max: Tamaño máximo del recurso (ancho/alto)
            
        Returns:
            dict: Diccionario con arrays numpy:
                - x, y: Posiciones (esquina superior izquierda)
                - w, h: Dimensiones (ancho y alto)
                - centers: Array (n, 2) con coordenadas del centro
                - charges: Array de cargas iniciales (por defecto 3)
        """
        # Generar anchos y altos aleatorios dentro del rango especificado
        w = np.random.randint(size_min, size_max + 1, size=n)
        h = np.random.randint(size_min, size_max + 1, size=n)

        # Generar posiciones aleatorias, evitando que los recursos se salgan del mapa
        x = np.random.randint(0, self.map_width  - size_max + 1, size=n)
        y = np.random.randint(0, self.map_height - size_max + 1, size=n)

        # Calcular los centros de cada recurso para búsquedas espaciales
        # Stack crea un array (n, 2) con [cx, cy] por cada recurso
        centers = np.stack([x + w / 2, y + h / 2], axis=1)
        
        # Inicializar todas las cargas en 3 (cada recurso puede ser consumido 3 veces)
        charges = np.full(n, 3, dtype=np.int32)
        
        return {"x": x, "y": y, "w": w, "h": h, "centers": centers, "charges": charges}

    def closest_vegetation(self, position: Tuple[float, float], k: int = 1):
        """
        Encuentra los k parches de vegetación más cercanos a una posición.
        
        Usa el KD-Tree para búsqueda eficiente.
        
        Args:
            position: Tupla (x, y) con la posición de consulta
            k: Número de vecinos más cercanos a retornar (por defecto 1)
            
        Returns:
            tuple: (índices, distancias) de los k recursos más cercanos
        """
        dist, idx = self._veg_tree.query(position, k=k)
        return idx, dist

    def closest_water(self, position: Tuple[float, float], k: int = 1):
        """
        Encuentra las k fuentes de agua más cercanas a una posición.

        Usa el KD-Tree para búsqueda eficiente.

        Args:
            position: Tupla (x, y) con la posición de consulta
            k: Número de vecinos más cercanos a retornar (por defecto 1)
            
        Returns:
            tuple: (índices, distancias) de los k recursos más cercanos
        """
        dist, idx = self._wat_tree.query(position, k=k)
        return idx, dist

    def consume_water(self, idx: int) -> bool:
        """
        Consume una unidad de carga de una fuente de agua.
        
        Reduce en 1 la carga del recurso de agua en el índice especificado.
        Cuando la carga llega a 0 o menos, el recurso se considera agotado.
        
        Args:
            idx: Índice del recurso de agua en el array
            
        Returns:
            bool: True si el recurso quedó agotado (carga <= 0), False en caso contrario
        """
        # Verificar que el índice sea válido (bounds check por seguridad)
        n = self.water_sources["charges"].shape[0]
        if idx < 0 or idx >= n:
            return False
        
        # Reducir la carga en 1
        self.water_sources["charges"][idx] -= 1
        
        # Retornar True si el recurso se agotó completamente
        return self.water_sources["charges"][idx] <= 0

    def consume_vegetation(self, idx: int) -> bool:
        """
        Consume una unidad de carga de un parche de vegetación.
        
        Reduce en 1 la carga del recurso de vegetación en el índice especificado.
        Cuando la carga llega a 0 o menos, el recurso se considera agotado.
        
        Args:
            idx: Índice del recurso de vegetación en el array
            
        Returns:
            bool: True si el recurso quedó agotado (carga <= 0), False en caso contrario
        """
        # Verificar que el índice sea válido
        n = self.vegetation["charges"].shape[0]
        if idx < 0 or idx >= n:
            return False
        
        # Reducir la carga en 1
        self.vegetation["charges"][idx] -= 1
        
        # Retornar True si el recurso se agotó completamente
        return self.vegetation["charges"][idx] <= 0

    def collide_resources(self, position: Tuple[float, float], size: float, resources: dict) -> np.ndarray:
        """
        Detecta colisiones entre una posición circular y recursos rectangulares.
        
        Utiliza detección de colisión AABB (Axis-Aligned Bounding Box) simplificada
        para verificar si un círculo (agente) intersecta con rectángulos (recursos).
        La implementación es vectorizada para eficiencia.
        
        Args:
            position: Tupla (x, y) con el centro del círculo (agente)
            size: Diámetro del círculo (tamaño del agente)
            resources: Diccionario con arrays de recursos a verificar
            
        Returns:
            np.ndarray: Array con los índices de recursos que colisionan
        """
        # Extraer coordenadas del centro del agente
        cx, cy = position
        half = size / 2  # Radio del círculo

        # Calcular distancias absolutas en X e Y entre el agente y todos los recursos
        # Usando broadcasting de numpy para operaciones vectorizadas
        dx = np.abs(resources["centers"][:, 0] - cx)  # Distancia horizontal
        dy = np.abs(resources["centers"][:, 1] - cy)  # Distancia vertical
        
        # Calcular semi-anchos y semi-altos de los recursos
        half_w = resources["w"] / 2
        half_h = resources["h"] / 2

        # Detección de colisión AABB: hay colisión si las distancias son menores
        # que la suma de los semi-tamaños en cada dimensión
        hits = (dx <= half_w + half) & (dy <= half_h + half)
        
        # Retornar los índices de los recursos que colisionan
        return np.nonzero(hits)[0]

    def spawn_vegetation(self, n: int = 1, size_min: int = 25, size_max: int = 45) -> None:
        """
        Genera nuevos parches de vegetación en el ecosistema.
        
        Añade n nuevos recursos de vegetación al array existente y reconstruye
        el KD-Tree para mantener las búsquedas espaciales eficientes.
        
        Args:
            n: Número de nuevos parches de vegetación a generar (por defecto 1)
            size_min: Tamaño mínimo de los nuevos parches (por defecto 25)
            size_max: Tamaño máximo de los nuevos parches (por defecto 45)
        """
        # Generar dimensiones aleatorias para los nuevos recursos
        w = np.random.randint(size_min, size_max + 1, size=n)
        h = np.random.randint(size_min, size_max + 1, size=n)
        
        # Generar posiciones aleatorias dentro del mapa
        x = np.random.randint(0, self.map_width  - size_max + 1, size=n)
        y = np.random.randint(0, self.map_height - size_max + 1, size=n)
        
        # Calcular los centros de los nuevos recursos
        centers = np.stack([x + w / 2, y + h / 2], axis=1)
        
        # Inicializar cargas en 3 para los nuevos recursos
        charges = np.full(n, 3, dtype=np.int32)

        # Concatenar los nuevos recursos con los existentes
        for k, arr in (("x", x), ("y", y), ("w", w), ("h", h)):
            self.vegetation[k] = np.concatenate([self.vegetation[k], arr])
        self.vegetation["centers"] = np.vstack([self.vegetation["centers"], centers])
        self.vegetation["charges"] = np.concatenate([self.vegetation["charges"], charges])
        
        # Reconstruir el KD-Tree con todos los recursos (viejos + nuevos)
        self._veg_tree = KDTree(self.vegetation["centers"])

    def spawn_water(self, n: int = 1, size_min: int = 20, size_max: int = 60):
        """
        Genera nuevas fuentes de agua en el ecosistema.
        
        Añade n nuevas fuentes de agua al array existente y reconstruye
        el KD-Tree para mantener las búsquedas espaciales eficientes.
        
        Args:
            n: Número de nuevas fuentes de agua a generar (por defecto 1)
            size_min: Tamaño mínimo de las nuevas fuentes (por defecto 20)
            size_max: Tamaño máximo de las nuevas fuentes (por defecto 60)
        """
        # Generar dimensiones aleatorias para los nuevos recursos
        w = np.random.randint(size_min, size_max + 1, size=n)
        h = np.random.randint(size_min, size_max + 1, size=n)
        
        # Generar posiciones aleatorias dentro del mapa
        x = np.random.randint(0, self.map_width  - size_max + 1, size=n)
        y = np.random.randint(0, self.map_height - size_max + 1, size=n)
        
        # Calcular los centros de los nuevos recursos
        centers = np.stack([x + w/2, y + h/2], axis=1)
        
        # Inicializar cargas en 3 para los nuevos recursos
        charges = np.full(n, 3, dtype=np.int32)
        
        # Concatenar los nuevos recursos con los existentes
        for k, arr in (("x", x), ("y", y), ("w", w), ("h", h)):
            self.water_sources[k] = np.concatenate([self.water_sources[k], arr])
        self.water_sources["centers"] = np.vstack([self.water_sources["centers"], centers])
        self.water_sources["charges"] = np.concatenate([self.water_sources["charges"], charges])
        
        # Reconstruir el KD-Tree con todos los recursos (viejos + nuevos)
        self._wat_tree = KDTree(self.water_sources["centers"])

    def _prune_depleted_resources(self):
        """
        Elimina recursos agotados y reconstruye los KD-Trees.
        
        Filtra los recursos que tienen carga > 0 (aún pueden ser consumidos)
        y elimina los que tienen carga <= 0 (agotados). Después reconstruye
        los KD-Trees para reflejar solo los recursos disponibles.
        
        Esta optimización mejora el rendimiento al reducir el número de recursos
        a considerar en las búsquedas espaciales.
        """
        # --- Procesar fuentes de agua ---
        if self.water_sources["charges"].size:
            # Crear máscara booleana: True para recursos con carga > 0
            mask_w = self.water_sources["charges"] > 0
            
            # Filtrar todos los arrays usando la máscara
            for k, arr in self.water_sources.items():
                # Los 'centers' son 2D (N, 2), el resto son 1D (N,)
                self.water_sources[k] = arr[mask_w] if arr.ndim == 1 else arr[mask_w, :]
            
            # Reconstruir KD-Tree con recursos restantes (o None si no quedan)
            self._wat_tree = (
                KDTree(self.water_sources["centers"])
                if self.water_sources["centers"].size
                else None
            )

        # --- Procesar vegetación ---
        if self.vegetation["charges"].size:
            # Crear máscara booleana: True para recursos con carga > 0
            mask_v = self.vegetation["charges"] > 0
            
            # Filtrar todos los arrays usando la máscara
            for k, arr in self.vegetation.items():
                # Los 'centers' son 2D (N, 2), el resto son 1D (N,)
                self.vegetation[k] = arr[mask_v] if arr.ndim == 1 else arr[mask_v, :]
            
            # Reconstruir KD-Tree con recursos restantes (o None si no quedan)
            self._veg_tree = (
                KDTree(self.vegetation["centers"])
                if self.vegetation["centers"].size
                else None
            )