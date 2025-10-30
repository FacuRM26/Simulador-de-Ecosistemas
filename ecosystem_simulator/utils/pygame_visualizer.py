"""
Visualizador en tiempo real con Pygame para el ecosistema.

Este módulo proporciona una visualización gráfica interactiva del ecosistema
durante el entrenamiento o simulación. Muestra:
- Recursos (vegetación y agua) con intensidad basada en cargas
- Agentes con colores según su estado de salud
- Barras de estado individual (comida y agua)
- Panel de información con estadísticas en tiempo real
"""
import pygame
import numpy as np
import sys
from typing import Dict, List, Any

class EcosystemVisualizer:
    """
    Visualizador interactivo del ecosistema usando Pygame.
    
    Renderiza el estado del ecosistema en una ventana gráfica con:
    - Vista del mapa con recursos y agentes
    - Panel de estadísticas en la parte inferior
    - Actualización en tiempo real durante el entrenamiento
    
    Atributos:
        map_width, map_height: Dimensiones del área de simulación
        screen: Superficie de Pygame para renderizado
        colors: Diccionario de colores para diferentes elementos
        font: Fuentes para texto
        clock: Control de FPS
    """
    
    def __init__(self, map_width: int = 1200, map_height: int = 800):
        """
        Inicializa el visualizador con Pygame.
        
        Args:
            map_width: Ancho del mapa en píxeles (por defecto 1200)
            map_height: Alto del mapa en píxeles (por defecto 800)
        """
        self.map_width = map_width
        self.map_height = map_height
        
        # Inicializar subsistema de Pygame
        pygame.init()
        
        # Crear ventana con espacio extra para panel de estadísticas (170px)
        self.screen = pygame.display.set_mode((map_width, map_height + 170))
        pygame.display.set_caption("Simulador de Ecosistemas Multi-Agente - Entrenamiento en Tiempo Real")
        
        # === PALETA DE COLORES ===
        self.colors = {
            'background': (194, 158, 153),    # Café claro para el suelo
            'vegetation': (34, 139, 34),      # Verde bosque para vegetación
            'water': (65, 105, 225),          # Azul real para agua
            'agent': (178, 34, 34),           # Rojo fuego para agente normal
            'agent_hungry': (255, 140, 0),    # Naranja para agente hambriento
            'agent_thirsty': (70, 130, 180),  # Azul acero para agente sediento
            'text': (0, 0, 0),                # Negro para texto
            'stats_bg': (255, 255, 255, 180)  # Blanco semi-transparente
        }
        
        # === CONFIGURACIÓN DE FUENTES ===
        self.font = pygame.font.SysFont('Arial', 16)           # Fuente regular
        self.title_font = pygame.font.SysFont('Arial', 30, bold=True)  # Título
        self.stats_font = pygame.font.SysFont('Souvenir', 22)  # Estadísticas
        
        # === CONTROL DE FRAMES POR SEGUNDO ===
        self.clock = pygame.time.Clock()
        self.fps = 20  # 20 FPS para visualización fluida
        
    def render(self, ecosystem, species: list, step: int, episode: int, rewards: Dict[str, float] = None, metrics: Dict[str, Any] = None):
        """
        Renderiza el estado actual del ecosistema en la ventana.
        
        Este método actualiza toda la ventana con el estado actual:
        1. Procesa eventos de cierre de ventana
        2. Limpia la pantalla
        3. Dibuja recursos (vegetación y agua)
        4. Dibuja agentes con sus estados
        5. Dibuja panel de información
        6. Actualiza la pantalla
        
        Args:
            ecosystem: Objeto Ecosystem con recursos
            species: Lista de objetos Specie (agentes)
            step: Número de paso actual en el episodio
            episode: Número de episodio actual
            rewards: Diccionario con recompensas por agente (opcional)
            metrics: Diccionario con métricas adicionales (opcional)
            
        Returns:
            bool: True si debe continuar, False si el usuario cerró la ventana
        """
        # === PROCESAR EVENTOS DE PYGAME ===
        for event in pygame.event.get():
            if event.type == pygame.QUIT:  # Usuario cierra la ventana
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:  # Presiona ESC
                    return False
        
        # === LIMPIAR PANTALLA ===
        self.screen.fill(self.colors['background'])
        
        # === RENDERIZAR ELEMENTOS ===
        self._draw_resources(ecosystem)  # Dibujar vegetación y agua
        self._draw_agents(species, rewards)  # Dibujar agentes
        self._draw_info_panel(step, episode, rewards, metrics, species)  # Panel de info
        
        # === ACTUALIZAR PANTALLA Y CONTROLAR FPS ===
        pygame.display.flip()  # Actualizar toda la pantalla
        self.clock.tick(self.fps)  # Limitar a FPS configurado
        
        return True
    
    def _draw_resources(self, ecosystem):
        """
        Dibuja todos los recursos del ecosistema (vegetación y agua).
        
        Los recursos se dibujan como rectángulos con:
        - Color que varía según la carga restante (más oscuro = más agotado)
        - Borde oscuro para delimitación
        
        Args:
            ecosystem: Objeto Ecosystem con recursos a dibujar
        """
        # === DIBUJAR VEGETACIÓN ===
        for i in range(len(ecosystem.vegetation['x'])):
            # Solo dibujar si el recurso aún tiene carga
            if ecosystem.vegetation['charges'][i] > 0:
                # Extraer propiedades del recurso
                x = ecosystem.vegetation['x'][i]
                y = ecosystem.vegetation['y'][i]
                w = ecosystem.vegetation['w'][i]
                h = ecosystem.vegetation['h'][i]
                
                # Calcular intensidad del color según cargas restantes (carga inicial = 3)
                charge_ratio = ecosystem.vegetation['charges'][i] / 3.0
                color = tuple(int(c * charge_ratio) for c in self.colors['vegetation'])
                
                # Dibujar rectángulo relleno
                pygame.draw.rect(self.screen, color, (x, y, w, h))
                # Dibujar borde oscuro
                pygame.draw.rect(self.screen, (0, 100, 0), (x, y, w, h), 1)
        
        # === DIBUJAR AGUA ===
        for i in range(len(ecosystem.water_sources['x'])):
            # Solo dibujar si el recurso aún tiene carga
            if ecosystem.water_sources['charges'][i] > 0:
                # Extraer propiedades del recurso
                x = ecosystem.water_sources['x'][i]
                y = ecosystem.water_sources['y'][i]
                w = ecosystem.water_sources['w'][i]
                h = ecosystem.water_sources['h'][i]
                
                # Calcular intensidad del color según cargas restantes
                charge_ratio = ecosystem.water_sources['charges'][i] / 3.0
                color = tuple(int(c * charge_ratio) for c in self.colors['water'])
                
                # Dibujar rectángulo relleno
                pygame.draw.rect(self.screen, color, (x, y, w, h))
                # Dibujar borde oscuro
                pygame.draw.rect(self.screen, (0, 0, 139), (x, y, w, h), 1)
    
    def _draw_agents(self, species, rewards: Dict[str, float] = None):
        """
        Dibuja todos los agentes en el mapa con indicadores de estado.
        
        Cada agente se representa como un círculo con:
        - Color según su estado de salud (normal, hambriento, sediento)
        - Barra de comida (verde)
        - Barra de agua (azul)
        - ID del agente
        - Recompensa actual (si está disponible)
        
        Args:
            species: Lista de objetos Specie a dibujar
            rewards: Diccionario con recompensas por agente (opcional)
        """
        for i, specie in enumerate(species):
            # === DETERMINAR COLOR SEGÚN ESTADO DE SALUD ===
            food_ratio = specie.food / specie.max_food    # Ratio de comida [0-1]
            water_ratio = specie.water / specie.max_water # Ratio de agua [0-1]
            
            # Verificar si el agente está en estado crítico (< 30%)
            if food_ratio < 0.3 or water_ratio < 0.3:
                # En estado crítico: determinar cuál recurso es más bajo
                if food_ratio < water_ratio:
                    color = self.colors['agent_hungry']  # Hambriento (naranja)
                else:
                    color = self.colors['agent_thirsty']  # Sediento (azul)
            else:
                color = self.colors['agent']  # Estado normal (rojo)
            
            # === DIBUJAR CÍRCULO DEL AGENTE ===
            agent_size = 10  # Radio del círculo
            # Dibujar círculo relleno
            pygame.draw.circle(self.screen, color, (int(specie.x), int(specie.y)), agent_size)
            # Dibujar borde negro
            pygame.draw.circle(self.screen, (0, 0, 0), (int(specie.x), int(specie.y)), agent_size, 2)
            
            # === DIBUJAR BARRAS DE ESTADO ===
            self._draw_agent_status(specie, i, rewards)
    
    def _draw_agent_status(self, specie, agent_id: int, rewards: Dict[str, float] = None):
        """
        Dibuja las barras de estado y etiquetas para un agente.
        
        Muestra:
        - Barra de comida (verde) proporcional al nivel
        - Barra de agua (azul) proporcional al nivel
        - ID del agente (ej. "A0")
        - Recompensa actual si está disponible
        
        Args:
            specie: Objeto Specie del agente
            agent_id: ID numérico del agente
            rewards: Diccionario con recompensas (opcional)
        """
        x, y = int(specie.x), int(specie.y)
        agent_size = 10  # Radio del círculo del agente
        
        # === BARRA DE COMIDA (Verde, arriba) ===
        # Calcular ancho proporcional al nivel de comida (máximo 30 píxeles)
        food_width = int((specie.food / specie.max_food) * 30)
        pygame.draw.rect(self.screen, (0, 255, 0), (x - 15, y - agent_size - 10, food_width, 4))
        
        # === BARRA DE AGUA (Azul, abajo de la barra de comida) ===
        # Calcular ancho proporcional al nivel de agua (máximo 30 píxeles)
        water_width = int((specie.water / specie.max_water) * 30)
        pygame.draw.rect(self.screen, (0, 0, 255), (x - 15, y - agent_size - 5, water_width, 4))
        
        # === ETIQUETA DE ID DEL AGENTE ===
        agent_text = self.font.render(f"A{agent_id}", True, self.colors['text'])
        self.screen.blit(agent_text, (x - 8, y - agent_size - 25))
        
        # === MOSTRAR RECOMPENSA SI ESTÁ DISPONIBLE ===
        if rewards and f"agent_{agent_id}" in rewards:
            reward = rewards[f"agent_{agent_id}"]
            # Color verde para recompensa positiva, rojo para negativa
            reward_color = (0, 100, 0) if reward >= 0 else (139, 0, 0)
            reward_text = self.font.render(f"{reward:.1f}", True, reward_color)
            self.screen.blit(reward_text, (x - 10, y + agent_size + 5))
    
    def _draw_info_panel(self, step: int, episode: int, rewards: Dict[str, float], metrics: Dict[str, Any], species: list):
        """
        Dibuja el panel de información en la parte inferior de la ventana.
        
        El panel muestra:
        - Título con número de episodio y paso
        - Estadísticas detalladas por agente (comida, agua, energía)
        - Métricas de entrenamiento (recompensas, agentes vivos)
        - Leyenda de colores
        
        Args:
            step: Número de paso actual
            episode: Número de episodio actual
            rewards: Diccionario con recompensas por agente
            metrics: Diccionario con métricas adicionales
            species: Lista de objetos Specie
        """
        # Posición y tamaño del panel
        panel_y = self.map_height
        panel_height = 200
        
        # === FONDO DEL PANEL ===
        pygame.draw.rect(self.screen, (220, 220, 220), (0, panel_y, self.map_width, panel_height))
        
        # === TÍTULO CON EPISODIO Y PASO ===
        title = self.title_font.render(f"Episodio: {episode} - Step: {step}", True, self.colors['text'])
        self.screen.blit(title, (20, panel_y + 10))
        
        # === ESTADÍSTICAS POR AGENTE ===
        y_offset = panel_y + 60  # Posición Y inicial para stats
        for i, specie in enumerate(species):
            # Calcular porcentajes de recursos
            food_pct = (specie.food / specie.max_food) * 100
            water_pct = (specie.water / specie.max_water) * 100
            energy = specie.total_energy
            
            # Renderizar texto con estadísticas del agente
            agent_text = self.stats_font.render(
                f"Agente {i}: Comida: {food_pct:.1f}% | Agua: {water_pct:.1f}% | Energía: {energy:.1f}", 
                True, self.colors['text']
            )
            
            # Organizar en dos columnas (agentes pares a la izquierda, impares a la derecha)
            if(i % 2 == 0):
                self.screen.blit(agent_text, (20, y_offset))
            else:
                self.screen.blit(agent_text, (450, y_offset))
                y_offset += 20  # Avanzar fila después del agente impar
        
        # === MÉTRICAS DE ENTRENAMIENTO ===
        if metrics:
            y_offset -= 80  # Ajustar posición hacia arriba
            # Renderizar métricas principales
            metric_text1 = self.stats_font.render(f"Recompensa media: {metrics.get('mean_reward', 0):.2f}", True, self.colors['text'])
            metric_text2 = self.stats_font.render(f"Recompensa total: {metrics.get('total_reward', 0):.2f}", True, self.colors['text'])
            metric_text3 = self.stats_font.render(f"Agentes vivos: {metrics.get('alive_agents', 0)}/{metrics.get('total_agents', 0)}", True, self.colors['text'])
            # Mostrar en la esquina derecha
            self.screen.blit(metric_text1, (920, y_offset))
            self.screen.blit(metric_text2, (920, y_offset + 20))
            self.screen.blit(metric_text3, (920, y_offset + 40))
        
        # === LEYENDA DE COLORES ===
        legend_x = 450  # Posición X inicial
        legend_y = panel_y + 15
        
        # Lista de elementos de la leyenda: (texto, color, offset_x_siguiente)
        legend_items = [
            ("Vegetación", self.colors['vegetation'], 120),
            ("Agua", self.colors['water'], 80),
            ("Agente normal", self.colors['agent'], 150),
            ("Agente hambriento", self.colors['agent_hungry'], 180),
            ("Agente sediento", self.colors['agent_thirsty'], 0)
        ]
        
        # Dibujar cada elemento de la leyenda
        for i, (text, color, offset_x) in enumerate(legend_items):
            # Dibujar cuadro de color (15x15 píxeles)
            pygame.draw.rect(self.screen, color, (legend_x, legend_y, 15, 15))
            # Dibujar texto explicativo
            legend_text = self.stats_font.render(text, True, self.colors['text'])
            self.screen.blit(legend_text, (legend_x + 20, legend_y))
            # Avanzar posición para siguiente elemento
            legend_x += offset_x
    
    def close(self):
        """
        Cierra la ventana de visualización y limpia recursos de Pygame.
        
        Debe llamarse al finalizar para liberar recursos correctamente.
        """
        pygame.quit()