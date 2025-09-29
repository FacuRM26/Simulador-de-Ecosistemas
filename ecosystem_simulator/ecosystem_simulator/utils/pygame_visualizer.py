"""
Visualizador en tiempo real con Pygame para el ecosistema
"""
import pygame
import numpy as np
import sys
from typing import Dict, List, Any

class EcosystemVisualizer:
    def __init__(self, map_width: int = 1200, map_height: int = 800):
        self.map_width = map_width
        self.map_height = map_height
        
        # Inicializar Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((map_width, map_height + 200))  # Espacio extra para stats
        pygame.display.set_caption("Simulador de Ecosistemas Multi-Agente - Entrenamiento en Tiempo Real")
        
        # Colores
        self.colors = {
            'background': (240, 240, 240),
            'vegetation': (34, 139, 34),    # Verde bosque
            'water': (65, 105, 225),        # Azul real
            'agent': (178, 34, 34),         # Rojo fuego
            'agent_hungry': (255, 140, 0),  # Naranja
            'agent_thirsty': (70, 130, 180), # Azul acero
            'text': (0, 0, 0),
            'stats_bg': (255, 255, 255, 180)
        }
        
        # Fuentes
        self.font = pygame.font.SysFont('Arial', 16)
        self.title_font = pygame.font.SysFont('Arial', 24, bold=True)
        self.stats_font = pygame.font.SysFont('Arial', 14)
        
        # Control de FPS
        self.clock = pygame.time.Clock()
        self.fps = 30
        
    def render(self, ecosystem, species: list, step: int, episode: int, rewards: Dict[str, float] = None, metrics: Dict[str, Any] = None):
        """Renderiza el estado actual del ecosistema"""
        # Manejar eventos
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
        
        # Limpiar pantalla
        self.screen.fill(self.colors['background'])
        
        # Dibujar recursos
        self._draw_resources(ecosystem)
        
        # Dibujar agentes
        self._draw_agents(species, rewards)
        
        # Dibujar información y estadísticas
        self._draw_info_panel(step, episode, rewards, metrics, species)
        
        # Actualizar pantalla
        pygame.display.flip()
        self.clock.tick(self.fps)
        
        return True
    
    def _draw_resources(self, ecosystem):
        """Dibuja la vegetación y fuentes de agua"""
        # Dibujar vegetación
        for i in range(len(ecosystem.vegetation['x'])):
            if ecosystem.vegetation['charges'][i] > 0:
                x = ecosystem.vegetation['x'][i]
                y = ecosystem.vegetation['y'][i]
                w = ecosystem.vegetation['w'][i]
                h = ecosystem.vegetation['h'][i]
                
                # Intensidad del color según las cargas restantes
                charge_ratio = ecosystem.vegetation['charges'][i] / 3.0
                color = tuple(int(c * charge_ratio) for c in self.colors['vegetation'])
                
                pygame.draw.rect(self.screen, color, (x, y, w, h))
                pygame.draw.rect(self.screen, (0, 100, 0), (x, y, w, h), 1)  # Borde
        
        # Dibujar agua
        for i in range(len(ecosystem.water_sources['x'])):
            if ecosystem.water_sources['charges'][i] > 0:
                x = ecosystem.water_sources['x'][i]
                y = ecosystem.water_sources['y'][i]
                w = ecosystem.water_sources['w'][i]
                h = ecosystem.water_sources['h'][i]
                
                charge_ratio = ecosystem.water_sources['charges'][i] / 3.0
                color = tuple(int(c * charge_ratio) for c in self.colors['water'])
                
                pygame.draw.rect(self.screen, color, (x, y, w, h))
                pygame.draw.rect(self.screen, (0, 0, 139), (x, y, w, h), 1)  # Borde
    
    def _draw_agents(self, species, rewards: Dict[str, float] = None):
        """Dibuja los agentes en el mapa"""
        for i, specie in enumerate(species):
            # Determinar color según necesidades
            food_ratio = specie.food / specie.max_food
            water_ratio = specie.water / specie.max_water
            
            if food_ratio < 0.3 or water_ratio < 0.3:
                # Agente en estado crítico
                if food_ratio < water_ratio:
                    color = self.colors['agent_hungry']  # Hambriento (naranja)
                else:
                    color = self.colors['agent_thirsty']  # Sediento (azul)
            else:
                color = self.colors['agent']  # Normal (rojo)
            
            # Dibujar agente
            agent_size = 10
            pygame.draw.circle(self.screen, color, (int(specie.x), int(specie.y)), agent_size)
            pygame.draw.circle(self.screen, (0, 0, 0), (int(specie.x), int(specie.y)), agent_size, 2)  # Borde
            
            # Dibujar barra de estado
            self._draw_agent_status(specie, i, rewards)
    
    def _draw_agent_status(self, specie, agent_id: int, rewards: Dict[str, float] = None):
        """Dibuja la barra de estado del agente"""
        x, y = int(specie.x), int(specie.y)
        agent_size = 10
        
        # Barra de comida (verde)
        food_width = int((specie.food / specie.max_food) * 30)
        pygame.draw.rect(self.screen, (0, 255, 0), (x - 15, y - agent_size - 10, food_width, 4))
        
        # Barra de agua (azul)
        water_width = int((specie.water / specie.max_water) * 30)
        pygame.draw.rect(self.screen, (0, 0, 255), (x - 15, y - agent_size - 5, water_width, 4))
        
        # ID del agente
        agent_text = self.font.render(f"A{agent_id}", True, self.colors['text'])
        self.screen.blit(agent_text, (x - 8, y - agent_size - 25))
        
        # Recompensa si está disponible
        if rewards and f"agent_{agent_id}" in rewards:
            reward = rewards[f"agent_{agent_id}"]
            reward_color = (0, 100, 0) if reward >= 0 else (139, 0, 0)
            reward_text = self.font.render(f"{reward:.1f}", True, reward_color)
            self.screen.blit(reward_text, (x - 10, y + agent_size + 5))
    
    def _draw_info_panel(self, step: int, episode: int, rewards: Dict[str, float], metrics: Dict[str, Any], species: list):
        """Dibuja el panel de información en la parte inferior"""
        panel_y = self.map_height
        panel_height = 200
        
        # Fondo del panel
        pygame.draw.rect(self.screen, (220, 220, 220), (0, panel_y, self.map_width, panel_height))
        
        # Título
        title = self.title_font.render(f"Episodio: {episode} - Step: {step}", True, self.colors['text'])
        self.screen.blit(title, (20, panel_y + 10))
        
        # Estadísticas de agentes
        y_offset = panel_y + 40
        for i, specie in enumerate(species):
            food_pct = (specie.food / specie.max_food) * 100
            water_pct = (specie.water / specie.max_water) * 100
            energy = specie.total_energy
            
            agent_text = self.stats_font.render(
                f"Agente {i}: Comida: {food_pct:.1f}% | Agua: {water_pct:.1f}% | Energía: {energy:.1f}", 
                True, self.colors['text']
            )
            self.screen.blit(agent_text, (20, y_offset))
            y_offset += 20
        
        # Métricas de entrenamiento
        if metrics:
            y_offset += 10
            metrics_text = self.stats_font.render(
                f"Recompensa media: {metrics.get('mean_reward', 0):.2f} | "
                f"Recompensa total: {metrics.get('total_reward', 0):.2f} | "
                f"Agentes vivos: {metrics.get('alive_agents', 0)}/{metrics.get('total_agents', 0)}",
                True, self.colors['text']
            )
            self.screen.blit(metrics_text, (20, y_offset))
        
        # Leyenda
        legend_x = self.map_width - 250
        legend_y = panel_y + 40
        
        legend_items = [
            ("🌿 Vegetación", self.colors['vegetation']),
            ("💧 Agua", self.colors['water']),
            ("🔴 Agente normal", self.colors['agent']),
            ("🟠 Agente hambriento", self.colors['agent_hungry']),
            ("🔵 Agente sediento", self.colors['agent_thirsty'])
        ]
        
        for i, (text, color) in enumerate(legend_items):
            # Dibujar cuadro de color
            pygame.draw.rect(self.screen, color, (legend_x, legend_y + i * 25, 15, 15))
            legend_text = self.stats_font.render(text, True, self.colors['text'])
            self.screen.blit(legend_text, (legend_x + 20, legend_y + i * 25))
    
    def close(self):
        """Cierra la visualización"""
        pygame.quit()