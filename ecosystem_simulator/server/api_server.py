"""
Servidor HTTP API para comunicar el estado del ecosistema a Godot.

Endpoints:
    GET /state - Obtiene el estado actual del ecosistema
    GET /health - Verifica que el servidor está activo
    
El cliente de Godot hace polling a http://localhost:5000/state
"""
from flask import Flask, jsonify
from flask_cors import CORS
import numpy as np
from typing import Dict, List, Any, Optional
import threading
import time

app = Flask(__name__)
CORS(app)  # Permitir peticiones desde Godot

# Estado global del ecosistema (thread-safe)
_ecosystem_state = {
    "ready": False,
    "episode": 0,
    "step": 0,
    "agents": [],
    "vegetation": [],
    "water_sources": [],
    "map_width": 800,
    "map_height": 600,
    "role": ""
}
_state_lock = threading.Lock()


@app.route('/health')
def health():
    """Verifica que el servidor está activo."""
    return jsonify({"status": "ok", "message": "Servidor activo"})


@app.route('/state')
def get_state():
    """Retorna el estado actual del ecosistema."""
    with _state_lock:
        return jsonify(_ecosystem_state)


def update_ecosystem_state(state_data: Dict[str, Any]):
    """
    Actualiza el estado global del ecosistema de forma thread-safe.
    
    Args:
        state_data: Diccionario con el estado del ecosistema
    """
    with _state_lock:
        _ecosystem_state.update(state_data)


def serialize_ecosystem_state(env, agents: List, episode: int, step: int) -> Dict:
    """
    Serializa el estado completo del ecosistema.
    
    Args:
        env: Instancia del ecosistema
        agents: Lista de agentes (especies)
        episode: Número de episodio actual
        step: Paso actual del episodio
        
    Returns:
        Diccionario con todos los datos del ecosistema
    """
    # Serializar agentes
    agents_data = []
    alive_count = 0
    for i, agent in enumerate(agents):
        is_alive = agent.food > 0 and agent.water > 0
        if is_alive:
            alive_count += 1
        agents_data.append({
            "id": i,
            "x": float(agent.x),
            "y": float(agent.y),
            "food": float(agent.food),
            "water": float(agent.water),
            "max_food": float(agent.max_food),
            "max_water": float(agent.max_water),
            "alive": is_alive,
            "role": agent.role.value
        })
        
    # Serializar vegetación
    vegetation_data = []
    for i in range(len(env.vegetation["x"])):
        if env.vegetation["charges"][i] > 0:  # Solo recursos activos
            vegetation_data.append({
                "x": float(env.vegetation["x"][i]),
                "y": float(env.vegetation["y"][i]),
                "w": float(env.vegetation["w"][i]),
                "h": float(env.vegetation["h"][i]),
                "charges": int(env.vegetation["charges"][i])
            })
            
    # Serializar fuentes de agua
    water_data = []
    for i in range(len(env.water_sources["x"])):
        if env.water_sources["charges"][i] > 0:  # Solo recursos activos
            water_data.append({
                "x": float(env.water_sources["x"][i]),
                "y": float(env.water_sources["y"][i]),
                "w": float(env.water_sources["w"][i]),
                "h": float(env.water_sources["h"][i]),
                "charges": int(env.water_sources["charges"][i])
            })
            
    return {
        "ready": True,
        "episode": episode,
        "step": step,
        "map_width": env.map_width,
        "map_height": env.map_height,
        "agents": agents_data,
        "vegetation": vegetation_data,
        "water_sources": water_data,
        "alive_count": alive_count,
        "total_agents": len(agents)
    }


def run_server(host: str = "0.0.0.0", port: int = 5000):
    """
    Inicia el servidor Flask en un thread separado.
    
    Args:
        host: Dirección del servidor
        port: Puerto del servidor
    """
    import logging

    # Silenciar logs de Flask
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR) # Solo mostrar errores

    app.run(host=host, port=port, debug=False, use_reloader=False)
