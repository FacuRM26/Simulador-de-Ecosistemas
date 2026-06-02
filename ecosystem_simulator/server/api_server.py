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
CORS(app)

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
    
    La serialización es consistente con lo que Godot espera:
    - role: 1 = PREDATOR, 0 = HERBIVORE (entero, no string)
    - alive: True solo si food > 0, water > 0 Y sp.alive == True (cubre muerte por depredación)
    - x, y: esquina superior izquierda del recurso (Godot calcula el centro)
    - Los recursos incluyen un 'id' estable basado en su posición para que Godot
      pueda rastrearlos correctamente entre pasos.

    Args:
        env: Instancia del ecosistema (MultiAgentEcosystem)
        agents: Lista de objetos Specie
        episode: Número de episodio actual
        step: Paso actual del episodio
        
    Returns:
        Diccionario con todos los datos del ecosistema listos para Godot
    """
    # ── Agentes ────────────────────────────────────────────────────────────
    agents_data = []
    alive_count = 0

    for i, agent in enumerate(agents):
        # Considera muerto si: recursos agotados O marcado como no-vivo (depredación)
        is_alive = bool(agent.food > 0 and agent.water > 0 and agent.alive)
        if is_alive:
            alive_count += 1

        # role como entero: 1 = PREDATOR, 0 = HERBIVORE
        # Godot hace: is_predator = (data.role == 1)
        try:
            from ecosystem_simulator.entities.specie import Role
            role_int = 1 if agent.role is Role.PREDATOR else 0
        except Exception:
            role_int = 0

        agents_data.append({
            "id": i,
            "x": float(agent.x),
            "y": float(agent.y),
            "food": float(agent.food),
            "water": float(agent.water),
            "max_food": float(agent.max_food),
            "max_water": float(agent.max_water),
            "alive": is_alive,
            "role": role_int,          # ← entero, no string
            "total_energy": float(getattr(agent, "total_energy", 0.0)),
        })

    # ── Vegetación ─────────────────────────────────────────────────────────
    # Usamos un ID estable basado en posición para que Godot pueda rastrear
    # el mismo recurso entre frames aunque el array se reordene.
    vegetation_data = []
    for i in range(len(env.vegetation["x"])):
        if env.vegetation["charges"][i] > 0:
            x = float(env.vegetation["x"][i])
            y = float(env.vegetation["y"][i])
            w = float(env.vegetation["w"][i])
            h = float(env.vegetation["h"][i])
            vegetation_data.append({
                # ID estable: hash de posición + tamaño (no cambia si el recurso no se mueve)
                "id": f"{int(x)}_{int(y)}_{int(w)}_{int(h)}",
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "charges": int(env.vegetation["charges"][i]),
            })

    # ── Agua ───────────────────────────────────────────────────────────────
    water_data = []
    for i in range(len(env.water_sources["x"])):
        if env.water_sources["charges"][i] > 0:
            x = float(env.water_sources["x"][i])
            y = float(env.water_sources["y"][i])
            w = float(env.water_sources["w"][i])
            h = float(env.water_sources["h"][i])
            water_data.append({
                "id": f"{int(x)}_{int(y)}_{int(w)}_{int(h)}",
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "charges": int(env.water_sources["charges"][i]),
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
        "total_agents": len(agents),
    }


def run_server(host: str = "0.0.0.0", port: int = 5000):
    """
    Inicia el servidor Flask.
    
    Args:
        host: Dirección del servidor
        port: Puerto del servidor
    """
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    app.run(host=host, port=port, debug=False, use_reloader=False)