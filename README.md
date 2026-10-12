# Simulador-de-Ecosistemas

### Para iniciar el proyecto

Tomando en cuenta que la versión de Python es Python 3.12.1:

1. cd ecosystem_simulator (Raíz)
2. py -3.12 -m venv .venv (Si aún no se tiene un venv)
3. .venv/Scripts/activate
4. pip install -r requirements.txt
5. python run.py

Enlace para instalar Python 3.12.1: https://www.python.org/downloads/release/python-3121/

### Lanzador unificado

`run.py` solo orquesta los entrypoints existentes. No cambia la lógica de entrenamiento.

Ejemplos:

```bash
python run.py
python run.py --godot
python run.py --reward-shaping --reflexion
python run.py --all
```

Si no pasa argumentos, ejecuta el entrenamiento base sin LLM.

