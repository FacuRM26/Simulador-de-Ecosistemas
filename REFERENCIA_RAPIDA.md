# 📚 Guía de Referencia Rápida

## Estructura del Proyecto

```
Simulador-de-Ecosistemas/
├── ecosystem_simulator/
│   ├── llm/                          ← NUEVO: Módulos LLM
│   │   ├── __init__.py
│   │   ├── ollama_utils.py           ← Comunicación con Ollama
│   │   ├── behavior_selector.py      ← ENFOQUE 1
│   │   ├── reward_shaping.py         ← ENFOQUE 2
│   │   └── reflexion.py              ← ENFOQUE 3
│   ├── entities/
│   ├── environment/
│   ├── training/
│   ├── server/
│   └── utils/
│
├── run_with_llm_behavior_selector.py ← Script ENFOQUE 1
├── run_with_llm_reward_shaping.py    ← Script ENFOQUE 2
├── run_with_llm_reflexion.py         ← Script ENFOQUE 3
│
├── examples_llm.py                   ← Ejemplos simples
├── QUICK_START.md                    ← Inicio rápido (5 min)
├── RESUMEN_EJECUTIVO.md              ← Resumen de todo
├── LLM_INTEGRATION_README.md         ← Documentación detallada
├── requirements.txt                  ← Actualizado con requests
│
├── run_training.py                   (original, sin cambios)
├── run_with_godot.py                 (original, sin cambios)
└── ... (otros archivos originales sin cambios)
```

---

## Matriz de Decisión

¿Cuál enfoque debo usar?

| Si quieres... | Usa | Razón |
|--------------|-----|-------|
| **Comportamientos explícitos** | Selector | Más interpretable |
| **Objetivos dinámicos** | Reward Shaping | Modula en tiempo real |
| **Aprender de errores** | Reflexion | Acumula lecciones |
| **Empezar rápido** | Selector | Más simple |
| **Mejor rendimiento** | Reward Shaping | Más fino tuning |
| **Long-term learning** | Reflexion | Mejora iterativa |

---

## Flujos de Trabajo

### **Workflow A: Prueba Rápida** (5-10 minutos)

```bash
# 1. Instalar Ollama
ollama run mistral

# 2. En otra terminal
pip install requests

# 3. Ejecutar ejemplos
python examples_llm.py

# 4. Ver resultados
# ✓ Todos funcionan y generan insights LLM
```

### **Workflow B: Comparar Enfoques** (1 hora)

```bash
# Terminal 1: Behavior Selector
python run_with_llm_behavior_selector.py
# monitor_llm_behavior.csv

# Terminal 2: Reward Shaping
python run_with_llm_reward_shaping.py
# monitor_llm_reward_shaping.csv

# Terminal 3: Reflexion
python run_with_llm_reflexion.py
# monitor_llm_reflexion.csv

# Luego: comparar CSV en Excel/Python
import pandas as pd
df1 = pd.read_csv('monitor_llm_behavior.csv')
df2 = pd.read_csv('monitor_llm_reward_shaping.csv')
df3 = pd.read_csv('monitor_llm_reflexion.csv')
```

### **Workflow C: Integración Personalizada** (1-2 horas)

```python
# En tu código de entrenamiento custom:
from ecosystem_simulator.llm.behavior_selector import BehaviorSelector
from ecosystem_simulator.llm.reward_shaping import RewardShaper
from ecosystem_simulator.llm.reflexion import ReflectionAgent

# Usar uno o combinar múltiples
selector = BehaviorSelector(...)
shaper = RewardShaper(...)
agent = ReflectionAgent(...)

# En tu loop de entrenamiento
action = selector.get_action(obs, agent_id)
# o
reward = shaper.shape_reward(obs, action, reward, agent_id)
# o
agent.process_episode(trajectory, success, reward)
```

---

## Parámetros Clave

### ollama_utils.py
```python
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_TIMEOUT = 30  # segundos
DEFAULT_MODEL = "mistral"
```

### BehaviorSelector
```python
selector = BehaviorSelector(
    available_behaviors=["hunt", "explore", "survive"],
    model="mistral",
    temperature=0.3,  # Conservador
)
```

### RewardShaper
```python
shaper = RewardShaper(
    model="mistral",
    temperature=0.5,   # Balanceado
    max_bonus_weight=0.5,  # 50% máx ajuste
)
```

### ReflectionAgent
```python
agent = ReflectionAgent(
    model="mistral",
    temperature=0.6,   # Creativo
)
```

---

## Comandos Útiles

```bash
# Verificar Ollama
curl http://localhost:11434/api/tags

# Ver modelos disponibles
ollama list

# Descargar otro modelo
ollama run neural-chat
ollama run llama2

# Eliminar modelo
ollama rm mistral

# Ver uso de memoria
ollama serve  # Muestra estadísticas
```

---

## Debugging

### Habilitar logging detallado

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Ahora verás:
# [2026-05-12 15:30:45] DEBUG: [LLM] agent_0 seleccionó: hunt
# [2026-05-12 15:30:46] DEBUG: [RewardShaper] agent_1: bonus=-0.0500
```

### Inspeccionar prompts del LLM

En `ollama_utils.py`, descomentar:
```python
# En la función call_ollama():
print(f"Prompt: {prompt}")
print(f"Response: {response}")
```

### Medir latencia

```python
import time
start = time.time()
response = call_ollama(prompt)
print(f"Latencia: {time.time() - start:.2f}s")
```

---

## Almacenamiento de Datos

### CSVs generados
```
monitor_llm_behavior.csv
  - Iteración, return_mean, len_mean
  - Per-agent rewards/lengths
  - LLM selection distribution

monitor_llm_reward_shaping.csv
  - Iteración, return_mean, len_mean
  - llm_reward_calls, cache_hits, efficiency

monitor_llm_reflexion.csv
  - Iteración, return_mean, len_mean
  - total_episodes, reflections_generated, success_rate
```

### Guardar estado del agent
```python
# Reflexion
import json
memory_dict = {
    "episodes": len(agent.memory.episodes),
    "lessons": list(agent.memory.lessons),
    "stats": agent.get_statistics()
}
with open("agent_memory.json", "w") as f:
    json.dump(memory_dict, f, indent=2)
```

---

## Optimizaciones

### Para latencia (speedup)

1. **Cambiar modelo:**
   ```python
   LLM_MODEL = "neural-chat"  # 30% más rápido
   ```

2. **Reducir temperatura:**
   ```python
   temperature = 0.2  # Procesa menos
   ```

3. **Aumentar cache:**
   ```python
   RewardShaper(cache_size=200)
   ```

4. **Usar GPU:**
   ```bash
   # Ollama automáticamente usa GPU si está disponible
   ```

### Para exactitud

1. **Aumentar temperatura:**
   ```python
   temperature = 0.8  # Más creativo/diverso
   ```

2. **Usar modelo mejor:**
   ```bash
   ollama run mistral:medium
   ```

3. **Custom system prompt:**
   ```python
   call_ollama(prompt, system_prompt="Eres un experto en ecosistemas...")
   ```

---

## Testing Rápido

```python
# Verificar setup
from ecosystem_simulator.llm.ollama_utils import check_ollama_available
assert check_ollama_available()

# Probar selector
from ecosystem_simulator.llm.behavior_selector import BehaviorSelector
selector = BehaviorSelector(["a", "b"])
selector.register_policy("a", lambda x: 0)
selector.select_behavior([1,2,3], "test")  # Debe retornar "a" o "b"

# Probar reward shaper
from ecosystem_simulator.llm.reward_shaping import RewardShaper
shaper = RewardShaper()
reward = shaper.shape_reward([1,2,3], 4, 1.0, "test")  # Debe retornar float

# Probar reflexion
from ecosystem_simulator.llm.reflexion import ReflectionAgent
agent = ReflectionAgent()
agent.process_episode([{"action": 0, "reward": 1.0}], True, 1.0)
print(agent.get_context_prompt())  # Debe mostrar contexto
```

---

## Troubleshooting Avanzado

### "RuntimeError: No module named..."
```bash
# Asegurar que estás en el directorio correcto
cd tu/ruta/Simulador-de-Ecosistemas
python -c "import ecosystem_simulator; print(ecosystem_simulator.__file__)"
```

### "Ollama genera respuestas cortas/vagas"
```python
# Aumentar max_tokens (si Ollama lo soporta)
# O cambiar el system prompt
```

### "Cache no funciona / siempre llama LLM"
```python
# Verificar que state_to_text() genera textos similares para estados similares
obs = np.array([1, 2, 3, 4, 5])
text = shaper.state_to_text(obs, "agent_0")
print(text)  # Si varía mucho, el cache no cachea bien
```

---

## Contacto / Soporte

Si algo no funciona:
1. Revisar [QUICK_START.md](QUICK_START.md)
2. Revisar [LLM_INTEGRATION_README.md](LLM_INTEGRATION_README.md)
3. Verificar que Ollama está corriendo: `curl http://localhost:11434/api/tags`
4. Revisar logs con `logging.basicConfig(level=logging.DEBUG)`

---

**Última actualización:** Mayo 2026
