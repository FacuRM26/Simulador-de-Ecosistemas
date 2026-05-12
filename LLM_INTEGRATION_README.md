# Integración de LLM en Entrenamiento RL - Simulador de Ecosistemas

Este documento explica los 3 enfoques implementados para integrar Large Language Models (LLMs) en el entrenamiento del ecosistema multi-agente.

## 🚀 Instalación y Configuración Rápida

### 1. Instalar Ollama (LLM Local Gratuito)

**Opción A: Windows/Mac/Linux**
- Descargar desde: https://ollama.ai
- Instalar normalmente
- Abrir terminal y ejecutar:
  ```bash
  ollama run mistral
  # O para modelo más ligero:
  ollama run neural-chat
  ```

**Diferencias de modelos:**
- `mistral` (7B): Equilibrado, ~4GB RAM, recomendado
- `neural-chat` (7B): Más rápido, ~3GB RAM, ideal para pruebas
- `llama2` (7B): Similar a Mistral, ~4GB RAM

### 2. Instalar dependencias Python

```bash
pip install requests  # Para comunicar con Ollama
# El resto ya están en requirements.txt
```

### 3. Verificar conexión a Ollama

```python
from ecosystem_simulator.llm.ollama_utils import check_ollama_available
if check_ollama_available():
    print("✓ Ollama está disponible en http://localhost:11434")
else:
    print("✗ Ollama no está disponible")
```

---

## 📋 Los 3 Enfoques Implementados

### **ENFOQUE 1: LLM como Selector de Comportamientos Pre-entrenados**

**Concepto:**
```
Estado del Agente (numérico)
        ↓
    [LLM - Analiza estado]
        ↓
Selecciona comportamiento: "huir" / "explorar" / "cazar"
        ↓
Política RL pre-entrenada ejecuta la acción
```

**Archivo:** [ecosystem_simulator/llm/behavior_selector.py](ecosystem_simulator/llm/behavior_selector.py)

**Script de entrenamiento:** [run_with_llm_behavior_selector.py](run_with_llm_behavior_selector.py)

**Ventajas:**
- ✅ No modifica el entrenamiento RL existente
- ✅ El LLM solo hace clasificación (tarea simple y confiable)
- ✅ Las políticas son explícitas e interpretables
- ✅ Bajo costo computacional

**Cómo funciona:**
1. Se entrenan 2+ políticas RL para comportamientos específicos
2. En cada paso, el LLM recibe descripción textual del estado
3. El LLM elige qué política activar
4. La política genera la acción numérica

**Ejemplo de uso:**
```python
from ecosystem_simulator.llm.behavior_selector import BehaviorSelector

# Crear selector con comportamientos disponibles
selector = BehaviorSelector(
    available_behaviors=["explore", "survival", "hunt"],
    model="mistral"
)

# Registrar políticas (en producción serían políticas RL entrenadas)
selector.register_policy("explore", my_explore_policy)
selector.register_policy("survival", my_survival_policy)

# En el loop de entrenamiento:
action = selector.get_action(obs, agent_id, use_llm=True)
```

**Ejecutar:**
```bash
python run_with_llm_behavior_selector.py
# Monitor: monitor_llm_behavior.csv
```

---

### **ENFOQUE 2: LLM para Reward Shaping Dinámico**

**Concepto:**
```
Acción tomada + Recompensa base
        ↓
    [LLM - Analiza contexto]
        ↓
Genera insight: "en este contexto, priorizar X"
        ↓
Ajusta recompensa: reward = reward_base + bonus_dinámico
        ↓
RL entrena con recompensa modulada
```

**Archivo:** [ecosystem_simulator/llm/reward_shaping.py](ecosystem_simulator/llm/reward_shaping.py)

**Script de entrenamiento:** [run_with_llm_reward_shaping.py](run_with_llm_reward_shaping.py)

**Ventajas:**
- ✅ El RL sigue siendo el motor principal
- ✅ El LLM solo modula (no controla)
- ✅ Permite comportamientos emergentes con objetivos dinámicos
- ✅ Menos overhead que cambiar acciones

**Cómo funciona:**
1. El LLM analiza el estado actual
2. Genera un "insight" (ej: "buscar comida", "evitar predador")
3. Según el insight y la acción, calcula bonus/penalización
4. La recompensa se ajusta: `r_final = r_base + bonus`

**Ejemplo de uso:**
```python
from ecosystem_simulator.llm.reward_shaping import RewardShaper

shaper = RewardShaper(
    model="mistral",
    max_bonus_weight=0.5  # 50% de ajuste máximo
)

# En cada paso de entrenamiento:
base_reward = env.get_reward()
shaped_reward = shaper.shape_reward(
    obs, action, base_reward, agent_id, use_llm=True
)
```

**Ejemplo de insights generados:**
```
- "buscar_comida"    → bonus si action == EAT
- "buscar_agua"      → bonus si action == DRINK
- "evitar_predador"  → bonus si action == MOVE_AWAY
- "acercarse_grupo"  → bonus si action == MOVE_TOWARDS_OTHERS
- "congelarse"       → bonus si action == IDLE
```

**Ejecutar:**
```bash
python run_with_llm_reward_shaping.py
# Monitor: monitor_llm_reward_shaping.csv
```

---

### **ENFOQUE 3: LLM con Reflexion (Aprendizaje entre Episodios)**

**Concepto:**
```
Episodio completo finaliza
        ↓
[LLM - Analiza trayectoria]
        ↓
Genera lección: "aprendí que..."
        ↓
Almacena en memoria a largo plazo
        ↓
Próximo episodio inyecta lecciones en contexto
        ↓
El agente toma mejores decisiones iterativamente
```

**Archivo:** [ecosystem_simulator/llm/reflexion.py](ecosystem_simulator/llm/reflexion.py)

**Script de entrenamiento:** [run_with_llm_reflexion.py](run_with_llm_reflexion.py)

**Ventajas:**
- ✅ Aprendizaje acumulativo a nivel episódico
- ✅ El agente "recuerda" y aprende de errores
- ✅ Explicaciones interpretables
- ✅ Puede reducir iteraciones necesarias

**Cómo funciona:**
1. Al finalizar cada episodio, el LLM analiza la trayectoria
2. Compara con lecciones previas aprendidas
3. Genera nueva lección en primera persona
4. Almacena en memoria (últimas 5 lecciones)
5. En futuros episodios, inyecta lecciones en prompts

**Ejemplo de lecciones generadas:**
```
- "Cuando los depredadores se acercan, debo buscar agua primero para tener energía de huida"
- "Agruparme con otros herbívoros reduce 60% de ataques exitosos"
- "Explorar el norte del mapa tiene menor densidad de predadores"
```

**Ejemplo de uso:**
```python
from ecosystem_simulator.llm.reflexion import ReflectionAgent

agent = ReflectionAgent(model="mistral")

# Al finalizar cada episodio:
trajectory = [  # Lista de pasos: {obs, action, reward, done}
    {"obs": ..., "action": 3, "reward": 0.5},
    {"obs": ..., "action": 4, "reward": 2.0},
    # ...
]
agent.process_episode(
    trajectory=trajectory,
    success=total_reward > 100,
    total_reward=total_reward
)

# Para obtener contexto con lecciones:
context = agent.get_context_prompt()
print(context)
```

**Ejecutar:**
```bash
python run_with_llm_reflexion.py
# Monitor: monitor_llm_reflexion.csv
```

---

## 📊 Monitoreo y Estadísticas

Cada script genera un archivo CSV con métricas:

**monitor_llm_behavior.csv:**
- `iter`: Iteración
- `r_mean`, `l_mean`: Recompensa y longitud promedio de episodios
- `llm_selections_*`: Conteo de selecciones por comportamiento
- Recompensas y longitudes por agente

**monitor_llm_reward_shaping.csv:**
- `iter`: Iteración
- Métricas RL estándar
- `llm_reward_calls`: Llamadas al LLM
- `cache_hits`: Aciertos de cache (evita llamadas repetidas)
- `cache_efficiency`: Porcentaje de eficiencia del cache

**monitor_llm_reflexion.csv:**
- `iter`: Iteración
- Métricas RL estándar
- `total_episodes`: Episodios procesados
- `reflections_generated`: Lecciones generadas
- `success_rate`: Tasa de éxito de episodios

---

## 🔧 Configuración Avanzada

### Cambiar modelo de LLM

En los scripts, cambiar:
```python
LLM_MODEL = "mistral"  # cambiar a:
# LLM_MODEL = "neural-chat"  # Más rápido
# LLM_MODEL = "llama2"       # Similar a mistral
```

### Ajustar temperatura

La temperatura controla la "creatividad" del LLM (0.0 = determinístico, 1.0 = creativo):

```python
# En behavior_selector.py
selector = BehaviorSelector(..., temperature=0.3)  # Más conservador

# En reward_shaping.py
shaper = RewardShaper(..., temperature=0.5)  # Balanceado

# En reflexion.py
agent = ReflectionAgent(..., temperature=0.6)  # Más creativo
```

### Ajustar peso de reward shaping

```python
# En reward_shaping.py
shaper = RewardShaper(..., max_bonus_weight=0.5)  # 50% máx ajuste
```

### Frecuencia de reflexiones

```python
# En run_with_llm_reflexion.py
REFLECTION_INTERVAL = 5  # Generar reflexión cada 5 episodios
```

---

## 🐛 Debugging y Troubleshooting

### "Ollama no está disponible"

```bash
# Verificar que Ollama está corriendo:
curl http://localhost:11434/api/tags

# Si no funciona, iniciar Ollama:
ollama run mistral
```

### Llamadas al LLM son lentas

```python
# Opción 1: Usar modelo más ligero
ollama run neural-chat

# Opción 2: Aumentar timeout (en ollama_utils.py)
OLLAMA_TIMEOUT = 60  # segundos

# Opción 3: Reducir temperatura (menos procesamiento)
temperature=0.2
```

### Memory error / Out of memory

```python
# Reducir tamaño del cache
RewardShaper(cache_size=50)  # default es 100

# Reducir número de lecciones en memoria
EpisodeMemory(max_history=20)  # default es 50
```

---

## 📚 Comparación de Enfoques

| Criterio | Behavior Selector | Reward Shaping | Reflexion |
|----------|-------------------|-----------------|-----------|
| **Complejidad** | Baja | Media | Alta |
| **Costo LLM** | Bajo | Medio | Bajo (post-episodio) |
| **Interpretabilidad** | Alta | Media | Alta |
| **Aprendizaje acumulativo** | No | No | Sí |
| **Modificación RL** | Mínima | Moderada | Nula |
| **Latencia** | ~1s por acción | ~0.1s (con cache) | Offline |
| **Mejor para** | Comportamientos discretos | Objetivos dinámicos | Aprendizaje a largo plazo |

---

## 📖 Referencias y Papers

- **Reflexion**: https://arxiv.org/abs/2303.11366
- **Reward Shaping**: https://www.jmlr.org/papers/volume13/silver12a/silver12a.pdf
- **LLM-based Decision Making**: https://arxiv.org/abs/2305.14405

---

## 💡 Próximas Mejoras

Sugerencias para extender estos enfoques:

1. **Hybrid Approach**: Combinar 2-3 enfoques simultáneamente
2. **Fine-tuning**: Entrenar modelos LLM específicos para el dominio
3. **Memory Augmentation**: Usar base de datos para lecciones (en lugar de lista)
4. **Multi-agent Cooperation**: Que los LLMs negocien entre agentes
5. **Online Learning**: Actualizar prompts según feedback en tiempo real

---

## 📝 Licencia

Estos módulos son parte del proyecto Simulador-de-Ecosistemas.

---

**Última actualización:** Mayo 2026
