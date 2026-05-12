# ⚡ Quick Start - LLM + RL en 5 Minutos

## Paso 1: Instalar Ollama (2 minutos)

### Windows / Mac / Linux
1. Descargar desde: https://ollama.ai
2. Instalar normalmente
3. Abrir terminal y ejecutar:
   ```bash
   ollama run mistral
   ```
4. Esperar a que descargue (~4GB) y aparezca el prompt `>>>`

### ¿Está funcionando?
```bash
curl http://localhost:11434/api/tags
# Debe mostrar JSON con los modelos disponibles
```

---

## Paso 2: Instalar dependencias (1 minuto)

```bash
cd tu/ruta/Simulador-de-Ecosistemas
pip install requests
```

---

## Paso 3: Probar con ejemplos (2 minutos)

```bash
python examples_llm.py
```

**Salida esperada:**
```
✓ Ollama disponible

EJEMPLO 1: LLM Behavior Selector
  Estado: Mucha energía, solo
  ✓ Comportamiento seleccionado: explore

EJEMPLO 2: LLM Reward Shaping
  Recompensa base: 1.0000
  Recompensa ajustada: 1.2500
  Bonus/penalización: +0.2500

EJEMPLO 3: LLM Reflexion
  [Lecciones generadas por LLM...]
```

---

## Paso 4: Elegir un Enfoque y Ejecutar

### **Opción A: Behavior Selector** (Fácil)
```bash
python run_with_llm_behavior_selector.py
```
El LLM elige qué comportamiento activar en cada paso.

### **Opción B: Reward Shaping** (Recomendado)
```bash
python run_with_llm_reward_shaping.py
```
El LLM modula dinámicamente la recompensa según el contexto.

### **Opción C: Reflexion** (Avanzado)
```bash
python run_with_llm_reflexion.py
```
El LLM aprende lecciones de episodios completos.

---

## Qué Esperar

- Entrenamiento: **~10-15 minutos** (100 iteraciones)
- Llamadas al LLM: **Transparentes** en el log
- Monitor: Se generan archivos CSV con métricas

```
Iter   0: return=5.23, len=120.45, LLM_selections=32
Iter   1: return=5.89, len=125.12, LLM_selections=28
...
```

---

## Examinar Resultados

```bash
# Mostrar métricas del CSV
python -c "import pandas as pd; df = pd.read_csv('monitor_llm_behavior.csv'); print(df.head(10))"

# Graficar en Jupyter
import pandas as pd
df = pd.read_csv('monitor_llm_behavior.csv')
df[['r_mean', 'l_mean']].plot()
```

---

## Si Algo Falla

### "Ollama no está disponible"
```bash
# Verifica que Ollama está corriendo
curl http://localhost:11434/api/tags

# Si no, inicia en otra terminal
ollama run mistral
```

### "Timeout esperando LLM"
```python
# Reduce la temperatura (menos procesamiento)
# En el script, cambiar:
temperature=0.2  # era 0.3 / 0.5 / 0.6
```

### "ImportError: No module named 'requests'"
```bash
pip install requests
```

---

## Archivos Importantes

| Archivo | Propósito |
|---------|-----------|
| `examples_llm.py` | Pruebas rápidas sin Ray |
| `run_with_llm_behavior_selector.py` | Enfoque 1 |
| `run_with_llm_reward_shaping.py` | Enfoque 2 |
| `run_with_llm_reflexion.py` | Enfoque 3 |
| `LLM_INTEGRATION_README.md` | Documentación completa |
| `RESUMEN_EJECUTIVO.md` | Resumen de todo |

---

## Próximos Pasos

1. **Leer:** [RESUMEN_EJECUTIVO.md](RESUMEN_EJECUTIVO.md) (5 min)
2. **Entender:** [LLM_INTEGRATION_README.md](LLM_INTEGRATION_README.md) (20 min)
3. **Experimentar:** Cambiar temperaturas, modelos, parámetros
4. **Comparar:** Ejecutar los 3 enfoques y analizar diferencias

---

## Modelos Alternativos

```bash
# Más rápido (recomendado si es lento)
ollama run neural-chat

# Similar a Mistral
ollama run llama2

# Más pesado (mejor calidad)
ollama run mistral:medium
```

Cambiar en los scripts:
```python
LLM_MODEL = "neural-chat"  # en lugar de "mistral"
```

---

## Resumen

```bash
# 1. Instalar Ollama (descargar + ollama run mistral)
# 2. pip install requests
# 3. python examples_llm.py
# 4. python run_with_llm_behavior_selector.py  # (o los otros)
# 5. Analizar monitor_llm_*.csv
```

**¡Listo! 🚀**
