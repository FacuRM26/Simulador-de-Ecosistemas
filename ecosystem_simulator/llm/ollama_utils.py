"""
Utilidades para interactuar con Ollama (LLM local).

Proporciona funciones para llamar a Ollama de forma segura, con manejo de
errores y timeouts.

Ollama debe estar ejecutándose en http://localhost:11434

Instalación:
  1. Descargar desde https://ollama.ai
  2. ollama run mistral (o neural-chat, llama2, etc)
"""
import requests
import json
import logging
from typing import Optional
from datetime import datetime

logger = logging.getLogger(__name__)

OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_TIMEOUT = 40  # segundos
DEFAULT_MODEL = "mistral"  # cambiar a neural-chat para menor latencia


class OllamaError(Exception):
    """Error al conectar con Ollama."""
    pass


def check_ollama_available() -> bool:
    """
    Verifica si Ollama está disponible y funcionando.
    
    Returns:
        bool: True si Ollama está disponible, False en caso contrario.
    """
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def call_ollama(
    prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    timeout: int = OLLAMA_TIMEOUT,
    system_prompt: Optional[str] = None,
) -> str:
    """
    Llama a Ollama y obtiene una respuesta.
    
    Args:
        prompt: El prompt para el LLM.
        model: Modelo a usar (default: mistral).
        temperature: Creatividad de la respuesta (0.0-1.0).
        timeout: Timeout en segundos.
        system_prompt: Prompt del sistema para contexto.
    
    Returns:
        str: La respuesta del LLM.
    
    Raises:
        OllamaError: Si hay error en la conexión o timeout.
    """
    if not check_ollama_available():
        raise OllamaError(
            f"Ollama no está disponible en {OLLAMA_API_URL}\n"
            "Instala Ollama desde https://ollama.ai e inicia con: ollama run mistral"
        )

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "temperature": temperature,
    }

    if system_prompt:
        payload["system"] = system_prompt

    try:
        response = requests.post(
            OLLAMA_API_URL,
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
        result = response.json()
        return result.get("response", "").strip()

    except requests.exceptions.Timeout:
        raise OllamaError(f"Timeout al llamar a Ollama (>{timeout}s)")
    except requests.exceptions.ConnectionError:
        raise OllamaError(f"No se puede conectar a Ollama en {OLLAMA_API_URL}")
    except json.JSONDecodeError:
        raise OllamaError("Respuesta inválida de Ollama")
    except Exception as e:
        raise OllamaError(f"Error inesperado: {e}")


def call_ollama_with_retry(
    prompt: str,
    model: str = DEFAULT_MODEL,
    max_retries: int = 2,
    **kwargs,
) -> str:
    """
    Llama a Ollama con reintentos automáticos en caso de error.
    
    Args:
        prompt: El prompt para el LLM.
        model: Modelo a usar.
        max_retries: Número máximo de reintentos.
        **kwargs: Otros argumentos para call_ollama.
    
    Returns:
        str: La respuesta del LLM.
    
    Raises:
        OllamaError: Si falla después de todos los reintentos.
    """
    for attempt in range(max_retries):
        try:
            return call_ollama(prompt, model=model, **kwargs)
        except OllamaError as e:
            if attempt == max_retries - 1:
                raise
            logger.warning(f"Intento {attempt + 1} falló: {e}. Reintentando...")


def extract_first_line(text: str) -> str:
    """Extrae la primera línea no vacía de un texto."""
    for line in text.split("\n"):
        line = line.strip()
        if line:
            return line
    return text.strip()


def extract_json_from_response(response: str) -> Optional[dict]:
    """
    Intenta extraer JSON de una respuesta del LLM.
    
    Args:
        response: Texto de respuesta del LLM.
    
    Returns:
        dict: El JSON parseado, o None si no se encuentra.
    """
    try:
        # Buscar contenido entre {}
        start = response.find("{")
        end = response.rfind("}") + 1
        if start >= 0 and end > start:
            json_str = response[start:end]
            return json.loads(json_str)
    except (json.JSONDecodeError, ValueError):
        pass
    return None


def log_llm_call(prompt: str, response: str, model: str) -> None:
    """
    Registra una llamada a LLM en el log (útil para debugging).
    """
    timestamp = datetime.now().isoformat()
    logger.debug(f"[{timestamp}] LLM Call ({model})")
    logger.debug(f"  Prompt: {prompt[:100]}...")
    logger.debug(f"  Response: {response[:100]}...")
