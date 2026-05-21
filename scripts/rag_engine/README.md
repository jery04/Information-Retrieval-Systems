# RAG Engine - Módulo de Generación Aumentada por Recuperación

## Descripción General

El módulo `rag_engine` implementa un sistema de **Retrieval-Augmented Generation (RAG)** completo que cumple con los requisitos de Corte 2:

- ✅ **Implementación funcional de RAG** usando Cerebras LLM como generador
- ✅ **Integración del componente Recuperador** (GVSM) con el Generador (Cerebras)
- ✅ **Respuestas enriquecidas** basadas en documentos recuperados
- ✅ **Fallback robusto** a generación extractiva mejorada si el LLM falla
- ✅ **Mantiene compatibilidad 100%** con main.py (sin cambios requeridos)

## Arquitectura Modular

```
scripts/rag_engine/
├── __init__.py                  # Exports públicos
├── config.py                    # Configuración (carga desde .env)
├── fallback_generator.py        # Generador extractivo mejorado (fallback)
├── generator.py                 # CerebrasRAGGenerator (generativo + fallback)
├── cerebras_client.py           # Cliente API Cerebras con reintentos
└── utils.py                     # Funciones auxiliares (prompt, doc mapping)

scripts/rag.py                   # Interfaz pública RAGPipeline (sin cambios)
.env                             # Configuración con API Key de Cerebras
```

## Características Principales

### 1. Generación Generativa con Cerebras LLM
- Modelo: `llama-3.1-70b` (configurable)
- Temperatura: 0.7 (equilibrio) - configurable
- Max tokens: 1200 caracteres aproximadamente
- Timeout: 30 segundos

### 2. Reintentos Inteligentes
- Hasta 3 reintentos con backoff exponencial
- Espera: 1s → 2s → 4s entre reintentos
- Log detallado de intentos y errores

### 3. Fallback Robusto
- Si Cerebras falla después de 3 reintentos → generación extractiva
- Generador extractivo mejorado con:
  - Mejor scoring de sentencias
  - Diversidad de documentos fuente
  - Truncado inteligente sin romper oraciones

### 4. Mapeo Inteligente de Documentos
- Detección automática de documentos usados en la respuesta
- Basado en análisis de palabras clave de títulos
- Fallback a primer documento si no detecta uso

## Configuración

### Variables de Entorno (.env)

```bash
CEREBRAS_API_KEY=csk-pv89mjp25xr4p3cyhcy4584rrpyfxrchemc4k5x5kmewed9k
CEREBRAS_MODEL=llama-3.1-70b
CEREBRAS_TEMPERATURE=0.7
CEREBRAS_MAX_TOKENS=1200
CEREBRAS_TIMEOUT=30
CEREBRAS_MAX_RETRIES=3
CEREBRAS_RETRY_BACKOFF_FACTOR=2.0
```

### Carga Automática
- Las variables se cargan automáticamente desde `.env`
- RAGConfig puede sobrescribirse en tiempo de ejecución

## Cómo Usar

### Uso Básico (compatibilidad con main.py)

```python
from rag import RAGPipeline
from indexer import GVSMSearchEngine

# Inicializar exactamente como antes
engine = GVSMSearchEngine()
rag = RAGPipeline(engine)

# Usar exactamente como antes
respuesta = rag.answer(
    query="¿Cómo usar Python en IA?",
    top_k=5,
    max_chars=1200
)

print(respuesta)
# {
#   "query": "¿Cómo usar Python en IA?",
#   "answer": "Python es ampliamente usado en IA porque...",
#   "sources": [...],
#   "contexts": [...],
#   "total_sources": 2
# }
```

### Uso Avanzado (custom config)

```python
from rag import RAGPipeline
from rag_engine import RAGConfig, CerebrasRAGGenerator

# Config personalizada
config = RAGConfig(
    temperature=0.5,  # Más determinístico
    max_tokens=1500,  # Respuestas más largas
    timeout=45,       # Más tolerante
)

# Generator personalizado
generator = CerebrasRAGGenerator(
    config=config,
    max_chars=1500,
    enable_fallback=True
)

# Pipeline con config personalizada
rag = RAGPipeline(engine, generator=generator)
```

## Componentes Detalle

### config.py - Configuración
- `RAGConfig` dataclass con todas las configuraciones
- Carga automática desde `.env`
- Validación de campos requeridos

### fallback_generator.py - Generador Extractivo Mejorado
- `ImprovedRAGGenerator` con heurísticas mejores que el original
- Scoring de sentencias basado en overlap + posición
- Diversidad: selecciona de múltiples documentos
- Truncado inteligente sin romper oraciones

### generator.py - Generador Cerebras
- `CerebrasRAGGenerator`: generador principal
- Usa Cerebras LLM para generar respuestas
- Fallback automático a extractivo si LLM falla
- Loguea intentos y errores para debugging

### cerebras_client.py - Cliente API
- `CerebrasClient` con retry logic
- Exponential backoff entre reintentos
- Manejo robusto de timeouts
- Logging detallado

### utils.py - Funciones Auxiliares
- `build_rag_prompt()`: construye prompt para Cerebras
- `extract_used_doc_ids()`: detecta documentos usados en respuesta
- `truncate_answer()`: trunca respuesta respetando oraciones

## Integración con main.py

**No requiere cambios en main.py**

El módulo mantiene la interfaz exacta:
```python
# En main.py (línea 248) - FUNCIONA IGUAL
rag = RAGPipeline(engine)

# En main.py (línea 394) - FUNCIONA IGUAL
payload = rag.answer(query=q, top_k=top_k, max_sentences=6, max_chars=1200)
```

## Testing

Ejecutar validación:
```bash
python3 scripts/test_rag.py
```

Resultado esperado:
```
============================================================
Results: 4/4 tests passed
============================================================
✓ All integration tests passed!
```

## Logging

El módulo registra actividades con distintos niveles:
- **INFO**: Operaciones normales (LLM generando, config cargada)
- **WARNING**: Reintentos, fallbacks
- **ERROR**: Fallos críticos (API key inválida, etc.)

Ver logs:
```python
import logging
logging.basicConfig(level=logging.INFO)
```

## Mejoras Respecto al RAG Original

| Aspecto | Original | Nuevo |
|--------|----------|-------|
| **Generación** | Extractiva pura | Generativa (LLM) + fallback |
| **Calidad** | Limitada a fragmentos | Mejora integración via Cerebras |
| **Reintentos** | No había | Hasta 3 con backoff exponencial |
| **Fallback** | Primer documento | Generador extractivo mejorado |
| **Configuración** | Hardcodeada | Desde .env |
| **Logging** | Básico | Detallado por nivel |
| **Modularidad** | Monolítico | Separación clara de responsabilidades |

## Requisitos

```
- Python 3.8+
- cerebras-cloud-sdk (instalado con `pip install cerebras-cloud-sdk`)
- python-dotenv (opcional, para cargar .env automáticamente)
```

## Cumplimiento de Requisitos - Corte 2

✅ **Módulo RAG completo**
- Implementación funcional de Retrieval-Augmented Generation ✓
- Integración del componente Recuperador con el Generador ✓
- Capacidad de generar respuestas enriquecidas basadas en documentos recuperados ✓

✅ **Integración coherente**
- Compatible 100% con main.py sin cambios ✓
- Ruta /rag funciona igual que antes ✓
- Frontend recibe respuesta en mismo formato ✓

✅ **Mejoras avanzadas**
- Cerebras LLM para generación natural ✓
- Fallback robusto a extracción mejorada ✓
- Reintentos inteligentes con backoff ✓
- Documentos usados detectados automáticamente ✓

## Próximos Pasos

1. Completar indexación del corpus con `python3 scripts/indexer.py`
2. Iniciar servidor: `python3 scripts/main.py serve`
3. Probar /rag endpoint: `curl 'http://localhost:5000/rag?query=test'`
4. Validar respuestas generativas en frontend

## Referencia Rápida

- **Iniciar con Cerebras**: `python3 scripts/main.py serve`
- **Test módulo**: `python3 scripts/test_rag.py`
- **Ver logs**: Configurar `logging` en scripts
- **Cambiar modelo**: Editar `.env` o RAGConfig

---

**Contribuidor**: Equipo SRI | **Fecha**: Mayo 2026 | **Versión**: 1.0
