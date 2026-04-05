# Information Retrieval Systems (IRS) 🔎

> ⚠️ **Estado del proyecto:** versión preliminar en desarrollo activo. La estructura, funcionalidades y resultados pueden evolucionar.

## 🧠 Descripción General

Este repositorio implementa un prototipo de **Sistema de Recuperación de Información (IRS)** orientado a:

- 📥 Ingesta y organización de documentos web.
- 🧹 Preprocesamiento de contenido para indexación.
- 🗂️ Construcción de índices especializados (trie invertido + coocurrencia).
- 📊 Ranking documental con **Generalized Vector Space Model (GVSM)**.
- 🌐 Exploración de resultados mediante interfaz web.

## 🏗️ Arquitectura del Sistema

El sistema está dividido en tres capas principales:

1. **Pipeline de datos**
   - Recolección de recursos en bruto (web, PDFs, imágenes).
   - Extracción y normalización del contenido.

2. **Motor de recuperación**
   - Construcción de índices en disco a partir del corpus procesado.
   - Cálculo de relaciones semánticas y puntuación de similitud con GVSM.

3. **Capa de presentación**
   - Aplicación frontend para búsqueda, filtros y visualización de resultados.

## 📁 Estructura del Repositorio

```text
data/
  raw/                # Recursos originales (webpages, pdfs, images)
  extracted/          # Contenido extraído y estructurado
  processed/          # Artefactos de indexación (JSON de índices)
scripts/
  indexer.py          # Construcción de índices
  gvsm_model.py       # Modelo de ranking basado en GVSM
webapp/
  src/                # Componentes React y estilos
  public/             # Assets estáticos
```

## ⚙️ Componentes Clave

- `scripts/indexer.py`
  - Genera estructuras de índice para acceso eficiente a términos y documentos.

- `scripts/gvsm_model.py`
  - Implementa el modelo vectorial generalizado para ranking por similitud.

- `webapp/`
  - Frontend con **Vite + React** para interacción de consulta y análisis visual.

## 🔄 Flujo de Trabajo de Datos

1. `data/raw` → entrada de recursos originales.
2. `data/extracted` → salida de extracción/parsing.
3. `scripts/indexer.py` → generación de índices en `data/processed`.
4. `scripts/gvsm_model.py` → cálculo de relevancia para consultas.
5. `webapp/` → consulta interactiva y visualización de resultados.

## 🚀 Estado y Objetivo

Proyecto enfocado en experimentación académica y mejora iterativa de técnicas de IR:

- ✅ Base funcional de indexación y ranking.
- 🧪 Espacio abierto para pruebas de calidad de recuperación.
- 📈 Evolución prevista en rendimiento, relevancia y UX de búsqueda.
