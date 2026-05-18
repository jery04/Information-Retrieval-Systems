from __future__ import annotations

from typing import Set


TECH_KEYWORDS: Set[str] = {
    "software", "programming", "programacion", "programación", "developer", "desarrollador",
    "development", "desarrollo", "coding", "codificacion", "codificación", "python", "java",
    "javascript", "typescript", "rust", "golang", "go", "react", "angular", "vue", "django",
    "flask", "node", "api", "backend", "frontend", "microservices", "microservicios",
    "kubernetes", "devops", "cloud", "nube", "database", "base de datos", "algorithms",
    "algoritmos", "architecture", "arquitectura", "security", "seguridad", "machine learning",
    "aprendizaje automatico", "aprendizaje automático", "artificial intelligence",
    "inteligencia artificial", "data science", "ciencia de datos", "web development",
    "desarrollo web", "open source", "opensource",
}


def score_relevance(title: str, text: str, url: str) -> float:
    """Compute bilingual tech relevance by keyword presence."""
    blob = f"{title} {text[:5000]} {url}".lower()
    return float(sum(1 for keyword in TECH_KEYWORDS if keyword in blob))
