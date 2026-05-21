"""RAG Engine configuration module."""

import os
from dataclasses import dataclass
from typing import Optional


def _load_env() -> dict:
    """Load environment variables from .env file if it exists."""
    env_vars = {}
    env_path = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
    
    if os.path.exists(env_path):
        try:
            with open(env_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        key, value = line.split("=", 1)
                        env_vars[key.strip()] = value.strip()
        except Exception:
            pass
    
    return env_vars


_ENV_VARS = _load_env()


@dataclass
class RAGConfig:
    """Configuration for RAG engine with Cerebras integration."""
    
    # Cerebras API settings
    api_key: str = None
    model: str = None
    temperature: float = None
    max_tokens: int = None
    timeout: int = None
    max_retries: int = None
    retry_backoff_factor: float = None
    
    # RAG generation settings
    max_chars: int = 1200
    enable_fallback: bool = True
    log_level: str = "INFO"
    
    def __post_init__(self):
        """Load defaults from environment variables if not provided."""
        self.api_key = self.api_key or _ENV_VARS.get(
            "CEREBRAS_API_KEY", 
            "csk-pv89mjp25xr4p3cyhcy4584rrpyfxrchemc4k5x5kmewed9k"
        )
        self.model = self.model or _ENV_VARS.get("CEREBRAS_MODEL", "llama-3.1-70b")
        self.temperature = self.temperature or float(_ENV_VARS.get("CEREBRAS_TEMPERATURE", "0.7"))
        self.max_tokens = self.max_tokens or int(_ENV_VARS.get("CEREBRAS_MAX_TOKENS", "1200"))
        self.timeout = self.timeout or int(_ENV_VARS.get("CEREBRAS_TIMEOUT", "30"))
        self.max_retries = self.max_retries or int(_ENV_VARS.get("CEREBRAS_MAX_RETRIES", "3"))
        self.retry_backoff_factor = self.retry_backoff_factor or float(
            _ENV_VARS.get("CEREBRAS_RETRY_BACKOFF_FACTOR", "2.0")
        )
    
    def validate(self) -> bool:
        """Validate that configuration has all required fields."""
        required = ["api_key", "model", "timeout"]
        return all(getattr(self, field) is not None for field in required)
