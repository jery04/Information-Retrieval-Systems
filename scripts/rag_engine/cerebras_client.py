"""Cerebras API client with retry logic."""

import logging
import time
from typing import Tuple

from .config import RAGConfig

logger = logging.getLogger(__name__)


class CerebrasClient:
    """Client for Cerebras API with exponential backoff retry logic."""
    
    def __init__(self, config: RAGConfig):
        """
        Initialize Cerebras client.
        
        Args:
            config: RAGConfig instance with API credentials
        """
        self.config = config
        self._client = None
        self._validate_and_init()
    
    def _validate_and_init(self):
        """Validate config and initialize Cerebras client."""
        if not self.config.validate():
            logger.error("RAG Config validation failed - missing required fields")
            raise ValueError("Invalid RAGConfig: missing required fields")
        
        try:
            from cerebras.cloud.sdk import Cerebras
            self._client = Cerebras(api_key=self.config.api_key)
            logger.info(f"Cerebras client initialized with model: {self.config.model}")
        except ImportError:
            logger.error(
                "cerebras_cloud_sdk not installed. "
                "Install with: pip install cerebras-cloud-sdk"
            )
            raise ImportError("cerebras_cloud_sdk is required for RAG with LLM")
        except Exception as e:
            logger.error(f"Failed to initialize Cerebras client: {e}")
            raise
    
    def generate(self, prompt: str) -> Tuple[str, bool]:
        """
        Generate response from Cerebras with exponential backoff retry.
        
        Args:
            prompt: The prompt to send to Cerebras
        
        Returns:
            (response_text, was_successful)
            - response_text: Generated text or empty string if failed
            - was_successful: True if API call succeeded, False if fallback is needed
        """
        if not self._client:
            logger.error("Cerebras client not initialized")
            return "", False
        
        for attempt in range(1, self.config.max_retries + 1):
            try:
                logger.info(
                    f"Sending request to Cerebras (attempt {attempt}/{self.config.max_retries})"
                )
                
                # Correct API: client.chat.completions.create()
                response = self._client.chat.completions.create(
                    model=self.config.model,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                )
                
                # Correct response structure: response.choices[0].message.content
                if response and response.choices and len(response.choices) > 0:
                    answer = response.choices[0].message.content.strip()
                    logger.info(f"Successfully generated response ({len(answer)} chars)")
                    return answer, True
                
                logger.warning("Empty response from Cerebras")
                return "", False
            
            except TimeoutError:
                logger.warning(f"Timeout on attempt {attempt}/{self.config.max_retries}")
                if attempt < self.config.max_retries:
                    self._backoff_wait(attempt)
                    continue
                return "", False
            
            except Exception as e:
                error_name = type(e).__name__
                logger.warning(
                    f"Error on attempt {attempt}/{self.config.max_retries}: {error_name}: {e}"
                )
                
                if attempt < self.config.max_retries:
                    self._backoff_wait(attempt)
                    continue
                
                logger.error(f"Failed after {self.config.max_retries} attempts, using fallback")
                return "", False
        
        return "", False
    
    def _backoff_wait(self, attempt: int):
        """Wait with exponential backoff before retry."""
        wait_time = self.config.retry_backoff_factor ** (attempt - 1)
        logger.info(f"Waiting {wait_time}s before retry...")
        time.sleep(wait_time)
