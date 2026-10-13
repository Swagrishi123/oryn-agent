"""Isolated LLM Provider Adapter with Exponential Backoff and OTPM Handling."""
import logging
import re
import time
from typing import Dict, List, Optional
from groq import Groq

logger = logging.getLogger("oryn.llm")


class GroqProvider:
    DEFAULT_MODELS = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "llama3-70b-8192",
        "llama3-8b-8192"
    ]

    def __init__(self, api_key: str, preferred_model: Optional[str] = None):
        self.api_key = api_key.strip()
        self.preferred_model = preferred_model or self.DEFAULT_MODELS[0]
        self._client: Optional[Groq] = None

    @property
    def client(self) -> Groq:
        if not self._client:
            if not self.api_key:
                raise ValueError("Groq API Key is missing.")
            self._client = Groq(api_key=self.api_key, max_retries=0)
        return self._client

    def complete(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 800,
        temperature: float = 0.1,
        retries: int = 2
    ) -> str:
        """Executes LLM completion with bounded backoff and automatic OTPM rate-limit mitigation."""
        current_model = self.preferred_model
        budget = min(max_tokens, 800)  # Bound to stay under Groq 1,000 OTPM ceiling

        for attempt in range(retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=current_model,
                    messages=messages,
                    max_tokens=budget,
                    temperature=temperature,
                    frequency_penalty=0.15,
                    presence_penalty=0.10
                )
                return resp.choices[0].message.content or ""
            except Exception as e:
                err_str = str(e).lower()
                logger.warning(f"Groq API error on {current_model} (attempt {attempt}): {err_str}")

                # Self-healing if output tokens per minute quota is exceeded
                if any(k in err_str for k in ["output tokens per minute", "otpm", "reduce max_tokens"]):
                    if budget > 450:
                        budget = 450
                        time.sleep(1.0)
                        continue

                # Rate-limit 429 delay handling
                if "429" in err_str and attempt < retries:
                    delay_match = re.search(r'try again in ([0-9.]+)s', err_str)
                    wait_time = float(delay_match.group(1)) + 0.5 if delay_match else 2.5
                    time.sleep(min(wait_time, 5.0))
                    continue

                # Fallback to backup model if model is retired, missing, or unavailable
                if any(k in err_str for k in ["404", "decommissioned", "model_not_found"]):
                    for fallback in self.DEFAULT_MODELS:
                        if fallback != current_model:
                            logger.info(f"Switching from {current_model} to fallback model {fallback}")
                            current_model = fallback
                            break
                    continue

                if attempt == retries:
                    raise RuntimeError(f"Groq inference failed: {e}") from e

        return "Inference failed to produce a valid response."
