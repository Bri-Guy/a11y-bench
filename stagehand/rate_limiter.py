"""
Rate limiter module for API calls to various providers (OpenAI, Anthropic, Google).

This module implements a token bucket algorithm with exponential backoff retry logic
to handle rate limiting across different AI providers.
"""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, TypeVar

T = TypeVar("T")


class Provider(Enum):
    """Supported AI providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting a specific provider."""
    requests_per_minute: int = 60  # RPM limit
    tokens_per_minute: int = 100000  # TPM limit (not always used)
    max_retries: int = 5  # Maximum number of retries on rate limit errors
    base_delay: float = 1.0  # Base delay in seconds for exponential backoff
    max_delay: float = 60.0  # Maximum delay in seconds


# Default rate limit configurations per provider
DEFAULT_RATE_LIMITS: dict[Provider, RateLimitConfig] = {
    Provider.OPENAI: RateLimitConfig(
        requests_per_minute=500,  # GPT-4o tier 1 is ~500 RPM
        tokens_per_minute=30000,
        max_retries=5,
        base_delay=1.0,
        max_delay=60.0,
    ),
    Provider.ANTHROPIC: RateLimitConfig(
        requests_per_minute=50,  # Claude is typically ~50 RPM for tier 1
        tokens_per_minute=40000,
        max_retries=5,
        base_delay=1.0,
        max_delay=60.0,
    ),
    Provider.GOOGLE: RateLimitConfig(
        requests_per_minute=60,  # Gemini is typically ~60 RPM
        tokens_per_minute=32000,
        max_retries=5,
        base_delay=1.0,
        max_delay=60.0,
    ),
}


@dataclass
class TokenBucket:
    """Token bucket for rate limiting requests."""
    capacity: float  # Maximum tokens in bucket
    tokens: float = field(init=False)  # Current tokens
    refill_rate: float = field(init=False)  # Tokens added per second
    last_refill: float = field(init=False)  # Last refill timestamp
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    
    def __post_init__(self):
        self.tokens = self.capacity
        self.refill_rate = self.capacity / 60.0  # Convert RPM to per-second
        self.last_refill = time.monotonic()
    
    async def acquire(self, tokens: int = 1) -> float:
        """
        Acquire tokens from the bucket.
        
        Returns the time to wait before the tokens are available.
        If tokens are available immediately, returns 0.
        """
        async with self._lock:
            self._refill()
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return 0.0
            
            tokens_needed = tokens - self.tokens
            wait_time = tokens_needed / self.refill_rate
            return wait_time
    
    def _refill(self):
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self.last_refill
        tokens_to_add = elapsed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill = now


class RateLimiter:
    """
    Rate limiter for API calls with exponential backoff retry logic.
    
    Usage:
        limiter = RateLimiter()
        result = await limiter.execute(Provider.OPENAI, api_call_func, *args, **kwargs)
    """
    
    def __init__(
        self,
        configs: Optional[dict[Provider, RateLimitConfig]] = None,
        logger: Optional[Any] = None,
    ):
        """
        Initialize the rate limiter.
        
        Args:
            configs: Optional custom rate limit configurations per provider.
                    Merges with defaults, allowing partial overrides.
            logger: Optional logger for debugging rate limit events.
        """
        self._configs = {**DEFAULT_RATE_LIMITS}
        if configs:
            self._configs.update(configs)
        
        self._buckets: dict[Provider, TokenBucket] = {}
        self._logger = logger
        
        for provider, config in self._configs.items():
            self._buckets[provider] = TokenBucket(capacity=config.requests_per_minute)
    
    def _log(self, message: str, level: str = "debug"):
        """Log a message if logger is available."""
        if self._logger:
            if hasattr(self._logger, level):
                log_func = getattr(self._logger, level)
                if callable(log_func):
                    try:
                        log_func(message, category="rate_limiter")
                    except TypeError:
                        # Fallback if category is not supported
                        log_func(message)
    
    def _is_rate_limit_error(self, error: Exception) -> bool:
        """Check if an exception is a rate limit error."""
        error_str = str(error).lower()
        error_type = type(error).__name__.lower()
        
        rate_limit_indicators = [
            "rate_limit",
            "ratelimit",
            "rate limit",
            "too many requests",
            "429",
            "quota exceeded",
            "quota_exceeded",
            "resource exhausted",
            "resource_exhausted",
        ]
        
        return any(indicator in error_str or indicator in error_type 
                   for indicator in rate_limit_indicators)
    
    def _calculate_backoff(
        self, 
        attempt: int, 
        config: RateLimitConfig
    ) -> float:
        """Calculate exponential backoff delay with jitter."""
        import random
        
        # Exponential backoff: base_delay * 2^attempt
        delay = config.base_delay * (2 ** attempt)
        
        # Add jitter (±25%)
        jitter = delay * 0.25 * (2 * random.random() - 1)
        delay += jitter
        
        return min(delay, config.max_delay)
    
    async def execute(
        self,
        provider: Provider,
        func: Callable[..., T],
        *args,
        **kwargs,
    ) -> T:
        """
        Execute an API call with rate limiting and retry logic.
        
        Args:
            provider: The API provider (OpenAI, Anthropic, Google)
            func: The async function to call
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            The result of the function call
            
        Raises:
            The last exception if all retries are exhausted
        """
        config = self._configs.get(provider, DEFAULT_RATE_LIMITS[Provider.OPENAI])
        bucket = self._buckets.get(provider)
        
        if bucket is None:
            bucket = TokenBucket(capacity=config.requests_per_minute)
            self._buckets[provider] = bucket
        
        last_exception: Optional[Exception] = None
        
        for attempt in range(config.max_retries + 1):
            wait_time = await bucket.acquire()
            if wait_time > 0:
                self._log(f"Rate limit: waiting {wait_time:.2f}s for {provider.value}")
                await asyncio.sleep(wait_time)
            
            try:
                # Check if func is a coroutine function
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    # Run sync function in executor
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(None, lambda: func(*args, **kwargs))
                
                return result
                
            except Exception as e:
                last_exception = e
                
                if self._is_rate_limit_error(e):
                    if attempt < config.max_retries:
                        backoff = self._calculate_backoff(attempt, config)
                        self._log(
                            f"Rate limit hit for {provider.value}, "
                            f"retry {attempt + 1}/{config.max_retries} "
                            f"after {backoff:.2f}s: {e}",
                            level="warning" if hasattr(self._logger, "warning") else "info"
                        )
                        await asyncio.sleep(backoff)
                        continue
                
                # Not a rate limit error or retries exhausted
                raise
        
        # All retries exhausted
        if last_exception:
            raise last_exception
        raise RuntimeError("Unexpected error in rate limiter")
    
    async def execute_sync(
        self,
        provider: Provider,
        func: Callable[..., T],
        *args,
        **kwargs,
    ) -> T:
        """
        Execute a synchronous API call with rate limiting and retry logic.
        Wrapper for sync functions that runs them in an executor.
        
        Args:
            provider: The API provider (OpenAI, Anthropic, Google)
            func: The synchronous function to call
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            The result of the function call
        """
        config = self._configs.get(provider, DEFAULT_RATE_LIMITS[Provider.OPENAI])
        bucket = self._buckets.get(provider)
        
        if bucket is None:
            bucket = TokenBucket(capacity=config.requests_per_minute)
            self._buckets[provider] = bucket
        
        last_exception: Optional[Exception] = None
        
        for attempt in range(config.max_retries + 1):
            wait_time = await bucket.acquire()
            if wait_time > 0:
                self._log(f"Rate limit: waiting {wait_time:.2f}s for {provider.value}")
                await asyncio.sleep(wait_time)
            
            try:
                # Run sync function in executor
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None, 
                    lambda: func(*args, **kwargs)
                )
                return result
                
            except Exception as e:
                last_exception = e
                
                if self._is_rate_limit_error(e):
                    if attempt < config.max_retries:
                        backoff = self._calculate_backoff(attempt, config)
                        self._log(
                            f"Rate limit hit for {provider.value}, "
                            f"retry {attempt + 1}/{config.max_retries} "
                            f"after {backoff:.2f}s: {e}",
                            level="warning" if hasattr(self._logger, "warning") else "info"
                        )
                        await asyncio.sleep(backoff)
                        continue
                
                # Not a rate limit error or retries exhausted
                raise
        
        # All retries exhausted
        if last_exception:
            raise last_exception
        raise RuntimeError("Unexpected error in rate limiter")


# Global rate limiter instance (can be configured at module level)
_global_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter(logger: Optional[Any] = None) -> RateLimiter:
    """
    Get or create the global rate limiter instance.
    
    Args:
        logger: Optional logger to use (only used when creating new instance)
        
    Returns:
        The global RateLimiter instance
    """
    global _global_rate_limiter
    if _global_rate_limiter is None:
        _global_rate_limiter = RateLimiter(logger=logger)
    return _global_rate_limiter


def configure_rate_limiter(
    configs: Optional[dict[Provider, RateLimitConfig]] = None,
    logger: Optional[Any] = None,
) -> RateLimiter:
    """
    Configure and return a new global rate limiter instance.
    
    This replaces any existing global rate limiter.
    
    Args:
        configs: Optional custom rate limit configurations per provider
        logger: Optional logger for debugging
        
    Returns:
        The new global RateLimiter instance
    """
    global _global_rate_limiter
    _global_rate_limiter = RateLimiter(configs=configs, logger=logger)
    return _global_rate_limiter


def get_provider_from_model(model: str) -> Provider:
    """
    Determine the provider from a model name.
    
    Args:
        model: The model name (e.g., "gpt-4o", "claude-3-5-sonnet", "gemini-2.0-flash")
        
    Returns:
        The corresponding Provider enum value
    """
    model_lower = model.lower()
    
    if any(name in model_lower for name in ["claude", "anthropic"]):
        return Provider.ANTHROPIC
    elif any(name in model_lower for name in ["gemini", "google"]):
        return Provider.GOOGLE
    else:
        # Default to OpenAI for GPT models and others
        return Provider.OPENAI

