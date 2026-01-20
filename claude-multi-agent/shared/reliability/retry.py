"""
Automatic Retry with Jitter Implementation (adv-rel-002)

Implements retry logic with exponential backoff and jitter to prevent
thundering herd problems. Supports configurable retry strategies.
"""

import asyncio
import random
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, Callable, Any, Type, Tuple, TypeVar, Union
from functools import wraps

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    initial_delay_seconds: float = 1.0
    max_delay_seconds: float = 60.0
    exponential_base: float = 2.0
    jitter_factor: float = 0.5  # 0.0 = no jitter, 1.0 = full jitter
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,)
    non_retryable_exceptions: Tuple[Type[Exception], ...] = ()


@dataclass
class RetryStats:
    """Statistics for retry operations."""
    total_attempts: int = 0
    successful_retries: int = 0
    failed_retries: int = 0
    total_retry_time: float = 0.0


class RetryWithJitter:
    """
    Retry mechanism with exponential backoff and jitter.

    Implements the "full jitter" algorithm recommended by AWS:
    delay = random_between(0, min(cap, base * 2 ** attempt))

    This prevents synchronized retry storms when multiple agents fail simultaneously.

    Usage:
        retry = RetryWithJitter(config=RetryConfig(max_attempts=5))

        # As decorator
        @retry
        async def unreliable_operation():
            ...

        # Or explicitly
        result = await retry.execute(unreliable_operation, arg1, arg2)
    """

    def __init__(
        self,
        config: Optional[RetryConfig] = None,
        on_retry: Optional[Callable[[int, Exception, float], None]] = None,
        on_success: Optional[Callable[[int], None]] = None,
        on_failure: Optional[Callable[[int, Exception], None]] = None,
    ):
        """
        Initialize retry mechanism.

        Args:
            config: Retry configuration
            on_retry: Callback before each retry (attempt, exception, delay)
            on_success: Callback on success (attempts_used)
            on_failure: Callback on final failure (attempts_used, exception)
        """
        self.config = config or RetryConfig()
        self.on_retry = on_retry
        self.on_success = on_success
        self.on_failure = on_failure
        self._stats = RetryStats()

    @property
    def stats(self) -> RetryStats:
        """Get retry statistics."""
        return self._stats

    def calculate_delay(self, attempt: int) -> float:
        """
        Calculate delay for a given attempt using exponential backoff with full jitter.

        Args:
            attempt: The current attempt number (0-indexed)

        Returns:
            Delay in seconds with jitter applied
        """
        # Calculate base exponential delay
        base_delay = self.config.initial_delay_seconds * (
            self.config.exponential_base ** attempt
        )

        # Cap the delay
        capped_delay = min(base_delay, self.config.max_delay_seconds)

        # Apply jitter
        if self.config.jitter_factor > 0:
            # Full jitter: random value between 0 and capped_delay
            min_delay = capped_delay * (1 - self.config.jitter_factor)
            max_delay = capped_delay
            return random.uniform(min_delay, max_delay)

        return capped_delay

    def should_retry(self, exception: Exception) -> bool:
        """
        Determine if an exception should trigger a retry.

        Args:
            exception: The exception that was raised

        Returns:
            True if the exception is retryable
        """
        # Check non-retryable first
        if isinstance(exception, self.config.non_retryable_exceptions):
            return False

        # Then check retryable
        return isinstance(exception, self.config.retryable_exceptions)

    async def execute_async(
        self,
        func: Callable[..., Any],
        *args,
        **kwargs
    ) -> Any:
        """
        Execute an async function with retry logic.

        Args:
            func: The async function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            The result of the function

        Raises:
            The last exception if all retries fail
        """
        last_exception: Optional[Exception] = None

        for attempt in range(self.config.max_attempts):
            self._stats.total_attempts += 1

            try:
                result = await func(*args, **kwargs)

                # Success!
                if attempt > 0:
                    self._stats.successful_retries += 1
                    logger.info(
                        f"Retry succeeded after {attempt + 1} attempts"
                    )

                if self.on_success:
                    try:
                        self.on_success(attempt + 1)
                    except Exception as e:
                        logger.error(f"on_success callback error: {e}")

                return result

            except Exception as e:
                last_exception = e

                # Check if we should retry
                if not self.should_retry(e):
                    logger.warning(
                        f"Non-retryable exception: {type(e).__name__}: {e}"
                    )
                    raise

                # Check if we have attempts left
                if attempt + 1 >= self.config.max_attempts:
                    self._stats.failed_retries += 1
                    logger.error(
                        f"All {self.config.max_attempts} attempts failed. "
                        f"Last error: {type(e).__name__}: {e}"
                    )

                    if self.on_failure:
                        try:
                            self.on_failure(attempt + 1, e)
                        except Exception as callback_error:
                            logger.error(f"on_failure callback error: {callback_error}")

                    raise

                # Calculate delay and wait
                delay = self.calculate_delay(attempt)
                self._stats.total_retry_time += delay

                logger.warning(
                    f"Attempt {attempt + 1} failed: {type(e).__name__}: {e}. "
                    f"Retrying in {delay:.2f}s..."
                )

                if self.on_retry:
                    try:
                        self.on_retry(attempt + 1, e, delay)
                    except Exception as callback_error:
                        logger.error(f"on_retry callback error: {callback_error}")

                await asyncio.sleep(delay)

        # Should never reach here, but just in case
        raise last_exception if last_exception else RuntimeError("Retry logic error")

    def execute_sync(
        self,
        func: Callable[..., Any],
        *args,
        **kwargs
    ) -> Any:
        """
        Execute a sync function with retry logic.

        Args:
            func: The sync function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            The result of the function

        Raises:
            The last exception if all retries fail
        """
        last_exception: Optional[Exception] = None

        for attempt in range(self.config.max_attempts):
            self._stats.total_attempts += 1

            try:
                result = func(*args, **kwargs)

                if attempt > 0:
                    self._stats.successful_retries += 1
                    logger.info(f"Retry succeeded after {attempt + 1} attempts")

                if self.on_success:
                    try:
                        self.on_success(attempt + 1)
                    except Exception as e:
                        logger.error(f"on_success callback error: {e}")

                return result

            except Exception as e:
                last_exception = e

                if not self.should_retry(e):
                    logger.warning(f"Non-retryable exception: {type(e).__name__}: {e}")
                    raise

                if attempt + 1 >= self.config.max_attempts:
                    self._stats.failed_retries += 1
                    logger.error(
                        f"All {self.config.max_attempts} attempts failed. "
                        f"Last error: {type(e).__name__}: {e}"
                    )

                    if self.on_failure:
                        try:
                            self.on_failure(attempt + 1, e)
                        except Exception as callback_error:
                            logger.error(f"on_failure callback error: {callback_error}")

                    raise

                delay = self.calculate_delay(attempt)
                self._stats.total_retry_time += delay

                logger.warning(
                    f"Attempt {attempt + 1} failed: {type(e).__name__}: {e}. "
                    f"Retrying in {delay:.2f}s..."
                )

                if self.on_retry:
                    try:
                        self.on_retry(attempt + 1, e, delay)
                    except Exception as callback_error:
                        logger.error(f"on_retry callback error: {callback_error}")

                time.sleep(delay)

        raise last_exception if last_exception else RuntimeError("Retry logic error")

    def __call__(self, func: Callable) -> Callable:
        """Use as a decorator."""
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await self.execute_async(func, *args, **kwargs)
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                return self.execute_sync(func, *args, **kwargs)
            return sync_wrapper


def retry_with_jitter(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter_factor: float = 0.5,
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,),
) -> Callable:
    """
    Convenience decorator for retry with jitter.

    Usage:
        @retry_with_jitter(max_attempts=5, initial_delay=2.0)
        async def my_function():
            ...
    """
    config = RetryConfig(
        max_attempts=max_attempts,
        initial_delay_seconds=initial_delay,
        max_delay_seconds=max_delay,
        jitter_factor=jitter_factor,
        retryable_exceptions=retryable_exceptions,
    )
    retry = RetryWithJitter(config=config)
    return retry
