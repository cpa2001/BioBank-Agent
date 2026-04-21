"""Auto-retry decorator with intelligent parameter mutation on failure.

This decorator provides self-healing for transient failures by intelligently
modifying parameters and retrying. Useful for resource-constrained operations.
"""

import logging
import time
from functools import wraps
from typing import Callable, Dict, Any, Optional

logger = logging.getLogger(__name__)


def retry_on_error(
    max_retries: int = 3,
    backoff_factor: float = 2.0,
    initial_delay: float = 1.0,
) -> Callable:
    """Decorator that retries a skill on error, modifying parameters intelligently.
    
    When a skill fails, this decorator:
    1. Reduces computational complexity (fewer folds, smaller samples, fewer items)
    2. Waits with exponential backoff before retrying
    3. Returns an error dict after exhausting retries
    
    Parameters
    ----------
    max_retries : int
        Maximum number of retry attempts (default: 3)
    backoff_factor : float
        Exponential backoff multiplier (default: 2.0)
        Delay = initial_delay * (backoff_factor ** attempt_number)
    initial_delay : float
        Initial delay in seconds before first retry (default: 1.0)
    
    Returns
    -------
    Callable
        Decorator function
    
    Examples
    --------
    ```python
    @retry_on_error(max_retries=3, backoff_factor=2.0)
    def my_skill(icd10_code: str, n_folds: int = 5, top_n: int = 20, *, ctx=None) -> dict:
        # First attempt uses n_folds=5, top_n=20
        # If error: retry with n_folds=4, top_n=10
        # If error: retry with n_folds=3, top_n=10
        ...
    ```
    """
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            ctx = kwargs.get("ctx")
            func_name = func.__name__
            last_error = None
            
            for attempt in range(1, max_retries + 1):
                try:
                    logger.debug(
                        f"Executing {func_name} (attempt {attempt}/{max_retries}) "
                        f"with kwargs: {kwargs}"
                    )
                    return func(*args, **kwargs)
                    
                except Exception as e:
                    last_error = e
                    error_msg = str(e)
                    
                    # If this is the last attempt, return error
                    if attempt == max_retries:
                        logger.error(
                            f"Failed after {max_retries} retries: {error_msg}"
                        )
                        return {
                            "error": f"Failed after {max_retries} retries: {error_msg}",
                            "skill": func_name,
                            "last_attempt": attempt,
                            "original_error": error_msg,
                        }
                    
                    # Log the failure and prepare for retry
                    logger.warning(
                        f"Attempt {attempt}/{max_retries} failed: {error_msg}. "
                        f"Retrying with modified parameters..."
                    )
                    
                    # Intelligently modify parameters to reduce complexity
                    _modify_parameters_for_retry(kwargs, attempt)
                    
                    # Exponential backoff before retry
                    delay = initial_delay * (backoff_factor ** (attempt - 1))
                    logger.debug(f"Waiting {delay:.1f}s before retry...")
                    time.sleep(delay)
            
            # Fallback (should not reach here, but just in case)
            return {
                "error": f"Unexpected state: exhausted retries",
                "skill": func_name,
            }
        
        return wrapper
    
    return decorator


def _modify_parameters_for_retry(kwargs: Dict[str, Any], attempt_number: int) -> None:
    """Modify kwargs to reduce computational complexity for retry.
    
    Intelligently reduces parameters that commonly cause failures:
    - n_folds: Cross-validation folds (reduce by 1, min 2)
    - n_repeats: Repetitions (reduce by 1, min 1)
    - sample_size: Sample size (divide by 2, min 10)
    - top_n: Number of top items (cap at 10)
    - n_cases_min: Minimum case threshold (reduce by 10%)
    
    Parameters
    ----------
    kwargs : dict
        Function keyword arguments (modified in-place)
    attempt_number : int
        Which retry attempt this is (1-indexed)
    """
    
    # Reduce cross-validation folds
    if "n_folds" in kwargs:
        old_val = kwargs["n_folds"]
        kwargs["n_folds"] = max(2, kwargs["n_folds"] - 1)
        logger.debug(
            f"n_folds: {old_val} → {kwargs['n_folds']}"
        )
    
    # Reduce repetitions
    if "n_repeats" in kwargs:
        old_val = kwargs["n_repeats"]
        kwargs["n_repeats"] = max(1, kwargs["n_repeats"] - 1)
        logger.debug(
            f"n_repeats: {old_val} → {kwargs['n_repeats']}"
        )
    
    # Reduce sample size
    if "sample_size" in kwargs:
        old_val = kwargs["sample_size"]
        kwargs["sample_size"] = max(10, kwargs["sample_size"] // 2)
        logger.debug(
            f"sample_size: {old_val} → {kwargs['sample_size']}"
        )
    
    # Cap top_n at reasonable value
    if "top_n" in kwargs:
        old_val = kwargs["top_n"]
        kwargs["top_n"] = min(old_val, 10)
        logger.debug(
            f"top_n: {old_val} → {kwargs['top_n']}"
        )
    
    # Reduce minimum case threshold if present
    if "n_cases_min" in kwargs:
        old_val = kwargs["n_cases_min"]
        kwargs["n_cases_min"] = max(50, int(kwargs["n_cases_min"] * 0.9))
        logger.debug(
            f"n_cases_min: {old_val} → {kwargs['n_cases_min']}"
        )
    
    # For XGBoost/LightGBM specific parameters
    if "max_depth" in kwargs:
        old_val = kwargs["max_depth"]
        kwargs["max_depth"] = max(2, kwargs["max_depth"] - 1)
        logger.debug(
            f"max_depth: {old_val} → {kwargs['max_depth']}"
        )
    
    if "n_estimators" in kwargs:
        old_val = kwargs["n_estimators"]
        kwargs["n_estimators"] = max(10, kwargs["n_estimators"] // 2)
        logger.debug(
            f"n_estimators: {old_val} → {kwargs['n_estimators']}"
        )


def should_retry_on_error(error: Exception) -> bool:
    """Check if an error is retryable (transient vs permanent).
    
    Some errors indicate transient issues that retry might fix:
    - Memory/resource errors (likely to succeed with fewer resources)
    - Timeout errors (likely to succeed with more time)
    - Temporary failures
    
    Permanent errors should not be retried:
    - ValueError (invalid input)
    - KeyError (missing data)
    - AttributeError (API mismatch)
    
    Parameters
    ----------
    error : Exception
        The error to check
    
    Returns
    -------
    bool
        True if retry is likely to help
    """
    
    error_type = type(error).__name__
    error_str = str(error).lower()
    
    # Permanent error types that should NOT retry
    PERMANENT = {
        "ValueError",
        "KeyError",
        "AttributeError",
        "TypeError",
        "NameError",
        "ImportError",
        "NotImplementedError",
    }
    
    # Check error message keywords for permanent conditions FIRST (highest priority)
    # These keywords indicate permanent issues even if error type seems retryable
    PERMANENT_KEYWORDS = [
        "not found",
        "no such",
        "invalid",
        "unsupported",
        "deprecated",
        "undefined",
        "unrecognized",
    ]
    
    if any(kw in error_str for kw in PERMANENT_KEYWORDS):
        return False
    
    # Check error type for permanent errors
    if error_type in PERMANENT:
        return False
    
    # Transient errors that should retry
    RETRYABLE = {
        "MemoryError",
        "RuntimeError",
        "IOError",
        "OSError",
        "TimeoutError",
        "BrokenPipeError",
    }
    
    # Check error type for retryable errors
    if error_type in RETRYABLE:
        return True
    
    # Check error message keywords for retryable conditions
    RETRYABLE_KEYWORDS = [
        "memory",
        "resource",
        "timeout",
        "busy",
        "try again",
        "temporary",
        "transient",
    ]
    
    if any(kw in error_str for kw in RETRYABLE_KEYWORDS):
        return True
    
    # Default: retry on unknown errors (conservative approach)
    return True
