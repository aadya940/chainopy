from functools import lru_cache, wraps


def _cache(class_method):
    """
    Method for caching results while the Transition-Matrix
    doesn't change.

    NOTE: Not to be called directly.

    """

    def decorator(method):
        @lru_cache()
        @wraps(method)
        def cached_wrapper(instance, *args, **kwargs):
            if "tpm" in method.__code__.co_varnames:
                key = (tuple(instance.tpm),) + args + tuple(kwargs.items())
            else:
                key = args + tuple(kwargs.items())
            return method(instance, *args, **kwargs)

        return cached_wrapper

    return decorator
