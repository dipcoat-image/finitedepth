"""Caching function."""

import functools

__all__ = [
    "attrcache",
]


def attrcache(attrname: str):
    """Cache the method into attribute.

    Method must not have any argument.

    Examples:
        >>> from dipcoatimage.finitedepth.cache import attrcache
        >>> class Foo:
        ...     @attrcache("_bar")
        ...     def bar(self):
        ...         print("spam")
        ...         return 1
        >>> f = Foo()
        >>> f.bar()
        spam
        1
        >>> f.bar()
        1
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(self):
            if not hasattr(self, attrname):
                setattr(self, attrname, func(self))
            return getattr(self, attrname)

        return wrapper

    return decorator
