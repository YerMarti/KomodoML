import inspect
from functools import wraps

def _forward_methods(from_cls, to_cls, methods=None):
    """
    Copy methods from `from_cls` into `to_cls` dynamically. Preserves docstrings and signatures.
    Forward calls to a matching attribute (default: self.model) if it exists.
    """
    for name, func in inspect.getmembers(from_cls, predicate=inspect.isfunction):
        if methods and name not in methods:
            continue
        if hasattr(to_cls, name):
            continue  # don’t overwrite existing

        @wraps(func)
        def wrapper(self, *args, _name=name, **kwargs):
            target = getattr(self, "model", None)
            if target is None:
                return None
            return getattr(target, _name)(*args, **kwargs)

        setattr(to_cls, name, wrapper)


class Wrapper(type):
    """
    Metaclass to wrap methods from an underlying model class into the wrapper class.
    The underlying model class should be specified in the `wrapped_cls` attribute.
    """

    def __new__(mcs, name, bases, namespace, **kwargs):
        cls = super().__new__(mcs, name, bases, namespace)
        wrapped_cls = namespace.get("wrapped_cls")

        if wrapped_cls is not None:
            _forward_methods(wrapped_cls, cls)

            # Merge docstrings
            cls.__doc__ = (cls.__doc__ or "") + "\n\nWrapped class docstring:\n" + (wrapped_cls.__doc__ or "")

            # Attach signature from wrapped class __init__ for IDEs
            try:
                cls.__signature__ = inspect.signature(wrapped_cls.__init__)
            except (TypeError, ValueError):
                pass

        return cls