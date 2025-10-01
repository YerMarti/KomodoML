import inspect
from functools import wraps

def forward_methods(from_cls, to_cls, methods=None):
    """
    Copy methods from `from_cls` into `to_cls` dynamically.
    Preserves docstrings, signatures and also appends
    KomodoML-specific extra docs if declared in `__extra_docs__`.
    """
    extra_docs = getattr(to_cls, "__extra_docs__", {})

    for name, func in inspect.getmembers(from_cls, predicate=inspect.isfunction):
        if methods and name not in methods:
            continue
        if hasattr(to_cls, name):
            continue  # don't overwrite existing

        @wraps(func)
        def wrapper(self, *args, _name=name, **kwargs):
            return getattr(self.model, _name)(*args, **kwargs)

        # Merge docstrings
        base_doc = func.__doc__ or ""
        addon = extra_docs.get(name, "")
        if addon:
            wrapper.__doc__ = base_doc + "\n\n[KomodoML Extension]\n" + addon
        else:
            wrapper.__doc__ = base_doc

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
            forward_methods(wrapped_cls, cls)
        return cls