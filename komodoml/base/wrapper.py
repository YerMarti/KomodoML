from functools import wraps

def forward_methods(from_cls, to_cls, methods=None):
    """
    Copy methods from `from_cls` into `to_cls` dynamically.
    Preserves docstrings and signatures using functools.wraps.
    """
    for name, func in inspect.getmembers(from_cls, predicate=inspect.isfunction):
        if methods and name not in methods:
            continue
        if hasattr(to_cls, name):
            continue  # don't overwrite existing

        @wraps(func)
        def wrapper(self, *args, _name=name, **kwargs):
            return getattr(self.model, _name)(*args, **kwargs)

        setattr(to_cls, name, wrapper)


class Wrapper(type):
    def __new__(mcs, name, bases, namespace, **kwargs):
        cls = super().__new__(mcs, name, bases, namespace)
        wrapped_cls = namespace.get("wrapped_cls")
        if wrapped_cls is not None:
            forward_methods(wrapped_cls, cls)
        return cls
