"""Open/Closed generic factory registry for pluggable ML components."""

from typing import Callable, Dict, Generic, List, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """Lightweight registry decoupling component consumers from concrete classes."""

    def __init__(self, name: str) -> None:
        self._name = name
        self._items: Dict[str, Callable[..., T]] = {}

    def register(self, *keys: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
        """Decorator registering a class or builder function under one or more names."""
        if not keys:
            raise ValueError(f"Registry '{self._name}' requires at least one key.")

        def decorator(target: Callable[..., T]) -> Callable[..., T]:
            for key in keys:
                norm_key = key.strip().lower()
                if norm_key in self._items:
                    raise KeyError(
                        f"Duplicate key '{norm_key}' in registry '{self._name}'."
                    )
                self._items[norm_key] = target
            return target

        return decorator

    def build(self, key: str, *args, **kwargs) -> T:
        norm_key = key.strip().lower()
        if norm_key not in self._items:
            available = ", ".join(sorted(self._items.keys()))
            raise KeyError(
                f"Unknown '{norm_key}' in registry '{self._name}'. Available: [{available}]"
            )
        return self._items[norm_key](*args, **kwargs)

    def available(self) -> List[str]:
        return sorted(self._items.keys())


STABILIZER_REGISTRY: Registry = Registry("stabilizer")
ANCHOR_REGISTRY: Registry = Registry("anchor")
LOSS_REGISTRY: Registry = Registry("loss")
