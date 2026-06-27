"""Environment (scope) management for MatPy."""

from __future__ import annotations
from typing import Any


class Environment:
    """Lexical scope with parent chain for variable lookup."""

    def __init__(self, parent: Environment | None = None, name: str = "global"):
        self.parent = parent
        self.name = name
        self._vars: dict[str, Any] = {}
        self._globals: set[str] = set()
        self._persistent: dict[str, Any] = {}

    def get(self, name: str) -> Any:
        if name in self._vars:
            return self._vars[name]
        if name in self._globals and self.parent is not None:
            return self._get_global(name)
        if self.parent is not None:
            return self.parent.get(name)
        raise NameError(f"Undefined variable '{name}'")

    def set(self, name: str, value: Any):
        if name in self._globals:
            self._set_global(name, value)
        else:
            self._vars[name] = value

    def define_global(self, name: str):
        self._globals.add(name)

    def define_persistent(self, name: str, default: Any = None):
        if name not in self._persistent:
            self._persistent[name] = default
        self._vars[name] = self._persistent[name]

    def has(self, name: str) -> bool:
        if name in self._vars:
            return True
        if name in self._globals and self.parent is not None:
            return self._has_global(name)
        if self.parent is not None:
            return self.parent.has(name)
        return False

    def _get_global(self, name: str) -> Any:
        env = self
        while env.parent is not None:
            env = env.parent
        if name not in env._vars:
            raise NameError(f"Undefined global variable '{name}'")
        return env._vars[name]

    def _set_global(self, name: str, value: Any):
        env = self
        while env.parent is not None:
            env = env.parent
        env._vars[name] = value

    def _has_global(self, name: str) -> bool:
        env = self
        while env.parent is not None:
            env = env.parent
        return name in env._vars

    def local_vars(self) -> dict[str, Any]:
        return dict(self._vars)

    def all_vars(self) -> dict[str, Any]:
        result = {}
        if self.parent is not None:
            result.update(self.parent.all_vars())
        result.update(self._vars)
        return result

    def child(self, name: str = "local") -> Environment:
        return Environment(parent=self, name=name)
