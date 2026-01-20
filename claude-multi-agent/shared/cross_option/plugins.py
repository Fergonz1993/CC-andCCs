"""
Plugin System for Extensibility (adv-cross-007)

Provides a plugin architecture that allows extending the coordination
system with custom functionality. Plugins can hook into various
lifecycle events and add new capabilities.
"""

import importlib
import importlib.util
import inspect
import json
import sys
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Type


class PluginHook(Enum):
    """Lifecycle hooks that plugins can implement."""
    # Initialization
    ON_INIT = "on_init"
    ON_SHUTDOWN = "on_shutdown"

    # Task lifecycle
    BEFORE_TASK_CREATE = "before_task_create"
    AFTER_TASK_CREATE = "after_task_create"
    BEFORE_TASK_CLAIM = "before_task_claim"
    AFTER_TASK_CLAIM = "after_task_claim"
    BEFORE_TASK_START = "before_task_start"
    AFTER_TASK_START = "after_task_start"
    BEFORE_TASK_COMPLETE = "before_task_complete"
    AFTER_TASK_COMPLETE = "after_task_complete"
    BEFORE_TASK_FAIL = "before_task_fail"
    AFTER_TASK_FAIL = "after_task_fail"

    # Agent lifecycle
    ON_AGENT_REGISTER = "on_agent_register"
    ON_AGENT_HEARTBEAT = "on_agent_heartbeat"
    ON_AGENT_DISCONNECT = "on_agent_disconnect"

    # Discovery
    ON_DISCOVERY_ADD = "on_discovery_add"

    # Sync
    BEFORE_SYNC = "before_sync"
    AFTER_SYNC = "after_sync"
    ON_SYNC_CONFLICT = "on_sync_conflict"

    # Migration
    BEFORE_MIGRATION = "before_migration"
    AFTER_MIGRATION = "after_migration"


@dataclass
class PluginConfig:
    """Configuration for a plugin."""
    name: str
    version: str = "1.0.0"
    description: str = ""
    author: str = ""
    enabled: bool = True
    priority: int = 100  # Lower = higher priority
    config: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PluginConfig":
        return cls(
            name=data["name"],
            version=data.get("version", "1.0.0"),
            description=data.get("description", ""),
            author=data.get("author", ""),
            enabled=data.get("enabled", True),
            priority=data.get("priority", 100),
            config=data.get("config", {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "enabled": self.enabled,
            "priority": self.priority,
            "config": self.config,
        }


class Plugin(ABC):
    """
    Base class for all plugins.

    To create a plugin:
    1. Subclass this class
    2. Implement the required methods
    3. Register hook handlers using the @hook decorator or register_hook method
    """

    def __init__(self, config: Optional[PluginConfig] = None):
        """Initialize the plugin."""
        self._config = config or PluginConfig(name=self.__class__.__name__)
        self._hooks: Dict[PluginHook, List[Callable]] = {}
        self._initialized = False

    @property
    def name(self) -> str:
        """Get the plugin name."""
        return self._config.name

    @property
    def version(self) -> str:
        """Get the plugin version."""
        return self._config.version

    @property
    def config(self) -> Dict[str, Any]:
        """Get the plugin configuration."""
        return self._config.config

    @property
    def enabled(self) -> bool:
        """Check if the plugin is enabled."""
        return self._config.enabled

    def enable(self) -> None:
        """Enable the plugin."""
        self._config.enabled = True

    def disable(self) -> None:
        """Disable the plugin."""
        self._config.enabled = False

    def register_hook(self, hook: PluginHook, handler: Callable) -> None:
        """Register a handler for a hook."""
        if hook not in self._hooks:
            self._hooks[hook] = []
        self._hooks[hook].append(handler)

    def get_hooks(self, hook: PluginHook) -> List[Callable]:
        """Get all handlers for a hook."""
        return self._hooks.get(hook, [])

    def has_hook(self, hook: PluginHook) -> bool:
        """Check if the plugin has handlers for a hook."""
        return hook in self._hooks and len(self._hooks[hook]) > 0

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the plugin. Called when the plugin is loaded."""
        pass

    def shutdown(self) -> None:
        """Shutdown the plugin. Called when the plugin is unloaded."""
        pass

    def get_info(self) -> Dict[str, Any]:
        """Get plugin information."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self._config.description,
            "author": self._config.author,
            "enabled": self.enabled,
            "hooks": [h.value for h in self._hooks.keys()],
        }


def hook(hook_type: PluginHook):
    """
    Decorator to register a method as a hook handler.

    Usage:
        class MyPlugin(Plugin):
            @hook(PluginHook.AFTER_TASK_CREATE)
            def on_task_created(self, task):
                print(f"Task created: {task.id}")
    """
    def decorator(func: Callable) -> Callable:
        if not hasattr(func, "_plugin_hooks"):
            func._plugin_hooks = []
        func._plugin_hooks.append(hook_type)
        return func
    return decorator


class PluginManager:
    """
    Manages the lifecycle of plugins.

    Features:
    - Plugin discovery and loading
    - Hook invocation
    - Plugin ordering by priority
    - Enable/disable plugins
    """

    def __init__(self, plugin_dirs: Optional[List[str]] = None):
        """
        Initialize the plugin manager.

        Args:
            plugin_dirs: Directories to search for plugins
        """
        self.plugin_dirs = [Path(d) for d in (plugin_dirs or [])]
        self._plugins: Dict[str, Plugin] = {}
        self._hook_cache: Dict[PluginHook, List[tuple]] = {}
        self._lock = threading.Lock()
        self._initialized = False

    def register(self, plugin: Plugin) -> None:
        """Register a plugin."""
        with self._lock:
            if plugin.name in self._plugins:
                raise ValueError(f"Plugin {plugin.name} is already registered")

            self._plugins[plugin.name] = plugin
            self._invalidate_cache()

    def unregister(self, name: str) -> Optional[Plugin]:
        """Unregister a plugin by name."""
        with self._lock:
            if name in self._plugins:
                plugin = self._plugins.pop(name)
                plugin.shutdown()
                self._invalidate_cache()
                return plugin
            return None

    def get(self, name: str) -> Optional[Plugin]:
        """Get a plugin by name."""
        return self._plugins.get(name)

    def get_all(self) -> List[Plugin]:
        """Get all registered plugins."""
        return list(self._plugins.values())

    def get_enabled(self) -> List[Plugin]:
        """Get all enabled plugins."""
        return [p for p in self._plugins.values() if p.enabled]

    def enable(self, name: str) -> bool:
        """Enable a plugin."""
        plugin = self._plugins.get(name)
        if plugin:
            plugin.enable()
            self._invalidate_cache()
            return True
        return False

    def disable(self, name: str) -> bool:
        """Disable a plugin."""
        plugin = self._plugins.get(name)
        if plugin:
            plugin.disable()
            self._invalidate_cache()
            return True
        return False

    def initialize_all(self) -> None:
        """Initialize all registered plugins."""
        with self._lock:
            for plugin in self._get_ordered_plugins():
                if plugin.enabled and not plugin._initialized:
                    try:
                        # Register decorated hooks
                        self._register_decorated_hooks(plugin)
                        plugin.initialize()
                        plugin._initialized = True
                    except Exception as e:
                        print(f"Failed to initialize plugin {plugin.name}: {e}")

            self._initialized = True

    def shutdown_all(self) -> None:
        """Shutdown all plugins."""
        with self._lock:
            for plugin in reversed(self._get_ordered_plugins()):
                if plugin._initialized:
                    try:
                        plugin.shutdown()
                        plugin._initialized = False
                    except Exception as e:
                        print(f"Failed to shutdown plugin {plugin.name}: {e}")

            self._initialized = False

    def invoke_hook(
        self,
        hook: PluginHook,
        *args,
        stop_on_error: bool = False,
        **kwargs,
    ) -> List[Any]:
        """
        Invoke all handlers for a hook.

        Args:
            hook: The hook to invoke
            *args: Arguments to pass to handlers
            stop_on_error: Whether to stop on the first error
            **kwargs: Keyword arguments to pass to handlers

        Returns:
            List of results from all handlers
        """
        results = []
        handlers = self._get_hook_handlers(hook)

        for plugin, handler in handlers:
            if not plugin.enabled:
                continue

            try:
                result = handler(*args, **kwargs)
                results.append(result)
            except Exception as e:
                if stop_on_error:
                    raise
                print(f"Error in hook {hook.value} from plugin {plugin.name}: {e}")

        return results

    def invoke_hook_chain(
        self,
        hook: PluginHook,
        value: Any,
        **kwargs,
    ) -> Any:
        """
        Invoke handlers as a chain, passing the result of each to the next.

        Args:
            hook: The hook to invoke
            value: Initial value to pass through the chain
            **kwargs: Keyword arguments to pass to handlers

        Returns:
            Final value after passing through all handlers
        """
        handlers = self._get_hook_handlers(hook)

        for plugin, handler in handlers:
            if not plugin.enabled:
                continue

            try:
                value = handler(value, **kwargs)
            except Exception as e:
                print(f"Error in hook chain {hook.value} from plugin {plugin.name}: {e}")

        return value

    def discover_plugins(self) -> List[str]:
        """
        Discover plugins in the configured directories.

        Returns:
            List of discovered plugin module paths
        """
        discovered = []

        for plugin_dir in self.plugin_dirs:
            if not plugin_dir.exists():
                continue

            # Look for Python files
            for py_file in plugin_dir.glob("*.py"):
                if py_file.name.startswith("_"):
                    continue
                discovered.append(str(py_file))

            # Look for plugin packages (directories with __init__.py)
            for subdir in plugin_dir.iterdir():
                if subdir.is_dir() and (subdir / "__init__.py").exists():
                    discovered.append(str(subdir / "__init__.py"))

        return discovered

    def load_plugin_from_file(self, filepath: str) -> Optional[Plugin]:
        """
        Load a plugin from a Python file.

        Args:
            filepath: Path to the plugin file

        Returns:
            Loaded plugin instance or None
        """
        path = Path(filepath)
        if not path.exists():
            return None

        try:
            # Load the module
            spec = importlib.util.spec_from_file_location(
                path.stem,
                path,
            )
            if spec is None or spec.loader is None:
                return None

            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)

            # Find Plugin subclasses
            for name, obj in inspect.getmembers(module):
                if (
                    inspect.isclass(obj)
                    and issubclass(obj, Plugin)
                    and obj is not Plugin
                ):
                    plugin = obj()
                    self.register(plugin)
                    return plugin

        except Exception as e:
            print(f"Failed to load plugin from {filepath}: {e}")

        return None

    def load_plugins_from_directory(self, directory: str) -> List[Plugin]:
        """Load all plugins from a directory."""
        loaded = []
        dir_path = Path(directory)

        if not dir_path.exists():
            return loaded

        for py_file in dir_path.glob("*.py"):
            if py_file.name.startswith("_"):
                continue

            plugin = self.load_plugin_from_file(str(py_file))
            if plugin:
                loaded.append(plugin)

        return loaded

    def load_config(self, config_file: str) -> None:
        """Load plugin configuration from a JSON file."""
        with open(config_file) as f:
            config = json.load(f)

        for plugin_config in config.get("plugins", []):
            name = plugin_config.get("name")
            if name and name in self._plugins:
                plugin = self._plugins[name]
                plugin._config = PluginConfig.from_dict(plugin_config)

    def save_config(self, config_file: str) -> None:
        """Save plugin configuration to a JSON file."""
        config = {
            "plugins": [
                plugin._config.to_dict()
                for plugin in self._plugins.values()
            ]
        }

        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)

    def _get_ordered_plugins(self) -> List[Plugin]:
        """Get plugins ordered by priority."""
        return sorted(
            self._plugins.values(),
            key=lambda p: p._config.priority,
        )

    def _register_decorated_hooks(self, plugin: Plugin) -> None:
        """Register hooks from decorated methods."""
        for name, method in inspect.getmembers(plugin, inspect.ismethod):
            if hasattr(method, "_plugin_hooks"):
                for hook_type in method._plugin_hooks:
                    plugin.register_hook(hook_type, method)

    def _get_hook_handlers(self, hook: PluginHook) -> List[tuple]:
        """Get all handlers for a hook, cached for performance."""
        with self._lock:
            if hook not in self._hook_cache:
                handlers = []
                for plugin in self._get_ordered_plugins():
                    for handler in plugin.get_hooks(hook):
                        handlers.append((plugin, handler))
                self._hook_cache[hook] = handlers

            return self._hook_cache[hook]

    def _invalidate_cache(self) -> None:
        """Invalidate the hook cache."""
        self._hook_cache.clear()


# Global plugin manager instance
_global_manager: Optional[PluginManager] = None
_global_lock = threading.Lock()


def get_plugin_manager() -> PluginManager:
    """Get or create the global plugin manager."""
    global _global_manager
    with _global_lock:
        if _global_manager is None:
            _global_manager = PluginManager()
        return _global_manager


def register_plugin(plugin: Plugin) -> None:
    """Register a plugin with the global manager."""
    get_plugin_manager().register(plugin)


def invoke_hook(hook: PluginHook, *args, **kwargs) -> List[Any]:
    """Invoke a hook on the global manager."""
    return get_plugin_manager().invoke_hook(hook, *args, **kwargs)


# Example plugins

class LoggingPlugin(Plugin):
    """Example plugin that logs all task events."""

    def __init__(self):
        super().__init__(PluginConfig(
            name="LoggingPlugin",
            version="1.0.0",
            description="Logs all task lifecycle events",
        ))

    def initialize(self) -> None:
        """Set up logging hooks."""
        self.register_hook(PluginHook.AFTER_TASK_CREATE, self._on_task_create)
        self.register_hook(PluginHook.AFTER_TASK_COMPLETE, self._on_task_complete)
        self.register_hook(PluginHook.AFTER_TASK_FAIL, self._on_task_fail)

    def _on_task_create(self, task: Any) -> None:
        print(f"[LoggingPlugin] Task created: {getattr(task, 'id', task)}")

    def _on_task_complete(self, task: Any) -> None:
        print(f"[LoggingPlugin] Task completed: {getattr(task, 'id', task)}")

    def _on_task_fail(self, task: Any, error: str) -> None:
        print(f"[LoggingPlugin] Task failed: {getattr(task, 'id', task)} - {error}")


class MetricsPlugin(Plugin):
    """Example plugin that collects metrics."""

    def __init__(self):
        super().__init__(PluginConfig(
            name="MetricsPlugin",
            version="1.0.0",
            description="Collects task metrics",
        ))
        self.task_counts = {"created": 0, "completed": 0, "failed": 0}

    def initialize(self) -> None:
        """Set up metric collection hooks."""
        self.register_hook(PluginHook.AFTER_TASK_CREATE, self._count_create)
        self.register_hook(PluginHook.AFTER_TASK_COMPLETE, self._count_complete)
        self.register_hook(PluginHook.AFTER_TASK_FAIL, self._count_fail)

    def _count_create(self, task: Any) -> None:
        self.task_counts["created"] += 1

    def _count_complete(self, task: Any) -> None:
        self.task_counts["completed"] += 1

    def _count_fail(self, task: Any, error: str) -> None:
        self.task_counts["failed"] += 1

    def get_counts(self) -> Dict[str, int]:
        return dict(self.task_counts)
