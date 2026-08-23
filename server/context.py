"""FastAPI 路由共享上下文。"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from server.builders import AppBuilders
from core.config import PluginServiceSettings
from runtime.engine import PluginRuntime
from runtime.sync import sync_runtime_with_config


@dataclass(slots=True)
class AppContext:
    runtime: PluginRuntime
    builders: AppBuilders
    _config_mutation_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)

    async def apply_config_mutation(self, mutator: Callable[[PluginServiceSettings], PluginServiceSettings]) -> dict[str, Any]:
        async with self._config_mutation_lock:
            configured_settings = PluginServiceSettings.from_storage()
            next_settings = mutator(configured_settings)
            next_settings.save_to_storage()
            reload_state = await sync_runtime_with_config(self.runtime, next_settings)
            return self.with_mutation_payload(reload_state)

    async def sync_configured_settings(self) -> dict[str, Any]:
        async with self._config_mutation_lock:
            configured_settings = PluginServiceSettings.from_storage()
            reload_state = await sync_runtime_with_config(self.runtime, configured_settings)
            return self.with_mutation_payload(reload_state)

    def with_mutation_payload(self, reload_state: dict[str, Any]) -> dict[str, Any]:
        return {
            **reload_state,
            "overview": self.builders.build_overview(),
            "plugins": self.builders.build_plugin_payload(),
            "settings": self.builders.build_settings_payload(),
        }
