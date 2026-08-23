import asyncio

import server.context as context_module
from core.config import PluginServiceSettings
from server.context import AppContext


class DummyBuilders:
    def build_overview(self) -> dict:
        return {}

    def build_plugin_payload(self) -> list:
        return []

    def build_settings_payload(self) -> dict:
        return {}


def test_app_context_serializes_config_reload(monkeypatch) -> None:
    settings = PluginServiceSettings()
    monkeypatch.setattr(PluginServiceSettings, "from_storage", classmethod(lambda cls: settings))
    monkeypatch.setattr(PluginServiceSettings, "save_to_storage", lambda self: None)
    active_reloads = 0
    max_active_reloads = 0

    async def fake_sync(runtime, configured_settings):
        nonlocal active_reloads, max_active_reloads
        active_reloads += 1
        max_active_reloads = max(max_active_reloads, active_reloads)
        await asyncio.sleep(0)
        active_reloads -= 1
        return {}

    monkeypatch.setattr(context_module, "sync_runtime_with_config", fake_sync)
    ctx = AppContext(object(), DummyBuilders())

    async def run_mutations() -> None:
        await asyncio.gather(
            ctx.apply_config_mutation(lambda current: current),
            ctx.apply_config_mutation(lambda current: current),
        )

    asyncio.run(run_mutations())
    assert max_active_reloads == 1
