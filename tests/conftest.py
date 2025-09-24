"""Test configuration and lightweight fallbacks for optional dependencies."""

from __future__ import annotations

import importlib.util
import sys
import types


def _install_rich_stub() -> None:
    if "rich" in sys.modules:
        return

    if importlib.util.find_spec("rich") is not None:
        return

    rich_module = types.ModuleType("rich")

    # Minimal box namespace with the attributes gtop expects
    box_module = types.ModuleType("rich.box")
    box_module.ROUNDED = "ROUNDED"
    box_module.SIMPLE = "SIMPLE"
    rich_module.box = box_module

    class _DummyConsole:
        def __init__(self, *args, **kwargs):
            self._is_stderr = kwargs.get("stderr", False)

        def print(self, *args, **kwargs):  # pragma: no cover - passthrough helper
            # Tests do not rely on console output, so swallow anything printed.
            return None

    console_module = types.ModuleType("rich.console")
    console_module.Console = _DummyConsole
    rich_module.console = console_module

    class _DummyTable:
        def __init__(self, *args, **kwargs):
            self.columns = []
            self.rows = []

        def add_column(self, *args, **kwargs):
            self.columns.append(args)

        def add_row(self, *args, **kwargs):
            self.rows.append(args)

    table_module = types.ModuleType("rich.table")
    table_module.Table = _DummyTable
    rich_module.table = table_module

    sys.modules["rich"] = rich_module
    sys.modules["rich.box"] = box_module
    sys.modules["rich.console"] = console_module
    sys.modules["rich.table"] = table_module


_install_rich_stub()
