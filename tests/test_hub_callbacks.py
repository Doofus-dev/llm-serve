"""Regression tests for Hub callbacks after the screen is closed."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from tui.data.models_json import Registry
from tui.data.settings import TUISettings
from tui.screens.hub import HubScreen


class HubCallbackTests(unittest.TestCase):
    def test_status_update_is_ignored_after_screen_is_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            screen = HubScreen(
                registry=Registry(),
                settings=TUISettings(),
                settings_path=root / "settings.json",
                models_json_path=root / "models.json",
                models_dir=root / "models",
            )
            screen._set_status = Mock()

            screen._set_status_if_mounted("download complete")

            screen._set_status.assert_not_called()


if __name__ == "__main__":
    unittest.main()
