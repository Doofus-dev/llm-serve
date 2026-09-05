"""Regression tests for Hub callbacks after the screen is closed."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, PropertyMock, patch

from textual.css.query import NoMatches

from tui.data.models_json import Registry
from tui.data.settings import TUISettings
from tui.screens.hub import HubScreen


class HubCallbackTests(unittest.TestCase):
    def _make_screen(self, root: Path) -> HubScreen:
        return HubScreen(
            registry=Registry(),
            settings=TUISettings(),
            settings_path=root / "settings.json",
            models_json_path=root / "models.json",
            models_dir=root / "models",
        )

    def test_status_update_is_ignored_after_screen_is_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            screen = self._make_screen(Path(tmp))
            screen._set_status = Mock()

            screen._set_status_if_mounted("download complete")

            screen._set_status.assert_not_called()

    def test_status_update_survives_missing_status_widget(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            screen = self._make_screen(Path(tmp))
            screen.query_one = Mock(
                side_effect=NoMatches("No nodes match '#hub-status' on HubScreen()")
            )

            with patch.object(
                type(screen), "is_mounted", new_callable=PropertyMock, return_value=True
            ):
                screen._set_status_if_mounted("Registered foo.gguf")


if __name__ == "__main__":
    unittest.main()
