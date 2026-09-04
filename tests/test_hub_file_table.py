"""Hub tables show a Disk column for repos and files."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from textual.widgets import DataTable

from tui.app import LLMServeApp
from tui.data.hf import HubFile, HubRepo
from tui.data.models_json import ModelConfig, Registry, downloaded_repo_ids
from tui.data.settings import TUISettings
from tui.screens.hub import DownloadProfileDialog, HubScreen


def _repo(repo_id: str) -> HubRepo:
    author = repo_id.split("/", 1)[0]
    return HubRepo(id=repo_id, author=author, downloads=0, likes=0)


class DownloadedRepoIdTests(unittest.TestCase):
    def test_repo_is_on_disk_when_any_quant_file_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            models_dir = Path(tmp)
            gguf = models_dir / "bartowski" / "Model-Q4_K_M.gguf"
            gguf.parent.mkdir(parents=True)
            gguf.write_bytes(b"gguf")

            reg = Registry(
                models={
                    "model": ModelConfig(
                        name="model",
                        params={
                            "file": "bartowski/Model-Q4_K_M.gguf",
                            "source": {"repo": "bartowski/Model-GGUF"},
                            "quants": {
                                "Q4_K_M": {
                                    "filename": "Model-Q4_K_M.gguf",
                                    "file": "bartowski/Model-Q4_K_M.gguf",
                                },
                            },
                        },
                    ),
                    "other": ModelConfig(
                        name="other",
                        params={
                            "file": "bartowski/Other-Q8_0.gguf",
                            "source": {"repo": "bartowski/Other-GGUF"},
                            "quants": {
                                "Q8_0": {
                                    "filename": "Other-Q8_0.gguf",
                                    "file": "bartowski/Other-Q8_0.gguf",
                                },
                            },
                        },
                    ),
                }
            )

            self.assertEqual(
                downloaded_repo_ids(reg, models_dir),
                {"bartowski/Model-GGUF"},
            )


class HubFileTableTests(unittest.IsolatedAsyncioTestCase):
    async def test_repo_table_marks_downloaded_repos(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models_dir = root / "models"
            gguf = models_dir / "bartowski" / "Model-Q4_K_M.gguf"
            gguf.parent.mkdir(parents=True)
            gguf.write_bytes(b"gguf")

            registry = Registry(
                models={
                    "model": ModelConfig(
                        name="model",
                        params={
                            "file": "bartowski/Model-Q4_K_M.gguf",
                            "source": {"repo": "bartowski/Model-GGUF"},
                            "quants": {
                                "Q4_K_M": {
                                    "filename": "Model-Q4_K_M.gguf",
                                    "file": "bartowski/Model-Q4_K_M.gguf",
                                },
                            },
                        },
                    )
                }
            )
            app = LLMServeApp()
            screen = HubScreen(
                registry=registry,
                settings=TUISettings(),
                settings_path=root / "settings.json",
                models_json_path=root / "models.json",
                models_dir=models_dir,
            )
            with (
                patch.object(HubScreen, "_request_repos", lambda *a, **k: None),
                patch.object(HubScreen, "_load_files", lambda *a, **k: None),
            ):
                async with app.run_test(size=(120, 45)) as pilot:
                    app.push_screen(screen)
                    await pilot.pause()

                    screen.repos = [
                        _repo("bartowski/Model-GGUF"),
                        _repo("bartowski/Other-GGUF"),
                    ]
                    screen._set_mode_repos()

                    table = screen.query_one("#hub-table", DataTable)
                    labels = [str(col.label) for col in table.columns.values()]
                    self.assertEqual(labels[0], "Disk")
                    self.assertIn("●", str(table.get_row_at(0)[0]))
                    self.assertEqual(str(table.get_row_at(1)[0]), "—")

                    table.move_cursor(row=0)
                    screen._handle_select()
                    self.assertEqual(screen.mode, "files")
                    self.assertEqual(screen.selected_repo.id, "bartowski/Model-GGUF")

    async def test_disk_column_and_select_behavior(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models_dir = root / "models"
            author_dir = models_dir / "bartowski"
            author_dir.mkdir(parents=True)
            (author_dir / "Model-Q4_K_M.gguf").write_bytes(b"gguf")

            app = LLMServeApp()
            screen = HubScreen(
                registry=Registry(),
                settings=TUISettings(),
                settings_path=root / "settings.json",
                models_json_path=root / "models.json",
                models_dir=models_dir,
            )
            with patch.object(HubScreen, "_request_repos", lambda *a, **k: None):
                async with app.run_test(size=(120, 45)) as pilot:
                    app.push_screen(screen)
                    await pilot.pause()

                    screen.mode = "files"
                    screen.selected_repo = _repo("bartowski/Model-GGUF")
                    screen.files = [
                        HubFile(path="Model-Q8_0.gguf", size=2000),
                        HubFile(path="Model-Q4_K_M.gguf", size=1000),
                    ]
                    screen._render_file_table()

                    table = screen.query_one("#hub-table", DataTable)
                    labels = [str(col.label) for col in table.columns.values()]
                    self.assertEqual(labels[0], "Disk")
                    self.assertEqual(str(table.get_row_at(0)[0]), "—")
                    self.assertIn("●", str(table.get_row_at(1)[0]))

                    table.move_cursor(row=1)
                    with (
                        patch.object(app, "start_model_download") as start,
                        patch.object(screen, "notify") as notify,
                    ):
                        screen._handle_select()
                    start.assert_not_called()
                    notify.assert_called_once()
                    self.assertIn("already on disk", notify.call_args.args[0])

                    table.move_cursor(row=0)
                    with patch.object(app, "push_screen") as push:
                        screen._handle_select()
                    push.assert_called_once()
                    self.assertIsInstance(push.call_args.args[0], DownloadProfileDialog)


if __name__ == "__main__":
    unittest.main()
