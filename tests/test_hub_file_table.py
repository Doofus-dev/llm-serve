"""Hub tables show a Disk column for repos and files."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.support import Harness
from textual.widgets import DataTable, Select

from tui.data.capabilities import resolve_capabilities
from tui.data.hf import HubFile, HubRepo
from tui.data.models_json import ModelConfig, Registry, downloaded_repo_ids
from tui.data.settings import TUISettings
from tui.screens.hub import DownloadProfileDialog, HubScreen
from tui.widgets.nav import EYEBALL


def _repo(repo_id: str, **kwargs) -> HubRepo:
    author = repo_id.split("/", 1)[0]
    return HubRepo(id=repo_id, author=author, downloads=0, likes=0, **kwargs)


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
    def setUp(self) -> None:
        self.harness = Harness()

    def tearDown(self) -> None:
        self.harness.cleanup()

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
            app = self.harness.app()
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
                        _repo(
                            "bartowski/Model-GGUF",
                            pipeline_tag="image-text-to-text",
                        ),
                        _repo("bartowski/Other-GGUF"),
                    ]
                    screen._set_mode_repos()

                    table = screen.query_one("#hub-table", DataTable)
                    labels = [str(col.label) for col in table.columns.values()]
                    self.assertEqual(labels[0], "Disk")
                    self.assertEqual(labels[1], "Repo / author")
                    self.assertEqual(labels[2], "Caps")
                    self.assertIn("vision", str(table.get_row_at(0)[2]).lower())
                    self.assertIn("●", str(table.get_row_at(0)[0]))
                    self.assertEqual(str(table.get_row_at(1)[0]), "—")
                    self.assertIsInstance(screen.query_one("#min_context", Select), Select)

                    table.move_cursor(row=0)
                    screen._handle_select()
                    self.assertEqual(screen.mode, "files")
                    self.assertEqual(screen.selected_repo.id, "bartowski/Model-GGUF")

    async def test_repo_table_shows_mtp_from_siblings_without_opening(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            app = self.harness.app()
            screen = HubScreen(
                registry=Registry(),
                settings=TUISettings(),
                settings_path=root / "settings.json",
                models_json_path=root / "models.json",
                models_dir=root / "models",
            )
            with (
                patch.object(HubScreen, "_request_repos", lambda *a, **k: None),
                patch.object(HubScreen, "_load_files", lambda *a, **k: None),
            ):
                async with app.run_test(size=(120, 45)) as pilot:
                    app.push_screen(screen)
                    await pilot.pause()

                    screen.repos = [
                        _repo(
                            "unsloth/Qwen3.8-Flash-Next-GGUF",
                            filenames=(
                                "Qwen3.8-Flash-Next-Q8_0.gguf",
                                "mmproj-F16.gguf",
                                "mtp-Qwen3.8-Flash-Next-Q8_0.gguf",
                            ),
                        ),
                    ]
                    screen._set_mode_repos()

                    table = screen.query_one("#hub-table", DataTable)
                    caps = str(table.get_row_at(0)[2]).lower()
                    self.assertIn("mtp", caps)
                    self.assertIn("vision", caps)

                    screen._set_mode_files(screen.repos[0])
                    self.assertTrue(screen._file_caps.mtp)
                    self.assertTrue(screen._file_caps.vision)
                    file_table = screen.query_one("#hub-table", DataTable)
                    labels = [
                        str(file_table.get_row_at(i)[1]).lower()
                        for i in range(file_table.row_count)
                    ]
                    self.assertTrue(any("mtp" in cell for cell in labels))
                    self.assertTrue(any("vision" in cell for cell in labels))

    async def test_file_table_labels_addons_and_hides_imatrix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            app = self.harness.app()
            screen = HubScreen(
                registry=Registry(),
                settings=TUISettings(),
                settings_path=root / "settings.json",
                models_json_path=root / "models.json",
                models_dir=root / "models",
            )
            with patch.object(HubScreen, "_request_repos", lambda *a, **k: None):
                async with app.run_test(size=(120, 45)) as pilot:
                    app.push_screen(screen)
                    await pilot.pause()

                    screen.mode = "files"
                    screen.selected_repo = _repo("unsloth/Qwen3.8-Flash-Next-GGUF")
                    screen.files = [
                        HubFile(path="Qwen3.8-Flash-Next-Q8_0.gguf", size=2000),
                        HubFile(path="mtp-Qwen3.8-Flash-Next-Q8_0.gguf", size=100),
                        HubFile(path="mmproj-F16.gguf", size=50),
                        HubFile(path="imatrix_unsloth.gguf", size=10),
                    ]
                    screen._file_caps = resolve_capabilities(
                        files=screen.files,
                        repo_id=screen.selected_repo.id,
                    )
                    screen._render_file_table()

                    table = screen.query_one("#hub-table", DataTable)
                    labels = [
                        str(table.get_row_at(i)[1])
                        for i in range(table.row_count)
                    ]
                    joined = " ".join(labels).lower()
                    self.assertIn("mtp", joined)
                    self.assertIn("vision", joined)
                    self.assertNotIn("imatrix", joined)
                    self.assertEqual(table.row_count, 3)

    async def test_disk_column_and_select_behavior(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models_dir = root / "models"
            author_dir = models_dir / "bartowski"
            author_dir.mkdir(parents=True)
            (author_dir / "Model-Q4_K_M.gguf").write_bytes(b"x" * 1000)

            app = self.harness.app()
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
                    self.assertEqual(labels[4], "Act. VRAM")
                    self.assertEqual(labels[5], f"Act. VRAM {EYEBALL}")
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

    async def test_first_model_dialog_skips_clone_select(self) -> None:
        """Empty models.json must not mount Select(value=False) (Textual crash)."""
        app = self.harness.app()
        dialog = DownloadProfileDialog(Registry(), "Qwen 3.8")
        async with app.run_test(size=(80, 30)) as pilot:
            app.push_screen(dialog)
            await pilot.pause()
            self.assertEqual(len(dialog.query("#clone_from")), 0)
            self.assertEqual(dialog.query_one("#name").value, "Qwen 3.8")


if __name__ == "__main__":
    unittest.main()
