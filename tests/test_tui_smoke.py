"""Headless smoke test for the TUI app using Textual's Pilot."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tui.app import LLMServeApp


async def main():
    app = LLMServeApp()
    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause(0.5)
        # Check the tree has models
        from tui.app import ModelTree, StatusPanel, ConfigPanel, LogPanel
        tree = app.query_one(ModelTree)
        labels = [str(c.label) for c in tree.root.children]
        assert labels == ["qwen25", "qwen36", "qwen36-deep", "Aliases"], labels
        print("tree models:", labels)

        status = app.query_one(StatusPanel)
        rendered = status.render()
        assert "NOT RUNNING" in rendered or "RUNNING" in rendered
        print("status OK:", "RUNNING" if "RUNNING" in rendered.replace("NOT RUNNING", "") else "not running")

        cfg = app.query_one(ConfigPanel)
        cfg.selected = "qwen25"
        cfg.registry = app.registry
        text = cfg.render()
        assert "gpu_layers" in text and "32768" in text
        print("config panel OK")

        # trigger refresh + gpu poll actions
        app._refresh_pid()
        app._poll_gpu()
        await pilot.pause(0.3)
        r2 = app.query_one(StatusPanel).render()
        assert "GPU" in r2 and ("AMD" in r2 or "RAM" in r2)
        print("gpu section OK")

        # press keys
        await pilot.press("r")
        await pilot.pause(0.2)
        print("smoke test PASSED")

asyncio.run(main())

