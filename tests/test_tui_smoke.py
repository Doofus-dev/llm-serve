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
        from tui.app import ModelTree, StatusPanel, ConfigPanel

        tree = app.query_one(ModelTree)
        model_slugs = [
            c.data[1]
            for c in tree.root.children
            if c.data and c.data[0] == "model"
        ]
        assert "qwen25" in model_slugs, model_slugs
        assert "qwen36" in model_slugs, model_slugs
        print("tree models:", model_slugs)

        # Tree shows display names, not slugs
        labels = [str(c.label) for c in tree.root.children if c.data and c.data[0] == "model"]
        assert any("qwen" in label.lower() for label in labels), labels
        print("tree labels:", labels)

        status = app.query_one(StatusPanel)
        rendered = status.render()
        assert "NOT RUNNING" in rendered or "RUNNING" in rendered
        print("status OK:", "RUNNING" if "RUNNING" in rendered.replace("NOT RUNNING", "") else "not running")

        cfg = app.query_one(ConfigPanel)
        cfg.selected = "qwen25"
        cfg.registry = app.registry
        text = cfg.render()
        assert "gpu_layers" in text and "32768" in text
        assert "display" in text
        print("config panel OK")

        app._refresh_pid()
        app._poll_gpu()
        await pilot.pause(0.3)
        r2 = app.query_one(StatusPanel).render()
        assert "GPU" in r2 and ("AMD" in r2 or "RAM" in r2)
        print("gpu section OK")

        await pilot.press("r")
        await pilot.pause(0.2)
        print("smoke test PASSED")

asyncio.run(main())
