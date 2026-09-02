"""Headless smoke test for the TUI app using Textual's Pilot."""
import asyncio
import io
import sys
from pathlib import Path

from rich.console import Console

sys.path.insert(0, str(Path(__file__).parent.parent))

from tui.app import LLMServeApp


def render_text(renderable) -> str:
    output = io.StringIO()
    console = Console(file=output, width=100, color_system=None)
    console.print(renderable)
    return output.getvalue()


async def main():
    app = LLMServeApp()
    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause(0.5)
        from tui.app import ConfigPanel, ModelNav, StatusPanel

        nav = app.query_one(ModelNav)
        model_slugs = [
            data[1]
            for data in nav._option_data.values()
            if data and data[0] == "model"
        ]
        assert "qwen38-27b-bartowski" in model_slugs, model_slugs
        print("nav models:", model_slugs)

        # Cards show friendly display names, not internal slugs.
        labels = [
            str(nav.get_option(option_id).prompt)
            for option_id, data in nav._option_data.items()
            if data and data[0] == "model"
        ]
        assert any("qwen 3.8" in label.lower() for label in labels), labels
        assert any("Quant" in label and "ctx:" in label for label in labels), labels
        print("nav cards:", labels)

        status = app.query_one(StatusPanel)
        rendered = render_text(status.render())
        assert "NOT RUNNING" in rendered or "RUNNING" in rendered
        print("status OK:", "RUNNING" if "RUNNING" in rendered.replace("NOT RUNNING", "") else "not running")

        cfg = app.query_one(ConfigPanel)
        cfg.selected = "qwen38-27b-bartowski"
        cfg.registry = app.registry
        cfg.preset_store = app.preset_store
        text = render_text(cfg.render())
        assert "Model" in text
        assert "Qwen 3.8" in text
        assert "Preset" in text
        print("config panel OK")

        app._refresh_pid()
        app._poll_gpu()
        await pilot.pause(0.3)
        r2 = render_text(app.query_one(StatusPanel).render())
        assert "GPU" in r2 and ("AMD" in r2 or "RAM" in r2)
        print("gpu section OK")

        await pilot.press("r")
        await pilot.pause(0.2)
        print("smoke test PASSED")

asyncio.run(main())
