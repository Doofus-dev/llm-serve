"""Footer binding groups and selection-aware visibility."""

from __future__ import annotations

import unittest

from tui.app import (
    AliasNav,
    LLMServeApp,
    LogPanel,
    ModelNav,
    build_app_bindings,
    selection_supports_action,
)


def shown_actions(app: LLMServeApp) -> set[str]:
    return {
        binding.action
        for _key, binding, _enabled, _tooltip in app.screen.active_bindings.values()
        if binding.show
    }


class FooterBindingTests(unittest.TestCase):
    def test_selection_kinds_that_enable_model_actions(self) -> None:
        self.assertTrue(selection_supports_action("edit", "model"))
        self.assertTrue(selection_supports_action("edit", "preset"))
        self.assertFalse(selection_supports_action("edit", "alias"))
        self.assertFalse(selection_supports_action("launch", None))
        self.assertTrue(selection_supports_action("launch", "alias"))
        self.assertTrue(selection_supports_action("pick_quant", "model", models_section=True))
        self.assertTrue(selection_supports_action("pick_quant", "preset", models_section=True))
        self.assertFalse(selection_supports_action("pick_quant", "model", models_section=False))
        self.assertFalse(selection_supports_action("pick_quant", "alias", models_section=True))
        self.assertFalse(selection_supports_action("pick_quant", None))
        self.assertFalse(selection_supports_action("launch", "model", log_section=True))
        self.assertFalse(selection_supports_action("new", "preset", log_section=True))
        self.assertFalse(selection_supports_action("delete", "alias", log_section=True))
        self.assertTrue(selection_supports_action("toggle_remote", None, log_section=True))

    def test_bindings_are_in_workflow_order(self) -> None:
        bindings = [b for b in build_app_bindings() if b.show]
        self.assertEqual(
            [b.action for b in bindings],
            [
                "launch",
                "stop",
                "edit",
                "pick_quant",
                "new",
                "delete",
                "toggle_remote",
                "cycle_log_verbosity",
                "toggle_log_source",
                "open_hub",
                "change_theme",
                "help",
                "quit",
            ],
        )
        self.assertTrue(all(b.group is None for b in bindings))

    def test_dynamic_labels(self) -> None:
        labels = {
            b.action: b.description
            for b in build_app_bindings(remote_on=True, log_label="DEBUG", info_label="Info ON")
        }
        self.assertEqual(labels["toggle_remote"], "Remote ON")
        self.assertEqual(labels["cycle_log_verbosity"], "Log DEBUG")
        self.assertEqual(labels["toggle_log_source"], "Info ON")


class FooterVisibilityTests(unittest.IsolatedAsyncioTestCase):
    async def test_model_actions_follow_selection(self) -> None:
        app = LLMServeApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause(0.3)
            app.query_one(ModelNav).focus()
            await pilot.pause(0.1)
            on_model = shown_actions(app)
            self.assertTrue({"launch", "edit", "pick_quant"} <= on_model)
            self.assertTrue({"stop", "new", "delete", "open_hub", "quit"} <= on_model)

            aliases = app.query_one(AliasNav)
            aliases.focus()
            await pilot.pause(0.1)
            on_alias = shown_actions(app)
            self.assertIn("launch", on_alias)
            self.assertNotIn("pick_quant", on_alias)
            self.assertNotIn("edit", on_alias)
            self.assertTrue({"stop", "new", "delete", "open_hub", "quit"} <= on_alias)

    async def test_log_panel_hides_selection_actions(self) -> None:
        app = LLMServeApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause(0.3)
            logs = app.query_one(LogPanel)
            self.assertFalse(logs.auto_scroll)
            logs.focus()
            await pilot.pause(0.1)
            on_logs = shown_actions(app)
            self.assertFalse({"launch", "edit", "pick_quant", "new", "delete"} & on_logs)
            self.assertTrue({"stop", "toggle_remote", "cycle_log_verbosity", "toggle_log_source"} <= on_logs)
