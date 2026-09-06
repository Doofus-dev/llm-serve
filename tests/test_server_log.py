"""Tests for llama.cpp / llm-serve log parsing and the event feed."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tui.data.server_log import (
    LogAggregator,
    LogTailer,
    format_elapsed,
    is_unformatted_event,
    parse_line,
    render_event,
    slice_to_session,
)

STARTUP = """\
── 2026-09-04 16:13:31 launch: qwen36-27b-bartowski (PID 172011) ──
0.00.045.655 W DEPRECATED: --defrag-thold is deprecated and no longer necessary to specify
0.00.045.715 I cmn  common_param: common_params_print_info: verbosity = 3 (adjust with the `-lv N` CLI arg)
0.00.045.909 W srv  llama_server: -----------------
0.00.045.910 W srv  llama_server: CORS is set to allow all origins ('*') and no API key is set
0.00.045.911 W srv  llama_server: this can be a security risk (cross-origin attacks)
0.00.045.911 W srv  llama_server: more info: https://github.com/ggml-org/llama.cpp/pull/25655
0.00.045.911 W srv  llama_server: -----------------
0.00.047.063 I srv    load_model: loading model '/home/doofus/software/llm-serve/models/bartowski/Qwen_Qwen3.6-27B-Q2_K.gguf'
0.00.499.379 W model has unused tensor blk.64.attn_norm.weight (size = 20480 bytes) -- ignoring
0.00.499.383 W model has unused tensor blk.64.post_attention_norm.weight (size = 20480 bytes) -- ignoring
0.00.499.387 W model has unused tensor blk.64.attn_q.weight (size = 66846720 bytes) -- ignoring
0.03.424.343 I cmn          init: llama threadpool init, n_threads = 8
0.03.499.061 I srv    load_model: initializing, n_slots = 1, n_ctx_slot = 65024, kv_unified = 'false'
0.03.501.436 I srv          init: chat template supports preserving reasoning, consider enabling it via --reasoning-preserve
0.03.501.447 I srv  llama_server: model loaded
0.03.501.451 I srv  llama_server: listening on http://0.0.0.0:8081
"""

LCP_REQUEST = """\
22.49.807.577 I slot get_availabl: id  0 | task -1 | selected slot by LCP similarity, f_sim_best = 0.998 (> 0.100 thold), f_keep = 1.000
22.49.808.066 I slot launch_slot_: id  0 | task 16291 | processing task, is_child = 0
22.50.655.442 I slot print_timing: id  0 | task 16291 | prompt eval time =     105.14 ms /    44 tokens (    2.39 ms per token,   418.51 tokens per second)
22.50.655.444 I slot print_timing: id  0 | task 16291 |        eval time =     742.23 ms /    37 tokens (   20.62 ms per token,    48.50 tokens per second)
22.50.655.444 I slot print_timing: id  0 | task 16291 |       total time =     847.36 ms /    81 tokens
22.50.655.444 I slot print_timing: id  0 | task 16291 |    graphs reused =      10568
22.50.656.087 I slot      release: id  0 | task 16291 | stop processing: n_tokens = 27589, truncated = 0
"""

LRU_REQUEST = """\
23.07.875.089 I slot get_availabl: id  0 | task -1 | selected slot by LRU, t_last = 43950067412
23.08.087.179 I slot launch_slot_: id  0 | task 17122 | processing task, is_child = 0
23.08.361.224 I slot print_timing: id  0 | task 17122 | prompt eval time =     215.84 ms /   202 tokens (    1.07 ms per token,   935.87 tokens per second)
23.08.361.226 I slot print_timing: id  0 | task 17122 |        eval time =      58.19 ms /     4 tokens (   19.40 ms per token,    51.56 tokens per second)
23.08.361.227 I slot print_timing: id  0 | task 17122 |    graphs reused =      11252
23.08.361.254 I slot      release: id  0 | task 17122 | stop processing: n_tokens = 400, truncated = 0
"""

TRUNCATED_REQUEST = """\
1.00.000.000 I slot get_availabl: id  0 | task -1 | selected slot by LRU, t_last = 1
1.00.100.000 I slot launch_slot_: id  0 | task 9 | processing task, is_child = 0
1.00.200.000 I slot print_timing: id  0 | task 9 | prompt eval time =     100.00 ms /   100 tokens (    1.00 ms per token,  1000.00 tokens per second)
1.00.300.000 I slot print_timing: id  0 | task 9 |        eval time =     200.00 ms /    10 tokens (   20.00 ms per token,    50.00 tokens per second)
1.00.400.000 I slot      release: id  0 | task 9 | stop processing: n_tokens = 65024, truncated = 1
"""

NOISY_PROGRESS = """\
0.12.467.200 I slot get_availabl: id  0 | task -1 | selected slot by LRU, t_last = -1
0.12.467.592 I slot launch_slot_: id  0 | task 36 | processing task, is_child = 0
0.16.267.873 I slot print_timing: id  0 | task 36 | prompt processing, n_tokens =   6144, progress = 0.17, t =   3.80 s / 1616.73 tokens per second
0.17.580.266 I slot print_timing: id  0 | task 36 | prompt processing, n_tokens =   8192, progress = 0.23, t =   5.11 s / 1602.30 tokens per second
6.08.385.187 I slot print_timing: id  0 | task 36 | n_gen =    144, tg =  47.66 t/s, tg_3s =  47.99 t/s
"""

INCOMPLETE_THEN_DONE = NOISY_PROGRESS + """\
0.20.000.000 I slot print_timing: id  0 | task 36 | prompt eval time =   26058.49 ms / 36381 tokens (    0.72 ms per token,  1396.13 tokens per second)
0.20.000.001 I slot print_timing: id  0 | task 36 |        eval time =    6340.14 ms /   310 tokens (   20.52 ms per token,    48.74 tokens per second)
0.20.000.002 I slot      release: id  0 | task 36 | stop processing: n_tokens = 36690, truncated = 0
"""

CANCEL = """\
23.14.683.060 W srv          stop: cancel task, id_task = 11429
23.15.160.700 I srv    operator(): operator(): cleaning up before exit...
"""


def events_from(text: str):
    agg = LogAggregator()
    found = agg.feed_text(text)
    found.extend(agg.flush())
    return found, agg


def request_event(events):
    found = [event for event in events if event.kind == "request"]
    if len(found) != 1:
        raise AssertionError(f"expected 1 request, got {len(found)} from {[e.kind for e in events]}")
    return found[0]


class ParseLineTests(unittest.TestCase):
    def test_launch_marker(self) -> None:
        parsed = parse_line(
            "── 2026-09-04 16:13:31 launch: qwen36-27b-bartowski (PID 172011) ──"
        )
        self.assertIsNotNone(parsed)
        self.assertTrue(parsed.is_launch)
        self.assertEqual(parsed.launch_model, "qwen36-27b-bartowski")
        self.assertEqual(parsed.launch_pid, 172011)

    def test_padded_slot_function(self) -> None:
        parsed = parse_line(
            "22.49.677.985 I slot      release: id  0 | task 16198 | stop processing: n_tokens = 27509, truncated = 0"
        )
        self.assertEqual(parsed.category, "slot")
        self.assertEqual(parsed.function, "release")

    def test_elapsed_minutes_seconds(self) -> None:
        parsed = parse_line(
            "3.52.149.070 I slot get_availabl: id  0 | task -1 | selected slot by LRU, t_last = -1"
        )
        self.assertAlmostEqual(parsed.elapsed_s, 3 * 60 + 52.149070, places=5)


class StartupCollapseTests(unittest.TestCase):
    def test_launch_cors_unused_ready(self) -> None:
        events, _ = events_from(STARTUP)
        kinds = [event.kind for event in events]
        self.assertEqual(
            kinds,
            [
                "launch",
                "hint",
                "cors",
                "loading",
                "unused_tensors",
                "hint",
                "hint",
                "ready",
            ],
        )
        unused = events[4]
        self.assertEqual(unused.unused_count, 3)
        self.assertIn("unused tensor", unused.message)
        cors = events[2]
        self.assertIn("CORS", cors.message)
        self.assertEqual(len(cors.raw_lines), 5)
        ready = events[-1]
        self.assertEqual(ready.n_slots, 1)
        self.assertEqual(ready.n_ctx_slot, 65024)
        self.assertIn("0.0.0.0:8081", ready.message)
        loading = events[3]
        self.assertEqual(loading.message, "Qwen_Qwen3.6-27B-Q2_K.gguf")
        self.assertTrue(all(event.kind != "info" for event in events))


class RequestRecapTests(unittest.TestCase):
    def test_lcp_cache_reuse(self) -> None:
        events, _ = events_from(LCP_REQUEST)
        event = request_event(events)
        self.assertEqual(event.kind, "request")
        self.assertEqual(event.cache_mode, "lcp")
        self.assertAlmostEqual(event.f_sim, 0.998)
        self.assertEqual(event.prompt_tokens, 44)
        self.assertAlmostEqual(event.prompt_tps, 418.51)
        self.assertEqual(event.gen_tokens, 37)
        self.assertAlmostEqual(event.gen_tps, 48.50)
        self.assertEqual(event.context_tokens, 27589)
        self.assertFalse(event.truncated)
        self.assertIn("reused cache 99.8%", event.message)
        self.assertIn("read 44 tok", event.message)
        self.assertIn("wrote 37 tok", event.message)
        self.assertIn("no truncation", event.detail)
        self.assertIsNone(event.client_addr)

    def test_request_recap_includes_client_ip(self) -> None:
        lines = LCP_REQUEST.splitlines()
        done = (
            "22.50.655.445 I srv  log_server_r: done request: "
            "POST /v1/chat/completions 100.86.55.45 200"
        )
        text = "\n".join(lines[:-1] + [done, lines[-1]])
        events, _ = events_from(text)
        event = request_event(events)
        self.assertEqual(event.client_addr, "100.86.55.45")
        self.assertIn("from 100.86.55.45", event.message)
        self.assertFalse(any(item.kind == "http" for item in events))
        rendered = render_event(event)
        self.assertIn("100.86.55.45", rendered)
        self.assertIn("bold yellow", rendered)

    def test_lru_fresh_prompt(self) -> None:
        events, _ = events_from(LRU_REQUEST)
        event = request_event(events)
        self.assertEqual(event.cache_mode, "lru")
        self.assertIn("fresh prompt", event.message)
        self.assertNotIn("reused cache", event.message)
        self.assertEqual(event.prompt_tokens, 202)
        self.assertEqual(event.gen_tokens, 4)

    def test_truncated_flagged(self) -> None:
        events, _ = events_from(TRUNCATED_REQUEST)
        event = request_event(events)
        self.assertTrue(event.truncated)
        self.assertIn("context overflow", event.detail)
        rendered = render_event(event)
        self.assertIn("bold red", rendered)

    def test_progress_ticks_are_shown_collapsed(self) -> None:
        events, agg = events_from(NOISY_PROGRESS)
        self.assertIn(36, agg._requests)
        self.assertFalse(any(event.kind == "request" for event in events))
        progress = [event for event in events if event.kind == "prefill"]
        self.assertEqual(len(progress), 1)
        self.assertEqual(progress[0].count, 2)
        self.assertEqual(progress[0].prompt_tokens, 8192)
        self.assertIn("8192 tok", progress[0].message.replace(",", ""))
        gen = [event for event in events if event.kind == "generate"]
        self.assertEqual(len(gen), 1)
        self.assertEqual(gen[0].gen_tokens, 144)
        rendered = render_event(progress[0])
        self.assertIn("prefill", rendered)
        self.assertIn("bold yellow", rendered)

    def test_incomplete_request_flushes_on_release(self) -> None:
        agg = LogAggregator()
        first = agg.feed_text(NOISY_PROGRESS)
        self.assertTrue(all(event.kind != "request" for event in first))
        done = agg.feed_text(
            "\n".join(INCOMPLETE_THEN_DONE.splitlines()[len(NOISY_PROGRESS.splitlines()) :])
        )
        event = request_event(done)
        self.assertEqual(event.prompt_tokens, 36381)
        self.assertEqual(event.gen_tokens, 310)
        self.assertEqual(agg._requests, {})

    def test_cancel_and_shutdown(self) -> None:
        events, _ = events_from(CANCEL)
        self.assertEqual([event.kind for event in events], ["cancel", "shutdown"])
        self.assertIn("cancelled", events[0].message)
        self.assertEqual(events[1].message, "server stopped")

    def test_http_done_request(self) -> None:
        events, _ = events_from(
            "1.02.000.000 I srv  log_server_r: done request: POST /v1/chat/completions 100.86.55.45 200\n"
        )
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].kind, "http")
        self.assertEqual(events[0].client_addr, "100.86.55.45")
        self.assertEqual(events[0].http_status, 200)
        self.assertIn("POST /v1/chat/completions", events[0].message)
        rendered = render_event(events[0])
        self.assertIn("100.86.55.45", rendered)
        self.assertIn("bold yellow", rendered)

    def test_local_slots_polls_are_http_noise(self) -> None:
        events, _ = events_from(
            "1.02.000.000 I srv  log_server_r: done request: GET /slots 127.0.0.1 200\n"
            "1.02.000.001 I srv  log_server_r: done request: GET /slots 127.0.0.1 503\n"
            "1.02.000.002 I srv  log_server_r: done request: GET /v1/props 127.0.0.1 404\n"
            "1.02.000.003 I srv  log_server_r: done request: GET /api/tags 100.86.55.45 404\n"
            "1.02.000.004 I srv  log_server_r: done request: POST /v1/chat/completions 127.0.0.1 200\n"
        )
        kinds = [event.kind for event in events]
        self.assertEqual(kinds, ["http", "http"])
        self.assertEqual(events[0].client_addr, "100.86.55.45")
        self.assertIn("GET /api/tags", events[0].message)
        self.assertEqual(events[1].client_addr, "127.0.0.1")
        self.assertIn("POST /v1/chat/completions", events[1].message)

    def test_debug_bodies_are_dropped(self) -> None:
        events, _ = events_from(
            '1.02.000.001 D srv  log_server_r: request:  {"messages":[{"role":"user","content":"hi"}]}\n'
            '1.02.000.002 D srv  log_server_r: response: {"choices":[]}\n'
        )
        self.assertEqual(events, [])

    def test_trace_slot_checks_do_not_clobber_lcp(self) -> None:
        events, _ = events_from(
            "22.49.807.500 I slot get_availabl: id  0 | task -1 |  - checking sim = 0.500 (10/20) > 0.100\n"
            + LCP_REQUEST
        )
        event = request_event(events)
        self.assertEqual(event.cache_mode, "lcp")
        self.assertAlmostEqual(event.f_sim, 0.998)

    def test_idle_collapses_then_request_keeps_both(self) -> None:
        idle = "0.01.469.096 I srv  update_slots: all slots are idle\n"
        events, _ = events_from(idle + idle + idle + LCP_REQUEST)
        idle_events = [event for event in events if event.kind == "idle"]
        self.assertEqual(len(idle_events), 1)
        self.assertEqual(idle_events[0].count, 3)
        self.assertEqual(idle_events[0].label, "idle")
        rendered = render_event(idle_events[0])
        self.assertIn("idle", rendered)
        self.assertIn("bold yellow", rendered)
        event = request_event(events)
        self.assertEqual(event.cache_mode, "lcp")

    def test_checkpoint_and_restore_are_formatted(self) -> None:
        events, _ = events_from(
            "0.50.870.576 I slot create_check: id  0 | task 162 | "
            "created context checkpoint 1 of 32 (pos_min = 13399, pos_max = 13399, "
            "n_tokens = 13400, size = 149.626 MiB)\n"
            "0.56.761.029 I slot create_check: id  0 | task 162 | "
            "created context checkpoint 2 of 32 (pos_min = 21744, pos_max = 21744, "
            "n_tokens = 21745, size = 149.626 MiB)\n"
            "3.10.507.219 I slot   operator(): id  0 | task 1084 | "
            "restored context checkpoint (pos_min = 21744, pos_max = 21744, "
            "n_tokens = 21745, n_past = 21745, size = 149.626 MiB)\n"
        )
        ckpts = [event for event in events if event.kind == "checkpoint"]
        self.assertEqual(len(ckpts), 2)
        self.assertEqual(ckpts[0].count, 2)
        self.assertEqual(ckpts[0].ckpt_index, 2)
        self.assertEqual(ckpts[0].context_tokens, 21745)
        self.assertTrue(ckpts[1].restored)
        self.assertIn("restored", ckpts[1].message)
        rendered = render_event(ckpts[0])
        self.assertIn("ckpt", rendered)
        self.assertIn("bold yellow", rendered)

    def test_cached_token_ticks_are_noise(self) -> None:
        events, _ = events_from(
            "0.42.109.971 I slot   operator(): id  0 | task 162 | "
            "cached n_tokens = 0, memory_seq_rm [0, end)\n"
            "0.43.376.384 I slot   operator(): id  0 | task 162 | "
            "cached n_tokens = 2048, memory_seq_rm [2048, end)\n"
        )
        self.assertEqual(events, [])

    def test_print_info_fields_are_not_collapsed(self) -> None:
        events, _ = events_from(
            "0.00.339.417 I print_info: file type   = Q2_K - Medium\n"
            "0.00.339.418 I print_info: file size   = 11.21 GiB (3.53 BPW)\n"
            "0.00.456.228 I print_info: model params          = 27.32 B\n"
        )
        self.assertEqual(len(events), 3)
        self.assertTrue(all(event.count == 1 for event in events))
        self.assertIn("file type", events[0].message)
        self.assertIn("file size", events[1].message)
        self.assertIn("model params", events[2].message)


class SessionSliceTests(unittest.TestCase):
    def test_keeps_from_last_launch(self) -> None:
        text = "old line\n" + STARTUP + LCP_REQUEST
        earlier = (
            "── 2026-09-04 13:40:10 launch: older-model (PID 1) ──\n"
            "0.00.001.000 I srv  llama_server: listening on http://127.0.0.1:8081\n"
        )
        sliced = slice_to_session(earlier + text)
        self.assertIn("qwen36-27b-bartowski", sliced)
        self.assertNotIn("older-model", sliced)

    def test_fallback_without_launch(self) -> None:
        lines = [f"{i} keep\n" for i in range(500)]
        sliced = slice_to_session("".join(lines), fallback_lines=10)
        self.assertEqual(sliced.count("\n"), 10)


class TailerTests(unittest.TestCase):
    def test_truncation_resets_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "llm-serve.log"
            path.write_text(STARTUP)
            tailer = LogTailer()
            first, reset = tailer.poll(path)
            self.assertTrue(reset)
            self.assertEqual(first[0].kind, "launch")
            self.assertEqual(first[0].model, "qwen36-27b-bartowski")

            path.write_text(
                "── 2026-09-04 17:00:00 launch: other-model (PID 99) ──\n"
                "0.00.100.000 I srv  llama_server: listening on http://127.0.0.1:9000\n"
            )
            second, reset = tailer.poll(path)
            self.assertTrue(reset)
            self.assertEqual(second[0].model, "other-model")
            self.assertTrue(any(event.kind == "ready" for event in second))

    def test_incremental_append(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "llm-serve.log"
            path.write_text(STARTUP)
            tailer = LogTailer()
            first, _ = tailer.poll(path)
            self.assertTrue(any(event.kind == "ready" for event in first))

            with path.open("a") as handle:
                handle.write(LCP_REQUEST)
            extra, reset = tailer.poll(path)
            self.assertFalse(reset)
            self.assertEqual(request_event(extra).kind, "request")
            self.assertEqual(request_event(extra).cache_mode, "lcp")


class RenderTests(unittest.TestCase):
    def test_elapsed_formatting(self) -> None:
        self.assertEqual(format_elapsed(3.2), "3s")
        self.assertEqual(format_elapsed(125), "2m")
        self.assertEqual(format_elapsed(7200), "2h")

    def test_show_raw_includes_original_line(self) -> None:
        events, _ = events_from(LCP_REQUEST)
        event = request_event(events)
        hidden = render_event(event, show_raw=False)
        shown = render_event(event, show_raw=True)
        self.assertNotIn("print_timing", hidden)
        self.assertIn("print_timing", shown)
        self.assertIn("reused cache", shown)

    def test_print_info_is_unformatted(self) -> None:
        events, _ = events_from(
            "0.00.339.417 I print_info: file type   = Q2_K - Medium\n"
            + LCP_REQUEST
        )
        leftover = [event for event in events if event.kind == "info"]
        self.assertTrue(leftover)
        self.assertTrue(all(is_unformatted_event(event) for event in leftover))
        self.assertFalse(is_unformatted_event(request_event(events)))


if __name__ == "__main__":
    unittest.main()
