import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

import app


class AppTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app.app)
        app.openrouter_client.cache_clear()
        self.conversation_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.conversation_directory.cleanup)
        original_conversation_file = app.CONVERSATION_FILE
        app.CONVERSATION_FILE = Path(self.conversation_directory.name) / "conversation.jsonl"
        self.addCleanup(setattr, app, "CONVERSATION_FILE", original_conversation_file)
        with app.PI_JOBS_LOCK:
            app.PI_JOBS.clear()
        with app.PI_SESSION_LOCKS_GUARD:
            app.PI_SESSION_LOCKS.clear()
        with app.PI_PROCESSES_LOCK:
            app.PI_PROCESSES.clear()

    def test_index(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("KINETIQ", response.text)

    def test_configured_client_ip_blocks_other_devices(self):
        with patch("app.ALLOWED_CLIENT_IPS", {"100.72.153.36"}):
            response = self.client.get("/")

        self.assertEqual(response.status_code, 403)

    def test_chat_rejects_empty_message(self):
        response = self.client.post("/api/chat", json={"message": "   "})
        self.assertEqual(response.status_code, 400)

    @patch("app.wake_word_model")
    def test_wake_word_stream_uses_dedicated_model(self, model_factory):
        model_factory.return_value.predict.return_value = {"hey_jarvis": 0.8}

        with self.client.websocket_connect("/api/wake-word") as websocket:
            self.assertEqual(websocket.receive_json(), {"ready": True})
            websocket.send_bytes(bytes(1280 * 2))
            detection = websocket.receive_json()

        self.assertTrue(detection["detected"])
        self.assertEqual(detection["score"], 0.8)
        model_factory.return_value.predict.assert_called_once()
        self.assertEqual(model_factory.return_value.reset.call_count, 2)

    @patch("app.transcribe_with_nemotron", return_value="Jarvis")
    def test_transcription_uses_nemotron(self, transcribe):
        response = self.client.post(
            "/api/transcribe",
            files={"audio": ("speech.wav", b"audio", "audio/wav")},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"text": "Jarvis"})
        transcribe.assert_called_once()

    @patch("app.synthesize_with_kokoro", return_value=b"RIFF-audio")
    def test_synthesis_uses_kokoro(self, synthesize):
        response = self.client.post("/api/synthesize", json={"text": "Hello"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "audio/wav")
        self.assertEqual(response.content, b"RIFF-audio")
        synthesize.assert_called_once_with("Hello")

    @patch("app.openrouter_client")
    def test_chat_uses_openrouter_client(self, client_factory):
        response_object = Mock()
        response_object.choices = [Mock(message=Mock(content="Hello.", tool_calls=None))]
        client_factory.return_value.chat.completions.create.return_value = response_object

        response = self.client.post("/api/chat", json={"message": "Hello"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["answer"], "Hello.")
        self.assertIn("conversation_id", response.json())
        request = client_factory.return_value.chat.completions.create.call_args.kwargs
        self.assertEqual(request["tools"][0]["function"]["name"], "ask_pi")

    def test_silent_marker_is_removed_after_spoken_text(self):
        response = "I’ll ask him and let you know what he says.\n\n__SILENT__"
        self.assertEqual(
            app.strip_silent_markers(response),
            "I’ll ask him and let you know what he says.",
        )

    @patch("app.openrouter_client")
    def test_chat_streams_openrouter_deltas(self, client_factory):
        first_delta = Mock(content="Hello", tool_calls=None)
        second_delta = Mock(content=" there.", tool_calls=None)
        client_factory.return_value.chat.completions.create.return_value = [
            Mock(choices=[Mock(delta=first_delta)]),
            Mock(choices=[Mock(delta=second_delta)]),
        ]

        response = self.client.post("/api/chat/stream", json={"message": "Hello", "history": []})

        self.assertEqual(response.status_code, 200)
        events = [json.loads(line) for line in response.text.splitlines()]
        self.assertEqual("".join(item.get("text", "") for item in events), "Hello there.")
        self.assertEqual(events[-1]["type"], "done")
        records = [json.loads(line) for line in app.CONVERSATION_FILE.read_text().splitlines()]
        visible_messages = [record["message"] for record in records if record.get("visible")]
        self.assertEqual([message["role"] for message in visible_messages], ["user", "assistant"])
        self.assertEqual(visible_messages[-1]["content"], "Hello there.")
        request = client_factory.return_value.chat.completions.create.call_args.kwargs
        self.assertTrue(request["stream"])

    @patch("app.launch_pi_job")
    @patch("app.openrouter_client")
    def test_stream_does_not_repeat_pre_tool_acknowledgement(self, client_factory, launch_pi):
        acknowledgement = "I’ll inspect the project and give you a clear breakdown."
        function_delta = Mock()
        function_delta.name = "ask_pi"
        function_delta.arguments = (
            '{"prompt":"inspect","directory":"/tmp","session_action":"new",'
            '"session_name":"inspection"}'
        )
        tool_delta = Mock(index=0, id="tool-1", function=function_delta)
        first_stream = [
            Mock(choices=[Mock(delta=Mock(content=acknowledgement, tool_calls=None))]),
            Mock(choices=[Mock(delta=Mock(content=None, tool_calls=[tool_delta]))]),
        ]
        second_stream = [Mock(choices=[Mock(delta=Mock(content=acknowledgement, tool_calls=None))])]
        client_factory.return_value.chat.completions.create.side_effect = [first_stream, second_stream]
        launch_pi.return_value = {
            "status": "queued",
            "session_name": "inspection",
            "directory": "/tmp",
        }

        response = self.client.post("/api/chat/stream", json={"message": "Inspect this project"})

        events = [json.loads(line) for line in response.text.splitlines()]
        spoken_text = "".join(item.get("text", "") for item in events)
        self.assertEqual(spoken_text, acknowledgement)
        self.assertEqual(events[-1]["answer"], acknowledgement)
        records = [json.loads(line) for line in app.CONVERSATION_FILE.read_text().splitlines()]
        visible_messages = [record["message"] for record in records if record.get("visible")]
        self.assertEqual([message["role"] for message in visible_messages], ["user", "assistant"])
        self.assertEqual(visible_messages[-1]["content"], acknowledgement)

    def test_conversation_history_hides_legacy_tool_acknowledgements(self):
        conversation = self.client.post("/api/conversations", json={"title": "Time"}).json()
        conversation_id = conversation["conversation_id"]
        acknowledgement = "Let me check the current time."
        app.append_conversation_message(
            conversation_id,
            {"role": "assistant", "content": acknowledgement, "tool_calls": [{"id": "tool-1"}]},
            visible=True,
            source="test",
        )
        app.append_conversation_message(
            conversation_id,
            {"role": "assistant", "content": acknowledgement},
            visible=True,
            source="test",
        )

        messages = self.client.get(f"/api/conversations/{conversation_id}").json()["messages"]

        self.assertEqual(messages, [{"role": "assistant", "content": acknowledgement}])

    @patch("app.launch_pi_job")
    @patch("app.openrouter_client")
    def test_chat_launches_pi_in_background(self, client_factory, launch_pi):
        tool_function = Mock()
        tool_function.name = "ask_pi"
        tool_function.arguments = (
            '{"prompt":"inspect","directory":"/tmp","session_action":"new",'
            '"session_name":"inspection"}'
        )
        tool_call = Mock(id="tool-1", function=tool_function)
        tool_response = Mock()
        tool_response.choices = [Mock(message=Mock(content=None, tool_calls=[tool_call]))]
        final_response = Mock()
        final_response.choices = [Mock(message=Mock(content="silent", tool_calls=None))]
        client_factory.return_value.chat.completions.create.side_effect = [tool_response, final_response]
        launch_pi.return_value = {
            "job_id": "job-1",
            "status": "queued",
            "session_name": "inspection",
            "directory": "/tmp",
        }

        response = self.client.post("/api/chat", json={"message": "Investigate the issue"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["answer"],
            "Let me check that. I’ll let you know when I have something useful.",
        )
        self.assertNotIn("silent", response.json())
        self.assertIn("Started background Pi job", response.json()["history_content"])
        second_request = client_factory.return_value.chat.completions.create.call_args_list[1].kwargs
        self.assertEqual(second_request["messages"][-1]["role"], "tool")
        records = [json.loads(line) for line in app.CONVERSATION_FILE.read_text().splitlines()]
        stored_messages = [record["message"] for record in records if record.get("type") == "message"]
        self.assertTrue(any(message.get("tool_calls") for message in stored_messages))
        self.assertTrue(any(message.get("role") == "tool" for message in stored_messages))

    def test_conversations_can_be_created_and_switched(self):
        first = self.client.post("/api/conversations", json={"title": "First"}).json()
        second = self.client.post("/api/conversations", json={"title": "Second"}).json()
        app.append_conversation_message(
            first["conversation_id"],
            {"role": "user", "content": "Message in first"},
            visible=True,
            source="test",
        )

        listing = self.client.get("/api/conversations").json()["conversations"]
        loaded_first = self.client.get(f"/api/conversations/{first['conversation_id']}").json()
        loaded_second = self.client.get(f"/api/conversations/{second['conversation_id']}").json()

        self.assertEqual(len(listing), 2)
        self.assertEqual(loaded_first["messages"][0]["content"], "Message in first")
        self.assertEqual(loaded_second["messages"], [])

    def test_pi_job_launch_returns_before_work_finishes(self):
        started = threading.Event()
        release = threading.Event()

        def slow_pi(_arguments, _job_id):
            started.set()
            release.wait(timeout=1)
            return {"summary": "done", "directory": "/tmp", "session_name": "slow"}

        with patch("app.run_pi_tool", side_effect=slow_pi):
            launched = app.launch_pi_job({
                "prompt": "Work slowly",
                "directory": "/tmp",
                "session_action": "new",
                "session_name": "slow",
            })
            self.assertTrue(started.wait(timeout=0.5))
            self.assertIn(launched["status"], {"queued", "running"})
            with app.PI_JOBS_LOCK:
                job_id = next(iter(app.PI_JOBS))
                self.assertNotEqual(app.PI_JOBS[job_id]["status"], "completed")
            release.set()
            deadline = time.monotonic() + 1
            while time.monotonic() < deadline:
                with app.PI_JOBS_LOCK:
                    if app.PI_JOBS[job_id]["status"] == "completed":
                        break
                time.sleep(0.01)
            with app.PI_JOBS_LOCK:
                self.assertEqual(app.PI_JOBS[job_id]["status"], "completed")

    def test_distinct_pi_sessions_execute_in_parallel(self):
        barrier = threading.Barrier(2)

        def create_process(*_args, **_kwargs):
            process = Mock(returncode=0, pid=123)

            def communicate(timeout=None):
                barrier.wait(timeout=0.5)
                return ("Pi result", "")

            process.communicate.side_effect = communicate
            return process

        with tempfile.TemporaryDirectory() as temporary_directory:
            runtime_directory = Path(temporary_directory) / "runtime"
            with (
                patch("app.RUNTIME_DIR", runtime_directory),
                patch("app.PI_SESSION_REGISTRY", runtime_directory / "sessions.json"),
                patch("app.subprocess.Popen", side_effect=create_process),
                patch("app.summarize_pi_output", return_value="summary"),
            ):
                app.launch_pi_job({
                    "prompt": "First task",
                    "directory": temporary_directory,
                    "session_action": "new",
                    "session_name": "first session",
                })
                app.launch_pi_job({
                    "prompt": "Second task",
                    "directory": temporary_directory,
                    "session_action": "new",
                    "session_name": "second session",
                })

                deadline = time.monotonic() + 1
                while time.monotonic() < deadline:
                    with app.PI_JOBS_LOCK:
                        statuses = {job["status"] for job in app.PI_JOBS.values()}
                    if statuses == {"completed"}:
                        break
                    time.sleep(0.01)

                with app.PI_JOBS_LOCK:
                    self.assertTrue(all(job["status"] == "completed" for job in app.PI_JOBS.values()))

    @patch("app.os.killpg")
    def test_running_pi_job_can_be_cancelled(self, killpg):
        process = Mock(pid=456)
        process.poll.return_value = None
        with app.PI_JOBS_LOCK:
            app.PI_JOBS["job-1"] = {"job_id": "job-1", "status": "running", "cancel_requested": False}
        with app.PI_PROCESSES_LOCK:
            app.PI_PROCESSES["job-1"] = process

        response = self.client.post("/api/pi/jobs/job-1/cancel")

        self.assertEqual(response.status_code, 200)
        with app.PI_JOBS_LOCK:
            self.assertTrue(app.PI_JOBS["job-1"]["cancel_requested"])
            self.assertEqual(app.PI_JOBS["job-1"]["status"], "cancelling")
        killpg.assert_called_once_with(456, app.signal.SIGTERM)

    @patch("app.openrouter_client")
    def test_completed_background_job_is_reported_to_jarvis(self, client_factory):
        with app.PI_JOBS_LOCK:
            app.PI_JOBS["job-1"] = {
                "job_id": "job-1",
                "status": "completed",
                "session_name": "inspection",
                "session_action": "new",
                "directory": "/tmp",
                "prompt": "Inspect the project",
                "summary": "Tests pass after fixing the configuration.",
                "error": None,
                "created_at": "2026-01-01T00:00:00Z",
                "started_at": "2026-01-01T00:00:01Z",
                "completed_at": "2026-01-01T00:00:02Z",
                "notified": False,
            }
        response_object = Mock()
        response_object.choices = [Mock(message=Mock(content="Pi finished and the tests pass.", tool_calls=None))]
        client_factory.return_value.chat.completions.create.return_value = response_object

        response = self.client.post(
            "/api/chat",
            json={"background_event": {"job_id": "job-1"}, "history": []},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["answer"], "Pi finished and the tests pass.")
        request_messages = client_factory.return_value.chat.completions.create.call_args.kwargs["messages"]
        self.assertIn("Tests pass after fixing", request_messages[-1]["content"])
        self.assertNotIn("job-1", request_messages[-1]["content"])

    @patch("app.openrouter_client")
    def test_routine_background_follow_up_can_remain_silent(self, client_factory):
        with app.PI_JOBS_LOCK:
            app.PI_JOBS["job-quiet"] = {
                "job_id": "job-quiet",
                "status": "completed",
                "session_name": "implementation",
                "session_action": "continue",
                "directory": "/tmp",
                "prompt": "Run the next internal check",
                "summary": "One check finished; another internal check is needed.",
                "error": None,
                "created_at": "2026-01-01T00:00:00Z",
                "started_at": "2026-01-01T00:00:01Z",
                "completed_at": "2026-01-01T00:00:02Z",
                "notified": False,
                "cancel_requested": False,
            }
        response_object = Mock()
        response_object.choices = [Mock(message=Mock(content="__SILENT__", tool_calls=None))]
        client_factory.return_value.chat.completions.create.return_value = response_object

        response = self.client.post(
            "/api/chat",
            json={"background_event": {"job_id": "job-quiet"}, "history": []},
        )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["silent"])
        self.assertEqual(response.json()["answer"], "")
        self.assertIn("another internal check", response.json()["history_content"])

    def test_pi_sessions_use_private_id_and_explicit_resume(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            runtime_directory = Path(temporary_directory) / "runtime"
            registry_file = runtime_directory / "sessions.json"
            processes = []
            for output in ("initial result", "follow-up result"):
                process = Mock(returncode=0, pid=123)
                process.communicate.return_value = (output, "")
                processes.append(process)

            with (
                patch("app.RUNTIME_DIR", runtime_directory),
                patch("app.PI_SESSION_REGISTRY", registry_file),
                patch("app.subprocess.Popen", side_effect=processes) as popen,
                patch("app.summarize_pi_output", return_value="summary"),
            ):
                first = app.run_pi_tool({
                    "prompt": "Inspect the project",
                    "directory": temporary_directory,
                    "session_action": "new",
                    "session_name": "project inspection",
                })
                second = app.run_pi_tool({
                    "prompt": "Check the tests now",
                    "directory": temporary_directory,
                    "session_action": "continue",
                    "session_name": "project inspection",
                })
                catalog = app.pi_session_catalog()

            first_command = popen.call_args_list[0].args[0]
            second_command = popen.call_args_list[1].args[0]
            self.assertIn("--session-id", first_command)
            self.assertIn("json", first_command)
            self.assertNotIn("--model", first_command)
            self.assertIn("--session", second_command)
            self.assertEqual(first["session_name"], "project inspection")
            self.assertEqual(second["session_name"], "project inspection")
            self.assertNotIn("id", first)
            self.assertIn("project inspection", catalog)
            self.assertNotIn(first_command[first_command.index("--session-id") + 1], catalog)


if __name__ == "__main__":
    unittest.main()
