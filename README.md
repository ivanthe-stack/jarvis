# J.A.R.V.I.S.

A local-first voice assistant proof of concept. The browser continuously streams microphone audio to a local openWakeWord detector after one permission click. Once “Hey Jarvis” activates it, the server transcribes the command with NVIDIA Nemotron Speech Streaming EN 0.6B, sends text to OpenRouter using the OpenAI Python library, and generates speech with Kokoro-82M. J.A.R.V.I.S. can delegate computer and coding work to headless Pi sessions.

## Requirements

- Python 3.11+
- A Chromium or Firefox browser
- An NVIDIA GPU with CUDA support recommended for Nemotron Speech
- `espeak-ng` and `libsndfile1`
- An OpenRouter API key
- Pi installed, authenticated, and available as `pi`

## Run

```bash
sudo apt install espeak-ng libsndfile1
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Add your OpenRouter key and summarizer model to .env
uvicorn app:app --reload
```

Open <http://127.0.0.1:8000>, click **Start listening**, allow microphone access, and say **“Hey Jarvis.”** After the acknowledgement, speak your request. The first start downloads openWakeWord's local `hey_jarvis` model. Set `WAKE_WORD_THRESHOLD` to tune its activation score (default `0.5`).

### Private phone access with Tailscale

This PC is configured to serve J.A.R.V.I.S. directly over Tailscale HTTPS:

```bash
./run-tailscale.sh
```

On the Redmi 13C or this PC, connect Tailscale and open <https://ubuntupc.folk-shaula.ts.net:8443>. The launcher renews the Tailscale certificate, binds only to the PC's Tailscale addresses, and configures the application to accept only this PC and the phone. The private certificate and key remain under the ignored `.tailscale/` directory. You can also say the wake word and request together, for example: “Hey Jarvis, explain recursion.” After each reply, you have eight seconds to ask a follow-up without saying “Jarvis” again. You can also speak over a response to stop it and immediately ask a new question.

The first transcription downloads `nvidia/nemotron-speech-streaming-en-0.6b`; the first spoken response downloads `hexgrad/Kokoro-82M` and the configured voice. Nemotron and Kokoro both run on CUDA by default. Audio remains local and the application sends only conversation text to OpenRouter.

## Streaming responses

Direct user requests use OpenRouter’s streaming API through `/api/chat/stream`. Text deltas appear in the conversation immediately. Complete sentences are sent to Kokoro and spoken in order while OpenRouter continues generating later text. Voice interruption aborts the active OpenRouter stream, stops current audio, and discards queued speech. Background Pi completion events continue using the non-streaming endpoint so silent coordination markers can be resolved before anything is displayed or spoken.

## Conversations

Conversations are stored in `.jarvis/conversation.jsonl`. The file includes conversation metadata and the complete model context: system prompt snapshots, user messages, assistant replies, assistant tool calls, tool results, background Pi events, silent coordination, and delivery-status notes. Records also include timestamps, source, visibility, and completion status.

The web header provides a conversation selector and **New conversation** button. Creating a conversation never deletes older conversations. Switching reloads its visible messages while the backend restores its complete hidden model and tool context. Records written by the earlier single-conversation format appear as **Imported conversation**.

## Pi delegation

The main OpenRouter model decides when Pi is useful, whether to create a new named Pi session or continue one, and which existing directory Pi should run in. The backend privately maps descriptive session names to Pi session UUIDs in `.jarvis/pi-sessions.json`.

A new session runs as:

```text
pi -p --mode json --session-id <private-id> --name "JARVIS: <name>" <prompt>
```

A follow-up runs as:

```text
pi -p --mode json --session <private-id> <prompt>
```

No provider or model flags are passed, so Pi uses its configured model and restores it with the session. Pi runs headlessly with its normal tools and system access. Prompts are passed as subprocess arguments without a shell.

Pi calls are scheduled as background jobs, so the main J.A.R.V.I.S. request returns immediately and conversation can continue. Jobs in different named sessions execute in parallel. Follow-ups within the same session execute in order to protect Pi’s persisted conversation state. The web interface polls job status and shows the session, directory, prompt, status, summary, or error.

Running and queued jobs have a **Stop** action. You can also ask J.A.R.V.I.S. to stop work by its descriptive Pi session name. Running processes receive `SIGTERM`; queued work is cancelled before launch. JSON event mode captures tool calls, tool results, and assistant progress before termination. `OPENROUTER_SUMMARY_MODEL` summarizes the work completed before cancellation.

Completed, failed, and cancelled results are sent to the main J.A.R.V.I.S. model as private context. Routine progress and agent-to-agent follow-ups remain silent. J.A.R.V.I.S. speaks only for a useful final result, a meaningful failure, or a blocker requiring user input. Silent coordination is retained in conversation history without adding or speaking a visible message.

## MVP limitations

- Browsers require a click before microphone capture can begin.
- The browser currently sends each detected utterance as one WAV request; it does not yet feed Nemotron’s cache-aware streaming state continuously.
- Kokoro voice interruption relies on browser echo cancellation. Headphones provide the most reliable separation between J.A.R.V.I.S. and the user.
- Interrupted responses are marked in conversation history with the approximate playback position, so the model knows which portion the user likely did not hear.
- Command and interruption detection adapt their volume thresholds to the room, but sustained background audio can still affect them.
