const MIN_VOICE_THRESHOLD = 0.008;
const PRE_ROLL_MS = 1200;
const WAKE_SAMPLE_RATE = 16000;
const WAKE_FRAME_SAMPLES = 1280;
const SILENCE_MS = 950;
const MAX_UTTERANCE_MS = 12000;
const WAKE_COMMAND_MS = 12000;
const FOLLOW_UP_MS = 8000;
const INTERRUPT_THRESHOLD = 0.04;
const INTERRUPT_HOLD_MS = 140;

const listenButton = document.querySelector("#listenButton");
const listenButtonLabel = listenButton.querySelector("span");
const statusText = document.querySelector("#statusText");
const statusIndicator = document.querySelector("#statusIndicator");
const modeLabel = document.querySelector("#modeLabel");
const modeHint = document.querySelector("#modeHint");
const conversation = document.querySelector("#conversation");
let emptyState = document.querySelector("#emptyState");
const conversationSelect = document.querySelector("#conversationSelect");
const newConversationButton = document.querySelector("#newConversationButton");
const meter = document.querySelector("#meter");
const meterContext = meter.getContext("2d");
const piJobsElement = document.querySelector("#piJobs");
const piJobCount = document.querySelector("#piJobCount");
const messageForm = document.querySelector("#messageForm");
const messageInput = document.querySelector("#messageInput");
const sendButton = document.querySelector("#sendButton");
const speechToggle = document.querySelector("#speechToggle");

let stream;
let audioContext;
let analyser;
let sampleBuffer;
let captureNode;
let wakeSocket;
let wakePcmBuffer = [];
let preRollChunks = [];
let preRollSamples = 0;
let utteranceChunks = [];
let listening = false;
let recording = false;
let busy = false;
let speaking = false;
let finishSpeech;
let activeSpeechAudio;
let activeSpeechUrl;
let activeStreamingTurn;
let interruptLoudSince = 0;
let noiseFloor = 0.003;
let lastSoundAt = 0;
let recordingStartedAt = 0;
let awaitingCommand = false;
let capturingWakeCommand = false;
let wakeTimer;
let animationFrame;
let history = [];
let currentConversationId;
let notificationInProgress = false;
let speechEnabled = localStorage.getItem("kinetiq-speech-enabled") !== "false";
const queuedJobIds = new Set();
const jobNotificationQueue = [];

function stripSilentMarkers(text = "") {
  return text
    .replace(/`{0,3}__SILENT__`{0,3}[.!]?/gi, "")
    .replace(/(?:<SILENT>|\[SILENT\])[.!]?/gi, "")
    .replace(/^[ \t]*(?:`{1,3}[ \t]*)?(?:SILENT|STAY[ \t]+SILENT|NO[ \t]+USER[ \t]+UPDATE)(?:[ \t]*`{1,3})?[.!]?[ \t]*(?:\r?\n|$)/gim, "")
    .replace(/\n{3,}/g, "\n\n");
}

function isSilentText(text = "") {
  const normalized = text.toLowerCase().replace(/[^a-z]/g, "");
  return ["silent", "staysilent", "nouserupdate"].includes(normalized) || !stripSilentMarkers(text).trim();
}

function showEmptyConversation() {
  conversation.replaceChildren();
  emptyState = document.createElement("div");
  emptyState.className = "empty-state";
  emptyState.id = "emptyState";
  const instruction = document.createElement("p");
  instruction.append("Type a message or say ");
  const wakeWord = document.createElement("strong");
  wakeWord.textContent = "“Hey Jarvis.”";
  instruction.append(wakeWord);
  const privacy = document.createElement("p");
  privacy.textContent = "Your speech is transcribed on this machine. Only text is sent to OpenRouter.";
  emptyState.append(instruction, privacy);
  conversation.append(emptyState);
}

async function refreshConversationList(selectedId = currentConversationId) {
  const response = await fetch("/api/conversations");
  const result = await readResponse(response);
  conversationSelect.replaceChildren();
  for (const item of result.conversations) {
    const option = document.createElement("option");
    option.value = item.conversation_id;
    option.textContent = item.title;
    conversationSelect.append(option);
  }
  if (selectedId) conversationSelect.value = selectedId;
  return result.conversations;
}

async function switchConversation(conversationId) {
  if (!conversationId) return;
  if (listening) stopListening();
  else interruptSpeech();
  currentConversationId = conversationId;
  conversationSelect.value = conversationId;
  const response = await fetch(`/api/conversations/${encodeURIComponent(conversationId)}`);
  const result = await readResponse(response);
  history = result.messages.map((message) => ({ role: message.role, content: message.content }));
  conversation.replaceChildren();
  emptyState = undefined;
  if (!result.messages.length) {
    showEmptyConversation();
  } else {
    for (const message of result.messages) {
      addMessage(message.role === "user" ? "You" : "KINETIQ", message.content, message.role);
    }
  }
  jobNotificationQueue.length = 0;
  queuedJobIds.clear();
  refreshPiJobs();
}

async function initializeConversations() {
  let conversations = await refreshConversationList();
  if (!conversations.length) {
    const response = await fetch("/api/conversations", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title: "New conversation" }),
    });
    const created = await readResponse(response);
    conversations = await refreshConversationList(created.conversation_id);
  }
  await switchConversation(currentConversationId || conversations[0].conversation_id);
}

async function createNewConversation() {
  newConversationButton.disabled = true;
  try {
    const response = await fetch("/api/conversations", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title: "New conversation" }),
    });
    const created = await readResponse(response);
    await refreshConversationList(created.conversation_id);
    await switchConversation(created.conversation_id);
  } finally {
    newConversationButton.disabled = false;
  }
}

function setMode(label, hint, state = "active") {
  modeLabel.textContent = label;
  modeHint.textContent = hint;
  statusText.textContent = label;
  statusIndicator.className = `status-indicator ${state}`;
}

function resetToArmed() {
  if (!listening) return;
  busy = false;
  setMode("Waiting for “Hey Jarvis”", "openWakeWord is listening locally.");
}

function drawMeter(level = 0) {
  const width = meter.width;
  const height = meter.height;
  meterContext.clearRect(0, 0, width, height);
  const styles = getComputedStyle(document.documentElement);
  meterContext.fillStyle = styles.getPropertyValue("--meter-line");
  meterContext.fillRect(0, Math.floor(height / 2), width, 1);

  const bars = 14;
  const activeBars = Math.min(bars, Math.round(level * 70));
  for (let index = 0; index < bars; index += 1) {
    const barHeight = index < activeBars ? Math.max(4, level * height) : 2;
    meterContext.fillStyle = index < activeBars ? styles.getPropertyValue("--accent") : styles.getPropertyValue("--meter-idle");
    meterContext.fillRect(index * 8, (height - barHeight) / 2, 4, barHeight);
  }
}

function monitorAudio() {
  if (!listening || !analyser) return;

  analyser.getFloatTimeDomainData(sampleBuffer);
  const sum = sampleBuffer.reduce((total, sample) => total + sample * sample, 0);
  const level = Math.sqrt(sum / sampleBuffer.length);
  drawMeter(level);

  const now = performance.now();
  const startThreshold = Math.max(MIN_VOICE_THRESHOLD, noiseFloor * 3);
  const continueThreshold = Math.max(MIN_VOICE_THRESHOLD * 0.75, noiseFloor * 1.8);

  if (!busy && !awaitingCommand && !speaking && !recording) {
    noiseFloor = noiseFloor * 0.97 + Math.min(level, noiseFloor * 1.5) * 0.03;
  }

  if (speaking && !recording) {
    const interruptionThreshold = Math.max(INTERRUPT_THRESHOLD, noiseFloor * 5);
    if (level > interruptionThreshold) {
      if (!interruptLoudSince) interruptLoudSince = now;
      if (now - interruptLoudSince >= INTERRUPT_HOLD_MS) {
        interruptSpeech();
        awaitingCommand = true;
        busy = false;
        startUtterance(now, 300);
      }
    } else {
      interruptLoudSince = 0;
    }
  } else if (!busy && awaitingCommand) {
    if (!recording) {
      noiseFloor = noiseFloor * 0.98 + Math.min(level, startThreshold) * 0.02;
      if (level > startThreshold) startUtterance(now);
    } else {
      if (level > continueThreshold) lastSoundAt = now;
      const silentLongEnough = now - lastSoundAt > SILENCE_MS;
      const tooLong = now - recordingStartedAt > MAX_UTTERANCE_MS;
      if (silentLongEnough || tooLong) stopUtterance();
    }
  }

  animationFrame = requestAnimationFrame(monitorAudio);
}

function captureAudio(event) {
  const chunk = new Float32Array(event.inputBuffer.getChannelData(0));
  preRollChunks.push(chunk);
  preRollSamples += chunk.length;

  const maxPreRollSamples = Math.round(audioContext.sampleRate * PRE_ROLL_MS / 1000);
  while (preRollSamples > maxPreRollSamples && preRollChunks.length > 1) {
    preRollSamples -= preRollChunks.shift().length;
  }

  if (recording) utteranceChunks.push(chunk);
  if (listening && !awaitingCommand && !busy && !speaking && wakeSocket?.readyState === WebSocket.OPEN) {
    const wakeSamples = downsampleAudio(chunk, audioContext.sampleRate);
    for (const sample of wakeSamples) wakePcmBuffer.push(sample);
    while (wakePcmBuffer.length >= WAKE_FRAME_SAMPLES) {
      const frame = new Int16Array(wakePcmBuffer.splice(0, WAKE_FRAME_SAMPLES));
      wakeSocket.send(frame.buffer);
    }
  }
}

function downsampleAudio(input, sourceRate) {
  if (sourceRate === WAKE_SAMPLE_RATE) {
    return Int16Array.from(input, (sample) => Math.max(-32768, Math.min(32767, sample * 32768)));
  }
  const ratio = sourceRate / WAKE_SAMPLE_RATE;
  const output = new Int16Array(Math.floor(input.length / ratio));
  for (let index = 0; index < output.length; index += 1) {
    const start = Math.floor(index * ratio);
    const end = Math.max(start + 1, Math.floor((index + 1) * ratio));
    let total = 0;
    for (let sourceIndex = start; sourceIndex < end && sourceIndex < input.length; sourceIndex += 1) {
      total += input[sourceIndex];
    }
    const sample = total / (end - start);
    output[index] = Math.max(-32768, Math.min(32767, sample * 32768));
  }
  return output;
}

function activateWakeWord() {
  if (!listening || awaitingCommand || busy || speaking) return;
  awaitingCommand = true;
  capturingWakeCommand = true;
  wakePcmBuffer = [];
  playWakeTone();
  startUtterance(performance.now());
  clearTimeout(wakeTimer);
  wakeTimer = setTimeout(() => {
    awaitingCommand = false;
    capturingWakeCommand = false;
    resetToArmed();
  }, WAKE_COMMAND_MS);
}

function startUtterance(now, preRollMs = PRE_ROLL_MS) {
  const wantedSamples = Math.round(audioContext.sampleRate * preRollMs / 1000);
  let selectedSamples = 0;
  const selectedChunks = [];
  for (let index = preRollChunks.length - 1; index >= 0 && selectedSamples < wantedSamples; index -= 1) {
    selectedChunks.unshift(preRollChunks[index].slice());
    selectedSamples += preRollChunks[index].length;
  }
  utteranceChunks = selectedChunks;
  recording = true;
  recordingStartedAt = now;
  lastSoundAt = now;
  setMode(awaitingCommand ? "Listening to your request" : "Hearing speech", "Waiting for the end of the phrase.");
}

function stopUtterance() {
  if (!recording) return;
  recording = false;
  busy = true;
  const chunks = utteranceChunks;
  utteranceChunks = [];
  const blob = encodeWav(chunks, audioContext.sampleRate);
  if (blob.size < 1000) return resetToArmed();
  handleSpeech(blob);
}

function encodeWav(chunks, sampleRate) {
  const sampleCount = chunks.reduce((total, chunk) => total + chunk.length, 0);
  const buffer = new ArrayBuffer(44 + sampleCount * 2);
  const view = new DataView(buffer);
  const writeText = (offset, text) => [...text].forEach((character, index) => view.setUint8(offset + index, character.charCodeAt(0)));

  writeText(0, "RIFF");
  view.setUint32(4, 36 + sampleCount * 2, true);
  writeText(8, "WAVE");
  writeText(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeText(36, "data");
  view.setUint32(40, sampleCount * 2, true);

  let offset = 44;
  for (const chunk of chunks) {
    for (const sample of chunk) {
      const clipped = Math.max(-1, Math.min(1, sample));
      view.setInt16(offset, clipped < 0 ? clipped * 0x8000 : clipped * 0x7fff, true);
      offset += 2;
    }
  }
  return new Blob([buffer], { type: "audio/wav" });
}

async function handleSpeech(blob) {
  setMode("Transcribing locally", "Nemotron Speech is processing the recording on this machine.");
  try {
    const form = new FormData();
    form.append("audio", blob, "speech.wav");
    const response = await fetch("/api/transcribe", { method: "POST", body: form });
    const result = await readResponse(response);
    const text = result.text.trim();
    if (!text && capturingWakeCommand) {
      capturingWakeCommand = false;
      busy = false;
      return setMode("Ready for your request", "Speak now.");
    }
    if (!text) return resetToArmed();

    if (awaitingCommand && !capturingWakeCommand) {
      awaitingCommand = false;
      clearTimeout(wakeTimer);
      return askJarvis(text);
    }

    const wakeMatch = text.match(/\b(?:hey\s+)?(?:jarvis|jervis)\b/i);
    const command = wakeMatch
      ? text.slice(wakeMatch.index + wakeMatch[0].length).replace(/^[,.:;!?\s-]+/, "").trim()
      : text;
    capturingWakeCommand = false;
    if (command) {
      awaitingCommand = false;
      clearTimeout(wakeTimer);
      return askJarvis(command);
    }

    setMode("Ready for your request", "Speak now.");
    setTimeout(() => {
      if (awaitingCommand) busy = false;
    }, 180);
    wakeTimer = setTimeout(() => {
      awaitingCommand = false;
      resetToArmed();
    }, WAKE_COMMAND_MS);
  } catch (error) {
    showError(error.message);
  }
}

function extractSpeechChunks(buffer, flush = false) {
  const chunks = [];
  let remainder = buffer;
  while (remainder) {
    const sentence = remainder.match(/^([\s\S]*?[.!?]["']?)(?=\s|$)/);
    if (sentence) {
      chunks.push(sentence[1].trim());
      remainder = remainder.slice(sentence[0].length).trimStart();
      continue;
    }
    if (remainder.length > 180) {
      const splitAt = remainder.lastIndexOf(" ", 160);
      if (splitAt > 60) {
        chunks.push(remainder.slice(0, splitAt).trim());
        remainder = remainder.slice(splitAt + 1);
        continue;
      }
    }
    break;
  }
  if (flush && remainder.trim()) {
    chunks.push(remainder.trim());
    remainder = "";
  }
  return { chunks, remainder };
}

async function readChatStream(payload, controller, onEvent) {
  const response = await fetch("/api/chat/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    signal: controller.signal,
  });
  if (!response.ok) return readResponse(response);

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let doneResult;
  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";
    for (const line of lines) {
      if (!line.trim()) continue;
      const streamEvent = JSON.parse(line);
      if (streamEvent.type === "error") throw new Error(streamEvent.detail);
      onEvent(streamEvent);
      if (streamEvent.type === "done") doneResult = streamEvent;
    }
    if (done) break;
  }
  if (!doneResult) throw new Error("OpenRouter stream ended before completion");
  return doneResult;
}

async function askJarvis(message) {
  addMessage("You", message, "user");
  const assistantText = addMessage("KINETIQ", "", "assistant");
  setMode("Thinking", "Waiting for the first OpenRouter token.");

  const turn = { controller: new AbortController(), interrupted: false };
  activeStreamingTurn = turn;
  let answer = "";
  let speechBuffer = "";
  let spokenChars = 0;
  let speechChain = Promise.resolve();
  let result;

  const queueSpeech = (text) => {
    if (!text || !speechEnabled) return;
    speechChain = speechChain.then(async () => {
      if (turn.interrupted || !speechEnabled) return;
      const speechResult = await speak(text);
      spokenChars += speechResult.spokenChars;
      if (speechResult.status === "interrupted") turn.interrupted = true;
    });
  };

  try {
    result = await readChatStream(
      { message, history, conversation_id: currentConversationId },
      turn.controller,
      (streamEvent) => {
      if (streamEvent.type !== "text_delta") return;
      answer = stripSilentMarkers(answer + streamEvent.text);
      assistantText.textContent = answer;
      speechBuffer = stripSilentMarkers(speechBuffer + streamEvent.text);
      const extracted = extractSpeechChunks(speechBuffer);
      speechBuffer = extracted.remainder;
        extracted.chunks.forEach(queueSpeech);
      },
    );
    if (result.conversation_id) currentConversationId = result.conversation_id;
    answer = stripSilentMarkers(result.answer);
    assistantText.textContent = answer;
    speechBuffer = stripSilentMarkers(speechBuffer);
    const finalChunks = extractSpeechChunks(speechBuffer, true);
    finalChunks.chunks.forEach(queueSpeech);
    await speechChain;

    let historyAnswer = result.history_content || result.answer;
    if (turn.interrupted) {
      const heardText = result.answer.slice(0, spokenChars).trim();
      const deliveryNote = `[Delivery status: Streamed speech was interrupted after approximately ${spokenChars} of ${result.answer.length} characters. The user did not hear anything after that point. Last words likely heard: “${heardText.slice(-120)}”]`;
      historyAnswer += `\n\n${deliveryNote}`;
      persistConversationContext(deliveryNote);
    }
    history.push({ role: "user", content: message }, { role: "assistant", content: historyAnswer });
    history = history.slice(-10);
    refreshConversationList(currentConversationId).catch(() => {});
    if (!turn.interrupted) openFollowUpWindow();
  } catch (error) {
    await speechChain.catch(() => {});
    if (turn.interrupted || error.name === "AbortError") {
      const deliveryNote = `[Delivery status: The streamed response was interrupted. The user heard approximately ${spokenChars} characters.]`;
      const historyAnswer = `${answer}\n\n${deliveryNote}`;
      persistConversationContext(deliveryNote);
      history.push({ role: "user", content: message }, { role: "assistant", content: historyAnswer });
      history = history.slice(-10);
      return;
    }
    showError(error.message);
  } finally {
    if (activeStreamingTurn === turn) activeStreamingTurn = undefined;
    messageInput.disabled = false;
    sendButton.disabled = false;
    messageInput.focus();
  }
}

function persistConversationContext(content) {
  if (!currentConversationId || !content) return;
  fetch(`/api/conversations/${encodeURIComponent(currentConversationId)}/context`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ content }),
  }).catch(() => {});
}

function openFollowUpWindow() {
  if (!listening) {
    busy = false;
    setMode("Not listening", "Type below or turn on the microphone.", "");
    return;
  }
  setTimeout(() => {
    awaitingCommand = true;
    busy = false;
    setMode("Listening for a follow-up", "Speak within 8 seconds, no wake word needed.");
    clearTimeout(wakeTimer);
    wakeTimer = setTimeout(() => {
      awaitingCommand = false;
      resetToArmed();
    }, FOLLOW_UP_MS);
  }, 250);
}

function renderPiJobs(jobs) {
  piJobCount.textContent = `${jobs.length} ${jobs.length === 1 ? "job" : "jobs"}`;
  piJobsElement.replaceChildren();
  if (!jobs.length) {
    const empty = document.createElement("p");
    empty.className = "pi-empty";
    empty.textContent = "No background work.";
    piJobsElement.append(empty);
    return;
  }

  for (const job of jobs) {
    const item = document.createElement("article");
    item.className = "pi-job";
    const line = document.createElement("div");
    line.className = "pi-job-line";
    const name = document.createElement("strong");
    name.className = "pi-job-name";
    name.textContent = job.session_name || "Unnamed session";
    const status = document.createElement("span");
    status.className = `pi-job-status ${job.status}`;
    status.textContent = job.status;
    line.append(name, status);
    if (["queued", "running", "cancelling"].includes(job.status)) {
      const stopButton = document.createElement("button");
      stopButton.className = "pi-stop";
      stopButton.type = "button";
      stopButton.textContent = job.status === "cancelling" ? "Stopping" : "Stop";
      stopButton.disabled = job.status === "cancelling";
      stopButton.setAttribute("aria-label", `Stop ${job.session_name}`);
      stopButton.addEventListener("click", async () => {
        stopButton.disabled = true;
        stopButton.textContent = "Stopping";
        try {
          const response = await fetch(`/api/pi/jobs/${encodeURIComponent(job.job_id)}/cancel`, { method: "POST" });
          await readResponse(response);
          refreshPiJobs();
        } catch (error) {
          stopButton.disabled = false;
          stopButton.textContent = "Stop";
        }
      });
      line.append(stopButton);
    }

    const directory = document.createElement("p");
    directory.className = "pi-job-directory";
    directory.textContent = job.directory;
    directory.title = job.directory;

    const details = document.createElement("details");
    const detailsLabel = document.createElement("summary");
    detailsLabel.textContent = "Details";
    const prompt = document.createElement("p");
    prompt.className = "pi-job-detail";
    prompt.textContent = `Prompt: ${job.prompt || "None"}`;
    details.append(detailsLabel, prompt);
    if (job.summary || job.error) {
      const result = document.createElement("p");
      result.className = "pi-job-detail";
      result.textContent = job.summary ? `Result: ${job.summary}` : `Error: ${job.error}`;
      details.append(result);
    }
    item.append(line, directory, details);
    piJobsElement.append(item);
  }
}

async function refreshPiJobs() {
  try {
    const response = await fetch("/api/pi/jobs");
    const result = await readResponse(response);
    renderPiJobs(result.jobs);
    for (const job of result.jobs) {
      const belongsToCurrentConversation = !job.conversation_id || job.conversation_id === currentConversationId;
      if (belongsToCurrentConversation && ["completed", "failed", "cancelled"].includes(job.status) && !job.notified && !queuedJobIds.has(job.job_id)) {
        queuedJobIds.add(job.job_id);
        jobNotificationQueue.push(job);
      }
    }
    processJobNotifications();
  } catch (error) {
    piJobCount.textContent = "unavailable";
  }
}

async function processJobNotifications() {
  if (notificationInProgress || busy || recording || speaking || !jobNotificationQueue.length) return;
  const job = jobNotificationQueue.shift();
  notificationInProgress = true;
  busy = true;
  setMode("Reviewing Pi’s result", `${job.session_name} finished in the background.`);

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: "",
        history,
        conversation_id: job.conversation_id || currentConversationId,
        background_event: { job_id: job.job_id },
      }),
    });
    const result = await readResponse(response);
    await fetch(`/api/pi/jobs/${encodeURIComponent(job.job_id)}/acknowledge`, { method: "POST" });
    if (result.silent || isSilentText(result.answer)) {
      if (result.history_content) {
        history.push({ role: "assistant", content: result.history_content });
        history = history.slice(-10);
      }
      queuedJobIds.delete(job.job_id);
      busy = false;
      if (listening && awaitingCommand) {
        setMode("Listening for a follow-up", "No wake word needed while the current window remains open.");
      } else if (listening) {
        resetToArmed();
      } else {
        setMode("Not listening", "Pi can continue working in the background.", "");
      }
      return;
    }

    addMessage("KINETIQ", result.answer, "assistant");
    const speechResult = speechEnabled
      ? await speak(result.answer)
      : { status: "finished", spokenChars: 0 };
    let historyAnswer = result.history_content || result.answer;
    if (speechResult.status === "interrupted") {
      const heardText = result.answer.slice(0, speechResult.spokenChars).trim();
      const deliveryNote = `[Delivery status: Background update was interrupted after approximately ${speechResult.spokenChars} of ${result.answer.length} characters. Last words likely heard: “${heardText.slice(-120)}”]`;
      historyAnswer += `\n\n${deliveryNote}`;
      persistConversationContext(deliveryNote);
    }
    history.push({ role: "assistant", content: historyAnswer });
    history = history.slice(-10);
    queuedJobIds.delete(job.job_id);
    if (speechResult.status !== "interrupted") openFollowUpWindow();
  } catch (error) {
    queuedJobIds.delete(job.job_id);
    showError(error.message);
  } finally {
    notificationInProgress = false;
  }
}

async function readResponse(response) {
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.detail || `Request failed (${response.status})`);
  return data;
}

function addMessage(role, text, type) {
  emptyState?.remove();
  const article = document.createElement("article");
  article.className = `message ${type}`;
  const roleElement = document.createElement("div");
  roleElement.className = "message-role";
  roleElement.textContent = role;
  const textElement = document.createElement("p");
  textElement.textContent = text;
  article.append(roleElement, textElement);
  conversation.append(article);
  article.scrollIntoView({ behavior: "smooth", block: "end" });
  return textElement;
}

async function speak(text) {
  text = stripSilentMarkers(text).trim();
  if (!text) return { status: "finished", spokenChars: 0 };
  setMode("Generating voice", "Kokoro is creating speech locally.");
  const response = await fetch("/api/synthesize", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  const audioBlob = await response.blob();
  if (!response.ok) {
    const error = await audioBlob.text().catch(() => "");
    throw new Error(error || `Kokoro request failed (${response.status})`);
  }
  if (!speechEnabled) return { status: "finished", spokenChars: 0 };

  return new Promise((resolve) => {
    setMode("Speaking", "Speak over the response to interrupt.");
    speaking = true;
    interruptLoudSince = 0;
    activeSpeechUrl = URL.createObjectURL(audioBlob);
    activeSpeechAudio = new Audio(activeSpeechUrl);
    let settled = false;
    const settle = (status) => {
      if (settled) return;
      settled = true;
      let spokenChars = text.length;
      if (status === "interrupted") {
        spokenChars = Number.isFinite(activeSpeechAudio.duration) && activeSpeechAudio.duration > 0
          ? Math.min(text.length, Math.round(text.length * activeSpeechAudio.currentTime / activeSpeechAudio.duration))
          : Math.min(text.length, Math.round(activeSpeechAudio.currentTime * 14));
      }
      activeSpeechAudio.pause();
      URL.revokeObjectURL(activeSpeechUrl);
      activeSpeechAudio = undefined;
      activeSpeechUrl = undefined;
      speaking = false;
      finishSpeech = undefined;
      resolve({ status, spokenChars });
    };
    finishSpeech = () => settle("interrupted");
    activeSpeechAudio.addEventListener("ended", () => settle("finished"), { once: true });
    activeSpeechAudio.addEventListener("error", () => settle("finished"), { once: true });
    activeSpeechAudio.play().catch(() => settle("finished"));
  });
}

function interruptSpeech() {
  if (activeStreamingTurn) {
    activeStreamingTurn.interrupted = true;
    activeStreamingTurn.controller.abort();
  }
  if (speaking) finishSpeech?.();
  interruptLoudSince = 0;
}

function playWakeTone() {
  if (!audioContext) return;
  const oscillator = audioContext.createOscillator();
  const gain = audioContext.createGain();
  oscillator.frequency.value = 660;
  gain.gain.setValueAtTime(0.06, audioContext.currentTime);
  gain.gain.exponentialRampToValueAtTime(0.001, audioContext.currentTime + 0.12);
  oscillator.connect(gain).connect(audioContext.destination);
  oscillator.start();
  oscillator.stop(audioContext.currentTime + 0.12);
}

function showError(message) {
  busy = false;
  awaitingCommand = false;
  capturingWakeCommand = false;
  setMode("Something went wrong", message, "error");
  setTimeout(resetToArmed, 3500);
}

function connectWakeWord() {
  return new Promise((resolve, reject) => {
    const protocol = location.protocol === "https:" ? "wss:" : "ws:";
    wakeSocket = new WebSocket(`${protocol}//${location.host}/api/wake-word`);
    wakeSocket.onmessage = (event) => {
      const message = JSON.parse(event.data);
      if (message.ready) resolve();
      else if (message.detected) activateWakeWord();
      else if (message.error) reject(new Error(message.error));
    };
    wakeSocket.onerror = () => reject(new Error("Could not connect to the local wake-word detector."));
    wakeSocket.onclose = () => {
      if (listening) {
        stopListening();
        setMode("Wake-word detector disconnected", "Start listening to reconnect.", "error");
      }
    };
  });
}

async function startListening() {
  listenButton.disabled = true;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
        channelCount: 1,
      },
    });
    audioContext = new AudioContext();
    await audioContext.resume();
    analyser = audioContext.createAnalyser();
    analyser.fftSize = 256;
    sampleBuffer = new Float32Array(analyser.fftSize);
    const source = audioContext.createMediaStreamSource(stream);
    const voiceFilter = audioContext.createBiquadFilter();
    voiceFilter.type = "highpass";
    voiceFilter.frequency.value = 100;
    voiceFilter.Q.value = 0.7;
    source.connect(voiceFilter);
    voiceFilter.connect(analyser);

    captureNode = audioContext.createScriptProcessor(2048, 1, 1);
    const silentOutput = audioContext.createGain();
    silentOutput.gain.value = 0;
    captureNode.onaudioprocess = captureAudio;
    voiceFilter.connect(captureNode);
    captureNode.connect(silentOutput).connect(audioContext.destination);
    preRollChunks = [];
    preRollSamples = 0;
    wakePcmBuffer = [];
    noiseFloor = 0.003;
    setMode("Loading wake-word detector", "Preparing openWakeWord on this machine.");
    await connectWakeWord();
    listening = true;
    awaitingCommand = true;
    listenButton.classList.add("listening");
    listenButtonLabel.textContent = "Stop listening";
    setMode("Ready for your request", "Speak now. You can use “Hey Jarvis” again afterward.");
    clearTimeout(wakeTimer);
    wakeTimer = setTimeout(() => {
      awaitingCommand = false;
      resetToArmed();
    }, WAKE_COMMAND_MS);
    monitorAudio();
  } catch (error) {
    wakeSocket?.close();
    wakeSocket = undefined;
    stream?.getTracks().forEach((track) => track.stop());
    await audioContext?.close();
    statusIndicator.className = "status-indicator error";
    statusText.textContent = "Listening unavailable";
    modeLabel.textContent = "Could not start";
    modeHint.textContent = error.message;
  } finally {
    listenButton.disabled = false;
  }
}

function stopListening() {
  listening = false;
  busy = false;
  awaitingCommand = false;
  capturingWakeCommand = false;
  clearTimeout(wakeTimer);
  cancelAnimationFrame(animationFrame);
  recording = false;
  utteranceChunks = [];
  if (captureNode) captureNode.onaudioprocess = null;
  wakeSocket?.close();
  wakeSocket = undefined;
  wakePcmBuffer = [];
  stream?.getTracks().forEach((track) => track.stop());
  audioContext?.close();
  interruptSpeech();
  drawMeter(0);
  listenButton.classList.remove("listening");
  listenButtonLabel.textContent = "Start listening";
  statusIndicator.className = "status-indicator";
  statusText.textContent = "Microphone off";
  modeLabel.textContent = "Not listening";
  modeHint.textContent = "Microphone permission is required once.";
}

async function submitTypedMessage(event) {
  event.preventDefault();
  const message = messageInput.value.trim();
  if (!message || busy || activeStreamingTurn || notificationInProgress) return;
  busy = true;
  messageInput.value = "";
  messageInput.disabled = true;
  sendButton.disabled = true;
  await askJarvis(message);
}

speechToggle.checked = speechEnabled;
speechToggle.addEventListener("change", () => {
  speechEnabled = speechToggle.checked;
  localStorage.setItem("kinetiq-speech-enabled", String(speechEnabled));
  if (!speechEnabled && speaking) finishSpeech?.();
});
messageForm.addEventListener("submit", submitTypedMessage);
listenButton.addEventListener("click", () => listening ? stopListening() : startListening());
conversationSelect.addEventListener("change", () => switchConversation(conversationSelect.value));
newConversationButton.addEventListener("click", createNewConversation);
drawMeter(0);
initializeConversations().catch((error) => showError(error.message));
setInterval(refreshPiJobs, 1500);
