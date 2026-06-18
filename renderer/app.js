/* ─── Ai4Me · renderer/app.js ──────────────────────────────────────── */

const WS_URL = 'ws://127.0.0.1:7823/ws';
const RECONNECT_DELAY = 3000;

let ws = null;
let connected = false;
let streamingBubble = null;
let streamingContent = '';
let reconnectTimer = null;

/* ─── DOM refs ─────────────────────────────────────────────────────── */
const messagesEl      = document.getElementById('messages');
const welcomeEl       = document.getElementById('welcome');
const inputEl         = document.getElementById('chat-input');
const sendBtn         = document.getElementById('send-btn');
const typingEl        = document.getElementById('typing-indicator');
const searchEl        = document.getElementById('search-indicator');
const orb             = document.getElementById('aitha-orb');
const aithaStatus     = document.getElementById('aitha-status');
const modelLabel      = document.getElementById('settings-model-label');

const clockTime   = document.getElementById('clock-time');
const clockDay    = document.getElementById('clock-day');

const voiceToggle = document.getElementById('voice-toggle');
const voiceLabel  = document.getElementById('voice-label');
let ttsEnabled = true;

/* ─── Voice toggle ─────────────────────────────────────────────────── */
function applyVoiceState(on) {
  ttsEnabled = on;
  voiceToggle.classList.toggle('muted', !on);
  voiceLabel.textContent = on ? 'Voice' : 'Muted';
  syncPassthroughControls();
}
voiceToggle.addEventListener('click', () => {
  const next = !ttsEnabled;
  applyVoiceState(next);
  if (connected && ws?.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: 'set_tts', enabled: next }));
  }
});

/* ─── Settings modal ───────────────────────────────────────────────── */
const settingsBtn    = document.getElementById('settings-btn');
const settingsModal  = document.getElementById('settings-modal');
const setModel       = document.getElementById('set-model');
const setVisionModel = document.getElementById('set-vision-model');
const setCtx         = document.getElementById('set-ctx');
const setCtxVal      = document.getElementById('set-ctx-val');
const setVoice       = document.getElementById('set-voice');
const setDevice      = document.getElementById('set-device');
const setTtsToggle   = document.getElementById('set-tts-toggle');

const setTagsToggle = document.getElementById('set-tags-toggle');
let pendingTts = true;  // staged speech on/off inside the modal

// Show raw directive tags (<note>, <journal>, <explore>…) — a local debug pref.
let showDirectives = localStorage.getItem('showDirectives') === '1';
function applyTagsToggleUI() {
  setTagsToggle.classList.toggle('off', !showDirectives);
  setTagsToggle.querySelector('span').textContent = showDirectives ? 'On' : 'Off';
}
setTagsToggle.addEventListener('click', () => {
  showDirectives = !showDirectives;
  localStorage.setItem('showDirectives', showDirectives ? '1' : '0');
  applyTagsToggleUI();
});
applyTagsToggleUI();

// Settings tabs
const modalTabs = document.getElementById('modal-tabs');
modalTabs?.addEventListener('click', (e) => {
  const tab = e.target.closest('.modal-tab');
  if (!tab) return;
  const name = tab.dataset.tab;
  modalTabs.querySelectorAll('.modal-tab').forEach(t => t.classList.toggle('active', t === tab));
  document.querySelectorAll('.modal-tabpane').forEach(p =>
    p.classList.toggle('active', p.dataset.pane === name));
});

// Clear recent chats — wipes the short-term conversation, keeps long-term memory.
const chatClearBtn  = document.getElementById('chat-clear');
const chatClearNote = document.getElementById('chat-clear-note');
chatClearBtn?.addEventListener('click', () => {
  if (connected && ws?.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: 'clear_chat' }));
    if (chatClearNote) {
      chatClearNote.textContent = 'Cleared. Aitha starts fresh.';
      setTimeout(() => { if (chatClearNote) chatClearNote.textContent = ''; }, 3000);
    }
  } else if (chatClearNote) {
    chatClearNote.textContent = 'Not connected — try again in a moment.';
  }
});

function openSettings() {
  if (connected && ws?.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: 'get_settings' }));
  }
  loadMemory();
  refreshSpotify();   // keep the Music capability connection note current
  // Point the Appearance editor at the tab you're currently looking at.
  editTarget = activeView;
  syncThemeControls();
  // Always open on the General tab.
  modalTabs?.querySelectorAll('.modal-tab').forEach(t => t.classList.toggle('active', t.dataset.tab === 'general'));
  document.querySelectorAll('.modal-tabpane').forEach(p => p.classList.toggle('active', p.dataset.pane === 'general'));
  settingsModal.classList.add('open');
}
function closeSettings() {
  settingsModal.classList.remove('open');
}

function fillSelect(el, items, selected) {
  el.innerHTML = '';
  for (const item of items) {
    const opt = document.createElement('option');
    opt.value = item;
    opt.textContent = el === setModel ? shortModel(item) : item;
    if (item === selected) opt.selected = true;
    el.appendChild(opt);
  }
}

function populateSettings(data) {
  const cur = data.current || {};
  const opt = data.options || {};

  const models = opt.models?.length ? opt.models : [cur.model];
  fillSelect(setModel, models, cur.model);
  if (setVisionModel) {
    setVisionModel.innerHTML = '';
    for (const item of ['', ...(opt.vision_models || [])]) {
      const o = document.createElement('option');
      o.value = item;
      o.textContent = item ? shortModel(item) : 'None — she can’t see images';
      if (item === (cur.vision_model || '')) o.selected = true;
      setVisionModel.appendChild(o);
    }
  }
  fillSelect(setVoice, opt.voices || [cur.tts_voice], cur.tts_voice);
  fillSelect(setDevice, opt.devices?.length ? opt.devices : [cur.tts_device], cur.tts_device);

  setCtx.value = cur.num_ctx ?? 6144;
  setCtxVal.textContent = setCtx.value;

  pendingTts = cur.tts_enabled !== false;
  setTtsToggle.classList.toggle('off', !pendingTts);
  setTtsToggle.querySelector('span').textContent = pendingTts ? 'On' : 'Off';

  if (cur.behavior && typeof cur.behavior === 'object') {
    Object.assign(pendingBehavior, cur.behavior);
    applyBehaviorUI();
  }

  if (cur.capabilities && typeof cur.capabilities === 'object') {
    pendingCaps = { ...pendingCaps, ...cur.capabilities };
  }
  applyCapsUI();

  pendingFileRoots = Array.isArray(cur.file_roots) ? [...cur.file_roots] : [];
  renderFileRoots();

  if (cur.char_name) applyCharName(cur.char_name);
}

settingsBtn.addEventListener('click', openSettings);
document.getElementById('settings-cancel').addEventListener('click', closeSettings);
document.getElementById('settings-cancel-2').addEventListener('click', closeSettings);
settingsModal.addEventListener('click', (e) => {
  if (e.target === settingsModal) closeSettings();
});
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && settingsModal.classList.contains('open')) closeSettings();
});

setCtx.addEventListener('input', () => { setCtxVal.textContent = setCtx.value; });

setTtsToggle.addEventListener('click', () => {
  pendingTts = !pendingTts;
  setTtsToggle.classList.toggle('off', !pendingTts);
  setTtsToggle.querySelector('span').textContent = pendingTts ? 'On' : 'Off';
});

/* ═══ Behavior tab — her self-directed drives (staged; sent on Save) ═══
   Toggles are hard on/off; *_freq are percent multipliers on her eagerness.
   heartbeat is the check-in cadence in seconds. */
const pendingBehavior = {
  proactive: true, journaling: true, curiosity: true,
  speak_freq: 1.0, journal_freq: 1.0, curiosity_freq: 1.0, heartbeat_seconds: 40,
};
const behToggles = {
  proactive: document.getElementById('beh-proactive-toggle'),
  journaling: document.getElementById('beh-journal-toggle'),
  curiosity: document.getElementById('beh-curiosity-toggle'),
};
const behFreqs = {
  speak_freq:     { el: document.getElementById('beh-speak-freq'),     val: document.getElementById('beh-speak-freq-val') },
  journal_freq:   { el: document.getElementById('beh-journal-freq'),   val: document.getElementById('beh-journal-freq-val') },
  curiosity_freq: { el: document.getElementById('beh-curiosity-freq'), val: document.getElementById('beh-curiosity-freq-val') },
};
const behHeartbeat    = document.getElementById('beh-heartbeat');
const behHeartbeatVal = document.getElementById('beh-heartbeat-val');

function applyBehaviorUI() {
  for (const [key, btn] of Object.entries(behToggles)) {
    if (!btn) continue;
    const on = pendingBehavior[key];
    btn.classList.toggle('off', !on);
    btn.querySelector('span').textContent = on ? 'On' : 'Off';
  }
  for (const [key, { el, val }] of Object.entries(behFreqs)) {
    if (!el) continue;
    const pct = Math.round(pendingBehavior[key] * 100);
    el.value = pct;
    if (val) val.textContent = pct + '%';
  }
  if (behHeartbeat) {
    behHeartbeat.value = pendingBehavior.heartbeat_seconds;
    if (behHeartbeatVal) behHeartbeatVal.textContent = pendingBehavior.heartbeat_seconds + 's';
  }
}

for (const [key, btn] of Object.entries(behToggles)) {
  btn?.addEventListener('click', () => {
    pendingBehavior[key] = !pendingBehavior[key];
    applyBehaviorUI();
  });
}
for (const [key, { el, val }] of Object.entries(behFreqs)) {
  el?.addEventListener('input', () => {
    pendingBehavior[key] = parseInt(el.value, 10) / 100;
    if (val) val.textContent = el.value + '%';
  });
}
behHeartbeat?.addEventListener('input', () => {
  pendingBehavior.heartbeat_seconds = parseInt(behHeartbeat.value, 10);
  if (behHeartbeatVal) behHeartbeatVal.textContent = behHeartbeat.value + 's';
});

/* ═══ Capability toggles — what's fed into her context (staged, sent on Save) ═══ */
const CAP_KEYS = ['notes', 'projects', 'calendar', 'files', 'images', 'web', 'themes', 'music'];
let pendingCaps = Object.fromEntries(CAP_KEYS.map(k => [k, true]));
const capToggles = document.querySelectorAll('[data-cap]');

function applyCapsUI() {
  capToggles.forEach(btn => {
    const on = pendingCaps[btn.dataset.cap] !== false;
    btn.classList.toggle('off', !on);
    btn.querySelector('span').textContent = on ? 'On' : 'Off';
  });
  // Hide the image item in the composer menu when images are off.
  const imgItem = document.getElementById('cmenu-image');
  if (imgItem) imgItem.style.display = pendingCaps.images === false ? 'none' : '';
}
capToggles.forEach(btn => btn.addEventListener('click', () => {
  const k = btn.dataset.cap;
  pendingCaps[k] = pendingCaps[k] === false;   // flip
  applyCapsUI();
}));

document.getElementById('settings-save').addEventListener('click', () => {
  const payload = {
    model: setModel.value,
    vision_model: setVisionModel ? setVisionModel.value : '',
    num_ctx: parseInt(setCtx.value, 10),
    tts_voice: setVoice.value,
    tts_device: setDevice.value,
    tts_enabled: pendingTts,
    char_name: (document.getElementById('set-name')?.value || '').trim() || undefined,
    behavior: { ...pendingBehavior },
    capabilities: { ...pendingCaps },
    file_roots: [...pendingFileRoots],
  };
  if (connected && ws?.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: 'set_settings', settings: payload }));
  }
  applyVoiceState(pendingTts);
  closeSettings();
});

/* ═══ Folders she can read (scoped, opt-in; lives in the composer + menu) ═══
   Changes here persist immediately (the menu isn't behind the Settings Save). */
let pendingFileRoots = [];
const cmenuRootsEl = document.getElementById('cmenu-roots');

function renderFileRoots() {
  if (!cmenuRootsEl) return;
  cmenuRootsEl.innerHTML = '';
  if (!pendingFileRoots.length) {
    cmenuRootsEl.innerHTML = '<div class="cmenu-empty">None yet — she can’t see your files.</div>';
    return;
  }
  pendingFileRoots.forEach((path, i) => {
    const row = document.createElement('div'); row.className = 'cmenu-root';
    const name = path.replace(/[\\/]+$/, '').split(/[\\/]/).pop() || path;
    const p = document.createElement('span'); p.className = 'cmenu-root-name'; p.textContent = name; p.title = path;
    const rm = document.createElement('button'); rm.className = 'cmenu-root-rm'; rm.textContent = '✕';
    rm.title = 'Stop sharing this folder';
    rm.addEventListener('click', (e) => {
      e.stopPropagation();
      pendingFileRoots.splice(i, 1); renderFileRoots(); saveFileRoots();
    });
    row.append(p, rm); cmenuRootsEl.appendChild(row);
  });
}

function addFileRoot(path) {
  path = (path || '').trim().replace(/[\\/]+$/, '');
  if (!path) return;
  if (!pendingFileRoots.some(p => p.toLowerCase() === path.toLowerCase())) {
    pendingFileRoots.push(path);
    saveFileRoots();
  }
  renderFileRoots();
}

function saveFileRoots() {
  if (connected && ws?.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: 'set_settings', settings: { file_roots: [...pendingFileRoots] } }));
  }
}

/* ═══ Spotify — connect, status, and the now-playing widget ═══ */
const npEl       = document.getElementById('nowplaying');
const npTrack    = document.getElementById('np-track');
const spotStatus = document.getElementById('spotify-status');
const spotConnect = document.getElementById('spotify-connect');
const spotDisc   = document.getElementById('spotify-disconnect');
const musicCapNote = document.getElementById('music-cap-note');
let spotConnected = false;

async function refreshSpotify() {
  try {
    const s = await (await fetch('/api/spotify/status')).json();
    spotConnected = !!s.connected;
    if (spotStatus) {
      spotStatus.textContent = !s.configured ? 'Not configured (set keys in .env)'
        : s.connected ? 'Connected ✓' : 'Not connected';
    }
    if (spotConnect) spotConnect.style.display = s.connected ? 'none' : '';
    if (spotDisc) spotDisc.style.display = s.connected ? '' : 'none';
    // Tell him whether the Music capability is actually live: she only sees the
    // music tags in her context when Spotify is connected.
    if (musicCapNote) {
      if (!s.configured) {
        musicCapNote.textContent = '⚠ Spotify not configured — set keys in .env.';
        musicCapNote.className = 'cap-note warn';
      } else if (!s.connected) {
        musicCapNote.textContent = '⚠ Spotify not connected — she can’t see music yet. Connect it in General.';
        musicCapNote.className = 'cap-note warn';
      } else if (s.premium) {
        musicCapNote.textContent = 'Spotify Premium connected ✓ — she can see, control, and build playlists.';
        musicCapNote.className = 'cap-note ok';
      } else {
        musicCapNote.textContent = 'Spotify connected ✓ (Free account) — she can see your taste and build playlists, but can’t control playback.';
        musicCapNote.className = 'cap-note warn';
      }
    }
    // Now-playing widget
    const np = s.now_playing;
    if (npEl) {
      if (s.connected && np) {
        npEl.style.display = '';
        if (npTrack) npTrack.textContent = np.text || '—';
        npEl.classList.toggle('paused', !np.playing);
      } else {
        npEl.style.display = 'none';
      }
    }
  } catch { /* backend not up yet */ }
}

spotConnect?.addEventListener('click', () => {
  window.open('/spotify/login');   // opens in the system browser (handled in main.js)
  // Poll for a bit so the UI flips once he finishes the consent flow.
  let n = 0; const t = setInterval(() => { refreshSpotify(); if (++n > 40 || spotConnected) clearInterval(t); }, 1500);
});
spotDisc?.addEventListener('click', async () => {
  await fetch('/api/spotify/disconnect', { method: 'POST' });
  refreshSpotify();
});

async function spotifyControl(action) {
  await fetch('/api/spotify/control', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action }),
  });
  setTimeout(refreshSpotify, 400);
}
document.getElementById('np-prev')?.addEventListener('click', () => spotifyControl('previous'));
document.getElementById('np-next')?.addEventListener('click', () => spotifyControl('next'));
document.getElementById('np-playpause')?.addEventListener('click', () => {
  spotifyControl(npEl?.classList.contains('paused') ? 'play' : 'pause');
});

// Keep the now-playing widget fresh while connected.
setInterval(() => { if (spotConnected) refreshSpotify(); }, 15000);
refreshSpotify();

/* ═══ Character name — a rename re-skins every visible mention ═══ */
let currentCharName = 'Aitha';
function applyCharName(name) {
  name = (name || '').trim() || 'Aitha';
  const input = document.getElementById('set-name');
  if (input && document.activeElement !== input) input.value = name;
  document.title = name;
  if (name === currentCharName) return;
  const old = currentCharName;
  const re = new RegExp(old.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'g');
  document.querySelectorAll('[placeholder]').forEach(el => {
    if (el.placeholder.includes(old)) el.placeholder = el.placeholder.replace(re, name);
  });
  document.querySelectorAll('[title]').forEach(el => {
    if (el.title.includes(old)) el.title = el.title.replace(re, name);
  });
  // Visible chrome text only — never chat bubbles, notes, or journal content.
  document.querySelectorAll('#char-name, .modal label, .field-hint, .note-empty p, .mantle-title, .mantle-sub')
    .forEach(el => {
      if (el.children.length === 0 && el.textContent.includes(old))
        el.textContent = el.textContent.replace(re, name);
    });
  currentCharName = name;
}

/* ─── Window controls (Electron IPC) ──────────────────────────────── */
document.getElementById('btn-close').addEventListener('click', () => {
  window.electron?.close();
});
document.getElementById('btn-minimize').addEventListener('click', () => {
  window.electron?.minimize();
});
document.getElementById('btn-maximize').addEventListener('click', () => {
  window.electron?.maximize();
});

/* ─── WebSocket ────────────────────────────────────────────────────── */
function connect() {
  if (ws) { try { ws.close(); } catch (_) {} }

  ws = new WebSocket(WS_URL);

  ws.addEventListener('open', () => {
    connected = true;
    clearError();
    setStatus('Watching');
    if (reconnectTimer) { clearTimeout(reconnectTimer); reconnectTimer = null; }
  });

  ws.addEventListener('close', () => {
    connected = false;
    setStatus('Disconnected');
    reconnectTimer = setTimeout(connect, RECONNECT_DELAY);
  });

  ws.addEventListener('error', () => {
    connected = false;
  });

  ws.addEventListener('message', (ev) => {
    let data;
    try { data = JSON.parse(ev.data); } catch { return; }
    handleMessage(data);
  });
}

function handleMessage(data) {
  switch (data.type) {

    case 'context_update':
      updateContext(data.context);
      break;

    case 'history':
      renderHistory(data.messages || []);
      break;

    case 'chat_echo':
      document.getElementById('welcome')?.remove();
      appendMessage(data.role === 'user' ? 'user' : 'aitha', data.content);
      scrollToBottom();
      break;

    case 'aitha_image':
      document.getElementById('welcome')?.remove();
      if (data.url) appendMessage('aitha', '', { images: [data.url] });
      scrollToBottom();
      break;

    case 'notes_changed':
      if (notesLoaded) loadNoteList();
      if (currentNote && (data.titles || []).includes(currentNote)) openNote(currentNote);
      break;

    case 'activity':
      showExploring(data.state === 'exploring', data.label);
      break;

    case 'hearth_state':
      onHearthState(data);
      break;

    case 'hearth_roll':
      showDiceResult(data.roll, data.who);
      break;

    case 'directives':
      if (showDirectives) renderDirectives(data.blocks || []);
      break;

    case 'memory_changed':
      if (memLaneOpen) loadMemLane(true);   // she kept/forgot one mid-chat — refresh the list
      break;

    case 'projects_changed':
      if (activeView === 'mantle') loadMind();   // she started/advanced a project — refresh her mind
      break;

    case 'calendar_changed':
      if (bedrockOpen) loadCalendar();           // she jotted an event — refresh Bedrock
      break;

    case 'spotify_changed':
      refreshSpotify();                          // playback/connection changed — refresh widget
      break;

    case 'tts_state':
      applyVoiceState(data.enabled);
      break;

    case 'speaking':
      setAithaSpeaking(!!data.on);
      break;

    case 'theme':
      applyTheme(data.theme || {}, data.by);
      break;

    case 'char_name':
      applyCharName(data.name);
      break;

    case 'settings':
      populateSettings(data);
      break;

    case 'searching':
      showSearching(true);
      setOrbState('searching');
      setStatus('Searching...');
      break;

    case 'token':
      showSearching(false);
      typingEl.style.display = 'none';
      setOrbState('thinking');
      document.getElementById('welcome')?.remove();  // clear placeholder (incl. greeting)

      if (!streamingBubble) {
        const container = memLaneOpen ? document.getElementById('memlane-thread') : messagesEl;
        streamingBubble = appendBubble('aitha', '', container);
        streamingBubble.classList.add('streaming');
        streamingContent = '';
      }
      streamingContent += data.content;
      streamingBubble.textContent = streamingContent;
      if (memLaneOpen) scrollThread(); else scrollToBottom();
      break;

    case 'done':
      if (streamingBubble) {
        streamingBubble.classList.remove('streaming');
        streamingBubble = null;
        streamingContent = '';
      }
      setOrbState('idle');
      setStatus(data.cancelled ? 'Cancelled' : 'Watching');
      showSearching(false);
      typingEl.style.display = 'none';
      setGenerating(false);
      scrollToBottom();
      break;
  }
}

/* ─── Context display ──────────────────────────────────────────────── */
function updateContext(ctx) {
  if (!ctx) return;
  if (ctx.time) clockTime.textContent = ctx.time;
  if (ctx.day) clockDay.textContent = ctx.day.slice(0, 3);
}

function shortModel(name) {
  return (name.split('/').pop() || name).replace(':latest', '');
}

/* ─── Send / cancel ────────────────────────────────────────────────── */
const SEND_SVG = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="22" y1="2" x2="11" y2="13"></line><polygon points="22 2 15 22 11 13 2 9 22 2"></polygon></svg>';
const STOP_SVG = '<svg width="15" height="15" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="6" width="12" height="12" rx="2.5"></rect></svg>';
let generating = false;

function setGenerating(on) {
  generating = on;
  inputEl.disabled = on;
  sendBtn.disabled = false;            // stays clickable so it can cancel
  sendBtn.classList.toggle('stop', on);
  sendBtn.title = on ? 'Stop' : 'Send';
  sendBtn.innerHTML = on ? STOP_SVG : SEND_SVG;
  if (!on) inputEl.focus();
}

function send() {
  const text = inputEl.value.trim();
  const images = pendingImages.slice();
  if ((!text && !images.length) || !connected) return;

  welcomeEl?.remove();
  document.getElementById('welcome')?.remove();
  appendMessage('user', text, { images });
  scrollToBottom();

  inputEl.value = '';
  pendingImages = [];
  renderAttachTray();
  autoResize();
  setGenerating(true);

  typingEl.style.display = 'flex';
  setOrbState('thinking');
  setStatus('Thinking...');

  ws.send(JSON.stringify({ type: 'chat', message: text, images }));
}

function cancelGeneration() {
  if (connected && ws?.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: 'cancel' }));
  }
  setStatus('Cancelled');
}

/* ─── Input handling ───────────────────────────────────────────────── */
inputEl.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    send();
  }
});

sendBtn.addEventListener('click', () => {
  if (generating) cancelGeneration();
  else send();
});

inputEl.addEventListener('input', autoResize);

function autoResize() {
  inputEl.style.height = 'auto';
  inputEl.style.height = Math.min(inputEl.scrollHeight, 140) + 'px';
}

/* ═══════════════════════════════════════════════════════════════════
   HANDS-FREE VOICE INPUT
   Mic → Web Audio VAD → record an utterance → POST /api/stt → send.
   She mutes herself while speaking (the backend 'speaking' signal) so she
   never hears or transcribes her own voice.
   ═══════════════════════════════════════════════════════════════════ */
const micToggle = document.getElementById('mic-toggle');
const micLabel  = document.getElementById('mic-label');
const vlEl      = document.getElementById('voice-listening');
const vlBars    = [...document.querySelectorAll('#vl-bars i')];
const vlLabel   = document.getElementById('vl-label');

let micOn        = localStorage.getItem('micOn') === '1';
let micDeviceId  = localStorage.getItem('micDeviceId') || '';
// Whether to mute the mic while she's speaking (default on). Off suits headphone
// users, where her TTS never bleeds into the mic.
let micGate      = localStorage.getItem('micGate') !== '0';
let audioStream = null, audioCtx = null, analyser = null, timeBuf = null, rafId = null;
let recorder = null, chunks = [], recording = false, recDiscard = false;
let aithaSpeaking = false;
let voiceStart = 0, lastVoiceTs = 0, recStart = 0;

const VAD = {
  START: 0.05,       // RMS to begin capturing speech
  STOP: 0.025,       // RMS below this counts as silence
  ONSET_MS: 70,      // sustained loudness before we commit to recording
  SILENCE_MS: 900,   // trailing silence that ends an utterance
  MIN_MS: 350,       // ignore blips shorter than this
  MAX_MS: 20000,     // hard safety cap on one utterance
};

function rmsOf(buf) {
  let s = 0;
  for (let i = 0; i < buf.length; i++) { const v = (buf[i] - 128) / 128; s += v * v; }
  return Math.sqrt(s / buf.length);
}

function setVlState(state, label) {
  vlEl.className = 'voice-listening ' + (state || '');
  vlLabel.textContent = label || (state === 'recording' ? 'Hearing you…'
                                : state === 'busy' ? '…'
                                : 'Listening…');
  if (state !== 'recording') vlBars.forEach(b => b.style.height = '');
}

function drawBars(level) {
  const amp = Math.min(1, level * 4.5);
  const t = performance.now() / 120;
  vlBars.forEach((b, i) => {
    const k = 0.45 + 0.55 * Math.abs(Math.sin(t + i * 0.7));
    b.style.height = (4 + amp * 18 * k).toFixed(1) + 'px';
  });
}

function micConstraints() {
  return {
    deviceId: micDeviceId ? { exact: micDeviceId } : undefined,
    echoCancellation: true, noiseSuppression: true, autoGainControl: true,
  };
}

async function startMic() {
  try {
    audioStream = await navigator.mediaDevices.getUserMedia({ audio: micConstraints() });
  } catch (e) {
    console.warn('[voice] mic unavailable:', e);
    micOn = false; localStorage.setItem('micOn', '0'); applyMicUI();
    return;
  }
  audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  const src = audioCtx.createMediaStreamSource(audioStream);
  analyser = audioCtx.createAnalyser();
  analyser.fftSize = 1024;
  timeBuf = new Uint8Array(analyser.fftSize);
  src.connect(analyser);
  recording = false; voiceStart = 0;
  vlEl.style.display = 'flex';
  setVlState(aithaSpeaking ? 'gated' : 'idle');
  rafId = requestAnimationFrame(tick);
  populateMicDevices();  // labels are available now that permission is granted
}

function stopMic() {
  if (rafId) { cancelAnimationFrame(rafId); rafId = null; }
  if (recording && recorder) { recDiscard = true; try { recorder.stop(); } catch (_) {} }
  recording = false;
  micToggle.classList.remove('live');
  if (audioStream) { audioStream.getTracks().forEach(t => t.stop()); audioStream = null; }
  if (audioCtx) { audioCtx.close().catch(() => {}); audioCtx = null; }
  analyser = null;
  vlEl.style.display = 'none';
}

function tick() {
  rafId = requestAnimationFrame(tick);
  if (!analyser) return;
  analyser.getByteTimeDomainData(timeBuf);
  const level = rmsOf(timeBuf);
  if (recording) drawBars(level);
  if (aithaSpeaking) return;   // she's talking — don't listen to her own voice
  const now = performance.now();
  if (!recording) {
    if (level > VAD.START) {
      if (!voiceStart) voiceStart = now;
      else if (now - voiceStart > VAD.ONSET_MS) startRec(now);
    } else voiceStart = 0;
  } else {
    if (level > VAD.STOP) lastVoiceTs = now;
    if (now - lastVoiceTs > VAD.SILENCE_MS || now - recStart > VAD.MAX_MS) stopRec();
  }
}

function startRec(now) {
  if (recording || !audioStream) return;
  try { recorder = new MediaRecorder(audioStream); } catch (e) { return; }
  chunks = []; recDiscard = false; recording = true; recStart = now; lastVoiceTs = now;
  recorder.ondataavailable = e => { if (e.data && e.data.size) chunks.push(e.data); };
  recorder.onstop = () => {
    const blob = new Blob(chunks, { type: recorder.mimeType || 'audio/webm' });
    const dur = performance.now() - recStart;
    if (recDiscard) { if (micOn && !aithaSpeaking) setVlState('idle'); return; }
    onUtterance(blob, dur);
  };
  recorder.start();
  micToggle.classList.add('live');
  setVlState('recording', 'Hearing you…');
}

function stopRec() {
  if (!recording) return;
  recording = false;
  micToggle.classList.remove('live');
  try { recorder.stop(); } catch (_) {}
}

function onUtterance(blob, durMs) {
  if (durMs < VAD.MIN_MS) { if (micOn && !aithaSpeaking) setVlState('idle'); return; }
  setVlState('busy', '…');
  transcribe(blob).then(text => {
    if (text) submitTranscript(text);
    if (micOn && !aithaSpeaking) setVlState('idle');
  });
}

async function transcribe(blob) {
  try {
    const res = await fetch('/api/stt', {
      method: 'POST',
      headers: { 'Content-Type': blob.type || 'application/octet-stream' },
      body: blob,
    });
    const data = await res.json();
    return (data.text || '').trim();
  } catch (e) {
    console.warn('[voice] transcribe failed:', e);
    return '';
  }
}

function submitTranscript(text) {
  if (!text || !connected) return;
  if (generating) cancelGeneration();   // barge-in: she yields to your voice
  inputEl.value = text;
  send();
}

// Called from the WS 'speaking' signal — mute the mic while she talks. The TTS
// queue can briefly drain between sentences, so we hold the mute for a short
// grace period and only release if she stays silent (no flicker mid-speech).
let speakingOffTimer = null;
function setAithaSpeaking(on) {
  if (!micGate) return;   // headphone mode — never gate the mic on her speech
  if (speakingOffTimer) { clearTimeout(speakingOffTimer); speakingOffTimer = null; }
  if (on) {
    aithaSpeaking = true;
    if (recording && recorder) { recDiscard = true; stopRec(); }
    if (micOn) setVlState('gated', 'Aitha’s speaking…');
  } else {
    speakingOffTimer = setTimeout(() => {
      aithaSpeaking = false;
      if (micOn && !recording) setVlState('idle');
    }, 1200);
  }
}

function applyMicUI() {
  micToggle.classList.toggle('off', !micOn);
  micLabel.textContent = micOn ? 'Listening' : 'Listen';
  syncPassthroughControls();
}

micToggle.addEventListener('click', async () => {
  micOn = !micOn;
  localStorage.setItem('micOn', micOn ? '1' : '0');
  applyMicUI();
  if (micOn) await startMic(); else stopMic();
});

/* ─── Settings: "mute mic while she speaks" gate ───────────────────────
   Lives here (not up in the settings block) so it runs after micGate and the
   voice state above are initialized — referencing them earlier would throw. */
const setMicGateToggle = document.getElementById('set-micgate-toggle');
function applyMicGateUI() {
  if (!setMicGateToggle) return;
  setMicGateToggle.classList.toggle('off', !micGate);
  setMicGateToggle.querySelector('span').textContent = micGate ? 'On' : 'Off';
}
applyMicGateUI();
setMicGateToggle?.addEventListener('click', () => {
  micGate = !micGate;
  localStorage.setItem('micGate', micGate ? '1' : '0');
  applyMicGateUI();
  // Turning the gate off mid-speech should free the mic immediately.
  if (!micGate && aithaSpeaking) {
    if (speakingOffTimer) { clearTimeout(speakingOffTimer); speakingOffTimer = null; }
    aithaSpeaking = false;
    if (micOn && !recording) setVlState('idle');
  }
});

async function populateMicDevices() {
  const sel = document.getElementById('set-mic');
  if (!sel) return;
  try {
    const devs = await navigator.mediaDevices.enumerateDevices();
    const mics = devs.filter(d => d.kind === 'audioinput');
    sel.innerHTML = '';
    const def = document.createElement('option');
    def.value = ''; def.textContent = 'System default';
    sel.appendChild(def);
    mics.forEach((m, i) => {
      const o = document.createElement('option');
      o.value = m.deviceId;
      o.textContent = m.label || ('Microphone ' + (i + 1));
      sel.appendChild(o);
    });
    sel.value = micDeviceId;
  } catch (_) {}
}

document.getElementById('set-mic')?.addEventListener('change', (e) => {
  micDeviceId = e.target.value;
  localStorage.setItem('micDeviceId', micDeviceId);
  if (micOn) { stopMic(); startMic(); }
});
settingsBtn.addEventListener('click', populateMicDevices);

// Restore saved preference on launch.
applyMicUI();
if (micOn) startMic();

/* ═══════════════════════════════════════════════════════════════════
   THEMING — shared between her and him. The backend is the source of truth:
   the UI only SENDS changes; we apply on the broadcast echo so both stay synced.
   ═══════════════════════════════════════════════════════════════════ */
const PRESET_DEFAULTS = {
  default: { accent: '#a78bfa', bg: '#07070e', orb: '#a78bfa' },
  sky:     { accent: '#5ca9e8', bg: '#0a1018', orb: '#5ca9e8' },
  warm:    { accent: '#f5b14b', bg: '#140a06', orb: '#f5b14b' },
  moody:   { accent: '#6d8bd0', bg: '#05060d', orb: '#6d8bd0' },
  magma:   { accent: '#f43f5e', bg: '#140609', orb: '#f43f5e' },
  hearth:  { accent: '#f5b14b', bg: '#140d05', orb: '#f5b14b' },
  forest:  { accent: '#43c59e', bg: '#08130d', orb: '#43c59e' },
  rose:    { accent: '#f472b6', bg: '#160810', orb: '#f472b6' },
  ocean:   { accent: '#2dd4bf', bg: '#061413', orb: '#2dd4bf' },
  mono:    { accent: '#9aa7b8', bg: '#0b0d12', orb: '#9aa7b8' },
};
const ALL_PRESET_CLASSES = ['sky', 'warm', 'moody', 'magma', 'hearth', 'forest', 'rose', 'ocean', 'mono']
  .map(p => 'chat-theme-' + p);

// Each app-tab remembers its own full theme. Sky (chat) mirrors the backend —
// it's shared with Aitha (two-way). Mantle/Magma/Hearth are local display prefs.
const TAB_DEFAULTS = {
  chat:   { preset: 'default', accent: null, bg: null, orb: null },
  mantle: { preset: 'moody',   accent: null, bg: null, orb: null },
  notes:  { preset: 'magma',   accent: null, bg: null, orb: null },
  hearth: { preset: 'hearth',  accent: null, bg: null, orb: null },
};
const VIEW_ORDER = ['chat', 'mantle', 'notes', 'hearth'];

function emptyTheme() { return { preset: 'default', accent: null, bg: null, orb: null }; }

function loadTabThemes() {
  let saved = {};
  try { saved = JSON.parse(localStorage.getItem('tabThemes') || '{}') || {}; } catch (_) {}
  const out = {};
  for (const v of VIEW_ORDER) out[v] = Object.assign({}, TAB_DEFAULTS[v], saved[v] || {});
  return out;
}
function saveTabThemes() {
  try { localStorage.setItem('tabThemes', JSON.stringify(tabThemes)); } catch (_) {}
}

let tabThemes  = loadTabThemes();
let activeView = 'chat';        // which app-tab is on screen
let editTarget = 'chat';        // which tab the Appearance editor is editing
let currentTheme = tabThemes.chat;   // alias for the Sky/chat theme (backend-synced)

function hexToRgb(hex) {
  const m = /^#?([0-9a-f]{6})$/i.exec((hex || '').trim());
  if (!m) return null;
  const n = parseInt(m[1], 16);
  return { r: (n >> 16) & 255, g: (n >> 8) & 255, b: n & 255 };
}
function rgbStr(hex) { const c = hexToRgb(hex); return c ? `${c.r}, ${c.g}, ${c.b}` : null; }
function mix(hex, target, amt) {
  const c = hexToRgb(hex); if (!c) return hex;
  const t = target === 'white' ? 255 : 0;
  const f = ch => Math.round(ch + (t - ch) * amt);
  return `${f(c.r)}, ${f(c.g)}, ${f(c.b)}`;
}

// Pure CSS applier — paints whatever theme object it's given onto <body>. The
// active app-tab decides which theme this gets called with.
function applyThemeObject(theme) {
  const t = Object.assign(emptyTheme(), theme || {});
  const root = document.body.style;
  // Big look: preset via body class (default needs none — it's :root).
  document.body.classList.remove(...ALL_PRESET_CLASSES);
  if (t.preset && t.preset !== 'default') {
    document.body.classList.add('chat-theme-' + t.preset);
  }
  // Clear prior fine-tune overrides, then re-apply.
  ['--accent-v', '--accent-v-bright', '--accent-v-dim', '--accent-v-glow',
   '--border-accent', '--bg-deep', '--bg-mid',
   '--orb-rgb', '--orb-core', '--orb-deep'].forEach(p => root.removeProperty(p));
  if (t.accent && rgbStr(t.accent)) {
    const rgb = rgbStr(t.accent);
    root.setProperty('--accent-v', t.accent);
    root.setProperty('--accent-v-bright', `rgb(${mix(t.accent,'white',0.3)})`);
    root.setProperty('--accent-v-dim', `rgba(${rgb}, 0.15)`);
    root.setProperty('--accent-v-glow', `rgba(${rgb}, 0.25)`);
    root.setProperty('--border-accent', `rgba(${rgb}, 0.30)`);
  }
  if (t.bg && hexToRgb(t.bg)) {
    root.setProperty('--bg-deep', t.bg);
    root.setProperty('--bg-mid', `rgb(${mix(t.bg,'white',0.06)})`);
  }
  if (t.orb && rgbStr(t.orb)) {
    root.setProperty('--orb-rgb', rgbStr(t.orb));
    root.setProperty('--orb-core', mix(t.orb, 'white', 0.4));
    root.setProperty('--orb-deep', mix(t.orb, 'black', 0.35));
  }
}

// Paint the theme belonging to a given app-tab (used on tab switch).
function showTabTheme(view) {
  applyThemeObject(tabThemes[view] || TAB_DEFAULTS[view] || {});
}

// Backend echo for the SHARED Sky/chat theme (she or he changed it).
function applyTheme(theme, by) {
  tabThemes.chat = Object.assign(emptyTheme(), theme || {});
  currentTheme = tabThemes.chat;
  saveTabThemes();
  if (activeView === 'chat') applyThemeObject(currentTheme);
  if (editTarget === 'chat') syncThemeControls();
  if (by === 'her') themeToast('Aitha changed the look');
}

function themeForEdit() { return tabThemes[editTarget] || TAB_DEFAULTS[editTarget] || emptyTheme(); }

function syncThemeControls() {
  const t = themeForEdit();
  document.querySelectorAll('.binder-tab').forEach(b =>
    b.classList.toggle('active', b.dataset.tabtarget === editTarget));
  document.querySelectorAll('.theme-card').forEach(c =>
    c.classList.toggle('active', c.dataset.preset === t.preset));
  const d = PRESET_DEFAULTS[t.preset] || PRESET_DEFAULTS.default;
  const accent = document.getElementById('theme-accent');
  const bg = document.getElementById('theme-bg');
  const orb = document.getElementById('theme-orb');
  if (accent) accent.value = t.accent || d.accent;
  if (bg) bg.value = t.bg || d.bg;
  if (orb) orb.value = t.orb || d.orb;
}

function sendTheme(patch) {
  if (connected && ws?.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: 'set_theme', theme: patch }));
  }
}

// Apply an edit to whichever tab the binder is pointed at. Sky routes through
// the backend (shared with Aitha); the rest are local + persisted.
function editTheme(patch) {
  if (editTarget === 'chat') { sendTheme(patch); return; }  // echoes back via applyTheme
  tabThemes[editTarget] = Object.assign({}, tabThemes[editTarget], patch);
  saveTabThemes();
  if (activeView === editTarget) applyThemeObject(tabThemes[editTarget]);
  syncThemeControls();
}

function restoreTabDefault() {
  const def = Object.assign(emptyTheme(), TAB_DEFAULTS[editTarget]);
  if (editTarget === 'chat') { sendTheme(def); return; }
  tabThemes[editTarget] = def;
  saveTabThemes();
  if (activeView === editTarget) applyThemeObject(def);
  syncThemeControls();
}

document.getElementById('theme-binder')?.addEventListener('click', (e) => {
  const b = e.target.closest('.binder-tab');
  if (!b) return;
  editTarget = b.dataset.tabtarget;
  syncThemeControls();
});
document.getElementById('theme-presets')?.addEventListener('click', (e) => {
  const card = e.target.closest('.theme-card');
  if (card) editTheme({ preset: card.dataset.preset });
});
document.getElementById('theme-accent')?.addEventListener('change', e => editTheme({ accent: e.target.value }));
document.getElementById('theme-bg')?.addEventListener('change', e => editTheme({ bg: e.target.value }));
document.getElementById('theme-orb')?.addEventListener('change', e => editTheme({ orb: e.target.value }));
document.getElementById('theme-reset')?.addEventListener('click', restoreTabDefault);
document.getElementById('mantle-refresh')?.addEventListener('click', () => loadMind());

let themeToastTimer = null;
function themeToast(msg) {
  let el = document.getElementById('theme-toast');
  if (!el) {
    el = document.createElement('div');
    el.id = 'theme-toast';
    el.className = 'theme-toast';
    document.body.appendChild(el);
  }
  el.textContent = msg;
  el.classList.add('show');
  if (themeToastTimer) clearTimeout(themeToastTimer);
  themeToastTimer = setTimeout(() => el.classList.remove('show'), 2600);
}

/* ─── Exploring indicator ──────────────────────────────────────────── */
const exploreEl = document.getElementById('explore-indicator');
function showExploring(on, label) {
  if (on && label) document.getElementById('explore-label').textContent = label;
  exploreEl.style.display = on ? 'flex' : 'none';
}

/* ─── Directive debug blocks (shown when "Show note tags" is on) ────── */
function renderDirectives(blocks) {
  if (!blocks.length) return;
  document.getElementById('welcome')?.remove();
  const wrap = document.createElement('div');
  wrap.className = 'directive-block';
  wrap.textContent = blocks.map(b => `‹${b.kind}›\n${b.text}`).join('\n\n');
  messagesEl.appendChild(wrap);
  scrollToBottom();
}

/* ─── Restore prior conversation on (re)connect ────────────────────── */
function renderHistory(msgs) {
  document.getElementById('welcome')?.remove();
  // Rebuild from scratch so a reconnect doesn't duplicate bubbles.
  messagesEl.innerHTML = '';
  streamingBubble = null;
  streamingContent = '';
  for (const m of msgs) {
    appendMessage(m.role === 'user' ? 'user' : 'aitha', m.content, { images: m.images });
  }
  scrollToBottom();
}

/* ─── Bubble factory ───────────────────────────────────────────────── */
function appendBubble(role, text, container = messagesEl) {
  const row = document.createElement('div');
  row.className = `message ${role}`;

  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.textContent = text;

  row.appendChild(bubble);
  container.appendChild(row);
  return bubble;
}

/* ─── Image-aware message (text bubble + any image bubbles, stacked) ──── */
const IMG_TAG_RE = /<image>\s*([^<\s][^<]*?)\s*<\/image>/gi;

function addBubbleImage(col, url) {
  const img = document.createElement('img');
  img.className = 'bubble-img';
  img.src = url;
  img.alt = '';
  img.loading = 'lazy';
  img.addEventListener('click', () => window.open(url, '_blank'));
  img.addEventListener('error', () => {
    img.replaceWith(Object.assign(document.createElement('div'),
      { className: 'bubble-img-broken', textContent: 'image couldn’t load' }));
  });
  col.appendChild(img);
}

// role 'user'|'aitha'; text may contain <image>…</image> tags (aitha history);
// opts.images is an explicit list of urls to render under the text.
function appendMessage(role, text, opts = {}) {
  const container = opts.container || messagesEl;
  let body = text || '';
  let images = opts.images ? opts.images.slice() : [];

  if (role === 'aitha' && body) {
    const found = [...body.matchAll(IMG_TAG_RE)].map(m => m[1].trim());
    if (found.length) { images = images.concat(found); body = body.replace(IMG_TAG_RE, '').trim(); }
  }

  const row = document.createElement('div');
  row.className = `message ${role}`;
  const col = document.createElement('div');
  col.className = 'msg-col';

  if (body) {
    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    bubble.textContent = body;
    col.appendChild(bubble);
  }
  for (const u of images) addBubbleImage(col, u);

  row.appendChild(col);
  container.appendChild(row);
  return row;
}

/* ─── Image attachments (drop / paste / pick → send to her) ──────────── */
const MAX_ATTACH = 4;
const MAX_IMG_DIM = 1024;          // downscale longest side before sending
let pendingImages = [];
const attachTray  = document.getElementById('attach-tray');
const attachInput = document.getElementById('attach-input');

function renderAttachTray() {
  if (!attachTray) return;
  attachTray.innerHTML = '';
  attachTray.style.display = pendingImages.length ? 'flex' : 'none';
  pendingImages.forEach((url, i) => {
    const chip = document.createElement('div');
    chip.className = 'attach-chip';
    const img = document.createElement('img'); img.src = url;
    const rm = document.createElement('button');
    rm.className = 'attach-rm'; rm.textContent = '✕'; rm.title = 'Remove';
    rm.addEventListener('click', () => { pendingImages.splice(i, 1); renderAttachTray(); });
    chip.append(img, rm);
    attachTray.appendChild(chip);
  });
}

function loadImg(src) {
  return new Promise((res, rej) => {
    const im = new Image(); im.onload = () => res(im); im.onerror = rej; im.src = src;
  });
}

// Read a file → data URL, downscaling big images (keeps base64 payload sane for
// both Ollama and cloud token cost).
async function fileToDataURL(file) {
  const raw = await new Promise((res, rej) => {
    const r = new FileReader(); r.onload = () => res(r.result); r.onerror = rej;
    r.readAsDataURL(file);
  });
  try {
    const img = await loadImg(raw);
    const { width: w, height: h } = img;
    if (Math.max(w, h) <= MAX_IMG_DIM && raw.length < 700000) return raw;
    const scale = Math.min(1, MAX_IMG_DIM / Math.max(w, h));
    const c = document.createElement('canvas');
    c.width = Math.round(w * scale); c.height = Math.round(h * scale);
    c.getContext('2d').drawImage(img, 0, 0, c.width, c.height);
    return c.toDataURL('image/jpeg', 0.85);
  } catch { return raw; }
}

async function addImageFile(file) {
  if (!file || !file.type?.startsWith('image/')) return;
  if (pendingImages.length >= MAX_ATTACH) return;
  const url = await fileToDataURL(file);
  if (pendingImages.length < MAX_ATTACH) { pendingImages.push(url); renderAttachTray(); }
}

attachInput?.addEventListener('change', () => {
  [...attachInput.files].forEach(addImageFile);
  attachInput.value = '';
});

/* ─── Composer "+" menu (hover-expand): image + the folders she can read ── */
const addWrap   = document.getElementById('composer-add-wrap');
const addBtn    = document.getElementById('composer-add');
const addMenu   = document.getElementById('composer-menu');
let menuOpenTimer = null, menuCloseTimer = null;

function openComposerMenu() {
  clearTimeout(menuCloseTimer);
  if (!addMenu || addMenu.classList.contains('open')) return;
  renderFileRoots();
  addMenu.classList.add('open');
  addMenu.setAttribute('aria-hidden', 'false');
  addBtn?.setAttribute('aria-expanded', 'true');
}
function closeComposerMenu() {
  clearTimeout(menuOpenTimer);
  if (!addMenu) return;
  addMenu.classList.remove('open');
  addMenu.setAttribute('aria-hidden', 'true');
  addBtn?.setAttribute('aria-expanded', 'false');
}
// Hover for half a second to expand (Claude-style); click toggles immediately.
addWrap?.addEventListener('mouseenter', () => {
  clearTimeout(menuCloseTimer);
  menuOpenTimer = setTimeout(openComposerMenu, 500);
});
addWrap?.addEventListener('mouseleave', () => {
  clearTimeout(menuOpenTimer);
  menuCloseTimer = setTimeout(closeComposerMenu, 220);
});
addBtn?.addEventListener('click', (e) => {
  e.stopPropagation();
  addMenu?.classList.contains('open') ? closeComposerMenu() : openComposerMenu();
});
document.addEventListener('click', (e) => {
  if (addMenu?.classList.contains('open') && !addWrap.contains(e.target)) closeComposerMenu();
});
document.getElementById('cmenu-image')?.addEventListener('click', () => {
  closeComposerMenu();
  attachInput?.click();
});
document.getElementById('cmenu-addfolder')?.addEventListener('click', async () => {
  const picked = await window.electron?.pickFolder?.();
  if (picked) addFileRoot(picked);
});

inputEl.addEventListener('paste', (e) => {
  const imgs = [...(e.clipboardData?.items || [])].filter(it => it.type.startsWith('image/'));
  if (imgs.length) { e.preventDefault(); imgs.forEach(it => addImageFile(it.getAsFile())); }
});

const dropZone = document.getElementById('view-chat');
['dragenter', 'dragover'].forEach(ev => dropZone?.addEventListener(ev, (e) => {
  if ([...(e.dataTransfer?.types || [])].includes('Files')) {
    e.preventDefault(); dropZone.classList.add('drag-over');
  }
}));
['dragleave', 'drop'].forEach(ev => dropZone?.addEventListener(ev, () => dropZone.classList.remove('drag-over')));
dropZone?.addEventListener('drop', (e) => {
  const files = [...(e.dataTransfer?.files || [])].filter(f => f.type.startsWith('image/'));
  if (files.length) { e.preventDefault(); files.forEach(addImageFile); }
});

/* ─── Orb states ───────────────────────────────────────────────────── */
function setOrbState(state) {
  orb.classList.remove('thinking', 'searching');
  if (state === 'thinking') orb.classList.add('thinking');
  if (state === 'searching') orb.classList.add('searching');
}

/* ─── Status label ─────────────────────────────────────────────────── */
function setStatus(text) {
  aithaStatus.textContent = text;
}

/* ─── Indicators ───────────────────────────────────────────────────── */
function showSearching(show) {
  searchEl.style.display = show ? 'flex' : 'none';
  // While searching, the "thinking" dots make no sense — hide them.
  if (show) typingEl.style.display = 'none';
}

/* ─── Error banner ─────────────────────────────────────────────────── */
function clearError() {
  document.querySelectorAll('.error-banner').forEach(el => el.remove());
}

/* ─── Scroll ───────────────────────────────────────────────────────── */
function scrollToBottom() {
  messagesEl.scrollTo({ top: messagesEl.scrollHeight, behavior: 'smooth' });
}

/* ─── Helpers ──────────────────────────────────────────────────────── */
function truncate(str, max) {
  return str.length > max ? str.slice(0, max - 1) + '…' : str;
}

function capitalize(str) {
  return str.charAt(0).toUpperCase() + str.slice(1);
}

/* ═══════════════════════════════════════════════════════════════════
   NAVIGATION HUB
   ═══════════════════════════════════════════════════════════════════ */
const navItems = document.querySelectorAll('.nav-item');
const views = {
  chat: document.getElementById('view-chat'),
  mantle: document.getElementById('view-mantle'),
  notes: document.getElementById('view-notes'),
  hearth: document.getElementById('view-hearth'),
};
let notesLoaded = false;
let hearthLoaded = false;

navItems.forEach(item => {
  item.addEventListener('click', () => switchView(item.dataset.view));
});

function switchView(name) {
  activeView = name;
  // Each tab wears its own theme; switching re-themes the whole view.
  showTabTheme(name);
  navItems.forEach(i => i.classList.toggle('active', i.dataset.view === name));
  Object.entries(views).forEach(([k, el]) => el && el.classList.toggle('active', k === name));
  if (name === 'notes' && !notesLoaded) { notesLoaded = true; loadNoteList(); }
  if (name === 'hearth' && !hearthLoaded) { hearthLoaded = true; loadHearth(); }
  if (name === 'mantle') loadMind();   // refresh each visit — her mind moves
  if (name !== 'mantle') closeMemLane();   // don't leave the lane (and its matrix) running
  if (name !== 'notes') closeBedrock();    // leaving Magma closes the calendar slide-over
  if (name === 'chat') inputEl.focus();
}

/* ═══════════════════════════════════════════════════════════════════
   MANTLE — a read-only window into her inner life (mood, thoughts, etc.)
   ═══════════════════════════════════════════════════════════════════ */
async function loadMind() {
  const moodEl = document.getElementById('mantle-mood');
  const jEl = document.getElementById('mantle-journal');
  const dEl = document.getElementById('mantle-discoveries');
  const cEl = document.getElementById('mantle-core');
  const pEl = document.getElementById('mantle-projects');
  if (!jEl) return;
  jEl.classList.add('loading');
  try {
    const data = await (await fetch('/api/mind')).json();
    if (moodEl) moodEl.textContent = data.mood || '—';

    jEl.innerHTML = '';
    const journal = data.journal || [];
    if (!journal.length) jEl.innerHTML = '<div class="mantle-empty">No thoughts written down yet.</div>';
    journal.forEach(e => {
      const row = document.createElement('div'); row.className = 'mantle-entry';
      const t = document.createElement('div'); t.className = 'me-time'; t.textContent = e.time;
      const b = document.createElement('div'); b.className = 'me-text'; b.textContent = e.text;
      row.append(t, b); jEl.appendChild(row);
    });

    dEl.innerHTML = '';
    const disc = data.discoveries || [];
    if (!disc.length) dEl.innerHTML = '<div class="mantle-empty">She hasn’t wandered off lately.</div>';
    disc.forEach(d => {
      const row = document.createElement('div'); row.className = 'mantle-disc';
      const h = document.createElement('div'); h.className = 'md-title'; h.textContent = d.title;
      const b = document.createElement('div'); b.className = 'md-text'; b.textContent = d.text;
      row.append(h, b);
      row.title = 'Open in Magma';
      row.addEventListener('click', () => {
        document.querySelector('.nav-item[data-view="notes"]')?.click();
        setTimeout(() => openNote("Aitha's Discoveries"), 120);
      });
      dEl.appendChild(row);
    });

    cEl.innerHTML = '';
    const core = data.core || { self: [], him: [] };
    if (!core.self.length && !core.him.length)
      cEl.innerHTML = '<div class="mantle-empty">Nothing marked core yet.</div>';
    const add = (label, arr, cls) => arr.forEach(t => {
      const row = document.createElement('div'); row.className = 'mantle-core-item ' + cls;
      const tag = document.createElement('span'); tag.className = 'mc-tag'; tag.textContent = label;
      const s = document.createElement('span'); s.textContent = t;
      row.append(tag, s); cEl.appendChild(row);
    });
    add('her', core.self, 'self');
    add('him', core.him, 'him');

    if (pEl) {
      pEl.innerHTML = '';
      const projects = data.projects || [];
      const active = projects.filter(p => p.status === 'active');
      const rest = projects.filter(p => p.status !== 'active');
      if (!projects.length) {
        pEl.innerHTML = '<div class="mantle-empty">Nothing she’s working on yet.</div>';
      }
      active.forEach(p => {
        const row = document.createElement('div'); row.className = 'mantle-project';
        const head = document.createElement('div'); head.className = 'mp-head';
        const ttl = document.createElement('span'); ttl.className = 'mp-title'; ttl.textContent = p.title;
        head.appendChild(ttl);
        if (p.private) {
          const lock = document.createElement('span'); lock.className = 'mp-private';
          lock.textContent = 'private'; lock.title = 'She’s keeping this one to herself';
          head.appendChild(lock);
        }
        row.appendChild(head);
        if (p.about) {
          const ab = document.createElement('div'); ab.className = 'mp-about'; ab.textContent = p.about;
          row.appendChild(ab);
        }
        const last = (p.log || [])[p.log.length - 1];
        if (last) {
          const lg = document.createElement('div'); lg.className = 'mp-last';
          lg.textContent = `${last.time} — ${last.note}`;
          row.appendChild(lg);
        }
        pEl.appendChild(row);
      });
      if (rest.length) {
        const done = rest.map(p => p.title);
        const foot = document.createElement('div'); foot.className = 'mp-foot';
        foot.textContent = (rest.some(p => p.status === 'done') ? 'finished/shelved: ' : 'shelved: ') + done.join(', ');
        pEl.appendChild(foot);
      }
    }
  } catch (e) {
    if (moodEl) moodEl.textContent = '(couldn’t reach her mind right now)';
  } finally {
    jEl.classList.remove('loading');
  }
}

/* ═══════════════════════════════════════════════════════════════════
   MEMORY LANE — a slide-in panel over Mantle: all her memories, a Matrix
   backdrop, and a chat box wired to the SAME conversation so you can
   reminisce and prune her context together.
   ═══════════════════════════════════════════════════════════════════ */
let memLaneOpen = false;
let matrixRAF = null;

const memLaneEl   = document.getElementById('memlane');
const memThreadEl = document.getElementById('memlane-thread');
const memInputEl  = document.getElementById('memlane-input');

function scrollThread() {
  if (memThreadEl) memThreadEl.scrollTo({ top: memThreadEl.scrollHeight, behavior: 'smooth' });
}

function openMemLane() {
  if (memLaneOpen) return;
  memLaneOpen = true;
  document.getElementById('view-mantle')?.classList.add('lane-open');
  startMatrix();
  loadMemLane();
  setTimeout(() => memInputEl?.focus(), 450);
}

function closeMemLane() {
  if (!memLaneOpen) return;
  memLaneOpen = false;
  document.getElementById('view-mantle')?.classList.remove('lane-open');
  stopMatrix();
}

async function loadMemLane(quiet) {
  const listsEl = document.getElementById('memlane-lists');
  const countEl = document.getElementById('memlane-count');
  if (!listsEl) return;
  try {
    const mem = await (await fetch('/api/memory')).json();
    renderMemList(listsEl, countEl, mem);
  } catch (e) {
    if (!quiet) listsEl.innerHTML = '<div class="memlane-empty">Couldn’t reach her memories right now.</div>';
  }
}

function renderMemList(listsEl, countEl, mem) {
  const self = mem.self_facts || [];
  const him  = mem.facts || [];
  if (countEl) {
    const core = [...self, ...him].filter(m => m.core).length;
    countEl.textContent = `${self.length + him.length} memories · ${core} kept`;
  }
  listsEl.innerHTML = '';
  const group = (title, sub, arr, kind) => {
    const h = document.createElement('div');
    h.className = 'memlane-group-title';
    h.innerHTML = `${title} <b>${sub}</b>`;
    listsEl.appendChild(h);
    if (!arr.length) {
      const e = document.createElement('div');
      e.className = 'memlane-empty';
      e.textContent = 'Nothing here yet.';
      listsEl.appendChild(e);
      return;
    }
    arr.forEach(m => listsEl.appendChild(memRow(m, kind)));
  };
  group('Who she is', '— her identity', self, 'self');
  group('What she knows about you', '— about him', him, 'him');
}

function memRow(m, kind) {
  const row = document.createElement('div');
  row.className = 'mem-row' + (m.core ? ' core' : '');

  const keep = document.createElement('button');
  keep.className = 'mem-act keep' + (m.core ? ' on' : '');
  keep.textContent = '★';
  keep.title = m.core ? 'Kept (core) — click to unkeep' : 'Keep this (mark core)';
  keep.addEventListener('click', () => memAction('core', kind, m.text, { core: !m.core }));

  const text = document.createElement('div');
  text.className = 'mem-text';
  text.textContent = m.text;

  const del = document.createElement('button');
  del.className = 'mem-act del';
  del.textContent = '✕';
  del.title = 'Forget this';
  del.addEventListener('click', () => {
    row.classList.add('removing');
    setTimeout(() => memAction('delete', kind, m.text), 280);
  });

  row.append(keep, text, del);
  return row;
}

async function memAction(op, kind, fact, extra = {}) {
  try {
    const mem = await (await fetch('/api/memory/' + op, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ kind, fact, ...extra }),
    })).json();
    renderMemList(document.getElementById('memlane-lists'),
                  document.getElementById('memlane-count'), mem);
  } catch (e) { /* leave the list as-is on failure */ }
}

function sendMemLane() {
  const text = memInputEl.value.trim();
  if (!text || !connected) return;
  document.querySelector('.memlane-hint')?.remove();
  appendBubble('user', text, memThreadEl);
  scrollThread();
  memInputEl.value = '';
  memInputEl.style.height = 'auto';
  setOrbState('thinking');
  setStatus('Thinking...');
  // review:true folds her FULL memory list into context for this turn.
  ws.send(JSON.stringify({ type: 'chat', message: text, review: true }));
}

/* ─── Matrix rain backdrop (themed to the current accent) ───────────── */
function startMatrix() {
  const canvas = document.getElementById('memlane-matrix');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const glyphs = 'ｱｲｳｴｵｶｷｸｹｺｻｼｽｾｿﾀﾁﾂﾃﾄﾅﾆﾇﾈ0123456789:.";=*+-<>'.split('');
  let cols, drops, fontSize;
  const accent = getComputedStyle(document.documentElement)
                   .getPropertyValue('--accent-v-bright').trim() || '#9db4ff';

  function resize() {
    canvas.width = canvas.clientWidth;
    canvas.height = canvas.clientHeight;
    fontSize = 15;
    cols = Math.max(1, Math.floor(canvas.width / fontSize));
    drops = Array.from({ length: cols }, () => Math.random() * -50);
  }
  resize();
  memLaneEl._matrixResize = resize;
  window.addEventListener('resize', resize);

  function tick() {
    ctx.fillStyle = 'rgba(0,0,0,0.08)';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.font = fontSize + 'px monospace';
    for (let i = 0; i < cols; i++) {
      const ch = glyphs[(Math.random() * glyphs.length) | 0];
      const x = i * fontSize, y = drops[i] * fontSize;
      ctx.fillStyle = accent;
      ctx.fillText(ch, x, y);
      if (y > canvas.height && Math.random() > 0.975) drops[i] = 0;
      drops[i]++;
    }
    matrixRAF = requestAnimationFrame(tick);
  }
  tick();
}

function stopMatrix() {
  if (matrixRAF) cancelAnimationFrame(matrixRAF);
  matrixRAF = null;
  if (memLaneEl?._matrixResize) window.removeEventListener('resize', memLaneEl._matrixResize);
}

document.getElementById('mantle-memlane')?.addEventListener('click', openMemLane);
document.getElementById('memlane-back')?.addEventListener('click', closeMemLane);
document.getElementById('memlane-send')?.addEventListener('click', sendMemLane);
memInputEl?.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMemLane(); }
});
memInputEl?.addEventListener('input', () => {
  memInputEl.style.height = 'auto';
  memInputEl.style.height = Math.min(memInputEl.scrollHeight, 120) + 'px';
});

/* ═══════════════════════════════════════════════════════════════════
   NOTES
   ═══════════════════════════════════════════════════════════════════ */
const notesItemsEl   = document.getElementById('notes-items');
const notesSearchEl  = document.getElementById('notes-search');
const noteEmptyEl    = document.getElementById('note-empty');
const notePaneEl     = document.getElementById('note-pane');
const noteTitleEl    = document.getElementById('note-title');
const noteContentEl  = document.getElementById('note-content');
const notePreviewEl  = document.getElementById('note-preview');
const noteBacklinksEl= document.getElementById('note-backlinks');
const noteSavedEl    = document.getElementById('note-saved');
const noteEditLabel  = document.getElementById('note-edit-label');
const assistInput    = document.getElementById('note-assist-input');
const assistBtn      = document.getElementById('note-assist-btn');

let allNotes = [];          // [{title, modified}]
let currentNote = null;     // title currently open
let previewMode = false;
let saveTimer = null;
let knownTitles = new Set();

async function loadNoteList() {
  try {
    allNotes = await (await fetch('/api/notes')).json();
  } catch { allNotes = []; }
  knownTitles = new Set(allNotes.map(n => n.title.toLowerCase()));
  renderNoteList();
}

function renderNoteList() {
  const q = (notesSearchEl.value || '').toLowerCase();
  const filtered = allNotes.filter(n => n.title.toLowerCase().includes(q));
  notesItemsEl.innerHTML = '';
  if (!filtered.length) {
    notesItemsEl.innerHTML = '<div class="notes-empty-hint">No notes yet.<br>Hit + to make one.</div>';
    return;
  }
  for (const n of filtered) {
    const el = document.createElement('div');
    el.className = 'note-item' + (n.title === currentNote ? ' active' : '');
    el.innerHTML = `<div class="note-item-title"></div><div class="note-item-meta">${relTime(n.modified)}</div>`;
    el.querySelector('.note-item-title').textContent = n.title;
    el.addEventListener('click', () => openNote(n.title));
    notesItemsEl.appendChild(el);
  }
}

notesSearchEl.addEventListener('input', renderNoteList);

async function openNote(title) {
  currentNote = title;
  noteEmptyEl.style.display = 'none';
  notePaneEl.style.display = 'flex';
  let data;
  try { data = await (await fetch('/api/notes/' + encodeURIComponent(title))).json(); }
  catch { return; }
  noteTitleEl.value = title;
  noteContentEl.value = data.content || '';
  renderBacklinks(data.backlinks || []);
  setPreviewMode(true);   // open in preview by default
  renderNoteList();
}

function newNote() {
  let base = 'Untitled', name = base, i = 1;
  while (knownTitles.has(name.toLowerCase())) name = `${base} ${++i}`;
  currentNote = name;
  knownTitles.add(name.toLowerCase());
  noteEmptyEl.style.display = 'none';
  notePaneEl.style.display = 'flex';
  noteTitleEl.value = name;
  noteContentEl.value = '';
  renderBacklinks([]);
  setPreviewMode(false);
  noteTitleEl.focus();
  noteTitleEl.select();
  saveNote();
  loadNoteList();
}
document.getElementById('note-new').addEventListener('click', newNote);

async function saveNote() {
  if (!currentNote) return;
  const newTitle = noteTitleEl.value.trim() || 'Untitled';
  const content = noteContentEl.value;
  // Title changed → delete old file, write new
  if (newTitle !== currentNote) {
    try { await fetch('/api/notes/' + encodeURIComponent(currentNote), { method: 'DELETE' }); } catch {}
    currentNote = newTitle;
  }
  try {
    await fetch('/api/notes/' + encodeURIComponent(currentNote), {
      method: 'PUT', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ content }),
    });
    flashSaved();
    await loadNoteList();
  } catch {}
}

function scheduleSave() {
  clearTimeout(saveTimer);
  saveTimer = setTimeout(saveNote, 700);
}
noteContentEl.addEventListener('input', scheduleSave);
noteTitleEl.addEventListener('change', saveNote);

async function deleteCurrentNote() {
  if (!currentNote) return;
  if (!confirm(`Delete "${currentNote}"?`)) return;
  try { await fetch('/api/notes/' + encodeURIComponent(currentNote), { method: 'DELETE' }); } catch {}
  currentNote = null;
  notePaneEl.style.display = 'none';
  noteEmptyEl.style.display = 'flex';
  loadNoteList();
}

document.getElementById('note-delete').addEventListener('click', deleteCurrentNote);

// Delete key removes the open note — but only when you're not typing in a field
// (title, body, or the assist box), so Delete still edits text while editing.
document.addEventListener('keydown', (e) => {
  if (e.key !== 'Delete' || activeView !== 'notes' || !currentNote) return;
  const a = document.activeElement;
  const editing = a && (a.tagName === 'INPUT' || a.tagName === 'TEXTAREA' || a.isContentEditable);
  if (editing) return;
  e.preventDefault();
  deleteCurrentNote();
});

/* edit / preview */
document.getElementById('note-edit-toggle').addEventListener('click', () => setPreviewMode(!previewMode));

function setPreviewMode(on) {
  previewMode = on;
  if (on) {
    notePreviewEl.innerHTML = renderMarkdown(noteContentEl.value);
    notePreviewEl.style.display = 'block';
    noteContentEl.style.display = 'none';
    noteEditLabel.textContent = 'Edit';
    wireWikiLinks();
  } else {
    notePreviewEl.style.display = 'none';
    noteContentEl.style.display = 'block';
    noteEditLabel.textContent = 'Preview';
    noteContentEl.focus();
  }
}

function renderBacklinks(list) {
  if (!list.length) { noteBacklinksEl.innerHTML = ''; return; }
  noteBacklinksEl.innerHTML = '<span class="bl-label">Linked from</span>';
  for (const t of list) {
    const chip = document.createElement('span');
    chip.className = 'bl-chip';
    chip.textContent = t;
    chip.addEventListener('click', () => openNote(t));
    noteBacklinksEl.appendChild(chip);
  }
}

function flashSaved() {
  noteSavedEl.textContent = 'Saved';
  noteSavedEl.classList.add('show');
  setTimeout(() => noteSavedEl.classList.remove('show'), 1200);
}

/* Aitha note assist */
async function askAitha() {
  if (!currentNote) return;
  const instruction = assistInput.value.trim();
  if (!instruction) { assistInput.focus(); return; }
  assistBtn.classList.add('working');
  assistBtn.disabled = true;
  document.getElementById('assist-label').textContent = 'Thinking…';
  try {
    const res = await fetch('/api/notes/assist', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ content: noteContentEl.value, instruction }),
    });
    const data = await res.json();
    if (data.content != null) {
      noteContentEl.value = data.content;
      assistInput.value = '';
      await saveNote();
      if (previewMode) setPreviewMode(true);
    }
  } catch {}
  assistBtn.classList.remove('working');
  assistBtn.disabled = false;
  document.getElementById('assist-label').textContent = 'Ask';
}
assistBtn.addEventListener('click', askAitha);
assistInput.addEventListener('keydown', e => { if (e.key === 'Enter') { e.preventDefault(); askAitha(); } });

/* minimal markdown renderer with [[wikilinks]] */
function renderMarkdown(src) {
  const esc = s => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  const lines = (src || '').split('\n');
  let html = '', inList = false;
  const inline = t => esc(t)
    .replace(/\[\[([^\]]+?)\]\]/g, (_, name) => {
      const miss = knownTitles.has(name.trim().toLowerCase()) ? '' : ' missing';
      return `<a class="wikilink${miss}" data-note="${esc(name.trim())}">${esc(name.trim())}</a>`;
    })
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    .replace(/(^|[^*])\*([^*]+)\*/g, '$1<em>$2</em>');
  for (let line of lines) {
    if (/^\s*[-*]\s+/.test(line)) {
      if (!inList) { html += '<ul>'; inList = true; }
      html += '<li>' + inline(line.replace(/^\s*[-*]\s+/, '')) + '</li>';
      continue;
    }
    if (inList) { html += '</ul>'; inList = false; }
    if (/^###\s+/.test(line)) html += '<h3>' + inline(line.slice(4)) + '</h3>';
    else if (/^##\s+/.test(line)) html += '<h2>' + inline(line.slice(3)) + '</h2>';
    else if (/^#\s+/.test(line)) html += '<h1>' + inline(line.slice(2)) + '</h1>';
    else if (line.trim() === '') html += '';
    else html += '<p>' + inline(line) + '</p>';
  }
  if (inList) html += '</ul>';
  return html;
}

function wireWikiLinks() {
  notePreviewEl.querySelectorAll('a.wikilink').forEach(a => {
    a.addEventListener('click', () => openNote(a.dataset.note));
  });
}

function relTime(mtime) {
  const s = Date.now() / 1000 - mtime;
  if (s < 60) return 'just now';
  if (s < 3600) return Math.round(s / 60) + 'm ago';
  if (s < 86400) return Math.round(s / 3600) + 'h ago';
  return Math.round(s / 86400) + 'd ago';
}

/* ─── Magma chat — talk to Aitha; she writes notes ─────────────────── */
const magmaInput = document.getElementById('magma-input');
const magmaSend  = document.getElementById('magma-send');
const magmaReply = document.getElementById('magma-reply');
let magmaHistory = [];

async function sendMagma() {
  const message = magmaInput.value.trim();
  if (!message) return;
  magmaInput.value = '';
  magmaSend.disabled = true;
  magmaReply.classList.add('show');
  magmaReply.textContent = '…';
  try {
    const res = await fetch('/api/magma_chat', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message, history: magmaHistory.slice(-12) }),
    });
    const data = await res.json();
    magmaReply.textContent = data.reply || '…';
    magmaHistory.push({ role: 'user', content: message });
    magmaHistory.push({ role: 'assistant', content: data.reply || '' });
    if (data.changed && data.changed.length) {
      await loadNoteList();
      // If she touched the open note, reload it; otherwise open the newest one.
      if (currentNote && data.changed.includes(currentNote)) openNote(currentNote);
      else openNote(data.changed[0]);
    }
  } catch {
    magmaReply.textContent = '...I couldn\'t reach you just now.';
  }
  magmaSend.disabled = false;
  magmaInput.focus();
}
magmaSend.addEventListener('click', sendMagma);
magmaInput.addEventListener('keydown', e => { if (e.key === 'Enter') { e.preventDefault(); sendMagma(); } });

/* ─── Memory viewer (settings) ─────────────────────────────────────── */
const memListEl  = document.getElementById('mem-list');
const memCountEl  = document.getElementById('mem-count');
const memAddInput = document.getElementById('mem-add-input');

async function loadMemory() {
  try {
    const data = await (await fetch('/api/memory')).json();
    renderMemory(data);
  } catch {}
}

function memRow(item, kind) {
  // Accept legacy strings as well as {text, core} objects.
  const text = typeof item === 'string' ? item : item.text;
  const core = typeof item === 'object' && item.core;

  const row = document.createElement('div');
  row.className = 'mem-row' + (core ? ' core' : '');

  const star = document.createElement('button');
  star.className = 'mem-star' + (core ? ' on' : '');
  star.title = core ? 'Core memory — protected. Click to unset.' : 'Mark as core memory';
  star.textContent = core ? '★' : '☆';
  star.addEventListener('click', () => toggleCore(text, kind, !core));

  const span = document.createElement('span');
  span.textContent = text;

  const del = document.createElement('button');
  del.className = 'mem-del';
  del.title = 'Delete';
  del.innerHTML = '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>';
  del.addEventListener('click', () => deleteMemory(text, kind));

  row.appendChild(star);
  row.appendChild(span);
  row.appendChild(del);
  return row;
}

function renderMemory(data) {
  // Core memories float to the top of each group.
  const byCore = (a, b) => (b.core ? 1 : 0) - (a.core ? 1 : 0);
  const him = (data.facts || []).slice().sort(byCore);
  const self = (data.self_facts || []).slice().sort(byCore);
  memCountEl.textContent = him.length + self.length;
  memListEl.innerHTML = '';
  if (!him.length && !self.length) {
    memListEl.innerHTML = '<div class="mem-empty">No memories yet — she\'ll form them as you talk.</div>';
    return;
  }
  if (self.length) {
    const h = document.createElement('div');
    h.className = 'mem-group-label';
    h.textContent = 'Her sense of self';
    memListEl.appendChild(h);
    self.forEach(f => memListEl.appendChild(memRow(f, 'self')));
  }
  if (him.length) {
    const h = document.createElement('div');
    h.className = 'mem-group-label';
    h.textContent = 'About you';
    memListEl.appendChild(h);
    him.forEach(f => memListEl.appendChild(memRow(f, 'him')));
  }
}

async function addMemory() {
  const fact = memAddInput.value.trim();
  if (!fact) return;
  memAddInput.value = '';
  try {
    const data = await (await fetch('/api/memory/add', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ fact, kind: 'him' }),
    })).json();
    renderMemory(data);
  } catch {}
}

async function deleteMemory(fact, kind) {
  try {
    const data = await (await fetch('/api/memory/delete', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ fact, kind }),
    })).json();
    renderMemory(data);
  } catch {}
}

async function toggleCore(fact, kind, core) {
  try {
    const data = await (await fetch('/api/memory/core', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ fact, kind, core }),
    })).json();
    renderMemory(data);
  } catch {}
}

async function clearMemory() {
  if (!confirm("Erase everything Aitha remembers about you? This can't be undone.")) return;
  try {
    const data = await (await fetch('/api/memory/clear', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ scope: 'all' }),
    })).json();
    renderMemory(data);
  } catch {}
}

document.getElementById('mem-add-btn').addEventListener('click', addMemory);
memAddInput.addEventListener('keydown', e => { if (e.key === 'Enter') { e.preventDefault(); addMemory(); } });
document.getElementById('mem-clear').addEventListener('click', clearMemory);

/* ═══════════════════════════════════════════════════════════════════
   HEARTH (D&D)
   ═══════════════════════════════════════════════════════════════════ */
let hearth = { dm: null, active: null, campaigns: [], campaign: null };

async function hpost(path, body) {
  try {
    return await (await fetch(path, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body || {}),
    })).json();
  } catch { return null; }
}

async function loadHearth() {
  try { hearth = await (await fetch('/api/hearth')).json(); } catch { return; }
  renderHearth();
  // First run: if there are no campaigns, make a starter one.
  if (!hearth.campaigns || !hearth.campaigns.length) {
    await hpost('/api/hearth/campaign/new', { name: 'A New Tale' });
  }
}

function onHearthState(data) {
  if (data.dm) hearth.dm = data.dm;
  if (data.campaign) {
    hearth.campaign = data.campaign;
    hearth.active = data.campaign.id;
  }
  // refresh campaign list lazily
  fetch('/api/hearth').then(r => r.json()).then(h => {
    hearth.campaigns = h.campaigns; hearth.active = h.active;
    if (h.campaign) hearth.campaign = h.campaign;
    hearth.dm = h.dm;
    renderHearth();
  }).catch(() => renderHearth());
}

const $h = id => document.getElementById(id);

function renderHearth() {
  const c = hearth.campaign;
  // campaign selector
  const sel = $h('hearth-campaign');
  if (sel) {
    sel.innerHTML = '';
    (hearth.campaigns || []).forEach(cp => {
      const o = document.createElement('option');
      o.value = cp.id; o.textContent = cp.name;
      if (cp.id === hearth.active) o.selected = true;
      sel.appendChild(o);
    });
  }
  // DM banner + turn
  if (hearth.dm) $h('dm-name').textContent = hearth.dm.name || 'The Keeper';
  const turn = c?.turn?.active;
  const turnLabel = { dm: 'DM’s turn', aitha: 'Aitha’s turn', me: 'Your turn' }[turn] || '—';
  $h('turn-pill').textContent = turnLabel;
  $h('seat-aitha').classList.toggle('active-turn', turn === 'aitha');
  $h('seat-me').classList.toggle('active-turn', turn === 'me');

  renderSheetMini('me', c?.sheets?.me);
  renderSheetMini('aitha', c?.sheets?.aitha);
  renderSceneLog(c);
  renderDiceTray();
  renderBoard(c);
}

function renderSheetMini(who, s) {
  const el = $h('sheet-' + who);
  if (!el) return;
  if (!s) { el.innerHTML = '<span style="color:var(--text-3)">No campaign.</span>'; return; }
  const st = s.stats || {};
  const stat = (k) => `<div class="sm-stat"><b>${st[k] ?? 10}</b><span>${k}</span></div>`;
  el.innerHTML = `
    <div class="sm-row"><span>${escapeHtml(s.race||'')} ${escapeHtml(s.class||'Adventurer')}</span><span>Lv ${s.level||1}</span></div>
    <div class="sm-row"><span class="sm-hp">HP ${s.hp?.cur ?? 0}/${s.hp?.max ?? 0}</span><span>AC ${s.ac ?? 10}</span></div>
    <div class="sm-stats">${['str','dex','con','int','wis','cha'].map(stat).join('')}</div>`;
}

function renderSceneLog(c) {
  const log = $h('scene-log');
  if (!log) return;
  log.innerHTML = '';
  if (!c || !c.log || !c.log.length) {
    log.innerHTML = '<div style="color:var(--text-3);text-align:center;margin-top:30px">The hearth is lit. Describe what you do to begin…</div>';
    return;
  }
  for (const e of c.log) {
    const row = document.createElement('div');
    const kind = (e.kind === 'roll' || e.kind === 'ask') ? e.kind : e.who;
    row.className = 'log-entry ' + kind;
    const whoLabel = { dm: hearth.dm?.name || 'DM', aitha: 'Aitha', me: 'You' }[e.who] || e.who;
    let body = escapeHtml(e.text || '');
    if (e.kind === 'roll' && e.roll) {
      const r = e.roll;
      body = `\u{1F3B2} ${escapeHtml(e.text || r.expr)} → <b>${r.total}</b> <span style="opacity:.6">[${r.rolls.join(', ')}${r.mod ? (r.mod>0?'+':'')+r.mod : ''}]</span>`;
    } else if (e.kind === 'ask') {
      body = '\u{1F3B2} ' + body;
    }
    row.innerHTML = `<div class="log-who">${escapeHtml(whoLabel)}</div><div class="log-body">${body}</div>`;
    log.appendChild(row);
  }
  log.scrollTop = log.scrollHeight;
}

const DICE = [4, 6, 8, 10, 12, 20];
function renderDiceTray() {
  const tray = $h('dice-tray');
  if (!tray || tray.dataset.built) return;
  tray.dataset.built = '1';
  DICE.forEach(d => {
    const b = document.createElement('button');
    b.className = 'die'; b.dataset.d = d;
    b.innerHTML = `<span class="die-shape"></span><span class="die-label">d${d}</span>`;
    b.addEventListener('click', () => rollDie(d, b));
    tray.appendChild(b);
  });
}

async function rollDie(sides, btn) {
  btn.classList.add('rolling');
  setTimeout(() => btn.classList.remove('rolling'), 650);
  await hpost('/api/hearth/roll', { expr: 'd' + sides, who: 'me' });
  // result animation arrives via hearth_roll broadcast
}

function showDiceResult(roll, who) {
  if (!roll) return;
  document.querySelectorAll('.dice-result').forEach(e => e.remove());
  const wrap = document.createElement('div');
  wrap.className = 'dice-result';
  const sides = roll.sides;
  const whoName = { dm: hearth.dm?.name || 'DM', aitha: 'Aitha', me: 'You' }[who] || '';
  wrap.innerHTML = `
    <div class="dr-die"><span class="die-shape" style="position:absolute;inset:0"></span><span class="dr-num">${roll.total}</span></div>
    <div class="dr-meta">${escapeHtml(whoName)} rolled ${escapeHtml(roll.expr)}</div>`;
  // shape it like the die
  const shape = wrap.querySelector('.die-shape');
  const probe = document.createElement('div'); probe.dataset.d = sides;
  shape.className = 'die-shape';
  if (sides === 4) shape.style.clipPath = 'polygon(50% 0,100% 100%,0 100%)';
  else if (sides === 6) shape.style.borderRadius = '16px';
  else if (sides === 8) shape.style.clipPath = 'polygon(50% 0,100% 50%,50% 100%,0 50%)';
  else if (sides === 10) shape.style.clipPath = 'polygon(50% 0,90% 38%,72% 100%,28% 100%,10% 38%)';
  else if (sides === 12) shape.style.clipPath = 'polygon(50% 0,79% 10%,98% 35%,98% 65%,79% 90%,50% 100%,21% 90%,2% 65%,2% 35%,21% 10%)';
  else if (sides === 20) shape.style.clipPath = 'polygon(50% 0,93% 25%,93% 75%,50% 100%,7% 75%,7% 25%)';
  document.body.appendChild(wrap);
  requestAnimationFrame(() => wrap.classList.add('show'));
  setTimeout(() => wrap.remove(), 1800);
}

function renderBoard(c) {
  const board = $h('battle-board');
  const toggle = $h('hearth-board-toggle');
  if (!board) return;
  const b = c?.board;
  const on = !!(b && b.enabled);
  if (toggle) toggle.classList.toggle('on', on);
  board.style.display = on ? 'grid' : 'none';
  if (!on || !b) return;
  const w = b.w || 14, h = b.h || 10;
  board.style.gridTemplateColumns = `repeat(${w}, 1fr)`;
  board.innerHTML = '';
  const cells = [];
  for (let i = 0; i < w * h; i++) {
    const cell = document.createElement('div'); cell.className = 'bb-cell'; cells.push(cell); board.appendChild(cell);
  }
  (b.tokens || []).forEach(t => {
    const x = Math.max(0, Math.min(w - 1, t.x || 0));
    const y = Math.max(0, Math.min(h - 1, t.y || 0));
    const cell = cells[y * w + x];
    if (!cell) return;
    const tok = document.createElement('div');
    tok.className = 'bb-token ' + (t.kind || 'npc');
    tok.title = t.label || '';
    tok.textContent = (t.label || '?').slice(0, 2);
    if (t.color) tok.style.background = t.color;
    cell.appendChild(tok);
  });
}

/* ── panels ───────────────────────────────────────────────────────── */
function openPanel(title, html) {
  $h('hearth-panel-title').textContent = title;
  $h('hearth-panel-body').innerHTML = '';
  if (typeof html === 'string') $h('hearth-panel-body').innerHTML = html;
  else $h('hearth-panel-body').appendChild(html);
  $h('hearth-panel').style.display = 'flex';
}
function closePanel() { $h('hearth-panel').style.display = 'none'; }

function panelSessions() {
  const body = document.createElement('div');
  body.style.cssText = 'display:flex;flex-direction:column;gap:10px';
  const c = hearth.campaign;
  body.innerHTML = '<div class="hp-field"><label>Campaign summary — auto-updates as you play (Aitha & the DM read this)</label>'
    + `<textarea id="hp-summary" rows="4">${escapeHtml(c?.summary || '')}</textarea></div>`
    + '<button class="hp-save" id="hp-summary-save">Save summary (overrides until more play)</button>'
    + '<div style="height:8px"></div><div style="color:var(--text-3);font-size:11px;text-transform:uppercase">Campaigns</div>';
  (hearth.campaigns || []).forEach(cp => {
    const item = document.createElement('div');
    item.className = 'hp-camp-item' + (cp.id === hearth.active ? ' active' : '');
    item.innerHTML = `<div class="hp-camp-name">${escapeHtml(cp.name)}</div>`
      + `<div class="hp-camp-sum">${escapeHtml(cp.summary || 'No summary yet.')}</div>`;
    item.addEventListener('click', () => hpost('/api/hearth/campaign/active', { id: cp.id }));
    body.appendChild(item);
  });
  openPanel('Sessions', body);
  $h('hp-summary-save').addEventListener('click', () =>
    hpost('/api/hearth/campaign/summary', { summary: $h('hp-summary').value }));
}

function panelMemory() {
  const c = hearth.campaign;
  const body = document.createElement('div');
  body.style.cssText = 'display:flex;flex-direction:column;gap:10px';
  body.innerHTML = '<div class="hp-row"><div class="hp-field" style="flex:2"><label>New memory</label>'
    + '<input id="hp-mem-text" placeholder="A goblin warren lies east…"></div>'
    + '<div class="hp-field"><label>Category</label><select id="hp-mem-cat">'
    + ['enemy','location','setting','npc','quest','misc'].map(x=>`<option>${x}</option>`).join('')
    + '</select></div></div><button class="hp-save" id="hp-mem-add">Add memory</button>'
    + '<div style="color:var(--text-3);font-size:11px">Hidden ones (\u{1F441}) are kept out of the DM’s & Aitha’s view.</div>';
  (c?.memory || []).forEach(m => {
    const item = document.createElement('div');
    item.className = 'hp-list-item' + (m.hidden ? ' hidden-mem' : '');
    item.innerHTML = `<div><div class="hp-cat">${escapeHtml(m.category||'misc')}</div>${escapeHtml(m.text)}</div>`
      + `<div style="display:flex;gap:4px"><button class="hp-mini-btn" data-act="toggle">${m.hidden?'\u{1F648}':'\u{1F441}'}</button>`
      + '<button class="hp-mini-btn" data-act="del">✕</button></div>';
    item.querySelector('[data-act="toggle"]').addEventListener('click', () => hpost('/api/hearth/memory', { op: 'toggle', id: m.id }));
    item.querySelector('[data-act="del"]').addEventListener('click', () => hpost('/api/hearth/memory', { op: 'delete', id: m.id }));
    body.appendChild(item);
  });
  openPanel('Session Memory', body);
  $h('hp-mem-add').addEventListener('click', () => {
    const t = $h('hp-mem-text').value.trim(); if (!t) return;
    hpost('/api/hearth/memory', { op: 'add', text: t, category: $h('hp-mem-cat').value });
  });
}

function panelNotes() {
  const c = hearth.campaign;
  const body = document.createElement('div');
  body.style.cssText = 'display:flex;flex-direction:column;gap:10px';
  body.innerHTML = '<button class="hp-save" id="hp-note-aitha">Ask Aitha to reflect on this session</button>'
    + '<div class="hp-field"><label>Or jot your own note</label><textarea id="hp-note-mine" rows="3"></textarea></div>'
    + '<button class="hp-save" id="hp-note-mine-save">Add my note</button><div style="height:6px"></div>';
  const notes = (c?.session_notes || []).slice().reverse();
  if (!notes.length) body.innerHTML += '<div style="color:var(--text-3);font-size:12px">No session notes yet.</div>';
  notes.forEach(n => {
    const item = document.createElement('div');
    item.className = 'hp-note-item';
    item.innerHTML = `<div class="hp-note-by">${n.author === 'aitha' ? 'Aitha' : 'You'} · ${relTime(n.ts)}</div>${escapeHtml(n.text)}`;
    body.appendChild(item);
  });
  openPanel('Session Notes', body);
  $h('hp-note-aitha').addEventListener('click', async (e) => {
    e.target.textContent = 'Aitha is reflecting…'; e.target.disabled = true;
    await hpost('/api/hearth/sessionnote', { mode: 'aitha' });
  });
  $h('hp-note-mine-save').addEventListener('click', () => {
    const t = $h('hp-note-mine').value.trim(); if (!t) return;
    hpost('/api/hearth/sessionnote', { mode: 'manual', text: t });
  });
}

function panelDM() {
  const dm = hearth.dm || {};
  const body = document.createElement('div');
  body.innerHTML = '<div class="hp-field"><label>DM name</label>'
    + `<input id="hp-dm-name" value="${escapeHtml(dm.name||'')}"></div>`
    + '<div class="hp-field"><label>DM persona & style</label>'
    + `<textarea id="hp-dm-persona" rows="6">${escapeHtml(dm.persona||'')}</textarea></div>`
    + '<button class="hp-save" id="hp-dm-save">Save DM</button>'
    + '<div style="color:var(--text-3);font-size:11px;margin-top:6px">The DM knows the core 5e rules and runs the table.</div>';
  openPanel('Dungeon Master', body);
  $h('hp-dm-save').addEventListener('click', () =>
    hpost('/api/hearth/dm', { name: $h('hp-dm-name').value, persona: $h('hp-dm-persona').value }));
}

function panelSheet(who) {
  const c = hearth.campaign; if (!c) return;
  const s = c.sheets[who] || {};
  const st = s.stats || {};
  const f = (id, label, val, type) => `<div class="hp-field"><label>${label}</label><input id="${id}" value="${escapeHtml(String(val ?? ''))}" ${type === 'num' ? 'type="number"' : ''}></div>`;
  const ta = (id, label, val) => `<div class="hp-field"><label>${label}</label><textarea id="${id}" rows="2">${escapeHtml(val || '')}</textarea></div>`;
  const stat = (k) => `<div class="hp-field"><label>${k.toUpperCase()}</label><input id="hp-st-${k}" type="number" value="${st[k] ?? 10}"></div>`;
  const body = document.createElement('div');
  body.innerHTML =
    f('hp-name', 'Name', s.name) +
    '<div class="hp-row">' + f('hp-race', 'Race', s.race) + f('hp-class', 'Class', s.class) + '</div>' +
    '<div class="hp-row">' + f('hp-level', 'Level', s.level, 'num') + f('hp-ac', 'AC', s.ac, 'num') + f('hp-speed', 'Speed', s.speed, 'num') + '</div>' +
    '<div class="hp-row">' + f('hp-hpcur', 'HP', s.hp?.cur, 'num') + f('hp-hpmax', 'Max HP', s.hp?.max, 'num') + f('hp-hptemp', 'Temp', s.hp?.temp, 'num') + '</div>' +
    '<label style="font-size:11px;color:var(--text-3)">Ability scores</label><div class="hp-stats-grid">' +
    ['str','dex','con','int','wis','cha'].map(stat).join('') + '</div>' +
    ta('hp-skills', 'Skills / Proficiencies', s.skills) +
    ta('hp-inventory', 'Inventory', s.inventory) +
    ta('hp-features', 'Features & Traits', s.features) +
    ta('hp-spells', 'Spells', s.spells) +
    ta('hp-notes', 'Notes', s.notes) +
    '<button class="hp-save" id="hp-sheet-save">Save sheet</button>';
  openPanel((who === 'aitha' ? 'Aitha' : 'Your') + ' Character', body);
  $h('hp-sheet-save').addEventListener('click', () => {
    const sheet = {
      name: $h('hp-name').value, race: $h('hp-race').value, class: $h('hp-class').value,
      level: +$h('hp-level').value || 1, ac: +$h('hp-ac').value || 10, speed: +$h('hp-speed').value || 30,
      hp: { cur: +$h('hp-hpcur').value || 0, max: +$h('hp-hpmax').value || 0, temp: +$h('hp-hptemp').value || 0 },
      stats: Object.fromEntries(['str','dex','con','int','wis','cha'].map(k => [k, +$h('hp-st-' + k).value || 10])),
      skills: $h('hp-skills').value, inventory: $h('hp-inventory').value,
      features: $h('hp-features').value, spells: $h('hp-spells').value, notes: $h('hp-notes').value,
    };
    hpost('/api/hearth/sheet', { who, sheet });
  });
}

function escapeHtml(s) {
  return String(s == null ? '' : s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

/* ── Session Zero: guided character creation ──────────────────────────── */
const DND_RACES = ['Human', 'Elf', 'Dwarf', 'Halfling', 'Half-Elf', 'Half-Orc', 'Tiefling', 'Dragonborn', 'Gnome'];
const DND_CLASSES = ['Fighter', 'Wizard', 'Rogue', 'Cleric', 'Ranger', 'Barbarian', 'Bard', 'Druid', 'Monk', 'Paladin', 'Sorcerer', 'Warlock'];
const DND_BACKGROUNDS = ['Acolyte', 'Charlatan', 'Criminal', 'Entertainer', 'Folk Hero', 'Guild Artisan', 'Hermit', 'Noble', 'Outlander', 'Sage', 'Soldier', 'Urchin'];
const STANDARD_ARRAY = [15, 14, 13, 12, 10, 8];
const HIT_DICE = { Barbarian: 12, Fighter: 10, Paladin: 10, Ranger: 10, Sorcerer: 6, Wizard: 6 };  // others d8
const SZ_STEPS = ['Concept', 'Race', 'Class', 'Abilities', 'Background & Gear', 'Review'];
const ABILS = ['str', 'dex', 'con', 'int', 'wis', 'cha'];
const abilMod = (score) => Math.floor(((+score || 10) - 10) / 2);

let wizard = null;

function openSessionZero() {
  if (!hearth.campaign) { return; }
  const body = document.createElement('div');
  body.style.cssText = 'display:flex;flex-direction:column;gap:12px';
  body.innerHTML =
    '<div class="sz-intro">Build your characters before the adventure begins. Walk through it step by step — or just start talking and the DM will guide you.</div>'
    + '<button class="hp-save" id="sz-build-me">⚔ Build your character</button>'
    + '<button class="hp-save sz-alt" id="sz-build-aitha">✦ Let Aitha create hers with the DM</button>';
  openPanel('Session Zero', body);
  $h('sz-build-me').addEventListener('click', () => startCharWizard('me'));
  $h('sz-build-aitha').addEventListener('click', () => {
    closePanel();
    hpost('/api/hearth/say', { text: "Let's begin session zero. DM, please interview Aitha to create her "
      + "character — ask her who she wants to be, one step at a time, and let her decide." });
  });
}

function startCharWizard(who) {
  const s = (hearth.campaign?.sheets?.[who]) || {};
  const known = (s.name && s.name !== 'You' && s.name !== 'Aitha') ? s.name : '';
  wizard = {
    who, step: 0, name: known, concept: '',
    race: s.race || '', cls: s.class || '', background: s.background || '',
    stats: { str: 10, dex: 10, con: 10, int: 10, wis: 10, cha: 10, ...(s.stats || {}) },
    gear: s.inventory || '',
  };
  renderWizard();
}

function captureWizardStep() {
  const w = wizard; if (!w) return;
  if (w.step === 0) {
    w.name = ($h('sz-name')?.value || '').trim() || w.name;
    w.concept = ($h('sz-concept')?.value || '').trim();
  } else if (w.step === 3) {
    ABILS.forEach(k => { const v = +($h('sz-st-' + k)?.value); if (v) w.stats[k] = v; });
  } else if (w.step === 4) {
    w.background = $h('sz-bg')?.value || '';
    w.gear = $h('sz-gear')?.value || '';
  }
}

function renderWizard() {
  const w = wizard; if (!w) return;
  const progress = SZ_STEPS.map((label, i) =>
    `<div class="sz-step${i === w.step ? ' active' : ''}${i < w.step ? ' done' : ''}">${label}</div>`).join('');
  let inner = '';
  if (w.step === 0) {
    inner = '<div class="hp-field"><label>Character name</label>'
      + `<input id="sz-name" value="${escapeHtml(w.name)}" placeholder="e.g. Kael Thornwood"></div>`
      + '<div class="hp-field"><label>Concept (one line — optional)</label>'
      + `<input id="sz-concept" value="${escapeHtml(w.concept)}" placeholder="a haunted ex-soldier seeking redemption"></div>`;
  } else if (w.step === 1) {
    inner = '<div class="sz-grid">' + DND_RACES.map(r =>
      `<button class="sz-pick${w.race === r ? ' sel' : ''}" data-race="${r}">${r}</button>`).join('') + '</div>';
  } else if (w.step === 2) {
    inner = '<div class="sz-grid">' + DND_CLASSES.map(r =>
      `<button class="sz-pick${w.cls === r ? ' sel' : ''}" data-cls="${r}">${r}</button>`).join('') + '</div>';
  } else if (w.step === 3) {
    inner = '<div class="sz-hint">Assign the standard array (15, 14, 13, 12, 10, 8) or type your own.</div>'
      + '<div class="hp-stats-grid">' + ABILS.map(k =>
        `<div class="hp-field"><label>${k.toUpperCase()} <span class="sz-mod" id="sz-mod-${k}"></span></label>`
        + `<input id="sz-st-${k}" type="number" value="${w.stats[k] ?? 10}"></div>`).join('') + '</div>'
      + '<button class="hp-mini-btn sz-array" id="sz-array">Fill standard array</button>';
  } else if (w.step === 4) {
    inner = '<div class="hp-field"><label>Background</label><select id="sz-bg">'
      + ['', ...DND_BACKGROUNDS].map(b => `<option${w.background === b ? ' selected' : ''}>${b}</option>`).join('')
      + '</select></div>'
      + '<div class="hp-field"><label>Starting gear / notes</label>'
      + `<textarea id="sz-gear" rows="3">${escapeHtml(w.gear)}</textarea></div>`;
  } else {
    const hd = HIT_DICE[w.cls] || 8;
    const hp = hd + abilMod(w.stats.con);
    inner = '<div class="sz-review">'
      + `<div class="sz-rev-name">${escapeHtml(w.name || '(unnamed)')}</div>`
      + `<div class="sz-rev-sub">${escapeHtml(w.race || '?')} ${escapeHtml(w.cls || '?')}${w.background ? ' · ' + escapeHtml(w.background) : ''}</div>`
      + '<div class="sz-rev-stats">' + ABILS.map(k =>
        `<span><b>${w.stats[k] ?? 10}</b> ${k.toUpperCase()} (${abilMod(w.stats[k]) >= 0 ? '+' : ''}${abilMod(w.stats[k])})</span>`).join('') + '</div>'
      + `<div class="sz-hint">Starting HP ≈ ${hp} (d${hd} + CON). ${w.concept ? '“' + escapeHtml(w.concept) + '”' : ''}</div>`
      + '</div>';
  }
  const body = document.createElement('div');
  body.className = 'sz-wizard';
  body.innerHTML = `<div class="sz-progress">${progress}</div><div class="sz-stepbody">${inner}</div>`
    + '<div class="sz-nav">'
    + `<button class="btn-secondary" id="sz-back"${w.step === 0 ? ' disabled' : ''}>Back</button>`
    + `<button class="btn-primary" id="sz-next">${w.step === SZ_STEPS.length - 1 ? 'Finish' : 'Next'}</button>`
    + '</div>';
  openPanel((w.who === 'aitha' ? 'Aitha' : 'Your') + ' Character — Session Zero', body);

  body.querySelectorAll('[data-race]').forEach(b => b.addEventListener('click', () => { w.race = b.dataset.race; renderWizard(); }));
  body.querySelectorAll('[data-cls]').forEach(b => b.addEventListener('click', () => { w.cls = b.dataset.cls; renderWizard(); }));
  const refreshMods = () => ABILS.forEach(k => {
    const el = $h('sz-mod-' + k); if (el) { const m = abilMod($h('sz-st-' + k)?.value); el.textContent = (m >= 0 ? '+' : '') + m; }
  });
  if (w.step === 3) {
    refreshMods();
    ABILS.forEach(k => $h('sz-st-' + k)?.addEventListener('input', refreshMods));
    $h('sz-array')?.addEventListener('click', () => { STANDARD_ARRAY.forEach((v, i) => w.stats[ABILS[i]] = v); renderWizard(); });
  }
  $h('sz-back')?.addEventListener('click', () => { captureWizardStep(); w.step = Math.max(0, w.step - 1); renderWizard(); });
  $h('sz-next')?.addEventListener('click', () => {
    captureWizardStep();
    if (w.step < SZ_STEPS.length - 1) { w.step++; renderWizard(); } else finishWizard();
  });
}

async function finishWizard() {
  const w = wizard; if (!w) return;
  const cur = (hearth.campaign?.sheets?.[w.who]) || {};
  const hd = HIT_DICE[w.cls] || 8;
  const hp = Math.max(1, hd + abilMod(w.stats.con));
  const notes = [cur.notes, w.concept ? 'Concept: ' + w.concept : ''].filter(Boolean).join('\n');
  const sheet = {
    ...cur,
    name: w.name || cur.name || (w.who === 'aitha' ? 'Aitha' : 'You'),
    race: w.race, class: w.cls, level: cur.level || 1, background: w.background,
    stats: w.stats,
    hp: { cur: hp, max: hp, temp: 0 },
    ac: cur.ac || (10 + abilMod(w.stats.dex)),
    inventory: w.gear || cur.inventory || '',
    notes,
  };
  await hpost('/api/hearth/sheet', { who: w.who, sheet });
  closePanel();
  hpost('/api/hearth/say', { text: `(${w.who === 'aitha' ? 'Aitha' : 'I'} finished a character: `
    + `${sheet.name}, a ${w.race || ''} ${w.cls || ''}.)` });
  wizard = null;
}

// Electron disables window.prompt(), so we roll our own small text dialog.
// Returns a Promise that resolves to the entered string, or null if cancelled.
function inlinePrompt(title, { label = '', value = '', placeholder = '', ok = 'OK' } = {}) {
  return new Promise(resolve => {
    const back = document.createElement('div');
    back.className = 'inq-backdrop';
    back.innerHTML =
      `<div class="inq-card">
         <div class="inq-title">${escapeHtml(title)}</div>
         ${label ? `<div class="inq-label">${escapeHtml(label)}</div>` : ''}
         <input class="inq-input" type="text" placeholder="${escapeHtml(placeholder)}" />
         <div class="inq-actions">
           <button class="btn-secondary inq-cancel">Cancel</button>
           <button class="btn-primary inq-ok">${escapeHtml(ok)}</button>
         </div>
       </div>`;
    document.body.appendChild(back);
    const input = back.querySelector('.inq-input');
    input.value = value;
    const done = (val) => { back.remove(); resolve(val); };
    back.querySelector('.inq-cancel').addEventListener('click', () => done(null));
    back.querySelector('.inq-ok').addEventListener('click', () => done(input.value.trim() || null));
    back.addEventListener('mousedown', e => { if (e.target === back) done(null); });
    input.addEventListener('keydown', e => {
      if (e.key === 'Enter') { e.preventDefault(); done(input.value.trim() || null); }
      if (e.key === 'Escape') { e.preventDefault(); done(null); }
    });
    requestAnimationFrame(() => { back.classList.add('open'); input.focus(); input.select(); });
  });
}

/* ── hearth wiring ────────────────────────────────────────────────── */
document.querySelectorAll('.hearth-tool[data-panel]').forEach(btn => {
  btn.addEventListener('click', () => {
    const p = btn.dataset.panel;
    if (p === 'sessions') panelSessions();
    else if (p === 'memory') panelMemory();
    else if (p === 'notes') panelNotes();
    else if (p === 'dm') panelDM();
  });
});
document.querySelectorAll('.seat-edit').forEach(b =>
  b.addEventListener('click', () => panelSheet(b.dataset.who)));
$h('hearth-panel-close').addEventListener('click', closePanel);
$h('hearth-new-camp').addEventListener('click', async () => {
  const name = await inlinePrompt('New campaign', {
    label: 'Name your campaign:', value: 'A New Tale', placeholder: 'A New Tale', ok: 'Create',
  });
  if (name) await hpost('/api/hearth/campaign/new', { name });
});
$h('hearth-campaign').addEventListener('change', e =>
  hpost('/api/hearth/campaign/active', { id: e.target.value }));
$h('hearth-board-toggle').addEventListener('click', () =>
  hpost('/api/hearth/board', { enabled: !(hearth.campaign?.board?.enabled) }));

function sendHearth() {
  const t = $h('hearth-input').value.trim();
  if (!t) return;
  $h('hearth-input').value = '';
  hpost('/api/hearth/say', { text: t });
}
$h('hearth-send').addEventListener('click', sendHearth);
$h('hearth-input').addEventListener('keydown', e => { if (e.key === 'Enter') { e.preventDefault(); sendHearth(); } });
$h('hearth-continue').addEventListener('click', () => hpost('/api/hearth/continue', {}));
$h('hearth-session-zero').addEventListener('click', openSessionZero);

/* ═══════════════════════════════════════════════════════════════════
   BEDROCK — a shared calendar that slides in over Magma. He edits events;
   she sees what's coming up and can add events herself (<event> directive).
   ═══════════════════════════════════════════════════════════════════ */
let bedrockOpen = false;
let calYear, calMonth;            // month currently shown
let calSelected = null;           // selected day (YYYY-MM-DD)
let calToday = null;              // today's ISO, from the server
let calEvents = [];              // events for the shown month
const MONTHS = ['January','February','March','April','May','June','July',
  'August','September','October','November','December'];

function openBedrock() {
  if (bedrockOpen) return;
  bedrockOpen = true;
  document.getElementById('view-notes')?.classList.add('bedrock-open');
  const now = new Date();
  if (calYear == null) { calYear = now.getFullYear(); calMonth = now.getMonth() + 1; }
  loadCalendar();
}
function closeBedrock() {
  if (!bedrockOpen) return;
  bedrockOpen = false;
  document.getElementById('view-notes')?.classList.remove('bedrock-open');
}

async function loadCalendar() {
  try {
    const data = await (await fetch(`/api/calendar?year=${calYear}&month=${calMonth}`)).json();
    calToday = data.today;
    calEvents = data.events || [];
    renderCalendar();
  } catch (e) {
    const grid = document.getElementById('cal-grid');
    if (grid) grid.innerHTML = '<div class="cal-empty">Couldn’t reach the calendar.</div>';
  }
}

function eventsOn(iso) {
  return calEvents.filter(e => e.date === iso)
    .sort((a, b) => (a.time || '00:00').localeCompare(b.time || '00:00'));
}

function renderCalendar() {
  const label = document.getElementById('bedrock-monthlabel');
  if (label) label.textContent = `${MONTHS[calMonth - 1]} ${calYear}`;
  const grid = document.getElementById('cal-grid');
  if (!grid) return;
  grid.innerHTML = '';
  const first = new Date(calYear, calMonth - 1, 1);
  const startDow = first.getDay();                          // 0=Sun
  const daysInMonth = new Date(calYear, calMonth, 0).getDate();
  for (let i = 0; i < startDow; i++) {
    const blank = document.createElement('div'); blank.className = 'cal-cell blank';
    grid.appendChild(blank);
  }
  for (let d = 1; d <= daysInMonth; d++) {
    const iso = `${calYear}-${String(calMonth).padStart(2,'0')}-${String(d).padStart(2,'0')}`;
    const cell = document.createElement('div');
    cell.className = 'cal-cell';
    if (iso === calToday) cell.classList.add('today');
    if (iso === calSelected) cell.classList.add('selected');
    const num = document.createElement('div'); num.className = 'cal-num'; num.textContent = d;
    cell.appendChild(num);
    const evs = eventsOn(iso);
    if (evs.length) {
      const dots = document.createElement('div'); dots.className = 'cal-dots';
      evs.slice(0, 3).forEach(() => { const s = document.createElement('span'); dots.appendChild(s); });
      cell.appendChild(dots);
    }
    cell.addEventListener('click', () => { calSelected = iso; renderCalendar(); renderCalDay(); });
    grid.appendChild(cell);
  }
  renderCalDay();
}

function renderCalDay() {
  const title = document.getElementById('cal-day-title');
  const list = document.getElementById('cal-day-events');
  const add = document.getElementById('cal-add');
  if (!title || !list) return;
  if (!calSelected) {
    title.textContent = 'Pick a day';
    list.innerHTML = '';
    if (add) add.style.display = 'none';
    return;
  }
  const [y, m, d] = calSelected.split('-').map(Number);
  const dateObj = new Date(y, m - 1, d);
  title.textContent = dateObj.toLocaleDateString(undefined,
    { weekday: 'long', month: 'long', day: 'numeric' });
  if (add) add.style.display = '';
  list.innerHTML = '';
  const evs = eventsOn(calSelected);
  if (!evs.length) list.innerHTML = '<div class="cal-empty">Nothing planned.</div>';
  evs.forEach(e => {
    const row = document.createElement('div'); row.className = 'cal-event';
    const t = document.createElement('span'); t.className = 'cal-event-time';
    t.textContent = e.time ? fmtTime(e.time) : 'all day';
    const body = document.createElement('div'); body.className = 'cal-event-body';
    const ttl = document.createElement('div'); ttl.className = 'cal-event-title'; ttl.textContent = e.title;
    body.appendChild(ttl);
    if (e.notes) { const n = document.createElement('div'); n.className = 'cal-event-notes'; n.textContent = e.notes; body.appendChild(n); }
    const rm = document.createElement('button'); rm.className = 'cal-event-rm'; rm.textContent = '✕'; rm.title = 'Delete';
    rm.addEventListener('click', () => deleteEvent(e.id));
    row.append(t, body, rm);
    list.appendChild(row);
  });
}

function fmtTime(hhmm) {
  const [h, m] = hhmm.split(':').map(Number);
  const ampm = h < 12 ? 'AM' : 'PM';
  const h12 = h % 12 || 12;
  return `${h12}:${String(m).padStart(2,'0')} ${ampm}`;
}

async function addCalEvent() {
  const titleEl = document.getElementById('cal-add-title');
  const timeEl = document.getElementById('cal-add-time');
  const title = (titleEl?.value || '').trim();
  if (!title || !calSelected) return;
  await fetch('/api/calendar/add', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ date: calSelected, title, time: timeEl?.value || '' }),
  });
  if (titleEl) titleEl.value = '';
  if (timeEl) timeEl.value = '';
  loadCalendar();
}
async function deleteEvent(id) {
  await fetch('/api/calendar/delete', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ id }),
  });
  loadCalendar();
}

document.getElementById('bedrock-open')?.addEventListener('click', openBedrock);
document.getElementById('bedrock-back')?.addEventListener('click', closeBedrock);
document.getElementById('bedrock-prev')?.addEventListener('click', () => {
  calMonth--; if (calMonth < 1) { calMonth = 12; calYear--; } loadCalendar();
});
document.getElementById('bedrock-next')?.addEventListener('click', () => {
  calMonth++; if (calMonth > 12) { calMonth = 1; calYear++; } loadCalendar();
});
document.getElementById('bedrock-today')?.addEventListener('click', () => {
  const now = new Date(); calYear = now.getFullYear(); calMonth = now.getMonth() + 1;
  calSelected = calToday; loadCalendar();
});
document.getElementById('cal-add-btn')?.addEventListener('click', addCalEvent);
document.getElementById('cal-add-title')?.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') { e.preventDefault(); addCalEvent(); }
});

/* ═══════════════════════════════════════════════════════════════════
   PASSTHROUGH MODE
   Ambient view: her ball + the conversation + minimal mic/voice controls.
   Reuses the chat view in place (body.passthrough); double-clicking her
   collapses the chat (body.chat-collapsed). The mini mic/voice buttons
   proxy the real sidebar toggles so all the logic stays in one place.
   ═══════════════════════════════════════════════════════════════════ */
// var (not const): syncPassthroughControls() is reachable from applyMicUI(),
// which runs at boot BEFORE this point — var hoists to undefined so the guard
// below works instead of throwing a temporal-dead-zone ReferenceError.
var passthroughBtn = document.getElementById('passthrough-btn');
var ptBar    = document.getElementById('pt-bar');
var ptMicBtn = document.getElementById('pt-mic');
var ptVoBtn  = document.getElementById('pt-voice');
var ptExit   = document.getElementById('pt-exit');
var ptHint   = document.getElementById('pt-hint');
let passthrough = false;

function syncPassthroughControls() {
  if (!ptMicBtn) return;   // refs not built yet (called from applyMicUI on boot)
  ptMicBtn.classList.toggle('off', micToggle.classList.contains('off'));
  ptVoBtn.classList.toggle('muted', voiceToggle.classList.contains('muted'));
}

function updatePtHint() {
  if (!ptHint) return;
  const collapsed = document.body.classList.contains('chat-collapsed');
  ptHint.textContent = collapsed
    ? 'double-click her to show the conversation'
    : 'double-click her to hide the conversation';
}

function enterPassthrough() {
  if (passthrough) return;
  passthrough = true;
  switchView('chat');
  document.body.classList.add('passthrough');
  document.body.classList.remove('chat-collapsed');
  window.electron?.passthroughEnter?.();   // reshape the window to a phone ratio
  syncPassthroughControls();
  updatePtHint();
  inputEl.focus();
}

function exitPassthrough() {
  if (!passthrough) return;
  passthrough = false;
  document.body.classList.remove('passthrough', 'chat-collapsed');
  window.electron?.passthroughExit?.();    // restore the normal window
}

function toggleChatCollapsed() {
  if (!passthrough) return;
  document.body.classList.toggle('chat-collapsed');
  updatePtHint();
  if (!document.body.classList.contains('chat-collapsed')) inputEl.focus();
}

passthroughBtn?.addEventListener('click', enterPassthrough);
ptExit?.addEventListener('click', exitPassthrough);
// Mini controls proxy the real toggles; sync visual state right after.
ptMicBtn?.addEventListener('click', () => { micToggle.click(); setTimeout(syncPassthroughControls, 0); });
ptVoBtn?.addEventListener('click', () => { voiceToggle.click(); });
// Double-click her ball to hide / reveal the conversation. Bind to the
// orb-container, not #aitha-orb: the orb-ring overlays the orb as a sibling
// and would otherwise swallow the clicks before they reach the orb.
document.querySelector('.orb-container')?.addEventListener('dblclick', toggleChatCollapsed);
// Esc leaves passthrough.
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && passthrough && !settingsModal.classList.contains('open')) {
    exitPassthrough();
  }
});

/* ─── Boot ─────────────────────────────────────────────────────────── */
connect();
inputEl.focus();

// Ping keepalive
setInterval(() => {
  if (connected && ws?.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: 'ping' }));
  }
}, 25000);
