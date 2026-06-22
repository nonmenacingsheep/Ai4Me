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

let awaitingSettings = false;   // true between asking for settings and the reply
function openSettings() {
  if (connected && ws?.readyState === WebSocket.OPEN) {
    awaitingSettings = true;   // let the next 'settings' message fully populate
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

  // A 'settings' broadcast can arrive from another client (or our own save)
  // while you're mid-edit. Re-syncing the staged buffers below would silently
  // discard unsaved toggles, so when the panel is already open (and this isn't
  // the reply to our own open request) we only refresh the dropdown option
  // lists and leave the in-progress edits untouched.
  if (settingsModal.classList.contains('open') && !awaitingSettings) {
    const models = opt.models?.length ? opt.models : [cur.model];
    fillSelect(setModel, models, setModel.value || cur.model);
    fillSelect(setVoice, opt.voices || [cur.tts_voice], setVoice.value || cur.tts_voice);
    fillSelect(setDevice, opt.devices?.length ? opt.devices : [cur.tts_device], setDevice.value || cur.tts_device);
    return;
  }

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
  awaitingSettings = false;
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
  voice_mood_map: true, voice_prosody: true, voice_micro_pauses: true, voice_whisper: false,
};
const behToggles = {
  proactive: document.getElementById('beh-proactive-toggle'),
  journaling: document.getElementById('beh-journal-toggle'),
  curiosity: document.getElementById('beh-curiosity-toggle'),
  voice_mood_map: document.getElementById('beh-voice-moodmap-toggle'),
  voice_prosody: document.getElementById('beh-voice-prosody-toggle'),
  voice_micro_pauses: document.getElementById('beh-voice-pauses-toggle'),
  voice_whisper: document.getElementById('beh-voice-whisper-toggle'),
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
const CAP_KEYS = ['notes', 'projects', 'calendar', 'files', 'images', 'web', 'themes', 'music', 'coding'];
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

    case 'company_changed':
      if (activeView === 'foundry') loadFoundry();      // she ran a CEO action — refresh the Foundry
      break;

    case 'company_chat':
      if (data.message) appendFoundryChat(data.message);  // a new group-chat message
      break;

    case 'world_tick':
      worldOnTick(data);   // entities + clock move forward (cheap; layers come via snapshot)
      break;
    case 'world_changed':
      worldOnChanged();   // terrain/flora/wildlife was reshaped — refresh the map
      break;
    case 'world_speed':
      setActiveSpeed(data.speed);   // fast-forward multiplier changed (maybe by the other god)
      break;
    case 'room_changed':
      if (activeView === 'room') loadRoom();   // she reshaped her space — refresh it live
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
  forge:   { accent: '#ff7a18', bg: '#0d0a08', orb: '#ff7a18' },
  foundry: { accent: '#5a8dd6', bg: '#0a0e16', orb: '#5a8dd6' },
  hearth:  { accent: '#f5b14b', bg: '#140d05', orb: '#f5b14b' },
  forest:  { accent: '#43c59e', bg: '#08130d', orb: '#43c59e' },
  rose:    { accent: '#f472b6', bg: '#160810', orb: '#f472b6' },
  ocean:   { accent: '#2dd4bf', bg: '#061413', orb: '#2dd4bf' },
  mono:    { accent: '#9aa7b8', bg: '#0b0d12', orb: '#9aa7b8' },
};
const ALL_PRESET_CLASSES = ['sky', 'warm', 'moody', 'magma', 'forge', 'foundry', 'hearth', 'forest', 'rose', 'ocean', 'mono']
  .map(p => 'chat-theme-' + p);

// Each app-tab remembers its own full theme. Sky (chat) mirrors the backend —
// it's shared with Aitha (two-way). Mantle/Magma/Hearth are local display prefs.
const TAB_DEFAULTS = {
  chat:   { preset: 'default', accent: null, bg: null, orb: null },
  mantle: { preset: 'moody',   accent: null, bg: null, orb: null },
  notes:  { preset: 'magma',   accent: null, bg: null, orb: null },
  forge:  { preset: 'forge',   accent: null, bg: null, orb: null },
  foundry:   { preset: 'moody',   accent: null, bg: null, orb: null },
  hearth: { preset: 'hearth',  accent: null, bg: null, orb: null },
  world:  { preset: 'forge',   accent: '#6fcf97', bg: '#0a1410', orb: '#6fcf97' },
};
const VIEW_ORDER = ['chat', 'mantle', 'notes', 'forge', 'foundry', 'hearth', 'world'];

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
  room: document.getElementById('view-room'),
  notes: document.getElementById('view-notes'),
  forge: document.getElementById('view-forge'),
  foundry: document.getElementById('view-foundry'),
  hearth: document.getElementById('view-hearth'),
  world: document.getElementById('view-world'),
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
  if (name === 'forge') loadForge();   // refresh each visit — her workspace changes
  if (name === 'foundry') loadFoundry();     // refresh each visit — her company moves
  if (name === 'mantle') loadMind();   // refresh each visit — her mind moves
  if (name === 'room') loadRoom();     // refresh each visit — her space changes
  if (name === 'world') loadWorld();   // (re)open the living map
  if (name !== 'world') stopWorld();   // pause the periodic refresh when off-tab
  if (name !== 'room') stopRoomCanvas();   // don't keep the ambient canvas animating off-tab
  if (name !== 'mantle') closeMemLane();   // don't leave the lane (and its matrix) running
  if (name !== 'notes') closeBedrock();    // leaving Magma closes the calendar slide-over
  if (name === 'chat') inputEl.focus();
}

/* ═══════════════════════════════════════════════════════════════════
   FORGE — everything she's making in her code workspace
   ═══════════════════════════════════════════════════════════════════ */
function forgeBytes(n) { n = n || 0; return n < 1024 ? n + ' B' : (n / 1024).toFixed(1) + ' KB'; }
function forgeWhen(sec) {
  if (!sec) return '';
  return new Date(sec * 1000).toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
}
async function loadForge() {
  const wrap = document.getElementById('forge-files');
  const sub = document.getElementById('forge-sub');
  if (!wrap) return;
  wrap.classList.add('loading');
  try {
    const data = await (await fetch('/api/forge')).json();
    wrap.innerHTML = '';
    if (!data.enabled) {
      wrap.innerHTML = '<div class="forge-empty">Her code workspace is off.<br>Turn on <strong>Code workspace</strong> in Settings → Behavior → Capabilities to let her build things here.</div>';
      if (sub) sub.textContent = 'workspace off';
      return;
    }
    const files = data.files || [];
    if (sub) sub.textContent = files.length ? `${files.length} file${files.length > 1 ? 's' : ''}` : 'nothing forged yet';
    if (!files.length) {
      wrap.innerHTML = '<div class="forge-empty">Nothing forged yet.<br>When she writes and runs code, it’ll appear here.</div>';
      return;
    }
    files.forEach(f => {
      const card = document.createElement('div'); card.className = 'forge-card';
      const head = document.createElement('div'); head.className = 'forge-card-head';
      const name = document.createElement('span'); name.className = 'forge-name'; name.textContent = f.name;
      const meta = document.createElement('span'); meta.className = 'forge-meta';
      meta.textContent = `${forgeBytes(f.size)} · ${forgeWhen(f.mtime)}`;
      const titles = document.createElement('div'); titles.className = 'forge-titles';
      titles.append(name, meta);
      const acts = document.createElement('div'); acts.className = 'forge-acts';
      const del = document.createElement('button'); del.className = 'forge-del';
      del.textContent = '✕'; del.title = 'Delete this file';
      del.addEventListener('click', () => deleteForgeFile(f.name, card, del));
      acts.append(del);
      head.append(titles, acts);
      card.append(head);

      if (f.kind === 'image') {
        // Something she drew/rendered — show it.
        const img = document.createElement('img'); img.className = 'forge-media';
        img.src = '/api/forge/raw/' + encodeURIComponent(f.name) + '?t=' + Math.floor(f.mtime || 0);
        img.alt = f.name;
        card.append(img);
      } else if (f.kind === 'html') {
        // A live visual/animation — render it sandboxed so it can run JS but can't
        // touch the app (no same-origin).
        const frame = document.createElement('iframe'); frame.className = 'forge-media forge-frame';
        frame.setAttribute('sandbox', 'allow-scripts');
        frame.srcdoc = f.content || '';
        card.append(frame);
      } else {
        const pre = document.createElement('pre'); pre.className = 'forge-code';
        pre.textContent = f.content || '(empty)';
        card.append(pre);
      }

      // .py files get a Run button + inline output.
      if (/\.py$/i.test(f.name)) {
        const run = document.createElement('button'); run.className = 'forge-run';
        run.textContent = '▸ Run';
        const out = document.createElement('pre'); out.className = 'forge-output'; out.style.display = 'none';
        run.addEventListener('click', () => runForgeFile(f.name, run, out));
        acts.prepend(run);  // Run sits left of the delete ✕
        card.append(out);
      }
      wrap.appendChild(card);
    });
  } catch (_) {
    wrap.innerHTML = '<div class="forge-empty">Couldn’t load the workspace.</div>';
  } finally {
    wrap.classList.remove('loading');
  }
}
async function runForgeFile(name, btn, out) {
  btn.disabled = true;
  const label = btn.textContent;
  btn.textContent = '… running';
  out.style.display = 'block';
  out.textContent = 'running…';
  try {
    const r = await fetch('/api/forge/run', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name }),
    });
    const data = await r.json();
    out.textContent = data.result || '(no output)';
  } catch (_) {
    out.textContent = "couldn't run it.";
  } finally {
    btn.disabled = false;
    btn.textContent = label;
  }
}
async function deleteForgeFile(name, card, btn) {
  if (!confirm(`Delete "${name}"? This can't be undone.`)) return;
  btn.disabled = true;
  try {
    const r = await fetch('/api/forge/delete', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name }),
    });
    const data = await r.json();
    if (data.ok) {
      card.remove();
      loadForge();  // refresh so the grid re-fills and the count updates
    } else {
      btn.disabled = false;
      alert(data.result || "Couldn't delete it.");
    }
  } catch (_) {
    btn.disabled = false;
    alert("Couldn't delete it.");
  }
}
document.getElementById('forge-refresh')?.addEventListener('click', loadForge);

/* ═══════════════════════════════════════════════════════════════════
   FOUNDRY — her company. She's the CEO; he's the Chairman who guides her.
   ═══════════════════════════════════════════════════════════════════ */
const FOUNDRY_COLS = [
  { key: 'backlog',     label: 'Backlog' },
  { key: 'in_progress', label: 'In progress' },
  { key: 'done',        label: 'Done' },
  { key: 'blocked',     label: 'Blocked' },
];
let foundryData = null;       // last loaded company snapshot
let foundrySub = 'overview';  // active sub-tab: overview | team | chat
function foundryEl(tag, cls, text) {
  const el = document.createElement(tag);
  if (cls) el.className = cls;
  if (text != null) el.textContent = text;
  return el;
}
async function loadFoundry() {
  const body = document.getElementById('foundry-body');
  const sub = document.getElementById('foundry-sub');
  const title = document.getElementById('foundry-title');
  const subtabs = document.getElementById('foundry-subtabs');
  const composer = document.getElementById('foundry-composer');
  const beat = document.getElementById('foundry-beat');
  if (!body) return;
  try {
    const data = await (await fetch('/api/company')).json();
    foundryData = (data.enabled && data.company && data.company.founded) ? data.company : null;
    if (!foundryData) {
      body.innerHTML = '';
      if (subtabs) subtabs.style.display = 'none';
      if (composer) composer.style.display = 'none';
      if (beat) beat.style.display = 'none';
      if (title) title.textContent = 'The Foundry';
      if (!data.enabled) {
        if (sub) sub.textContent = 'company off';
        body.appendChild(foundryEl('div', 'foundry-empty',
          'Her company is off. Turn on Company in Settings → Behavior → Capabilities to let her run it as CEO — you’re the Chairman who guides her.'));
      } else {
        if (sub) sub.textContent = 'not founded yet';
        body.appendChild(foundryEl('div', 'foundry-empty',
          'No company yet. Talk to her about what she’d build if she ran her own company — when it clicks, she’ll found it, and it’ll take shape here.'));
      }
      return;
    }
    const co = foundryData;
    if (title) title.textContent = co.name || 'Her Company';
    if (sub) sub.textContent = co.industry || 'her company';
    if (subtabs) subtabs.style.display = 'flex';
    if (beat) beat.style.display = 'inline-flex';
    updateBeatButton(!!co.heartbeat);
    renderFoundrySub();
  } catch (_) {
    body.innerHTML = '';
    body.appendChild(foundryEl('div', 'foundry-empty', 'Couldn’t load her company.'));
  }
}
function updateBeatButton(on) {
  const beat = document.getElementById('foundry-beat');
  const label = document.getElementById('foundry-beat-label');
  if (!beat) return;
  beat.classList.toggle('on', on);
  if (label) label.textContent = on ? 'Heartbeat on' : 'Heartbeat off';
}
function setFoundrySub(name) {
  foundrySub = name;
  document.querySelectorAll('.foundry-subtab').forEach(b =>
    b.classList.toggle('active', b.dataset.sub === name));
  renderFoundrySub();
}
let foundryBoardStatus = 'in_progress';  // which board column is expanded
let foundryOpenTaskId = null;            // which task title is expanded inline
function renderFoundrySub() {
  const body = document.getElementById('foundry-body');
  const composer = document.getElementById('foundry-composer');
  if (!body || !foundryData) return;
  body.innerHTML = '';
  if (composer) composer.style.display = (foundrySub === 'chat') ? 'flex' : 'none';
  if (foundrySub === 'team') renderFoundryTeam(body, foundryData);
  else if (foundrySub === 'chat') renderFoundryChat(body, foundryData);
  else if (foundrySub === 'board') renderFoundryBoard(body, foundryData);
  else if (foundrySub === 'projects') renderFoundryProjects(body, foundryData);
  else renderFoundryOverview(body, foundryData);
}
function renderFoundryOverview(body, co) {
  const head = foundryEl('div', 'foundry-overview');
  if (co.mission) head.appendChild(foundryEl('div', 'foundry-mission', co.mission));
  const c = co.counts || {};
  const stats = foundryEl('div', 'foundry-stats');
  [['Team', c.employees], ['In progress', c.in_progress],
   ['Done', c.done], ['Blocked', c.blocked]].forEach(([k, v]) => {
    const s = foundryEl('div', 'foundry-stat');
    s.appendChild(foundryEl('span', 'foundry-stat-n', String(v || 0)));
    s.appendChild(foundryEl('span', 'foundry-stat-l', k));
    stats.appendChild(s);
  });
  head.appendChild(stats);
  body.appendChild(head);

  // A compact peek at the board; the Board sub-tab is the full interactive view.
  const tasks = co.tasks || [];
  const recent = [...tasks].sort((a, b) => (b.updated || 0) - (a.updated || 0)).slice(0, 5);
  if (recent.length) {
    const sec = foundryEl('div', 'foundry-section');
    const t = foundryEl('div', 'foundry-section-title', 'Latest activity');
    sec.appendChild(t);
    const list = foundryEl('div', 'foundry-mini-tasks');
    recent.forEach(tk => {
      const row = foundryEl('div', `foundry-mini-task foundry-task-${tk.status || 'backlog'}`);
      row.appendChild(foundryEl('span', 'foundry-mini-dot'));
      row.appendChild(foundryEl('span', 'foundry-mini-title', tk.title));
      row.appendChild(foundryEl('span', 'foundry-mini-status', (tk.status || 'backlog').replace('_', ' ')));
      row.title = 'Open';
      row.addEventListener('click', () => openTaskModal(tk));
      list.appendChild(row);
    });
    sec.appendChild(list);
    body.appendChild(sec);
  }

  const projs = (co.projects || []).filter(p => p.status === 'active');
  if (projs.length) {
    const sec = foundryEl('div', 'foundry-section');
    sec.appendChild(foundryEl('div', 'foundry-section-title', 'Active projects'));
    const list = foundryEl('div', 'foundry-mini-tasks');
    projs.slice(0, 4).forEach(p => {
      const row = foundryEl('div', 'foundry-mini-task');
      row.appendChild(foundryEl('span', 'foundry-mini-title', p.title));
      list.appendChild(row);
    });
    sec.appendChild(list);
    body.appendChild(sec);
  }

  const decs = co.decisions || [];
  if (decs.length) {
    const sec = foundryEl('div', 'foundry-section');
    sec.appendChild(foundryEl('div', 'foundry-section-title', 'Decisions'));
    const list = foundryEl('div', 'foundry-decisions');
    decs.forEach(d => {
      const row = foundryEl('div', 'foundry-decision');
      row.appendChild(foundryEl('div', 'foundry-decision-text', d.summary));
      if (d.time) row.appendChild(foundryEl('div', 'foundry-decision-time', d.time));
      list.appendChild(row);
    });
    sec.appendChild(list);
    body.appendChild(sec);
  }
}
/* The multi-step board: status counts → click to expand a column's task titles →
   click a title to expand its contents inline → double-click to fullscreen it. */
function renderFoundryBoard(body, co) {
  const tasks = co.tasks || [];
  const counts = foundryEl('div', 'foundry-count-row');
  FOUNDRY_COLS.forEach(col => {
    const inCol = tasks.filter(t => (t.status || 'backlog') === col.key);
    const btn = foundryEl('button', `foundry-count foundry-task-${col.key}`);
    if (col.key === foundryBoardStatus) btn.classList.add('active');
    btn.appendChild(foundryEl('span', 'foundry-count-n', String(inCol.length)));
    btn.appendChild(foundryEl('span', 'foundry-count-l', col.label));
    btn.addEventListener('click', () => {
      foundryBoardStatus = col.key; foundryOpenTaskId = null; renderFoundrySub();
    });
    counts.appendChild(btn);
  });
  body.appendChild(counts);

  const col = FOUNDRY_COLS.find(c => c.key === foundryBoardStatus) || FOUNDRY_COLS[0];
  const inCol = tasks.filter(t => (t.status || 'backlog') === col.key)
                     .sort((a, b) => (b.updated || 0) - (a.updated || 0));
  const list = foundryEl('div', 'foundry-tasklist');
  list.appendChild(foundryEl('div', 'foundry-tasklist-head',
    `${col.label} — ${inCol.length} ${inCol.length === 1 ? 'task' : 'tasks'}`));
  if (!inCol.length) {
    list.appendChild(foundryEl('div', 'foundry-empty', 'Nothing here yet.'));
  }
  inCol.forEach(t => {
    const item = foundryEl('div', 'foundry-titem' + (t.id === foundryOpenTaskId ? ' open' : ''));
    const row = foundryEl('div', 'foundry-titem-row');
    row.appendChild(foundryEl('span', 'foundry-titem-caret', t.id === foundryOpenTaskId ? '▾' : '▸'));
    row.appendChild(foundryEl('span', 'foundry-titem-title', t.title));
    row.appendChild(foundryEl('span', 'foundry-titem-who', t.assignee || 'unassigned'));
    // single click = expand inline; double click = fullscreen
    let clickTimer = null;
    row.addEventListener('click', () => {
      if (clickTimer) return;
      clickTimer = setTimeout(() => {
        clickTimer = null;
        foundryOpenTaskId = (t.id === foundryOpenTaskId) ? null : t.id;
        renderFoundrySub();
      }, 200);
    });
    row.addEventListener('dblclick', () => {
      if (clickTimer) { clearTimeout(clickTimer); clickTimer = null; }
      openTaskModal(t);
    });
    item.appendChild(row);
    if (t.id === foundryOpenTaskId) item.appendChild(taskDetailEl(t));
    list.appendChild(item);
  });
  body.appendChild(list);
}
/* The expanded contents of a task: detail, contributors, output, and its log. */
function taskDetailEl(t) {
  const d = foundryEl('div', 'foundry-titem-body');
  if (t.detail) d.appendChild(foundryEl('div', 'foundry-titem-detail', t.detail));
  const meta = foundryEl('div', 'foundry-titem-meta');
  meta.appendChild(foundryEl('span', 'foundry-chip', (t.status || 'backlog').replace('_', ' ')));
  meta.appendChild(foundryEl('span', 'foundry-chip', '@ ' + (t.assignee || 'unassigned')));
  (t.contributors || []).forEach(c => meta.appendChild(foundryEl('span', 'foundry-chip alt', '+ ' + c)));
  d.appendChild(meta);
  if (t.output) {
    d.appendChild(foundryEl('div', 'foundry-titem-label', 'Work product'));
    d.appendChild(foundryEl('div', 'foundry-titem-output', t.output));
  }
  const log = t.log || [];
  if (log.length) {
    d.appendChild(foundryEl('div', 'foundry-titem-label', 'Log'));
    const l = foundryEl('div', 'foundry-titem-log');
    log.slice(-6).forEach(e => {
      const r = foundryEl('div', 'foundry-log-row');
      if (e.time) r.appendChild(foundryEl('span', 'foundry-log-time', e.time));
      r.appendChild(foundryEl('span', 'foundry-log-note', e.note || ''));
      l.appendChild(r);
    });
    d.appendChild(l);
  }
  const hint = foundryEl('div', 'foundry-titem-hint', 'Double-click the title for fullscreen');
  d.appendChild(hint);
  return d;
}
function openTaskModal(t) {
  const modal = document.getElementById('foundry-modal');
  const body = document.getElementById('foundry-modal-body');
  if (!modal || !body) return;
  body.innerHTML = '';
  body.appendChild(foundryEl('div', 'foundry-modal-title', t.title));
  body.appendChild(taskDetailEl(t));
  modal.style.display = 'flex';
}
function closeTaskModal() {
  const modal = document.getElementById('foundry-modal');
  if (modal) modal.style.display = 'none';
}
function renderFoundryProjects(body, co) {
  const projs = co.projects || [];
  if (!projs.length) {
    body.appendChild(foundryEl('div', 'foundry-empty',
      'No company projects yet. As her company finds bigger initiatives to rally around, they’ll show up here.'));
    return;
  }
  const list = foundryEl('div', 'foundry-projlist');
  projs.forEach(p => {
    const card = foundryEl('div', `foundry-proj foundry-proj-${p.status || 'active'}`);
    const head = foundryEl('div', 'foundry-proj-head');
    head.appendChild(foundryEl('span', 'foundry-proj-title', p.title));
    head.appendChild(foundryEl('span', 'foundry-chip', p.status || 'active'));
    card.appendChild(head);
    if (p.about) card.appendChild(foundryEl('div', 'foundry-proj-about', p.about));
    const log = p.log || [];
    if (log.length) {
      const l = foundryEl('div', 'foundry-titem-log');
      log.slice(-4).forEach(e => {
        const r = foundryEl('div', 'foundry-log-row');
        if (e.time) r.appendChild(foundryEl('span', 'foundry-log-time', e.time));
        r.appendChild(foundryEl('span', 'foundry-log-note', e.note || ''));
        l.appendChild(r);
      });
      card.appendChild(l);
    }
    list.appendChild(card);
  });
  body.appendChild(list);
}
function renderFoundryTeam(body, co) {
  const emps = co.employees || [];
  if (!emps.length) {
    body.appendChild(foundryEl('div', 'foundry-empty', 'No one hired yet. She’ll bring teammates on as her company grows.'));
    return;
  }
  const tasks = co.tasks || [];
  const grid = foundryEl('div', 'foundry-team-grid');
  emps.forEach(e => {
    const card = foundryEl('div', 'foundry-teamcard');
    const head = foundryEl('div', 'foundry-teamcard-head');
    const initials = (e.name || e.role || '?').trim().slice(0, 2).toUpperCase();
    head.appendChild(foundryEl('div', 'foundry-avatar', initials));
    const info = foundryEl('div', 'foundry-emp-info');
    info.appendChild(foundryEl('div', 'foundry-emp-name', e.name || e.role));
    info.appendChild(foundryEl('div', 'foundry-emp-role', e.role || ''));
    head.appendChild(info);
    card.appendChild(head);
    if (e.brief) card.appendChild(foundryEl('div', 'foundry-emp-brief', e.brief));
    const mine = tasks.filter(t => t.assignee_id === e.id);
    const tl = foundryEl('div', 'foundry-emp-tasks');
    if (!mine.length) tl.appendChild(foundryEl('div', 'foundry-col-empty', 'No tasks yet'));
    mine.forEach(t => {
      const row = foundryEl('div', `foundry-emp-task foundry-task-${t.status || 'backlog'}`);
      row.appendChild(foundryEl('span', 'foundry-emp-task-title', t.title));
      row.appendChild(foundryEl('span', 'foundry-emp-task-status', (t.status || 'backlog').replace('_', ' ')));
      tl.appendChild(row);
    });
    card.appendChild(tl);
    grid.appendChild(card);
  });
  body.appendChild(grid);
}
function foundryMsgRow(m) {
  if (m.author_id === 'system' || m.role === 'system') {
    const row = foundryEl('div', 'foundry-msg foundry-msg-system');
    row.appendChild(foundryEl('div', 'foundry-msg-text', m.text));
    return row;
  }
  const who = m.author_id === 'chairman' ? 'chairman' : (m.author_id === 'ceo' ? 'ceo' : 'emp');
  const row = foundryEl('div', `foundry-msg foundry-msg-${who}`);
  const meta = foundryEl('div', 'foundry-msg-meta');
  meta.appendChild(foundryEl('span', 'foundry-msg-author', m.author));
  if (m.role && m.role !== m.author) meta.appendChild(foundryEl('span', 'foundry-msg-role', m.role));
  if (m.time) meta.appendChild(foundryEl('span', 'foundry-msg-time', m.time));
  row.appendChild(meta);
  row.appendChild(foundryEl('div', 'foundry-msg-text', m.text));
  return row;
}
function renderFoundryChat(body, co) {
  const list = foundryEl('div', 'foundry-chat');
  list.id = 'foundry-chat-list';
  const msgs = co.chat || [];
  if (!msgs.length) {
    list.appendChild(foundryEl('div', 'foundry-empty',
      'Quiet so far. Post a message as the Chairman, or flip on the Heartbeat and let the team start passing ideas around.'));
  }
  msgs.forEach(m => list.appendChild(foundryMsgRow(m)));
  body.appendChild(list);
  list.scrollTop = list.scrollHeight;
}
function appendFoundryChat(m) {
  if (foundryData) { foundryData.chat = (foundryData.chat || []); foundryData.chat.push(m); }
  if (activeView !== 'foundry' || foundrySub !== 'chat') return;
  const list = document.getElementById('foundry-chat-list');
  if (!list) return;
  const empty = list.querySelector('.foundry-empty');
  if (empty) empty.remove();
  list.appendChild(foundryMsgRow(m));
  list.scrollTop = list.scrollHeight;
}
async function toggleFoundryBeat() {
  if (!foundryData) return;
  const next = !foundryData.heartbeat;
  updateBeatButton(next);  // optimistic
  try {
    const r = await (await fetch('/api/company/heartbeat', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ on: next }),
    })).json();
    foundryData.heartbeat = !!r.heartbeat;
    updateBeatButton(foundryData.heartbeat);
  } catch (_) { updateBeatButton(foundryData.heartbeat); }
}
async function sendFoundryChat() {
  const input = document.getElementById('foundry-chat-input');
  if (!input) return;
  const text = input.value.trim();
  if (!text) return;
  input.value = '';
  input.style.height = 'auto';
  try {
    await fetch('/api/company/chat', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });
    // the chairman message + team replies arrive via the company_chat socket event
  } catch (_) {}
}
document.getElementById('foundry-modal-close')?.addEventListener('click', closeTaskModal);
document.getElementById('foundry-modal')?.addEventListener('click', (e) => {
  if (e.target.id === 'foundry-modal') closeTaskModal();
});
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') {
    const modal = document.getElementById('foundry-modal');
    if (modal && modal.style.display !== 'none') closeTaskModal();
  }
});
document.getElementById('foundry-refresh')?.addEventListener('click', loadFoundry);
document.getElementById('foundry-beat')?.addEventListener('click', toggleFoundryBeat);
document.getElementById('foundry-subtabs')?.addEventListener('click', (e) => {
  const b = e.target.closest('.foundry-subtab');
  if (b) setFoundrySub(b.dataset.sub);
});
async function conveneFoundryMeeting() {
  const topic = (prompt('Meeting agenda — what should the team settle?') || '').trim();
  if (!topic) return;
  try {
    await fetch('/api/company/meeting', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ topic }),
    });
    // the meeting plays out via the company_chat socket events
  } catch (_) {}
}
document.getElementById('foundry-meeting')?.addEventListener('click', conveneFoundryMeeting);
document.getElementById('foundry-chat-send')?.addEventListener('click', sendFoundryChat);
document.getElementById('foundry-chat-input')?.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendFoundryChat(); }
});

/* ═══════════════════════════════════════════════════════════════════
   ROOM — Aitha's own self-authored space. She sets its name, light,
   atmosphere, and the objects in it; we render it as a living scene.
   ═══════════════════════════════════════════════════════════════════ */
let roomData = null;
let roomRAF = null;          // requestAnimationFrame handle for the ambient canvas
let roomParticles = [];
const ROOM_VIOLET = '#a78bfa';

async function loadRoom() {
  const stage = document.getElementById('room-stage');
  const empty = document.getElementById('room-empty');
  if (!stage || !empty) return;
  try {
    const data = await (await fetch('/api/room')).json();
    if (!data.enabled) {
      stage.style.display = 'none';
      empty.style.display = 'flex';
      empty.textContent = 'Her Room is off. Turn on “Her Room” in Settings → Behavior → Capabilities to give her a space of her own.';
      stopRoomCanvas();
      return;
    }
    roomData = data.room || {};
    const a = roomData.atmosphere || {};
    const named = (roomData.name || '').trim();
    const hasContent = named || (roomData.objects || []).length || (roomData.description || '').trim();
    if (!hasContent) {
      stage.style.display = 'none';
      empty.style.display = 'flex';
      empty.textContent = 'She hasn’t made her room yet. Give her a little while — when something pulls at her, she’ll start shaping a space of her own here.';
    } else {
      empty.style.display = 'none';
      stage.style.display = 'flex';
      document.getElementById('room-name').textContent = named || 'Her Room';
      const vibeEl = document.getElementById('room-vibe');
      vibeEl.textContent = (roomData.vibe || '').trim();
      vibeEl.style.display = vibeEl.textContent ? 'block' : 'none';
      const descEl = document.getElementById('room-desc');
      descEl.textContent = (roomData.description || '').trim();
      descEl.style.display = descEl.textContent ? 'block' : 'none';
      renderRoomObjects(roomData.objects || []);
    }
    applyRoomAtmosphere(a);
    startRoomCanvas(a);
  } catch (_) {
    stage.style.display = 'none';
    empty.style.display = 'flex';
    empty.textContent = 'Couldn’t reach her room just now.';
  }
}

function renderRoomObjects(objects) {
  const wrap = document.getElementById('room-objects');
  wrap.innerHTML = '';
  objects.forEach(o => {
    const card = document.createElement('div');
    card.className = 'room-object';
    const icon = document.createElement('div');
    icon.className = 'room-object-icon';
    icon.textContent = (o.icon || '').trim() || '◦';
    const name = document.createElement('div');
    name.className = 'room-object-name';
    name.textContent = o.name || '';
    card.append(icon, name);
    if ((o.note || '').trim()) {
      const note = document.createElement('div');
      note.className = 'room-object-note';
      note.textContent = o.note;
      card.appendChild(note);
      card.classList.add('has-note');
    }
    wrap.appendChild(card);
  });
}

// Paint her chosen palette onto the whole view (accent, bg, orb) + a light wash.
function applyRoomAtmosphere(a) {
  const accent = a.accent || ROOM_VIOLET;
  const theme = { preset: 'default', accent, bg: a.bg || null, orb: accent };
  applyThemeObject(theme);
  const light = document.getElementById('room-light');
  if (light) {
    const glow = a.glow || accent;
    const lighting = a.lighting || 'soft';
    const INT = { dim: 0.10, soft: 0.18, cool: 0.18, warm: 0.26, candle: 0.30, bright: 0.40 };
    const amt = INT[lighting] ?? 0.18;
    const pos = (lighting === 'candle' || lighting === 'warm') ? '50% 85%' : '50% 18%';
    light.style.background = `radial-gradient(60% 55% at ${pos}, ${hexA(glow, amt)}, transparent 70%)`;
  }
}
function hexA(hex, alpha) {
  const c = hexToRgb(hex) || hexToRgb(ROOM_VIOLET);
  return `rgba(${c.r}, ${c.g}, ${c.b}, ${alpha})`;
}

/* The ambient canvas — a living scene driven by her chosen motion mode. Each mode
   is a small particle behaviour, tinted by her accent/glow so the space breathes
   in her colours. */
function startRoomCanvas(a) {
  const canvas = document.getElementById('room-canvas');
  if (!canvas) return;
  stopRoomCanvas();
  const ctx = canvas.getContext('2d');
  const mode = a.motion || 'drift';
  const accent = hexToRgb(a.accent || ROOM_VIOLET);
  const glow = hexToRgb(a.glow || a.accent || ROOM_VIOLET);
  let W = 0, H = 0, dpr = Math.min(window.devicePixelRatio || 1, 2);
  function resize() {
    const r = canvas.getBoundingClientRect();
    W = r.width; H = r.height;
    canvas.width = Math.max(1, W * dpr); canvas.height = Math.max(1, H * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  resize();
  const COUNTS = { still: 0, drift: 46, embers: 60, rain: 110, stars: 90, mist: 14 };
  const n = COUNTS[mode] ?? 40;
  const col = (mode === 'embers') ? glow : accent;
  roomParticles = [];
  const rnd = (a2, b) => a2 + Math.random() * (b - a2);
  for (let i = 0; i < n; i++) roomParticles.push(spawn(mode, W, H, rnd, true));

  function spawn(m, w, h, rnd, initial) {
    if (m === 'embers') return { x: rnd(0, w), y: initial ? rnd(0, h) : h + 8, vy: -rnd(8, 26), vx: rnd(-6, 6), r: rnd(1, 2.6), a: rnd(0.3, 0.9), tw: rnd(0, 6.28) };
    if (m === 'rain')   return { x: rnd(0, w), y: initial ? rnd(0, h) : -10, vy: rnd(220, 380), vx: rnd(-10, 0), len: rnd(8, 18), a: rnd(0.15, 0.4) };
    if (m === 'stars')  return { x: rnd(0, w), y: rnd(0, h), r: rnd(0.5, 1.6), a: rnd(0.2, 0.9), tw: rnd(0, 6.28), ts: rnd(0.6, 2.0) };
    if (m === 'mist')   return { x: rnd(-0.2 * w, w), y: rnd(0.15 * h, 0.95 * h), vx: rnd(4, 14), r: rnd(60, 160), a: rnd(0.02, 0.06) };
    /* drift */          return { x: rnd(0, w), y: rnd(0, h), vy: -rnd(4, 14), vx: rnd(-6, 6), r: rnd(0.8, 2.4), a: rnd(0.15, 0.55), tw: rnd(0, 6.28) };
  }

  let last = performance.now();
  function frame(now) {
    const dt = Math.min(0.05, (now - last) / 1000); last = now;
    ctx.clearRect(0, 0, W, H);
    const c = `${col.r}, ${col.g}, ${col.b}`;
    for (const p of roomParticles) {
      if (mode === 'rain') {
        p.y += p.vy * dt; p.x += p.vx * dt;
        ctx.strokeStyle = `rgba(${c}, ${p.a})`; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(p.x, p.y); ctx.lineTo(p.x + p.vx * 0.04, p.y + p.len); ctx.stroke();
        if (p.y > H + 20) Object.assign(p, spawn(mode, W, H, rnd, false));
      } else if (mode === 'mist') {
        p.x += p.vx * dt;
        const g = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, p.r);
        g.addColorStop(0, `rgba(${c}, ${p.a})`); g.addColorStop(1, `rgba(${c}, 0)`);
        ctx.fillStyle = g; ctx.beginPath(); ctx.arc(p.x, p.y, p.r, 0, 6.2832); ctx.fill();
        if (p.x - p.r > W) { p.x = -p.r; p.y = rnd(0.15 * H, 0.95 * H); }
      } else if (mode === 'stars') {
        p.tw += p.ts * dt; const tw = (Math.sin(p.tw) + 1) / 2;
        ctx.fillStyle = `rgba(${c}, ${p.a * (0.35 + 0.65 * tw)})`;
        ctx.beginPath(); ctx.arc(p.x, p.y, p.r, 0, 6.2832); ctx.fill();
      } else if (mode !== 'still') { /* drift / embers */
        p.y += p.vy * dt; p.x += p.vx * dt; p.tw += dt;
        const fl = mode === 'embers' ? (0.7 + 0.3 * Math.sin(p.tw * 3)) : 1;
        ctx.fillStyle = `rgba(${c}, ${p.a * fl})`;
        ctx.beginPath(); ctx.arc(p.x, p.y, p.r, 0, 6.2832); ctx.fill();
        if (p.y < -10 || p.x < -20 || p.x > W + 20) Object.assign(p, spawn(mode, W, H, rnd, false));
      }
    }
    roomRAF = requestAnimationFrame(frame);
  }
  canvas._roomResize = resize;
  window.addEventListener('resize', resize);
  if (mode === 'still') { ctx.clearRect(0, 0, W, H); return; }
  roomRAF = requestAnimationFrame(frame);
}
function stopRoomCanvas() {
  if (roomRAF) { cancelAnimationFrame(roomRAF); roomRAF = null; }
  const canvas = document.getElementById('room-canvas');
  if (canvas && canvas._roomResize) { window.removeEventListener('resize', canvas._roomResize); canvas._roomResize = null; }
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

/* ═══════════════════════════════════════════════════════════════════
   WORLD — the living god-sim map + brush god-tools
   ═══════════════════════════════════════════════════════════════════ */
const WBIOME_RGB = {
  ocean: [29, 58, 95], beach: [217, 200, 154], grassland: [106, 153, 78],
  forest: [56, 102, 65], rainforest: [30, 86, 49], desert: [224, 192, 104],
  savanna: [179, 161, 66], tundra: [138, 160, 160], snow: [232, 238, 242],
  swamp: [74, 93, 58], mountain: [138, 128, 115], rock: [107, 107, 107],
  shingle: [150, 140, 120],            // gravel/pebble shore
};
const WWATER_RGB = { 1: [79, 155, 217], 2: [47, 111, 176], 3: [22, 52, 90], 4: [86, 176, 196] }; // river/lake/ocean/shallow
const WPLANT_TINT = {
  grass: [120, 180, 70], shrub: [110, 150, 70], oak: [40, 95, 45], pine: [28, 72, 50],
  cactus: [80, 140, 90], reeds: [95, 155, 80], palm: [50, 125, 70],
};
const WANIMAL_RGB = { rabbit: [232, 230, 224], deer: [201, 160, 106], wolf: [86, 92, 104] };
// Rendered footprint in TILES per entity (bigger beasts read as ~2×2 = 4 tiles, a
// rabbit as 1). People share the large size. New large creatures should be added at 2.
const WANIMAL_SPAN = { rabbit: 1, deer: 2, wolf: 2 };
const WPERSON_SPAN = 2;
// Trees drawn as ~2×2-tile canopy sprites (per the requested sizing) when zoomed in,
// sampled from the crisp detail window's veg layer. Darker than the terrain tint so
// they read as foliage above the ground.
const WTREE_RGB = { oak: [34, 82, 38], pine: [26, 66, 46], palm: [40, 112, 64] };
const WTREE_SPAN = 2;
// Forageable, non-tree flora — the grasses, reeds, shrubs and cacti people actually pull
// leaves, fibre and food from. Drawn as small clumps (≈1 tile) so these resources have a
// VISIBLE place on the map rather than reading as bare ground a person mysteriously feeds at.
const WBUSH_RGB = {
  grass: [96, 156, 66], shrub: [80, 122, 54], reeds: [108, 150, 74],
  cactus: [74, 126, 86],
};
// Placed-block colours, keyed by block code (1 floor, 2 wall, 3 door, 4 window, 5 fence).
const WBLOCK_RGB = {
  1: [171, 132, 86], 2: [120, 82, 45], 3: [196, 158, 92], 4: [150, 196, 214],
  5: [150, 120, 78], 6: [78, 138, 64],   // 6 = leaf panel (green)
};
const WORE_RGB = {
  copper_ore: [200, 118, 64], tin_ore: [206, 210, 214], iron_ore: [90, 92, 100],
  gold_ore: [226, 196, 78], coal: [40, 40, 46],
};

const WORLD = {
  data: null, base: null, baseCtx: null,
  cam: null,                       // { zoom: px/tile, camX, camY (top-left tile, float) }
  tool: 'inspect', arg: null,
  dragging: false, panning: false, panLX: 0, panLY: 0, bound: false,
  refreshTimer: null, refreshPending: false, lastPaint: 0,
};

function _wb64(b64) {
  const bin = atob(b64); const u = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i);
  return u;
}

async function loadWorld() {
  const off = document.getElementById('world-off');
  const stage = document.getElementById('world-stage');
  if (!stage) return;
  bindWorld();
  try {
    const j = await (await fetch('/api/world')).json();
    if (!j.enabled || !j.world) {
      off.style.display = 'flex'; stage.style.display = 'none';
      stopWorld();
      return;
    }
    off.style.display = 'none'; stage.style.display = 'flex';
    startWorldRefresh();            // (re)arm the slow terrain refresh first…
    setWorld(j.world);              // …then load + kick off the interpolated anim loop
  } catch (e) { console.warn('[world] load failed', e); }
}

function setWorld(s) {
  WORLD.data = {
    // True world size (entities + camera live in this 0..w/0..h tile space)…
    w: s.w, h: s.h,
    // …while terrain ships as a downsampled OVERVIEW (ovw×ovh, each cell = ovStep tiles).
    ovw: s.ovw || s.w, ovh: s.ovh || s.h, ovStep: s.ov_step || 1,
    biomes: s.biomes, plants: s.plants || {}, animals: s.animals || [],
    people: s.people || [], structures: s.structures || [],
    blocks: s.blocks || [], roofs: s.roofs || [], sites: s.sites || [],
    ore: s.ore || [], blockNames: s.block_names || {},
    elevation: _wb64(s.layers.elevation), biome: _wb64(s.layers.biome),
    water: _wb64(s.layers.water), vegSp: _wb64(s.layers.veg_sp),
    vegGrowth: _wb64(s.layers.veg_growth),
    // Keep any crisp detail window already streamed (a periodic terrain refresh shares
    // the same camera, so reusing it avoids a blurry flash); re-stream replaces it below.
    detail: (WORLD.data && WORLD.data.detail) || null,
    day: s.day, time: s.time, season: s.season, weather: s.weather,
    census: null, version: s.version,
  };
  _wDetailKey = '';                         // force refreshWorldDetail to re-stream crisp tiles
  buildWorldBase();
  computeWorldCamera();
  refreshWorldDetail();
  updateWorldHud();
  setActiveSpeed(s.speed || 1);             // reflect the saved fast-forward multiplier
  startWorldAnim();                         // continuous, interpolated rendering loop
}

// ── Smooth, real-time rendering ──────────────────────────────────────────────
// World ticks arrive a few times a second; rather than redraw only on each tick
// (which looks like discrete tile-hops), we run a requestAnimationFrame loop that
// eases every entity toward its latest reported tile each frame, so motion is fluid
// at the display's refresh rate even when the server tick rate is modest.
function worldSmooth(id, tx, ty) {
  let m = WORLD.smooth;
  if (!m) m = WORLD.smooth = new Map();
  let s = m.get(id);
  if (!s) { s = { x: tx, y: ty }; m.set(id, s); return s; }   // new entity → start in place
  return s;
}
function easeWorldEntities() {
  const d = WORLD.data; if (!d) return;
  const k = 0.28;                           // per-frame approach factor (≈ smooth follow)
  const seen = new Set();
  for (const e of [...(d.animals || []), ...(d.people || [])]) {
    const s = worldSmooth(e.id, e.x, e.y); seen.add(e.id);
    const dx = e.x - s.x, dy = e.y - s.y;
    s.x += dx * k; s.y += dy * k;
    if (Math.abs(dx) > 6 || Math.abs(dy) > 6) { s.x = e.x; s.y = e.y; }  // teleport (placed)
  }
  if (WORLD.smooth) for (const id of WORLD.smooth.keys()) if (!seen.has(id)) WORLD.smooth.delete(id);
}
function worldAnimFrame() {
  if (activeView !== 'world' || !WORLD.data) { WORLD.animRAF = null; return; }
  easeWorldEntities();
  renderWorld();
  updateWorldNet();
  WORLD.animRAF = requestAnimationFrame(worldAnimFrame);
}

// Surfaces a warning in the HUD only when live world_tick updates are lagging — so it's
// immediately visible (no DevTools) whether the freeze is the server not sending or the
// renderer not receiving. Hidden when updates are flowing normally.
function updateWorldNet() {
  const el = document.getElementById('whud-net'); if (!el) return;
  const age = WORLD._lastTickWall ? (performance.now() - WORLD._lastTickWall) / 1000 : 0;
  if (age > 2) { el.textContent = `⚠ last update ${age.toFixed(0)}s ago`; el.style.display = ''; }
  else { el.style.display = 'none'; }
}
function startWorldAnim() {
  if (WORLD.animRAF || activeView !== 'world') return;
  WORLD.animRAF = requestAnimationFrame(worldAnimFrame);
}

// Colour one tile pixel (biome + west-light hillshade + veg tint, or shaded water).
function _wPaintPixel(px, o, i, el, bi, wa, vs, vg, leftElev, biomes, plants) {
  const elev = el[i] / 255;
  let r, g, b;
  const water = wa[i];
  if (water) {
    const c = WWATER_RGB[water] || WWATER_RGB[2];
    const sh = 0.7 + 0.6 * elev;
    r = c[0] * sh; g = c[1] * sh; b = c[2] * sh;
  } else {
    const c = WBIOME_RGB[biomes[bi[i]]] || [120, 120, 120];
    let shade = 0.74 + elev * 0.34 + (elev - leftElev) * 1.7;
    shade = Math.max(0.42, Math.min(1.4, shade));
    r = c[0] * shade; g = c[1] * shade; b = c[2] * shade;
    const sp = vs[i];
    if (sp) {
      const t = WPLANT_TINT[plants[sp]] || [90, 150, 70];
      const k = 0.14 + 0.5 * (vg[i] / 255);
      r = r * (1 - k) + t[0] * k; g = g * (1 - k) + t[1] * k; b = b * (1 - k) + t[2] * k;
    }
  }
  px[o] = r; px[o + 1] = g; px[o + 2] = b; px[o + 3] = 255;
}

// Paint the downsampled overview into an offscreen canvas (ovw×ovh). renderWorld
// stretches it across the whole world; a crisp detail window overlays it when zoomed.
function buildWorldBase() {
  const d = WORLD.data; if (!d) return;
  if (!WORLD.base) { WORLD.base = document.createElement('canvas'); }
  WORLD.base.width = d.ovw; WORLD.base.height = d.ovh;
  WORLD.baseCtx = WORLD.base.getContext('2d');
  const img = WORLD.baseCtx.createImageData(d.ovw, d.ovh);
  const px = img.data;
  for (let y = 0; y < d.ovh; y++) {
    for (let x = 0; x < d.ovw; x++) {
      const i = y * d.ovw + x;
      const leftElev = (x > 0 ? d.elevation[i - 1] : d.elevation[i]) / 255;
      _wPaintPixel(px, i * 4, i, d.elevation, d.biome, d.water, d.vegSp, d.vegGrowth,
        leftElev, d.biomes, d.plants);
    }
  }
  WORLD.baseCtx.putImageData(img, 0, 0);
}

// Build a crisp tile-accurate canvas from a streamed view window.
function buildDetailCanvas(v) {
  const vw = v.vw, vh = v.vh;
  const el = _wb64(v.layers.elevation), bi = _wb64(v.layers.biome), wa = _wb64(v.layers.water);
  const vs = _wb64(v.layers.veg_sp), vg = _wb64(v.layers.veg_growth);
  const cvs = document.createElement('canvas'); cvs.width = vw; cvs.height = vh;
  const cx = cvs.getContext('2d');
  const img = cx.createImageData(vw, vh); const px = img.data;
  const d = WORLD.data;
  for (let y = 0; y < vh; y++) {
    for (let x = 0; x < vw; x++) {
      const i = y * vw + x;
      const leftElev = (x > 0 ? el[i - 1] : el[i]) / 255;
      _wPaintPixel(px, i * 4, i, el, bi, wa, vs, vg, leftElev, d.biomes, d.plants);
    }
  }
  cx.putImageData(img, 0, 0);
  // Keep the veg layer + step so renderWorld can draw crisp tree-canopy sprites on top.
  // _id changes each time a fresh window is streamed, invalidating the foliage cache.
  return { canvas: cvs, x0: v.x0, y0: v.y0, x1: v.x1, y1: v.y1, vw, vh,
           step: v.step || 1, vegSp: vs, vegGrowth: vg, _id: ++_wDetailId };
}

// When zoomed in enough that the overview looks blocky, stream a crisp window of the
// visible area at an appropriate level-of-detail (debounced; skipped when overview suffices).
let _wDetailTimer = null, _wDetailKey = '', _wDetailId = 0;
function refreshWorldDetail() {
  const d = WORLD.data, cv = document.getElementById('world-canvas');
  if (!d || !cv || !WORLD.cam) return;
  const cam = WORLD.cam, z = cam.zoom;
  if (z * d.ovStep < 6) { d.detail = null; _wDetailKey = ''; return; }   // overview is crisp enough
  const x0 = Math.max(0, Math.floor(cam.camX));
  const y0 = Math.max(0, Math.floor(cam.camY));
  const x1 = Math.min(d.w, Math.ceil(cam.camX + cv.width / z) + 1);
  const y1 = Math.min(d.h, Math.ceil(cam.camY + cv.height / z) + 1);
  const step = Math.max(1, Math.ceil(Math.max(x1 - x0, y1 - y0) / 512));
  const key = `${x0},${y0},${x1},${y1},${step}`;
  if (key === _wDetailKey) return;
  clearTimeout(_wDetailTimer);
  _wDetailTimer = setTimeout(async () => {
    try {
      const j = await (await fetch(`/api/world/view?x0=${x0}&y0=${y0}&x1=${x1}&y1=${y1}&step=${step}`)).json();
      if (!j.view || !j.view.vw) return;
      _wDetailKey = key;
      WORLD.data.detail = buildDetailCanvas(j.view);
      renderWorld();
    } catch (_) {}
  }, 120);
}

// Size the canvas to fill its wrapper and set up a camera that *covers* the view
// (the map always fills the stage; never a tiny centered square). Keeps the
// existing camera (zoom/pan) across refreshes; only re-seeds for a new map.
function computeWorldCamera() {
  const d = WORLD.data; if (!d) return;
  const wrap = document.getElementById('world-canvas-wrap');
  const cv = document.getElementById('world-canvas');
  if (!wrap || !cv) return;
  const cw = wrap.clientWidth, ch = wrap.clientHeight;
  if (cw <= 0 || ch <= 0) {
    // Layout not settled yet (tab just became visible) — try again next frame.
    requestAnimationFrame(() => { computeWorldCamera(); renderWorld(); });
    return;
  }
  cv.width = cw; cv.height = ch;
  if (!WORLD.cam || WORLD.cam._w !== d.w || WORLD.cam._h !== d.h) {
    const z = Math.max(cw / d.w, ch / d.h);           // cover
    WORLD.cam = { zoom: z, camX: (d.w - cw / z) / 2, camY: (d.h - ch / z) / 2, _w: d.w, _h: d.h };
  }
  clampCamera();
}

// Keep zoom ≥ cover and the viewport inside the map bounds.
function clampCamera() {
  const d = WORLD.data, cv = document.getElementById('world-canvas');
  if (!d || !cv || !WORLD.cam) return;
  const cam = WORLD.cam, cw = cv.width, ch = cv.height;
  const minZoom = Math.max(cw / d.w, ch / d.h);
  cam.zoom = Math.max(minZoom, Math.min(48, cam.zoom));
  const viewW = cw / cam.zoom, viewH = ch / cam.zoom;
  cam.camX = Math.max(0, Math.min(d.w - viewW, cam.camX));
  cam.camY = Math.max(0, Math.min(d.h - viewH, cam.camY));
}

// Render every tree-canopy + forage-clump sprite in the current detail window onto an
// offscreen canvas (screen-space), so renderWorld can blit it instead of recomputing the
// whole field every frame. Rebuilt only when the view changes (see the caller's key).
function buildFoliageLayer(cv, cam, z) {
  const d = WORLD.data, dt = d.detail; if (!dt) return;
  let fc = WORLD.foliage;
  if (!fc) fc = WORLD.foliage = document.createElement('canvas');
  if (fc.width !== cv.width || fc.height !== cv.height) { fc.width = cv.width; fc.height = cv.height; }
  const ctx = fc.getContext('2d');
  ctx.clearRect(0, 0, fc.width, fc.height);
  const vs = dt.vegSp, vg = dt.vegGrowth;
  const onScreen = (sx, sy, span) => !(sx < -span * z || sy < -span * z || sx > cv.width || sy > cv.height);
  for (let cy = 0; cy < dt.vh; cy++) {
    for (let cx2 = 0; cx2 < dt.vw; cx2++) {
      const sp = vs[cy * dt.vw + cx2];
      const name = sp ? d.plants[sp] : null;
      if (!name) continue;
      const treeCol = WTREE_RGB[name];
      const bushCol = !treeCol ? WBUSH_RGB[name] : null;
      if (!treeCol && !bushCol) continue;
      const gnorm = vg[cy * dt.vw + cx2] / 255;
      const sx = (dt.x0 + cx2 - cam.camX) * z, sy = (dt.y0 + cy - cam.camY) * z;
      if (!onScreen(sx, sy, WTREE_SPAN)) continue;
      const ccx = sx + z / 2, ccy = sy + z / 2;
      if (treeCol) {
        const grow = 0.62 + 0.38 * gnorm;
        const rad = WTREE_SPAN * z * 0.5 * grow;
        ctx.fillStyle = 'rgba(40,28,16,0.9)';               // short trunk
        ctx.fillRect(ccx - z * 0.08, ccy, z * 0.16, rad * 0.8);
        ctx.fillStyle = `rgb(${treeCol[0]},${treeCol[1]},${treeCol[2]})`; // canopy
        ctx.beginPath();
        if (name === 'pine') {                              // conifer: a tall triangle
          ctx.moveTo(ccx, ccy - rad); ctx.lineTo(ccx + rad * 0.8, ccy + rad * 0.6);
          ctx.lineTo(ccx - rad * 0.8, ccy + rad * 0.6); ctx.closePath();
        } else {
          ctx.arc(ccx, ccy, rad, 0, 6.283);
        }
        ctx.fill();
        ctx.strokeStyle = 'rgba(0,0,0,0.35)'; ctx.lineWidth = 1; ctx.stroke();
      } else {
        // A low forage clump (grass tuft / shrub / reeds / cactus) — only once it's grown
        // enough to be worth gathering, so a sparse tile still reads as bare.
        if (gnorm < 0.22) continue;
        const r = z * (0.2 + 0.16 * gnorm);
        ctx.fillStyle = `rgb(${bushCol[0]},${bushCol[1]},${bushCol[2]})`;
        if (name === 'cactus') {                            // a stout upright pad
          ctx.fillRect(ccx - r * 0.5, ccy - r, r, r * 1.8);
          ctx.fillRect(ccx - r, ccy - r * 0.2, r * 0.5, r * 0.9);
          ctx.fillRect(ccx + r * 0.5, ccy - r * 0.4, r * 0.5, r);
        } else {
          for (const [ox, oy] of [[-0.18, 0.06], [0.18, 0.06], [0, -0.12]]) {
            ctx.beginPath();
            ctx.arc(ccx + ox * z, ccy + oy * z, r, 0, 6.283); ctx.fill();
          }
        }
        ctx.strokeStyle = 'rgba(0,0,0,0.22)'; ctx.lineWidth = 1; ctx.stroke();
      }
    }
  }
}

function renderWorld() {
  const d = WORLD.data; if (!d || !WORLD.base || !WORLD.cam) return;
  const cv = document.getElementById('world-canvas');
  if (!cv || activeView !== 'world') return;
  const ctx = cv.getContext('2d');
  const cam = WORLD.cam, z = cam.zoom;
  ctx.imageSmoothingEnabled = false;
  ctx.fillStyle = '#06100c';
  ctx.fillRect(0, 0, cv.width, cv.height);
  // Overview terrain: the ovw×ovh base stretched across the whole world extent.
  ctx.drawImage(WORLD.base, 0, 0, d.ovw, d.ovh, -cam.camX * z, -cam.camY * z, d.w * z, d.h * z);
  // Crisp detail window overlaid where we've streamed it (when zoomed in).
  if (d.detail) {
    const dt = d.detail;
    ctx.drawImage(dt.canvas, 0, 0, dt.vw, dt.vh,
      (dt.x0 - cam.camX) * z, (dt.y0 - cam.camY) * z, (dt.x1 - dt.x0) * z, (dt.y1 - dt.y0) * z);
  }
  const onScreen = (sx, sy, span) => !(sx < -span * z || sy < -span * z || sx > cv.width || sy > cv.height);
  // Foliage sprites (tree canopies + forage clumps) are STATIC relative to the terrain, but
  // there can be tens of thousands of them in view — redrawing them every animation frame
  // tanked the framerate, which backed up the websocket and starved the world's tick loop
  // (entities then "teleported"). So we render them ONCE to an offscreen layer and just blit
  // it each frame, rebuilding only when the view (zoom/pan/detail window) actually changes.
  if (d.detail && d.detail.step === 1 && z >= 6) {
    const key = `${d.detail._id || 0}|${z.toFixed(2)}|${cam.camX.toFixed(1)}|${cam.camY.toFixed(1)}|${cv.width}x${cv.height}`;
    if (key !== WORLD._foliageKey) { buildFoliageLayer(cv, cam, z); WORLD._foliageKey = key; }
    if (WORLD.foliage) ctx.drawImage(WORLD.foliage, 0, 0);
  } else {
    WORLD._foliageKey = null;
  }
  // Ore deposits: a small gem on the rock once you've zoomed in enough to mine-scout.
  if (z >= 3) {
    for (const o of (d.ore || [])) {
      const sx = (o.x - cam.camX) * z, sy = (o.y - cam.camY) * z;
      if (!onScreen(sx, sy, 1)) continue;
      const c = WORE_RGB[o.kind] || [200, 200, 200];
      const s = Math.max(2, z * 0.5), ox = sx + (z - s) / 2, oy = sy + (z - s) / 2;
      ctx.fillStyle = `rgb(${c[0]},${c[1]},${c[2]})`;
      ctx.fillRect(ox, oy, s, s);
      ctx.strokeStyle = 'rgba(0,0,0,0.6)'; ctx.lineWidth = 1;
      ctx.strokeRect(ox + 0.5, oy + 0.5, s - 1, s - 1);
    }
  }
  // Placed building tiles: floors first (so walls/doors sit on top), then the shell.
  const drawBlock = (bx, by, code) => {
    const sx = (bx - cam.camX) * z, sy = (by - cam.camY) * z;
    if (!onScreen(sx, sy, 1)) return;
    const c = WBLOCK_RGB[code] || [150, 150, 150];
    ctx.fillStyle = `rgb(${c[0]},${c[1]},${c[2]})`;
    ctx.fillRect(sx, sy, z + 0.5, z + 0.5);
    if (code === 2 || code === 4) {                          // wall / window: dark mortar lines
      ctx.strokeStyle = 'rgba(0,0,0,0.45)'; ctx.lineWidth = 1;
      ctx.strokeRect(sx + 0.5, sy + 0.5, z - 1, z - 1);
    }
    if (code === 3) {                                        // door: a lighter slab + handle gap
      ctx.fillStyle = 'rgba(60,40,20,0.85)';
      ctx.fillRect(sx + z * 0.2, sy + z * 0.15, z * 0.6, z * 0.7);
    }
  };
  for (const b of (d.blocks || [])) if (b[2] === 1) drawBlock(b[0], b[1], b[2]);
  for (const b of (d.blocks || [])) if (b[2] !== 1) drawBlock(b[0], b[1], b[2]);
  // Legacy point-shelters (pre tile-building saves): a small hut under people.
  for (const st of (d.structures || [])) {
    const sx = (st.x - cam.camX) * z, sy = (st.y - cam.camY) * z;
    if (!onScreen(sx, sy, 1)) continue;
    const s = Math.max(4, z * 0.92);
    const ox = sx + (z - s) / 2, oy = sy + (z - s) / 2;
    ctx.fillStyle = '#8a5a33'; ctx.fillRect(ox, oy + s * 0.42, s, s * 0.58);
    ctx.fillStyle = '#5c3a20';
    ctx.beginPath();
    ctx.moveTo(ox - s * 0.12, oy + s * 0.46); ctx.lineTo(ox + s / 2, oy);
    ctx.lineTo(ox + s * 1.12, oy + s * 0.46); ctx.closePath(); ctx.fill();
  }
  // Wildlife as crisp pixel sprites, sized per species (deer/wolf ≈ 2×2 tiles, rabbit 1).
  const sm = WORLD.smooth;
  for (const a of d.animals) {
    const span = WANIMAL_SPAN[a.sp] || 1;
    const sp0 = sm && sm.get(a.id);
    const ax = sp0 ? sp0.x : a.x, ay = sp0 ? sp0.y : a.y;
    const sx = (ax - cam.camX) * z, sy = (ay - cam.camY) * z;
    if (!onScreen(sx, sy, span)) continue;
    const c = WANIMAL_RGB[a.sp] || [220, 220, 220];
    const s = Math.max(2.5, z * span * 0.82);
    const ox = sx + (z - s) / 2, oy = sy + (z - s) / 2;     // centred on the entity's tile
    ctx.fillStyle = `rgb(${c[0]},${c[1]},${c[2]})`;
    ctx.fillRect(ox, oy, s, s);
    ctx.strokeStyle = 'rgba(0,0,0,0.55)'; ctx.lineWidth = 1;
    ctx.strokeRect(ox + 0.5, oy + 0.5, s - 1, s - 1);
  }
  // People: a two-tone figure (body + head) ≈ 2 tiles tall, tinted by health.
  for (const p of (d.people || [])) {
    const span = WPERSON_SPAN;
    const sp0 = sm && sm.get(p.id);
    const px2 = sp0 ? sp0.x : p.x, py2 = sp0 ? sp0.y : p.y;
    const sx = (px2 - cam.camX) * z, sy = (py2 - cam.camY) * z;
    if (!onScreen(sx, sy, span)) continue;
    const hp = p.hp == null ? 1 : p.hp;
    const body = hp > 0.5 ? '#f2c14e' : '#e0683c';          // gold when well, rust when failing
    const bw = Math.max(2, z * span * 0.34), bh = Math.max(3, z * span * 0.62);
    const bx = sx + (z - bw) / 2, by = sy + (z - bh) / 2;
    ctx.fillStyle = 'rgba(0,0,0,0.45)';
    ctx.fillRect(bx - 1, by - 1, bw + 2, bh + 2);
    ctx.fillStyle = body;
    ctx.fillRect(bx, by + bh * 0.32, bw, bh * 0.68);        // torso
    const hr = Math.max(1.2, bw * 0.6);
    ctx.beginPath();
    ctx.arc(bx + bw / 2, by + hr, hr, 0, 6.283); ctx.fill(); // head
    // A ⚙ floats over anyone mid-craft (crafting now takes in-world time), with a small
    // ring filling to show progress — there are no work animations yet, so this is the tell.
    if (p.crafting && z >= 4) {
      const gx = bx + bw / 2, gy = by - 8, gr = Math.max(5, Math.min(11, z * 0.5));
      ctx.font = `${gr * 1.7}px system-ui, sans-serif`;
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText('⚙', gx, gy);
      const pct = Math.max(0, Math.min(1, p.crafting.pct || 0));
      ctx.beginPath();
      ctx.strokeStyle = 'rgba(0,0,0,0.5)'; ctx.lineWidth = 3;
      ctx.arc(gx, gy, gr + 3, -Math.PI / 2, -Math.PI / 2 + 6.283); ctx.stroke();
      ctx.beginPath();
      ctx.strokeStyle = '#7ed79b'; ctx.lineWidth = 2.4;
      ctx.arc(gx, gy, gr + 3, -Math.PI / 2, -Math.PI / 2 + 6.283 * pct); ctx.stroke();
      ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic';
    }
    // A spoken line floats above the head for a short while after they say it (only when
    // zoomed in enough to read it). say_t is in game-minutes; ~25 of those ≈ a minute live.
    if (p.say && z >= 6 && d.clock != null && (d.clock - (p.say_t || 0)) < 25) {
      const txt = p.say.length > 42 ? p.say.slice(0, 41) + '…' : p.say;
      ctx.font = `${Math.max(9, Math.min(13, z * 0.5))}px system-ui, sans-serif`;
      const tw = ctx.measureText(txt).width, padx = 5;
      const bxc = sx + z / 2, byb = by - 7;
      ctx.fillStyle = 'rgba(20,22,28,0.86)';
      ctx.beginPath();
      if (ctx.roundRect) ctx.roundRect(bxc - tw / 2 - padx, byb - 15, tw + padx * 2, 16, 4);
      else ctx.rect(bxc - tw / 2 - padx, byb - 15, tw + padx * 2, 16);
      ctx.fill();
      ctx.fillStyle = '#eef1f6'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText(txt, bxc, byb - 7);
      ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic';
    }
  }
  // Thatch roofs last, translucent, so a covered building reads as "indoors" but the
  // folk sheltering under it still show through faintly.
  for (const r of (d.roofs || [])) {
    const sx = (r[0] - cam.camX) * z, sy = (r[1] - cam.camY) * z;
    if (!onScreen(sx, sy, 1)) continue;
    ctx.fillStyle = 'rgba(126,94,52,0.5)';
    ctx.fillRect(sx, sy, z + 0.5, z + 0.5);
  }
}

function setActiveSpeed(speed) {
  const wrap = document.getElementById('whud-speed');
  if (!wrap) return;
  wrap.querySelectorAll('button[data-speed]').forEach(b =>
    b.classList.toggle('active', +b.dataset.speed === Math.round(speed)));
}

function updateWorldHud() {
  const d = WORLD.data; if (!d) return;
  const hh = Math.floor(d.time), mm = Math.round((d.time - hh) * 60);
  const set = (id, v) => { const el = document.getElementById(id); if (el) el.textContent = v; };
  set('whud-day', `Day ${d.day}`);
  set('whud-time', `${String(hh).padStart(2, '0')}:${String(mm).padStart(2, '0')}`);
  set('whud-season', d.season);
  set('whud-weather', d.weather);
  const cen = d.census;
  if (cen && cen.animals) {
    const a = cen.animals;
    let html = Object.entries(a).map(([k, v]) => `${k} <span>${v}</span>`).join('');
    const ppl = cen.people != null ? cen.people : (d.people || []).length;
    html += `people <span>${ppl}</span>`;
    document.getElementById('whud-census').innerHTML = html;
  }
}

// A light per-tick update (entities + clock move; terrain doesn't).
function worldOnTick(msg) {
  if (!WORLD.data) return;
  Object.assign(WORLD.data, {
    day: msg.day, time: msg.time, season: msg.season,
    weather: msg.weather, animals: msg.animals || [], people: msg.people || [],
    structures: msg.structures || [],
    blocks: msg.blocks || WORLD.data.blocks || [],
    roofs: msg.roofs || WORLD.data.roofs || [],
    census: msg.census, version: msg.version,
  });
  WORLD._lastTickWall = performance.now();   // for the delivery-lag indicator
  if (activeView === 'world') { updateWorldHud(); startWorldAnim(); }
  if (WORLD._personId != null) refreshPersonPanel(false);   // keep the open inspector live
}

// Terrain/flora was reshaped (a god acted) — pull a fresh snapshot, debounced.
function worldOnChanged() {
  if (activeView !== 'world') return;
  if (WORLD.refreshPending) return;
  WORLD.refreshPending = true;
  setTimeout(async () => {
    WORLD.refreshPending = false;
    try {
      const j = await (await fetch('/api/world')).json();
      if (j.enabled && j.world) setWorld(j.world);
    } catch (_) {}
    if (_ledgerOpen()) loadLedger();   // a discovery/teaching may have just been logged
  }, 250);
}

function startWorldRefresh() {
  stopWorld();
  // Periodically refresh the full snapshot so slow changes (plant growth, spread,
  // seasonal recolouring) show up even without an explicit god-action.
  WORLD.refreshTimer = setInterval(() => { if (activeView === 'world') worldOnChanged(); }, 20000);
}
function stopWorld() {
  if (WORLD.refreshTimer) { clearInterval(WORLD.refreshTimer); WORLD.refreshTimer = null; }
  if (WORLD.animRAF) { cancelAnimationFrame(WORLD.animRAF); WORLD.animRAF = null; }
}

// Translate a pointer event to a tile coordinate via the camera.
function worldTileAt(ev) {
  const cv = document.getElementById('world-canvas');
  if (!cv || !WORLD.cam) return { x: -1, y: -1 };
  const r = cv.getBoundingClientRect();
  const px = (ev.clientX - r.left) * (cv.width / r.width);
  const py = (ev.clientY - r.top) * (cv.height / r.height);
  return {
    x: Math.floor(WORLD.cam.camX + px / WORLD.cam.zoom),
    y: Math.floor(WORLD.cam.camY + py / WORLD.cam.zoom),
  };
}

function worldInspect(x, y) {
  const d = WORLD.data; if (!d) return;
  if (x < 0 || y < 0 || x >= d.w || y >= d.h) return;
  // Terrain layers are the downsampled overview — map the true tile to its overview cell.
  const ox = Math.min(d.ovw - 1, Math.floor(x / d.ovStep));
  const oy = Math.min(d.ovh - 1, Math.floor(y / d.ovStep));
  const i = oy * d.ovw + ox;
  const ro = document.getElementById('world-readout');
  const water = d.water[i];
  const biome = water ? ['', 'river', 'lake', 'ocean', 'shallow'][water] : d.biomes[d.biome[i]];
  const sp = d.vegSp[i];
  const flora = sp ? `${d.plants[sp]} (${Math.round(d.vegGrowth[i] / 255 * 100)}%)` : '—';
  const here = d.animals.filter(a => a.x === x && a.y === y).map(a => a.sp);
  const folk = (d.people || []).filter(p => p.x === x && p.y === y);
  const pct = (v) => Math.round((v || 0) * 100);
  const invStr = (inv) => {
    const e = Object.entries(inv || {}).filter(([, n]) => n);
    return e.length ? ' · carrying ' + e.map(([k, n]) => `${n} ${k}`).join(', ') : '';
  };
  const knows = (p) => {
    const r = Object.values(p.rel || {});
    if (!r.length) return '';
    const friend = r.reduce((a, b) => (b.trust > (a ? a.trust : 0) ? b : a), null);
    return friend ? ` · knows ${r.length} (closest ${friend.name}, ${Math.round(friend.trust * 100)}% trust)` : '';
  };
  // Character: the temperament trait a life has leaned into most (their emerging identity).
  const character = (p) => {
    const t = p.traits; if (!t) return '';
    const v = p.values || {};
    const tot = {}; for (const k in t) tot[k] = (t[k] || 0) + (v[k] || 0);
    const top = Object.keys(tot).reduce((a, b) => (tot[b] > tot[a] ? b : a));
    const word = { sociability: 'sociable', ambition: 'ambitious', curiosity: 'curious', caution: 'cautious' }[top];
    return word ? ` · ${word}` : '';
  };
  const folkLine = folk.map(p =>
    `<br><strong>${p.name}</strong> — ${p.action}${character(p)} · nourishment ${pct(p.satiety ?? 1)}% ` +
    `hydration ${pct(p.hydration ?? 1)}% vigour ${pct(p.stamina ?? 1)}% · health ${pct(p.hp)}%${invStr(p.inv)}` +
    (p.intent ? `<br><em>${escapeHtml(p.intent)}</em>` : '') +
    (p.say ? `<br>💬 “${escapeHtml(p.say)}”` : '') + knows(p)).join('');
  const builds = (d.structures || []).filter(st => st.x === x && st.y === y);
  let buildLine = builds.map(st => `<br>🛖 ${st.kind}${st.by ? ` (built by ${st.by})` : ''}`).join('');
  // Placed tile (wall/door/floor/…), roof, ore deposit, and any construction site here.
  const blk = (d.blocks || []).find(b => b[0] === x && b[1] === y);
  if (blk) buildLine += `<br>🧱 ${d.blockNames[blk[2]] || 'block'}` +
    ((d.roofs || []).some(r => r[0] === x && r[1] === y) ? ' (roofed)' : '');
  const ore = (d.ore || []).find(o => o.x === x && o.y === y);
  if (ore) buildLine += `<br>⛏️ ${ore.kind.replace('_', ' ')} deposit`;
  const site = (d.sites || []).find(s => !s.done && x >= s.ox && y >= s.oy && x < s.ox + 6 && y < s.oy + 6);
  if (site) buildLine += `<br>🏗️ ${site.name} under construction — ${site.built}/${site.total} tiles (by ${site.by})`;
  ro.innerHTML = `<strong>(${x}, ${y})</strong> · ${biome}<br>` +
    `elevation ${Math.round(d.elevation[i] / 255 * 100)}% · flora: ${flora}` +
    (here.length ? `<br>here: ${here.join(', ')}` : '') + buildLine + folkLine;
  ro.classList.add('show');
  clearTimeout(WORLD._roTimer);
  WORLD._roTimer = setTimeout(() => ro.classList.remove('show'), 3500);
}

// ── Person inspector — the full stats + mind of one soul ─────────────────────
async function worldOpenPerson(pid) {
  WORLD._personId = pid;
  const panel = document.getElementById('world-person');
  if (panel) { panel.style.display = 'flex'; panel.setAttribute('aria-hidden', 'false'); }
  await refreshPersonPanel(true);
}
function worldClosePerson() {
  WORLD._personId = null;
  const panel = document.getElementById('world-person');
  if (panel) { panel.style.display = 'none'; panel.setAttribute('aria-hidden', 'true'); }
}
async function refreshPersonPanel(force) {
  const pid = WORLD._personId; if (pid == null) return;
  const now = Date.now();
  if (!force && now - (WORLD._personFetch || 0) < 1400) return;
  WORLD._personFetch = now;
  try {
    const j = await (await fetch(`/api/world/person/${encodeURIComponent(pid)}`)).json();
    if (!j.person) { worldClosePerson(); return; }
    if (WORLD._personId === pid) renderPersonPanel(j.person);
  } catch (_) {}
}
function renderPersonPanel(p) {
  const body = document.getElementById('world-person-body') || document.getElementById('wperson-body');
  const title = document.getElementById('wperson-title');
  if (!body) return;
  const pct = (v) => Math.round((v || 0) * 100);
  const bar = (label, v, col) =>
    `<div class="wp-bar"><span>${label}</span><div class="wp-track"><i style="width:${Math.min(100, pct(v))}%;background:${col}"></i></div><b>${pct(v)}%</b></div>`;
  const standingLine = (r) => {
    r = r || 0;
    const tier = r < 0.05 ? 'unknown' : r < 0.2 ? 'known' : r < 0.5 ? 'respected'
               : r < 1.0 ? 'esteemed' : 'renowned';
    return `<div class="wp-standing">Standing <b>${tier}</b><em>renown ${r.toFixed(2)}</em></div>`;
  };
  const kinLine = (k) => {
    if (!k) return '';
    const bits = [];
    if (k.partner) bits.push(`partnered with ${escapeHtml(k.partner)}`);
    if (k.parents && k.parents.length) bits.push(`child of ${k.parents.map(escapeHtml).join(' & ')}`);
    if (k.children && k.children.length) bits.push(`${k.children.length} child${k.children.length > 1 ? 'ren' : ''}: ${k.children.map(escapeHtml).join(', ')}`);
    if (!bits.length) return '';
    const line = bits.join(' · ') + (k.lineage ? ` · of the ${escapeHtml(k.lineage)} line` : '');
    return `<div class="wp-kin">${line}</div>`;
  };
  const traits = p.traits || {}, values = p.values || {};
  const trait = (k) => {
    const base = traits[k] || 0, drift = values[k] || 0;
    const sign = drift > 0.001 ? `+${drift.toFixed(2)}` : drift < -0.001 ? drift.toFixed(2) : '·';
    return `<div class="wp-trait"><span>${k}</span><b>${(base + drift).toFixed(2)}</b><em>${sign}</em></div>`;
  };
  const rels = Object.values(p.rel || {}).sort((a, b) => (b.trust || 0) - (a.trust || 0));
  const relRows = rels.length ? rels.map(r =>
    `<div class="wp-rel"><span>${escapeHtml(r.name || '?')}</span>` +
    `<em>trust ${pct(r.trust)}% · ${r.sentiment >= 0 ? 'warm' : 'cold'} ${pct(Math.abs(r.sentiment))}%` +
    `${r.trades ? ` · ${r.trades} trade${r.trades > 1 ? 's' : ''}` : ''}</em></div>`).join('')
    : '<div class="wp-empty">No one yet.</div>';
  const mem = (p.memory || []).slice(-10).reverse();
  const memRows = mem.length ? mem.map(m =>
    `<div class="wp-mem"><i>${escapeHtml(m.kind || '')}</i> ${escapeHtml(m.text || '')}</div>`).join('')
    : '<div class="wp-empty">No memories yet.</div>';
  const refl = (p.reflections || []).slice(-5).reverse();
  const reflRows = refl.length ? refl.map(r =>
    `<div class="wp-refl">“${escapeHtml(typeof r === 'string' ? r : (r.text || ''))}”</div>`).join('') : '';
  const inv = Object.entries(p.inv || {}).filter(([, n]) => n)
    .map(([k, n]) => `${n}× ${k.replace(/_/g, ' ')}`).join(', ') || 'nothing';
  const craftLine = p.crafting
    ? `<div class="wp-craft">⚙ crafting <b>${(p.crafting.out || p.crafting.rid).replace(/_/g, ' ')}</b> — ${pct(p.crafting.pct)}%` +
      `${p.crafting.left_min != null ? ` (${Math.round(p.crafting.left_min)} game-min left)` : ''}</div>` : '';
  const illLine = (p.illness && p.illness.known)
    ? `<div class="wp-ill">🤒 ailing — ${escapeHtml((p.illness.d || '').replace(/_/g, ' '))}</div>` : '';
  if (title) title.textContent = p.name || 'Soul';
  const age = p.age != null ? `${p.age.toFixed(1)} days` : '—';
  body.innerHTML =
    `<div class="wp-sub">${escapeHtml(p.action || 'idle')} · ${p.stage ? escapeHtml(p.stage) + ' ' : ''}age ${age}${p.vocation ? ' · ' + escapeHtml(p.vocation) : ''}</div>` +
    kinLine(p.kin) +
    craftLine +
    illLine +
    (p.intent ? `<div class="wp-intent">“${escapeHtml(p.intent)}”</div>` : '') +
    (p.say ? `<div class="wp-say">💬 ${escapeHtml(p.say)}</div>` : '') +
    `<div class="wp-sec">Body</div>` +
    bar('health', p.hp, '#7ed79b') +
    `<div class="wp-sub2">Comfort <em>— how much they want relief (drives behaviour)</em></div>` +
    bar('hunger', p.hunger, '#e0a13c') +
    bar('thirst', p.thirst, '#56b0c4') +
    bar('fatigue', p.fatigue, '#b07cd8') +
    `<div class="wp-sub2">Reserves <em>— the body's true store (full = days of margin)</em></div>` +
    bar('nourishment', p.satiety, '#caa15a') +
    bar('hydration', p.hydration, '#4a90a4') +
    bar('vigour', p.stamina, '#8c6abf') +
    standingLine(p.renown) +
    `<div class="wp-sec">Temperament <em>(born · lived drift)</em></div>` +
    `<div class="wp-traits">${['sociability', 'ambition', 'curiosity', 'caution'].map(trait).join('')}</div>` +
    `<div class="wp-sec">Carrying</div><div class="wp-inv">${escapeHtml(inv)}</div>` +
    (() => {
      const SURV = { leaf_flask: 'leaf flask', forage_sack: 'forage sack', sleeping_mat: 'sleeping mat', campfire: 'campfire' };
      const knows = (p.recipes || []).filter(r => SURV[r]).map(r => SURV[r]);
      const open = Object.keys(SURV).filter(r => !(p.recipes || []).includes(r)).map(r => SURV[r]);
      return `<div class="wp-sec">Make-shift craft</div>` +
        `<div class="wp-inv">knows: ${knows.length ? escapeHtml(knows.join(', ')) : '—'}</div>` +
        (open.length ? `<div class="wp-inv" style="opacity:.6">still puzzling: ${escapeHtml(open.join(', '))}</div>` : '');
    })() +
    `<div class="wp-sec">Relationships <em>${rels.length}</em></div>${relRows}` +
    (reflRows ? `<div class="wp-sec">Beliefs</div>${reflRows}` : '') +
    `<div class="wp-sec">Memory stream <em>recent</em></div>${memRows}`;
}

async function worldPaint(x, y) {
  const d = WORLD.data; if (!d || x < 0 || y < 0 || x >= d.w || y >= d.h) return;
  const r = +(document.getElementById('wbrush')?.value || 4);
  let body = { x, y, r };
  switch (WORLD.tool) {
    case 'raise': body.tool = 'sculpt'; body.d = 0.12; break;
    case 'lower': body.tool = 'sculpt'; body.d = -0.12; break;
    case 'water': body.tool = 'water'; body.kind = WORLD.arg; break;
    case 'biome': body.tool = 'biome'; body.name = WORLD.arg; break;
    case 'plant': body.tool = 'plant'; body.species = WORLD.arg; break;
    case 'spawn': body.tool = 'spawn'; body.species = WORLD.arg; body.n = 1; break;
    case 'person': body.tool = 'person'; body.n = +(WORLD.arg || 1); break;
    default: return;
  }
  try {
    await fetch('/api/world/action', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
    });
    // The server broadcasts world_changed; worldOnChanged() refreshes the map.
  } catch (_) {}
}

const _wRgb = (c) => `rgb(${c[0]},${c[1]},${c[2]})`;

// Build one swatch+label tool button.
function _wToolBtn(tool, arg, label, color) {
  const b = document.createElement('button');
  b.className = 'wbtn'; b.dataset.tool = tool; b.dataset.arg = arg;
  const sw = document.createElement('span'); sw.className = 'wsw'; sw.style.background = _wRgb(color);
  const t = document.createElement('span'); t.textContent = label;
  b.append(sw, t);
  return b;
}

// Fill the four accordion grids from the palettes (once).
function populateWorldTools() {
  if (WORLD._toolsBuilt) return;
  const cap = (s) => s.charAt(0).toUpperCase() + s.slice(1);
  const biome = document.getElementById('wbiome-grid');
  const water = document.getElementById('wwater-grid');
  const flora = document.getElementById('wflora-grid');
  const wild = document.getElementById('wwild-grid');
  if (!biome || !water || !flora || !wild) return;
  for (const [name, c] of Object.entries(WBIOME_RGB)) biome.appendChild(_wToolBtn('biome', name, cap(name), c));
  for (const [kind, code] of [['river', 1], ['lake', 2], ['ocean', 3]])
    water.appendChild(_wToolBtn('water', kind, cap(kind), WWATER_RGB[code]));
  for (const [sp, c] of Object.entries(WPLANT_TINT)) flora.appendChild(_wToolBtn('plant', sp, cap(sp), c));
  for (const [sp, c] of Object.entries(WANIMAL_RGB)) wild.appendChild(_wToolBtn('spawn', sp, cap(sp), c));
  WORLD._toolsBuilt = true;
}

// ── Crafting browser: the 128-recipe registry from GET /api/world/recipes ──────
async function loadCraftingRecipes() {
  if (WORLD._catalog || WORLD._catalogLoading) { renderCraftingList(); return; }
  WORLD._catalogLoading = true;
  const list = document.getElementById('wcraft-list');
  if (list) list.innerHTML = '<div class="wcraft-empty">Loading recipes…</div>';
  try {
    const j = await (await fetch('/api/world/recipes')).json();
    WORLD._catalog = j.enabled ? j.catalog : null;
  } catch (_) { WORLD._catalog = null; }
  WORLD._catalogLoading = false;
  renderCraftingList();
}

// ── Ledger of Making — discoveries (who/when/why) + failed inventions ──────────
async function loadLedger() {
  const list = document.getElementById('wledger-list');
  if (list && !list.innerHTML) list.innerHTML = '<div class="wledger-empty">Loading…</div>';
  try {
    const j = await (await fetch('/api/world/ledger')).json();
    WORLD._ledger = j.enabled ? (j.ledger || []) : null;
  } catch (_) { WORLD._ledger = null; }
  renderLedger();
}
function _ledgerOpen() {
  const sec = document.querySelector('.wmenu[data-menu="ledger"]');
  return sec && sec.classList.contains('open');
}
function renderLedger() {
  const list = document.getElementById('wledger-list');
  if (!list) return;
  const led = WORLD._ledger;
  if (!led) { list.innerHTML = '<div class="wledger-empty">Ledger unavailable.</div>'; return; }
  const made = led.filter(e => e.kind === 'made');
  const failed = led.filter(e => e.kind === 'failed');
  const viaLabel = (v) => v && v.startsWith('taught') ? v : (v === 'reasoned out' ? 'reasoned it out' : 'worked it out');
  let html = '';
  if (!made.length && !failed.length) {
    list.innerHTML = '<div class="wledger-empty">Nothing made yet — the band is still figuring things out.</div>';
    return;
  }
  html += `<div class="wledger-grp">Discoveries <span>${made.length}</span></div>`;
  for (const e of made.slice().reverse()) {
    html += `<div class="wledger-row${e.first ? ' first' : ''}">` +
      `<div class="wledger-main"><b>${escapeHtml(e.name || e.rid || '')}</b>` +
      `${e.first ? '<em class="wledger-first">first!</em>' : ''}</div>` +
      `<div class="wledger-sub">${escapeHtml(e.who || '?')} ${escapeHtml(viaLabel(e.via))} · Day ${e.day} ${escapeHtml(e.time || '')}</div>` +
      (e.rationale ? `<div class="wledger-why">“${escapeHtml(e.rationale)}”</div>` : '') +
      `</div>`;
  }
  if (failed.length) {
    html += `<div class="wledger-grp">Failed inventions <span>${failed.length}</span></div>`;
    for (const e of failed.slice().reverse().slice(0, 40)) {
      html += `<div class="wledger-row failed">` +
        `<div class="wledger-main"><b>${escapeHtml(e.combo || '')}</b> — nothing</div>` +
        `<div class="wledger-sub">${escapeHtml(e.who || '?')} tried · Day ${e.day} ${escapeHtml(e.time || '')}</div></div>`;
    }
  }
  list.innerHTML = html;
}

function renderCraftingList(filter = '') {
  const list = document.getElementById('wcraft-list');
  const cat = WORLD._catalog;
  if (!list) return;
  if (!cat) { list.innerHTML = '<div class="wcraft-empty">Recipes unavailable.</div>'; return; }
  const items = cat.items || {};
  const nameOf = (id) => (items[id] && items[id].name) || id.replace(/_/g, ' ');
  const iconOf = (id) => (items[id] && items[id].icon) || '▪';
  // Group by station (handheld first), each sorted by tier then name.
  const order = ['', 'workbench', 'campfire', 'kiln', 'furnace', 'forge', 'loom', 'tannery', 'anvil', 'well'];
  const label = { '': 'By hand', workbench: 'Workbench', campfire: 'Campfire', kiln: 'Kiln',
    furnace: 'Furnace', forge: 'Forge', loom: 'Loom', tannery: 'Tannery', anvil: 'Anvil', well: 'Well' };
  const groups = {};
  for (const r of cat.recipes) {
    if (filter && !(nameOf(r.out).toLowerCase().includes(filter) || r.out.includes(filter))) continue;
    (groups[r.station || ''] = groups[r.station || ''] || []).push(r);
  }
  let html = '';
  for (const st of order) {
    const rs = groups[st]; if (!rs || !rs.length) continue;
    rs.sort((a, b) => a.tier - b.tier || nameOf(a.out).localeCompare(nameOf(b.out)));
    html += `<div class="wcraft-grp">${label[st] || st} <span>${rs.length}</span></div>`;
    for (const r of rs) {
      const ins = Object.entries(r.inp).map(([k, n]) => `${n}× ${nameOf(k)}`).join(', ');
      const tool = r.tool ? ` · needs ${r.tool}` : '';
      const qty = r.qty > 1 ? ` ×${r.qty}` : '';
      html += `<div class="wcraft-row"><span class="wcraft-ic">${iconOf(r.out)}</span>` +
        `<span class="wcraft-main"><b>${nameOf(r.out)}${qty}</b>` +
        `<span class="wcraft-sub">${ins}${tool}</span></span>` +
        `<span class="wcraft-tier">T${r.tier}</span></div>`;
    }
  }
  list.innerHTML = html || '<div class="wcraft-empty">No recipes match.</div>';
}

function bindWorld() {
  // The sidebar toggle lives outside the world view — bind it independently so it
  // works even before the World tab is ever opened.
  bindSidebarToggle();
  if (WORLD.bound) return;
  WORLD.bound = true;
  const cv = document.getElementById('world-canvas');
  const panel = document.getElementById('world-panel');
  const fab = document.getElementById('world-fab');

  populateWorldTools();

  // FAB opens/closes the god-tools panel.
  fab?.addEventListener('click', () => {
    const open = panel.classList.toggle('open');
    fab.setAttribute('aria-expanded', String(open));
    panel.setAttribute('aria-hidden', String(!open));
  });
  document.getElementById('wpanel-close')?.addEventListener('click', () => {
    panel.classList.remove('open');
    fab?.setAttribute('aria-expanded', 'false');
    panel.setAttribute('aria-hidden', 'true');
  });

  // Accordion menu headers.
  panel?.querySelectorAll('.wmenu-head').forEach(head => {
    head.addEventListener('click', () => {
      const sec = head.parentElement;
      const open = sec.classList.toggle('open');
      if (open && sec.dataset.menu === 'crafting') loadCraftingRecipes();   // lazy-load on first open
      if (open && sec.dataset.menu === 'ledger') loadLedger();
    });
  });
  const csearch = document.getElementById('wcraft-search');
  csearch?.addEventListener('input', () => renderCraftingList(csearch.value.trim().toLowerCase()));

  // Tool selection (delegated — covers both static and generated buttons).
  panel?.addEventListener('click', (ev) => {
    const btn = ev.target.closest('.wbtn[data-tool]');
    if (!btn) return;
    panel.querySelectorAll('.wbtn[data-tool]').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    WORLD.tool = btn.dataset.tool;
    WORLD.arg = btn.dataset.arg || null;
  });

  const brush = document.getElementById('wbrush');
  brush?.addEventListener('input', () => { document.getElementById('wbrush-val').textContent = brush.value; });

  document.getElementById('wreset')?.addEventListener('click', async () => {
    if (!confirm('Generate a brand-new world? The current one will be replaced.')) return;
    try {
      const j = await (await fetch('/api/world/reset', {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}',
      })).json();
      if (j.world) { WORLD.cam = null; setWorld(j.world); }
    } catch (_) {}
  });

  // Pointer: inspect-tool drags pan the camera; any paint tool paints.
  cv?.addEventListener('pointerdown', (ev) => {
    const { x, y } = worldTileAt(ev);
    if (WORLD.tool === 'inspect') {
      WORLD.panning = true; WORLD.panLX = ev.clientX; WORLD.panLY = ev.clientY;
      cv.setPointerCapture(ev.pointerId);
      worldInspect(x, y);
      return;
    }
    WORLD.dragging = true; cv.setPointerCapture(ev.pointerId);
    worldPaint(x, y);
  });
  cv?.addEventListener('pointermove', (ev) => {
    const { x, y } = worldTileAt(ev);
    const co = document.getElementById('whud-coords');
    if (co) co.textContent = `${x}, ${y}`;
    if (WORLD.panning && WORLD.cam) {
      const r = cv.getBoundingClientRect();
      const dx = (ev.clientX - WORLD.panLX) * (cv.width / r.width);
      const dy = (ev.clientY - WORLD.panLY) * (cv.height / r.height);
      WORLD.panLX = ev.clientX; WORLD.panLY = ev.clientY;
      WORLD.cam.camX -= dx / WORLD.cam.zoom;
      WORLD.cam.camY -= dy / WORLD.cam.zoom;
      clampCamera(); renderWorld(); refreshWorldDetail();
      return;
    }
    if (WORLD.dragging && WORLD.tool !== 'spawn' && WORLD.tool !== 'person') {
      const now = Date.now();
      if (now - WORLD.lastPaint > 90) { WORLD.lastPaint = now; worldPaint(x, y); }
    }
  });
  const stop = () => { WORLD.dragging = false; WORLD.panning = false; };
  cv?.addEventListener('pointerup', stop);
  cv?.addEventListener('pointercancel', stop);

  // Double-click a soul to lay bare its stats and mind.
  cv?.addEventListener('dblclick', (ev) => {
    const { x, y } = worldTileAt(ev);
    const d = WORLD.data; if (!d) return;
    let best = null, bd = 9;
    for (const p of (d.people || [])) {
      const s = WORLD.smooth && WORLD.smooth.get(p.id);
      const pxp = s ? s.x : p.x, pyp = s ? s.y : p.y;
      const dd = Math.abs(pxp - x) + Math.abs(pyp - y);
      if (dd < bd) { bd = dd; best = p; }
    }
    if (best && bd <= 3) worldOpenPerson(best.id);
  });
  document.getElementById('wperson-close')?.addEventListener('click', worldClosePerson);

  // World speed (fast-forward) selector.
  document.getElementById('whud-speed')?.addEventListener('click', async (ev) => {
    const btn = ev.target.closest('button[data-speed]');
    if (!btn) return;
    const speed = +btn.dataset.speed;
    setActiveSpeed(speed);                       // optimistic
    try {
      await fetch('/api/world/speed', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ speed }),
      });
    } catch (_) {}
  });
  cv?.addEventListener('pointerleave', () => {
    const co = document.getElementById('whud-coords'); if (co) co.textContent = '';
  });

  // Wheel zooms toward the cursor.
  cv?.addEventListener('wheel', (ev) => {
    if (!WORLD.cam) return;
    ev.preventDefault();
    const r = cv.getBoundingClientRect();
    const px = (ev.clientX - r.left) * (cv.width / r.width);
    const py = (ev.clientY - r.top) * (cv.height / r.height);
    const cam = WORLD.cam;
    const tileX = cam.camX + px / cam.zoom, tileY = cam.camY + py / cam.zoom;
    cam.zoom *= ev.deltaY < 0 ? 1.15 : 1 / 1.15;
    const minZoom = Math.max(cv.width / WORLD.data.w, cv.height / WORLD.data.h);
    cam.zoom = Math.max(minZoom, Math.min(48, cam.zoom));
    cam.camX = tileX - px / cam.zoom;
    cam.camY = tileY - py / cam.zoom;
    clampCamera(); renderWorld(); refreshWorldDetail();
  }, { passive: false });

  window.addEventListener('resize', () => {
    if (activeView === 'world' && WORLD.data) { computeWorldCamera(); renderWorld(); }
  });
}

let _sidebarToggleBound = false;
function bindSidebarToggle() {
  if (_sidebarToggleBound) return;
  const btn = document.getElementById('sidebar-toggle');
  if (!btn) return;
  _sidebarToggleBound = true;
  btn.addEventListener('click', () => {
    document.body.classList.toggle('sidebar-collapsed');
    // The sidebar width animates (~.22s); recompute the map after it settles.
    if (activeView === 'world' && WORLD.data) {
      requestAnimationFrame(() => { computeWorldCamera(); renderWorld(); });
      setTimeout(() => { computeWorldCamera(); renderWorld(); }, 260);
    }
  });
}

// Wire the sidebar collapse toggle at startup (independent of the World tab).
bindSidebarToggle();
