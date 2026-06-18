const { app, BrowserWindow, ipcMain, Tray, Menu, nativeImage, shell, screen, dialog } = require('electron');
const { spawn } = require('child_process');
const path = require('path');
const http = require('http');
const net = require('net');
const fs = require('fs');

let mainWindow;
let backendProcess;
let tray;

// Passthrough (ambient phone-shaped) mode: remember the normal bounds so the
// temporary phone size never leaks into the saved window state.
let passthroughActive = false;
let prePassthroughBounds = null;
let prePassthroughMaximized = false;

const BACKEND_PORT = 7823;
const BACKEND_HOST = '127.0.0.1';

/* ─── Window size/position memory ──────────────────────────────────── */
const WINDOW_STATE_FILE = path.join(app.getPath('userData'), 'window-state.json');

function loadWindowState() {
  try { return JSON.parse(fs.readFileSync(WINDOW_STATE_FILE, 'utf8')); }
  catch (_) { return null; }
}

function saveWindowState() {
  if (!mainWindow) return;
  try {
    // In passthrough the window is phone-shaped — persist the normal bounds we
    // stashed on entry instead, so a restart reopens at the real size.
    const b = (passthroughActive && prePassthroughBounds)
      ? prePassthroughBounds
      : mainWindow.getNormalBounds();
    const maximized = passthroughActive ? prePassthroughMaximized : mainWindow.isMaximized();
    fs.writeFileSync(WINDOW_STATE_FILE, JSON.stringify({ ...b, maximized }));
  } catch (_) {}
}

// Guard against restoring onto a monitor that's no longer attached.
function isVisibleOnSomeDisplay(b) {
  if (!Number.isFinite(b.x) || !Number.isFinite(b.y)) return false;
  return screen.getAllDisplays().some(d => {
    const a = d.workArea;
    return b.x < a.x + a.width && b.x + b.width > a.x &&
           b.y < a.y + a.height && b.y + b.height > a.y;
  });
}

// Voice changer (MMVCServerSIO) — optional. Point VOICE_CHANGER_BAT in your .env
// at its start_http.bat to have Ai4Me launch it for you; otherwise this is skipped.
const VC_PORT = 18888;
const VC_BAT = process.env.VOICE_CHANGER_BAT || '';

// Local dev app — never cache the UI, so edits to index.html/css/js always show.
app.commandLine.appendSwitch('disable-http-cache');

// Give Windows a distinct app identity so the taskbar uses OUR icon instead of
// grouping under electron.exe's default icon.
if (process.platform === 'win32') {
  app.setAppUserModelId('com.ai4me.aitha');
}

/* ─── Find Python ──────────────────────────────────────────────────── */
function findPython() {
  const candidates = ['python', 'python3'];
  // In packaged builds, look for bundled interpreter first
  const bundled = path.join(process.resourcesPath || '', 'backend', 'python.exe');
  if (fs.existsSync(bundled)) return bundled;
  return candidates[0]; // fallback to PATH
}

/* ─── Voice changer (MMVCServerSIO) ────────────────────────────────── */
function isPortOpen(port) {
  return new Promise((resolve) => {
    const sock = net.connect({ host: '127.0.0.1', port }, () => {
      sock.destroy();
      resolve(true);
    });
    sock.on('error', () => resolve(false));
    sock.setTimeout(700, () => { sock.destroy(); resolve(false); });
  });
}

async function startVoiceChanger() {
  try {
    if (!VC_BAT) {
      console.log('[vc] no VOICE_CHANGER_BAT set — skipping (TTS still works without it)');
      return;
    }
    if (await isPortOpen(VC_PORT)) {
      console.log(`[vc] already running on ${VC_PORT} — not starting another`);
      return;
    }
    if (!fs.existsSync(VC_BAT)) {
      console.log(`[vc] batch not found, skipping: ${VC_BAT}`);
      return;
    }
    const dir = path.dirname(VC_BAT);
    // Launch in its own console window, detached so it lives independently of Ai4Me.
    const cmd = `start "Voice Changer" /D "${dir}" "${path.basename(VC_BAT)}"`;
    const child = spawn(cmd, { shell: true, detached: true, stdio: 'ignore' });
    child.unref();
    console.log('[vc] launched MMVCServerSIO');
  } catch (e) {
    console.log(`[vc] launch failed: ${e.message}`);
  }
}

/* ─── Start FastAPI backend ────────────────────────────────────────── */
function startBackend() {
  return new Promise((resolve, reject) => {
    const python = findPython();
    const scriptDir = app.isPackaged
      ? path.join(process.resourcesPath, 'backend')
      : path.join(__dirname, 'backend');
    const script = path.join(scriptDir, 'server.py');

    backendProcess = spawn(python, [script], {
      cwd: scriptDir,
      env: { ...process.env },
      windowsHide: true,
    });

    backendProcess.stdout.on('data', d => console.log('[backend]', d.toString().trim()));
    backendProcess.stderr.on('data', d => console.log('[backend]', d.toString().trim()));
    backendProcess.on('error', err => console.error('[backend error]', err));

    // Poll until backend is ready
    let attempts = 0;
    const poll = setInterval(() => {
      attempts++;
      const req = http.get(`http://${BACKEND_HOST}:${BACKEND_PORT}/health`, (res) => {
        if (res.statusCode === 200) {
          clearInterval(poll);
          resolve();
        }
      });
      req.on('error', () => {});
      req.setTimeout(400, () => req.destroy());

      if (attempts > 60) {
        clearInterval(poll);
        reject(new Error('Backend did not start within 30 seconds'));
      }
    }, 500);
  });
}

/* ─── Create window ────────────────────────────────────────────────── */
function createWindow() {
  const saved = loadWindowState();
  const bounds = { width: 900, height: 660 };
  if (saved && Number.isFinite(saved.width) && Number.isFinite(saved.height)) {
    bounds.width = Math.max(680, saved.width);
    bounds.height = Math.max(480, saved.height);
    if (isVisibleOnSomeDisplay(saved)) { bounds.x = saved.x; bounds.y = saved.y; }
  }

  mainWindow = new BrowserWindow({
    ...bounds,
    minWidth: 680,
    minHeight: 480,
    frame: false,
    transparent: true,            // lets passthrough mode float over the desktop
    backgroundColor: '#00000000', // normal mode stays opaque via the page's body bg
    titleBarStyle: 'hidden',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
    show: false,
    icon: path.join(__dirname, 'assets', 'icon.ico'),
  });

  // Let the renderer use the microphone (hands-free voice input). localhost is a
  // secure context, so getUserMedia just needs the Electron permission grant.
  const ses = mainWindow.webContents.session;
  const allowMic = (perm) => perm === 'media' || perm === 'microphone' || perm === 'audioCapture';
  ses.setPermissionRequestHandler((_wc, permission, cb) => cb(allowMic(permission)));
  ses.setPermissionCheckHandler((_wc, permission) => allowMic(permission));

  // Purge any stale cached UI from a previous run, then load fresh.
  mainWindow.webContents.session.clearCache().finally(() => {
    mainWindow.loadURL(`http://${BACKEND_HOST}:${BACKEND_PORT}/`);
  });

  mainWindow.once('ready-to-show', () => {
    if (saved && saved.maximized) mainWindow.maximize();
    mainWindow.show();
  });

  // Remember size/position. Debounced so we're not writing on every pixel of a
  // drag, plus a final save as it closes.
  let saveTimer = null;
  const scheduleSave = () => {
    if (passthroughActive) return;   // don't persist the temporary phone size
    clearTimeout(saveTimer);
    saveTimer = setTimeout(saveWindowState, 400);
  };
  mainWindow.on('resize', scheduleSave);
  mainWindow.on('move', scheduleSave);
  mainWindow.on('close', saveWindowState);

  mainWindow.on('closed', () => {
    mainWindow = null;
  });

  // Open external links in default browser
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url);
    return { action: 'deny' };
  });
}

/* ─── Tray ─────────────────────────────────────────────────────────── */
function createTray() {
  const iconPath = path.join(__dirname, 'assets', 'icon.ico');
  const icon = fs.existsSync(iconPath)
    ? nativeImage.createFromPath(iconPath)
    : nativeImage.createEmpty();

  tray = new Tray(icon);
  tray.setToolTip('Ai4Me — Aitha is watching');

  const menu = Menu.buildFromTemplate([
    { label: 'Show', click: () => mainWindow?.show() },
    { type: 'separator' },
    { label: 'Quit', click: () => app.quit() },
  ]);

  tray.setContextMenu(menu);
  tray.on('double-click', () => mainWindow?.show());
}

/* ─── IPC: window controls ─────────────────────────────────────────── */
ipcMain.on('window-close',    () => { app.quit(); });  // close button quits the app
ipcMain.on('window-minimize', () => { mainWindow?.minimize(); });

// Native folder picker for granting Aitha read-only access to a folder.
ipcMain.handle('pick-folder', async () => {
  if (!mainWindow) return null;
  const res = await dialog.showOpenDialog(mainWindow, {
    title: 'Choose a folder Aitha can read',
    properties: ['openDirectory'],
  });
  return res.canceled || !res.filePaths.length ? null : res.filePaths[0];
});
ipcMain.on('window-maximize', () => {
  if (mainWindow?.isMaximized()) mainWindow.unmaximize();
  else mainWindow?.maximize();
});

/* ─── IPC: passthrough (ambient phone-shaped) mode ─────────────────── */
ipcMain.on('passthrough-enter', () => {
  if (!mainWindow || passthroughActive) return;
  passthroughActive = true;
  prePassthroughMaximized = mainWindow.isMaximized();
  if (prePassthroughMaximized) mainWindow.unmaximize();
  prePassthroughBounds = mainWindow.getNormalBounds();

  const disp = screen.getDisplayMatching(prePassthroughBounds).workArea;
  const h = Math.min(840, disp.height - 40);
  const w = Math.round(h * 0.50);   // ~phone aspect ratio
  const x = Math.round(prePassthroughBounds.x + (prePassthroughBounds.width - w) / 2);
  const y = Math.max(disp.y + 10, Math.round(prePassthroughBounds.y + (prePassthroughBounds.height - h) / 2));

  mainWindow.setMinimumSize(280, 460);
  mainWindow.setBounds({ x, y, width: w, height: h }, true);
  mainWindow.setAlwaysOnTop(true, 'floating');   // she floats over other windows
});

ipcMain.on('passthrough-exit', () => {
  if (!mainWindow || !passthroughActive) return;
  passthroughActive = false;
  mainWindow.setAlwaysOnTop(false);
  mainWindow.setMinimumSize(680, 480);
  if (prePassthroughBounds) mainWindow.setBounds(prePassthroughBounds, true);
  if (prePassthroughMaximized) mainWindow.maximize();
});

/* ─── Single instance ──────────────────────────────────────────────── */
// Prevent a second launch (shortcut + run.bat) from fighting over port 7823.
if (!app.requestSingleInstanceLock()) {
  app.quit();
} else {
  app.on('second-instance', () => {
    if (mainWindow) {
      if (mainWindow.isMinimized()) mainWindow.restore();
      mainWindow.show();
      mainWindow.focus();
    }
  });
}

/* ─── App lifecycle ────────────────────────────────────────────────── */
app.whenReady().then(async () => {
  if (!app.hasSingleInstanceLock()) return;  // losing instance — do nothing

  // Kick off the voice changer first (slow GPU model) — don't wait on it.
  startVoiceChanger();

  // Ollama is no longer auto-started — we run on the cloud model (DeepSeek).
  // (To use a local model again, call startOllama() here and wait for its port.)

  try {
    await startBackend();
  } catch (err) {
    console.error('Backend startup failed:', err.message);
    // Still open the window — it will show disconnected state
  }

  createWindow();
  createTray();

  app.on('activate', () => {
    if (!mainWindow) createWindow();
    else mainWindow.show();
  });
});

app.on('window-all-closed', () => {
  // Don't quit on close — live in tray
  // On macOS this is expected; on Windows we're doing it intentionally
});

app.on('before-quit', () => {
  if (backendProcess) {
    backendProcess.kill('SIGTERM');
    setTimeout(() => backendProcess?.kill('SIGKILL'), 2000);
  }
});

app.on('will-quit', () => {
  if (backendProcess) {
    try { backendProcess.kill(); } catch (_) {}
  }
});
