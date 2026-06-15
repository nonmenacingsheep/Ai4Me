const { app, BrowserWindow, ipcMain, Tray, Menu, nativeImage, shell } = require('electron');
const { spawn } = require('child_process');
const path = require('path');
const http = require('http');
const net = require('net');
const fs = require('fs');

let mainWindow;
let backendProcess;
let tray;

const BACKEND_PORT = 7823;
const BACKEND_HOST = '127.0.0.1';

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
  mainWindow = new BrowserWindow({
    width: 900,
    height: 660,
    minWidth: 680,
    minHeight: 480,
    frame: false,
    transparent: false,
    backgroundColor: '#07070e',
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
    mainWindow.show();
  });

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
ipcMain.on('window-maximize', () => {
  if (mainWindow?.isMaximized()) mainWindow.unmaximize();
  else mainWindow?.maximize();
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
