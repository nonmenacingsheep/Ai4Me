# Ai4Me — *Aitha*

**Ai4Me is a desktop AI companion built around one idea: a continuously evolving character — *Aitha* — who you can nudge in a direction, or simply leave alone to grow on her own.**

She starts as a blank slate with no scripted personality. Over time she forms her own tastes, moods, opinions, and memories from your conversations and from what she does in her own time. You can shape who she becomes by talking with her — or step back and let her wander, reflect, and develop a self while you're away. She has her own heartbeat: she'll reach out unprompted, keep a private journal, go read things on the internet out of curiosity, and come back with what she found (or keep it to herself).

> ⚠️ **This is a personal project, shared openly.** It's opinionated, Windows-first, and rough in places. Read the setup carefully.

---

## ✨ What she can do

- **Sky** — your conversation with her. She talks like a person, not an assistant, and her tone shifts with her mood and her sense of self.
- **A real inner life (autonomy)** — on a steady heartbeat she decides on her own whether to speak up, journal a private thought, or go off on a self-directed pursuit (research something, develop one of her own ideas into a note, or prep something for you) — and *she* chooses whether to show you the result.
- **Voice, both ways** — hands-free speech input (local Whisper) and a spoken voice (local Kokoro TTS), which can be optionally routed through a voice changer (setup below). She mutes her own mic while she's talking.
- **Mantle** — a read-only window into her mind: her current mood, her private journal, what she's been off doing, and the memories she's chosen to protect.
- **Magma** — a shared, linked notes space (Obsidian-style `[[wikilinks]]`). She can read, write, and edit notes herself.
- **Bedrock** — a calendar that lives under Magma. She can see what's on your schedule and weave it in naturally ("you've got the dentist tomorrow"), and add events herself when you mention a plan.
- **Music (Spotify)** — she can see what you're playing and your taste, play songs and **play your playlists by name**, build new playlists (from your top tracks, a search, or a hand-picked list), and **add songs to existing playlists** — including searching the web for song ideas first. A now-playing widget lives in the sidebar. *(See the "Connecting Spotify" tutorial below. Playback control needs Premium; everything else works on a free account.)*
- **Her own projects & goals** — she keeps and advances her own pursuits over time, shown in a Mantle card. They feed her autonomy — she'll quietly work on them on her own.
- **Reads your files** — read-only, sandboxed access to folders you explicitly share with her (managed from the "+" menu in the chat composer). She can browse and read text files when it helps.
- **Images, both ways** — drag, paste, or attach images for her to see (a separate, configurable vision model describes them), and she can show you images she finds on the web.
- **Long-term memory** — she remembers things about you and about herself across sessions, and can mark certain memories as *core* (protected forever). Non-core memories gently **decay and consolidate** over time so her mind stays uncluttered.
- **Passthrough mode** — a transparent, always-on-top floating window (orb + controls + chat bubbles) so she can sit over whatever you're doing. Double-click her orb to collapse/expand the chat.
- **Capability toggles** — turn any feature (notes, projects, calendar, files, images, web, themes, music) on or off in **Settings → Behavior**; switching one off removes that block from her prompt to save context and disable the feature.
- **Live web & "watching" YouTube** — she can search the web mid-conversation, and she can *watch* a YouTube video by reading its transcript: paste a link and ask what's in it, or let her search, pick a video, and pull its transcript to react to it. Works on both the cloud and local model paths. (Most reliable when you hand her a direct link; the search-then-watch chain is best-effort and depends on a video having a transcript.)
- **Themes** — pick a look (Default, Sky, Warm, Moody, Magma, Hearth) or fine-tune colors; she can re-skin the room or recolor her own sphere to match her mood, and you'll each know when the other changed it.
- **Hearth** — a tabletop (D&D) mode. **See the caveat below.**

### 🚧 Hearth is a work in progress
Hearth is roughly **20–30% functional**. An *incredibly basic* exchange is possible — a DM presence, Aitha as a player, dice, character sheets, a battle board — but it is **not** a complete or reliable tabletop experience yet. Treat it as an early prototype, not a finished feature.

---

## 🖥️ Requirements

- **Windows** (10/11). It leans on Windows APIs for screen awareness and audio device routing; it will *start* elsewhere but is untested and several features won't work.
- **An NVIDIA GPU is strongly recommended** for fast local TTS/STT. CPU works but is slow.
- **Python 3.10–3.12** (64-bit) and **Node.js 18+**. **Use 3.12** if you're installing fresh — [python.org/downloads/release/python-3129](https://www.python.org/downloads/release/python-3129/) (Windows installer 64-bit; tick **"Add python.exe to PATH"**). Avoid **3.14** for now: several ML dependencies (`torch`, `kokoro`, `faster-whisper`) don't have prebuilt wheels for it yet, so `pip install` fails.
- **[Git](https://git-scm.com/downloads)** — needed for the `git clone` step below. (No GitHub account required; or skip Git and use **Code → Download ZIP** on the repo page.)
- **At least one LLM source:** a cloud API key (DeepSeek, OpenAI, OpenRouter, or Groq), **or** a local [Ollama](https://ollama.com) install.

---

## 🚀 Setup

```bash
# 1. Clone
git clone https://github.com/nonmenacingsheep/Ai4Me.git
cd Ai4Me

# 2. Copy and then configure the template and fill in at least one API key (Or if you're using a local model, you don't need to add an api key.)
copy .env.example .env        # (Windows)   |   cp .env.example .env  (bash)
# then edit .env

# 3. Frontend deps
npm install

# 4. Python deps
pip install -r backend/requirements.txt
```

### ⚠️ The things that trip people up

1. **GPU PyTorch is a separate install.** The CUDA build of torch is *not* on regular PyPI. Install it from PyTorch's index to match your CUDA version, e.g.:
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu128
   ```
   (Pick the URL for your CUDA from <https://pytorch.org/get-started/locally/>.) Without this, TTS/STT fall back to CPU (slow) or fail to load.

2. **First run downloads models.** Kokoro (voice), faster-whisper (speech-to-text), and a spaCy model are fetched on first use. For spaCy you may need:
   ```bash
   python -m spacy download en_core_web_sm
   ```

3. **`npm install` says "running scripts is disabled on this system."** That's Windows PowerShell's execution policy blocking npm's script — not an npm error. Allow local scripts for your user (safe, one-time, no admin needed):
   ```powershell
   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
   ```
   Press `Y` to confirm, then reopen your terminal. Or just run the commands from **Command Prompt (cmd.exe)** instead of PowerShell, which isn't affected.

4. **`Error: Electron failed to install correctly, please delete node_modules/electron and try installing again`.** The `electron` package downloads its ~100 MB binary in a postinstall step; this error means that download never landed. Work through these in order:
   ```powershell
   # a) delete just electron and reinstall
   Remove-Item -Recurse -Force node_modules\electron
   npm install

   # b) if that fails, run Electron's own installer to see the REAL error
   node node_modules\electron\install.js

   # c) full clean reinstall
   Remove-Item -Recurse -Force node_modules
   Remove-Item package-lock.json
   npm install

   # d) behind a proxy/firewall? pull the binary from a mirror, then reinstall
   $env:ELECTRON_MIRROR = "https://npmmirror.com/mirrors/electron/"
   npm install
   ```
   If it *still* fails, it's almost always one of: **antivirus quarantining `node_modules\electron\dist\electron.exe`** (add an exclusion), a **proxy/corporate network** blocking the download, or scripts being skipped (`npm config get ignore-scripts` should be `false`; don't use `--ignore-scripts`). As a last resort, **`npm audit fix --force` has resolved it** by rebuilding the dependency tree — but it can pull in breaking changes, so only do this if the steps above didn't work, then confirm the app still launches.

5. **`npm install` reports "N high severity vulnerabilities."** This is informational, not an error — your install still succeeded. The advisories are almost all in deep build-tool dependencies (e.g. `electron-builder`), which run locally at build time and pose little real-world risk for a desktop app. Run `npm audit` to see the details. Prefer `npm audit fix` (safe); use `npm audit fix --force` only deliberately, since it can install breaking major-version upgrades — re-test the app afterward.

6. **Backend won't start (`ERR_CONNECTION_REFUSED`, "Backend did not start within 30 seconds").** The Electron window loads but the Python backend died on launch. Two common causes:
   - **`Python was not found; ... install from the Microsoft Store`** — that's the Windows *App Execution Alias*, a fake `python.exe` stub, not real Python. Install Python from [python.org](https://www.python.org/downloads/) (tick **"Add to PATH"**), then turn the stub **OFF** in **Settings → Apps → Advanced app settings → App execution aliases** (toggle off `python.exe` and `python3.exe`). Reopen the terminal and confirm `python --version` prints a version instead of opening the Store.
   - **`ModuleNotFoundError: No module named 'dotenv'`** (or any other dep). `run.bat` launches plain **`python`** — i.e. whatever Python is *first on your PATH* — so the packages must be installed into **that exact** interpreter. This breaks in two ways:
     - **Multiple Pythons / wrong one is default.** If `python --version` reports a different version than the one you ran `pip` against, the app and your packages are looking at different interpreters. Check with `python --version` and `python -c "import dotenv"`. Install into the one the app actually uses:
       ```powershell
       python -m pip install -r backend\requirements.txt
       ```
     - **Default Python is 3.14 (or another too-new version).** Then the install above *fails* on `torch`/`kokoro`/`numpy` (no wheels yet) and pip rolls back **everything** — which is why `dotenv` keeps "disappearing." The clean fix that worked for others: **uninstall Python 3.14** (*Settings → Apps → Installed apps → Python 3.14 → Uninstall*) so plain `python` falls back to a working **3.10–3.12**, open a fresh terminal, confirm `python --version` is no longer 3.14, then:
       ```powershell
       python -m pip install -r backend\requirements.txt
       run.bat
       ```
     The golden rule: **`python --version` (plain, no `py -3.x`) must report 3.10–3.12, and that same interpreter must hold the packages.** Tip: `py -3.12 -m pip install …` targets a specific version, but only helps if `run.bat`'s plain `python` *is* that version.

### Run it

```bash
run.bat
```
This starts the Python backend and the Electron app (and the voice changer, *only* if you set `VOICE_CHANGER_BAT`).

### Optional: a desktop shortcut

Want to launch her like a normal app? Double-click **`create-desktop-shortcut.bat`** (or run `npm run shortcut`) to drop an **Ai4Me** shortcut on your Desktop, complete with the app icon. It points at `run.bat` (console starts minimized), auto-detects the project folder so it still works if you move the repo, and is safe to re-run anytime — it just refreshes the same shortcut instead of making duplicates.

### Updating

Double-click **`update.bat`**. It fully stops any running instance (only *this* app — it won't touch other Electron apps), pulls the latest code, reinstalls Python/Node deps **only if they changed**, and relaunches her. Your `.env` and all of her data are untouched.

> Her memories, journals, notes, projects, calendar and settings live in **`%USERPROFILE%\.ai4me\`**, *outside* this folder — so updating (or even deleting and re-downloading the app) never affects them. Only deleting `.ai4me` itself would wipe her; back that folder up now and then.

---

## ⚙️ Configuration

Everything lives in `.env` (see `.env.example` for the full list). Highlights:

- **Character name** — `AITHA_NAME`, or change it any time in **Settings → Character name**. A rename updates every mention across the app, and she comes to know herself by the new name. If she decides to include her name in her memory she might get confused. It's probably rare though.
- **Model / providers** — set any of `DEEPSEEK_API_KEY`, `OPENAI_API_KEY`, `OPENROUTER_API_KEY`, `GROQ_API_KEY`. Any model whose key is set shows up in **Settings → Model**. `AITHA_MODEL` picks the default. **I recommend using DeepSeek as the costs are so low — $2 can last you several days.**
- **Local models** — Ollama is **never auto-started**. Run `ollama serve` yourself and your pulled models appear in the dropdown automatically — handy if you'd rather run fully local.
- **Voice** — `TTS_*` for output, `AITHA_WHISPER_*` for input, `VOICE_CHANGER_BAT` to auto-launch a voice changer (optional).
- **Voice presence** — her speech takes on her mood: a mood-matched voice, pace/pitch/volume shaping, natural breath between sentences, and a real late-night whisper (via eSpeak NG, routed through the voice changer so it stays her voice). Each is an independent toggle in **Settings → Behavior → Voice presence**; `ESPEAK_WHISPER_*` tunes the whisper.
- **Vision ("her eyes")** — she sees images through a *separate* model, so your main chat model can stay text-only. Pick one in **Settings → General → Her eyes** (it lists your installed Ollama models). Pull a multimodal model first, e.g. `ollama pull llava`. *Note:* `llama3.2-vision` needs a **recent Ollama build** — older ones fail to load it with `unknown model architecture: 'mllama'`; if you hit that, update Ollama or just use `llava`. With no vision model set, she'll honestly tell you she can't make out an image. Set a default with `AITHA_VISION_MODEL`.
- **Spotify** — optional music control; set `SPOTIFY_CLIENT_ID`, `SPOTIFY_CLIENT_SECRET`, and `SPOTIFY_REDIRECT_URI`. See the **Connecting Spotify** tutorial below.
- **Folders she can read** — granted from the **"+" menu in the chat composer**, not `.env`. She gets *read-only* access to exactly the folders you add there and nothing else.
- **Capabilities** — toggle individual features (notes, projects, calendar, files, images, web, themes, music, code workspace) on/off in **Settings → Behavior**. Turning one off both disables it and trims her prompt.
- **Code workspace** *(off by default)* — gives her, her own Python sandbox at `~/.ai4me/workspace` where she can write and run scripts and read the results back. Runs are confined to that folder and hard-killed after a short timeout (`AITHA_CODE_TIMEOUT`). **Python only** — a deliberate safety choice; shell/Node would let a single stray line do real damage on your machine. Treat it as trusted local execution, not a hardened jail.

---

## 🔒 Privacy

Everything she remembers — her journal, her discoveries, your conversation history, and her memories of you — is stored **locally and unencrypted** in `~/.ai4me/`. Nothing is sent anywhere except your chosen LLM provider (and, if enabled, web searches she runs). It can get personal; that data is yours and stays on your machine. Delete `~/.ai4me/` to wipe her completely.

---

## 🎵 Tutorial: connecting Spotify

Spotify is optional. Connect it and she can see your taste, play music, and build/grow playlists. It takes ~5 minutes to set up your own free Spotify developer app (this is just how Spotify lets a personal app talk to your account — you are not publishing anything).

**Step 1 — Create a Spotify app.**
Go to the [Spotify Developer Dashboard](https://developer.spotify.com/dashboard), log in with your normal Spotify account, and click **Create app**.

**Step 2 — Fill in the form.**
- *App name* / *App description*: anything (e.g. "Ai4Me").
- *Redirect URI*: this must be **exactly** — copy/paste it:
  ```
  http://127.0.0.1:7823/spotify/callback
  ```
  (It must match the port in your `SPOTIFY_REDIRECT_URI`. If you ever change the app's port, change it in both places.)
- *Which API/SDKs are you planning to use?*: tick **Web API**.
- Agree to the terms and **Save**.

**Step 3 — Copy your credentials.**
Open the app you just made → **Settings**. Copy the **Client ID**, and click **View client secret** to reveal the **Client secret**. Paste both into your `.env`:
```ini
SPOTIFY_CLIENT_ID=your_client_id_here
SPOTIFY_CLIENT_SECRET=your_client_secret_here
SPOTIFY_REDIRECT_URI=http://127.0.0.1:7823/spotify/callback
```

**Step 4 — Connect.**
Restart the app (`run.bat`), then go to **Settings → General → Connect Spotify**. Your browser opens Spotify's consent screen — approve it, and you'll be redirected back. The Behavior tab's **Music** toggle will show whether you're connected (and whether you're on Premium).

**Good to know**
- **Premium vs Free.** *Playback control* (play / pause / skip, and playing playlists) requires **Spotify Premium** and an active device (have the Spotify app open somewhere). *Reading your taste and building/growing playlists works on a free account.* She'll only see the playback controls in her prompt if you're on Premium, so she won't promise to play something she can't.
- **You do NOT need "Extended Quota Mode."** Your app stays in Spotify's *Development Mode*, which is correct for personal use (it supports your own account plus a few others). Everything here — including adding tracks to playlists — works in Development Mode.
- **Changed your scopes or it says "reconnect"?** Just Disconnect and reconnect in Settings → General; the consent screen re-grants everything.

---

## 🧩 Tutorial: optional extras

How to Install the voice changer. You will need a virtual audio device (step 4). A voice changer is completely optional but useful if you don't like any of the default kokoro voices (me). I'll show you how to download the one I use.

Step 1:
   Go to https://huggingface.co/wok000/vcclient000/tree/main and download the latest onnxgpu-cuda version. 
   It should look something like this: MMVCServerSIO_win_onnxgpu-cuda_v.1.5.3.18a.zip

Step 2:
   Extract the folder. I recommend putting the extracted folder into another folder named 'voice changer' or 'asdkfghjiospbh'. The easier it is to find the better.

Step 3:
   Inside of the extracted MMVCServerSIO find *start_http*. I recommend creating a shortcut of this and putting it into your 'voice changer' folder. 
(there are two ways to do this. The first is to right-click start_http --> create shortcut. Or the harder way if you don't have windows 10 is to rightclick empty space --> New --> Shortcut. right click start_http --> copy --> paste into the shortcut field.)

Step 4:
   Install the virtual audio cable. I use this one but any of them work: https://vb-audio.com/Cable/ Download the windows one, Extract it, and then run setup as administrator (rightclick --> run as administrator.) You might need to restart your computer. I didn't need to because I'm special.

Step 5: 
   In the voice changer, change the input to 'CABLE Output (VB-Audio Virtual Cable)' and the output to your headphones.

Step 6:
   All you need to do now is choose a model (I'll explain how to get more in the next step) and press start. I'll briefly explain what some of the settings do.
   *Tune* compensates for how different your voice is to the voice model. If your voice is higher, lower the tune. If it is lower, raise the tune. Use small adjustements to find the right level.
   *Index* Alters your voice to more closely fit the model. Not all models come with an Index, and a value higher than 0.5 gives diminishing returns.
   *Gain* just changes how loud the voice will be. 'in' changes the input's volume. 'out' changes the output's volume. Both increase volume but a balance is probably a good idea (based on raw natural instinct alone).
   *S.Thresh* Changes the level at which sound will be picked up. Useful if your room is loud. Not useful in this usecase however, keep it at 0.
   *Chunk* Pretty much how long the model cooks in the oven. 128 is a good balance between quality and speed.
   *Extra* No clue, max it out. Almost no difference to delay (probably).
   *GPU* Switch to CPU if you want a worse experience.

Step 7:
   The best place (according to me) to get new models is the 'AI Hub' Discord server. Not sure if this link will work but you can just search it up. https://discord.com/invite/ai-hub-1159260121998827560 Once you find a model in the 'voice models' tab, download it. I recommend creating a 'Models' folder in your 'Voice Changer' folder and putting all models in there in order to keep things organized. Inside the model you just downloaded there should be a .pth file and possibly a .index file.

Step 8:
   In the voice changer Client, find the 'edit' button. Find a blank Model slot and click 'upload'. Upload the .pth file to the 'Model', and the .index file to the 'Index', if it has one. Click upload and you're done. You can change the name and add a profile picture in the edit menu, and save the settings you apply to the model by clicking 'save setting' below *Index*.

---

## License

[MIT](LICENSE) — do what you like, no warranty.
