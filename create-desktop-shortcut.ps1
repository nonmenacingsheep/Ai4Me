# Creates (or refreshes) a Desktop shortcut to Ai4Me, with the app icon.
# Run it directly, via create-desktop-shortcut.bat, or `npm run shortcut`.

$ErrorActionPreference = 'Stop'

# This script lives in the project root, so its own folder IS the project root.
$root    = Split-Path -Parent $MyInvocation.MyCommand.Path
$target  = Join-Path $root 'run.bat'
$icon    = Join-Path $root 'assets\icon.ico'
$desktop = [Environment]::GetFolderPath('Desktop')
$lnkPath = Join-Path $desktop 'Ai4Me.lnk'

if (-not (Test-Path $target)) { throw "Can't find run.bat at $target" }
if (-not (Test-Path $icon))   { throw "Can't find the icon at $icon" }

$shell = New-Object -ComObject WScript.Shell
$sc = $shell.CreateShortcut($lnkPath)
$sc.TargetPath       = $target
$sc.WorkingDirectory = $root
$sc.IconLocation     = "$icon,0"   # the app icon (assets\icon.ico)
$sc.WindowStyle      = 7           # start the launcher console minimized
$sc.Description      = 'Ai4Me - Aitha, your AI companion'
$sc.Save()

Write-Host "[Ai4Me] Desktop shortcut created with icon:" -ForegroundColor Green
Write-Host "        $lnkPath"
