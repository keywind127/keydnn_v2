@echo off
setlocal ENABLEEXTENSIONS ENABLEDELAYEDEXPANSION

set "ROOT_DIR=%~dp0.."
set "ENV_FILE=%ROOT_DIR%\.env"

set "SRC_DIR=%ROOT_DIR%\src\keydnn\infrastructure\native\src"
set "INC_DIR=%ROOT_DIR%\src\keydnn\infrastructure\native\include"
set "OUT_DIR=%ROOT_DIR%\src\keydnn\infrastructure\native\python"
set "OUT_LIB=%OUT_DIR%\keydnn_native.dll"

if not exist "%ENV_FILE%" (
  echo [KeyDNN] ERROR: .env not found at "%ENV_FILE%"
  exit /b 1
)

REM Load .env via PowerShell (BOM-safe)
for /f "usebackq tokens=1,* delims==" %%A in (`
  powershell -NoProfile -Command ^
    "$p='%ENV_FILE%'; " ^
    "$lines = Get-Content -LiteralPath $p -ErrorAction Stop; " ^
    "foreach($line in $lines) { " ^
    "  $line = $line.Trim(); " ^
    "  if($line -eq '' -or $line.StartsWith('#')) { continue } " ^
    "  if($line.StartsWith([char]0xFEFF)) { $line = $line.TrimStart([char]0xFEFF) } " ^
    "  Write-Output $line " ^
    "}"
`) do (
  set "K=%%A"
  set "V=%%B"
  for /f "tokens=* delims= " %%K in ("!K!") do set "K=%%K"
  if defined V (
    if "!V:~0,1!"=="^"" if "!V:~-1!"=="^"" set "V=!V:~1,-1!"
  )
  set "!K!=!V!"
)

REM Prefer KEYDNN_GPP; fallback to KEYDNN_MINGW_BIN\g++.exe
set "GPP=%KEYDNN_GPP%"
if not defined GPP (
  if defined KEYDNN_MINGW_BIN set "GPP=%KEYDNN_MINGW_BIN%\g++.exe"
)

if not defined GPP (
  echo [KeyDNN] ERROR: KEYDNN_GPP is not set and KEYDNN_MINGW_BIN is not set in .env
  exit /b 1
)

if defined KEYDNN_MINGW_BIN (
  set "PATH=%KEYDNN_MINGW_BIN%;%PATH%"
)

echo [KeyDNN] Building native maxpool kernel (Windows / MinGW-w64)
echo   Compiler: %GPP%
echo   Source  : %SRC_DIR%
echo   Output  : %OUT_LIB%

if not exist "%GPP%" (
  echo [KeyDNN] ERROR: g++.exe not found at "%GPP%"
  exit /b 1
)

if not exist "%OUT_DIR%" mkdir "%OUT_DIR%"

"%GPP%" -O3 -std=c++17 -shared ^
  -I"%INC_DIR%" ^
  "%SRC_DIR%\keydnn_maxpool2d.cpp" ^
  "%SRC_DIR%\keydnn_avgpool2d.cpp" ^
  "%SRC_DIR%\keydnn_conv2d.cpp" ^
  -o "%OUT_LIB%"

if errorlevel 1 (
  echo [KeyDNN] Build failed
  exit /b 1
)

echo [KeyDNN] Build successful
endlocal
