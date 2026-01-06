@echo off
setlocal ENABLEEXTENSIONS ENABLEDELAYEDEXPANSION

set "ROOT_DIR=%~dp0.."
set "ENV_FILE=%ROOT_DIR%\.env"

set "SRC_DIR=%ROOT_DIR%\src\keydnn\infrastructure\native\src"
set "INC_DIR=%ROOT_DIR%\src\keydnn\infrastructure\native\include"
set "OUT_DIR=%ROOT_DIR%\src\keydnn\infrastructure\native\python"

REM Baseline + OpenMP outputs
set "OUT_LIB_BASE=%OUT_DIR%\keydnn_native_noomp.dll"
set "OUT_LIB_OMP=%OUT_DIR%\keydnn_native_omp.dll"

REM Back-compat default
set "OUT_LIB_DEFAULT=%OUT_DIR%\keydnn_native.dll"

REM -------------------------
REM Load .env (BOM-safe)
REM -------------------------
if not exist "%ENV_FILE%" (
  echo [KeyDNN] ERROR: .env not found at "%ENV_FILE%"
  exit /b 1
)

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

echo [KeyDNN] Building native kernels (Windows / MinGW-w64)
echo   Compiler: %GPP%
echo   Source  : %SRC_DIR%
echo   Include : %INC_DIR%
echo   Output  : %OUT_DIR%

if not exist "%GPP%" (
  echo [KeyDNN] ERROR: g++.exe not found at "%GPP%"
  exit /b 1
)

if not exist "%OUT_DIR%" mkdir "%OUT_DIR%"

set "COMMON_FLAGS=-O3 -std=c++17 -shared"
set "SRC1=%SRC_DIR%\keydnn_maxpool2d.cpp"
set "SRC2=%SRC_DIR%\keydnn_avgpool2d.cpp"
set "SRC3=%SRC_DIR%\keydnn_conv2d.cpp"
set "SRC4=%SRC_DIR%\keydnn_conv2d_transpose.cpp"

for %%F in ("%SRC1%" "%SRC2%" "%SRC3%" "%SRC4%") do (
  if not exist "%%~F" (
    echo [KeyDNN] ERROR: source file not found: %%~F
    exit /b 1
  )
)

REM -------------------------
REM 1) Baseline build (no OpenMP)
REM -------------------------
echo.
echo [KeyDNN] Build: baseline (no OpenMP)
echo   ^> %OUT_LIB_BASE%

"%GPP%" %COMMON_FLAGS% ^
  "-I%INC_DIR%" ^
  "%SRC1%" ^
  "%SRC2%" ^
  "%SRC3%" ^
  "%SRC4%" ^
  -o "%OUT_LIB_BASE%" || goto :fail_base

REM -------------------------
REM 2) OpenMP build
REM -------------------------
echo.
echo [KeyDNN] Build: OpenMP (-fopenmp)
echo   ^> %OUT_LIB_OMP%

"%GPP%" %COMMON_FLAGS% -fopenmp ^
  "-I%INC_DIR%" ^
  "%SRC1%" ^
  "%SRC2%" ^
  "%SRC3%" ^
  "%SRC4%" ^
  -o "%OUT_LIB_OMP%" || goto :fail_omp

REM -------------------------
REM 2b) Copy OpenMP runtime deps next to the DLL
REM     (required for ctypes.CDLL on Windows)
REM -------------------------
echo.
echo [KeyDNN] Staging OpenMP runtime DLLs next to output (required for loading)

REM Determine MinGW bin dir:
REM 1) KEYDNN_MINGW_BIN if non-empty
REM 2) derive from %GPP% (e.g., ...\bin\g++.exe -> ...\bin)
set "MINGW_BIN=%KEYDNN_MINGW_BIN%"
if "%MINGW_BIN%"=="" (
  for %%I in ("%GPP%") do set "MINGW_BIN=%%~dpI"
)

REM Normalize: remove trailing backslash if present
if not "%MINGW_BIN%"=="" (
  if "%MINGW_BIN:~-1%"=="\" set "MINGW_BIN=%MINGW_BIN:~0,-1%"
)

echo [KeyDNN] MinGW bin: %MINGW_BIN%

if "%MINGW_BIN%"=="" (
  echo [KeyDNN] WARN: Could not determine MinGW bin directory. Skipping runtime DLL copy.
) else (
  if exist "%MINGW_BIN%\libgomp-1.dll" (
    copy /Y "%MINGW_BIN%\libgomp-1.dll" "%OUT_DIR%\" >nul
  ) else (
    echo [KeyDNN] WARN: libgomp-1.dll not found in %MINGW_BIN%
  )

  if exist "%MINGW_BIN%\libgcc_s_seh-1.dll" (
    copy /Y "%MINGW_BIN%\libgcc_s_seh-1.dll" "%OUT_DIR%\" >nul
  ) else (
    echo [KeyDNN] WARN: libgcc_s_seh-1.dll not found in %MINGW_BIN%
  )

  if exist "%MINGW_BIN%\libwinpthread-1.dll" (
    copy /Y "%MINGW_BIN%\libwinpthread-1.dll" "%OUT_DIR%\" >nul
  ) else (
    echo [KeyDNN] WARN: libwinpthread-1.dll not found in %MINGW_BIN%
  )
)


REM -------------------------
REM 3) Select default DLL (back-compat)
REM -------------------------
copy /Y "%OUT_LIB_OMP%" "%OUT_LIB_DEFAULT%" >nul || goto :fail_default

echo.
echo [KeyDNN] Build successful
echo   Baseline: %OUT_LIB_BASE%
echo   OpenMP  : %OUT_LIB_OMP%
echo   Default : %OUT_LIB_DEFAULT%  (currently points to OpenMP build)
echo.
echo [KeyDNN] Note: OpenMP runtime DLLs should exist in:
echo   %OUT_DIR%
echo   - libgomp-1.dll
echo   - libgcc_s_seh-1.dll
echo   - libwinpthread-1.dll

endlocal
exit /b 0

:fail_base
echo [KeyDNN] Build failed (baseline)
endlocal
exit /b 1

:fail_omp
echo [KeyDNN] Build failed (OpenMP)
endlocal
exit /b 1

:fail_default
echo [KeyDNN] ERROR: could not update default DLL: %OUT_LIB_DEFAULT%
endlocal
exit /b 1
