#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SRC_DIR="$ROOT_DIR/src/keydnn/infrastructure/native/src"
INC_DIR="$ROOT_DIR/src/keydnn/infrastructure/native/include"
OUT_DIR="$ROOT_DIR/src/keydnn/infrastructure/native/python"

# Outputs
OUT_LIB_NOOMP="$OUT_DIR/libkeydnn_native_noomp.dylib"
OUT_LIB_OMP="$OUT_DIR/libkeydnn_native_omp.dylib"
OUT_LIB_DEFAULT="$OUT_DIR/libkeydnn_native.dylib"

# -------------------------
# Load compiler settings from repo_root/.env (optional)
# Supports lines like:
#   KEYDNN_CXX=/path/to/clang++
#   KEYDNN_GPP=/path/to/c++   (kept for symmetry with Linux/Windows)
#   KEYDNN_MINGW_BIN=/path/to/bin (fallback: uses KEYDNN_MINGW_BIN/g++)
#
# Notes:
# - This is macOS. Default compiler is clang++.
# - If KEYDNN_CXX is provided, it takes highest priority.
# -------------------------
ENV_FILE="$ROOT_DIR/.env"

KEYDNN_CXX="${KEYDNN_CXX:-}"
KEYDNN_GPP="${KEYDNN_GPP:-}"
KEYDNN_MINGW_BIN="${KEYDNN_MINGW_BIN:-}"

if [[ -f "$ENV_FILE" ]]; then
  while IFS= read -r line || [[ -n "$line" ]]; do
    # trim leading/trailing spaces
    line="${line#"${line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"

    [[ -z "$line" ]] && continue
    [[ "${line:0:1}" == "#" ]] && continue

    if [[ "$line" == *"="* ]]; then
      key="${line%%=*}"
      val="${line#*=}"

      key="${key#"${key%%[![:space:]]*}"}"
      key="${key%"${key##*[![:space:]]}"}"

      # strip surrounding quotes
      if [[ "${val:0:1}" == "\"" && "${val: -1}" == "\"" ]]; then
        val="${val:1:-1}"
      fi

      case "$key" in
        KEYDNN_CXX|KEYDNN_GPP|KEYDNN_MINGW_BIN)
          printf -v "$key" '%s' "$val"
          ;;
      esac
    fi
  done < "$ENV_FILE"
fi

# Determine compiler priority:
# 1) KEYDNN_CXX
# 2) KEYDNN_GPP
# 3) KEYDNN_MINGW_BIN/g++  (rare on macOS, but allowed)
# 4) clang++
CXX="${KEYDNN_CXX:-}"
if [[ -z "$CXX" ]]; then
  CXX="${KEYDNN_GPP:-}"
fi
if [[ -z "$CXX" && -n "${KEYDNN_MINGW_BIN:-}" ]]; then
  CXX="$KEYDNN_MINGW_BIN/g++"
fi
if [[ -z "$CXX" ]]; then
  CXX="clang++"
fi

echo "[KeyDNN] Building native kernels (macOS)"
echo "  Source   : $SRC_DIR"
echo "  Output   : $OUT_DIR"
echo "  Compiler : $CXX"

mkdir -p "$OUT_DIR"

COMMON_FLAGS=(-O3 -std=c++17 -fPIC -shared)
INCLUDES=(-I"$INC_DIR")
SOURCES=(
  "$SRC_DIR/keydnn_maxpool2d.cpp"
  "$SRC_DIR/keydnn_avgpool2d.cpp"
  "$SRC_DIR/keydnn_conv2d.cpp"
)

# -------------------------
# 1) Baseline build (no OpenMP)
# -------------------------
echo
echo "[KeyDNN] Build: baseline (no OpenMP)"
echo "  -> $OUT_LIB_NOOMP"
"$CXX" "${COMMON_FLAGS[@]}" \
  "${INCLUDES[@]}" \
  "${SOURCES[@]}" \
  -o "$OUT_LIB_NOOMP"

# -------------------------
# 2) OpenMP build
# -------------------------
echo
echo "[KeyDNN] Build: OpenMP"
echo "  -> $OUT_LIB_OMP"
"$CXX" "${COMMON_FLAGS[@]}" \
  -Xpreprocessor -fopenmp \
  -lomp \
  "${INCLUDES[@]}" \
  "${SOURCES[@]}" \
  -o "$OUT_LIB_OMP"

# -------------------------
# 3) Select default (back-compat)
#    Change to NOOMP if you want baseline as default.
# -------------------------
cp -f "$OUT_LIB_OMP" "$OUT_LIB_DEFAULT"

echo
echo "[KeyDNN] Build successful"
echo "  Baseline: $OUT_LIB_NOOMP"
echo "  OpenMP  : $OUT_LIB_OMP"
echo "  Default : $OUT_LIB_DEFAULT  (currently points to OpenMP build)"
