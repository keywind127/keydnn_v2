#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SRC_DIR="$ROOT_DIR/src/keydnn/infrastructure/native/src"
INC_DIR="$ROOT_DIR/src/keydnn/infrastructure/native/include"
OUT_DIR="$ROOT_DIR/src/keydnn/infrastructure/native/python"

OUT_LIB="$OUT_DIR/libkeydnn_native.so"

# -------------------------
# Load compiler settings from repo_root/.env (optional)
# Supports lines like:
#   KEYDNN_GPP=/path/to/g++
#   KEYDNN_MINGW_BIN=/path/to/bin   (fallback: uses KEYDNN_MINGW_BIN/g++)
#
# Notes:
# - This script is for Linux. If you happen to be using a custom g++ toolchain
#   path stored in .env (even if it's named "MINGW"), we honor it.
# - We ignore blank lines and comments starting with '#'.
# -------------------------
ENV_FILE="$ROOT_DIR/.env"

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

      if [[ "${val:0:1}" == "\"" && "${val: -1}" == "\"" ]]; then
        val="${val:1:-1}"
      fi

      case "$key" in
        KEYDNN_GPP|KEYDNN_MINGW_BIN)
          printf -v "$key" '%s' "$val"
          ;;
      esac
    fi
  done < "$ENV_FILE"
fi

# Determine compiler
GPP="${KEYDNN_GPP:-}"
if [[ -z "$GPP" && -n "${KEYDNN_MINGW_BIN:-}" ]]; then
  GPP="$KEYDNN_MINGW_BIN/g++"
fi
if [[ -z "$GPP" ]]; then
  GPP="g++"
fi

echo "[KeyDNN] Building native pooling kernels (Linux)"
echo "  Source   : $SRC_DIR"
echo "  Output   : $OUT_LIB"
echo "  Compiler : $GPP"

mkdir -p "$OUT_DIR"

"$GPP" -O3 -std=c++17 -fPIC -shared \
  -I"$INC_DIR" \
  "$SRC_DIR/keydnn_maxpool2d.cpp" \
  "$SRC_DIR/keydnn_avgpool2d.cpp" \
  "$SRC_DIR/keydnn_conv2d.cpp" \
  -o "$OUT_LIB"

echo "[KeyDNN] Build successful"
