#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SRC_DIR="$ROOT_DIR/src/keydnn/infrastructure/native/src"
INC_DIR="$ROOT_DIR/src/keydnn/infrastructure/native/include"
OUT_DIR="$ROOT_DIR/src/keydnn/infrastructure/native/python"

# Outputs
OUT_LIB_NOOMP="$OUT_DIR/libkeydnn_native_noomp.so"
OUT_LIB_OMP="$OUT_DIR/libkeydnn_native_omp.so"
OUT_LIB_DEFAULT="$OUT_DIR/libkeydnn_native.so"

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

echo "[KeyDNN] Building native kernels (Linux)"
echo "  Source   : $SRC_DIR"
echo "  Output   : $OUT_DIR"
echo "  Compiler : $GPP"

mkdir -p "$OUT_DIR"

COMMON_FLAGS=(-O3 -std=c++17 -fPIC -shared)
INCLUDES=(-I"$INC_DIR")
SOURCES=(
  "$SRC_DIR/keydnn_maxpool2d.cpp"
  "$SRC_DIR/keydnn_avgpool2d.cpp"
  "$SRC_DIR/keydnn_conv2d.cpp"
  "$SRC_DIR/keydnn_conv2d_transpose.cpp"
)

# Optional: fail fast if any source is missing
for f in "${SOURCES[@]}"; do
  if [[ ! -f "$f" ]]; then
    echo "[KeyDNN] ERROR: source file not found: $f" >&2
    exit 1
  fi
done

# -------------------------
# 1) Baseline build (no OpenMP)
# -------------------------
echo
echo "[KeyDNN] Build: baseline (no OpenMP)"
echo "  -> $OUT_LIB_NOOMP"
"$GPP" "${COMMON_FLAGS[@]}" \
  "${INCLUDES[@]}" \
  "${SOURCES[@]}" \
  -o "$OUT_LIB_NOOMP"

# -------------------------
# 2) OpenMP build
# -------------------------
echo
echo "[KeyDNN] Build: OpenMP (-fopenmp)"
echo "  -> $OUT_LIB_OMP"
"$GPP" "${COMMON_FLAGS[@]}" -fopenmp \
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
