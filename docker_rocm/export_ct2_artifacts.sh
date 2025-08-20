#!/usr/bin/env sh
set -eu

# Configurable paths via env vars
OUT_DIR=${OUT_DIR:-/out}
SRC_WHEEL_DIR=${SRC_WHEEL_DIR:-/home/CTranslate2/python/dist}
SRC_CT2=${CTRANSLATE2_ROOT:?CTRANSLATE2_ROOT is not set}

echo "== Export CT2 Artifacts =="
echo "OUT_DIR=${OUT_DIR}"
echo "SRC_WHEEL_DIR=${SRC_WHEEL_DIR}"
echo "SRC_CT2=${SRC_CT2}"

# Ensure output directory
mkdir -p "${OUT_DIR}"
echo "-- Ensured output directory: ${OUT_DIR}"

# 1) Copy built Python wheel(s)
if [ -d "${SRC_WHEEL_DIR}" ]; then
  # Count wheels
  WHEEL_COUNT=$(find "${SRC_WHEEL_DIR}" -maxdepth 1 -type f -name '*.whl' 2>/dev/null | wc -l | tr -d ' ')
  echo "-- Searching for wheels in ${SRC_WHEEL_DIR} (found: ${WHEEL_COUNT})"
  FOUNDANY=0
  for f in "${SRC_WHEEL_DIR}"/*.whl; do
    if [ -e "$f" ]; then
      FOUNDANY=1
      dest="${OUT_DIR}/$(basename "$f")"
      if [ -e "$dest" ]; then
        echo "[wheel] Overwriting: $dest (from: $f)"
      else
        echo "[wheel] Copying: $f -> $dest"
      fi
      cp -v "$f" "$dest"
    fi
  done
  if [ "$FOUNDANY" -eq 0 ]; then
    echo "-- No wheel files found in ${SRC_WHEEL_DIR}"
  fi
else
  echo "-- Wheel directory does not exist: ${SRC_WHEEL_DIR}"
fi

# 2) Copy CTranslate2 install tree
DEST_CT2="${OUT_DIR}/ctranslate2_root"
mkdir -p "${DEST_CT2}"
if [ "${DEST_CT2}" != "${OUT_DIR}/ctranslate2_root" ]; then
  echo "-- Warning: unexpected CT2 dest path: ${DEST_CT2}"
fi

if [ -d "${DEST_CT2}" ]; then
  # Quick estimate of pre-existing files
  PRE_EXISTING=$(find "${DEST_CT2}" -type f 2>/dev/null | wc -l | tr -d ' ')
  echo "-- Destination CT2 tree exists (${DEST_CT2}), files may be overwritten (existing files: ${PRE_EXISTING})"
fi

echo "-- Copying CT2 install tree (verbose)"
cp -a -v "${SRC_CT2}"/. "${DEST_CT2}"/

# Summary
POST_CT2_COUNT=$(find "${DEST_CT2}" -type f 2>/dev/null | wc -l | tr -d ' ')
echo "== Export complete =="
echo "Wheels copied to: ${OUT_DIR}"
echo "CT2 tree copied to: ${DEST_CT2} (total files now: ${POST_CT2_COUNT})"
