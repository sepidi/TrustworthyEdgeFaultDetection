#!/usr/bin/env bash
# Download + extract the 29 Paderborn bearings used in the paper (~17 GB of .rar: ~10 GB real + ~7 GB artificial; the extracted .mat files need about the same again):
#   17 healthy / real-damage bearings (training, validation, calibration, test)
#   12 artificial-damage bearings (training only)
#   bash scripts/download_pu.sh                                   # all 29 into data/raw
#   DEST=/big/disk/pu bash scripts/download_pu.sh                 # then set data.raw_dir in config.yaml
#   BEARINGS="K001 KA04" bash scripts/download_pu.sh              # any subset
# Known issue in the original data: N15_M01_F10_KA08_2.mat is damaged and cannot be read; the
# preprocessing skips it with a warning (this is expected and matches the paper).
# Needs: wget (or curl) and one of: unrar | unar | 7z  (sudo apt install unrar  or  p7zip-full)
# If the server URL changed, download the same .rar files manually from the KAt DataCenter page
# (https://mb.uni-paderborn.de/en/kat/research/bearing-datacenter/data-sets-and-download)
# into $DEST/_rar/ and re-run this script: it will only extract.
set -euo pipefail
BASE="${BASE:-https://groups.uni-paderborn.de/kat/BearingDataCenter}"
DEST="${DEST:-data/raw}"
REAL="K001 K002 K003 K004 K005 K006 KA04 KA15 KA16 KA22 KA30 KI04 KI14 KI16 KI17 KI18 KI21"
ARTIFICIAL="KA01 KA03 KA05 KA06 KA07 KA08 KA09 KI01 KI03 KI05 KI07 KI08"
BEARINGS="${BEARINGS:-$REAL $ARTIFICIAL}"
mkdir -p "$DEST/_rar"

extract() {  # $1 = rar file, $2 = target dir
  if command -v unrar >/dev/null; then unrar x -o+ -inul "$1" "$2/"
  elif command -v unar >/dev/null; then unar -q -f -o "$2" "$1"
  elif command -v 7z >/dev/null; then 7z x -y -bd -o"$2" "$1" >/dev/null
  else echo "Install unrar, unar or p7zip-full first"; exit 1; fi
}

for b in $BEARINGS; do
  rar="$DEST/_rar/$b.rar"
  if [ ! -s "$rar" ]; then
    echo "downloading $b ..."
    if command -v wget >/dev/null; then wget -c -q --show-progress -O "$rar" "$BASE/$b.rar"
    else curl -L -C - -o "$rar" "$BASE/$b.rar"; fi
  fi
  if [ ! -d "$DEST/$b" ] || [ "$(find "$DEST/$b" -name '*.mat' | wc -l)" -lt 80 ]; then
    echo "extracting $b ..."; extract "$rar" "$DEST"
  fi
  n=$(find "$DEST/$b" -name '*.mat' 2>/dev/null | wc -l)
  echo "  $b: $n .mat files (expected 80 = 4 conditions x 20)"
done
echo "Done. Optional: rm -r $DEST/_rar to free space."
echo "Check: every bearing should report 80 files (KA08 reports 80 too; one of them is unreadable, see above)."
