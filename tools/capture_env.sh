#!/usr/bin/env bash
# Record the exact software/hardware environment of a machine into env/<name>.txt
#   bash tools/capture_env.sh workstation      # on the training workstation
#   bash tools/capture_env.sh jetson           # on the Jetson Orin Nano
# Commit the resulting file(s) so others can see exactly what produced the shipped results.
set -uo pipefail
cd "$(dirname "$0")/.."
NAME="${1:-$(hostname)}"
mkdir -p env
OUT="env/${NAME}.txt"
{
  echo "# captured $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "## OS";       uname -srm; (lsb_release -ds 2>/dev/null || grep PRETTY_NAME /etc/os-release)
  echo "## CPU";      (lscpu 2>/dev/null | grep -E "Model name|^CPU\(s\)") || true
  echo "## GPU";      (nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null) || echo "nvidia-smi n/a"
  echo "## Jetson";   (cat /etc/nv_tegra_release 2>/dev/null; sudo -n nvpmodel -q 2>/dev/null || nvpmodel -q 2>/dev/null) || echo "not a Jetson"
  echo "## Python";   python3 --version
  python3 - <<'PY'
import importlib
for m in ["torch", "torch_pruning", "onnx", "onnxruntime", "numpy", "scipy", "sklearn", "pandas", "matplotlib",
          "yaml", "tensorrt", "polygraphy"]:
    try:
        mod = importlib.import_module(m)
        v = getattr(mod, "__version__", "?")
        extra = ""
        if m == "torch":
            extra = f"  (CUDA {mod.version.cuda}, cuDNN {mod.backends.cudnn.version()})"
        print(f"{m}=={v}{extra}")
    except Exception:
        print(f"{m}: not installed")
PY
  echo "## pip freeze"; python3 -m pip freeze 2>/dev/null
} > "$OUT" 2>&1
echo "Wrote $OUT"
