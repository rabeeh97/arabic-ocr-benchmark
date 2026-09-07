# Source this before working locally:  source env.sh
#
# Pins EVERY cache to the external SSD so nothing is written to the internal
# drive. The internal disk on this machine is near-full; keep it that way.

_REPO="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
_CACHE="${_REPO}/.cache"
mkdir -p "${_CACHE}"/{uv,hf,pip,torch,hf_transformers}

# uv: build/venv/link behaviour
export UV_CACHE_DIR="${_CACHE}/uv"
export UV_LINK_MODE=copy            # cache and target are on the same FS, but be explicit

# Hugging Face: datasets + model weights + hub
export HF_HOME="${_CACHE}/hf"
export HF_HUB_DISABLE_TELEMETRY=1
# Xet transfer backend is unreliable from this box: whole passes of the Gemini /
# free-tier sweeps die on "File reconstruction error: CAS Client Error" when
# fetching arocrbench_historyar / arocrbench_onlinekhatt. Force the plain-LFS
# download path (slower first fetch, then cached and fine).
export HF_HUB_DISABLE_XET=1
export TRANSFORMERS_CACHE="${_CACHE}/hf_transformers"

# pip / torch (in case anything shells out to them)
export PIP_CACHE_DIR="${_CACHE}/pip"
export TORCH_HOME="${_CACHE}/torch"

echo "caches -> ${_CACHE}"
echo "  UV_CACHE_DIR=${UV_CACHE_DIR}"
echo "  HF_HOME=${HF_HOME}"
