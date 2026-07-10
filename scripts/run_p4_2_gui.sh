#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
micromamba_bin="${MICROMAMBA_BIN:-$HOME/.local/bin/micromamba}"
keep_open_after_rollout_s="${KEEP_OPEN_AFTER_ROLLOUT_S:-20}"
isaaclab_launcher="$HOME/IsaacLab/isaaclab.sh"

if [[ ! -x "$micromamba_bin" ]]; then
    echo "micromamba was not found at: $micromamba_bin" >&2
    exit 1
fi
if [[ ! -x "$isaaclab_launcher" ]]; then
    echo "Isaac Lab launcher was not found at: $isaaclab_launcher" >&2
    exit 1
fi

export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"

exec "$micromamba_bin" run -n isaaclab3 env \
    "PYTHONPATH=$PYTHONPATH" \
    "OMNI_KIT_ACCEPT_EULA=$OMNI_KIT_ACCEPT_EULA" \
    python3 "$repo_root/scripts/p4_2_deterministic_rollout.py" \
    --config "$repo_root/configs/training/p4_2_deterministic_rollout.yaml" \
    --real \
    --viewer kit \
    --realtime-playback \
    --keep-open-after-rollout-s "$keep_open_after_rollout_s" \
    "$@"
