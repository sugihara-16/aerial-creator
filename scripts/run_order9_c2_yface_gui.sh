#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MICROMAMBA_EXECUTABLE="${MICROMAMBA_EXECUTABLE:-$HOME/.local/bin/micromamba}"
STOP_AFTER_PHASE="${STOP_AFTER_PHASE:-contact_acquisition}"
MAX_WINDOWS="${MAX_WINDOWS:-30}"
KEEP_OPEN_S="${KEEP_OPEN_S:-10}"
OUTPUT_JSON="${OUTPUT_JSON:-$REPOSITORY_ROOT/artifacts/p4_full/order9/posture_resolver/c2_yface_contact_acquisition_gui.json}"

CHECKPOINT="$REPOSITORY_ROOT/artifacts/p4_full/order9/c3_preparation/pi_l_active_knot_initializer.pt"
BUCKET_MANIFEST="$REPOSITORY_ROOT/artifacts/p4_full/order9/stages/c2_pi_l_ppo_fixed_conservative/rollout_buckets/manifest.json"
ARTICULATED_URDF="$REPOSITORY_ROOT/artifacts/p4_full/order9/fixed_nominal_asset/1817317197beefe5b1356e365d52a96f546168d99883280c8ecd0617aafe2080/holon_order9_fixed_nominal_articulated_v3_1817317197be.urdf"

if [[ ! -x "$MICROMAMBA_EXECUTABLE" ]]; then
    echo "micromamba is unavailable: $MICROMAMBA_EXECUTABLE" >&2
    exit 2
fi
for required_path in "$CHECKPOINT" "$BUCKET_MANIFEST" "$ARTICULATED_URDF"; do
    if [[ ! -f "$required_path" ]]; then
        echo "required Order 9 input is missing: $required_path" >&2
        exit 2
    fi
done

mkdir -p "$(dirname "$OUTPUT_JSON")"
cd "$REPOSITORY_ROOT"

echo "Order 9 GUI: Isaac/Kitの初期化には約1～2分かかります。" >&2
echo "端末の ORDER9_SHADOW_STATUS=scene_reset_complete 後に物理再生が始まります。" >&2
echo "GUI観測専用: Isaac接触・制御は実行し、重いmesh証拠集計だけ省略します。" >&2

PYTHONUNBUFFERED=1 PYTHONPATH="$REPOSITORY_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    "$MICROMAMBA_EXECUTABLE" run -a "" -n isaaclab3 -- \
    python "$REPOSITORY_ROOT/scripts/order9_isaac_shadow_smoke.py" \
    --pi-l-checkpoint "$CHECKPOINT" \
    --pi-l-checkpoint-sha256 \
        8dc6157f1e506a6c776a173b41ee6b2a8e51e18537334c273fde9d18951f2da3 \
    --c2-bucket-manifest "$BUCKET_MANIFEST" \
    --c2-bucket-id validation-000008-138352e449ab \
    --c2-articulated-urdf "$ARTICULATED_URDF" \
    --c2-stop-after-phase "$STOP_AFTER_PHASE" \
    --c2-rolling-window-count "$MAX_WINDOWS" \
    --skip-mesh-collision-evidence \
    --gui-realtime \
    --gui-keep-open-s "$KEEP_OPEN_S" \
    --output-json "$OUTPUT_JSON"
