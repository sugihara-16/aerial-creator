#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MICROMAMBA_EXECUTABLE="${MICROMAMBA_EXECUTABLE:-$HOME/.local/bin/micromamba}"
INITIALIZER="$REPOSITORY_ROOT/artifacts/p4_full/order9/c3_preparation/pi_l_active_knot_initializer_current_physical_v1.pt"
BUCKETS="$REPOSITORY_ROOT/artifacts/p4_full/order9/stages/c3_pi_l_ppo_arbitrary_morphology/rollout_buckets_current_lineage_v5/manifest_c3_nominal_replay_v1.json"

for required_path in "$INITIALIZER" "$BUCKETS"; do
    if [[ ! -f "$required_path" ]]; then
        echo "required Order 9 C3 input is missing: $required_path" >&2
        exit 2
    fi
done

cd "$REPOSITORY_ROOT"
exec "$MICROMAMBA_EXECUTABLE" run -a "" -n isaaclab3 -- \
    python scripts/order9_run_pi_l_ppo_stage.py \
    --config configs/training/order9_learning_curriculum.yaml \
    --stage c3_pi_l_ppo_arbitrary_morphology \
    --initial-checkpoint "$INITIALIZER" \
    --bucket-manifest "$BUCKETS" \
    --prior-stage-manifest artifacts/p4_full/order9/stages/c0_order8_teacher_collection/stage_promoted.json \
    --prior-stage-manifest artifacts/p4_full/order9/stages/c1_pi_l_bc_fixed_nominal/stage_promoted.json \
    --prior-stage-manifest artifacts/p4_full/order9/stages/c2_pi_l_ppo_fixed_conservative/evaluations/extension_update_000049_final_200/stage_promoted.json \
    "$@"
