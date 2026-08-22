#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MICROMAMBA_EXECUTABLE="${MICROMAMBA_EXECUTABLE:-$HOME/.local/bin/micromamba}"
PARENT_CHECKPOINT="$REPOSITORY_ROOT/artifacts/p4_full/order9/stages/c3_pi_l_ppo_arbitrary_morphology/training_lineages/contact_space_projected_qpid_physical_feasibility_from_zero_command_v1/modules_2_3/update_000002/checkpoint_update_000002.pt"
BUCKETS="$REPOSITORY_ROOT/artifacts/p4_full/order9/stages/c3_pi_l_ppo_arbitrary_morphology/rollout_buckets_current_lineage_v5/manifest_c3_2_8_joint_only_release_smooth_val33_open35_v19.json"
STAGE_ROOT="${C3_STAGE_ROOT:-$REPOSITORY_ROOT/artifacts/p4_full/order9/stages/c3_pi_l_ppo_arbitrary_morphology/training_lineages/contact_space_projected_qpid_physical_feasibility_from_zero_command_v1/modules_2_4}"

for required_path in "$PARENT_CHECKPOINT" "$BUCKETS"; do
    if [[ ! -f "$required_path" ]]; then
        echo "required Order 9 C3 2--4-module input is missing: $required_path" >&2
        exit 2
    fi
done

cd "$REPOSITORY_ROOT"
exec taskset -c 0-31 "$MICROMAMBA_EXECUTABLE" run -a "" -n isaaclab3 -- \
    python scripts/order9_run_pi_l_ppo_stage.py \
    --config configs/training/order9_learning_curriculum.yaml \
    --stage c3_pi_l_ppo_arbitrary_morphology \
    --initial-checkpoint "$PARENT_CHECKPOINT" \
    --bucket-manifest "$BUCKETS" \
    --stage-root "$STAGE_ROOT" \
    --training-min-module-count 2 \
    --training-max-module-count 4 \
    --branch-parent-update-index 2 \
    --c3-action-contract contact_space_projected_policy_command \
    --additional-update-count 80 \
    --maximum-parallel-collector-process-count 2 \
    --collector-process-start-stagger-s 8 \
    --prior-stage-manifest artifacts/p4_full/order9/stages/c0_order8_teacher_collection/stage_promoted.json \
    --prior-stage-manifest artifacts/p4_full/order9/stages/c1_pi_l_bc_fixed_nominal/stage_promoted.json \
    --prior-stage-manifest artifacts/p4_full/order9/stages/c2_pi_l_ppo_fixed_conservative/evaluations/extension_update_000049_final_200/stage_promoted.json \
    "$@"
