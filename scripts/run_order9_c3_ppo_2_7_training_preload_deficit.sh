#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE:-/home/leus/.local/share/mamba/envs/isaaclab3/bin/python}"
PARENT_CHECKPOINT="${C3_PARENT_CHECKPOINT:-$REPOSITORY_ROOT/artifacts/p4_full/order9/stages/c3_pi_l_ppo_arbitrary_morphology/training_lineages/incremental_full_action_min2_clear300_preload_screened_from_update3_v2/modules_2_7/update_000016/checkpoint_update_000016.pt}"
BUCKETS="${C3_BUCKET_MANIFEST:-$REPOSITORY_ROOT/artifacts/p4_full/order9/stages/c3_pi_l_ppo_arbitrary_morphology/rollout_buckets_current_lineage_v5/manifest_c3_release_smooth_clear300_current_config_min2_preload_screened_v2.json}"
STAGE_ROOT="${C3_STAGE_ROOT:-$REPOSITORY_ROOT/artifacts/p4_full/order9/stages/c3_pi_l_ppo_arbitrary_morphology/training_lineages/incremental_full_action_min2_clear300_preload_screened_from_update3_v2/modules_2_7_training_preload_deficit_v1_from_update16}"
STOP_AFTER_UPDATE_INDEX="${C3_STOP_AFTER_UPDATE_INDEX:-17}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
ulimit -s "${C3_PPO_STACK_LIMIT_KIB:-65536}"

for required_path in "$PARENT_CHECKPOINT" "$BUCKETS"; do
    if [[ ! -f "$required_path" ]]; then
        echo "required Order 9 C3 training-preload-deficit input is missing: $required_path" >&2
        exit 2
    fi
done

cd "$REPOSITORY_ROOT"
exec taskset -c 0-31 "$PYTHON_EXECUTABLE" \
    scripts/order9_run_pi_l_ppo_stage.py \
    --config configs/training/order9_learning_curriculum.yaml \
    --stage c3_pi_l_ppo_arbitrary_morphology \
    --initial-checkpoint "$PARENT_CHECKPOINT" \
    --bucket-manifest "$BUCKETS" \
    --stage-root "$STAGE_ROOT" \
    --training-min-module-count 2 \
    --training-max-module-count 7 \
    --branch-parent-update-index 16 \
    --additional-update-count 8 \
    --c3-action-contract contact_space_projected_policy_command \
    --c3-training-nominal-preload-deficit-mm 0,2,4,6,8 \
    --c3-training-nominal-preload-deficit-module-count 7 \
    --stop-after-update-index "$STOP_AFTER_UPDATE_INDEX" \
    --maximum-parallel-collector-process-count 2 \
    --collector-process-start-stagger-s 8 \
    --prior-stage-manifest artifacts/p4_full/order9/stages/c0_order8_teacher_collection/stage_promoted.json \
    --prior-stage-manifest artifacts/p4_full/order9/stages/c1_pi_l_bc_fixed_nominal/stage_promoted.json \
    --prior-stage-manifest artifacts/p4_full/order9/stages/c2_pi_l_ppo_fixed_conservative/evaluations/extension_update_000049_final_200/stage_promoted.json \
    "$@"
