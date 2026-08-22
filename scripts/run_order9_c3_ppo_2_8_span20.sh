#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE:-/home/leus/.local/share/mamba/envs/isaaclab3/bin/python}"
PARENT_CHECKPOINT="${C3_PARENT_CHECKPOINT:-$REPOSITORY_ROOT/artifacts/p4_full/order9/stages/c3_pi_l_ppo_arbitrary_morphology/training_lineages/module_2_8_span20_from_2_7_u1_v1/initializer_unscaled/checkpoint.pt}"
BUCKETS="${C3_BUCKET_MANIFEST:-$REPOSITORY_ROOT/artifacts/p4_full/order9/stages/c3_pi_l_ppo_arbitrary_morphology/rollout_buckets_current_lineage_v5/manifest_c3_2_8_morphology_invariant_screened_v1.json}"
STAGE_ROOT="${C3_STAGE_ROOT:-$REPOSITORY_ROOT/artifacts/p4_full/order9/stages/c3_pi_l_ppo_arbitrary_morphology/training_lineages/module_2_8_span20_from_2_7_u1_v1/ppo_unscaled_fresh_v1}"
ADDITIONAL_UPDATE_COUNT="${C3_ADDITIONAL_UPDATE_COUNT:-0}"
STOP_AFTER_UPDATE_INDEX="${C3_STOP_AFTER_UPDATE_INDEX:-0}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
ulimit -s "${C3_PPO_STACK_LIMIT_KIB:-65536}"

for required_path in "$PARENT_CHECKPOINT" "$BUCKETS"; do
    if [[ ! -f "$required_path" ]]; then
        echo "required Order 9 C3 2--8-module span-20 input is missing: $required_path" >&2
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
    --training-max-module-count 8 \
    --additional-update-count "$ADDITIONAL_UPDATE_COUNT" \
    --stop-after-update-index "$STOP_AFTER_UPDATE_INDEX" \
    --prior-stage-manifest artifacts/p4_full/order9/stages/c0_order8_teacher_collection/stage_promoted.json \
    --prior-stage-manifest artifacts/p4_full/order9/stages/c1_pi_l_bc_fixed_nominal/stage_promoted.json \
    --prior-stage-manifest artifacts/p4_full/order9/stages/c2_pi_l_ppo_fixed_conservative/evaluations/extension_update_000049_final_200/stage_promoted.json \
    "$@"
