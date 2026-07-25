#!/usr/bin/env bash
set -euo pipefail

benchmark_root="${AMSRR_VIM4_BENCHMARK_ROOT:-${HOME}/amsrr_vim4_benchmark}"
source_root="${benchmark_root}/source"
numpy_root="${benchmark_root}/python_packages_numpy_1_24_4"

if [[ ! -d "${source_root}" || ! -d "${numpy_root}" ]]; then
    echo "VIM4 posture benchmark environment is incomplete: ${benchmark_root}" >&2
    exit 1
fi

export PYTHONPATH="${source_root}:${numpy_root}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "${source_root}"
exec python3 scripts/order9_benchmark_posture_resolver.py \
    --fixture artifacts/p4_full/order9/posture_resolver/c3a_12_case_fixture_vim4_v4.json \
    --output artifacts/p4_full/order9/posture_resolver/vim4_benchmark.json \
    --repeats 4 \
    "$@"
