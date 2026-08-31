#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_path="${repository_root}/amsrr/feasibility/native/order9_posture_native.cpp"
default_build_script="${repository_root}/scripts/build_order9_posture_native.sh"
output_dir="${AMSRR_ORDER9_POSTURE_NATIVE_V13_OUTPUT_DIR:-${repository_root}/build/order9_posture_native_v13}"
generated_source="${output_dir}/order9_posture_native_v13.cpp"
python_executable="${PYTHON:-python3}"
pybind11_include="$(
  "${python_executable}" - <<'PY'
from pathlib import Path

try:
    import pybind11
except ModuleNotFoundError:
    try:
        import torch
    except ModuleNotFoundError:
        print("/usr/include")
    else:
        include = Path(torch.__file__).resolve().parent / "include"
        if not (include / "pybind11" / "pybind11.h").is_file():
            raise SystemExit("PyTorch does not provide pybind11 headers")
        print(include)
else:
    print(pybind11.get_include())
PY
)"
extension_suffix="$(
  "${python_executable}" - <<'PY'
import sysconfig

value = sysconfig.get_config_var("EXT_SUFFIX")
if not value:
    raise SystemExit("Python extension suffix is unavailable")
print(value)
PY
)"
python_includes="$(
  "${python_executable}" - <<'PY'
import sysconfig

paths = sysconfig.get_paths()
values = []
for key in ("include", "platinclude"):
    value = paths.get(key)
    if value and value not in values:
        values.append(value)
print(" ".join(f"-I{value}" for value in values))
PY
)"
mkdir -p "${output_dir}"

"${python_executable}" - "${source_path}" "${generated_source}" <<'PY'
from pathlib import Path
import sys

source = Path(sys.argv[1]).read_text(encoding="utf-8")
old = """      const bool broad_phase_safe =
          exact
              ? broad_phase_clearance > 0.0
              : broad_phase_clearance > activation_distance;
"""
new = """      // v13: an AABB distance proves a requested clearance only when
      // the lower bound exceeds that clearance.  A positive but smaller
      // lower bound must continue to the exact convex/STL distance query.
      const bool broad_phase_safe =
          broad_phase_clearance > activation_distance;
"""
if source.count(old) != 1:
    raise SystemExit("v13 native broad-phase source anchor differs")
Path(sys.argv[2]).write_text(source.replace(old, new), encoding="utf-8")
PY

g++ \
  -O3 \
  -DNDEBUG \
  -march=native \
  -Wall \
  -Wextra \
  -Wpedantic \
  -fPIC \
  -shared \
  -std=c++17 \
  ${python_includes} \
  -I"${pybind11_include}" \
  -I/usr/include/eigen3 \
  "${generated_source}" \
  $(pkg-config --libs fcl) \
  -o "${output_dir}/_order9_posture_native${extension_suffix}"

extension_path="${output_dir}/_order9_posture_native${extension_suffix}"
source_sha256="$(sha256sum "${source_path}" | awk '{print $1}')"
default_build_script_sha256="$(sha256sum "${default_build_script}" | awk '{print $1}')"
v13_build_script_sha256="$(sha256sum "${BASH_SOURCE[0]}" | awk '{print $1}')"
generated_source_sha256="$(sha256sum "${generated_source}" | awk '{print $1}')"
printf '%s\n%s\n' \
  "${source_sha256}" \
  "${default_build_script_sha256}" \
  > "${extension_path}.build-identity"
printf '%s\n%s\n%s\n%s\n' \
  "order9_posture_native_exact_margin_v13" \
  "${source_sha256}" \
  "${v13_build_script_sha256}" \
  "${generated_source_sha256}" \
  > "${extension_path}.v13-build-identity"

echo "${extension_path}"
