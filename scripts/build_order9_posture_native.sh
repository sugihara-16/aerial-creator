#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_path="${repository_root}/amsrr/feasibility/native/order9_posture_native.cpp"
output_dir="${repository_root}/build/order9_posture_native"
python_executable="${PYTHON:-python3}"
# Compile against the interpreter that will load the extension. A system
# python3-config may describe a different ABI from the selected environment.
python_include="$("${python_executable}" -c 'import sysconfig; print(sysconfig.get_path("include"))')"
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
extension_suffix="$("${python_executable}" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
mkdir -p "${output_dir}"

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
  -I"${python_include}" \
  -I"${pybind11_include}" \
  -I/usr/include/eigen3 \
  "${source_path}" \
  $(pkg-config --libs fcl) \
  -o "${output_dir}/_order9_posture_native${extension_suffix}"

extension_path="${output_dir}/_order9_posture_native${extension_suffix}"
source_sha256="$(sha256sum "${source_path}" | awk '{print $1}')"
build_script_sha256="$(sha256sum "${BASH_SOURCE[0]}" | awk '{print $1}')"
printf '%s\n%s\n' \
  "${source_sha256}" \
  "${build_script_sha256}" \
  > "${extension_path}.build-identity"

echo "${extension_path}"
