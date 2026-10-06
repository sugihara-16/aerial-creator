#!/usr/bin/env bash
set -euo pipefail
repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_path="${repository_root}/amsrr/controllers/native/request_dynamics_native.cpp"
output_dir="${repository_root}/build/request_dynamics_native"
python_executable="${PYTHON:-python3}"
python_include="$("${python_executable}" -c 'import sysconfig; print(sysconfig.get_path("include"))')"
pybind_include="$("${python_executable}" -c 'from pathlib import Path; import torch; print(Path(torch.__file__).parent / "include")')"
suffix="$("${python_executable}" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
mkdir -p "${output_dir}"
extension_path="${output_dir}/_request_dynamics_native${suffix}"
g++ -O3 -DNDEBUG -ffp-contract=off -march=native -Wall -Wextra -fvisibility=hidden \
  -shared -fPIC -std=c++17 -I"${python_include}" -I"${pybind_include}" \
  -I/usr/include/eigen3 "${source_path}" -o "${extension_path}.tmp"
mv "${extension_path}.tmp" "${extension_path}"
sha256sum "${source_path}" "${BASH_SOURCE[0]}" | awk '{print $1}' > "${extension_path}.build-identity"
echo "${extension_path}"
