# VIM4 Compute Test Log

## 2026-07-24: Production promotion of the validated native posture IK

- Scope: Promote the already-approved C++/Eigen complete fixed-centroidal IK loop and proxy-only convex/FCL collision gate from the isolated experiment into the production posture-resolver path.  No solver tolerance, iteration budget, regularization, joint/CoM/anchor/timing authority, policy action, QPID/QP authority, or learned-policy schema changed.
- Production layout:
  - native source: `amsrr/feasibility/native/order9_posture_native.cpp`;
  - host build: `scripts/build_order9_posture_native.sh`;
  - ABI-specific output: ignored `build/order9_posture_native/_order9_posture_native<EXT_SUFFIX>`;
  - hash-bound allowed-pair manifest: `configs/robot/holon_nominal_collision_pairs_v1.json`;
  - Python adapters: `amsrr/feasibility/order9_native_loader.py`, `order9_native_posture_ik.py`, `order9_posture_collision.py`.
- Runtime boundary: Ordinary offline resolution prefers the native extension and exposes the readable Python solver only as a non-collision fallback.  Production `C_H` supplies the target box ID/size from its authenticated bucket and current pose from the runtime observation or knot object target; it requires the native convex solver and fails closed if the host extension/build identity, box scene, URDF/mesh manifest, IK feasibility, or independent post-solve `5 mm` convex admission is missing or inconsistent.  Runtime never compiles code.
- Build hardening: The host extension is bound to the current native source and build-script SHA-256 by a generated sidecar.  Stale or manually copied binaries without that identity fail before load.  The promoted build omits `-ffast-math`; this prevents the extension from changing the Python process's subnormal floating-point mode while retaining the tested C++ algorithm.
- PC correctness/performance:
  - direct C++ versus NumPy FK/anchors/CoM: `384` samples over `12` morphologies, maximum difference `8.882e-16`, threshold `1e-12`;
  - complete production resolver: all `12` immutable cases passed the `1e-8` comparison, maximum trajectory difference `1.585e-10`; three repetitions per case measured mean/max cold `0.09355 / 0.10998 s` and mean/max warm `0.08779 / 0.09482 s`;
  - eight-module convex path: `9/9` raw knots IK-feasible and independently admitted; mean scene/IK/admission `0.00487 / 0.00610 / 0.00149 s`, with mean `10,711/10,747` pairs conservatively AABB-pruned;
  - exact-capable versus proxy-only comparison: all `12` cases/`36` knots matched exactly in q, position/attitude errors, feasibility, iterations, and proxy admission; an exact query against the proxy-only kernel failed closed;
  - allowed-contact test: the two explicitly assigned object-contact links were admitted, while removing those allowances rejected the same configuration with exactly two colliding pairs.
- Evidence:
  - resolver report `artifacts/p4_full/order9/posture_resolver/production_native_promotion_pc_12case.json`, SHA-256 `d36db57ec260c5885f254531d4e09b09ff691f248d4edd013b23b8be76798132`;
  - convex report `artifacts/p4_full/order9/posture_resolver/production_native_convex_pc_c3a_08_chain.json`, SHA-256 `a81a88fd07006fde351d942e15e9db12b02a8e60c11177c2a457e720eaf5568a`;
  - x86-64 extension SHA-256 `a4705e35406c52d513a6d6fe6f6db07ba0d6c49567855254f1b3def994567df4`.
- VIM4 status: The algorithmic native and proxy-only implementations were already built and measured on VIM4 in the entries below.  The new production module name/build-identity wrapper and the removal of `-ffast-math` have not yet been rebuilt/rebenchmarked on the board; deployers must run the production build script natively on VIM4 before inference.  No x86-64 binary is portable to aarch64.
- Test status: Targeted native/resolver/production-gate tests passed.  The repository unit suite completed `1225 passed, 3 skipped`; two unrelated pre-existing environment/test-contract failures remained: missing optional `trimesh`, and a visualization test that does not provide the runtime observation now required by the existing phase-local teacher.

## 2026-07-24: Minimal posture-resolver benchmark environment

- Scope: Prepare a minimal, non-Isaac environment on the Khadas VIM4 for deterministic Order 9 posture-trajectory resolver correctness and latency measurements.
- Source specification: `A-MSRR_codex_ready_spec_v0_4_ja.md` v0.4 plus the approved `learned pi_H -> deterministic IK -> learned pi_L -> QPID` amendment recorded in `AMSRR_design_modification_by_codex.md`.
- Target: `khadas@192.168.96.47`.
- Authorization: The user authorized command execution on the target for this setup.
- Remote access: Installed the PC's existing SSH public key for the `khadas` account; passwordless SSH from this PC now succeeds.
- Host audit:
  - OS: Ubuntu 22.04.5 LTS, Linux 5.15.137.
  - Architecture: `aarch64`.
  - CPU: 8 logical CPUs.
  - Memory: 7,990,944 KiB reported by `/proc/meminfo`.
  - Storage: approximately 25 GiB free on `/`.
  - Python: CPython 3.10.12 with `venv` support.
  - NumPy: not initially installed.
- Benchmark boundary: `scripts/order9_benchmark_posture_resolver.py` consumes a PC-prepared, hash-bound 12-case fixture and measures only deterministic raw-trajectory posture resolution. It does not require or exercise Isaac, PyTorch, learned policy inference, QPID, collision checking, grasp dynamics, or transport.
- Local dependency audit:
  - Fixture: `artifacts/p4_full/order9/posture_resolver/c3a_12_case_fixture.json`, SHA-256 `90d4fbd75bab7f2881952765867517d80ea2cf602ab0ebee2e14869215f7a441`.
  - Fixture preparation remains PC-only.
  - The target calculation requires NumPy at IK solve time, the imported Python subset, robot-model YAML files, and `assets/robots/holon/holon.urdf`.
- Target environment:
  - Root: `/home/khadas/amsrr_vim4_benchmark`.
  - Transferred source/input subset: 117 files and approximately 3.1 MiB. The Python files were selected from modules imported by a successful local execution of the fixed 12-case benchmark; no repository history, Isaac, PyTorch, checkpoint, training dataset, mesh set, or unrelated artifact tree was copied.
  - Runtime dependency: NumPy 1.24.4 aarch64 wheel, SHA-256 `79fc682a374c4a8ed08b331bef9c5f582585d1048fa6d80bc6c35bc384eee9b4`, extracted under the benchmark root. The system-provided PyYAML 5.4.1 is reused.
  - Isolation: Runtime lookup is restricted with `PYTHONPATH` to the transferred source and extracted NumPy. BLAS/OpenMP-related thread counts are fixed to one for repeatable latency measurements.
  - An initial attempt to create a standard venv exposed a missing `python3.10-venv`. The Ubuntu package-index refresh was stopped because the board's mirror was unusually slow; a subsequent package install encountered stale-mirror 404 responses and installed nothing. `dpkg --audit` is clean, `python3.10-venv` remains uninstalled, and the final environment does not depend on apt or pip.
- Portability corrections:
  - The original v1 fixture encoded absolute PC source paths through `PhysicalModel.stable_hash()`, so a correctly copied model failed before calculation on a different project root.
  - Added a benchmark-only, content-backed PhysicalModel identity that replaces the URDF and joint-actuator source paths with their existing SHA-256 identities. Production `PhysicalModel.stable_hash()` and control/runtime contracts are unchanged.
  - Preliminary aarch64 solves differed from x86_64 only in floating-point tails. A nine-decimal quantized hash passed the first case but failed on a rounding boundary in the fourth case despite only approximately `2.6e-11 rad` joint-value difference. The quantized comparison was therefore superseded rather than loosened.
  - The current fixture stores the expected PC trajectory. Cross-platform validation requires identical structure and discrete values and measures every floating-point field directly with an absolute tolerance of `1e-8`; same-host repeats still require an exact resolved-trajectory hash. Resolver output is not rounded or altered.
  - Current fixture version: `order9_posture_resolver_benchmark_fixture_v4`; VIM4 fixture SHA-256 `d05f165411bf15ab23ea3512dbd0b38eaf7b4bd88843e346af1602d774ed20ef`.
- Target smoke result:
  - Case: `c3a-02-chain`, two modules, 50 resolved knots, two repeats.
  - Cold/warm-or-cache time: `2.399838 s / 0.753066 s`.
  - Exact same-host result hash: stable across repeats.
  - Preliminary PC/VIM4 quantized trajectory hash: matched; this check was subsequently replaced by the direct v4 comparison above.
  - Maximum anchor position/attitude error: `1.751981303414928e-05 m / 1.0839840187038403e-04 rad`.
  - Minimum joint-rate margin: `0.1352062163985952 rad/s`.
  - The relocated PhysicalModel content identity passed.
- Full 12-case timing:
  - Command: `/home/khadas/amsrr_vim4_benchmark/order9_run_vim4_posture_benchmark.sh`.
  - Conditions: four repeats per case, one BLAS/OpenMP thread, default `conservative` CPU governors, GNOME desktop left running, no CPU affinity or performance-governor override.
  - Cold solve mean/median/minimum/maximum: `13.592939 / 11.732337 / 2.447275 / 31.533293 s`, corresponding to `0.0736 / 0.0852 / 0.4086 / 0.0317 Hz`.
  - Exact-cache mean/median/minimum/maximum: `1.063046 / 1.069837 / 0.784405 / 1.299368 s`, corresponding to `0.9407 / 0.9347 / 1.2749 / 0.7696 Hz`.
  - The slowest cold case was the seven-module chain (`31.533293 s`); the eight-module chain/branched cases were `17.652342 / 23.496128 s`.
  - All 12 cases reproduced an exact hash across the four VIM4 repeats. Every PC/VIM4 trajectory passed the direct tolerance check; the largest measured cross-platform float difference was `1.532e-10`, approximately 65 times below the `1e-8` limit.
  - Maximum anchor position/attitude error: `2.8129503e-5 m / 1.1065284e-4 rad`. Maximum joint rate was `2.572136 rad/s`; minimum rate margin was positive at `0.127864 rad/s`.
  - Process peak RSS: `53.145 MiB`. Post-run thermal readings were approximately `47.1--48.7 C`; no thermal failure was observed.
  - Result path: `/home/khadas/amsrr_vim4_benchmark/source/artifacts/p4_full/order9/posture_resolver/vim4_benchmark.json`; SHA-256 `760473a3670148393d12ac3c0769f981e2b19d4a1be6b4d49116a9d642a34ec4`.
  - Interpretation: uncached full-horizon resolution does not meet the approved `1--2 Hz` pi_H cadence on the VIM4. Exact-cache reuse reaches about `0.94 Hz` on average, but an exact cache hit is not a substitute for solving a newly sampled pi_H trajectory.
- Runner: `/home/khadas/amsrr_vim4_benchmark/order9_run_vim4_posture_benchmark.sh`. It runs all 12 cases with four repeats and writes `source/artifacts/p4_full/order9/posture_resolver/vim4_benchmark.json`.
- Local files changed:
  - `scripts/order9_benchmark_posture_resolver.py`
  - `scripts/order9_run_vim4_posture_benchmark.sh`
  - `tests/unit/training/test_order9_posture_benchmark.py`
  - `for_codex/VIM4_COMPUTE_TEST_LOG.md`
- Tests/checks:
  - Focused portability/cross-platform comparison tests: `3 passed`.
  - Python compilation, runner `bash -n`, and `git diff --check`: passed.
  - Fresh PC preparation and all-12-case execution of the v3 fixture: passed; mean/max cold time `1.891470 s / 4.383418 s`.
- Schema/interface changes: None to production schemas, learned policies, resolver output, feasibility checking, QPID, or runtime interfaces. Changes are confined to the standalone benchmark fixture/report versions and runner.
- Repository `WORKLOG.md`: intentionally not modified for this task to avoid conflicts with another active chat.
- Status: Minimal VIM4 environment and the full 12-case timing run are complete. The measured uncached implementation requires performance optimization before it can satisfy the intended online cadence on this SBC.

## 2026-07-24: Collision-free posture-resolver fast-path experiment on PC

- User direction:
  - Optimize the collision-free resolver as far as practical using
    algorithmic, implementation, and implementation-language improvements.
  - Perform development and builds on the PC because repeated VIM4 builds are
    inconvenient.
  - Do not modify the production implementation; use a separate temporary
    workspace inside the repository.
- Isolation:
  - Workspace: `tmp/order9_posture_fastpath`.
  - No source under `amsrr/`, no production config, and no production test was
    modified for this experiment.
  - Correctness reference: immutable VIM4 v4 12-case fixture, SHA-256
    `d05f165411bf15ab23ea3512dbd0b38eaf7b4bd88843e346af1602d774ed20ef`.
- Baseline profile:
  - Worst PC case (`c3a-07-chain`) required approximately `4.7 s`.
  - Dominant cost was repeated object-based whole-structure FK inside
    finite-difference Jacobians, plus construction of a complete rigid-body
    control model merely to obtain the assembled CoM.
  - Linear solves were not a material bottleneck.
- Optimizations implemented in the temporary workspace:
  - CoM-only evaluator, with module-local CoM caching for the pure-Python
    fallback.
  - NumPy batching of all finite-difference columns for fixed-base and
    fixed-centroidal Jacobians.
  - Vectorized SO(3) finite differences.
  - C++17/Eigen native kernel for batched physical-link FK, morphology-tree
    propagation, anchor poses, and assembled CoM.
  - Cached dataclass type-hint resolution.
  - Validation-preserving ownership copies using `copy.deepcopy` instead of
    repeated dict/schema reconstruction.
  - IK limits, regularization, tolerances, iteration logic, mutation check,
    hashing, output validation, raw `pi_H` authority, and the trajectory
    contract remain unchanged.
  - Collision checks remain intentionally absent.
- Native-kernel correctness:
  - Compared C++ and independent NumPy kernels on 32 random joint vectors for
    each of all 12 morphologies, 384 samples total.
  - Compared module roots, anchor poses, and assembled CoM.
  - Maximum absolute difference: `1.110223025e-15`; threshold: `1e-12`;
    passed.
- End-to-end fixture correctness:
  - All 12 cases retained the reference solver-iteration sequences.
  - Maximum resolved-trajectory float difference:
    `1.103043817e-10`, below the cross-platform limit of `1e-8`.
  - Exact hashes differ because the native arithmetic changes floating-point
    tails; no rounding was applied.
- Final PC timing:
  - Conditions: 12 cases, 10 repetitions each, BLAS/OpenMP-related thread
    counts fixed at one.
  - Existing v4 PC cold baseline mean: `1.984976489 s`.
  - First-pass mean/median: `0.040943792 / 0.038862610 s`.
  - Steady mean/median: `0.038977350 / 0.036491647 s`.
  - Mean speedup: `48.48x` first pass and `50.93x` steady.
  - Maximum observed latency: `0.064264598 s`.
  - Peak RSS: `54.809 MiB`.
  - The seven-module chain improved from `4.770362 s` to a median of roughly
    `0.049 s` in the final ownership-copy configuration, approximately
    `97x`.
- Artifacts:
  - Result:
    `tmp/order9_posture_fastpath/cpp_pc_benchmark_final.json`.
  - Result SHA-256:
    `caf067bfc9818d67c9e6bcb50deae3fe59fb7f27062f2e1e9a69f97c0e36fcc1`.
  - Build entry point: `tmp/order9_posture_fastpath/build_cpp.sh`.
  - Kernel cross-check: `tmp/order9_posture_fastpath/validate_cpp_kernel.py`.
- VIM4 portability:
  - The generated PC `.so` is x86-64 and cannot run on the aarch64 VIM4.
  - Rebuild on VIM4 or cross-compile against its CPython 3.10 ABI.
  - A VIM4 timing run is still required; the PC result demonstrates the
    algorithm and native implementation but is not a VIM4 latency
    measurement.
- Promotion status:
  - Temporary experiment only. Production promotion requires explicit
    dependency injection instead of monkey patches, supported native
    packaging/fallbacks, production tests, and a separate decision on the
    validated `deepcopy` runtime optimization.
- Repository `WORKLOG.md`: intentionally not modified to avoid conflict with
  another active chat.

## 2026-07-24: Collision-aware C++/Eigen/FCL path measured on VIM4

- Scope:
  - Build and exercise the temporary two-level collision implementation on
    the VIM4.
  - Use lightweight convex geometry during IK and original URDF STL
    triangles for final hard admission.
  - Production source remains unchanged; work is confined to
    `tmp/order9_posture_fastpath`.
- Target and dependencies:
  - The board remained up, but DHCP changed its address from
    `192.168.96.47` to `192.168.96.50` during the build. Subsequent commands
    used the new address.
  - Installed `libfcl-dev 0.7.0-3`, `libccd-dev 2.1-2`,
    `liboctomap-dev 1.9.7+dfsg-3`, and
    `python3-scipy 1.8.0-1exp2ubuntu1`.
  - Copied all 20 original STL files from `module_urdf/mesh`.
  - Native extension:
    `fast_kinematics_ext.cpython-310-aarch64-linux-gnu.so`.
  - Extension SHA-256:
    `40aa1629602ba2da30aa22d5abb0cf08b3fc5efd4d298dd9f696881fc568e5c1`.
  - The SSH connection was lost when the DHCP address changed, so an exact
    wall-clock build duration was not recovered. The resulting extension
    timestamp is newer than the deployed source and links against FCL,
    libccd, and OctoMap.
- Functional validation:
  - The focused contact-contract test passed.
  - A selected anchor/object grasp contact was accepted.
  - The identical geometry with no allowed grasp anchors was rejected with
    two colliding pairs.
  - The representative eight-module chain (`c3a-08-chain`) solved all three
    active raw `pi_H` knots and the exact checker accepted all three.
  - Maximum anchor position/attitude errors were approximately
    `3.29e-6 m / 2.03e-5 rad`.
- Sparse-knot timing, eight modules, one BLAS/OpenMP thread:
  - Default `conservative` governor:
    - collision-aware IK mean: `0.659804 s` (`1.516 Hz`);
    - first solve: `1.572211 s` (`0.636 Hz`);
    - subsequent zero-iteration solves:
      `0.203947 / 0.203255 s` (approximately `4.91 Hz`);
    - exact STL admission mean: `2.619446 s` (`0.382 Hz`);
    - sequential IK plus exact admission from the means:
      `3.279250 s` (`0.305 Hz`);
    - peak RSS: `376.840 MiB`.
  - Temporary `performance` governor:
    - collision-aware IK mean: `0.657538 s` (`1.521 Hz`);
    - exact STL admission mean: `2.599477 s` (`0.385 Hz`);
    - sequential mean: `3.257014 s` (`0.307 Hz`);
    - peak RSS: `376.375 MiB`.
  - The performance-governor difference was below one percent. Both CPU
    policies were restored to `conservative` after the test.
- Dense path validation:
  - Checked every one of the 41 samples in the two-second, 20 Hz resolved
    trajectory for `c3a-08-chain`.
  - Convex accepted: `41/41`; exact STL accepted: `41/41`.
  - Maximum exact colliding-pair count: `0`.
  - Minimum proxy clearance: `0.017000005 m`.
  - Geometry construction: `9.502595 s`.
  - Collision checks: `122.214691 s`, or `2.980846 s/sample`
    (`0.335 Hz`).
  - Whole process wall time: approximately `133.01 s`.
  - Peak RSS: `373.980 MiB`.
- Load and thermal state:
  - Post-run memory: approximately `1.1 GiB` used, `6.0 GiB` available;
    swap unused.
  - Post-run thermal readings: approximately `45.9--47.6 C`.
  - No thermal or memory failure was observed.
- Result copies on the PC:
  - `tmp/order9_posture_fastpath/vim4_collision_aware_c3a_08_chain.json`,
    SHA-256
    `9b717ae25eca74a69b7a0661d7822f16105b149c404dcffc55d5331e7bcd7b05`.
  - `tmp/order9_posture_fastpath/vim4_collision_aware_c3a_08_chain_performance.json`,
    SHA-256
    `bec6ae644db3cfc1e8f0c38860e5fd6986bd759cc4fb6a5432c198a2de24e45b`.
  - `tmp/order9_posture_fastpath/vim4_dense_collision_path_c3a_08_chain.json`,
    SHA-256
    `47d912eb3eb4680d3dbb2855fe23a077d1fbe24489db400b22096bd3c08c1517`.
- Interpretation:
  - Lightweight collision-aware IK alone reaches the low end of the approved
    `1--2 Hz` high-level cadence on average, and warm zero-iteration knots are
    faster.
  - The current exhaustive exact-mesh checker does not meet a `1--2 Hz`
    per-candidate admission budget on the VIM4.
  - Exact validation of every 20 Hz interpolated sample is an offline
    validation operation in the current implementation, not a real-time
    control-loop operation.
  - The next performance target is the exact checker: persistent broad-phase
    managers, candidate-pair pruning, and checking only newly affected
    module/object pairs should be evaluated before production promotion.
- Repository `WORKLOG.md`: intentionally not modified to avoid conflict with
  another active chat.

## 2026-07-24: Convex-only broad-phase fast path on PC and VIM4

- Scope and method boundary:
  - The user approved removing original-STL admission from the interim online
    posture path and using the `5 mm` convex gate.
  - Production source remains unchanged; implementation is isolated under
    `tmp/order9_posture_fastpath`.
  - Added a conservative world-AABB lower-bound broad phase, per-evaluation
    collision-pose cache, persistent object shape, and morphology/allowed-set
    pair-table cache.
  - A pair is pruned only when its AABB separation already exceeds the active
    threshold. Near pairs retain the same FCL convex signed-distance query.
  - The proxy-only kernel does not load exact STL BVHs. An accidental exact
    query fails closed.
- Allowed-collision provenance:
  - Proxy hull overlap is not used to invent exclusions.
  - The six authored nominal within-module overlap pairs were generated by
    the exact-capable PC diagnostic and stored in
    `holon_nominal_collision_pairs_v1.json`.
  - The manifest binds the URDF SHA-256, all ordered collision-link IDs, and
    every ordered mesh SHA-256. Missing/mismatched input rejects
    initialization.
- PC correctness:
  - The optimized exact-capable build reproduced the prior 12-case result:
    `23/36` exact-clear after optimization, `18/36` IK-feasible, and
    `18/36` baseline exact-clear.
  - All 12 cases and 36 active knots were then compared between the
    exact-capable and proxy-only builds.
  - Maximum joint, position-error, and attitude-error differences were
    exactly zero.
  - Feasibility, solver iterations, and proxy admission matched every knot.
  - The focused allowed-contact/prohibited-contact test also passed.
- PC performance:
  - All-knot collision-aware IK mean:
    `7.303345 -> 0.338929 s` (`21.55x`).
  - Feasible-knot mean:
    `0.057417 -> 0.004007 s` (`14.33x`).
  - Failed-search mean:
    `14.549273 -> 0.673850 s` (`21.59x`).
  - Eight-module proxy-only end-to-end knot mean over 30 knots:
    `0.009688 s` (`103.22 Hz`).
  - Proxy-only peak RSS: `157.50 MiB`.
- VIM4 build and conditions:
  - Final native build wall time: `85.666 s`.
  - Extension SHA-256:
    `5941befc149376f991a5287286d559c180c7129dfb4efacc7027001e0a821332`.
  - CPU policies remained `conservative`; BLAS/OpenMP thread counts were one.
- VIM4 eight-module accepted-case result:
  - Case `c3a-08-chain`, 20 repetitions and 60 active knots.
  - IK feasible: `60/60`; proxy admission: `60/60`.
  - Mean scene setup / IK / separate proxy check:
    `0.009167 / 0.047780 / 0.009746 s`.
  - Full per-knot mean:
    `0.066694 s` (`14.99 Hz`).
  - Excluding the first pair-table construction:
    `0.061492 s` (`16.26 Hz`).
  - Zero-iteration knots:
    `0.034139 s` (`29.29 Hz`).
  - Eight-iteration knots after pair-cache warm-up:
    `0.119077 s` (`8.40 Hz`).
  - One-time proxy geometry construction: `4.352423 s`.
  - Mean FCL narrow phase: `36` of `10,747` enabled candidate pairs;
    `10,711` were safely pruned by AABB.
  - Peak RSS: `106.535 MiB`.
- Same-boundary VIM4 collision-free A/B:
  - Used the same `c3a-08-chain` raw knots, 20 repetitions, 60 solves,
    explicit solution-cache clearing, one BLAS/OpenMP thread, and the same
    native kinematics.
  - Collision-free IK-only mean:
    `0.027454 s` (`36.42 Hz`).
  - Convex IK-only mean:
    `0.047780 s` (`20.93 Hz`).
  - Convex scene update plus IK plus separate admission mean:
    `0.066694 s` (`14.99 Hz`).
  - Therefore convex collision handling is not faster than collision-free
    IK under an equal measurement boundary. The earlier `3.88 Hz`
    collision-free number measured complete resolver work rather than one
    direct knot solve and is not a like-for-like denominator.
- VIM4 collision-infeasible result:
  - Case `c3a-05-chain`, three active knots.
  - IK feasible and proxy admitted: `0/3`; all failed closed.
  - Mean full-knot time:
    `1.694100 s` (`0.590 Hz`) because the bounded unsuccessful search was
    exhausted.
  - PC/VIM4 maximum joint difference for this case:
    `9.373e-11 rad`.
- Cross-architecture agreement:
  - Accepted eight-module first-repeat maximum PC/VIM4 joint difference:
    `1.540e-11 rad`.
- Result copies on the PC:
  - `convex_only_vim4_c3a_08_chain_final.json`, SHA-256
    `f03ff130c016da652f40c53b837bfd2087c312fb69f64848de115b59734b4d01`.
  - `convex_only_vim4_c3a_05_chain_failed_final.json`, SHA-256
    `ef28270acb84c0520cfd3cd0378f9ce8a499e130a5d37a3953f8c72a306ce156`.
  - `proxy_only_equivalence_pc_final.txt`, SHA-256
    `edea1e8195e309575d6089ff3e9f303ecde043d165c0bf963ac11e72ca792ccc`.
- Interpretation:
  - The accepted trajectory path now exceeds the intended `1--2 Hz`
    high-level cadence with substantial margin on the VIM4.
  - Collision-infeasible proposals remain the latency tail. A bounded
    fail-fast budget is the next optimization before production promotion;
    lowering that budget is a method/calibration decision and was not done
    here.
  - Geometry construction remains a one-time `4.35 s` cost and is not
    included in per-knot throughput.
- Schema/interface changes: none.
- Repository `WORKLOG.md`: intentionally not modified to avoid conflict with
  another active chat.

## 2026-07-24: Collision-aware IK and exact-mesh PC experiment

- Scope:
  - User approved the two-level method: lightweight convex geometry inside
    IK and original URDF collision meshes for final hard admission.
  - Work remains isolated in `tmp/order9_posture_fastpath`; production
    schemas, interfaces, resolver, and controller code were not changed.
  - This section records PC results only. The collision-aware extension has
    not yet been built or timed on VIM4.
- Implementation:
  - Each unique binary STL is reduced to a SciPy/Qhull convex hull for the IK
    objective.
  - C++/FCL signed distances supply collision residuals to the existing
    fixed-centroidal C++/Eigen solve.
  - The proxy margin is `5 mm`; pairs within `20 mm` are activated for
    finite-difference Jacobian evaluation.
  - Original STL triangles are loaded into FCL `BVHModel<OBBRSSd>` objects for
    final intersection rejection.
  - Selected object-contact links and intended dock-interface pairs are
    excluded. Authored within-module nominal overlaps generate the standard
    allowed-collision matrix. Within-module pairs invariant to the controlled
    Dock joints are omitted from the IK cost but remain covered by the
    independent exact check unless they are authored nominal overlaps.
  - A conservative world-AABB broad phase avoids unnecessary exact BVH
    queries.
  - Current scene support covers robot self-collision and the task-declared
    box object. Generic obstacles and support-surface collision inputs remain
    future work.
- Correctness regression:
  - Collision disabled, all immutable 12 cases passed.
  - Maximum trajectory difference:
    `1.637889546e-10`, below `1e-8`.
  - Mean collision-disabled solve:
    `0.035431549 s`.
  - Direct C++ versus NumPy FK/anchor/CoM comparison:
    384 random states, maximum difference `1.110223025e-15`, passed at
    `1e-12`.
- Collision-aware 12-case PC result:
  - Active raw knots: `36`.
  - Existing baseline knots passing exact STL rejection: `18 / 36`.
  - Collision-aware IK-feasible knots: `18 / 36`.
  - Joint admission (IK feasible, convex margin, exact STL): `18 / 36`.
  - Independently exact-clear after optimization: `23 / 36`; the additional
    exact-clear knots are still rejected because IK/anchor or convex-margin
    feasibility failed.
  - Maximum exact collision count found in one baseline knot: `34`
    prohibited pairs.
  - Mean/max solve time among feasible knots:
    `0.057417 / 0.303213 s`.
  - Mean/max failed-search time:
    `14.549273 / 25.553958 s`.
  - Mean/max exact-STL check time:
    `0.499524 / 0.888153 s`.
  - Peak RSS: `421.402 MiB`.
- Dense-path check:
  - Case: `c3a-08-chain`, 8 modules, 20 Hz resolved trajectory.
  - All `41 / 41` dense samples passed the convex `5 mm` gate.
  - All `41 / 41` dense samples passed exact-STL intersection rejection.
  - Minimum convex clearance: `0.017000005 m`.
  - Maximum exact colliding-pair count: `0`.
  - Exact-plus-proxy validation time:
    `23.658921 s`, or `0.577047 s/sample`.
  - Peak RSS: `386.453 MiB`.
- Allowed-contact contract check:
  - With the two selected grasp anchors allowed, the exact state passed with
    `0` prohibited colliding pairs.
  - With the same geometry and no allowed object contacts, the exact checker
    rejected the state with `2` colliding pairs.
- Artifacts:
  - `tmp/order9_posture_fastpath/collision_aware_pc_12case_final.json`
    - SHA-256:
      `0f6eab2971486b2ffe1e8e432de341a154c6dd82070d16f7c8d351485d881dd6`
  - `tmp/order9_posture_fastpath/dense_collision_path_pc_c3a_08_chain.json`
    - SHA-256:
      `834220d08cde00ed2ef1233d40f351c9716a43dabc78c583acb89c72dce4ac15`
  - `tmp/order9_posture_fastpath/collision_disabled_regression_final.json`
    - SHA-256:
      `4411c7ea26081d5243bdda427538cf8dda2456d9e5e2507635e9e7e85324a6a9`
  - `tmp/order9_posture_fastpath/collision_contract_validation.txt`
    - SHA-256:
      `d2dd9da16fcf9d3fdc0b8980c61ff9169f73e3bc572c6fa3d8f8c332978fe607`
- Interpretation / limitations:
  - The two stages are materially different: exact STL rejects collisions
    that convex proxy status alone cannot establish, and exact-clear status
    does not override failed kinematic feasibility.
  - Collision-free candidates remain fast on PC, but exhaustive failed
    searches are far too slow and require a production fail-fast budget.
  - The exact checker is suitable as an offline/high-level hard gate at the
    measured latency, not a 20 Hz inner-loop checker.
  - VIM4 lacks a collision-aware measurement in this entry; FCL/Qhull
    availability and aarch64 timing must be verified before promotion.
- Repository `WORKLOG.md`: intentionally not modified to avoid conflict with
  another active chat.

## 2026-07-24: Complete IK loop moved to C++/Eigen

- Scope:
  - Extend the collision-free native fast path from FK/CoM/Jacobian
    evaluation to the complete deterministic IK solver.
  - Production source remains unchanged; work is confined to
    `tmp/order9_posture_fastpath`.
- Native implementation now includes:
  - initial fixed-q feasibility evaluation;
  - relaxed free-base seed loop;
  - fixed-centroidal IK loop;
  - central/one-sided finite-difference selection at joint limits;
  - position and attitude residual assembly;
  - continuity and pitch-joint regularization;
  - damped normal-equation construction;
  - Eigen partial-pivot LU with complete-orthogonal-decomposition fallback;
  - joint-step clipping, joint limits, and bounded base deltas;
  - feasible-refinement and best-solution selection.
- Unchanged behavior:
  - Raw `pi_H` authority and resolved-trajectory contract.
  - Solver configuration values, finite-difference semantics, tolerances,
    regularization, maximum iteration counts, and solution cache behavior.
  - Trajectory validation, mutation checks, and hashes.
  - Collision checking remains outside this benchmark.
- PC validation:
  - 12 cases, 10 repetitions each.
  - All solver iteration sequences matched the reference.
  - Maximum trajectory difference: `1.637889546e-10`.
  - First-pass/steady mean:
    `0.033858471 / 0.033501019 s`.
  - Speedup over the PC cold baseline:
    `58.63x / 59.25x`.
  - Native-solver PC result:
    `tmp/order9_posture_fastpath/cpp_native_solver_pc_benchmark.json`.
  - SHA-256:
    `aa2fd85693fb3413d1fcd82f128f43d422b273935d9d3076decdfc6f9087cc21`.
- VIM4 rebuild:
  - Expanded aarch64 extension build time: `61.522 s`.
  - The compiler emitted two warnings inside Eigen 3.4.0's NEON packet header
    about `memcpy` and a wrapper type; no warning originated in the
    fast-path source.
  - Extension SHA-256:
    `4968772d56be44c1925cd74bdfb47ae4823dfad88bef3a2ccbc190b4571e9d5f`.
- VIM4 timing:
  - 12 cases, 10 repetitions each, `conservative` governors, one
    BLAS/OpenMP thread.
  - First-pass mean/median:
    `0.261935807 / 0.257006014 s`.
  - Steady mean/median:
    `0.257965963 / 0.256680517 s`.
  - Original-cold-baseline speedup:
    `51.89x / 52.69x`.
  - Mean/median throughput: `3.88 / 3.90 Hz`.
  - Maximum observed latency: `0.372660278 s`, corresponding to `2.68 Hz`.
  - Speedup over the old exact-cache mean: `4.12x`.
  - Improvement over the prior C++ FK plus Python IK loop steady mean:
    `16.1%`.
  - Peak RSS: `52.949 MiB`.
  - Post-run thermal readings: approximately `46.1--47.7 C`.
- VIM4 correctness:
  - All 12 solver iteration sequences matched.
  - Maximum trajectory difference:
    `1.594587795e-10`, below `1e-8`.
  - The measured maximum latency now satisfies a strict 2 Hz deadline under
    these test conditions.
- Result:
  - VIM4:
    `/home/khadas/amsrr_vim4_benchmark/source/tmp/order9_posture_fastpath/vim4_cpp_benchmark_final.json`.
  - PC copy:
    `tmp/order9_posture_fastpath/vim4_cpp_native_solver_benchmark.json`.
  - SHA-256:
    `5aac9672b017fb2421558b20ce8e4aeeae84024aa20c583510aa0c70e69b7040`.
- Repository `WORKLOG.md`: intentionally not modified to avoid conflict with
  another active chat.

## 2026-07-24: Native C++ fast path built and measured on VIM4

- Deployment target:
  `/home/khadas/amsrr_vim4_benchmark/source/tmp/order9_posture_fastpath`.
- Production-source identity:
  - The five directly relevant resolver, reachability, kinematics,
    rigid-body-model, and trajectory-runtime source hashes were checked before
    deployment.
  - PC and VIM4 copies matched exactly.
  - Only the temporary fast-path sources and the PC baseline report were
    added to the target source tree.
- Build environment:
  - Existing compiler: `g++ 11.4.0`.
  - The first install attempt failed because the VIM4 APT index still
    referenced removed package revisions.
  - `apt-get update` completed successfully; the subsequent install
    succeeded.
  - Installed:
    - `python3.10-dev 3.10.12-1~22.04.16`
    - `libeigen3-dev 3.4.0-2ubuntu2`
    - `pybind11-dev 2.9.1-2`
  - Python 3.10 packages were upgraded from the board's prior Ubuntu security
    revision to `3.10.12-1~22.04.16`; the Python major/minor and extension ABI
    remain CPython 3.10. No reboot is required.
  - `build_cpp.sh` now accepts either `python3-config` or
    `python3.10-config`.
  - Native aarch64 build time: `24.696 s`.
  - Extension:
    `fast_kinematics_ext.cpython-310-aarch64-linux-gnu.so`.
  - Extension SHA-256:
    `ce5bb3c50ef08d09ab99fc63c9f1ca680d875baec95cc92c98a45b5f6dc47afb`.
- Native-kernel validation on VIM4:
  - 12 morphologies, 32 random joint vectors each, 384 total.
  - Compared C++ and independent NumPy module roots, anchors, and CoM.
  - Maximum difference: `8.881784197e-16`; threshold `1e-12`; passed.
- Timing conditions:
  - 12 cases, 10 repetitions each.
  - Eight CPU governors remained `conservative`.
  - BLAS/OpenMP-related thread counts fixed at one.
  - GNOME desktop left running.
  - Temperature before the first measurement was approximately
    `47.9--49.3 C`; after the final measurement it was approximately
    `41.8--43.5 C`.
- Timing result:
  - Original VIM4 cold mean/median/max:
    `13.592939 / 11.732337 / 31.533293 s`.
  - Fast-path first-pass mean/median:
    `0.310089 / 0.290343 s`.
  - Fast-path steady mean/median:
    `0.307485 / 0.285863 s`.
  - First-pass/steady speedup against the old cold mean:
    `43.84x / 44.21x`.
  - Mean/median throughput: `3.25 / 3.49 Hz`.
  - Maximum observed latency: `0.524782 s`, corresponding to `1.91 Hz`.
  - Speedup over the prior exact-cache mean: `3.46x`.
  - Per-case cold-baseline speedups ranged from `11.23x` to `80.06x`.
  - Peak RSS: `53.754 MiB`.
- Correctness:
  - All 12 solver iteration sequences matched.
  - Maximum full-trajectory float difference from the PC fixture:
    `1.099894198e-10`, below the `1e-8` threshold.
  - Exact result hashes differ across architectures due only to
    floating-point tails; no output rounding was introduced.
- Result:
  - VIM4 path:
    `/home/khadas/amsrr_vim4_benchmark/source/tmp/order9_posture_fastpath/vim4_cpp_benchmark_final.json`.
  - PC copy:
    `tmp/order9_posture_fastpath/vim4_cpp_benchmark_final.json`.
  - SHA-256:
    `c86ed6dff3e023f391e9011a39fb8530169436558272232af981bda2ff9d4432`.
- Interpretation:
  - The requested `27.2x` improvement was exceeded on VIM4.
  - Average throughput exceeds the intended `1--2 Hz` high-level replanning
    cadence.
  - The single largest observed sample was `0.5248 s`, so a strict
    worst-case `2.00 Hz` deadline is not yet guaranteed; `1 Hz` has ample
    margin.
  - Collision checking remains intentionally outside this timing.
- Repository `WORKLOG.md`: intentionally not modified to avoid conflict with
  another active chat.
