# WORKLOG.md

## Global Worklog

### 2026-07-22 (Order 9 C2 live TensorBoard telemetry)
- Active specification/work package: `A-MSRR_codex_ready_spec_v0_4_ja.md` v0.4 plus the approved Order 9 curriculum/runtime supplements; Agent J/K C2 learned `pi_L` execution and observability.
- Summary: Added default-on, live TensorBoard observation for C2 real-Isaac rollout and recurrent PPO updates. The implementation records the total reward and all eleven tensor reward terms at every control step, with current/running and global/per-phase views; it also records phase occupancy/success, task terminals and failure causes, QP feasibility, throughput, GPU/load telemetry, and every PPO minibatch's loss/entropy/KL/clip metrics. It does not alter reward values, gradients, actions, controller authority, stage gates, or artifacts used as learning evidence.
- Files changed: `amsrr/training/order9_tensorboard.py`; `amsrr/training/{order9_online_training,order9_ppo,order9_runtime_load,order9_tensor_reward}.py`; `scripts/order9_{vectorized_isaac_rollout,train_ppo}.py`; focused policy/runtime/TensorBoard tests; design supplement; this worklog.
- Schema/interface changes: None to persisted schemas, policy observations/actions, `PolicyCommand`, reward values, PPO replay, QPID/QP, safety, actuator authority, checkpoint selection, or promotion. Additive Python callback and CLI observability options only.
- Runtime behavior: Default stage root is `artifacts/p4_full/order9/stages/<stage>/tensorboard`, with `train` and `validation` sub-runs. Both rollout and PPO append to the train run using hash-lineage-derived update indices; each live write is flushed. `--tensorboard-log-dir` overrides the root and `--no-tensorboard` supports explicit diagnostics. TensorBoard `2.21.0` is already installed in `isaaclab3`.
- Tests/commands: Focused TensorBoard/PPO/runtime selection passed `35`; the dependency-complete `isaaclab3` focused set passed `39`; the full `isaaclab3` unit suite passed `1192` with `1` skip in `93.86 s`; compilation and `git diff --check` passed. The minimal base Python full suite reached `1189 passed, 3 skipped` and failed only the existing `trimesh`-dependent held-out mass-property test because that environment lacks `trimesh`; the same test passed in `isaaclab3`.
- Real integration evidence: A default-writer event-file smoke was read back through TensorBoard's `EventAccumulator`. A real-Isaac diagnostic using the promoted C1 checkpoint then completed `2 environments x 2 steps`, finite and without terminal failure, at `artifacts/p4_full/order9/runtime_diagnostics/c2_tensorboard_2x2_smoke.pt` (SHA-256 `93afe5af372429bf9a67d5ca2f42eb80590cdd91e189f550e154d86d65e174ec`). Its live run exposed `142` scalar tags and the expected reward, phase, QP, and GPU series at environment steps `2` and `4`. This CLI-overridden smoke is not C2 training data or policy-quality evidence.
- Assumptions/limitations: C2 has not started. With the configured `2048 x 16` generation, reward curves update once per control step (16 live points per rollout) and PPO curves once per optimizer minibatch. TensorBoard observation is local and does not replace immutable JSON/tensor metrics or promotion evaluation.
- Blockers/open questions: None. No method-level decision was introduced.
- Next steps: Start C2 from the promoted C1 checkpoint, run TensorBoard against the stage root, and use recorded full-generation throughput/load and learning curves before selecting C3+ parallelism.

### 2026-07-22 (Order 9 C0/C1 completed and C1 promoted)
- Active specification/work package: `A-MSRR_codex_ready_spec_v0_4_ja.md` v0.4 plus the approved Order 9 complete-`PolicyCommand`, bounded-diversity C0, actor-only C1, and articulated-graph runtime supplements; Agent J/K low-level learning and real-Isaac execution boundary.
- Summary: Completed the corrected C0/C1 lineage. C0 collected and promoted 20/20 bounded-diversity real-Isaac teacher episodes with 26,103 low-level records and no safety failure. C1 trained a 595,197-parameter phase-conditioned `pi_L` for 24 phase-balanced recurrent-window epochs, restored best epoch 22, and achieved validation actor loss `5.27308176247542e-06`. The promoted checkpoint is `artifacts/p4_full/order9/stages/c1_pi_l_bc_fixed_nominal/checkpoint.pt`, SHA-256 `533d23e921a8e047878784a523c8c93124f1ae18fbaa476704df88bee67a37ce`.
- Root-cause correction: The earlier fixed-root asset mismatch was corrected with the graph-derived articulated child-reroot asset. A second implementation mismatch was then isolated in the tensor graph observation: copied-environment origins leaked into module positions, all 28 PhysicalModel joints (including 16 fixed joints) were summarized instead of the 12 non-fixed joints stored by C0, and raw Isaac quaternion signs were not canonicalized. The controller continues to consume true world-frame state; only the policy graph view is environment-local and C0-compatible. A saved C0 row and the corrected tensor path matched across all graph features to maximum absolute error `1.1920929e-07`.
- Files changed: `amsrr/training/order9_tensor_pi_l_runtime.py`, `amsrr/training/order9_tensor_runtime.py`, `scripts/order9_vectorized_isaac_rollout.py`, `tests/unit/training/test_order9_tensor_pi_l_runtime.py`, plus the previously accumulated C0/C1 policy, decoder, dataset, curriculum, training, fixed/articulated-asset, rollout, evaluation, and test changes. `amsrr/training/order9_pipeline.py` now marks finalized promotion evaluation complete, and `scripts/order9_stage.py` resolves the repository package when invoked directly.
- Schema/interface changes: No unversioned persisted schema, `PolicyCommand`, QPID/QP, safety, or actuator-authority change. Tensor-runtime version identifiers and observation metadata were advanced for the corrected implementation. Stage promotion uses the existing typed evaluation/report/manifest interfaces.
- Upstream dependencies used: C0 dataset manifest SHA-256 `65993f666eda326a91cc2d896244f366ac0fe690b0f8578f50c4f996d88de33f`; accepted Order 8 deterministic natural-contact teacher; PhysicalModel SHA lineage; fixed-nominal articulated asset manifest v3; active-knot teacher reference; complete Order 9 command decoder; batched QPID/QP and Isaac contact runtime.
- C0 execution: 20 requested/20 successful, zero failures, two concurrent Isaac processes, stride five for high/low teacher records, wall time `11370.8688 s`, and maximum absolute unclipped normalized action `0.127413`. Dataset split is 18,263/3,900/3,940 train/validation/held-out records. C0 stage promotion is 20/20 with no fallback or safety failure.
- C1 training load: wall time `4818.1137 s`; effective record visits `110.399/s`; GPU utilization mean/peak `14.27/43%`; device VRAM mean/peak `2178.51/2534 MiB`. The superseded eight-epoch and fixed-root diagnostic checkpoints remain historical diagnostics and are not promotion evidence.
- C1 physical evidence: The corrected learned checkpoint first passed 4/4 real-Isaac closed-loop episodes. The formal copied-environment evaluation then passed 100/100 deterministic first-terminal episodes, with fallback count 0, safety failures 0, all runtime phases 0--7 unlocked, terminal step range 4564--4573 (median 4568), and mean episode return `6624.4855`. Raw rollout SHA-256 is `d27a8029ca8df33f8dc97fb809eccbfc418e0c4b879e4fc16569015b4259adf7`; evaluation JSONL SHA-256 is `cf6b5de0375bfa3350c2a76a57edddab8cc3572b7ed7f04a6b3e0b80f4f398e3`. The formal run completed 457,300 environment steps in `626.1604 s`, `746.507 env-step/s`, with GPU utilization mean/peak `58.70/68%`, VRAM mean/peak `5024.30/5918 MiB`, and temperature mean/peak `47.66/57 C`.
- Promotion/artifacts: `stage_evaluation.json` reports success rate/no-fallback success rate `1.0/1.0` and fallback rate `0.0`. `stage_promoted.json` has `status=promoted`, `promotion_evaluation_completed=true`, no failed gates, and SHA-256 `91a7090b53f8eae831f76d2dccc62620e2cad358ab95e7e891e51702db19bb46`. Generated runtime artifacts remain under ignored `artifacts/` and are hash-bound by the manifests.
- Reliability note: The first immediate restart for the 100-environment evaluation exited 139 during Isaac Kit startup in `XOpenDisplay`, before scene creation or rollout. No model/artifact was written and GPU/RAM were idle. After verifying no residual Isaac process, the explicit headless retry completed normally; this was a simulator startup incident, not a task rollout, OOM, PC crash, or learning failure.
- Tests/commands: focused Order 9/runtime/pipeline/asset tests `30 passed`; complete unit suite `1190 passed, 1 skipped` in `94.19 s`; complete acceptance suite `75 passed` in `241.73 s`; `compileall` and `git diff --check` passed. The initial full-unit invocation without repository `PYTHONPATH` collided with ROS Humble's unrelated `scripts` package and was rerun successfully with the repository root explicitly first.
- Assumptions/limitations: C1 validates fixed three-module, conservative Order8-anchor behavior only. It does not establish broader object/morphology robustness, learned `pi_H`, learned `pi_D`, C2 PPO, or P4-full completion. The 100 environments share the C1 nominal task family and differ by independent copied physics state/seed; broader curriculum variation begins downstream.
- Blockers/open questions: None at method level for starting C2. C2 remains provisionally configured for 2048 parallel environments, and its measured collection/update speed and load must be retained before selecting C3+ parallelism.
- Next steps: Start C2 fixed-morphology conservative PPO from the promoted C1 checkpoint, record per-generation throughput/load, evaluate promotion on validation episodes, and stop before any method-level curriculum or objective change.

### 2026-07-22 (Order 9 C1 corrected BC and fixed-asset kinematic blocker)
- Active specification/work package: `A-MSRR_codex_ready_spec_v0_4_ja.md` v0.4 plus the approved Order 9 curriculum/complete-`PolicyCommand` supplements; Agent J/K low-level learning and real-Isaac execution boundary.
- Summary: Corrected C1 checkpoint selection so the `pi_L` actor is selected by validation global-action plus joint-action loss rather than the numerically dominant value loss. Set the C1 value-loss weight to zero, retained eight phase-balanced recurrent-window epochs, archived the superseded checkpoint recoverably, and retrained from the current 20-episode C0 dataset. The corrected checkpoint is `artifacts/p4_full/order9/stages/c1_pi_l_bc_fixed_nominal/checkpoint.pt`, SHA-256 `96195840e0d51f47e64de6bd610fcc0817584032a37a5fcbf2c3de38f649f35d`; best epoch `7`, validation global/joint/combined actor losses `7.11918e-06 / 1.57694e-06 / 8.69612e-06`.
- Files changed: `amsrr/training/order9_offline_training.py`; `configs/training/order9_learning_curriculum.yaml`; `tests/unit/training/test_order9_offline_training.py`; `for_codex/AMSRR_design_modification_by_codex.md`; this worklog. Existing broader Order 9 C0/C1 implementation changes remain in the same uncommitted worktree and were not reverted.
- Schema/interface changes: None to persisted data, `PolicyCommand`, observation/action tensors, QPID/QP, safety, or actuator ownership. The offline trainer version and checkpoint metadata now identify the actor-only C1 selection metric.
- Upstream dependencies used: Hash-bound C0 dataset `artifacts/p4_full/order9/c0_teacher/dataset/manifest.json` (20 task-disjoint episodes, 26,103 records; 18,263/3,900/3,940 train/validation/held-out), the accepted Order 8 deterministic teacher, current PhysicalModel, and the current Order 9 fixed-nominal generated asset.
- Downstream impact: Offline C1 training is complete but C1 is not promoted. C2 and all later learned stages remain blocked because the current fixed-nominal/topology-bucket asset does not reproduce the teacher's structural Dock-joint kinematics.
- Tests/commands: `env PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/unit/training/test_order9_offline_training.py tests/unit/training/test_order9_curriculum.py tests/unit/training/test_order9_teacher_collection.py tests/unit/simulation/test_order9_fixed_nominal_asset.py` passed `11`; `git diff --check` passed. C1 CUDA training completed in `1616.962 s`; GPU utilization mean/peak was `12.20/37%`, VRAM `2021.61/2117 MiB`, temperature `39.17/50 C`, power `54.11/64 W`, and process RSS peak `1860.61 MiB`.
- Real-Isaac diagnostics: A four-copied-environment learned-checkpoint rollout ran 5,000 steps with no safety/QP failures but all environments timed out in phase 1 with zero selected contact. A separate diagnostic-only zero-normalized-action checkpoint, which sends the exact active-knot pose/twist/joint references and zero wrench/torque residuals, ran 7,500 steps and reproduced the same phase-1 timeout and zero contact. It is not promotion eligible. The second rollout took `581.305 s` after `10.702 s` setup and ended normally; no Isaac/Kit process remains. The BIOS update and user authorization permit concurrent Isaac processes for later runs, but parallelism cannot correct this deterministic mismatch.
- Root cause: C0/Order 8 represents the three modules as independent articulations joined at occupied graph Dock edges by external fixed constraints; structural Dock-joint motion therefore moves child module roots. The current Order 9 generated asset instead fixes module roots together, so the same Dock commands rotate only local Dock links. During phase 1, the teacher changes the two grasp-module centre separation from about `1.1079 m` to `1.0387 m`; the fixed asset remains about `1.0691 m`, leaving the selected links approximately `14--16 mm` too far outward per side and unable to contact the object. The exact-teacher diagnostic had maximum controller-QP residual `1.68e-05`, all QPs feasible, and zero joint-command error, excluding actor fitting and QPID as causes.
- Assumptions: The zero-action diagnostic is used only as a causal check of the current active-knot reference/decoder path. It is not evidence of learned performance or a replacement for the 100-episode C1 promotion evaluation.
- Blocker/open question: Resolving the incompatible structural representation requires a method/runtime-architecture decision: preserve Order 8 structural Dock articulation in Order 9 topology assets/runtime, or redefine/regenerate the C0 teacher for rigid-root assets. Per the user's stop rule, no implementation choice was made silently.
- Next steps: Obtain approval for the structural-representation direction. Then regenerate/revalidate affected assets, rerun a minimal exact-teacher contact diagnostic, benchmark the selected runtime (including authorized process parallelism), and only after that run the 100 learned C1 promotion episodes.

### 2026-07-21 (Order 9 C0 bounded-diversity budget and C0/C1 execution)
- Active specification/work package: v0.4 plus the user-approved Order 9 curriculum and complete-`PolicyCommand` corrections. The user approved replacing 100 near-duplicate high-fidelity C0 teacher repeats with a bounded-diversity 20-episode teacher set and retaining 100 episodes as the learned C1 evaluation gate.
- Method/config change: Advanced the curriculum to `order9_curriculum_v2`. C0/C1 keep the fixed three-module morphology but use `conservative_order8_anchor` conditions. C0 is `20` episodes (`14/3/3` train/validation/held-out): nominal, four conservative corners, and 15 deterministic seeded samples over Order8-anchored size, mass, standoff, selected-contact friction, and compliance. C0 low-level labels use stride five (10 Hz with interval-aggregated reward). C1 uses phase-balanced recurrent-window sampling; its existing 100-episode quota now belongs to learned-checkpoint evaluation rather than teacher collection.
- Runtime/reliability: The collector performs one serialized current-source asset conversion and then defaults to two concurrent Isaac processes. It is resumable only for exact condition/config/stride/profile matches. Runtime telemetry now includes GPU, process RSS, system load, and system memory for collection and BC. The user reported a BIOS update and explicitly authorized multiple Isaac processes; this does not relax fail-closed physical or data checks.
- Files changed: `configs/training/order9_learning_curriculum.yaml`; `amsrr/training/{order9_c0_curriculum,order9_curriculum,order9_dataset,order9_offline_training,order9_runtime_load,order9_teacher_collection}.py`; `amsrr/simulation/order8_isaac_runtime.py`; `scripts/order9_collect_teacher.py`; focused tests; design modification log and this worklog.
- Schema/interface changes: Added a typed C0 collection profile and a BC `phase_balanced_sampling` option; advanced curriculum, C0 collection/builder, offline trainer, and runtime-load telemetry versions. Persisted episode schemas are unchanged, but current profile/condition/stride metadata is now mandatory for C0 dataset construction and C1 replay.
- Verification/status: Focused implementation tests and compilation are in progress. The two corrected but fixed-nominal stride-one episodes from the superseded attempt must be archived before the v2 C0 run; they are not compatible training evidence. Final C0/C1 dataset, checkpoint, evaluation, promotion, timing, and load results remain pending and will be appended after execution.
- Blockers/open questions: None at method level. Any conservative C0 condition that fails physical Order8 acceptance will stop C0 rather than trigger an unapproved range change.

### 2026-07-21 (Order 9 pi_L complete-command correction and C0 rerun)
- Active specification/work package: `A-MSRR_codex_ready_spec_v0_4_ja.md` plus the approved centroidal-only QPID supplement and Order 9 curriculum decisions. This corrects C0/C1 at the learned `pi_L` boundary; it does not change QPID/QP or actuator authority and does not yet claim C0 completion.
- Root cause/correction: The initial Order 9 scaffold treated `pi_L` as a residual added to `BaselineLowLevelPolicy`, which contradicted the approved requirement that learned `pi_L` itself produce the complete bounded `PolicyCommand`. The corrected normal path uses only the active `pi_H` knot as reference/context and emits desired assembled-centroidal pose/twist, body-frame centroidal wrench bias, absolute non-vectoring joint position/velocity targets, and bounded joint torque bias. Deterministic `BaselineLowLevelPolicy` is now substitution-only fallback and is never mixed into, or credited as, a successful learned action.
- Interface/version changes: The Order 9 global actor action is now 18-dimensional: world position correction 3, body rotation-vector correction 3, mixed-frame twist correction 6, and zero-centred body-frame wrench bias 6. Local joint outputs remain source-ID/mask aware and decode to absolute targets plus torque bias. Order 9 policy, runtime, learning, checkpoint/action semantics, tensor runtime, tensor rollout, and collection versions were advanced; incompatible pre-correction C0/checkpoint/rollout artifacts are rejected rather than reinterpreted. `trust_region_blend=1.0` prevents accidental restoration of baseline-plus-residual semantics.
- Teacher correction: C0 now records the actual assembled-centroidal pose/twist reference used by the Order 8 teacher instead of the legacy base-root `CentroidalTarget`, and installs its actual absolute joint position/velocity targets into `PostureTarget`. Each completed or reused episode is decoded/encoded through the production Order 9 action contract and must have every normalized teacher action within `[-1,1]` before it can enter the dataset.
- Major files: `amsrr/policies/{morphology_conditioned_low_level_policy,order9_low_level_policy,order9_low_level_runtime,order9_policy_command,order9_tensor_command_decoder}.py`; `amsrr/training/{order9_offline_training,order9_pi_l_learning,order9_ppo,order9_teacher_collection,order9_tensor_pi_l_runtime,order9_tensor_rollout_artifact,order9_tensor_runtime}.py`; `amsrr/simulation/order8_isaac_runtime.py`; Order 9 collection/rollout/benchmark scripts; associated policy/training unit tests; the design-modification log and this worklog.
- Verification before rerun: focused correction tests passed `36`; the broader Order 9 selection passed `134` with `1040` deselected; the complete unit suite passed `1173` with `1` skip in `127.83 s`. The obsolete partial C0 directory was moved recoverably to `artifacts/p4_full/order9/c0_teacher_pre_complete_policy_command_v2_20260721`. The initial fixed-nominal 100-episode attempt was subsequently superseded by the separately approved bounded-diversity C0 budget above; none of its partial shards may be mixed into that profile.

### 2026-07-21 (Order 9 C2 provisional 2048-environment runtime)
- User decision/config: Set only `c2_pi_l_ppo_fixed_conservative` to a provisional `2048` parallel Isaac environments with `16` control steps/environment. The `32768`-sample train rollout shard therefore matches the former `128 x 256` shard size. C3 and later stages remain on the hash-bound `128`-environment production fallback until measured C2 learning behavior, memory, and end-to-end speed are reviewed.
- Implementation: Added validated stage-local PPO runtime overrides and a single resolver used by collection and stage preflight. Overrides must supply both positive environment and step counts, are forbidden outside PPO, and must produce a sample count divisible by the policy-family minibatch. Collector CLI counts are now diagnostic overrides; omitting them uses the stage-resolved production values. New-collector artifacts mark any CLI override, and the production dataset builder rejects such diagnostic shards.
- Telemetry: Added one-second GPU/load sampling to real-Isaac collection and all single/joint PPO updates. Raw rollout artifacts retain setup/rollout/combined time, rollout-only/end-to-end throughput, total device VRAM, GPU utilization, memory utilization, power, temperature, process RSS, PyTorch allocator peaks, and the sample time series. Dataset manifests retain compact per-source summaries plus aggregate collection rates. PPO metrics retain update time, consumed-step rate, and load samples. A missing GPU probe is represented explicitly and cannot fabricate measurements or change learning/control behavior.
- Real-Isaac verification: A no-runtime-override C2 smoke resolved to `2048 x 16`, wrote `32768` finite environment steps, retained the canonical phase/reset path and PhysicalModel actuator readback, and passed. Setup/rollout/combined time was `61.986/2.619/64.605 s`, giving `12513.931` rollout-only and `507.205` cold end-to-end env-step/s. Total-device VRAM peaked at `6392 MiB`; GPU utilization averaged/peaked at `13.29/61%`; power averaged/peaked at `65.83/116.12 W`; process RSS peaked at `8675.59 MiB`; PyTorch reserved memory peaked at `666 MiB`. The 21-sample ignored raw diagnostic is `artifacts/p4_full/order9/runtime_diagnostics/c2_2048x16_telemetry_smoke.pt`, SHA-256 `c55b92b5b89ea5384f49d2c1fb4d1d55034e24cde0af48f03178b57161c78c4f`. It uses an untrained diagnostic checkpoint and is not task-success/training evidence.
- Verification: All focused Order 9 unit tests passed `115`; the complete unit suite passed `1173` with `1` skip in `91.89 s`; the complete acceptance suite passed `75` in `236.82 s`; and the post-change real-Isaac C2 runtime smoke passed. Python compilation and repository whitespace checks also passed.

### 2026-07-20 (P4-full Order 9 production learning preparation complete; training not started)
- Active specification/work package: `A-MSRR_codex_ready_spec_v0_4_ja.md` v0.4 plus the user-approved Order 9 supplement in `AMSRR_design_modification_by_codex.md`. Ownership spans the Agent C/D/E/F/G/H/I/J/K/L end-to-end learning boundary. This entry completes the implementation and runtime preparation needed to start the curriculum; it does **not** claim that C0--C10 training, learned TaskSpec delivery, distributional success, or P4-full acceptance has run.
- Curriculum/policy implementation: Added validated C0--C10 stages for fixed-to-arbitrary-morphology `pi_L`, assignment then complete-trajectory `pi_H`, masked autoregressive graph-edit `pi_D`, joint object-task PPO, and held-out full-system evaluation. One phase-conditioned policy receives explicit task/adapter/phase/progress context; reward is common safety plus a registered task adapter. Dynamic assembly remains deterministic and separately evaluated except for the configured 10 percent C9/C10 end-to-end subset. `pi_H` means the learned policy only and emits a complete `ContactWrenchTrajectory`; `pi_L` emits bounded `PolicyCommand`; `pi_D` emits masked graph edits/`DesignOutput` and retains its auxiliary learned feasibility head without replacing the deterministic checker.
- Production `C_H`: Implemented the approved `hybrid_lightweight_qp_persistent_isaac_shadow` backend. The analytic layer solves a small convex per-knot contact-wrench witness QP over the policy-proposed wrench boxes, force/torque capabilities, friction/patch-moment constraints, and numeric object-wrench requirements. It neither selects contacts nor plans the task. A persistent isolated real-Isaac worker restores a copied immutable state and executes the same proposal with the current hash-bound `pi_L` plus QPID/QP, then returns controller/wrench residuals, explicit-pair collision evidence, finite-state evidence, and main-state digest preservation. Any missing/stale/mutating backend fails closed. Proposal projection remains forbidden; maximum attempts are two; rejected proposals are independent terminal GAE segments; deterministic fallback credit is never assigned to a learned policy. `warmup_proxy` remains teacher/BC-only.
- Production object-task collector: Replaced the inference scaffold with copied real PhysX environments, real contact sensors, phase-aware object-task state/reward/terminal logic, morphology-conditioned `pi_L`, batched rigid-body/QPID/thrust-QP execution, topology/object buckets, phase reset banks, and immutable GPU tensor rollout artifacts. Fixed joints collapsed by PhysX are mapped explicitly without inventing DOFs. The object link pose, true/estimated CoM separation, estimator-randomized mass/inertia/CoM, physical object readback, actor/privileged boundary, and exact policy/checkpoint/config/model/URDF/USD/simulator provenance are enforced. A post-simulation builder namespaces shards and reconstructs the existing `LowLevelControlRecord` contract; train/validation coverage is required and held-out data are rejected from PPO training.
- Actuator/runtime consistency correction: The cached fixed USD still exposed the historical generic `6.6 Nm` gimbal limit, whereas the authoritative PhysicalModel specifies XC330 `0.76 Nm` peak and a `3 rad/s` simulation-safe limit. Both the production collector and the persistent shadow worker now derive gimbal/Dock gains, armature, effort, and velocity limits from `joint_actuator_specs`, override stale USD values, and fail startup unless Isaac readback matches. Real readback is `0.75999999 Nm / 3 rad/s` for 12 gimbals and `4.09999990 Nm / 3 rad/s` for 12 Dock joints in the canonical three-module worker.
- Conservative/randomized runtime evidence: A randomized C2 train plus validation collection produced `16` exact low-level records across `4` episodes and `4` TaskSpecs; merged dataset replay passed with no failures. The C3 production bucket path selected an eight-module train topology and a disjoint three-module validation topology. The eight-module real-Isaac collector completed a four-step finite-state smoke using the generated arbitrary-morphology USD. The persistent `C_H` worker completed real synchronize -> two-knot execute -> reset with main-state preservation and PhysicalModel actuator readback. Its untrained smoke checkpoint produced residual `1.0` and would therefore be rejected by the hard gate; the RPC smoke pass is execution/integrity evidence, not policy-feasibility or task-success evidence.
- Production throughput gate: The obsolete inference-only probe is replaced by a cold-start production-collector measurement that includes real contact reduction, phase-conditioned `pi_L`, batched QPID/QP, reward/terminal handling, and GPU rollout buffering for 64 control steps at `dt=0.02 s`. Aggregate rates for 32/64/128 environments are `242.881100 / 482.572843 / 952.727310 env-step/s`; only 128 passes the configured `500` gate and is selected. `artifacts/p4_full/order9/runtime_benchmark.json` is bound at SHA-256 `a70e6264aa10464177a25157c5ce5d303130c33e4bb5f94355369430658b3d9a`. Preflight verifies the report version/config, selected count, production evidence flags, and source raw-artifact hashes. This remains a throughput result, not a learned task-success result or replacement for full-mesh Order 8 acceptance.
- Generated immutable inputs: `artifacts/p4_full/order9/morphology_pool.json` contains 80 split-safe 2--8-module entries (SHA-256 `5b6dd383212ec59c2ba5350ffa48a842d808b6893aecc266f8278cd7e3036080`). All 80 were converted to hash-bound arbitrary-morphology URDF/USD bundles; `artifacts/p4_full/order9/morphology_assets/manifest.json` has SHA-256 `186edbf6ef350b71bc747ff036a009ead5c33b754312327e71b630cc7bd57008`. The C7 teacher manifest still contains 500 complete masked design traces (400 train/50 validation/50 held out; SHA-256 `67dbf654036a80ac1548a0617a51278c9bc3beff628d375e3d8d982cbcae967a`). Canonical Order 8 remains SHA-256 `d0f75cca2ae540c79971766ab722d4530dd4fb44842276256bac40aafdb8cc49`; current C0 preflight succeeds against it and the new production benchmark.
- Pipeline/data safeguards: Added strict checkpoint metadata/loaders, causal teacher collection, BC, exact recurrent/masked PPO replay, one-generation/one-update online lineage, joint PPO orchestration, typed rejection transitions, fail-closed stage preflight/finalization, raw-evidence evaluation rows, and promotion gates derived from typed episodes rather than caller-provided aggregates. The new Order 9 sequential design-action dataset kind is isolated from the archived P4.3 v1/v2 required shard set so the enum extension does not retroactively change P4.3 acceptance.
- Major files changed: `amsrr/schemas/{datasets,feasibility,order9,policies,task_spec}.py`; `amsrr/geometry/{convex_clearance,mass_properties,wrench}.py`; `amsrr/feasibility/contact_wrench_*.py`; `amsrr/policies/order9_*.py`; `amsrr/controllers/batched_*.py`; `amsrr/simulation/order9_*.py`; `amsrr/training/order9_*.py`; C0 hooks in `amsrr/simulation/order8_{isaac_runtime,natural_contact}.py`; `configs/training/order9_learning_curriculum.yaml`; `scripts/order9_*.py`; associated unit/acceptance tests; this worklog and the design-modification log.
- Verification: Focused Order 9 selection passed `118` tests. The final complete unit suite passed `1169` with `1` skip in `95.57 s`; after the P4.3 legacy-kind compatibility correction, the complete acceptance suite passed `75` in `240.15 s`. Additional focused actuator/shadow tests passed `18`, and the real persistent-worker smoke passed with two observations and exact actuator readback. The randomized C2 dataset replay, C3 eight-module real-Isaac smoke, production benchmark build, and current C0 preflight all passed. Full `amsrr`/`scripts` Python compilation and repository `git diff --check` passed.
- Assumptions/open boundary: No method-level undefined item remains for beginning the approved curriculum. Canonical Order 8 evidence and acceptance are unchanged; raw contact remains critic/reward/safety/evaluation-only; `warmup_proxy` cannot authorize online policy execution. Generated assets/reports are reproducible ignored artifacts. This preparation does not establish learned policy quality and must not be cited as C0--C10 or P4 completion.
- Next step: Begin actual execution at C0 deterministic teacher collection and C1 fixed-morphology `pi_L` BC, then advance only through the configured C2--C10 promotion gates. Every PPO update must collect a fresh generation from its immediate parent checkpoint; do not reuse the smoke checkpoint or any benchmark rollout as training evidence.

### 2026-07-19 (P4-full Order 8 formal real-Isaac acceptance complete)
- Spec/work package/status: `A-MSRR_codex_ready_spec_v0_4_ja.md` remains the source of truth; this closes the representative deterministic Order 8 natural-contact substrate owned by Agent H/I/J/K/L. Order 9 learned full-TaskSpec delivery and Order 10 P4-full acceptance remain pending. No learned policy, statistical robustness matrix, or P4-full completion is claimed here.
- Final implementation: The normal path now completes floor-supported staging, simple previous-target-integrated all-relevant-Dock closure, non-privileged simultaneous proximity/load arrest, verified two-surface grasp, load-synchronized payload feed-forward, lift, `0.2 m` transport, place, independently bounded release back to the measured closure-start `q_open`, contact-free retreat, settle, and terminal evidence. The source settings are selected authored-Dock friction `4.5`, uniform authored-mesh compliance `7500 N/m / 75 Ns/m`, a free `1.0 kg` object, actual mesh collision, no proxy, no post-grasp torque bias, and normal fail-closed safety. Formal execution uses `dt=0.020 s` (`50 Hz`), a `150 s` rollout budget, and `30 s` per-phase timeout. Terminal evidence uses a strictly later timestamp than the final physics observation, so a no-extra-step transition to `COMPLETE` remains strictly monotonic.
- Controller/actuator contract: Simple closure preserves the user-approved `q_target[k+1]=clip(q_target[k]+qdot_command*dt)` rule and physical joint limits; an experimental ordinary-IK target-lead clamp was removed after it prevented second-side load acquisition. The report distinguishes finite pre-saturation implicit-drive torque from physically applied actuator torque. Acceptance gates the measured/applied AK40-10 envelope, with exact zero envelope violations; the computed pre-saturation value is diagnostic only. Raw Isaac contact remains privileged validation/safety evidence and is not a normal actor, planner, or QPID input.
- Canonical evidence: `artifacts/p4_full/order8_natural_contact/order8_mu4p5_dt20ms_full_v406.json`, SHA-256 `d0f75cca2ae540c79971766ab722d4530dd4fb44842276256bac40aafdb8cc49`. The wrapper reports `passed=true`, `monitor_passed=true`, and `report_validation_failures=[]`. Ordered phases are `reset, approach, contact_acquisition, lift, transport, place, release, retreat, settle, complete`; duration is `128.9 s` (`6445` evidence steps). Maximum grasp-reference displacement is `19.406/15.461 mm` (`30 mm` limit), selected-contact force `10.046 N` (`30 N` limit), penetration `0.719 mm` (`2 mm` limit), unintended-contact count `0`, and object-drop false. The telemetry-only tangential slip-speed peak is `47.665 mm/s`; per the user-approved v12 contract it is not a dwell, safe-hold, or acceptance gate. Release-contact-free, retreat-clearance, and settle gates all pass.
- Actuator evidence: Maximum requested closure/release velocity command is `0.02315 rad/s`; maximum measured Dock speed is `0.61954 rad/s` (`3 rad/s` limit); maximum applied torque/current are `4.100 Nm / 7.300 A` (configured hard limits), with zero audit violations. Maximum finite computed implicit-drive torque before saturation is `8.947 Nm` and maximum position-target lead is `0.04266 rad`; neither is mislabeled as delivered actuator torque.
- Independent acceptance: `run_order8_acceptance(...)` reloaded v406 and returned `artifact_loaded=true`, `real_isaac_passed=true`, `evidence_integrity=true`, `no_mislabeling=true`, `completion=true`, an empty failure list, and all acceptance metrics `1.0`. The live GUI path uses the same formal runtime: `micromamba run -n isaaclab3 python scripts/order8_natural_contact.py --real --viewer kit --realtime-playback --keep-open-after-rollout-s 20 --seed 0 --report-path artifacts/p4_full/order8_natural_contact/order8_gui_report.json`. GUI inspection remains diagnostic; the headless hash-bound artifact is the acceptance evidence.
- Verification/commits: The final focused Order 8 unit/acceptance suite passed `290` tests. The repository-wide command `PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/unit tests/acceptance` then passed `1079` with `3` skips in `374.89 s`; Python compilation and `git diff --check` passed. Source is split into `a228b3a` (`[P4-full][Order8] Add natural-contact control core`), `51baf27` (`[P4-full][Order8] Integrate Isaac natural-contact runtime`), and `c053ede` (`[P4-full][Order8] Add natural-contact acceptance gate`); this documentation closeout is committed separately.

### 2026-07-19 (P4-full Order 8 selected-Dock friction set to 4.5)
- User decision/config change: Promote the representative Order 8 selected authored-Dock static/dynamic friction from `2.0` to `4.5` under the existing PhysX `max` combine mode. The object remains `mu=0.6`, the floor remains `mu=0.8`, and neither material is changed. `Order8NaturalContactConfig` and `configs/training/order8_natural_contact.yaml` now agree on `4.5`; the frozen-config unit assertion was updated. This changes a configured physical value and its config hash, not the persisted field shape, so the `v12` schema identifier is retained.
- Basis/scope: The proxy-free sweep measured the retained-lift transition inside `3.5625 < mu <= 3.625`; `4.5` is approximately `24%` above the lowest verified coefficient. This is deliberate margin for the representative smoke, not a universal friction or robustness claim. No proxy, contact/slip threshold change, object constraint, raw-contact control input, actuator-limit change, or direct object force/pose write was introduced.
- Order 8 completion boundary: Friction promotion alone does not yet establish acceptance. Every authored-mesh lift pass used the acceptance-ineligible `+/-0.5 Nm` post-grasp diagnostic bias and `7000/75` compliant-contact override, while the persisted path still uses `7500/75` and zero post-grasp torque bias. Before the full floor-initialized run can count, the accepted non-diagnostic command path must be verified under the persisted settings (or the already approved policy joint-torque-bias channel must be separately promoted/versioned), then complete `grasp -> lift -> transport -> place -> release -> retreat -> settle` with empty monitor and wrapper failures. No such full acceptance pass is claimed by this config-only increment.

### 2026-07-19 (P4-full Order 8 authored-mesh friction boundary sweep)
- Scope/authorization: With the successful v381 no-pad setup as the reference, vary only the selected authored-Dock static/dynamic friction coefficient and find the lift boundary. Every run retained the free `1.0 kg`, `0.30 x 0.40 x 0.15 m` object (`mu=0.6`), actual authored Dock meshes, `max` friction combine mode, `7000 N/m / 75 Ns/m` compliant contact, `0.005 rad/s` diagnostic closure, `+/-0.5 Nm` four-yaw post-grasp bias, `30 mm` grasp-reference displacement gate, disabled slip-speed gate, and smooth lift-bias removal. No proxy, object constraint, direct object force/pose write, production default change, or acceptance relaxation was introduced.
- Boundary method/result: A lift was counted as acquired at the unchanged `100 mm` object-bottom-clearance gate; a **retained lift** additionally had to avoid clearance-loss/drop through the `45 s` diagnostic ceiling. `mu=2.0` did not lift off and failed the `30 mm` displacement gate; `mu=3.0` lifted off but reached only `16.681 mm` before the same gate. `mu=3.5` reached `100.301 mm` at `42.90 s` but lost clearance after `0.14 s`; `mu=3.5625` reached `101.894 mm` at `43.70 s` but lost clearance after `0.54 s`. `mu=3.625` reached the lift gate at `39.04 s`, reached `119.429 mm`, and remained drop-free through `45 s`; `mu=3.75` likewise passed at `38.08 s`, reached `122.178 mm`, and remained drop-free. The sampled retained-lift boundary is therefore `3.5625 < mu <= 3.625`, and the lowest verified coefficient is `mu=3.625`. This is an empirical bracket for this exact deterministic diagnostic, not a universal Coulomb-friction threshold; contact dynamics are nonlinear and no monotonicity proof or multi-seed robustness claim is made.
- Safety/contact evidence: At the lowest retained-lift pass (`mu=3.625`), maximum per-link grasp-reference displacement was `0.442/0.644 mm`, selected-contact force was `9.960 N`, penetration was `1.205 mm`, and instantaneous slip-speed telemetry peaked at `18.151 mm/s` without gating. There was no object drop and the monitor failure-reason list was empty. Its top-level `order8_monitor_gate_failed` only means the `45 s` ceiling ended during `TRANSPORT`, before transport/place/release/retreat/settle completion; it is not a lift failure and is not Order 8/P4-full acceptance.
- Artifacts/commands: The bracket is backed by `authored_mesh_mu2_contact_point_slip_v382_40s.json`, `authored_mesh_mu3_contact_point_slip_v384_40s.json`, `authored_mesh_mu3p5_contact_point_slip_v386_45s.json`, `authored_mesh_mu3p5625_contact_point_slip_v389_45s.json`, `authored_mesh_mu3p625_contact_point_slip_v388_45s.json`, `authored_mesh_mu3p75_contact_point_slip_v387_45s.json`, and `authored_mesh_mu4_contact_point_slip_v383_40s.json` under `artifacts/p4_full/order8_natural_contact/diagnostics/`. Each used `scripts/order8_contact_force_diagnostic.py` with only `--selected-gripper-friction` and the `40/45 s` ceiling differing. The success-side `mu=3.625` setting remains a diagnostic override; production/config friction was not silently promoted.

### 2026-07-19 (P4-full Order 8 authored-mesh no-pad lift A/B)
- Scope/authorization: Remove only the cone proxy from the v380 diagnostic and repeat the same free `1.0 kg` object, selected-surface friction `10`, compliant contact `7000 N/m / 75 Ns/m`, `0.005 rad/s` diagnostic closure, `+/-0.5 Nm` four-yaw post-grasp bias, `30 mm` grasp-reference displacement limit, smooth lift-bias removal, and `40 s` ceiling. Slip speed remained telemetry-only. No production default or acceptance gate was changed.
- Real-Isaac result: `artifacts/p4_full/order8_natural_contact/diagnostics/authored_mesh_contact_point_slip_smooth_bias_v381_40s.json` ran `40.000 s` in `164.414 s` wall time, with trace `authored_mesh_contact_point_slip_smooth_bias_v381_40s_state_trace.json`. Runtime evidence explicitly reports `proxy_pad_enabled=false`, authored collision enabled/retained, and `selected_surface_actual_dock_mesh=true`. Grasp completed and `LIFT` began at `26.12 s`; physical support separation was confirmed at `28.36 s`; smooth bias removal completed at `28.88 s`; the `100 mm` lift gate passed at `35.20 s`; and the planner entered `TRANSPORT`. Maximum/final object-bottom clearance was `115.439 mm`.
- Contact/safety evidence: Per-link maximum grasp-reference displacement was `0.468/1.309 mm` against `30 mm`. Maintained-contact slip-speed telemetry peaked at `16.535 mm/s` but was non-gating as specified. Maximum selected-contact force/torque/penetration was `7.334 N / 0.223 Nm / 1.109 mm`; both selected authored meshes remained in contact, and there were zero monitor failure reasons, object drop, unintended contacts, QP infeasibility, controller failure, actuator-envelope violation, joint-limit violation, or robot-environment unsafe contact. The lift therefore succeeds without a proxy under these otherwise unchanged diagnostic conditions.
- Comparison/boundary: Relative to cone-pad v380, no-pad grasp and lift-gate completion were about `1.80/1.76 s` later, maximum clearance was `4.90 mm` lower, and the worse selected-link displacement increased from `0.493` to `1.309 mm`, still only `4.4%` of the `30 mm` limit. `monitor_passed=false` and `order8_monitor_gate_failed` only reflect the `40 s` ceiling ending during `TRANSPORT` (`66.19 mm` of the required `200 mm`), before place/release/retreat/settle; the monitor's `failure_reasons` list is empty. This A/B removes the proxy from the successful lift result, but friction `10`, `7000/75` compliance, `+/-0.5 Nm` diagnostic bias, and the incomplete downstream phases still prevent Order 8/P4-full acceptance.

### 2026-07-19 (P4-full Order 8 grasp-referenced slip contract and smooth lift-bias removal)
- Scope/authorization: Replace path-integrated slip with the user-defined grasp-reference metric, disable slip-speed gating, remove the lift inertial bias with a ramp, and determine whether the existing `1.0 kg` cone-pad diagnostic can clear the `100 mm` lift gate. The cone proxy, friction `10`, `7000/75` compliance, and `+/-0.5 Nm` four-yaw bias remain explicit acceptance-ineligible diagnostic conditions; they are not production promotion.
- Contract/implementation: Advanced the Order 8 config/observation/step/result contracts to `v12/v4/v4/v3`. At the exact verified-grasp transition, each selected Dock link now latches its force-weighted raw-contact centre in the measured object frame. Maintained-contact slip is the current 3D Euclidean displacement from that immutable reference, with a `30 mm` limit; the monitor stores current and per-link maximum values. Instantaneous tangential slip speed remains reported but cannot fail contact dwell, request safe hold, or fail acceptance. The former absolute-speed integral is retained only under explicitly diagnostic telemetry names and is not a safety metric. Lift-bias removal retains the configured `0.5 s` duration but now uses cubic smoothstep (`3p^2-2p^3`) so both ramp endpoints have zero slope. Report/acceptance method identities and the diagnostic CLI override were versioned accordingly; the old `--max-cumulative-slip-m` spelling is only a deprecated CLI alias.
- Regression verification: `PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/unit` passed `1004`, with `3` skips. Focused Order 8 tests passed `275`; they prove that arbitrarily high slip-speed telemetry is non-gating, `31 mm` grasp-reference displacement fails, the reference is not re-latched after grasp, and the lift-bias smoothstep has the intended intermediate/end values. `compileall` and `git diff --check` passed.
- Real-Isaac result: `artifacts/p4_full/order8_natural_contact/diagnostics/cone_proxy_contact_point_slip_smooth_bias_v380_40s.json` ran `40.000 s` in `196.012 s` wall time, with trace `cone_proxy_contact_point_slip_smooth_bias_v380_40s_state_trace.json`. Grasp completed and `LIFT` began at `24.32 s`; support separation was confirmed at `26.58 s`; smooth bias removal completed at `27.10 s`; the `100 mm` lift gate passed at `33.44 s` with `100.303 mm` clearance and the planner entered `TRANSPORT`. Maximum/final clearance was `120.343 mm`. Per-link maximum grasp-reference displacement was only `0.466/0.493 mm` against `30 mm`, although instantaneous maintained-contact speed telemetry peaked at `21.176 mm/s`; no safe-hold transition/request occurred. Force/torque/penetration maxima were `5.215 N / 0.149 Nm / 0.985 mm`, with zero QP/controller/actuator-envelope/environment-contact failure.
- Interpretation/handoff: The requested lift gate is verified under the stated diagnostic conditions. `monitor_passed=false` is expected because the deliberately short `40 s` run ended during `TRANSPORT`, before the `200 mm` transport, place, release, retreat, and settle gates; `failure_reasons` is empty and this is not a lift failure. Do not interpret the historical v379 velocity-integral paths (`259/297 mm`) as current contact-point slip. Next Order 8 work may resume from downstream transport/place/release/settle and then the unchanged complete authored-mesh environment; cone-pad v380 is not Order 8 or P4-full acceptance.

### 2026-07-18 (P4-full Order 8 cone-pad +/-0.5 Nm 50 s no-safe-hold diagnostic)
- Scope/authorization: Repeat the v378 full-duration diagnostic with every condition fixed except restoring the four grasp-contributing yaw-Dock closure biases from `+/-1.0` to `+/-0.5 Nm`. Run exactly `50.0 s`, retain all monitor evidence, and prevent every `SAFE_HOLD` request from changing the physical rollout. Production defaults, source config, normal safety behavior, and acceptance remain unchanged.
- Physical result/artifacts: `artifacts/p4_full/order8_natural_contact/diagnostics/cone_proxy_pad_tau0p5_all_safehold_v379_50s.json` completed `2500 x 0.02 s = 50.000 s` in `244.669 s` wall time. The measured grasp entered `LIFT` at `24.36 s`, physical support separation was confirmed at `26.58 s`, the object cleared the required `100 mm` lift at about `33.44 s`, `TRANSPORT` began at `33.48 s`, and the object moved `200.20 mm` during transport before `PLACE` began at `47.08 s`. Maximum object-COM rise was `141.82 mm`; the run ended during place with `104.88 mm` COM rise, so release/retreat/settle were outside the 50 s budget.
- Failure interpretation: This is not an acceptance pass. Maximum provisional/maintained slip speeds were `30.51/21.18 mm/s`, and cumulative paths reached `259.21/296.85 mm`, far above the diagnostic `30 mm` bound. The first suppressed request occurred at `26.74 s`; `1163` safe-hold requests were recorded but none acted. The monitor's `required_contact_break` and `object_drop` latches follow from the cumulative-slip-invalidated required-contact gate, not literal disappearance of raw contacts: both selected links were still reported throughout lift/transport/place. Grasp/lift/transport were acquired; release-contact-free, retreat, and settle were not.
- Contact/actuator evidence: Peak selected force/contact torque/penetration were `5.215 N / 0.149 Nm / 0.985 mm`, with zero unintended contact. The requested/applied bias remained exactly `0.5 Nm`; combined position-drive/effort torque peaked at `2.524 Nm`, estimated current at `4.494 A`, and Dock speed at `0.164 rad/s`. These are below the AK40-10 hard audit but the combined torque is still above the `1.3 Nm` continuous rating, and the severe accumulated slip prevents promotion to production or acceptance.
- Verification/GUI: The expanded Order 8 suite passed `354`; compilation passed. Added `scripts/order8_tau0p5_all_safehold_replay_gui.py`, bound to the hash-validated `1251`-frame v379 trace. A real balanced Kit replay exited normally, displayed all `150` cone pads, reproduced `5.43 deg` Dock motion, and reported zero joint/object write error. User command: `python3 scripts/order8_tau0p5_all_safehold_replay_gui.py`. This cached low-load replay is visual-only; it does not recompute contact physics.

### 2026-07-18 (P4-full Order 8 cone-pad +/-1 Nm 50 s no-safe-hold diagnostic)
- Scope/authorization: Keep the v375 cone-pad `1.0 kg` diagnostic conditions fixed, change only the four grasp-contributing yaw-Dock closure biases from `+/-0.5` to `+/-1.0 Nm`, run exactly `50.0 s`, and prevent every safe-hold from stopping the diagnostic. Production, source config, normal safety behavior, and acceptance remain unchanged.
- Implementation/files: Added the default-off diagnostic flags `--disable-all-safe-hold` / `--order8-diagnostic-disable-all-safe-hold` in `scripts/order8_contact_force_diagnostic.py`, `scripts/p4_control_holon_spawn_probe.py`, and `amsrr/simulation/order8_isaac_runtime.py`. The mode retains all raw evidence/monitor calculations, records supervisor requests instead of executing them, extends planner phase timeouts beyond the requested step budget, and does not terminate early on `COMPLETE`; an unexpected bypass into `SAFE_HOLD` fails loudly. Added `scripts/order8_tau1_all_safehold_replay_gui.py` as a cached low-load replay of this exact run. No persisted schema or production default changed.
- Physical result/artifacts: `artifacts/p4_full/order8_natural_contact/diagnostics/cone_proxy_pad_tau1_all_safehold_v378_50s.json` completed `2500 x 0.02 s = 50.000 s` in `234.150 s` wall time. q_close and load-limited positional preload completed, and the signed bias was active for `1312` steps (`26.24 s`) with the exact last map `+1/+1 Nm` on module 1 yaw joints and `-1/-1 Nm` on module 2 yaw joints. The larger bias caused sustained articulated motion: maximum acquisition-relative speed was `0.16316 m/s`, final per-anchor speeds were `0.01151/0.11270 m/s`, and the required `0.01 m/s` pre-LIFT dwell never completed. Final phase remained `contact_acquisition`; grasp/lift/lift-off were false and object COM rise was only `0.0037 mm` maximum. This is a gate failure, not a safe-hold termination.
- Safety/numerics: No `SAFE_HOLD` appeared in the phase trace, no direct supervisor safe-hold request occurred, and the nominal `35 s` contact-acquisition timeout was deliberately bypassed by the diagnostic timeout extension. Peak selected force/contact torque/penetration were `5.215 N / 0.149 Nm / 0.985 mm`; no raw-contact invalidity/saturation, drop, unintended contact, robot-floor/support contact, QP infeasibility, controller failure, or joint-limit violation occurred. The requested effort bias was exactly `1.0 Nm`, but its position-drive combination reached `3.278 Nm`, `5.836 A` estimated current, and `0.425 rad/s`. This stays inside the AK40-10 hard `4.1 Nm / 7.3 A / 3 rad/s` audit but exceeds the `1.3 Nm` continuous torque rating, so the long bias is not production-viable evidence.
- Verification/GUI: Focused compilation and `235` tests passed. The hash-validated state trace `cone_proxy_pad_tau1_all_safehold_v378_50s_state_trace.json` contains `1251` frames over `50.0 s`. A real balanced Kit replay at `50x` exited normally, authored `150` pad visuals, reproduced `5.55 deg` independent PhysX Dock motion, and reported zero joint/object write error. User command: `python3 scripts/order8_tau1_all_safehold_replay_gui.py` (cached `0.5x` replay; no contact physics is recomputed). Conclusion: `+/-1 Nm` is worse than v375's `+/-0.5 Nm` because it prevents stable grasp-dwell entry rather than improving lift.

### 2026-07-18 (P4-full Order 8 cone-pad lift low-load GUI replay)
- Spec/work package: v0.4 plus the approved Order 8 diagnostic visualization boundary; Agent H/I/J/K/L. The complete Order 8 worktree remains uncommitted.
- Requested outcome: Let the user inspect the v375 cone-pad physical-lift behavior in Kit without combining the expensive contact simulation and RTX viewer, while avoiding the previous static-joint, black-viewport, and warning-flood failure modes.
- Implementation/files: Added `scripts/order8_cone_proxy_lift_replay_gui.py` and `tests/unit/simulation/test_order8_cone_proxy_lift_replay_gui.py`; extended the probe/runtime so cone-pad geometry may be authored during state replay only when `--order8-state-trace-replay-sync-physics` is also active. The wrapper reuses a cached exact physical trace, defaults to one `0.5x` loop, balanced rendering, the existing replay Dome Light, and errors-only child output. It displays all `150` cone pads but disables object/environment collision, gravity, self-collision, and graph constraints during the single-step synchronization; each recorded state is reapplied exactly after that step. Routine Kit startup output is filtered while phase/time/independent PhysX Dock motion and abnormal-exit logs remain visible.
- Physical trace: `artifacts/p4_full/order8_natural_contact/diagnostics/cone_proxy_pad_effective_gate_slowclose005_v375_visual_state_trace.json` was captured headlessly from the exact v375 `probe_command`. It contains `670` hash-validated frames over `26.74 s`, all three module roots, all joints, and the free-object state. Recorded maximum Dock-joint motion is `0.094722 rad` (`5.43 deg`) and maximum object COM rise is `11.66 mm`.
- Verification: The focused runtime/wrapper/geometry selection passed `244`; focused compilation passed. Two real balanced-renderer Kit replays exited normally. Independent PhysX DOF readback reproduced the full `5.43 deg` Dock motion with `0 rad` error, object/root write errors were zero, and the report audited `150` cone visual prims with object collision disabled. A desktop capture of the held final replay showed normally rendered Holon meshes, the blue object, support, floor, and lighting rather than a black viewport. No Isaac/Kit process remains.
- Interface/boundary: No persisted schema, policy, QPID, production contact, or acceptance interface changed. One diagnostic CLI combination is newly allowed: cone proxy plus state replay only in contact-minimized PhysX-sync mode. The replay is a visual reproduction of already captured physics, not a rerun of contact, force, slip, or lift dynamics and cannot be used as Order 8 evidence.
- User command/next step: Run `python3 scripts/order8_cone_proxy_lift_replay_gui.py`. The cached trace means no physical recapture occurs. Use `--speed 0.75` or `--speed 1.0` only if a faster display is preferred; use `--normal-kit-log` only for troubleshooting. User visual review of grasp, lift, tilt, and slip is the next step.

### 2026-07-18 (P4-full Order 8 cone micro-pad physical lift diagnostic)
- Scope/authorization: After visual approval of the cone-only merged micro-pad placement, connect exactly that fixed link-local geometry to the acceptance-ineligible Order 8 diagnostic runtime and determine whether it can physically lift the free `1.0 kg` object, not merely establish two contacts. Production contact and P4-full acceptance remain unchanged.
- Implementation/files: `amsrr/simulation/order8_isaac_runtime.py`, `scripts/p4_control_holon_spawn_probe.py`, and `scripts/order8_contact_force_diagnostic.py` now support a default-off cone-proxy runtime path. It authors `75` thin boxes on each selected yaw-Dock cone (`150` total) under the existing rigid links, assigns the selected compliant/high-friction diagnostic material, and disables only the corresponding authored collision descendants to avoid double contact while retaining their visuals. The proxy creates no independent rigid body and is explicitly reported as `selected_surface_actual_dock_mesh=false`. A diagnostic-only positive closure-speed override was added so the already approved integrated-target velocity closure can be slowed without changing production. The q_close geometric gate now samples the active proxy boxes rather than the disabled authored mesh; raw Isaac contact remains privileged diagnostic/safety evidence and does not enter the policy or QPID command.
- Verification: Focused Order 8 runtime/wrapper/geometry tests passed `234`; focused compilation passed. Spawn audit `cone_proxy_pad_spawn_audit_v370_0p1s.json` proved all `150` colliders were attached to the intended rigid links, selected authored collision was disabled, and no independent rigid body was introduced. Earlier v371/v374 runs exposed and isolated the stale authored-mesh q_close gate; the corrected effective-surface gate is covered by unit tests and v375 physical evidence.
- Best physical result: `artifacts/p4_full/order8_natural_contact/diagnostics/cone_proxy_pad_effective_gate_slowclose005_v375_30s.json` used the unchanged free `1.0 kg` object, `0.005 rad/s` diagnostic closure, selected friction `10`, compliant contact `7000 N/m / 75 Ns/m`, existing `Kp/Kd=200/5`, and `0.5 Nm` post-grasp closure bias. It acquired two-contact grasp, entered `LIFT`, and confirmed physical support separation at `26.58 s`. Object COM rose `11.663 mm`; the tilted object OBB achieved `3.009 mm` bottom clearance. Peak selected-contact force was `5.215 N`, penetration `0.985 mm`, there was no drop or unintended contact, and the object was never directly written, forced, or constrained.
- Remaining failure/boundary: v375 did not pass the `100 mm` lift gate. It safe-held at `26.76 s` when cumulative selected-contact slip reached `18.194/30.233 mm`; instantaneous slip-speed stopping was disabled only for this diagnostic and the diagnostic cumulative bound was `30 mm`. A slower `2.5 s` payload-transfer A/B (`cone_proxy_pad_slowtransfer2p5_v377_35s.json`) was worse: no confirmed lift-off, only `0.402 mm` maximum OBB clearance, and the same cumulative-slip stop at `19.157/30.113 mm`. Therefore the cone pad makes the grasp mechanically capable of lifting 1 kg, but does not yet maintain the grasp through the required 100 mm lift and is not Order 8/P4-full acceptance evidence. Do not promote the proxy, friction `10`, torque bias, or relaxed slip diagnostics into production implicitly. The next step is a separately approved contact-retention method/tuning decision, followed by the shortest q_close-to-lift A/B before transport/place/release/settle.

### 2026-07-18 (P4-full Order 8 cone-only merged micro-pad preview)
- Scope/user feedback: The surface-following v2 placement was directionally correct, but pads on the non-conical cylindrical/mounting structure were unnecessary. The user also requested merging adjacent pads without materially changing the conical coverage. No contact/lift rollout was authorized.
- Implementation: Preview v3 restricts eligible authored STL triangles to the physical cone (`link-local x=0.068--0.1158 m`, outward-normal axial component `0.30--0.85`). It replaces the former `12 x 24` whole-side grid with a cone-only `4 x 20` grid, which merges neighbouring locally similar regions while fitting every resulting pad to its own outer triangle patch. The canonical result is `75` pads per yaw Dock (`150` total, down from `474`, approximately `68%` fewer). Pads are approximately `8.15--18.67 mm` tangentially, remain `0.8 mm` thick, and retain the `0.2 mm` mesh clearance. The right-side non-conical geometry receives no pad.
- Verification: Surface points remain within the configured cone range; fitted normal axial components are `0.418--0.716`; the local fit gap improves to `0.951 mm` at the 95th percentile and `1.406 mm` maximum. The focused selection passed `14` tests; compilation and `git diff --check` passed. A real single-module Isaac stage smoke exited `0` in about `5.1 s`, authored all `150` colliders under the two existing rigid links, and took zero post-reset physics steps.
- Boundary/handoff: Preview remains `acceptance_eligible=false` and `contact_runtime_enabled=false`. Run `python3 scripts/order8_side_proxy_pad_gui.py` to approve the cone-only geometry. Do not run proxy contact physics or promote this shape set until that visual approval is explicit.

### 2026-07-18 (P4-full Order 8 collision-following micro-pad preview correction)
- Scope/problem: The user rejected the first full-side preview because each `135--139 x 106 x 2 mm` plate was much too large, its plane was visibly rotated away from the local collision surface, and one plane could cover only one projection rather than the complete Dock side. No contact/lift run was authorized.
- Implementation/files: `order8_side_proxy_pad_preview_v2` replaces each full plate with an axial-by-circumferential tiling derived directly from the real URDF collision STL triangles. For every occupied `12 x 24` surface cell, the builder selects the outer local triangle patch, fits an outward normal, and authors one link-local box with that individual orientation. The canonical Holon mesh produces `237` pads per yaw Dock (`474` total); tangential dimensions are `6--16 mm`, thickness is `0.8 mm`, and the inner face begins `0.2 mm` beyond the fitted mesh surface. Pads span all existing lateral-surface bands and all 24 circumferential sectors; empty cells on physically open portions of the authored mesh are not artificially filled. The single-module viewer and focused tests were updated. The ordinary Order 8 runtime remains disconnected.
- Numerical/Isaac verification: Local surface-fit gap is `1.316 mm` at the 95th percentile and `2.277 mm` maximum; fitted normals span both signs of link-local Y and Z rather than one hard-coded plane. `PYTHONPATH= PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests/unit/simulation/test_order8_side_proxy_pad.py tests/unit/simulation/test_order8_side_proxy_pad_gui.py tests/unit/simulation/test_order8_proxy_pad_gui.py` passed `14`. Compilation passed. `python3 scripts/order8_side_proxy_pad_gui.py --headless --keep-open-s 0.2` exited `0` in about `4.8 s`, authored all `474` cube colliders under the two existing yaw-Dock rigid bodies, and performed zero post-reset physics steps.
- Boundary/handoff: This remains placement-only, `acceptance_eligible=false`, and `contact_runtime_enabled=false`. Run `python3 scripts/order8_side_proxy_pad_gui.py` (optionally `--focus-link yaw_dock_mech1` or `yaw_dock_mech2`) and visually inspect the orange micro-tiles. Do not connect them to the contact runtime or claim natural-contact evidence until the user approves this revised placement.

### 2026-07-18 (P4-full Order 8 full-side proxy-pad placement preview)
- Spec/work package: v0.4 plus the Order 8 natural-contact diagnostic supplements; Agent H/I/J/K/L contact-geometry preview. The complete pre-existing Order 8 worktree remains uncommitted.
- User-approved scope: Replace the old connect-frame-tip placement for the next proxy experiment with a fixed pad covering the complete grasp-facing side of each selected yaw Dock mechanism. Before any contact/lift simulation, expose the exact geometry in the smallest useful GUI scene so the user can approve or reject placement.
- Implementation/files: Added `configs/training/order8_side_proxy_pad_preview.yaml`, pure mesh/URDF geometry in `amsrr/simulation/order8_side_proxy_pad.py`, single-module render-only launcher `scripts/order8_side_proxy_pad_gui.py`, and focused tests. Each plate is fixed in its Dock-link frame, uses no object pose at runtime, covers the complete orthographic projection of the authored yaw-Dock collision mesh with a `3 mm` tangential margin, is `2 mm` thick, and begins `1 mm` outside the mesh support plane. Resulting plates are approximately `135.899 x 106.133 x 2 mm` on `yaw_dock_mech1` and `139.049 x 106.111 x 2 mm` on `yaw_dock_mech2`. They are translucent orange collision-enabled cube children of the existing rigid links, with no independent rigid body.
- Interface/schema boundary: The preview config is fail-closed with `acceptance_eligible=false` and `contact_runtime_enabled=false`. The ordinary Order 8 rollout, production contact representation, QPID, joint controller, object, and acceptance validator are unchanged. Visual approval is required before connecting this builder to the contact runtime.
- Verification/commands: `PYTHONPATH= PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests/unit/simulation/test_order8_side_proxy_pad.py tests/unit/simulation/test_order8_side_proxy_pad_gui.py tests/unit/simulation/test_order8_proxy_pad_gui.py` passed `14`; `py_compile` and `git diff --check` passed. A `python3 scripts/order8_side_proxy_pad_gui.py --headless --keep-open-s 0.2` single-module stage smoke exited `0`, resolved one rigid-body child collider under each yaw Dock link, reported zero physics steps after reset, and did not run grasp/contact dynamics.
- Handoff: Run `python3 scripts/order8_side_proxy_pad_gui.py`; the wrapper selects `--viz kit` automatically and keeps the physics-paused viewer open until Kit is closed. `--focus-link yaw_dock_mech1` or `yaw_dock_mech2` changes the initial camera target. Inspect whether the large orange plates cover the intended physical side without unacceptable interference. Do not run or interpret a proxy contact/lift test until this placement is approved.

### 2026-07-18 (P4-full Order 8 low-load GUI black-viewport correction)
- Spec/work package: v0.4 plus the Order 8 diagnostic-GUI supplement; Agent H/I/J/K/L diagnostic visualization. The full Order 8 worktree remains uncommitted.
- User-observed problem: The synchronized replay's normal viewport was black, while enabling collision visualization revealed the collision geometry. This established that scene/camera/PhysX data existed and isolated the fault to normal visual rendering/illumination.
- Implementation/files: `scripts/order8_lift_symptom_replay_gui.py` now defaults to Isaac Lab's `balanced` rendering preset instead of the explicit `performance` preset that produced the black viewport, exposes `--rendering-mode performance|balanced|quality`, and validates the selection. `amsrr/simulation/order8_isaac_runtime.py` adds a replay-only uniform `1200`-intensity Dome Light when the synchronized Kit viewer is active; the existing Distant Light remains. Focused wrapper tests, design supplement, and WORKLOG were updated. Physics, collision geometry, state trace, controller, production config, and the `1.0 kg` mass are unchanged.
- Verification: Focused runtime/wrapper tests passed `195` before the real GUI check; compilation and `git diff --check` passed. A short real Kit launch using `balanced`, one `100x` replay loop, and a 25 s hold completed with `spawn_passed=true`, independent PhysX Dock motion `0.126427 rad`, and zero joint error. A captured `2560x1440` desktop image visibly contains all three normal Holon meshes, the blue object, support, and floor under stage lighting; the previously enabled green collision overlay is also visible and does not replace the normal mesh. The app then shut down normally, and no Isaac process remains.
- Interface/downstream boundary: One diagnostic wrapper option was added; no persisted schema or production interface changed. Rendering repair is not contact/controller/acceptance evidence.
- Handoff: Run `python3 scripts/order8_lift_symptom_replay_gui.py` again. The default is now `balanced`; collision overlay may be turned off independently in the PhysX visualization controls. Use `--rendering-mode quality` only if the workstation still has a renderer-specific display issue, and do not use `performance` on this workstation for this replay.

### 2026-07-18 (P4-full Order 8 low-load GUI joint-state synchronization correction)
- Spec version: `A-MSRR_codex_ready_spec_v0_4_ja.md` v0.4 plus the approved Order 8 diagnostic-GUI boundary in `AMSRR_design_modification_by_codex.md`.
- Work package / Agent label: Agent H/I/J/K/L diagnostic visualization. The complete Order 8 worktree remains uncommitted.
- Problem/cause: The user correctly reported that the first low-load replay showed no Dock-joint motion visually or numerically. The stored v359 trace itself contains up to `0.126456 rad` (`7.247 deg`) Dock motion, but replay called the direct state writer and then validated `robot.data.joint_pos`, the same Isaac Lab cache updated by that writer. Its previous zero-error result was circular and did not prove PhysX/Fabric/Kit synchronization.
- Summary/files: Added the diagnostic-only `--order8-state-trace-replay-sync-physics` option in `scripts/p4_control_holon_spawn_probe.py` and implemented it in `amsrr/simulation/order8_isaac_runtime.py`. For every displayed frame it writes the recorded state, sets matching position targets plus zero velocity/effort targets, advances one gravity-free synchronization step, re-applies the exact recorded state, forwards kinematics, and renders. Graph fixed constraints, contact reporting, object/floor/support collision, and robot self-collision are disabled in this replay; authored cross-module collision geometry remains enabled. `scripts/order8_lift_symptom_replay_gui.py` enables this mode by default and retains Kit performance rendering/error-only logs. `scripts/order8_current_grasp_gui.py` strips the new flag when reconstructing another capture/live command. Focused tests were updated, and the design supplement now records the corrected boundary.
- Schema/interface changes: No persisted schema or production controller/config change. One optional diagnostic CLI flag and additional diagnostic report telemetry were added. Production object mass remains exactly `1.0 kg`.
- Verification/commands: `PYTHONPATH= PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests/unit/simulation/test_order8_lift_symptom_replay_gui.py tests/unit/simulation/test_order8_current_grasp_gui.py tests/unit/simulation/test_order8_isaac_runtime.py` passed `197`; `py_compile` and `git diff --check` passed. A real default-headless PhysX replay at `100x` completed with `spawn_passed=true`; independent `root_view.get_dof_positions()` readback measured `0.1264272034 rad` (`7.244 deg`) maximum Dock motion, versus `0.1264564395 rad` (`7.247 deg`) recorded, with `0.0 rad` maximum PhysX joint write error. The replay loop itself took about `0.19 s` for the sampled high-speed frames after Kit startup.
- Assumptions/downstream impact: This step exists solely to synchronize runtime/UI transforms; it is not a simulation of the recorded contact dynamics. It does not replace the live authored-mesh physical run or qualify as Order 8 acceptance evidence. Terminal `physx_dock_delta_max` is the independent numerical check and should grow from approximately `0` to `7.24 deg` during replay.
- Blockers/next steps: No code-level blocker remains. Rerun `python3 scripts/order8_lift_symptom_replay_gui.py`; user-side visual confirmation is still required because the headless check cannot inspect the local Kit viewport. If the terminal value grows but the viewport still remains static, the remaining fault is specifically Kit viewport/Fabric presentation and should be diagnosed without rerunning the slow contact simulation.

### 2026-07-18 (P4-full Order 8 low-load 1 kg lift-symptom GUI replay)
- Spec version: `A-MSRR_codex_ready_spec_v0_4_ja.md` v0.4 plus the approved Order 8 diagnostic-GUI boundary in `AMSRR_design_modification_by_codex.md`.
- Work package / Agent label: Agent H/I/J/K/L diagnostic visualization. The complete Order 8 worktree remains uncommitted.
- Summary/files: Added `scripts/order8_lift_symptom_replay_gui.py` and focused tests. The helper refuses any source report or trace whose embedded Order 8 config is not exactly `1.0 kg`; production `configs/training/order8_natural_contact.yaml` was already and remains `1.0 kg`. It captures the exact current v359 physical trajectory once in headless mode, then replaces its launcher process with a Kit `performance`-rendering replay that applies only the recorded roots, all joint states, and object state. Replay advances no physics, suppresses ordinary Kit warning output by default, and remains diagnostic-only/acceptance-ineligible.
- Physical capture: `loaded_rebase_1kg_v359_visual_state_trace.json` contains `395` hash-validated frames from `0.00` through `15.76 s` and the `reset -> contact_acquisition -> lift` sequence. It embeds `object_mass_kg=1.0`, the current graph/config/URDF/USD identities, all three module roots and twelve joints per module, and the free-object pose/twist. The captured run reproduces the v359 symptom and stops at the retained `30 mm` cumulative-slip bound during loaded-state settle.
- Verification: New/current GUI/state-trace unit selection passed `8`, and the expanded Order 8-related unit/acceptance selection passed `388`. A real Isaac no-viewer replay of the exact trace passed in one launch with `0.0` maximum joint, module-root-position, and object-position write error; it recorded `0.126456 rad` (`7.25 deg`) maximum Dock-joint motion. This verifies asset/index binding and state application but does not claim that the user's Kit renderer has been visually inspected or that replay is physical evidence. `py_compile`, direct CLI dry-run, and `git diff --check` passed.
- Handoff/run command: `python3 scripts/order8_lift_symptom_replay_gui.py`. The existing cached 1 kg trace is reused, so this opens only the low-load GUI replay. Use `--refresh-trace` only after a relevant physical/controller change; use `--normal-kit-log` only when full Kit warnings are needed. If this no-physics performance replay still crashes, treat it as a Kit/GPU/driver problem rather than an Order 8 contact-simulation throughput problem and retain the headless physical report as authoritative.

### 2026-07-18 (P4-full Order 8 diagnostic object-mass sweep in 100 g steps)
- Spec version: `A-MSRR_codex_ready_spec_v0_4_ja.md` v0.4 plus the approved Order 8 natural-contact/simple-closure supplements in `AMSRR_design_modification_by_codex.md`.
- Work package / Agent label: Agent H/I/J/K/L lift-transition mass diagnostic. The complete Order 8 worktree remains uncommitted.
- Requested scope and implementation: Fast diagnostic contract v28 adds an explicit `--object-mass-kg` override so object mass can be changed without editing the source/production configuration. It validates a finite positive value, records the override in the report, and leaves `configs/training/order8_natural_contact.yaml` at the unchanged production baseline `1.0 kg`. All physical runs otherwise retained the corrected v359 conditions: actual compliant authored meshes, selected friction `10`, `7000/75` compliance, `Kp/Kd=200/5`, `0.5 Nm` diagnostic closure bias, one-shot loaded-state rebase, continuous joint correction off, instantaneous-slip stop disabled, and the diagnostic `30 mm` cumulative limit.
- Real-Isaac sweep: The `1.0 kg` v359 baseline and new v360-v368 reports cover `0.9` through `0.1 kg` in exact `0.1 kg` steps. `1.0/0.9/0.8 kg` confirmed lift-off but reached only `1.035/1.162/1.178 mm` maximum support clearance and failed the cumulative-slip gate before rebase completion. `0.7/0.6/0.5 kg` remained in `CONTACT_ACQUISITION` for the full `30 s`. `0.4/0.3 kg` completed the loaded-state settle/rebase and resumed the main lift, reaching `70.334/70.998 mm` maximum clearance before cumulative slip reached `30.063/30.042 mm`. `0.2 kg` remained in acquisition for `30 s`. `0.1 kg` completed rebase and reached `35.849 mm`, then stopped at `30.132 mm` cumulative slip.
- Conclusion/handoff: No tested mass passed the unchanged `100 mm` lift gate, so lowering mass alone does not complete Order 8. The response is strongly non-monotonic: the best physical result is `0.3 kg`, but even it consumes the complete diagnostic slip budget by about `71 mm` clearance. Do not promote a lighter mass or infer a monotonic payload threshold from this sweep. Production remains `1.0 kg`, selected Dock friction `2`, compliance `7500/75`, zero post-grasp bias, normal slip gates, and loaded-state rebase disabled.
- Verification: `py_compile` passed, the override/source-immutability test plus complete Order 8-related unit/acceptance selection passed `386`, and evidence assertions passed all nine mass-override reports. Every report binds its requested mass, a free object, actual Dock meshes, no object intervention/constraint, finite controller/actuator state, and the same diagnostic settings. No Isaac/Kit process remains.

### 2026-07-18 (P4-full Order 8 one-shot loaded-state rebase diagnostic)
- Spec version: `A-MSRR_codex_ready_spec_v0_4_ja.md` v0.4 plus the approved Order 8 natural-contact/simple-closure supplements in `AMSRR_design_modification_by_codex.md`.
- Work package / Agent label: Agent H/I/J/K/L lift-transition diagnostic. The complete Order 8 worktree remains uncommitted.
- Approved implementation: Added a default-off, acceptance-ineligible `--loaded-state-rebase` / `--order8-diagnostic-loaded-state-rebase` path to fast diagnostic contract v27. At the existing first `1 mm` geometric support-separation event, it snapshots the measured base-root pose and all twelve measured Dock positions exactly once, replaces the absolute Dock targets with that measured `q`, keeps zero joint-velocity targets and the already requested `0.5 Nm` closure bias, resets QPID integrators, and holds the measured base target. It requires at least `0.5 s` hold plus `0.1 s` continuous all-anchor relative speed at or below the existing `10 mm/s` pre-LIFT threshold before resuming the main lift. Payload gravity feed-forward remains active; the transient micro-lift acceleration bias is zero during the settle. The mode uses no raw-contact control input, object pose/velocity write, direct object force, or robot-object constraint, is mutually exclusive with continuous joint correction and the privileged rotation lock, and changes no production default.
- First Real-Isaac result and correction: v358 (`compliant_authored_mesh_mu10_k7000_c75_loaded_rebase_v358_30s.json`) exposed an implementation error: the transient lift-acceleration bias remained active during the nominal settle. The rebase triggered at `14.96 s` but never accumulated settle dwell and safe-held at `15.56 s` on cumulative slip `28.419/31.101 mm`. The runtime now suppresses only that transient bias while retaining payload gravity compensation; focused unit coverage binds this split.
- Corrected Real-Isaac result: v359 (`compliant_authored_mesh_mu10_k7000_c75_loaded_rebase_no_accel_v359_30s.json`) used the unchanged v355 comparison conditions: actual compliant authored meshes, selected friction `10`, `7000/75` compliance, `Kp/Kd=200/5`, `0.5 Nm` bias, continuous joint correction off, instantaneous-slip stop disabled, and retained `30 mm` cumulative bound. At the `14.96 s` trigger, cumulative paths were already `20.393/14.181 mm` and anchor-relative speeds were `13.127/1.863 mm/s`. The base root was `40.583 mm` above its measured q_close pose and still moving upward at `48.706 mm/s`. All `40` rebase steps correctly suppressed the nominal bias (peak suppressed scale `0.95`), but the one-shot pose/q reset cannot remove this kinetic state: clearance peaked at `1.035 mm`, returned to support by `15.06 s`, settle dwell remained zero, and the cumulative gate safe-held at `15.76 s` on `30.044/24.450 mm`.
- Conclusion/handoff: The exact approved instantaneous one-shot method is implemented and causally rejected for this system. It does not complete the loaded settle, main lift, transport, place, release, or final settle and is not promoted. A further correction needs a new method-level choice, most plausibly a separately bounded velocity-continuous micro-lift/deceleration trajectory before the one-shot capture; do not silently turn this into continuous joint correction, reset the slip path, relax the `30 mm` diagnostic bound, or intervene on the object. Production remains selected Dock friction `2`, compliance `7500/75`, zero post-grasp torque bias, normal slip gates, and the one-shot mode disabled.
- Verification: `py_compile` passed and focused runtime/wrapper tests passed `225`. Both real-Isaac reports bind a free object, actual Dock meshes, no continuous joint correction, no diagnostic object intervention, finite controller/actuator state, and explicit rebase telemetry. No Isaac/Kit process remains.

### 2026-07-18 (P4-full Order 8 staged Dock-compliance sweep at diagnostic friction 10)
- Spec version: `A-MSRR_codex_ready_spec_v0_4_ja.md` v0.4 plus the approved Order 8 natural-contact/simple-closure supplements in `AMSRR_design_modification_by_codex.md`.
- Work package / Agent label: Agent H/I/J/K/L contact-material diagnostic. The complete Order 8 worktree remains uncommitted.
- Requested A/B scope: Retained the previous best diagnostic selected-Dock friction `static/dynamic mu=10.0`, PhysX `max` combine mode, damping `75 Ns/m`, joint-target correction off, actual compliant authored Dock meshes, free object, `Kp/Kd=200/5`, `0.5 Nm` diagnostic closing bias, disabled instantaneous-slip stop, and retained `30 mm` cumulative diagnostic limit. Starting from v354's `7500 N/m` reference, only compliant-contact stiffness was lowered to `7000`, `6500`, and `6000 N/m`. These were explicit diagnostic overrides; production remained `object mu=0.6`, selected Dock `mu=2.0`, and `7500/75` compliance throughout the sweep.
- `7000/75` result: v355 is `artifacts/p4_full/order8_natural_contact/diagnostics/compliant_authored_mesh_mu10_k7000_c75_no_joint_correction_v355_30s.json`. It remained stable, acquired grasp, and confirmed support separation at `14.96 s`, versus `15.30 s` for the `7500/75` v354 reference. Object rise improved from `8.470` to `9.195 mm`, peak maintained-contact slip decreased from `16.951` to `16.287 mm/s`, and penetration increased only from `1.157` to `1.200 mm`. It still safe-held at `16.02 s` on cumulative slip `30.187/21.187 mm`, before the full lift/transport sequence.
- `6500/75` result: v356 is `artifacts/p4_full/order8_natural_contact/diagnostics/compliant_authored_mesh_mu10_k6500_c75_no_joint_correction_v356_30s.json`. The simulation remained finite and penetration stayed at `1.247 mm`, but one selected link's object-relative speed settled near `21.755 mm/s`, above the unchanged `10 mm/s` pre-LIFT settle threshold. It remained in `CONTACT_ACQUISITION` through the `30.00 s` ceiling; the verified grasp dwell and LIFT were not acquired. Monitor slip is consequently zero because maintained-contact slip enforcement begins only after verified grasp.
- `6000/75` result: v357 is `artifacts/p4_full/order8_natural_contact/diagnostics/compliant_authored_mesh_mu10_k6000_c75_no_joint_correction_v357_30s.json`. It acquired grasp and entered LIFT, but did not separate from support. Peak maintained-contact slip increased to `79.353 mm/s`, object rise was only `1.148 mm`, and it safe-held at `15.68 s` on cumulative slip `30.041/9.654 mm`. Penetration remained finite and below the hard ceiling at `1.305 mm`.
- Conclusion/handoff: Compliance response is nonlinear and is not improved by monotonically reducing stiffness. Of the tested values, `7000/75` is the only candidate that improves the v354 lift-off timing, rise, and peak maintained-contact slip while preserving margin to the `2 mm` penetration ceiling. `6500` and `6000 N/m` are rejected. No production value is promoted by this sweep alone; if the user approves promotion, the candidate is selected Dock `mu=10`, `7000/75`, followed by a clean production-gate rerun without diagnostic slip/bias relaxations. Order 8 remains incomplete.
- Verification: All four reports bind joint correction off, selected material readback, actual Dock mesh use, proxy off, finite force/penetration, and no drop/unintended contact. Evidence comparison assertions and `git diff --check` passed; no Isaac/Kit process remains.

### 2026-07-18 (P4-full Order 8 maximum-stable diagnostic friction, joint correction off)
- Spec version: `A-MSRR_codex_ready_spec_v0_4_ja.md` v0.4 plus the approved Order 8 natural-contact/simple-closure supplements in `AMSRR_design_modification_by_codex.md`.
- Work package / Agent label: Agent H/I/J/K/L lift-contact friction diagnostic. The complete Order 8 worktree remains uncommitted.
- User-requested condition and boundary: The v26 multi-anchor joint-target correction was explicitly left off. Only the selected Dock contact material was overridden to `static/dynamic friction = 10.0`, using PhysX `max` combine mode. This is the highest deliberately extreme fault-isolation value already shown stable in the Order 8 diagnostic history, not a claimed universal PhysX numerical maximum. The actual compliant authored Dock meshes, free object, `Kp/Kd=200/5`, `0.5 Nm` diagnostic closure-direction bias, disabled instantaneous-slip stop, retained `30 mm` cumulative diagnostic limit, and saved v314 fixture were otherwise unchanged. Production remains `object_friction=0.6` and `selected_gripper_friction=2.0`; no persisted config, controller, schema, or source default was changed.
- Real-Isaac result: `artifacts/p4_full/order8_natural_contact/diagnostics/compliant_authored_mesh_mu10_tau_bias0p5_no_joint_correction_v354_30s.json`. The material readback/audit confirmed selected static/dynamic friction `10.0`, `max` combine mode, actual Dock mesh use, proxy disabled, and `diagnostic_anchor_hold_joint_correction=false`. The simulation remained finite and controller-feasible, acquired a two-contact grasp, entered `LIFT`, and confirmed physical support separation at `15.30 s`. Object COM rose `8.470 mm` from q_close, versus `1.512 mm` with production selected friction `2.0` in v349. Peak maintained-contact tangential slip fell from `154.750` to `16.951 mm/s`; peak force/penetration were `10.153 N / 1.157 mm`, with zero unintended contact, drop, or joint-limit violation.
- Remaining failure/conclusion: The run still safe-held at `16.44 s` on the retained cumulative-slip gate (`30.001/21.692 mm`) before the full `0.1 m` lift gate, transport, place, release, or settle. Therefore extreme friction demonstrates that friction capacity is materially relevant and can produce lift-off without joint-target correction, but it does not complete Order 8 and is not promoted to production. The next production correction must preserve the `0.6/2.0` material assumptions and address articulated contact-relative motion rather than relying on `mu=10`.
- Verification/handoff: The run took `70.69 s` wall time, applied no joint-target correction, and left no Isaac/Kit process. Evidence assertions and `git diff --check` passed after documentation. Resume from v354 only as a diagnostic comparison; v349 remains the production-friction/no-correction baseline.

### 2026-07-18 (P4-full Order 8 diagnostic multi-anchor joint-target correction)
- Spec version: `A-MSRR_codex_ready_spec_v0_4_ja.md` v0.4 plus the approved Order 8 natural-contact/simple-closure supplements in `AMSRR_design_modification_by_codex.md`.
- Work package / Agent label: Agent H/I/J/K/L lift-contact control diagnostic. The complete Order 8 worktree remains uncommitted.
- Approved scope and implementation: Added a default-off, acceptance-ineligible `--anchor-hold-joint-correction` / `--order8-diagnostic-anchor-hold-joint-correction` path. Grasp acquisition and load-limited q_close remain unchanged. During `LIFT/TRANSPORT/PLACE`, the existing full-Dock whole-structure Jacobians and simultaneous two-anchor DLS produce bounded absolute joint position/velocity targets instead of discarding that result in favour of the fixed q_close target. The active controller has no neutral/null-space posture pull, uses no raw contact or contact wrench, and does not change QPID. Pitch remains fixed during the simple acquisition diagnostic but all influential upstream Dock columns, including upstream pitch, become available after LIFT entry. The established optional `0.5 Nm` closure-direction torque bias and every AK40-10 position/speed/torque/current audit remain downstream and independent. Fast diagnostic evidence advances to v26.
- Unit/minimum verification: Added parser coverage and a controller test proving simultaneous anchor correction integrates from the prior absolute target without moving an uninfluential null-space joint. Focused controller/runtime/wrapper/natural-contact/acceptance tests passed `273`; the smaller edit loop passed `238`. The saved v314 near-contact fixture avoided takeoff/asset-debug repetition for all physical A/B runs.
- Preliminary Real-Isaac diagnostics: v351 (`compliant_authored_mesh_tau_bias0p5_anchor_hold_v351_30s.json`) used the existing `1/s` task gain and exposed a command-speed mismatch: maximum correction was only `0.0044 rad/s` while load-induced joint motion was approximately `0.047 rad/s`. v352 (`compliant_authored_mesh_tau_bias0p5_anchor_hold_gain10_v352_30s.json`) used the bounded `10/s` diagnostic gain, reduced base-relative anchor translation error from `1.970` to `0.396 mm`, and remained DLS-reachable, but still safe-held at `14.52 s` on `30.260/19.290 mm` cumulative slip. Object rise/load transfer worsened to `0.910 mm / 5.04%` versus v349's `1.512 mm / 9.13%`. This revealed that merely holding q_close relative to the measured live base does not command the anchor/object lift path.
- Corrected commanded-path result: v353 is `artifacts/p4_full/order8_natural_contact/diagnostics/compliant_authored_mesh_tau_bias0p5_anchor_path_follow_v353_30s.json`. Its target is the rigid q_close anchor pair transported by the current commanded centroidal manipulation pose, matching the approved planned object/anchor trajectory rather than the preliminary live-base-relative target. The run acquired grasp and entered LIFT, but safe-held at `14.26 s` on cumulative slip `20.990/30.760 mm`; peak slip was `106.015 mm/s`, object rise only `0.033 mm`, and measured payload transfer `4.42%`. The outer loop ran `68` steps, reached the diagnostic `0.1 rad/s` joint-command ceiling, accumulated a `2.210 deg` target offset, and ended `unreachable_residual` with maximum translation/attitude error `7.026 mm / 2.228 deg`. At termination selected normal forces were only `2.040/2.175 N`; the measured vertical friction reactions available to the object summed to approximately `8.428 N`, below its `9.81 N` weight. Force/penetration maxima remained `10.030 N / 1.151 mm`, no unintended contact/drop occurred, and the AK40-10 audit passed.
- Diagnosis/open method choice: The approved outer-loop idea was implemented and exercised, but neither base-relative shape holding nor exact two-anchor commanded-path tracking fixes lift. The final task asks the Dock joints to make up common-mode base-path lag while also preserving two 6D contacts and normal preload; that stacked task is outside the bounded joint subspace and trades away the normal-load/friction margin. Further gain increase is not justified. A next method must explicitly choose a hierarchy, for example (a) QPID owns common-mode lift, joint DLS corrects only differential anchor error, plus a non-raw-contact joint-load loop maintains normal preload; or (b) a weighted task that prioritizes normal compression and vertical/tangential transport while relaxing full orientation. This is a method-level decision and is not implemented without user direction. Do not promote the diagnostic correction, `0.5 Nm` bias, disabled speed safe hold, or `30 mm` path limit to production.
- Files changed in this increment: `amsrr/simulation/order8_isaac_runtime.py`, `scripts/p4_control_holon_spawn_probe.py`, `scripts/order8_contact_force_diagnostic.py`, focused controller/wrapper tests, this worklog, and `AMSRR_design_modification_by_codex.md`. Persisted schemas/configs and the normal QPID/controller split are unchanged; only an explicit diagnostic CLI/report interface was added.
- Verification/handoff: Focused `py_compile`, evidence assertions, and `git diff --check` passed. No Isaac/Kit process remains. Resume only after selecting the task hierarchy/load-maintenance rule above; v349 remains the unchanged no-correction baseline and v353 is not acceptance evidence.

### 2026-07-18 (P4-full Order 8 object-rotation causal diagnostic)
- Spec version: `A-MSRR_codex_ready_spec_v0_4_ja.md` v0.4 plus the approved Order 8 natural-contact/simple-closure supplements in `AMSRR_design_modification_by_codex.md`.
- Work package / Agent label: Agent H/I/J/K/L lift-contact causal diagnostic. The complete Order 8 worktree remains uncommitted.
- Diagnostic implementation: Added a default-off, diagnostic-only object-rotation projection to the probe/runtime/wrapper. It captures the free object's orientation exactly when `LIFT` begins and, at each subsequent `LIFT/TRANSPORT/PLACE` physics-step boundary, restores only that quaternion and zeros only angular velocity. Object XYZ and linear velocity remain untouched, the object is never attached to the robot, and grasp acquisition is unchanged. The mode is mutually exclusive with a world-fixed object, reports every projection, increments the object-pose-write audit, is explicitly acceptance-ineligible, and advances the fast diagnostic contract to v25. This is a privileged causal A/B and not a proposed controller or P4-full solution.
- Matched Real-Isaac A/B: Baseline v349 is `artifacts/p4_full/order8_natural_contact/diagnostics/compliant_authored_mesh_tau_bias0p5_no_speed_hold_cumulative30mm_v349_30s.json`; rotation-projected v350 is `artifacts/p4_full/order8_natural_contact/diagnostics/compliant_authored_mesh_tau_bias0p5_rotation_lock_v350_30s.json`. Both used the saved v314 fixture, free translation, proxy disabled, actual compliant Dock meshes, production friction `2`, Dock gains `200/5`, four-yaw-joint closure bias `0.5 Nm`, disabled instantaneous-slip safe hold, diagnostic cumulative limit `30 mm`, and a `30 s` ceiling. v350 projected rotation for `89` steps; the largest one-step pre-projection deviation/speed was `0.001649 rad / 0.196308 rad/s`, confirming that the intervention was active.
- Result: Both runs acquired the two-contact grasp, entered `LIFT`, and safe-held on cumulative slip without support separation. Rotation projection reduced peak maintained-contact slip from `154.750` to `93.860 mm/s`, but merely moved the limiting cumulative path from module 1 to module 2: v349 `30.388/21.566 mm`, v350 `21.805/30.663 mm`. Object-COM rise from q_close decreased from approximately `1.512` to `0.365 mm`, and peak measured payload-load transfer decreased from `9.133%` to `2.696%`, despite commanded feed-forward reaching `1.0`. At v350 termination the grasp-chain yaw joints still had approximately `0.094/0.066 rad/s` relative articulation on the largest-moving pair.
- Diagnosis/handoff: Object rotation contributes to the largest instantaneous slip transient, but it is neither the sole nor the principal lift blocker. With rotation removed, articulated Dock/contact-point translation still accumulates tangential path on the opposite contact and the object remains support-borne. Do not add a production object-orientation constraint or further weaken slip gates. Resume from the contact-relative translational/articulation boundary—specifically preservation of both selected-anchor poses under payload load—while keeping QPID joint-unaware and leaving the learned `pi_L` ownership boundary intact. A deterministic anchor-hold compensator would be a new method-level choice and therefore is not implemented here.
- Verification: Focused `py_compile` passed; runtime/wrapper/natural-contact/acceptance tests passed `256`; the matched real-Isaac v350 run completed in `61.42 s` wall time with force/penetration `10.030 N / 1.151 mm`, no unintended contact or drop, and a clean AK40-10 actuator audit. Repository `git diff --check` passed after documentation updates. No Isaac/Kit process remains.

### 2026-07-18 (P4-full Order 8 0.5 Nm bias / slip-speed safe-hold disabled diagnostic)
- Spec version: `A-MSRR_codex_ready_spec_v0_4_ja.md` v0.4 plus the approved Order 8 natural-contact/simple-closure supplements in `AMSRR_design_modification_by_codex.md`.
- Work package / Agent label: Agent H/I/J/K/L contact-maintenance diagnostic. The complete Order 8 worktree remains uncommitted.
- User-requested diagnostic condition: Retained the four-joint closure-direction `0.5 Nm` post-preload bias from v347, disabled only the maintained-contact instantaneous-slip-speed safe hold, and raised only the diagnostic cumulative tangential-slip path limit from `10 mm` to `30 mm`. Slip speed remains measured and reported. The production `20 mm/s` value remains in the config and continues to derive the unchanged `10 mm/s` pre-LIFT relative-motion settle threshold. Force, torque, penetration, cumulative slip, contact break/drop, raw-contact validity, controller, collision, and AK40-10 actuator protections remain active. This path is diagnostic-only, is off by default, and advances the fast diagnostic contract to v24.
- Implementation: `NaturalContactEvidenceMonitor` has a default-true internal speed-enforcement switch; only the explicit probe flag `--order8-diagnostic-disable-slip-speed-safe-hold` sets it false. The separate wrapper option is `--disable-slip-speed-safe-hold`. The monitor still accumulates maximum speed and path length, while its contact-safe predicate ignores speed only in that explicit mode. Reports bind both the requested diagnostic flag and the effective safe-hold state. Persisted production config/schema values are unchanged.
- Real-Isaac result: `artifacts/p4_full/order8_natural_contact/diagnostics/compliant_authored_mesh_tau_bias0p5_no_speed_hold_cumulative30mm_v349_30s.json`. Proxy remained disabled; actual compliant Dock meshes, production friction `2`, Dock gains `200/5`, and every other production setting were retained. The run acquired the two-contact grasp, entered `LIFT`, reached full payload feed-forward/progress, and then safe-held at `14.58 s` on `selected_contact_cumulative_slip_limit_exceeded`, before the `30 s` ceiling. Cumulative paths were `30.388/21.566 mm`; measured maintained-contact slip peaked at `154.750 mm/s`. The object COM rose approximately `1.512 mm` from q_close, but the conservative support-separation event was not confirmed and `lift_acquired=false`.
- Safety evidence/conclusion: Peak selected-contact force was `10.030 N`, peak penetration `1.151 mm`, no unintended contact or object drop was reported, and the Dock actuator envelope audit passed with zero violation steps. Removing the instantaneous-speed stop does not reveal a self-recovering transient: the same motion accumulates to the explicitly retained `30 mm` displacement/path bound before lift-off. Do not promote this safe-hold exemption, the `30 mm` bound, or the `0.5 Nm` bias to production. Resume from the articulated contact-relative-motion correction rather than weakening the remaining cumulative gate.
- Verification: Focused `py_compile` passed; monitor/runtime/wrapper/natural-contact/acceptance tests passed `253`; repository `git diff --check` passed after documentation updates. No Isaac/Kit process remains.

### 2026-07-18 (P4-full Order 8 post-grasp Dock torque-bias A/B)
- Spec version: `A-MSRR_codex_ready_spec_v0_4_ja.md` v0.4 plus the approved Order 8 natural-contact/simple-closure supplements in `AMSRR_design_modification_by_codex.md`.
- Work package / Agent label: Agent H/I/J/K/L contact-maintenance diagnostic. The complete Order 8 worktree remains uncommitted.
- User-requested diagnostic implementation: Added an acceptance-ineligible `--post-grasp-joint-torque-bias-nm` path to the fast diagnostic and Isaac probe. After both load-limited positional preloads freeze, the scalar bias is applied to the four grasp-contributing yaw Dock joints using the established fixed-closure signs (`module_1` `+`, `module_2` `-`); all pitch and unrelated joints remain at zero bias. The command is added to the existing absolute position hold, remains bounded by the AK40-10 `1.3 Nm` continuous offset limit and independent `4.1 Nm / 7.3 A / 3 rad/s` hard envelope, and is reported per joint. Production behavior remains exactly zero torque bias when the explicit diagnostic option is absent. The fast diagnostic evidence contract is v23.
- `0.5 Nm` result: `artifacts/p4_full/order8_natural_contact/diagnostics/compliant_authored_mesh_tau_bias0p5_v347_16s.json`. Proxy was disabled and production friction/gains/slip limits plus the actual compliant Dock meshes were retained. A two-contact grasp was acquired and `LIFT` started, but support separation was not confirmed. At `14.02 s`, maintained-contact slip reached `25.306 mm/s` and correctly triggered `safety_supervisor:selected_contact_slip_speed_limit_exceeded` against the unchanged `20 mm/s` limit. Cumulative paths were `8.517/7.897 mm`, peak per-contact force `10.030 N`, peak penetration `1.151 mm`, and the Dock actuator audit passed with zero violation steps.
- `1.0 Nm` result: `artifacts/p4_full/order8_natural_contact/diagnostics/compliant_authored_mesh_tau_bias1p0_v348_16s.json`. The same proxy-free production conditions were used. Both selected mesh contacts and full preload were present, but the constant bias sustained selected-link/object relative motion above the `10 mm/s` pre-LIFT settle threshold; the run remained in `CONTACT_ACQUISITION` through the `16.00 s` diagnostic ceiling. Consequently monitor `grasp_acquired=false`, `LIFT` was never entered, and support separation was not confirmed. Peak per-contact force and penetration remained `10.030 N / 1.151 mm`; the Dock actuator audit again passed with zero violation steps.
- Conclusion/handoff: Neither requested value fixes production lift contact maintenance. `0.5 Nm` reaches lift but still fails the instantaneous-slip gate; `1.0 Nm` is worse because it prevents the pre-lift relative-motion dwell. Do not promote either value to production or continue increasing constant bias merely because the AK40 limit permits it. Resume from the lift-transition/articulated relative-motion issue using actual compliant meshes and the saved v314 fixture; the proxy remains legacy diagnostic-only.
- Verification: Focused `py_compile` passed. Runtime/wrapper/natural-contact/acceptance tests passed `228`; repository `git diff --check` passed after documentation updates. Both real-Isaac A/B runs used the same fixture and production safety settings, and no Isaac/Kit process remains.

### 2026-07-18 (P4-full Order 8 authored-mesh compliant-contact replacement)
- Spec version: `A-MSRR_codex_ready_spec_v0_4_ja.md` v0.4 plus the approved Order 8 natural-contact/simple-closure supplements in `AMSRR_design_modification_by_codex.md`.
- Work package / Agent label: Agent J/K/L contact-physics boundary. The complete Order 8 worktree remains uncommitted.
- Approved decision/implementation: The finite-area proxy pad is no longer an active P4-full solution. It remains only as an explicitly acceptance-ineligible legacy diagnostic. `order8_natural_contact_config_v11` instead applies a spatially uniform PhysX implicit compliant-contact spring to each complete selected authored Dock collision-mesh rigid body. The persisted values are `7500 N/m` stiffness and `75 Ns/m` damping; the object remains one free rigid body, no task-specific pad/contact location is supplied, and raw contact remains privileged diagnostic/safety evidence only. Runtime reads back `physxMaterial:compliantContactStiffness` and `physxMaterial:compliantContactDamping`, fails closed on mismatch, reports the selected body/material paths and values, and normal acceptance requires `selected_surface_actual_dock_mesh=true`, proxy disabled, and the compliant-material audit to pass. The fast diagnostic contract advances to v22 and exposes temporary stiffness/damping overrides without changing the source config. The legacy proxy GUI narrowly migrates recorded v10 config JSON by adding only the two v11 compliance fields, so it remains inspectable without treating old evidence as current acceptance.
- Files changed for this increment: `amsrr/schemas/order8.py`, `configs/training/order8_natural_contact.yaml`, `amsrr/simulation/order8_isaac_runtime.py`, `amsrr/simulation/order8_natural_contact.py`, `scripts/order8_contact_force_diagnostic.py`, `scripts/order8_proxy_pad_gui.py`, focused schema/runtime/wrapper/acceptance tests, this worklog, and `AMSRR_design_modification_by_codex.md`. A short-path fixture initialization bug exposed by the new one-second material audit was also fixed by seeding stationary phase pose/velocity maps before the first runtime loop; it does not change the normal per-step planner path.
- Isaac-independent verification: focused `py_compile` passed and the schema/runtime/natural-contact/wrapper/acceptance suite passed `235` tests. The final expanded Order 8 controller/policy/robot-model/schema/simulation/acceptance suite passed `344` tests, including the legacy GUI migration; repository `git diff --check` and the migrated proxy GUI `--print-command` smoke also passed.
- Minimum real-Isaac material audit: `compliant_mesh_audit_k7500_c75_v343_1s.json` confirmed the authored material retained exactly `7500/75`, proxy was disabled, selected material audit passed, and no Isaac process remained. The preloaded neutral fixture itself had positive mesh gaps and was not used to infer force/penetration performance.
- Proxy-free grasp calibration: `compliant_authored_mesh_k7500_c75_v344_13s.json` reused the saved collision-free v314 state, the representative-contact diagnostic `Kp/Kd=200/8`, selected friction `10`, and temporary `60 mm/s / 30 mm` slip limits. It used no proxy, acquired two actual Dock-mesh contacts and a measured grasp, entered `LIFT`, reached `9.661 N` maximum per-contact force, `1.157 mm` maximum penetration (below `2 mm`), `4.009 mm/s` maintained-contact slip through `13 s`, zero unintended contact/drop/raw-contact error, and a clean actuator-envelope audit. The observed `F/k` scale is consistent with the intended approximately `1--2 mm` compliant overlap, so no fixture-specific stiffness search was continued.
- Proxy-free initial-lift result: `compliant_authored_mesh_k7500_c75_v345_16s.json` under the same explicitly relaxed diagnostic conditions confirmed support separation at `15.02 s` with both authored-mesh contacts maintained. Through `16 s`, maximum penetration remained `1.157 mm`, maintained-contact slip was `14.234 mm/s`, cumulative paths were `23.534/18.110 mm`, no object drop/unintended contact occurred, and the actuator audit passed. `lift_acquired` remains false because the full `0.1 m` lift-clearance gate was intentionally not reached within this prefix; this result proves only grasp and initial physical lift-off.
- Production-setting A/B and remaining issue: `compliant_authored_mesh_production_v346_16s.json` restored configured selected friction `2`, Dock `200/5`, and unchanged `20 mm/s / 10 mm` safety limits. It still acquired a two-contact authored-mesh grasp with `10.030 N` peak force and `1.151 mm` penetration, then safe-held at `13.74 s` on a `41.627 mm/s` lift-entry slip transient before support separation. Cumulative paths were only `5.208/4.932 mm`; actuator and raw-contact audits passed. Therefore the proxy/generalization defect is resolved, but Order 8/P4-full acceptance is not complete: the previously documented production lift-transition/contact-maintenance problem remains and must not be hidden by the relaxed diagnostic. Resume from that lift transient, using the v314 fixture and actual compliant meshes; do not restore the proxy as the production solution or retune mesh-following IK.

### 2026-07-18 (P4-full Order 8 proxy-pad GUI inspector)
- Spec version: `A-MSRR_codex_ready_spec_v0_4_ja.md` v0.4 plus the approved diagnostic finite-area proxy supplement in `AMSRR_design_modification_by_codex.md`.
- Work package / Agent label: Agent J/K/L diagnostic tooling; user-requested GUI placement inspection only. The complete Order 8 worktree remains uncommitted.
- Implemented: Added `scripts/order8_proxy_pad_gui.py`. It loads a recorded acceptance-ineligible proxy-pad report, validates the required Order 8/diagnostic/proxy flags, strips headless/replay/pacing options, opens Kit, stops after a selected physics budget, and then updates rendering without advancing physics for a configurable wall-clock hold. The exact proxy cubes remain orange, `30 x 30 x 2 mm`, attached to the selected Dock rigid links. The default `--state grasp` (with `contact` as an alias) now re-executes the recorded near-contact rollout continuously through the first report sample with `grasp_acquired=true` and two selected contact links; for v342 this is `604` steps / approximately `12.08 s`. `--state qclose` retains exact checkpoint restore only as a dynamically fragile comparison, and `--state open` shows the deliberately collision-free starting fixture. Default hold is `120 s`.
- Files changed: `scripts/order8_proxy_pad_gui.py`, `tests/unit/simulation/test_order8_proxy_pad_gui.py`, `amsrr/simulation/order8_isaac_runtime.py`, and `for_codex/WORKLOG.md`.
- Schema/interface changes: None. No controller, runtime physics, persisted config, proxy geometry, or acceptance rule changed.
- Diagnosis/tests/commands: The restored q_close comparison was not a valid stable-contact display: a five-step real-Isaac check measured no initialized contact at `0.00 s`, one selected contact at `0.02 s` (`2.341 N`), and no selected contacts from `0.04 s` onward as the restored graph/contact constraints re-resolved. This explains the visibly separated restored screenshot and does not contradict the original continuous v342 rollout, which had both selected contacts when q_close latched and again throughout the final grasp dwell. `PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests/unit/simulation/test_order8_proxy_pad_gui.py tests/unit/simulation/test_order8_isaac_runtime.py` passed `193` tests; focused `py_compile`, direct `--print-command`, and `git diff --check` passed. A real Isaac/Kit smoke of the new continuous default completed all `604` steps and exited `0` in about `23 s` wall time. The earlier q_close smoke also exposed a first-sample `nominal_base_target` initialization bug unique to already-latched replay; the runtime now seeds it from the initialized fixture command before the loop. No Isaac/Kit process remains.
- Usage: `python3 scripts/order8_proxy_pad_gui.py` continuously produces and freezes the two-pad grasp. Use `--state qclose` only to inspect the fragile restored checkpoint, `--state open` for the intentionally separated start, `--keep-open-s <seconds>` to change the viewer hold, and `--print-command` to inspect without launching. This diagnostic cannot satisfy Order 8 acceptance.

### 2026-07-18 (P4-full Order 8 instantaneous-slip 60 mm/s diagnostic)
- Spec version: `A-MSRR_codex_ready_spec_v0_4_ja.md` v0.4 plus the approved Order 8 natural-contact/simple-closure supplements in `AMSRR_design_modification_by_codex.md`.
- Work package / Agent label: Agent H/I/J/K/L boundary; user-requested continuation of the temporary slip-limit A/B. The complete Order 8 worktree remains uncommitted.
- Scope/configuration: Repeated the v341 saved-v314 near-contact/proxy/free-object diagnostic with `Kp=200 Nm/rad`, `Kd=8 Nms/rad`, selected-gripper friction `10`, cumulative-slip limit `30 mm`, and instantaneous-slip limit raised diagnostically from `20` to `60 mm/s`. Production limits remain unchanged. The existing runtime derives the pre-LIFT relative-motion settle threshold as half the configured instantaneous limit, so this diagnostic also reports `30 mm/s` rather than the production `10 mm/s`; the actual phase trace still entered LIFT at approximately the same time as v341.
- Result: `artifacts/p4_full/order8_natural_contact/diagnostics/slip60mmps_cumulative30mm_kp200_kd8_v342_30s.json`. The run reached `reset -> approach -> contact_acquisition -> lift -> safe_hold` and stopped at `14.18 s`, before the `30 s` ceiling, on `selected_contact_slip_speed_limit_exceeded`. Peak maintained-contact slip increased to `160.883 mm/s`, above the temporary `60 mm/s` limit. Cumulative paths were only `17.984/19.944 mm`, below `30 mm`. Object COM rose `7.031 mm` relative to q_close, but lift was not confirmed before safe hold.
- Stop-transient diagnosis: The final spike is not merely a signed-displacement accumulation or the cumulative gate. Both selected raw contacts were still present. At `14.18 s`, object angular velocity about world X jumped to approximately `0.266 rad/s`; at the second contact the object-side point moved downward at `-69.60 mm/s` while its Dock point moved upward at `+91.28 mm/s`, yielding approximately `160.88 mm/s` tangential relative speed. This is a physical articulated/object-rotation transient in the measured kinematics, although the raw contact application points also move within the finite patch.
- Safety/physics evidence: Peak selected-contact force was `10.282 N`, peak penetration `0.079 mm`, no object drop occurred, both contact links were present at termination, raw-contact failure reasons were empty, and the Dock actuator envelope audit passed with zero violation steps.
- Files changed: `for_codex/WORKLOG.md` only for this increment; generated evidence is under the artifact tree. No production source/config value was changed.
- Schema/interface changes: None.
- Verification/commands: Real-Isaac diagnostic through `micromamba run -n isaaclab3`; focused JSON assertions confirmed `0.060/0.030` overrides, `200/8` gains, failure identity, and report integrity; `git diff --check` passed. No Isaac/Kit process remains.
- Conclusion/next step: Threshold relaxation alone does not make lift succeed. It permits a larger object-roll/contact-relative-velocity transient and then trips the raised instantaneous gate. Further increasing the gate would be a new safety/method decision rather than a bug fix; the next control work should prevent the articulated/object rotation or stabilize the selected anchors during load transfer.

### 2026-07-18 (P4-full Order 8 cumulative-slip 30 mm diagnostic)
- Spec version: `A-MSRR_codex_ready_spec_v0_4_ja.md` v0.4 plus the approved Order 8 natural-contact/simple-closure supplements in `AMSRR_design_modification_by_codex.md`.
- Work package / Agent label: Agent H/I/J/K/L boundary; user-requested temporary cumulative-slip relaxation using the best representative-contact Dock gain from the prior sweep. The complete Order 8 worktree remains uncommitted.
- Scope/configuration: Reused the saved v314 near-contact fixture, diagnostic finite-area proxy, selected-gripper friction `10`, free object, normal QPID/payload schedule, and AK40-10 limits. Used `Kp=200 Nm/rad`, `Kd=8 Nms/rad`, which had produced the largest object-COM rise among the representative-contact gain candidates. Only the diagnostic cumulative-slip override changed from `10 mm` to `30 mm`; the production config remains `10 mm` and the instantaneous limit remains `20 mm/s`.
- Result: `artifacts/p4_full/order8_natural_contact/diagnostics/cumulative_slip30mm_kp200_kd8_v341_30s.json`. The run reached `reset -> approach -> contact_acquisition -> lift -> safe_hold` and stopped at `14.08 s`, before the requested `30 s` ceiling, on `selected_contact_slip_speed_limit_exceeded`. Peak maintained-contact slip was `24.843 mm/s`, above the unchanged `20 mm/s` instantaneous limit, while cumulative paths were only `16.626/15.101 mm`, below the temporary `30 mm` limit. The object COM rose `5.818 mm` relative to the q_close checkpoint, but conservative support-clearance lift-off was not confirmed and `lift_acquired=false`.
- Safety/physics evidence: Both selected contacts remained represented at termination; peak selected-contact force was `10.282 N`, peak penetration was `0.079 mm`, no object drop occurred, and the Dock actuator torque/current/speed envelope audit passed with zero violation steps. The run is diagnostic-only and acceptance-ineligible.
- Files changed: `for_codex/WORKLOG.md` only for this increment; the generated diagnostic report is under the artifact tree. No production source/config value was changed.
- Schema/interface changes: None.
- Tests/commands: `PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests/unit/simulation/test_order8_contact_force_diagnostic.py -k 'override_raw_slip_limit or parser'` passed `13` tests (`14` deselected). The first host-Python launch stopped before simulation because `tomllib` was unavailable in the selected interpreter; rerunning through `micromamba run -n isaaclab3` completed the real-Isaac diagnostic in `61.27 s` wall time. No Isaac/Kit process remains.
- Conclusion/next step: Raising only cumulative-slip tolerance to `30 mm` does not make the lift succeed; it exposes an independent instantaneous-slip failure during the extra lift-bias ramp. Do not weaken the instantaneous gate without an explicit method decision. Resume from the articulated anchor-hold/control correction issue rather than further threshold-only relaxation.

### 2026-07-17 (P4-full Order 8 constrained Dock-drive tuning complete / production 200/5 retained)
- Spec version: `A-MSRR_codex_ready_spec_v0_4_ja.md` v0.4 plus the approved Order 8 natural-contact/simple-closure supplements in `AMSRR_design_modification_by_codex.md`.
- Work package / Agent label: Agent H/I/J/K/L boundary; user-requested AK40-10 Dock-joint `Kp/Kd` tuning with minimum-environment-first execution. The complete Order 8 worktree remains uncommitted.
- Implementation: Added `amsrr/simulation/dock_joint_drive_tuning.py`, `scripts/tune_dock_joint_drive.py`, and focused unit tests. The diagnostic reuses the existing Holon USD and runs one fixed-base module with gravity, ground, self-collision, contact sensors, and conversion disabled. All four Dock joints are excited concurrently by a `0.01 rad` step/return and `1.2 Nm` local child-link disturbance. The search is deterministic, coarse-to-fine, and constrained by unchanged AK40-10 `4.1 Nm` peak torque, `7.3 A` peak-current estimate, and `3 rad/s` safe simulation speed. It uses time-domain tracking/settling/disturbance metrics instead of ultimate-sensitivity tuning because PhysX implicit-drive saturation makes the latter unsuitable.
- Minimum-bench result: `artifacts/p4_full/order8_natural_contact/joint_drive_tuning/dock_joint_drive_tuning_v1.json`; `96` candidates plus one repeat, `184.3 s` aggregate simulated candidate duration in `24.75 s` wall time. Baseline `200/5` scored `2.6505`; the contact-free numerical minimum `800/4.375` scored `0.2251`, settled every joint step in `0.04 s`, stayed below `0.094 rad/s`, `2.022 Nm`, and estimated `3.60 A`, and repeated within `1.8e-8`. The report now labels this as `contact_free_bench_candidate_only` and requires separate representative-contact validation.
- Representative contact A/B: Identical saved v314 fixture, proxy geometry/friction, QPID/payload schedule, safety gates, and `16 s` ceiling were used. `800/4.375` (v336) and `650/4.375` (v337) hit the instantaneous-slip gate at absolute simulation time `9.00/9.22 s`; `300/5` (v339) and `200/8` (v340) hit cumulative slip at `11.68/13.36 s`. Configured `200/5` (v338) hit the same cumulative gate at `14.54 s` after `0.642 mm` object-COM rise. `200/8` reached `1.065 mm` COM rise, but the conservative support-clearance lift-off confirmation remained absent and one side had already accumulated `10.075 mm` slip. Every run passed the Dock actuator-envelope audit. Because acquisition duration changes with gain, absolute stop time alone is not treated as a phase-normalized score; the hard result is that none completed verified lift-off with slip margin.
- Decision: Retain production `Kp=200 Nm/rad`, `Kd=5 Nms/rad`, armature `0.01 kg m^2` because no alternative passed representative contact validation. Higher stiffness improves isolated tracking but fails early load transfer, and the bench-favored `Kd=8` does not clear the task safety gate. No contact/safety limit was weakened and no production gain was changed by the search.
- Verification/status: The focused tuning/runtime/wrapper/actuator suite passed `218`; the expanded Order 8 suite passed `333`; focused `py_compile`, YAML reload, and repository `git diff --check` passed. The final bench rerun exited `0`, its deterministic repeat passed, and no Isaac/Kit process remains. Order 8 remains incomplete. Resume from the previously documented articulated anchor-hold decision; do not continue gain sweeping or reinterpret the contact-free optimum as a deployment gain.

### 2026-07-17 (P4-full Order 8 payload-component A/B complete / paused at articulated anchor-hold decision)
- Spec version: `A-MSRR_codex_ready_spec_v0_4_ja.md` v0.4 plus the approved Order 8 natural-contact/simple-closure supplements in `AMSRR_design_modification_by_codex.md`.
- Work package / Agent label: Agent H/I/J/K/L boundary; user-approved diagnostic separation of payload translational force, COM-offset moment, and rotational-inertia coupling. The repository is on `master`; the complete Order 8 worktree remains uncommitted, and no Isaac/Kit process is running.
- Diagnostic-only implementation: Added three payload-coupling component modes to the separated minimum fixture: `full`, `translational_force_only`, and `translational_force_and_com_offset_moment`. The latter two respectively zero both payload offset/inertia and payload inertia only while preserving the same measured payload mass, normal payload-coupling/QPID/allocator path, transition gates, free object, contact monitor, and AK40-10 limits. Non-full modes are acceptance-ineligible and fail closed outside the explicitly separated diagnostic. Reports retain both the applied component view and the unmodified full coupling. The fast diagnostic contract is now v21; no production payload schedule or persisted acceptance contract changed.
- Isaac-independent verification: focused runtime/wrapper tests passed `211`; the final expanded Order 8 controller/planner/schema/runtime/acceptance suite passed `345`; focused `py_compile` and repository `git diff --check` passed.
- Full reference v333: `artifacts/p4_full/order8_natural_contact/diagnostics/separated_lift_kinematics_v333_14p5s.json`. Peak maintained-contact slip was `16.675 mm/s`; the two selected Dock contact points reached approximately `-15.224/-16.825 mm/s` downward. At `14.20 s`, centroidal rigid-point kinematics instead predicted `+12.38/+13.79 mm/s`, so the residual is articulated motion rather than a centroidal wrench sign error.
- Force-only v334: `artifacts/p4_full/order8_natural_contact/diagnostics/payload_force_only_v334_14p5s.json`. Removing both COM-offset moment and rotational inertia did not restore symmetric upward Dock motion. One Dock reached approximately `-19.287 mm/s`, pitch evolved nose-down, and selected slip reached `20.129 mm/s`, causing the unchanged safety gate to stop the run at `13.96 s`. The diagnostic payload torque was exactly zero. Thus the `-Y` offset moment is not the cause; removing it makes the load-transfer response less balanced.
- Force plus COM-offset moment, no rotational inertia v335: `artifacts/p4_full/order8_natural_contact/diagnostics/payload_force_offset_no_inertia_v335_14p5s.json`. Its centroidal and selected-Dock histories are effectively identical to full v333: Dock minima were approximately `-15.195/-16.865 mm/s` and peak slip was `16.674 mm/s`. The object rose only about `0.325 mm` by the diagnostic ceiling. Therefore rotational-inertia coupling is immaterial here, while the correctly signed nose-up offset moment is beneficial but cannot prevent the relative articulation.
- Corrected diagnosis: The causal trigger is the payload translational-force/load-transfer transient interacting with the articulated morphology, not the COM-offset moment sign or rotational inertia. QPID realizes the requested centroidal wrench, but fixed local Dock position targets do not guarantee that the upstream Dock chain preserves the selected anchor pose under the changed load. In the full/no-inertia runs, relevant upstream pitch joints still moved at about `0.12-0.14 rad/s` despite fixed targets, while the actuator-envelope audit remained valid. The deterministic smoke currently lacks the morphology-conditioned joint-position correction that learned `pi_L` is ultimately expected to provide.
- Method-level boundary / exact resume point: Do not alter payload-moment sign, remove the physically necessary offset moment, weaken slip/contact gates, use raw contact in nominal control, or apply force directly to the object. The minimum next discriminator is one diagnostic-only increase of Isaac Dock drive stiffness/damping while retaining the exact AK40-10 torque/current/speed limits. If it cannot hold the upstream chain, continuing requires an explicitly approved simple non-privileged kinematic anchor-hold compensator that updates relevant absolute Dock position targets during load transfer; QPID remains centroidal and joint-unaware. Run only the saved v314 fixture until this choice is resolved, then return to complete lift/transport/place/release/settle and the unchanged full authored-mesh environment.
- Status: Payload-component A/B is complete. Order 8 and P4-full acceptance are not complete; paused before a new control-method change.

### 2026-07-17 (P4-full Order 8 separated LIFT transition A/B / paused at payload-wrench scheduling decision)
- Spec version: `A-MSRR_codex_ready_spec_v0_4_ja.md` v0.4 plus the approved Order 8 natural-contact/simple-closure supplements in `AMSRR_design_modification_by_codex.md`.
- Work package / Agent label: Agent H/I/J/K/L boundary; diagnostic separation of contact-yield recovery, ordinary LIFT/payload coupling, and the approved transient upward bias. The repository is on `master`; the complete Order 8 worktree remains uncommitted, and no Isaac/Kit process is running.
- Diagnostic-only implementation:
  - Added a pre-LIFT restore gate that requires the measured-grasp centroidal rebase, centroidal-yield blend and joint-drive-yield blend both at zero, admittance disabled, and base speed within the existing pre-grasp tolerance. The raw two-contact dwell begins only after that gate.
  - Added exact selected-Dock and object contact-point velocities, centroidal/target twists, transition stage, restore readiness, and per-stage vertical-velocity extrema to diagnostic evidence. This distinguishes link-COM motion from the actual contact-point motion.
  - Added independently selectable delays for the extra LIFT acceleration bias and an acceptance-ineligible A/B switch that suppresses payload feed-forward while retaining the identical LIFT pose trajectory. The fast diagnostic contract is now v20. These switches do not change production acceptance behavior.
- Isaac-independent verification: the focused runtime/wrapper tests passed `206`; the final expanded Order 8 controller/planner/schema/runtime/acceptance suite passed `340`; focused `py_compile` and repository `git diff --check` passed.
- v331 separated-transition diagnostic: `artifacts/p4_full/order8_natural_contact/diagnostics/separated_lift_transition_v331_18s.json` used the saved v314 near-contact fixture, diagnostic finite-area proxy, `mu=10`, unchanged production payload ramp and safety gates, and a delayed extra bias. `LIFT` began at `13.28 s`, only after full QPID/yield restoration and a fresh dwell. During restore and pre-LIFT dwell, the two exact Dock contact points continued upward at approximately `+2.80..+2.93 mm/s` and `+3.22..+3.56 mm/s`. After ordinary payload coupling began, both reversed downward, first at `13.46/13.48 s` when feed-forward progress was only `0.16/0.18`, reaching approximately `-15.22/-16.83 mm/s` at progress `0.94`. The extra `1 N` acceleration bias never activated. The run safe-held at `14.54 s` on the unchanged cumulative-slip gate.
- v332 payload-coupling A/B diagnostic: `artifacts/p4_full/order8_natural_contact/diagnostics/lift_pose_only_no_payload_ff_v332_15s.json` retained the same restored transition and upward LIFT pose trajectory but diagnostically disabled payload coupling and delayed the extra bias beyond the run. Both Dock contact-point vertical velocities remained upward throughout LIFT, approximately `+1.56..+2.84 mm/s` and `+2.46..+3.44 mm/s`; no hard safety failure occurred through `15.0 s`. The terminal wrapper result is only the expected incomplete-monitor failure because the diagnostic ended before lift/transport/place/release completion. No lift-off was expected without payload support.
- Corrected diagnosis: The v330 conclusion that the free object simply outran an upward-moving gripper is superseded. The downward Dock-contact-point reversal is caused by enabling payload coupling, not by QPID/yield restoration, the upward pose-trajectory sign, or the extra acceleration bias. At v331, the QPID payload term included about `9.77 N` upward effective force and `-4.13 Nm` pitch moment from the approximately `0.423 m` payload-COM offset, changing the target pitch moment from about `+0.31` to `-3.82 Nm` before geometric lift-off. Full payload coupling is isolated as the trigger, but force versus moment/inertia is not yet isolated.
- v333 frame/kinematics correction: `artifacts/p4_full/order8_natural_contact/diagnostics/separated_lift_kinematics_v333_14p5s.json` repeated the shortest fixture with centroidal pose/twist and target telemetry. Because the payload lies on centroidal body `+X`, `r x (+Fz)` is `-Y`; this is correctly a nose-up moment that raises the object-side Dock under rigid-body motion. The measured centroidal body did rise and rotate nose-up. At `14.20 s`, centroidal rigid-point kinematics predicted the two Dock contact points moving upward at approximately `+12.38/+13.79 mm/s`, while their measured velocities were `-14.77/-16.15 mm/s`. The exact articulated residual was therefore approximately `-27.15/-29.94 mm/s`. The previous wording that the `-4.13 Nm` moment itself drove the Dock downward was wrong: payload coupling triggers large opposite relative motion inside the articulated morphology. QPID's achieved-wrench metric is allocator/model evidence at the centroidal frame, not evidence that every compliant Dock link follows the corresponding rigid-body velocity.
- Method-level blocker / recommended next diagnostic: Preserve the validated pre-LIFT restore gate. Before selecting a production payload schedule, isolate vertical payload force from COM-offset moment/inertia with diagnostic-only A/B runs using the same v314 fixture. If force-only retains upward Dock motion while moment activation recreates the articulated residual, then schedule the vertical force from bounded commanded progress and the moment/inertia from non-privileged measured load transfer/lift-off. If force-only also recreates it, the payload-force allocation/local-joint interaction must be addressed instead. Keep raw PhysX contact out of nominal control, keep the object free, and retain all force/slip/penetration/actuator gates.
- Exact resume point after explicit approval: add diagnostic-only force/moment component switches and evidence; run the minimum force-only and moment-addition A/B; only then version a production payload schedule. Require upward Dock contact-point motion, geometric lift-off, and margin under both slip gates before any downstream transport/place/release/settle run.
- Status: Paused at the required method-level choice. Order 8 and P4-full acceptance are not complete.

### 2026-07-17 (P4-full Order 8 lift-acceleration intent / paused at velocity governor)
- Spec version: `A-MSRR_codex_ready_spec_v0_4_ja.md` v0.4 plus the approved Order 8 natural-contact/simple-closure supplements in `AMSRR_design_modification_by_codex.md`.
- Work package / Agent label: Agent H/I/J/K/L boundary; user-approved LIFT-only payload inertial bias, versioned evidence, minimum saved-fixture diagnostic, and fail-closed stop when the result exposed a new control-method decision. The repository is on `master`; the complete Order 8 worktree remains uncommitted.
- Implemented approved method:
  - Advanced the persisted Order 8 config to `order8_natural_contact_config_v10` with `lift_payload_acceleration_mps2=1.0` and `lift_acceleration_bias_removal_s=0.5`.
  - During `LIFT`, the transient world-up force is `m_payload * a_lift * commanded_lift_progress`. It is rotated through the measured centroidal pose into body coordinates and enters the existing `PolicyCommand.residual_wrench_body` -> QPID CoM-wrench path. It is not applied directly to rotors and is not a contact/internal-wrench command.
  - The first existing 1 mm geometric lift-off event latches the current scale; the bias then falls linearly to zero over 0.5 s. It is exactly zero outside `LIFT`. Payload gravity/inertia coupling, local Dock position control, raw-contact privilege, force/slip/penetration limits, and AK40-10 limits are unchanged.
  - Added fail-closed report/acceptance evidence for method identity, configured mass/acceleration/removal time, scheduled and PolicyCommand-active counts, LIFT-external activation count, scale/force/body-wrench norm, lift-off latch, removal completion, and terminal zero wrench. State trace and live terminal telemetry expose the scale and force. The fast diagnostic contract is now v19.
- Isaac-independent verification: focused schema/runtime/wrapper suite passed `194`; the expanded Order 8 controller/planner/measurement/GUI/wrapper suite passed `333`; focused `py_compile` and `git diff --check` passed. Host pytest required `PYTHONPATH=.` and `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` because the external `launch_testing` plugin is incompatible with the installed pytest hookspec.
- Minimum real-Isaac diagnostic: `artifacts/p4_full/order8_natural_contact/diagnostics/lift_acceleration_bias_proxy_mu10_v330_16s.json`, using the saved v314 near-contact fixture, diagnostic proxy, `mu=10`, unchanged production `payload_load_transfer_s=1.0`, unchanged `20 mm/s` instantaneous and `10 mm` cumulative slip gates, and a 16 s ceiling. It ran 13.92 simulation seconds in 59.92 wall seconds and safe-held on `selected_contact_slip_speed_limit_exceeded`; no Isaac/Kit process remains.
- v330 evidence / result:
  - Bias scheduling and application were correct: 56 scheduled active steps, 56 PolicyCommand-active steps, zero non-LIFT active steps, peak scale `1.0`, peak world-up force `1.0 N`, and peak body residual-force norm `1.0 N`. QP/actuator mapping remained on the intended path.
  - The object rose `3.041 mm` from the measured q_close COM and was moving upward at `39.04 mm/s`. Its tilted OBB lower support clearance was only `0.679 mm`, so the conservative 1 mm lift-off audit had not yet latched.
  - The selected Dock patches lagged the rising object: terminal object-frame tangential relative velocity was predominantly `-Z`, reaching `34.07 mm/s`; cumulative paths were already `9.243/9.555 mm`. The bias therefore created real upward motion but violated the unchanged instantaneous slip gate just before lift-off.
  - For comparison, v326 with no acceleration bias reached only `+0.533 mm` COM rise at `22.61 mm/s`; instantaneous slip stayed at `17.16 mm/s`, but cumulative paths reached `10.097/9.953 mm`. Thus neither zero bias nor the approved open-loop 1 N bias has margin, and selecting an intermediate constant solely for this fixture is not a robust solution.
- Current diagnosis: This is not a frame/sign/plumbing bug. A full-second constant-amplitude `m_payload * 1 m/s^2` open-loop intent accelerates the 1 kg free object faster than the maintained-contact gripper motion, whose centroidal command is capped at `10 mm/s`. Merely reducing the constant trades the v330 instantaneous-slip failure back toward the v326 cumulative-slip failure.
- Method-level blocker / recommended next decision: Govern the *increase* of payload feed-forward and acceleration bias with non-privileged object/Dock kinematic relative velocity. Keep the base lift target progressing at the existing `10 mm/s`; when the object begins to outrun the selected Dock surfaces, hold feed-forward progress and remove the acceleration bias until the surfaces catch up, then resume with hysteresis. Do not use raw PhysX contact/slip, weaken acceptance limits, increase base speed, or change the free-object model. This changes the approved time-only progress driver and therefore requires explicit approval before implementation.
- Exact resume point after approval: version the bounded kinematic velocity governor; add pure schedule/hysteresis tests plus report evidence; run one v314 proxy early-lift diagnostic only; require 1 mm lift-off with margin under both slip gates before downstream transport/place/release/settle or a complete environment run.
- Status: Paused at the required method-level choice. Order 8 and P4-full acceptance are not complete.

### 2026-07-17 (P4-full Order 8 synchronized lift progress / paused at lift-acceleration intent)
- Spec version: `A-MSRR_codex_ready_spec_v0_4_ja.md` v0.4 plus the approved Order 8 natural-contact/simple-closure supplements in `AMSRR_design_modification_by_codex.md`.
- Work package / Agent label: Agent H/I/J/K/L boundary; approved commanded-progress payload feed-forward floor, versioned evidence, shortest proxy diagnostic, and fail-closed stop at the next control-method decision. The repository is on `master`, the complete Order 8 worktree remains uncommitted, and no Isaac/Kit process is running.
- Implemented approved method:
  - Bumped the persisted config contract to `order8_natural_contact_config_v9`.
  - During `LIFT`, the shared phase progress `clip(lift_elapsed / payload_load_transfer_s, 0, 1)` now drives both maintained-contact motion entry and the minimum known-payload feed-forward target. The feed-forward target is the monotonic maximum of commanded progress, aggregate centroidal observed-load share, measured object-rise share, and verified lift-off; the existing slew bound remains active.
  - Aggregate centroidal external-wrench estimation remains a non-privileged lower-bound/audit and raw Isaac contact remains privileged-only. No contact/internal wrench was added to `PolicyCommand` or QPID, no object constraint/pose write was introduced, and all slip/force/penetration/AK40-10 gates remain unchanged.
  - Added report/acceptance evidence for the progress method, last/peak commanded progress, permitted lead over observed load, and zero maximum lag behind commanded progress. The exact method/driver strings are versioned. Added a diagnostic-only `--payload-load-transfer-s` override and advanced the fast diagnostic contract to v18 without changing the production default (`1.0 s`).
- Files changed in this increment: `amsrr/schemas/order8.py`, `configs/training/order8_natural_contact.yaml`, `amsrr/simulation/order8_isaac_runtime.py`, `amsrr/simulation/order8_natural_contact.py`, `scripts/order8_contact_force_diagnostic.py`, and focused schema/runtime/wrapper tests; design supplement and WORKLOG updated.
- Schema/interface changes: Approved versioned Order 8 config/report contract update from v8 to v9. Normal policy/QPID/actuator schemas are unchanged. The duration override is diagnostic-only.
- Tests/commands: focused schema/runtime/measurement/wrapper suite passed `222`; focused `py_compile` passed; `git diff --check` passed. Real diagnostics ran through `micromamba run -n isaaclab3` from the saved v314 near-contact fixture with the diagnostic proxy and `mu=10` solely to isolate lift scheduling.
- Real-Isaac results under unchanged `20 mm/s` instantaneous and `10 mm` cumulative slip limits:
  - v326 (`1.0 s`): progress/feed-forward reached `1.0` with zero lag and observer peak `0.945`. The object began rising (`+0.533 mm`, terminal `+22.6 mm/s`) but safe-held at `14.02 s` on cumulative paths `10.097/9.953 mm`, about `0.47 mm` before the lift-off audit.
  - v327 (`0.5 s`): cumulative paths fell to `4.120/4.567 mm`, but the faster ramp caused `22.76 mm/s` instantaneous slip and safe-held at `13.32 s`.
  - v328 (`0.75 s`): peak slip remained safe at `19.51 mm/s`; the object rose `0.678 mm` at `22.3 mm/s`, but one side reached `10.080 mm` cumulative slip about `0.32 mm` before lift-off.
  - v329 (`0.72 s`): cumulative paths were only `5.971/6.917 mm` when `20.73 mm/s` instantaneous slip stopped the run. QP and actuator envelopes remained valid in every run and no lift-off was claimed.
- Current diagnosis: The approved progress floor removes the observed-load deadlock and produces actual upward object motion. However, a single linear duration has no robust gate margin: `0.72 s` is too abrupt for instantaneous slip, while `0.75 s` is already too slow for cumulative slip. Searching a narrow `0.73-0.75 s` value could create a fixture-specific pass but would not be a robust control solution.
- Blocker / open question: A physical lift requires not only payload gravity compensation but also positive upward acceleration. The current QPID target at full payload accounting remains feasible/non-saturated yet supplies almost exactly payload weight; contact transmits about `9.3-9.8 N`, so the supported object begins rising too late. The recommended next method is a small bounded `LIFT`-only upward CoM wrench bias (for example `Fz = m_payload * a_lift`, initially `a_lift = 1.0 m/s^2`), ramped with the same progress and removed after lift-off/steady lift. This uses the existing CoM-wrench intent path and does not make QPID contact-aware.
- Next steps after explicit approval: add/version the lift-acceleration bias and evidence; unit-test zero bias outside LIFT plus bounded ramp/removal; retain the safer production `1.0 s` progress duration and run one v314-fixture q_close-to-early-lift proxy diagnostic; require lift-off with margin below both unchanged slip gates before lowering proxy friction or entering downstream phases.
- Status: Paused as required at a new method-level choice. Order 8 and P4-full acceptance are not complete.

### 2026-07-17 (P4-full Order 8 finite-area proxy diagnostic / paused at lift feed-forward rule)
- Spec version: `A-MSRR_codex_ready_spec_v0_4_ja.md` v0.4 plus the approved Order 8 natural-contact/simple-closure supplements in `AMSRR_design_modification_by_codex.md`.
- Work package / Agent label: Agent H/I/J/K/L boundary; user-approved diagnostic finite-area selected-contact proxy, shortest real-Isaac calibration, and fail-closed handoff at the next method decision. The repository is on `master`, the full in-progress Order 8 worktree remains uncommitted, and no Isaac/Kit process is running.
- Summary of changes:
  - Added an acceptance-ineligible `--order8-diagnostic-proxy-pad` / `--proxy-pad` path. Each selected Dock rigid body receives one connect-frame-aligned `30 x 30 x 2 mm` thin cube collider fitted to the sampled outer face of the actual selected Dock mesh. Its outer face is `3 mm` beyond the mesh and its inner face remains `1 mm` outside it, so the unchanged `2 mm` penetration limit makes proxy and retained authored mesh mutually exclusive at the object surface.
  - The proxy has no independent rigid body, uses the explicitly selected high-friction material, and retains the actual Dock mesh for visuals and non-proxy collision. Reports audit authored paths, collision traversal, material binding, independent-rigid-body count, retained mesh, and geometric exclusivity. Proxy reports explicitly set `selected_surface_actual_dock_mesh=false`, use `diagnostic_finite_area_proxy_pad_v1`, and cannot satisfy Order 8 acceptance.
  - Added focused helper/parser coverage and diagnostic report metadata. No normal policy observation, QPID command, raw-contact privilege boundary, actuator envelope, slip/force/penetration threshold, or persisted Order 8 config/result schema was changed.
- Files changed in this increment: `amsrr/simulation/order8_isaac_runtime.py`, `scripts/p4_control_holon_spawn_probe.py`, `scripts/order8_contact_force_diagnostic.py`, `tests/unit/simulation/test_order8_isaac_runtime.py`, and `tests/unit/simulation/test_order8_contact_force_diagnostic.py`; this entry and the matching design supplement were also updated.
- Schema/interface changes: None to persisted schemas/interfaces. The new CLI/runtime switch and report keys are diagnostic-only and acceptance-ineligible. Promoting a proxy surface into the persisted Order 8 contract would require a separately approved, versioned schema/config change.
- Upstream dependencies used: current selected Dock mesh sampling/connect frames, v8 load-limited positional preload, aggregate centroidal external-wrench load observer, free-object constraint audit, AK40-10 envelope, and unchanged privileged raw-contact monitor.
- Isaac-independent verification: focused Order 8 schema/runtime/measurement/wrapper suite passed `221` tests. The proxy helper also fails closed when a sampled selected face cannot contain the required finite pad. Focused `py_compile` and repository `git diff --check` passed.
- Commands run: the five-file focused pytest matrix for Order 8 schema/runtime/measurement/wrapper; focused `python3 -m py_compile`; `git diff --check`; v321 USD audit; and v322-v325 real-Isaac proxy diagnostics through `micromamba run -n isaaclab3`.
- Short real-Isaac evidence from the saved v314 near-contact fixture:
  - v321 (`0.2 s`) proved both pads were authored under the intended selected rigid bodies with collision/material/exclusivity audits passing and zero independent rigid bodies: `artifacts/p4_full/order8_natural_contact/diagnostics/proxy_pad_usd_audit_v321_0p2s.json`.
  - v322 (`mu=3.0`) and v323 (`mu=4.5`) reached `LIFT` but did not lift off before the unchanged cumulative-slip gate. Peak inferred payload transfer was approximately `0.695/0.741`; reports are `proxy_pad_friction3_v322_14s.json` and `proxy_pad_friction4p5_v323_14s.json` in the same diagnostic directory.
  - v324 (`mu=10.0`, `14 s`) was diagnostic-time-limited only `1.22 s` after LIFT entry and therefore was not used as a success/failure conclusion.
  - v325 (`mu=10.0`, `18 s` ceiling) reached `reset -> approach -> contact_acquisition -> lift -> safe_hold`. Both proxy contacts remained present; peak/terminal selected normal forces were about `10.03/10.33 N` and `2.64/2.65 N`, QP allocation remained feasible and unsaturated, the Dock actuator envelope passed, penetration peaked at only `0.086 mm`, and no robot/environment contact occurred. Nevertheless the object rose only about `0.002 mm`; inferred/feed-forward transfer reached about `0.795/0.792`, then the selected pads accumulated `10.077/7.789 mm` of tangential path and safe-held at `14.50 s`. Report: `artifacts/p4_full/order8_natural_contact/diagnostics/proxy_pad_friction10_v325_18s.json`.
- Current diagnosis: finite area and friction are no longer the limiting factors. The observed-load-only feed-forward rule is circular while the object remains supported: full object weight must be supplied before lift-off, but the support carries the untransferred fraction, so the observer commands only the already transferred fraction. At v325 the controller added approximately `0.792 m_payload g`, leaving the support to carry the remainder; the QP itself was feasible/non-saturated while the selected contact patches continued tangential sliding (cumulative displacement predominantly object `+X/-Z`). Increasing friction further cannot remove this load-transfer deadlock.
- Assumptions: The proxy is a temporary fault-isolation representation only. The current `10 mm` slip bound, free object, no kinematic attach, raw-contact diagnostic-only rule, `1 kg` payload, and AK40-10 torque/current/speed limits remain unchanged.
- Blocker / open question: Continuing requires a method-level LIFT scheduling decision. The direct correction is to introduce a bounded commanded lift-progress floor for payload feed-forward (ramping centroidal upward motion and known-payload compensation together to full support), while retaining the aggregate observer as a lower-bound/audit. This differs from the currently implemented observed-load-only driver and must not be changed silently.
- Next steps after explicit approval: implement and unit-test the synchronized lift-progress/feed-forward rule; rerun only the saved-fixture q_close-through-early-lift slice with the proxy; require geometric lift-off below unchanged slip/force/penetration/actuator gates; then decide separately whether the proxy becomes a versioned production contact representation before transport/place/release/settle and unchanged full-environment acceptance.
- Status: Paused at the required method decision. Order 8 and P4-full acceptance are not complete.

### 2026-07-17 (P4-full Order 8 load-limited positional preload / paused at contact-capacity choice)
- Spec version: `A-MSRR_codex_ready_spec_v0_4_ja.md` v0.4 plus the approved Order 8 natural-contact/simple-closure supplements in `AMSRR_design_modification_by_codex.md`.
- Work package / Agent label: Agent H/I/J/K/L boundary; approved replacement of the post-`q_close` open-loop Jacobian-transpose torque-bias ramp, shortest real-Isaac verification, and fail-closed handoff. The repository is on `master`, the full in-progress Order 8 worktree remains uncommitted, and no Isaac/Kit process is running.
- Implemented approved grasp-preload rule:
  - After the measured `q_close` velocity settle, the fixed closure ratio continues from the previous absolute position target at `0.002 rad/s`. It is never rebased from measured `q` during preload.
  - Each selected side uses only its moving influential Dock joints, observes the maximum damping-compensated actuator load, and freezes independently after the configured load threshold persists for `contact_stall_dwell_s`. A shared joint freezes conservatively when any owning side freezes.
  - Once both sides freeze, every preload velocity becomes exactly zero. `LIFT` is withheld until the existing full selected-link/object relative-speed dwell also passes.
  - Policy, unclipped/limited mapping, and Isaac effort-target torque bias are forced to exactly zero from preload through carriage. QPID remains centroidal-only, local position servos remain independent, and raw Isaac contact remains privileged diagnostic/safety evidence only.
  - Config/report schema is now `order8_natural_contact_config_v8`; the wrapper validates the preload method, moving-joint sets, threshold/dwell/freeze evidence, terminal zero velocities, and zero offset-torque path. The safe default threshold remains `1.2 Nm`, below the AK40-10 `1.3 Nm` continuous rating.
- Isaac-independent verification:
  - Schema/runtime focused gate: `165 passed` after the final `1.2 Nm` default was restored.
  - Earlier complete focused Order 8 set for this increment: `218 passed`; rerun the same complete set after any next method choice. Python compilation and `git diff --check` must also be rerun before handoff/commit.
- Short real-Isaac diagnostics, all using the saved v314 near-contact state, the complete three-module/free-object/authored-mesh path, unchanged `10 mm` cumulative-slip safety gate, and no kinematic payload constraint:
  - v318 (`1.0 Nm`): both sides froze at `1.060/1.008 Nm`; selected normal force was about `2.27 N/side`; inferred payload transfer peaked at `0.514`; no lift-off; safe hold at the slip gate. Report: `artifacts/p4_full/order8_natural_contact/diagnostics/load_limited_position_preload_v318_14s.json` (`52.71 s` wall).
  - v319 (`1.2 Nm`, retained default): both sides froze at `1.295/1.210 Nm`; selected normal force was about `2.63 N/side`; inferred feed-forward followed observed transfer to `0.572` with zero lead; no lift-off; paths were `10.013/6.624 mm`. Report: `artifacts/p4_full/order8_natural_contact/diagnostics/load_limited_position_preload_v319_14s.json` (`58.92 s` wall).
  - v320 (`1.3 Nm` boundary diagnostic only): both sides froze, but dwell/servo overshoot reached `1.405/1.315 Nm`, so this threshold was not retained. Selected normal force was about `2.86 N/side`; inferred transfer rose only to `0.601`; no lift-off; paths were `10.110/5.646 mm`. Report: `artifacts/p4_full/order8_natural_contact/diagnostics/load_limited_position_preload_v320_14s.json` (`58.41 s` wall).
- Current diagnosis / method-level boundary:
  - The new control rule itself works: closure is monotonic, both sides independently load/freeze, offset torque stays zero, pre-lift motion dwell passes, `LIFT` begins, and payload feed-forward never leads observed support-load transfer.
  - The remaining failure is physical contact capacity/representation. At the AK40-10 continuous-load boundary the authored selected meshes generate only about `2.86 N` normal force per side; the supported `1 kg` object transfers only about 60% of its weight before the grippers slide predominantly in object `+X/-Z`. Raising the same load threshold further is disallowed and is not a valid next step.
  - Continuing now requires a user-approved method choice: (A) an explicitly labelled thin finite-area high-friction proxy pad on each selected Dock surface, with actual meshes retained for visual/non-selected collision and raw-contact acceptance applied to the proxy; (B) a justified change to contact material/payload assumptions; or (C) a different grasp mechanics/control strategy. Do not silently relax slip, use raw contact in nominal control, exceed AK40-10 limits, or add a kinematic attach.
- Exact resume point: obtain the method choice above. If proxy/material is approved, first run one acceptance-ineligible minimum diagnostic from the saved near-contact state to calibrate the lowest sufficient friction/contact representation, then repeat only q_close-through-early-lift. Proceed to lift/transport/place/release/settle and the unchanged complete acceptance environment only after geometric lift-off occurs below every existing safety bound.
- Status: Paused as required at a non-bug method decision. Order 8 and P4-full acceptance are not complete.

### 2026-07-17 (P4-full Order 8 observed-load payload feed-forward / paused at grasp-hold equilibrium)
- Spec version: `A-MSRR_codex_ready_spec_v0_4_ja.md` v0.4 plus the approved Order 8 natural-contact/simple-closure supplements in `AMSRR_design_modification_by_codex.md`.
- Work package / Agent label: Agent H/I/J/K/L boundary; approved q_close-to-lift load-transfer correction, focused real-Isaac diagnosis, and handoff before a new grasp-hold method choice. The repository remains on `master`, the complete in-progress Order 8 worktree is uncommitted, and no Isaac/Kit process is running.
- Implemented approved lift/load-transfer behavior:
  - `LIFT` now starts its bounded upward centroidal motion immediately instead of holding the pose while payload compensation ramps independently.
  - Payload feed-forward follows the monotonic fraction of known payload weight inferred from the aggregate centroidal external-wrench estimate. The LIFT-entry external vertical force is the zero-load baseline; inferred transferred load is normalized by `m_payload * g`, clipped to `[0, 1]`, and slew-limited by `payload_load_transfer_s`.
  - Geometric object rise remains a lower-bound audit and verified support separation forces full payload accounting. Raw Isaac contact is still privileged diagnostic/safety evidence only and is not an observer, planner, actor, or QPID input.
  - Added observer validity/baseline/load/feed-forward-lead/lift-off telemetry, corresponding acceptance checks, and reusable initial near-contact state loading from completed diagnostic reports.
- Files changed in this increment:
  - `amsrr/simulation/order8_isaac_runtime.py`
  - `amsrr/simulation/order8_natural_contact.py`
  - `scripts/order8_contact_force_diagnostic.py`
  - `configs/training/order8_natural_contact.yaml`
  - `tests/unit/simulation/test_order8_isaac_runtime.py`
  - `tests/unit/simulation/test_order8_natural_contact.py`
  - `tests/unit/simulation/test_order8_contact_force_diagnostic.py`
- Isaac-independent verification:
  - `PYTHONPATH= PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests/unit/simulation/test_order8_contact_force_diagnostic.py tests/unit/simulation/test_order8_isaac_runtime.py tests/unit/simulation/test_order8_natural_contact.py tests/unit/simulation/test_order8_contact_measurement.py`: `208 passed`.
  - Python compilation and `git diff --check` passed before the real diagnostics.
- Real-Isaac v314, standard `0.25 s` contact dwell:
  - Report: `artifacts/p4_full/order8_natural_contact/diagnostics/load_observer_lift_v314_14s.json`; `14 s` ceiling, `101.35 s` wall time.
  - q_close and full-force preload passed; `LIFT` began at approximately `11.65 s`. The load observer remained valid, feed-forward never led the observed load (`max lead = 0`), and by safe hold inferred/commanded load fractions were approximately `0.795/0.783`.
  - The object did not leave the support. The unchanged `10 mm` cumulative-slip limit stopped the run at about `13.12 s`; module paths were `8.609/10.011 mm`, with module 2 signed object-frame displacement approximately `[-9.131, 0, -1.388] mm`.
- Real-Isaac v315, diagnostic `2.0 s` full-force dwell:
  - Report: `artifacts/p4_full/order8_natural_contact/diagnostics/load_observer_settle_v315_16s.json`; `16 s` ceiling, `115.56 s` wall time.
  - Extending the dwell did not settle the grasp. During full-force hold the object remained stationary, but selected contact-link/object relative speeds converged to neither zero nor a decaying transient: immediately before `LIFT` they remained approximately `7.4/6.5 mm/s`, while base-module linear speed remained about `5.8 mm/s`.
  - `LIFT` began at `13.40 s` and safe hold occurred at `14.87 s` for the same cumulative-slip gate. Observer/feed-forward fractions reached approximately `0.802/0.789`, feed-forward lead remained zero, and no geometric lift-off occurred. Module paths were `8.606/10.048 mm`; module 2 again accumulated predominantly object `-X` displacement (`-9.154 mm`).
- Real-Isaac v316, user-approved AK40-10 peak-torque fault-isolation diagnostic:
  - Report: `artifacts/p4_full/order8_natural_contact/diagnostics/peak_torque_lift_v316_14p5s.json`; diagnostic-only `6.0 s` post-q_close peak-limit window, normal-force target `6 N/contact`, `14.5 s` ceiling, and `105.14 s` wall time.
  - The active torque-bias limit reached the configured AK40-10 peak `4.1 Nm`; maximum applied torque/current were exactly bounded at approximately `4.1 Nm/7.3 A`, maximum measured Dock speed was `0.0843 rad/s`, and the actuator-envelope audit recorded zero violation steps.
  - Full-force selected normal loads reached approximately `5.72/5.74 N`, versus approximately `2.75 N` under the continuous `1.3 Nm` limit. Higher preload did not settle the grasp: base speed rose to about `8.2 mm/s`, and selected-link/object relative speeds reached approximately `6.6/13.1 mm/s`. The normal-force increase therefore strengthened, rather than removed, the persistent articulated shear motion.
  - The pre-lift non-privileged relative-motion gate correctly withheld `LIFT`. As the peak window returned to continuous torque, only `0.06 s` of the required `0.25 s` dwell was acquired before a numerical `clipped_actuator_targets` safe hold at `13.23 s`; no payload feed-forward or lift-off occurred.
  - The terminal clip was not a physical AK40-10 envelope violation. Accumulated simulation time left the scheduled limit only about `1.4e-12 Nm` above `1.3 Nm`, which was then rounded/clipped by the bridge. The peak-window handoff now snaps sub-`1e-9 Nm` residue to the exact continuous limit, with focused regression coverage. Repeating the heavy v316 run is unnecessary for the method conclusion because `LIFT` was already blocked throughout the actual high-force interval.
- Real-Isaac v317, final simulation-gain A/B before method change:
  - Report: `artifacts/p4_full/order8_natural_contact/diagnostics/high_damping_lift_v317_14p5s.json`; unchanged continuous `1.3 Nm` torque bias, Dock stiffness `200 Nm/rad`, diagnostic damping increased from `5` to `20 Nms/rad`, `14.5 s` ceiling, and `102.94 s` wall time.
  - The actuator torque/current/speed envelope remained valid. The grasp dwell passed and `LIFT` began at `11.63 s`, but the object never left the support and the unchanged cumulative-slip gate stopped the run at `12.95 s`.
  - Higher damping reduced module 1 slip path from v314's `8.609 mm` to `5.424 mm`, but the failing module 2 path was unchanged (`10.011 -> 10.037 mm`). Its signed displacement remained dominated by object `-X` (`-9.872 mm`). The problematic side still entered lift near `8.1 mm/s` relative speed and ended near `7.6 mm/s`; maximum Dock speed reached the diagnostic `0.1 rad/s` ceiling during acquisition.
  - Therefore the failure is not removable by further normal-force or simulation-damping gain sweep. No `20 Nms/rad` production/default change is adopted.
- Current diagnosis / method-level boundary:
  - The synchronized motion and load observer are functioning as designed; neither premature payload compensation nor insufficient fixed dwell explains the remaining failure.
  - The maintained slip is already seeded by persistent articulated/contact-link motion during the full-force hold. The measured q_close position targets are fixed and the object is stationary, but the nominal Dock impedance plus saturated `1.3 Nm` continuous torque bias and QPID pose hold do not converge to a sufficiently stationary physical equilibrium. The current pre-lift gate accepts up to `10 mm/s`, so it authorizes lift despite a persistent `6-7 mm/s` motion that consumes the `10 mm` path budget in roughly `1.5 s`.
  - Merely extending dwell, tightening the gate, increasing normal-force torque, or increasing simulator drive damping would stall/timeout or leave the failing side unchanged rather than create equilibrium. Do not relax the slip limit or use raw contact in nominal control.
- Exact resume point after user choice:
  1. Revise the grasp-hold command/control rule rather than adding torque. The current evidence requires a bounded non-privileged stabilization rule that lets the fixed measured q_close morphology and centroidal hold reach low relative velocity before authorizing payload motion; it must preserve the independent joint-command/QPID split and may not consume raw per-patch Isaac contact.
  2. Use the saved v314-v316 initial near-contact state and the shortest q_close-through-early-lift slice; require both selected link/object relative speeds to decay materially before another downstream/full run.
  3. Only after a safe early lift, verify lift completion, transport, place, release, and settle in short slices, then rerun the unchanged complete environment and independent Order 8 acceptance.
- Status: Paused at the required grasp-hold method decision. No acceptance claim, threshold relaxation, proxy collision, kinematic attach, or grasp-hold law change has been made. The only post-v316 source correction is the diagnostic peak-window floating-point handoff clamp.

### 2026-07-17 (P4-full Order 8 lift-slip vector diagnosis / paused before method change)
- Spec version: `A-MSRR_codex_ready_spec_v0_4_ja.md` v0.4 plus the approved Order 8 natural-contact/simple-closure supplements in `AMSRR_design_modification_by_codex.md`.
- Work package / Agent label: Agent H/I/J/K/L boundary; privileged contact-kinematics diagnosis of the first verified-grasp-to-lift failure. This entry supersedes the earlier temporary-branch handoff: the repository is on `master`, the in-progress Order 8 worktree remains uncommitted, and no Isaac/Kit process is running.
- Implemented diagnostic instrumentation:
  - Added deterministic per-contact-patch body/object relative velocity, tangential velocity, contact point, and contact normal measurement without changing the persisted raw-contact or policy/QPID schemas.
  - Added selected-patch signed tangential slip velocity in world/object frames, signed cumulative displacement, cumulative path length, dominant axis, and phase/object/payload-feed-forward step telemetry. Raw Isaac contact remains privileged validation/diagnostic input only.
  - Added an acceptance-ineligible `--qclose-fixture-state-trace` diagnostic path that reconstructs the first recorded `lift` frame from the hash-bound v308 trace. Exact and zero-velocity restores both excite graph/contact-constraint transients, so this path cannot faithfully replace a continuous physical rollout and must not be used as acceptance evidence.
- Files changed in this diagnostic increment:
  - `amsrr/simulation/order8_contact_measurement.py`
  - `amsrr/simulation/order8_isaac_runtime.py`
  - `scripts/order8_contact_force_diagnostic.py`
  - `tests/unit/simulation/test_order8_contact_measurement.py`
  - `tests/unit/simulation/test_order8_isaac_runtime.py`
  - `tests/unit/simulation/test_order8_contact_force_diagnostic.py`
- Verification:
  - `PYTHONPATH= PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests/unit/simulation/test_order8_contact_force_diagnostic.py tests/unit/simulation/test_order8_contact_measurement.py tests/unit/simulation/test_order8_isaac_runtime.py`: `197 passed`.
  - Python compilation for the edited source/test files and `git diff --check` passed.
  - Short restored-state diagnostics v309-v312 established that checkpoint restore is not dynamically faithful enough to enter lift: restored graph/contact constraints produce relative-motion oscillation and prevent the pre-lift dwell. These runs were used only to reject that shortcut before repeating a full rollout.
- Faithful real-Isaac reproduction:
  - Re-ran the complete v308 near-contact three-module/free-object/authored-mesh path continuously for a `14 s` ceiling with the new vector telemetry. Report: `artifacts/p4_full/order8_natural_contact/diagnostics/full_near_contact_slip_vector_v313_14s.json`.
  - The run reproduced `reset -> approach -> contact_acquisition -> lift -> safe_hold`: verified q_close/force dwell completed, `LIFT` began at approximately `11.65 s`, and unchanged safety logic stopped at `12.83 s` because module 2 cumulative tangential slip reached `10.156 mm` against the `10 mm` limit.
  - Signed cumulative slip in object coordinates at failure was module 1 `[-3.874, +0.002, -0.353] mm` and module 2 `[-6.488, -0.001, -6.064] mm`. Thus the failing side slips diagonally in object `-X/-Z`, not in the contact-normal direction; its cumulative path is `10.156 mm`.
  - Both selected contacts remained present. Final selected normal forces were approximately `2.54/2.55 N`; no controller failure, unintended environment contact, actuator-envelope violation, or joint-limit violation caused the stop.
- Root-cause evidence:
  - During the first `1.0 s` of `LIFT`, payload feed-forward ramps from `0` to `1`, but the current motion-entry schedule deliberately keeps the centroidal lift target stationary for that same interval. The object height consequently remains essentially unchanged until the feed-forward ramp completes.
  - By that point, module 2 had already accumulated `7.663 mm` of slip path (about 75% of the eventual failure value) while the object was still supported. The subsequent physical rise adds the remainder. This strongly identifies the dominant cause as preloading the grasp against the support before initiating lift, rather than insufficient simulation duration or a raw-contact gate defect.
  - Dock torque bias remained clipped to the AK40-10 continuous `1.3 Nm` limit while requested bias was about `3.09 Nm`; this did not violate the configured torque/speed/current envelope. Peak-torque scheduling is a possible later controlled experiment, but it is not the first correction and was not changed here.
- Method-level blocker / proposed change (not yet implemented): Synchronize lift motion entry and payload feed-forward from the start of `LIFT`, using one bounded load-transfer progress over `payload_load_transfer_s`, instead of holding the robot target stationary while payload compensation alone ramps. QPID then supplies the initial upward trajectory while the known-payload model is introduced concurrently. Preserve the `10 mm` slip limit, raw-contact diagnostic-only boundary, free object, contact/friction/penetration gates, and AK40-10 limits.
- Exact resume point after user approval:
  1. Change only the lift/load-transfer scheduling above and add focused unit coverage for the synchronized ramp.
  2. Run the shortest continuous q_close-through-early-lift diagnostic; confirm object rise starts with the feed-forward ramp and pre-rise slip is reduced without weakening any gate.
  3. If it passes, test lift completion, transport, place, release, and settle as short slices, then rerun the unchanged full environment and independent Order 8 acceptance. If synchronized scheduling still fails, pause again before considering a brief AK40-10 peak-torque window.
- Status: Paused at the required method-level decision. No acceptance claim, threshold relaxation, proxy collision, kinematic attach, or control-schedule change has been made in this increment.

### 2026-07-17 (P4-full Order 8 temporary checkpoint after host crash)
- Spec version: `A-MSRR_codex_ready_spec_v0_4_ja.md` v0.4 plus the approved Order 8 natural-contact/simple-closure supplements in `AMSRR_design_modification_by_codex.md`.
- Work package / Agent label: Agent H/I/J/K/L boundary; temporary source-control checkpoint and exact resumption handoff only.
- Status: Work is deliberately paused on branch `temp` for a temporary commit/push. This snapshot is not a completed Order 8 implementation and is intended to be reset/recommitted cleanly after the host is made stable. No Isaac/Kit process should be assumed to remain running.
- Preserved implementation state:
  - The current worktree contains the complete in-progress Order 8 schema/controller/planner/runtime/evidence/acceptance/CLI/test surface described by the preceding entries, including the diagnostic-only pitch hold and previous-target joint-position integration correction.
  - Real-Isaac diagnostic v308 completed monotonic yaw closure, formed both selected authored-mesh contacts, completed the non-privileged simultaneous-arrest dwell, latched measured `q_close`, completed force ramp and grasp dwell, and entered `LIFT`.
  - The v308 state trace is preserved at `artifacts/p4_full/order8_natural_contact/diagnostics/pitch_hold_integrated_target_v308_30s_state_trace.json`. This is diagnostic evidence only, not Order 8 acceptance.
- Verification already completed before this checkpoint:
  - `tests/unit/simulation/test_order8_isaac_runtime.py`: `151 passed` with host plugin autoload disabled.
  - Focused joint-controller/evidence/GUI/runtime companion set: `59 passed`.
  - `python3 -m py_compile ...` for the corrected runtime/tests and `git diff --check` passed.
  - v308 entered fail-closed `SAFE_HOLD` at `12.83 s` because selected-contact cumulative slip reached `10.156 mm`, just above the unchanged `10 mm` safety limit. No acceptance threshold was relaxed and no proxy collision pad was added.
- Current implementation problem: Closure and the grasp gate now succeed. The next code/simulator problem is asymmetric contact/load transfer and cumulative tangential slip at the start of lift. Transport, place, release, settle, wrapper acceptance, and independent Order 8 acceptance remain unverified.
- Host-stability incident:
  - A subsequent live-GUI attempt was followed by a full host crash/reboot, not a normal Isaac exception. The retained previous-boot kernel record contains a CPU Machine Check immediately followed by a supervisor-mode NX page fault. It contains no corresponding OOM kill, VRAM-exhaustion evidence, or NVIDIA Xid.
  - The host is an Intel Core i9-14900KF on an ASRock Z790 Steel Legend WiFi with BIOS `9.03` dated 2023-06. Heavy Isaac execution may expose the instability, but normal simulation load must not be treated as the root cause of a kernel Machine Check.
- Exact resume instructions:
  1. Before another heavy or GUI Isaac run, back up the repository, update to a current stable motherboard BIOS, load Intel default/baseline limits, temporarily disable XMP/overclock/undervolt, and test CPU/RAM stability. If Machine Check events recur at stock settings, stop simulator work and diagnose/RMA the affected hardware.
  2. Restore this temporary snapshot from `origin/temp`, confirm `git status`, and rerun `git diff --check` plus the focused Isaac-independent tests above. Do not discard the v308 trace.
  3. Inspect the v308 q_close-to-lift trace and build/run only the shortest q_close-to-lift diagnostic needed to observe per-side tangential reaction, object twist, centroidal target/load transfer, selected-contact slip, and requested/limited/applied joint commands. Do not repeat the complete 30 s/high-fidelity path while isolating this fault.
  4. Correct the lift/load-transfer cause without relaxing the `10 mm` cumulative-slip limit or using privileged raw Isaac contact to generate nominal policy/QPID intent. Preserve the diagnostic pitch mask as acceptance-ineligible.
  5. After a short lift slice passes, verify transport, place, release, and settle as separate short slices; then rerun the unchanged complete environment, wrapper validation, independent acceptance, and finally GUI inspection.
- Schema/interface changes in this checkpoint: None; this entry only records the temporary source-control and recovery boundary.
- Blockers/open questions: Host hardware/firmware stability must be established before simulator-heavy work resumes. The Order 8 lift-slip fault remains open after that prerequisite.

### 2026-07-17 (P4-full Order 8 pitch hold / previous-target closure integration)
- Spec version: `A-MSRR_codex_ready_spec_v0_4_ja.md` v0.4 plus the approved Order 8 natural-contact/simple-closure supplements in `AMSRR_design_modification_by_codex.md`.
- Work package / Agent label: Agent H/I/J/K/L boundary; diagnostic-only joint-command correction and real-Isaac natural-contact evidence.
- Summary:
  - Preserved every physical Dock articulation, but for this acceptance-ineligible diagnostic only, captured all pitch Dock angles at initialization and overrode their final commands to the same absolute position with zero velocity and zero torque bias.
  - Changed the simple fixed-velocity joint command from measured-state rebasing to `position_target[k+1] = position_target[k] + velocity_command * dt`, with joint position-limit clipping.
  - Zeroed pitch components before normalizing the one-shot closure velocity ratio. Yaw components remain the one-shot whole-structure direction; no online/receding mesh IK was restored.
- Files changed:
  - `amsrr/simulation/order8_isaac_runtime.py`
  - `tests/unit/simulation/test_order8_isaac_runtime.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: None. Added only diagnostic report fields for the pitch-hold positions/method/error and final integrated closure targets. Normal `PolicyCommand`, QPID, TaskSpec, morphology, and learned-policy contracts are unchanged.
- Upstream dependencies used: Current v307 raised-support near-contact fixture, representative three-module morphology, authored Dock meshes, nominal `200 Nm/rad` / `5 Nms/rad` drives, AK40-10 limits, CoM-only admittance, and unchanged contact/slip safety thresholds.
- Tests and commands:
  - `python3 -m py_compile amsrr/simulation/order8_isaac_runtime.py tests/unit/simulation/test_order8_isaac_runtime.py` passed.
  - Initial plain pytest launch failed before collection because the host ROS `launch_testing` plugin is incompatible with the installed pytest hook API.
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests/unit/simulation/test_order8_isaac_runtime.py`: `151 passed`.
  - `PYTHONPATH=/home/leus/amsrr PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests/unit/simulation/test_order8_contact_force_diagnostic.py tests/unit/simulation/test_order8_current_grasp_gui.py tests/unit/simulation/test_natural_contact_evidence.py tests/unit/controllers/test_natural_contact_joint_controller.py`: `59 passed`.
  - `git diff --check` passed before the real run.
  - The first real-run attempt used the system Python and stopped before simulator startup because `tomllib` was unavailable; the sandboxed micromamba retry likewise stopped before startup because its cache lock was not writable. The approved external `isaaclab3` run then executed normally.
- Real-Isaac command/result:
  - Reused the exact v307 diagnostic conditions: near-contact source `/tmp/order8_near_contact_measured_q_contact_v296.json`, `+0.15 m` support/fixture shift, `3.0 s` opening from `/tmp/order8_fixed_whole_structure_closure_v305_30s.json`, `dt=0.01`, speed scale `2.0`, Dock velocity ceiling `0.1 rad/s`, selected-gripper friction `3.0`, target force `6 N`, and a `30 s` simulation ceiling.
  - Report: `/tmp/order8_pitch_hold_integrated_target_v308_30s.json`.
  - State trace: `artifacts/p4_full/order8_natural_contact/diagnostics/pitch_hold_integrated_target_v308_30s_state_trace.json`.
  - Closure initialized at `2.04 s`; fixed pitch velocity targets were all exactly zero. Both selected raw contacts formed by about `6 s`; the unchanged simultaneous non-privileged arrest dwell reached `0.10 s`, `q_close` latched, the force ramp reached `1.0`, and the verified `0.25 s` contact-command dwell transitioned to `LIFT` at `11.65 s`.
  - The run entered `SAFE_HOLD` at `12.83 s` rather than consuming the full ceiling. The exact reason was `selected_contact_cumulative_slip_limit_exceeded`: module 2 accumulated `0.010156 m` against the unchanged `0.010 m` limit during lift. No object drop, unintended contact, actuator-envelope violation, or joint-limit violation occurred.
  - The previous large yaw reversal was removed during closure. Module 2 `yaw_dock_mech_joint2` progressed from about `-0.543 deg` to `-3.714 deg`; only sub-`0.008 deg/step` contact jitter occurred. Pitch command targets remained fixed; maximum measured pitch deflection was about `0.086 deg` through q_close and `0.357 deg` across the complete run.
  - Peak selected normal force was `9.493 N`, maximum penetration `0.0166 mm`, maximum maintained-contact slip speed `0.01597 m/s`, and AK40-10 actuator-envelope violations were zero.
- Assumptions: “Pitch fixed” means an absolute local position command under the existing finite drive, not a welded/kinematic joint. Small physical deflection under graph/contact load remains observable and is not hidden.
- Current issue: Closure and q_close now pass. The next isolated problem is cumulative tangential slip during the start of lift; lift/transport/place/release/settle and Order 8 acceptance remain incomplete.
- Next steps: Do not tune or relax the `10 mm` safety limit implicitly. First inspect the lift load-transfer/centroidal trajectory and the asymmetric selected-contact friction reaction using a short q_close-to-lift diagnostic before another full rollout.

### 2026-07-17 (P4-full Order 8 raised-support / CoM-only yield 30 s diagnostic)
- Spec version: `A-MSRR_codex_ready_spec_v0_4_ja.md` v0.4 plus the approved Order 8 natural-contact/simple-closure supplements in `AMSRR_design_modification_by_codex.md`.
- Work package / Agent label: Agent H/I/J/K/L boundary; free-object natural-contact fixture, centroidal yield, local Dock command/load observation, real-Isaac evidence, and diagnostic GUI replay.
- Status: Paused after the user-requested `30 s` diagnostic at a method-level choice. No Isaac/Kit process remains running. Proxy collision pads have **not** been added, and no two-contact grasp, downstream lift/transport/place/release, or Order 8 acceptance pass is claimed.
- Implemented control/safety correction:
  - Raised both the object work surface and commanded robot working height by `0.15 m`. The object remains a free dynamic rigid body and rests on a fixed support platform; it is neither constrained nor pose-written during rollout.
  - Added all-robot-body contact monitoring against both the floor and the object support. Any such contact during contact/manipulation phases enters immediate safe hold and is reported separately from intended selected-Dock/object contact.
  - Removed active all-Dock low-stiffness scheduling. Every Dock drive remains at the nominal diagnostic gains (`200 Nm/rad`, `5 Nms/rad`), and the QPID pose controller retains full P/I/D authority, including altitude and roll/pitch stabilization.
  - Restricted contact admittance to a bounded CoM translation along the horizontal selected-contact reaction axis. It applies no angular admittance and no vertical displacement, so it cannot disable gravity/height or attitude stabilization.
  - Contact/load arming now uses terminal-joint applied torque after subtracting the estimated virtual-drive damping contribution, together with selected-mesh proximity. Requested, raw applied, damping-estimated, and compensated loads are all reported.
  - The diagnostic fixture begins from a deliberately more open joint state generated by replaying the fixed closure velocity ratio backward for `3.0 s`; actual initial selected-mesh clearances were `30.584 mm` and `26.560 mm`.
- Schema/interface changes:
  - Advanced the internal Order 8 diagnostic config to `order8_natural_contact_config_v7` and added `object_support_height_m=0.15`.
  - Extended `CentroidalAdmittanceController.update` with optional world-axis force projection and angular-admittance disable controls.
  - Added diagnostic-only support/contact/load/admittance/state-trace evidence. No TaskSpec, MorphologyGraph, normal `PolicyCommand`, QPID command, learned-policy, or checkpoint contract changed.
- Verification:
  - Python compilation and `git diff --check` passed.
  - Focused Isaac-independent regression passed `189` tests; subsequent focused runtime/diagnostic regression passed `162`, and GUI/state-trace coverage passed `5`.
  - A `3 s` real-Isaac safety slice completed in `30.76 s` wall time. It preserved the opened geometry, had zero robot-floor/support contact, zero joint-drive-yield writes, nominal Dock gains, and zero controller/QP/actuator-envelope violation.
  - The requested real-Isaac run advanced the full `30.000 s` in `230.232 s` wall time. Report: `/tmp/order8_safe_open_closure_v307_30s.json`. It recorded `1,501` trace frames at nominal `50 fps`, hash `16eeba51cd729727257411d10416ceff4b8034d1bf69f8fa866d3a8cce3f81d1`, under `artifacts/p4_full/order8_natural_contact/diagnostics/safe_open_closure_v307_30s_state_trace.json`.
- `30 s` result:
  - Phase trace was `reset -> approach -> contact_acquisition`; `q_close` was never latched and contact-configuration dwell ended at `0`.
  - Both selected terminal-load/proximity triggers occurred at `5.69 s`. The fixed-ratio joint closure briefly brought both authored meshes to roughly millimetre-scale proximity and produced raw selected forces, but the simultaneous non-privileged arrest conditions persisted for only about one `0.01 s` step, below the unchanged `0.10 s` arrest dwell. The same monotonic fixed joint-space line then carried one side away; final clearances were `23.561 mm` and `89.873 mm`.
  - Peak selected-link normal force remained safe at `3.655 N`; raw evidence remained valid/unsaturated. The provisional pre-grasp slip peak (`0.1081 m/s`) is recorded but does not count as accepted-grasp slip because no grasp dwell was established.
  - Robot-floor/support contact count and unsafe-contact count were zero. QP infeasibility, controller failure, Dock actuator-envelope violation, and joint-limit violation were zero. Joint-drive yield remained inactive for all steps; minimum stiffness/maximum damping were exactly `200/5`. Minimum QPID PI scale was `1.0`.
  - CoM admittance became active after load/proximity detection but remained horizontal, with maximum translation offset `0.818 mm` and zero z offset. The free object translated only `0.169 mm` before q_close would have occurred and was not dropped.
- GUI inspection correction:
  - User inspection found that the former default state-trace replay displayed no visible Dock-joint motion. Offline audit of the real-physics trace nevertheless found substantial measured motion: module 1 `yaw_dock_mech_joint2` moved from `0.1776` to `0.7265 rad` (maximum span `31.45 deg`), module 2 `yaw_dock_mech_joint1` moved from `-0.1536` to `-0.5782 rad` (`24.34 deg`), and the relevant pitch joints moved about `13 deg`.
  - The previous replay verification compared the requested values with `robot.data.joint_pos` immediately after the same write. Because that cache is updated by the write operation, its zero error did not prove that the rendered articulated link transforms moved. The earlier statement that this verified GUI articulation motion is withdrawn.
  - `python3 scripts/order8_current_grasp_gui.py` now defaults to the exact `30 s` **live-physics** diagnostic in Kit and prints the recorded maximum Dock motion before launch. This is slower than wall clock but is the required path for judging physical joint/contact behavior.
  - The old fast path remains only as `python3 scripts/order8_current_grasp_gui.py --mode replay`. It is explicitly visual-only, acceptance-ineligible, and not sufficient for judging articulated physical behavior. `--refresh-trace` and `--capture-only` apply only to replay mode.
- GUI black-screen / warning correction:
  - The first live-GUI helper launch opened a black viewport because `AppLauncher` normalizes `--viz kit` into `args.visualizer=["kit"]`, while the runtime incorrectly inspected `args.viz`. Consequently the diagnostic close camera and stage light were skipped even though Kit was active. The runtime now recognizes the normalized visualizer destination (with a legacy fallback), and a captured real window confirms the lit three-module/object/support scene is visible from the intended close camera.
  - Repeated `GPU contact filter for collider ... is not supported` warnings came from using bare static floor/support colliders as PhysX GPU contact-filter targets for every robot body. Floor and support are now fixed kinematic rigid bodies: their mechanical role remains immovable, while the explicit robot-environment safety views use supported rigid-body filter targets. The repeated warning stream disappeared in the real GUI rerun.
  - Remaining terminal warnings are finite Isaac/host startup diagnostics (extension metadata, duplicate protobuf registration, CPU power profile, PCIe/IOMMU, renderer/TGS, and the pre-existing disjoint Dock-edge authoring notices); they are no longer emitted once per monitored robot body. Focused runtime/GUI/natural-contact regression passed `160` tests, compilation and `git diff --check` passed, and the real GUI rerun was intentionally interrupted after visual verification rather than completing the slow `30 s` diagnostic.
- Current interpretation pending corrected GUI inspection: Real-physics telemetry proves that Dock joints were commanded and their measured states changed substantially, while q_close was not acquired. The report is consistent with a fixed all-Dock velocity ratio crossing rather than dwelling in the simultaneous two-surface region, but the user's former GUI run did not display that physical motion because it used the insufficient replay path. Do not finalize the proxy-pad decision until the corrected live-physics GUI run confirms the visible kinematics.
- Exact resume point:
  1. Run `python3 scripts/order8_current_grasp_gui.py` and inspect the exact live-physics closure; do not use replay mode for this decision.
  2. Reconcile the visible motion with the recorded joint angles and selected-mesh clearances.
  3. Only then obtain the user's choice before changing the contact representation or closure method.
  4. If finite-area proxy pads are approved, first verify proxy two-contact/q_close in the minimum collision slice, then in the current free-base/support/QPID fixture.
  5. Only after safe q_close passes, run short force, lift, transport, place, release, and settle slices before the unchanged complete-environment Order 8 acceptance.

### 2026-07-16 (P4-full Order 8 paused at approved-simple-closure method decision)
- Status: Paused at a method-level decision as explicitly requested by the user. No Isaac/Kit/Order 8 process remains running. The uncommitted Order 8 worktree is preserved; no two-contact, downstream, or Order 8 acceptance pass is claimed.
- Approved boundary implemented in the current worktree:
  - Removed the active receding mesh-point / material-point IK path from contact closure.
  - Closure commands are measured-q receding joint commands with zero torque bias; load/current-equivalent remains the non-privileged q_close input, while raw Isaac contact remains privileged validation only.
  - Added direct joint-space release toward the measured open configuration captured at closure onset.
  - Added versioned report fields for the closure driver, fixed velocity ratio, open joint state, active steps, and release steps.
  - The current experimental closure driver evaluates whole-structure IK exactly once toward the known terminal anchor pose, normalizes that one joint-velocity ratio to `0.02 rad/s` maximum, then holds the ratio fixed. It does not re-solve IK and does not track mesh geometry during closure.
- Static verification:
  - `python -m py_compile amsrr/simulation/order8_isaac_runtime.py tests/unit/simulation/test_order8_isaac_runtime.py` passed.
  - `git diff --check` passed.
  - `tests/unit/simulation/test_order8_isaac_runtime.py`: `142 passed`.
  - Joint-controller/evidence/Order 8 focused companion set: `45 passed`.
- Fast real-Isaac evidence:
  - v298-v300 proved that selecting only the two terminal-joint signs from anchor or one-time mesh-point Jacobians is not valid for this whole structure; one or both selected surfaces move away from the object.
  - v301 used the explicit representative mechanism signs (`joint1` negative, `joint2` positive) from the earlier successful receding-IK direction. Terminal-joint-only closure still diverged because upstream Dock joints are required for grasp morphing.
  - v302 restarted from the prior v297 all-joint-shaped near-contact state. Terminal-joint-only closure still did not reach raw contact.
  - v303 used one fixed all-Dock velocity ratio from a single closure-onset IK solve. In the free-base/QPID diagnostic both sampled clearances initially decreased; the best observed region was approximately `0.46 mm / 1.17 mm`, but the same fixed joint-space line then moved away. At `6 s`, clearances were approximately `0.754 mm / 1.186 mm`, raw selected contacts remained zero, and q_close was not acquired.
  - v304 fixed both base and object to isolate joint geometry. The same fixed ratio monotonically increased the clearances; the run was deliberately interrupted after the result was conclusive rather than spending the full budget.
- Method-level blocker:
  - For the current authored Dock collision meshes and representative initial morphology, neither terminal-joint-only constant velocity nor a single fixed all-Dock velocity ratio intersects both object faces simultaneously. Further work now requires choosing a different method, not correcting a local bug or tuning force/friction/load thresholds.
  - Continuing with multi-segment/receding joint-space planning would reintroduce the grasp-motion-planning problem that the user explicitly said is not worth refining for Order 8 because learned pi_L will ultimately provide those joint targets.
- Recommended decision:
  - Prefer simple finite-area proxy collision pads rigidly attached to the two selected Dock links for the Order 8 controller/contact substrate smoke. Keep the authored Dock mesh visible and keep raw Isaac proxy contact privileged-only. This directly tests natural contact, joint-load q_close, centroidal yield/admittance, QPID, force/slip/penetration, lift/transport/place/release, and avoids claiming a general mesh grasp planner. The user previously stated that actual Dock collision need not be forced for this check, but explicit confirmation is required before implementing this method-level change.
  - Alternative: retain authored meshes and add a deterministic multi-segment whole-structure closure trajectory or a solved known q_precontact. This is less aligned with the latest scope because it spends implementation effort on the temporary motion planner.
- Exact resume point after user decision:
  1. If proxy pads are approved, add two thin finite-area collision primitives under the selected Dock rigid bodies with fully reported local transforms/materials; do not modify visual meshes, object pose during rollout, QPID inputs, or safety thresholds.
  2. Reuse the existing measured-q fixed-velocity/load-q_close path, first in a base/object-fixed collision-only slice, then free-base QPID contact acquisition.
  3. Once safe two-point contact/q_close passes, run short force, lift, transport, place, release, and settle slices before the unchanged complete-environment acceptance.
  4. Reconcile runtime, `order8_natural_contact.py` validator, fake-report method strings, design supplement, and final WORKLOG only after the selected method passes.

### 2026-07-16 (P4-full Order 8 paused at fixed material-point closure handoff)
- Status: Paused on user request. No Isaac/Kit/Order 8 process remains running. The existing uncommitted Order 8 worktree is preserved in place; no end-to-end or Order 8 acceptance pass is claimed.
- Progress since the preceding pause:
  - Removed false contact-yield/q_close triggers caused by upstream structural-joint saturation. Contact acquisition now uses each anchor's terminal Dock-mechanism joint load; influential whole-chain loads remain only for post-grasp structural stability/preload evidence.
  - Armed non-privileged load detection only after all-Dock contact-impedance blending settles, added per-anchor load dwell, and exposed requested/clipped/applied load-related telemetry.
  - Prevented position-servo preload from masquerading as contact by using a measured-q receding position reference throughout active closure. In v295, maximum target lead fell from about `0.0205 rad` to `0.000283 rad`, non-contact terminal loads stayed below about `0.1 Nm`, and false yield/q_close did not recur.
  - Added a bounded common selected-surface translation to the centroidal target and stopped external-wrench bias calibration once closure starts. This remained within the approved boundary: deterministic pi_L fallback emits CoM pose plus joint posture, while QPID remains unaware of joint dynamics.
  - Diagnosed a remaining kinematic inconsistency: the controller was recomputing the closest authored mesh sample every tick. Dock rotation can change that argmin, so differential IK may chase different physical points. Began replacing it with a single mesh material point latched in each mechanism-link frame at closure onset and transformed rigidly thereafter.
- Fast real-Isaac diagnostics:
  - v291/v294 isolated false load detection to structural-joint saturation and accumulated position-target lead.
  - v293 fixed-base and v295 free-base confirmed the false-yield correction.
  - v295 reached sampled clearances of approximately `0.91/1.16 mm` by `6 s` without raw contact.
  - v296 initially reached about `0.70 mm` on one side, then one clearance diverged to about `5.78 mm` after roughly `9 s`.
  - v297 ended near `0.817/1.175 mm` at `8 s`; the common centroidal correction was only about `0.2 mm` and did not resolve the remaining convergence defect.
  - The Isaac-independent focused suite passed before the latest material-point partial edit (`183` tests in the broad focused run). The partial material-point edit has not yet been compiled, unit-tested, or run in Isaac.
- Current problem:
  - Stable two-point raw contact is still not established. The strongest current hypothesis is discontinuous switching of the closest sampled mesh point during Dock rotation.
  - `amsrr/simulation/order8_isaac_runtime.py` already contains the partial local-frame material-point latch and several control-site substitutions, but its report fields/method version are unfinished.
  - `amsrr/simulation/order8_natural_contact.py` and `tests/unit/simulation/test_order8_natural_contact.py` still contain stale expected method strings for contact yield, contact centering, closure detection/provisional contact, and wrench-application mapping. Full validation must not be run until runtime, validator, and fake-report contracts agree.
  - Lift, transport, place, release, and settle have not been reached with the current acquisition path, so downstream physical defects may still exist.
- Exact safe resume procedure:
  1. Finish report evidence for the fixed control point: local/world point per anchor, maximum live-query-to-material-point distance, and a versioned fixed-material-point wrench/Jacobian mapping string.
  2. Reconcile all stale runtime/validator/fake-report method strings, then run `py_compile`, `git diff --check`, and the focused runtime/controller/evidence tests.
  3. Run a temporary `8-10 s` near-contact diagnostic as v298. Confirm that the fixed material point approaches monotonically, no non-contact yield/q_close reappears, and inspect whether live-query/material-point separation proves the argmin-switch hypothesis.
  4. If v298 reaches safe two-point contact, use short checkpointed slices for force dwell and then lift/transport/place/release/settle. If not, inspect the fixed-point task error before changing force, friction, acceptance thresholds, or the full environment.
  5. Only after the reduced slices pass, execute the unchanged three-module/all-Dock/authored-mesh/free-object complete-environment Order 8 acceptance and update the design supplement.
- Intended next diagnostic command after steps 1-2:
  `env CONDA_PREFIX=/home/leus/.local/share/mamba/envs/isaaclab3 /home/leus/.local/share/mamba/envs/isaaclab3/bin/python scripts/order8_contact_force_diagnostic.py --near-contact-fixture-report /tmp/order8_split_yield_admittance_v264.json --continue-after-force-ramp --object-width-padding-m 0 --dt 0.01 --speed-scale 2.0 --force-ramp-s 4.0 --contact-dwell-s 0.25 --dock-velocity-limit 0.1 --selected-gripper-friction 3.0 --normal-force-target-n 6.0 --max-simulation-time-s 10.0 --timeout-s 180 --report-path /tmp/order8_near_contact_material_point_v298.json`

### 2026-07-16 (P4-full Order 8 current-grasp wall-clock GUI handoff)
- Spec version: `A-MSRR_codex_ready_spec_v0_4_ja.md` v0.4 plus the approved Order 8 natural-contact and final-configuration-IK supplements in `AMSRR_design_modification_by_codex.md`.
- Work package / Agent label: Agent H/I/J/K/L boundary; Order 8 natural-contact diagnosis and visual inspection tooling.
- Status: Paused for the requested user GUI inspection before changing the grasp planner.  No Isaac/Kit process remains running.  The existing uncommitted Order 8 worktree is preserved; no end-to-end or Order 8 acceptance pass is claimed.
- Summary: Added a versioned, hash-bound, diagnostic-only Order 8 state trace; real-physics state capture; wall-clock kinematic Kit replay with late-frame dropping; strict graph/config/URDF/USD/module/joint-order binding; and `scripts/order8_current_grasp_gui.py`.  The helper automatically uses the existing `isaaclab3` micromamba environment, captures only if the trace is absent or explicitly refreshed, and defaults to three 1.0x GUI loops plus a five-second hold.  Replay advances no physics and is permanently acceptance-ineligible.
- Files changed: Added `amsrr/simulation/order8_state_trace.py`, `scripts/order8_current_grasp_gui.py`, and two focused test files; additive capture/replay integration in `amsrr/simulation/order8_isaac_runtime.py` and `scripts/p4_control_holon_spawn_probe.py`; updated the design supplement and this worklog.  All earlier Order 8 uncommitted files remain in place.
- Schema/interface changes: Added only internal `order8_diagnostic_state_trace_v1` and additive diagnostic probe flags/report fields.  No TaskSpec, MorphologyGraph, policy, QPID, actuator, checkpoint, or acceptance contract changed.  The replay trace cannot be supplied as acceptance evidence.
- Real-physics capture: Re-ran the exact v246 precontact fixture (`0.2 rad/s` temporary Dock velocity limit, `dt=0.01 s`) and recorded `10.45 s`, `524` frames at nominal `50 fps`, `1,616,955` bytes, trace hash `b8e72fed097b304223aab2e019e5c9376e929bbd51b44c4b87b31bf0e64c0b8f`.  The final physical state reproduces the current defect: simultaneous contact configuration, force scale `0.03`, selected forces `30.287 N / 1.070 N`, followed by safety `safe_hold`.  The ignored local trace is `artifacts/p4_full/order8_natural_contact/diagnostics/current_grasp_v246_state_trace.json`.
- Replay verification: A real Isaac headless replay of the same trace at `20x` passed, replayed the `10.45 s` source in `0.5256 s`, and reported `replay_advances_physics=false`; `196/524` frames were rendered and `328` intentionally dropped to preserve wall-clock timing.  This verifies the runtime replay path, not GUI appearance or contact physics.  The user-facing 1.0x Kit observation remains the next action.
- GUI visibility follow-up: The first user observation appeared static.  Trace audit showed that the current v246 differential-IK closure really changes the largest Dock joint by only `0.083402 rad` (`4.78 deg`) over `10.45 s`; the selected arm joints otherwise move approximately `0.050-0.061 rad`, so the former distant camera hid most of the motion.  Replay now uses a close grasp-centred camera, holds the first and last poses for `1.5 s` by default, prints the recorded maximum Dock delta, and audits measured post-write robot root/joint/object state on every rendered frame.  A fresh real-Isaac headless replay measured exactly zero joint/root/object write error, proving the recorded states are being applied; the small amplitude belongs to the current controller behavior rather than a replay write failure.
- Tests/commands: Focused state-trace/current-GUI/runtime/diagnostic suite passed `140`; focused new files passed `5`; Python compilation, CLI help, command dry-run, trace reload/hash validation, and `git diff --check` passed.  The first capture launch failed before physics because the normal interpreter selected an old Isaac-side Python without `tomllib`; the helper now explicitly enters `isaaclab3`.  The next launch exposed and fixed a function-local `Path` name collision before physics; neither failed launch produced or altered accepted evidence.
- Current problem: The current controller still uses a receding differential-IK target rather than a globally solved final collision-aware `q_grasp`.  v246 reaches the current contact event too energetically.  The prior v247 handoff instruction to merely extend the run is superseded: lowering only Isaac's velocity ceiling to `0.1 rad/s` while leaving controller/IK motion limits unchanged created target backlog and position-drive saturation, so controller and simulator limits must be made consistent if that diagnostic is revisited.
- Approved pending correction: After the user inspects this baseline, implement explicit simultaneous whole-structure final grasp configuration solving, a bounded `q_open -> q_precontact` trajectory, final millimetre-scale natural closure, measured `q_close` hold, and moving-object re-solve.  Preserve all force/friction/penetration/slip/controller/AK40-10 gates and the free-object/no-pose-write invariant.
- User command: `python3 scripts/order8_current_grasp_gui.py`.  The existing trace makes this a GUI-only launch; use `--refresh-trace` only to deliberately repeat the slow physical capture.  `--speed`, `--loops`, and `--keep-open-s` control visual playback.
- Blockers/open questions: No implementation blocker.  Human GUI judgement of the current motion is intentionally required before the approved planner correction resumes.

### 2026-07-15 (P4-full Order 8 paused after v247 low-speed acquisition A/B)
- Status: Paused on user request with the uncommitted Order 8 worktree preserved in place. No Isaac/Kit diagnostic process remains running. No end-to-end or Order 8 acceptance pass is claimed.
- Progress since the previous pause: Diagnosed v245 as a circular stop condition: `q_close` required low relative velocity while the position target was still advancing. Implemented a two-stage non-privileged gate: simultaneous terminal-Dock load/current-equivalent observation + sampled authored-mesh proximity + componentwise `+/-50 mm` surface-region membership causes same-cycle measured `q_close`; filtered settling, force ramp, contact dwell, and privileged raw-contact safety remain separate post-arrest requirements. Raw contact is still excluded from actor/QPID/arrest inputs. Focused runtime/diagnostic tests pass `135`; compilation and `git diff --check` pass.
- Real diagnostic evidence: v246 (`/tmp/order8_pair_center_arrest_gpu_v246.json`, temporary Dock velocity ceiling `0.2 rad/s`) reached simultaneous `q_close` and emitted a complete exact checkpoint, with QPID and AK40-10 envelope audits passing. It reduced the former `39.926 N` runaway but residual joint motion raised peak selected force to `30.287 N` at force scale `0.03`, exceeding the unchanged `30 N` hard gate by `0.287 N` (`0.098 mm` maximum penetration). v247 (`/tmp/order8_pair_center_arrest_vlimit01_gpu_v247.json`, `0.1 rad/s`) stayed well inside the force gate (`0.962 N` peak), penetration gate (`0.064 mm`), QPID, and AK envelope, but its `11.1 s` temporary budget ended just before simultaneous closure; terminal sampled clearances were about `3.001/5.303 mm`, so it is safe incomplete evidence rather than a failure of the new arrest rule.
- Current problem: The reduced-speed candidate needs a slightly longer acquisition budget to reach q_close. It has not yet proved stable force scale `1.0`, lift, transport, place, release, or settle. v246's exact checkpoint is dynamically too energetic for downstream validation and must not be reused as success evidence.
- Exact resume point: Do not add another controller change first. Rerun the v247 fixture unchanged except extend `--max-simulation-time-s` from `11` to approximately `14`; if it q_closes below `30 N`, use its exact checkpoint for short force-ramp and lift-only continuations. If repeated precontact A/B becomes necessary, implement a generic pre-impact full-state checkpoint/restart before further slow runs. Only after reduced force/lift/transport/place/release slices pass should the unchanged three-module/all-Dock/authored-mesh/free-object acceptance sequence be run.
- Remaining-time estimate: approximately `8-16` focused working hours if the `0.1 rad/s` acquisition reaches a safe checkpoint and no new lift/release defect appears: `1-3 h` for safe q_close/force verification, `3-7 h` for lift through settle slices, and `4-6 h` for unchanged full-environment verification, regressions, and documentation. New downstream physical defects would increase this estimate and must be reported rather than hidden by relaxed gates.

### 2026-07-15 (P4-full Order 8 resumed with moving-object surface-region acquisition)
- Status: In progress in the existing uncommitted Order 8 worktree. No acceptance or end-to-end natural-contact pass is claimed yet. Debug remains minimum-fixture-first; the unchanged three-module/all-Dock/authored-mesh/free-object environment is reserved for final verification.
- Approved method correction implemented so far: Reach the final object-relative centroidal pose with all Dock meshes open, then close all relevant Dock joints simultaneously. Treat each nominal object contact as a componentwise `+/-0.050 m` tangential surface region; allow provisional contacts to slide, separate, and reacquire before the verified two-contact grasp dwell. Retarget the centroidal and anchor goals from the measured free-object pose without object pose writes or constraints. `q_close` holds only the measured articulated shape; maintained-contact slip/cumulative-slip/contact-break enforcement starts after the stable two-contact dwell and continues until planned release. Force, torque, penetration, unintended-contact, raw-truth, controller, and actuator hard gates remain active throughout acquisition.
- Implementation/evidence changes: Added config v3 tangential tolerance and a `0.015 m` near-surface slowdown; object-follow and surface-region target helpers; simultaneous joint-only closure and simultaneous `q_close`; typed q_close visibility in observation/evidence; separate provisional-acquisition versus maintained-contact slip telemetry; updated fail-closed report validation and the design supplement. Superseded one-sided world arrest, centroidal recentering, and alternating reacquire counters must remain zero.
- Verification: Focused schema/evidence/runtime/wrapper tests pass `85`. Real fast diagnostics v115-v121 isolated the sequence without running lift/transport/release: fixing the precontact fixture and base-settle gate produced true base-first/joint-second closure; widening the slowdown boundary from `0.006 m` to `0.015 m` reduced the initial two-contact slip from about `20.1 mm/s` to `3.4 mm/s`. The subsequent one-frame provisional separation reached about `21.3 mm/s`, exposing the obsolete pre-grasp slip hard-failure semantics now corrected. Kp `50` was too slow and Kd `5` worsened the transient; production remains Kp `200`, Kd `2`, with observed torque/speed below AK40-10 constraints.
- Exact resume point: Run the same short free-object precontact fixture with the corrected evidence semantics. It must recover from provisional separation, reach simultaneous `q_close`, and establish the stable two-contact dwell without exceeding force/torque/penetration/controller/actuator hard limits. Diagnose only that reduced sequence until it passes; then run the broader Isaac-independent matrix, complete force ramp in a short fixture, and only afterward attempt the unchanged full lift/transport/place/release/settle environment and independent acceptance.

#### 2026-07-15 continuation (authored-mesh precenter A/B and lift-first resume point)
- Implemented diagnostic branch: Contact-region offsets are now measured at the selected authored collision-mesh samples rather than at Dock connect frames. Added a collision-clear `15 mm` mesh-precenter dwell, componentwise `+/-50 mm` tangential-region targets, object-relative retargeting, symmetric normal creep, and a bounded horizontal pair-mean base correction. These remain uncommitted and are not accepted production behavior yet.
- Verification: The focused Isaac-independent runtime/wrapper set passed `130` tests before the latest real runs. Real v238-v240 used only the post-axial precontact fixture. They proved the clear precenter and symmetric creep, but the pair-centred geometry produced a two-contact impact (about `20 N` total in v240), transient base/angular motion, one QPID infeasible step, and two AK velocity-envelope audit violations. Raising simulator Dock damping from `5` to `20 Nms/rad` did not solve it and is rejected as the primary correction.
- Reassessment: The earlier v227 exact checkpoint already reached simultaneous `q_close` and stable two-contact acquisition with production damping `5`, no controller failure, and the AK40-10 torque/current/speed audit passing. The v234 continuation from that checkpoint remained controller/actuator safe and retained both contacts; it stopped after only `3 s` while the object COM had risen about `52.7 mm`, so failure to reach the `100 mm` bottom-clearance gate does not yet prove a lift-control defect.
- Exact resume point: Use the v227 exact `q_close` checkpoint in the short fixture and extend only the lift time. If lift succeeds, discard or narrow the pair-centering experiment and restore the known-good acquisition geometry while retaining the approved moving-object/contact-region semantics. Then test transport/place/release from short checkpoints before the single unchanged full-environment acceptance run.

#### 2026-07-15 continuation (v245 force-spike diagnosis and same-cycle q_close arrest)
- Diagnosis: The velocity-limited v245 precontact fixture kept QPID and the AK40-10 torque/current/speed envelope valid, but the first simultaneous physical contact was not allowed to stop because `q_close` required filtered low relative velocity before holding the joints. The position targets therefore continued advancing and selected contact force rose over successive `0.01 s` steps from about `8.9 N` to `14.5`, `21.2`, `26.5`, `28.8`, then `39.9 N`; penetration remained only `0.141 mm`. This is a stop-condition control deadlock, not a QPID, torque-envelope, or gross PhysX-penetration failure.
- Implementation: Split the non-privileged `q_close` arrest event from the later stable-grasp proof. The first control cycle with simultaneous selected terminal-Dock load/current-equivalent observation, authored-mesh proximity, and componentwise `+/-50 mm` tangential-region membership now snapshots the measured base and complete Dock joint state. Filtered settling, force ramp, contact dwell, and privileged two-contact safety authorization remain independent post-arrest gates before payload motion. One-sided provisional contact is not latched and remains free to separate/reacquire; raw Isaac contact truth remains excluded from the arrest decision and QPID.
- Verification: Focused runtime/diagnostic tests pass `135`; Python compilation and `git diff --check` pass. Real-Isaac verification of the new arrest condition is pending; no acceptance claim is made by this entry.
- Exact resume point: Rerun the v245 post-axial precontact fixture with the `0.2 rad/s` temporary Dock velocity ceiling. It should latch before the `30 N` force gate and emit an exact q_close checkpoint. Use that checkpoint for short force/lift A/B runs; do not repeat the full approach path unless the reduced checks pass.

### 2026-07-15 (P4-full Order 8 paused at exact q_close checkpoint handoff)
- Status: Paused immediately on user request. No Isaac/Kit process remains running; the large Order 8 worktree is still uncommitted and must be resumed in place.
- Implemented in this slice: Restored the empirically successful v57 per-anchor arrest/alternating-reacquire logic from the local Codex session record; added a diagnostic q_close checkpoint path and diagnostic-only Dock-stiffness override. Removed the invalid sampled-mesh-clearance centering calibration. Began an additive exact-checkpoint parser (`_QCloseCheckpointState`) for full simulator-state restoration, but it is deliberately **not yet connected** to the runtime/CLI.
- Verification: Focused Isaac-independent tests passed `74`; Python compilation and `git diff --check` passed. Real diagnostic `/tmp/order8_qclose_checkpoint_v77.json` reproduced the old successful q_close at simulation time `16.06 s` with no selected contact and sampled clearances approximately `2.699 mm / 0.218 mm` (wall time `162.7 s`).
- Current problem: The first fast restart `/tmp/order8_qclose_kp200_v78.json` used only base pose plus Dock positions. Graph-FK reconstruction did not preserve the small constraint-loaded offsets of all three articulations; reset produced an initial collision/transient, moved the free object, and opened the clearances. Therefore v78 is invalid for stiffness comparison and no Kp conclusion may be drawn from it.
- Exact resume point: Finish wiring the exact diagnostic checkpoint state through `scripts/order8_contact_force_diagnostic.py` and `scripts/p4_control_holon_spawn_probe.py`; capture/restore every module articulation root pose/twist, global Dock q/qdot, free-object pose/twist, and measured anchor-hold poses. Add parser/report tests and rerun the focused set. Then perform one final slow precontact run to generate the exact state, prove restart equivalence at force scale near zero, and only then compare Kp candidates in the 10-40 s fast fixture. Do not run another full sequence until a fast candidate passes through force scale 1.0.

### 2026-07-14 (P4-full Order 8 paused after real-Isaac v52 diagnostic)
- Spec version: `A-MSRR_codex_ready_spec_v0_4_ja.md` v0.4 plus the frozen and implementation-clarified Order 8 supplement in `AMSRR_design_modification_by_codex.md`.
- Work package / Agent label: Agent H/I/J/K/L; deterministic natural-contact planner, full-Dock local control, real-Isaac runtime/evidence, wrapper, and acceptance.
- Status: Paused on user request with no simulator process left running.  The Order 8 worktree remains uncommitted.  v52 was intentionally interrupted during force-ramp diagnosis and produced no report, so it is not acceptance evidence; Order 8 end-to-end completion is not claimed.
- Implemented in this resumed slice: The representative symmetric three-module morphology now searches compatible Dock-edge port assignments and deterministically selects a mesh-backed, exactly opposed lateral gripper pair (current selected global port ids `7/10`) instead of accepting the previous oblique pair.  Planned contact force is now applied at the nearest authored Dock-mesh surface sample and power-equivalently shifted to the connect-frame Jacobian as `W_connect = [F, M + (p_mesh - p_connect) x F]`.  The mesh application point comes only from authored collision geometry plus observed body/object poses; raw PhysX contact truth remains safety/diagnostic-only.  Added fail-closed report evidence for this mapping and regressions for topology, inward virtual work, wrench shift, finite/exact application-point coverage, and deterministic surface-sample choice.
- Verification: The focused runtime/wrapper set passed `47` tests.  The broader Isaac-independent Order 8/morphology/controller/policy/schema/acceptance matrix passed `130` tests in `2.61 s`; Python compilation and `git diff --check` passed.  Real v52 completed approach, axial insert/settle, balanced full-Dock close, both per-anchor dynamic `q_close` arrests, and simultaneous reacquire without QP, force, slip, penetration, or joint-limit failure before the manual interrupt.
- v52 diagnostic state at interrupt: At simulation time `57.100 s`, phase was `contact_acquisition/force_ramp_hold`, force scale `0.288`, sampled surface clearances were approximately `1.449 mm / 0.473 mm`, selected raw contact count and measured contact force were still zero, and slip/penetration/joint-limit violations were zero.  The shifted-wrench mapping improved the farther gap from about `2.65 mm` at ramp start to `1.45 mm`, but it then plateaued from roughly force scale `0.238-0.288`; the nearer gap remained about `0.47 mm`.  This is progress over v51 but not yet a natural-contact pass.
- Current problem: The remaining gap plateau is most likely at the interaction between fixed measured `q_close` position targets, position-servo stiffness, shifted torque bias, and per-joint effort limiting.  That hypothesis is not yet proven because the report currently lacks per-joint requested/clipped/applied torque-bias telemetry at the force ramp.  Safety thresholds and privileged-contact boundaries must not be relaxed to hide it.
- Exact resume point: Add compact force-ramp telemetry for per-joint requested torque bias, post-limit torque bias, position-reference error/servo contribution where Isaac exposes it, and mesh moment arm/shifted wrench.  Cover the telemetry with unit/report-validation tests, rerun the `130`-test matrix, then use a deliberately short diagnostic-only real run to locate contact onset or torque saturation before spending another full `40 s` ramp.  Once the mechanism is corrected, rerun the unchanged full ramp through lift/transport/place/release/settle, wrapper validation, independent acceptance, and GUI inspection.

### 2026-07-14 (P4-full Order 8 real-Isaac diagnosis resumed)
- Spec version: `A-MSRR_codex_ready_spec_v0_4_ja.md` v0.4 plus the frozen and implementation-clarified Order 8 supplement in `AMSRR_design_modification_by_codex.md`.
- Work package / Agent label: Agent H/I/J/K/L; deterministic natural-contact planner, full-Dock local control, real-Isaac runtime/evidence, wrapper, and acceptance.
- Status: In progress from the v16 handoff.  The Order 8 worktree remains uncommitted.  Real-Isaac v24 is currently executing the representative three-module/free-object rollout; no end-to-end acceptance is claimed until it reaches `complete`, the wrapper reports no validation failures, and independent artifact acceptance passes.
- Implemented since v16: Added a distinct `0.003 m` sampled-mesh `q_close` arming gate and final creep tier; per-anchor arrest/latch history; full anchor-orientation task weights; zero unilateral force while the other side is still being acquired; simultaneous non-privileged reacquisition before the force ramp; staged whole-shape freeze; and bounded whole-structure clearance balancing.  The centering target uses only the difference of the two sampled authored-mesh clearances.  Because the current one-axis-vectoring Holon cannot translate along the representative world-`y` correction while exactly level, the centroidal target now converts the existing QPID horizontal pose/velocity error into bounded roll/pitch (`<=0.020 rad`) while retaining the `0.030 m` recenter-offset limit.  Raw Isaac contact truth remains safety/diagnostic evidence only.
- Diagnostic evidence: v18 demonstrated that applying force after only the first contact causes excess slip, so unilateral force was removed.  v20 exposed insufficient orientation weighting and v21 showed the opposite side stalled roughly `44 mm` away under Dock-only morphing.  v22 computed the expected approximately `20.8 mm` centroidal correction but joint motion conflicted with it.  v23 froze the Dock shape and reduced anchor-pose error from roughly `100 mm` to `1-5 mm`, but fixed-level lateral motion remained underactuated.  These are fail-closed diagnostic runs, not acceptance artifacts.
- Verification: The latest focused helper/runtime/schema matrix passed `39` tests.  The complete Isaac-independent Order 8-related matrix passed `113` tests in `1.75 s`; Python compilation and `git diff --check` passed before the v24 launch.  The live v24 run has so far remained within force/slip/penetration/joint limits; its terminal result is pending.
- Runtime/learning handoff: The observed approximately `0.03-0.06x` real-time factor belongs to the high-fidelity three-articulation/authored-mesh/per-patch acceptance environment.  It must not become the Order 9 per-update training path.  Order 9 needs vectorized GPU environments, cached assets, reduced-cost aggregate/proxy contact curriculum, and periodic unchanged full-mesh evaluation; fast-path evidence cannot be relabelled as Order 8 acceptance.
- Next steps: Finish v24 and verify that bounded tilt produces measured centroidal recentering.  If it passes end to end, run the independent report acceptance and prepare the GUI command.  If it fails, preserve the unchanged safety thresholds, diagnose from the hash-bound report, make the smallest controller/planner correction, rerun the focused tests, and repeat the real-Isaac smoke.

### 2026-07-14 (P4-full Order 8 paused after real-Isaac v16 diagnostic)
- Spec version: `A-MSRR_codex_ready_spec_v0_4_ja.md` v0.4 plus the frozen Order 8 natural-contact supplement in `AMSRR_design_modification_by_codex.md`.
- Work package / Agent label: Agent H/I/J/K/L; deterministic natural-contact planner, full-Dock local control, real-Isaac runtime/evidence, wrapper, and acceptance.
- Status: Paused at a clean diagnostic boundary at the user's request.  No simulator process is intentionally left running, no commit was made, and the attempted v16 correction described below has **not** yet been implemented.  Resume from the current uncommitted worktree.
- Implemented so far: Versioned Order 8 config/observation/evidence/result contracts; exact three-module/two-anchor morphology; actual Dock collision-mesh surface selection; free `1 kg` object with no pose slaving or robot constraint; all-Dock whole-structure kinematics and independent absolute position/velocity/torque-bias control; centroidal-only QPID integration; deterministic approach/grasp/lift/transport/place/release planner; privileged per-patch Isaac contact evidence and fail-closed safety monitor; hash-bound real/dry wrapper, CLI/GUI flags, and acceptance; plus runtime corrections for module-frame/root-frame targets, measured axial-hold reconciliation, per-anchor dynamic `q_close`, sampled mesh-surface distance, and global near-surface slowdown.
- Verification completed: Focused schema/runtime tests after the latest surface-distance/slowdown changes passed `34` tests.  The broader Order 8-related command emitted passing progress for all `108` collected test positions, but its terminal pytest summary was not recovered, so it must be rerun before a final claim.  `git diff --check` passed.  Real-Isaac v16 ran to a deterministic fail-closed terminal and wrote `/tmp/order8_real_report_staging_v16.json`; it is diagnostic failure evidence, not acceptance.
- Current problem: v16's sampled mesh-surface estimator overestimated the first physical contact gap by about `2 mm`, while the nominal per-anchor `q_close` arming gate reused the `0.1 mm` penetration noise floor.  Therefore the first side was not latched when a safe initial selected contact appeared at simulation time `76.160 s` (`0.502 N`, `0.0007 m/s` slip).  The position command continued to close and at `76.820 s` produced `9.444 N` with `0.0422 m/s` slip, exceeding the unchanged `0.020 m/s` hard limit and correctly entering `safe_hold`.  No grasp/lift/transport/release acceptance is claimed.
- Resume action: Add a distinct, versioned non-privileged sampled-surface `q_close` arming distance of `0.003 m` (rather than reusing the penetration noise floor), and add a final creep-speed tier inside that distance.  Bind both values/method versions into report validation and tests.  Do **not** relax force, slip, penetration, dwell, or other acceptance thresholds; raw contact truth must remain diagnostic/safety-only.  Then rerun focused tests, real-Isaac headless, artifact acceptance, and only after a headless pass prepare the GUI diagnostic.
- Files/state: The Order 8 implementation and tests remain uncommitted in the paths shown by `git status`; existing user/earlier work is preserved.  `AMSRR_design_modification_by_codex.md` still contains the earlier object-size typo (`0.30 x 0.20 x 0.15 m`) and must be corrected to the active config value `0.30 x 0.40 x 0.15 m` during closeout.

### 2026-07-13 (P4-full Order 8 natural-contact implementation and final contract audit)
- Spec version: `A-MSRR_codex_ready_spec_v0_4_ja.md` v0.4 plus the frozen Order 8 supplement in `AMSRR_design_modification_by_codex.md`.
- Work package / Agent label: Agent H/I/J/K/L; deterministic contact planner, full-Dock local control, free-object Isaac runtime, privileged contact evidence, wrapper/provenance, and acceptance.
- Status: The versioned Order 8 substrate, fail-closed wrapper, CLI, and artifact acceptance are implemented and unit/acceptance-tested.  The producer now applies and reports the requested seed across Python random, Torch, applicable CUDA, and NumPy, and performs the independent free-object audits required by the strengthened wrapper.  No real-Isaac natural-contact pass or GUI success is claimed by this entry.
- Implemented: Exact current symmetric two-anchor three-module morphology; actual Dock mesh surface selection with force-regenerated `Convex Decomposition`; a free dynamic box with no kinematic attachment/pre-contact hold; deterministic reset/approach/contact-acquisition/lift/transport/place/release/retreat/settle/complete planning; per-patch selected/unintended contact measurement; phase-sensitive dwell, load, slip, penetration, contact-break, drop, release, retreat, and settle monitoring; whole-structure full-Dock Jacobians; absolute Dock position/velocity plus bounded torque-bias commands; payload feed-forward accounting; live phase progress; dry/real wrapper and GUI/realtime/hold flags; and hash-bound acceptance/no-mislabeling.
- Final contract strengthening: Acceptance now requires applied-seed evidence, exact measured `simulation_dt`, `requested_steps == ceil(rollout_budget/simulation_dt)`, local source-URDF/collision regeneration, local generated-URDF/USD/bundle rehashing, exact global Dock-joint identity equality across observation and all three command channels, full-Dock Jacobian coverage, both selected anchor IDs, empty raw-contact failure reasons, and terminal monitor metrics within the active config.  Counts or self-reported top-level booleans alone cannot pass.
- Policy boundary: Raw Isaac force/penetration/slip patches remain privileged diagnostic/acceptance evidence.  They are not actor input, nominal planner feedback, a `PolicyCommand` field, or a QPID contact-wrench target.  A raw-evidence hard failure may only force safe hold; it cannot generate nominal actuator intent or convert a failed episode into success.
- Object boundary: Static runtime audit found no object pose write after the spawn `InitialState` and no object-to-robot constraint authoring; only occupied robot DockEdges receive external FixedJoints.  The strengthened acceptance additionally requires an instrumented post-spawn object-pose-write counter and an independent final-stage scan of all `UsdPhysics` Joint `body0`/`body1` targets, with zero references to the object subtree and an empty offending-prim-path list.
- Primary files: `amsrr/schemas/order8.py`, `amsrr/robot_model/gripper_surfaces.py`, `amsrr/robot_model/whole_structure_kinematics.py`, `amsrr/controllers/natural_contact_joint_controller.py`, `amsrr/policies/deterministic_natural_contact_planner.py`, `amsrr/simulation/order8_contact_measurement.py`, `natural_contact_evidence.py`, `order8_isaac_runtime.py`, `order8_natural_contact.py`, `amsrr/acceptance/order8_acceptance.py`, `configs/training/order8_natural_contact.yaml`, `scripts/order8_natural_contact.py`, and the Order 8 probe integration/tests.
- Verification: The complete Isaac-independent Order 8 matrix passes `83` tests with host pytest plugin auto-loading disabled; focused wrapper/acceptance coverage is `9` passing tests.  `py_compile`, Black check for the wrapper/acceptance scope, and repository diff whitespace validation pass.  Real Isaac and GUI execution remain separate required evidence; dry-run or fake reports are permanently ineligible for acceptance.
- Handoff: Run headless real Isaac and inspect any fail-closed report before changing thresholds or control.  Only a report with empty wrapper validation failures may be passed to `run_order8_acceptance`; afterward use `--real --viewer kit --realtime-playback --keep-open-after-rollout-s ...` for visual diagnosis.  Keep Order 9 learned/full-TaskSpec delivery and Order 10 P4-full acceptance separate.

### 2026-07-13 (P4-full Order 8 object natural-contact smoke started)
- Spec version: `A-MSRR_codex_ready_spec_v0_4_ja.md` v0.4 plus the user-approved Order 8 natural-contact smoke supplement recorded in `AMSRR_design_modification_by_codex.md`
- Work package / Agent label: Agent H/I/J/K/L boundary; selected contact planning/runtime integration, articulated local-joint control, real-Isaac natural-contact environment, evidence logging, and acceptance
- Status: In progress.  The method contract and numeric smoke gates are frozen; no Order 8 implementation, real-Isaac acceptance result, or completed-test claim is recorded by this entry.
- Summary of changes: Defined a deterministic/controlled natural-contact substrate smoke using a free `1.0 kg`, `0.30 x 0.40 x 0.15 m` box (`mu_object=0.6`, `mu_floor~=0.8`) and actual Dock-mechanism collision meshes.  Removed kinematic slaving/pre-contact pose hold/fixed object attachment from the acceptance model; required at least two selected contacts on distinct Dock links; and separated this smoke from Order 9 learned full-TaskSpec delivery.
- Frozen thresholds: selected contact normal force `>=0.5 N`; simultaneous continuous contact dwell `0.25 s`; nominal normal-force target approximately `11 N/contact`; hard force `<=30 N/contact`; hard contact torque `<=5 N m/contact`; penetration `<=0.002 m`; tangential slip speed `<=0.02 m/s`; accumulated tangential slip `<=0.010 m`; contact-break grace `0.05 s`; object-bottom lift clearance `>=0.100 m`; short transport `0.200 m`; release contact-free dwell `0.10 s`; gripper-retreat clearance `>=0.050 m`; and post-release object speed `<=0.05 m/s` linear plus `<=0.10 rad/s` angular continuously for `1.0 s`.
- Safety/phase contract: Only configured selected Dock-link/object pairs are intended; meaningful non-selected robot-object contact is unintended.  Drop is evaluated only after lift and before intended place from floor recontact, sufficient fall, or required-contact loss beyond grace; floor contact during intentional place/release is expected.  Controller/QP feasibility and complete, resolved, unclipped actuator targets remain mandatory.
- Dock-joint invariant: Every Dock joint remains a physically movable articulation DOF and remains observed and commandable through complete position/velocity/torque-bias channels.  No weld, lock, zero-range substitution, deleted DOF, or fixed-transform shortcut is allowed.  Any diagnostic command/policy mask is non-structural, explicitly reported, and disabled for acceptance; upstream morphology joints are not excluded merely because they are not adjacent to the selected contact link.
- Files changed: `for_codex/AMSRR_design_modification_by_codex.md`; `for_codex/WORKLOG.md`
- Schema/interface changes: None in this documentation-only start.  Any later Order 8 schema/report additions must be additive and versioned; normal `PolicyCommand` and QPID remain free of contact/internal-wrench commands, and raw Isaac contact truth remains privileged critic/reward/diagnostic evidence rather than a normal actor input.
- Upstream dependencies used: v0.4 Sections 22.4, 23, 24.5, 25, and 26; completed Orders 1-4/2.5 and Orders 5-7 fallback acceptance; current Holon URDF/PhysicalModel and actuator mapping; existing contact-candidate/trajectory/controller/archive interfaces; user-approved numeric and articulation constraints.
- Downstream impact: Order 9 may use the accepted natural-contact substrate for staged BC/RL and varied TaskSpec delivery; Order 10 may aggregate only hash-bound real-Isaac Order 8 evidence.  Neither may relabel kinematic P4.2, dynamic module docking, debug-mask, or structural-joint-lock evidence as natural-contact success.
- Tests added or run: None for implementation or simulation at this in-progress documentation stage.  Documentation whitespace/diff validation is the only intended check.
- Commands run: Re-read `AGENTS.md`, relevant v0.4 contact/reward/P4/backend/ownership sections, the current design supplement, and existing WORKLOG handoffs; inspected repository status and Order 8 references; run `git diff --check` after this edit.
- Assumptions: The `11 N/contact` value is a nominal frictional no-slip target with approximately `25%` margin for the baseline object, not the hard-force acceptance ceiling.  Exact Isaac per-patch force/impulse/penetration/slip truth is privileged.  A diagnostic mask changes commands only and never the physical articulation topology.
- Blockers / open questions: No method-level blocker remains for starting implementation under the frozen baseline.  The implementation must make its phase-sensitive sufficient-fall predicate and meaningful-contact noise floor explicit/configuration-backed without weakening the frozen gates; no value is invented in this documentation entry.
- Next steps: Add versioned Order 8 config/result/evidence contracts and fail-closed validation first; implement free-object/Dock-mesh spawning and per-link contact evidence; integrate articulated full-chain joint commands and deterministic approach/grasp/lift/transport/place/release execution; add fake/unit no-mislabel tests; then run a separate real-Isaac headless smoke and GUI diagnostic before making any acceptance claim.

### 2026-07-13 (P4-full Orders 5-7 dynamic assembly fallback accepted)
- Spec version: `A-MSRR_codex_ready_spec_v0_4_ja.md` v0.4 plus the approved Orders 5-7 dynamic-assembly and funnel-mating supplements
- Work package / Agent label: Agent B/G/I/J/K/L boundary; deterministic `pi_A` component motion, exact-frame dynamic attach, unload-gated detach, and real-Isaac round-trip evidence
- Status: Order 5 remains complete. Orders 6 and 7 pass their independent real-Isaac gates under `selected_pair_collision_filter_fallback_v1`, which is now the configuration, dataclass, and CLI-helper default. `physical_funnel_contact_v1` remains an explicit non-accepted diagnostic override and is not required for the active path.
- Primary source commits: `51bc152` (`[P4-full][Order5] Refine assembly approach control`) and `0180504` (`[P4-full][Orders6-7] Add dynamic dock roundtrip`). The final live-heartbeat refinement is kept with this closeout because it changes only runtime observability, not acceptance semantics.
- Summary: Completed the two-single-Holon floor/takeoff/hover/staging/prealignment/axial-approach path, exact selected-connect-frame FixedJoint creation and identity verification, bidirectional controller handover, follower-subtree unload gating, exact removal, measured separation, delayed collision-filter removal, and continuous post-release hover. In the default mode only the selected pitch/yaw Dock rigid-body pair is filtered, the filter is applied and verified during prealignment before axial physics begins, and every other body/environment collision remains enabled. Closure requires zero selected-pair contact plus the unchanged strict final pose/twist/dwell gate; it never claims physical funnel guidance contact.
- Separation correction: Added a measured `DynamicSeparationLifecycle`. It holds the final separation target beyond the nominal 4 s ramp when needed, removes the exact-pair filter only after both nominal duration and actual gap/clearance gates pass, verifies removal ownership, and then requires a resettable continuous 1 s stable window within a 2x acquisition budget. This fixes the earlier run that reached sufficient clearance only during the post-ramp hold and therefore never rechecked filter removal.
- Runtime numerics: Dynamic assembly uses PhysX solver iterations `8/8`, Dock implicit-drive stiffness/damping `200/2`, and explicit simulation effort/velocity limits `4.1 Nm / 3.0 rad/s`. The axial evidence contains 6,676 samples with every Dock joint position, velocity, and torque-bias target exactly zero. Historical Order 3 artifacts used damping `1`; they are not current Orders 5-7 numerics evidence.
- Real-Isaac attach-only evidence: `artifacts/p4_full/order5_7_dynamic_assembly/attach_only_filter_fallback_report.json` passed with axial/transverse/attitude error `0.000160594 m / 0.001994758 m / 0.000182269 rad`, relative linear/angular speed `0.000769583 m/s / 0.017335538 rad/s`, and `0.100000 s` continuous final dwell; validation failures were empty.
- Real-Isaac round-trip evidence: `artifacts/p4_full/order5_7_dynamic_assembly/roundtrip_filter_fallback_report.json` passed attach and detach with empty validation failures. The filter was removed after 954 separation steps at actual gap/selected-body clearance `0.200870784 m / 0.030092942 m`; post-unfilter minima were `0.200996401 m / 0.030207242 m`, selected-pair recontact count was zero, and 200/200 stable samples established the required 1 s post-release dwell. Unload readiness passed with `0.006328 N / 0.000633 Nm` estimated load and follower external-contact-free evidence.
- Physical funnel result: The force-regenerated/post-verified `convexDecomposition` route was exercised but did not reach the strict final seated gate; contact-induced stall/rotation remained. It is retained as a fail-closed diagnostic path and has no attach or round-trip acceptance claim. Do not move the authored connect frames, relax the `3 mm` axial gate, or describe fallback evidence as natural/physical Dock contact.
- Live progress: Dynamic assembly now prints flushed `[dynamic-assembly] simulation_time=... phase=... event=...` lines at every phase transition and every `1.0` s of simulation time while the phase remains active. User-facing aliases expose `staging`, `axial`, `fixed`, `unload`, and `separation`; the synchronous outer runner concurrently drains both child streams and forwards those progress lines immediately instead of hiding them behind final JSON capture.
- Files changed: `amsrr/assembly/assembly_control_bridge.py`, `assembly_motion_planner.py`, `closed_loop_executor.py`; `amsrr/controllers/controller_handover.py` and exports; `amsrr/simulation/dynamic_assembly.py`, `dynamic_contact_evidence.py`, `dynamic_dock_constraint.py`, `isaac_usd_collision.py`; `scripts/order5_7_dynamic_assembly.py`, dynamic integration in `scripts/p4_control_holon_spawn_probe.py`; joint/dynamic config, hashing support, focused tests, design supplement, and this worklog.
- Schema/interface changes: No existing persisted morphology, policy, checkpoint, TaskSpec, or QPID command schema changed. Added versioned internal Orders 5-7 config/result/evidence contracts, explicit `attach_only|roundtrip` gates, and explicit physical/fallback mating modes. Reports are hash-bound and mode-specific.
- Upstream dependencies used: v0.4 Sections 17, 20, and 24.5; completed Orders 1-4 and 2.5; `centroidal_local_joint_v2`; `FACE_TO_FACE_DOCK_RELATION`; current PhysicalModel/URDF; existing follower-subtree estimator and unload gate.
- Tests added or run: The final focused assembly/controller/simulation/hashing regression, including fallback-default, transition forwarding, and simulation-time heartbeat coverage, passed `167` in `1.99 s` under `isaaclab3`; both formal seed-2 real-Isaac commands exited zero and passed their typed validators. Python compilation and whitespace checks pass at closeout.
- Commands run: Formal `scripts/order5_7_dynamic_assembly.py --real --mating-contact-mode selected_pair_collision_filter_fallback` runs for `attach_only` and `roundtrip`; focused pytest with external plugin autoload disabled; source/report/status/diff audits.
- Assumptions: Leader holds while one follower moves; both start as independent single-module Articulations at neutral Dock targets; exact connect frames are final seated frames; the approved pair-only filter is a fallback for simulator collision representation, not a mechanical latch/contact model.
- Blockers / open questions: No method-level blocker for handing the accepted default boundary to the next order. Arbitrary preassembled components, intra-component self-collision validation, and generic `AssemblyStep(detach)`/`ConstructionState` split integration remain open limitations. Object natural-contact grasping is still Order 8 and is not covered here. Physical funnel contact is not an active blocker.
- Next steps: Commit the implementation/documentation split, then begin Order 8 only on explicit request. Any optional future physical-funnel diagnostic must keep the same connect frames and strict final gates and must earn its own separately named physical-contact acceptance.

### 2026-07-13 (P4-full Orders 6-7 funnel-guided mating correction started)
- Spec version: `A-MSRR_codex_ready_spec_v0_4_ja.md` v0.4 plus the approved Orders 5-7 dynamic-assembly supplement and the user-confirmed funnel-mating geometry
- Work package / Agent label: Agent B/G/J/K/L boundary; Dock collision representation, deterministic `pi_A` guidance contact, Isaac attach, and round-trip evidence
- Status: Historical in-progress entry, superseded by the fallback-acceptance closeout above. No physical funnel attach/detach success is claimed here or by the later fallback result.
- User-confirmed geometry: `pitch_dock_mech` is a receiving funnel; `yaw_dock_mech` enters it while bounded contact with the funnel interior guides alignment.  Pitch `connect_point` is the final seated frame inside the funnel, so first selected-body contact normally occurs before the final connect-frame pose tolerance is met.
- Diagnostic correction: The latest seed-2 GUI run force-regenerated the isolated bundle and requested `Convex Decomposition`, but the generated USD actually authored each selected Dock collider as one `physics:approximation="convexHull"`.  This filled the pitch funnel.  The observed `3.443 N` contact at a `0.046536 m` connect-frame gap is expected funnel-rim contact made blocking by that runtime collider, not evidence that the authored pitch connect frame should move.
- Implementation contract: Preserve the authored connect frames and strict final `3 mm` gate.  Separate bounded selected-pair guidance contact from final seated/fix-ready evidence; continue axial insertion through safe guidance contact; fail on non-selected contact, excessive force/penetration, invalid raw evidence, controller failure, or timeout.  Verify the generated USD collision approximation rather than trusting the requested converter string.  Prefer verified funnel-preserving collision; if unavailable, filter only the selected pitch/yaw rigid-body pair during insertion and report that fallback without claiming physical funnel-contact evidence.
- Files changed so far: `for_codex/AMSRR_design_modification_by_codex.md`, `for_codex/WORKLOG.md`.
- Schema/interface changes: None yet.  Additive internal phase/evidence/report fields may be introduced without changing persisted morphology, policy, checkpoint, or QPID command schemas.
- Tests run: None yet for this correction.
- Blockers / open questions: None at method level.  The installed Isaac converter's ability to author a true compound convex decomposition is under local inspection.
- Next steps: Superseded. Do not follow this old pending-work instruction; use the closeout entry above and keep physical-funnel versus fallback evidence separate.

### 2026-07-13 (P4-full Orders 5-7 dynamic assembly started)
- Spec version: `A-MSRR_codex_ready_spec_v0_4_ja.md` v0.4 plus the approved centroidal-control and Orders 5-7 dynamic-assembly supplement
- Work package / Agent label: Agent B/G/I/J/K/L boundary; deterministic `pi_A` component control, dynamic attach, unload-gated detach, and Isaac round trip
- Status: Historical in-progress entry, superseded by the fallback-acceptance closeout above. Order 5 was complete at this point; later Orders 6-7 fallback acceptance does not create a physical funnel-contact claim.
- Summary: Implemented two separately spawned single-Holon Articulations, explicit floor initialization, vectoring pre-trim, takeoff/hover acquisition, collision-aware component motion, face-to-face pre-alignment, bounded axial approach, same-patch selected-surface contact validation, preauthored-disabled external `UsdPhysics.FixedJoint`, exact runtime identity verification, verified collision-filter lifetime, bidirectional controller handover, raw-patch follower external-contact evidence, follower-subtree unload estimation, exact removal, current-Dock-body collider clearance, separation, and continuous post-release stability. Every dynamic run force-regenerates its isolated `Convex Decomposition` USD from the current resolved URDF. `AssemblyControlBridge` remains the v0.4 Section 17.7 implementation; Dock joints remain morphology joints, not latches.
- Files changed so far: Order 5 files plus `amsrr/simulation/dynamic_assembly.py`, `dynamic_dock_constraint.py`, `amsrr/controllers/controller_handover.py`, `scripts/order5_7_dynamic_assembly.py`, the additive dynamic path in `scripts/p4_control_holon_spawn_probe.py`, controller/simulation tests, config, design supplement, and this worklog.
- Schema/interface changes: Additive `attach_only|roundtrip` acceptance gate and typed result; versioned exact-constraint spec/record/residual; complete actuator-domain merge/blend helpers. The CLI stores seed, sampling, graph, effective config, backend config, and physical-model provenance inside the typed result's free-form `report`, and defaults to separate `attach_only_report.json`/`roundtrip_report.json` paths. Existing persisted morphology, policy, checkpoint, and QPID command schemas are unchanged.
- Upstream dependencies used: v0.4 Sections 17, 20, and 24.5; completed Orders 1-4 and 2.5; `centroidal_local_joint_v2`; `FACE_TO_FACE_DOCK_RELATION`; current neutral Dock geometry; existing follower-subtree estimator/unload dwell gate.
- Downstream impact: Order 8 may use full-chain morphology kinematics and all upstream Dock joints for object grasping, but normal QPID remains quasi-static centroidal thrust allocation plus independent local joint servos.
- Tests added or run: Order 5 coverage plus exact connect-frame construction/residual/JointEnabled identity; attach-only versus round-trip validation; source/URDF/USD hashes and forced conversion; phase/filter/raw-contact/unload/no-mislabel gates; controller command merge/blend; bounded staging reference; floor/contact/hover/constraint/joint/clearance dwell evidence. The final focused Order 5-7/controller suite passed `71`; the complete regression passed `605` with `1` skipped in `300.24 s`.
- Commands run: clean-tree/log audit; source-of-truth and supplement re-audit; focused pytest with external plugin autoload disabled; `py_compile`; CLI dry runs for both gates; two real Isaac seed-2 attach-only diagnostic runs on CUDA; `git diff --check`.
- Assumptions: Leader holds while follower moves during initial deterministic assembly; module docking starts at canonical neutral Dock configuration and permits only bounded final alignment correction; the constraint fixes selected connect frames while upstream Dock joints remain articulated.
- Real-Isaac diagnostic (non-authoritative for current collision type): The second seed-2 diagnostic passed floor settle, vectoring pre-trim, takeoff, hover acquisition, staging, prealign dwell, and axial approach with zero operational QP infeasibility. First non-speculative selected-body contact was `3.443 N`, separation `0.000062 m` at the collider patches, but connect-frame relative translation remained `[0.046536, -0.003481, 0.002785] m`; selected-surface validation rejected it and no FixedJoint was enabled. The earlier `0.15 m/s` staging limit produced one QP-infeasible tick; configuration-backed `0.10 m/s` then removed that failure. Both runs used the former shared `Convex Hull` conversion lineage and prove fail-closed behavior only.
- Blockers / open questions: Rerun from the isolated `artifacts/isaac/robots/holon_dynamic_assembly` `Convex Decomposition` bundle before concluding that the approximately `47 mm` mismatch belongs to the authoritative URDF/xacro. Do not relax the `3 mm` fix gate or snap/fix at `46.5 mm`. Intra-articulation self-collision is disabled in this first smoke and remains unvalidated. Generic `AssemblyStep(detach)`/`ConstructionState` split support is downstream P4-full integration debt rather than evidence supplied by the dedicated Order 7 smoke.
- Next steps: Superseded. Do not correct or move the authoritative connect frames based on this historical diagnostic. Use the accepted pair-only fallback reports above; the separate physical-funnel path remains open and fail-closed.

### 2026-07-13 (P4-full Order 4 deterministic free-flight pi_H completed)
- Spec version: `A-MSRR_codex_ready_spec_v0_4_ja.md` v0.4 plus the approved Order 4 free-flight deterministic `pi_H`/trajectory-runtime supplement
- Work package / Agent label: Agent H/I/K boundary; P4-full Order 4 deterministic free-flight planner, trajectory executor, low-level integration, and GUI verification path
- Status: Implementation and representative verification complete for the deterministic free-flight slice. This is not learned/contact-aware `pi_H` completion and not P4-full completion.
- Summary: Implemented a retained production deterministic free-flight `pi_H` fallback plus the shared `ContactWrenchTrajectory` runtime. The planner consumes an unchanged `HighLevelPolicyContext`, evaluates measured centroidal/controller/floor-contact guards, emits rolling 2 s/0.25 s multi-knot plans at 2 Hz, and executes floor settle, takeoff, hover acquisition, three translation/attitude waypoints, final hover, timeout/abort, and safe hold. The common executor owns plan-relative time origin, knot validation/selection/interpolation, expiry, and explicit active-knot delivery to deterministic baseline or optional compatible learned `pi_L`, then QPID and the Isaac bridge.
- Files changed: New `amsrr/schemas/order4.py`, `amsrr/policies/contact_wrench_trajectory_runtime.py`, `amsrr/policies/deterministic_free_flight_planner.py`, `amsrr/simulation/order4_free_flight.py`, `configs/training/order4_deterministic_pi_h.yaml`, `scripts/order4_free_flight_pi_h.py`, four focused test files; additive Order 4 integration in `scripts/p4_control_holon_spawn_probe.py`; design supplement and this worklog.
- Schema/interface changes: No existing persisted schema or existing policy/checkpoint contract changed. Added versioned Order 4 mission, planner config, runtime-step, Isaac config/result, and report contracts. Existing `HighLevelPolicyContext`, `ContactWrenchTrajectory`, `InteractionKnot`, `PolicyCommand`, `centroidal_local_joint_v2`, and Order 3 checkpoint format remain authoritative.
- Upstream dependencies used: v0.4 Sections 19, 20, and 24.5; completed Orders 1-3 and Order 2.5; existing random-morphology floor takeoff, morphology-conditioned `pi_L`, QPID, actuator bridge, and corrected zero-dock-hold contracts.
- Downstream impact: Order 8 may extend the retained fallback with natural-contact planning; Order 9 may train/evaluate learned `pi_H` through the same executor. Order 4 itself cannot establish contact-aware or learned-policy quality.
- Tests added or run: Added schema hash/timing validation; executor origin/interpolation/expiry validation; planner state guard, multi-waypoint, low-level handoff, controller-infeasible safe hold, no-contact boundary, optional checkpoint hash/propagation, and Isaac command/viewer/report/tamper tests. Final focused suite passed `12`; relevant Order 2/3/4 regression passed `112`; final complete `tests/unit tests/acceptance` regression passed `530` with `1` skipped in `297.44 s`. Python compilation, CLI help, and `git diff --check` passed.
- Commands run: Repository/spec/interface audits; focused and full pytest; compileall; CLI dry run/help; real Isaac Lab runs through `micromamba run -n isaaclab3`; Kit viewer with real-time playback; raw report metric extraction; status/diff/whitespace checks.
- Real-Isaac evidence: N=2 seed 2, N=3 seed 25, and N=8 seed 8 each passed the default three-waypoint mission in `18.005 s`, with `37` replans and `5.505 s` final hover. Final position errors were `0.000423/0.000484/0.000547 m`; final attitude errors `0.005200/0.005250/0.005292 rad`; maximum Dock angles `0.001056/0.001780/0.002504 rad`. Every case had zero QP infeasibility, unintended cross-module contact, missing/unsupported actuator, and nonzero Dock position/velocity/torque-bias command. N=3 endurance passed `20.505 s` final hover after `67` replans/`33.005 s`, with final position/attitude error `0.000305 m / 0.000189 rad` and zero safety failures.
- GUI evidence: N=3 seed 25 passed through Kit with real-time playback and a 5 s post-rollout hold. The same typed validator reported no failures. User command: `/home/leus/.local/bin/micromamba run -n isaaclab3 python scripts/order4_free_flight_pi_h.py --real --viewer kit --realtime-playback --module-count 3`; omit `--seed` for a fresh morphology or add it for reproducibility.
- Failure/recovery record: The first 20 s endurance attempt reached the inherited 300 s subprocess timeout before a raw report was produced. This was an execution-time budget failure, not accepted motion evidence. Added an explicit validated Order 4 `command_timeout_s=600`; the identical rerun passed. No failed rollout was relabelled as success.
- Assumptions: Automated acceptance uses a 5 s continuous final hover; representative endurance uses 20 s. Existing Order 3 checkpoint-visible `task_progress` is not mutated. Empty assignments report simultaneous reachability as `not_applicable_no_active_assignments`. The optional Order 3 `pi_L` checkpoint route is hash checked and shares the executor, but the recorded Order 4 real-Isaac evidence intentionally uses deterministic baseline `pi_L` to isolate fallback/runtime behavior.
- Artifacts: Local ignored evidence is under `artifacts/p4_full/order4_deterministic_pi_h/`, including N=2/N=3/N=8 headless, N=3 endurance, and N=3 GUI reports. These are representative evidence, not source-controlled fresh-clone dependencies.
- Blockers / open questions: None for Order 4. Contact assignment/reachability, natural contact, learned `pi_H`, and full TaskSpec evaluation remain explicitly untested here.
- Next steps: Order 5 (`pi_A` AssemblyControlBridge) is the next roadmap entry, only on explicit request. Do not use the Order 4 reset-time fixed morphology as evidence of dynamic docking.

### 2026-07-12 (P4-full Orders 1-10 roadmap and Order 3 commit handoff)
- Spec version: `A-MSRR_codex_ready_spec_v0_4_ja.md` v0.4 plus the approved centroidal controller, random-morphology, and Order 3 supplements
- Work package / Agent label: Agent E/F/G/H/I/J/K/L handoff boundary; P4-full roadmap governance
- Summary: Recorded the canonical local P4-full Order 1-10 roadmap, distinguished it from older P4-control/P4.1/P4.2/P4.3 internal order numbering, marked Orders 1-3 complete at the approved boundary, and made Order 4 the unambiguous next implementation entry point. Split the accumulated Order 3 source into a policy/training-core commit and an Isaac/pipeline/acceptance commit.
- Files changed: `for_codex/AMSRR_design_modification_by_codex.md`, `for_codex/WORKLOG.md`; source committed separately as listed below.
- Schema/interface changes: Documentation only in this handoff entry. Order 3 schema/runtime changes are versioned in commits `59aba6d` and `629008c` and described by the preceding worklog entries.
- Upstream dependencies used: v0.4 P4 full acceptance and ownership sections; completed Order 0, Orders 1-2, Order 2.5, and Order 3 source/evidence; approved separation of natural-contact smoke from full TaskSpec delivery; approved dynamic assembly and controller responsibility boundaries.
- Downstream impact: A fresh chat should begin at Order 4, not restart Order 3 and not jump directly to dynamic docking or natural contact. Later work can identify exact ownership, entry gates, primary files, and no-mislabeling boundaries for every remaining order from the design-modification roadmap.
- Tests added or run: Documentation adds no tests. Before this handoff, the focused Order 3/runtime suite passed `166`; the complete `tests/unit tests/acceptance` suite passed `518` with `1` skipped in `296.41 s`; Python compilation and `git diff --check` passed.
- Commands run: `git log`/status/diff audits; focused and full pytest recorded in the preceding entry; `python3 -m compileall -q amsrr scripts tests`; `git diff --check`; logical staging/commit inspection.
- Assumptions: `Order 1-10` in this roadmap always means the local P4-full sequence recorded on 2026-07-12. Order 3 completion means source implementation plus representative hash-bound real-Isaac verification; it is not Order 10 and does not assert that ignored local artifacts exist in a fresh clone.
- Blockers / open questions: No blocker for beginning Order 4. Before Order 8 implementation, configuration-backed numeric natural-contact smoke thresholds must be confirmed in the design log. The additional regenerated disturbed floor-takeoff evaluation ended before producing a report and is not accepted evidence.
- Next steps: Start Order 4 only after confirming its production deterministic `pi_H` scheduler output contract and simultaneous-reachability inputs. Do not begin Order 5-10 in parallel merely because they are listed.

### 2026-07-12 (corrected dock frames, absolute-zero hold, and Order 3 regeneration)
- Spec version: `A-MSRR_codex_ready_spec_v0_4_ja.md` v0.4 plus the approved centroidal controller and Order 3 supplements
- Work package / Agent label: Agent B/I/J/K/L boundary, pre-next-order URDF/PhysicalModel/runtime/training regeneration
- Status: Requested regeneration and representative real-Isaac/GUI verification complete. This is not the configured full Order 3 statistical acceptance matrix and not P4 full completion.
- User-supplied upstream change: `assets/robots/holon/holon.urdf` and `module_urdf/holon.urdf.xacro` now use neutral yaw-dock origins and zero `pitch_dock_mech_joint2` origin rotation. These edits were consumed as authoritative inputs; they were not authored by this work package.
- Runtime changes: Added a fail-closed current-PhysicalModel connect-frame alignment gate before placement; complete absolute-zero dock position/velocity/torque-bias commands during settle, deterministic takeoff, and safety-masked learned free flight; measured dock angles and command maxima in real reports; and validation of the one-backlash `0.0053 rad` neutral-hold gate. BC/PPO collectors now accept explicit zero maps and reject non-zero dock intent. Learned decoding no longer ratchets measured deflection into a new target.
- Drive tuning: Raised only the AK40-10 dock simulation stiffness from `20` to `200 Nm/rad`, retaining damping `1 Nms/rad`. The selected stiffness corresponds to about `1.047 Nm` at the configured `0.005236 rad` backlash, below the `1.3 Nm` rated torque. A real-Isaac A/B run reduced maximum dock deflection from `0.0080748` to `0.00252045 rad`; the passing run ended at `0.000257712 rad`, with all three commanded maxima exactly zero.
- Files changed: Order 3 policy/collectors, random takeoff environment and Isaac probe, takeoff configs, joint actuator config, focused policy/simulation/collector tests, design supplement, and this worklog. The two URDF/xacro changes are user-owned upstream edits.
- Regenerated lineage: Pool `7f06399db7b27d495f154aa0dc7223718f755634d5a28c4df0d2f398a296f0ed`, 80 unique structures with split counts `52/14/14`; three real N=3 BC reports, 917 transitions per split; BC dataset hash `2f12c10c...`, checkpoint `8813b03e...`; three stochastic real-Isaac PPO reports, 50 transitions per split; PPO dataset hash `5eb64950...`; exactly one PPO update checkpoint `70e5b88f...`. Previous lineages were preserved under `artifacts/p4_full/order3_pi_l_v2_pre_urdf_fix_20260712/` and `artifacts/p4_full/order3_pi_l_v2_drive20_20260712/`.
- Real-Isaac evidence: All three regenerated deterministic BC sources passed; maximum/final dock angles were train `0.002520/0.000258`, validation `0.002974/0.000509`, held-out `0.001436/0.000231 rad`. All three checkpoint-bound stochastic PPO sources passed. The PPO held-out deterministic headless evaluation passed with final position/attitude error `0.012502 m / 0.004087 rad`, 50 learned decisions, fallback 0, maximum/final dock angle `0.000234/0.000232 rad`, and zero dock position/velocity/torque-bias command maxima.
- GUI evidence: Launched the regenerated held-out PPO checkpoint through Kit with real-time playback and a 20 s post-rollout hold. It exited 0 with `real_isaac_passed=true` and no report-validation failures. The GUI and headless reports bind the corrected URDF hash `21dbb35a...`, regenerated USD path, asset cache key `de55c69c...`, and dock stiffness `200`.
- Tests added or run: Added absolute-zero decoder/fallback tests, stale-frame rejection, fixed-dock report gates, explicit-zero BC/PPO collector acceptance plus non-zero rejection, and updated the fixed-morphology coordinate expectation for the corrected URDF. The final focused Order 3/runtime suite passed `166`; the complete `tests/unit tests/acceptance` regression passed `518` with `1` skipped in `296.41 s`. Python compilation and `git diff --check` passed.
- Artifacts: Active ignored evidence is under `artifacts/p4_full/order3_pi_l_v2/`; generated USD is under `artifacts/isaac/robots/holon/`. These artifacts are representative integration evidence and are not source-controlled acceptance evidence.
- Assumptions / limits: The `0.0053 rad` gate is based on the configured AK40-10 backlash and must be revisited with hardware measurements. Automated evidence proves frame consistency, zero commands, bounded measured joint motion, collision/contact gates, and successful Kit execution; final visual judgement of mesh clearance remains a human GUI observation. The full 2-8-module training/evaluation quota remains outstanding.
- Next steps: The user accepted the GUI result. Preserve the committed Order 3 boundary and begin Order 4 with its scheduler-contract review; do not start later orders in parallel.

### 2026-07-12 (Order 3 learned-policy GUI evaluation)
- Spec version: `A-MSRR_codex_ready_spec_v0_4_ja.md` v0.4 plus the approved Order 3 morphology-conditioned `pi_L` supplement
- Work package / Agent label: Agent I/K/L boundary, Order 3 real-Isaac learned-policy evaluation UX
- Summary: Added optional Kit viewer, real-time playback, and post-rollout hold controls to deterministic `evaluate-learned` execution. The public CLI propagates these options through the staged single-rollout command to the existing Isaac probe flags while preserving headless defaults and the same checkpoint/morphology/condition/report-validation path.
- Files changed: `amsrr/simulation/order3_policy_rollout.py`, `amsrr/training/order3_pipeline_runner.py`, `scripts/order3_morphology_pi_l.py`, focused Order 3 simulation/pipeline tests, design supplement, and this worklog.
- Schema/interface changes: No persisted schema/checkpoint/dataset/report change. Additive optional runtime constructor/method/CLI arguments only.
- Upstream dependencies used: Existing Order 3 checkpoint-bound rollout, Order 3 evaluation planner, `RandomMorphologyTakeoffEnv`, Isaac probe `--viz kit` / real-time / keep-open support, and P4.2 viewer-option precedent.
- Downstream impact: A user can visually inspect the same deterministic learned-policy evaluation used by the headless pipeline. GUI observation remains diagnostic and does not alter acceptance evidence or claim P4 full completion.
- Tests added or run: Added probe-command propagation/rejection coverage and evaluation-plan propagation/rejection coverage. Focused two-file suite passed `15`; complete Order 3 unit/acceptance suite passed `117` in `18.36 s`; CLI help parsing and Python compilation passed.
- Commands run: `python3 -m py_compile ...`; focused and complete Order 3 pytest invocations with `PYTHONPATH=.` and plugin autoload disabled; both relevant CLI `--help` commands; `git diff --check`/status inspection.
- Assumptions: `viewer="kit"` remains the only supported visualizer; the launching desktop session supplies a usable display. Automated training/collection remains headless.
- Blockers / open questions: No implementation blocker. A GUI process was not launched during automated verification because it requires the user's interactive desktop session.
- Next steps: Run one held-out checkpoint evaluation with `--real --viewer kit --realtime-playback`; continue full Order 3 curriculum training separately.

### 2026-07-12 (Order 3 morphology-conditioned pi_L started)
- Spec version: `A-MSRR_codex_ready_spec_v0_4_ja.md` v0.4 plus normative controller supplement Section 14 and the approved Order 3 design-modification entry
- Work package / Agent label: Agent I/K/L boundary, Order 3 morphology-conditioned `pi_L` free-flight training
- Status: In progress. User approved the proposed scope, BC-to-PPO sequence, graph encoder, morphology split, safety/fallback boundary, and initial acceptance targets.
- Fixed decisions: Preserve `MorphologyGraph`; tensorize each morphology as one homogeneous module-node/DockEdge-edge graph with port-derived edge features. Create new v2 dataset/checkpoint/runtime metadata rather than modifying or relabelling P4.3 v1 artifacts. Use true centroidal state, actor-visible deployable observations, privileged critic/reward separation, allocator-owned vectoring, and safety-masked non-vectoring joint decoding during the initial free-flight curriculum.
- Current work: Auditing existing encoder/workspace, random-morphology Isaac runtime, and P4.3 v1 artifact boundaries before schema-first implementation.
- Tests run: None yet for this order.
- Blockers / open questions: None after approval. Full acceptance runtime depends on measured Isaac training throughput; implementation will retain smaller deterministic smoke budgets without weakening the configured full gate.
- Next steps: Add v2 contracts and graph-disjoint pool, implement morphology encoder/recurrent policy and BC/PPO trainer, integrate online Isaac rollout/evaluation, then run focused/full/real validation.

### 2026-07-12 (Order 3 implementation checkpoint: v2 policy/training and real-Isaac smoke)
- Status: In progress; this checkpoint is not Order 3 statistical acceptance and not P4 full completion.
- Implemented so far: Added the hash-disjoint 2-8 module morphology pool (`4/2/2` at N=2 and `8/2/2` at N=3-8), versioned v2 dataset/checkpoint schemas and strict artifact I/O, homogeneous module/DockEdge graph tensorization and message passing, a graph-conditioned GRU actor/critic, bounded centroidal twist/wrench residual decoding, masked absolute dock-joint hold, strict deterministic-v2 fallback, BC and PPO stage APIs, privileged free-flight reward, online decision-trace collection, and Order 3-specific acceptance arithmetic.
- Runtime/controller boundary: The learned actor passes the centroidal pose target through exactly and emits intent only. QPID/allocator/bridge retain actuator authority. Actor inputs exclude measured contact wrench and simulator disturbance truth; the critic/reward may receive the applied aggregate disturbance. Vectoring remains allocator-owned and the initial free-flight dock decoder remains safety-masked to current-position hold.
- Provenance hardening added during review: Runtime now validates checkpoint PhysicalModel/URDF/fallback configuration, uses checkpoint morphology hashes as the in-distribution allowlist, and returns explicit `structural_hash_ood` fallback for valid unseen structures. Disturbed rollout `old_value` now uses the same critic-only privileged wrench later consumed by PPO while actor action/recurrent output remains invariant. Production PPO is one fresh checkpoint-matched rollout generation per update; recurrent likelihood/value evaluation is episode-sequential and does not optimize isolated rows using stale recorded hidden states.
- Evaluation hardening in progress: Acceptance now binds learned and paired deterministic-baseline raw report paths/hashes and canonical rollout conditions, requires held-out morphology x hover/waypoint/takeoff x nominal/randomized coverage, and requires OOD fallback reason `structural_hash_ood`. The executable in-air waypoint/model-randomization runtime and pipeline generation of these enriched evaluation records are still being connected.
- Representative real-Isaac collection: Deterministic v2 train/validation/held-out N=3 floor takeoff reports each passed 917 steps with zero QP infeasibility, missing/unsupported actuator, clipping, or unintended cross-module contact. They produced a three-split 2,751-transition smoke BC dataset. A two-epoch smoke BC run reduced held-out actor MSE from `0.0385888` to `0.000655133`; this is a representative integration artifact, not the configured full-pool training run.
- Learned real-Isaac smoke: The first environment-correct learned rollout executed 180 policy decisions with 180 learned applications, zero fallback, final position error `0.0621575 m`, and valid final critic bootstrap. A base-report scope parser initially rejected the correct Order 3 phase label; the parser was fixed and the same raw report revalidated with no failures. The pool/config provenance was then regenerated using both configured mesh search directories, so the smoke checkpoint/report must be rerun against the final regenerated lineage before final handoff.
- Tests at this checkpoint: Focused graph/policy/schema/dataset/reward/collector/training/rollout/acceptance groups have passed individually; latest policy tests `10 passed`, training tests `8 passed`, raw-bound acceptance tests `20 passed`, and the Order 3 takeoff scope regression `5 passed`. Full repository tests and final real-Isaac learned/PPO verification remain pending.
- Review findings being resolved: Initial code had schema-only curriculum declarations, takeoff-only online collection, repeated stale-rollout PPO updates, zero-privilege live critic values, no structural OOD allowlist, no paired evaluation producer, and no runtime fallback-config check. The latter four are fixed; executable hover/waypoint/randomization, generalized collection, iterative pipeline orchestration, and final evaluation production remain active work.
- Ignored verification artifacts: `artifacts/p4_full/order3_pi_l_v2/` contains the regenerated pool, representative graph/report files, smoke datasets, and smoke training checkpoints. These are local validation outputs and are not source-controlled acceptance evidence.

### 2026-07-12 (Order 3 implementation and end-to-end verification)
- Status: Implementation path complete and representative real-Isaac verification complete. The configured full 2-8-module training/evaluation matrix has not been run, so Order 3 statistical acceptance and P4 full completion are not claimed.
- Runtime/curriculum completion: Added hash-bound hover/waypoint/takeoff conditions, deterministic initial pose/twist perturbations, full-RPY waypoint ramp, mass/inertia/thrust randomization, scheduled external wrench, post-reset canonical state reapplication, and strict requested/applied realization evidence. Terminal dwell now starts only after target ramp and disturbance completion/onset as appropriate; pre-disturbance dwell cannot terminate an episode. Tracking cost is accumulated over a reported true-centroidal paired window rather than using only the final snapshot.
- Policy/PPO hardening: The serialized PPO observation is exactly the causal actor input and carries prior, not current, controller status. Production collection replays stored action/log-probability/value/GRU state against the behavior checkpoint. PPO orchestration defaults to the configured curriculum, derives fresh deterministic conditions per update and structural hash, consumes one fresh generation per update, selects complete episodes with seeded module-count balancing before budget truncation, and normalizes advantages only over selected rows.
- Evaluation/acceptance hardening: Learned/baseline raw reports are paired by structural and condition hash, bind seed/realization/backend/PhysicalModel/collision hashes and raw file hashes, and learned report paths include checkpoint hash. Acceptance independently revalidates applied randomization and seed evidence, binds raw safety fields for both ID and OOD episodes before fallback handling, and requires explicit `structural_hash_ood` evidence.
- Representative real-Isaac hover: The final BC checkpoint (`d0354f...`) passed a held-out N=3 in-air hover with randomized initial state: 200 physics steps, 50 policy decisions, fallback 0, QP/collision/non-finite/unsupported failures all 0, final position error `0.01073 m`, and exact requested/applied initial pose/twist evidence. Its behavior replay reproduced all 50 transitions. A paired deterministic baseline under the same condition also passed; strict raw pairing produced a smoke evaluation episode with learned/baseline tracking cost `0.09594 / 0.09514`.
- Representative PPO path: Three real-Isaac stochastic hover reports (train/validation/held-out N=3) all passed and produced 50 transitions per split. The production collector marked behavior replay verified for all three sources. Exactly one fresh PPO update produced checkpoint `ef32cd...` with finite losses, immediate-parent hash match, selected-transition-only advantage normalization, and no held-out optimization. That PPO checkpoint then passed held-out hover with 50 decisions, fallback 0, no safety failures, final position error `0.01347 m`, and tracking cost `0.09927`.
- Representative floor takeoff: The PPO checkpoint passed the randomized/disturbed floor takeoff-to-hover condition after 999 physics steps. Requested/applied mass, inertia, and thrust scales matched (`1.08262`, `0.99747`, `0.96022`), the external wrench ran from 3-4 s, terminal dwell was collected from 4-5 s, height gain ratio was `0.82448`, fallback/QP/collision/non-finite/unsupported failures were all 0, and the task-specific takeoff gate passed.
- Honest negative waypoint result: The aggressive disturbed-waypoint smoke applied all requested initial/model/wrench conditions and waited through post-disturbance dwell, but the BC-only checkpoint caused 24 initial QP-infeasible steps and 6 `controller_infeasible` fallbacks. It was correctly rejected despite terminal position error `0.14029 m`. This is expected evidence that the full waypoint/disturbance curriculum still needs actual PPO training; it was not relabelled as acceptance success.
- Verification: Order 3 focused suite passed `96` tests; the collector/pipeline/acceptance subset passed `45` tests. Final full `tests/unit tests/acceptance` regression passed `513` tests with `1` skipped in `297.06 s`. Python compilation and `git diff --check` passed.
- Artifacts: Local ignored evidence is under `artifacts/p4_full/order3_pi_l_v2/`, including pool/graphs, BC and PPO smoke datasets/checkpoints, raw rollout plans/reports, and paired smoke episodes. These are integration evidence only and do not satisfy the full held-out matrix/quota gate.

### 2026-07-12 (Order 2.5 centroidal controller contract implemented)
- Spec version: `A-MSRR_codex_ready_spec_v0_4_ja.md` v0.4 plus normative `A-MSRR_QP_PID_controller_design_spec_v0_1_ja.md` Section 14
- Work package / Agent label: Agent I/J boundary, Order 2.5: versioned `pi_L` command contract, centroidal QPID, local joint servo/bridge, privileged-contact boundary, and detach-only estimator
- Summary: Implemented `centroidal_local_joint_v2` without changing historical `legacy_contact_bias_v1` semantics. The v2 normal path tracks true morphology CoM pose/twist, applies policy CoM wrench bias through the rotor/vectoring allocator, ignores contact-wrench bias, keeps vectoring allocator-owned, and routes absolute non-vectoring position/velocity plus bounded offset torque through validated controller/Isaac bridge targets. Added follower-subtree detach wrench estimation and a fail-closed unload dwell gate.
- Files changed: `amsrr/schemas/policies.py`; controller modules `policy_command_builder.py`, `rigid_body_model.py`, `qpid_controller.py`, `actuator_mapping.py`, `isaac_controller_bridge.py`, and new `detach_wrench_estimator.py`; `amsrr/policies/low_level_policy_base.py`; shared controller smoke/takeoff simulation; Isaac probe; legacy and Order 2.5 configs; focused schema/controller/policy/simulation tests; normative controller supplement; design supplement; and this worklog.
- Schema/interface changes: Additive versioned fields `control_contract_version`, `joint_position_targets`, `joint_velocity_targets`, and `joint_torque_bias` on policy/controller commands. Added CoM-origin `RigidBodyControlModel.body_twist_world`. Added bridge actuator modes `joint_position`, `joint_velocity`, and `joint_effort_bias`. Old serialized commands parse with the legacy version/default-empty new fields.
- Upstream dependencies used: user-approved Section 14 contract; v0.4 controller responsibility boundary; Order 0 motor limits/control modes; Order 1 feasible morphology sampler; Order 2 graph-specific floor takeoff runner; current rigid-body QP and Isaac articulation APIs.
- Downstream impact: Order 3 can now generate/train a new morphology-conditioned `pi_L` against v2 absolute joint targets without contact-wrench actor leakage. Existing v1 checkpoints/artifacts remain legacy and are not v2 evidence. Dynamic `pi_A` detach still needs to invoke the new estimator/gate and implement latch/post-release execution.
- Tests added or run: Added round-trip/non-finite schema tests; v2 builder/contact no-op; true CoM pose/twist and QPID tracking; vectoring ownership; current joint hold; dock position/velocity/continuous-torque-bias mapping and unsupported-mode failure; privileged contact-wrench non-leakage; follower cut sign/gravity/frame behavior; and unload dwell/reset tests. Full `tests/unit tests/acceptance` passed 395 tests with 1 skipped in 282.41 s; after adding the final pure CoM-to-dock moment-shift sign regression, the final focused set passed 122 tests. Python compilation and `git diff --check` passed.
- Commands run: focused/full pytest with plugin autoload disabled and `PYTHONPATH=.`; `python -m compileall`; two real Isaac invocations through `micromamba run -n isaaclab3`; Git status/diff/whitespace inspection.
- Real Isaac evidence: Seed 25 sampled an accepted three-module graph on attempt 1. The final v2 run passed 917 steps / 4.585 s with 1.0 s hover dwell; final position/attitude error 0.062446 m / 0.000368 rad; final linear/angular speed 0.047828 m/s / 0.000657 rad/s; zero QP infeasibility, controller/bridge clipping, missing/unsupported/unresolved actuator target, unintended cross-module contact, and report-validation failures. Ignored artifacts are `artifacts/p4_full/order2_5_centroidal_control.json{,l}`.
- Assumptions: Version 1 non-vectoring motion remains quasi-static; offset torque uses the configured continuous dock torque limit (1.3 Nm); default detach thresholds are provisional configurable values pending measured noise/load characterization; `external_contact_free` must be independently evidenced and unknown fails closed.
- Blockers / open questions: None for the Order 2.5 controller migration. Actual `pi_A` latch release and post-release stability execution are intentionally downstream, as are v2 `pi_L` training and natural-contact TaskSpec completion.
- Next steps: Begin Order 3 morphology-conditioned `pi_L` training only on explicit request, using `centroidal_local_joint_v2` and a new checkpoint version.

### 2026-07-12 (Centroidal-only QPID design revision)
- Spec version: `A-MSRR_codex_ready_spec_v0_4_ja.md` v0.4 plus user-approved `A-MSRR_QP_PID_controller_design_spec_v0_1_ja.md` Section 14 and the matching design-modification entry
- Work package / Agent label: Agent I/J boundary: `π_L` PolicyCommand, centroidal QPID, local joint servo, controller bridge, contact-reward, and detach-unload design contract
- Summary: Documented the approved replacement of the normal contact-aware/internal-wrench QP path with a centroidal-only thrust/vectoring QP plus independent non-vectoring joint servo. Normal `PolicyCommand`/QP no longer carries contact or dock internal wrench intent. `π_L` supplies centroidal pose/twist, CoM wrench bias, absolute joint position/velocity targets, and bounded torque bias; contact wrench remains high-level context/reward/safety evidence, while dock internal wrench is detach-only.
- Files changed: `for_codex/A-MSRR_QP_PID_controller_design_spec_v0_1_ja.md`, `for_codex/AMSRR_design_modification_by_codex.md`, and `for_codex/WORKLOG.md`.
- Schema/interface changes: Documentation contract only in this work package; no Python schema or runtime interface was changed. Future schema-first migration adds absolute `joint_position_targets`, `joint_velocity_targets`, and `joint_torque_bias`, clarifies `desired_body_pose`/`desired_body_twist` as centroidal-control-frame targets and `residual_wrench_body` as CoM wrench bias, and deprecates legacy joint/contact bias fields with versioned compatibility.
- Controller decision: Normal QP variables are rotor thrust, thrust-vectoring variables/targets, and slack only. Generic non-vectoring joints use local position/velocity servo plus offset torque. Vectoring joints remain allocator-owned. Contact/internal wrench variables and tracking objectives are excluded from normal QPID.
- Centroidal decision: The revised contract requires actual current-morphology CoM pose/twist derived from the quasi-static rigid-body model; existing base-module `fc` tracking must not be reported as true centroidal tracking.
- Contact/training decision: `π_H` schema remains unchanged. Contact wrench requirements feed `π_L` context, feasibility, privileged Isaac reward/critic, safety, task success, and logging, but not direct QP tracking. Privileged per-contact data must not leak into actor observations.
- Detach decision: Internal wrench is a special detach-only estimate computed on the contact-free follower subtree from momentum balance, known actuator/gravity loads, and a CoM-to-dock wrench transform. Release remains fail-closed on contact evidence, estimator validity, pose/velocity, force/torque, component feasibility, dwell, and separation stability.
- Commands run: Inspected existing controller/spec/design/worklog references with `sed` and `rg`; updated documentation with `apply_patch`; checked Markdown structure, supersession wording, staged whitespace, and Git status/diff before commit.
- Tests run: Documentation-only consistency checks and `git diff --check`; no Python or Isaac behavior changed, so runtime tests were not rerun.
- Upstream dependencies used: v0.4 Sections 17, 19-20, 22, and 24; QP/PID supplement Sections 2-10; existing `PolicyCommand`, `ControllerCommand`, `RigidBodyControlModel`, QPID/QP, actuator model/bridge, Order 1-2 morphology flight evidence, and the user-approved detach/contact assumptions.
- Downstream impact: Order 3 morphology-conditioned `π_L` training is paused until the revised schema/controller/bridge path is implemented and versioned. Existing P4 artifacts remain valid only under their recorded legacy contracts and cannot prove this revised design.
- Assumptions: follower-side dock-wrench observability applies only when the candidate-edge follower component is verified free of other external contacts/loads; Version 1 non-vectoring joint motion remains bounded and quasi-static.
- Blockers / open questions: No design blocker. Implementation, migration, and new acceptance evidence remain pending and are not part of this documentation commit.
- Next steps: Implement the revised schema-first controller contract before resuming Order 3 training.

### 2026-07-11 (Random morphology GUI teleop utility)
- Spec version: `A-MSRR_codex_ready_spec_v0_4_ja.md` v0.4 plus the completed P4 full Orders 1-2 deterministic morphology-flight boundary
- Work package / Agent label: Agent I/J verification utility for random morphology floor takeoff, hover, and terminal pose-target commands
- Summary: Added an interactive Isaac Lab Kit launcher that samples a feasible connected morphology at a requested module count, spawns it on the floor, completes the existing deterministic takeoff-to-hover gate, and then accepts bounded terminal keyboard increments for translation and attitude. This is a manual diagnostic utility and does not extend P4 full acceptance.
- Files changed: `amsrr/simulation/random_morphology_teleop.py`, `scripts/random_morphology_teleop.py`, `scripts/p4_control_holon_spawn_probe.py`, `tests/unit/simulation/test_random_morphology_teleop.py`, design supplement, and this worklog.
- Controls: `W/S` forward/back, `A/D` left/right, `R/F` up/down, `J/L` yaw, `I/K` pitch, `U/O` roll, `H` or space hold measured pose, `0` reset the initial hover target, `P` print the target, `?` help, and `Q` quit. Horizontal translation is relative to the commanded yaw.
- Safety/runtime behavior: Position targets cannot lead the measured body pose by more than 0.50 m by default; roll/pitch targets are bounded to 30 degrees; descent is bounded above the settled floor pose. The existing QPID/rigid-body QP/Isaac actuator bridge remains the only actuator path. QP infeasibility, controller/bridge clipping, unresolved application targets, unintended cross-module contact, or raw contact-buffer saturation terminates the interactive run as a failure.
- Learning boundary: No learner, policy checkpoint, `pi_L`, `pi_H`, or training process is loaded. The utility uses the Order 1 deterministic sampler/feasibility gate and the Order 2 deterministic controller path only. Keyboard input updates the desired body pose in `PolicyCommand`; it never writes actuators directly.
- Interface changes: Added a standalone TTY CLI with `--module-count {2,3,4,5,6,7,8}`, optional reproducible `--seed`, sampling-attempt and teleop step/bound options. Added opt-in probe flags and compact interactive summary keys; normal non-interactive probe/report behavior is unchanged.
- Validation: Focused teleop/takeoff/runner regression passed before the 2-8 range extension (78 tests); the extended Order 1/2 focused set passed (118 tests). A real GUI/TTY smoke with two modules and seed 2 sampled and spawned the structure, passed floor takeoff/hover, entered interactive control, reflected `P`, `W`, and `J` commands in the target, and exited on `Q`. Real headless 7- and 8-module Order 2 smokes also passed. Final full-suite and static-check results are recorded in the Orders 1-2 entry below.
- Assumptions / limitations: Kit camera control remains manual; the terminal must retain keyboard focus. This utility checks deterministic free-flight control for one sampled structure per invocation. It does not demonstrate learned robustness, object interaction, dynamic module attach/detach, or full TaskSpec delivery.
- Next steps: User manual inspection. No later P4 full implementation order is started by this utility.

### 2026-07-11 (P4 full Orders 1-2 completed)
- Spec version: `A-MSRR_codex_ready_spec_v0_4_ja.md` v0.4 plus the user-approved P4 full implementation order and approved random-morphology feasibility definition
- Work package / Agent label: Agent E/F/I/J/L boundary: Orders 1-2 random feasible connected morphology distribution, floor initialization, deterministic takeoff-to-hover, and fail-closed evidence/archive binding
- Summary: Completed Orders 1 and 2 together. Added a seeded task-independent feasible distribution over connected 2-8 Holon trees, deterministic structural/collision/flight gates, graph-specific floor placement, and a real-Isaac zero-thrust settle -> deterministic takeoff ramp -> hover-hold path. Runtime collision evidence now monitors every cross-module rigid-body matrix while exempting only the graph-selected dock_mech body pair for each dock edge. This does not start morphology-conditioned policy training or claim P4 full completion.
- Approved Order 1 method: Sample 2-8 Holons with a seeded deterministic constructive tree generator; keep module 0 as base; use each compatible pitch/yaw port at most once; derive module poses from dock frames at the URDF reference joint state; reject duplicate canonical structures, structural invalidity, non-adjacent coarse self-collision with a 0.03 m margin, and morphology-level hover allocation infeasibility. The distribution output is task-independent `MorphologyGraph`; TaskSpec/`DesignOutput` adaptation remains downstream.
- Approved Order 2 boundary: Initialize every connected morphology on the floor, then execute deterministic takeoff-to-hover through the existing controller/QP and Isaac actuator bridge. Floor/mesh/physics evidence is an Order 2 runtime gate rather than an Order 1 graph-only claim.
- Files changed:
  - `amsrr/morphology/random_connected.py`, `amsrr/morphology/random_feasible.py`
  - `amsrr/feasibility/morphology_flight.py`
  - `amsrr/robot_model/fixed_morphology_urdf.py`, `amsrr/robot_model/physical_model_builder.py`
  - `amsrr/simulation/random_morphology_takeoff.py`
  - `amsrr/training/random_morphology_takeoff_runner.py`
  - `configs/training/random_morphology_takeoff.yaml`
  - `scripts/random_morphology_takeoff.py`, `scripts/p4_control_holon_spawn_probe.py`
  - focused morphology, feasibility, fixed-URDF, simulation, and runner tests
  - `for_codex/AMSRR_design_modification_by_codex.md`, `for_codex/WORKLOG.md`
- Schema/interface changes: No existing base schema changed. Added internal configuration/result APIs and additive probe CLI/report keys. `RandomMorphologyTakeoffRunnerResult` stores the accepted graph, deterministic `FeasibilityResult`, sampling provenance, backend-config hash, model/URDF/collision hashes, and dry/real takeoff result.
- Order 1 implementation: Constructive proposals sample module count uniformly over the configured inclusive range and select uniformly from compatible free parent/child port pairs. Canonical base-rooted port-labelled hashing is invariant to non-base module IDs, edge direction/order, and list order. Bounded rejection is fail-closed and records proposal seed, attempt count, duplicate count, and violation-code counts.
- Feasibility implementation: Structural checks derive full port inventory/local poses/compatibility masks from `PhysicalModel`; nominal-q collision AABBs are constructed from URDF collision origins/rotations/scales and binary/ASCII STL bounds; only non-adjacent inter-module pairs require 0.03 m clearance. The production rigid-body builder and `VirtualThrustQPAllocator` separately solve gravity hover and 1.15x gravity, reporting the configured 15% margin as a QP-certified lower bound. This curriculum-specific initial-flight gate does not replace the general task-level `FeasibilityChecker`.
- Order 1 distribution audit: A 50-seed proposal audit over the original 2-6 range accepted 39/50 candidates (78%); all 17 hard violations were conservative non-adjacent coarse-collision findings and none were hover-QP failures. Fixed-count acceptance over 100 seeds was 100/100, 100/100, 82/100, 56/100, 43/100, 25/100, and 16/100 for 2, 3, 4, 5, 6, 7, and 8 modules respectively. The default bounded rejection budget is therefore 256 attempts.
- Order 2 implementation: Graph `fc`-frame poses are conjugated into copied-URDF `root` poses before fixed-tree asset generation; a rotated-frame regression verifies the correction. Collision bounds determine a root spawn height with 2 mm initial floor clearance. The real probe applies no rotor thrust during a 1 s settle, then uses deterministic QPID plus the existing controller/actuator bridge for a 2 s takeoff ramp and 1 s hover hold. Runtime observations use every module's resolved Isaac `fc` body pose/twist. QP infeasibility, controller/bridge clipping, missing/unsupported actuators, excessive vertical speed, non-finite state, unresolved module frames, failed settle/ramp/hold, incomplete logging, or collision-evidence mismatch fails the gate.
- Exact collision/contact decision: Isaac articulation self-collision is enabled. All same-module rigid-body pairs are filtered. For each graph edge, `PortNode.port_local_id -> PhysicalModel.DockPortSpec.parent_link` resolves exactly one intended cross-module dock_mech body pair, and only that pair is additionally filtered. Reset-time `get_initial_collider_pairs` rejects unintended adjacent/non-adjacent contacts. One PhysX tensor view per unordered module pair monitors every other cross-module body pair on every simulation step using both the aggregate force matrix and non-aggregated `get_contact_data`. Raw capacity is eight contact patches per rigid-body pair; any raw patch count, per-patch/aggregate force above 0.001 N, or capacity saturation fails. View/pair keys, aggregate/raw update counts, capacity, raw counts, saturation, dock link/path bindings, and thresholds are graph/config bound in the fail-closed parser.
- Floor/contact and state thresholds: 2 mm initial floor clearance; floor-contact aggregate force >= 0.5 N for >= 0.10 s; zero-thrust settle 1.0 s with linear/angular thresholds 0.20 m/s and 0.50 rad/s and >= 0.25 s low-speed dwell; takeoff ramp 2.0 s; hover target +0.5 m; final error thresholds 0.20 m and 0.25 rad; final speed thresholds 0.15 m/s and 0.25 rad/s; 1.0 s hover hold with 2.0 s acquisition allowance; vertical-speed ceiling 3.0 m/s.
- Evidence/provenance implementation: Successful real reports are reconstructed into aligned typed `RuntimeObservation`, `PolicyCommand`, `ControllerCommand`, and `IsaacActuatorTargetRecord` sequences and written as one `EpisodeArchive` JSONL record. Failed/non-attempted runs replace the owned archive with an empty file, preventing stale success evidence. Graph/model/URDF/collision-geometry/backend-config/config hashes and exact phase/config/runtime bindings are persisted. Physical aggregate mass uses `math.fsum`, making the provenance-bearing `PhysicalModel` hash identical under host Python and Isaac's Python (`c1e0cb743e96ed2335b2f7cf2f373758f353942c5261d281764a216aee084be7`).
- Real Isaac evidence: One feasible random morphology at every supported size 2 through 8 passed the final selective-dock/all-cross-module raw-contact gate. Executed steps were 917, 915, 917, 917, 917, 917, and 912; monitored module-pair views were 1, 3, 6, 10, 15, 21, and 28; aggregate and raw view updates were 917, 2,745, 5,502, 9,170, 13,755, 19,257, and 25,536. Raw patch capacities were 5,000, 15,000, 30,000, 50,000, 75,000, 105,000, and 140,000. Every case reported raw contact count 0, saturation 0, maximum aggregate/per-patch unintended force 0.0 N, zero adjacent-unintended/non-adjacent contacts, zero violation steps, and dock exemptions exactly equal to its 1 through 7 graph edges. Final position error was 0.06214, 0.06162, 0.06226, 0.06231, 0.06217, 0.06273, and 0.05940 m; final attitude error was 0.001260, 0.001897, 0.000121, 0.000364, 0.000100, 0.000122, and 0.000423 rad. Every case held hover for 1.0 s and had zero QP-infeasible, controller-clipped, bridge-clipped, missing/unsupported-actuator, and unresolved-application counts. Detailed reports/typed archives are `/tmp/amsrr_order12_raw_contact_{2,3,4,5,6,7,8}.json{,l}` and are intentionally not repository artifacts.
- Commands run: Focused `pytest` sets; dry CLI/contract checks; real Isaac CLI through `micromamba run -n isaaclab3` for final 2-8 module smokes; host/Isaac model-hash comparison; full unit/acceptance suite; Python compilation; `git diff --check`.
- Tests run: Extended focused Order 1/2 regression passed: 118 passed. Full repository unit/acceptance suite passed: 382 passed, 1 skipped in 279.57 s. Python compilation and diff whitespace checks passed.
- Upstream dependencies: v0.4 Sections 14-16 and 24.5.2, PhysicalModel/URDF dock frames, Order 0 actuator limits, existing rigid-body QP/QPID, Isaac actuator bridge, fixed graph URDF generation, and IsaacLab launcher.
- Downstream impact: Order 3 can draw unique feasible `MorphologyGraph` batches with deterministic provenance for morphology-conditioned `pi_L` training. The reset-time fixed morphology representation remains distinct from later dynamic `pi_A` attach/detach work.
- Assumptions / limitations: The Order 1 coarse module AABB gate is conservative and may reject collision-free shapes; adjacent dock-connected pairs are excluded only from that coarse graph gate. Order 2 separately checks all cross-module physics pairs and exempts only the selected dock_mech pair per edge. The seven real cases establish a supported-size deterministic smoke cohort, not statistical robustness, learned-policy quality, dynamic attach/detach, object-contact success, or full TaskSpec delivery.
- Blockers / open questions: None for Orders 1-2.
- Next steps: Stop here. Begin Order 3 morphology-conditioned `pi_L` training only after an explicit user request.

### 2026-07-11
- Spec version: `A-MSRR_codex_ready_spec_v0_4_ja.md` v0.4 plus approved P4 full Order 0 joint-actuator performance clarification
- Work package / Agent label: Agent B/I/J boundary: P4 full Order 0 joint actuator model and Isaac drive configuration
- Summary: Completed Order 0 only. Replaced generic vectoring/dock URDF performance limits with installed-motor values, added a validated joint actuator performance config with official-source provenance, integrated it into PhysicalModel and actuator-channel metadata, and made the Isaac probe consume configured drive gains by default. No random morphology, takeoff/hover, learning, assembly bridge, dynamic docking, natural contact, or P4 full rollout work was started.
- Motor identity and limits: Resolved user-reported DYNAMIXEL `SC330-T181` to official `XC330-T181-T`. Vectoring hard limits are `0.76 Nm` and `10.890854 rad/s` at the recommended 11.1 V operating point; its `0.152 Nm` continuous value is explicitly a conservative ROBOTIS US estimate, not a published continuous rating. Dock joints use CubeMars AK40-10 KV170 rated `1.3 Nm / 38.746309 rad/s`, peak `4.1 Nm`, and no-load `45.553093 rad/s`; the MIT protocol `5.0 Nm` range is recorded but not used as the safety hard limit.
- Files changed:
  - `configs/robot/joint_actuators.yaml`, `configs/robot/robot_model.yaml`
  - `module_urdf/holon.urdf.xacro`, `assets/robots/holon/holon.urdf`, `module_urdf/README_for_codex.md`
  - `amsrr/robot_model/joint_actuator_model.py`, `amsrr/robot_model/physical_model_builder.py`
  - `amsrr/controllers/actuator_mapping.py`, `scripts/p4_control_holon_spawn_probe.py`
  - joint actuator, PhysicalModel, and actuator mapping unit tests
  - `for_codex/AMSRR_design_modification_by_codex.md`, `for_codex/WORKLOG.md`
- Schema/interface changes: No persisted schema change. Added an internal validated YAML config and additive nested PhysicalModel/DockPort/ActuatorChannel metadata. `build_physical_model` and `build_physical_model_from_config` gain optional joint-actuator config path inputs. Probe drive-gain CLI arguments now default to the actuator config while retaining explicit overrides.
- Source provenance: ROBOTIS official XC330-T181-T model reference and product page; CubeMars official AK40-10 product specification and AK-series driver manual. URLs and basis notes are persisted in `configs/robot/joint_actuators.yaml`.
- Tests run: Targeted joint actuator/robot model/controller mapping regression passed: 26 passed; final source/runtime URDF consistency focus passed: 10 passed. Full unit/acceptance suite passed: 262 passed, 1 skipped before the final additive consistency test was added.
- Assumptions: The typed model name supplied as `SC330-T181` is a typo/alias for official `XC330-T181-T`. At this historical Order 0 boundary, `20/1` Isaac stiffness/damping and the `3 rad/s` safe motion limit were retained as simulation tuning rather than inferred manufacturer data. The Dock drive was later superseded by current `200/2`; the `3 rad/s` safe limit remains.
- Next steps: Stop after Order 0. Begin random feasible connected morphology distribution only on a later explicit request.

### 2026-07-10
- Spec version: `A-MSRR_codex_ready_spec_v0_4_ja.md` v0.4, especially Sections 22.3-22.4, 24.5.5, 26.10, and implementation-order items 25-26
- Work package / Agent label: Agent K P4.3 minimum learning bootstrap, Agent J learned `pi_L` Isaac injection boundary, Agent L P4.3 artifact/acceptance audit
- Summary: Completed the ordered P4.3a-d minimum learning bootstrap without starting joint RL. Added task-disjoint real-Isaac datasets and reward alignment, trained bounded `pi_L`, teacher-imitation `pi_H`, and outcome-conditioned `pi_D`, ran one checkpoint-bound online `pi_L` Isaac rollout, and wrote a fail-closed acceptance/summary archive. Deterministic `pi_D`/`pi_H`/`pi_L` fallbacks, `FeasibilityChecker`, controller/QP, and actuator ownership remain intact. This is P4.3 minimum completion only, not P4 full completion.
- Files changed:
  - `amsrr/schemas/datasets.py`, `amsrr/schemas/__init__.py`
  - `amsrr/training/p4_2_deterministic_rollout_runner.py`, `p4_3_reward.py`, `p4_3_rollout_runner.py`, `p4_3_dataset_builder.py`, `p4_3_pi_l_training.py`, `p4_3_pi_h_training.py`, `p4_3_pi_d_training.py`, `p4_3_learning_archive.py`
  - `amsrr/policies/learned_low_level_policy.py`, `contact_candidate_encoder.py`, `learned_high_level_policy.py`, `learned_design_selector.py`
  - `amsrr/simulation/isaac_lab_backend.py`, `amsrr/simulation/p4_2_isaac_env.py`
  - `amsrr/acceptance/p4_3_acceptance.py`, `amsrr/acceptance/__init__.py`
  - `configs/training/p4_3_learning_bootstrap.yaml`
  - `scripts/p4_3_collect_dataset.py`, `p4_3_build_dataset.py`, `p4_3_learning_bootstrap.py`, `p4_control_holon_spawn_probe.py`
  - P4.3 schema/policy/training/acceptance tests and the P4.2 environment propagation test
  - `for_codex/AMSRR_design_modification_by_codex.md`, `for_codex/WORKLOG.md`
- Schema/interface changes: Added the additive `p4_3_dataset_v1` schemas (`StageDecisionMasks`, low-level, interaction-trajectory, design-outcome records, shards, and manifest). Existing persisted schemas are unchanged. The existing P4.2 environment/backend/probe interface gains optional learned-`pi_L` checkpoint and runtime blend inputs; behavior is unchanged when no checkpoint is supplied.
- Upstream dependencies used: P2/P2.5 deterministic candidate generation and scorer checkpoint, deterministic `FeasibilityChecker`, P3 assembled morphology, P4.2 `ContactCandidateSet` / deterministic trajectory / kinematic payload-coupled real Isaac rollout, `BaselineLowLevelPolicy`, controller/QP, actuator bridge, EpisodeArchive hashing/provenance, and TaskSpec full-goal tolerances.
- Dataset/reward result: Collected 24 real-Isaac rollouts from six seeds and four hard-feasible designs per task. P4.2 bounded-carry outcomes were 17 pass / 7 fail and all outcomes were retained. Task split is 4 train / 1 validation / 1 held-out. Final shards contain 24 rollout, 24 interaction-trajectory, 24 design-outcome, and 4,486 low-level records at effective 50 Hz. Reward uses `obs[i] -> obs[i+1]` with `command[i]`; terminal belongs to `N-2`, final command state terms are unavailable, and stride aggregation preserves the full 200 Hz return.
- Learning result: `pi_L` final train/validation normalized MSE is 0.0009354 / 0.0027732 with zero clipped and zero unrepresentable target values. Its manifest-held-out real-Isaac gate passed with 680 learned non-zero subset overlays, blend 0.10, fallback 0, max overlay norm 0.10750, and zero drop/collision/controller-QP terminal. `pi_H` validation exact selection, schema, and assignment-feasible rates are 1.0 with fallback rate 0; evaluation remains offline teacher decode. `pi_D` used the compatible P2.5 initializer, has 20 train / 5 validation within-task ranking pairs, and validation pairwise ranking accuracy 0.8; evaluation remains offline held-out outcome regression.
- Acceptance/artifacts: Final report has `dataset_passed`, `pi_l_passed`, `pi_h_passed`, `pi_d_passed`, deterministic fallbacks, no-mislabeling, and `completion_passed` all true for 24 source episodes. The gate binds all shard hashes/counts/splits/masks, head-specific checkpoint/config/dataset metadata, and a held-out online checkpoint/archive. It recomputes command-level non-learned-field/orientation preservation, active-knot guards, and overlay norms from pre/post evidence and rejects P4-full/natural-contact claims. The P4.3-only summary removes copied step logs, fixes task/metric phase metadata, and post-validates its hashes/flags. Local ignored artifacts are under `artifacts/p4_3/`, including `p4_3_minimum_learning_summary.jsonl`.
- Commands run:
  - `/home/leus/.local/bin/micromamba run -n isaaclab3 python scripts/p4_3_collect_dataset.py --real` (first two candidates per task)
  - `/home/leus/.local/bin/micromamba run -n isaaclab3 python scripts/p4_3_collect_dataset.py --real --candidate-offset 2 --candidates-per-task 2 --archive-path artifacts/p4_3/rollouts/deterministic_isaac_extra.jsonl`
  - `PYTHONPATH=. python scripts/p4_3_build_dataset.py artifacts/p4_3/rollouts/deterministic_isaac.jsonl artifacts/p4_3/rollouts/deterministic_isaac_extra.jsonl`
  - `PYTHONPATH=. python scripts/p4_3_learning_bootstrap.py`
  - `/home/leus/.local/bin/micromamba run -n isaaclab3 python scripts/p4_3_collect_dataset.py --real --task-start-index 4 --task-count 1 --candidates-per-task 1 --archive-path artifacts/p4_3/pi_l/online_rollout_archive.jsonl --pi-l-checkpoint artifacts/p4_3/pi_l/checkpoint.pt`
  - `PYTHONPATH=. python scripts/p4_3_learning_bootstrap.py --acceptance-only`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. pytest -q` for the 94-test P4.2/P4.3 focused set
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. pytest -qq tests/unit tests/acceptance`
  - `python -m py_compile ...`, `git diff --check`
- Tests run: Focused P4.2/P4.3 regression passed: 94 passed. Full unit/acceptance suite passed: 259 passed, 1 skipped (260 collected). Real Isaac dataset collection, learned `pi_L` online rollout, final acceptance, and summary archive creation all exited 0.
- Assumptions: `ModuleCapabilityToken` fields are the minimum serialized PhysicalModel summary for `pi_L`. The minimum `pi_H` ranker learns candidate/group tokens while envelope/morphology/runtime/cache stay runtime/decode/safety context. The `pi_L` deployment trust region is 0.10 and changes only twist, body-position, and residual-wrench intent; the P4.2 controller knot and non-learned command fields are preserved.
- Blockers / open questions: None for the bounded P4.3a-d minimum run. Production learned-policy quality is not established; `pi_H` and `pi_D` are not online-deployed, and the minimum `pi_H` candidate-head top-k recall is 0.0 even though its learned group branch decodes the exact feasible teacher assignments. Natural contact, slip/friction quality, full TaskSpec delivery, and P4 full completion remain unproven.
- Next steps: Commit/review P4.3a-d. Then treat P4.3e joint fine-tuning as a separate later work package, run P4.4 natural-contact validation, and only then evaluate P4 full acceptance.

### 2026-07-10
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4 plus approved P4.2 payload-coupled deterministic rollout clarification
- Work package / Agent label: Agent J/K/L boundary: P4.2 GUI observation launcher
- Summary: Added a hand-executable Kit viewer launcher for the existing real P4.2 deterministic rollout. The P4.2 parent CLI now forwards optional viewer, real-time playback, and post-rollout hold settings through the environment/backend command boundary to the existing Isaac probe; the default headless acceptance path is unchanged.
- Files changed:
  - `amsrr/simulation/isaac_lab_backend.py`
  - `amsrr/simulation/p4_2_isaac_env.py`
  - `scripts/p4_2_deterministic_rollout.py`
  - `scripts/run_p4_2_gui.sh`
  - `tests/unit/simulation/test_p4_2_isaac_env.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: No persisted schema change. Added optional P4.2 visual-observation arguments only: `viewer`, `realtime_playback`, and `keep_open_after_rollout_s` / probe `keep_open_after_smoke_s`.
- Upstream dependencies used: Existing P4.2 parent CLI, `P4_2IsaacEnv`, IsaacLab backend command builder, probe support for `--viz kit`, real-time playback, and viewer hold.
- Downstream impact: `scripts/run_p4_2_gui.sh` runs the same P2/P3-sourced deterministic payload-carry rollout under `kinematic_payload_coupled_attach_v1` in the Kit GUI. Visualization does not change attach conditions, payload coupling, archives, success semantics, or split acceptance.
- Tests added or run:
  - Added P4.2 command and environment propagation coverage for Kit viewer, real-time playback, and viewer hold.
  - `python3 -m py_compile amsrr/simulation/isaac_lab_backend.py amsrr/simulation/p4_2_isaac_env.py scripts/p4_2_deterministic_rollout.py`
  - `bash -n scripts/run_p4_2_gui.sh`
  - `PYTHONPATH=/home/leus/amsrr PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/unit/simulation/test_p4_2_isaac_env.py tests/unit/training/test_p4_2_deterministic_rollout_runner.py tests/acceptance/test_p4_2_acceptance.py`
  - `scripts/run_p4_2_gui.sh --help`
- Tests run: Targeted P4.2 / shared Isaac command tests passed: 21 passed. The launcher help was verified from the `isaaclab3` micromamba environment without opening Isaac Lab. The headless real Isaac P4.2 rollout passed with `completion_passed=true` and `real_isaac_rollout_passed=true`.
- Assumptions: `${HOME}/IsaacLab/isaaclab.sh` and `${HOME}/.local/bin/micromamba` retain the paths specified by the existing `configs/env/isaac_lab.yaml` / AGENTS environment setup.
- Blockers: None. GUI rendering is intentionally left for the user's local display session; `--help` does not launch a simulator.
- Next steps: Run `scripts/run_p4_2_gui.sh` on the desktop session and inspect the selected `module_1__pitch_dock_mech1` anchor during attach. Natural contact validation remains P4.4 work.

### 2026-07-10
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4 plus approved P4.2 payload-coupled deterministic rollout clarification
- Work package / Agent label: Agent J/K/L boundary: P4.2 link-backed RobotAnchor attach gate hardening
- Summary: Hardened the P4.2 v1 attach gate so successful attach evidence must be tied to the selected `RobotAnchor.link_id` resolved as an Isaac articulation body. The probe now computes the selected anchor pose from the resolved connector-link body pose plus `RobotAnchor.local_pose`, uses that pose for attach distance, relative velocity, object slaving, and body target computation, and archives link-backed attach evidence. Acceptance now rejects attach archives that only prove the older module-state fallback anchor.
- Files changed:
  - `amsrr/simulation/p4_2_rollout.py`
  - `amsrr/simulation/p4_2_isaac_env.py`
  - `amsrr/acceptance/p4_2_acceptance.py`
  - `amsrr/training/p4_2_deterministic_rollout_runner.py`
  - `scripts/p4_control_holon_spawn_probe.py`
  - P4.2 unit/acceptance tests
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: No persisted base schema change. Additive P4.2 runtime/archive fields were added to `P4_2AttachEvent` for `anchor_link_id`, resolved Isaac body name, anchor pose source, link pose, local pose in link frame, link twist, and link-resolution metadata. P4.2 acceptance now requires a link-backed attach event.
- Upstream dependencies used: P2 selected `DesignOutput`, P3 assembled `MorphologyGraph`, `RobotAnchor.link_id` / `local_pose`, P4.2 selected contact assignments, existing Isaac body-name prefixing, and the approved `contact_model="kinematic_payload_coupled_attach_v1"` scope.
- Downstream impact: P4.2 completion can no longer pass on an attach event that is only based on module-frame virtual anchors. The archive now contains enough evidence to inspect which connector-link body triggered attach. This still does not validate natural contact or connector mesh contact.
- Tests added or run:
  - Added schema coverage for link-backed attach event evidence.
  - Added acceptance coverage rejecting attach events without `anchor_pose_source="isaac_link"`.
  - `python3 -m py_compile amsrr/simulation/p4_2_rollout.py amsrr/acceptance/p4_2_acceptance.py amsrr/simulation/p4_2_isaac_env.py amsrr/training/p4_2_deterministic_rollout_runner.py scripts/p4_control_holon_spawn_probe.py tests/acceptance/test_p4_2_acceptance.py tests/unit/simulation/test_p4_2_rollout.py tests/unit/simulation/test_p4_2_isaac_env.py tests/unit/training/test_p4_2_deterministic_rollout_runner.py`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/unit/simulation/test_p4_2_rollout.py tests/unit/simulation/test_p4_2_isaac_env.py tests/unit/training/test_p4_2_deterministic_rollout_runner.py tests/acceptance/test_p4_2_acceptance.py`
  - `PYTHONPATH=/home/leus/amsrr python3 scripts/p4_2_deterministic_rollout.py --config configs/training/p4_2_deterministic_rollout.yaml --real`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/unit tests/acceptance`
  - `git diff --check`
- Tests run: Targeted P4.2 tests passed: 21 passed. Full unit/acceptance suite passed: 186 passed, 1 skipped. Real Isaac P4.2 gate passed with `completion_passed=true`, `real_isaac_rollout_passed=true`, one link-backed attach event, one release event, and no acceptance failures.
- Real Isaac evidence: The attach event archived `anchor_id=0`, `anchor_link_id="pitch_dock_mech1"`, `anchor_resolved_body_name="module_1__pitch_dock_mech1"`, `anchor_pose_source="isaac_link"`, `distance_m=0.10354209267884071`, `relative_velocity_mps=0.19955582230704955`, `p4_2_attach_event_link_backed_count=1.0`, `p4_2_payload_controller_metric_record_count=101.0`, and `p4_2_payload_wrench_delta_norm=14.017291907331392`.
- Assumptions: `RobotAnchor.local_pose` is interpreted as a link-relative P4.2 runtime offset when `link_id` resolves in Isaac. This is an implementation-time supplement for P4.2 v1 and does not alter the meaning of π_D as graph-level design.
- Blockers: None for this hardening step.
- Next steps / handoff: The GUI should be rerun with the generated command to visually inspect whether the connector-link-backed anchor is acceptable. P4.2 remains a kinematic payload-coupled attach approximation, not natural grasp. Keep `P4.4 / P4-natural-contact-grasp validation` open for free rigid-body contact/friction/contact maintenance, slip/contact break/drop/unintended collision/penetration/contact-force logging, and comparison against `natural_contact_grasp_v1`.

### 2026-07-10
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4 plus approved P4.2 payload-coupled deterministic rollout clarification
- Work package / Agent label: Agent J/K/L boundary: P4.2 Order 6b real Isaac gated attach / payload transport / release
- Summary: Completed the real Isaac-backed P4.2 v1 rollout for `contact_model="kinematic_payload_coupled_attach_v1"`. The P4.2 probe now consumes P2/P3-selected `ContactCandidateSet` and `ContactWrenchTrajectory`, reflects the P3 assembled graph into the reset-time Isaac morphology asset, evaluates gated attach against selected candidates/anchors/slots/contact regions, applies controller-side payload coupling after attach, transports the slaved payload through a bounded P4.2 v1 payload-carry segment, emits an intended release event, and returns terminal `success` without claiming natural grasp, true fixed-joint dynamics, learned policy success, P4.3 bootstrap, or P4 full completion.
- Files changed:
  - `amsrr/simulation/p4_2_rollout.py`
  - `amsrr/simulation/p4_2_isaac_env.py`
  - `amsrr/simulation/isaac_lab_backend.py`
  - `amsrr/simulation/__init__.py`
  - `amsrr/training/p4_2_deterministic_rollout_runner.py`
  - `configs/training/p4_2_deterministic_rollout.yaml`
  - `scripts/p4_control_holon_spawn_probe.py`
  - P4.2 / Isaac backend unit tests
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: No persisted base schema change. Additive runtime/config changes include P4.2 `transport_min_displacement_m`, optional P4.2 env object pose/size/mass overrides from the sampled P2 TaskSpec, backend/CLI propagation of pregrasp distance, real-probe candidate/trajectory JSON path inputs, CPU-only Isaac/Warp environment fallbacks, and P4.2 metrics for payload controller record count, payload wrench delta, bounded transport displacement, and pre-attach pose hold.
- Upstream dependencies used: P2 selected `DesignOutput`, P3 assembled `MorphologyGraph`, selected ContactCandidate / RobotAnchor assignments from the P4.2 deterministic trajectory, QPID payload coupling from Order 6a, split acceptance from Order 5, and IsaacLab backend command surface.
- Implementation notes: The sampled P2 object pose/size/mass now drive Isaac object spawning instead of static env defaults. Before attach, the probe holds the object at the deterministic TaskSpec-derived pose so selected contact regions do not drift from gravity before the attach gate; this is not an attach and does not constrain the object to the robot. After gated attach, the object is slaved to the selected anchor-relative pose and the payload gravity/effective wrench is added exactly once in the controller. Transport release uses a bounded P4.2 v1 displacement gate (`transport_min_displacement_m=0.25`) so this rollout validates payload-carry/release behavior without claiming full object-goal completion.
- Real Isaac result: `PYTHONPATH=/home/leus/amsrr python3 scripts/p4_2_deterministic_rollout.py --config configs/training/p4_2_deterministic_rollout.yaml --real` exited 0 with `completion_passed=true`, `real_isaac_rollout_passed=true`, one attach event, one release event, 1047 per-step records, graph asset/module/actuator reflection true, `p4_2_payload_controller_metric_record_count=109`, `p4_2_payload_wrench_delta_norm=12.721902086272879`, `p4_2_transport_displacement_m=0.25349208644967913`, and no object_drop / hard_collision / controller_qp_infeasible_terminal / timeout_failure.
- Tests added or run:
  - Added/updated coverage for P4.2 backend command candidate/trajectory/pregrasp arguments, runner propagation of P2 sampled object pose/size/mass, non-fatal QP/controller attach status semantics, and writable Warp cache setup.
  - `python3 -m py_compile scripts/p4_control_holon_spawn_probe.py amsrr/simulation/isaac_lab_backend.py amsrr/simulation/p4_2_isaac_env.py amsrr/simulation/p4_2_rollout.py amsrr/training/p4_2_deterministic_rollout_runner.py`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/unit/simulation/test_p4_2_rollout.py tests/unit/simulation/test_p4_2_isaac_env.py tests/unit/training/test_p4_2_deterministic_rollout_runner.py tests/acceptance/test_p4_2_acceptance.py`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/unit tests/acceptance`
  - `PYTHONPATH=/home/leus/amsrr python3 scripts/p4_2_deterministic_rollout.py --config configs/training/p4_2_deterministic_rollout.yaml --real`
- Tests run: Targeted P4.2 tests passed: 19 passed. Full unit/acceptance suite passed: 184 passed, 1 skipped. Real CLI gate passed and wrote `artifacts/p4_2/p4_2_deterministic_rollout.jsonl`.
- Assumptions: P4.2 v1 completion is success under `kinematic_payload_coupled_attach_v1` and the bounded payload-carry displacement gate only. It is not natural contact grasp success, not true fixed-joint dynamics success, not learned policy success, not P4.3 bootstrap, and not P4 full completion.
- Blockers: None for P4.2 v1 real gate.
- Next steps / handoff: Keep `P4.4 / P4-natural-contact-grasp validation` open. P4.4 should remove the pre-attach pose hold and kinematic/slaved object constraint, treat the object as a free rigid body, evaluate real contact/friction/contact maintenance from the selected anchors/candidates/assignments, log slip/contact break/drop/unintended collision/excess penetration/contact force, and compare `kinematic_payload_coupled_attach_v1` with `natural_contact_grasp_v1`.

### 2026-07-10
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4 plus approved P4.2 payload-coupled kinematic attach clarification
- Work package / Agent label: Agent J/K/L boundary: P4.2 Order 6a contact-model contract, payload coupling, release/archive acceptance
- Summary: Updated P4.2 v1 from fixed-joint wording to `contact_model="kinematic_payload_coupled_attach_v1"`. The rollout contract now requires gated attach evidence with snap distance / relative pose / phase-timeout checks, release event archives, and controller-side payload coupling evidence. The QPID controller now applies payload mass/inertia/CoM gravity/effective wrench to the target wrench before allocation and archives before/after target wrench, payload wrench, achieved wrench, residual, QP, clipped, missing, and unsupported actuator metrics. Acceptance now rejects archives that lack release events or fail to prove payload-coupled controller computation.
- Files changed:
  - `amsrr/controllers/controller_base.py`
  - `amsrr/controllers/qpid_controller.py`
  - `amsrr/controllers/__init__.py`
  - `amsrr/simulation/p4_2_rollout.py`
  - `amsrr/simulation/p4_2_isaac_env.py`
  - `amsrr/simulation/isaac_lab_backend.py`
  - `amsrr/simulation/__init__.py`
  - `amsrr/training/p4_2_deterministic_rollout_runner.py`
  - `amsrr/acceptance/p4_2_acceptance.py`
  - `amsrr/policies/contact_wrench_trajectory.py`
  - `configs/training/p4_2_deterministic_rollout.yaml`
  - `scripts/p4_control_holon_spawn_probe.py`
  - P4.2 unit/acceptance tests
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: No base persisted schema change. Added `PayloadCoupling` to controller context, `P4_2ReleaseEvent`, stricter P4.2 attach event fields, snap-distance config, release-event archive fields, and acceptance-report payload/release booleans.
- Upstream dependencies used: User-approved P4.2 v1 payload-coupled kinematic attach requirements, prior P4.2 state-machine/runner/split-acceptance contracts, P2 selected design, P3 assembled morphology, existing controller/QP allocation APIs, and Isaac backend command boundary.
- Downstream impact: P4.2 fake-backend tests can still exercise the archive fast gate, but P4.2 completion still requires a real Isaac-backed deterministic rollout. A successful archive must prove payload load changed controller computation, not merely log payload metadata.
- Tests added or run:
  - Added unit coverage that `PayloadCoupling` changes QPID target wrench and archives payload/achieved wrench metrics.
  - Added P4.2 contract/env/runner/acceptance coverage for release events, snap-distance attach fields, no true fixed-joint success claim, and payload-coupling fast-gate rejection.
  - `python3 -m py_compile amsrr/simulation/p4_2_rollout.py amsrr/simulation/p4_2_isaac_env.py amsrr/simulation/isaac_lab_backend.py amsrr/training/p4_2_deterministic_rollout_runner.py amsrr/acceptance/p4_2_acceptance.py amsrr/controllers/controller_base.py amsrr/controllers/qpid_controller.py scripts/p4_control_holon_spawn_probe.py tests/unit/simulation/test_p4_2_rollout.py tests/unit/simulation/test_p4_2_isaac_env.py tests/unit/training/test_p4_2_deterministic_rollout_runner.py tests/acceptance/test_p4_2_acceptance.py tests/unit/controllers/test_qpid_controller.py`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/unit/simulation/test_p4_2_rollout.py tests/unit/simulation/test_p4_2_isaac_env.py tests/unit/training/test_p4_2_deterministic_rollout_runner.py tests/acceptance/test_p4_2_acceptance.py tests/unit/controllers/test_qpid_controller.py`
  - `git diff --check`
- Tests run: Targeted P4.2/controller tests passed: 34 passed. Py compile and diff check passed. Plain pytest without disabling external plugin autoload failed before collecting tests because the installed `launch_testing` pytest plugin has an incompatible hook signature; rerun with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` passed.
- Assumptions: Order 6a updates contracts, controller computation, archive fields, fake-gate acceptance, and command surfaces only. The real Isaac probe still needs Order 6b to execute gated attach/release and payload-coupled rollout behavior in the simulator.
- Blockers: None for Order 6a.
- Next steps / handoff: Commit Order 6a, then implement Order 6b real Isaac gated attach/release rollout. Keep `P4.4 / P4-natural-contact-grasp validation` explicitly open for later; P4.4 should compare `kinematic_payload_coupled_attach_v1` with `natural_contact_grasp_v1` using a free rigid-body object, real contact/friction/contact maintenance, slip/contact-break/drop/unintended-collision/penetration/contact-force logging, and no kinematic object constraint.

### 2026-07-10
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4 plus P4.2 deterministic rollout user clarifications
- Work package / Agent label: Agent L boundary: P4.2 split acceptance
- Summary: Added P4.2 split acceptance. The fast gate checks archives for P2 selected design, P3 assembled morphology, deterministic phase trajectory, selected contact candidates/assignments, per-step runtime/policy/controller/actuator logs, gated object attach events, graph-specific morphology reflection, and no-mislabeling artifacts. The real gate separately requires the named rollout `p2_p3_deterministic_grasp_carry` to be attempted, passed, Isaac-backed, P2/P3-sourced, graph-reflected, final `success`, and backed by attach/per-step records.
- Files changed:
  - `amsrr/acceptance/p4_2_acceptance.py`
  - `amsrr/acceptance/__init__.py`
  - `amsrr/training/p4_2_deterministic_rollout_runner.py`
  - `scripts/p4_2_deterministic_rollout.py`
  - `tests/acceptance/test_p4_2_acceptance.py`
  - `tests/unit/training/test_p4_2_deterministic_rollout_runner.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: No persisted schema changes. Added acceptance report dataclass, package export, runner acceptance output, and CLI completion gating.
- Upstream dependencies used: P4.2 contract/result, P4.2 runner archives, P4.2 required real rollout name, and user clarification that fake gate and real Isaac rollout gate must remain separate.
- Downstream impact: P4.2 can now have fast archive tests without allowing completion. Actual P4.2 completion requires a real Isaac-backed successful rollout result and cannot be passed by fake backend archives alone.
- Tests added or run:
  - Added acceptance coverage that fake-backed archives pass the fast gate but cannot pass completion, that a real Isaac-backed rollout result is required for completion, and that missing attach events fail the fast gate.
  - `PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/acceptance/test_p4_2_acceptance.py tests/unit/training/test_p4_2_deterministic_rollout_runner.py -q`
  - `PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/acceptance/test_p4_2_acceptance.py tests/acceptance/test_p4_1_acceptance.py tests/unit/training/test_p4_2_deterministic_rollout_runner.py tests/unit/training/test_p4_1_backend_smoke_runner.py -q`
  - `PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/simulation/test_p4_2_isaac_env.py tests/unit/simulation/test_p4_2_rollout.py tests/unit/policies/test_p4_2_deterministic_policies.py -q`
  - `PYTHONDONTWRITEBYTECODE=1 python3 scripts/p4_2_deterministic_rollout.py --config configs/training/p4_2_deterministic_rollout.yaml --archive-path /tmp/amsrr_p4_2_dry.jsonl`
  - `python3 -m compileall amsrr/acceptance amsrr/training scripts tests/acceptance/test_p4_2_acceptance.py tests/unit/training/test_p4_2_deterministic_rollout_runner.py -q`
  - `git diff --check`
- Tests run: New acceptance/runner tests passed: 6 passed. Related acceptance/runner tests passed: 12 passed. Related P4.2 simulation/policy tests passed: 14 passed. CLI dry-run exited 0 with `completion_passed=false`. Compileall and diff check passed.
- Assumptions: This order implements the split gate only. It does not implement the real Isaac object fixed-joint attach/release mechanics or claim P4.2 completion.
- Blockers: None for Order 5.
- Next steps: Commit Order 5. The next implementation area is real P4.2 object attach/transport/release in Isaac; before coding that, confirm any remaining method-level undefined details around Isaac fixed-joint/kinematic attach mechanics.

### 2026-07-10
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4 plus P4.2 deterministic rollout user clarifications
- Work package / Agent label: Agent K boundary: P4.2 deterministic rollout runner/archive
- Summary: Added the P4.2 deterministic rollout runner and CLI. The runner builds a P2 selected design and P3 assembled morphology, samples contact candidates against the assembled graph, creates the P4.2 deterministic `ContactWrenchTrajectory`, calls the P4.2 Isaac env with the assembled `MorphologyGraph`, and archives per-step runtime/policy/controller/actuator logs plus phase transitions, attach events, candidate set, selected assignments, and no-mislabeling artifacts.
- Files changed:
  - `amsrr/training/p4_2_deterministic_rollout_runner.py`
  - `amsrr/training/__init__.py`
  - `scripts/p4_2_deterministic_rollout.py`
  - `tests/unit/training/test_p4_2_deterministic_rollout_runner.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: No persisted schema changes. Added runner/config/result dataclasses, package exports, and a CLI wrapper.
- Upstream dependencies used: P4.2 Order 1 contract/result, Order 2 deterministic planner, Order 3 Isaac env/backend boundary, existing P2 design policy, P3 assembly runner, contact candidate sampler, and `EpisodeArchive`.
- Downstream impact: P4.2 acceptance can now inspect archives for P2/P3 provenance, assembled graph usage, selected contact candidates, deterministic phase trajectory, attach event records, per-step logs, and no P4.3/P4-full/learning claims. Fake backend archives are possible for fast gate tests but record `isaac_backed=0` and `real_isaac_completion_claim=0`.
- Tests added or run:
  - Added unit coverage for config loading, P2/P3 rollout case construction, candidate/trajectory generation, fake-backend archive writing, archive JSONL roundtrip, and no real-completion/no-learning claims.
  - `PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/training/test_p4_2_deterministic_rollout_runner.py -q`
  - `PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/training/test_p4_2_deterministic_rollout_runner.py tests/unit/training/test_p4_1_backend_smoke_runner.py tests/unit/simulation/test_p4_2_isaac_env.py tests/unit/simulation/test_p4_2_rollout.py -q`
  - `PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/policies/test_p4_2_deterministic_policies.py tests/unit/policies/test_high_level_baseline.py tests/unit/policies/test_low_level_baseline.py -q`
  - `PYTHONDONTWRITEBYTECODE=1 python3 scripts/p4_2_deterministic_rollout.py --config configs/training/p4_2_deterministic_rollout.yaml --archive-path /tmp/amsrr_p4_2_dry.jsonl`
  - `python3 -m compileall amsrr/training scripts tests/unit/training/test_p4_2_deterministic_rollout_runner.py -q`
  - `git diff --check`
- Tests run: P4.2 runner tests passed: 3 passed. Combined runner/simulation tests passed: 18 passed. P4.2 policy tests passed: 8 passed. CLI dry-run exited 0 with no archives and no real rollout claim. Compileall and diff check passed.
- Assumptions: This order archives deterministic rollout evidence and fake-backend results for later fast acceptance, but does not implement split acceptance, run real Isaac completion, or claim P4.2 completion without the real gate.
- Blockers: None for Order 4.
- Next steps: Commit Order 4, then assess P4.2 split acceptance/real gate order for method-level undefined items before implementation.

### 2026-07-10
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4 plus P4.2 deterministic rollout user clarifications
- Work package / Agent label: Agent J boundary: P4.2 graph-specific Isaac env/probe
- Summary: Added the P4.2 Isaac env/backend/probe boundary for graph-specific deterministic rollout assets. The backend command now passes a serialized P3 assembled `MorphologyGraph` to the Isaac probe, and the probe generates a reset-time fixed graph morphology URDF from the graph's module poses and dock edges. This is explicitly not a π_A dynamic construction/update path: robot morphology is frozen during rollout, and any P4.2 attach/release semantics are object attach/release only.
- Files changed:
  - `amsrr/robot_model/fixed_morphology_urdf.py`
  - `amsrr/simulation/p4_2_isaac_env.py`
  - `amsrr/simulation/isaac_lab_backend.py`
  - `amsrr/simulation/__init__.py`
  - `scripts/p4_control_holon_spawn_probe.py`
  - `tests/unit/robot_model/test_fixed_morphology_urdf.py`
  - `tests/unit/simulation/test_p4_2_isaac_env.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: No persisted schema changes. Added graph-specific fixed-URDF helper, P4.2 env/report parser, backend command wrappers, and additive probe/report fields.
- Upstream dependencies used: P4.2 Order 1 contract/state machine, Order 2 deterministic policy phase metadata, existing P3 `MorphologyGraph`, Isaac backend command surface, and controller/bridge actuator mapping.
- Downstream impact: P4.2 runner/acceptance can now call a real Isaac command surface that requires P3 graph JSON and reflects graph module placement and actuator mapping. The current real probe deliberately cannot pass P4.2 completion without selected contact candidates and a gated attach event.
- Tests added or run:
  - Added unit coverage for graph-specific fixed morphology URDF generation from `MorphologyGraph`, P4.2 backend command JSON usage without `--fixed-module-count`, P4.2 env dry-run/missing-graph behavior, fake report parsing, no dynamic morphology claim propagation, and no-attach timeout rejection.
  - `PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/simulation/test_p4_2_isaac_env.py tests/unit/simulation/test_p4_2_rollout.py tests/unit/robot_model/test_fixed_morphology_urdf.py -q`
  - `PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/simulation/test_p4_1_isaac_env.py tests/unit/simulation/test_p4_1_backend_smoke.py tests/unit/robot_model/test_fixed_morphology_urdf.py -q`
  - `python3 -m py_compile scripts/p4_control_holon_spawn_probe.py`
  - `python3 -m compileall amsrr/simulation amsrr/robot_model tests/unit/simulation/test_p4_2_isaac_env.py tests/unit/robot_model/test_fixed_morphology_urdf.py -q`
- Tests run: P4.2 env/contract/URDF tests passed: 17 passed. P4.1 related regression tests passed: 14 passed. Py compile and compileall passed.
- Assumptions: This order establishes the graph-specific Isaac rollout boundary and nonpassing probe surface only. It does not implement selected contact-candidate attach gating, object fixed-joint creation/release, rollout archive writing, split P4.2 acceptance, real P4.2 completion, learning bootstrap, checkpoints, or reward-curve training.
- Blockers: None for Order 3.
- Next steps: Commit Order 3, then assess the P4.2 runner/archive order for method-level undefined items before implementation.

### 2026-07-10
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4 plus P4.2 deterministic rollout user clarifications
- Work package / Agent label: Agent H/I boundary: P4.2 deterministic π_H/π_L phase adaptation
- Summary: Added a P4.2-specific deterministic grasp/carry planner that emits explicit phase-labeled knots for `approach`, `pregrasp_align`, `attach_attempt`, `attached_maintain`, `transport`, and `release`. The planner reuses selected assignment feasibility and stores P4.2 phase/contact-model guard metadata without changing policy schemas. Updated baseline `π_L` so P4.2 phase guards are reflected as numeric `PolicyCommand.priority_weights` intent fields, keeping final actuator authority in the controller/bridge layer.
- Files changed:
  - `amsrr/policies/contact_wrench_trajectory.py`
  - `amsrr/policies/low_level_policy_base.py`
  - `amsrr/policies/__init__.py`
  - `tests/unit/policies/test_p4_2_deterministic_policies.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: No persisted schema changes. Added policy-side classes/helpers and package exports only.
- Upstream dependencies used: v0.4 Sections 19, 20, 24.5.4; P4.2 Order 1 phase/contact model contract; existing `ContactCandidateSet`, assignment feasibility, `ContactWrenchTrajectory`, `InteractionKnot`, `PolicyCommand`, and baseline low-level policy contracts.
- Downstream impact: P4.2 env/runner can now request a deterministic phase-aware π_H trajectory and per-step π_L `PolicyCommand` intent. Attach gating remains the later env/runner responsibility; this order does not perform attach or claim P4.2 completion.
- Tests added or run:
  - Added unit coverage for P4.2 phase-labeled trajectory output, schedule-state mapping, object/centroidal targets, and `π_L` phase priority intent without actuator commands.
  - `PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/policies/test_p4_2_deterministic_policies.py tests/unit/policies/test_high_level_baseline.py tests/unit/policies/test_low_level_baseline.py -q`
  - `python3 -m compileall amsrr/policies tests/unit/policies/test_p4_2_deterministic_policies.py -q`
  - `PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/policies -q`
  - `git diff --check`
- Tests run: P4.2/related high-low baseline tests passed: 8 passed. Full policy unit tests passed: 22 passed. Compileall and diff check passed.
- Assumptions: Approach/pregrasp body targets use the selected contact-candidate centroid with a conservative height offset; transport/release preserve current body-object offset when runtime observation is available. This is deterministic rollout targeting metadata and not a high-fidelity grasp controller.
- Blockers: None for Order 2.
- Next steps: Commit Order 2, then assess Order 3 P4.2 Isaac env/probe for method-level undefined items before implementation.

### 2026-07-10
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4 plus P4.2 deterministic rollout user clarifications
- Work package / Agent label: Agent J/K/L boundary: P4.2 deterministic rollout Order 1 contract/state-machine
- Summary: Added the initial P4.2 deterministic grasp/carry rollout contract. P4.2 is now explicitly a deterministic rollout with `reset`, `approach`, `pregrasp_align`, `attach_attempt`, `attached_maintain`, `transport`, `release`, and terminal success/failure phases, not a P4.1 full-scene smoke extension. The contract now uses `contact_model="kinematic_payload_coupled_attach_v1"` after the Order 6a supplement, gated attach conditions, attach event records, metric definitions for success/drop/collision/controller failure, P2/P3 morphology-reflection requirements, and no-mislabeling artifacts that reject P4.3 learning and P4 full-completion claims.
- Files changed:
  - `amsrr/simulation/p4_2_rollout.py`
  - `amsrr/simulation/__init__.py`
  - `configs/training/p4_2_deterministic_rollout.yaml`
  - `tests/unit/simulation/test_p4_2_rollout.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: No persisted schema changes. Added additive P4.2 simulation-side dataclasses, config loader, constants, helper contracts, and package exports.
- Upstream dependencies used: v0.4 Sections 23.3, 23.5, 24.5.4, 25, 26.10, 27.1; P4.2 user clarifications on state machine, gated kinematic attach, P2/P3 morphology reflection, metric definitions, split acceptance, and no P4.3/P4-full claims.
- Downstream impact: Agent J can implement a fake/real P4.2 Isaac env against the phase/attach/metric contract. Agent K can build a runner/archive path that must include P2 selected `DesignOutput`, P3 assembled `MorphologyGraph`, per-step logs, attach events, and no-mislabeling artifacts. Agent L can later implement split fast/real P4.2 acceptance.
- Tests added or run:
  - Added unit coverage for P4.2 config loading, phase state-machine definitions, gated attach conditions, metric definitions, no-mislabeling artifacts, success-result morphology requirements, and terminal failure metrics.
  - `PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/simulation/test_p4_2_rollout.py -q`
  - `python3 -m compileall amsrr/simulation tests/unit/simulation/test_p4_2_rollout.py -q`
- Tests run: P4.2 contract tests passed: 6 passed. Compileall passed.
- Assumptions: Order 1 is a contract-only order. It does not run Isaac, create rollout archives, claim object grasp/carry completion evidence, claim high-fidelity natural grasp success, claim learned policy success, claim P4.3, or claim P4 full completion.
- Blockers: None for Order 1.
- Next steps: Commit Order 1, then assess Order 2 deterministic π_H/π_L rollout target adaptation for any method-level undefined items before implementation.

### 2026-07-09
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4 plus P4.1 full-scene backend smoke user clarifications
- Work package / Agent label: Agent L boundary: P4.1 real Isaac backend smoke
- Summary: Ran the real P4.1 Isaac full-scene backend smoke through `micromamba run -n isaaclab3`. The split acceptance report passed both fast and real gates for P4.1: `fast_gate_passed=true`, `real_isaac_smoke_passed=true`, `completion_passed=true`. The smoke used the P2 selected / P3 assembled 3-module case, spawned robot + object + floor, stepped 80 frames, and saved per-step runtime observations, controller commands, actuator target records, and object pose history. Added `/artifacts/` to `.gitignore` because Isaac USD conversion generated local artifacts.
- Files changed:
  - `.gitignore`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: None.
- Tests added or run:
  - `/home/leus/.local/bin/micromamba run -n isaaclab3 python scripts/p4_1_backend_smoke.py --real --archive-path /tmp/amsrr_p4_1_backend_smoke.jsonl`
  - `PYTHONDONTWRITEBYTECODE=1 python3 - <<'PY' ... read_episode_archives_jsonl('/tmp/amsrr_p4_1_backend_smoke.jsonl') ...`
- Tests run: Real P4.1 smoke passed with one archive. Archive summary: success `true`, 80 runtime observations, 80 controller commands, 80 actuator target records, 80 object poses, assembled module count `3`, `fixed_two_module_only=0`, `p2_selected_design_used=1`, `p3_assembly_result_used=1`, `isaac_backed=1`, `p4_1_backend_smoke_passed=1`, `p4_1_full_scene_spawned=1`, vectoring joint key count `960`, dock joint key count `960`, `p4_full_completion=0`, and `object_grasp_carry_success_claim=0`.
- Assumptions: This completes the P4.1 full-scene backend smoke scope only. It does not claim object grasp/carry success, learned policy success, P4.2 rollout, or P4 full completion.
- Blockers: None for P4.1 backend smoke.
- Next steps: Future work should proceed to P4.2 contact-rich rollout/trajectory work, keeping the no-mislabeling boundary intact.

### 2026-07-09
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4 plus P4.1 full-scene backend smoke user clarifications
- Work package / Agent label: Agent L boundary: P4.1 CLI probe fix before real smoke
- Summary: Fixed `scripts/p4_1_backend_smoke.py` so direct script execution adds the repository root to `sys.path`, matching the existing Isaac probe script pattern. Verified normal Python probe reports missing current-interpreter Isaac modules, and micromamba `isaaclab3` probe reports the real backend as available.
- Files changed:
  - `scripts/p4_1_backend_smoke.py`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: None.
- Tests added or run:
  - `PYTHONDONTWRITEBYTECODE=1 python3 scripts/p4_1_backend_smoke.py --probe`
  - `/home/leus/.local/bin/micromamba run -n isaaclab3 python scripts/p4_1_backend_smoke.py --probe`
  - `PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/acceptance/test_p4_1_acceptance.py tests/unit/training/test_p4_1_backend_smoke_runner.py -q`
  - `python3 -m compileall scripts amsrr/acceptance amsrr/training -q`
- Tests run: P4.1 acceptance/runner tests passed: 6 passed. Compileall passed. Micromamba probe reported `available=true`.
- Next steps: Commit the CLI probe fix, then run the real P4.1 Isaac smoke.

### 2026-07-09
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4 plus P4.1 full-scene backend smoke user clarifications
- Work package / Agent label: Agent L boundary: P4.1 full-scene backend smoke Order 4 split acceptance
- Summary: Added P4.1 split acceptance. The fast gate checks archives for P2 selected design, P3 assembled morphology, non-2-module case evidence, per-step runtime/controller/actuator/object-pose logs, RuntimeObservation joint-state preservation, full-scene robot/object/floor spawn evidence, and no-mislabeling fields. The real gate separately requires the named real smoke `p2_p3_full_scene_backend` to be attempted, passed, Isaac-backed, full-scene, and P2/P3-sourced. `completion_passed` is only true when both gates pass.
- Files changed:
  - `amsrr/acceptance/p4_1_acceptance.py`
  - `amsrr/acceptance/__init__.py`
  - `amsrr/training/p4_1_backend_smoke_runner.py`
  - `scripts/p4_1_backend_smoke.py`
  - `tests/acceptance/test_p4_1_acceptance.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: No persisted schema changes. Added acceptance report dataclass and wired the runner/CLI to include acceptance output.
- Upstream dependencies used: P4.1 required real smoke name, P4.1 backend smoke result contract, `EpisodeArchive`, and the RuntimeObservation joint-state checker.
- Downstream impact: Fake-backend unit gates can validate archive/logging behavior, but P4.1 cannot complete unless the real Isaac smoke result is `isaac_backed=True` and passed. The CLI exits successfully for `--real` only when completion passes.
- Tests added or run:
  - Added acceptance coverage that fake-backed archives pass the fast gate but cannot pass completion, that a real Isaac-backed smoke result is required for completion, and that missing joint positions fail the fast gate.
  - `PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/acceptance/test_p4_1_acceptance.py tests/unit/training/test_p4_1_backend_smoke_runner.py -q`
  - `python3 -m compileall amsrr/acceptance amsrr/training scripts -q`
  - `PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/acceptance/test_p4_control_acceptance.py tests/acceptance/test_p4_0_acceptance.py tests/unit/simulation/test_p4_1_backend_smoke.py tests/unit/simulation/test_p4_1_isaac_env.py -q`
  - `git diff --check`
- Tests run: P4.1 acceptance/runner tests passed: 6 passed. Related acceptance/simulation tests passed: 12 passed. Compileall and diff check passed.
- Assumptions: The real Isaac smoke itself has not been run in Order 4. This order implements and tests the split gate; actual P4.1 completion still requires running the real smoke.
- Blockers: None for Order 4 implementation.
- Next steps: Commit Order 4, then run the real Isaac P4.1 smoke if the environment is available and record the result without claiming P4.2, learned policy success, object grasp/carry success, or P4 full completion.

### 2026-07-09
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4 plus P4.1 full-scene backend smoke user clarifications
- Work package / Agent label: Agent K/L boundary: P4.1 full-scene backend smoke Order 3 runner/archive
- Summary: Added the P4.1 backend smoke runner and CLI. The runner deterministically builds a P2-selected design and P3 assembled morphology before calling the backend smoke. The default seed selects the accepted `tri_anchor_support_grasp` case and the P3 assembled morphology has 3 modules, so the runner is not a fixed 2-module-only case. The runner stores per-step runtime observations, controller commands, actuator target records, and object pose history inside `EpisodeArchive` artifacts.
- Files changed:
  - `amsrr/training/p4_1_backend_smoke_runner.py`
  - `amsrr/training/__init__.py`
  - `scripts/p4_1_backend_smoke.py`
  - `tests/unit/training/test_p4_1_backend_smoke_runner.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: No persisted schema changes. Added P4.1 runner dataclasses, training exports, and a CLI wrapper only.
- Upstream dependencies used: P3 assembly runner config, P2 design distribution and deterministic `P2DesignPolicy`, `AssemblyRunner` with `SimplifiedAssemblyExecutor`, P4.1 backend/env result contract, and existing `EpisodeArchive`.
- Downstream impact: P4.1 acceptance can now inspect archive evidence that the backend smoke used P2 selected design and P3 assembled morphology and saved per-step logs. This still does not satisfy P4.1 completion without the later real Isaac gate.
- Tests added or run:
  - Added unit coverage for P4.1 runner config loading, P2/P3 case materialization, non-2-module module count, fake-backend per-step archive writing, archive roundtrip, and no-mislabeling fields.
  - `PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/training/test_p4_1_backend_smoke_runner.py tests/unit/simulation/test_p4_1_backend_smoke.py tests/unit/simulation/test_p4_1_isaac_env.py -q`
  - `python3 -m compileall amsrr/training amsrr/simulation scripts -q`
  - `PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/training/test_p3_assembly_runner.py tests/unit/training/test_p4_0_full_pipeline_runner.py tests/unit/training/test_p4_control_runner.py -q`
  - `git diff --check`
- Tests run: P4.1 targeted tests passed: 12 passed. Related runner tests passed: 8 passed. Compileall and diff check passed.
- Assumptions: The Order 3 backend command surface still passes assembled morphology to Isaac through module count/provenance because the current probe does not yet accept arbitrary P2/P3 graph geometry. The archive preserves the actual P3 assembled graph and plan as the source of truth.
- Blockers: None for Order 3.
- Next steps: Commit Order 3, then implement split P4.1 acceptance with a fast fake-backend archive gate and a real Isaac smoke gate.

### 2026-07-09
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4 plus P4.1 full-scene backend smoke user clarifications
- Work package / Agent label: Agent J/K/L boundary: P4.1 full-scene backend smoke Order 2 backend/probe boundary
- Summary: Added the P4.1 backend command/env path and extended the Isaac Holon probe with a separate full-scene smoke mode. The probe can spawn robot, object, and floor in the same stage, run a short QPID/bridge step loop, and emit per-step `RuntimeObservation`, `ControllerCommand`, actuator target record, and object pose history fields under `p4_1_*` report keys. This path is intentionally separate from P4-control hover pass/fail metrics.
- Files changed:
  - `amsrr/simulation/isaac_lab_backend.py`
  - `amsrr/simulation/p4_1_isaac_env.py`
  - `amsrr/simulation/__init__.py`
  - `scripts/p4_control_holon_spawn_probe.py`
  - `tests/unit/simulation/test_p4_1_isaac_env.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: No persisted schema changes. Added P4.1 backend/env helpers and probe CLI/report keys only.
- Upstream dependencies used: Existing IsaacLab backend command wrapper, P4-control probe conversion/spawn utilities, `build_fixed_morphology`, QPID controller, Isaac bridge, and the P4.1 joint-state contract from Order 1.
- Downstream impact: P4.1 runner/acceptance can now consume a fake or real backend report with full-scene and per-step evidence. The Order 2 fake gate does not satisfy P4.1 completion; a real Isaac full-scene smoke gate remains required.
- Tests added or run:
  - Added unit coverage for P4.1 config loading, backend command construction, dry-run/missing-backend skip behavior, fake backend report parsing, per-step log parsing, P2/P3 flag propagation, and joint-position rejection.
  - `PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/simulation/test_p4_1_backend_smoke.py tests/unit/simulation/test_p4_1_isaac_env.py -q`
  - `python3 -m compileall amsrr/simulation scripts -q`
  - `PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/simulation/test_p4_control_isaac_env.py tests/unit/simulation/test_p4_control_controller_smoke.py -q`
  - `git diff --check`
- Tests run: P4.1 targeted tests passed: 9 passed. Related P4-control simulation tests passed: 8 passed. Compileall and diff check passed.
- Assumptions: Order 2 uses the existing generated fixed-morphology asset path as the backend smoke command surface. The later P4.1 runner order must select at least one case from P2 `DesignOutput` and P3 assembled morphology before P4.1 can be considered complete.
- Blockers: None for Order 2.
- Next steps: Commit Order 2, then implement the P4.1 runner/archive order with P2/P3 selected case materialization and artifact logging.

### 2026-07-09
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4 plus P4.1 full-scene backend smoke user clarifications
- Work package / Agent label: Agent J/K/L boundary: P4.1 full-scene backend smoke Order 1 contract
- Summary: Added the initial P4.1 backend smoke contract. P4.1 is scoped as a full-scene Isaac backend smoke, not a P4-control hover rerun. The new contract records the required real smoke name, config defaults for robot/object/floor full-scene smoke, per-step runtime/controller/actuator/object-history result fields, and a RuntimeObservation joint-state checker that requires module pose/twist plus vectoring/gimbal and dock mechanism joint positions. Articulated P4.1 observations must additionally prove RigidBodyControlModel B(q)-style update metrics when the articulated flag is set.
- Files changed:
  - `amsrr/simulation/p4_1_backend_smoke.py`
  - `amsrr/simulation/__init__.py`
  - `configs/training/p4_1_backend_smoke.yaml`
  - `tests/unit/simulation/test_p4_1_backend_smoke.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: No persisted schema changes. Added P4.1 simulation-side dataclasses and helper exports only.
- Upstream dependencies used: v0.4 Sections 23.5, 24.5.3, 25, 26.10, 27.1; user P4.1 clarifications; existing `RuntimeObservation`, `ControllerCommand`, and P4-control bridge/archive contracts.
- Downstream impact: Later P4.1 orders can implement fake/real backend runners and acceptance against the new smoke result and joint-observation contract. Completion must still require a real Isaac full-scene smoke.
- Tests added or run:
  - Added unit coverage for P4.1 config loading, vectoring/dock joint position checks, empty joint-state rejection, and articulated B(q) update metric checks.
  - `PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/simulation/test_p4_1_backend_smoke.py -q`
  - `PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/simulation/test_p4_1_backend_smoke.py tests/unit/simulation/test_p4_control_isaac_env.py -q`
- Tests run: P4.1 unit contract tests passed: 4 passed. Related simulation tests passed: 10 passed.
- Assumptions: P4.1 full-scene smoke records object pose history as observation/logging evidence only; it does not claim object grasp/carry success.
- Blockers: None for Order 1.
- Next steps: Commit Order 1, then implement the P4.1 backend/env fake contract and real command path.

### 2026-07-09
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4 plus P4-control/P4a closeout after articulated multi-link correction
- Work package / Agent label: Agent I/J/K/L boundary: P4-control / P4a final closeout
- Summary: Re-ran the P4-control/P4a fast gates and real Isaac smoke gates after the articulated multi-link correction. The first real runner rerun exposed a probe branching bug where `--fixed-morphology-waypoint-smoke` referenced `fixed_articulated_joint_names` without initializing it; fixed the non-articulated waypoint branch to pass `None`. The corrected runner then passed all required real Isaac smokes (`single_module_hover`, `fixed_morphology_hover`, `fixed_morphology_waypoint`) and produced a passing P4-control/P4a acceptance report. Re-ran the articulated multi-link hover smoke as a separate correction-specific regression and confirmed 20 s hover with real module motion and q-dependent model updates.
- Files changed:
  - `scripts/p4_control_holon_spawn_probe.py`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: None.
- Upstream dependencies used: Latest articulated fixed-morphology URDF path, QPID rigid-body QP controller path, real Isaac smoke runner, P4-control split fast/real acceptance gate, and `/tmp` generated USD/archive output paths.
- Downstream impact: P4-control/P4a low-level closeout now has fresh fast pytest, acceptance, compile, diff, real runner, and articulated multi-link smoke evidence. This does not claim object grasp/carry success, learned `π_D`/`π_H`/`π_L`, dynamic closed-loop docking constraints, P4.1/P4.2/P4.3, or P4 full completion.
- Tests added or run:
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/acceptance -q`
  - `python3 -m compileall amsrr scripts -q`
  - `git diff --check`
  - Real Isaac P4-control runner with generated USD at `/tmp/amsrr_p4a_closeout_runner_usd`, archive JSONL at `/tmp/amsrr_p4a_closeout_archives.jsonl`, and result JSON at `/tmp/amsrr_p4a_closeout_runner_result.json`.
  - Real Isaac 20 s fixed-morphology articulated hover smoke with result JSON at `/tmp/amsrr_p4a_closeout_articulated_result.json`.
- Tests run: After the probe fix, unit tests passed: 136 passed, 1 skipped. Acceptance tests passed: 9 passed. `compileall` passed. `git diff --check` passed. Real runner passed all required smokes with `fast_gate_passed=true`, `real_isaac_smoke_passed=true`, and `completion_passed=true` for P4-control/P4a only. Runner metrics: `single_module_hover` final position error `0.013609 m`, final attitude error `0.002473 rad`, QP infeasible `0`, no missing/unsupported/clipped targets; `fixed_morphology_hover` final position error `0.013599 m`, final attitude error `0.000576 rad`, QP infeasible `0`, no missing/unsupported/clipped targets; `fixed_morphology_waypoint` final position error `0.016774 m`, final attitude error `0.000847 rad`, QP infeasible `0`, no missing/unsupported/clipped targets. Articulated 20 s smoke passed with hold time `20.0 s`, final position error `0.004215 m`, final attitude error `0.001329 rad`, max position error `0.022913 m`, relative module position change `0.056426 m`, relative module attitude change `0.122662 rad`, model rotor-origin change `0.047150 m`, allocation-matrix change `0.088292`, max joint position `0.122662 rad`, max joint tracking error `0.008668 rad`, QP infeasible `0`, and no missing/unsupported/clipped targets.
- Assumptions: The articulated regression remains the approved URDF-tree approximation: the selected parent-side dock mechanism moves the child module subtree, while the mating side is held to avoid an unsupported closed kinematic loop.
- Blockers: None for P4-control/P4a closeout.
- Next steps: Commit the closeout fix/log. Later work should start a new scope for closed-loop dock constraints, object grasp/carry, learned policies, or P4 full completion rather than extending this P4-control/P4a closeout.

### 2026-07-09
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4 plus articulated assembly correction
- Work package / Agent label: Agent I/J boundary: P4-control articulated multi-link assembly correction
- Summary: Corrected the prior articulated hover smoke so the fixed-morphology articulated case is a real multi-link assembly rather than a rigid root-to-root fixed morphology with independently moving dock links. Added an articulated morphology URDF generator that attaches the child module root to the parent module's selected connect dummy frame, so the parent dock mechanism joint moves the whole child module subtree. The fixed articulated smoke now observes module poses from Isaac body poses (`module_i__fc`) and requires both real relative module motion and q-dependent control-model updates before it can pass.
- Files changed:
  - `amsrr/robot_model/fixed_morphology_urdf.py`
  - `amsrr/controllers/qpid_controller.py`
  - `scripts/p4_control_holon_spawn_probe.py`
  - `tests/unit/robot_model/test_fixed_morphology_urdf.py`
  - `tests/unit/controllers/test_qpid_controller.py`
  - `tests/unit/controllers/test_rigid_body_model.py`
  - `for_codex/WORKLOG.md`
  - `for_codex/AMSRR_design_modification_by_codex.md`
- Schema/interface changes: No persisted schema change. Added `write_articulated_morphology_urdf()` and `articulated_morphology_connections()` helpers. `QPIDController` can now emit module-scoped dock mechanism commands such as `module_0:pitch_dock_mech_joint1`, avoiding accidental broadcast to every module when only one structural dock joint should move.
- Upstream dependencies used: Holon connect dummy frames, face-to-face dock relation `Rz(pi)`, Isaac body pose observations, existing QPID rigid-body model builder, and existing actuator mapping aliases for module-scoped commands.
- Downstream impact: `--fixed-morphology-articulated-hover-smoke` now validates a tree-structured articulated assembly. Non-articulated fixed hover/waypoint smokes continue to use the rigid fixed URDF path.
- Tests added or run:
  - Added unit coverage for articulated URDF frame-tree generation and q=0 connect point alignment.
  - Added unit coverage for module-scoped dock mechanism commands.
  - Added unit coverage that module pose changes alter rigid-body rotor origins, allocation matrix, and inertia.
  - `python3 -m py_compile amsrr/robot_model/fixed_morphology_urdf.py amsrr/controllers/qpid_controller.py scripts/p4_control_holon_spawn_probe.py tests/unit/robot_model/test_fixed_morphology_urdf.py tests/unit/controllers/test_qpid_controller.py tests/unit/controllers/test_rigid_body_model.py`
  - `PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/robot_model/test_fixed_morphology_urdf.py tests/unit/controllers/test_qpid_controller.py tests/unit/controllers/test_rigid_body_model.py -q`
  - `PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q`
  - Real Isaac 10 s and 20 s fixed-morphology articulated multi-link hover smokes.
- Tests run: Targeted unit tests passed: 23 passed. Full unit suite passed: 136 passed, 1 skipped. Real Isaac 10 s smoke passed with relative module position change `0.056426 m`, relative module attitude change `0.122662 rad`, model rotor-origin change `0.047150 m`, allocation-matrix change `0.088292`, final position error `0.009462 m`, and QP infeasible count `0`. Real Isaac 20 s smoke passed with hold time `20.0 s`, final position error `0.004215 m`, max position error `0.022913 m`, relative module position change `0.056426 m`, relative module attitude change `0.122662 rad`, model rotor-origin change `0.047150 m`, allocation-matrix change `0.088292`, QP infeasible count `0`, and no missing/unsupported/clipped bridge targets.
- Assumptions: The first articulated two-module smoke uses the selected parent-side connection mechanism (`module_0:pitch_dock_mech_joint1`) as the structural joint that moves the child module; the mating child-side dock mechanism is held at zero to keep a URDF tree rather than a closed kinematic loop.
- Blockers: None for the corrected two-module articulated hover smoke.
- Next steps: If closed-loop dock constraints with both mating mechanisms active are required, that should be treated as a later physics-modeling task because URDF cannot represent the closed kinematic loop directly.

### 2026-07-09
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4 plus P4-control articulated hover smoke supplement
- Work package / Agent label: Agent I/J boundary: P4-control articulated joint flight smoke
- Summary: Added explicit articulated hover smoke paths for single-module and fixed-morphology P4-control checks. `PostureTarget.joint_pos_target` can now command dock mechanism position targets through `QPIDController.dock_mechanism_commands` while unspecified dock joints still default to nominal zero hold. The Isaac probe exposes `--single-module-articulated-hover-smoke` and `--fixed-morphology-articulated-hover-smoke`, drives selected dock mechanism joints with a bounded sinusoidal target during hover, and reports both hover stability and actual joint-motion/tracking metrics.
- Files changed:
  - `amsrr/controllers/qpid_controller.py`
  - `amsrr/simulation/isaac_lab_backend.py`
  - `scripts/p4_control_holon_spawn_probe.py`
  - `tests/unit/controllers/test_qpid_controller.py`
  - `tests/unit/simulation/test_p4_control_isaac_env.py`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: No persisted schema change. Added CLI/backend helper options only. `InteractionKnot.posture_target` remains the controller-facing route for commanded internal joint posture; `PolicyCommand` still does not directly emit actuator targets.
- Upstream dependencies used: Existing P4-control hover smoke loop, QPID posture reference builder, dock mechanism actuator mapping, Isaac bridge position target path, and fixed-morphology runtime observation reconstruction.
- Downstream impact: The new smoke reports are optional diagnostic/acceptance aids for articulated low-level flight and are not added to the existing P4-control full acceptance set by default. They do not claim dynamic docking, object grasp/carry, learned policy success, or P4 full completion.
- Tests added or run:
  - Added unit coverage for QPID dock mechanism posture targets and backend command construction for articulated hover smokes.
  - `python3 -m py_compile amsrr/controllers/qpid_controller.py amsrr/simulation/isaac_lab_backend.py scripts/p4_control_holon_spawn_probe.py tests/unit/controllers/test_qpid_controller.py tests/unit/simulation/test_p4_control_isaac_env.py`
  - `PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/controllers/test_qpid_controller.py tests/unit/simulation/test_p4_control_isaac_env.py -q`
  - `PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q`
  - Real Isaac 20 s single-module articulated hover smoke.
  - Real Isaac 20 s fixed-morphology articulated hover smoke.
- Tests run: Targeted unit tests passed: 19 passed. Full unit suite passed: 133 passed, 1 skipped. Real Isaac single-module articulated hover passed with `single_module_articulated_hover_smoke_passed=true`, hold time `20.0 s`, final position error `0.000881 m`, max position error `0.022904 m`, QP infeasible count `0`, max observed joint motion `0.122956 rad`, and max joint tracking error `0.007602 rad`. Real Isaac fixed-morphology articulated hover passed with `fixed_morphology_articulated_hover_smoke_passed=true`, hold time `20.0 s`, final position error `0.000131 m`, max position error `0.022951 m`, QP infeasible count `0`, max observed joint motion `0.124773 rad`, and max joint tracking error `0.009495 rad`.
- Assumptions: Default articulated trajectory is small-amplitude sinusoidal dock mechanism motion (`0.12 rad`, `8 s` period, `1 s` warmup) to verify q-dependent control updates without deliberately destabilizing the hover.
- Blockers: None for the added articulated-hover smoke paths.
- Next steps: Provide user-facing commands for headless and GUI runs; do not treat these smokes as object grasp/carry, learned policy, dynamic docking, or P4 full completion evidence.

### 2026-07-09
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4 plus dock frame alignment supplement
- Work package / Agent label: Agent D/I/J boundary: π_D dock geometry and P4-control fixed-morphology spawn alignment
- Summary: Corrected the fixed-morphology docked pose and the upstream design geometry. Dock connections now use the face-to-face relation where pitch/yaw connect point origins coincide, z remains aligned, and x/y are reversed via `Rz(pi)`. `DockPortSpec.local_pose` now stores the connect dummy frame in the module/base frame, morphology builders compute dock-edge relative poses from selected port pairs, and the fixed-morphology URDF/probe/runner use the same dock-aligned module poses.
- Files changed:
  - `amsrr/geometry/pose_math.py`
  - `amsrr/robot_model/urdf_transforms.py`
  - `amsrr/robot_model/physical_model_builder.py`
  - `amsrr/robot_model/fixed_morphology_urdf.py`
  - `amsrr/morphology/dock_geometry.py`
  - `amsrr/morphology/graph.py`
  - `amsrr/morphology/grasp_carry_designs.py`
  - `amsrr/simulation/p4_control_controller_smoke.py`
  - `amsrr/training/p4_control_runner.py`
  - `scripts/p4_control_holon_spawn_probe.py`
  - related unit tests under `tests/unit/`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: No dataclass field changes. Semantics of `DockPortSpec.local_pose` / `PortNode.local_pose` are corrected to module-frame connect-port poses. Added internal geometry helpers only.
- Upstream dependencies used: Holon URDF connect point frame tree, `pitch_connect_point_*` / `yaw_connect_point_*`, existing pitch/yaw dock compatibility masks, fixed-morphology URDF generation, and P4-control smoke runner/probe paths.
- Downstream impact: π_D design outputs, feasibility-visible dock edges, generated fixed-morphology URDF/USDs, runtime observations, and smoke summary archives now agree on docked module geometry. Existing generated fixed USDs should be regenerated.
- Tests added or run:
  - Added unit checks that physical dock ports are module-frame poses and satisfy `src_port * Rz(pi) == relative * dst_port`.
  - Added fixed-URDF checks that generated module connect dummy frames satisfy the face-to-face relation.
  - Added minimal and grasp/carry morphology checks that all dock edges are port-aligned.
  - `python3 -m py_compile amsrr/geometry/pose_math.py amsrr/robot_model/urdf_transforms.py amsrr/robot_model/physical_model_builder.py amsrr/robot_model/fixed_morphology_urdf.py amsrr/morphology/dock_geometry.py amsrr/morphology/graph.py amsrr/morphology/grasp_carry_designs.py amsrr/simulation/p4_control_controller_smoke.py amsrr/training/p4_control_runner.py scripts/p4_control_holon_spawn_probe.py`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/robot_model/test_physical_model_builder.py tests/unit/robot_model/test_fixed_morphology_urdf.py tests/unit/morphology/test_minimal_morphology_builder.py tests/unit/morphology/test_grasp_carry_variants.py tests/unit/simulation/test_p4_control_controller_smoke.py tests/unit/training/test_p4_control_runner.py`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q`
  - Real Isaac fixed-morphology hover smoke for 20 s and fixed-morphology waypoint smoke with regenerated USDs.
- Tests run: Related unit subset passed: 17 passed. Full unit suite passed: 132 passed, 1 skipped. Real Isaac fixed hover passed with `fixed_morphology_hover_smoke_passed=true`, hold time `20.0 s`, final position error `6.41e-05 m`, max position error `0.02295 m`, and QP infeasible count `0`. Real Isaac fixed waypoint passed with final position error `0.01677 m`, hold time `1.0 s`, QP infeasible count `0`, and no missing/unsupported/clipped targets.
- Assumptions: The selected default fixed connection pairs are the first compatible free pitch/yaw pairs in sorted connect point order. `--fixed-module-spacing-m` remains as fallback metadata/legacy input, not the primary docked-pose definition when ports are available.
- Blockers: None for two-module fixed-morphology spawn geometry after regeneration.
- Next steps: User can rerun GUI fixed hover/waypoint commands with `--force-convert` to visually inspect the corrected facing dock pose.

### 2026-07-09
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4 plus P4-control hover drift fix supplement
- Work package / Agent label: Agent I/J boundary: P4-control hover stabilization debug
- Summary: Fixed the causes of the observed 5-10 s single-module hover drift. The physical model now treats each rotor's local thrust direction as thrust-frame `+z`; the URDF rotor continuous joint axis sign is used to derive reaction torque from `<m_f_rate>` instead of being treated as thrust direction. The rigid-body model's vectoring virtual lateral channel now follows the actual positive gimbal motion direction, computed from `gimbal_axis x thrust_z`, instead of assuming rotor-arm x. Dock mechanism hold commands now command the nominal zero position instead of following the current passive joint angle.
- Files changed:
  - `amsrr/robot_model/physical_model_builder.py`
  - `amsrr/controllers/rigid_body_model.py`
  - `amsrr/controllers/qpid_controller.py`
  - `tests/unit/robot_model/test_physical_model_builder.py`
  - `tests/unit/controllers/test_rigid_body_model.py`
  - `tests/unit/controllers/test_qpid_controller.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: None. This corrects URDF interpretation and controller-internal target construction only.
- Upstream dependencies used: Reference `aerial_robot_base` gimbalrotor controller, which uses rotor direction sign for `m_f_rate` reaction torque and thrust-frame z for translational force; Holon URDF gimbal/rotor joint axes; existing P4-control QP primary path.
- Downstream impact: Single-module hover uses the same primary `rigid_body_qp` path but now matches Isaac's actual thrust/vectoring geometry. Pseudoinverse remains a debug comparison path, not a completion path.
- Tests added or run:
  - Added physical-model assertions for all-local-`+z` thrust axes and alternating reaction torque coefficients `[-0.0172, 0.0172, -0.0172, 0.0172]`.
  - Added a finite-difference rigid-body model test requiring the vectoring virtual lateral axis to match positive gimbal joint motion.
  - Updated QPID dock hold expectation to zero and relaxed the tiny QP residual assertion to `1e-4`.
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/robot_model/test_physical_model_builder.py tests/unit/controllers/test_rigid_body_model.py tests/unit/controllers/test_qpid_controller.py`
  - Real Isaac 20 s single-module hover smoke via `/home/leus/.local/bin/micromamba run -n isaaclab3 /home/leus/IsaacLab/isaaclab.sh -p scripts/p4_control_holon_spawn_probe.py ... --steps 4000 --single-module-hover-smoke --hover-hold-duration-s 20.0 --no-hover-stop-on-hold --allocation-mode rigid_body_qp`
- Tests run: Unit subset passed: 19 passed. Real Isaac 20 s hover passed with `single_module_hover_smoke_passed=true`, `single_module_hover_hold_time_s=20.0`, final position error `0.000966 m`, max position error `0.022904 m`, final attitude error `0.001038 rad`, QP infeasible count `0`, clipped target count `0`, and controller clipped count `0`.
- Assumptions: This validates the single-module 20 s hover case only. It does not claim object grasp/carry, policy learning, fixed-morphology 20 s hover, or P4 full completion.
- Blockers: None for the reported single-module hover drift reproduction after this fix.
- Next steps: Let the user inspect the GUI run, then rerun broader P4-control smoke gates if desired after review.

### 2026-07-09
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4 plus P4-control hover drift diagnostic supplement
- Work package / Agent label: Agent I/J boundary: P4-control hover drift diagnostics
- Summary: Added a debug-only pseudoinverse allocation path and a conversion-time vectoring velocity limit override to investigate the user's observed 5-10 s hover drift/crash. Ran real Isaac 10 s no-stop single-module hover comparisons for QP default, pseudoinverse default, QP with 20 rad/s vectoring velocity, and QP with 20 rad/s vectoring plus higher gimbal stiffness/damping.
- Files changed:
  - `amsrr/controllers/qp_allocator_interface.py`
  - `amsrr/controllers/qpid_controller.py`
  - `amsrr/controllers/__init__.py`
  - `amsrr/robot_model/fixed_morphology_urdf.py`
  - `scripts/p4_control_holon_spawn_probe.py`
  - `tests/unit/controllers/test_qpid_controller.py`
  - `tests/unit/robot_model/test_fixed_morphology_urdf.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: None. Added debug allocator/CLI options only: `allocation_mode="rigid_body_pseudoinverse"` and `--vectoring-velocity-limit-rad-s`.
- Upstream dependencies used: Existing virtual x/z thrust channel representation, rigid-body model builder, IsaacLab URDF converter, and P4-control hover smoke probe.
- Downstream impact: The QP path remains the primary P4-control path. Pseudoinverse is available only for comparison and must not be used to claim P4-control completion.
- Tests added or run:
  - Added pseudoinverse allocator and controller selection unit tests.
  - Added fixed-module local-name joint velocity override unit test.
  - `python3 -m py_compile amsrr/controllers/qp_allocator_interface.py amsrr/controllers/qpid_controller.py amsrr/controllers/__init__.py amsrr/robot_model/fixed_morphology_urdf.py scripts/p4_control_holon_spawn_probe.py`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/controllers/test_qpid_controller.py tests/unit/robot_model/test_fixed_morphology_urdf.py tests/unit/simulation/test_p4_control_isaac_env.py -q`
  - Real Isaac no-stop 10 s single-module hover comparisons under `micromamba run -n isaaclab3`
- Commands run: Real Isaac comparisons used `/home/leus/.local/bin/micromamba run -n isaaclab3 /home/leus/IsaacLab/isaaclab.sh -p /home/leus/amsrr/scripts/p4_control_holon_spawn_probe.py ... --steps 2000 --single-module-hover-smoke --no-hover-stop-on-hold`.
- Tests run: Related unit tests passed: 21 passed. QP default reproduced drift with final position error `4.94 m`, hold time `1.745 s`, QP infeasible count `1690`, clipped count `1446`. Pseudoinverse default ended with smaller lateral error but altitude loss, final position error `0.424 m`, hold time `0.28 s`, infeasible/clipped every step. QP with vectoring velocity `20 rad/s` worsened final position error to `8.87 m`. QP with velocity `20 rad/s`, gimbal stiffness `200`, damping `20` reduced final position error to `0.469 m` but ended with attitude error `2.89 rad`; it still failed the 10 s hover.
- Assumptions: These are diagnostic comparisons, not acceptance gates. Old 1 s hold smoke remains a narrow smoke, not a long hover claim.
- Blockers: Stable 10 s hover likely needs the next controller investigation; vectoring speed alone is not sufficient.
- Next steps: Inspect rotor/vectoring sign conventions, reaction torque/yaw authority, dock joint passive motion, and PID/integrator behavior before changing acceptance thresholds.

### 2026-07-09
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4 plus P4-control Holon USD visual mesh resolution supplement
- Work package / Agent label: Agent J/K boundary: P4-control Isaac/GUI asset visibility support
- Summary: Fixed the GUI case where Kit opened and `/World/Holon` existed but the aircraft body was invisible. The runtime Holon URDF referenced `mesh/*.STL` relative to `assets/robots/holon`, but the STL files live under `module_urdf/mesh`; Isaac conversion therefore produced articulation/link Xforms without visible mesh payloads. Added a conversion-only mesh resolver that writes a temporary URDF with existing absolute STL paths before single-module conversion and also resolves mesh paths for fixed-morphology URDF generation.
- Files changed:
  - `amsrr/robot_model/fixed_morphology_urdf.py`
  - `scripts/p4_control_holon_spawn_probe.py`
  - `tests/unit/robot_model/test_fixed_morphology_urdf.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: None. Added a Python helper and probe-side conversion preparation only; no persisted schema or controller contract changed.
- Upstream dependencies used: Existing runtime Holon URDF, `module_urdf/mesh` STL assets, IsaacLab URDF converter, and previously added GUI observation options.
- Downstream impact: Users should regenerate Holon USDs with `--force-convert` to see visual geometry. Existing controller/QP smoke metrics are unchanged; generated USD artifacts remain reproducible and uncommitted.
- Tests added or run:
  - Added `test_resolved_mesh_urdf_points_asset_meshes_to_existing_files`.
  - Updated fixed-morphology URDF test to require resolved mesh paths to exist.
  - `python3 -m py_compile amsrr/robot_model/fixed_morphology_urdf.py scripts/p4_control_holon_spawn_probe.py`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/robot_model/test_fixed_morphology_urdf.py tests/unit/robot_model/test_urdf_loader.py -q`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/simulation/test_p4_control_isaac_env.py -q`
  - Real Isaac conversion under `micromamba run -n isaaclab3` with output `/tmp/amsrr_visible_mesh_probe`
  - `pxr.Usd` instance-proxy mesh count check on `/tmp/amsrr_visible_mesh_probe/holon/holon.usda`
  - Real Isaac GUI spawn with `--viz kit --keep-open-after-smoke-s 10` and output `/tmp/amsrr_visible_mesh_gui`
  - Real Isaac GUI single-module hover smoke with `--viz kit --realtime-playback --keep-open-after-smoke-s 5` and output `/tmp/amsrr_visible_mesh_gui_hover`
- Commands run: Real Isaac conversion/spawn/hover used `/home/leus/.local/bin/micromamba run -n isaaclab3 /home/leus/IsaacLab/isaaclab.sh -p /home/leus/amsrr/scripts/p4_control_holon_spawn_probe.py ... --force-convert`. The mesh-count check found 38 mesh prims when traversing USD instance proxies. The GUI hover run passed with `single_module_hover_smoke_passed=true`, `single_module_hover_steps=200`, final position error `0.014463 m`, final attitude error `0.004510 rad`, and QP infeasible count `0`.
- Assumptions: The asset mesh source of truth remains `module_urdf/mesh`; the converted USD is a generated artifact and should not be committed.
- Blockers: None for visible Holon geometry after regeneration. Kit's Stage tree may still show instanceable Xforms unless instance proxies are expanded, which is normal USD composition behavior.
- Next steps: Use the updated GUI command with `--force-convert` for inspection, then continue P4-control work only if the user requests the next phase.

### 2026-07-09
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4 plus P4-control GUI observation smoke supplement
- Work package / Agent label: Agent K/J boundary: P4-control GUI observation support
- Summary: Investigated why the user-facing GUI hover command did not open a visible window. IsaacLab 3 forces headless mode when no visualizer is selected, so `HEADLESS=0` alone is insufficient; the command must include `--viz kit`. Also fixed the earlier command guidance by adding GUI-observation probe options: `--realtime-playback` and `--keep-open-after-smoke-s`.
- Files changed:
  - `scripts/p4_control_holon_spawn_probe.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: No persisted schema change. Added CLI-only probe options for watchable GUI playback and post-smoke Kit hold-open.
- Tests added or run:
  - `python3 -m py_compile scripts/p4_control_holon_spawn_probe.py`
  - `eval "$(~/.local/bin/micromamba shell hook -s bash)" && micromamba activate isaaclab3 && PYTHONPATH=/home/leus/amsrr:$PYTHONPATH python3 scripts/p4_control_holon_spawn_probe.py --help`
  - Real Isaac GUI smoke with `--viz kit --realtime-playback --keep-open-after-smoke-s 0.2`
  - `git diff --check`
- Tests run: Help output showed `--realtime-playback` and `--keep-open-after-smoke-s`. Real Isaac GUI single-module hover smoke launched the Kit visualizer backend, passed with `single_module_hover_smoke_passed=true`, `single_module_hover_steps=200`, final position error `0.014463 m`, final attitude error `0.004510 rad`, QP infeasible count `0`, and report fields `realtime_playback=true`, `keep_open_after_smoke_s=0.2`.
- Assumptions: GUI observation is for human inspection only and does not expand acceptance. Long-duration hover remains unclaimed; earlier `--no-hover-stop-on-hold` 30 s exploratory run was observed to fail after initially holding within tolerance.
- Blockers: None for GUI observation of the existing smoke.
- Next steps: Use `--viz kit`, `--realtime-playback`, and a positive `--keep-open-after-smoke-s` for user-visible inspection commands.

### 2026-07-09
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4 plus P4-control split acceptance and smoke summary archive supplement
- Work package / Agent label: Agent K/L boundary: P4-control Order 18 smoke summary archive production
- Summary: Implemented the P4-control fast-gate archive production path for real smoke runs. `P4ControlLowLevelRunner` now builds one `EpisodeArchive` smoke summary for each attempted non-skipped real smoke when no external archives are supplied. Each archive records a free-flight smoke task, Holon morphology graph from the configured physical model, desired body pose policy command, summary controller status, summary runtime observation, summary actuator target metrics, and explicit no-P4-full-completion labels. Dry-run still produces no archives and cannot complete.
- Files changed:
  - `amsrr/simulation/p4_control_isaac_env.py`
  - `amsrr/training/p4_control_runner.py`
  - `configs/training/p4_control_low_level.yaml`
  - `tests/unit/training/test_p4_control_runner.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: No persisted schema change. Added `robot_model_config_path` to `P4ControlLowLevelRunnerConfig` with a default of `configs/robot/robot_model.yaml`. Added runner-side summary archive construction and flattened nested real-smoke controller/bridge numeric metrics into `P4ControlSmokeResult.metrics`.
- Upstream dependencies used: P4-control acceptance split, Order 17 all-three real smoke runner, `EpisodeArchive` runtime/controller/actuator fields, Holon physical model builder, and fixed-morphology graph helper.
- Downstream impact: When real Isaac smokes pass, `P4ControlLowLevelRunner.run()` can now produce archives that satisfy the fast gate, so `run_p4_control_acceptance` can report P4-control low-level `completion_passed=true`. This is not P4 full completion and does not cover object grasp/carry or learning.
- Tests added or run:
  - Added `test_p4_control_runner_real_smoke_builds_summary_archives`.
  - `PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/training/test_p4_control_runner.py tests/unit/simulation/test_p4_control_isaac_env.py tests/acceptance/test_p4_control_acceptance.py -q`
  - `python3 -m py_compile amsrr/training/p4_control_runner.py amsrr/simulation/p4_control_isaac_env.py tests/unit/training/test_p4_control_runner.py`
  - `PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q`
  - `git diff --check`
- Commands run:
  - `eval "$(~/.local/bin/micromamba shell hook -s bash)" && micromamba activate isaaclab3 && PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/home/leus/amsrr:$PYTHONPATH python3 -c '...'` to run `P4ControlLowLevelRunner(dry_run=False)` with real Isaac smokes, generated USD under `/tmp/amsrr_isaac_holon_runner_archive`, and summary archive output at `/tmp/amsrr_p4_control_smoke_summary.jsonl`
- Tests run: Related runner/simulation/acceptance tests passed: 12 passed. Full unit suite passed: 127 passed, 1 skipped. Diff check passed. Real Isaac runner plus summary archive passed with archive count `3`, `fast_gate_passed=true`, `real_isaac_smoke_passed=true`, `completion_passed=true`, and no acceptance failure reasons.
- Assumptions: Summary archives intentionally store smoke-level controller/runtime/actuator metrics, not full per-step actuator target replay. The archives use `rollout_artifacts.archive_type="smoke_summary"` and keep `is_p4_full_completion=false`, `physical_success_claim=false`, `object_grasp_carry_claim=false`, and `learning_claim=false`.
- Blockers: None for P4-control low-level completion. Remaining work for later phases includes per-step Isaac rollout archives, object grasp/carry rollout, learned policies, checkpoints, reward curves, and P4 full completion.
- Next steps: Commit Order 18, then pause before moving from P4-control/P4a into P4.1/P4.2/P4.3 unless the user wants the next phase started.

### 2026-07-09
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4 plus approved fixed-morphology rigid assembly representation
- Work package / Agent label: Agent J/K boundary: P4-control Order 17 fixed-morphology waypoint smoke
- Summary: Implemented and validated the real Isaac fixed-morphology waypoint smoke for the rigid 2-module Holon assembly. The Holon probe now supports `--fixed-morphology-waypoint-smoke`, ramps the direct `PolicyCommand.desired_body_pose` target from the initial root pose to the final waypoint, applies module-prefixed rotor/vectoring/dock targets through the existing bridge path, and reports `fixed_morphology_waypoint_*` metrics. The P4-control real smoke runner now executes all three low-level real smokes: single-module hover, fixed-morphology hover, and fixed-morphology waypoint.
- Files changed:
  - `amsrr/simulation/isaac_lab_backend.py`
  - `amsrr/simulation/p4_control_isaac_env.py`
  - `configs/training/p4_control_low_level.yaml`
  - `scripts/p4_control_holon_spawn_probe.py`
  - `tests/unit/simulation/test_p4_control_isaac_env.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: No persisted schema change. Added fixed-waypoint CLI/report fields, backend command/run helpers, runner execution for `fixed_morphology_waypoint`, and `waypoint_ramp_duration_s` runner config.
- Upstream dependencies used: Order 15 fixed assembly URDF generator, Order 16 fixed-morphology runtime/actuator mapping, rigid-body QP allocator, controller PID target builder, and the split P4-control acceptance rule requiring three real Isaac-backed smoke results.
- Downstream impact: The real Isaac smoke side of P4-control acceptance can now pass all three required low-level smokes. P4-control completion still remains blocked by the fast archive/interface gate until runtime observations, controller commands, actuator target records, and residual/clipping metrics are archived for the acceptance runner.
- Tests added or run:
  - Extended backend/env tests for fixed-waypoint command construction and runner result mapping.
  - `PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/simulation/test_p4_control_isaac_env.py tests/unit/simulation/test_p4_control_controller_smoke.py -q`
  - `python3 -m py_compile scripts/p4_control_holon_spawn_probe.py amsrr/simulation/isaac_lab_backend.py amsrr/simulation/p4_control_isaac_env.py tests/unit/simulation/test_p4_control_isaac_env.py`
  - `PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q`
  - `git diff --check`
- Commands run:
  - `eval "$(~/.local/bin/micromamba shell hook -s bash)" && micromamba activate isaaclab3 && PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/home/leus/amsrr:$PYTHONPATH /home/leus/IsaacLab/isaaclab.sh -p /home/leus/amsrr/scripts/p4_control_holon_spawn_probe.py ... --fixed-morphology-waypoint-smoke --fixed-module-count 2 --fixed-module-spacing-m 0.45 --waypoint-target-position-m 0.05 0.0 0.5 --waypoint-target-yaw-rad 0.0 --waypoint-ramp-duration-s 0.1 ...`
  - `eval "$(~/.local/bin/micromamba shell hook -s bash)" && micromamba activate isaaclab3 && PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/home/leus/amsrr:$PYTHONPATH python3 -c '...'` to run `P4ControlIsaacEnv.run_smokes(dry_run=False)` with generated USD under `/tmp/amsrr_isaac_holon_runner_waypoint_compact`
- Tests run: Related simulation tests passed: 8 passed. Full unit suite passed: 126 passed, 1 skipped. Diff check passed. Direct real Isaac fixed-waypoint smoke passed with `fixed_morphology_waypoint_smoke_passed=true`, `fixed_morphology_waypoint_steps=220`, hold time `1.0 s`, ramp duration `0.1 s`, final position error `0.018288 m`, final attitude error `0.018032 rad`, max position error `0.051401 m`, QP infeasible count `0`, controller clipped count `0`, and no missing/unsupported/clipped bridge targets. Runner real smoke passed all three smoke scenarios with final position errors `0.014463 m` for `single_module_hover`, `0.014247 m` for `fixed_morphology_hover`, and `0.018288 m` for `fixed_morphology_waypoint`.
- Assumptions: The waypoint target is an absolute world-frame target for the fixed assembly root. The default waypoint is intentionally small: `(0.05, 0.0, 0.5)` with `0.1 s` ramp, `0.20 m` position tolerance, `0.25 rad` attitude tolerance, and `1.0 s` hold. Larger exploratory lateral/z targets were observed to fail and are not claimed by this order.
- Blockers: None for the fixed-morphology waypoint smoke. P4-control completion remains blocked by EpisodeArchive runtime/controller/actuator logging for the real smokes; object grasp/carry success, learned policies, and P4 full completion remain unimplemented.
- Next steps: Commit Order 17, then implement the P4-control fast-gate archive production path for the real smoke runner.

### 2026-07-09
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4 plus approved fixed-morphology rigid assembly representation
- Work package / Agent label: Agent J/K boundary: P4-control Order 16 fixed-morphology hover smoke
- Summary: Implemented and validated the real Isaac fixed-morphology hover smoke for a rigid 2-module Holon assembly. The Holon probe now generates a combined fixed URDF, converts/spawns it as one Isaac articulation, reconstructs module-local runtime state from prefixed Isaac joints, applies module-prefixed rotor/vectoring/dock targets, and reports `fixed_morphology_hover_*` metrics. The runner now executes real `single_module_hover` and real `fixed_morphology_hover`; waypoint remains skipped.
- Files changed:
  - `amsrr/simulation/p4_control_controller_smoke.py`
  - `amsrr/simulation/isaac_lab_backend.py`
  - `amsrr/simulation/p4_control_isaac_env.py`
  - `configs/training/p4_control_low_level.yaml`
  - `scripts/p4_control_holon_spawn_probe.py`
  - `tests/unit/simulation/test_p4_control_controller_smoke.py`
  - `tests/unit/simulation/test_p4_control_isaac_env.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: No persisted schema change. Added fixed-hover CLI/report fields, backend command/run helpers, `fixed_morphology_module_spacing_m` runner config, and a fixed-morphology controller smoke helper.
- Upstream dependencies used: Order 15 fixed assembly URDF generator, rigid-body QP allocator, module-prefixed actuator mapping, Isaac wrench composer/body force application, and the user-approved rigid combined URDF/USD representation.
- Downstream impact: P4-control real smoke runner now has two of the three required low-level Isaac smoke results available as real Isaac-backed passes. P4-control completion remains blocked by fixed-morphology waypoint tracking and archive completeness.
- Tests added or run:
  - Added `test_fixed_morphology_controller_command_smoke_builds_multi_module_bridge_record`.
  - Extended backend/env tests for fixed-hover command construction and runner result mapping.
  - `PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/simulation/test_p4_control_isaac_env.py tests/unit/simulation/test_p4_control_controller_smoke.py -q`
  - `python3 -m py_compile scripts/p4_control_holon_spawn_probe.py amsrr/simulation/isaac_lab_backend.py amsrr/simulation/p4_control_isaac_env.py amsrr/simulation/p4_control_controller_smoke.py`
  - `PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q`
  - `git diff --check`
- Commands run:
  - `eval "$(~/.local/bin/micromamba shell hook -s bash)" && micromamba activate isaaclab3 && PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/home/leus/amsrr:$PYTHONPATH /home/leus/IsaacLab/isaaclab.sh -p /home/leus/amsrr/scripts/p4_control_holon_spawn_probe.py --config /home/leus/amsrr/configs/env/isaac_lab.yaml --force-convert --generated-usd-dir /tmp/amsrr_isaac_holon_fixed_hover --generated-usd-path /tmp/amsrr_isaac_holon_fixed_hover/holon_fixed_2/holon_fixed_2.usda --steps 600 --fixed-morphology-hover-smoke --fixed-module-count 2 --fixed-module-spacing-m 0.45 --hover-target-height 0.5 --hover-position-tolerance-m 0.20 --hover-attitude-tolerance-rad 0.25 --hover-hold-duration-s 1.0`
  - `eval "$(~/.local/bin/micromamba shell hook -s bash)" && micromamba activate isaaclab3 && PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/home/leus/amsrr:$PYTHONPATH python3 -c '...'` to run `P4ControlIsaacEnv.run_smokes(dry_run=False)` with generated USD under `/tmp/amsrr_isaac_holon_runner_fixed`
- Tests run: Related simulation tests passed: 8 passed. Full unit suite passed: 126 passed, 1 skipped. Diff check passed. Direct real Isaac fixed-hover smoke passed with `fixed_morphology_hover_smoke_passed=true`, `fixed_morphology_hover_steps=200`, hold time `1.0 s`, final position error `0.014247 m`, final attitude error `0.006313 rad`, max position error `0.022915 m`, QP infeasible count `0`, controller clipped count `0`, and no missing/unsupported/clipped bridge targets. Runner real smoke passed `single_module_hover` and `fixed_morphology_hover`; `fixed_morphology_waypoint` remained skipped with `real_isaac_execution_not_implemented`.
- Assumptions: The fixed assembly uses two modules, `0.45 m` spacing along module-0 x, and fixed root-to-root connection. This is a low-level flight validation asset, not a physical docking or P3 assembly success artifact.
- Blockers: Fixed-morphology waypoint tracking is still unimplemented. P4-control completion also remains blocked by EpisodeArchive runtime/controller/actuator logging for these real smokes.
- Next steps: Commit Order 16, then implement fixed-morphology waypoint smoke using the same generated rigid assembly and direct controller target path.

### 2026-07-09
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4 plus approved fixed-morphology rigid assembly representation
- Work package / Agent label: Agent I/J boundary: P4-control Order 15 fixed-morphology assembly asset preparation
- Summary: Prepared the fixed-morphology path without running Isaac yet. Added a deterministic URDF generator for rigid multi-module Holon assemblies and corrected `QPIDController` multi-module gravity compensation so hover/body-target wrench generation uses the current `RigidBodyControlModel.total_mass_kg` rather than single-module `PhysicalModel.aggregate_mass_kg`.
- Files changed:
  - `amsrr/robot_model/fixed_morphology_urdf.py`
  - `amsrr/controllers/qpid_controller.py`
  - `tests/unit/robot_model/test_fixed_morphology_urdf.py`
  - `tests/unit/controllers/test_qpid_controller.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: No persisted schema change. Added a robot-model utility for generating fixed-morphology URDF assets with `module_<id>__` link/joint prefixes.
- Upstream dependencies used: User approval that the first fixed-morphology smoke may use a pre-generated combined URDF/USD with fixed-joint-equivalent dock connection; existing `RigidBodyControlModelBuilder` multi-module support; existing `QPIDController` rigid-body QP path.
- Downstream impact: The next order can convert/spawn a 2-module fixed assembly in Isaac and map prefixed Isaac body/joint names back to module-local controller actuator ids. Controller hover force generation is now physically scaled for multi-module rigid assemblies.
- Tests added or run:
  - Added `test_fixed_morphology_urdf_prefixes_modules_and_keeps_single_tree`.
  - Added `test_qpid_controller_default_hover_uses_rigid_body_total_mass_for_multi_module`.
  - `PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/robot_model/test_fixed_morphology_urdf.py tests/unit/controllers/test_qpid_controller.py -q`
  - `python3 -m py_compile amsrr/robot_model/fixed_morphology_urdf.py amsrr/controllers/qpid_controller.py tests/unit/robot_model/test_fixed_morphology_urdf.py tests/unit/controllers/test_qpid_controller.py`
  - `PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q`
  - `git diff --check`
- Tests run: Related controller/robot-model tests passed: 11 passed. Full unit suite passed: 125 passed, 1 skipped. Diff check passed.
- Assumptions: Initial fixed assembly uses two Holon modules separated along the module-0 x axis by a configurable spacing, with additional module roots fixed to `module_0__root`. This is an asset-level rigid approximation for low-level controller validation, not a physical docking/detach implementation.
- Blockers: None for asset preparation. Real fixed-morphology hover/waypoint still need Isaac probe integration and prefixed actuator/body mapping.
- Next steps: Commit Order 15, then implement the real fixed-morphology hover smoke using the generated combined URDF/USD.

### 2026-07-09
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4 plus P4-control controller supplement and split real-smoke acceptance
- Work package / Agent label: Agent J/K boundary: P4-control Order 14 single-module real smoke runner integration
- Summary: Connected the validated real single-module hover smoke into the P4-control runner path. `P4ControlIsaacEnv.run_smokes(dry_run=False)` now executes `single_module_hover` through `IsaacLabBackend.run_holon_single_module_hover_smoke`, parses the probe JSON, and converts the closed-loop smoke result into `P4ControlSmokeResult` metrics. Fixed-morphology hover and waypoint results remain explicit skipped entries until their Isaac semantics are implemented.
- Files changed:
  - `amsrr/simulation/isaac_lab_backend.py`
  - `amsrr/simulation/p4_control_isaac_env.py`
  - `configs/training/p4_control_low_level.yaml`
  - `tests/unit/simulation/test_p4_control_isaac_env.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: Added internal config field `hover_target_height_m` to `P4ControlLowLevelEnvConfig` and config YAML. Added backend helper `run_holon_single_module_hover_smoke` plus force-convert support on Holon probe command builders. No persisted archive schema change.
- Upstream dependencies used: Order 13 single-module closed-loop hover probe, existing `P4ControlSmokeResult` contract, split acceptance rule requiring all three real Isaac smokes, and backend generated-USD config paths.
- Downstream impact: Runner/acceptance consumers can now see a real Isaac-backed passed `single_module_hover` smoke. P4-control completion still remains false because `fixed_morphology_hover` and `fixed_morphology_waypoint` are skipped.
- Tests added or run:
  - Added a fake-backend unit test verifying that real runner mode executes `single_module_hover`, maps JSON metrics, and skips fixed-morphology cases.
  - Extended backend command tests for `--force-convert`.
  - `PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/simulation/test_p4_control_isaac_env.py tests/unit/training/test_p4_control_runner.py -q`
  - `python3 -m py_compile amsrr/simulation/isaac_lab_backend.py amsrr/simulation/p4_control_isaac_env.py tests/unit/simulation/test_p4_control_isaac_env.py`
  - `PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q`
  - `git diff --check`
- Commands run:
  - `sed -n ... amsrr/training/p4_control_runner.py`
  - `sed -n ... amsrr/simulation/p4_control_isaac_env.py`
  - `sed -n ... amsrr/simulation/isaac_lab_backend.py`
  - `eval "$(~/.local/bin/micromamba shell hook -s bash)" && micromamba activate isaaclab3 && PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/home/leus/amsrr:$PYTHONPATH python3 -c '...'` to run `P4ControlIsaacEnv.run_smokes(dry_run=False)` with generated USD under `/tmp/amsrr_isaac_holon_runner`
- Tests run: Related simulation/runner unit tests passed: 9 passed. Full unit suite passed: 123 passed, 1 skipped. Diff check passed. Real runner smoke passed for `single_module_hover` with final position error `0.014463 m`, final attitude error `0.004510 rad`, hold time `1.0 s`, QP infeasible count `0`, and no missing/unsupported/clipped bridge targets; `fixed_morphology_hover` and `fixed_morphology_waypoint` were intentionally skipped with `real_isaac_execution_not_implemented`.
- Assumptions: The runner uses force-conversion for the single-module smoke so the real probe consumes current URDF assets. Generated USD paths are backend-configurable; the real runner verification used `/tmp` to avoid committing generated artifacts.
- Blockers: Fixed-morphology hover and waypoint smoke remain unimplemented. Their Isaac module placement, physical connection/docking representation, and target morphology semantics need a method-level decision before implementation.
- Next steps: Commit Order 14, then stop for the fixed-morphology smoke definition unless the user confirms the intended Isaac representation.

### 2026-07-09
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4 plus P4-control controller supplement and user-provided initial PID gains
- Work package / Agent label: Agent I/J/K boundary: P4-control Order 13 single-module closed-loop hover smoke
- Summary: Added a real Isaac-backed closed-loop single-module hover smoke. The probe now keeps one persistent `QPIDController(allocation_mode="rigid_body_qp")`, rebuilds runtime observation from Isaac root/joint state every step, sends a direct `PolicyCommand.desired_body_pose` / `desired_body_twist` hover target, converts the bridge-supported `ControllerCommand` through `IsaacControllerBridge`, and applies rotor thrust plus vectoring/dock joint position targets in Isaac until the configured hold duration is achieved.
- Files changed:
  - `amsrr/controllers/qpid_controller.py`
  - `amsrr/simulation/isaac_lab_backend.py`
  - `amsrr/simulation/p4_control_controller_smoke.py`
  - `scripts/p4_control_holon_spawn_probe.py`
  - `tests/unit/controllers/test_qpid_controller.py`
  - `tests/unit/simulation/test_p4_control_isaac_env.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: No persisted schema change. Added `IsaacLabBackend.holon_single_module_hover_smoke_command`, a public `bridge_supported_controller_command` helper, and Isaac probe CLI/report fields for `--single-module-hover-smoke`.
- Upstream dependencies used: PolicyCommand PID target builder, Agent I rigid-body/QP allocation path, Agent J controller-to-Isaac bridge path, Holon corrected URDF/USD, and user-approved normal Isaac Lab launch in the `isaaclab3` environment.
- Downstream impact: P4-control runner/acceptance can now consume a real single-module hover smoke artifact. Fixed-morphology hover and fixed-morphology waypoint smoke gates remain outstanding and must still fail P4-control completion until implemented and passed.
- Tests added or run:
  - Extended `tests/unit/simulation/test_p4_control_isaac_env.py` for the single-module hover command-line contract.
  - Updated `test_qpid_controller_builds_pid_wrench_from_policy_body_target_and_feedforward` because attitude PID output is now angular acceleration converted through the current composite inertia.
  - `PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/simulation/test_p4_control_isaac_env.py tests/unit/simulation/test_p4_control_controller_smoke.py tests/unit/controllers/test_qpid_controller.py -q`
  - `PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q`
  - `python3 -m compileall amsrr scripts -q`
  - `git diff --check`
- Commands run:
  - `sed -n ... scripts/p4_control_holon_spawn_probe.py`
  - `sed -n ... amsrr/simulation/p4_control_controller_smoke.py`
  - `sed -n ... amsrr/controllers/qpid_controller.py`
  - `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/home/leus/amsrr python3 - <<'PY' ...` for hover/controller diagnostics
  - `eval "$(~/.local/bin/micromamba shell hook -s bash)" && micromamba activate isaaclab3 && PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/home/leus/amsrr:$PYTHONPATH /home/leus/IsaacLab/isaaclab.sh -p /home/leus/amsrr/scripts/p4_control_holon_spawn_probe.py --config /home/leus/amsrr/configs/env/isaac_lab.yaml --force-convert --generated-usd-dir /tmp/amsrr_isaac_holon_hover --generated-usd-path /tmp/amsrr_isaac_holon_hover/holon/holon.usda --steps 600 --single-module-hover-smoke --hover-target-height 0.5 --hover-position-tolerance-m 0.20 --hover-attitude-tolerance-rad 0.25 --hover-hold-duration-s 1.0`
- Tests run: Related unit tests passed: 15 passed. Full unit suite passed: 122 passed, 1 skipped. Compileall and diff check passed. Real Isaac single-module closed-loop hover smoke passed with `single_module_hover_smoke_passed=true`, `single_module_hover_steps=200`, `single_module_hover_requested_steps=600`, `single_module_hover_duration_s=1.0`, `single_module_hover_hold_time_s=1.0`, final position error `0.014463 m`, final attitude error `0.004510 rad`, max position error `0.022829 m`, max attitude error `0.004510 rad`, `single_module_hover_qp_infeasible_count=0`, no controller clipping, and no missing/unsupported/clipped bridge targets.
- Assumptions: The smoke stops early after the configured 1.0 s hold by default; this is a smoke gate, not a long-duration hover claim. `unsupported_wrench_tolerance=1e-2` is a controller-local infeasibility cutoff for small QP/back-conversion residuals, while residuals above `1e-3` still report tracking warnings. Dock mechanism hold stiffness/damping are probe settings to keep passive dock joints from drifting during single-module hover.
- Blockers: None for single-module hover smoke. Fixed-morphology hover, fixed-morphology waypoint tracking, object grasp/carry, learned policies, P4-control completion, and P4 full completion are still not implemented or claimed.
- Next steps: Commit Order 13, then wire the real smoke result into the P4-control runner/acceptance path or proceed to fixed-morphology hover once multi-module spawn/docking semantics are defined.

### 2026-07-09
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4 plus P4-control controller supplement and user-provided initial PID gains
- Work package / Agent label: Agent I boundary: P4-control Order 12 PolicyCommand PID target builder
- Summary: Added the deterministic controller-side PID target builder needed before closed-loop hover. `PolicyCommand.desired_body_pose` / `desired_body_twist` now generate a desired body wrench with gravity compensation, user-specified xy/z/roll-pitch/yaw PID gains, body-frame quaternion attitude error, feedforward/residual wrench addition, target tracking metrics, and integral anti-windup that commits only after feasible unclipped allocation.
- Files changed:
  - `amsrr/controllers/qpid_controller.py`
  - `tests/unit/controllers/test_qpid_controller.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: None.
- Upstream dependencies used: P4-control controller supplement Section 6, user-provided PID gains, existing `PolicyCommand` body target/residual wrench fields, `RuntimeObservation` pose/twist, and Agent I QP allocator contract.
- Downstream impact: The upcoming single-module closed-loop Isaac smoke can reuse one persistent `QPIDController` instance and feed direct hover/waypoint targets through `PolicyCommand` instead of bypassing π_L/controller ownership boundaries.
- Tests added or run:
  - Added `test_qpid_controller_builds_pid_wrench_from_policy_body_target_and_feedforward`.
  - `PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/controllers/test_qpid_controller.py -q`
  - `PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q`
  - `python3 -m compileall amsrr scripts -q`
  - `git diff --check`
- Commands run:
  - `sed -n ... amsrr/controllers/qpid_controller.py`
  - `sed -n ... tests/unit/controllers/test_qpid_controller.py`
  - `rg -n "root_pose_w|root_quat_w|xyzw" /home/leus/IsaacLab/source/isaaclab -S` to confirm IsaacLab 3 quaternion ordering is XYZW
- Tests run: Controller unit tests passed: 9 passed. Full unit suite passed: 122 passed, 1 skipped. Compileall and diff check passed.
- Assumptions: No fixed acceleration or torque saturation values were introduced because the user specified gains but not wrench saturation limits; QP actuator bounds and infeasible/clipped metrics remain the safety enforcement layer for this order.
- Blockers: None for PID target builder. Closed-loop Isaac hover is still not implemented and no P4-control completion is claimed.
- Next steps: Commit Order 12, then implement the real single-module closed-loop smoke using a persistent controller and direct `PolicyCommand` hover target.

### 2026-07-09
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4 plus P4-control Isaac environment recommendation
- Work package / Agent label: Agent I/J boundary: P4-control Order 11 QP feasibility tuning for controller-command smoke
- Summary: Resolved the controller smoke's false QP infeasible status. The QP solver was succeeding, but default smoothing weights pulled the first hover allocation toward the previous zero-thrust command and the post-solve hard check counted a zero-thrust vectoring angle singularity as clipping. Reduced the primary allocator regularization/previous-command weights, raised controller-level unsupported-wrench tolerance only to the small back-conversion residual scale, and held vectoring joints at current position when the back-converted rotor thrust is effectively zero.
- Files changed:
  - `amsrr/controllers/qp_allocator_interface.py`
  - `amsrr/controllers/qpid_controller.py`
  - `tests/unit/controllers/test_qpid_controller.py`
  - `tests/unit/simulation/test_p4_control_controller_smoke.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: None.
- Upstream dependencies used: Agent I `VirtualThrustQPAllocator`, controller hard check/clamp rules, Agent J controller-command smoke helper, corrected Holon URDF/USD, and real Isaac command probe path.
- Downstream impact: Controller-to-Isaac command smoke now reports QP feasible/ok with no residual or clipping violations under the single-module initial hover command. This remains a command-routing/QP-feasibility smoke, not closed-loop hover or P4-control completion.
- Tests added or run:
  - Added `test_qpid_controller_rigid_body_qp_hover_is_feasible_with_default_tolerance`.
  - Strengthened `test_single_module_controller_command_smoke_builds_bridge_record` to assert QP feasible/status ok, residual `< 1e-5`, and no clipping.
  - `PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/controllers/test_qpid_controller.py tests/unit/simulation/test_p4_control_controller_smoke.py -q`
  - `PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q`
  - `python3 -m compileall amsrr scripts -q`
  - `git diff --check`
- Commands run:
  - `sed -n ... amsrr/controllers/qp_allocator_interface.py`
  - `sed -n ... amsrr/controllers/qpid_controller.py`
  - `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/home/leus/amsrr python3 - <<'PY' ...` for QP weight, residual, and clipping diagnostics
  - `eval "$(~/.local/bin/micromamba shell hook -s bash)" && micromamba activate isaaclab3 && PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/home/leus/amsrr:$PYTHONPATH /home/leus/IsaacLab/isaaclab.sh -p /home/leus/amsrr/scripts/p4_control_holon_spawn_probe.py --config /home/leus/amsrr/configs/env/isaac_lab.yaml --force-convert --generated-usd-dir /tmp/amsrr_isaac_holon_controller_feasible --generated-usd-path /tmp/amsrr_isaac_holon_controller_feasible/holon/holon.usda --steps 80 --controller-command-smoke`
- Tests run: Focused controller/smoke tests passed: 9 passed. Full unit suite passed: 121 passed, 1 skipped. Compileall and diff check passed. The first sandboxed Isaac run failed because no CUDA GPU was visible; the approved external Isaac run passed with `command_probe_passed=true`, `controller_status.status="ok"`, `controller_status.qp_feasible=true`, `controller_qp_feasible=1.0`, `allocation_residual_norm ~= 3.83e-6`, `clipped_target_count=0.0`, `violation_count=0.0`, no missing/unsupported bridge targets, and no battery2 invalid-inertia warning.
- Assumptions: A `1e-5` controller unsupported-wrench tolerance is a numerical back-conversion tolerance, not a physical tracking-success threshold. Near-zero-thrust vectoring angles are actuator-neutral, so holding current joint position is preferable to commanding an arbitrary rate-limit boundary.
- Blockers: None for controller-command QP feasibility. This does not validate closed-loop hover, fixed-morphology hover, waypoint tracking, object carry, learned policies, or P4 full completion.
- Next steps: Commit Order 11, then proceed toward the real closed-loop single-module hover smoke path unless a method-level undefined item appears.

### 2026-07-09
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4 plus P4-control Isaac environment recommendation
- Work package / Agent label: Agent I/J/K boundary: P4-control Order 10 controller-to-Isaac command smoke
- Summary: Added a unit-testable controller command smoke builder and connected it to the real Isaac Holon probe. The new path builds a single-module morphology/runtime observation, computes a `QPIDController` command with `allocation_mode="rigid_body_qp"`, converts it through `IsaacControllerBridge`, and applies the bridge record to Isaac thrust bodies and gimbal/dock joints via the existing probe script.
- Files changed:
  - `amsrr/simulation/p4_control_controller_smoke.py`
  - `amsrr/simulation/isaac_lab_backend.py`
  - `scripts/p4_control_holon_spawn_probe.py`
  - `tests/unit/simulation/test_p4_control_controller_smoke.py`
  - `tests/unit/simulation/test_p4_control_isaac_env.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: No persisted schema change. Added controller smoke helper dataclasses/functions and backend command helper `holon_controller_command_probe_command`.
- Upstream dependencies used: `QPIDController`, `VirtualThrustQPAllocator`, `ActuatorMapping`, `IsaacControllerBridge`, corrected Holon URDF/USD, Isaac Lab wrench composer and joint target APIs.
- Downstream impact: Later real-smoke runner work can consume the controller smoke bundle and bridge record instead of hand-authored force/joint commands. The same application path can be reused for single-module closed-loop hover once QP feasibility and control loop updates are fixed.
- Tests added or run:
  - Added `tests/unit/simulation/test_p4_control_controller_smoke.py`.
  - Updated `tests/unit/simulation/test_p4_control_isaac_env.py` for the controller command probe command contract.
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/simulation/test_p4_control_controller_smoke.py tests/unit/simulation/test_p4_control_isaac_env.py tests/unit/training/test_p4_control_runner.py -q`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q`
  - `python3 -m compileall amsrr scripts -q`
  - `git diff --check`
- Commands run:
  - `sed -n ... amsrr/controllers/qpid_controller.py`
  - `sed -n ... amsrr/controllers/isaac_controller_bridge.py`
  - `sed -n ... amsrr/simulation/p4_control_isaac_env.py`
  - `PYTHONPATH=/home/leus/amsrr python3 - <<'PY' ...` to inspect command/bridge record contents
  - `eval "$(~/.local/bin/micromamba shell hook -s bash)" && micromamba activate isaaclab3 && PYTHONPATH=/home/leus/amsrr:$PYTHONPATH /home/leus/IsaacLab/isaaclab.sh -p /home/leus/amsrr/scripts/p4_control_holon_spawn_probe.py --config /home/leus/amsrr/configs/env/isaac_lab.yaml --force-convert --generated-usd-dir /tmp/amsrr_isaac_holon_controller --generated-usd-path /tmp/amsrr_isaac_holon_controller/holon/holon.usda --steps 80 --controller-command-smoke`
  - `find amsrr scripts tests -type d -name __pycache__ -prune -exec rm -rf {} +`
- Tests run: New/related controller smoke tests passed: 9 passed. Full unit suite passed: 120 passed, 1 skipped. Compileall and diff check passed. Real Isaac controller-command smoke passed with `command_probe_passed=true`, `controller_command_smoke=true`, `controller_bridge_missing_actuators=[]`, `controller_bridge_unsupported_actuators=[]`, `controller_bridge_clipped_targets=[]`, bridge target count `12`, and no battery2 invalid-inertia warning.
- Assumptions: This order intentionally filters raw controller joint-torque commands out of the bridge record because this smoke validates P4-control rotor thrust, vectoring joint target, and dock joint position surfaces. The raw zero/rotor/fixed joint torque command behavior remains visible in `raw_joint_torque_command_count`.
- Blockers: None for controller-to-Isaac command routing. The controller QP still reports `qp_feasible=false` with small residual/clipping violations (`allocation_residual_norm ~= 0.00692`), so this is not a hover pass or completion artifact.
- Next steps: Commit Order 10, then investigate/resolve the single-module hover QP feasibility and closed-loop control update path before attempting real P4-control smoke acceptance.

### 2026-07-09
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4 plus P4-control Isaac environment recommendation
- Work package / Agent label: Agent B/J boundary: P4-control Order 9 Holon battery2 inertial correction
- Summary: Investigated the recurring real Isaac PhysX warning for `battery2`. The runtime Holon URDF and reference xacro had `battery2` inertial data set to zero mass and zero inertia. Copied the symmetric `battery1` inertial origin/mass/inertia to `battery2`, added a unit guard for mesh-bearing runtime URDF links, regenerated USD under `/tmp`, and verified the battery2 warning is gone in a real Isaac command probe.
- Files changed:
  - `assets/robots/holon/holon.urdf`
  - `module_urdf/holon.urdf.xacro`
  - `tests/unit/robot_model/test_urdf_loader.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: None.
- Upstream dependencies used: Real Isaac probe logs from Orders 7/8; Holon URDF/xacro inertial data; `battery1` symmetric inertial values.
- Downstream impact: Real Isaac smoke should regenerate USD from the corrected URDF before physical tests. Holon's aggregate mass/inertia changes from the previous zero-mass battery2 asset, so controller mass/gravity compensation and QP allocation should use the corrected physical model.
- Tests added or run:
  - Added `test_asset_urdf_inertials_are_positive_for_physics_import`.
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/robot_model -q`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q`
  - `python3 -m compileall amsrr scripts -q`
  - `git diff --check`
- Commands run:
  - `rg -n "battery2|battery1|inertial|mass|ixx|iyy|izz" assets/robots/holon module_urdf configs/robot tests/unit/robot_model`
  - `sed -n ... assets/robots/holon/holon.urdf`
  - `sed -n ... module_urdf/holon.urdf.xacro`
  - `eval "$(~/.local/bin/micromamba shell hook -s bash)" && micromamba activate isaaclab3 && PYTHONPATH=/home/leus/amsrr:$PYTHONPATH /home/leus/IsaacLab/isaaclab.sh -p /home/leus/amsrr/scripts/p4_control_holon_spawn_probe.py --config /home/leus/amsrr/configs/env/isaac_lab.yaml --force-convert --generated-usd-dir /tmp/amsrr_isaac_holon_inertial --generated-usd-path /tmp/amsrr_isaac_holon_inertial/holon/holon.usda --steps 20 --hover-force-scale 0.5 --gimbal-target-rad 0.1`
  - `find amsrr scripts tests -type d -name __pycache__ -prune -exec rm -rf {} +`
- Tests run: Robot model unit tests passed: 8 passed, 1 skipped. Full unit suite passed: 119 passed, 1 skipped. Compileall and diff check passed. Real Isaac command probe with regenerated USD passed with `command_probe_passed=true`, `robot_mass_kg=1.8667402267456055`, `num_bodies=25`, `num_joints=12`, max gimbal target error `0.014317 rad`, and no battery2 invalid inertia/negative mass warning in the log.
- Assumptions: `battery2` is the symmetric counterpart of `battery1`; until a CAD-exported inertial override is provided, mirroring `battery1` inertial properties is the most conservative correction.
- Blockers: None for the inertial correction. Remaining Isaac warnings are general extension/USD/TGS notices, not the prior battery2 invalid-mass warning.
- Next steps: Commit Order 9, then wire corrected physical model/controller output into real Isaac single-module closed-loop smoke.

### 2026-07-09
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4 plus P4-control Isaac environment recommendation
- Work package / Agent label: Agent J/K boundary: P4-control Order 8 Holon Isaac wrench and joint command probe
- Summary: Extended the real Isaac Holon probe from spawn-only to command-path validation. The probe now applies world-frame `+z` wrenches to the four `thrust_.*` bodies and position targets to `gimbal.*` joints, then reports command ids, force totals, root-state deltas, gimbal actual/target positions, and a tolerance-based command pass flag.
- Files changed:
  - `amsrr/simulation/isaac_lab_backend.py`
  - `scripts/p4_control_holon_spawn_probe.py`
  - `tests/unit/simulation/test_p4_control_isaac_env.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: No persisted schema change. Added `IsaacLabBackend.holon_command_probe_command` and command-probe JSON fields in the Isaac-only script.
- Upstream dependencies used: IsaacLab `WrenchComposer.set_forces_and_torques_index`, `Articulation.set_joint_position_target_index`, Holon real body names `thrust_1..4`, Holon real gimbal joints `gimbal1..4`.
- Downstream impact: The next real-smoke order can connect `ControllerCommand` / `IsaacControllerBridge` outputs to the same Isaac body/joint command surfaces. The command probe also gives a concrete observation extraction payload for root state and joint state, but still does not evaluate closed-loop hover.
- Tests added or run:
  - Updated `tests/unit/simulation/test_p4_control_isaac_env.py` to assert the command-probe command line contract.
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/simulation/test_p4_control_isaac_env.py tests/unit/training/test_p4_control_runner.py -q`
  - `python3 -m py_compile scripts/p4_control_holon_spawn_probe.py amsrr/simulation/isaac_lab_backend.py tests/unit/simulation/test_p4_control_isaac_env.py`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q`
  - `python3 -m compileall amsrr scripts -q`
  - `git diff --check`
- Commands run:
  - `sed -n ... /home/leus/IsaacLab/source/isaaclab/isaaclab/utils/wrench_composer.py`
  - `sed -n ... /home/leus/IsaacLab/source/isaaclab_physx/isaaclab_physx/assets/articulation/articulation.py`
  - `eval "$(~/.local/bin/micromamba shell hook -s bash)" && micromamba activate isaaclab3 && PYTHONPATH=/home/leus/amsrr:$PYTHONPATH /home/leus/IsaacLab/isaaclab.sh -p /home/leus/amsrr/scripts/p4_control_holon_spawn_probe.py --config /home/leus/amsrr/configs/env/isaac_lab.yaml --convert-if-missing --generated-usd-dir /tmp/amsrr_isaac_holon_spawn --generated-usd-path /tmp/amsrr_isaac_holon_spawn/holon/holon.usda --steps 80 --hover-force-scale 0.5 --gimbal-target-rad 0.1`
  - `find amsrr scripts tests -type d -name __pycache__ -prune -exec rm -rf {} +`
- Tests run: Related P4-control env/runner tests passed: 8 passed. Full unit suite passed: 118 passed, 1 skipped. Compileall, py_compile, and diff check passed. Real Isaac command probe passed with `command_probe_passed=true`, `thrust_body_names=["thrust_1","thrust_2","thrust_3","thrust_4"]`, `gimbal_joint_names=["gimbal1","gimbal2","gimbal3","gimbal4"]`, total commanded force `13.0386 N`, and max gimbal target error `0.001052 rad` against a `0.02 rad` tolerance.
- Assumptions: The command probe's `hover-force-scale` applies a global `+z` wrench for Isaac API validation only. It is not the final rotor-axis force semantics, not QP allocation, and not a hover success criterion.
- Blockers: None for command API routing. The real logs still show the `battery2` PhysX inertia/mass warning, and the half-hover open-loop probe falls as expected; both should be handled before claiming physical hover performance.
- Next steps: Commit Order 8, then wire `ControllerCommand` / `IsaacControllerBridge` outputs into the real Isaac probe or runner and begin single-module closed-loop smoke implementation.

### 2026-07-09
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4 plus P4-control Isaac environment recommendation
- Work package / Agent label: Agent J boundary: P4-control Order 7 Holon Isaac articulation spawn probe
- Summary: Added a real Isaac Lab spawn probe for the generated Holon USD. The probe runs under `isaaclab3` / `isaaclab.sh -p`, optionally converts the URDF, creates a fresh stage, spawns `/World/Holon` as an `Articulation`, steps a few frames, and emits JSON metadata. Real execution passed with Holon reported as 25 bodies and 12 joints.
- Files changed:
  - `amsrr/simulation/isaac_lab_backend.py`
  - `scripts/p4_control_holon_spawn_probe.py`
  - `tests/unit/simulation/test_p4_control_isaac_env.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: No persisted schema change. Added an Isaac backend helper method for constructing the Holon spawn probe command.
- Upstream dependencies used: AGENTS.md micromamba/IsaacLab instructions; IsaacLab `AppLauncher`, `SimulationContext`, `ArticulationCfg`, `UsdFileCfg`, `UrdfConverter`; Holon URDF from Agent B.
- Downstream impact: Later real-smoke code can rely on the generated Holon USD being spawnable as an Isaac articulation and can consume the body/joint names from the probe. The next real-smoke order still needs wrench/force application, joint target application, and controller observation extraction.
- Tests added or run:
  - Updated `tests/unit/simulation/test_p4_control_isaac_env.py` to assert the spawn probe command contract and avoid deprecated `--headless`.
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/simulation/test_p4_control_isaac_env.py tests/unit/training/test_p4_control_runner.py -q`
  - `python3 -m py_compile scripts/p4_control_holon_spawn_probe.py amsrr/simulation/isaac_lab_backend.py tests/unit/simulation/test_p4_control_isaac_env.py`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q`
  - `python3 -m compileall amsrr scripts -q`
  - `git diff --check`
- Commands run:
  - `sed -n ... /home/leus/IsaacLab/scripts/tutorials/01_assets/add_new_robot.py`
  - `sed -n ... /home/leus/IsaacLab/source/isaaclab_assets/isaaclab_assets/robots/quadcopter.py`
  - `sed -n ... /home/leus/IsaacLab/scripts/tools/convert_urdf.py`
  - `sed -n ... /home/leus/IsaacLab/source/isaaclab/isaaclab/assets/asset_base.py`
  - `eval "$(~/.local/bin/micromamba shell hook -s bash)" && micromamba activate isaaclab3 && PYTHONPATH=/home/leus/amsrr:$PYTHONPATH /home/leus/IsaacLab/isaaclab.sh -p /home/leus/amsrr/scripts/p4_control_holon_spawn_probe.py --config /home/leus/amsrr/configs/env/isaac_lab.yaml --force-convert --generated-usd-dir /tmp/amsrr_isaac_holon_spawn --generated-usd-path /tmp/amsrr_isaac_holon_spawn/holon/holon.usda --steps 3`
  - `find amsrr scripts tests -type d -name __pycache__ -prune -exec rm -rf {} +`
- Tests run: Related P4-control env/runner tests passed: 8 passed. Full unit suite passed: 118 passed, 1 skipped. Compileall, py_compile, and diff check passed. Real Isaac spawn probe passed with `spawn_passed=true`, `isaac_backed=true`, `converted=true`, `num_bodies=25`, `num_joints=12`, and root pose near `[0, 0, 0.496]` after 3 steps.
- Assumptions: Generated USD artifacts are reproducible outputs and were written under `/tmp/amsrr_isaac_holon_spawn` for the real probe, not committed. The spawn probe is a prerequisite smoke and not a hover/control result.
- Blockers: None for single-module articulation spawn. Real Isaac logs still warn about `battery2` invalid inertia/negative mass fallback; investigate or correct URDF inertial data before trusting physical hover performance.
- Next steps: Commit Order 7, then implement a minimal Isaac wrench/joint-command probe using the spawned Holon articulation and the existing controller bridge records.

### 2026-07-09
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4 plus P4-control Isaac environment recommendation
- Work package / Agent label: Agent J boundary: P4-control Order 6 Isaac URDF conversion probe
- Summary: Probed the real IsaacLab URDF converter path for Holon. The `isaaclab3` environment exposes `isaaclab` and `torch`; `isaaclab.sh -p scripts/p4_control_smoke.py --probe` reports the backend available in Isaac Python; `convert_urdf.py` successfully converted `assets/robots/holon/holon.urdf` to `/tmp/amsrr_isaac_holon/holon/holon.usda`. Updated generated USD config/default path to match Isaac importer's output layout.
- Files changed:
  - `configs/env/isaac_lab.yaml`
  - `amsrr/simulation/isaac_lab_backend.py`
  - `tests/unit/simulation/test_p4_control_isaac_env.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: None.
- Upstream dependencies used: AGENTS.md micromamba/IsaacLab instructions; IsaacLab `scripts/tools/convert_urdf.py`; Holon URDF from Agent B.
- Downstream impact: Real smoke implementation can rely on URDF conversion succeeding and should expect generated USD under `<generated_usd_dir>/holon/holon.usda`.
- Tests added or run:
  - Updated generated USD path assertion in `tests/unit/simulation/test_p4_control_isaac_env.py`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/simulation/test_p4_control_isaac_env.py tests/unit/training/test_p4_control_runner.py -q`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q`
  - `python3 -m compileall amsrr -q`
  - `git diff --check`
- Commands run:
  - `eval "$(~/.local/bin/micromamba shell hook -s bash)" && micromamba activate isaaclab3 && python -c ...`
  - `eval "$(~/.local/bin/micromamba shell hook -s bash)" && micromamba activate isaaclab3 && PYTHONPATH=/home/leus/amsrr:$PYTHONPATH /home/leus/IsaacLab/isaaclab.sh -p /home/leus/amsrr/scripts/p4_control_smoke.py --probe`
  - `sed -n ... /home/leus/IsaacLab/scripts/tools/convert_urdf.py`
  - `eval "$(~/.local/bin/micromamba shell hook -s bash)" && micromamba activate isaaclab3 && PYTHONPATH=/home/leus/amsrr:$PYTHONPATH /home/leus/IsaacLab/isaaclab.sh -p /home/leus/IsaacLab/scripts/tools/convert_urdf.py /home/leus/amsrr/assets/robots/holon/holon.urdf /tmp/amsrr_isaac_holon --headless`
  - `find /tmp/amsrr_isaac_holon -maxdepth 4 -type f -printf ...`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q`
  - `python3 -m compileall amsrr -q`
  - `git diff --check`
  - `find amsrr tests -type d -name __pycache__ -prune -exec rm -rf {} +`
- Tests run: Related P4-control env/runner tests passed: 8 passed. Full unit suite passed: 118 passed, 1 skipped. Compileall and diff check passed. Isaac URDF conversion probe passed and produced `/tmp/amsrr_isaac_holon/holon/holon.usda`.
- Assumptions: Generated USD artifacts are reproducible outputs and are not committed. The repo config records where generated USD should live when produced under the workspace.
- Blockers: None for URDF conversion. Next real-smoke order still needs Isaac articulation spawn, force/wrench application, joint command application, and observation extraction implementation.
- Next steps: Commit Order 6, then implement or probe a minimal Isaac articulation spawn script for the generated Holon USD.

### 2026-07-09
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4 plus P4-control Isaac environment recommendation
- Work package / Agent label: Agent J/K boundary: P4-control Order 5 smoke runner configuration and dry-run harness
- Summary: Added configurable P4-control Isaac Lab backend settings, low-level smoke scenario config, smoke environment boundary, dry-run runner, and CLI probe/dry-run script. The runner defines the three required smoke names and thresholds while keeping real Isaac execution unimplemented and completion false unless later real smoke artifacts are supplied.
- Files changed:
  - `configs/env/isaac_lab.yaml`
  - `configs/training/p4_control_low_level.yaml`
  - `amsrr/simulation/isaac_lab_backend.py`
  - `amsrr/simulation/p4_control_smoke.py`
  - `amsrr/simulation/p4_control_isaac_env.py`
  - `amsrr/simulation/__init__.py`
  - `amsrr/training/p4_control_runner.py`
  - `amsrr/training/__init__.py`
  - `scripts/p4_control_smoke.py`
  - `tests/unit/simulation/test_p4_control_isaac_env.py`
  - `tests/unit/training/test_p4_control_runner.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: No persisted schema change. Added simulation/training config/result dataclasses and shared P4-control smoke-result contract.
- Upstream dependencies used: AGENTS.md Isaac Lab micromamba instructions; v0.4 Sections 23.5 and 24.5.2; controller supplement Sections 9, 11, and 13; user approval for URDF-to-USD custom articulation and wrench-composer initial path.
- Downstream impact: Later real Isaac execution code can use the config and scenario contracts. Current dry-run/probe path is safe for fast pytest and cannot claim P4-control completion.
- Tests added or run:
  - Added `tests/unit/simulation/test_p4_control_isaac_env.py`
  - Added `tests/unit/training/test_p4_control_runner.py`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/simulation/test_p4_control_isaac_env.py tests/unit/training/test_p4_control_runner.py -q`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/acceptance/test_p4_control_acceptance.py -q`
  - `python3 -m compileall amsrr -q`
  - `git diff --check`
  - `PYTHONPATH=/home/leus/amsrr python3 scripts/p4_control_smoke.py --probe`
  - `PYTHONPATH=/home/leus/amsrr python3 scripts/p4_control_smoke.py`
- Commands run:
  - `sed -n ... amsrr/utils/config.py amsrr/simulation/base.py amsrr/training/p4_0_full_pipeline_runner.py`
  - `find configs ...`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/simulation/test_p4_control_isaac_env.py tests/unit/training/test_p4_control_runner.py -q`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/acceptance/test_p4_control_acceptance.py -q`
  - `python3 -m compileall amsrr -q`
  - `git diff --check`
  - `PYTHONPATH=/home/leus/amsrr python3 scripts/p4_control_smoke.py --probe`
  - `PYTHONPATH=/home/leus/amsrr python3 scripts/p4_control_smoke.py`
  - `find amsrr tests -type d -name __pycache__ -prune -exec rm -rf {} +`
- Tests run: New P4-control env/runner tests passed: 8 passed. Full unit suite passed: 118 passed, 1 skipped. Targeted P4-control acceptance tests passed: 2 passed. Compileall, diff check, CLI probe, and CLI dry-run passed.
- Assumptions: Normal repo Python is not the Isaac runtime; `scripts/p4_control_smoke.py --probe` is expected to report Isaac Python modules unavailable unless run through the `isaaclab3` environment / `isaaclab.sh -p` path.
- Blockers: Real Isaac physics execution, URDF conversion, Holon spawn, wrench application, and observation extraction are still unimplemented.
- Next steps: Commit Order 5, then probe the actual `isaaclab3` / `isaaclab.sh -p` environment and proceed to URDF conversion or stop if Isaac API details are still undefined.

### 2026-07-09
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4 plus P4-control split acceptance requirement
- Work package / Agent label: Agent L boundary: P4-control Order 4 fast/real acceptance split
- Summary: Implemented P4-control acceptance reporting that separates fast archive/interface checks from real Isaac smoke completion. The new report exposes `fast_gate_passed`, `real_isaac_smoke_passed`, and `completion_passed`; completion cannot pass unless single-module hover, fixed-morphology hover, and fixed-morphology waypoint smoke results are all Isaac-backed and passed.
- Files changed:
  - `amsrr/acceptance/p4_control_acceptance.py`
  - `amsrr/acceptance/__init__.py`
  - `tests/acceptance/test_p4_control_acceptance.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: No persisted schema change. Added acceptance/report dataclasses only.
- Upstream dependencies used: v0.4 Sections 24.5.2 and 24.5.6; user clarification that fast pytest and real Isaac smoke gates must be separate and completion must not pass without real smoke; existing `EpisodeArchive` P4 fields.
- Downstream impact: Later Agent J/K runners can feed real smoke results into this acceptance gate. Until then, P4-control fast gate may pass but P4-control completion remains false.
- Tests added or run:
  - Added `tests/acceptance/test_p4_control_acceptance.py`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/acceptance/test_p4_control_acceptance.py -q`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/acceptance -q`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q`
  - `python3 -m compileall amsrr -q`
  - `git diff --check`
- Commands run:
  - `git status --short`
  - `sed -n ... amsrr/acceptance/p4_0_acceptance.py tests/acceptance/test_p4_0_acceptance.py`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/acceptance/test_p4_control_acceptance.py -q`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/acceptance -q`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q`
  - `python3 -m compileall amsrr -q`
  - `git diff --check`
  - `find amsrr tests -type d -name __pycache__ -prune -exec rm -rf {} +`
- Tests run: P4-control acceptance tests passed: 2 passed. Full acceptance suite passed: 9 passed. Full unit suite passed: 110 passed, 1 skipped. Compileall and diff check passed.
- Assumptions: This order does not execute Isaac; it only evaluates smoke result records supplied by a later real Isaac runner. Synthetic smoke results in tests exercise aggregation only and are not a completion artifact.
- Blockers: Real Isaac single-module/fixed-morphology hover and waypoint smoke remain unimplemented/unrun.
- Next steps: Commit Order 4. The next implementation order should create the P4-control runner/config and/or real Isaac smoke harness; method-level Isaac execution details may require user confirmation before implementation.

### 2026-07-09
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4 plus P4-control controller bridge requirements
- Work package / Agent label: Agent I/J boundary: P4-control Order 3 actuator mapping and bridge target records
- Summary: Implemented simulator-independent actuator mapping and Isaac target record conversion for P4-control. Added deterministic active actuator channels for rotor thrusts, vectoring joints, dock mechanism joints, and effort-limited joints; added a bridge that converts `ControllerCommand` into clipped actuator target records with missing/unsupported/clipped metrics and controller residual status.
- Files changed:
  - `amsrr/controllers/actuator_mapping.py`
  - `amsrr/controllers/isaac_controller_bridge.py`
  - `amsrr/controllers/__init__.py`
  - `tests/unit/controllers/test_actuator_mapping.py`
  - `tests/unit/controllers/test_isaac_controller_bridge.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: No persisted schema change. Added controller-local bridge dataclasses and JSON-compatible target record conversion for use in existing `EpisodeArchive.actuator_target_records`.
- Upstream dependencies used: v0.4 Sections 20.8, 23.5, 24.5.2, 25.1; Agent B `PhysicalModel`; Agent I `ControllerCommand`; existing `EpisodeArchive.actuator_target_records` dict field.
- Downstream impact: Agent J/K/L can consume `ActuatorMapping` and `IsaacActuatorTargetRecord` as the fast-testable bridge contract before real Isaac execution. This does not satisfy the real Isaac smoke gate by itself.
- Tests added or run:
  - Added `tests/unit/controllers/test_actuator_mapping.py`
  - Added `tests/unit/controllers/test_isaac_controller_bridge.py`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/controllers/test_actuator_mapping.py tests/unit/controllers/test_isaac_controller_bridge.py tests/unit/controllers -q`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q`
  - `python3 -m compileall amsrr -q`
  - `git diff --check`
- Commands run:
  - `git status --short`
  - `rg -n ... actuator_target ...`
  - `sed -n ... for_codex/A-MSRR_codex_ready_spec_v0_4_ja.md`
  - `sed -n ... amsrr/logging/episode_archive.py amsrr/schemas/runtime.py`
  - `python3 - <<'PY' ... physical model actuator summary ...`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/controllers/test_actuator_mapping.py tests/unit/controllers/test_isaac_controller_bridge.py tests/unit/controllers -q`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q`
  - `python3 -m compileall amsrr -q`
  - `git diff --check`
  - `find amsrr tests -type d -name __pycache__ -prune -exec rm -rf {} +`
- Tests run: Controller/mapping/bridge tests passed: 16 passed. Full unit suite passed: 110 passed, 1 skipped. Compileall and diff check passed.
- Assumptions: Single-module bridge accepts local command keys for backward compatibility; multi-module bridge requires deterministic global `module_<module_id>:<local_id>` keys. The bridge records target conversion only and does not call Isaac APIs.
- Blockers: None for the fast pytest bridge contract. Real Isaac smoke remains unrun and P4-control completion is not claimed.
- Next steps: Commit Order 3, then proceed to the next implementation order for P4-control runner/config or Isaac smoke harness, stopping if real Isaac environment details are undefined.

### 2026-07-09
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4 plus P4-control controller supplement and user virtual rotor clarification
- Work package / Agent label: Agent I: P4-control Order 2 primary virtual-thrust QP allocator
- Summary: Implemented the primary P4-control allocator path. Added rotor-arm-fixed virtual x/z thrust channels for vectoring rotors, SciPy-backed quadratic allocation with actuator bounds and linearized vectoring joint/rate constraints, back-conversion to non-negative rotor thrusts and absolute vectoring joint targets, achieved-wrench recomputation after hard check/clamp, and controller integration behind `QPIDControllerConfig(allocation_mode="rigid_body_qp")`.
- Files changed:
  - `amsrr/controllers/rigid_body_model.py`
  - `amsrr/controllers/qp_allocator_interface.py`
  - `amsrr/controllers/qpid_controller.py`
  - `amsrr/controllers/__init__.py`
  - `tests/unit/controllers/test_qpid_controller.py`
  - `tests/unit/controllers/test_rigid_body_model.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: No persisted schema change. Backward-compatible optional controller-local fields were added to `QPAllocationProblem`, `QPAllocationResult`, and `RigidBodyControlModel`.
- Upstream dependencies used: Agent I Order 1 `RigidBodyControlModel`, v0.4 Sections 20 and 24.5.2, controller supplement Section 7, and user clarification that virtual directions are rotor-arm x/z fixed and limits should be in QP constraints plus hard check/clamp.
- Downstream impact: P4-control now has a QP-primary allocator interface for controller-side unit tests. `BoundedVerticalRotorAllocator` remains available but is explicitly tagged as degraded fallback and must not be used to claim P4-control completion.
- Tests added or run:
  - Added virtual-thrust allocator back-conversion and limit/clamp tests.
  - Added controller integration test for `allocation_mode="rigid_body_qp"`.
  - Updated rigid-body model test for current q and virtual axes.
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/controllers/test_qpid_controller.py tests/unit/controllers/test_rigid_body_model.py -q`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/controllers -q`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q`
  - `python3 -m compileall amsrr -q`
  - `git diff --check`
- Commands run:
  - `sed -n ... amsrr/controllers/*.py`
  - `rg -n ... for_codex/*.md amsrr tests`
  - `python3 - <<'PY' ... import scipy ...`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/controllers/test_qpid_controller.py tests/unit/controllers/test_rigid_body_model.py -q`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/controllers -q`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q`
  - `python3 -m compileall amsrr -q`
  - `git diff --check`
  - `find amsrr tests -type d -name __pycache__ -prune -exec rm -rf {} +`
- Tests run: Targeted QP/controller/rigid-body tests passed: 10 passed. Controller unit tests passed: 11 passed. Full unit suite passed: 105 passed, 1 skipped. Compileall and diff check passed.
- Assumptions: The virtual z channel is rotor-arm fixed but sign-aligned to each rotor's positive thrust direction, because Holon includes both `+z` and `-z` rotor thrust axes. True rotor thrust magnitude is hard checked after solving because exact `sqrt(Fx^2+Fz^2)` thrust bounds are not linear QP constraints in the virtual channel parameterization.
- Blockers: None for Agent I Order 2 fast pytest gate. Real Isaac smoke is still not run and P4-control completion is not claimed.
- Next steps: Commit Agent I Order 2, then proceed to the next implementation order, likely controller bridge / actuator mapping, unless method-level details are undefined.

### 2026-07-09
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4 plus P4-control controller supplement and virtual thrust channel supplement
- Work package / Agent label: Agent I: P4-control Order 1 RigidBodyControlModel
- Summary: Implemented deterministic per-control-step rigid-body model update for P4-control. Added controller-local `RigidBodyControlModel`, `RotorControlElement`, and `RigidBodyControlModelBuilder` that compute link-level composite mass/COM/inertia, current rotor origins and axes, scalar rotor allocation columns, vectoring joint axes, dock actuator ids, and actuator limits from `PhysicalModel`, `MorphologyGraph`, and `RuntimeObservation.module_states[*].joint_positions`.
- Files changed:
  - `amsrr/controllers/rigid_body_model.py`
  - `amsrr/controllers/__init__.py`
  - `tests/unit/controllers/test_rigid_body_model.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: None to persisted schemas. Added controller-local dataclasses and exports.
- Upstream dependencies used: v0.4 Sections 20, 24.5.2, 26.9, 27.1; `for_codex/A-MSRR_QP_PID_controller_design_spec_v0_1_ja.md`; user P4-control clarification; Agent B `PhysicalModel`; Agent I controller scaffolds.
- Downstream impact: The next Agent I order can build the primary QP allocator on top of `RigidBodyControlModel.allocation_matrix_body` and rotor/vectoring metadata, while keeping `BoundedVerticalRotorAllocator` as degraded fallback only.
- Tests added or run:
  - Added `tests/unit/controllers/test_rigid_body_model.py`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/controllers/test_rigid_body_model.py -q`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/controllers -q`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q`
  - `python3 -m compileall amsrr -q`
- Commands run:
  - `git status --short`
  - `sed -n ... amsrr/controllers/*.py`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/controllers/test_rigid_body_model.py -q`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/controllers -q`
  - `python3 -m compileall amsrr -q`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q`
  - `find amsrr tests -type d -name __pycache__ -prune -exec rm -rf {} +`
- Tests run: Targeted rigid-body model tests passed: 3 passed. Controller unit tests passed: 8 passed. Full unit suite passed: 102 passed, 1 skipped in 4.83s. Compileall passed.
- Assumptions: Controller-local body frame uses composite COM as origin and base/control module orientation as attitude. Multi-module actuator ids use `module_<module_id>:<local_id>` strings.
- Blockers: None.
- Next steps: Commit Agent I Order 1, then proceed to Agent I Order 2 primary QP allocator unless method-level undefined issues appear.

### 2026-07-09
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4 plus P4-control virtual thrust channel supplement
- Work package / Agent label: P4-Control / P4a planning: virtual thrust channel and acceptance split
- Summary: Recorded the user's P4-control implementation clarifications before code changes. The supplement fixes per-control-step `q`-conditioned rigid-body model and allocation matrix updates, QP-primary allocation, vectoring rotor virtual thrust channel expansion with back-conversion to `ControllerCommand`, degraded-only status for `BoundedVerticalRotorAllocator`, and separate fast pytest versus real Isaac smoke gates.
- Files changed:
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: None. Documentation/planning only.
- Upstream dependencies used: User clarification, v0.4 Sections 20 and 24.5.2, and `for_codex/A-MSRR_QP_PID_controller_design_spec_v0_1_ja.md`.
- Downstream impact: Agent I implementation must rebuild composite inertia, COM, rotor origins, rotor axes, and allocation matrix `B(q)` every control cycle from `RuntimeObservation` joint positions. Agent L must keep fast pytest checks separate from real Isaac smoke completion.
- Tests added or run: No tests added; documentation-only supplement. `git diff --check` passed.
- Commands run:
  - `git diff --check`
  - `git diff --stat`
  - `git diff -- for_codex/AMSRR_design_modification_by_codex.md for_codex/WORKLOG.md`
- Assumptions: Virtual thrust channels are internal QP variables only; public output remains `ControllerCommand`.
- Blockers: None.
- Next steps: Commit this planning supplement, then start Agent I Order 1 `RigidBodyControlModel`.

### 2026-07-09
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4 plus controller supplement cross-reference
- Work package / Agent label: Documentation integration: main spec reference to P4-control QP/PID supplement
- Summary: Inserted references to the controller supplement into the main v0.4 design spec. Section 20.1 now points controller implementers to `for_codex/A-MSRR_QP_PID_controller_design_spec_v0_1_ja.md` for QP/PID, quasi-static rigid-body model update, QP allocation, Isaac bridge, and P4-control acceptance details. Section 24.5.2 also references the supplement and summarizes the resolved P4-control implementation decisions.
- Files changed:
  - `for_codex/A-MSRR_codex_ready_spec_v0_4_ja.md`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: None. Documentation only.
- Upstream dependencies used: User request, revised controller supplement, v0.4 Sections 20 and 24.5.2.
- Downstream impact: Future P4-control work should consult the controller supplement before changing Agent I/J/K/L implementation files. The main spec still preserves the original `π_L` intent-only rule.
- Tests added or run: No tests added or run; documentation-only change. `git diff --check` was run.
- Commands run:
  - `sed -n ... for_codex/A-MSRR_codex_ready_spec_v0_4_ja.md`
  - `git diff --check`
- Assumptions: The supplement is a controller-specific supplement, not a replacement for the full v0.4 source of truth.
- Blockers: None.
- Next steps: Proceed to detailed implementation planning for Agent I only after any remaining controller supplement details are accepted.

### 2026-07-09
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4 plus controller design draft v0.1 resolved decisions
- Work package / Agent label: Agent I/J/K/L planning: P4-control QP/PID controller decision recording
- Summary: Recorded the user's answers to the controller-spec open questions. The controller spec now fixes Python/library-based QP as the initial backend, per-thruster thrust target as the primary Isaac representation with wrench-composer fallback for custom Holon articulation, absolute vectoring joint targets, reaction torque inclusion, link-level quasi-static inertia aggregation, and configurable initial waypoint thresholds.
- Files changed:
  - `for_codex/A-MSRR_QP_PID_controller_design_spec_v0_1_ja.md`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: None. Documentation only.
- Upstream dependencies used: User answers for controller open questions; local Isaac Lab examples/docs showing both per-thruster multirotor support and wrench-composer force application paths.
- Downstream impact: Agent I can implement link-level rigid-body model update and QP allocation without asking these questions again. Agent J bridge should preserve per-rotor thrust target records even if the custom Holon Isaac backend applies them through wrench composer.
- Tests added or run: No tests added or run; documentation-only decision recording. `git diff --check` was run after edits.
- Commands run:
  - `sed -n ... /home/leus/IsaacLab/scripts/demos/quadcopter.py`
  - `sed -n ... /home/leus/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/quadcopter/quadcopter_env.py`
  - `sed -n ... /home/leus/IsaacLab/source/isaaclab_contrib/isaaclab_contrib/assets/multirotor/multirotor.py`
  - `sed -n ... for_codex/A-MSRR_QP_PID_controller_design_spec_v0_1_ja.md`
  - `git diff --check`
- Assumptions: For Isaac, the A-MSRR archive contract uses per-thruster thrust targets as the stable representation even if a backend implementation applies equivalent forces through wrench composer.
- Blockers: None for this documentation update.
- Next steps: Begin Agent I implementation planning for `rigid_body_model.py`, QP allocation, and bridge-facing actuator target records.

### 2026-07-09
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4 plus revised controller design draft v0.1
- Work package / Agent label: Agent I/J/K/L planning: P4-control QP/PID controller specification revision
- Summary: Revised the controller-specific spec per user guidance. Removed the reference-implementation notes section, rewrote the draft in Japanese, made QP allocation normative instead of pseudoinverse allocation, and added the quasi-static assembled-morphology rule: every control step updates inertia, CoM, rotor origins, and rotor axes from joint angles, then controls the whole morphology as a single rigid body.
- Files changed:
  - `for_codex/A-MSRR_QP_PID_controller_design_spec_v0_1_ja.md`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: None. Documentation only.
- Upstream dependencies used: User clarification, v0.4 Sections 20, 23.5, 24.5.2, 25, 26.9-26.10, 27.1, and the existing controller draft.
- Downstream impact: Future Agent I/J/K/L implementation should treat the revised spec as the active controller supplement and should not encode `aerial_robot_base` as a source dependency or spec reference. Allocation is QP-owned; any fallback must be explicitly labeled degraded/non-QP.
- Tests added or run: No tests added or run; documentation-only revision. `git diff --check` was run after edits.
- Commands run:
  - `sed -n ... for_codex/A-MSRR_QP_PID_controller_design_spec_v0_1_ja.md`
  - `sed -n ... for_codex/AMSRR_design_modification_by_codex.md`
  - `sed -n ... for_codex/WORKLOG.md`
  - `git diff --check`
- Assumptions: The unresolved controller details are now explicitly listed as implementation-before-coding questions in the spec.
- Blockers: None for this documentation revision.
- Next steps: Resolve the listed open questions, then begin Agent I implementation of rigid-body model update, QP allocation, and Isaac bridge contracts.

### 2026-07-09
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4 plus controller design draft v0.1
- Work package / Agent label: Agent I/J/K/L planning: P4-control QP/PID controller specification draft
- Summary: Read the temporary `aerial_robot_base` gimbal rotor controller reference at a high level under the requested branch assumptions (`underactuate_=false`, `gimbal_calc_in_fc_=true`, `gimbal_dof_=1`) and added a controller-specific design spec skeleton for the upcoming near-complete QP/PID controller work.
- Files changed:
  - `for_codex/A-MSRR_QP_PID_controller_design_spec_v0_1_ja.md`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: None. Documentation only.
- Upstream dependencies used: v0.4 Sections 20, 23.5, 24.5.2, 25, 26.9-26.10, 27.1; existing `amsrr/controllers` scaffolds; reference file `/home/leus/ros2/aerial_robot_base_ws/src/aerial_robot_base/robots/gimbalrotor/src/control/gimbalrotor_controller.cpp`.
- Downstream impact: Future P4-control implementation can refine the controller draft before editing Agent I/J/K/L source files. The draft highlights required decisions around frames, allocation matrix equations, actuator target records, Isaac bridge semantics, and QP fallback behavior.
- Tests added or run: No tests added or run; this step created documentation only.
- Commands run:
  - `wc -l /home/leus/ros2/aerial_robot_base_ws/src/aerial_robot_base/robots/gimbalrotor/src/control/gimbalrotor_controller.cpp`
  - `rg -n ... /home/leus/ros2/aerial_robot_base_ws/src/aerial_robot_base/robots/gimbalrotor/src/control/gimbalrotor_controller.cpp`
  - `sed -n ... /home/leus/ros2/aerial_robot_base_ws/src/aerial_robot_base/robots/gimbalrotor/src/control/gimbalrotor_controller.cpp`
  - `sed -n ... /home/leus/ros2/aerial_robot_base_ws/src/aerial_robot_base/robots/gimbalrotor/include/gimbalrotor/control/gimbalrotor_controller.h`
- Assumptions: The reference reading was intentionally shallow and branch-specific; exact frame conventions, Isaac actuator semantics, and QP solver choice remain open in the draft.
- Blockers: None for the documentation skeleton.
- Next steps: Refine the controller spec into concrete equations/config/tests, then start Agent I controller allocation and Isaac bridge implementation.

### 2026-07-08
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4 plus P4.0 implementation order
- Work package / Agent label: Agent K/L: P4.0 final docs and verification
- Summary: Completed P4.0 documentation handoff after Orders 1-5. Added the P4.0 implementation supplement to the design modification log, confirmed full unit and acceptance suites pass, and kept P4.0 explicitly scoped to simplified full-pipeline wiring rather than P4 full completion.
- Files changed:
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: Documentation only in this order. Earlier Order 1 added backward-compatible `EpisodeArchive` fields.
- Upstream dependencies used: Completed P4.0 Orders 1-5, v0.4 Sections 24.5.1, 25.1, 26.10, and 27.3.
- Downstream impact: Future work can start P4-control / controller bridge / actuator mapping with P4.0 simplified acceptance as a prerequisite, without confusing P4.0 with Isaac-backed P4 full completion.
- Tests added or run: No new tests added in this order.
- Commands run:
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/acceptance -q`
- Tests run: Full unit suite passed: 99 passed, 1 skipped in 4.67s. Full acceptance suite passed: 7 passed in 118.22s.
- Assumptions: P4.0 acceptance remains a simplified backend wiring gate only; Isaac-backed physical success metrics belong to later P4-control / P4.1 / P4.2 / P4 full acceptance.
- Blockers: None.
- Next steps: Proceed to controller bridge / actuator mapping and P4-control Isaac low-level flight validation before any P4 full completion claim.

### 2026-07-08
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4 plus P4.0 implementation order
- Work package / Agent label: Agent L: P4.0 simplified acceptance gate
- Summary: Added a P4.0 acceptance gate that runs the simplified full-pipeline runner, verifies P2 selected design and P3 assembly result usage, checks contact candidate / pi_H / pi_L / controller / archive completeness, records simplified success/drop/collision/QP metrics, and enforces no-mislabeling as non-Isaac and non-P4-full.
- Files changed:
  - `amsrr/acceptance/p4_0_acceptance.py`
  - `amsrr/acceptance/__init__.py`
  - `amsrr/training/p4_0_full_pipeline_runner.py`
  - `tests/acceptance/test_p4_0_acceptance.py`
- Schema/interface changes: None to persisted schemas. Added acceptance-side report/criteria dataclasses and metric aliases `collision_rate` / `qp_infeasible_rate`.
- Upstream dependencies used: Order 3 runner, Order 4 runner tests, v0.4 Section 24.5.1 P4.0 simplified acceptance and no-mislabeling text.
- Downstream impact: P4.0 can now be mechanically accepted without claiming P4 full completion. Later P4-control / P4.1 work must remain separate.
- Tests added or run: Added P4.0 acceptance test.
- Commands run:
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/acceptance/test_p4_0_acceptance.py -q`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/training/test_p4_0_full_pipeline_runner.py tests/acceptance/test_p4_0_acceptance.py -q`
  - `python3 -m compileall amsrr -q`
  - `git diff --check`
- Tests run: P4.0 acceptance targeted test passed: 1 passed. P4.0 unit + acceptance passed: 3 passed. Compileall passed. `git diff --check` passed.
- Assumptions: Because v0.4 does not set a P4.0 success-rate threshold, this gate checks wiring/archive completeness and metric recording rather than adding a new success threshold.
- Blockers: None.
- Next steps: Order 6, update docs / WORKLOG / design modification log and run final verification.

### 2026-07-08
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4 plus P4.0 implementation order
- Work package / Agent label: Agent K/L: P4.0 unit, archive completeness, and no-mislabeling tests
- Summary: Added P4.0 runner unit tests that execute the simplified full pipeline, validate archive completeness for design/feasibility/assembly/trajectory/policy/controller/runtime/reward records, and verify explicit no-mislabeling metadata stating that P4.0 is simplified, not Isaac-backed, and not P4 full completion.
- Files changed:
  - `tests/unit/training/test_p4_0_full_pipeline_runner.py`
- Schema/interface changes: None.
- Upstream dependencies used: Order 3 P4.0 runner, Order 1 archive compatibility, Order 2 simplified env injection, and v0.4 P4.0 no-mislabeling requirement.
- Downstream impact: Order 5 acceptance can build on these tested runner/archive invariants rather than duplicating every low-level field assertion.
- Tests added or run: Added P4.0 runner archive/no-mislabeling unit tests.
- Commands run:
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/training/test_p4_0_full_pipeline_runner.py -q`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/simulation/test_simplified_grasp_carry_env.py tests/unit/training/test_p1_runner.py tests/unit/training/test_p4_0_full_pipeline_runner.py -q`
  - `python3 -m compileall amsrr -q`
  - `git diff --check`
- Tests run: P4.0 runner tests passed: 2 passed. Related env/P1/P4.0 tests passed: 9 passed. Compileall passed. `git diff --check` passed.
- Assumptions: P4.0 archives intentionally leave `actuator_target_records` empty because no Isaac actuator target conversion is performed in the simplified backend.
- Blockers: None.
- Next steps: Order 5, implement the P4.0 simplified acceptance gate over the tested runner.

### 2026-07-08
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4 plus P4.0 implementation order
- Work package / Agent label: Agent K: P4.0 full-pipeline runner
- Summary: Added a P4.0 simplified full-pipeline runner that wires P2 selected `DesignOutput`, P3 simplified assembly result, morphology-conditioned contact candidates, baseline pi_H trajectory, baseline pi_L policy commands, controller commands, rewards, metrics, and `EpisodeArchive` logging. The runner records explicit simplified-backend / not-Isaac / not-P4-full metadata.
- Files changed:
  - `amsrr/training/p4_0_full_pipeline_runner.py`
  - `amsrr/training/__init__.py`
  - `configs/training/p4_0_grasp_carry.yaml`
- Schema/interface changes: None to persisted schemas. Uses the additive `EpisodeArchive` fields from Order 1.
- Upstream dependencies used: P2 design distribution/policy, P3 assembly runner/executor semantics, Order 2 simplified env external design injection, Agent H pi_H baseline, Agent I pi_L/controller scaffolds, v0.4 Section 24.5.1 P4.0 requirements.
- Downstream impact: Order 4 can add archive completeness and no-mislabeling tests against the new runner. Order 5 can implement the P4.0 acceptance gate over this runner.
- Tests added or run: No unit test files added in this order; import/config smoke and compile checks passed.
- Commands run:
  - `python3 -m compileall amsrr -q`
  - `python3 -c "from amsrr.training import load_p4_0_full_pipeline_runner_config, P4_0FullPipelineRunner; ..."`
  - `git diff --check`
- Tests run: Compileall passed. P4.0 runner config/import smoke passed. `git diff --check` passed.
- Assumptions: P3 `AssemblyRunReport.final_state.physical_graph` is the simplified assembled morphology for P4.0 wiring only and does not imply physical docking success.
- Blockers: None.
- Next steps: Order 4, add unit/archive/no-mislabeling tests for the P4.0 runner.

### 2026-07-08
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4 plus P4.0 implementation order
- Work package / Agent label: Agent JP1/K: simplified env external DesignOutput / assembled morphology injection
- Summary: Added a P4.0-compatible injection path to `SimplifiedGraspCarryEnv` so callers can provide a selected external `DesignOutput` and optional assembled morphology. The existing P1 fixed/simple default path remains unchanged, while external design paths bypass `FixedSimpleDesignPolicy`.
- Files changed:
  - `amsrr/simulation/simplified_grasp_carry_env.py`
  - `tests/unit/simulation/test_simplified_grasp_carry_env.py`
- Schema/interface changes: None to persisted schemas. Added optional concrete-env arguments and an internal build-artifact `design_source` label.
- Upstream dependencies used: P4.0 requirement to use P2 selected morphology / P3 assembled morphology downstream and avoid `FixedSimpleDesignPolicy` fixed path.
- Downstream impact: The P4.0 runner can instantiate or reset the simplified env with the P2 selected design and P3 assembled morphology before contact candidate generation, π_H, π_L, and controller execution.
- Tests added or run: Added external design/assembled morphology injection test.
- Commands run:
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/simulation/test_simplified_grasp_carry_env.py -q`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/training/test_p1_runner.py -q`
  - `python3 -m compileall amsrr -q`
  - `git diff --check`
- Tests run: Simplified env tests passed: 4 passed. P1 runner tests passed: 3 passed. Compileall passed. `git diff --check` passed.
- Assumptions: In P4.0, "assembled morphology" is represented by the successful P3 construction state's physical graph or an equivalent `MorphologyGraph`; this does not claim physical docking success.
- Blockers: None.
- Next steps: Order 3, implement the P4.0 full-pipeline runner over P2 selected design, P3 assembly result, contact candidates, π_H, π_L, controller, rewards, metrics, and archive logging.

### 2026-07-08
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4 plus P4.0 implementation order
- Work package / Agent label: Agent A/K: P4.0 archive compatibility
- Summary: Added backward-compatible P4 archive fields to `EpisodeArchive` for runtime observations, actuator target records, rollout artifacts, and learning artifacts. Existing P1/P2/P3 archive construction remains valid because the new fields use defaults.
- Files changed:
  - `amsrr/logging/episode_archive.py`
  - `tests/unit/training/test_p1_runner.py`
- Schema/interface changes: Additive `EpisodeArchive` interface fields with default empty list/dict values.
- Upstream dependencies used: v0.4 Section 25.1 EpisodeArchive contract and P4.0/P4 logging requirements.
- Downstream impact: P4.0 can archive simplified rollout records now, while later P4-control / Isaac-backed runs can fill runtime observations and actuator target records without changing the archive type again.
- Tests added or run: Added legacy archive default-field restoration assertions in the P1 runner archive roundtrip test.
- Commands run:
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/training/test_p1_runner.py -q`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/training/test_p2_design_runner.py tests/unit/training/test_p3_assembly_runner.py -q`
  - `python3 -m compileall amsrr -q`
  - `git diff --check`
- Tests run: P1 runner tests passed: 3 passed. P2/P3 runner tests passed: 5 passed. Compileall passed. `git diff --check` passed.
- Assumptions: P1-P4.0 simplified archives may leave `runtime_observations` and `actuator_target_records` empty unless a runner explicitly records them; Isaac-backed P4 must populate them per the source spec.
- Blockers: None.
- Next steps: Order 2, add external `DesignOutput` / assembled morphology injection to the simplified env without using `FixedSimpleDesignPolicy` on the P4.0 path.

### 2026-07-08
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4 plus P4.3 learning target clarification
- Work package / Agent label: P4.3 learning design revision / source-spec update
- Summary: Clarified that P4.3 learning bootstrap targets π_L/residual controller learning, π_H contact/trajectory policy learning, and π_D outcome-conditioned design scorer/selector fine-tuning, not π_L alone. Added P4.3a-P4.3e recommended order, expanded P4 full acceptance learning artifacts for all three policy families, and updated the P4 Mermaid diagram so the training loop points back to π_D, π_H, and π_L with their separate output responsibilities.
- Files changed:
  - `for_codex/A-MSRR_codex_ready_spec_v0_4_ja.md`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: Source spec only. No Python implementation files were changed.
- Upstream dependencies used: User-provided追加修正 request, current P4.3 design text, v0.4 Sections 15, 19, 20, 24.5, and P2.5 learning bootstrap status.
- Downstream impact: Future P4.3 implementation must collect deterministic Isaac rollout datasets, then stage learning through π_L/residual control, π_H trajectory/contact policy, and π_D scorer fine-tuning before any optional joint fine-tuning. Deterministic fallbacks and `FeasibilityChecker` hard safety remain required.
- Tests added or run: No tests added; this is a design-spec revision only.
- Commands run:
  - `rg -n "P4.3|learning bootstrap|Training loop|π_D|π_H|π_L|P4 full acceptance|minimum learning" for_codex/A-MSRR_codex_ready_spec_v0_4_ja.md`
  - `sed -n ... for_codex/A-MSRR_codex_ready_spec_v0_4_ja.md`
  - `sed -n ... for_codex/AMSRR_design_modification_by_codex.md`
  - `sed -n ... for_codex/WORKLOG.md`
  - `git status --short`
  - `rg -n "P4\\.3a|P4\\.3b|P4\\.3c|P4\\.3d|P4\\.3e|π_L / residual controller|π_H contact / trajectory|π_D outcome-conditioned|updates π_D|updates π_H|updates π_L|deterministic safety gate|FeasibilityChecker" for_codex/A-MSRR_codex_ready_spec_v0_4_ja.md`
  - `git diff --check`
  - `git diff --stat`
- Tests run: Documentation verification only. `rg` verification found the new P4.3 learning-target terms and training-loop update arrows in the source spec. `git diff --check` passed.
- Assumptions: P2.5 π_D scorer can be used as an initializer or auxiliary model, but deterministic `P2DesignPolicy` and `FeasibilityChecker` remain the production fallback and hard-safety source of truth.
- Blockers: None.
- Next steps: When P4.3 implementation starts, collect deterministic Isaac rollout datasets before staging π_L/residual, π_H, and π_D scorer learning.

### 2026-07-08
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4 plus P4 Isaac-backed completion clarification
- Work package / Agent label: P4 design revision / source-spec update
- Summary: Updated the source design spec per the user-provided P4 design revision instruction. P4 is now split into P4.0 simplified full-pipeline integration, P4-control/P4a low-level Isaac flight validation, P4.1 Isaac backend smoke, P4.2 Isaac deterministic full grasp/carry rollout, P4.3 Isaac learning bootstrap, and P4 full completion. The spec now states that P4.0 is necessary but not P4 complete, and that P4 full completion requires Isaac-backed rollout plus minimum learning run artifacts.
- Files changed:
  - `for_codex/A-MSRR_codex_ready_spec_v0_4_ja.md`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: Source spec only. Future implementation will need EpisodeArchive additions for `runtime_observations`, `actuator_target_records`, rollout artifacts, and learning artifacts; no Python implementation files were changed in this task.
- Upstream dependencies used: User-provided `/home/leus/Downloads/p4_design_revision_instruction.md`, v0.4 Sections 17, 20, 23, 24, 25, 26, and 27.
- Downstream impact: P4 implementation must not mark simplified backend acceptance as P4 complete. Future P4 work must implement controller bridge / actuator mapping, π_A docking/detach/separation handoff to controller targets, P4-control Isaac low-level flight validation, Isaac-backed rollout, and a minimum learning run before P4 full completion.
- Tests added or run: No tests added; this is a design-spec revision only.
- Commands run:
  - `wc -l /home/leus/Downloads/p4_design_revision_instruction.md`
  - `sed -n ... /home/leus/Downloads/p4_design_revision_instruction.md`
  - `rg -n "P4|full grasp|SimplifiedGraspCarryEnv|π_H|π_L|QP|Controller|Simulation|Training Curriculum|Acceptance|Agent J|Agent K|Agent L|Implementation order|EpisodeArchive" for_codex/A-MSRR_codex_ready_spec_v0_4_ja.md`
  - `sed -n ... for_codex/A-MSRR_codex_ready_spec_v0_4_ja.md`
  - `rg -n "P4.0|P4-control|low-level flight|Isaac|Controller bridge|actuator mapping|P4 full completion" for_codex/A-MSRR_codex_ready_spec_v0_4_ja.md`
  - `git diff --check`
  - `find amsrr -type d -name __pycache__ -prune -exec rm -rf {} +`
  - `git status --short`
  - `git diff --stat`
- Tests run: Documentation verification only. `rg` verification found the new P4 terms in the source spec. `git diff --check` passed.
- Assumptions: The P4 revision changes the source design contract but intentionally does not implement any P4 code yet.
- Blockers: None.
- Next steps: When implementation resumes, begin with P4.0 simplified full-pipeline integration, then implement controller bridge / actuator mapping and P4-control Isaac low-level flight validation before claiming P4 full completion.

### 2026-07-08
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4 plus P3 assembly integration supplements
- Work package / Agent label: P3 final verification and handoff
- Summary: Completed final verification after the P3 assembly runner/executor/retry/acceptance sequence. Full unit and acceptance suites passed, compile checks passed, and diff whitespace checks passed.
- Files changed:
  - `for_codex/WORKLOG.md`
- Schema/interface changes: None.
- Upstream dependencies used: Completed P3 order 1-5 commits, full unit suite, full acceptance suite, and existing AGENTS.md handoff rules.
- Downstream impact: P3 deterministic assembly integration is now mechanically checked. Future P4 work can start from the P3 acceptance gate, while remembering that this remains simplified assembly integration and does not run Isaac, π_H, π_L, QP/PID, or actuator commands.
- Tests added or run: No new tests added in this final handoff step.
- Commands run:
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/acceptance -q`
  - `python3 -m compileall amsrr -q`
  - `git diff --check`
  - `find amsrr tests -type d -name __pycache__ -prune -exec rm -rf {} +`
- Tests run: Full unit suite passed: 96 passed, 1 skipped. Full acceptance suite passed: 6 passed in 115.39s. `python3 -m compileall amsrr -q` passed. `git diff --check` passed.
- Assumptions: P3 acceptance is satisfied by deterministic simplified assembly integration per v0.4 Section 24.4. Physical docking, Isaac, π_H, π_L, QP/PID, actuator commands, and full grasp/carry task execution remain P4/later work.
- Blockers: None.
- Next steps: Proceed to P4 full grasp/carry integration after reviewing P3 acceptance outputs and deciding whether Isaac-backed assembly validation is needed before P4.

### 2026-07-08
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4 plus P3 Acceptance Gate Supplement
- Work package / Agent label: Agent L: P3 acceptance gate
- Summary: Added a P3 acceptance gate for Section 24.4. It runs the P3 assembly evaluation runner, checks assembly success rate, verifies construction-state/physical-graph consistency for successful assemblies, and exercises explicit retry and abort probes through the simplified executor.
- Files changed:
  - `amsrr/acceptance/p3_acceptance.py`
  - `amsrr/acceptance/__init__.py`
  - `tests/acceptance/test_p3_acceptance.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: None to persisted schemas. Added acceptance-side `P3AcceptanceCriteria`, `P3AcceptanceReport`, and `run_p3_acceptance`.
- Upstream dependencies used: v0.4 Section 24.4; Agent K P3 runner; Agent G assembly runner/retry/abort and simplified executor; P2 design distribution/policy for probe target graphs.
- Downstream impact: P3 can now be mechanically checked before moving to P4 full grasp/carry integration.
- Tests added or run:
  - Added `test_p3_acceptance_section_24_4`
- Commands run:
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/training/test_p3_assembly_runner.py tests/acceptance/test_p3_acceptance.py -q`
  - `python3 -m compileall amsrr -q`
- Tests run: P3 runner and P3 acceptance targeted tests passed: 3 passed. `python3 -m compileall amsrr -q` passed.
- Assumptions: Retry/abort path testing uses explicit deterministic failure probes because the normal simplified executor succeeds deterministically.
- Blockers: None.
- Next steps: Run P3 acceptance and related targeted tests, commit order 5, then perform final docs/worklog verification.

### 2026-07-08
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4 plus P3 Assembly Evaluation Runner Supplement
- Work package / Agent label: Agent K: P3 assembly evaluation runner
- Summary: Added a P3 assembly evaluation runner/config that samples grasp/carry tasks, reuses deterministic P2 design selection, executes the selected target morphology through `AssemblyRunner` and `SimplifiedAssemblyExecutor`, stores `AssemblyPlan` in `EpisodeArchive.assembly_plan`, and records assembly success/state/retry/abort metrics.
- Files changed:
  - `amsrr/training/p3_assembly_runner.py`
  - `amsrr/training/__init__.py`
  - `configs/training/p3_assembly_grasp_carry.yaml`
  - `tests/unit/training/test_p3_assembly_runner.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: None to persisted schemas. Added training-side `P3AssemblyRunnerConfig`, `P3AssemblyRunnerResult`, and `P3AssemblyEvaluationRunner`.
- Upstream dependencies used: P2 task distribution/config, `P2DesignPolicy`, `FeasibilityChecker` labels through selected candidate results, Agent G assembly runner/executor, and `EpisodeArchive`.
- Downstream impact: Agent L P3 acceptance can aggregate assembly success rate, retry/abort coverage, and construction-state consistency from runner archives/reports.
- Tests added or run:
  - Added `test_p3_assembly_runner_collects_successful_assembly_archives`
  - Added `test_p3_assembly_runner_config_loader`
- Commands run:
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/training/test_p2_design_runner.py tests/unit/training/test_p3_assembly_runner.py tests/unit/assembly/test_graph_edit_planner.py tests/unit/assembly/test_assembly_runner.py tests/unit/assembly/test_simplified_executor.py -q`
  - `python3 -m compileall amsrr -q`
- Tests run: P3 runner plus related P2/assembly targeted tests passed: 17 passed. `python3 -m compileall amsrr -q` passed.
- Assumptions: P3 assembly runner remains simplified and intentionally does not run contact candidates, π_H, π_L, QP/PID, actuator commands, or Isaac.
- Blockers: None.
- Next steps: Run targeted Agent K tests, commit order 4, then implement P3 acceptance gate.

### 2026-07-08
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4 plus P3 Retry/Abort State-Machine Supplement
- Work package / Agent label: Agent G: P3 retry/abort behavior
- Summary: Extended `AssemblyRunner` with deterministic retry/abort behavior. Failed planned steps now emit synthetic `retry` steps up to a configurable retry limit, then emit a synthetic `abort` step if the planned step still fails. `AssemblyRunReport` now records retry/abort counts, aborted status, and executed step types. The simplified executor can now fail matching steps once for transient failure tests.
- Files changed:
  - `amsrr/assembly/assembly_runner.py`
  - `amsrr/assembly/simplified_executor.py`
  - `tests/unit/assembly/test_assembly_runner.py`
  - `tests/unit/assembly/test_simplified_executor.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: None to persisted schemas. Extended assembly-local runner/executor dataclasses only.
- Upstream dependencies used: Existing Agent G runner/executor scaffolding and v0.4 valid AssemblyStep types `retry` and `abort`.
- Downstream impact: P3 runner/acceptance can now measure retry and abort path coverage directly from `AssemblyRunReport`.
- Tests added or run:
  - Added `test_assembly_runner_can_disable_retry_for_single_failure_stop`
  - Added `test_simplified_executor_fail_once_allows_runner_retry_success`
  - Updated failure-path tests to assert synthetic retry/abort records.
- Commands run:
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/assembly/test_graph_edit_planner.py tests/unit/assembly/test_assembly_runner.py tests/unit/assembly/test_simplified_executor.py -q`
  - `python3 -m compileall amsrr -q`
- Tests run: Agent G targeted assembly tests passed: 12 passed. `python3 -m compileall amsrr -q` passed.
- Assumptions: Retry/abort steps are synthetic runtime steps and are not inserted into the source `AssemblyPlan.steps`, preserving the original deterministic graph-edit plan.
- Blockers: None.
- Next steps: Run targeted Agent G tests, commit order 3, then implement P3 assembly evaluation runner.

### 2026-07-08
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4 plus P3 Simplified Assembly Executor Supplement
- Work package / Agent label: Agent G: P3 simplified assembly executor
- Summary: Added a deterministic `SimplifiedAssemblyExecutor` backend for the assembly executor interface. It succeeds assembly steps by default, can return updated construction state on `verify_attach`, records per-step smoke metrics, and supports explicit failure injection for later retry/abort probes.
- Files changed:
  - `amsrr/assembly/simplified_executor.py`
  - `amsrr/assembly/__init__.py`
  - `tests/unit/assembly/test_simplified_executor.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: None to persisted schemas. Added assembly-local `SimplifiedAssemblyExecutorConfig` and `SimplifiedAssemblyExecutor`.
- Upstream dependencies used: Existing Agent G `AssemblyRunner`, `AssemblyExecutorInterface`, `mark_edge_attached`, and v0.4 P3 simplified sim acceptance guidance.
- Downstream impact: Order 3 retry/abort behavior and Order 4 P3 runner can use the simplified executor to exercise success and failure paths without Isaac or controller dependencies.
- Tests added or run:
  - Added `test_simplified_executor_runs_full_assembly_and_returns_updated_state`
  - Added `test_simplified_executor_can_inject_step_type_failure`
  - Added `test_simplified_executor_success_without_target_graph_uses_runner_state_transition`
- Commands run:
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/assembly/test_graph_edit_planner.py tests/unit/assembly/test_assembly_runner.py tests/unit/assembly/test_simplified_executor.py -q`
  - `python3 -m compileall amsrr -q`
- Tests run: Agent G targeted assembly tests passed: 10 passed. `python3 -m compileall amsrr -q` passed.
- Assumptions: Simplified executor metrics are smoke values only and do not imply physical docking feasibility.
- Blockers: None.
- Next steps: Run targeted Agent G tests, commit order 2, then implement retry/abort behavior.

### 2026-07-08
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4 plus P3 Assembly Runner Core Supplement
- Work package / Agent label: Agent G: P3 assembly state execution core
- Summary: Added a deterministic `AssemblyRunner` core that runs `AssemblyPlan` steps through an `AssemblyExecutorInterface`, updates `ConstructionState` after successful `verify_attach` steps, records per-step results, and reports final physical-graph consistency metrics against the target `MorphologyGraph`.
- Files changed:
  - `amsrr/assembly/assembly_runner.py`
  - `amsrr/assembly/__init__.py`
  - `tests/unit/assembly/test_assembly_runner.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: None to persisted schemas. Added assembly-local `AssemblyRunnerConfig` and `AssemblyRunReport` dataclasses.
- Upstream dependencies used: v0.4 Sections 17 and 24.4; existing Agent G `GraphEditAssemblyPlanner`, `ConstructionState`, `AssemblyExecutorInterface`, and P2 grasp/carry morphology variants.
- Downstream impact: P3 simplified executor and acceptance work can now execute deterministic assembly plans and evaluate whether construction-state physical graph changes match the target graph.
- Tests added or run:
  - Added `test_assembly_runner_completes_plan_and_updates_construction_state`
  - Added `test_assembly_runner_stops_on_failed_step_without_completing_graph`
  - Added `test_assembly_runner_resumes_from_partial_construction_state`
- Commands run:
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/assembly/test_graph_edit_planner.py tests/unit/assembly/test_assembly_runner.py -q`
  - `python3 -m compileall amsrr -q`
- Tests run: Agent G targeted assembly tests passed: 7 passed. `python3 -m compileall amsrr -q` passed.
- Assumptions: Successful `verify_attach` is the deterministic point at which the core can mark a target dock edge attached if the executor does not provide a richer updated state.
- Blockers: None.
- Next steps: Run targeted Agent G tests, commit order 1, then implement the simplified assembly executor.

### 2026-07-08
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4 plus user-requested P2.5 learning bootstrap
- Work package / Agent label: P2.5: Supervised learning bootstrap for π_D scorer and feasibility head
- Summary: Added a P2.5 learning bootstrap that turns deterministic `P2DesignPolicy` candidate evaluations and `FeasibilityChecker` labels into a supervised dataset, trains a minimal learned π_D candidate scorer, trains a minimal learned feasibility head, saves checkpoints/metrics/loss curves, and updates the P2.5 report/acceptance gate. This is not full RL and does not replace deterministic design selection or hard safety checks.
- Files changed:
  - `amsrr/training/p2_candidate_trace_export.py`
  - `amsrr/training/p2_learning_dataset.py`
  - `amsrr/training/p2_learned_scorer.py`
  - `amsrr/training/p2_feasibility_head_training.py`
  - `amsrr/reporting/p2_5_inspection_report.py`
  - `amsrr/acceptance/p2_5_inspection.py`
  - `amsrr/acceptance/p2_5_learning_bootstrap.py`
  - `amsrr/acceptance/__init__.py`
  - `tests/unit/training/test_p2_learning_dataset.py`
  - `tests/unit/training/test_p2_learned_scorer.py`
  - `tests/unit/training/test_p2_feasibility_head_training.py`
  - `tests/unit/reporting/test_p2_5_inspection_report.py`
  - `tests/acceptance/test_p2_5_inspection.py`
  - `tests/acceptance/test_p2_5_learning_bootstrap.py`
  - `outputs/p2_5/datasets/p2_candidate_dataset.jsonl`
  - `outputs/p2_5/datasets/p2_candidate_dataset_summary.json`
  - `outputs/p2_5/datasets/train_ids.json`
  - `outputs/p2_5/datasets/val_ids.json`
  - `outputs/p2_5/training/pi_d_scorer/checkpoint.pt`
  - `outputs/p2_5/training/pi_d_scorer/metrics.json`
  - `outputs/p2_5/training/pi_d_scorer/loss_curve.csv`
  - `outputs/p2_5/training/feasibility_head/checkpoint.pt`
  - `outputs/p2_5/training/feasibility_head/metrics.json`
  - `outputs/p2_5/training/feasibility_head/loss_curve.csv`
  - `outputs/p2_5/report/p2_5_inspection_report.md`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: None to persisted schemas. Added training/acceptance helper dataclasses only.
- Upstream dependencies used: Existing P2 task distribution/config, `P2DesignPolicy.evaluate_candidates()`/`evaluate_design_outputs()`, deterministic `FeasibilityChecker`, P2.5 candidate trace export, and P2.5 report/acceptance scaffolding.
- Dataset output: `outputs/p2_5/datasets/p2_candidate_dataset.jsonl`
- Dataset counts: 320 candidate records from 64 task samples; train=255, val=65; accepted=256, rejected=64, selected=64.
- Dataset labels/features: all normal P2 candidates plus closed-loop invalid probes are stored with selected/accepted/feasible labels, teacher scores, design scores, violation labels/codes, feasibility margins, slot/capability coverage, thrust/payload/reachability margins, module count, and dock edge count.
- Training commands:
  - `python3 -m amsrr.training.p2_learning_dataset --config configs/training/p2_design_grasp_carry.yaml --output-dir outputs/p2_5/datasets --sample-count 64 --seed 0`
  - `python3 -m amsrr.training.p2_learned_scorer --dataset outputs/p2_5/datasets/p2_candidate_dataset.jsonl --train-ids outputs/p2_5/datasets/train_ids.json --val-ids outputs/p2_5/datasets/val_ids.json --output-dir outputs/p2_5/training/pi_d_scorer --epochs 40 --seed 0`
  - `python3 -m amsrr.training.p2_feasibility_head_training --dataset outputs/p2_5/datasets/p2_candidate_dataset.jsonl --train-ids outputs/p2_5/datasets/train_ids.json --val-ids outputs/p2_5/datasets/val_ids.json --output-dir outputs/p2_5/training/feasibility_head --epochs 40 --seed 1`
- π_D scorer checkpoint: `outputs/p2_5/training/pi_d_scorer/checkpoint.pt`
- π_D scorer metrics: train_loss=0.10842715948820114, val_loss=0.10839308053255081, selected_accuracy=1.0, num_train_samples=255, num_val_samples=65.
- Feasibility head checkpoint: `outputs/p2_5/training/feasibility_head/checkpoint.pt`
- Feasibility head metrics: train_loss=0.00012452361988835037, val_loss=0.00012500998855102807, binary_accuracy=1.0, precision=1.0, recall=1.0, num_train_samples=255, num_val_samples=65.
- Report update: `outputs/p2_5/report/p2_5_inspection_report.md` now records dataset paths/counts, scorer/head checkpoint paths, metrics, and explicitly states that learned models are NOT used in production path and deterministic `P2DesignPolicy` / `FeasibilityChecker` remain source of truth.
- Tests added or run:
  - Added `test_p2_learning_dataset_builds_records_and_split`
  - Added `test_p2_learned_scorer_training_writes_checkpoint_and_metrics`
  - Added `test_p2_feasibility_head_training_writes_checkpoint_and_metrics`
  - Added `test_p2_5_learning_bootstrap_acceptance_gate`
- Commands run:
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/training/test_p2_learning_dataset.py tests/unit/training/test_p2_learned_scorer.py tests/unit/training/test_p2_feasibility_head_training.py tests/unit/reporting/test_p2_5_inspection_report.py tests/acceptance/test_p2_5_learning_bootstrap.py -q`
  - `python3 -m amsrr.training.p2_learning_dataset --config configs/training/p2_design_grasp_carry.yaml --output-dir outputs/p2_5/datasets --sample-count 64 --seed 0`
  - `python3 -m amsrr.training.p2_learned_scorer --dataset outputs/p2_5/datasets/p2_candidate_dataset.jsonl --train-ids outputs/p2_5/datasets/train_ids.json --val-ids outputs/p2_5/datasets/val_ids.json --output-dir outputs/p2_5/training/pi_d_scorer --epochs 40 --seed 0`
  - `python3 -m amsrr.training.p2_feasibility_head_training --dataset outputs/p2_5/datasets/p2_candidate_dataset.jsonl --train-ids outputs/p2_5/datasets/train_ids.json --val-ids outputs/p2_5/datasets/val_ids.json --output-dir outputs/p2_5/training/feasibility_head --epochs 40 --seed 1`
  - `python3 -m amsrr.reporting.p2_5_inspection_report --trace-dir outputs/p2_5/candidate_traces --visualization-dir outputs/p2_5/visualization --output-dir outputs/p2_5/report --config configs/training/p2_design_grasp_carry.yaml --dataset-dir outputs/p2_5/datasets --training-dir outputs/p2_5/training`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/acceptance -q`
  - `python3 -m compileall amsrr -q`
  - `git diff --check`
- Tests run: Targeted learning/report/acceptance tests passed: 5 passed. Full unit suite passed: 86 passed, 1 skipped. Full acceptance suite passed: 5 passed in 89.94s. `python3 -m compileall amsrr -q` passed. `git diff --check` passed.
- Production-path status: The learned π_D scorer and learned feasibility head are not used in production path. Deterministic `P2DesignPolicy` remains the design-selection source of truth, and deterministic `FeasibilityChecker` remains the hard safety/source-of-truth checker.
- Explicitly not executed: full RL, Isaac, π_H, π_L, QP/PID, actuator command execution.
- Assumptions: Minimal MLPs are sufficient for bootstrap acceptance because this task proves dataset -> training -> checkpoint -> metrics wiring rather than model quality. The high accuracy reflects deterministic teacher labels and simple diagnostic features, not a claim of general learned policy performance.
- Blockers: None.
- Next steps: Commit P2.5 learning bootstrap changes if accepted, then proceed toward P3 only after human review of P2.5 report/visualizations and confirmation that learned artifacts remain auxiliary.

### 2026-07-08
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4 clarification
- Work package / Agent label: Design clarification: π_D structural scope and joint-angle non-design rule
- Summary: Clarified the source design spec to state that π_D designs graph-level A-MSRR structure only and must not treat movable joint angles or runtime module relative poses as design freedoms. Clarified `ModuleNode.pose_in_design_frame` and `DockEdge.relative_pose_src_to_dst` as nominal/canonical metadata, not optimized joint configurations, and clarified that design-level feasibility must not score a single nominal joint configuration.
- Files changed:
  - `for_codex/A-MSRR_codex_ready_spec_v0_4_ja.md`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: None.
- Upstream dependencies used: User clarification, current P2/P2.5 implementation, Section 14 MorphologyGraph, Section 15 π_D, and Section 16 FeasibilityChecker.
- Downstream impact: Future π_D, feasibility, visualization, and report work must treat pose fields as canonical/nominal metadata only. Joint-angle optimization and runtime relative-pose trajectories belong to π_H, π_L, QP/PID, controller/runtime state, or simulator logic.
- Tests added or run: No code tests added; this is a source-spec/documentation clarification.
- Commands run:
  - `git status --short`
  - `rg -n "π_D|MorphologyGraph|pose_in_design_frame|DockEdge|Design-level|Feasibility|feasibility|relative_pose" for_codex/A-MSRR_codex_ready_spec_v0_4_ja.md`
  - `rg -n "pose_in_design_frame|relative_pose_src_to_dst|joint angle|joint_angle|joint target|vectoring|rotor thrust|torque" amsrr tests for_codex -g '*.py' -g '*.md'`
  - `sed -n ...` inspections for Sections 14, 15, 16, and worked example text
- Tests run: Not run because no source code changed. Existing code inspection indicates current P2 design-level checker does not score single joint angles or a single nominal module-relative pose.
- Assumptions: Current P2 scaffold `pose_in_design_frame` values are nominal layout/reference values for visualization/debugging and not learned or optimized continuous joint outputs.
- Blockers: None.
- Next steps: If desired, regenerate P2.5 report text to include this clarification, but functional code changes are not required for correctness.

### 2026-07-08
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4 plus user-requested P2.5 inspection phase
- Work package / Agent label: P2.5: Post-P2 inspection, visualization, and candidate trace export
- Summary: Added P2.5 as an additional pre-P3 inspection/debugging phase that visualizes all P2 grasp/carry morphology variants, exports every evaluated candidate including accepted/rejected/selected labels, generates a human-readable inspection report, and provides a P2.5 acceptance gate.
- Files changed:
  - `amsrr/training/p2_inspection_context.py`
  - `amsrr/training/p2_candidate_trace_export.py`
  - `amsrr/visualization/__init__.py`
  - `amsrr/visualization/p2_morphology.py`
  - `amsrr/reporting/__init__.py`
  - `amsrr/reporting/p2_5_inspection_report.py`
  - `amsrr/acceptance/__init__.py`
  - `amsrr/acceptance/p2_5_inspection.py`
  - `tests/unit/visualization/test_p2_morphology_visualization.py`
  - `tests/unit/training/test_p2_candidate_trace_export.py`
  - `tests/unit/reporting/test_p2_5_inspection_report.py`
  - `tests/acceptance/test_p2_5_inspection.py`
  - `outputs/p2_5/visualization/*.svg`
  - `outputs/p2_5/candidate_traces/p2_candidate_trace.jsonl`
  - `outputs/p2_5/candidate_traces/p2_candidate_summary.csv`
  - `outputs/p2_5/report/p2_5_inspection_report.md`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: None to persisted schemas. Added inspection/report/acceptance-side helper dataclasses only.
- Upstream dependencies used: Existing P2 completion, Agent E grasp/carry variants and `P2DesignPolicy`, Agent F feasibility labels/margins, Agent K P2 design config/distribution, `DesignOutput`, `FeasibilityResult`, and current P2 runner context.
- Downstream impact: P3 should not start until a human has inspected `outputs/p2_5/report/p2_5_inspection_report.md` and the SVG visualizations. P2 completion semantics remain unchanged; P2.5 is an additional inspection gate.
- Generated visualization files:
  - `outputs/p2_5/visualization/chain_grasp_graph.svg`
  - `outputs/p2_5/visualization/chain_grasp_layout.svg`
  - `outputs/p2_5/visualization/symmetric_two_anchor_grasp_graph.svg`
  - `outputs/p2_5/visualization/symmetric_two_anchor_grasp_layout.svg`
  - `outputs/p2_5/visualization/tri_anchor_support_grasp_graph.svg`
  - `outputs/p2_5/visualization/tri_anchor_support_grasp_layout.svg`
  - `outputs/p2_5/visualization/central_base_plus_two_grasp_arms_graph.svg`
  - `outputs/p2_5/visualization/central_base_plus_two_grasp_arms_layout.svg`
- Candidate trace outputs:
  - `outputs/p2_5/candidate_traces/p2_candidate_trace.jsonl`
  - `outputs/p2_5/candidate_traces/p2_candidate_summary.csv`
- Inspection report: `outputs/p2_5/report/p2_5_inspection_report.md`
- Candidate counts in generated trace: 5 records total; 4 accepted; 1 rejected; 1 selected.
- Representative violation code: `F_CLOSED_LOOP_REJECT_V1` from the explicit `tri_anchor_support_grasp_closed_loop_probe` rejected candidate.
- Tests added or run:
  - Added `test_p2_morphology_visualization_outputs_graph_and_layout_svgs`
  - Added `test_p2_candidate_trace_export_writes_all_candidates_and_probe`
  - Added `test_p2_5_inspection_report_contains_summary_and_scope_notes`
  - Added `test_p2_5_inspection_acceptance_gate`
- Commands run:
  - Read attached request text from `/home/leus/.codex/attachments/.../pasted-text.txt`
  - `git status --short`, `git diff --stat`, `find ...`, `sed -n ...`, `rg ...`, and `git log ...` inspections
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/visualization/test_p2_morphology_visualization.py -q`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/training/test_p2_candidate_trace_export.py -q`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/reporting/test_p2_5_inspection_report.py -q`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/acceptance/test_p2_5_inspection.py -q`
  - `python3 -m amsrr.visualization.p2_morphology --config configs/training/p2_design_grasp_carry.yaml --output-dir outputs/p2_5/visualization`
  - `python3 -m amsrr.training.p2_candidate_trace_export --config configs/training/p2_design_grasp_carry.yaml --output-dir outputs/p2_5/candidate_traces`
  - `python3 -m amsrr.reporting.p2_5_inspection_report --trace-dir outputs/p2_5/candidate_traces --visualization-dir outputs/p2_5/visualization --output-dir outputs/p2_5/report --config configs/training/p2_design_grasp_carry.yaml`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/acceptance -q`
  - `python3 -m compileall amsrr -q`
  - `git diff --check`
  - `find amsrr tests -type d -name __pycache__ -prune -exec rm -rf {} +`
- Tests run: P2.5 targeted tests passed individually: visualization 1 passed, trace export 1 passed, report 1 passed, P2.5 acceptance 1 passed. Full unit suite passed: 83 passed, 1 skipped. Full acceptance suite passed: 4 passed in 88.72s. `python3 -m compileall amsrr -q` passed. `git diff --check` passed.
- P2.5 explicitly not executed: Isaac, π_H, π_L, QP/PID, actuator commands, learned training.
- Assumptions: The normal P2 variant set currently yields accepted candidates for the default sample, so P2.5 appends an explicit closed-loop invalid probe through `P2DesignPolicy.evaluate_design_outputs()` to externalize a rejected candidate and its labels without changing P2 completion.
- Blockers: None.
- Next steps: Commit final P2.5 report/acceptance changes. Human review of the report and SVGs is recommended before P3.

### 2026-07-08
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4
- Work package / Agent label: Agent L: P2 completion gate
- Summary: Added a P2 milestone completion wrapper that runs the Section 24.3 P2 acceptance gate and emits explicit boolean completion checks for valid design rate, required slot coverage, closed-loop invalid rejection, and feasibility label storage.
- Files changed:
  - `amsrr/acceptance/__init__.py`
  - `amsrr/acceptance/p2_completion.py`
  - `tests/acceptance/test_p2_completion.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: None to persisted schemas. Added acceptance-side `P2CompletionCriteria`, `P2CompletionReport`, and `run_p2_completion`.
- Upstream dependencies used: v0.4 Section 24.3; existing `run_p2_acceptance`, `P2AcceptanceReport`, P2 design runner archives, and Agent F feasibility labels/margins.
- Downstream impact: Downstream P3/P4 work can treat `run_p2_completion(...).passed` as the local P2 milestone gate before assembly/end-to-end integration. This remains design-level and does not run π_H, π_L, QP/PID, actuator commands, Isaac, or learned training.
- Tests added or run:
  - Added `test_p2_completion_milestone_section_24_3`
- Commands run:
  - `git status --short`, `sed -n ...`, `rg -n ...`, `ls -la ...`, and `git log -5 --oneline` inspections for spec/worklog/acceptance state
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/acceptance/test_p2_completion.py -q`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/acceptance -q`
  - `python3 -m compileall amsrr -q`
  - `git diff --check`
  - `find amsrr tests -type d -name __pycache__ -prune -exec rm -rf {} +`
- Tests run: Targeted P2 completion test passed: 1 passed in 23.62s. Full unit suite passed: 80 passed, 1 skipped. Full acceptance suite passed: 3 passed in 88.20s. `python3 -m compileall amsrr -q` passed. `git diff --check` passed.
- Assumptions: P2 completion is defined as successful Section 24.3 design-level acceptance. It intentionally does not imply assembly execution, π_H/π_L/controller execution, Isaac Sim execution, or full grasp/carry success; those begin in P3/P4.
- Blockers: None.
- Next steps: Commit P2 completion changes if accepted, then advance to P3 assembly integration.

### 2026-07-08
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4
- Work package / Agent label: Agent L: P2 acceptance gate
- Summary: Added a mechanical P2 acceptance gate for Section 24.3 that runs the P2 design evaluation runner, checks `valid_design_rate >= 70%`, verifies accepted-design required slot coverage, probes closed-loop invalid rejection, and validates feasibility label storage in `EpisodeArchive.feasibility_result`.
- Files changed:
  - `amsrr/acceptance/__init__.py`
  - `amsrr/acceptance/p2_acceptance.py`
  - `tests/acceptance/test_p2_acceptance.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: None. Added acceptance-side `P2AcceptanceCriteria`, `P2AcceptanceReport`, and `run_p2_acceptance`.
- Upstream dependencies used: v0.4 Section 24.3; Agent K P2 design runner/archive output; Agent E P2 `P2DesignPolicy` and grasp/carry variants; Agent F `FeasibilityChecker` labels/margins.
- Downstream impact: P2 now has a reproducible pass/fail gate before moving to later assembly/end-to-end phases. The gate remains design-level only and does not run π_H, π_L, QP/PID/controller commands, Isaac, or learned training.
- Tests added or run:
  - Added `test_p2_acceptance_section_24_3`
- Commands run:
  - `sed -n ...`, `rg -n ...`, `git status --short`, `git diff --stat`, and `git log -3 --oneline` inspections for acceptance, feasibility labels, Section 24.3, and current commit format
  - `git add ...`
  - `git commit -m "[P2][Agent K] Add design evaluation runner"`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/acceptance/test_p2_acceptance.py -q`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/acceptance -q`
  - `python3 -m compileall amsrr -q`
  - `git diff --check`
  - `find amsrr tests -type d -name __pycache__ -prune -exec rm -rf {} +`
- Tests run: Targeted P2 acceptance test passed: 1 passed in 23.45s. Full unit suite passed: 80 passed, 1 skipped. Full acceptance suite passed: 2 passed in 64.82s. `python3 -m compileall amsrr -q` passed. `git diff --check` passed.
- Assumptions: `required_slot_coverage >= 90% for accepted designs` is enforced as a minimum over accepted archived designs, which is stricter than an average-only interpretation. The normal P2 distribution produces tree morphologies, so closed-loop invalid rejection is tested through an explicit synthetic closed-loop design probe.
- Blockers: None.
- Next steps: Commit Agent L P2 acceptance changes if accepted, then continue to the next P2/P3 work package.

### 2026-07-08
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4
- Work package / Agent label: Agent K: P2 design evaluation runner
- Summary: Added a P2 design-evaluation runner that samples diverse grasp/carry TaskSpecs, runs TaskSpec -> Geometry/IRG -> InteractionEnvelope -> P2 π_D candidate evaluation -> FeasibilityChecker, and stores selected `DesignOutput` plus selected `FeasibilityResult` labels/margins in `EpisodeArchive` JSONL records.
- Files changed:
  - `amsrr/training/__init__.py`
  - `amsrr/training/p2_design_distribution.py`
  - `amsrr/training/p2_design_runner.py`
  - `configs/training/p2_design_grasp_carry.yaml`
  - `tests/unit/training/test_p2_design_runner.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: None. Existing `EpisodeArchive.feasibility_result`, `FeasibilityResult.proxy_scores`, `FeasibilityResult.margins`, and `DesignOutput.design_scores` are used unchanged.
- Upstream dependencies used: v0.4 Sections 23.4, 24.3, 25.1, 26.10; Agent E P2 variant builder and `P2DesignPolicy`; Agent F P2 feasibility labels/margins; existing `IRGBuilder`, `InteractionEnvelopeExtractor`, `PhysicalModel`, and `EpisodeArchive` logging.
- Downstream impact: P2 acceptance and dataset generation can now read archived design-level labels directly from `EpisodeArchive.feasibility_result`. The runner remains design-level only and does not run π_H, π_L, controller allocation, actuator commands, Isaac, or learned training.
- Tests added or run:
  - Added `test_p2_design_distribution_randomizes_and_marks_metadata`
  - Added `test_p2_design_runner_collects_feasibility_archives`
  - Added `test_p2_design_runner_config_loader`
- Commands run:
  - `sed -n ...`, `rg --files ...`, `rg -n ...`, `git status --short`, and `git diff --stat` inspections for training runners, policy/checker interfaces, schema/logging utilities, config files, and worklogs
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/training/test_p2_design_runner.py -q`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q`
  - `python3 -m compileall amsrr -q`
  - `git diff --check`
  - `find amsrr tests -type d -name __pycache__ -prune -exec rm -rf {} +`
- Tests run: Targeted P2 design runner tests passed: 3 passed. Full unit suite passed: 80 passed, 1 skipped. `python3 -m compileall amsrr -q` passed. `git diff --check` passed.
- Assumptions: P2 object diversity reuses and slightly widens the P1 box grasp/carry randomization fields for this slice. The runner archives the selected candidate's feasibility labels; full per-candidate dataset rows can be added later if P2 training needs rejected-candidate supervision beyond the current selected-design archive.
- Blockers: None.
- Next steps: Add an Agent L P2 acceptance gate over this runner, or extend archive output to store per-candidate feasibility traces if required by the training dataset format.

### 2026-07-08
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4
- Work package / Agent label: Agent E: P2 π_D candidate selection scaffold
- Summary: Added deterministic P2 design-policy scaffold that enumerates multiple grasp/carry candidate morphologies, evaluates each with `FeasibilityChecker`, separates accepted/rejected candidates, computes deterministic soft scores, and returns the best accepted design.
- Files changed:
  - `amsrr/policies/__init__.py`
  - `amsrr/policies/design_policy_p2.py`
  - `tests/unit/policies/test_p2_design_policy.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: None. Existing `DesignOutput.design_scores` stores P2 selection metadata as float keys with `p2_design_policy_*` prefixes.
- Upstream dependencies used: v0.4 Sections 15, 16, 24.3, 26.5, 27.1; Agent E grasp/carry variant builder; Agent F P2 FeasibilityChecker labels/margins; existing `DesignPolicyContext` and `DesignOutput` schemas.
- Downstream impact: P2 runner/acceptance can now call `P2DesignPolicy.evaluate_candidates()` to obtain all candidates plus accepted/rejected splits, or `design()` to get the deterministic selected design. Later learned π_D heads can replace scoring while preserving the candidate/evaluation boundary.
- Tests added or run:
  - Added `test_p2_design_policy_enumerates_variants_and_selects_best_accepted`
  - Added `test_p2_design_policy_splits_rejected_candidates_with_feasibility_checker`
  - Added `test_p2_design_policy_falls_back_to_best_rejected_when_none_accepted`
- Commands run:
  - `sed -n ...` inspections for design policy, teacher, candidate generator, package exports, and existing tests
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/policies/test_p2_design_policy.py tests/unit/policies/test_design_teacher.py -q`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q`
  - `python3 -m compileall amsrr -q`
  - `git diff --check`
  - `find amsrr tests -type d -name __pycache__ -prune -exec rm -rf {} +`
- Tests run: Targeted P2 design policy/design teacher tests passed: 6 passed. Full unit suite passed: 77 passed, 1 skipped. `python3 -m compileall amsrr -q` passed. `git diff --check` passed.
- Assumptions: This is deterministic π_D scaffolding, not a learned policy head. The soft score is a hand-coded P2 baseline combining feasibility margins with small support/complexity/variant priors; it is documented as replaceable by learned scoring later.
- Blockers: None.
- Next steps: Continue with Agent K/L P2 design runner and acceptance gate, or add dataset logging around the new candidate selection results.

### 2026-07-08
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4
- Work package / Agent label: Agent F: P2 design-level FeasibilityChecker strengthening
- Summary: Strengthened design-level `FeasibilityChecker` for P2 acceptance by stabilizing hard-check labels and numeric margins for slot coverage, anchor capability, closed-loop rejection, port conflicts, thrust/payload margins, and coarse reachability.
- Files changed:
  - `amsrr/feasibility/checker.py`
  - `tests/unit/feasibility/test_feasibility_checker.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: None. Existing `FeasibilityResult` schema is unchanged; P2 labels are stored as float entries in `proxy_scores` with `L_...` keys, and acceptance margins are stored in the existing `margins` map.
- Upstream dependencies used: v0.4 Sections 16.2-16.8, 24.3, 26.6, 27.1; existing Agent E P2 grasp/carry morphology variants, IRG ContactSlot and CapabilityRequirement edges, PhysicalModel thrust data, and MorphologyGraph/DesignOutput schemas.
- Downstream impact: P2 runners/acceptance can aggregate `L_FEASIBLE`, `L_<hard_check_code>`, required-slot coverage ratios, closed-loop rejection, port conflict counts, thrust margin, and payload margin directly from archived `FeasibilityResult` records.
- Tests added or run:
  - Added `test_p2_feasibility_checker_records_acceptance_margins_for_variant`
  - Added `test_p2_feasibility_checker_uses_capability_requirement_force_label`
  - Added `test_p2_feasibility_checker_records_port_conflict_margins`
  - Added `test_p2_feasibility_checker_records_reachability_margins`
  - Updated existing missing slot coverage and closed-loop tests to assert labels/margins
- Commands run:
  - `sed -n ...` and `rg -n ...` inspections for spec Sections 16/24/26, checker, IRG templates, and tests
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/feasibility/test_feasibility_checker.py -q`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q`
  - `python3 -m compileall amsrr -q`
  - `git diff --check`
  - `find amsrr tests -type d -name __pycache__ -prune -exec rm -rf {} +`
- Tests run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/feasibility/test_feasibility_checker.py -q` passed: 7 passed. Full unit suite passed: 74 passed, 1 skipped. `python3 -m compileall amsrr -q` passed. `git diff --check` passed.
- Assumptions: `FeasibilityResult` has no dedicated label field in v0.4, so deterministic P2 labels are represented as `proxy_scores["L_..."]` floats. These labels do not replace hard violations and are intended for acceptance/dataset aggregation.
- Blockers: None.
- Next steps: Continue with Agent E P2 candidate/evaluation policy scaffolding or Agent K/L P2 design runner and acceptance gate.

### 2026-07-08
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4
- Work package / Agent label: Agent E: P2 grasp/carry morphology variant builder
- Summary: Implemented real deterministic P2 grasp/carry morphology variants for `chain_grasp`, `symmetric_two_anchor_grasp`, `tri_anchor_support_grasp`, and `central_base_plus_two_grasp_arms`, and routed object grasp/carry `DeterministicDesignTeacher` output through the new variant builder.
- Files changed:
  - `amsrr/morphology/__init__.py`
  - `amsrr/morphology/grasp_carry_designs.py`
  - `amsrr/policies/design_teacher.py`
  - `tests/unit/morphology/test_grasp_carry_variants.py`
  - `tests/unit/policies/test_design_teacher.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: None. Existing `MorphologyGraph`, `DesignOutput`, `DesignAction`, `RobotAnchor`, and `ControlGroup` schemas were used unchanged.
- Upstream dependencies used: v0.4 Sections 14, 15.3, 15.4, 16, 24.3, 26.5, 27.1; existing IRG ContactSlot semantics, PhysicalModel dock ports/capability token, FeasibilityChecker, and design teacher/candidate trace boundaries.
- Downstream impact: P2 design evaluation can now sample/evaluate distinct teacher morphology demonstrations instead of four labels over one minimal seed graph. ContactCandidateSampler and FeasibilityChecker continue to consume the same schema objects.
- Tests added or run:
  - Added `test_grasp_carry_variants_build_distinct_feasible_morphologies`
  - Added `test_grasp_carry_variant_topology_shapes`
  - Added `test_grasp_carry_variants_cover_required_slot_min_count`
  - Updated `test_design_teacher_selects_p1_grasp_support_variant`
- Commands run:
  - `sed -n ...` inspections for morphology builder, design teacher, and existing tests
  - `python3 -c "from amsrr.robot_model.physical_model_builder import build_physical_model_from_config; ..."` to inspect Holon dock ports
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/morphology/test_grasp_carry_variants.py tests/unit/policies/test_design_teacher.py -q`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q`
  - `python3 -m compileall amsrr -q`
  - `git diff --check`
  - `find amsrr tests -type d -name __pycache__ -prune -exec rm -rf {} +`
- Tests run: Targeted morphology/design teacher tests passed: 6 passed. Full unit suite passed: 70 passed, 1 skipped. `python3 -m compileall amsrr -q` passed. `git diff --check` passed.
- Assumptions: Exact variant poses/topologies are not specified by v0.4, so this implementation defines deterministic scaffold layouts for P2 teacher/evaluation use. These variants are not optimized morphology search results and are not learned π_D outputs yet.
- Blockers: None.
- Next steps: Continue with Agent F P2 FeasibilityChecker strengthening or Agent E P2 candidate/evaluation policy scaffolding, then add the P2 acceptance runner/gate.

### 2026-07-08
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4
- Work package / Agent label: Agent L: P1 tests and acceptance
- Summary: Added an explicit P1 acceptance gate for v0.4 Section 24.2 using the configured simplified grasp/carry runner, EpisodeArchive JSONL output, and randomized contact-candidate smoke checks.
- Files changed:
  - `amsrr/acceptance/__init__.py`
  - `amsrr/acceptance/p1_acceptance.py`
  - `tests/acceptance/test_p1_acceptance.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: No persisted schema changes. Added acceptance-side `P1AcceptanceCriteria`, `P1AcceptanceReport`, and `run_p1_acceptance`.
- Upstream dependencies used: v0.4 Sections 24.2, 25.1, 26.12, 27.3; existing P1 task distribution config, `P1SimplifiedRunner`, `EpisodeArchive`, fixed/simple design policy, ContactCandidateSampler, pi_H baseline, pi_L baseline, and QPID controller.
- Downstream impact: P1 has a reproducible pass/fail acceptance harness before Isaac Lab integration. Later simulator backends can add equivalent acceptance coverage without changing the Section 24.2 criteria.
- Tests added or run:
  - Added `test_p1_acceptance_section_24_2`
- Commands run:
  - `sed -n ...` and `rg -n ...` inspections for spec Sections 24/26/27, acceptance ownership, runner/env/logging modules, and existing tests
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/acceptance/test_p1_acceptance.py -q`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q`
  - `python3 -m compileall amsrr -q`
  - `git diff --check`
  - `find amsrr tests -type d -name __pycache__ -prune -exec rm -rf {} +`
- Tests run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/acceptance/test_p1_acceptance.py -q` passed: 1 passed in 41.26s. `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q` passed: 67 passed, 1 skipped. `python3 -m compileall amsrr -q` passed. `git diff --check` passed.
- Assumptions: Section 24.2 can be evaluated on the interface-backed simplified env for P1. Isaac Lab remains a later simulator-backend validation step, not a prerequisite for this acceptance gate.
- Blockers: None.
- Next steps: Commit Agent L acceptance changes if accepted, then move to the next post-P1 work package.

### 2026-07-08
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4
- Work package / Agent label: Agent K: P1 task distribution, runner, metrics, and EpisodeArchive logging
- Summary: Implemented P1 order 8 task randomization config, grasp/carry task distribution, simplified env runner, EpisodeArchive schema/logging, batch metrics, and archive JSONL roundtrip tests.
- Files changed:
  - `amsrr/logging/__init__.py`
  - `amsrr/logging/episode_archive.py`
  - `amsrr/training/__init__.py`
  - `amsrr/training/p1_task_distribution.py`
  - `amsrr/training/p1_runner.py`
  - `configs/training/p1_grasp_carry_distribution.yaml`
  - `tests/unit/training/test_p1_runner.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: No persisted schema changes. Added logging/training-side `EpisodeArchive`, `P1TaskDistributionConfig`, `P1TaskSample`, `P1GraspCarryTaskDistribution`, `P1RunnerConfig`, `P1RunnerResult`, and `P1SimplifiedRunner`.
- Upstream dependencies used: v0.4 Sections 23.4, 24.2, 25.1, 25.3, 26.10; existing TaskSpec, IRG, InteractionEnvelope, DesignOutput, PolicyCommand, ControllerCommand, simplified env, and config/hash utilities.
- Downstream impact: P1 simplified runs can now be sampled from a configured object distribution, summarized by metrics, and serialized as EpisodeArchive JSONL records for later dataset/training work.
- Tests added or run:
  - Added `test_p1_distribution_randomizes_configured_fields`
  - Added `test_p1_runner_collects_metrics_and_archives`
  - Added `test_p1_runner_config_loader`
- Commands run:
  - `sed -n ...` and `rg -n ...` inspections for spec Sections 23/24/25/26, TaskSpec schemas, simplified env, config/hash utilities, and existing tests
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/training/test_p1_runner.py -q`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q`
  - `python3 -m compileall amsrr -q`
  - `find amsrr tests -type d -name __pycache__ -prune -exec rm -rf {} +`
- Tests run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/training/test_p1_runner.py -q` passed: 3 passed. `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q` passed: 67 passed, 1 skipped. `python3 -m compileall amsrr -q` passed.
- Assumptions: P1 randomization currently covers box primitive size, object mass, object friction, initial object pose, and target pose. Object shape stays box for this slice; wind, sensor noise, thrust scale error, and contact break threshold randomization are deferred.
- Blockers: None.
- Next steps: Continue with broader dataset/logging integration or Isaac Lab backend binding after this P1 runner is accepted.

### 2026-07-08
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4
- Work package / Agent label: Agent JP1: simplified grasp-carry simulation env
- Summary: Implemented P1 order 7 interface-backed simplified grasp/carry environment that runs the existing TaskSpec -> IRG -> Envelope -> fixed/simple morphology -> ContactCandidateSampler -> pi_H -> pi_L -> QPID controller loop without Isaac dependencies, plus 1000-episode crash-free unit coverage.
- Files changed:
  - `amsrr/simulation/__init__.py`
  - `amsrr/simulation/base.py`
  - `amsrr/simulation/simplified_grasp_carry_env.py`
  - `tests/unit/simulation/test_simplified_grasp_carry_env.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: No persisted schema changes. Added simulation-side `SimulationEnvBase`, `SimplifiedGraspCarryEnvConfig`, `SimplifiedGraspCarryBuildArtifacts`, `SimplifiedEpisodeResult`, `SimplifiedBatchRunResult`, `SimplifiedGraspCarryEnv`, and `run_crash_free_episodes`.
- Upstream dependencies used: v0.4 Sections 23, 24.2, 25.1, 26.10; Agent D IRGBuilder/EnvelopeExtractor; Agent E fixed/simple design policy; Agent H ContactCandidateSampler and pi_H baseline; Agent I pi_L and QPID controller interfaces.
- Downstream impact: P1 can validate the schema/runtime/controller loop before Isaac Lab integration. Later Agent J Isaac environments can implement the same `SimulationEnvBase` boundary while reusing policy/controller interfaces.
- Tests added or run:
  - Added `test_simplified_grasp_carry_env_matches_base_protocol`
  - Added `test_simplified_grasp_carry_env_runs_policy_controller_episode`
  - Added `test_simplified_grasp_carry_1000_episodes_crash_free`
- Commands run:
  - `sed -n ...` and `rg -n ...` inspections for spec Sections 23/24/25/26, existing pi_H/pi_L/controller tests, and worklog/design notes
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/simulation/test_simplified_grasp_carry_env.py -q`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q`
  - `python3 -m compileall amsrr -q`
  - `find amsrr tests -type d -name __pycache__ -prune -exec rm -rf {} +`
- Tests run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/simulation/test_simplified_grasp_carry_env.py -q` passed: 3 passed, including 1000 simplified episodes with 0 crashes. `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q` passed: 64 passed, 1 skipped. `python3 -m compileall amsrr -q` passed.
- Assumptions: The P1 simplified env uses kinematic/fixed-joint contact after attach, high-level object target tracking, deterministic small initial XY jitter, and controller status checks. It is not an Isaac Lab environment and does not model high-fidelity contact, friction, aerodynamic, or collision dynamics.
- Blockers: None.
- Next steps: Continue with Isaac Lab environment integration or dataset/logging once P1 simplified env behavior is accepted.

### 2026-07-08
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4
- Work package / Agent label: Agent I: pi_L + QP/PID interfaces
- Summary: Implemented P1 order 6 Agent I interfaces: deterministic pi_L baseline, controller context/base protocol, QP allocator interface, dependency-free bounded vertical rotor allocator, QPID controller scaffold, package exports, and policy/controller unit tests.
- Files changed:
  - `amsrr/controllers/__init__.py`
  - `amsrr/controllers/controller_base.py`
  - `amsrr/controllers/qp_allocator_interface.py`
  - `amsrr/controllers/qpid_controller.py`
  - `amsrr/policies/__init__.py`
  - `amsrr/policies/low_level_policy_base.py`
  - `tests/unit/controllers/test_qpid_controller.py`
  - `tests/unit/policies/test_low_level_baseline.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: No persisted schema changes. Added policy-side `LowLevelPolicyContext`, `LowLevelPolicyBase`, `BaselineLowLevelPolicyConfig`, `BaselineLowLevelPolicy`, and `select_active_knot`; added controller-side `ControllerContext`, `ControllerBase`, `QPAllocationProblem`, `QPAllocationResult`, `QPAllocatorInterface`, `BoundedVerticalRotorAllocator`, `RotorAllocationSpec`, `QPIDControllerConfig`, and `QPIDController`.
- Upstream dependencies used: v0.4 Sections 20, 26.9, 27.1, 28.11; existing `PolicyCommand`, `RuntimeObservation`, `PhysicalModel`, `ContactWrenchTrajectory`, and `PolicyCommandBiasBuilder`.
- Downstream impact: P1 simplified grasp-carry simulation can consume deterministic `PolicyCommand` and `ControllerCommand` outputs through stable interfaces. Later learned pi_L heads and exact QP backends can replace the baseline/allocator without changing the context boundaries.
- Tests added or run:
  - Added `test_baseline_low_level_policy_outputs_policy_command`
  - Added `test_baseline_low_level_policy_selects_knot_from_runtime_time`
  - Added `test_baseline_low_level_policy_suppresses_residual_when_controller_infeasible`
  - Added `test_select_active_knot_rejects_empty_trajectory`
  - Added `test_bounded_vertical_rotor_allocator_feasible_and_unsupported_residual`
  - Added `test_bounded_vertical_rotor_allocator_reports_infeasible_clip`
  - Added `test_qpid_controller_outputs_controller_command`
  - Added `test_qpid_controller_reports_infeasible_vertical_wrench`
- Commands run:
  - `sed -n ...` and `rg -n ...` inspections for spec Section 20/26/27, policy schemas, pi_H planner, controller bias builder, physical-model builder, and existing tests
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/policies/test_low_level_baseline.py -q`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/controllers/test_qpid_controller.py -q`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q`
  - `python3 -m compileall amsrr -q`
  - `git diff --check`
  - `find amsrr tests -type d -name __pycache__ -prune -exec rm -rf {} +`
- Tests run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/policies/test_low_level_baseline.py -q` passed: 4 passed. `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/controllers/test_qpid_controller.py -q` passed: 4 passed. `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q` passed: 61 passed, 1 skipped. `python3 -m compileall amsrr -q` passed. `git diff --check` passed.
- Assumptions: P1 pi_L baseline is a deterministic tracking-intent scaffold. Object pose error is converted to a clipped residual wrench proxy; contact tracking bias is a small scaled copy of active assignment wrench targets. The P1 controller allocator supports bounded vertical thrust allocation only and reports unsupported lateral/torque wrench residuals as metrics/violations.
- Blockers: None.
- Next steps: Continue with P1 order 7, simplified grasp-carry simulation environment.

### 2026-07-08
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4
- Work package / Agent label: Agent H: pi_H baseline planner
- Summary: Implemented a deterministic P1 grasp/carry high-level planner that selects feasible contact assignments from `ContactCandidateSet` group proposals, caches assignment feasibility labels, and emits a schema-valid `ContactWrenchTrajectory`.
- Files changed:
  - `amsrr/policies/__init__.py`
  - `amsrr/policies/high_level_policy_base.py`
  - `amsrr/policies/contact_wrench_trajectory.py`
  - `tests/unit/policies/test_high_level_baseline.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: No persisted schema changes. Added policy-side `HighLevelPolicyContext`, `HighLevelPolicyBase`, `BaselineTrajectoryPlannerConfig`, and `GraspCarryBaselinePlanner`.
- Upstream dependencies used: v0.4 Sections 19, 26.8, 27.1, 28.10; Agent H ContactCandidateSampler; selected assignment feasibility evaluator; existing policy schemas.
- Downstream impact: Agent I pi_L baseline can now consume a deterministic `ContactWrenchTrajectory` with approach/attach/maintain/release assignments, posture anchor targets, object goal targets, and priority weights.
- Tests added or run:
  - Added `test_grasp_carry_baseline_planner_outputs_contact_wrench_trajectory`
  - Added `test_select_feasible_assignments_uses_grasp_pair_group`
- Commands run:
  - `sed -n ...` inspections for spec Section 19, Agent H deliverables, policy schemas, and existing tests
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/policies/test_high_level_baseline.py -q`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q`
  - `python3 -m compileall amsrr -q`
  - `find amsrr tests -type d -name __pycache__ -prune -exec rm -rf {} +`
- Tests run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/policies/test_high_level_baseline.py -q` passed: 2 passed. `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q` passed: 53 passed, 1 skipped. `python3 -m compileall amsrr -q` passed.
- Assumptions: P1 pi_H baseline prefers `grasp_pair` proposals and emits a fixed five-knot grasp/carry schedule. It is deterministic scaffold logic, not a learned high-level policy or exhaustive assignment search.
- Blockers: None.
- Next steps: Continue with implementation order item 15 / P1 order 6, Agent I pi_L baseline policy and controller interface work.

### 2026-07-08
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4
- Work package / Agent label: Agent H/F: Selected assignment feasibility proxy
- Summary: Implemented selected-assignment feasibility evaluation for π_H-selected `ContactAssignment` sets, including candidate consistency, slot cardinality, pairwise conflict, grasp-opposition wrench proxy, friction/collision/QP residual hooks, cache updates, exports, and unit tests.
- Files changed:
  - `amsrr/policies/__init__.py`
  - `amsrr/policies/assignment_feasibility.py`
  - `tests/unit/policies/test_contact_candidate_interfaces.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: No persisted schema changes. Added a policy-side evaluator function that returns the existing `AssignmentFeasibilityResult` schema.
- Upstream dependencies used: v0.4 Sections 18.6, 18.7, 19.3, Appendix B.4, Appendix C; existing `ContactCandidateSet`, `ContactAssignment`, pairwise conflict matrices, and assignment-feasibility cache.
- Downstream impact: Agent H π_H baseline can evaluate only its selected assignments and cache infeasible selections without enumerating arbitrary candidate subsets. Later exact QP/collision/wrench evaluators can pass residuals/margins through the same result schema.
- Tests added or run:
  - Added `test_selected_assignment_feasibility_accepts_opposing_grasp_pair`
  - Added `test_selected_assignment_feasibility_rejects_cardinality_and_pair_conflict`
  - Added `test_selected_assignment_feasibility_rejects_non_opposing_grasp_normals`
- Commands run:
  - `sed -n ...` inspections for spec Sections 18.6, 18.7, 19.3, Appendix B/C, and existing assignment feasibility code
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/policies/test_contact_candidate_interfaces.py -q`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q`
  - `python3 -m compileall amsrr -q`
  - `find amsrr tests -type d -name __pycache__ -prune -exec rm -rf {} +`
- Tests run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/policies/test_contact_candidate_interfaces.py -q` passed: 5 passed. `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q` passed: 51 passed, 1 skipped. `python3 -m compileall amsrr -q` passed.
- Assumptions: Assignment-level hard checks here are deterministic proxies: selected cardinality, pair conflicts, friction margin, and opposing-normal grasp proxy. They are not exact wrench closure, exact collision, or exact QP solving.
- Blockers: None.
- Next steps: Continue with implementation order item 14, Agent H π_H trajectory schema/baseline planner.

### 2026-07-08
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4
- Work package / Agent label: Agent H: ContactCandidateSampler
- Summary: Implemented a deterministic morphology-conditioned `ContactCandidateSampler` for P1 grasp/carry, optional group proposal support in `build_contact_candidate_set`, package exports, and unit tests covering non-empty candidate generation, grasp-pair proposals, anchor association preservation, and serialization.
- Files changed:
  - `amsrr/policies/__init__.py`
  - `amsrr/policies/contact_candidate_set.py`
  - `amsrr/policies/contact_candidate_sampler.py`
  - `tests/unit/policies/test_contact_candidate_sampler.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: No persisted schema changes. Extended the existing contact-candidate helper function with optional `group_proposals` and `sampler_version` arguments while preserving prior defaults.
- Upstream dependencies used: v0.4 Sections 18, 24.2, 26.8, 27.1, 28.9; Agent D IRGBuilder and EnvelopeExtractor; Agent E fixed/simple `DesignOutput`; GeometryProcessor descriptors and ContactRegionGraph; existing ContactCandidate schemas.
- Downstream impact: Agent H π_H baseline planner can now consume finite morphology-conditioned `ContactCandidateSet` objects with slot coverage, pairwise matrices, and small grasp/support group proposals.
- Tests added or run:
  - Added `test_contact_candidate_sampler_returns_non_empty_grasp_carry_candidates`
  - Added `test_contact_candidate_sampler_builds_grasp_pair_group_proposals`
  - Added `test_contact_candidate_sampler_uses_robot_anchor_associations`
- Commands run:
  - `sed -n ...` inspections for spec Section 18, Agent H deliverables, geometry, IRG, morphology, and existing candidate helpers
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/policies/test_contact_candidate_sampler.py -q`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q`
  - `python3 -m compileall amsrr -q`
  - `find amsrr tests -type d -name __pycache__ -prune -exec rm -rf {} +`
- Tests run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/policies/test_contact_candidate_sampler.py -q` passed: 3 passed. `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q` passed: 48 passed, 1 skipped. `python3 -m compileall amsrr -q` passed.
- Assumptions: P1 sampler emits deterministic smoke candidates and unary scores, not exact reachability/collision/QP feasibility. Default quota is one candidate per ContactSlot × ContactRegion × RobotAnchor. Grasp-pair proposals are small pairwise/group hints and are not task-feasibility proofs.
- Blockers: None.
- Next steps: Continue with implementation order item 14, Agent H π_H trajectory schema/baseline planner and selected-assignment feasibility interface.

### 2026-07-08
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4
- Work package / Agent label: Agent G: π_A GraphEditAssemblyPlanner
- Summary: Implemented deterministic graph-edit assembly planning over target `MorphologyGraph` dock edges, construction-state helpers, control handoff request scaffolding, executor interface records, package exports, and Agent G unit tests.
- Files changed:
  - `amsrr/assembly/__init__.py`
  - `amsrr/assembly/construction_state.py`
  - `amsrr/assembly/graph_edit_planner.py`
  - `amsrr/assembly/control_handoff.py`
  - `amsrr/assembly/executor_interface.py`
  - `tests/unit/assembly/test_graph_edit_planner.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: No changes to existing persisted schema modules. Added implementation-local assembly dataclasses/interfaces that match v0.4 Section 17 contracts inside `amsrr/assembly`.
- Upstream dependencies used: v0.4 Sections 17, 26.7, 27.1; existing `MorphologyGraph`, `DockEdge`, `Violation`, `MinimalMorphologyBuilder`, IRGBuilder, and PhysicalModel builder.
- Downstream impact: P1/P3 can now derive deterministic assembly step sequences from fixed/simple target morphologies. Agent H can proceed to ContactCandidateSampler using assembled/target graph contracts without needing learned assembly.
- Tests added or run:
  - Added `test_initial_construction_state_contains_base_only`
  - Added `test_graph_edit_planner_builds_deterministic_attach_sequence`
  - Added `test_graph_edit_planner_resumes_from_construction_state`
  - Added `test_control_handoff_request_for_docking_step`
- Commands run:
  - `sed -n ...` and `rg -n ...` inspections for spec Section 17, 26.7, 27.1, existing schemas, and tests
  - `mkdir -p amsrr/assembly tests/unit/assembly`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/assembly/test_graph_edit_planner.py -q`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q`
  - `python3 -m compileall amsrr -q`
  - `find amsrr tests -type d -name __pycache__ -prune -exec rm -rf {} +`
- Tests run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/assembly/test_graph_edit_planner.py -q` passed: 4 passed. `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q` passed: 45 passed, 1 skipped. `python3 -m compileall amsrr -q` passed.
- Assumptions: Target morphology is treated as a connected tree rooted at `base_module_id` for v1/P1 scaffold planning. Each new dock edge expands to `move_to_staging -> align_ports -> dock -> verify_attach`. Exact assembly motion planning, retry execution, learned assembly, and simulator verification are out of scope for this slice.
- Blockers: None.
- Next steps: Continue with implementation order item 13, Agent H ContactCandidateSampler and ContactCandidateSet group proposal generation.

### 2026-07-07
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4
- Work package / Agent label: Agent E: Deterministic design teacher and π_D scaffolding
- Summary: Implemented the P1 fixed/simple morphology provider surface for π_D by adding a `DesignPolicyContext`, fixed-simple baseline design policy, deterministic design teacher variants, and a small action-candidate/STOP-mask generator over teacher traces.
- Files changed:
  - `amsrr/policies/__init__.py`
  - `amsrr/policies/design_policy_base.py`
  - `amsrr/policies/design_candidate_generator.py`
  - `amsrr/policies/design_teacher.py`
  - `tests/unit/policies/test_design_teacher.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: No persisted schema changes. Added policy-side interface/scaffold modules that consume existing `TaskSpec`, `InteractionRequirementGraph`, `InteractionEnvelope`, `PhysicalModel`, and `DesignOutput` schemas.
- Upstream dependencies used: v0.4 Sections 14, 15, 24.2, 26.5, 27.1; existing Agent E/F minimal morphology builder; Agent F FeasibilityChecker; Agent D IRGBuilder and EnvelopeExtractor; Agent B PhysicalModel.
- Downstream impact: Agent G can consume a deterministic target `DesignOutput` for assembly planning; Agent H can consume fixed/simple morphology and RobotAnchors for ContactCandidateSampler implementation; later learned π_D heads can replace the teacher scorer while keeping the same `DesignPolicyContext -> DesignOutput` boundary.
- Tests added or run:
  - Added `test_design_teacher_selects_p1_grasp_support_variant`
  - Added `test_design_candidate_trace_masks_stop_until_final_step`
  - Added `test_fixed_simple_design_policy_outputs_feasible_stop`
- Commands run:
  - `sed -n ...` inspections for spec Sections 14, 15, 26.5, 27.1, schema, morphology builder, and existing tests
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/policies/test_design_teacher.py -q`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q`
  - `python3 -m compileall amsrr -q`
  - `find amsrr tests -type d -name __pycache__ -prune -exec rm -rf {} +`
- Tests run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/policies/test_design_teacher.py -q` passed: 3 passed. `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q` passed: 41 passed, 1 skipped. `python3 -m compileall amsrr -q` passed.
- Assumptions: Teacher variants are deterministic labels over the existing minimal connected-tree morphology scaffold. P1 object grasp/carry defaults to `tri_anchor_support_grasp` when the IRG contains required grasp slots plus an optional support slot. The candidate generator is an action-mask scaffold, not a learned scorer.
- Blockers: None.
- Next steps: Continue with implementation order item 12, Agent G π_A GraphEditAssemblyPlanner, then Agent H ContactCandidateSampler.

### 2026-07-07
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4
- Work package / Agent label: Agent H/I: P0 interface-only smoke pieces
- Summary: Implemented P0 interface-only helpers for ContactCandidateSet pairwise compatibility, assignment-level QP infeasibility reporting, and PolicyCommand-to-QP/PID reference bias building.
- Files changed:
  - `amsrr/policies/__init__.py`
  - `amsrr/policies/contact_candidate_set.py`
  - `amsrr/policies/assignment_feasibility.py`
  - `amsrr/controllers/__init__.py`
  - `amsrr/controllers/policy_command_builder.py`
  - `tests/unit/policies/test_contact_candidate_interfaces.py`
  - `tests/unit/controllers/test_policy_command_builder.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: No persisted schema changes. Added interface helper modules that consume existing `ContactCandidateSet`, `AssignmentFeasibilityResult`, `ContactAssignment`, `InteractionKnot`, and `PolicyCommand` schemas.
- Upstream dependencies used: v0.4 Sections 18, 19, 20, 26.8, 26.9, 27.2, 28.9, 28.10, 28.11, Appendix B.4; existing policy/contact candidate schemas.
- Downstream impact: ContactCandidateSampler, π_H trajectory planners, π_L policies, and controller backends have smoke-tested interface contracts for candidate pairwise matrices, selected-assignment feasibility cache entries, and desired bias references.
- Tests added or run:
  - Added `test_contact_candidate_pairwise_conflict_matrix`
  - Added `test_assignment_level_qp_infeasible_case`
  - Added `test_policy_command_bias_builder`
- Commands run:
  - `rg -n ...` and `sed -n ...` inspections for spec Sections 18-20, 26.8, 26.9, and 27.2
  - `mkdir -p amsrr/policies amsrr/controllers tests/unit/policies tests/unit/controllers`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q`
  - `python3 -m compileall amsrr -q`
  - `find amsrr tests -type d -name __pycache__ -prune -exec rm -rf {} +`
- Tests run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q` passed: 38 passed, 1 skipped. `python3 -m compileall amsrr -q` passed.
- Assumptions: Pairwise conflict is limited to immediate candidate conflicts such as shared robot anchor; no exhaustive subset feasibility is performed. Assignment-level QP infeasibility only evaluates a selected assignment set. PolicyCommandBiasBuilder emits references for QP/PID and never final actuator commands.
- Blockers: None.
- Next steps: P0 Section 27.2 unit-test smoke coverage is now complete; later phases can implement full ContactCandidateSampler, π_H baseline trajectory planner, π_L baseline policy, and controller interfaces.

### 2026-07-07
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4
- Work package / Agent label: Agent E/F: Minimal MorphologyGraph + Feasibility hard-check scaffolding
- Summary: Implemented a deterministic minimal MorphologyGraph/DesignOutput builder and a design-level FeasibilityChecker scaffold for schema, connected graph, module count, port compatibility, closed-loop rejection, required slot coverage, coarse reachability, thrust margin, payload margin, and hover proxy checks.
- Files changed:
  - `amsrr/morphology/__init__.py`
  - `amsrr/morphology/graph.py`
  - `amsrr/feasibility/__init__.py`
  - `amsrr/feasibility/checker.py`
  - `amsrr/feasibility/violation_codes.py`
  - `tests/unit/morphology/test_minimal_morphology_builder.py`
  - `tests/unit/feasibility/test_feasibility_checker.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: No persisted schema changes. Added implementation modules that consume existing `MorphologyGraph`, `DesignOutput`, and `FeasibilityResult` schemas.
- Upstream dependencies used: v0.4 Sections 14, 15.2, 15.3, 16, 26.5, 26.6, 27.1, 27.2, 28.6, 28.7; Agent B PhysicalModel; Agent D IRGBuilder.
- Downstream impact: Later design-policy scaffolding, assembly planning, contact candidate sampling, and assignment-level feasibility can consume a deterministic seed morphology and checker result.
- Tests added or run:
  - Added `test_minimal_morphology_builder_grasp_carry_design_output`
  - Added `test_minimal_morphology_design_output_roundtrip`
  - Added `test_feasibility_checker_accepts_minimal_design`
  - Added `test_feasibility_checker_rejects_missing_required_slot_coverage`
  - Added `test_feasibility_checker_rejects_closed_loop_v1`
- Commands run:
  - `rg -n ...` and `sed -n ...` inspections for spec Sections 14/16/26/28, schemas, and robot model utilities
  - `mkdir -p amsrr/morphology amsrr/feasibility/checks tests/unit/morphology tests/unit/feasibility`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q`
  - `python3 -m compileall amsrr -q`
  - `find amsrr tests -type d -name __pycache__ -prune -exec rm -rf {} +`
- Tests run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q` passed: 35 passed, 1 skipped. `python3 -m compileall amsrr -q` passed.
- Assumptions: Minimal morphology is a deterministic seed/teacher scaffold, not an optimized policy output. Coarse thrust margin uses `abs(thrust_axis_local.z) * thrust_max_n` for the vectoring-capable Holon proxy. Coarse collision and QP hover are represented by necessary-condition scaffold checks, not exact simulation/QP.
- Blockers: None.
- Next steps: Continue with design policy scaffolding / deterministic teacher generator, then assembly planning and contact candidate sampling.

### 2026-07-07
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4
- Work package / Agent label: Agent A/L: SharedInteractionWorkspace tensor/mask contract
- Summary: Implemented workspace token group schema, strict group mask/slice validation, recommended learned-query specs, and a SharedInteractionWorkspaceBuilder that assembles modality token groups into a padded shared workspace with required empty groups.
- Files changed:
  - `amsrr/schemas/workspace.py`
  - `amsrr/encoders/__init__.py`
  - `amsrr/encoders/workspace_builder.py`
  - `tests/unit/schemas/test_workspace.py`
  - `tests/unit/encoders/test_workspace_builder.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: Strengthened internal workspace validation by adding `WorkspaceTokenGroup`, `OPTIONAL_WORKSPACE_GROUPS`, `WORKSPACE_GROUPS`, required `group_masks` shape checks, optional `contact_candidates`, and `recommended_learned_query_specs`.
- Upstream dependencies used: v0.4 Sections 21.6, 21.7, 26.1, 27.1, 27.2; prior InteractionEnvelopeEncoder token-group output.
- Downstream impact: Future modality encoders can produce `WorkspaceTokenGroup` objects and use `SharedInteractionWorkspaceBuilder` to assemble a single tensor/mask/source-id contract for π_D/π_H/π_L/critic/feasibility heads.
- Tests added or run:
  - Added `test_workspace_rejects_group_mask_mismatch`
  - Added `test_workspace_token_group_shapes`
  - Added `test_learned_query_spec_contract`
  - Added `test_workspace_builder_assembles_required_group_slices`
  - Added `test_workspace_builder_supports_optional_contact_candidate_group`
  - Added `test_workspace_builder_rejects_mismatched_d_model`
  - Added `test_empty_workspace_token_group_contract`
- Commands run:
  - `rg -n ...` and `sed -n ...` inspections for spec Section 21, workspace schema, and encoder outputs
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q`
  - `python3 -m compileall amsrr -q`
  - `find amsrr tests -type d -name __pycache__ -prune -exec rm -rf {} +`
- Tests run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q` passed: 30 passed, 1 skipped. `python3 -m compileall amsrr -q` passed.
- Assumptions: Empty modality groups are represented as `[B, 0]` nested-list rows plus explicit `d_model`, then become zero-width slices in the assembled workspace. Query specs are contracts only; learned query parameters are not implemented here.
- Blockers: None.
- Next steps: Implementation order item 9 can build MorphologyGraph and DesignOutput; later modality encoders can feed additional non-empty groups into the workspace builder.

### 2026-07-07
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4
- Work package / Agent label: Agent D/A: InteractionEnvelopeExtractor + InteractionEnvelopeEncoder
- Summary: Implemented deterministic InteractionEnvelope extraction from IRG and a dependency-free InteractionEnvelopeEncoder contract that emits padded token tensors, masks, token type ids, source type ids, and source ids for the `interaction_envelope` workspace group.
- Files changed:
  - `amsrr/irg/__init__.py`
  - `amsrr/irg/envelope_extractor.py`
  - `amsrr/encoders/__init__.py`
  - `amsrr/encoders/interaction_envelope_encoder.py`
  - `tests/unit/irg/test_envelope_extractor.py`
  - `tests/unit/encoders/test_interaction_envelope_encoder.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: No persisted schema changes. Added internal encoder output contract `InteractionEnvelopeEncoderOutput` for P0 token/mask/source-id handoff.
- Upstream dependencies used: v0.4 Sections 13, 21, 26.4, 27.1, 27.2, 28.5; Agent A schemas and workspace tensor shape helpers; Agent D IRGBuilder output.
- Downstream impact: π_D/π_H scaffolding and future SharedInteractionWorkspace assembly can consume deterministic envelope tokens. ContactCandidateSampler can use envelope target region sets, contact count ranges, and modes without reinterpreting TaskSpec directly.
- Tests added or run:
  - Added `test_interaction_envelope_extract`
  - Added `test_interaction_envelope_extracts_all_task_families`
  - Added `test_interaction_envelope_encoder_contract`
  - Added `test_interaction_envelope_encoder_batch_padding`
- Commands run:
  - `rg -n ...` and `sed -n ...` inspections for spec Sections 13/21, schemas, and IRG templates
  - `mkdir -p amsrr/encoders tests/unit/encoders`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q`
  - `python3 -m compileall amsrr -q`
  - `find amsrr tests -type d -name __pycache__ -prune -exec rm -rf {} +`
- Tests run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q` passed: 23 passed, 1 skipped. `python3 -m compileall amsrr -q` passed.
- Assumptions: Required contact count range aggregates required ContactSlots only; optional slots still contribute contact mode and target-region tokens. The encoder implements the deterministic contract and `mlp_embedding` fallback metadata, not learned parameters.
- Blockers: None.
- Next steps: Implementation order item 8 can assemble modality token groups into full SharedInteractionWorkspace and learned query pooling contracts.

### 2026-07-07
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4
- Work package / Agent label: Agent D: IRGBuilder + InteractionTemplates
- Summary: Implemented deterministic IRGBuilder, SceneGraph normalization, IRG structural validator, and all five P0 task-family templates: free-flight navigation, object grasp/carry, valve operation, perching manipulation, and contact-mediated locomotion.
- Files changed:
  - `amsrr/irg/__init__.py`
  - `amsrr/irg/irg_builder.py`
  - `amsrr/irg/validator.py`
  - `amsrr/irg/templates/__init__.py`
  - `amsrr/irg/templates/base.py`
  - `amsrr/irg/templates/free_flight.py`
  - `amsrr/irg/templates/object_grasp_carry.py`
  - `amsrr/irg/templates/valve_operation.py`
  - `amsrr/irg/templates/perching_manipulation.py`
  - `amsrr/irg/templates/contact_mediated_locomotion.py`
  - `tests/unit/irg/test_irg_builder.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: None. Existing TaskSpec, GeometryDescriptor, InteractionRequirementGraph, IRGNode, IRGEdge, PhaseType, ConstraintType, CapabilityType, and ContactMode schemas were used unchanged.
- Upstream dependencies used: v0.4 Sections 10, 11, 12, 26.4, 27.1, 27.2, 28.3, 28.4; Agent A schemas; Agent C GeometryProcessor outputs.
- Downstream impact: Agent E EnvelopeExtractor and downstream policy/feasibility work can now consume valid IRGs for every P0 task family. IRGs remain abstract and do not include final contact poses, robot anchors, morphology, trajectories, or actuator commands.
- Tests added or run:
  - Added `test_phase_label_to_phase_type_mapping`
  - Added `test_irg_builder_grasp_carry_valid`
  - Added `test_irg_builder_all_task_families_smoke`
- Commands run:
  - `find amsrr/irg tests/unit/irg -type f | sort`
  - `sed -n ...` inspections for IRGBuilder, templates, validator, schemas, and spec sections
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q`
  - `python3 -m compileall amsrr -q`
  - `python3 - <<'PY' ...` smoke inspection of object grasp/carry IRG node and edge counts
  - `find amsrr tests -type d -name __pycache__ -prune -exec rm -rf {} +`
- Tests run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q` passed: 19 passed, 1 skipped. `python3 -m compileall amsrr -q` passed.
- Assumptions: Template-local phase labels are preserved in `phase_label` and mapped into existing `PhaseType` values. Template-local constraint concepts that are not v0.4 `ConstraintType` enum values are represented by the closest standard enum and preserved in `parameters["template_constraint"]`.
- Blockers: None.
- Next steps: Agent E EnvelopeExtractor should compute compact summaries from these IRGs without treating the envelope as source of truth. The Section 26.4 `envelope_extractor.py` item remains for the next work package because this task explicitly targeted item 6, IRGBuilder and all task templates.

### 2026-07-07
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4
- Work package / Agent label: Agent C: GeometryProcessor
- Summary: Implemented deterministic GeometryProcessor for primitives and mesh smoke, including asset resolution, primitive analytic surface decomposition, STL/OBJ mesh summary loading, surface patch graph construction, and contact region graph construction.
- Files changed:
  - `amsrr/geometry/__init__.py`
  - `amsrr/geometry/asset_resolver.py`
  - `amsrr/geometry/surface_patch_graph.py`
  - `amsrr/geometry/contact_region_extractor.py`
  - `amsrr/geometry/geometry_processor.py`
  - `tests/unit/geometry/test_geometry_processor.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: None. Existing `GeometryDescriptor`, `GlobalShapeFeatures`, `SurfacePatchGraph`, `ContactRegionGraph`, `SurfacePatchToken`, and `ContactRegion` schemas were used unchanged.
- Upstream dependencies used: v0.4 Sections 8.1-8.10, 26.3, 27.1, 27.2; Agent A schemas; existing mesh assets under `module_urdf/mesh/`.
- Downstream impact: Agent D IRGBuilder can consume primitive and mesh `GeometryDescriptor` outputs. P0 now has box primitive regions and mesh smoke coverage.
- Tests added or run:
  - Added `test_geometry_processor_box_regions`
  - Added `test_geometry_processor_mesh_smoke`
- Commands run:
  - `mkdir -p amsrr/geometry tests/unit/geometry assets/objects/primitives assets/objects/meshes`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q`
  - `python3 -m compileall amsrr -q`
  - `find amsrr tests -type d -name __pycache__ -prune -exec rm -rf {} +`
- Tests run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q` passed: 16 passed, 1 skipped. `python3 -m compileall amsrr -q` passed.
- Assumptions: P0 mesh support is a deterministic smoke implementation, not full mesh repair/segmentation. Mesh descriptors expose hashed refs such as `mesh://sha256:<hash>` instead of raw asset paths.
- Blockers: None.
- Next steps: Agent D IRGBuilder and templates can use these descriptors; later mesh work can replace smoke normal-cluster aggregation with richer segmentation without schema changes.

### 2026-07-07
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4
- Work package / Agent label: Agent B: URDF / PhysicalModel
- Summary: Implemented URDF/xacro XML loader, thrust model YAML loader, PhysicalModel builder, ModuleCapabilityToken builder, normalized runtime Holon URDF asset, and Agent B unit tests.
- Files changed:
  - `assets/robots/holon/holon.urdf`
  - `amsrr/robot_model/__init__.py`
  - `amsrr/robot_model/urdf_loader.py`
  - `amsrr/robot_model/thrust_model.py`
  - `amsrr/robot_model/physical_model_builder.py`
  - `tests/unit/robot_model/test_urdf_loader.py`
  - `tests/unit/robot_model/test_thrust_model.py`
  - `tests/unit/robot_model/test_physical_model_builder.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: None. Existing `PhysicalModel`, `LinkModel`, `JointModel`, `RotorModel`, `DockPortSpec`, and `ModuleCapabilityToken` schemas were used unchanged.
- Upstream dependencies used: v0.4 Sections 3.1, 3.2, 9.1-9.8, 26.2, 27.1, 27.2; `module_urdf/README_for_codex.md`; existing Agent A schemas.
- Downstream impact: Agent C/D/F and later controller work can now load a configurable runtime URDF path and receive structured `PhysicalModel` plus compact module capability features.
- Tests added or run:
  - Added `test_urdf_parse_holon_if_present`
  - Added `test_urdf_parse_holon_xacro_reference`
  - Added `test_asset_urdf_uses_config_thrust_link_names`
  - Added `test_thrust_model_loads_config`
  - Added `test_thrust_model_rejects_duplicate_rotor_ids`
  - Added `test_physical_model_total_mass_positive`
  - Added `test_physical_model_rotors_and_dock_ports`
  - Added `test_module_capability_token_from_physical_model`
- Commands run:
  - `mkdir -p assets/robots/holon`
  - `cp module_urdf/holon.urdf.xacro assets/robots/holon/holon.urdf`
  - `perl -0pi -e 's/thrust([1-4])\b/thrust_$1/g' assets/robots/holon/holon.urdf`
  - `mkdir -p amsrr/robot_model tests/unit/robot_model`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q`
  - `python3 -m compileall amsrr -q`
  - `find amsrr tests -type d -name __pycache__ -prune -exec rm -rf {} +`
- Tests run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q` passed: 14 passed, 1 skipped. `python3 -m compileall amsrr -q` passed.
- Assumptions: `module_urdf/holon.urdf.xacro` can be parsed as XML without ROS/xacro macro expansion. Runtime asset path remains configurable and uses `assets/robots/holon/holon.urdf`. `thrust_1` config IDs are preserved as schema rotor IDs.
- Blockers: None. `module_urdf/holon.urdf` is absent, so its explicit test is skipped by design.
- Next steps: Agent C GeometryProcessor for primitives and mesh smoke, or Agent D IRGBuilder once geometry descriptors exist.

### 2026-07-07
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4
- Work package / Agent label: Repository organization / handoff documentation
- Summary: Moved Codex-facing project documents into `for_codex/` and prepared current implementation files for git tracking.
- Files changed:
  - `for_codex/A-MSRR_codex_ready_spec_v0_4_ja.md`
  - `for_codex/AGENTS.md`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Schema/interface changes: None.
- Upstream dependencies used: User request to move Codex-facing documents under `for_codex/` and commit current implementation.
- Downstream impact: Future coding assistants should read `for_codex/AGENTS.md`, `for_codex/A-MSRR_codex_ready_spec_v0_4_ja.md`, and `for_codex/WORKLOG.md` from the new directory.
- Tests added or run: Reran current unit tests before commit.
- Commands run:
  - `git status --short`
  - `git status --short --untracked-files=all`
  - `git log --oneline --max-count=5`
  - `git ls-files`
  - `mkdir -p for_codex`
  - `git mv A-MSRR_codex_ready_spec_v0_4_ja.md AGENTS.md AMSRR_design_modification_by_codex.md WORKLOG.md for_codex/`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q`
  - `find amsrr tests -type d -name __pycache__ -prune -exec rm -rf {} +`
- Tests run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q` passed: 7 passed.
- Assumptions: Documentation relocation does not change runtime import paths or test behavior.
- Blockers: None.
- Next steps: Stage and commit moved documentation plus current schema/config/test implementation.

### 2026-07-07
- Spec version: A-MSRR_codex_ready_spec_v0_4_ja.md v0.4
- Work package / Agent label: Agent A: Schemas and validation; Agent A/L support: config loading, hashing, initial test harness
- Summary: Implemented schema-first dataclass models, strict JSON/YAML coercion and validation helpers, config loading, stable hashing, minimum robot/training config files, and unit tests for schema roundtrip, TaskSpec parsing, workspace masks, and config hashing.
- Files changed:
  - `amsrr/__init__.py`
  - `amsrr/schemas/__init__.py`
  - `amsrr/schemas/common.py`
  - `amsrr/schemas/task_spec.py`
  - `amsrr/schemas/geometry.py`
  - `amsrr/schemas/irg.py`
  - `amsrr/schemas/interaction_envelope.py`
  - `amsrr/schemas/morphology.py`
  - `amsrr/schemas/physical_model.py`
  - `amsrr/schemas/runtime.py`
  - `amsrr/schemas/policies.py`
  - `amsrr/schemas/feasibility.py`
  - `amsrr/schemas/workspace.py`
  - `amsrr/schemas/contact_candidates.py`
  - `amsrr/utils/__init__.py`
  - `amsrr/utils/config.py`
  - `amsrr/utils/hashing.py`
  - `configs/robot/robot_model.yaml`
  - `configs/robot/thrust_model.yaml`
  - `configs/training/p0_schema_tests.yaml`
  - `tests/conftest.py`
  - `tests/unit/schemas/test_task_spec.py`
  - `tests/unit/schemas/test_schema_roundtrip.py`
  - `tests/unit/schemas/test_workspace.py`
  - `tests/unit/utils/test_config_hashing.py`
- Schema/interface changes: Initial schema/interface implementation. Added approved supplement `IRGEdgeType.ALLOWS = "allows"` because v0.4 uses `allows` edges in diagrams and examples but omits it from the enum listing.
- Upstream dependencies used: v0.4 Sections 7, 8, 9, 10, 13, 14, 16, 18, 19, 20, 21, 23, 25, 26.1, 27.1, 27.2; AGENTS.md implementation rules.
- Downstream impact: Agent B/C/D can now consume stable dataclass schemas and config/hash utilities. IRGBuilder can emit `contact_region --allows--> contact_slot` edges without inventing a local edge string.
- Tests added or run:
  - Added `test_task_spec_parse_grasp_carry_yaml`
  - Added `test_task_spec_rejects_missing_grasp_carry_mass`
  - Added `test_schema_roundtrip_json`
  - Added `test_irg_edge_type_includes_allows`
  - Added `test_shared_interaction_workspace_tensor_shapes`
  - Added `test_padded_tensor_masks`
  - Added `test_config_loading_and_hashing`
- Commands run:
  - `python3 --version`
  - `python3 -m pytest --version`
  - `python3 -c "import yaml; print(yaml.__version__)"`
  - `python3 -m pytest tests/unit -q` failed before collection due external pytest plugin `launch_testing` hook incompatibility.
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q` passed: 7 passed.
  - `python3 -m compileall amsrr -q` passed.
  - `find amsrr tests -type d -name __pycache__ -prune -exec rm -rf {} +` removed generated Python cache directories after tests/compile checks.
- Tests run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q` passed; `python3 -m compileall amsrr -q` passed.
- Assumptions: No new package installation; use standard-library dataclasses plus installed PyYAML. Spec examples may omit optional fields, so optional schema fields default to `None` where needed for the provided YAML.
- Blockers: None for Agent A items 1-2. Full P0 tests for URDF, GeometryProcessor, IRGBuilder, EnvelopeExtractor, and downstream policy/controller behavior remain unimplemented.
- Next steps: Agent B URDF/PhysicalModel loader and/or Agent C primitive GeometryProcessor, using the schemas added here.

---

## Work Package Logs

### Agent H/I/J/K/L: Order 8 Object Natural-Contact Smoke

#### 2026-07-18 (cone-pad lift low-load GUI replay)
- Scope/result: Added a one-command balanced Kit replay for the exact v375 cone-pad physical trajectory. The cached `670`-frame trace records `5.43 deg` Dock motion and `11.66 mm` object COM rise; a real replay reproduced the full PhysX DOF motion with zero write error and a normally lit, non-black viewport.
- Safety/boundary: Replay disables contact dynamics, gravity, graph constraints, and object/environment collision and is visual-only. It displays all `150` cone pads but creates no new lift/contact evidence and changes no production interface.
- Handoff: Run `python3 scripts/order8_cone_proxy_lift_replay_gui.py`; the default one-loop `0.5x` playback suppresses routine Kit logs and prints live phase/time/PhysX joint motion. User visual review is pending.

#### 2026-07-18 (cone micro-pad physical lift diagnostic)
- Scope/result: Connected the visually approved cone-only `75/link` proxy geometry to the default-off diagnostic runtime, corrected q_close geometry to use the active proxy surfaces, and ran the full acquisition-to-lift slice with a free `1.0 kg` object. v375 physically lifted the object: support separation was confirmed at `26.58 s`, COM rose `11.663 mm`, and OBB bottom clearance reached `3.009 mm`.
- Failure/boundary: The run safe-held on cumulative slip `18.194/30.233 mm` before the unchanged `100 mm` lift gate. Slower payload transfer reduced rather than improved clearance. This is acceptance-ineligible proof that the pad grasp can lift, not an Order 8 pass; production remains authored-mesh, proxy-disabled, and unchanged.
- Handoff: No Isaac/Kit process remains. Stop at the contact-retention method/tuning choice; after approval, use the shortest effective-surface q_close-to-lift A/B before attempting transport/place/release/settle.

#### 2026-07-18 (cone-only merged micro-pad preview)
- Scope/result: Removed every pad outside the identified conical contact shell and merged neighbouring surface cells from the previous whole-side `12 x 24` partition into a cone-only `4 x 20` partition. Shape count is now `75/link`, `150` total, versus `474` in v2; the per-pad local fit remains within `1.406 mm` maximum.
- Verification/boundary: `14` focused tests plus a real render-only Isaac spawn passed. Runtime and acceptance remain disconnected pending visual approval via `python3 scripts/order8_side_proxy_pad_gui.py`.

#### 2026-07-18 (collision-following micro-pad preview correction)
- Scope/result: Visual feedback rejected the one-plane full-side geometry. Preview v2 now uses `237` independently oriented `6--16 x 6--16 x 0.8 mm` local surface tiles on each selected yaw Dock, fitted from the actual collision STL over `12` axial bands and `24` circumferential sectors with a `0.2 mm` clearance.
- Verification/boundary: Geometry/GUI selection passed `14` tests and a real single-module Isaac stage smoke authored `474` colliders with zero rollout steps. This is still preview-only and cannot enter runtime or acceptance before user visual approval.
- Handoff: Run `python3 scripts/order8_side_proxy_pad_gui.py`; use `--focus-link` for close inspection. If approved, the next task is a separately authorized minimum contact slice, not a full Order 8 run.

#### 2026-07-18 (full-side proxy-pad placement preview)
- Scope: Build and visualize the user-requested complete side-covering proxy geometry before any physical contact run.
- Files changed: Added the preview config, pure URDF/STL full-projection geometry builder, one-module Kit viewer, two focused test files, design supplement, and this log.
- Implemented: Two fixed link-local, translucent-orange, collision-enabled plates covering the complete grasp-facing projections of `yaw_dock_mech1/2`; explicit mesh search/provenance; no object-conditioned placement; automatic `isaaclab3`/Kit launch; render-only hold after one reset.
- Schema/interface changes: No production or persisted policy/controller schema change. Preview is explicitly acceptance-ineligible and runtime-disabled pending visual approval.
- Tests/evidence: `14` focused tests, compilation, and diff checks passed. A real Isaac single-module stage authoring smoke resolved both pad prims and exited normally with no post-reset physics step.
- Handoff/open question: User must inspect `python3 scripts/order8_side_proxy_pad_gui.py`. Only after placement approval should the same specs replace the legacy tip pad in a minimum contact slice; current production remains actual compliant authored meshes.

#### 2026-07-18 (low-load replay black-viewport correction)
- Scope/result: Correct normal-mesh black output in the synchronized 1 kg Kit replay. Default rendering is now `balanced`, a replay-only Dome Light supplements the existing stage light, and rendering mode remains explicitly selectable.
- Verification: Focused `195` tests plus compilation/diff checks passed; a real Kit desktop capture visibly verified three normal Holon meshes, object, support, floor, and the separately enabled collision overlay. PhysX motion remained `7.244 deg` with zero joint error, and Kit shut down normally.
- Boundary/handoff: Visual-only correction; no physical/controller/acceptance change. Rerun the unchanged wrapper command and optionally disable the green collision overlay in the GUI.

#### 2026-07-18 (low-load replay PhysX/Fabric synchronization correction)
- Scope: Correct the user's verified no-motion failure in the cached 1 kg GUI replay without rerunning slow contact dynamics.
- Files changed: `amsrr/simulation/order8_isaac_runtime.py`, `scripts/p4_control_holon_spawn_probe.py`, `scripts/order8_current_grasp_gui.py`, `scripts/order8_lift_symptom_replay_gui.py`, focused tests, design supplement, and WORKLOG.
- Upstream dependencies: Existing hash-bound v359 1 kg state trace, current generated Holon USD, PhysX articulation tensor view, and Kit/Fabric rendering path.
- Implemented: Optional diagnostic one-step synchronization with exact-state reapply; matching drive targets; zero synchronization velocities/efforts/wrenches; disabled gravity, graph constraints, contact reports, object/environment collision, and self-collision; independent PhysX DOF readback and live `physx_dock_delta_max` telemetry. Authored cross-module collision geometry remains enabled.
- Schema/interface changes: No persisted schema/production interface change; one additive acceptance-ineligible CLI option and report fields.
- Tests/evidence: Focused `197` tests, compilation, and diff check passed. A real headless run independently measured `7.244 deg` PhysX Dock motion versus `7.247 deg` recorded with `0 rad` maximum error.
- Not implemented/limitation: Synchronization replay does not reproduce contact forces/dynamics and cannot support acceptance. Headless verification does not replace user-side viewport confirmation.
- Handoff: Rerun `python3 scripts/order8_lift_symptom_replay_gui.py`. The terminal's PhysX displacement must grow to about `7.24 deg`; if it does but the viewport remains static, isolate Kit/Fabric presentation rather than changing control or rerunning contact physics.

#### 2026-07-18 (low-load 1 kg lift-symptom GUI replay)
- Scope: Restore/lock the visual diagnostic to the unchanged `1.0 kg` payload and provide a GUI observation route that does not combine slow three-module mesh-contact physics with RTX rendering.
- Files changed: `scripts/order8_lift_symptom_replay_gui.py`, its focused unit test, design supplement, and WORKLOG. The generated 1 kg state trace is a diagnostic artifact.
- Upstream dependencies: Corrected v359 source report, hash-bound `order8_diagnostic_state_trace_v1`, existing trace capture/replay runtime, current generated Holon USD, and `isaaclab3`.
- Implemented: Fail-closed 1 kg source/trace validation; one-time headless physical capture; no-physics Kit replay; performance rendering; default warning suppression; direct process replacement; capture/print/refresh controls.
- Schema/interface changes: None. No policy, QPID, actuator, acceptance, persisted config, or production mass value changed.
- Tests/evidence: Focused suite `8` and expanded Order 8 suite `388` passed. Real no-viewer replay bound all scene identities and returned zero joint/root/object write error over a 395-frame, 15.76-second 1 kg trace.
- Not implemented/limitation: A kinematic replay cannot validate force, friction, contact stability, controller feasibility, or natural-contact acceptance. Final physical conclusions remain those of the headless v359 report; the user must still make the requested visual judgement in Kit.
- Handoff: Run `python3 scripts/order8_lift_symptom_replay_gui.py`; do not add `--refresh-trace` unless the underlying physical implementation changes.

#### 2026-07-17 (constrained Dock-drive gain tuning; production 200/5 retained)
- Scope: Tune Dock implicit-drive `Kp/Kd` in the fastest representative single-module bench, then validate only selected candidates in the unchanged free-object contact fixture.
- Files added/changed: reusable tuning/evaluation module, one-launch real-Isaac sweep CLI, focused tests, Dock actuator provenance, design supplement, and WORKLOG. No policy, QPID, contact gate, or production gain value changed.
- Result: The contact-free bench minimum was `800/4.375`, but `800/4.375`, `650/4.375`, `300/5`, and `200/8` all reached unchanged slip gates and none completed verified lift-off with margin. Configured `200/5` also reaches the cumulative gate, so it is retained as the last validated baseline rather than claimed as a globally optimal or passing contact gain. The bench report is diagnostic-only and explicitly requires contact validation.
- Evidence: `dock_joint_drive_tuning_v1.json` plus v336-v340 under `artifacts/p4_full/order8_natural_contact/`; all gain candidates respected AK40-10 torque/current/speed constraints and none established lift-off or Order 8 acceptance.
- Tests: focused suite `218` passed; expanded Order 8 suite `333` passed; compilation, YAML reload, and diff checks passed.
- Handoff: End gain tuning here. Resume at the articulated anchor-hold method decision in the latest global Order 8 entry, using the v314 fixture for the shortest discriminator before any downstream/full-environment run.

#### 2026-07-17 (lift-acceleration intent; paused at velocity governor)
- Scope: Implement and verify the approved LIFT-only known-payload inertial bias without changing QPID contact awareness, local-joint control, raw-contact privilege, or safety limits.
- Files changed in this increment: `amsrr/schemas/order8.py`, `configs/training/order8_natural_contact.yaml`, `amsrr/simulation/order8_isaac_runtime.py`, `amsrr/simulation/order8_natural_contact.py`, `scripts/order8_contact_force_diagnostic.py`, focused schema/runtime/wrapper tests, design supplement, and WORKLOG.
- Schema/interface changes: approved config/report contract v10; normal `PolicyCommand` schema is unchanged because the existing centroidal residual-wrench field is used.
- Verification: expanded Order 8 suite `333` passed; v330 proved one-to-one schedule/PolicyCommand application and zero non-LIFT leakage, but safe-held at `34.07 mm/s` slip with `0.679 mm` OBB lift clearance.
- Not completed: verified lift-off, bias-removal runtime evidence, proxy promotion, transport, place, release, settle, complete-environment evidence, or Order 8/P4-full acceptance.
- Handoff: Do not sweep a narrow constant acceleration or relax gates. Resume only after approval of the non-privileged object/Dock relative-velocity governor described in the global entry above; use the v314 fixture for one shortest early-lift test before any downstream phase.

#### 2026-07-17 (synchronized lift progress; paused before acceleration bias)
- Scope: Implement the approved shared commanded-progress payload feed-forward floor, version its evidence, verify it with the minimum proxy fixture, and stop before any additional method change.
- Files changed: Order 8 schema/config, Isaac runtime, wrapper validator, fast diagnostic CLI, focused tests, design supplement, and WORKLOG.
- Upstream dependencies: v9 natural-contact contract, v8 load-limited positional preload, aggregate centroidal external-wrench observer, shared LIFT motion-entry progress, payload QPID coupling, saved v314 fixture, diagnostic proxy, and unchanged safety/actuator gates.
- Implemented: `max(commanded_progress, observed_load, measured_rise, lift_off)` feed-forward target with bounded slew; shared progress telemetry and acceptance validation; diagnostic duration override; v9/v18 version updates.
- Not implemented: LIFT acceleration/CoM wrench bias, production duration change from `1.0 s`, proxy promotion, safe lift-off, transport, place, release, settle, or Order 8/P4-full acceptance.
- Schema/interface changes: Approved persisted Order 8 config/report v9 update. Normal `PolicyCommand`, QPID, and actuator interfaces remain unchanged.
- Tests passed: focused suite `222`; `py_compile` and `git diff --check` passed.
- Real evidence: `synchronized_lift_proxy_mu10_v326_16s.json`, `synchronized_lift_0p5s_proxy_mu10_v327_15s.json`, `synchronized_lift_0p75s_proxy_mu10_v328_15s.json`, and `synchronized_lift_0p72s_proxy_mu10_v329_15s.json` under `artifacts/p4_full/order8_natural_contact/diagnostics/`.
- Downstream impact: The observer deadlock is fixed, but gravity compensation alone does not create a robust takeoff transient inside both unchanged slip gates. Duration tuning is closed; the next work must supply an explicit bounded upward acceleration intent.
- Handoff notes: No simulator process is running. After user approval, prefer a `1.0 N` lift-only CoM wrench bias for the `1 kg` object (`a_lift=1.0 m/s^2`), ramp it with the retained production `1.0 s` shared progress, remove it after verified lift-off/steady upward motion, and run one saved-fixture diagnostic before any further parameter search.
- Open questions: Exact bias removal gate and whether the bias magnitude should be fixed as acceleration times known payload mass or exposed as a bounded planner/config value.

#### 2026-07-17 (finite-area proxy fault isolation; paused at lift schedule)
- Scope: Implement the user-approved thin finite-area selected-surface proxy as an acceptance-ineligible diagnostic, calibrate contact representation with the shortest saved-fixture real-Isaac runs, and stop if the next failure requires a method change.
- Files changed: `amsrr/simulation/order8_isaac_runtime.py`, `scripts/p4_control_holon_spawn_probe.py`, `scripts/order8_contact_force_diagnostic.py`, focused runtime/wrapper tests, design supplement, and WORKLOG.
- Upstream dependencies: sampled selected Dock mesh outer faces/connect frames, v8 load-limited positional preload, saved v314 near-contact fixture, free-object audit, aggregate load observer, QPID, AK40-10 actuator envelope, and unchanged privileged contact monitor.
- Implemented: diagnostic-only `30 x 30 x 2 mm` rigid-body-child proxy pads; explicit material/collision/retained-mesh/exclusivity auditing; CLI/report plumbing; fail-closed face-fit validation; no-mislabel metadata and tests.
- Not implemented: persisted proxy/config promotion, any normal actor/QPID raw-contact input, lift feed-forward scheduling change, lift-off, transport, place, release, settle, complete-environment rerun, or Order 8/P4-full acceptance.
- Schema/interface changes: None to persisted schemas or normal interfaces; diagnostic CLI/report additions only.
- Tests passed: focused Order 8 schema/runtime/measurement/wrapper set `221`; final compilation and diff checks are recorded in the matching global entry.
- Real evidence: v321 USD audit passed. v325 at diagnostic `mu=10.0` kept both proxy contacts, all actuator/QP/penetration/environment gates valid, but the object remained supported while observer/feed-forward plateaued near `0.795/0.792`; cumulative slip then reached `10.077/7.789 mm` and safe-held.
- Downstream impact: Contact area/friction has been isolated from the remaining problem. No further friction sweep is justified. The next implementation must resolve the observed-load/full-support deadlock before downstream phases are exercised.
- Handoff notes: No Isaac process is running. Resume only after explicit approval of a bounded commanded lift-progress floor (or another stated load-transfer rule). Then use the v314 fixture and proxy for one q_close-through-early-lift run; do not weaken safety limits or promote the proxy into acceptance implicitly.
- Open questions: Whether payload feed-forward should use `max(observed_transfer, commanded_lift_progress)` and ramp to full alongside the upward target; whether a successful diagnostic proxy should later become a separately versioned production contact representation.

#### 2026-07-17 (diagnostic pitch hold and previous-target closure integration)
- Scope: Apply the user-approved diagnostic simplification, rerun the same raised-support natural-contact fixture with a `30 s` ceiling, and determine whether the prior yaw reversal and q_close failure persist.
- Files changed: `amsrr/simulation/order8_isaac_runtime.py`, its focused unit tests, design supplement, and WORKLOG.
- Upstream dependencies: v307 near-contact/opening fixture, all physical Dock DOFs, one-shot whole-structure closure direction, local Dock drives, QPID, CoM-only admittance, authored meshes, and unchanged safety gates.
- Implemented: Diagnostic-only absolute pitch target hold; zero pitch velocity/torque-bias command; previous-position-target velocity integration for closure/release; report evidence for fixed pitch IDs/targets/error; focused regression tests.
- Schema/interface changes: None outside additive diagnostic report keys.
- Tests passed: Runtime unit suite `151`; diagnostic/GUI/evidence/joint-controller companion suite `59`; compilation and diff checks passed.
- Real evidence: `/tmp/order8_pitch_hold_integrated_target_v308_30s.json` and `artifacts/p4_full/order8_natural_contact/diagnostics/pitch_hold_integrated_target_v308_30s_state_trace.json`. Two selected contacts, `0.10 s` q_close arrest dwell, full force ramp, grasp dwell, and transition to `LIFT` passed. The former multi-degree yaw reversal did not recur.
- Not completed: The run stopped safely at `12.83 s` when selected-contact cumulative slip reached `10.156 mm`; lift clearance, transport, place, release, settle, and Order 8 acceptance remain unverified.
- Handoff notes: Treat pitch hold as a temporary command mask, never as a structural joint lock or acceptance condition. Resume with a short q_close-to-lift load-transfer/slip diagnostic; do not return to mesh-following closure work and do not relax the existing slip threshold without explicit approval.

#### 2026-07-16 (baseline state-trace capture and wall-clock GUI replay)
- Scope: Let the user inspect the current v246 grasp motion near real time before replacing its receding differential-IK closure.  This is diagnostic visualization only.
- Files changed: `amsrr/simulation/order8_state_trace.py`, `amsrr/simulation/order8_isaac_runtime.py`, `scripts/p4_control_holon_spawn_probe.py`, `scripts/order8_current_grasp_gui.py`, focused state-trace/GUI tests, design supplement, and WORKLOG.
- Upstream dependencies: Current three-module representative morphology, authored Dock mesh/USD lineage, v246 precontact fixture, existing all-Dock articulation/controller state, and the configured `isaaclab3` environment.
- Implemented: Hash-bound full module-root/all-joint/object state capture; atomic trace storage; exact scene/index validation; physics-free Kit replay paced from wall time; late-frame dropping; terminal phase/time display; self-contained micromamba launch; trace reuse/refresh controls.
- Not implemented: The approved final collision-aware `q_grasp` solver/trajectory, stable grasp, lift, transport, place, release, settle, GUI visual judgement, or Order 8 acceptance.
- Schema/interface changes: Additive diagnostic `order8_diagnostic_state_trace_v1` and probe flags only.  Normal policy/QPID/actuator/acceptance interfaces are unchanged.
- Tests passed: Focused combined suite `140`; new trace/GUI suite `5`; real v246 capture completed; real headless `20x` replay passed with source duration `10.45 s` and wall duration `0.5256 s`; compilation/help/dry-run/hash/diff checks passed.
- Follow-up verification: After the visibility/telemetry change, the focused suite still passed `140`; real headless `20x` replay passed in `0.5380 s`, recorded maximum Dock delta `0.083402 rad`, and reported `0.0` maximum joint, root-position, and object-position write error.
- Handoff notes: Run `python3 scripts/order8_current_grasp_gui.py` and inspect the closing motion.  Do not use `--realtime-playback`; that sleeps after slow physics and cannot accelerate it.  The default trace already exists locally, so normal use opens only the replay.  After the observation, resume at explicit `q_grasp` plus bounded `q_open -> q_precontact` implementation, not another long v247 run.
- Open questions: Awaiting only the user's visual observation.  The current replay intentionally ends at the physical v246 safety hold (`30.287 N` selected peak), making the defect visible rather than hiding it.

#### 2026-07-13 (implementation contract frozen; in progress)
- Scope: Implement a separately accepted real-Isaac natural-contact substrate for deterministic/controlled grasp, lift, `0.200 m` transport, place, release, and settle of the baseline free box.  This entry freezes the contract only and does not claim implementation completion.
- Files changed: `for_codex/AMSRR_design_modification_by_codex.md`; `for_codex/WORKLOG.md`.
- Upstream dependencies: v0.4 contact/reward/backend/P4/archive contracts; completed Orders 1-7 and Order 2.5; actual Holon Dock collision meshes; current PhysicalModel, QPID/local-servo/actuator bridge, contact candidates, trajectory runtime, and archive provenance.
- Contract recorded: Free `1.0 kg`, `0.30 x 0.20 x 0.15 m` box; object/floor friction `0.6/~0.8`; at least two selected contacts on distinct Dock links; `0.5 N`, `0.25 s`, approximately `11 N/contact`, `30 N`, `5 N m`, `2 mm`, `0.02 m/s`, `10 mm`, and `0.05 s` contact/load/slip/break gates; `100 mm` lift, `200 mm` transport, `0.10 s` release dwell, `50 mm` retreat, and continuous `0.05 m/s` / `0.10 rad/s` / `1.0 s` post-release settle gates.
- Dock-joint acceptance invariant: Every Dock joint stays physically articulated, observed, and commandable; no structural lock/fixed substitution is accepted.  Any debug command mask is non-structural, explicitly logged, and disabled for acceptance.  Complete actuator mapping and controller/QP feasibility remain hard gates.
- Implemented: Documentation-only Order 8 scope, thresholds, policy/controller privilege boundary, phase-sensitive drop/contact semantics, evidence provenance, and no-mislabeling contract.
- Not implemented: Order 8 schemas/config, Isaac environment/runtime, contact classifier, controller/policy integration, archive/result validator, unit/fake/real tests, GUI path, real-Isaac rollout, or acceptance artifact.
- Schema/interface changes: None.  Future additions are expected to be additive/versioned and must not add contact/internal-wrench commands to normal QPID input.
- Downstream impact: Supplies a handoff-ready boundary for Order 8 implementation; accepted evidence will be an input to Order 9 learning/TaskSpec delivery and Order 10 aggregation, but this documentation entry is not such evidence.
- Tests added: None.
- Tests passed: None claimed.  Only documentation diff/whitespace validation is planned for this edit.
- Handoff notes: Start schema/config/evidence validation before simulator code.  Preserve the free-object constraint, actual Dock collision surfaces, selected-link identity, all Dock articulation DOFs, privileged raw-contact boundary, exact thresholds, phase trace, and hash provenance.  Reject debug-mask-enabled or structural-joint-lock runs rather than labelling them natural contact.
- Open questions: The sufficient-fall detector and meaningful-contact noise floor must be made explicit and configuration-backed during implementation; they may not reinterpret selected contact existence (`>=0.5 N`) or the phase-aware floor-contact rule.

### Agent B/G/J/K/L: Orders 6-7 Dynamic Dock Constraint and Round Trip

#### 2026-07-13 (selected-pair fallback acceptance closeout)
- Scope: Close the independently gated real-Isaac `attach_only` and `roundtrip` paths using the user-approved, exactly scoped collision-filter fallback while preserving a separate fail-closed physical-funnel mode.
- Source commits: Order 5 approach/control refinement `51bc152`; Orders 6-7 runtime, fallback default, live progress, and tests `0180504`.
- Files changed: `amsrr/simulation/dynamic_assembly.py`, `dynamic_contact_evidence.py`, `dynamic_dock_constraint.py`, `isaac_usd_collision.py`; `amsrr/controllers/controller_handover.py`; dynamic integration in `scripts/p4_control_holon_spawn_probe.py`; `scripts/order5_7_dynamic_assembly.py`; dynamic/joint configuration, hashing support, focused tests, design supplement, and WORKLOG.
- Upstream dependencies: Completed Order 5 bridge/planner/executor; exact connect-frame relation; current URDF/PhysicalModel; QPID/Isaac bridge; follower-subtree estimator/unload gate; user-approved pair-only collision-filter fallback.
- Implemented: Explicit `physical_funnel_contact_v1` and `selected_pair_collision_filter_fallback_v1` evidence contracts; prealignment-time exact-pair filter apply/verification; strict contactless final seated pose/twist/dwell gate for fallback; exact FixedJoint identity; complete actuator-domain handover; measured unload, separation, delayed filter removal, ownership verification, global post-unfilter no-recontact checks, and resettable continuous post-release stability. Only the selected pitch/yaw Dock rigid-body pair is filtered, never either Dock against the environment or other bodies.
- Schema/interface changes: Additive versioned internal Orders 5-7 config/result/evidence fields only. Existing persisted morphology/policy/checkpoint/TaskSpec/QPID contracts are unchanged.
- Accepted evidence: Seed-2 attach-only and roundtrip fallback reports both passed with empty validation failures. Attach retained the strict `3 mm / 2 mm / 0.5 deg / 0.01 m/s / 0.03 rad/s` gates. Roundtrip removed the filter after 954 steps at `0.200871 m` gap and `0.030093 m` selected-body clearance, recorded zero post-unfilter selected contact, and completed 200 continuous stable samples (`1 s`).
- Tests passed: Final focused assembly/controller/simulation/hashing suite `167 passed in 1.99 s`; formal real-Isaac attach-only and roundtrip commands both exited zero. Python compilation and whitespace checks pass at closeout.
- Not implemented/accepted: Physical funnel-contact attach/roundtrip, arbitrary preassembled multi-module component spawning, intra-component self-collision validation, generic graph-state detach integration, object contact, learned `pi_A`, TaskSpec delivery, and P4-full acceptance.
- Handoff notes: Treat the two fallback JSON reports as Orders 6-7 evidence only under their explicit mode. Never relabel them as physical funnel contact. Preserve authored connect frames and strict fix gates. Current dynamic assembly numerics are solver `8/8`, Dock drive `200/2`, and explicit limits `4.1 Nm / 3 rad/s`; historical Order 3 damping-1 artifacts are stale for this runtime.
- Open questions: Physical funnel collision/contact fidelity remains optional follow-up work. No method-level blocker remains for an explicitly requested Order 8 start under the accepted fallback boundary.

#### 2026-07-13 (initial implementation; superseded handoff)
- Scope: Implement independently gated real-Isaac module attach and attach/detach execution after the completed Order 5 bridge, without claiming success when the authored contact surface and connect frame disagree.
- Files changed: `amsrr/simulation/dynamic_assembly.py`, `dynamic_dock_constraint.py`, `amsrr/controllers/controller_handover.py`, controller exports, `scripts/order5_7_dynamic_assembly.py`, dynamic integration in `scripts/p4_control_holon_spawn_probe.py`, `configs/training/order5_7_dynamic_assembly.yaml`, focused controller/simulation/assembly tests, design supplement, and WORKLOG.
- Upstream dependencies: Order 5 component commands/planner/executor; current Holon URDF/USD and PhysicalModel; `centroidal_local_joint_v2`; QPID/Isaac bridge; `FACE_TO_FACE_DOCK_RELATION`; follower-subtree detach estimator and unload gate.
- Implemented: `attach_only|roundtrip` report separation and gate-specific artifact paths; CLI seed/sampling/effective-config/backend/model provenance; forced isolated `Convex Decomposition` conversion; two single-module Articulations; floor contact/low-speed dwell with explicit zero Dock commands; preflight vectoring and measured hover acquisition; URDF-origin-aware per-collider broad phase; bounded staging; finite/unsaturated raw selected-patch point/normal/force/penetration evidence; external FixedJoint exact local frames, `JointEnabled`, `excludeFromArticulation`, identity, verified filter apply/remove lifecycle; complete actuator-domain controller command blending in both directions; raw all-external follower contact monitoring; unload gate; current selected-Dock-body AABB clearance; delayed unfilter; zero-recontact accumulation; continuous attached/post-release pose/attitude/twist/Dock-joint/clearance dwell; config/graph/model/resolved-URDF/USD-bundle/backend hashes and exact ordered phases.
- Not implemented/accepted: No current-`Convex Decomposition` physical attach pass, detach pass, arbitrary preassembled multi-module component representation, intra-component self-collision validation, generic `AssemblyStep(detach)`/`ConstructionState` split integration, object contact, natural-contact grasping, learned `pi_A`, TaskSpec delivery, or P4-full acceptance. External FixedJoint reaction is not used as a release input; privileged reaction comparison remains optional evidence because the runtime API has not been established.
- Fast verification: Focused Order 5-7/controller suite passed `71`; both CLI gates produce distinct hash-bound dry commands containing `--force-convert`; Python compile and whitespace checks pass. Final full regression passed `605` with `1` skipped in `300.24 s`.
- Real-Isaac diagnostic: Under the former shared `Convex Hull` lineage, seed 2 reached axial approach with zero operational QP infeasibility after setting staging to `0.10 m/s`. At first physical contact, force was `3.443 N`, patch separation `0.000062 m`, and connect-frame translation `[0.046536, -0.003481, 0.002785] m`; the invalid selected-surface evidence correctly entered safe hold and never enabled the constraint. This is non-authoritative for the new isolated `Convex Decomposition` path.
- Blocker/handoff: Historical and superseded by the closeout above. Do not move/correct the authored connect frames based on this diagnostic. The pair-only fallback now has separate accepted attach-only and roundtrip reports; the physical-funnel path remains unaccepted.

#### 2026-07-13 (explicit Dock collision approximation repair)
- Scope: Verify and repair the generated Holon Dock collider approximation before the Orders 6-7 asset is spawned; no AssemblyControlBridge/contact-state behavior was changed in this subtask.
- Files changed: Added `amsrr/simulation/isaac_usd_collision.py`; added the post-conversion hook and additive evidence fields in `scripts/p4_control_holon_spawn_probe.py`; added `tests/unit/simulation/test_isaac_usd_collision.py`.
- Upstream dependencies: Current isolated dynamic-assembly USD bundle, local Isaac Lab/Isaac Sim URDF importer 3.0, current explicit URDF collision meshes, and the configured `Convex Decomposition` request.
- Root cause: Local importer code `urdf_usd_converter/_impl/geometry.py::apply_physics_collision` unconditionally authors `physics:approximation="convexHull"` for explicit URDF `<collision>` meshes. `URDFImporterConfig.collision_type` is consumed only when `collision_from_visuals` is enabled, so the prior config value was recorded but did not affect Holon's explicit Dock collision meshes.
- Implemented: After every forced dynamic-assembly conversion and before spawn, locate only local generated layers that own Holon pitch/yaw Dock mesh collision APIs, author `convexDecomposition`, save them, reopen the root asset, traverse instance proxies, and fail closed unless every composed Dock collider resolves to that token. The report records requested token, pre-repair tokens, two unique authored source prims, four composed Dock collider paths, and verification status; the complete post-repair USD directory remains hash-bound by the existing bundle provenance.
- Schema/interface changes: No persisted morphology/policy/controller schema changed. The real-Isaac probe gains additive free-form collision-approximation evidence fields only.
- Tests/commands: Host focused test passed `5` with `2` Isaac-only tests skipped; the same file under the `isaaclab3` Python passed all `7`; Python compilation and focused whitespace checks passed. Direct inspection of the current generated package confirmed two source opinions were changed from `convexHull` and all four composed Dock collider paths resolve to `convexDecomposition`.
- Assumptions/limitations: The authored USD token requests PhysX convex-decomposition cooking; it does not prove the cooked hull count or that the pitch funnel cavity is sufficiently preserved. Physical funnel validity still requires the real insertion/contact trajectory. A selected-pair collision-filter fallback, if used, must be an explicit separately reported mode and must not claim physical funnel-contact closure.
- Handoff: Physical-funnel validation still requires the approximation verification/token/count fields. This historical sequencing applies within that mode only; the separately contracted pair-only fallback subsequently passed its own attach-only gate before its own roundtrip gate.

### Agent G/I/J: Order 5 AssemblyControlBridge and Component Motion

#### 2026-07-13
- Scope: Implement the deterministic `pi_A` controller-facing bridge, collision-aware staging planner, and stateful attach-sequence runtime without claiming physical attachment.
- Files changed: `amsrr/assembly/assembly_control_bridge.py`, `assembly_motion_planner.py`, `closed_loop_executor.py`, `control_handoff.py`, assembly exports and focused tests, `configs/training/order5_7_dynamic_assembly.yaml`, design supplement, and WORKLOG.
- Upstream dependencies: v0.4 Section 17.7; `AssemblyStep`, `ConstructionState`, and `ControlHandoffRequest`; canonical connect-frame geometry; `centroidal_local_joint_v2`; QPID/local joint/Isaac bridge authority.
- Implemented: Leader/follower component partitioning; component-scoped PolicyCommands; staging +X and face-to-face targets; axial/transverse/attitude/connect-twist metrics; contact evidence/dwell/force/penetration gates; canonical neutral Dock joint targets with bounded correction; exact constraint intent; direct/bounded-via SE(3) planner; four-step stateful executor compatibility; retry/abort safe hold.
- Not implemented: Isaac contact observation, dynamic FixedJoint activation, physical graph/controller handover, constraint removal, detach, multi-module preassembled-component physical representation, object contact, or P4-full acceptance.
- Schema/interface changes: Additive versioned Order 5 contracts and one additive `ControlHandoffManager` conversion method; no existing persisted schema/checkpoint/controller field changed.
- Downstream impact: Order 6 supplies real component/contact/constraint observations and consumes exact constraint intent. Order 7 supplies unload/release. Later grasping must coordinate all upstream morphology joints, not only a contacting Dock link's adjacent joint.
- Tests added: Bridge/session/gate/constraint-intent, handoff conversion, direct/via/no-path planning, four-step closed-loop integration, collision-oracle failure.
- Tests passed: Assembly gate `27 passed`; assembly plus controller boundary regression `53 passed`; compileall and whitespace checks passed.
- Handoff notes: Dock joints are articulated morphology joints, not latches. The first physical gate must use two single-module Articulations and external exact-connect-frame constraint; existing fixed-root graph assets are not upstream-joint morphing evidence.
- Open questions: None for Order 5. The fallback thresholds now have Order 6/7 real-Isaac evidence; physical-funnel contact tuning/evidence remains open under its separate contract.

### Agent H/I/K: Order 4 Deterministic Free-Flight pi_H and Trajectory Runtime

#### 2026-07-13
- Scope: Implement and verify the retained deterministic free-flight `pi_H` fallback and policy-agnostic rolling `ContactWrenchTrajectory` executor without claiming learned/contact-aware `pi_H` completion.
- Files changed: Order 4 schemas, trajectory runtime, deterministic planner/context factory, Isaac environment/report gate, config, CLI, focused tests, additive Isaac probe path, design supplement, and WORKLOG.
- Upstream dependencies: v0.4 Sections 19/20/24.5; Orders 1-3 and 2.5; current PhysicalModel/URDF; `HighLevelPolicyContext`; baseline/optional learned `pi_L`; QPID/local joint servo/Isaac bridge.
- Implemented: Hash-bound multiple-waypoint mission; state-dependent settle/takeoff/hover/waypoint/final-hold guards; 2 Hz rolling replanning; explicit relative plan origin and centroidal interpolation; phase/timeout/safe-hold reporting; zero-contact reachability N/A; Dock absolute-zero posture; N=2/N=3/N=8 headless, N=3 20 s endurance, and N=3 Kit GUI validation.
- Not implemented: Learned `pi_H` training/evaluation, contact candidates/assignments/wrenches, simultaneous multi-anchor reachability, object contact, dynamic module attach/detach, full TaskSpec delivery, or P4-full acceptance.
- Schema/interface changes: Additive versioned Order 4-only contracts; no change to existing persisted policy/morphology/checkpoint/controller schemas.
- Downstream impact: Order 8 may extend the deterministic fallback with contact planning and Order 9 may place learned `pi_H` behind the same executor. Order 5 should preserve the controller boundary and keep dynamic assembly evidence separate.
- Tests added: Order 4 schema, executor, planner, safe-hold, low-level handoff, Isaac command/report/tamper, GUI-option coverage.
- Tests passed: Final focused `12`; related `112`; final full regression `530 passed, 1 skipped`; compile/help/diff checks; all recorded real Isaac and Kit gates passed after the documented timeout correction.
- Handoff notes: Use `scripts/order4_free_flight_pi_h.py`. Default execution samples a fresh feasible morphology; `--seed` reproduces it, `--module-count` accepts 2-8, `--endurance` requests 20 s, repeated `--waypoint` overrides the mission, and `--pi-l-checkpoint-path` selects a compatible Order 3 actor. Ignored reports are not required by source.
- Open questions: None for the free-flight slice. Contact-side thresholds and reachability remain Order 8 design work.

### Agent E/F/G/H/I/J/K/L: P4-Full Orders 1-10 Handoff

#### 2026-07-21 (Order 9 pi_L/C0 correction in progress)
- Scope: Correct the learned Order 9 `pi_L` boundary to the approved complete bounded `PolicyCommand` contract, invalidate incompatible residual-baseline artifacts, and regenerate C0 deterministic-teacher data before any C1 retraining.
- Files changed: Order 9 low-level policy/runtime/reference/decoder; BC/PPO/tensor-runtime/rollout/teacher paths; Order 8 teacher hook; collection/rollout/benchmark scripts; focused tests; design supplement; WORKLOG.
- Upstream dependencies: v0.4 `PolicyCommand` ownership, the approved centroidal-only QPID supplement, Order 8 canonical deterministic teacher, current PhysicalModel and `centroidal_local_joint_v2`, deterministic safety/fallback rules, and the approved Order 9 C0--C10 curriculum.
- Implemented: Complete 18-D centroidal action decoding plus source-ID/mask-aware absolute local-joint outputs; `pi_H`-knot-only learned reference; substitution-only deterministic fallback; exact assembled-centroidal/posture teacher relabelling; per-episode production inverse-decoder representability validation; incompatible contract/version rejection; resumable streaming C0 collection.
- Not implemented: No C1 checkpoint has been trained under the corrected contract; C0 is not complete until all 20 bounded-diversity physical episodes, the 14/3/3 dataset, evaluation row, and promotion gate pass. No C2 or later learning result is claimed.
- Schema/interface changes: No unversioned persisted schema field was changed. Order 9 action/tensor/runtime/checkpoint semantic contracts were deliberately versioned from the incompatible baseline-plus-residual interpretation; old artifacts are historical diagnostics only.
- Downstream impact: Every Order 9 `pi_L` BC/PPO checkpoint and rollout must be regenerated. C1 may start only from the corrected C0 dataset. `BaselineLowLevelPolicy` may still execute after learned-path rejection but may not be mixed into a learned command or receive actor credit.
- Tests added: Complete pose/twist/wrench/joint decode, scalar/tensor parity, complete-command runtime/fallback separation, teacher reference correction and action representability, rollout/checkpoint version compatibility.
- Tests passed: Focused `36`; broader Order 9 `134` with `1040` deselected; complete unit suite `1173 passed, 1 skipped` in `127.83 s`; repository whitespace check passed before C0 start.
- Handoff notes: The superseded fixed-nominal run used seeds beginning at 9009; its first corrected real-Isaac episode passed in `871.3 s` with maximum normalized teacher action `0.1213`, but it predates the bounded-diversity/stride/profile contract and is historical diagnostic evidence only. The replacement C0 uses seeds 9009--9028 with immutable condition identities and exact 14/3/3 membership.
- Open questions: None at method level. Final C0 wall time, dataset ID/hash, sample counts, evaluation metrics, and promotion outcome remain pending runtime evidence.

#### 2026-07-12
- Scope: Preserve the agreed P4-full implementation sequence and current completion point for future chats.
- Files changed: design-modification roadmap and WORKLOG handoff records.
- Upstream dependencies: completed Order 0 actuator model; Orders 1-2 random morphology/takeoff; Order 2.5 controller contract; Order 3 morphology-conditioned `pi_L`; v0.4 P4 full acceptance.
- Implemented: Canonical local numbering; completion states; ownership; primary files; per-order goals and acceptance boundaries; explicit Order 4 entry checklist.
- Not implemented: Orders 4-10 code or acceptance evidence.
- Schema/interface changes: None.
- Downstream impact: Order 4 is next. Later chats must keep contact/internal wrench outside normal QPID, preserve deterministic safety/fallbacks, and treat Orders 8, 9, and 10 as distinct gates.
- Tests added: None for documentation.
- Tests passed: Relies on the immediately preceding clean `518 passed, 1 skipped` repository regression and compile/diff checks.
- Handoff notes: Read the top roadmap in `AMSRR_design_modification_by_codex.md`. Ignore older local `Order N` labels unless their P4-control/P4.1/P4.2/P4.3 prefix is stated. Do not assume ignored `artifacts/` exist. Begin with the Order 4 scheduler contract review.
- Open questions: Natural-contact numeric smoke thresholds must be frozen before Order 8; no blocker for Order 4.

### Agent I/K/L: Order 3 Morphology-Conditioned pi_L

#### 2026-07-12 (implementation complete and committed handoff)
- Scope: Complete the approved morphology-conditioned `pi_L` free-flight implementation and representative real-Isaac verification before advancing to Order 4.
- Files changed: versioned Order 3 schemas; graph encoder/recurrent policy; datasets/reward/BC/PPO trainers; morphology pool; rollout conditions; online/takeoff collectors; real-Isaac rollout/pipeline/acceptance; CLI/config; corrected Dock neutral runtime; focused tests.
- Upstream dependencies: Orders 1-2 morphology/takeoff, Order 2.5 `centroidal_local_joint_v2`, current PhysicalModel/URDF, QPID/allocator/Isaac bridge, deterministic safety and fallback contracts.
- Implemented: Source commits `59aba6d` (`pi_L` contracts/encoder/policy/training core) and `629008c` (Isaac rollout/collectors/pipeline/acceptance and Dock-neutral integration); representative BC, stochastic PPO, one PPO update, held-out headless and GUI evaluation; absolute-zero Dock hold and stale-frame rejection.
- Not implemented: Production deterministic `pi_H` scheduler, dynamic `pi_A` execution, runtime constraint create/remove, natural object contact, full TaskSpec training/delivery, or P4 full acceptance.
- Schema/interface changes: New versioned Order 3-only dataset/checkpoint/condition/report contracts; existing MorphologyGraph is unchanged; normal controller remains `centroidal_local_joint_v2` with no contact/internal-wrench targets.
- Downstream impact: Order 4 may consume the deployed Order 3 actor input/target contract. It must replace only the training target source, not actor/QPID authority. Orders 5-10 remain sequential downstream work.
- Tests added: graph invariance/masks; policy decoding/fallback; schema/hash/tamper checks; BC/PPO correctness; curriculum conditions; real-report collectors; behavior replay; pipeline planning; acceptance positive/negative cases; Dock zero-hold and stale-frame regressions.
- Tests passed: focused `166`; full repository `518 passed, 1 skipped`; compilation and whitespace checks passed; representative real-Isaac and Kit GUI reports passed.
- Handoff notes: Local ignored artifacts are optional evidence, not repository dependencies. Regenerate checkpoints from `configs/training/order3_morphology_pi_l.yaml` when needed. The next code task is Order 4 and should reuse the exact actor target semantics demonstrated by Order 3.
- Open questions: Full statistical quota remains part of later P4-full evidence aggregation; no method-level blocker for deterministic Order 4 scheduler design review.

#### 2026-07-12 (learned-policy GUI evaluation)
- Scope: Expose visual observation for a real checkpoint-bound Order 3 evaluation without changing policy/controller semantics.
- Files changed: Order 3 rollout environment, pipeline runner, CLI, focused tests, design supplement, and worklog.
- Upstream dependencies: Existing real-Isaac Order 3 execution and probe visualization options.
- Implemented: Kit viewer propagation; physics-dt-paced playback; configurable post-rollout viewer hold; fail-fast validation for headless misuse; unit coverage from public plan to final probe command.
- Not implemented: Interactive teleoperation during learned execution, viewport keyboard capture, acceptance changes, or an automated GUI test.
- Schema/interface changes: Additive runtime arguments only; no persisted schema or artifact change.
- Downstream impact: The user can inspect checkpoint behavior visually while report validation and safety remain unchanged.
- Tests added: Order 3 GUI command/plan propagation and invalid-option rejection.
- Tests passed: Focused `15`; full Order 3 `117`; compile/help checks passed.
- Handoff notes: Use deterministic `evaluate-learned` with one explicit graph and one selected condition for a single convenient GUI session.
- Open questions: None.

### Agent I/J: Centroidal QPID and Local Joint Servo Contract

#### 2026-07-12 (Order 2.5 implementation)
- Scope: Implement and validate the approved Section 14 migration before Order 3.
- Files changed: versioned policy/controller schemas; policy reference builder; rigid-body model/QPID; actuator mapping/Isaac bridge/probe; baseline policy v2 option; detach estimator/gate; Order 2.5 takeoff config/report gate; tests; design/worklog records.
- Upstream dependencies: approved controller supplement Section 14, Order 0 actuator capabilities, existing rotor/vectoring QP, Order 1 sampler, and Order 2 real Isaac runner.
- Implemented: Legacy-compatible v2 command fields; true centroidal pose/twist; contact bias no-op; rotor/vectoring-only QP reporting; allocator-owned vectoring; deterministic non-vectoring hold and absolute targets; native dock position/velocity/offset torque with continuous-limit clipping; privileged wrench non-leak test; follower-subtree cut estimator; fail-closed unload dwell gate; dedicated real-Isaac v2 evidence.
- Not implemented: New learned v2 checkpoint/training, dynamic `pi_A` latch release, post-release separation execution, natural-contact task completion, or P4 full acceptance.
- Schema/interface changes: Additive and versioned as recorded in the global entry; omitted version fields resolve to `legacy_contact_bias_v1`.
- Downstream impact: Order 3 must explicitly select `centroidal_local_joint_v2`; v1 checkpoints and prior base-`fc` takeoff artifacts cannot be relabelled.
- Tests added: schema/round-trip, reference semantics, centroidal kinematics/control, actuator modes/limits, policy privileged-data isolation, detach estimator/sign/gate, and real-report no-mislabeling coverage.
- Tests passed: final focused 122; full unit/acceptance 395 passed and 1 skipped before the final isolated frame-sign regression; real three-module Isaac v2 takeoff/hover passed with no report failures.
- Handoff notes: Keep vectoring allocator-owned and normal contact/internal wrench outside QPID. Use the estimator only on an independently contact-free follower subtree and keep release fail-closed.
- Open questions: Detach threshold hardware tuning remains future empirical work; no method-level blocker for starting Order 3.

#### 2026-07-12
- Scope: Record the approved normal-operation controller simplification and detach-only internal-wrench boundary before Order 3 learning implementation.
- Files changed: QP/PID controller supplement, design-modification record, and worklog only.
- Upstream dependencies: v0.4 policy/controller responsibility split, current QPID/rigid-body model/bridge contracts, installed joint actuator capabilities, and completed random-morphology Orders 1-2.
- Implemented: Normative documentation for centroidal-control-frame targets; rotor/vectoring-only QP; absolute non-vectoring joint position/velocity targets plus torque bias; local joint servo; contact-wrench privileged reward/safety semantics; follower-subtree detach wrench estimation; legacy contract versioning and no-mislabeling requirements.
- Not implemented: Python schema changes, true centroidal observation/controller changes, local joint target/offset-torque bridge, contact-bias no-op migration, detach estimator, new reward, policy retraining, or revised acceptance runs.
- Schema/interface changes: Proposed and approved at documentation level only; runtime schemas are unchanged in this commit.
- Downstream impact: Agent I/J implementation must precede morphology-conditioned `π_L` training; Agent K/L training/acceptance must bind the new contract version and prevent privileged-contact leakage.
- Tests added: None; documentation-only change.
- Tests passed: Markdown/reference consistency and `git diff --check` only.
- Handoff notes: Do not add normal-operation contact/internal-wrench QP variables. Keep vectoring allocator-owned, non-vectoring joints local-servo-controlled, internal wrench detach-only, and legacy artifacts explicitly versioned.
- Open questions: None at design level.

### Agent I/J: Random Morphology GUI Teleop Verification Utility

#### 2026-07-11
- Scope: Provide a user-operated GUI check of the completed Orders 1-2 path for an explicitly selected module count.
- Files changed: `amsrr/simulation/random_morphology_teleop.py`, `scripts/random_morphology_teleop.py`, `scripts/p4_control_holon_spawn_probe.py`, teleop unit tests, and implementation records.
- Implemented: Fresh-random or reproducible feasible graph sampling; Kit GUI launch; existing floor settle/takeoff/hover; nonblocking TTY input; yaw-relative position and bounded attitude targets; controller/contact fail-closed monitoring; compact exit summary.
- Not implemented: A learned policy, joystick/Kit-viewport keyboard capture, automatic camera following, task/object execution, morphology changes in flight, or any new P4 full acceptance gate.
- Tests passed: 78 focused teleop/takeoff/runner tests before range extension; extended Order 1/2 focus 118 passed; full repository suite 382 passed, 1 skipped; real two-module GUI/TTY `P`/`W`/`J`/`Q` smoke; real 7/8-module headless takeoff/hover/contact smokes.
- Handoff: Run `python3 scripts/random_morphology_teleop.py --module-count N`; keep the launching terminal focused while sending commands. Add `--seed S` to reproduce a graph.

### Agent E/F/I/J/L: P4 Full Orders 1-2 Random Morphology Flight Curriculum

#### 2026-07-11
- Scope: Implement the approved random feasible connected morphology distribution and the floor initialization plus deterministic takeoff-to-hover runtime gate as one ordered work package.
- Files changed: `amsrr/morphology/random_connected.py`, `random_feasible.py`, `amsrr/feasibility/morphology_flight.py`, `amsrr/robot_model/fixed_morphology_urdf.py`, `physical_model_builder.py`, `amsrr/simulation/random_morphology_takeoff.py`, `amsrr/training/random_morphology_takeoff_runner.py`, training config/CLIs, focused tests, design supplement, and worklog.
- Upstream dependencies: Source spec Sections 14-16 and P4-control, `PhysicalModel` dock/collision/rotor data, Order 0 actuator data, dock transform helpers, rigid-body model/QP/QPID, actuator mapping/bridge, and graph-specific URDF/Isaac infrastructure.
- Implemented: Seeded 2-8 module constructive tree proposals; canonical structural deduplication; bounded fail-closed feasibility rejection; PhysicalModel-derived structural checks; URDF/STL coarse collision bounds; separate nominal hover and 15%-margin QPs; floor-contact placement; graph `fc` to URDF `root` frame correction; deterministic settle/ramp/hold scheduler; selective graph-bound dock_mech collision filtering; reset-time and all-cross-module per-step PhysX collision gates; real Isaac graph asset execution; typed per-step archive reconstruction; model/URDF/collision/backend/config provenance binding.
- Not implemented: Learned morphology-conditioned `pi_L`, learned `pi_H`, task-conditioned `DesignOutput` adaptation, dynamic module docking, object contact, or P4 full acceptance.
- Schema/interface changes: No existing persisted schema change. New work-package-local dataclasses/configs and additive probe/runner interfaces only.
- Downstream impact: Order 3 can consume unique, feasible, provenance-bound morphology samples and use the same runner for deterministic curriculum/evaluation rollouts.
- Tests added: Sampler/hash/distribution tests, negative structural/collision/QP tests, collision-floor bounds, scheduler/dry/real-report contracts, fail-closed runner tests, and rotated graph-frame URDF regression.
- Tests passed: Extended focused Order 1/2 set 118 passed; full repository suite 382 passed, 1 skipped; seven final selective-dock/all-cross-module raw-contact real Isaac cases covering every module count 2-8 passed; Python compilation and whitespace checks passed.
- Handoff notes: Use `RandomFeasibleConnectedMorphologyDistribution`, not the raw constructive proposal, for training inputs. Run `scripts/random_morphology_takeoff.py` from the configured `isaaclab3` environment with `--real` for physics evidence. `sampling_metadata` preserves the master seed and accepted proposal seed.
- Open questions: None for this work package. Statistical robustness across a larger held-out morphology cohort belongs to later training/evaluation, not this deterministic smoke completion.

### Agent J/K: Order 9 learned curriculum and physical execution

#### 2026-07-22 (C2 live TensorBoard observability)
- Scope: Make C2 reward, phase, safety, performance, load, and PPO optimization progress visible during execution without changing learning semantics.
- Files changed: live TensorBoard logger; runtime latest-sample accessor; recurrent PPO progress callback; rollout/PPO CLIs; reward-term constant; unit tests; design/work logs.
- Upstream dependencies: C2 `2048 x 16` runtime, tensor reward engine, immutable tensor rollout, one-generation/one-update PPO, existing runtime-load monitor, and promoted C1 checkpoint.
- Implemented: Per-step total/all-term reward logging with global and per-phase current/running means; terminal/QP/load telemetry; per-minibatch PPO metrics; final summaries; train/validation run separation; default stage log directory and diagnostic opt-out.
- Not implemented: No C2 training generation or update, no reward tuning, no dashboard-server process management, and no C3+ parallelism decision.
- Schema/interface changes: None to persisted or controller-facing contracts; additive observer callback/CLI only.
- Tests passed: `35` focused; `39` dependency-complete focused; full `isaaclab3` unit suite `1192 passed, 1 skipped`; real event readback and learned-checkpoint `2 x 2` real-Isaac integration smoke passed.
- Handoff notes: Launch `tensorboard --logdir artifacts/p4_full/order9/stages/c2_pi_l_ppo_fixed_conservative/tensorboard --port 6006`; open `http://localhost:6006`. The diagnostic smoke under `runtime_diagnostics` is excluded from training.
- Open questions: None.

#### 2026-07-22 (C0/C1 completion and promotion)
- Scope: Finish C0/C1 after the approved articulated graph-asset direction, correct tensor policy observations to the C0 contract, run the learned physical gate, and finalize hash-bound C1 promotion.
- Files changed: Order 9 fixed/arbitrary articulated asset generation and manifests; tensor `pi_L`/graph runtime; vectorized Isaac rollout; stage finalization/direct CLI; focused runtime/pipeline/asset tests; design supplement; this worklog. Generated C0/C1 data, checkpoint, evaluation, and promotion artifacts are ignored but hash-bound under `artifacts/p4_full/order9/`.
- Upstream dependencies: promoted C0 teacher data; Order 8 natural-contact physical teacher; complete bounded `PolicyCommand`; PhysicalModel non-fixed-joint identity; articulated graph Dock kinematics; active-knot reference; QPID/QP and copied Isaac environments.
- Implemented: 20-episode C0 promotion; 24-epoch actor-only C1 BC; child-reroot fixed-nominal asset; environment-local graph translations; non-fixed-only joint summaries; scalar-compatible quaternion canonicalization; 4/4 learned smoke; 100/100 formal learned evaluation; typed evaluation report and promoted stage manifest.
- Not implemented: C2 PPO or later curriculum stages, arbitrary-morphology learned validation, learned `pi_H`, learned `pi_D`, joint fine-tuning, held-out C10, or P4-full acceptance.
- Schema/interface changes: No unversioned persisted schema or controller interface change. Corrected tensor runtime versions/metadata only; final actuator authority remains QPID/QP/local servos.
- Downstream impact: C2 may now consume checkpoint SHA-256 `533d23e921a8e047878784a523c8c93124f1ae18fbaa476704df88bee67a37ce` and the promoted C1 manifest. Later copied-environment policy runtimes must preserve the environment-local/non-fixed/canonical-quaternion graph contract.
- Tests added: copied-origin removal without changing controller world state, non-fixed joint count, quaternion-sign canonicalization, invalid-origin rejection, and promotion-evaluation completion metadata.
- Tests passed: focused `30`; full unit `1190 passed, 1 skipped`; acceptance `75`; compileall and whitespace checks.
- Handoff notes: Formal C1 evaluation is 100/100 success with no fallback/safety failure at `746.507 env-step/s`; raw rollout SHA-256 `d27a8029ca8df33f8dc97fb809eccbfc418e0c4b879e4fc16569015b4259adf7`; promoted manifest SHA-256 `91a7090b53f8eae831f76d2dccc62620e2cad358ab95e7e891e51702db19bb46`. The prior phase-1 failure was an asset/observation implementation mismatch, not evidence that the actor or method was invalid.
- Open questions: None for C0/C1. Preserve C2's provisional 2048-environment telemetry requirement and review measured learning behavior before changing later-stage parallelism.

#### 2026-07-22
- Scope: Correct and rerun C1 `pi_L` behavior cloning, then perform the smallest real-Isaac causal checks needed before the 100-episode promotion evaluation.
- Files changed: `amsrr/training/order9_offline_training.py`, C1 curriculum config, focused offline-training test, design supplement, and worklog; generated checkpoint/metrics and ignored diagnostic artifacts under `artifacts/p4_full/order9/stages/c1_pi_l_bc_fixed_nominal/`.
- Upstream dependencies: Current 20-episode C0 teacher dataset, accepted Order 8 deterministic teacher/runtime, PhysicalModel, complete-`PolicyCommand` decoder, fixed-nominal Order 9 asset, QPID/QP, and Isaac contact observations.
- Implemented: Actor-only C1 checkpoint selection, zero C1 critic weight, corrected eight-epoch CUDA retraining, load telemetry, superseded-checkpoint archive, learned-checkpoint physical smoke, and exact-teacher-reference zero-action causal smoke.
- Not implemented: No C1 promotion, C2 PPO, asset kinematic redesign, teacher redesign, or 100-episode learned evaluation was attempted after the structural mismatch became deterministic.
- Schema/interface changes: No persisted or controller-facing interface change; trainer/checkpoint selection metadata only.
- Downstream impact: C1 training checkpoint exists but C1--C10 are blocked until Order 9's structural Dock representation is selected and shown compatible with its teacher.
- Tests added/passed: Actor-selection regression added; focused selection passed `11`; `git diff --check` passed; both causal Isaac diagnostics ended without process/GPU failure, safety terminal, or controller-QP failure.
- Handoff notes: The fixed asset cannot reach phase-1 object contact even with exact teacher references. Do not spend the 100-episode promotion budget or tune the actor against this runtime. Preserve the current artifacts as diagnostics and first approve either an articulated Order 9 runtime consistent with Order 8 (recommended) or a new rigid-asset-compatible teacher/data lineage.
- Open questions: Which structural representation is authoritative for Order 9 training: Order 8-compatible articulated graph execution, or rigid module-root assets with a newly defined teacher?

### Agent J: P4.3 learned pi_L Isaac execution boundary

#### 2026-07-10
- Scope: Add an optional learned low-level policy to the existing P4.2 Isaac path without changing controller/QP/actuator ownership or the default deterministic rollout.
- Files changed: `amsrr/simulation/isaac_lab_backend.py`, `amsrr/simulation/p4_2_isaac_env.py`, `scripts/p4_control_holon_spawn_probe.py`, `amsrr/policies/learned_low_level_policy.py`, and related tests.
- Upstream dependencies: P4.2 graph-specific probe, dynamic controller knot, deterministic P4.2 `PolicyCommand`, `BaselineLowLevelPolicy`, QPID controller, and actuator bridge.
- Implemented: Checkpoint loading with load-failure fallback; source-trajectory feature context; learned twist/body-position/residual subset overlay; preservation of the P4.2 controller knot and all non-learned command fields; configurable 0.10 trust-region blend; per-step learned/fallback/non-zero-overlay metrics.
- Not implemented: Learned actuator commands, learned QP/safety replacement, online `pi_H`/`pi_D`, or natural-contact control.
- Schema/interface changes: Optional P4.2 checkpoint/blend arguments only; default P4.2 behavior remains deterministic.
- Tests passed: Included in the 94 focused and 259-pass full suites. Final held-out real-Isaac learned rollout passed with 680 non-zero learned overlays and zero safety terminals.
- Handoff notes: Acceptance binds the checkpoint and online archive hashes and cross-checks episode-level metrics. Keep the command-field and controller-knot separation in future fine-tuning.

### Agent K: P4.3 minimum learning bootstrap and artifacts

#### 2026-07-10
- Scope: Implement v0.4 P4.3a-d in order: datasets/reward, `pi_L`, `pi_H`, then outcome-conditioned `pi_D`.
- Files changed: P4.3 schemas, reward/dataset/runner/training/archive modules, learned policy modules, training config/CLIs, and nearby tests listed in the Global Worklog.
- Upstream dependencies: P2/P2.5 candidates/checkpoint, hard feasibility, P3 morphology, P4.2 deterministic real-Isaac archives, contact/trajectory schemas, controller status, and TaskSpec goal semantics.
- Implemented: 24-episode task-disjoint real-Isaac dataset; causal reward and return-preserving stride; bounded `pi_L` checkpoint/metrics/reward curve; `pi_H` teacher imitation and feasible decode; P2-initialized `pi_D` outcome regression/ranking; hashes/fallback metadata; acceptance-gated summary archive.
- Not implemented: P4.3e joint fine-tuning, RL, production learned-policy claim, online `pi_H`/`pi_D`, natural-contact grasp, or P4 full completion.
- Schema/interface changes: Additive `p4_3_dataset_v1`; no breaking upstream schema change.
- Tests passed: 94 focused; full suite 259 passed / 1 skipped; all real Isaac and final artifact commands exited 0.
- Handoff notes: Preserve task-disjoint splits and within-task candidate outcomes. Do not collapse P4.2 bounded carry into full TaskSpec success.

### Agent L: P4.3 artifact completeness and no-mislabeling gate

#### 2026-07-10
- Scope: Fail closed unless real data, meaningful head-specific training evidence, deterministic fallbacks, and checkpoint-bound online `pi_L` safety evidence are all present.
- Files changed: `amsrr/acceptance/p4_3_acceptance.py`, `amsrr/acceptance/__init__.py`, `tests/acceptance/test_p4_3_acceptance.py`, and summary-archive integration.
- Upstream dependencies: `P4_3DatasetManifest`, P4.3 head checkpoint contracts, hashes, online EpisodeArchive fields, fallback metadata, and no-mislabeling flags.
- Implemented: Dataset shard hash/count/split/mask/provenance checks; head-specific checkpoint metadata/config hashes; `pi_H` zero-fallback decode gate; `pi_D` within-task/validation ranking gate; held-out online `pi_L` archive/checkpoint/aggregate/safety cross-check; command-level overlay/knot recomputation; sanitized self-validating acceptance-only summary creation.
- Not implemented: P4 full acceptance or natural-contact success acceptance.
- Schema/interface changes: No persisted schema change; new P4.3 report type only.
- Tests passed: Acceptance positive and negative/tamper cases are included in the 94 focused and full 259-pass suites. Final report has no failures.
- Handoff notes: A file-presence-only gate is insufficient. Future learned heads must add equivalent online evidence before their deployment flags can become true.

### P4.2 Implementation: Isaac Deterministic Grasp-Carry Rollout

#### 2026-07-10
- Scope: Hand-executable GUI observation path for the real P4.2 deterministic rollout.
- Files changed:
  - `amsrr/simulation/isaac_lab_backend.py`
  - `amsrr/simulation/p4_2_isaac_env.py`
  - `scripts/p4_2_deterministic_rollout.py`
  - `scripts/run_p4_2_gui.sh`
  - `tests/unit/simulation/test_p4_2_isaac_env.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Upstream dependencies: Existing P4.2 real rollout command, Isaac probe GUI controls, and the P4.2 link-backed attach gate.
- Implemented: Optional `--viewer kit`, real-time playback, and post-rollout hold propagate from the P4.2 CLI through `P4_2IsaacEnv` / `IsaacLabBackend` to the Isaac probe. `run_p4_2_gui.sh` supplies those controls with a 20-second default hold.
- Not implemented: Any change to P4.2 state-machine behavior, geometry, contact model, attach threshold, controller, payload coupling, natural grasp, or acceptance logic.
- Schema/interface changes: No persisted schemas. Optional observation parameters only; defaults preserve headless automation.
- Downstream impact: Users can inspect the exact P4.2 path that is archived and accepted, rather than a separate graphical smoke. The GUI run remains `kinematic_payload_coupled_attach_v1`, not natural-contact-grasp validation.
- Tests added: Backend command and environment propagation assertions for the three observation controls.
- Tests passed:
  - `py_compile` and `bash -n` passed.
  - Targeted P4.2 env/runner/acceptance plus shared Isaac command tests: 21 passed.
  - `scripts/run_p4_2_gui.sh --help` passed in the `isaaclab3` environment without launching Isaac Lab.
  - Headless real Isaac P4.2 rollout passed with `completion_passed=true` and `real_isaac_rollout_passed=true`.
- Handoff notes: Run `scripts/run_p4_2_gui.sh`; set `KEEP_OPEN_AFTER_ROLLOUT_S=60` for a longer post-rollout inspection. GUI controls do not add success evidence.
- Open questions: None for the launcher. The separate P4.4 natural-contact validation remains open.

#### 2026-07-10
- Scope: Link-backed RobotAnchor attach gate hardening for the completed P4.2 v1 deterministic rollout.
- Files changed:
  - `amsrr/simulation/p4_2_rollout.py`
  - `amsrr/simulation/p4_2_isaac_env.py`
  - `amsrr/acceptance/p4_2_acceptance.py`
  - `amsrr/training/p4_2_deterministic_rollout_runner.py`
  - `scripts/p4_control_holon_spawn_probe.py`
  - `tests/unit/simulation/test_p4_2_rollout.py`
  - `tests/unit/simulation/test_p4_2_isaac_env.py`
  - `tests/unit/training/test_p4_2_deterministic_rollout_runner.py`
  - `tests/acceptance/test_p4_2_acceptance.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Upstream dependencies: P4.2 selected ContactCandidate / RobotAnchor assignments, P3 assembled graph module ids, Isaac body-name prefixing, and P4.2 payload-coupled attach contract.
- Implemented: Link-backed anchor pose resolution from `RobotAnchor.link_id`, attach/root-target/slaving based on the resolved Isaac body pose plus `RobotAnchor.local_pose`, attach event link evidence, rollout anchor debug samples, and acceptance rejection for module-state-only attach evidence.
- Not implemented: Natural contact grasp, connector mesh contact validation, friction/slip validation, contact-force logging, or joint-closure grasping.
- Schema/interface changes: No persisted base schema changes. Additive P4.2 attach-event/report/archive fields only.
- Downstream impact: P4.2 fake and real gates now require link-backed attach evidence. A fallback anchor pose can be logged for diagnostics but cannot complete P4.2.
- Tests added:
  - Link-backed attach event schema checks.
  - Acceptance rejection for attach events lacking `anchor_pose_source="isaac_link"`.
- Tests passed:
  - Targeted P4.2 tests: 21 passed.
  - Full unit/acceptance suite: 186 passed, 1 skipped.
  - Real Isaac P4.2 rollout: `completion_passed=true`, `anchor_resolved_body_name="module_1__pitch_dock_mech1"`, `p4_2_attach_event_link_backed_count=1.0`.
  - `git diff --check` passed.
- Handoff notes: This is still `kinematic_payload_coupled_attach_v1`; it is connector-link-backed kinematic attach, not natural grasp or high-fidelity contact.
- Open questions: Whether later P4.2/P4.4 should replace heuristic `RobotAnchor.local_pose` with a dock-port/contact-surface-derived anchor frame before natural-contact validation.

#### 2026-07-10
- Scope: Order 2 deterministic `π_H` / `π_L` P4.2 phase adaptation.
- Files changed:
  - `amsrr/policies/contact_wrench_trajectory.py`
  - `amsrr/policies/low_level_policy_base.py`
  - `amsrr/policies/__init__.py`
  - `tests/unit/policies/test_p4_2_deterministic_policies.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Upstream dependencies: v0.4 π_H/π_L role split, P4.2 rollout contract, existing deterministic assignment feasibility and baseline low-level policy.
- Implemented: `P4_2DeterministicGraspCarryPlanner`, `P4_2DeterministicPlannerConfig`, phase guard helper, contact-centroid/body-target helpers, and P4.2 phase priority weights in `BaselineLowLevelPolicy`.
- Not implemented: Isaac-side attach gating, contact state updates, object kinematic attach/release, real rollout execution, archive generation, or split acceptance.
- Schema/interface changes: None. `ContactWrenchTrajectory`, `InteractionKnot`, and `PolicyCommand` schemas are unchanged.
- Downstream impact: Agent J/K can use the P4.2 planner in rollout env/runner code and can inspect `InteractionKnot.guard_conditions` / `PolicyCommand.priority_weights` for phase intent.
- Tests added: `tests/unit/policies/test_p4_2_deterministic_policies.py`
- Tests passed:
  - P4.2/related high-low baseline tests: 8 passed.
  - Full policy unit tests: 22 passed.
  - Compileall and `git diff --check` passed.
- Handoff notes: This order does not attach objects by itself. Later P4.2 env code must still enforce the gated attach contract from Order 1.
- Open questions: None for Order 2.

#### 2026-07-10
- Scope: Order 1 P4.2 rollout contract, phase state machine, gated attach, metric definitions, no-mislabeling artifacts, config, and unit tests.
- Files changed:
  - `amsrr/simulation/p4_2_rollout.py`
  - `amsrr/simulation/__init__.py`
  - `configs/training/p4_2_deterministic_rollout.yaml`
  - `tests/unit/simulation/test_p4_2_rollout.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Upstream dependencies: v0.4 P4.2 deterministic rollout requirements and user clarifications for explicit phases, gated kinematic attach, P2/P3 morphology reflection, drop/collision/controller failure definitions, split acceptance, and no learning/full-completion claims.
- Implemented: `P4_2RolloutPhase`, default phase definitions with entry/exit/timeout metadata, `P4_2DeterministicRolloutConfig`, attach condition and attach event contracts, metric definition contract, no-mislabeling helper, terminal failure metrics, config file, package exports, and unit tests.
- Not implemented: P4.2 Isaac env/probe, P2/P3 graph-to-asset/module placement execution, per-step rollout runner/archive generation, split P4.2 acceptance, or real Isaac rollout execution.
- Schema/interface changes: No persisted schema changes. Additive P4.2 simulation-side contracts only.
- Downstream impact: Later P4.2 env/runner/acceptance orders must use this contract and cannot pass completion with unconditional attach, module-count-only provenance, fake backend only, or learning/P4-full claims.
- Tests added: `tests/unit/simulation/test_p4_2_rollout.py`
- Tests passed:
  - `PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/simulation/test_p4_2_rollout.py -q` -> 6 passed.
  - `python3 -m compileall amsrr/simulation tests/unit/simulation/test_p4_2_rollout.py -q` -> passed.
- Handoff notes: P4.2 success rate must be labeled as deterministic payload-carry rollout success under `contact_model="kinematic_payload_coupled_attach_v1"`, not high-fidelity natural grasp success, true fixed-joint dynamics success, learned policy success, P4.3, or P4 full completion.
- Open questions: None for Order 1.

### P4.1 Implementation: Isaac Full-Scene Backend Smoke

#### 2026-07-09
- Scope: Order 1 P4.1 smoke contract, config, and RuntimeObservation joint-state checks.
- Files changed:
  - `amsrr/simulation/p4_1_backend_smoke.py`
  - `amsrr/simulation/__init__.py`
  - `configs/training/p4_1_backend_smoke.yaml`
  - `tests/unit/simulation/test_p4_1_backend_smoke.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Upstream dependencies: v0.4 P4.1 backend smoke requirements, user clarifications on full-scene robot/object/floor scope, P2/P3 usage, real-smoke completion gate, per-step archive logging, and joint-state preservation.
- Implemented: `P4_1FullSceneBackendConfig`, `P4_1BackendSmokeResult`, `P4_1RuntimeJointStateMetrics`, required real smoke name `p2_p3_full_scene_backend`, P4.1 config file, and `evaluate_runtime_observation_joint_state` for vectoring/dock joint checks and articulated B(q) update evidence.
- Not implemented: Real Isaac full-scene probe, P2/P3 runner integration, per-step archive writer, P4.1 acceptance gate, or real Isaac smoke execution.
- Schema/interface changes: None to persisted schemas.
- Downstream impact: Agent J/K/L P4.1 orders can now share a stable fake/real smoke result contract. Fake unit gates can pass contract checks, but P4.1 completion remains impossible without real Isaac smoke evidence.
- Tests added: `test_p4_1_backend_smoke_config_loader_contract`, `test_p4_1_runtime_observation_joint_state_requires_vectoring_and_dock_joints`, `test_p4_1_runtime_observation_joint_state_rejects_empty_joint_positions`, and `test_p4_1_articulated_joint_state_requires_model_update_metric`.
- Tests passed: P4.1 contract tests passed: 4 passed. Related simulation tests passed: 10 passed.
- Handoff notes: P4.1 must not be reported as object grasp/carry success, learned policy success, P4.2 rollout, or P4 full completion.
- Open questions: None for Order 1.

### P4-Control / P4a: QP/PID Controller Specification

#### 2026-07-09
- Scope: Final closeout for P4-control/P4a after the articulated multi-link correction.
- Files changed:
  - `scripts/p4_control_holon_spawn_probe.py`
  - `for_codex/WORKLOG.md`
- Upstream dependencies: Articulated multi-link smoke correction, real Isaac smoke runner, QPID rigid-body QP path, and P4-control split acceptance report.
- Implemented: Fixed the non-articulated `--fixed-morphology-waypoint-smoke` probe branch so it initializes `fixed_articulated_joint_names=None` before calling the shared fixed-morphology smoke runner. Recorded final closeout gates and real Isaac results.
- Not implemented: No new solver backend, no closed-loop docking constraint model, no object grasp/carry rollout, no learned policy training, and no P4 full completion claim.
- Schema/interface changes: None.
- Downstream impact: P4-control/P4a low-level runner can complete its required smoke set again after the articulated correction. The optional articulated smoke remains separate regression evidence for multi-link deformation and q-dependent model updates.
- Tests added: None; this was a closeout rerun plus a small branch-initialization bug fix.
- Tests passed: Unit suite passed: 136 passed, 1 skipped. Acceptance suite passed: 9 passed. `compileall` and `git diff --check` passed. Real runner passed required smokes (`single_module_hover`, `fixed_morphology_hover`, `fixed_morphology_waypoint`) with P4-control/P4a `completion_passed=true`. Real articulated 20 s hover smoke passed with module motion and control-model update checks.
- Handoff notes: This closeout is scoped to P4-control/P4a. Do not interpret the acceptance report as P4 full completion, object manipulation success, or learned policy success.
- Open questions: None for P4-control/P4a closeout.

#### 2026-07-09
- Scope: Add Agent I/J diagnostic controls for long-hover drift investigation.
- Files changed:
  - `amsrr/controllers/qp_allocator_interface.py`
  - `amsrr/controllers/qpid_controller.py`
  - `amsrr/controllers/__init__.py`
  - `amsrr/robot_model/fixed_morphology_urdf.py`
  - `scripts/p4_control_holon_spawn_probe.py`
  - `tests/unit/controllers/test_qpid_controller.py`
  - `tests/unit/robot_model/test_fixed_morphology_urdf.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Upstream dependencies: Virtual thrust channel allocator, rigid-body model, Holon URDF conversion helper, and single-module hover smoke.
- Implemented: Debug-only `RigidBodyPseudoinverseAllocator`, `QPIDControllerConfig(allocation_mode="rigid_body_pseudoinverse")`, probe `--allocation-mode`, and probe `--vectoring-velocity-limit-rad-s` for conversion-time gimbal velocity-limit overrides.
- Not implemented: No controller retuning, no acceptance-threshold change, no replacement of QP primary allocation, no P4 full completion claim.
- Schema/interface changes: None.
- Downstream impact: Future hover debugging can compare QP and pseudoinverse with the same smoke runner and can regenerate faster-vectoring USDs without changing source URDF assets.
- Tests added: Pseudoinverse allocator back-conversion test, controller pseudoinverse selection test, fixed-module joint velocity override test.
- Tests passed: Related unit tests passed: 21 passed. Real Isaac 10 s no-stop comparisons reproduced QP drift and showed vectoring velocity/stiffness changes alone do not solve stability.
- Handoff notes: Pseudoinverse is diagnostic-only; QP remains the required primary path for P4-control acceptance.
- Open questions: Need deeper investigation of axis/sign conventions, passive dock joint motion, yaw/reaction torque authority, and PID/integrator behavior.

#### 2026-07-09
- Scope: Fix Agent J/K Isaac GUI asset visibility for Holon generated USDs.
- Files changed:
  - `amsrr/robot_model/fixed_morphology_urdf.py`
  - `scripts/p4_control_holon_spawn_probe.py`
  - `tests/unit/robot_model/test_fixed_morphology_urdf.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Upstream dependencies: Runtime Holon URDF, `module_urdf/mesh` STL assets, IsaacLab URDF converter, and GUI observation probe options.
- Implemented: Added `write_resolved_mesh_urdf`, added mesh-search support to fixed-morphology URDF generation, routed P4-control probe conversion through resolved mesh URDFs, and validated that regenerated USDs contain instance-proxy mesh prims.
- Not implemented: No change to controller dynamics, QP allocation, hover thresholds, physical docking, object grasp/carry, learning, or P4 full completion.
- Schema/interface changes: None.
- Downstream impact: Regenerate USDs with `--force-convert` before GUI inspection; old generated USDs may still be invisible because they were produced from unresolved relative mesh paths.
- Tests added: `test_resolved_mesh_urdf_points_asset_meshes_to_existing_files`; fixed-morphology mesh refs now must exist.
- Tests passed: Focused robot-model tests passed: 5 passed, 1 skipped. Simulation env tests passed: 6 passed. `py_compile` passed. Real Isaac conversion, GUI spawn, and GUI single-module hover passed; USD instance-proxy traversal found 38 mesh prims.
- Handoff notes: Kit's Stage tree may show instanceable Xforms rather than expanded mesh children unless instance proxies are inspected; that is normal after conversion.
- Open questions: None for asset visibility.

#### 2026-07-09
- Scope: Implement Agent I Order 12 `PolicyCommand` PID target builder.
- Files changed:
  - `amsrr/controllers/qpid_controller.py`
  - `tests/unit/controllers/test_qpid_controller.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Upstream dependencies: Controller supplement PID target-builder section, user-provided PID gains, `PolicyCommand` direct body target fields, runtime pose/twist, and QP allocation result status.
- Implemented: PID gains in `QPIDControllerConfig`, body-pose/twist target wrench generation, world-frame gravity/position PID force converted to body frame, quaternion body-frame attitude PID torque, feedforward residual wrench addition, target tracking metrics, and allocation-gated integral anti-windup.
- Not implemented: Closed-loop Isaac execution, fixed-morphology hover, waypoint smoke, explicit desired-wrench saturation limits, or P4-control completion.
- Schema/interface changes: None.
- Downstream impact: Agent J/K closed-loop smoke can now compute `ControllerCommand` from observed pose/twist and `PolicyCommand.desired_body_pose` without bypassing the controller boundary.
- Tests added: `test_qpid_controller_builds_pid_wrench_from_policy_body_target_and_feedforward`.
- Tests passed: Controller unit tests passed: 9 passed. Full unit suite passed: 122 passed, 1 skipped. Compileall and `git diff --check` passed.
- Handoff notes: IsaacLab 3 root poses use XYZW quaternions, matching A-MSRR `Pose7D`. Integral state is held by the controller instance, so closed-loop smoke must reuse the controller across steps.
- Open questions: None for this order.

#### 2026-07-09
- Scope: Tune Agent I/J controller-command QP feasibility after the first real controller-to-Isaac smoke.
- Files changed:
  - `amsrr/controllers/qp_allocator_interface.py`
  - `amsrr/controllers/qpid_controller.py`
  - `tests/unit/controllers/test_qpid_controller.py`
  - `tests/unit/simulation/test_p4_control_controller_smoke.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Upstream dependencies: Agent I virtual-thrust QP allocator and RigidBodyControlModel, Agent J controller-command smoke, Holon actuator limits and real Isaac command surface.
- Implemented: Reduced QP smoothing/regularization weights so the primary objective tracks hover wrench before smoothing, set controller residual tolerance to `1e-5`, added tolerance-aware hard-check clipping detection, and held vectoring targets at current position for effectively zero-thrust rotors.
- Not implemented: Closed-loop hover controller, fixed-morphology smoke, waypoint smoke, object grasp/carry, or P4 full completion.
- Schema/interface changes: None.
- Downstream impact: The controller-command smoke can now be used as a clean precondition for closed-loop work because QP infeasibility is no longer caused by numerical tuning or zero-thrust vectoring angle singularity.
- Tests added: Rigid-body QP hover feasibility unit assertion and strengthened controller-command smoke assertions.
- Tests passed: Focused tests passed: 9 passed. Full unit suite passed: 121 passed, 1 skipped. Compileall and `git diff --check` passed. Real Isaac controller-command smoke passed with QP feasible/ok and no bridge target violations after an approved external run.
- Handoff notes: `BoundedVerticalRotorAllocator` remains only degraded fallback. The `1e-5` tolerance is still below the controller warning threshold and should not be reused as a success metric for real hover.
- Open questions: None for this tuning order.

#### 2026-07-09
- Scope: Probe real IsaacLab URDF conversion for Holon and correct generated USD path.
- Files changed:
  - `configs/env/isaac_lab.yaml`
  - `amsrr/simulation/isaac_lab_backend.py`
  - `tests/unit/simulation/test_p4_control_isaac_env.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Upstream dependencies: AGENTS.md `isaaclab3` environment, IsaacLab `convert_urdf.py`, Holon URDF, and Order 5 backend config.
- Implemented: Verified `isaaclab` Python module visibility under micromamba, verified `isaaclab.sh -p` can run the P4-control probe script, ran IsaacLab URDF conversion to `/tmp`, and corrected generated USD expected path to `<generated_usd_dir>/holon/holon.usda`.
- Not implemented: Workspace USD generation, committed USD assets, articulation spawn, force application, joint target execution, or real smoke simulation.
- Schema/interface changes: None.
- Downstream impact: Later spawn code should consume `artifacts/isaac/robots/holon/holon/holon.usda` after conversion rather than expecting `holon.usd` directly in the generated root.
- Tests added: Updated generated USD path assertion.
- Tests passed: Related env/runner tests passed: 8 passed. Full unit suite passed: 118 passed, 1 skipped. Compileall and `git diff --check` passed. Real Isaac URDF conversion probe passed.
- Handoff notes: The converter warned that `--headless` is deprecated and that omitting `--viz` is now default headless. Future command builders should avoid depending on `--headless`.
- Open questions: None for conversion. Real spawn/control APIs are next.

#### 2026-07-09
- Scope: Implement Agent J/K boundary Order 5 smoke runner configuration and dry-run harness.
- Files changed:
  - `configs/env/isaac_lab.yaml`
  - `configs/training/p4_control_low_level.yaml`
  - `amsrr/simulation/isaac_lab_backend.py`
  - `amsrr/simulation/p4_control_smoke.py`
  - `amsrr/simulation/p4_control_isaac_env.py`
  - `amsrr/simulation/__init__.py`
  - `amsrr/training/p4_control_runner.py`
  - `amsrr/training/__init__.py`
  - `scripts/p4_control_smoke.py`
  - `tests/unit/simulation/test_p4_control_isaac_env.py`
  - `tests/unit/training/test_p4_control_runner.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Upstream dependencies: AGENTS.md Isaac environment notes, P4-control controller supplement, prior acceptance split, and approved recommendations for URDF-to-USD custom articulation plus wrench-composer thrust application.
- Implemented: Config-driven IsaacLab backend probe, URDF conversion command construction, low-level smoke thresholds/scenarios, shared smoke result dataclass, dry-run smoke runner, CLI probe/dry-run entry point, and tests proving dry-run cannot pass completion.
- Not implemented: Real Isaac API execution, URDF-to-USD conversion run, Holon spawn, rotor force application, joint command application, runtime observation extraction, or real smoke pass/fail measurement.
- Schema/interface changes: None to persisted schemas. Added simulation/training-side contracts.
- Downstream impact: The next order can run the CLI under `isaaclab3` / `isaaclab.sh -p` to validate environment imports and then implement real execution behind the existing scenario/result contract.
- Tests added: P4-control backend/env config tests and P4-control runner dry-run tests.
- Tests passed: New P4-control env/runner tests passed: 8 passed. Full unit suite passed: 118 passed, 1 skipped. Targeted P4-control acceptance tests passed: 2 passed. Compileall, `git diff --check`, CLI probe, and CLI dry-run passed.
- Handoff notes: Running the CLI with normal repo Python reports `isaac_python_modules_unavailable_in_current_interpreter`, which is expected. Real probes should use the AGENTS.md micromamba/IsaacLab launch path.
- Open questions: None for config/dry-run. Real Isaac execution details remain to be probed in the next order.

#### 2026-07-09
- Scope: Implement Agent L boundary Order 4 P4-control fast/real acceptance split.
- Files changed:
  - `amsrr/acceptance/p4_control_acceptance.py`
  - `amsrr/acceptance/__init__.py`
  - `tests/acceptance/test_p4_control_acceptance.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Upstream dependencies: P4-control acceptance requirements, user clarification on real Isaac smoke gating, existing P4.0 acceptance/report style, and `EpisodeArchive` P4 logging fields.
- Implemented: `P4ControlSmokeResult`, `P4ControlAcceptanceReport`, `run_p4_control_acceptance`, fast archive checks for controller/runtime/actuator/residual metrics, real smoke aggregation for the three required Isaac smoke cases, and tests proving fast gate does not imply completion.
- Not implemented: Real Isaac smoke runner, Isaac environment spawn/step, hover stabilization, fixed-morphology waypoint tracking, or artifact collection from actual Isaac.
- Schema/interface changes: None to persisted schemas. Added acceptance-only dataclasses and exports.
- Downstream impact: Later P4-control smoke runners can call this acceptance function with real Isaac-backed smoke results. The gate will keep completion false if smoke is missing, skipped, synthetic-only, or failed.
- Tests added: `test_p4_control_fast_gate_does_not_complete_without_real_isaac_smoke` and `test_p4_control_completion_requires_all_real_isaac_smokes`.
- Tests passed: P4-control acceptance tests passed: 2 passed. Full acceptance suite passed: 9 passed. Full unit suite passed: 110 passed, 1 skipped. Compileall and `git diff --check` passed.
- Handoff notes: This order intentionally does not claim P4-control completion. It codifies the split gate so later real smoke artifacts cannot be accidentally replaced by unit tests.
- Open questions: Real Isaac execution details remain for the next runner/backend order.

#### 2026-07-09
- Scope: Implement Agent I/J boundary Order 3 actuator mapping and bridge target records.
- Files changed:
  - `amsrr/controllers/actuator_mapping.py`
  - `amsrr/controllers/isaac_controller_bridge.py`
  - `amsrr/controllers/__init__.py`
  - `tests/unit/controllers/test_actuator_mapping.py`
  - `tests/unit/controllers/test_isaac_controller_bridge.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Upstream dependencies: v0.4 controller bridge requirements, controller supplement bridge/logging section, `PhysicalModel`, `MorphologyGraph`, `ControllerCommand`, and `EpisodeArchive.actuator_target_records`.
- Implemented: Active actuator channel extraction for rotors/vectoring joints/dock joints/joint efforts, deterministic global actuator ids, single-module local aliases, actuator limit clipping, `IsaacControllerBridge` conversion to target records, missing/unsupported/clipped metrics, controller residual/QP status propagation, and JSON-compatible record roundtrip tests.
- Not implemented: Isaac Lab API calls, robot spawning, force application, P4-control runner, real single-module hover, fixed-morphology hover, or waypoint tracking smoke.
- Schema/interface changes: None to persisted schemas. Added controller-local bridge dataclasses and exports.
- Downstream impact: Later Isaac backend and runner code can use these records as the stable bridge contract and store their dict form in `EpisodeArchive.actuator_target_records`.
- Tests added: `test_actuator_mapping_builds_single_module_aliases_and_limits`, `test_actuator_mapping_uses_global_keys_for_multiple_modules`, `test_clip_to_channel_reports_clipped_value`, `test_isaac_controller_bridge_converts_and_clips_targets`, and target-record roundtrip test.
- Tests passed: Controller/mapping/bridge tests passed: 16 passed. Full unit suite passed: 110 passed, 1 skipped. Compileall and `git diff --check` passed.
- Handoff notes: This is a fast pytest bridge contract only. P4-control completion still requires real Isaac smoke with these records or equivalent actuator targets actually executed.
- Open questions: None for this bridge-contract order.

#### 2026-07-09
- Scope: Implement Agent I Order 2 primary virtual-thrust QP allocator and controller integration.
- Files changed:
  - `amsrr/controllers/rigid_body_model.py`
  - `amsrr/controllers/qp_allocator_interface.py`
  - `amsrr/controllers/qpid_controller.py`
  - `amsrr/controllers/__init__.py`
  - `tests/unit/controllers/test_qpid_controller.py`
  - `tests/unit/controllers/test_rigid_body_model.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Upstream dependencies: Agent I Order 1 rigid-body model, P4-control controller supplement, user clarification on rotor-arm-fixed x/z virtual channels and QP constraints plus hard check/clamp.
- Implemented: `VirtualThrustQPAllocator`, optional rigid-body-model allocation inputs, optional vectoring outputs and achieved wrench in allocation results, virtual x/z channel construction, vectoring joint limit/rate linear constraints, thrust/joint hard check and clamp after back-conversion, QP/degraded metrics, controller `allocation_mode="rigid_body_qp"`, and focused unit tests.
- Not implemented: Isaac controller bridge, actuator target record conversion, P4-control runner, real Isaac single-module/fixed-morphology smoke, or waypoint tracking smoke.
- Schema/interface changes: None to persisted schemas. Controller-local dataclasses gained backward-compatible optional fields.
- Downstream impact: Agent J/K/L can now consume a primary QP controller path for fast tests and bridge development, while treating `BoundedVerticalRotorAllocator` as degraded fallback only.
- Tests added: `test_virtual_thrust_qp_allocator_back_converts_vectoring_channel`, `test_virtual_thrust_qp_allocator_applies_limits_and_hard_clamp`, `test_qpid_controller_can_select_rigid_body_qp_primary_path`, plus rigid-body virtual axis/current-q assertions.
- Tests passed: Targeted QP/controller/rigid-body tests passed: 10 passed. Controller unit tests passed: 11 passed. Full unit suite passed: 105 passed, 1 skipped. Compileall and `git diff --check` passed.
- Handoff notes: Virtual z is rotor-arm fixed but sign-aligned with the rotor's positive thrust axis. Exact thrust magnitude bounds are rechecked after solving because the virtual-channel magnitude constraint is nonlinear; QP still includes actuator box bounds and vectoring angle/rate linear constraints.
- Open questions: None for the fast pytest gate. P4-control completion remains blocked on real Isaac smoke in later orders.

#### 2026-07-09
- Scope: Implement Agent I Order 1 deterministic `RigidBodyControlModel` update.
- Files changed:
  - `amsrr/controllers/rigid_body_model.py`
  - `amsrr/controllers/__init__.py`
  - `tests/unit/controllers/test_rigid_body_model.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Upstream dependencies: `PhysicalModel`, `MorphologyGraph`, `RuntimeObservation`, Holon URDF-derived joints/links/rotors, P4-control controller supplement, and user virtual-thrust-channel clarification.
- Implemented: Link-tree forward kinematics from current joint positions, `fc`/baselink-relative module transforms, composite COM and link-level inertia aggregation, body frame at COM with base module orientation, current rotor origin/axis extraction, scalar rotor allocation matrix columns, vectoring joint axes, dock actuator ids, active actuator limits, controller package exports, and unit tests.
- Not implemented: Virtual thrust channel QP expansion, QP solve, back-conversion to `ControllerCommand`, Isaac controller bridge, P4-control runner, or real Isaac smoke.
- Schema/interface changes: None to persisted schemas. Added controller-local internal dataclasses only.
- Downstream impact: The QP allocator can now consume current `B(q)`-style rotor geometry and per-module actuator keys without re-parsing URDF or hard-coding robot paths.
- Tests added: `test_rigid_body_model_builds_single_module_allocation_matrix`, `test_rigid_body_model_updates_rotor_axis_from_joint_position`, `test_rigid_body_model_handles_multiple_modules_with_unique_actuator_ids`.
- Tests passed: Targeted rigid-body tests passed: 3 passed. Controller unit tests passed: 8 passed. Full unit suite passed: 102 passed, 1 skipped. Compileall passed.
- Handoff notes: Zero-axis fixed joints are intentionally skipped in the joint-axis map while still participating in kinematic transforms. Rotor allocation columns are scalar-thrust columns; virtual channel expansion belongs to the next QP allocator order.
- Open questions: None currently known.

#### 2026-07-09
- Scope: Record P4-control virtual thrust channel, QP-primary allocation, and split acceptance-gate clarifications before implementation.
- Files changed:
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Upstream dependencies: User clarification, v0.4 controller/P4-control sections, and the controller supplement.
- Implemented: Documentation supplement requiring per-step `q`-conditioned rigid-body model and `B(q)` updates, virtual thrust channels inside QP with back-conversion to rotor thrust/vectoring targets, degraded-only bounded vertical fallback, and separate fast pytest / real Isaac smoke acceptance gates.
- Not implemented: Controller code, rigid-body model, QP allocator, Isaac bridge, P4-control runner, or acceptance gate.
- Schema/interface changes: None.
- Downstream impact: Agent I/J/K/L implementation should treat these clarified requirements as active P4-control constraints.
- Tests added: None.
- Tests passed: `git diff --check` passed.
- Handoff notes: P4-control completion requires real Isaac smoke; Isaac-unavailable skips are acceptable only for non-completion unit smoke paths.
- Open questions: None currently known.

#### 2026-07-09
- Scope: Add main-spec cross-references to the P4-control QP/PID controller supplement.
- Files changed:
  - `for_codex/A-MSRR_codex_ready_spec_v0_4_ja.md`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Upstream dependencies: User request, revised controller supplement, v0.4 Section 20 and Section 24.5.2.
- Implemented: Main spec references to `for_codex/A-MSRR_QP_PID_controller_design_spec_v0_1_ja.md` from low-level control and P4-control sections.
- Not implemented: Controller code, schema changes, Isaac bridge, P4-control runner, or acceptance gate.
- Schema/interface changes: None.
- Downstream impact: Controller implementers now have an explicit pointer from the source design spec to the controller supplement.
- Tests added: None.
- Tests passed: Not run; documentation-only change. `git diff --check` passed.
- Handoff notes: The cross-reference does not weaken the main spec rule that `π_L` outputs intent only.
- Open questions: None currently known at this documentation level.

#### 2026-07-09
- Scope: Record resolved controller implementation decisions before coding.
- Files changed:
  - `for_codex/A-MSRR_QP_PID_controller_design_spec_v0_1_ja.md`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Upstream dependencies: User answers to open questions, revised P4-control controller spec, local Isaac Lab multirotor/quadcopter examples.
- Implemented: Initial Python/library QP backend decision, per-thruster thrust target primary representation with wrench-composer fallback, absolute vectoring joint targets, reaction torque inclusion, link-level quasi-static inertia aggregation, and initial waypoint thresholds.
- Not implemented: Controller code, QP backend, Isaac bridge, P4-control runner, or acceptance gate.
- Schema/interface changes: None.
- Downstream impact: Agent I/J can now implement against resolved controller assumptions; additional undefined details still require user confirmation before incompatible assumptions are encoded.
- Tests added: None.
- Tests passed: Not run; documentation-only decision recording. `git diff --check` passed after edits.
- Handoff notes: Archive per-rotor thrust targets even if Isaac execution uses wrench composer internally.
- Open questions: None currently known at this planning level.

#### 2026-07-09
- Scope: Revise the P4-control controller spec after user clarification.
- Files changed:
  - `for_codex/A-MSRR_QP_PID_controller_design_spec_v0_1_ja.md`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Upstream dependencies: User clarification, v0.4 controller/P4-control sections, existing controller draft.
- Implemented: Japanese-first controller spec, no reference-implementation notes section, QP allocation as the required allocator path, quasi-static rigid-body model update for assembled morphologies, ControllerCommand / bridge / archive logging requirements, and implementation-before-coding questions.
- Not implemented: Controller code, QP solver backend, rigid-body model code, actuator mapping, Isaac bridge, P4-control runner, or acceptance gate.
- Schema/interface changes: None.
- Downstream impact: Agent I implementation should build deterministic model-update and QP allocation code from the revised spec and keep any non-QP fallback explicitly marked as degraded.
- Tests added: None.
- Tests passed: Not run; documentation-only revision. `git diff --check` passed after edits.
- Handoff notes: Combined morphology control assumes joint motion is quasi-static. Each control cycle updates inertia/CoM/rotor geometry from current joint angles, then controls the current shape as one rigid body.
- Open questions: QP solver/backend choice, Isaac actuator semantics, vectoring target semantics, reaction torque treatment, inertia aggregation fidelity, waypoint error thresholds.

#### 2026-07-09
- Scope: Create a controller-specific draft spec and read the gimbal rotor reference controller at a high level before implementation.
- Files changed:
  - `for_codex/A-MSRR_QP_PID_controller_design_spec_v0_1_ja.md`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Upstream dependencies: v0.4 controller/P4-control sections, current Agent I controller scaffolds, temporary `aerial_robot_base` reference source.
- Implemented: Draft controller spec skeleton with reference branch notes, QP/PID layer placeholders, allocation model placeholders, Isaac bridge/logging expectations, proposed Agent I/J/K/L files, and open questions.
- Not implemented: Controller code, actuator mapping, Isaac bridge, P4-control runner, acceptance gate, QP solver backend, or Isaac execution.
- Schema/interface changes: None.
- Downstream impact: Agent I implementation should begin only after this draft is refined into concrete equations and config/test requirements.
- Tests added: None.
- Tests passed: Not run; documentation-only task.
- Handoff notes: Keep `π_L` authority limited to `PolicyCommand`; controller/bridge remains final actuator authority. The reference branch sends base thrust plus torque allocation matrix to FC rather than directly publishing gimbal commands.
- Open questions: Frame conventions, actuator target semantics in Isaac, reaction torque handling, solver/fallback choice, multi-module inertia aggregation, and waypoint error thresholds.

### P4.0 Implementation: Simplified Full-Pipeline Integration

#### 2026-07-08
- Scope: Order 6 final docs, design modification log, and full verification.
- Files changed:
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Upstream dependencies: P4.0 Orders 1-5 and full suite verification.
- Implemented: P4.0 design modification supplement and final worklog handoff entry with full unit/acceptance results.
- Not implemented: Isaac Lab backend, controller bridge / actuator mapping, P4-control, P4.1/P4.2 deterministic Isaac rollout, P4.3 learning bootstrap, or P4 full acceptance.
- Schema/interface changes: None in this order.
- Downstream impact: Next work package should start with controller bridge / actuator mapping and P4-control, using P4.0 acceptance as the simplified wiring prerequisite.
- Tests added: None.
- Tests passed: Full unit suite passed: 99 passed, 1 skipped in 4.67s. Full acceptance suite passed: 7 passed in 118.22s.
- Handoff notes: P4.0 is accepted as simplified full-pipeline wiring only; it must not be reported as Isaac-backed full grasp/carry completion.
- Open questions: None currently.

#### 2026-07-08
- Scope: Order 5 P4.0 simplified acceptance gate.
- Files changed:
  - `amsrr/acceptance/p4_0_acceptance.py`
  - `amsrr/acceptance/__init__.py`
  - `amsrr/training/p4_0_full_pipeline_runner.py`
  - `tests/acceptance/test_p4_0_acceptance.py`
- Upstream dependencies: P4.0 runner/archive tests, P4.0 no-mislabeling metadata, v0.4 Section 24.5.1 and Section 27.3 acceptance items.
- Implemented: `P4_0AcceptanceCriteria`, `P4_0AcceptanceReport`, `run_p4_0_acceptance`, P2/P3 usage checks, FixedSimple absence check, candidate/trajectory/policy/controller/archive completeness checks, simplified metric recording checks, no-mislabeling checks, backend-note report field, acceptance export, and acceptance test.
- Not implemented: Isaac controller bridge, actuator mapping, P4-control, P4.1/P4.2/P4.3, or P4 full acceptance.
- Schema/interface changes: None to persisted schemas.
- Downstream impact: P4.0 simplified full-pipeline wiring can be accepted independently, and later P4 stages can require this gate as a prerequisite.
- Tests added: `test_p4_0_acceptance_simplified_full_pipeline`.
- Tests passed: P4.0 acceptance targeted test passed: 1 passed. P4.0 unit + acceptance passed: 3 passed. `python3 -m compileall amsrr -q` passed. `git diff --check` passed.
- Handoff notes: The report explicitly says the metrics are not Isaac-backed physical success rates; this is not P4 full completion.
- Open questions: None currently.

#### 2026-07-08
- Scope: Order 4 P4.0 unit, archive completeness, and no-mislabeling tests.
- Files changed:
  - `tests/unit/training/test_p4_0_full_pipeline_runner.py`
- Upstream dependencies: Order 3 runner and config, `EpisodeArchive` defaults, simplified env external design injection, P2/P3 scaffolds.
- Implemented: Unit test coverage for P4.0 runner config loading, full simplified pipeline archive contents, P2 selected design usage, P3 assembly result usage, contact candidates, trajectory records, policy/controller commands, runtime observations, rewards, and explicit no-Isaac/no-P4-full rollout metadata.
- Not implemented: P4.0 acceptance gate, report schema, full acceptance test, Isaac backend, or learning bootstrap.
- Schema/interface changes: None.
- Downstream impact: Agent L acceptance can aggregate runner metrics and check phase-level pass/fail criteria with confidence that per-archive fields are complete.
- Tests added: `test_p4_0_full_pipeline_runner_archives_full_simplified_pipeline`, `test_p4_0_full_pipeline_runner_config_loader`.
- Tests passed: P4.0 runner tests passed: 2 passed. Related env/P1/P4.0 tests passed: 9 passed. `python3 -m compileall amsrr -q` passed. `git diff --check` passed.
- Handoff notes: The test asserts `rollout_artifacts["is_p4_full_completion"] is False`, `isaac_backed is False`, and `physical_success_claim is False`.
- Open questions: None currently.

#### 2026-07-08
- Scope: Order 3 P4.0 full-pipeline runner implementation.
- Files changed:
  - `amsrr/training/p4_0_full_pipeline_runner.py`
  - `amsrr/training/__init__.py`
  - `configs/training/p4_0_grasp_carry.yaml`
- Upstream dependencies: P2 selected design path, P3 simplified assembly result, Order 2 env injection, ContactCandidateSampler, `GraspCarryBaselinePlanner`, `BaselineLowLevelPolicy`, `QPIDController`, and `EpisodeArchive`.
- Implemented: `P4_0FullPipelineRunnerConfig`, `P4_0FullPipelineRunnerResult`, config loader, `P4_0FullPipelineRunner`, deterministic episode sampling, P2 selection, P3 assembly execution, simplified rollout execution, reward/metric aggregation, archive writing, and explicit simplified-backend no-P4-full metadata.
- Not implemented: Unit/archive/no-mislabeling tests, P4.0 acceptance gate, Isaac backend, controller bridge, actuator mapping, or learning bootstrap.
- Schema/interface changes: None to persisted schemas.
- Downstream impact: Order 4 can assert the runner uses P2/P3 outputs, generates candidates/trajectory/policy/controller records, and does not label P4.0 as Isaac-backed or full P4 completion.
- Tests added: None in this order.
- Tests passed: `python3 -m compileall amsrr -q` passed. P4.0 config/import smoke passed. `git diff --check` passed.
- Handoff notes: Archive `rollout_artifacts["note"]` states that P4.0 metrics are simplified backend indicators, not Isaac-backed physical success rates.
- Open questions: None currently.

#### 2026-07-08
- Scope: Order 2 simplified env external `DesignOutput` / assembled morphology injection.
- Files changed:
  - `amsrr/simulation/simplified_grasp_carry_env.py`
  - `tests/unit/simulation/test_simplified_grasp_carry_env.py`
- Upstream dependencies: P4.0 selected design / assembled morphology handoff requirement, existing `SimplifiedGraspCarryEnv`, P2 deterministic `P2DesignPolicy`, and P3 simplified assembly boundary.
- Implemented: Optional `design_output` and `assembled_morphology` injection on env construction/reset, internal `design_source` labeling, external-design build path that bypasses `FixedSimpleDesignPolicy`, and assembled morphology replacement while preserving design metadata.
- Not implemented: P4.0 runner, archive completeness checks, P4.0 acceptance, Isaac backend, controller bridge, or actuator mapping.
- Schema/interface changes: None to persisted schemas.
- Downstream impact: Agent K can build P4.0 episodes using P2 selected `DesignOutput` and P3 assembled `MorphologyGraph` before sampling contacts and planning trajectories.
- Tests added: `test_simplified_grasp_carry_env_accepts_external_design_output`.
- Tests passed: Simplified env tests passed: 4 passed. P1 runner tests passed: 3 passed. `python3 -m compileall amsrr -q` passed. `git diff --check` passed.
- Handoff notes: Existing P1 `FixedSimpleDesignPolicy` path remains the default for callers that do not provide an external design.
- Open questions: None currently.

#### 2026-07-08
- Scope: Order 1 archive compatibility for P4.0/P4 logging fields.
- Files changed:
  - `amsrr/logging/episode_archive.py`
  - `tests/unit/training/test_p1_runner.py`
- Upstream dependencies: v0.4 Section 25.1, P4.0 simplified archive requirements, existing P1/P2/P3 runner archive behavior.
- Implemented: Defaulted `runtime_observations`, `actuator_target_records`, `rollout_artifacts`, and `learning_artifacts` on `EpisodeArchive`; added a legacy dict restoration check for archives missing those fields.
- Not implemented: P4.0 runner, simplified env injection, P4.0 acceptance, Isaac actuator target conversion, or learned training artifacts.
- Schema/interface changes: Additive archive fields only; existing archives deserialize with defaults.
- Downstream impact: Later P4.0 runner can store trajectory/policy/controller/reward metrics immediately and can optionally include simplified runtime observations, while P4-control/Isaac work can fill actuator records.
- Tests added: Legacy `EpisodeArchive.from_dict` default restoration assertions in `test_p1_runner_collects_metrics_and_archives`.
- Tests passed: P1 runner tests passed: 3 passed. P2/P3 runner tests passed: 5 passed. `python3 -m compileall amsrr -q` passed. `git diff --check` passed.
- Handoff notes: Keep P4.0 no-mislabeling checks separate: these fields enable P4 logging but do not imply Isaac-backed rollout or P4 full completion.
- Open questions: None currently.

### P4.3 Design Revision: Learning Target Clarification

#### 2026-07-08
- Scope: Clarify P4.3 learning bootstrap targets in the source design spec only.
- Files changed:
  - `for_codex/A-MSRR_codex_ready_spec_v0_4_ja.md`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Upstream dependencies: User追加修正 request, existing P4 Isaac-backed completion clarification, P2.5 learned π_D scorer / feasibility head notes, and v0.4 π_D / π_H / π_L ownership boundaries.
- Implemented: Three staged P4.3 learning target families, P4.3a-P4.3e recommended sequence, expanded P4 full acceptance learning artifacts for π_L/residual control, π_H, and π_D scorer fine-tuning, and updated Mermaid training-loop arrows to π_D / π_H / π_L.
- Not implemented: Any training code, checkpoints, policy heads, dataset builders, acceptance code, or Isaac rollout code.
- Schema/interface changes: Source spec only.
- Downstream impact: Future P4.3 work must not interpret learning bootstrap as π_L-only. Learned π_D scorer usage remains outcome-conditioned scoring/ranking only, π_H learning owns contact assignment / trajectory timing, and π_L learning owns PolicyCommand / residual intent.
- Tests added: None.
- Tests passed: Documentation verification only: required P4.3 terms were found in the revised source spec; `git diff --check` passed.
- Handoff notes: Learned models may enter production only through deterministic safety gates; hard feasibility remains owned by `FeasibilityChecker`.
- Open questions: None currently.

### P4 Design Revision: Isaac-Backed Full Completion Clarification

#### 2026-07-08
- Scope: Revise the source design spec only, clarifying P4 staging and preventing simplified full-pipeline acceptance from being treated as P4 completion.
- Files changed:
  - `for_codex/A-MSRR_codex_ready_spec_v0_4_ja.md`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Upstream dependencies: User-provided P4 design revision instruction, v0.4 Sections 17, 20, 23, 24, 25, 26, 27, and current P1/P2/P3 handoff state.
- Implemented: P4 phase split, P4.0 simplified integration scope, P4-control low-level Isaac flight validation prerequisites, controller bridge / actuator mapping requirements, π_A docking/detach/separation bridge requirement, Isaac backend requirements, split P4 acceptance, P4 learning bootstrap requirements, P4 mermaid flow, EpisodeArchive P4 logging fields, Agent I/J/K/L P4 ownership notes, and revised implementation order.
- Not implemented: Any P4 code, Isaac Lab backend, controller bridge, actuator mapping, P4 runner, P4 acceptance gate, or learning run.
- Schema/interface changes: Source spec only. Future schema/code changes are implied for P4 archive logging, but no implementation module was changed.
- Downstream impact: Future P4 implementation must proceed through P4.0, P4-control/P4a, P4.1, P4.2, P4.3, and P4 full acceptance rather than claiming completion after simplified backend wiring.
- Tests added: None.
- Tests passed: Documentation checks only: required P4 terms were found in the revised source spec; `git diff --check` passed.
- Handoff notes: P2.5 learned models remain auxiliary and deterministic `P2DesignPolicy` / `FeasibilityChecker` fallback remains required at P4 start.
- Open questions: None currently.

### P2.5: Post-P2 Inspection, Visualization, and Candidate Trace Export

#### 2026-07-08
- Scope: Add a pre-P3 inspection/debugging phase without replacing the existing P2 completion gate.
- Files changed:
  - `amsrr/training/p2_inspection_context.py`
  - `amsrr/training/p2_candidate_trace_export.py`
  - `amsrr/visualization/__init__.py`
  - `amsrr/visualization/p2_morphology.py`
  - `amsrr/reporting/__init__.py`
  - `amsrr/reporting/p2_5_inspection_report.py`
  - `amsrr/acceptance/__init__.py`
  - `amsrr/acceptance/p2_5_inspection.py`
  - `tests/unit/visualization/test_p2_morphology_visualization.py`
  - `tests/unit/training/test_p2_candidate_trace_export.py`
  - `tests/unit/reporting/test_p2_5_inspection_report.py`
  - `tests/acceptance/test_p2_5_inspection.py`
  - `outputs/p2_5/`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Upstream dependencies: User-requested P2.5 phase, existing P2 design policy/variants, FeasibilityChecker labels/margins, P2 design config/distribution, and Section 24.3 completion.
- Implemented: SVG morphology graph/layout visualization for all four P2 variants, JSONL/CSV per-candidate trace export, explicit closed-loop rejected probe, markdown inspection report, and P2.5 acceptance gate.
- Not implemented: Isaac, π_H, π_L, QP/PID, actuator commands, learned training, P3 assembly integration.
- Schema/interface changes: None to persisted schemas.
- Downstream impact: Human reviewers can inspect `outputs/p2_5/report/p2_5_inspection_report.md`, SVG layouts, and candidate traces before P3. P2 completion remains unchanged.
- Tests added: `test_p2_morphology_visualization_outputs_graph_and_layout_svgs`, `test_p2_candidate_trace_export_writes_all_candidates_and_probe`, `test_p2_5_inspection_report_contains_summary_and_scope_notes`, `test_p2_5_inspection_acceptance_gate`.
- Tests passed: Targeted P2.5 tests passed individually. Full unit suite passed: 83 passed, 1 skipped. Full acceptance suite passed: 4 passed in 88.72s. `python3 -m compileall amsrr -q` passed. `git diff --check` passed.
- Handoff notes: The generated trace contains 5 records: four normal P2 policy variants plus one closed-loop invalid probe, with counts accepted=4, rejected=1, selected=1.
- Open questions: Human review of P2.5 visualization/report is still recommended before P3 starts.

### Agent E: P2 π_D Candidate Selection Scaffold

#### 2026-07-08
- Scope: Add deterministic P2 π_D scaffold that enumerates candidate morphology designs, labels them with FeasibilityChecker results, and deterministically selects a design by soft score.
- Files changed:
  - `amsrr/policies/__init__.py`
  - `amsrr/policies/design_policy_p2.py`
  - `tests/unit/policies/test_p2_design_policy.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Upstream dependencies: Agent E grasp/carry morphology variants, Agent F P2 FeasibilityChecker labels/margins, existing `DesignPolicyContext`, `DesignOutput`, and v0.4 π_D / P2 acceptance guidance.
- Implemented: `P2DesignPolicyConfig`, `P2DesignCandidateEvaluation`, `P2DesignSelection`, `P2DesignPolicy`, variant enumeration, candidate feasibility evaluation, accepted/rejected split, deterministic soft scoring, selected-design annotation, and package exports.
- Not implemented: Learned π_D neural scoring, policy-gradient training, replay/dataset generation, P2 runner/acceptance gate, or simulator execution.
- Schema/interface changes: None to persisted schemas.
- Downstream impact: P2 runners can inspect `selection.candidates`, `accepted_candidates`, `rejected_candidates`, and `selected_candidate`, while callers that only need a design can use `P2DesignPolicy.design(context)`.
- Tests added: `test_p2_design_policy_enumerates_variants_and_selects_best_accepted`, `test_p2_design_policy_splits_rejected_candidates_with_feasibility_checker`, `test_p2_design_policy_falls_back_to_best_rejected_when_none_accepted`.
- Tests passed: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/policies/test_p2_design_policy.py tests/unit/policies/test_design_teacher.py -q` passed: 6 passed. `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q` passed: 77 passed, 1 skipped. `python3 -m compileall amsrr -q` passed. `git diff --check` passed.
- Handoff notes: The current hand-coded soft score intentionally prefers accepted candidates first, then balances slot/capability coverage, reachability, thrust/payload margins, optional support, variant prior, and complexity. It is a deterministic baseline for P2 before learned scoring.
- Open questions: None currently.

### Agent F: P2 Design-Level FeasibilityChecker

#### 2026-07-08
- Scope: Strengthen design-level feasibility outputs for P2 grasp/carry design evaluation and acceptance aggregation.
- Files changed:
  - `amsrr/feasibility/checker.py`
  - `tests/unit/feasibility/test_feasibility_checker.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Upstream dependencies: v0.4 hard-check list and P2 acceptance criteria, existing `FeasibilityResult`, IRG `CapabilityRequirement --applies_to--> ContactSlot` edges, Agent E P2 grasp/carry morphology variants, and PhysicalModel thrust data.
- Implemented: Checker version `p2_agent_f_design_v1`, stable `L_FEASIBLE` / `L_HARD_VIOLATION` / `L_<hard_check_code>` labels, coverage and capability ratios, CapabilityRequirement min-force checks, reachability ratios, port conflict counts, closed-loop rejection margins, detailed thrust/payload force margins, and metadata violation counts.
- Not implemented: Exact collision checking, exact QP hover solve, learned feasibility head, P2 runner/acceptance harness, or simulator validation.
- Schema/interface changes: None to persisted schemas.
- Downstream impact: Agent K/L P2 runners can store and aggregate feasibility labels directly from `EpisodeArchive.feasibility_result`. Agent E design policy work can use deterministic rejection labels for candidate evaluation.
- Tests added: `test_p2_feasibility_checker_records_acceptance_margins_for_variant`, `test_p2_feasibility_checker_uses_capability_requirement_force_label`, `test_p2_feasibility_checker_records_port_conflict_margins`, `test_p2_feasibility_checker_records_reachability_margins`.
- Tests passed: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/feasibility/test_feasibility_checker.py -q` passed: 7 passed. `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q` passed: 74 passed, 1 skipped. `python3 -m compileall amsrr -q` passed. `git diff --check` passed.
- Handoff notes: `proxy_scores["L_..."]` entries are deterministic labels encoded in the available float map, not learned proxy values. Hard safety remains owned by `hard_violations` and `feasible`.
- Open questions: None currently.

### Agent E: P2 Grasp-Carry Morphology Variant Builder

#### 2026-07-08
- Scope: Implement P2 order 1 real object grasp/carry morphology variants as distinct `MorphologyGraph` outputs, without changing schemas or downstream policy/controller interfaces.
- Files changed:
  - `amsrr/morphology/__init__.py`
  - `amsrr/morphology/grasp_carry_designs.py`
  - `amsrr/policies/design_teacher.py`
  - `tests/unit/morphology/test_grasp_carry_variants.py`
  - `tests/unit/policies/test_design_teacher.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Upstream dependencies: v0.4 MorphologyGraph/DesignOutput schemas, π_D teacher variant names, STOP validity constraints, IRG ContactSlots, Holon PhysicalModel dock ports, and existing FeasibilityChecker hard-check scaffold.
- Implemented: `GraspCarryMorphologyVariant`, `GraspCarryMorphologyVariantBuilder`, `build_grasp_carry_variant_design_output`, four deterministic connected-tree layouts, variant-specific module roles/poses/edges/control groups, required/optional RobotAnchor placement, design action traces, and design teacher routing for object grasp/carry variants.
- Not implemented: Learned π_D scorer/sampler, optimized morphology search, P2 design runner, P2 acceptance gate, exact collision/QP feasibility, or Isaac execution.
- Schema/interface changes: None to persisted schemas.
- Downstream impact: Future P2 design datasets and feasibility labeling can distinguish topology variants while preserving existing `DesignOutput` and `RobotAnchor` contracts. Existing P1 simplified flow continues to use `FixedSimpleDesignPolicy` through the same interface.
- Tests added: `test_grasp_carry_variants_build_distinct_feasible_morphologies`, `test_grasp_carry_variant_topology_shapes`, `test_grasp_carry_variants_cover_required_slot_min_count`.
- Tests passed: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/morphology/test_grasp_carry_variants.py tests/unit/policies/test_design_teacher.py -q` passed: 6 passed. `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q` passed: 70 passed, 1 skipped. `python3 -m compileall amsrr -q` passed. `git diff --check` passed.
- Handoff notes: `central_base_plus_two_grasp_arms` requires enough module budget for a five-module two-link-arm layout. The default grasp/carry teacher selection still chooses `tri_anchor_support_grasp` when an optional support slot exists and `max_modules >= 3`.
- Open questions: None currently.

### Agent L: Tests and Acceptance

#### 2026-07-08
- Scope: Final P3 verification and handoff after order 1-5 implementation commits.
- Files changed:
  - `for_codex/WORKLOG.md`
- Upstream dependencies: P3 Agent G runner/executor/retry work, Agent K P3 runner, Agent L P3 acceptance, and full repo tests.
- Implemented: Final worklog handoff entry with full verification commands and results.
- Not implemented: No new functionality in this handoff step.
- Schema/interface changes: None.
- Downstream impact: Future P4 work can treat P3 acceptance as passing in this checkout.
- Tests added: None.
- Tests passed: Full unit suite passed: 96 passed, 1 skipped. Full acceptance suite passed: 6 passed in 115.39s. `python3 -m compileall amsrr -q` passed. `git diff --check` passed.
- Handoff notes: P3 remains simplified deterministic assembly integration; P4 must still integrate contact candidates, π_H, π_L, QP/PID/controller execution, and full grasp/carry success criteria.
- Open questions: Whether to add Isaac-backed assembly validation before or during P4 remains a planning decision, not a blocker for the current simplified P3 gate.

#### 2026-07-08
- Scope: Add P3 order 5 acceptance gate for v0.4 Section 24.4.
- Files changed:
  - `amsrr/acceptance/p3_acceptance.py`
  - `amsrr/acceptance/__init__.py`
  - `tests/acceptance/test_p3_acceptance.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Upstream dependencies: Agent K P3 runner, Agent G assembly runner/executor, P2 design distribution/policy, and `EpisodeArchive` JSONL roundtrip helpers.
- Implemented: `P3AcceptanceCriteria`, `P3AcceptanceReport`, `run_p3_acceptance`, assembly success-rate gate, construction-state consistency gate, explicit retry probe, explicit abort probe, archive roundtrip validation, and acceptance test.
- Not implemented: P4 full grasp/carry, Isaac execution, learned assembly, π_H/π_L/QP/PID execution, or actuator commands.
- Schema/interface changes: None to persisted schemas.
- Downstream impact: P3 deterministic assembly integration now has a reproducible pass/fail milestone gate.
- Tests added: `test_p3_acceptance_section_24_4`.
- Tests passed: P3 runner and P3 acceptance targeted tests passed: 3 passed. `python3 -m compileall amsrr -q` passed.
- Handoff notes: The acceptance gate intentionally treats retry/abort probes separately from normal success-rate episodes so deterministic success runs do not need random failures.
- Open questions: None currently.

#### 2026-07-08
- Scope: Mark P2 complete by wrapping the Section 24.3 design-level acceptance gate in an explicit milestone completion report.
- Files changed:
  - `amsrr/acceptance/__init__.py`
  - `amsrr/acceptance/p2_completion.py`
  - `tests/acceptance/test_p2_completion.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Upstream dependencies: `run_p2_acceptance`, Agent K P2 design runner, Agent E P2 design policy/variants, Agent F feasibility labels, and v0.4 Section 24.3.
- Implemented: `P2CompletionCriteria`, `P2CompletionReport`, `run_p2_completion`, explicit completion checks, and a 1000-episode P2 completion acceptance test.
- Not implemented: P3 assembly execution, P4 end-to-end grasp/carry success, Isaac Sim execution, learned π_D training, π_H/π_L/controller execution inside the P2 gate.
- Schema/interface changes: None to persisted schemas.
- Downstream impact: Future work can use `run_p2_completion(...).passed` as the local signal that the P2 design-level milestone is complete before advancing to P3/P4.
- Tests added: `test_p2_completion_milestone_section_24_3`.
- Tests passed: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/acceptance/test_p2_completion.py -q` passed: 1 passed in 23.62s. Full unit suite passed: 80 passed, 1 skipped. Full acceptance suite passed: 3 passed in 88.20s. `python3 -m compileall amsrr -q` passed. `git diff --check` passed.
- Handoff notes: The completion report mirrors Section 24.3 exactly and deliberately does not claim actuator-command or simulator-task success.
- Open questions: None currently.

#### 2026-07-08
- Scope: Implement P2 Section 24.3 acceptance reporting and tests.
- Files changed:
  - `amsrr/acceptance/__init__.py`
  - `amsrr/acceptance/p2_acceptance.py`
  - `tests/acceptance/test_p2_acceptance.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Upstream dependencies: v0.4 P2 acceptance criteria, Agent K P2 design runner, Agent E P2 design policy and variants, Agent F FeasibilityChecker labels/margins, and `EpisodeArchive`.
- Implemented: `P2AcceptanceCriteria`, `P2AcceptanceReport`, `run_p2_acceptance`, Section 24.3 metric checks, synthetic closed-loop invalid probe, archive label validation, and a 1000-episode acceptance test.
- Not implemented: P2 completion wrapper in this entry, learned π_D training, Isaac validation, π_H/π_L/controller execution, or assembly integration.
- Schema/interface changes: None to persisted schemas.
- Downstream impact: P2 design-level validity and feasibility-label persistence can be checked mechanically before later phases.
- Tests added: `test_p2_acceptance_section_24_3`.
- Tests passed: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/acceptance/test_p2_acceptance.py -q` passed: 1 passed in 23.45s. Full unit suite passed: 80 passed, 1 skipped. Full acceptance suite passed: 2 passed in 64.82s. `python3 -m compileall amsrr -q` passed. `git diff --check` passed.
- Handoff notes: Closed-loop rejection is tested with an explicit synthetic invalid design because the normal P2 candidate builders intentionally emit connected trees.
- Open questions: None currently.

#### 2026-07-08
- Scope: Implement P1 Section 24.2 acceptance reporting and tests.
- Files changed:
  - `amsrr/acceptance/__init__.py`
  - `amsrr/acceptance/p1_acceptance.py`
  - `tests/acceptance/test_p1_acceptance.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Upstream dependencies: v0.4 P1 acceptance criteria, `P1SimplifiedRunner`, P1 task distribution config, `EpisodeArchive`, `SimplifiedGraspCarryEnv`, ContactCandidateSampler, pi_H/pi_L/controller baselines.
- Implemented: `P1AcceptanceCriteria`, `P1AcceptanceReport`, `run_p1_acceptance`, and a 1000-episode acceptance test that checks success rate, zero crashes, non-empty contact candidates on randomized valid objects, and archive roundtrip counts.
- Not implemented: Isaac Lab backend validation, learned pi_L training, held-out object evaluation, and high-fidelity contact physics checks.
- Schema/interface changes: None to persisted schemas.
- Downstream impact: P1 completion can now be verified by running `tests/acceptance/test_p1_acceptance.py`; future simulator backends can reuse the same criteria.
- Tests added: `test_p1_acceptance_section_24_2`.
- Tests passed: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/acceptance/test_p1_acceptance.py -q` passed: 1 passed in 41.26s. `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q` passed: 67 passed, 1 skipped. `python3 -m compileall amsrr -q` passed. `git diff --check` passed.
- Handoff notes: Acceptance currently targets the simplified backend explicitly. Keep Isaac Lab checks separate under Agent J / simulator integration.
- Open questions: None currently.

### Agent K: P1 Task Distribution, Runner, Metrics, and Logging

#### 2026-07-08
- Scope: Add P3 order 4 assembly evaluation runner, config, archive metrics, and tests.
- Files changed:
  - `amsrr/training/p3_assembly_runner.py`
  - `amsrr/training/__init__.py`
  - `configs/training/p3_assembly_grasp_carry.yaml`
  - `tests/unit/training/test_p3_assembly_runner.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Upstream dependencies: P2 design distribution/policy, Agent G assembly runner/executor, PhysicalModel builder, IRGBuilder, InteractionEnvelopeExtractor, and `EpisodeArchive` JSONL helpers.
- Implemented: `P3_ASSEMBLY_RUNNER_VERSION`, `P3AssemblyRunnerConfig`, `P3AssemblyRunnerResult`, `load_p3_assembly_runner_config`, `P3AssemblyEvaluationRunner`, P3 config file, assembly archive metrics, and unit tests.
- Not implemented: P3 acceptance gate, retry/abort acceptance probes, Isaac execution, π_H/π_L/QP/PID execution, or actuator commands.
- Schema/interface changes: None to persisted schemas.
- Downstream impact: Agent L can implement P3 Section 24.4 acceptance over this runner.
- Tests added: `test_p3_assembly_runner_collects_successful_assembly_archives`, `test_p3_assembly_runner_config_loader`.
- Tests passed: P3 runner plus related P2/assembly targeted tests passed: 17 passed. `python3 -m compileall amsrr -q` passed.
- Handoff notes: The runner stores the source `AssemblyPlan` as JSON-compatible data in `EpisodeArchive.assembly_plan`; full `AssemblyRunReport` records remain runtime evaluation objects.
- Open questions: None currently.

#### 2026-07-08
- Scope: Implement P2 design evaluation distribution, runner, metrics, and EpisodeArchive feasibility-label logging.
- Files changed:
  - `amsrr/training/__init__.py`
  - `amsrr/training/p2_design_distribution.py`
  - `amsrr/training/p2_design_runner.py`
  - `configs/training/p2_design_grasp_carry.yaml`
  - `tests/unit/training/test_p2_design_runner.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Upstream dependencies: v0.4 P2 acceptance criteria and logging guidance, Agent E P2 design policy/variants, Agent F FeasibilityChecker labels/margins, IRGBuilder, InteractionEnvelopeExtractor, PhysicalModel, and EpisodeArchive.
- Implemented: P2 grasp/carry design distribution, config loader, design evaluation runner, selected-design archive writing, feasibility label/margin metric extraction, P2 config file, package exports, and unit tests.
- Not implemented: P2 completion wrapper in this entry, per-candidate archive rows, learned training loops, Isaac recorder, π_H/π_L/controller execution, or actuator-command logging.
- Schema/interface changes: None to persisted schemas.
- Downstream impact: Agent L P2 acceptance/completion can aggregate design-level validity and labels directly from archived `EpisodeArchive.feasibility_result` values.
- Tests added: `test_p2_design_distribution_randomizes_and_marks_metadata`, `test_p2_design_runner_collects_feasibility_archives`, `test_p2_design_runner_config_loader`.
- Tests passed: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/training/test_p2_design_runner.py -q` passed: 3 passed. Full unit suite passed: 80 passed, 1 skipped. `python3 -m compileall amsrr -q` passed. `git diff --check` passed.
- Handoff notes: The runner archives the selected candidate's feasibility result; rejected-candidate supervision is available through `P2DesignSelection` but not yet emitted as separate dataset rows.
- Open questions: None currently.

#### 2026-07-08
- Scope: Implement P1 order 8 task distribution, runner, metrics, and EpisodeArchive logging for the simplified grasp/carry backend.
- Files changed:
  - `amsrr/logging/__init__.py`
  - `amsrr/logging/episode_archive.py`
  - `amsrr/training/__init__.py`
  - `amsrr/training/p1_task_distribution.py`
  - `amsrr/training/p1_runner.py`
  - `configs/training/p1_grasp_carry_distribution.yaml`
  - `tests/unit/training/test_p1_runner.py`
- Upstream dependencies: v0.4 domain randomization and EpisodeArchive guidance, existing TaskSpec and policy/controller schemas, `SimplifiedGraspCarryEnv`, and config/hash utilities.
- Implemented: Config-loaded P1 grasp/carry distribution, object size/mass/friction/initial-pose/target-pose sampling, per-episode runner over the simplified env, batch success/crash/failure metrics, EpisodeArchive dataclass, reproducibility metadata, JSONL write/read helpers, and unit tests.
- Not implemented: Learned training loop, replay buffer, dataset sharding, Isaac recorder, wind/sensor/thrust-scale randomization, non-box object shape sampling, or large-scale filesystem dataset management.
- Schema/interface changes: None to persisted schemas.
- Downstream impact: Simplified P1 runs now produce archives and metrics suitable for debugging and later dataset/training integration.
- Tests added: `test_p1_distribution_randomizes_configured_fields`, `test_p1_runner_collects_metrics_and_archives`, `test_p1_runner_config_loader`.
- Tests passed: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/training/test_p1_runner.py -q` passed: 3 passed. `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q` passed: 67 passed, 1 skipped. `python3 -m compileall amsrr -q` passed.
- Handoff notes: `EpisodeArchive` includes a `reproducibility` map as an implementation supplement for Section 25.3. Config currently lives at `configs/training/p1_grasp_carry_distribution.yaml`.
- Open questions: None currently.

### Agent JP1: Simplified Grasp-Carry Simulation Env

#### 2026-07-08
- Scope: Implement P1 order 7 simplified grasp/carry simulation environment for interface-backed crash-free validation before Isaac Lab binding.
- Files changed:
  - `amsrr/simulation/__init__.py`
  - `amsrr/simulation/base.py`
  - `amsrr/simulation/simplified_grasp_carry_env.py`
  - `tests/unit/simulation/test_simplified_grasp_carry_env.py`
- Upstream dependencies: v0.4 simplified contact and P1 acceptance requirements, existing TaskSpec/RuntimeObservation schemas, IRGBuilder, InteractionEnvelopeExtractor, fixed/simple design policy, ContactCandidateSampler, GraspCarryBaselinePlanner, BaselineLowLevelPolicy, and QPIDController.
- Implemented: `SimulationEnvBase`, simplified reset/step/get-runtime-observation boundary, deterministic pipeline build artifacts, kinematic/fixed-joint grasp attach approximation, active object-target tracking, contact-state emission, task-progress metrics, per-episode result summaries, batch 1000-episode runner, and unit tests.
- Not implemented: Isaac Lab/Isaac Sim integration, physics contact solver, friction/slip dynamics, collision geometry stepping, stochastic actuator faults, dataset archive writer, or training loop integration.
- Schema/interface changes: None to persisted schemas.
- Downstream impact: P1 can validate no schema/checker/controller crashes over 1000 simplified episodes. Later simulator backends can implement `SimulationEnvBase` while preserving existing policy/controller contracts.
- Tests added: `test_simplified_grasp_carry_env_matches_base_protocol`, `test_simplified_grasp_carry_env_runs_policy_controller_episode`, `test_simplified_grasp_carry_1000_episodes_crash_free`.
- Tests passed: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/simulation/test_simplified_grasp_carry_env.py -q` passed: 3 passed, including 1000 simplified episodes with 0 crashes. `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q` passed: 64 passed, 1 skipped. `python3 -m compileall amsrr -q` passed.
- Handoff notes: The env deliberately keeps simulator-specific dependencies out of `amsrr/simulation`; the current backend is deterministic and suitable for interface smoke and acceptance checks, not physics validation.
- Open questions: None currently.

### Agent I: pi_L + QP/PID Interfaces

#### 2026-07-08
- Scope: Implement P1 order 6 Agent I interfaces that map active pi_H knots/runtime observations to `PolicyCommand`, then to controller-owned `ControllerCommand` outputs.
- Files changed:
  - `amsrr/controllers/__init__.py`
  - `amsrr/controllers/controller_base.py`
  - `amsrr/controllers/qp_allocator_interface.py`
  - `amsrr/controllers/qpid_controller.py`
  - `amsrr/policies/__init__.py`
  - `amsrr/policies/low_level_policy_base.py`
  - `tests/unit/controllers/test_qpid_controller.py`
  - `tests/unit/policies/test_low_level_baseline.py`
- Upstream dependencies: v0.4 Section 20 pi_L/controller split, Agent H `ContactWrenchTrajectory`, existing runtime/physical-model/policy schemas, and `PolicyCommandBiasBuilder`.
- Implemented: `LowLevelPolicyContext`, `LowLevelPolicyBase`, `BaselineLowLevelPolicyConfig`, `BaselineLowLevelPolicy`, runtime-time active knot selection, object target residual wrench proxy, active contact tracking bias, controller-status residual suppression, `ControllerContext`, `ControllerBase`, QP allocator problem/result/backend protocol, bounded vertical rotor allocator, QPID controller scaffold, vectoring joint clipping, PD joint torque proxy, dock-mechanism hold commands, and focused tests.
- Not implemented: Learned pi_L head, exact multi-axis/vectoring/contact QP, OSQP/C++ backend, high-fidelity object/contact dynamics, simulator execution, or training integration.
- Schema/interface changes: None to persisted schemas.
- Downstream impact: P1 simplified grasp-carry simulation can run through pi_H trajectory, pi_L intent, desired-reference builder, and controller command scaffolding without introducing a simulator dependency yet.
- Tests added: `test_baseline_low_level_policy_outputs_policy_command`, `test_baseline_low_level_policy_selects_knot_from_runtime_time`, `test_baseline_low_level_policy_suppresses_residual_when_controller_infeasible`, `test_select_active_knot_rejects_empty_trajectory`, `test_bounded_vertical_rotor_allocator_feasible_and_unsupported_residual`, `test_bounded_vertical_rotor_allocator_reports_infeasible_clip`, `test_qpid_controller_outputs_controller_command`, `test_qpid_controller_reports_infeasible_vertical_wrench`.
- Tests passed: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/policies/test_low_level_baseline.py -q` passed: 4 passed. `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit/controllers/test_qpid_controller.py -q` passed: 4 passed. `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q` passed: 61 passed, 1 skipped. `python3 -m compileall amsrr -q` passed.
- Handoff notes: The pi_L baseline intentionally emits residual intent only. `ControllerCommand` fields are produced only in the controller layer. The current allocator is a simplified bounded vertical allocator and reports unsupported wrench residuals for future exact QP replacement.
- Open questions: None currently.

### Agent H: pi_H Baseline Planner

#### 2026-07-08
- Scope: Implement a deterministic baseline pi_H planner for P1 grasp/carry after ContactCandidateSampler and selected-assignment feasibility.
- Files changed:
  - `amsrr/policies/__init__.py`
  - `amsrr/policies/high_level_policy_base.py`
  - `amsrr/policies/contact_wrench_trajectory.py`
  - `tests/unit/policies/test_high_level_baseline.py`
- Upstream dependencies: `ContactCandidateSet` group proposals, selected-assignment feasibility, IRG state targets/contact slots, InteractionEnvelope, MorphologyGraph, and existing policy schemas.
- Implemented: `HighLevelPolicyContext`, `HighLevelPolicyBase`, `BaselineTrajectoryPlannerConfig`, `GraspCarryBaselinePlanner`, `select_feasible_assignments`, five-knot deterministic grasp/carry trajectory generation, object goal extraction, free-anchor pose targets, wrench target scaffolding, and feasibility cache integration.
- Not implemented: Learned pi_H heads, trajectory optimization, multi-knot re-planning from live observations, exact wrench/QP optimization, contact schedule search beyond group-proposal attempts, or simulator execution.
- Schema/interface changes: None to persisted schemas.
- Downstream impact: pi_L and controller interfaces can now consume active `InteractionKnot`s and `ContactAssignment`s from a full `ContactWrenchTrajectory`.
- Tests added: `test_grasp_carry_baseline_planner_outputs_contact_wrench_trajectory`, `test_select_feasible_assignments_uses_grasp_pair_group`.
- Tests passed: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q` passed: 53 passed, 1 skipped. `python3 -m compileall amsrr -q` passed.
- Handoff notes: Planner selection uses `evaluate_selected_assignment_feasibility`, so infeasible attempted groups are cached on the candidate set. The returned trajectory never contains final actuator commands.
- Open questions: None currently.

### Agent H/F: Selected Assignment Feasibility Proxy

#### 2026-07-08
- Scope: Add assignment-level feasibility checks for selected `ContactAssignment` sets only, without subset enumeration.
- Files changed:
  - `amsrr/policies/__init__.py`
  - `amsrr/policies/assignment_feasibility.py`
  - `tests/unit/policies/test_contact_candidate_interfaces.py`
- Upstream dependencies: Existing `ContactCandidateSet`, selected `ContactAssignment`, pairwise conflict matrix, candidate unary validity, and v0.4 assignment-level feasibility guidance.
- Implemented: `evaluate_selected_assignment_feasibility`, violation code constants, slot min/max cardinality checks, assignment/candidate consistency checks, selected pairwise conflict checks, duplicate selected-candidate checks, grasp-opposition residual proxy, friction margin proxy, optional explicit wrench/QP/collision residual hooks, and deterministic cache update.
- Not implemented: Exact force closure, exact support polygon/contact support ratio, full multi-contact collision, exact QP allocation, or π_H trajectory generation.
- Schema/interface changes: None to persisted schemas.
- Downstream impact: π_H can now select candidate assignments and receive deterministic feasibility/cache labels before later exact solver integration.
- Tests added: `test_selected_assignment_feasibility_accepts_opposing_grasp_pair`, `test_selected_assignment_feasibility_rejects_cardinality_and_pair_conflict`, `test_selected_assignment_feasibility_rejects_non_opposing_grasp_normals`.
- Tests passed: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q` passed: 51 passed, 1 skipped. `python3 -m compileall amsrr -q` passed.
- Handoff notes: The existing `evaluate_assignment_level_qp` smoke helper remains backward compatible. The new evaluator should be used when π_H has a selected assignment set and wants cardinality/pairwise/wrench-proxy labels in addition to optional QP residual labels.
- Open questions: None currently.

### Agent H: ContactCandidateSampler

#### 2026-07-08
- Scope: Implement P1 morphology-conditioned contact candidate sampling and group-proposal scaffolding only.
- Files changed:
  - `amsrr/policies/__init__.py`
  - `amsrr/policies/contact_candidate_set.py`
  - `amsrr/policies/contact_candidate_sampler.py`
  - `tests/unit/policies/test_contact_candidate_sampler.py`
- Upstream dependencies: `TaskSpec`, IRG ContactSlots, `InteractionEnvelope`, `MorphologyGraph` RobotAnchors, `GeometryDescriptor` / ContactRegionGraph, and existing `ContactCandidateSet` helper functions.
- Implemented: `ContactCandidateSamplerConfig`, `ContactCandidateSampler`, deterministic candidate IDs, entity-pose world transform, unary smoke scores, compatible-anchor filtering, `build_group_proposals`, `grasp_pair` group proposals, `support_set` fallback proposals, optional group-proposal support in `build_contact_candidate_set`, and package exports.
- Not implemented: Learned candidate encoder/scorer, task-specific advanced sampling quotas, exact reachability, exact local collision/clearance, assignment-level wrench/friction/QP feasibility, π_H selection, or simulator/runtime contact verification.
- Schema/interface changes: None to persisted schemas.
- Downstream impact: π_H baseline can now select over finite candidate pools with slot coverage and group hints. Assignment-level evaluators can later populate `assignment_feasibility_cache`.
- Tests added: `test_contact_candidate_sampler_returns_non_empty_grasp_carry_candidates`, `test_contact_candidate_sampler_builds_grasp_pair_group_proposals`, `test_contact_candidate_sampler_uses_robot_anchor_associations`.
- Tests passed: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q` passed: 48 passed, 1 skipped. `python3 -m compileall amsrr -q` passed.
- Handoff notes: Candidate generation preserves `ContactSlotID -> RobotAnchorID -> ContactCandidateID`; candidates are generated only for anchors already associated with the slot by π_D. Group proposals deliberately do not imply full task feasibility.
- Open questions: None currently.

### Agent G: π_A GraphEditAssemblyPlanner

#### 2026-07-08
- Scope: Add P3 order 3 retry/abort behavior to the deterministic assembly runner.
- Files changed:
  - `amsrr/assembly/assembly_runner.py`
  - `amsrr/assembly/simplified_executor.py`
  - `tests/unit/assembly/test_assembly_runner.py`
  - `tests/unit/assembly/test_simplified_executor.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Upstream dependencies: Existing Agent G assembly runner/executor and v0.4 `AssemblyStep.step_type` values.
- Implemented: Configurable retry limit, synthetic retry steps, synthetic abort steps, retry/abort counts, aborted status, executed step-type tracing, and fail-once support in the simplified executor.
- Not implemented: Motion replanning, learned recovery policy, detach release gates, physical docking verification, controller/QP integration, or Isaac execution.
- Schema/interface changes: None to persisted schemas.
- Downstream impact: P3 acceptance can test successful transient retry and persistent-failure abort paths without changing source assembly plans.
- Tests added: `test_assembly_runner_can_disable_retry_for_single_failure_stop`, `test_simplified_executor_fail_once_allows_runner_retry_success`.
- Tests passed: Agent G targeted assembly tests passed: 12 passed. `python3 -m compileall amsrr -q` passed.
- Handoff notes: Runtime retry/abort steps are represented in `AssemblyRunReport.executed_step_types`; `AssemblyPlan.steps` remains the source graph-edit plan.
- Open questions: None currently.

#### 2026-07-08
- Scope: Add P3 order 2 simplified assembly executor backend.
- Files changed:
  - `amsrr/assembly/simplified_executor.py`
  - `amsrr/assembly/__init__.py`
  - `tests/unit/assembly/test_simplified_executor.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Upstream dependencies: Agent G runner/core state transitions, `AssemblyExecutorInterface`, and existing construction-state helpers.
- Implemented: `SimplifiedAssemblyExecutorConfig`, `SimplifiedAssemblyExecutor`, default successful step execution, `verify_attach` updated-state return when target graph is provided, per-step smoke metrics, and deterministic failure injection by step id/type.
- Not implemented: Retry/abort state-machine execution, P3 runner/acceptance, Isaac execution, physical docking dynamics, or controller/QP integration.
- Schema/interface changes: None to persisted schemas.
- Downstream impact: P3 runner and acceptance can use this executor for deterministic success probes and controlled failure probes.
- Tests added: `test_simplified_executor_runs_full_assembly_and_returns_updated_state`, `test_simplified_executor_can_inject_step_type_failure`, `test_simplified_executor_success_without_target_graph_uses_runner_state_transition`.
- Tests passed: Agent G targeted assembly tests passed: 10 passed. `python3 -m compileall amsrr -q` passed.
- Handoff notes: Failure injection is executor-local; policy-level retry/abort handling remains the next order.
- Open questions: None currently.

#### 2026-07-08
- Scope: Add P3 order 1 assembly execution core on top of the existing graph-edit planner.
- Files changed:
  - `amsrr/assembly/assembly_runner.py`
  - `amsrr/assembly/__init__.py`
  - `tests/unit/assembly/test_assembly_runner.py`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Upstream dependencies: Existing Agent G planner/state dataclasses, `AssemblyExecutorInterface`, v0.4 Section 17 contracts, and P2 morphology variants.
- Implemented: `AssemblyRunnerConfig`, `AssemblyRunReport`, `AssemblyRunner`, automatic successful `verify_attach` state transition, final target graph consistency metrics, success/failure report serialization, and focused unit tests.
- Not implemented: Simplified executor, retry/abort policy execution, P3 runner/acceptance, Isaac execution, or physical docking verification.
- Schema/interface changes: None to persisted schemas.
- Downstream impact: Later P3 work can plug in deterministic simplified executors and aggregate assembly success/state-consistency metrics.
- Tests added: `test_assembly_runner_completes_plan_and_updates_construction_state`, `test_assembly_runner_stops_on_failed_step_without_completing_graph`, `test_assembly_runner_resumes_from_partial_construction_state`.
- Tests passed: Agent G targeted assembly tests passed: 7 passed. `python3 -m compileall amsrr -q` passed.
- Handoff notes: `state_matches_target` checks assembled module IDs, dock edge endpoint/port keys, and occupied target ports; target edge latch states remain target metadata rather than the equality criterion.
- Open questions: None currently.

#### 2026-07-08
- Scope: Implement implementation-order item 12: deterministic π_A assembly planner and construction/execution interface scaffolding.
- Files changed:
  - `amsrr/assembly/__init__.py`
  - `amsrr/assembly/construction_state.py`
  - `amsrr/assembly/graph_edit_planner.py`
  - `amsrr/assembly/control_handoff.py`
  - `amsrr/assembly/executor_interface.py`
  - `tests/unit/assembly/test_graph_edit_planner.py`
- Upstream dependencies: Agent E target `MorphologyGraph` / `DesignOutput`, v0.4 assembly contracts, `Violation`, and existing schema serialization helpers.
- Implemented: `AssemblyStep`, `AssemblyPlan`, `ConstructionState`, `initial_construction_state`, `construction_state_from_current_graph`, `mark_edge_attached`, `GraphEditAssemblyPlanner`, `AssemblyPlannerConfig`, `ControlHandoffManager`, `ControlHandoffRequest`, `AssemblyExecutionResult`, and `AssemblyExecutorInterface`.
- Not implemented: Learned assembly policy, simulator executor, path/motion planner, retry/abort state-machine execution, detach execution gates, QP/PID controller integration, or physical docking verification.
- Schema/interface changes: None to existing persisted schemas. Added assembly-local dataclasses/interfaces matching v0.4.
- Downstream impact: Later P1/P3 code can request deterministic assembly plans for target morphologies and can hand assembly steps to simulator/controller interfaces once those exist.
- Tests added: `test_initial_construction_state_contains_base_only`, `test_graph_edit_planner_builds_deterministic_attach_sequence`, `test_graph_edit_planner_resumes_from_construction_state`, `test_control_handoff_request_for_docking_step`.
- Tests passed: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q` passed: 45 passed, 1 skipped. `python3 -m compileall amsrr -q` passed.
- Handoff notes: The planner expands each target dock edge into four deterministic steps and returns no `next_step` when the target graph already has no remaining dock edges. Construction subgraphs keep only assembled modules/edges, while unattached modules remain in `ConstructionState.unattached_modules` and singleton components.
- Open questions: None currently.

### Agent E: Deterministic Design Teacher + π_D Scaffolding

#### 2026-07-07
- Scope: Implement implementation-order item 11 for P1 fixed/simple morphology: deterministic design teacher and π_D scaffolding only.
- Files changed:
  - `amsrr/policies/__init__.py`
  - `amsrr/policies/design_policy_base.py`
  - `amsrr/policies/design_candidate_generator.py`
  - `amsrr/policies/design_teacher.py`
  - `tests/unit/policies/test_design_teacher.py`
- Upstream dependencies: Existing `MinimalMorphologyBuilder`, `DesignOutput`, IRG ContactSlot semantics, InteractionEnvelopeExtractor, PhysicalModel builder, FeasibilityChecker, and v0.4 π_D action vocabulary.
- Implemented: `DesignPolicyContext`, `DesignPolicyBase` protocol, `FixedSimpleDesignPolicy`, `DesignTeacherVariant`, `DeterministicDesignTeacher`, `DesignTeacherExample`, `DesignCandidateGenerator`, `DesignActionCandidate`, `DesignCandidateStep`, P1 grasp/support teacher variant selection, and STOP-mask smoke checks.
- Not implemented: Learned π_D scoring/sampling, optimized teacher geometry variants, policy training, assembly planning, contact candidate sampling, π_H, π_L, QP/PID controller behavior, simulator integration.
- Schema/interface changes: None to persisted schemas.
- Downstream impact: P1 and Agent H can now request a stable fixed/simple `DesignOutput` with RobotAnchors before generating contact candidates. Agent G can plan assembly against the same target graph.
- Tests added: `test_design_teacher_selects_p1_grasp_support_variant`, `test_design_candidate_trace_masks_stop_until_final_step`, `test_fixed_simple_design_policy_outputs_feasible_stop`.
- Tests passed: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q` passed: 41 passed, 1 skipped. `python3 -m compileall amsrr -q` passed.
- Handoff notes: Teacher trace STOP is masked until the final teacher step. Final STOP validity performs scaffold checks and optionally respects a FeasibilityChecker result. π_D still emits `DesignOutput` only and never controller or actuator commands.
- Open questions: None currently.

### Agent H/I: P0 Interface-Only Smoke Pieces

#### 2026-07-07
- Scope: Add the remaining P0 Section 27.2 smoke pieces for contact candidates, selected assignment feasibility, and policy command bias references.
- Files changed:
  - `amsrr/policies/__init__.py`
  - `amsrr/policies/contact_candidate_set.py`
  - `amsrr/policies/assignment_feasibility.py`
  - `amsrr/controllers/__init__.py`
  - `amsrr/controllers/policy_command_builder.py`
  - `tests/unit/policies/test_contact_candidate_interfaces.py`
  - `tests/unit/controllers/test_policy_command_builder.py`
- Upstream dependencies: Existing contact candidate and policy schemas, v0.4 candidate/π_H/π_L/controller interface contracts.
- Implemented: `build_pairwise_conflict_matrix`, `build_pairwise_compatibility_score`, `build_contact_candidate_set`, deterministic `assignment_key_from_assignments`, `evaluate_assignment_level_qp`, and `PolicyCommandBiasBuilder`.
- Not implemented: Full morphology-conditioned candidate sampling, learned candidate encoder/scorer, π_H baseline planner, π_L baseline policy, actual QP allocation, PID/controller actuator outputs.
- Schema/interface changes: None to persisted schemas.
- Downstream impact: Future Agent H/I implementations can replace helpers with richer implementations while keeping tested schema boundaries and no direct actuator output from π_L.
- Tests added: `test_contact_candidate_pairwise_conflict_matrix`, `test_assignment_level_qp_infeasible_case`, `test_policy_command_bias_builder`.
- Tests passed: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q` passed: 38 passed, 1 skipped. `python3 -m compileall amsrr -q` passed.
- Handoff notes: `ASSIGNMENT_QP_INFEASIBLE_CODE` is `E_ASSIGNMENT_QP_INFEASIBLE`, matching v0.4 Appendix C. `PolicyCommandBiasBuilder` merges π_H priority weights with π_L command weights, with PolicyCommand taking precedence.
- Open questions: None currently.

### Agent E/F: Minimal MorphologyGraph + Feasibility Hard-Check Scaffolding

#### 2026-07-07
- Scope: Build a connected minimal MorphologyGraph/DesignOutput from TaskSpec + IRG + PhysicalModel, and evaluate design-level hard feasibility checks.
- Files changed:
  - `amsrr/morphology/__init__.py`
  - `amsrr/morphology/graph.py`
  - `amsrr/feasibility/__init__.py`
  - `amsrr/feasibility/checker.py`
  - `amsrr/feasibility/violation_codes.py`
  - `tests/unit/morphology/test_minimal_morphology_builder.py`
  - `tests/unit/feasibility/test_feasibility_checker.py`
- Upstream dependencies: Agent B PhysicalModel and ModuleCapabilityToken, Agent D IRG ContactSlots, v0.4 MorphologyGraph/DesignOutput/FeasibilityResult schemas.
- Implemented: Minimal module chain generation, dock port replication/compatibility masking, structural dock edges, robot anchor creation from ContactSlots, slot-anchor binding priors, design action trace, violation code constants, and design-level hard checks for required P0 validity conditions.
- Not implemented: Learned π_D, candidate enumeration policy head, deterministic design teacher variants beyond the minimal seed, exact collision checking, exact QP hover feasibility, assignment-level feasibility, simulator integration.
- Schema/interface changes: None to persisted schemas.
- Downstream impact: ContactCandidateSampler can start from known RobotAnchors and slot-anchor priors; later FeasibilityChecker work can refine coarse checks without changing result schema.
- Tests added: Morphology builder and feasibility checker tests listed in the global entry.
- Tests passed: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q` passed: 35 passed, 1 skipped. `python3 -m compileall amsrr -q` passed.
- Handoff notes: The checker version is `p0_agent_ef_v1`. The minimal builder creates optional slot anchors too when a ContactSlot allows them, but required coverage is checked against required slots and their `min_count_group`.
- Open questions: None currently.

### Agent A/L: SharedInteractionWorkspace Tensor/Mask Contract

#### 2026-07-07
- Scope: Define and validate the internal NN tensor contract that fuses per-modality token groups with masks, source ids, group slices, and learned-query specs.
- Files changed:
  - `amsrr/schemas/workspace.py`
  - `amsrr/encoders/__init__.py`
  - `amsrr/encoders/workspace_builder.py`
  - `tests/unit/schemas/test_workspace.py`
  - `tests/unit/encoders/test_workspace_builder.py`
- Upstream dependencies: Agent A workspace schema foundation, Agent D/A InteractionEnvelopeEncoder output, v0.4 SharedInteractionWorkspace and LearnedQuerySpec contract.
- Implemented: `WorkspaceTokenGroup`, stricter `SharedInteractionWorkspace` group mask validation, optional contact candidate group support, recommended query specs, empty group factory, encoder-output-to-group adapter, and shared workspace assembly.
- Not implemented: Learned query tensors/parameters, attention pooling, fusion encoder, policy heads, modality-specific encoders beyond the existing InteractionEnvelopeEncoder.
- Schema/interface changes: Internal workspace schema validation was strengthened. `group_masks` are now required for every group slice and must match the corresponding global mask slice.
- Downstream impact: Heads can rely on `source_ids` and `group_slices` to map outputs back to source schema ids. π_H contexts can opt into the optional `contact_candidates` group.
- Tests added: Workspace group/mask/query tests and workspace builder tests listed in the global entry.
- Tests passed: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q` passed: 30 passed, 1 skipped. `python3 -m compileall amsrr -q` passed.
- Handoff notes: `SharedInteractionWorkspaceBuilder` fills missing required groups with zero-width empty groups, so partial modality implementations can still produce a valid full workspace.
- Open questions: None currently.

### Agent D/A: InteractionEnvelopeExtractor + InteractionEnvelopeEncoder

#### 2026-07-07
- Scope: Aggregate compact interaction requirements from IRG and expose deterministic encoder tokens for the envelope modality.
- Files changed:
  - `amsrr/irg/__init__.py`
  - `amsrr/irg/envelope_extractor.py`
  - `amsrr/encoders/__init__.py`
  - `amsrr/encoders/interaction_envelope_encoder.py`
  - `tests/unit/irg/test_envelope_extractor.py`
  - `tests/unit/encoders/test_interaction_envelope_encoder.py`
- Upstream dependencies: Agent A `InteractionEnvelope` and `SharedInteractionWorkspace` shape helper schemas; Agent D IRG node/edge conventions; v0.4 envelope and encoder contracts.
- Implemented: Contact count range aggregation, contact mode aggregation, target region set extraction, wrench summary extraction, support/vertical thrust ratio summary hooks, precision/duration/capability extraction, branch option extraction for future fallback/mutually-exclusive IRGs, padded envelope token contract with masks and source ids.
- Not implemented: Full multimodal SharedInteractionWorkspace assembly, learned MLP/Transformer modules, query pooling parameters, generic constraint-threshold schema beyond fields currently available in `InteractionEnvelope`.
- Schema/interface changes: No persisted schema changes. Added internal encoder output dataclass for the interaction-envelope modality.
- Downstream impact: Future policy scaffolding can use envelope token groups without raw dict reinterpretation. Full workspace assembly remains the next implementation-order step.
- Tests added: `test_interaction_envelope_extract`, `test_interaction_envelope_extracts_all_task_families`, `test_interaction_envelope_encoder_contract`, `test_interaction_envelope_encoder_batch_padding`.
- Tests passed: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q` passed: 23 passed, 1 skipped. `python3 -m compileall amsrr -q` passed.
- Handoff notes: `InteractionEnvelopeEncoder` defaults to `backend_type="mlp_embedding"` when no dedicated backend key is provided. It emits deterministic scalar features; learned weights belong to later model code.
- Open questions: None currently.

### Agent D: IRGBuilder + InteractionTemplates

#### 2026-07-07
- Scope: Compile TaskSpec plus GeometryDescriptor-derived contact regions into a single typed InteractionRequirementGraph for all P0 task families.
- Files changed:
  - `amsrr/irg/__init__.py`
  - `amsrr/irg/irg_builder.py`
  - `amsrr/irg/validator.py`
  - `amsrr/irg/templates/__init__.py`
  - `amsrr/irg/templates/base.py`
  - `amsrr/irg/templates/free_flight.py`
  - `amsrr/irg/templates/object_grasp_carry.py`
  - `amsrr/irg/templates/valve_operation.py`
  - `amsrr/irg/templates/perching_manipulation.py`
  - `amsrr/irg/templates/contact_mediated_locomotion.py`
  - `tests/unit/irg/test_irg_builder.py`
- Upstream dependencies: Agent A schema dataclasses and enum validation, Agent C GeometryProcessor contact regions, v0.4 IRG and template contracts.
- Implemented: Deterministic node IDs and edge ordering, task/phase/contact-region/contact-slot/wrench/state/constraint/capability node generation, typed cross edges, structural validation, phase-label mapping, and smoke-valid IRGs for all five P0 task families.
- Not implemented: InteractionEnvelope extraction, task-aware geometry re-extraction beyond current GeometryProcessor descriptors, exact valve rim/handle segmentation, final contact/candidate selection, robot anchor assignment, morphology generation, trajectory generation, actuator commands.
- Schema/interface changes: None.
- Downstream impact: Envelope extraction can derive contact count ranges, modes, region sets, wrench requirements, state targets, constraints, and capability requirements directly from the IRG.
- Tests added: `test_phase_label_to_phase_type_mapping`, `test_irg_builder_grasp_carry_valid`, `test_irg_builder_all_task_families_smoke`.
- Tests passed: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q` passed: 19 passed, 1 skipped. `python3 -m compileall amsrr -q` passed.
- Handoff notes: `IRGBuilder.build_with_scene_graph()` returns both IRG and normalized SceneGraph for debugging. Non-template-required environment/obstacle descriptors are resolved lazily, so the v0.4 grasp/carry example can build even though it references `floor_geom` without declaring that geometry.
- Open questions: None currently.

### Agent C: GeometryProcessor

#### 2026-07-07
- Scope: Convert `GeometrySpec` references into `GeometryDescriptor`, learning-side patch/region tokens, and hashed exact/collision geometry refs for P0 primitives and mesh smoke.
- Files changed:
  - `amsrr/geometry/__init__.py`
  - `amsrr/geometry/asset_resolver.py`
  - `amsrr/geometry/surface_patch_graph.py`
  - `amsrr/geometry/contact_region_extractor.py`
  - `amsrr/geometry/geometry_processor.py`
  - `tests/unit/geometry/test_geometry_processor.py`
- Upstream dependencies: Agent A schema dataclasses, v0.4 GeometryProcessor contract, existing `module_urdf/mesh/battery_1.STL` smoke asset.
- Implemented: Primitive analytic decomposition for box, sphere, cylinder, and capsule; box face region coverage; STL binary/ascii and OBJ smoke mesh summary; normal-cluster mesh patch aggregation; path-free descriptor refs; deterministic surface/contact graph edge construction.
- Not implemented: Full mesh repair, curvature estimation, rim extraction, convex decomposition, SDF surface sampling, point cloud reconstruction, task-template-specific rim/edge extraction.
- Schema/interface changes: None.
- Downstream impact: IRGBuilder can request object surface contact regions for primitives and receive non-empty mesh patch clusters for mesh objects.
- Tests added: `test_geometry_processor_box_regions`, `test_geometry_processor_mesh_smoke`.
- Tests passed: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q` passed: 16 passed, 1 skipped.
- Handoff notes: `GeometryDescriptor.collision_ref` and `exact_geometry_ref` use hash URIs rather than raw filesystem paths. Asset paths remain resolver inputs only.
- Open questions: None currently.

### Agent B: URDF / PhysicalModel

#### 2026-07-07
- Scope: Parse Holon URDF/xacro XML, load thrust limits, build `PhysicalModel`, derive dock ports and rotor models, and report module capability features.
- Files changed:
  - `assets/robots/holon/holon.urdf`
  - `amsrr/robot_model/__init__.py`
  - `amsrr/robot_model/urdf_loader.py`
  - `amsrr/robot_model/thrust_model.py`
  - `amsrr/robot_model/physical_model_builder.py`
  - `tests/unit/robot_model/test_urdf_loader.py`
  - `tests/unit/robot_model/test_thrust_model.py`
  - `tests/unit/robot_model/test_physical_model_builder.py`
- Upstream dependencies: Agent A schemas, `configs/robot/robot_model.yaml`, `configs/robot/thrust_model.yaml`, `module_urdf/holon.urdf.xacro`, `module_urdf/README_for_codex.md`.
- Implemented: XML loader for URDF/xacro-derived files, link/joint/inertial/mesh extraction, frame-tree validation, rotor and dock candidate reporting, thrust model validation, runtime `PhysicalModel` builder, dock port derivation from connect point joints, rotor vectoring joint association, capability token derivation.
- Not implemented: Full xacro macro expansion, transform-accurate aggregate inertia, non-mesh collision primitive reconstruction, external metadata config for dock ports beyond name-pattern derivation.
- Schema/interface changes: None.
- Downstream impact: Feasibility and controller work can consume exact link/joint/rotor/dock schema objects. Design/policy work can consume `ModuleCapabilityToken`.
- Tests added: Agent B tests listed above.
- Tests passed: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q` passed: 14 passed, 1 skipped.
- Handoff notes: The normalized runtime URDF is at `assets/robots/holon/holon.urdf`, matching `configs/robot/robot_model.yaml`. The original developer reference xacro remains under `module_urdf/`.
- Open questions: None currently.

### Repository Organization: Codex Handoff Docs

#### 2026-07-07
- Scope: Move Codex-facing specification, instructions, design modification log, and worklog under `for_codex/`.
- Files changed:
  - `for_codex/A-MSRR_codex_ready_spec_v0_4_ja.md`
  - `for_codex/AGENTS.md`
  - `for_codex/AMSRR_design_modification_by_codex.md`
  - `for_codex/WORKLOG.md`
- Upstream dependencies: User request.
- Implemented: Documentation relocation and git staging/commit preparation.
- Not implemented: No source code changes in this worklog entry.
- Schema/interface changes: None.
- Downstream impact: Future handoff readers should look under `for_codex/`.
- Tests added: None.
- Tests passed: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q` passed: 7 passed.
- Handoff notes: Keep design-spec deviations in `for_codex/AMSRR_design_modification_by_codex.md`, separate from chronological worklog entries.
- Open questions: None.

### Agent A: Schemas and Validation

#### 2026-07-07
- Scope: Implement schema dataclasses, enums, serialization/deserialization, and validation helpers for P0 foundation.
- Files changed:
  - `amsrr/schemas/common.py`
  - `amsrr/schemas/task_spec.py`
  - `amsrr/schemas/geometry.py`
  - `amsrr/schemas/irg.py`
  - `amsrr/schemas/interaction_envelope.py`
  - `amsrr/schemas/morphology.py`
  - `amsrr/schemas/physical_model.py`
  - `amsrr/schemas/runtime.py`
  - `amsrr/schemas/policies.py`
  - `amsrr/schemas/feasibility.py`
  - `amsrr/schemas/workspace.py`
  - `amsrr/schemas/contact_candidates.py`
- Upstream dependencies: v0.4 schema sections and P0 acceptance requirements.
- Implemented: Strict dataclass `from_dict` / `to_dict` / JSON roundtrip, enum coercion, nested schema coercion, TaskSpec validation, abstract ContactSlot guard, phase_type validation, workspace tensor shape checks.
- Not implemented: URDF parsing, geometry processing, IRG building, envelope extraction logic, controller/QP logic, simulator integration.
- Schema/interface changes: Initial implementation plus approved `IRGEdgeType.ALLOWS` supplement.
- Downstream impact: Downstream work packages should import schema objects from `amsrr.schemas.*` rather than redefining local dataclasses.
- Tests added: Schema roundtrip, TaskSpec parsing/validation, workspace shape/mask, `allows` edge presence.
- Tests passed: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q` passed: 7 passed.
- Handoff notes: `ContactSlotNode` validation rejects final contact/candidate fields to preserve the IRG abstraction boundary. `SharedInteractionWorkspace` stores `group_slices` as Python `slice` objects and serializes them as JSON mappings.
- Open questions: None currently.

### Agent A/L: Config Loading, Hashing, and Test Harness

#### 2026-07-07
- Scope: Add minimum config and hash utilities needed before robot/geometry/IRG work.
- Files changed:
  - `amsrr/utils/config.py`
  - `amsrr/utils/hashing.py`
  - `configs/robot/robot_model.yaml`
  - `configs/robot/thrust_model.yaml`
  - `configs/training/p0_schema_tests.yaml`
  - `tests/conftest.py`
  - `tests/unit/utils/test_config_hashing.py`
- Upstream dependencies: v0.4 Appendix E minimum example files and Section 25 reproducibility metadata.
- Implemented: YAML/JSON config loading, deterministic canonical JSON hashing, SHA-256 file hashing, minimum robot/thrust/P0 config files.
- Not implemented: Full config schema classes, config merge/override system, command-line runners.
- Schema/interface changes: None beyond initial utility contracts.
- Downstream impact: Robot loader and future dataset/cache keys can use `stable_hash` / `hash_file`.
- Tests added: Config loading and stable hash ordering test.
- Tests passed: Included in 7 passing unit tests.
- Handoff notes: PyYAML is already available in the environment; no dependency install was performed.
- Open questions: None currently.
