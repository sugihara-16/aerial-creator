# AMSRR_design_modification_by_codex.md

This file records implementation-time supplements or deviations that were
originally accumulated against `A-MSRR_codex_ready_spec_v0_4_ja.md`. Current
normative state is consolidated in `A-MSRR_codex_ready_spec_v0_5_ja.md`; this
file remains the chronological decision/evidence log.

## 2026-08-24

### R1 support-collision diagnosis and rejected clearance revisions

- The v1 `hard_collision` is now identified as a support collision, not a
  movable-object collision. In both replays, during contact acquisition at
  rollout index 2790 and phase elapsed about 1.78 s, robot body
  `module_1__battery1` had about 2.42 N environment force and zero object
  force. In this scene the only external environment collider is the support.
- A controlled diagnostic held teacher trajectory, deterministic IK, reset,
  physical conditions, QPID/QP, local servo, and real Isaac fixed and bypassed
  only the promoted C3 `pi_L` command. Both episodes completed all eight phases
  with task success, zero safety failure, and zero fallback. Teacher references
  were identical through the learned failure index. A conservative oriented-
  box reconstruction attributes about 6.82 mm of lost support clearance and
  about 15.4 mm lateral battery displacement to the learned-command case. This
  establishes the promoted correction as decisive for the observed failure,
  while remaining diagnostic-only and promotion/collection/training
  ineligible.
- A user-requested second diagnostic retained only direct joint position/
  velocity action and contact-normal compression action. Centroidal pose/twist,
  residual wrench, contact-tangential translation, and contact-rotation action
  were zero. Persisted command values verified zero global/body/wrench and
  non-normal contact corrections. Both environments nevertheless reproduced
  the battery/support collision during contact acquisition at rollout index
  2791, one step later than the full-policy run, with about 2.48 N environment
  force and zero object force. The remaining causal set is therefore direct
  joint correction, normal-compression correction, or their combination. This
  result is diagnostic-only and authorizes no promotion, calibration,
  collection, or learning.
- R1 randomization moves object start and goal but not the physical support.
  The generic C3 nominal helper had reconstructed its support proxy from the
  moved object poses. An additive optional collision-object override now lets
  R1 generation combine the randomized object with the frozen source support.
  The first implementation touched the ledger-bound C3 nominal-generator
  source, so its resulting intermediate evidence was invalidated. That source
  was restored byte-for-byte to ledger SHA-256 `252c30fd...7887`; the override
  is now process-scoped in R1-only preparation and the default C3 path is
  unchanged. A post-restoration audit passed all 180 protected files and 159
  critical implementation files.
- An R1-only 12 mm frozen-support proposal passed the cheap exact-tracking
  screen with about 11.803 mm proxy clearance, 589 knots, and zero collision
  violations. A newly materialized two-environment full-layer replay through
  the protected update-18 `pi_L` nevertheless reproduced the same battery/
  support collision twice. The proposal is rejected.
- An execution-error-aware proposal sums about 8.47 mm measured nominal
  tracking consumption, about 6.83 mm learned-command consumption, and a 5 mm
  retained reserve, requiring at least 20.30 mm and requesting 22 mm. Under the
  existing contact targets and IK conditions, every available contact/posture
  alternative failed before screening or Isaac. The proposal is rejected.
- These results rule out clearance-only teacher-trajectory repair for this
  candidate under the current contact/IK contract. They do not modify the v0.5
  normative contract, the promoted checkpoint, or the formal v1 rejection.
  The recommended pending design choice is a derived low-level robustness
  branch from the immutable promoted checkpoint that preserves current tensor,
  action, and controller interfaces; a support-aware residual safety
  projection is the alternative. Either is a new contract/stage decision and
  requires explicit approval before implementation, learning, or R1
  recalibration.
- Only post-restoration result roots `diagnostic_selection_final_v3` (12 mm)
  and `preparation_result_final_v2` (22 mm) are current. Earlier similarly
  named roots are retained as superseded diagnostics and have no promotion,
  calibration, collection, or learning authority.

### R1 reachable-pose calibration v1 rejection

- The approved v1 calibration was executed through the real-Isaac boundary.
  Its first L1 lattice case used the train-owned two-module source bucket at
  residual x/y/yaw `-10 mm/-10 mm/-5 degrees`. Deterministic teacher plus IK
  completed, and the Isaac-free exact-tracking screen accepted all 589 resolved
  knots with no violation before any controller or simulator was invoked.
- The admitted trajectory was then run twice through deterministic teacher,
  deterministic IK, the protected update-18 C3 `pi_L`, QPID/QP, local servo,
  and real Isaac. Both episodes terminated in contact acquisition after 2791
  environment steps with `hard_collision`, task success false, safety failure
  true, and zero fallback.
- L1 requires every lattice case and both associated replays to pass and
  requires zero safety failures. This first monotonic failure makes the L1
  gates mathematically impossible regardless of later outcomes. The approved
  stop-on-first-failed-level rule therefore ended the remaining 681 L1 cases,
  L2--L4, and the 14-bucket confirmation. V1 is `rejected`; no envelope,
  distribution, collection authority, or learning authority exists.
- The protected C3 rollout implementation was restored to and executed at
  ledger SHA-256 `e813f950...b764c2`; the protected checkpoint remains
  `6ea412cc...57b`. Selection-source `train` ownership is recorded separately
  from the `validation` execution alias required by the unmodified C3 formal
  phase-zero runner. The alias is not counted as 14-bucket confirmation.
- The formal rejection manifest is
  `artifacts/p4_full/order9/r1_teacher/calibration_v1/formal_decision_final_v1/calibration_decision.json`
  at SHA-256 `d31e3fe4...af440b`. It also binds the five R1 implementation
  files used for enumeration, randomization, screening, and execution.
  Earlier smoke/intermediate records are retained but superseded. The complete
  result summary is `for_codex/R1_CALIBRATION_V1_RESULT.md`.

### R1 mandatory exact-tracking screening before full-layer testing

- User direction adds a strict two-stage admission order for every newly
  generated R1 calibration or collection candidate. First, the deterministic
  high-level teacher plus already-resolved IK trajectory is evaluated under
  ideal exact tracking only through the end of contact acquisition. Only a
  passing path may proceed to the protected C3 `pi_L`, QPID/QP, local-servo,
  and real-Isaac test.
- `order9_r1_exact_tracking_fast_screen_v1` is intentionally a linear scan of
  existing dense trajectory values and hash-bound resolver/planner evidence.
  It does not rerun IK, invoke trajectory search/optimization, step any
  controller, or start Isaac. It checks window/trajectory identity and
  continuity, finite and complete joint targets, joint-rate margin, collision
  acceptance recorded at the resolved knots, and arrival at a maintained
  two-anchor grasp pose.
- The collision evidence covers robot self-collision, unintended movable-
  object collision, support/ground, and authored environment boxes; only the
  selected grasp anchor/object contacts are permitted at acquisition. The
  screen does not claim physical force closure, frictional retention, or
  closed-loop tracking. Those claims remain exclusively with the later
  full-layer Isaac replay.
- An extreme-posture guard requires at least 1% of each authored joint range
  between every resolved target and its nearest hard limit and limits
  assembled-body tilt from world-up to 60 degrees. Read-only inspection of the
  promoted C3 nominal set found one train trajectory reaching about 91.6
  degrees, demonstrating that the tilt check is not redundant. The normalized
  reserve/tilt thresholds, screen identity, prohibited operations,
  attempt/pass counts, and selection/confirmation results are explicit
  proposal/manifest fields rather than implicit runner behavior. These values
  are part of the still-proposed v1 approval package and are not an executed
  calibration result.
- Isaac replay counts are now derived from fast-screen passes, not merely from
  teacher-generation successes. The accepted-distribution schema requires a
  separate `fast_kinematic_screen` artifact and proves that every screen
  attempt corresponds to a teacher-feasible path and every Isaac attempt to a
  screen-admitted path. The same ordering applies to formal collection.

## 2026-08-23

### Order-9 R1 Teacher-Collection Preparation and Calibration Gate

- v0.5 Section 24.5.9 and the current curriculum define R1 as
  `reachable_pose_expansion`, but intentionally do not define approved x/y/yaw
  bounds. The prior expanded-object sampler's `+/-0.015 m` position setting is
  not promoted into R1. No R1 teacher collection or learning may begin from an
  uncalibrated range.
- The additive R1 adapter uses world-frame offsets centered on the input
  task's `object.pose_world`, with versioned independent-uniform sampling. It
  applies the same planar translation and yaw rotation to the object-pose goal,
  preserving the relative transport task, box geometry, mass, CoM, inertia,
  friction, support height, and every non-pose condition. This makes R1 a
  placement/yaw ring rather than an accidental R2 property expansion.
- An acceptance-eligible adapter requires a hash-bound distribution manifest.
  The manifest records its envelope, seed/sampling semantics, complete 2--8
  module and topology coverage, center/axis/corner outcomes, deterministic
  teacher-feasibility yield, real-Isaac success and wall time, zero safety
  failures, and exact artifact bindings for the calibration config, complete
  teacher trajectory, production `C_H` checks, Isaac replay, and topology
  coverage. Calibration thresholds are not guessed in source: the manifest
  must bind a separately approved gate artifact and demonstrate its recorded
  thresholds.
- The preflight has two fail-closed modes. `calibration` verifies the immutable
  C3 base, current schedule/lineage, v9 checkpoint/action/observation contract,
  current physical model, both source and protected 43-entry nominal dependency
  sets, CUDA, and storage. `collection` additionally requires the accepted R1
  distribution manifest and re-hashes all evidence before it may publish the
  normal stage `PREPARED` manifest. Neither mode starts teacher collection or
  learning.
- The local operational storage floor is 128 GiB and may be raised but not
  weakened by the CLI. The first audit had 683.8 GiB free, verified all 180
  release-ledger files and 159 promotion-critical implementation files, and
  passed calibration readiness. Collection readiness correctly remains false
  because no x/y/yaw envelope or calibration gate has yet been approved.
- R1 orchestration must reuse the generic teacher episode/dataset writer,
  articulated trajectory teacher, deterministic posture resolver, and
  production checker. The C0-fixed collector, C3 conservative wrapper, and C3
  accepted nominal replay are not valid R1 entry points without a new
  versioned orchestration layer.

### Order-9 R1 Calibration Protocol v1 Proposal (Not Approved)

- A reviewable, schema-validated proposal now defines four symmetric
  world-frame x/y/yaw ladders beyond the C3 conservative envelope: +/-10 mm and
  +/-5 degrees, +/-20 mm and +/-10 degrees, +/-30 mm and +/-15 degrees, and
  +/-40 mm and +/-20 degrees. These are proposed calibration probes, not
  accepted R1 bounds.
- Ladder selection uses only the 22 C3 train-owned buckets. The selected
  largest consecutive pass is confirmed once on the untouched 14
  validation-owned buckets; both splits independently cover module counts
  2--8. Confirmation failure rejects the complete v1 result and may not be
  converted into a smaller accepted range by tuning against validation.
- Each bucket/level evaluates the complete 27-point lower/center/upper lattice
  (center, six face centers, twelve edge centers, eight corners), four seeded
  interior points, and two real-Isaac replays per teacher-feasible case. The
  lattice pass rate is 100%; aggregate and independently computed interior
  deterministic-teacher/Isaac rates are at least 95%; safety/fallback counts
  are zero. A first failed selection level stops the ladder, L1 failure accepts
  no envelope, and L4 success does not authorize extrapolation.
- Every sampled start/goal is audited against the frozen source-bucket support.
  Projected CoM must remain within that support and both projected-CoM and
  full-footprint margins are recorded. A negative full-footprint margin is not
  rejected by itself because promoted C3 intentionally uses lateral support
  inset; the unchanged support and real-Isaac gates remain mandatory.
- The distribution-manifest contract now separately binds the approved
  protocol and approval record, plus selection and confirmation counts/rates.
  Collection preflight re-hashes both. This prevents selection-only evidence
  or an unapproved proposal from publishing an accepted R1 envelope.
- The proposed post-acceptance collection contains 140 episodes, allocated
  98/21/21 across train/validation/held-out and 14/3/3 per module count. Its
  seeds are disjoint from calibration. Task ids, seeds, and pose hashes must be
  globally unique, validation/held-out poses disjoint, and the 95% success plus
  zero-safety/fallback gates apply to every split-by-module cell. Calibration
  records remain permanently ineligible for imitation learning.
- The complete proposal is
  `for_codex/R1_CALIBRATION_PROTOCOL_V1_PROPOSAL.md` and the executable
  proposal is
  `configs/training/order9_r1_calibration_protocol_v1_proposal.yaml`. Their
  current `proposed` state authorizes neither calibration nor collection.

## 2026-08-22

### v0.5 Standalone Specification Consolidation

- The user approved consolidating v0.4 and this chronological modification
  log into `A-MSRR_codex_ready_spec_v0_5_ja.md` while preserving v0.4 as
  history. The new document is the active standalone source of truth.
- The consolidation distinguishes current normative contracts from measured
  evidence and superseded/failed diagnostics. In particular, the promoted
  common C3 v9 contract and release ledger supersede module-count action masks,
  compression-only lineages, exact-wrench hard gates, rounded-Gaussian action
  trials, fixed 12 mm preload, and diagnostic checkpoints.
- v0.5 incorporates the centroidal-only QPID/independent-joint-servo boundary,
  joint-free `pi_H`, the common contact-space v9 action, deployable-versus-
  privileged observation boundary, morphology-aware nominal preload, C3
  factorized credit, the actual C0--C3 lineage, R1--R4 continuation, and the
  promoted update-18 artifact identity.
- `for_codex/AGENTS.md` and the QP/PID supplement now identify v0.5 as their
  parent source. Historical WORKLOG and experiment-log references to v0.4 are
  intentionally retained as records of the version active at those times.
- This consolidation changes documentation precedence only. It introduces no
  runtime schema, tensor shape, policy parameter, checkpoint, controller,
  curriculum configuration, dataset, or promotion result.

## 2026-08-07

### Order 9 C3 Morphology-Invariant Contact-Compression Action

- The user approved replacing the topology-dependent compression action
  coordinate with one dedicated scalar
  `contact_compression_residual_action` in `pi_L`.  The prior v5 adapter used
  whichever local joint happened to contribute most to the current IK
  compression direction as the scalar coordinate.  That coordinate changed
  with morphology and posture, so the same learned action did not have stable
  physical meaning across module counts.
- The v6 scalar always means additional grasp closure.  The command decoder
  adds it to the preserved v5 compression scalar, clamps the combined value,
  and distributes it over all local joints along the morphology-specific IK
  compression direction.  The old selected joint coordinate is removed from
  the ordinary per-joint residual before that distribution; applying it both
  directly and through the IK direction would double-count one joint and
  would not preserve the v5 command.
- The new mean head may condition on recurrent state, graph embedding, and
  deployable module count.  It does not consume raw PhysX contact or wrench
  tensors.  Its scalar action, mean, log probability, and entropy are stored
  explicitly in the v18 tensor-rollout contract so PPO exact replay includes
  the new stochastic dimension.
- v5-to-v6 migration copies every existing parameter exactly and initializes
  the new output to zero.  A training-only privileged warm start may calibrate
  the zero-initialized scalar for the entering module count.  Per-module-count
  calibration entries isolate this bootstrap: changing the five-module entry
  leaves the two--four-module deterministic command exactly unchanged.  Fresh
  on-policy PPO may subsequently train the graph-conditioned common head and
  all count entries under the normal C3 objective.
- The privileged warm-start label is training-only and remains
  acceptance-ineligible.  Raw contact truth is not added to actor input or to
  deployment.  Promotion still requires fresh on-policy training evidence and
  held-out continuous Isaac validation.
- On the five-module continuous training target, count-5 calibration reduced
  scalar target RMSE from `0.3982` to `0.0183`, while two--four-module anchor
  RMSE and maximum error remained exactly `0.0`.  A held-out five-module
  phase-zero Isaac run then completed all task phases in 6,353 steps with no
  collision, drop, QP failure, or fallback.
- The same entry procedure applies when a later module count is unlocked:
  screen accepted nominal trajectories first, calibrate only that module
  count from train-only privileged continuous targets, and require a fresh
  topology-stratified PPO update before evaluating the policy.  For six
  modules, a train-only `0.6/0.8` compression sweep selected `0.8`; both
  settings reached place without collision, QP failure, rotor saturation, or
  object drop, while `0.8` entered place sooner.  The count-6 calibration
  leaves counts two--five unchanged to numerical precision.  This is an
  extension of the approved morphology-entry bootstrap, not a new actor
  observation or deployment-time privileged signal.

## 2026-08-04

### Order 9 C3 Two--Three-Module PPO Subcurriculum

- The user approved beginning the revised C3 learning sequence with the
  available two- and three-module morphologies before returning to arbitrary
  two--eight-module training.  This is a staged optimization curriculum, not
  a change to the canonical C3 morphology range or promotion criteria.
- Production C3 rollout selection accepts explicit inclusive training module
  bounds.  The selected train distribution must contain every requested
  module count, while validation and formal promotion remain held-out and are
  not narrowed implicitly.  The 2--3-module lineage balances the two module
  counts in each topology-equal PPO update.
- Each update retains the approved mixture of phase-reset and continuous
  state-inheritance rollouts.  Policy observations/actions, recurrent-state
  semantics, outcome-only reward, nominal IK, `PolicyCommand`, QPID/QP, and
  safety gates are unchanged.
- Boundary-preserving PPO packing may merge an incomplete phase-reset tail
  with an adjacent sequence batch when the tail lacks the required target or
  anchor phase.  The operation must preserve every transition exactly once
  and may not relabel phases, cross recurrent episode boundaries, or weaken
  the existing target/anchor validation.  Continuous target-only shards
  remain legal only when phase-reset shards provide the corresponding anchor
  supervision.
- The bounded subcurriculum budget is 13 updates (indices 0--12), totaling
  7,215,104 fresh transitions.  Completion of this budget is not C3
  promotion.  Advancement still requires fixed held-out safety comparison
  and full-horizon continuous physical evidence; later 2--8-module training
  must not claim that a 2--3-only checkpoint satisfies arbitrary-morphology
  coverage.
- The completed update-12 checkpoint passed a full-horizon physical check on
  representative held-out two- and three-module buckets in `8/8` deterministic
  episodes.  Every episode completed release/settle without fallback,
  collision, drop, QP-infeasible terminal, or timeout.  This admits the
  checkpoint as the output of the 2--3-module subcurriculum only; it does not
  waive the later 4--8-module train/evaluation requirements.

## 2026-08-02

### Order 9 C3 Mixed State-Inheritance PPO Diagnostic

- The user approved retaining the outcome-only C3 objective while mixing
  phase-reset and continuous state-inheritance rollouts.  Every update now
  contains the existing 14 phase-reset shards plus one continuous shard for
  every module count 2--8.  Each continuous shard uses 12 environments for
  1,280 simulator steps, starts equally from `contact_acquisition` and
  `transport`, preserves physical and recurrent state across real phase
  transitions, and resets recurrent state only at an episode boundary.
- The first diagnostic generation contained 262,144 phase-reset train steps,
  107,520 continuous train steps, and 262,144 validation steps.  Exact replay
  passed for all 21 train shards.  Update 0 completed one epoch without KL
  stop or rollback; aggregate KL was `0.0005767`, maximum applied topology-
  phase KL was `0.032493 < 0.04`, and checkpoint SHA was
  `d380828b...bdb4d4`.
- Fixed-14 validation did not establish improvement over the initializer.
  Mean reward changed by `-0.000998`, collision rate by `-0.000166`, QP-
  feasible rate by `-0.000693`, and phase-success rate by `-1.31e-5`.
  Six of 14 buckets improved, median paired reward delta was `-4.02e-5`, and
  successful terminals remained `216`.
- Paired continuous validation also showed no phase-outcome improvement.
  Initializer and update 0 both reached `transport` and horizon-timed-out in
  `4/4` bucket-28 episodes; both dropped during `lift` in `4/4` bucket-36
  episodes.  Bucket-36 mean return changed by `-208.41`.  Update 0 is not
  promoted, and no update 1 is authorized from this diagnostic checkpoint.
- A trainer inefficiency was found during this run.  Fixed sequence-count
  minibatches made a fragmented seven-module failure shard with 2,169 short
  recurrent sequences dictate 181 optimizer attempts and cyclically reuse
  smaller shards.  Production packing now preserves complete phase-
  contiguous recurrent sequences but fills each topology minibatch by an
  equal transition budget.  On the immutable update-0 generation this reduces
  the computed attempt count from 181 to 93 without changing rollout data,
  reward, GAE, recurrent-state, topology-equal loss, or KL contracts.  Any
  further mixed-state experiment must restart at corrected update 0 from the
  clean initializer; the slower diagnostic checkpoint is evidence only.

### Order 9 C3 Outcome-Only Learning Result and State-Inheritance Blocker

- The approved outcome-only C3 contract was exercised from a clean promoted-C2
  initializer through updates 0--3.  Exact per-contact 6D wrench-range credit
  was zero in both PPO return and auxiliary actor supervision; its measured
  reward contribution was exactly zero in paired validation.
- All four updates completed one PPO epoch without KL early stop.  Update 3
  had aggregate KL `0.0001033`, maximum topology-phase KL `0.0009784`, clipped
  fraction `3.82e-6`, and checkpoint SHA `40d3c7c3...a2953ad`.
- On the fixed 14 held-out phase-reset buckets, update 3 minus the clean
  initializer changed mean reward by `+0.003278`, collision rate by
  `-8.72e-5`, QP-feasible rate by `-4.53e-4`, and phase-success rate by
  `+8.72e-6`.  Seven buckets improved and seven regressed; the median reward
  delta was `-0.000159`, and successful terminals remained `216`.
- The required continuous physical check did not pass.  Bucket 28 reached
  `place` without drop, collision, or QP terminal in all four episodes but
  exhausted the 3,200-step horizon before `release`.  Bucket 36 reached
  `transport`; two episodes dropped the object and two timed out in that
  phase.  Overall task success was `0/8`, so update 3 is not promoted and no
  later update is authorized from this lineage.
- This establishes a state-distribution mismatch: the current topology-
  stratified training batch is composed of 256-step phase-reset rollouts,
  whereas promotion starts at `approach` and carries physical and recurrent
  state across phase boundaries.  A higher fixed-reset score is therefore not
  sufficient evidence for continuous task completion.
- The next recommended method change is to retain the outcome-only reward and
  mix fresh continuous state-inheritance rollouts into every topology-
  stratified update, preserving recurrent state and GAE across real phase
  transitions.  Phase-reset shards remain useful for coverage, but cannot be
  the sole train distribution.  This is a method-level change and must be
  approved before a new lineage is started.

## 2026-08-01

### Order 9 C3 Outcome-Only Pi-L Learning

- The user approved removing exact per-contact 6D wrench-range error from the
  C3 `pi_L` learning objective.  `w_wrench_range=0.0`, and the privileged
  compression-teacher weight is also `0.0`; therefore neither PPO return nor
  an auxiliary actor loss receives exact-wrench credit.
- Raw PhysX wrench reduction and range-membership telemetry remain available
  for diagnosis, critic-independent evaluation, and `pi_H` feasibility data.
  They remain excluded from actor observations, `PolicyCommand`, IK, QPID/QP,
  and deployable phase decisions.  Existing contact maintenance, slip, object
  progress/pose, terminal success/failure, QP, collision, actuator saturation,
  energy, drop, and timeout terms/gates are unchanged.
- Formal learning starts from the clean C3 initializer derived from promoted
  C2, not from any prior wrench-trained C3 update.  Its v4 actor is migrated
  behavior-preservingly to v5 with a zero-output contact-residual branch; the
  migration now explicitly accepts the canonical initializer update index
  `-1` only when `initializer_only=true`.
- Update 0 is evaluated against the same clean initializer on the fixed 14
  held-out buckets.  Updates 1--3 proceed only if update 0 has a healthy PPO/KL
  trace and no material task/safety regression.  Continuation requires task
  outcome improvement independent of wrench telemetry.

### Order 9 C3 Wrench-Range Soft-Regularizer Diagnostic

- After removal of exact 6D wrench-box membership from the hard success gate,
  the first outcome-gate update improved fixed-14 mean reward by `0.04162`,
  but `0.03952` (about 95 percent) came from the still-configured
  `w_wrench_range=10.0` term.  Successful terminals were unchanged and the
  weak-grasp validation bucket still dropped in `4/4` continuous trials.
- The user approved a bounded diagnostic with `w_wrench_range=1.0`.  During
  that diagnostic, exact wrench membership remained privileged
  reward/telemetry and `pi_H` feasibility information but was reduced to a
  unit-weight soft regularizer.  Contact maintenance, slip, object/task
  outcome, QP feasibility, collision, drop, and actuator constraints were
  unchanged.
- This changes only the scalar C3 reward weighting.  Actor observations and
  actions, contact-residual architecture, nominal IK, `PolicyCommand`, QPID/QP,
  phase-success gates, safety terminals, and tensor shapes are unchanged.
- The diagnostic must first branch for one fresh update from the
  behavior-preserving update-9 v5 initializer.  A clean update-0 lineage is
  allowed only if paired fixed validation demonstrates task-outcome benefit;
  total-reward improvement caused only by the wrench term is insufficient.
- The one-update diagnostic consumed 524,288 fresh transitions across two
  train topologies for every module count 2--8.  On the paired fixed-14
  validation set, total reward changed by `-0.00437`; reward with the wrench
  term algebraically removed changed by `-0.00112`; phase-success rate changed
  by `-8.28e-5`; transport/place reward changed by `-0.02134/-0.03509`.
  QP-feasible rate improved by `+0.000331`, while collision rate worsened by
  `+2.62e-5` and successful terminals remained `215`.
- Therefore the unit-weight condition did not demonstrate task-outcome
  benefit.  It is rejected as the formal C3 setting, no clean update-0 lineage
  is started from it, and the configuration returns to the last accepted
  `w_wrench_range=10.0` pending a different method-level learning decision.

### Order 9 C3 Outcome-Based Grasp Gate

- The user approved removing exact per-assignment 6D contact-wrench box
  membership from the C3 grasp-success hard gate.  This supersedes only the
  hard-gate and phase-authority clauses of the 2026-07-30 `C3 Privileged PhysX
  Wrench-Range Phase Supervision` section; raw PhysX wrench reduction and the
  existing dense wrench-range reward/diagnostic remain unchanged.
- C3 contact readiness now requires the configured number of selected physical
  contacts, QP feasibility, and the existing continuous dwell.  `lift`,
  `transport`, and `place` require maintained physical contacts, QP
  feasibility, their existing object-pose/task conditions, and all unchanged
  collision/drop/timeout safety rules.  Wrench-box membership is recorded but
  cannot block phase progression or task success.
- Actor observations/actions, nominal trajectory IK and inward contact lead,
  `PolicyCommand`, QPID/QP, actuator limits, tensor shapes, and checkpoint
  shapes are unchanged.  Raw PhysX contact remains privileged C3
  reward/supervision evidence and is not an actor or controller input.
- A fixed-checkpoint A/B on update 9 and validation buckets 28/36 confirmed
  that both pass contact acquisition in `4/4` environments after removing the
  range gate.  Bucket 28 completed lift in `4/4`, raised the object by about
  `104 mm`, entered transport, maintained both contacts, and had no drop,
  collision, rotor saturation, or QP terminal.  Bucket 36 attempted lift but
  lost one selected contact and dropped the object in `4/4`; it therefore
  remains a genuine grasp/task failure rather than a wrench-box false reject.
- Contract identities advance to
  `order9_c3_privileged_two_physx_contacts_qp_dwell_v7`,
  `order9_c3_privileged_physx_contact_outcome_phase_supervisor_v2`, collector
  v40, and promotion runner v16.  Evidence explicitly records
  `wrench_range_hard_gate_enabled=false` so exact-gate artifacts cannot be
  reused under the outcome contract.

## 2026-07-31

### Order 9 C3 Contact-Compression Credit Assignment

- The existing C3 privileged wrench-range reward remains weighted by
  `w_wrench_range=10.0`; raw PhysX contact wrench remains training/evaluation
  truth only and is not added to actor observations, `PolicyCommand`, nominal
  IK, QPID/QP, or actuator commands.
- In the `establish_contact` and `lift` actor objective, the PPO advantage is
  assigned only to the existing morphology-conditioned contact-compression
  action coordinate.  During these boundary updates, actor optimization is
  restricted to the node-wise `joint_decoder`; the shared/global actor is
  frozen, while the critic remains trainable on all valid phase data.  This
  changes training-time credit assignment, not the deployed policy interface
  or action tensor shape.
- Same-rollout comparison from the prior update-9 checkpoint and paired
  fixed-14-bucket real-Isaac evaluation selected this combination: the
  intended 10x wrench weight plus joint/compression-limited credit assignment
  improved mean total reward and mean wrench-range reward; QP-feasible rate
  changed by `+8.72e-6` and collision rate by only `+4.80e-5`.  Applying
  another 10x multiplier on top of the configured value was diagnosed as an
  unintended effective 100x condition and rejected.
- A fresh official C3 lineage must therefore start from the hash-bound C2/C3
  initializer rather than continue the diagnostic update-9 branch.  Exact
  behavior replay, topology-stratified 2--8-module coverage, one-epoch
  boundary optimization, parent/non-target KL preservation, and hard
  topology-phase KL rollback remain mandatory.

## 2026-07-30

### Order 9 C3 Piecewise-Log Wrench-Range Penalty

- The C3 privileged supervisor and its exact wrench-box success gate remain
  unchanged.  However, the former dense penalty clipped each assignment's
  normalized L-infinity violation at `1.0`.  Fresh update 4--6 evidence showed
  that approximately 79--98 percent of contact-phase assignment samples were
  on this plateau, so wrenches farther than one interval width outside the
  teacher box lost their severity ordering even though they remained failures.
- For a per-assignment normalized violation `v`, the user-approved shaping is
  now `p(v)=v` for `v<=1` and `p(v)=1+log(v)` for `v>1`.  This preserves the
  previous reward exactly over its formerly informative range, remains zero
  exactly when all six wrench components lie inside the teacher box, and adds
  a monotonic but robust logarithmic tail for larger PhysX deviations.  The
  bounded-assignment mean and existing `w_wrench_range` weight are retained.
- This changes reward shaping only.  The two-contact/complete-6D-range/QP/dwell
  hard gate, timeout behavior, actor observations, `pi_H` teacher command,
  nominal IK, `PolicyCommand`, QPID/QP, actuator commands, and tensor shapes are
  unchanged.  Raw PhysX wrench remains privileged training/evaluation evidence
  and is not made deployable input.
- The reward contract advances to
  `order9_privileged_contact_wrench_range_reward_v3_patch_moment_piecewise_log`,
  the production collector to v37, and the promotion runner to v14 so saturated
  v2 rollout evidence cannot be reused as fresh v3 training or promotion data.

### Order 9 C3 Privileged PhysX Wrench-Range Phase Supervision

- C3 is a `pi_L` teacher-trajectory training/evaluation stage, not the final
  deployable `pi_H` phase controller.  During C3 only, runtime phase progression
  therefore uses a privileged supervisor built from PhysX contact truth.  This
  supersedes the C3 phase-authority portions of the 2026-07-29 Jacobian
  force-observer and deployable-phase-gate sections; the observer remains useful
  diagnostic evidence but is not allowed to block or admit a C3 transition.
- For every active teacher assignment, the raw normal and friction patches are
  reduced about the corresponding `ContactCandidate` frame to the realized 6D
  wrench.  Grasp readiness requires at least two selected physical contacts,
  every bounded assignment's complete 6D wrench to lie inside the active
  `pi_H` teacher lower/upper box, QP feasibility, and a continuous `0.25 s`
  dwell.  The existing `0.5 N` sensor floor only rejects numerical/no-contact
  noise when counting physical contacts; it is not a wrench target and cannot
  substitute for the teacher range.
- `contact_acquisition` completes only after the above dwell.  `lift`,
  `transport`, and `place` also require the current wrench-range and QP
  conditions in addition to their existing object-pose/contact conditions.
  A wrench-range violation retains its normalized privileged reward penalty and
  prevents phase success; sustained nonachievement therefore ends through the
  existing phase timeout/failure penalty rather than being silently accepted.
  Controller-side preload completion is not part of this gate.
- PhysX wrench values remain excluded from `pi_L` actor observations,
  `PolicyCommand`, nominal IK, QPID/QP inputs, and actuator commands.  They are
  training/evaluation supervision, reward/critic evidence, and diagnostics
  only.  Hard collision, QP, drop, and other safety authorities remain in
  force.  The C3 artifact records the privileged-supervisor contract and the
  collector version prevents old deployable-gate evidence from being mixed
  with new C3 promotion evidence.
- From R1 onward, learned `pi_H` owns phase/timing decisions from deployable
  observations.  The privileged supervisor then returns to label/reward/critic
  and evaluation use and is removed from runtime phase progression; hard
  deterministic feasibility and safety checks are not removed.

## 2026-07-29

### Order 9 C3 Deployable Jacobian Normal-Force Observer

- 同日付の`Surface-Semantic Virtual Contact and Deployable Phase Gate`で定義した
  contact-acquisition判定のうち、最大joint loadとjoint tracking errorを接触力の代用にする
  部分を置き換える。これらは機体内部の姿勢保持負荷でも増加し、物体への接触法線力を
  一意に示さないため、phase遷移のforce evidenceには使用しない。
- 実行時は選択grasp frameにおける現在Jacobian `J_p`、`pi_H`が指定した接触法線
  `n`、符号付き実joint torque、robot modelから得るgravity torque、および非接触
  free-motion torque baselineを使用する。baselineは`approach`中は全jointで更新し、
  `contact_acquisition`中は各anchorが幾何的contact bandの外側にある間だけ、そのanchorが
  所有するjoint枝を更新する。contact bandへ入った枝は凍結し、実接触反力をbaselineへ
  学習しない。接触によるactuator
  residualを `tau_r = tau_applied - tau_gravity - tau_baseline`、各anchorの写像を
  `A_i = -J_p_i^T n_i`とし、ridge付き非負最小二乗
  `argmin_{f >= 0} ||A^T f - tau_r||^2 + lambda ||f||^2`でanchor法線力を推定する。
  小規模なcoordinate NNLSで解き、外部QPライブラリやraw contact sensorを要求しない。
- 推定confidenceは、接触法線Jacobianの可観測性と、推定後のtorque誤差を同Jacobian
  行空間へ戻した残差から算出する。法線接触モデルと直交する姿勢保持torqueを残差へ
  含めてはならない。これはobserverであり、joint target、torque biasまたは最終actuator
  commandを生成しない。wrench-range内へ実レンチを収める責務は引き続きlearned
  `pi_L`にあり、QPID/QPが最終制御・安全authorityを保持する。
- `contact_acquisition -> lift`のproduction gateは、全選択anchorについて従来の
  pose/relative-speed条件に加え、推定法線力、推定confidence、QP feasibleおよびdwellを
  要求する。必要法線力は`pi_H` wrench rangeが同符号ならゼロに近い境界値を使用し、
  rangeがゼロを跨ぐ場合でも少なくとも等分担Coulomb支持条件
  `m_payload*g/(mu*N_contact)`を要求する。根拠のない固定`0.5 N`閾値は使用しない。
- raw PhysX contact wrenchはprivileged reward、critic、診断および評価にだけ残し、
  observer入力、phase遷移入力、controller分岐には使用しない。代表実Isaac診断で物理推定が
  弱接触と十分な接触を分離できた。十分な接触の16環境は推定`2.17--2.41 N`、
  privileged実力`1.70--2.27 N`、必要`1.18 N`で`16/16`がliftへ遷移した。弱接触の
  phase-zero 16環境は推定`0.145--0.450 N`、privileged実力`0.043--0.208 N`、
  必要`1.097 N`でlift遷移`0/16`だった。この分離結果によりlearned residual headは
  現時点では追加しない。
  複数形態で系統的な推定biasが観測された場合だけ、同じ物理推定に加算する校正項として
  再検討し、deterministic gate自体をlearned判定へ置き換えない。
- baseline校正は接触獲得中のfree-motion閉鎖torqueを接触力と誤認する問題を軽減するが、
  接触法線Jacobian `J_p^T n`がゼロまたは極小のanchorを可観測にはしない。この場合は
  confidence 0でfail closedとする。任意形態のproduction gateを完遂するには、motor
  currentだけに依存しないdeployable contact evidenceを別途定義する必要があり、baselineを
  privileged contact truthや推定値への固定加算で代用してはならない。
- 内部`Order9DeployablePhaseGateInput`はjoint-load/tracking evidenceから
  `estimated_normal_force / required_normal_force / estimator_confidence`へ変更する。
  保存済みrollout tensor schema、`pi_L` observation/action shape、checkpoint shapeおよび
  `pi_H -> nominal IK -> pi_L -> QPID/QP`契約は変更しない。

### Order 9 C3 Surface-Semantic Virtual Contact and Deployable Phase Gate

- `pi_H`のanchor poseは物体表面上の物理的な接触目標を表し、押込み量や接触力を
  暗黙に含めない。実行時nominal IKだけが、選択anchorの接触法線方向へ有界な
  virtual target leadを加えてjoint nominalを生成する。leadは`pi_H`出力、保存済み
  teacher軌道、wrench rangeおよび物体geometryの意味を変更しない。
- `pi_L`を無効化した実Isaac/PhysX sweepを、2・5・8モジュールの代表train bucketで
  `2 / 5 / 10 / 15 / 20 mm`について実施した。2モジュールでは2 mmで両anchor約
  `5.1 N`、5 mm以上で約`10 N`となった一方、5・8モジュールではbody/joint tracking
  offsetにより20 mmでも接触力ゼロの例があった。したがって、形態別のdeterministic
  force controllerは導入せず、安全な共通初期値として2 mmだけを採用し、強すぎる・
  弱すぎる実接触をwrench-range rewardで`pi_L`に補償させる。
- virtual leadはreview済み最終把持姿勢の局所differential IKでjoint targetへ変換し、
  1 jointあたりの追加量を`0.15 rad`に制限する。`approach`ではゼロ、
  `contact_acquisition`で滑らかに0から1、`lift / transport / place`で1、
  `release`で1から0へ戻し、`retreat / settle`ではゼロとする。controller側でraw contact
  forceに応じてjoint targetを積分するpreloadはproduction経路から削除する。本節は同日付の
  `C3 controller-side preload production integration`記録を置き換える。
- raw PhysX contact force、slip、penetrationおよびcollision truthは、privileged reward、
  critic、評価、TensorBoard、診断artifactにだけ使用できる。actorのraw input、runtime
  phase遷移、joint target生成、QPID/QP controller分岐には使用しない。phase遷移は
  anchor/objectの推定poseと相対速度、motor-current-equivalent joint load、joint tracking
  error、QP feasibleおよびdwellだけから判定する。
- `pi_L`のdeployable observationへ各module 4 jointの符号付きlog loadを追加した。実機では
  motor currentから換算し、Isaacでは前control stepに実際に適用したmotor torqueから同じ
  featureを作る。policy contractはactive-knot feature v2 / `pi_L` v4とし、旧checkpointの
  対応input weightをゼロ拡張したC2由来initializerから新しいC3 lineageを開始する。旧C3
  checkpointは診断資料として保持するが、新contractの継続学習parentにはしない。
- PPO actor更新は最初のcontact-control bootstrapでは`establish_contact / lift`を対象とし、
  criticは全phaseを学習する。非対象phaseは同じSHA拘束behavior policyへKLで固定し、
  `topology x phase`上限を越えたoptimizer stepはrollbackする。これは2 mm leadを接触力
  controllerへ固定化せず、wrench-range/slipの改善責務をlearned `pi_L`へ残すための
  training-only設定である。
- 7モジュールへの拡張で、共通2 mm leadと10 mmのlearned compression spanでは、
  held-out bucket 40がlift中にdropし、通常PPO updateを追加しても改善しないことを確認した。
  nominal 2 mmは変更せず、`pi_L`の形態共通compression action spanを20 mmへ拡張する。
  既習5・6モジュールはspan比に応じてmodule-count calibration biasをbehavior-preserving
  変換し、7モジュールは強い補正を維持する。変換checkpointはacceptance-ineligibleで、
  新しいschedule hashのfresh on-policy PPOを1 update以上通すまでpromoteしてはならない。
  変換後の実Isaac診断では、bucket 40が2 mm leadのまま全8 phaseを6601 stepで完走し、
  最大接触力10.862 N、collision/QP infeasible/drop/fallbackはいずれも0だった。5モジュール
  held-outも6335 stepで完走し、6モジュールは既知のpost-release deterministic retreat
  collisionまで把持・運搬・place・releaseを完了した。raw PhysX contactは引き続きactor入力
  または実行時compression生成に使用しない。
- 20 mmのlearned compressionは、review済みnominal collision-aware IKの後段に加わる局所
  補正なので、nominal姿勢近傍から無制限に外れてはならない。形態・review済みnominal・
  task geometryだけから、convex object/support/ground/self collisionと最大局所関節変位
  `35 mrad`を満たす正規化action上限を形態ロード時に一度計算し、実行時は`pi_L`が要求した
  共通compression scalarをその範囲へclampする。IK方向そのものは縮小しないため、上限内の
  既習actionは不変である。この処理はraw PhysX contact/wrenchを使わず、force controllerでも
  phase supervisorでもない。QPID制約とnominal collision-aware IKの間に置くdeployableな
  局所trust regionである。
- 5モジュールへの段階拡張後、held-out bucket 31で把持後のlift中にobject--robot相対回転が
  `0.1509--0.1716 rad`まで増え、物体が滑ることを確認した。policy checkpointとaction契約を
  固定した切り分けでは、virtual-contact total leadを3 mmから4 mmへ増やすと同回転は
  `0.0117--0.0125 rad`へ低下し、bucket 31/38はいずれも`4/4`成功した。さらに同一checkpointで
  2--5モジュールの固定validation 8 bucketを4 mmで評価し、全`32/32` episodeが成功、timeout、
  fallback、安全失敗はいずれも0だった。よってproduction共通nominal leadを2 mmから4 mmへ
  改定する。これは`pi_H`のsurface pose、保存nominal軌道、`pi_L`のaction shape、20 mm
  compression spanを変更せず、実行時nominal IKの中立押込み量だけを変更する。

### Order 9 C3 Complete Eight-Phase Nominal Replay Contract

- C3で実行可能なnominal trajectoryは、`approach / contact_acquisition / lift /
  transport / place / release / retreat / settle`の8 phaseをbucketごとに事前計算し、
  phase別軌道と一つのtimelineをhash-bound artifactとして保存する。C3学習・評価では
  これをcontrol周期で補間再生し、deterministic configuration-space plannerを実行しない。
  実機推論ではこのbucket固有artifactを再生せず、従来どおり`pi_H -> runtime nominal IK ->
  pi_L -> QPID/QP`を使用する。
- `approach`と`contact_acquisition`はhuman-acceptedかつcollision-aware plannerで得た経路を
  そのまま使用する。`lift / transport / place`は把持後のobject--robot相対SE(3)を厳密に
  保持し、`release`はaccepted contact pathを支持面上で時間反転、`retreat`はaccepted
  approach pathを上方clearanceで時間反転する。`settle`はその安全な退避終端を保持する。
- 全8 phaseを、bucket固有のobject/supportと`z >= 0`地面条件、convex proxyによる
  robot self/object/support collisionに対してoffline検査する。数値境界には`0.5 mm`だけを
  許容するが、実際のproxy交差、ground violation、選択接触の許容penetration超過は
  一切許可しない。42 bucketすべてがこの完全軌道検査に合格したlineageだけをC3へbindする。
- Human-accepted最終把持姿勢を保ったまま旧`approach`のsupport clearanceだけが不足する場合、
  offline teacher repairとしてrobot側のcentroidal/free-anchor目標を剛体並進してよい。
  関節目標、contact assignment、object目標は変更せず、上方clearanceを持つ`approach`から
  `contact_acquisition`中に把持高さへ連続的に降下させる。修復後は通常と同じfull-mesh
  collision検査とIsaac nominal-QPID screeningを独立に再実行する。この修復はbucket生成時
  だけに限定し、C3学習または推論時のprojection/plannerとして使用しない。
- Vectorized collectorで一部環境をterminal resetするとき、target tensorはbatch全体について
  再構築される。このため、reset対象だけでなく未終了環境を含む全行へ、保存済みC3 nominal
  conditionを同じstep内で必ず再適用する。一stepでも汎用task targetへ戻すことを禁止する。
  collector識別子は
  `order9_vectorized_isaac_complete_pi_l_collector_v25_complete_nominal_target_rebind`とする。
- Promotion時の学習throughput provenanceは、現在のdirectoryへupdate 0..Nが全て存在すると
  仮定しない。active stage manifestの`parent_checkpoint`と`dataset_manifest`のhash bindingを
  update Nから0まで逆向きに辿り、branchを跨いだ実際のgeneration列だけを集計する。
  promotion runner識別子は
  `order9_c3_promotion_runner_v3_complete_nominal_lineage`とする。
- 本変更はnominal/reference provenanceとcollectorのtarget保持不備を修正するものであり、
  policy入出力、reward、PPO、phase成功条件、QPID/QP、runtime nominal IK周期、deterministic
  safety authorityは変更しない。

## 2026-07-28

### Order 9 C3 Boundary-Preserving PPO Fine-Tune Contract

- 固定2 bucketの境界評価で、update 14は`release / retreat / settle`を改善した一方、
  通常の全phase PPOをさらに1 update進めたupdate 15では非対象phaseの共有actorが
  driftし、境界成功が`24/64 -> 10/64`へ後退した。aggregate phase KLだけでは
  morphologyごとの局所driftを拘束できなかったため、ユーザー承認によりupdate 14を
  親とする境界保持fine-tuneを追加する。
- ActorのPPO surrogateとentropyは`release / retreat / settle`だけへ適用し、criticは
  従来どおり全有効phaseを学習する。非対象phaseは同じSHA-256拘束on-policy rolloutに
  保存されたbehavior log probabilityをfrozen parentとして、exact Monte-Carlo KLを
  lossへ重み`1.0`で加える。別teacher policyや推論時fallbackは追加しない。
- updateは1 epochに限定する。各optimizer step後、同じtopology-stratified batchを
  再評価し、非対象の最大`topology x phase KL > 0.005`、または全phaseの最大
  `topology x phase KL > 0.04`ならparameterをstep前へrollbackして当該updateを停止する。
  適用済みstepだけから成るcheckpointが両上限を満たすことをstage runnerがfail-closedに
  検証する。
- 通常C3の`1e-5`で行った最初のcalibration stepは、rollback前の非対象最大KLが
  `0.01907`となり`0.005`上限を超えた。このstepは適用・保存されていない。境界
  fine-tuneだけはlearning rateを`0.25`倍（`2.5e-6`）へ校正し、hard rollbackと
  1 epoch上限は維持する。これは未消費の同一fresh rolloutで再現確認する。
- この変更はC3の境界性能を保持しながら後段phaseを追加学習するtraining-only契約である。
  actor/criticのtensor shape、`pi_H -> nominal IK -> pi_L -> PolicyCommand -> QPID/QP`、
  reward weight、phase成功条件、推論時周期・処理およびdeterministic safety authorityは
  変更しない。識別子は
  `order9_c3_boundary_fine_tune_v1_parent_kl_rollback`である。

### Order 9 C3 Release/Retreat Boundary-Tail Sampling Contract

- C3の通常rollout長`256 step x 0.02 s`では、30秒のphase後半から次phaseへ
  遷移した実状態を十分収集できない。そこで従来のintra-phase reset進捗
  `1/6, 1/2, 2/3`は全phaseで維持し、追加の`0.9` stratumだけを`release`と
  `retreat`で選択可能にする。他phaseでは`0.9`をreset samplingへ使用しない。
- Reset後は通常のlearned `pi_L -> PolicyCommand -> QPID/QP`を実行し、phase成功時には
  recurrent state、controller state、実関節・機体・物体stateをresetせず後続phaseへ
  引き継ぐ。これにより`release -> retreat`と`retreat -> settle`のsuccessor actionを、
  実際の境界履歴および既存のcollision terminal/rewardで学習する。境界をteleportして
  後続phaseだけを単独評価する方式ではない。
- 各C3 train/validation shardはtail sampling契約と境界遷移回数をartifactへ保存する。
  境界到達前に衝突した形態は有効なon-policy terminal学習データなので、そのshard単独に
  遷移成功を要求しない。代わりに1 train generation全体では両境界を少なくとも1回
  通過したことをrunnerがfail-closedに確認する。固定phase-zero promotion evaluationは
  従来どおりapproachから全taskを実行する。
- actor/critic入出力、reward weight、PPO hyperparameter、phase成功条件、QPID/QP、
  nominal IK、configuration-space teacherおよび推論時処理は変更しない。識別子は
  boundary-tail contract
  `order9_c3_boundary_tail_sampling_v1_release_retreat`、phase-reset reference v3、
  reset bank v2、collector v24、stage runner v9である。update 13からの検証学習は、
  旧lineageを上書きせず明示的なbranch parent indexを持つ新lineageへ保存する。

### Order 9 C3 Supported-Release Boundary Contract

- place完了後の物体は支持面が保持するため、QPID/QPの推定payload質量・慣性
  feedforwardは`lift / transport / place`に限って有効とし、`release`開始時に無効化する。
  これは物体接触が消えた後も機体へpayload重力補償を加え続ける旧境界を置き換える。
- `release`成功は、従来のcontact-free dwellと物体pose条件に加え、実関節角が
  release軌道の最終関節姿勢へ最大絶対誤差`0.05 rad`以内で到達したことを要求する。
  これによりopening軌道途中からfully-openな`retreat`目標へ不連続に遷移しない。
- QPID/QP、衝突・QP terminal、policy入出力、reward weight、PPO、nominal IK、
  phase順序および物体pose許容値は変更しない。新しい評価証拠はpayload契約
  `order9_payload_feedforward_lift_transport_place_v1`とrelease契約
  `order9_release_success_contact_free_object_pose_final_joint_posture_v1`を必須とし、
  collector v23で識別する。rollout tensor schemaはartifact v17のままである。

### Order 9 C3 Accepted-Nominal Static Phase-Reset Bank

- C3のphase resetは、human-acceptedかつhash-boundなnominal軌道から得た物理stateを
  形態別の固定reset bankへ一度だけ保存し、以後のtrain/evaluationでbyte-identicalに
  再利用する。reset bank構築中にはlearned `pi_L`、initializer、QPID/QP、fixture、
  settling simulationを実行しない。candidate `pi_L`はrolloutのstep 0から通常どおり
  `PolicyCommand -> QPID/QP`経路を担当する。
- Reset admissionは静的整合性に限定する。必須条件は、accepted nominal provenanceと
  morphology graph / physical model / robot USD / task specのhash一致、finiteな
  root/body/joint/object state、全phase/stratumの存在、accepted offline collision review、
  joint limit、地面・支持台条件である。reset時に「2点接触維持、QP feasible、一定時間の
  物体変位、downward speed」を再度成立させるdynamic stability gateは設けない。
  接触維持、wrench-range、slip、QP、衝突、物体追従はpolicy rolloutのreward・terminal・
  evaluation対象であり、reset bankの合否条件ではない。
- initializerとcandidateのpaired比較は同一reset-bank fileとSHA-256を共有する。
  rollout artifactは`c3_reset_bank_sha256`を必須とし、旧互換field
  `c3_reset_stabilization_checkpoint_sha256`は常に`null`、
  `dynamic_stability_gate_required`は`false`とする。bankのcontract/hash/tensor shape/
  phase identityが異なる場合はfail closedとする。
- 本契約は、下記の`Actor-Free Nominal-QPID Reset Construction`およびそれ以前の
  reset-time learned actor / QPID settling方式を置き換える。reward weight、actor I/O、
  PPO hyperparameter、nominal IK runtime、deployed inference、QPID/QPの最終権限は
  変更しない。識別子はreset bank
  `order9_c3_accepted_nominal_physical_reset_bank_v1`、artifact v17、collector v22、
  stage runner v8である。

## 2026-07-27

### Order 9 C3 Actor-Free Nominal-QPID Reset Construction

- **Superseded:** 本節は上記`Accepted-Nominal Static Phase-Reset Bank`により置き換えた
  旧実装記録であり、production collectorからは呼び出さない。
- C3の`lift / transport / place` intra-phase reset構築では、learned `pi_L`
  actorを実行しない。完成把持姿勢へのteleportによるmeshのdepenetration impulseを
  避けるため、accepted contact-acquisition軌道の最後`0.1 s`を`0.5 s`へ低速化して
  nominal `PolicyCommand -> production QPID/QP`で再生する。軌道終端が押付け力ゼロの
  幾何接触になることを避けるため、最後`0.1 s`のjoint-space閉鎖差分をもう10回分
  同じ方向へ延長し、QPIDで小さなcontact preloadを形成する。接触形成中は物体へ
  3軸の減衰fixtureを適用する。接触後の`0.5 s`でfixture支持率を`1 -> 0`、QPIDの
  payload mass/inertia feedforwardを`0 -> 1`へ連続的にhandoffし、最終nominal
  姿勢ではfixtureを完全に外す。これにより荷重feedforwardのstep入力で機体だけが
  上昇して接触を失うreset artifactを作らない。residual wrenchとjoint torque biasは
  常にゼロとし、learned actor/recurrent stateは評価も更新もしない。
- Reset構築のsimulation時間はtaskの`phase_elapsed`へ加算しない。したがって、
  物体変位gateはtask軌道ではなくreset候補の接触形成・物理settling変位を測る。
  2点selected contact、prohibited collisionなし、QP feasible、configured
  displacement boundというadmission条件は変更しない。admissionはfixture解除後
  のみ許可し、採用snapshotのbody/joint/object速度はゼロへ投影する。
- この変更はreset分布をpolicy-independentにし、3モジュール中心で学習された
  initializerを任意形態resetの成立条件にしないためのもの。PPO rollout開始後は
  学習対象`pi_L -> PolicyCommand -> QPID/QP`を通常どおり使用する。reward、actor
  入出力、PPO hyperparameter、runtime IK、deployed inferenceは変更しない。
- 新しいprovenanceはtensor runtime
  `order9_tensor_complete_pi_l_qpid_runtime_v5`、reset stabilizer
  `order9_c3_contact_reset_stabilization_v7_continuous_payload_handoff`、artifact v16、
  production collector v21、stage runner v7で識別する。本番runnerと固定
  held-out比較runnerはreset用actor checkpointをcollectorへ渡さない。artifactの
  旧provenance field `c3_reset_stabilization_checkpoint_sha256`は互換性のため残すが、
  この方式では常に`null`とする。

### Order 9 C3 Topology-Stratified On-Policy Update

- C3の1回のPPO updateは、単一形態のrolloutから任意形態actor全体を更新してはならない。
  各updateで2--8モジュールをそれぞれ1個ずつ含む、合計7個のfreshな
  topology-homogeneous train shardを収集する。bucketは各モジュール数のtrain集合内で
  updateごとに決定論的に巡回する。validationは従来どおり、更新には使わない独立した
  1 bucketである。
- trainの総環境数は従来と同じ`1024`、各shardのrollout長も`256` stepとする。
  `1024 / 7`の余りはupdateごとに巡回し、各形態へ`146`または`147`環境を割り当てる。
  したがって、1 updateのtrain/validationを合わせた総environment step数、C3全体の
  update数、PPO learning rate、clip、KL閾値、reward、action schemaは変更しない。
- graph/recurrent forwardとphase-local GAE・advantage正規化は形態ごとに行う。
  1 optimizer stepでは各形態から同数のphase-balanced recurrent sequenceを取り、
  7個の形態別lossを等重みでgradient accumulationした後に、一度だけgradient clipと
  optimizer stepを実行する。これにより形態間でgraphを混在させず、モジュール数による
  transition数差にも学習重みを支配させない。
- KL early-stopは従来のactor-phase別aggregate KLの最大値で判定する。形態×phase別KLも
  TensorBoardおよびupdate metadataへ記録するが、少数標本セルの瞬間値だけで更新を
  停止するgateにはしない。
- production C3 datasetは、7 train shardと1 validation shardをSHA-256で束ねる
  `order9_tensor_native_pi_l_on_policy_dataset_v2_topology_stratified`とする。
  collectorはGPUメモリとhost負荷を抑えるため最大2 processを並列実行する。この変更は
  offline configuration-space teacher、保存済みnominal軌道、runtime IK、`pi_L`の
  入出力、QPID/QP、deployed inferenceを変更しない。

### Order 9 C3 PPO Regression Corrections and Restart Contract

- The invalid `phase_reset_v2/update_000006` lineage must not be continued.
  Corrected C3 PPO restarts from the unchanged active-knot initializer and
  uses three interior reset-progress strata (`1/6`, `1/2`, and `2/3`) for
  lift, transport, and place.  Avoiding exact phase boundaries prevents a
  boundary-only support/collision state from dominating a stratum.  A reset
  state is admitted only after real Isaac contact reports at least two selected
  contacts, no prohibited collision, and feasible QPID/QP allocation within
  the configured object-displacement bound.  The first admitted contact state
  may be velocity-projected to a stationary reset; temporary settling support
  is reset-construction-only and is cleared before rollout actions or rewards.
- Reset construction is not part of the policy being evaluated or trained.
  The later actor-free nominal-QPID reset contract above supersedes the former
  immutable-initializer reset policy.  C3 generation and paired comparison
  execute no learned actor during reset construction; they use the same task
  seed, physics, accepted nominal trajectory, and production QPID/QP instead.
  Thus later `pi_L` checkpoints cannot improve their own initial-state
  distribution by acting during reset construction.
- Recurrent PPO sequences do not cross phase boundaries.  Advantages are
  normalized within actor phase, sequence minibatches are sampled in
  phase-balanced round-robin order, and the KL guard is evaluated per actor
  phase using the worst observed phase rather than only a global average.
  These are required C3 settings, not optional diagnostic modes.
- The critic may consume the actor recurrent state as a value feature, but
  value-loss gradients are detached at that boundary.  Actor/GRU parameters
  are therefore updated by the policy objective and entropy term, while the
  privileged critic is updated by the value objective without a
  value-dominated gradient into the shared actor representation.
- Active contact-frame force boxes retain the friction-cone/capability check.
  Moment boxes are derived from candidate contact-patch area, the bounded
  normal-force range, friction, and anchor torque capability: tilt capacity is
  patch radius times normal load, and torsional capacity is additionally
  scaled by friction.  Historical accepted C3 trajectories are calibrated at
  load time without changing their candidates, force bounds, targets,
  schedules, or nominal joint paths.
- The corrected implementation is identified by tensor PPO
  `order9_tensor_native_pi_l_ppo_v2_phase_balanced_detached_critic`, contact
  reset stabilization
  `order9_c3_contact_reset_stabilization_v3_contact_admitted_projection`, and
  production collector
  `order9_vectorized_isaac_complete_pi_l_collector_v16_early_contact_reset_projection`.
  No reward weight, action schema, `PolicyCommand`, nominal IK runtime, QPID/QP
  ownership, or deployed actor input is changed.

### Order 9 C3 Morphology-Specific Phase Reset Correction

- C3 production PPO must use phase-specific resets for every arbitrary
  morphology, matching the existing tensorized-collector and phase-conditioned
  actor contract.  The earlier implementation enabled the reset distribution
  only when an arbitrary graph happened to equal the canonical Order 8 graph.
  C3 graphs therefore installed only phase zero; because one `256 x 0.02 s`
  shard is shorter than the accepted `18--45 s` approach paths, updates 0--6
  collected almost exclusively approach data.  Those checkpoints are retained
  as diagnostic evidence and are not C3 promotion/training authority.
- The accepted offline nominal path now supplies morphology-specific approach
  and contact starts plus the final grasp state.  The existing deterministic
  object-task translations derive lift, transport, place, release, retreat,
  and settle starts from that same state.  Release/retreat use the accepted
  contact-pregrasp posture as the arbitrary-morphology open posture.  Each USD
  articulation root is solved from its realized root-to-assembled-body
  transform after installing the corresponding Dock posture; no online IK or
  configuration-space planner is introduced.
- Normal C3 training initializes copied environments round-robin over all
  eight runtime phases, while deterministic promotion evaluation remains
  phase-zero-only.  Subsequent terminal resets advance from each environment's
  own phase offset rather than collapsing all environments onto phase one.
  Production validation now fails closed unless all eight actor-mapped runtime
  phases occur in both train and validation shards and the complete reset bank
  is reported.
- The arbitrary-morphology posture banks now distinguish accepted initial,
  pregrasp, and final-grasp joint states.  Lift through place hold final grasp;
  release moves final grasp to pregrasp; retreat/settle hold pregrasp.  This
  corrects the earlier constant-posture release target without changing
  `pi_L`, `PolicyCommand`, QPID/QP, reward weights, phase gates, or task goals.
- Exact behavior replay must archive the phase progress that was actually
  consumed by the actor.  In C3 approach/contact, the accepted nominal IK
  conditioner replaces the generic task-timeout progress with progress along
  that morphology's nominal trajectory.  Recomputing the archived value as
  `phase_elapsed / task_timeout` changes the phase and actor features during
  replay.  Production evidence therefore stores
  `Order9TensorObjectTaskTarget.phase_progress` directly and identifies its
  semantics as `exact_policy_actor_input`.
- New production evidence uses tensor artifact v11, collector v12, phase-reset
  reference v1, and stage-runner v4.  The first reset-corrected attempt in
  `phase_reset_v1` exposed the phase-progress archive defect during strict
  replay and is retained as failed diagnostic evidence.  The independent
  `phase_reset_v2` lineage contains the corrected update-0--6 training.  The
  accepted nominal set, initializer, curriculum schedule, bucket split, PPO
  hyperparameters, and environment-step budget are unchanged.

### Order 9 `pi_L` PPO Tensor-Native Dataset and Replay Boundary

- The real-Isaac rollout artifact is the canonical stochastic transition
  payload for production `pi_L` PPO.  The stage runner now binds the train and
  validation `.pt` files through a small versioned, SHA-256-verified manifest
  and feeds the train tensors directly to recurrent PPO.  It must not expand
  each transition into nested `LowLevelControlRecord` objects or write/read a
  redundant compressed JSONL copy in the learning hot path.  Validation raw
  data remains immutable and hash-bound as generation/split evidence but is
  not converted into training records.
- This is an execution/representation correction, not a learning-method
  change: one fresh train+validation generation still produces exactly one PPO
  update; GAE, sequence length, minibatch size, epoch count, clipping, KL stop,
  phase/active-knot actor inputs, privileged critic input, actions, QPID/QP
  ownership, checkpoint lineage, and TensorBoard metrics are unchanged.
- Exact replay reconstructs actor features tensorially and re-evaluates all
  train transitions in the original collection-shaped time slices.  Stored
  `recurrent_state_out -> recurrent_state_in` and
  `global_action -> previous_global_action` continuity is checked separately
  at `2e-5` and was byte-exact for C3 update 0.  Because the v9 artifact does
  not store the derived centroidal actor-feature vector, long articulated
  trajectories amplify float reconstruction differences: update-0 full-data
  maxima were `0.0012494` log-probability, `1.1445e-5` value, and
  `0.0002461` recurrent state.  The replay evidence therefore carries and the
  validator enforces field-specific bounds (`0.0025`, `5e-5`, `5e-4`),
  rather than weakening value or temporal-continuity checks to one broad
  threshold.
- The initial `2.5e-5` critic-value bound was calibrated only on the
  three-module/update-0 replay.  The first five-module collection reproduced
  log probability, recurrent state, and stored temporal continuity well
  within their unchanged bounds, but showed a deterministic maximum critic
  difference of `3.43323e-5` at critic values around `17.3`.  The value-only
  bound is therefore `5e-5` (about `2.9e-6` relative at that scale); this is a
  GPU graph-reduction numerical allowance and does not change PPO, reward,
  physics, or recurrent/action continuity semantics.
- The additive tensor dataset manifest and tensor trainer are internal Order 9
  production interfaces.  Existing canonical record/JSONL dataset support is
  retained for behavior cloning, offline inspection, and other policy
  families; it is simply removed from the `pi_L` PPO runtime-critical path.

### Order 9 `pi_L` Privileged Contact-Wrench Range Reward Wiring

- This implementation closes the already specified reward wiring in the
  2026-07-07 policy/controller supplement; it does not redefine the method.
  During `contact_acquisition`, `lift`, `transport`, and `place`, the measured
  six-dimensional wrench for each selected assignment is compared with the
  active `pi_H`/teacher `wrench_lower` and `wrench_upper` in that candidate's
  moving contact frame.  Values anywhere inside the box receive zero range
  penalty.  Only normalized lower/upper overflow is penalized, averaged over
  the six axes and then the bounded assignments.  No point target or arbitrary
  multi-contact force decomposition is introduced.
- The measured force and moment are privileged-only.  The Isaac collector
  combines raw normal and friction patches and evaluates each moment about the
  corresponding object-following candidate contact-frame origin.  These values
  are consumed by reward and recorded in privileged evidence/TensorBoard.  The
  current critic observation contract is unchanged, and the values are not
  added to actor observations, `PolicyCommand`, QPID targets, or the deployable
  inference input.
- The existing `0.5 N` value remains solely the contact-existence threshold
  used for active-contact count, dwell, break, and release semantics.  It is
  not a desired contact force and is not a substitute for the policy-proposed
  wrench range.  The range reward has its own configurable non-negative weight
  `w_wrench_range` (initial value `1.0`).
- Raw rollout evidence advances to
  `order9_tensor_isaac_complete_pi_l_rollout_v9_privileged_wrench_range` and
  stores the measured contact-frame wrench, lower/upper bounds, and active
  bound mask.  The contract remains backward-readable for pre-change active-
  knot artifacts, while new production artifacts fail closed if the reward
  contract or required tensors are missing.

### Order 9 C3 Accepted Nominal Replay and Runtime Planner Boundary

- The deterministic configuration-space planner is an offline `pi_H` teacher
  generator only.  C3 learning and deployed inference must not instantiate or
  execute that planner.  The final human-reviewed approach/contact trajectories
  are immutable, SHA-256-bound training inputs.
- The accepted C3 set contains all `42` buckets (`28` train, `14` validation).
  User-reviewed replacements supersede aggregate indices `19`, `27`, and `41`;
  the final set manifest SHA-256 is
  `117065d301b680b660ae138ba94824b83ba1bae7a3544e2ad499a3bee1d31f4c`.
  The separate human-review record binds every exact artifact and animation
  scene hash and records `42/42` accepted decisions.  This review establishes
  ideal-tracking nominal geometry only, not learned-policy or dynamics success.
- Each production rollout bucket binds one accepted component artifact,
  timeline, collision-validation record, animation scene, and selected surface
  pair by hash.  Runtime loading fails closed on a changed bucket/task/
  structure/PhysicalModel identity, missing or rejected human decision,
  changed bytes, non-accepted collision record, or a request to enable the
  configuration-space planner.
- During C3, the saved nominal trajectory is materialized as device tensors at
  `10 Hz`.  Its CoM pose/twist and nominal Dock-joint position/velocity are
  linearly sampled on the `50 Hz` controller clock; after the accepted phase
  trajectory ends, the final reference is held.  `pi_L` consumes the current
  nominal reference and produces its learned correction, after which QPID/QP
  remains the sole owner of final actuator commands.
- This replay rule is C3-specific.  Once learned `pi_H` is introduced in later
  curriculum stages, its `1--2 Hz`, `1--3 s` horizon of CoM/anchor poses and
  wrench ranges is resolved by the deterministic trajectory IK into the same
  nominal-reference contract.  The offline configuration-space teacher is
  still not run at learning or inference time.

### Order 9 C3 Initializer Physical-Model Rebind

- The C2-derived active-knot initializer was created before the approved pitch
  grasp-frame and collision-mesh URDF update.  Its tensors and policy contracts
  remain applicable, but its checkpoint metadata carried the obsolete
  PhysicalModel hash.  A C3 PPO parent must match the current PhysicalModel
  fail-closed; silently accepting the old hash is prohibited.
- A separate metadata-only rebind artifact copies every policy tensor exactly,
  preserves the C2 parent, schedule, stage, feature/action contracts, random
  seed, and initializer status, and changes only the target PhysicalModel
  provenance plus explicit rebind ancestry.  Source and target state-dict hash
  are both
  `aeeef39db0b72f36571913f60d2266dbb828b250b81e1bde647baac18075b45d`.
  The rebound checkpoint SHA-256 is
  `c926f783bd5b868ac393d8d3f8d6b9edd988c8cfb441649218fded88c27d8e52`;
  its current PhysicalModel hash is
  `23dfd5a15d170b356e00a188f952afe98b5646b728a245493c36de36f789591a`.
- This is an artifact-lineage correction, not retraining, transfer learning,
  reward modification, architecture modification, or a relaxation of physical
  validation.  Stage startup now checks the initializer PhysicalModel hash
  before collection rather than waiting for the first PPO update.

### Order 9 C3 Production Preflight Boundary

- The production setting is `1024` Isaac environments per collector process
  and `256` rollout steps per environment.  One update consists of one train
  and one validation shard (`524,288` environment steps); the configured C3
  target resolves to `14` complete updates (`7,340,032` environment steps).
- No-training real-Isaac smoke evidence covers the rebound initializer on the
  accepted nominal path and a worst-case eight-module accepted bucket at
  `1024` environments.  TensorBoard evidence contains reward terms by phase,
  phase occupancy, task/safety/QP rates, throughput, GPU load, process memory,
  and system load.  The complete preflight is frozen in
  `c3_preflight_accepted_nominal_v1/preflight_manifest_v1.json`, SHA-256
  `53916f9c2731c22ad0fa3b40362ed2ce5de0be46edfb302493acc2e88d5da4f4`.
- These checks authorize starting C3 collection/training; they do not claim a
  PPO update, grasp, lift, transport, release, or task-success result.

## 2026-07-26

### Order 9 Contact-Acquisition Local Configuration-Space Planner

- The deterministic warm-up teacher uses different configuration-space
  planning semantics by task phase.  Detached `approach` motion continues to
  use the global overhead planner.  Once a collision-free pregrasp state has
  been reached, `contact_acquisition` is a local closure problem and must not
  use overhead or full-domain base samples merely because they are
  collision-free.
- Contact acquisition searches in the following deterministic order: the
  direct pregrasp-to-grasp edge, a fixed coordinate-wise bridge, and a
  seed-fixed local bidirectional RRT-Connect fallback.  Every candidate edge
  remains densely checked by the same convex self/object/support/ground
  oracle.  The local base domain is restricted to a `0.03 m` tube around the
  pregrasp-to-grasp translation segment and a ceiling of the higher endpoint
  plus `0.02 m`; the fallback is capped at `160` samples.  These constraints
  express contact-phase locality, not a relaxation of collision clearance or
  final anchor accuracy.
- A contact assignment/group is eligible for nominal C3 preparation only if
  the teacher can produce the complete collision-checked approach and contact
  path.  Final-pose feasibility or a successful approach alone is
  insufficient.  On local-contact failure, generation tries the next
  deterministic contact group and fails closed if none has a complete path.
- Human review may reject a collision-clear final posture for an undesirable
  morphology/anchor combination, such as an extreme vertical stacking that
  is not itself a proxy collision.  Such a decision excludes only the exact
  unordered surface-port pair for the hash-bound bucket structure; it does
  not introduce a generalized posture heuristic, change the task or
  structure, or relax any collision/path gate.  Replacement preparation must
  reselect a different surface pair and regenerate the complete approach plus
  contact trajectory.  Rejection files bind the source bucket manifest and
  structural hash, and multiple reviewed rejections for one bucket are
  cumulative.
- When a bucket has a hash-bound human-reviewed final grasp pose, its accepted
  Dock-joint vector may be reused only as an internal contact-goal IK seed.
  The stored vector must reproduce the reviewed joint-solution hash before it
  is admitted.  It does not become a learned `pi_H` output or bypass contact,
  collision, joint-limit, or complete-path checks.  Once IK refines that seed
  to an actually feasible goal, the local corridor is defined between the
  reached pregrasp state and that refined feasible goal, rather than an
  unrefined desired base pose.  The goal itself is still checked by the full
  collision oracle before it defines the corridor endpoint.
- This amendment is internal to the deterministic teacher and nominal IK
  preparation.  Learned/runtime `pi_H` remains joint-command-free and still
  outputs anchor assignment/poses, wrench ranges, and CoM poses; no
  `PostureTarget`, `PolicyCommand`, observation/action, or checkpoint schema
  changes.  The accepted configuration states remain optional internal IK
  seeds/references as specified by the route-preservation amendment below.
- Index 04 and 09 are the representative acceptance evidence.  Their contact
  phases now use three-state local paths, with maximum centroidal-CoM rises of
  `0.860 mm` and below `0.001 mm`, respectively, and both pass independent
  saved-frame replay against current convex proxies.  This remains
  ideal-tracking preparation evidence: exact visual-mesh swept collision,
  learned `pi_L`, QPID, Isaac dynamics/contact, grasp/transport/release, and
  C3 learning remain separate boundaries.
- The completed current-lineage preparation contains all `42` buckets
  (`28` train and `14` validation), with exactly six buckets at every module
  count from `2` through `8`.  Independent replay accepted all `14,652`
  saved frames against the current convex self/object/support/ground oracle.
  The aggregate manifest SHA-256 is
  `423188ed10a981fde97621d1655456253b789b92a9727855adb2adcf29d6b568`.
  This completes nominal-trajectory preparation only; it does not broaden the
  ideal-tracking evidence into learned-controller or physical task success.

### Order 9 Configuration-Route Preservation Across the Joint-Free `pi_H` Boundary

- `pi_H` remains joint-command-free: its learned/runtime output contains the
  selected anchor assignment, anchor-pose trajectory, contact-wrench range,
  and CoM-pose trajectory.  A deterministic configuration-space teacher may
  use Dock-joint states internally to construct a collision-free route, but
  those states are not added to the `pi_H` action, `PostureTarget`, or another
  persisted learned-policy command.
- For morphologies with redundant kinematics, projecting a valid full-state
  planner waypoint to CoM and selected-anchor poses is not sufficient to
  preserve the planned homotopy/posture: trajectory IK may converge to a
  different task-space-equivalent joint solution and reintroduce a collision
  or fail to advance.  Therefore the deterministic planner's joint state may
  be passed across the internal teacher/resolver boundary as a per-raw-knot
  IK seed/reference.  It initializes the nominal IK solve but does not replace
  its task-space constraints, collision gate, rate limits, or final solved
  joint trajectory.  Resolver evidence records the seed-trajectory hash when
  this path is used.
- A rolling-horizon teacher may cache the remaining accepted configuration
  route to avoid replanning the same expensive full-state path.  Cache reuse
  is permitted only when phase, morphology, allowed contacts, obstacle scene,
  and exact desired goal identities match.  Before each reuse, the active edge
  from the measured/current configuration is densely rechecked by the same
  self/object/support/ground collision oracle; failure invalidates the cache
  and triggers fresh planning.  Overhead shortcutting may remove only
  collision-checked interior waypoints and must preserve the initial lift and
  terminal descent semantics.
- Contact acquisition goal search may include the immediately preceding
  collision-free pregrasp state as an IK continuation seed in addition to the
  desired-state and zero-state seeds.  This changes solver initialization,
  not the contact target, anchor tolerance, collision margin, or feasibility
  criterion.
- The 5--8-module C3 nominal artifacts are ideal-tracking preparation
  evidence only.  Independent convex-proxy replay accepted every saved frame,
  but this does not substitute for learned `pi_L`/QPID/Isaac dynamics or
  exact-mesh production admission.

### Order 9 Bucket-Specific Support and Ground-Safe Nominal C3 Trajectories

- The earlier three-bucket configuration-space pilot is superseded for C3
  preparation by
  `nominal_trajectories_bucket_support_ground_v4`.  Its diagnostic scene had
  mixed a large virtual support with the bucket task geometry, so neither its
  rejected support poses nor its HTML scene may be used as support/ground
  evidence.
- Each bucket now owns one finite support collision box derived from the same
  object start/goal geometry used by the Order 8 raised-support task:
  X size is transport distance plus object X size plus `0.05 m`, Y size is
  object Y size minus `0.04 m` (with a `0.05 m` lower bound), height is the
  task support height, and its centre is the object start/goal midpoint.  The
  object bottom coincides with the support top at its initial pose.  The
  object and support are checked as separate native convex obstacles so that
  selected object contacts remain allowed while every robot/support contact
  remains prohibited.
- Ground collision is represented by the infinite half-space `z >= 0`, not a
  second oversized box.  The native collision oracle conservatively checks
  the world AABB of every current convex robot proxy and rejects a state when
  its minimum Z is below the plane after applying the configured numerical
  tolerance.  The finite blue ground patch in HTML is visualization only.
  This is an additive internal collision-checker input; it changes no
  persisted `TaskSpec`, policy observation/action, `PostureTarget`, or
  `PolicyCommand` schema.
- Receding-horizon configuration-space planning still validates every dense
  edge sample against self/object/support/ground collisions.  To prevent the
  former lateral/preform waypoint oscillation, a state already aligned above
  the exact goal may take the fully checked vertical descent edge, and only
  final-waypoint selection uses small equivalence tolerances (`5 mm`
  translation, `1 mrad` attitude, `35 mrad` joint angle).  These tolerances
  do not relax collision sampling or the exact final goal.  Detached approach
  projection is capped at `0.20 rad/s` and the pilot-only intermediate anchor
  reconstruction tolerance is `11 mm`; the final contact target/gate is
  unchanged.
- The regenerated 2/3/4-module trajectories contain `12/15/14` rolling
  windows and last `36/45/42 s`.  An independent replay checked all
  `361/451/421` saved frames (`1233/1233` total) with the current native
  convex oracle.  Every frame passed self/object/support/ground checks with
  zero violating pairs/proxies; minimum ground clearances are
  `0.133512/0.131705/0.132969 m`.  Maximum intermediate anchor errors are
  `10.901/9.093/9.734 mm`, within the explicitly pilot-scoped `11 mm` bound.
- This evidence assumes exact tracking of the deterministic `pi_H` teacher
  plus nominal IK trajectory.  It does not claim learned `pi_L`, QPID,
  contact-force, dynamics, exact visual-mesh admission, Isaac execution,
  grasp/transport success, production bucket promotion, or C3 learning.

### Order 9 Deterministic Configuration-Space π_H Teacher

- User-approved method amendment: the deterministic warm-up `pi_H` teacher
  no longer authors approach/contact-acquisition motion with fixed
  up/lateral/down staging heuristics.  It plans in the combined
  configuration space of assembled-base `SE(3)` and every global Dock joint
  using seed-fixed bidirectional RRT-Connect.  Every tree edge is sampled
  against the native convex collision proxy before it may enter a teacher
  path.  The learned/runtime `pi_H` interface is unchanged: the accepted
  full-state path is projected back to phase-local CoM and free-anchor pose
  targets, and deterministic dense IK still produces the nominal joint
  trajectory downstream.
- Convex collision is mandatory for C3 nominal generation.  A caller may no
  longer produce or archive these trajectories with
  `collision_gate_status=not_configured`.  Both the configuration-space edge
  oracle and the 10 Hz posture resolver use the same current-lineage native
  proxy geometry.  The existing requested `5 mm` non-contact margin,
  `0.2 mm` numerical/proxy feasibility tolerance, and `2 mm` selected-contact
  penetration bound remain unchanged.
- Per the user's requested permissive pilot boundary, no new exact-STL,
  visual-mesh, swept-volume, Isaac, contact-force, learned-policy, or dynamic
  task-success gate was added.  Existing anchor tolerances also remain
  unchanged (`10 mm` position and `0.1 rad` attitude in the C3 teacher
  resolver).  This pilot proves convex-proxy collision-free nominal
  construction only; later offline and Isaac admission remain separate.
- Three current-lineage pilot buckets (2/3/4 modules) were generated under
  `nominal_trajectories_configuration_space_v2`.  All `21` rolling windows
  report convex collision acceptance and zero violating pairs; minimum
  recorded clearance is `4.807--4.818 mm`, consistent with the requested
  margin minus its unchanged numerical tolerance.  The respective plans use
  `6/8/7` windows and last `18/24/21 s`.  No 42-bucket regeneration, C3
  learning, checkpoint, reward, controller, or production promotion occurred.

## 2026-07-25

### Order 9 Pitch-Dock Object-Grasp Contact Frame

- The user-defined object-grasp frame for each pitch Dock is a fixed frame
  translated exactly `+0.0424086 m` along the local X axis of
  `pitch_connect_point_{1,2}`.  The authored
  `pitch_grasp_contact_frame_{1,2}` links are object-contact semantics only:
  structural Dock assembly continues to use the unchanged
  `pitch_connect_point_{1,2}` frames.  Yaw Dock object-grasp frames remain
  identical to their assembly connect frames.
- `DockPortSpec` gains the backward-compatible optional transform
  `grasp_contact_frame_from_connect`.  The PhysicalModel builder requires the
  authored pitch frame and records its fixed transform; gripper-surface
  resolution, grammar-created `RobotAnchor.local_pose`, fixed-morphology
  teacher anchors, whole-structure kinematic checks, and curation markers all
  consume the composed object-grasp frame.  The grammar identity advances to
  `order9_holon_sequential_design_grammar_v3_grasp_contact_frame`; old v2
  anchor records must not be silently reinterpreted.
- This frame correction does not relax the existing `5 mm` non-contact
  clearance or `2 mm` selected-contact penetration gates.  The regenerated
  80-entry morphology pool preserves every split/module-count/structural-hash
  tuple, while PhysicalModel-bound artifacts move to a separate
  `morphology_assets_grasp_frame_v1` lineage.
- A diagnostic four-module pitch/pitch case proves that IK can reach the new
  frames (`0.897 mm` maximum contact-position error) while retaining
  `6.511 mm` minimum non-contact clearance.  Final admission still fails:
  selected pitch-mesh penetration is `3.388 mm`, above the unchanged `2 mm`
  limit.  This residual matches the known approximately `5 mm` mesh
  protrusion beyond the authored grasp frame and cannot be removed by changing
  joint angles while keeping that frame on the requested object contact.
  Therefore no pitch candidate is admitted from this diagnostic; pitch
  candidate generation remains blocked until the mesh protrusion is removed
  and the PhysicalModel-bound assets are regenerated.
- Mesh-correction resolution (supersedes only the blocker in the preceding
  bullet): the user removed the pitch-mesh protrusion and corrected the pitch
  mesh orientation in the xacro, while also making rotor-arm 1/4 collision
  geometry match their visual mesh.  The normalized runtime URDF now carries
  the same geometry changes while preserving its intentional `thrust_1--4`
  link normalization.  Exact-STL recomputation confirms that the six authored
  same-module overlap pairs are unchanged; their manifest is rebound to the
  new URDF and collision-mesh hashes rather than trusting the old exclusions.
- The same four-module pitch/pitch case, surface ports `[0,12]` and contact
  group `slot_0:grasp_pair:1`, now passes on the first collision-aware solve:
  maximum selected-contact penetration is `0.864 mm` against the unchanged
  `2 mm` limit, minimum non-contact clearance is `6.768 mm` against the
  unchanged `5 mm` margin, and both violation counts are zero.  The earlier
  pitch-candidate blocker is therefore resolved at the offline native gate.
  This remains convex/FCL static-pose evidence; human review and subsequent
  real-Isaac path/contact admission remain separate requirements.
- C3a batch 004 resumes human curation on the corrected geometry with five
  previously unused train-pool structures, one for every module count
  `4--8`.  Every selected pair uses at least one pitch grasp frame: the
  four- and five-module cases use pitch/yaw, while the six-, seven-, and
  eight-module cases use pitch/pitch.  All five pass the unchanged native
  gates with maximum selected-contact penetration `0.813--1.503 mm`,
  minimum non-contact clearance `5.189--11.272 mm`, and zero violations.
  They remain pending scene-hash-bound human Accept/Reject/Recalculate
  decisions and are not production bucket or Isaac evidence.
- Batch-004 disposition: the user subsequently accepted all five exact
  scene hashes.  Hash-bound real-Isaac static-pose replay passed all five
  saved base/joint solutions with selected penetration `<=2 mm`, no
  non-selected object contact, and no prohibited cross-module physical
  contact.  This does not supersede the required trajectory boundary:
  no hash-bound closure path was replayed, so all five remain ineligible for
  production promotion.  Static selected-contact force also exceeded the
  unchanged Order 8 `30 N` diagnostic threshold in three cases
  (`115.812`, `57.430`, and `61.737 N`); this is retained as a separate
  limitation rather than silently converting a static collision pass into
  grasp/transport evidence.
- C3a batch 005 contains another five previously unused train-pool structures,
  one for every module count `4--8`, and every pair again uses at least one
  pitch grasp frame.  The four- and five-module cases use yaw/pitch; the
  six-, seven-, and eight-module cases use pitch/pitch.  All five pass the
  unchanged offline native gates with maximum selected-contact penetration
  `1.502 mm`, minimum non-contact clearance `5.657 mm`, and zero violation
  counts.  The user subsequently accepted all five exact scene hashes, and
  hash-bound real-Isaac static-pose replay passed all five saved states.
  Cases 04/05 measured `33.151/50.479 N`, above the unchanged `30 N`
  diagnostic threshold; no closure trajectory was replayed, so none is
  production-eligible.
- C3a batch 006 contains five further unused train-pool structures, one for
  every module count `4--8`.  Four- and eight-module cases use pitch/yaw;
  five-, six-, and seven-module cases use pitch/pitch.  All five pass the
  offline native gates with maximum selected-contact penetration `1.513 mm`,
  minimum non-contact clearance `5.152 mm`, and zero violation counts.  They
  were all accepted by the user.  The accepted four-module scene is
  deterministic recalculation index `3`, with a user note that the whole
  vehicle is strongly tilted.  Exact hash-bound Isaac replay passed the other
  four scenes, but that four-module scene failed because the selected pitch
  body had no physical contact observation; only its yaw-selected body
  contacted.  This is a static admission failure even though the offline
  convex gate and human geometry review passed.
- C3a batch 007 uses the final unused train-pool structure in every
  `4--8`-module stratum.  Four- and five-module cases use pitch/yaw; six-,
  seven-, and eight-module cases use pitch/pitch.  The seven-module case
  requires deterministic collision seed `6` and a `-20 mm` Y centroidal
  offset.  All five pass the offline native gates with maximum selected
  penetration `1.500 mm`, minimum non-contact clearance `5.132 mm`, and zero
  violation counts.  The user accepted four scenes and rejected the
  seven-module scene for visually excessive inter-body proximity/overlap.
  Isaac did not execute the rejected scene; all four accepted scenes passed
  static admission.  The four-/five-module selected-contact forces were
  `30.866/35.434 N`, above the unchanged `30 N` diagnostic threshold, and no
  trajectory was replayed, so none is production-eligible.
- After batch 007, the unused existing-train-pool count is zero in every
  `4--8`-module stratum (eight total train structures per stratum).  Further
  curation cannot silently continue under the same provenance rule.  It must
  either expand/regenerate the split-owned morphology pool or explicitly
  admit isolated generated train candidates, with the resulting split and
  artifact-lineage consequences recorded first.
- Curation may pipeline computation across that human boundary: while one
  batch remains open in the browser, the next batch may search unused
  train-pool structures and prepare offline candidates.  The prepared batch
  cannot replace the active review page, inherit acceptance, enter Isaac, or
  alter a production bucket until the preceding exact-scene decisions are
  recorded.  This is an operational scheduling rule only; it changes no
  policy, feasibility, trajectory, dataset, or promotion semantics.

### Order 9 C3a Accepted-Pose Real-Isaac Admission Boundary

- The first five scene-hash-bound human-accepted C3a poses were replayed in
  real Isaac from their exact final base/joint/object state.  The runner
  authenticates the review decision, scene, morphology, generated URDF/USD,
  PhysicalModel, Order 8 contact thresholds, and its own immutable bytes.
  The IK pose is a generated-URDF `baselink` pose, so Isaac spawn uses the
  hash-bound generated-URDF root-to-baselink transform rather than writing it
  directly as the articulation-root pose.  Isaac collision filtering matches
  the existing production arbitrary-morphology runtime: all same-module
  internal pairs and the intended inter-module Dock pairs are filtered, while
  every cross-module non-Dock pair and every non-selected robot/object pair is
  monitored.  This avoids treating the known Isaac convex/importer
  `main_body`/`thrust_1` same-module overlap as morphology self-collision; the
  original exact-mesh/FCL gate remains responsible for the static geometry
  check.
- A raw PhysX patch alone is not physical-contact evidence.  The static
  diagnostic reuses the hash-bound Order 8 thresholds and classifies a finite,
  unsaturated patch as physical only when absolute solver force is at least
  `0.5 N` or separation is at most `-0.1 mm`.  It archives pair-level raw and
  physical counts, maximum force, minimum separation, exact-pose readback, GPU
  resource evidence, and the missing-path limitation.  Five fixed-SHA runs
  used runner SHA-256
  `f247752193ac51821237dfd431c8816b344b84e28ba3bfded88b15a7e89320f2`;
  all five passed the narrowly named
  `static_pose_collision_contact_pass`: both assigned bodies had physical
  object contact, selected penetration remained below `2 mm`, and there was
  no cross-module non-Dock or non-selected robot/object contact.
- This is **not** production C3 bucket admission.  The reviewed artifacts bind
  only one final pose and contain no collision-admitted approach/closure path,
  so all five retain `not_run_missing_bound_path` and the production-eligible
  count is zero.  Static-spawn peak selected force also exceeded the unchanged
  Order 8 `30 N` hard limit in three cases: `52.049 N` for the six-module
  branched pose, `249.477 N` for the seven-module chain pose, and `37.492 N`
  for the new eight-module branched pose.  The six-module chain and retried
  eight-module branched poses measured `23.061 N` and `29.488 N`,
  respectively.  These direct-penetration spawn transients are reported as a
  limitation rather than silently used as either a static-pose rejection gate
  or a runtime force pass.  No bucket was promoted and no learning/checkpoint
  state changed.
- Method-level decision remains open: either reduce/regenerate the nominal
  contact penetration and require the `30 N` bound in this direct static
  diagnostic, or bind and replay the actual collision-admitted
  approach/closure trajectory and apply the unchanged runtime force gate
  there.  Until that choice is approved and the missing path is supplied,
  static human/Isaac results cannot be promoted to C3 production buckets.

### Order 9 Selected-Contact Penetration Hard Gate

- The ordinary convex/FCL collision gate continues to exempt only the
  explicitly assigned anchor-link/object pairs from its non-contact `5 mm`
  clearance requirement so that intended contact remains representable.
  Those exempt pairs are no longer unchecked: the native kernel separately
  evaluates their convex signed distance and fails final configuration
  admission if maximum penetration exceeds `2 mm`.  This limit reuses the
  existing Order 8 selected-contact `max_penetration_m` contract rather than
  introducing a new task-specific tolerance.  A pair at positive separation
  is not rejected by this penetration-only gate; physical contact existence
  and force remain later Isaac/runtime evidence.
- Admission now archives the penetration limit, maximum selected-contact
  penetration, violating-pair count, and per-selected-link signed-distance
  records.  The selected pairs remain outside the ordinary collision penalty,
  so an invalid contact geometry is rejected rather than silently converted
  into a `5 mm` non-contact gap.  The collision-aware solver/gate identity
  advances to
  `centroidal_posture_ik_cpp_eigen_fcl_convex_v2_contact_penetration`;
  deployments must rebuild the hash-bound native extension.  No policy,
  TaskSpec, morphology, trajectory, controller, reward, dataset, bucket, or
  checkpoint schema changes.
- The corrected gate separated the completed first human review exactly as
  intended: accepted yaw contact links measured approximately `1.5 mm`
  nominal convex penetration, while the four user-rejected poses contained
  selected pitch-link penetration of approximately `42--47 mm`.  This is
  convex-proxy static-pose evidence, not original-STL or Isaac contact,
  dynamics, path, grasp, or transport evidence.
- A separate five-case review batch preserves all four rejected structural
  hashes and substitutes penetration-admitted yaw/yaw assignments; its fifth
  case is a previously unused train-pool eight-module branched morphology.
  All five have zero selected-contact and ordinary collision violations.
  Maximum selected-contact penetration spans `1.472--1.542 mm`, and minimum
  non-contact convex clearance spans `5.077--6.246 mm`.  The batch is
  hash-bound and remains pending human review; it does not mutate the
  completed twelve-case decision archive or any production C3 bucket.

### Order 9 Controller-Side Load-Limited Contact Preload

- User-approved responsibility supplement: `pi_H` remains joint-free and the
  deterministic posture resolver still supplies the nominal joint trajectory
  consumed by learned `pi_L`.  After both assigned contacts are physically
  observed, a deterministic controller-side preload stage runs after `pi_L`
  and before QPID.  It is not a learned-policy output and does not change the
  `PolicyCommand` or `ControllerCommand` schema.
- The preload rule is the proven Order 8 rule, without an added wrench or
  torque bias.  It retains the fixed IK closure ratio, advances the previous
  absolute Dock position target at a maximum `0.002 rad/s`, observes
  damping-compensated actuator load `abs(applied torque - virtual damping
  torque)`, and freezes each anchor's moving kinematic branch after its load
  remains at or above `1.2 Nm` for `0.10 s`.  A joint shared by multiple
  branches freezes conservatively when the first owning branch freezes.
- Order 8 armed this stage only after simultaneous mesh-proximity/load arrest
  and a settled q-close.  The Order 9 Isaac/runtime boundary represents that
  surface-arrest prerequisite with both assigned measured contact forces at
  or above the existing maintained-contact floor `0.5 N` for `0.10 s`; a
  single contact sample or one-sided contact cannot arm preload.
- The frozen absolute Dock targets remain the deterministic nominal hold
  through lift, transport, and place; release returns authority to the normal
  resolver/`pi_L`/QPID path.  Raw Isaac contact is not added to the learned
  actor observation.  The contact-acquisition phase gate additionally
  requires preload completion, so lift cannot begin from contact-force
  evidence alone.
- Arbitrary-morphology production refinement (2026-07-29): the tensor
  collector now implements this previously approved controller stage rather
  than leaving it only in the scalar copied/shadow runtime.  Multi-morphology
  Isaac evidence showed that a selected contact can be loaded by whole-
  structure motion while its anchor-local Dock axis remains below `1.2 Nm`;
  requiring only that local-axis threshold caused a false contact timeout in
  the two-module validation morphology.  An anchor therefore freezes after
  either the unchanged `1.2 Nm / 0.10 s` moving-branch load dwell or a
  `0.10 s` dwell above the signed normal-force magnitude at the outer edge of
  the `pi_H`-proposed wrench interval.  Using the interval outer edge rather
  than its lower edge leaves relaxation margin after the absolute target is
  frozen.  This is controller/privileged evidence only: raw contact remains
  absent from the actor, the full six-axis wrench interval remains a reward
  objective rather than an instantaneous gate, and `pi_L`, QPID, and `pi_H`
  output schemas are unchanged.  Representative real-Isaac evidence changed
  the former three-module failure from `0/8` contact timeouts to `8/8` full-
  task success; the two-module regression crossed contact and all subsequent
  task phases within the formal `15000`-step budget.

## 2026-07-24

### Order 9 Interim Convex-Only Posture Collision Gate

- The user approved suspending the original-STL final-admission path for the deterministic posture resolver and using the lightweight convex collision model as the interim online hard gate.  Each authored URDF collision mesh is conservatively replaced by its convex hull, the configured `5 mm` clearance remains fail-closed, and prohibited robot/self and robot/object pairs are checked during IK and once again before admission.  The exact-mesh checker may remain as an offline comparison oracle, but it is not on this interim runtime path and its success is not required for runtime admission.  This supersedes the earlier same-day statements that required original-STL admission specifically at the posture-resolver boundary; it does not disable independent Isaac collision events/telemetry during simulator training or evaluation.
- The convex fast path may use mathematically conservative AABB lower bounds to prune pairs only when the AABB separation already exceeds the active clearance threshold.  World poses, object shapes, morphology pair topology, and warm-start state may be cached without changing contact, joint-limit, CoM, anchor, timing, or margin semantics.  Intended docks, kinematically adjacent links, authored nominal within-module overlaps, and explicitly assigned anchor/object contact links retain the same allowed-collision treatment as before.
- A proxy-only runtime must not infer its authored-overlap exclusions from the convex hulls themselves.  The current Holon exclusion table is generated once with the exact-capable PC diagnostic, binds the URDF and every collision-mesh SHA-256 plus ordered collision-link identity, and is then consumed fail-closed on PC/aarch64.  Missing or mismatched identity data rejects initialization; an accidental exact-geometry query on a proxy-only kernel also fails closed.
- This amendment changes no persisted schema or policy/controller interface.  Following the explicit production-promotion request, the complete fixed-centroidal IK loop now lives in the host-built C++/Eigen kernel under `amsrr/feasibility/native`, and the proxy-only convex/FCL gate plus its hash-bound allowed-collision manifest are invoked by the production posture-resolving `C_H` path.  Ordinary offline resolver use prefers this native solver but may retain the readable Python solver when no host extension is installed; production collision admission never falls back and fails closed if the extension, build identity, object box identity/size/current pose, URDF/mesh-bound exclusion manifest, or separate post-solve `5 mm` convex admission is unavailable or inconsistent.  The compiler is never invoked by runtime code; each x86-64/aarch64 deployment builds once through `scripts/build_order9_posture_native.sh`.

### Order 9 Learned pi_H to Deterministic Posture-Resolver Boundary

- User-approved responsibility split: learned `pi_H` continues to run at the high-level rate and proposes the task/contact trajectory only: contact assignment and schedule, contact-wrench targets/bounds, assembled-centroidal pose/twist targets, selected free-anchor pose targets, object targets, priorities, guards, and timing.  Learned `pi_H` does not propose Dock joint position or velocity targets.  A deterministic posture resolver consumes the unchanged `pi_H` proposal plus the measured current joint state and produces a detached, controller-rate nominal joint trajectory.  The existing learned `pi_L` then emits bounded corrections around that nominal trajectory, and QPID/QP plus the local servos retain final actuator authority.
- User-approved rolling-teacher correction: each `pi_H` proposal is a phase-local `1--3 s` receding-horizon trajectory whose first knot is the measured current CoM/anchor state.  The teacher advances CoM and free-anchor targets gradually and replans from the next measured state; it must not compress the complete approach-through-settle task into one short proposal or jump its first knot to a terminal IK pose.  Pregrasp approach, contact acquisition, lift, transport, place, release, retreat, and settle remain distinct phase semantics.  The deterministic approach teacher may stage vertical clearance, lateral clearance, articulation, and final pregrasp motion, but real-mesh/persistent-Isaac checking remains the collision authority.
- The persisted v0.4 `PostureTarget` schema is intentionally unchanged.  In a raw learned-`pi_H` proposal, `joint_pos_target` and `joint_vel_target` must both be absent while `free_anchor_pose_targets` remains policy-owned.  The resolver deep-copies the proposal and is the only component allowed to populate those two joint fields on the resolved copy.  Raw and resolved trajectory hashes, resolver version/config, per-knot feasibility, and timing evidence are recorded separately.  The raw proposal is never mutated, projected, or silently replaced, preserving the existing archive-before-`C_H`, rejection-transition, and no-fallback-credit rules.
- The resolver holds each proposed assembled CoM position and body orientation fixed, warm-starts from the measured/previous joint state, solves the selected anchor constraints subject to PhysicalModel joint limits, rejects any solution that would violate the unchanged policy-owned timing or configured joint-rate limit, and emits nominal samples at the configured `10--20 Hz` posture rate.  If the exact proposed CoM/anchor/timing combination cannot be realized, it fails closed; changing the `pi_H` proposal to make it feasible is not permitted.  Collision and contact-wrench authority remain independent hard checks.  The initial resolver implementation supplies deterministic kinematic resolution plus a fail-closed validation hook; production admission still requires the existing real-geometry/persistent-Isaac collision path and does not infer collision clearance from IK success.
- The deterministic articulated teacher may use its free-base IK internally to construct feasible supervision for CoM and free-anchor targets, but its exported raw `pi_H` teacher trajectory contains no joint targets.  It then invokes the same posture resolver used at the learned-policy boundary to create the resolved trajectory consumed by `pi_L`.  The teacher-only IK seed is evidence, not a hidden learned-`pi_H` output or a runtime action.
- Learned-`pi_H` optimization receives downstream task return together with resolver acceptance and quality signals such as IK failure, contact/anchor residual, joint-limit/rate margin, collision-check result, and solve cost.  It is not trained to reproduce one exact joint-angle solution.  The deterministic resolver is outside learned `pi_H` and outside `C_H`; `C_H` still evaluates the unmodified raw high-level proposal and the resolved execution candidate at their respective contract boundaries without becoming a planner.
- Production shadow admission accepts the full already-specified `1--3 s` `pi_H` horizon.  The hard-checker maximum copied rollout is therefore `3.0 s`, not the earlier `2.0 s` implementation default.  This does not retime or project a proposal: a proposal above `3.0 s` still fails closed, while resolver-produced trajectories such as the current `2.226 s` conservative teacher can reach the unchanged real-geometry/wrench/controller checks instead of being rejected before physics.

### Order 9 C3a Twelve-Case Human Actual-Mesh Curation Pilot

- Before regenerating production C3 buckets, the user approved a human-in-the-loop two-contact pilot covering `12` morphology cases: one chain each at `2` and `3` modules, and one chain plus one branched graph at every module count `4--8`.  Candidate graphs come from the split-owned random-connected train pool except for the eight-module chain: the current train pool contains no eight-module chain, so that one case is generated deterministically from the same morphology distribution, passes the existing morphology-flight feasibility check, and remains an isolated train candidate rather than being inserted into the source pool or a production bucket.
- Curation now has two explicit inspection stages.  The neutral stage renders a top view of the actual URDF visual meshes with every free mesh-backed Dock surface labelled.  After a surface pair and object contact-candidate group are pinned, the existing articulated deterministic teacher solves the final grasp pose and an offline interactive viewer renders the actual URDF visual and collision STL instances, object geometry, selected surfaces, contact targets, and normals.  The browser viewer is an inspection aid only: it does not establish collision clearance, collision-free interpolation, controller tracking, dynamics, grasp, transport, or task success.
- Human selections are reproducible rather than advisory.  The articulated-teacher configuration accepts optional exact contact-candidate-group and surface-port-pair selectors; when present, it fails closed instead of silently choosing another candidate.  Defaults preserve the existing exhaustive automatic search, and no persisted morphology, task, trajectory, feasibility, policy, controller, dataset, or checkpoint schema changes.
- The pilot pins yaw-surface pairs for all `12` cases.  Every exact assignment passed the unchanged articulated IK tolerances and the independent hard reachability recheck.  Across the set, maximum contact position error is `0.538 mm`, maximum contact-normal error is `1.0683e-5 rad`, and maximum absolute pitch target is `0.000578 deg`.  These are kinematic results only.  Every case remains `pending_user_accept_or_reject`; only user-accepted poses may proceed to real-Isaac mesh collision/path replay, and only a subsequent Isaac pass may make a case eligible for production bucket admission.

- Collision-aware curation v2 supplement (supersedes the v1 final-pose
  selection, not its historical evidence): each pinned surface/contact
  assignment now passes through the production native fixed-centroidal
  convex/FCL posture IK and a separate post-solve `5 mm` collision admission
  before it is presented for human review.  The curation-only build does not
  require the production transport/release trajectory recheck because the
  reviewed artifact is one static final grasp pose; this exclusion is
  explicit in its metadata and cannot authorize a runtime trajectory.
  Production teacher calls retain the full recheck by default.
- Five v1 free-base-only assignments were rejected by the new collision-aware
  admission and replaced by the first feasible pair in the unchanged
  deterministic surface-pair ordering: 5-module branched `[14,18] / group 2`,
  6-module chain `[13,21] / group 3`, 6-module branched `[8,19] / group 1`,
  7-module chain `[16,27] / group 0`, and 8-module branched
  `[18,29] / group 1`.  All twelve v2 candidates have zero prohibited
  violating pairs; minimum convex clearance ranges from `5.099` to
  `10.534 mm`.  This remains convex model evidence rather than original-mesh
  or Isaac path/dynamics evidence.
- The local interactive viewer now owns a review-only control plane:
  `accept`, `reject`, and deterministic `recalculate` actions are persisted
  to a scene-hash-bound `review_decisions.json`.  Recalculation keeps the
  pinned morphology/surface/contact assignment and searches the next
  deterministic CoM/joint seed, then returns the case to pending review.
  These decisions never bypass collision admission, mutate a production
  bucket, or constitute Isaac acceptance.
- Curation-rendering frame correction: the collision-aware posture solution
  expresses `base_pose_world` at the PhysicalModel `fc`/generated-URDF
  `baselink`, whereas the WebGL scene places the generated URDF's root link.
  The first v2 render passed the former directly as the latter and therefore
  displayed every robot approximately `60.935 mm` above its solved pose even
  though native IK/collision admission used the correct frame.  Those review
  images are invalidated.  The viewer now derives `root_T_baselink` from each
  generated URDF and applies
  `world_T_root = world_T_baselink * inverse(root_T_baselink)`; scene
  generation also fails closed unless every rendered selected-anchor pose
  reproduces the IK solution.  The corrected viewer identity is
  `order9_c3_curation_mesh_viewer_v3_baselink_frame_review`.  Across the
  regenerated 12 cases, all 24 selected-anchor position residuals to their
  planned object contacts remain within the unchanged native `5 mm`
  tolerance (maximum `3.944 mm`, mean `0.341 mm`).  This correction changes
  only human-review geometry and provenance; it does not upgrade planned
  contact markers to measured Isaac contact evidence.

### Order 9 Articulated IK Pitch-Posture Preference

- User inspection of the offline articulated-teacher animation found avoidable whole-body twisting in which pitch Dock joints were driven to large angles even when yaw articulation could satisfy the grasp.  The deterministic IK teacher now supplements its existing all-joint `1e-6 * sum(q^2)` regularizer with `1e-2 * sum(q_pitch^2)`.  Pitch joints are resolved from the PhysicalModel `pitch_dock` port semantics and their mechanism-joint bindings, not from a hard-coded module count or runtime robot path.  This is a soft posture preference: contact position/normal tolerances, joint limits, the independent hard reachability recheck, and real-geometry collision authority are unchanged, so a morphology may still use pitch when required.
- The solver no longer returns the first iterate that enters the contact tolerances.  It completes the bounded 60-iteration damped-least-squares search, retains only hard-contact-feasible iterates, and returns the one with the smallest weighted contact-plus-joint regularized objective.  This makes the added posture objective participate in solution selection while preserving fail-closed contact feasibility.  IK/trajectory/C3/precheck identities advance to their respective v2 versions; no persisted trajectory, feasibility, policy, controller, or learning schema changes.
- A deterministic recomputation over all 42 existing C3 task/structural conditions admitted every condition.  Maximum contact position/normal errors were `4.563 mm / 0.01005 rad`; the largest absolute pitch target was `1.882 deg` and the largest pitch-vector L2 norm was `2.702 deg`.  In the representative eight-module condition `train-000006-05afa5d407ec`, selected surfaces changed from `16/28` to `16/30`, maximum absolute pitch fell to approximately `1.1e-7 deg`, yaw-vector L2 was `117.33 deg`, and contact position/normal errors were approximately `0.00138 mm / 4.26e-6 rad`.
- The recomputation above is model-level teacher evidence, not Isaac collision, tracking, or grasp-success evidence.  The old production bucket manifest intentionally remains immutable while the user inspects the new animation; its v1 teacher evidence is now stale and the v2 production runtime fails closed against it.  After posture inspection is approved, regenerate and revalidate the 42 production buckets and all downstream hash-bound runtime evidence before starting C3 learning.

## 2026-07-23

### Order 9 C3 Split-Safe Teacher Buckets and Measured 1024-Environment Runtime

- C3 topology buckets now preserve the split-owned pool graph as the structural source and defer task-owned anchor creation to the approved articulated C3 teacher.  The old generic topology-provider anchor choice is not persisted as C3 authority.  Within each split, bucket selection is deterministically stratified over module counts `2--8`; every candidate must pass the complete articulated IK teacher and its independent reachability recheck before the bucket is admitted.  Compact evidence binds the teacher/IK versions, selected mesh-backed surface IDs, task-conditioned morphology, candidate set, complete trajectory, reachability margins, and IK errors by hash.  Runtime recomputes the teacher from the immutable task/structural graph and fails closed unless that evidence is exactly equal.
- The production C3 bucket set contains `28` train and `14` validation buckets: four and two conditions, respectively, for every module count `2--8`.  It is task- and structural-split safe and binds the v2 articulated USD asset manifest.  There are `25` unique train and `13` unique validation structures.  Repetition occurs only in the two-module stratum: exhaustive available-pair teacher search admitted only one structure in each split, so that structure is retained under distinct conservative object randomizations rather than silently substituting a different module count.  Across all 42 admitted conditions, maximum IK position/normal errors are `4.970 mm / 0.08170 rad`, maximum iterations are `60`, and every hard reachability margin is non-negative.
- C3 parallelism is selected from the actual eight-module active-knot production collector, not inherited from C2 or from the earlier three-module benchmark.  Single-process `512/1024/2048 x 64` diagnostics measured `2515.30/4431.70/6757.78 env-step/s`, setup times `76.87/165.74/350.05 s`, and peak GPU memory `6032/7903/11792 MiB`.  Although 2048 is valid alone, it is rejected for production because C3 collection launches train and validation simultaneously and retains a 256-step GPU rollout buffer per process.
- The production-sized capacity gate ran two distinct eight-module topologies concurrently at `1024 environments x 256 steps` per process.  Both finite artifacts completed; global peak GPU memory was `17162 / 24564 MiB` (`0.6987`), headroom was `7402 MiB`, GPU utilization peaked at `99%`, and conservative pair throughput was `5431.83 env-step/s` rollout-only and `2034.70 env-step/s` including each slower collector's setup.  The selected production count is therefore `1024` per process.  The standard runtime report is bound at SHA-256 `1d0592c4d623a6fd70efb537b1b4f87e3c9a95713cd0abde88d8299d28ed1bef`; its source artifact hashes and rejected 2048 diagnostic are retained.
- This is an operational runtime/bucket amendment, not a learning-method change.  `pi_L`, reward, PPO objective/hyperparameters, QPID/QP/safety, object randomization, C3 stage budget, and promotion gates are unchanged.  `production_runtime.selected_environment_count` and its benchmark binding move from the historical fixed-topology `128` result to the measured C3 `1024` result.  The curriculum schedule hash remains `47e7fb64303333c5207192382dfbb7c85c497d96c6df7e4070988534667f81bd`, preserving the C2-to-C3 initializer lineage.  At `1024 x 256` per split, one paired generation contains `524288` recorded interactions and the `7,000,000` budget resolves to `14` complete updates (`7,340,032` recorded interactions).

### Order 9 C3 Articulated Trajectory Teacher and Production Reachability Boundary

- This section resolves and supersedes the C3 reachability blocker recorded below.  The earlier inference that a roughly `3.30 m` neutral selected-surface separation made the sampled eight-module morphology rigidly incapable of matching a roughly `0.297 m` contact pair was incorrect: the morphology is articulated through all structural and free Dock mechanism joints.  Neutral-frame separation is now used only to order deterministic search, never as a hard feasibility test.  A full-Dock-column IK solve on the sampled eight-module structure reached a conservative contact pair with `4.49 mm` maximum position error and `0.0068 rad` maximum normal error.  Exact feasibility is decided at the resulting joint pose, not at the authored neutral pose.
- Terminology and authority are explicit.  `pi_H` continues to mean the learned high-level policy only.  The new `Order9ArticulatedTrajectoryTeacher` is a deterministic/offline trajectory teacher, not a pseudo-policy deployed as `pi_H`.  It enumerates grammar-valid mesh-backed surface assignments, solves deterministic damped-least-squares IK over every global Dock joint plus the free assembled-body base, and emits the complete existing `ContactWrenchTrajectory` fields: contact assignments/wrench ranges, assembled-CoM position/velocity/orientation, the full absolute Dock joint position/velocity maps, object targets, and active free-anchor pose targets.  It retimes the pregrasp segment against the PhysicalModel-bound `3 rad/s` safe Dock velocity.  Candidate contact poses move rigidly with their target object's knot pose.
- The same whole-structure kinematics is used by a separate `ArticulatedTrajectoryReachabilityEvaluator` in production `C_H`.  The checker never runs IK or changes/ranks/projects a proposal.  It fails closed on missing/incomplete posture or centroidal targets, joint-set/limit/rate violations, inconsistent CoM-to-base reconstruction, active-anchor target mismatch, contact-position error above `10 mm`, or contact-normal error above `0.20 rad`.  Its codes and signed margins are merged into the existing trajectory feasibility result and checker metadata, so no persisted trajectory/result schema was changed.  The production factory requires this evaluator in addition to the existing contact-wrench QP and persistent Isaac shadow collision/wrench evaluator.
- RobotAnchor frame semantics are corrected at the source.  Sequential grammar anchors now store the connect-frame joint origin relative to the declared Dock mechanism `link_id`, rather than copying the module-frame `DockPortSpec.local_pose`.  They also carry the same mesh/collision/port provenance as the accepted fixed-grasp morphology path.  The grammar identity is advanced to `order9_holon_sequential_design_grammar_v2_mesh_anchor_frame`; older sequential records are not silently reinterpreted.
- For topology-randomized C3, the structural split-owned graph remains unchanged.  A task-conditioned adapter enumerates distinct-module free Dock-surface pairs, asks the grammar to bind the selected pair, samples contact candidates, runs the articulated trajectory teacher, and independently rechecks the completed trajectory.  Structures for which no tried pair is feasible fail closed and are resampled by the existing curriculum/topology boundary; object dimensions or module-count strata are not silently widened.  The C3 tensor runtime consumes the teacher's complete active-knot trajectory and derives arbitrary-morphology joint reference banks from its posture targets instead of requiring the three-module canonical joint table.
- Collision authority remains with real geometry.  The articulated evaluator proves joint/CoM/contact kinematic consistency and velocity/limit compliance; it does not approximate full-body collision clearance.  Teacher admission still requires real-Isaac replay, while learned proposals additionally pass the existing persistent-shadow collision/wrench path in production `C_H`.  A one-environment/two-step, no-training real-Isaac C3 smoke on an eight-module articulated USD passed with finite state, exact body/joint identity, QPID execution, and a written raw rollout artifact.  This smoke establishes runtime compatibility only; it is not C3 learning or a task-success/promotion result.

### Order 9 C3 Active-Knot Actor Migration Boundary

- The successor C3 `pi_L` actor consumes the deployable active `InteractionKnot` contract in addition to the unchanged C2 observation path.  Its added global/node residual branches cover assignment schedule/mode, centroidal/posture/object targets, contact-wrench target/bounds, phase timing/progress, morphology/physical-model summaries, and controller status.  Raw contact forces and simulator-only truth remain outside the actor input and remain available only to reward, safety, and evidence paths.
- The promoted v2 C2 actor may enter C3 only through the explicit `order9_pi_l_c2_v2_to_c3_active_knot_v1` transformation.  Every legacy parameter is copied exactly and every new residual output is initialized to zero, so the migrated initializer exactly reproduces the source actor before C3 updates.  The migration manifest binds source/target checkpoint bytes, both curriculum hashes, the v2-to-v3 lineage import, and the C2 promotion manifest.  The result is an initializer only, not a relabelled C2 promotion; the C3 physical gate remains the first v3 promotion.
- Historical blocker (resolved by the section above): preparation exposed that graph-distance anchor selection plus `F_COARSE_REACHABILITY` did not prove joint-aware contact reachability.  The user subsequently approved task-conditioned surface enumeration, articulated IK teaching, and hard proposal rechecking; these are now implemented and have passed the stated model-level and no-training Isaac preflights.  This historical paragraph no longer blocks C3.

### Order 9 User-Approved Progressive pi_L/pi_H Object-Condition Curriculum Amendment

- Scope and supersession: the user approved replacing the post-C3 portion of the 2026-07-20 Order 9 curriculum with a progressive object-condition curriculum that explicitly resolves the learned `pi_L`/learned `pi_H` bootstrap dependency.  C0--C2 results and the intent of C3 are unchanged: C3 first expands `pi_L` over grammar-valid 2--8-module morphologies while retaining the conservative Order 8 object anchor.  The old executable-v2 C4--C10 ordering, in which `pi_L`/`pi_H` stayed conservative before object diversity appeared first in `pi_D` training, is superseded and must not be started as the active post-C3 curriculum.
- Dependency resolution: neither untrained learned policy may serve as the other's teacher.  For every new object-condition ring, a shared deterministic/offline trajectory teacher first produces paired supervision: a complete schema-level `ContactWrenchTrajectory` for `pi_H`, and the corresponding controller-facing low-level reference/rollout for `pi_L`.  Teacher output must be geometry-, task-phase-, and morphology-aware, must pass deterministic reachability/contact-wrench/actuator/collision checks, and must succeed in real-Isaac replay before entering the learning archive.  The checker `C_H` remains an accept/reject gate for learned `pi_H` proposals; it does not generate, repair, project, or rank teacher trajectories.
- `pi_L` entry contract: before broad object-condition learning, the production tensor actor path must consume the deployable active-knot information from the existing `ContactWrenchTrajectory` contract, rather than depending on an Order-8-canonical body/joint-target shortcut.  The required information includes the active contact assignment/mode, centroidal and posture targets, object target, contact-wrench target/bounds, phase/timing/progress, morphology/physical-model summary, and controller status, subject to the existing actor-versus-privileged-input boundary.  This is alignment with the v0.4 `pi_H -> pi_L` interface, not authority for `pi_L` to bypass QPID/QP, safety, actuator mapping, or local servos.
- Per-ring training order: each object-condition ring executes the same ordered cycle.  (1) Generate, hard-check, real-Isaac replay, split, and archive teacher trajectories without silently replacing failed conditions.  (2) Train `pi_L` with the archived ground-truth high-level trajectories by teacher forcing/BC and, where physical tracking requires it, `pi_L` PPO while the high-level source remains the fixed teacher.  (3) Train learned `pi_H` by assignment warm-up and complete-trajectory BC from the same archive; this offline step may overlap `pi_L` training computationally, but it cannot authorize online learned-`pi_H` execution.  (4) Promote `pi_L` on held-out teacher trajectories, freeze it, then train/evaluate learned `pi_H` online through production `C_H` and the actual `pi_L`/QPID/QP/Isaac stack.  (5) Freeze the accepted learned `pi_H` snapshot, collect its accepted non-fallback output distribution, and readapt `pi_L` with DAgger-style supervised data and/or bounded on-policy training.  (6) Run a coupled physical gate over the new ring and all earlier rings before widening the distribution again.  Rejected `pi_H` proposals train `pi_H` through the existing rejection-transition rule and are never passed to `pi_L`; deterministic fallback return is credited to neither learned policy.
- Progressive rings: `R0` is the current conservative Order 8 anchor.  `R1` retains the canonical box size/shape and expands reachable object position and yaw.  `R2` adds box size, aspect ratio, density, mass, CoM, and inertia variation while retaining the validated R1 pose envelope.  `R3` adds the box/sphere/cylinder training primitives and their feasible pose ranges.  `R4` trains the cross-product of morphology, position, size/aspect, shape, mass properties, and estimator error rather than only one-factor-at-a-time cases.  `R5` is evaluation-only and contains disjoint capsule/shape/aspect/inertia and reachable-pose conditions.  True mass properties continue to come from analytic or mesh-based constant-density computation; estimator perturbations remain separate.  Exact reachable bounds and per-ring sample/step budgets are calibrated from teacher feasibility yield, Isaac success, and measured runtime rather than copied from the insufficient `+/-0.015 m` expanded-position placeholder.
- Downstream order: structured `pi_D` BC and masked PPO begin only after the `R4` learned `pi_L`/`pi_H` stack is promoted, so morphology candidates are evaluated with downstream object-task policies that already cover the training distribution.  They are followed by the existing limited joint object-task fine-tune and held-out full-system evaluation.  Dynamic assembly remains deterministic and separately evaluated, with only the already approved configured subset executed end to end.
- Promotion and regression boundary: every ring must bind task/morphology/object-condition identities, teacher/checker/simulator/config hashes, train/validation/held-out membership, policy checkpoints, fallback decisions, runtime telemetry, and per-ring plus cumulative success.  A later ring cannot promote by averaging away regression on an earlier ring.  Learned `pi_H` online promotion still requires unmodified proposals, production `C_H`, at most two learned attempts, zero fallback credit, and the configured safety/fallback gates.
- Version/lineage boundary: the parsed `order9_curriculum_v2` schedule semantics and hash remain available only to validate the already promoted C0--C2 lineage.  They must not be rewritten or relabelled under the amendment.  Before C3 starts, an executable successor schedule and a provenance-bound lineage import must reference the promoted C2 checkpoint SHA-256 `85474d9da96a6eb729e7f21fae5f7628fd0d9defc39488c74fe71e10476c6274` and promotion-manifest SHA-256 `1572587d5a27e90a92ba3f2575dfc7c3fb722513618c1c368c78aff098df11ed`.  If active-knot actor inputs require a model-config migration, compatible weights may initialize the successor C3 model only through an explicit recorded transformation; the imported initializer is not relabelled as a v3 C2 promotion.  The successor C3 physical gate establishes the first promoted checkpoint under the new schedule.

### Order 9 Progressive Curriculum v3 Materialization and v2 Lineage Import

- The canonical `configs/training/order9_learning_curriculum.yaml` is now `order9_curriculum_v3_progressive_object_conditions`, with schedule hash `47e7fb64303333c5207192382dfbb7c85c497d96c6df7e4070988534667f81bd`.  The historical parsed v2 schedule is retained at `configs/training/order9_learning_curriculum_v2.yaml` and still hashes to `c88d6816f754804621b74dca269f04488d469d9d709b2aee334fab7affc0cf9d`.  The v3 C0--C2 stage dictionaries are exactly equal to the v2 prefix; v3 execution rejects those imported stages rather than creating falsely relabelled v3 C0--C2 artifacts.
- The executable ordering contains 36 stages: unchanged C0--C3; seven ordered stages for each of R1--R4 (`teacher collection -> pi_L BC -> teacher-driven pi_L PPO -> pi_H assignment BC -> pi_H full-trajectory BC -> pi_H PPO with pi_L frozen -> pi_L readaptation with pi_H frozen`); post-R4 `pi_D` BC/PPO; joint object-task PPO; and held-out full-system evaluation.  Each readaptation stage retains a no-fallback cumulative physical gate, and schema validation rejects missing, reordered, or distribution-mismatched ring stages.
- Initial schedule-bound online budgets are `3,000,000` teacher-driven `pi_L` PPO steps, `5,000,000` frozen-`pi_L` `pi_H` PPO steps, and `2,000,000` frozen-`pi_H` `pi_L` readaptation steps per ring.  Teacher episode minima increase across R1--R4 as `100/150/250/400`; BC/evaluation minima likewise increase in the YAML.  These are initial bounded budgets, not evidence of sufficiency and not permission to bypass a failed ring gate.  Any later budget extension follows an explicit recorded boundary like C2 rather than silently changing an active checkpoint lineage.
- `configs/training/order9_curriculum_lineage_v3.yaml` binds both schedule hashes, the exact promoted v2 C2 manifest, and the exact v2 C2 `pi_L` checkpoint.  Validation checks the v2/v3 prefix equality, source and successor versions/hashes, promoted status/stage identity, manifest-to-checkpoint agreement, checkpoint bytes/family/metadata, and the C3 boundary.  The legacy checkpoint may initialize only v3 C3 update zero; every resulting child is written under the v3 schedule hash.  `scripts/order9_stage.py validate-lineage` performs this read-only audit.
- This materialization does not start C3 and does not claim the missing broad-condition teacher/randomizer/runtime adapters or complete active-knot tensor input are implemented.  C3 remains blocked from execution in this work package by the user's explicit instruction; before a later C3 start, the active-knot model/runtime compatibility decision and C3 parallelism selection remain required.

### Order 9 C2 User-Approved Bounded On-Policy Extension

- After the configured `3,000,000`-interaction C2 quota failed promotion and the checkpoint audit found continued improvement rather than checkpoint regression, the user approved additional C2 on-policy training with the current learning rate, PPO objective/hyperparameters, phase-balanced collection, randomization, controller, safety, reward, and promotion gate unchanged.  The first bounded control interval is four complete paired generations, updates `46--49`, adding `4 x 65,536 = 262,144` fresh recorded train-plus-validation interactions and ending at `3,276,800` cumulative recorded interactions before another bucket evaluation.
- The canonical curriculum YAML remains unchanged because `environment_steps` participates in the schedule and stage hashes already bound into the C1/C2 checkpoints and rollout-bucket manifest.  The resumable C2 runner instead accepts an explicit non-negative complete-generation extension count, records the configured target, base/added/total update counts, and actual generation-aligned target, and continues the existing contiguous checkpoint lineage.  A negative extension fails closed; an omitted or zero extension retains the original update count.
- This authorization adds training budget only.  It does not promote C2, authorize C3, lower the `0.85` success threshold, alter checkpoint selection, reuse an old rollout, or permit partial generations.  Every added update still requires a fresh hash-bound train/validation rollout pair, exact behavior replay, one PPO update, runtime telemetry, and an immutable child checkpoint.
- Execution result: updates `46--49` completed all four epochs without KL early stopping; mean approximate KL decreased from `0.007399` to `0.006055`, and the final checkpoint SHA-256 is `85474d9da96a6eb729e7f21fae5f7628fd0d9defc39488c74fe71e10476c6274`.  The same fixed 8+8 full-task seed control improved from update 45's `10/16` to `16/16`, after which the acceptance-eligible deterministic phase-zero full-mesh evaluation passed bucket 8/9 at `100/100` and `100/100`.  Success and no-fallback success rates are both `1.0`, fallback and safety-failure counts are zero, and measured aggregate training-rollout throughput is `5854.253 env-step/s`.
- The unchanged promotion pipeline therefore promoted C2 with no failed gate.  The evaluation report and promotion-manifest SHA-256 values are `f1e123308f37669795a6206fd2eb64191c87c78c62d7b00ac719db57f25ef2f1` and `1572587d5a27e90a92ba3f2575dfc7c3fb722513618c1c368c78aff098df11ed`; the ignored machine-readable extension summary has SHA-256 `ca7b5779d4479bc7e0aa22f8ed539f56f84f271ac95c2fa40d412a19aa990698`.  C3 was not started by this promotion operation.

### Order 9 C2 Diagnostic Checkpoint-Audit Boundary

- Checkpoint diagnostics may initialize every copied environment from one existing hash-bound canonical phase only when explicitly requested.  The diagnostic records the chosen runtime phase, deterministic-policy flag, and whether all environments began at phase zero.  It is forbidden to combine a nonzero diagnostic phase with promotion JSONL output, and normal C2 collection retains its phase-balanced reset distribution while every promotion evaluation remains deterministic, first-terminal, and phase-zero.  This is an additive diagnostic interface; it changes no policy input/action, recurrent state, task plan, reward/gate, PPO update, QPID/QP, safety, randomization, or promotion rule.
- Phase-isolated results are not checkpoint-ranking evidence.  With final update 45, direct transport initialization passed bucket 8/9 at `32/100` and `100/100`; lift initialization followed by transport likewise passed at `32/100` and `100/100`.  This ordering is the reverse of the full task (`100/100` versus `24/100`) because a canonical mid-task reset omits the physical state and GRU history produced by learned approach/contact/lift execution.  Such runs may diagnose one phase but cannot replace an end-to-end checkpoint comparison.
- The acceptance-ineligible full-task audit used deterministic actions, canonical phase-zero starts, the same validation buckets, and fixed seed subsets.  Bucket 9 results progressed from update 24 `0/8` with eight drops, through update 32 `0/8` with seven drops, update 40 `0/8` with eight timeouts and no safety failure, and the most promising late intermediate update 44 `0/8` with eight timeouts and no safety failure.  Update 40 passed bucket 8 at `8/8`; the already archived update-24 bucket-8 check also passed `8/8`.  The same first eight formal-evaluation seeds at update 45 passed bucket 8/9 at `8/8` and `2/8`, so its `10/16 = 0.625` diagnostic subset exceeds the at-most `0.5` aggregate of every checked intermediate.
- The audit therefore does not support mistaken final-checkpoint selection as the cause of rejection.  It shows continued improvement from contact loss and position error toward retained-contact orientation-limited transport, with update 45 the best observed checkpoint but still only `0.62` over the formal 200 episodes.  This representative audit does not exhaustively execute every update and cannot introduce a best-checkpoint selection method, promote C2, lower the `0.85` success gate, add training budget, or authorize C3.  Its ignored machine-readable summary is hash-bound under `artifacts/p4_full/order9/stages/c2_pi_l_ppo_fixed_conservative/checkpoint_audit/` with SHA-256 `6b1e0a337c6385923e3c1be23a230b6ae03c640203b4135b96b0a19ff418a362`.

### Order 9 C2 Planned Phase-Reference Continuity Correction

- A deterministic full-task check after accepted C2 update 24 exposed a runtime bookkeeping defect rather than a policy or PPO failure.  On every successful phase transition, the tensor rollout had initialized the successor phase from the observed physical state.  Because each phase-success gate intentionally permits a bounded non-zero tracking error, that choice accumulated the tolerated error over `lift -> transport -> place`; in the eight checked validation episodes it moved the final place target below the support surface and produced eight phase-5 timeouts.  The same C2 adapter/bucket/seeds with the promoted C1 checkpoint failed earlier at lift, excluding update-24 policy collapse as the cause.
- A successful transition now advances the deterministic task plan from the completed phase's planned robot/object endpoint.  The physical state is not overwritten, reset, attached, projected, or hidden from the controller; the learned `pi_L` and QPID/QP continue to close the bounded physical tracking error.  The semantics are explicitly archived as `planned_phase_goal_v1` in rollout metadata and the simulator hash so artifacts using observed-state re-anchoring cannot be silently mixed with corrected evidence.
- This is an implementation-continuity correction.  It changes no phase target, phase-success tolerance, reward term, policy observation/action, PPO sample or objective, QPID/QP authority, safety gate, randomization, curriculum quota, or promotion threshold.  Phase-specific reset-bank collection remains unchanged; only a genuine in-episode successful transition uses the completed planned endpoint as the next phase's reference.
- An exact post-correction A/B rerun used update-24 checkpoint SHA-256 `2eb1f60b8481d5a0067f6bad33eff32ab58cc84f84c308f1775a6560232c827a`, validation bucket 8, seeds `9019--9026`, deterministic actions, and first-terminal full-mesh execution.  All `8/8` episodes reached settle successfully with zero fallback, safety failure, collision, drop, QP-infeasible terminal, or timeout.  The corrected raw artifact and evaluation JSONL SHA-256 values are `0c7efe396fcf87449d2731ddd78ff48926fc43f2f708ffdb648dcbc6812659e1` and `f123a397627b5baa4c31d0f684510e1b3eb41e45e7d6f1e90f9d7d65f7ee527b`.  This bounded check authorizes continued C2 updates but does not replace the required final minimum-200-episode promotion evaluation.

## 2026-07-22

### Order 9 C2 Exact Behavior Replay Correction

- The first production C2 generation exposed an implementation mismatch before continued training: the copied-environment tensor collector encoded module poses after subtracting each Isaac environment origin and summarized non-fixed joints only, while schema reconstruction replayed absolute world poses and all present joints.  The unchanged C1 behavior checkpoint therefore produced a first-minibatch approximate KL of `3.3323` and clip fraction `0.9539`; target-KL early stopping worked, but that child checkpoint is rejected diagnostic output and contributes no accepted C2 update or training-step quota.
- The `pi_L` behavior payload now binds both `actor_graph_frame_origin_world` and `actor_graph_joint_summary_semantics`.  Tensor-generated traces use `non_fixed_joints_only`; ordinary single-world traces use `all_present_joints`.  PPO reconstructs only the graph-encoder view in that recorded frame while retaining the original world-frame observation for centroidal actor features, QPID/QP records, reward, and evaluation.
- Before any optimizer step, every recurrent train transition must numerically reproduce the parent checkpoint's stored log probability, value, recurrent input/output chain, and previous action within an absolute tolerance of `2e-5`.  A mismatch fails closed and produces no child checkpoint.  This fulfills the already approved exact behavior-replay contract; it does not change the policy architecture, observation information, action authority, reward, PPO objective/hyperparameters, controller, task randomization, or curriculum method.
- A real generation-0 two-environment subset replayed after the correction with maximum log-probability error `7.6294e-6`, value error `3.3528e-8`, and recurrent-state error `3.5763e-7`.  Full-generation replay and the corrected update remain required before C2 may continue.

### Order 9 C1 Behavior-Cloning Actor Selection Correction

- C1 checkpoint selection is based on the learned `pi_L` actor errors only: normalized global-action loss plus normalized joint-action loss on the natural validation split.  The critic/value loss is not part of C1 checkpoint selection, and its C1 weight is zero; critic fitting begins with the on-policy PPO stage C2.  This prevents the much larger return scale from selecting a checkpoint whose controller-facing action is worse even though the actor is the sole C1 learning target.
- The completed C1 run uses 24 phase-balanced recurrent-window epochs and restores the minimum validation actor-loss state.  Its checkpoint metadata names `global_action_plus_joint_action` as the selection metric and separately retains total/actor diagnostics.  This changes no `PolicyCommand`, QPID/QP, safety, actuator, dataset-record, or policy-observation/action interface.
- C1 promotion remains an online physical gate.  Low offline actor loss alone cannot promote the checkpoint, and diagnostic checkpoints that emit zero normalized action are explicitly non-promotable even when they reproduce a teacher reference exactly.

### Order 9 Fixed-Nominal Articulation and Graph-Observation Contract Correction

- The C1 fixed three-module asset must reproduce the Order 8/C0 structural kinematics: each occupied graph Dock edge attaches the child module through the selected parent Dock mechanism, while the mating child Dock mechanism remains represented in the articulated tree.  A root-to-root fixed assembly is incompatible because structural Dock motion would move only a local link instead of the child module subtree.  Fixed C1 and cached arbitrary-morphology assets therefore use the graph-derived articulated child-reroot generator with hash-bound URDF/USD manifests.
- The scalar C0 graph tensorizer defines the learned observation contract.  Copied Isaac environment offsets are removed from module translations before graph encoding; controller/QPID state remains in the actual Isaac world frame.  Runtime joint summaries contain only non-fixed PhysicalModel joints (12 per Holon in the current model), matching the C0 `RuntimeObservation`, and runtime quaternions are normalized with the same non-negative-`qw` convention as the scalar tensorizer.  These are representation/invariance corrections, not new policy inputs or a change to `pi_L` authority.
- C1 physical promotion remains actor-only and fallback-free.  The corrected fixed asset and observation path were first checked with the exact C0 reference and a four-environment learned-policy smoke, then with 100 deterministic first-terminal real-Isaac episodes.  Neither deterministic baseline blending nor privileged contact input is introduced by this correction.

## 2026-07-21

### Order 9 pi_L Complete-PolicyCommand Correction

- Corrected policy ownership: the learned Order 9 `pi_L` emits the complete controller-facing `PolicyCommand` intent defined by the approved centroidal-only QPID supplement. The normal learned path is not `BaselineLowLevelPolicy + learned residual`. It consumes the active `pi_H` knot as its reference/context and directly emits desired assembled-centroidal pose/twist, additive centroidal wrench bias, absolute non-vectoring joint position/velocity targets, and bounded joint torque bias. QPID/QP, the local joint servos, actuator mapping, and safety remain downstream final authorities. `BaselineLowLevelPolicy` is retained only as a separately identified substitution fallback after learned-policy rejection/OOD/non-finite failure; fallback output and return are never credited to the learned policy.
- Versioned global action contract: Order 9 uses an 18-dimensional global action consisting of world-frame centroidal position correction (3), control-body-frame orientation rotation-vector correction (3), mixed-frame centroidal twist correction (6), and zero-centred body-frame centroidal wrench bias (6). Local joint heads emit reference-relative bounded corrections which are decoded to absolute position/velocity targets plus bounded torque bias. The reference is the active knot's centroidal and posture targets, with measured current-joint hold only where the knot omits a target. `trust_region_blend` is fixed at `1.0` for this contract because blending with or shrinking a deterministic command would restore the rejected residual-policy semantics; action bounds, downstream clipping, safety validation, and fallback still apply.
- C0 teacher-label correction: legacy Order 8 teacher collection stored a base-root pose in `CentroidalTarget` while the actual teacher `PolicyCommand` tracked the assembled morphology centroidal pose, and it did not archive the actual absolute local-joint targets in `PostureTarget`. Order 9 C0 collection now replaces those learning references with the exact assembled-centroidal target pose/twist and teacher joint position/velocity targets before writing each causal record. It then encodes every teacher command through the production Order 9 decoder inverse and fails closed if any normalized target lies outside `[-1, 1]` (apart from numerical tolerance).
- Compatibility boundary: the corrected policy/runtime/checkpoint/action semantics and tensor-rollout artifact contracts have new version identifiers. C0 data, C1 checkpoints, and tensor rollouts produced under the former baseline-plus-residual interpretation are incompatible and may not be mixed with the corrected lineage. The earlier partial C0 output is retained only as a recoverable historical diagnostic at `artifacts/p4_full/order9/c0_teacher_pre_complete_policy_command_v2_20260721`; it is not training evidence.

### Order 9 C0 Bounded-Diversity Teacher Budget

- User-approved quota correction: `100 episodes` is no longer interpreted as 100 high-fidelity deterministic Order 8 teacher repetitions. C0 now requires 20 successful, task-disjoint real-Isaac episodes: one exact nominal condition, four paired conservative boundary conditions, and 15 reproducible seeded conditions. The fixed three-module morphology is unchanged. Object dimensions stay within `0.975--1.025` of nominal, mass within `0.95--1.05`, initial standoff within `+/-0.005 m`, selected-Dock friction within `4.275--4.725`, and compliant-contact stiffness within `0.95--1.05`; damping is co-scaled by `sqrt(stiffness_scale * mass_scale)`. These are the already approved conservative Order 8 anchor bounds, not the broader later shape distribution. Any failed physical condition blocks C0 instead of being silently replaced or narrowed.
- Split/sampling contract: The 20 independent tasks are split `14 train / 3 validation / 3 held-out` by episode, never by adjacent frames. Low-level teacher labels are archived every five 50 Hz controller steps (10 Hz); skipped rewards are interval-aggregated, so return semantics are retained. C1 recurrent windows are then sampled uniformly across represented task phases while preserving the configured per-epoch window budget. Natural-distribution validation remains unbalanced. Every episode binds the condition ID, actual Order 8 config hash, strides, source hashes, and complete-policy-command representability result; incompatible earlier C0 shards are not reusable.
- Evaluation separation: C0 promotion uses the 20 physical teacher episodes and requires no safety failures. C1 still requires 100 learned-policy evaluation episodes for a useful success-rate estimate, but these are deterministic first-terminal evaluation rollouts of the learned checkpoint in copied real-Isaac environments, not additional teacher demonstrations. Thus teacher diversity/coverage and learned-policy statistical evaluation are separate budgets.
- Execution/telemetry: The production default is two concurrent Isaac teacher processes after one serialized asset conversion/hash audit. C0 and BC runtime records wall time, throughput, GPU utilization/VRAM/power/temperature, process RSS, system load, and system memory. Parallelism is an execution choice only and does not change conditions, split membership, or acceptance.

### Order 9 C2 Provisional Parallelism and Runtime Telemetry

- User-approved runtime selection: C2 alone uses a stage-local provisional `2048` copied Isaac environments and `16` control steps per environment. This preserves `32768` train samples per rollout shard, equal to the previous `128 x 256` PPO shard, so the parallelism experiment does not also change the PPO train-batch size. The formal `32/64/128` cold-start benchmark and its hash-bound report remain valid evidence for the conservative production fallback; `production_runtime.selected_environment_count` remains `128`, and C3 onward deliberately inherits that fallback until C2 learning evidence is reviewed. Later-stage parallelism must be selected from measured runtime/load data rather than silently inheriting `2048`.
- Runtime resolution contract: PPO stages may declare a paired positive `parallel_environment_count` and `rollout_steps_per_environment` override. Both values must be present together, are rejected on non-PPO stages, and their product must be divisible by the relevant PPO minibatch. Stage preflight records the resolved count, step count, sample product, and whether each value came from the stage override or policy-family/default runtime.
- Telemetry contract: Every production Isaac rollout records setup, rollout, and combined wall time; rollout-only and end-to-end environment-step throughput; one-second time-series samples of total device VRAM, GPU/core-memory utilization, power, temperature, and process RSS; and PyTorch allocator peak allocation/reservation. The immutable raw tensor artifact retains the time series. The on-policy dataset manifest retains per-source summaries and combined rollout/end-to-end throughput. Every single-family and joint PPO update separately records update wall time, consumed-environment-step rate, the same GPU/load time series, and process RSS in its training metrics. Missing GPU monitoring is explicit rather than fabricated, while telemetry failure does not alter policy behavior or safety semantics.
- Real-Isaac implementation smoke: The config-resolved, no-CLI-override C2 path completed `2048 x 16 = 32768` environment steps with finite state and unchanged physical/actuator readback. Setup was `61.986 s`, rollout was `2.619 s`, rollout-only throughput was `12513.931 env-step/s`, and cold end-to-end throughput was `507.205 env-step/s`. Across 21 samples, total device VRAM peaked at `6392 MiB`, GPU utilization averaged/peaked at `13.29/61%`, power averaged/peaked at `65.83/116.12 W`, process RSS peaked at `8675.59 MiB`, and the PyTorch allocator reserved `666 MiB` peak. The ignored raw diagnostic is `artifacts/p4_full/order9/runtime_diagnostics/c2_2048x16_telemetry_smoke.pt` with SHA-256 `c55b92b5b89ea5384f49d2c1fb4d1d55034e24cde0af48f03178b57161c78c4f`. This used an untrained diagnostic checkpoint and establishes runtime/integrity only, not learned task success. The large setup fraction is why later parallelism will be decided from full generation and update telemetry, not rollout-only throughput.

### Order 9 C2 Live TensorBoard Observation

- C2 production rollout and single-family PPO update emit live TensorBoard events by default under `artifacts/p4_full/order9/stages/<stage>/tensorboard/{train,validation}`. Train and validation are separate TensorBoard runs so equal environment-step indices cannot overwrite or visually merge their statistics. `--tensorboard-log-dir` changes the shared root and `--no-tensorboard` is restricted to explicit diagnostic runs; neither option changes a rollout, reward, optimizer, checkpoint, or promotion result.

### Order 9 C2 Initial PPO Calibration and Batched Recurrent Execution

- The first exact-replay C2 update established that the scaffold `pi_L` PPO learning rate `1e-4` was too large specifically at the C1-to-C2 boundary: C1 intentionally leaves the critic untrained, and the first shared-trunk optimizer step moved the following minibatch to approximate KL `0.25083` and clip fraction `0.76918` against target KL `0.02`. That child is rejected calibration evidence, not C2 lineage. The production `pi_L` PPO rate is therefore `1e-5`; this is a numerical calibration of the already approved PPO method, not a reward, policy-interface, authority, or curriculum change.
- Recurrent PPO and its pre-optimizer exact-behavior gate execute all active independent episode sequences as one batch at each recurrent timestep. Hidden state and previous action still start from each sequence's stored behavior state, remain isolated by episode, and advance in the same order; variable tails simply leave the active batch. This replaces thousands of batch-size-one graph/policy calls with mathematically equivalent GPU batches without changing the loss, optimizer-step count, GAE, clipping, or sequence boundaries.
- Runtime observations, reconstructed graph-only actor views, centroidal actor features, phase features, and privileged critic inputs are invariant for a stored rollout record during one PPO update. They are therefore constructed once, indexed by the unique record id, and reused by the exact gate and all four epochs. Only current-policy recurrent state, distribution evaluation, values, losses, and gradients are recomputed per minibatch. The cache is an execution optimization and cannot substitute stale policy outputs or hidden states.
- The CLI loads and hash-verifies each rollout dataset once and supplies the same immutable in-memory bundle to stage preflight and training. Both consumers revalidate its manifest path and SHA-256. This removes a redundant full compressed-JSON parse only; artifact validation and replay gates are unchanged.
- For a newly built production generation, the tensor dataset builder may hand the exact same validated `LowLevelControlRecord` tuple directly to the PPO stage executor in the same runner process.  Canonical gzip JSONL shards and their manifest are still fully written and SHA-256 verified first; the in-memory bundle is bound to that serialized manifest/index, and preflight plus training repeat the manifest-path/SHA, stage, behavior-checkpoint, and exact-replay checks.  A resumed generation without a resident bundle falls back to the canonical loader.  This removes one redundant gzip/JSON reconstruction pass and cannot omit or replace persisted evidence.
- Canonical on-policy JSONL remains gzip-compressed, but its storage-only encoding uses compression level `1`, zero gzip mtime, and no original filename.  The level and header contract are recorded in every manifest, every record is still schema-validated before writing, shard bytes are deterministically reproducible for an identical ordered record sequence, and all shard/manifest SHA-256 checks remain mandatory.  This changes neither the logical records nor replay/training/evaluation semantics; it only avoids expensive level-9 recompression of evidence that is already hash-bound.  On one `1,236,107,936`-byte C2 train shard, stream recompression measured `42.79 s` at level 9 versus `7.81 s` at level 1; the first complete production-generation timing under the new writer must still be recorded separately.
- Calibration evidence: the sequential reference rerun at `1e-5` completed all `4` epochs / `40` optimizer minibatches without KL early stop, with mean KL `0.009414`, mean clip fraction `0.13940`, and final minibatch KL `0.01109`. The committed full-generation batched rerun preserved those metrics to numerical precision, replayed all `32768` train records below the unchanged `2e-5` gate, and reduced update wall time from `1290.09 s` to `634.84 s`. The accepted update-0 checkpoint and detailed load evidence are recorded in `WORKLOG.md`.
- Every Isaac control step flushes the total reward and all eleven archived weighted terms: object-goal progress, object-pose accuracy, grasp maintenance, centroidal stability, energy, QP residual, slip, collision, actuator saturation, phase-success bonus, and terminal-failure penalty. Each is reported as current-step and rollout-running means, both globally and per active task phase. Phase occupancy, phase/task success, terminal and failure causes, QP feasibility, environment-step throughput, GPU/VRAM/power/temperature, process RSS, system load, and memory are logged alongside them.
- C2 recurrent PPO exposes an observation-only progress callback after every optimizer minibatch and flushes actor/value/total loss, entropy, approximate KL, clipped fraction, epoch/update index, and current load telemetry. The final update aggregate is logged separately. The callback cannot alter gradients, optimizer stepping, KL stopping, action replay, or checkpoint selection; TensorBoard is an observer of the existing immutable rollout/update lineage.
- The production writer is loaded lazily because TensorBoard belongs to the `isaaclab3` execution environment rather than the repository's minimal base Python. Unit tests use an injected writer and therefore do not fabricate event compatibility. A real `isaaclab3` event-file readback and a learned-checkpoint real-Isaac `2 x 2` integration smoke both verified immediately visible scalar events; the latter is a diagnostic override and is excluded from C2 training evidence.
- Production generation accounting retains the implemented paired-split contract: the resolved tensor runtime product `2048 x 16 = 32768` is the size of each train or validation shard, while one immutable on-policy generation contains both shards and records `65536` consumed environment interactions in its dataset/checkpoint lineage.  PPO gradients use only the `32768` train records; validation remains excluded from optimization.  The C2 `3,000,000` interaction budget therefore requires `ceil(3,000,000 / 65,536) = 46` fresh paired generations (updates `0--45`) and ends at `3,014,656` recorded interactions.  This clarifies the existing artifact/accounting behavior; it does not relabel validation records as optimizer samples.

## 2026-07-20

### Order 9 Learned Full-TaskSpec Curriculum Contract

- Scope/status: Order 9 is the staged learning phase for the existing deterministic Order 8 natural-contact substrate. It trains newly initialized `pi_L`, `pi_H`, and `pi_D` through C0--C10 and does not reinterpret the single canonical Order 8 run as learned-policy, distributional, or P4-full evidence.
- Policy ownership: `pi_H` denotes the learned policy only, never the integrated deterministic layer. It outputs the complete schema-level `ContactWrenchTrajectory`: contact assignment/mode/schedule, contact-frame wrench target and bounds, centroidal target, posture target, object target, priorities, guards, and timing. `pi_L` emits bounded phase-conditioned `PolicyCommand` intent only; QPID/QP, safety, actuator mapping, and Isaac own later commands. `pi_D` is a learned masked autoregressive graph-edit policy that produces `DesignOutput`; it is not reduced to deterministic-candidate ranking and never emits joint angles, runtime poses, or actuator commands.
- Hard-gate/credit decision: Every stochastic `pi_H` proposal is archived before deterministic `C_H`. `C_H` accepts or rejects the unmodified trajectory and never projects it. At most two learned attempts are made before a separately checked deterministic fallback. Each rejected learned proposal is an independent terminal GAE segment with configurable reward `-1.0`; only an accepted proposal receives its actual Isaac return. Fallback reward is never credited to `pi_H`. Exact action tensors, old log-probability/value, checkpoint hash, and feasibility result are replayed by PPO. Equivalent no-fallback-credit and independent rejected-proposal rules apply to `pi_D`; `pi_L` safety fallback ends the current learned GAE segment and creates no actor transition.
- Feasibility-head boundary: The existing P2.5 `pi_D` learned feasibility head remains an auxiliary approximation trained from deterministic feasibility labels. It does not replace the grammar mask, final `FeasibilityChecker`, or fallback. A future `pi_H` feasibility head may be added only as an auxiliary predictor after representative hard-check data exist; Order 9 starts with hard-checking every `pi_H` proposal and learning directly from checker-rejection transitions. A learned head may not become the final gate or silently filter behavior-policy actions.
- Curriculum: C0 collects hash-bound deterministic Order 8 teacher records from the separately approved bounded-diversity C0 profile. C1 trains fixed three-module, conservative-Order8-anchor `pi_L` by phase-balanced BC and evaluates the learned checkpoint over 100 episodes; C2 applies conservative fixed-morphology PPO; C3 expands `pi_L` to arbitrary grammar-valid 2--8-module topologies. C4 trains assignment-only `pi_H` BC, C5 the complete trajectory by BC, and C6 complete-trajectory PPO. C7 trains structured autoregressive `pi_D` BC, C8 masked `pi_D` PPO, C9 jointly fine-tunes the object-task policies, and C10 is held-out full-system evaluation. Assembly remains deterministic and separately evaluated; only a configured 10 percent C9/C10 subset executes assembly plus object task end to end.
- Phase/task reward decision: One policy may serve multiple phases only when task type, adapter identity, phase index/count, and progress are explicit actor inputs. Common safety terms remain active in every phase. Task-specific reward and terminal semantics are owned by a registered task adapter; `object_grasp_carry_v1` is the initial adapter rather than hard-coded global reward logic. Raw contact truth remains privileged critic/reward/safety/evaluation input and is excluded from actor observations.
- Object distribution: Conservative C2--C6 randomization remains close to canonical Order 8 (`1 kg`, `0.30 x 0.40 x 0.15 m`, selected-Dock friction `4.5`, compliant contact `7500/75`) and perturbs dimensions, mass, pose, friction, compliance, and estimated mass properties only within bounded ranges. C7 onward expands to disjoint box/sphere/cylinder training families and capsule/aspect-ratio held-out families. True mass, CoM, and inertia come from analytic/trimesh mass properties at sampled density; bounded estimator error is applied separately rather than inventing unrelated inertia tensors.
- Production `C_H` resolution: The approved backend is `hybrid_lightweight_qp_persistent_isaac_shadow`. For every unchanged `pi_H` proposal and every original knot, a small convex QP searches only for a contact-wrench witness inside the policy-proposed wrench boxes and enforces force/torque capability, friction, contact-patch moment, and active numeric object-wrench requirements. Its residual is a minimum physical-feasibility check, not a task planner, contact selector, trajectory generator, or substitute for simulation. An isolated persistent real-Isaac worker then restores a copied immutable main state, executes the same trajectory with the current hash-bound `pi_L`/QPID/QP stack for `2.0 s` at `0.02 s`, and reports controller-QP residual, realized contact-wrench residual, finite state, main-state digest preservation, and collision margin. Only explicit active assignment pairs or task-declared contacts are excluded from the collision margin. Both layers are mandatory and any missing worker, stale identity, state mutation, non-finite result, wrong knot count, or backend exception rejects fail-closed. Projection remains forbidden and the two-attempt/no-fallback-credit rules above are unchanged. `warmup_proxy` remains BC/teacher-only and cannot authorize C6/C9 online proposals.
- Runtime/artifact decision and measured gate: Fast training uses the implemented tensorized, topology-bucketed real-Isaac collector with phase-specific resets and no per-step JSON. Its hot path includes copied PhysX environments, real contact-sensor reduction, phase-conditioned `pi_L`, batched QPID and thrust QP, object-task state/reward/terminal logic, and a GPU tensor rollout buffer; schema/JSONL reconstruction occurs after simulation. The formal cold-start production-collector measurement used `64` control steps at `0.02 s` and measured `242.881/482.573/952.727` aggregate environment steps/s for `32/64/128` environments. Only `128` passes the conservative `500` gate and is selected. `artifacts/p4_full/order9/runtime_benchmark.json` is bound at SHA-256 `a70e6264aa10464177a25157c5ce5d303130c33e4bb5f94355369430658b3d9a`; preflight additionally verifies its raw tensor artifact hashes, collector version, true object mass/inertia/CoM readback, and PhysicalModel-derived actuator-limit readback. The gimbal runtime is explicitly bound to `0.76 Nm / 3 rad/s` and Dock to `4.1 Nm / 3 rad/s` for both the cached fixed USD and generated arbitrary-morphology USDs. Periodic evaluation must still retain the unchanged full-mesh Order 8 path and cannot replace its acceptance.
- Arbitrary-morphology/data runtime: The current split-safe morphology pool contains `80` grammar-valid 2--8-module structures, and all `80` have hash-bound URDF/USD assets in the production morphology manifest. Object/topology buckets bind TaskSpec, task-conditioned morphology graph, split, USD, contact material/compliance, and separately randomized estimated mass/inertia/CoM. The real collector validates exact morphology body/joint identity even when PhysX collapses fixed joints, validates spawned physical properties, and writes immutable tensor artifacts with complete policy/config/model/simulator provenance. Multiple train/validation shards are namespaced and merged into the existing `LowLevelControlRecord` dataset contract; held-out data are rejected from training. Every dataset, rollout generation, checkpoint, evaluation row, and stage promotion is byte-hash-bound, and one stochastic rollout generation may feed exactly one PPO update.

## 2026-07-13

### P4 Full Order 8 Object Natural-Contact Smoke Contract

- Order 8 formal completion (2026-07-19): The accepted representative real-Isaac run is `artifacts/p4_full/order8_natural_contact/order8_mu4p5_dt20ms_full_v406.json` (SHA-256 `d0f75cca2ae540c79971766ab722d4530dd4fb44842276256bac40aafdb8cc49`). It uses the source-configured selected authored-Dock friction `4.5`, spatially uniform `7500 N/m / 75 Ns/m` compliant contact, the free `1.0 kg` object, actual Dock collision meshes, no proxy, no post-grasp diagnostic torque bias, and the normal fail-closed supervisor. The ordered real-physics trace completed `reset -> approach -> contact_acquisition -> lift -> transport -> place -> release -> retreat -> settle -> complete` in `128.9 s` simulated time. Wrapper validation and independent artifact acceptance both passed with empty failure lists. Maximum selected-link grasp-reference displacement was `19.406/15.461 mm` against `30 mm`, force/penetration maxima were `10.046 N / 0.719 mm` against `30 N / 2 mm`, the object did not drop, release became contact-free, retreat clearance passed, settle passed, and unintended contact count was zero. The telemetry-only tangential slip-speed peak was `47.665 mm/s`; v12 intentionally does not gate on instantaneous slip speed. This closes Order 8's deterministic natural-contact/QPID substrate smoke only; it does not establish learned-policy robustness, full TaskSpec delivery, or P4-full acceptance.
- Formal runtime/actuator boundary (2026-07-19): The accepted run uses `simulation_dt=0.020 s` (`50 Hz`), inside the specified `pi_L` execution range, rather than the earlier debugging `0.005 s`; this reduces wall-clock cost without changing the physical task or acceptance gates. Simple closure retains the user-approved previous-target integration `q_target[k+1]=clip(q_target[k]+qdot_command*dt)` and physical joint limits. The authoritative AK40-10 audit is on applied actuator torque/current and measured speed: v406 reached `4.100 Nm / 7.300 A / 0.620 rad/s` and recorded zero envelope violations. The finite implicit-drive torque computed before simulator saturation reached `8.947 Nm`; it is retained as diagnostic telemetry, not incorrectly treated as delivered actuator torque. Release uses an independently bounded `0.020 rad/s` opening command and returns to the actually attained closure-start `q_open` snapshot, rather than an unreachable idealized geometry target. The per-phase timeout is `30 s`, while the full rollout budget is `150 s`, so the configured `0.2 m` transport at `0.01 m/s` can complete.
- Validation-contract correction (2026-07-19): Formal validation now describes the implemented simple-closure controller rather than stale historical mesh-IK/preload semantics. The one-shot mesh direction seed is explicitly not a completion gate; provisional first contact may separate and reacquire; `q_close` is established by simultaneous non-privileged proximity/load arrest followed by the stable-contact monitor; raw Isaac contact remains validation/safety evidence and is not a nominal planner/QPID input. Terminal evidence is sampled at a strictly later timestamp than the last physics observation, preventing a duplicate-timestamp false rejection when `COMPLETE` is reached without another physics step. A temporary ordinary-IK position-target-lead clamp was tested and rejected because it prevented the second load threshold from being reached; it is not part of the accepted controller.
- Source commit boundary (2026-07-19): Order 8 source is split into `a228b3a` for the natural-contact controller/schema/model foundation, `51baf27` for the real-Isaac runtime plus diagnostic/GUI integrations, and `c053ede` for independent fail-closed artifact acceptance. Generated USDs, state traces, and acceptance reports remain reproducible ignored artifacts rather than committed source.
- Selected-Dock friction setting (user-approved 2026-07-19): The representative Order 8 source config and `Order8NaturalContactConfig` default now use selected authored-Dock static/dynamic friction `mu=4.5` with the existing PhysX `max` combine mode; object/floor friction remains `0.6/0.8`. This provides approximately `24%` coefficient margin over the lowest retained-lift pass at `mu=3.625`. It changes no contact geometry, safety gate, actor input, QPID interface, object dynamics, or actuator envelope. The exact sweep used `7000/75` compliant contact plus an acceptance-ineligible `+/-0.5 Nm` post-grasp diagnostic joint bias, so selecting `4.5` was not by itself an Order 8 acceptance result and did not silently promote that bias. The subsequently completed v406 run above separately verifies the non-diagnostic full-phase path with `7500/75` and zero post-grasp bias.
- Authored-mesh friction boundary diagnostic (2026-07-19): Keeping the proxy-free v381 physical/control setup fixed and varying only the selected authored-Dock static/dynamic friction under `max` combine mode produced a sampled retained-lift boundary of `3.5625 < mu <= 3.625`. The unchanged lift gate is `100 mm` object-bottom clearance; retained lift additionally means no clearance-loss/drop through the `45 s` diagnostic ceiling. `mu=3.5` and `3.5625` briefly crossed the lift gate but dropped `0.14/0.54 s` later, whereas `mu=3.625` crossed at `39.04 s`, reached `119.429 mm`, and remained drop-free through `45 s`; `mu=3.75` also retained lift. At `mu=3.625`, grasp-reference displacement was only `0.442/0.644 mm`, force/penetration maxima were `9.960 N / 1.205 mm`, and the monitor failure-reason list was empty. This bracket is specific to the deterministic free-`1 kg`, actual-mesh, `7000/75`, `+/-0.5 Nm`, slip-speed-nongating diagnostic and is not a universal friction threshold or multi-seed robustness result. The diagnostic override is not promoted to production, and the run ended during transport, so downstream phases and Order 8/P4-full acceptance remain open.
- No-pad authored-mesh lift A/B (2026-07-19): Removing only the acceptance-ineligible cone proxy from v380 while retaining the same free `1.0 kg` object, friction `10`, compliant contact `7000/75`, `+/-0.5 Nm` diagnostic yaw-Dock bias, `30 mm` grasp-reference displacement gate, and smooth lift-bias removal still passed the physical `100 mm` lift gate. Real-Isaac v381 reports actual authored selected-Dock mesh collision with no proxy; grasp/LIFT began at `26.12 s`, support separation at `28.36 s`, bias removal at `28.88 s`, and the lift gate at `35.20 s`. Maximum clearance was `115.439 mm`, per-link displacement was only `0.468/1.309 mm`, maximum penetration was `1.109 mm`, and no safety/controller/actuator failure occurred. The `40 s` ceiling ended `66.19 mm` into transport, so place/release/retreat/settle remain unverified. This establishes that the proxy is unnecessary for the current lift demonstration, but does not promote the diagnostic friction/compliance/bias settings or complete Order 8/P4-full acceptance.
- Grasp-referenced slip and lift-bias ramp correction (user-approved and implemented 2026-07-19): Maintained-contact slip is no longer the time integral of the absolute tangential relative-speed magnitude. At verified grasp completion, each selected Dock link latches the force-weighted raw-contact centre in the measured object frame; its slip is the current 3D displacement norm from that immutable reference. Every selected link must remain within `30 mm`. Instantaneous tangential slip speed remains privileged validation telemetry but has no dwell, safe-hold, or acceptance gate. The persisted config/observation/step/result contracts are `v12/v4/v4/v3`, and acceptance binds the exact measurement method. The transient LIFT inertial bias is removed over the existing `0.5 s` using cubic smoothstep rather than a constant-slope ramp endpoint, preserving value and slope continuity. In the acceptance-ineligible cone-pad v380 diagnostic (`1 kg`, friction `10`, `7000/75`, `+/-0.5 Nm`), grasp completed at `24.32 s`, support separation at `26.58 s`, bias removal at `27.10 s`, and the required `100 mm` lift at `33.44 s`; maximum selected-link displacement was only `0.466/0.493 mm`, no safe hold occurred, and maximum clearance reached `120.343 mm` by the `40 s` ceiling. The run ended during transport, so it proves the requested diagnostic lift gate only and does not promote the proxy/material/bias or complete Order 8/P4-full acceptance. Historical velocity-integral path values remain diagnostic history, not the current slip gate.
- Dock-drive gain-tuning decision (2026-07-17): Use a constrained time-domain search rather than ultimate-sensitivity/Ziegler-Nichols tuning because the Isaac implicit drive is torque/speed saturated and the target task contains contact; sustained-oscillation tuning would identify the saturation/contact loop rather than a useful linear joint model. The reusable diagnostic is one fixed-base Holon module with gravity, floor, self-collision, contact sensors, and asset conversion disabled. It excites all four Dock joints concurrently with a `0.01 rad` step/return and a `1.2 Nm` child-link disturbance, enforces the unchanged `4.1 Nm / 7.3 A / 3 rad/s` limits, and scores worst-joint tracking, overshoot, settling, disturbance deflection/recovery, speed, torque, and saturation. Ninety-six coarse/fine candidates plus a deterministic repeat required about `24.75 s` wall time for `184.3 s` simulated candidate time. The contact-free numerical minimum `800/4.375` is explicitly only a bench candidate, never a deployment result. Representative free-object contact A/B under identical v314 fixture/config/safety conditions did not validate `800/4.375`, `650/4.375`, `300/5`, or `200/8`: they reached an unchanged instantaneous/cumulative slip gate at absolute simulation time `9.00/9.22/11.68/13.36 s`, respectively, while configured `200/5` reached the cumulative gate at `14.54 s`. Acquisition durations differ, so these absolute times are not treated as a phase-normalized ranking. Every actuator-envelope audit passed, but no candidate completed geometric lift-off with slip margin. Production therefore retains `Kp=200 Nm/rad`, `Kd=5 Nms/rad`, and `0.01 kg m^2` armature because no replacement passed representative contact validation—not because the contact-free numerical optimum is deployable. This gain result neither relaxes the contact gates nor completes Order 8.
- Simulator-debug execution decision (2026-07-15): Final Order 8 acceptance remains the unchanged complete three-module, all-Dock, free-object, authored-mesh environment.  Debugging must instead begin with a separately labelled temporary minimum diagnostic that removes unrelated phases/assets/sensors, reuses unchanged generated assets, and records requested/limited/applied actuator telemetry.  A fast diagnostic may use shorter ramps, direct pre-contact initialization, reduced scene scope, or privileged measurements solely to identify a fault; it can neither satisfy nor weaken Order 8 acceptance.  This minimum-diagnostic-first, full-environment-last workflow applies to later simulator work generally, with wall-clock completion time and measured simulation throughput treated as planning constraints.
- Simple-closure scope correction (2026-07-16): The active Order 8 diagnostic does not own a general mesh-following grasp planner or a learned-policy substitute. It uses a known graspable morphology/fixture, reaches the object-relative centroidal staging pose through ordinary QPID, and then commands all relevant Dock joints monotonically along one fixed velocity ratio with zero torque bias. Contact acquisition may allow a provisional first contact to separate and reacquire. Non-privileged terminal-joint load/current-equivalent plus selected-surface proximity establishes measured `q_close`; raw Isaac contact remains privileged validation/safety evidence. Learned `pi_L`, not this smoke, ultimately owns task-conditioned joint trajectories.
- Raised-support and yield-control correction (2026-07-17): For floor-clearance safety, the free box rests on a fixed `0.15 m`-high support platform and the robot work target is raised by the same amount. The support is an environment surface, not an object constraint; object pose writes and robot-object joints remain forbidden. All robot rigid bodies are monitored against both floor and support and any contact during manipulation causes safe hold. Contact yield must not reduce all-Dock position-drive stiffness and must not scale down the QPID P/I loops. Dock joints retain their nominal independent local drives. Admittance acts only on the centroidal target, projects the estimated external force onto the horizontal selected-contact reaction axis, preserves z-height plus roll/pitch authority, and applies no angular admittance. The contact load signal subtracts the estimated virtual-drive damping torque before thresholding so simulator damping is not treated as object contact.
- Raised-support diagnostic result (2026-07-17): A deliberately opened fixture (`3.0 s` backward along the fixed closure ratio; initial authored-mesh clearances `30.584/26.560 mm`) was run for the requested `30.000 s`. The run had zero robot-floor/support contact, QP/controller/AK40-10-envelope/joint-limit violation, no joint-drive yield, full PI scale `1.0`, and only `0.818 mm` maximum horizontal CoM-admittance offset. Both load/proximity triggers occurred at `5.69 s`, but the fixed joint-space line crossed the approximate simultaneous-contact region for only about one `0.01 s` step and did not satisfy the unchanged `0.10 s` non-privileged q_close dwell; final clearances were `23.561/89.873 mm`. Peak selected force remained `3.655 N`. Thus the safety/control correction is verified, but two-contact acquisition and every downstream phase remain unverified. No proxy collision pad was added. The next method-level choice is either an explicitly labelled finite-area diagnostic proxy pad or a different closure trajectory; timeout, friction, force, joint-softening, or height-loop tuning must not be presented as fixing this geometric-path limitation.
- Diagnostic pitch-hold / target-integration correction (2026-07-17): The user clarified that pitch lock was not a prior general requirement, but approved it as a simplification for the current diagnostic. In diagnostic-only runs, every pitch Dock joint is therefore commanded at its measured initial absolute position with zero velocity and zero torque bias; this command mask remains non-structural and acceptance-ineligible. The fixed closure yaw command now uses `q_target[k+1] = clip(q_target[k] + qdot_command * dt)` rather than rebasing from measured `q` each step. In the resulting v308 run, closure began at `2.04 s`, both selected authored meshes reached contact, the unchanged `0.10 s` non-privileged simultaneous-arrest dwell completed, measured `q_close` was latched, and the force ramp plus `0.25 s` grasp dwell progressed to `LIFT`. The prior multi-degree yaw reversal was removed: module 2 `yaw_dock_mech_joint2` moved from approximately `-0.54 deg` to `-3.71 deg` through closure. Pitch target velocity/torque bias remained zero; finite-drive/constraint deflection was at most `0.086 deg` through q_close and `0.357 deg` over the whole run. The diagnostic then entered safe hold at `12.83 s`, before its `30 s` ceiling, because module 2 selected-contact cumulative slip reached `10.156 mm` during lift (limit `10 mm`). This is a downstream lift/contact-maintenance issue, not a failure of the corrected closure gate, and no proxy collision pad or acceptance relaxation was introduced.
- Lift/load-transfer correction (approved and implemented 2026-07-17): `LIFT` begins its bounded upward centroidal trajectory immediately. Natural-contact payload feed-forward does not advance from elapsed time or raw Isaac contact; it follows a monotonic, slew-limited transferred-load fraction inferred from the aggregate centroidal external-wrench estimate. The LIFT-entry world-vertical external force is the zero-load baseline, transferred load is normalized by known `m_payload * g` and clipped to `[0, 1]`, measured object rise is retained as a lower-bound audit, and verified support separation forces full payload accounting. This keeps raw per-patch contact privileged-only and avoids adding full payload inertia while the support still carries the object.
- Lift/load-transfer diagnostic result and open boundary (2026-07-17): Continuous real-Isaac v314/v315 runs verified that the observer remained valid and feed-forward never led inferred load, but both stopped at the unchanged `10 mm` cumulative-slip gate before geometric lift-off. Extending the full-force dwell from `0.25 s` to `2.0 s` did not settle the grasp: the stationary supported object still saw approximately `6.5-7.4 mm/s` selected-link relative motion immediately before lift. Therefore the remaining problem is the physical grasp-hold equilibrium under nominal Dock impedance, bounded torque bias, and QPID pose hold, not payload feed-forward timing. A longer dwell or stricter gate alone cannot establish equilibrium, and the slip threshold/raw-contact control boundary must not be relaxed.
- Peak-torque diagnostic result (approved and executed 2026-07-17): The acceptance-ineligible v316 diagnostic temporarily permitted the configured AK40-10 `4.1 Nm/7.3 A` peak envelope after q_close. It raised the two selected normal loads from approximately `2.75 N` to `5.72/5.74 N` while remaining inside the measured torque/current/speed envelope, but base speed increased to about `8.2 mm/s` and selected-link/object relative motion reached approximately `6.6/13.1 mm/s`. The non-privileged pre-lift motion gate correctly withheld LIFT throughout the high-force interval. Thus insufficient normal-force torque is rejected as the primary cause: increasing torque strengthens the articulated shear motion and cannot be the production correction. The next change must address grasp-hold equilibrium without raw contact input, threshold relaxation, or merging joint dynamics into QPID. No such grasp-hold rule change is yet approved or implemented.
- Simulation-damping diagnostic result (2026-07-17): Before changing the hold method, v317 exercised the previously authorized simulation-gain freedom by increasing Dock damping from `5` to `20 Nms/rad` while retaining `200 Nm/rad` stiffness and the continuous `1.3 Nm` torque-bias limit. It reduced the non-failing side's cumulative slip path from `8.609` to `5.424 mm`, but the failing side remained effectively unchanged (`10.011 -> 10.037 mm`) and again stopped before lift-off. Its relative speed was still approximately `8.1 mm/s` at LIFT entry. Torque/current/speed audits passed, but acquisition reached the diagnostic `0.1 rad/s` Dock speed ceiling. This rejects simple drive-damping insufficiency and further gain sweeping; the `20 Nms/rad` value is diagnostic-only and is not adopted as a default.
- Load-limited positional-preload decision (approved and implemented 2026-07-17): The post-`q_close` Jacobian-transpose/contact-wrench torque-bias ramp is superseded. After measured `q_close` settles, the same fixed closure ratio advances slowly from its previous absolute position target. Each selected side freezes independently after the maximum damping-compensated load over its moving influential Dock joints remains above threshold for `contact_stall_dwell_s`; shared joints freeze with the first completed owner. Both sides must freeze and the full selected-link/object relative-speed dwell must pass before `LIFT`. The offset-torque command is exactly zero throughout preload and carriage. This uses only hardware-available joint position, velocity, and load/current-equivalent signals, preserves the independent local-joint/QPID split, and does not consume raw Isaac contact in nominal control.
- Load-limited positional-preload diagnostic result and open boundary (2026-07-17): Real-Isaac v318-v320 verified correct independent freeze, zero offset torque, relative-motion settling, synchronized lift entry, and observed-load-following payload feed-forward. Nevertheless, the authored-mesh free-object grasp did not lift off. At retained `1.2 Nm` threshold, selected normal load was about `2.63 N/side` and only `57.2%` of payload weight transferred before the unchanged `10 mm` slip gate. A diagnostic at the AK40-10 `1.3 Nm` continuous boundary increased normal load only to about `2.86 N/side` and transfer to `60.1%`; it also showed dwell/servo overshoot, so `1.3 Nm` is not retained as the threshold. Further torque increase is forbidden. Continuing requires an explicit choice between a thin finite-area high-friction selected-surface proxy, a justified material/payload assumption change, or different grasp mechanics/control. None is approved or implemented yet; no slip relaxation, raw-contact feedback, kinematic attach, or Order 8 acceptance claim is made.
- Finite-area proxy diagnostic decision (approved and implemented 2026-07-17): A temporary, acceptance-ineligible selected-surface proxy may be used to distinguish authored-mesh contact-capacity limits from lift-control limits. Each selected Dock rigid body receives one connect-frame-aligned `30 x 30 x 2 mm` collision pad fitted from sampled outer-face mesh geometry. The pad has no independent rigid body, uses an explicitly reported high-friction material, retains the authored Dock mesh, and protrudes far enough that an object respecting the unchanged `2 mm` penetration bound cannot contact the retained mesh simultaneously. The report must identify proxy paths/specs, collision/material coverage, retained-mesh status, zero independent rigid bodies, and `selected_surface_actual_dock_mesh=false`; such a run cannot be relabelled as normal Order 8 acceptance.
- Finite-area proxy diagnostic result and load-transfer boundary (2026-07-17): Real-Isaac v321 audited the authored USD representation, while v322-v325 calibrated friction from `mu=3.0` through the deliberately extreme fault-isolation value `mu=10.0`. At `mu=10.0`, both selected proxy contacts remained valid, QPID allocation was feasible and non-saturated, Dock actuator limits passed, peak penetration was only `0.086 mm`, and no robot/environment collision occurred. Yet the object remained on its support; inferred/feed-forward payload share reached only about `0.795/0.792`, after which the pad/object tangential paths reached `10.077/7.789 mm` and the unchanged safety gate stopped the run. This rejects contact area or friction as the remaining primary cause. The observed-load-only payload feed-forward rule is load-support circular: before lift-off the support carries the fraction the observer has not yet seen, but lift-off requires compensation of the full known payload. The next method decision is whether to add a bounded commanded lift-progress floor so upward target motion and known-payload compensation rise together to full support, retaining aggregate external-wrench estimation as a lower-bound/audit. No such scheduling change or persisted proxy/config promotion is approved or implemented by this entry.
- Authored-mesh compliant-contact replacement (approved and implemented 2026-07-18): The finite-area proxy remains a legacy fault-isolation/placement diagnostic only and is not a P4-full contact representation. Normal Order 8 instead binds one spatially uniform PhysX compliant-contact material to each complete selected authored Dock collision-mesh rigid body. This is equivalent to a thin compliant coating at every possible mesh contact location; it is not a dynamically moved pad, does not require knowing the task's final contact point, does not deform or constrain the object, and preserves the actual Dock mesh plus the free rigid object. The persisted `order8_natural_contact_config_v11` values are `7500 N/m` stiffness and `75 Ns/m` damping. Acceptance fails closed unless both PhysX attributes read back exactly, every selected collision descendant inherits the binding, proxy use is false, and `selected_surface_actual_dock_mesh` is true. The unchanged `2 mm` penetration ceiling remains a hard safety/acceptance gate.
- Full-side proxy-pad placement preview (user-approved for inspection, not yet for runtime, 2026-07-18): Reanalysis of the legacy `30 x 30 x 2 mm` connect-frame pad showed that its normal was about `54--58 deg` from the relevant object-face normal and that contact occurred at pad corners rather than across the intended face. The replacement preview therefore defines one fixed flat plate on the grasp-facing side of each selected yaw Dock link. A pure URDF/STL builder projects the complete authored collision mesh onto fixed link-local horizontal/vertical axes, adds a `3 mm` tangential margin, and places a `2 mm` plate with its inner face `1 mm` beyond the mesh support plane. This produces approximately `135.899 x 106.133 x 2 mm` and `139.049 x 106.111 x 2 mm` plates. The normals are fixed link geometry, not object-conditioned runtime values. The preview runs one Holon module, authors the plates as translucent orange collision children with no independent rigid body, performs no post-reset physics step, and keeps only rendering active. Its config requires `acceptance_eligible=false` and `contact_runtime_enabled=false`; normal Order 8 continues to use actual compliant meshes until the user visually approves placement and separately authorizes runtime integration. A preview image or stage spawn is not natural-contact or acceptance evidence.
- Collision-following micro-pad preview correction (awaiting visual approval, 2026-07-18; supersedes the preceding one-plane geometry): User inspection established that a single `135--139 x 106 x 2 mm` support-plane plate is too large, is not aligned with curved/local collision faces, and cannot represent the complete side. Preview contract v2 therefore partitions each actual yaw-Dock collision STL into `12` axial bands and `24` circumferential segments. Every occupied outer lateral-surface cell receives an independently oriented thin box fitted to the local authored triangles; physically empty/open cells are left empty. The canonical mesh yields `237` pads per link (`474` total), each `6--16 mm` tangentially and `0.8 mm` thick, with its inner face `0.2 mm` outside the fitted surface. Local fit gap is `1.316 mm` at the 95th percentile and `2.277 mm` maximum, and normals cover both signs of local Y/Z. Geometry remains link-local and object-independent. This is strictly a render-only placement preview with zero post-reset physics steps, `acceptance_eligible=false`, and `contact_runtime_enabled=false`; it changes neither production contact nor Order 8 acceptance until a separate user approval and runtime-integration decision.
- Cone-only merged micro-pad correction (awaiting visual approval, 2026-07-18; supersedes v2 placement): Pads are unnecessary on the cylindrical/mounting structure outside the cone. Preview v3 selects only authored collision triangles in link-local `x=0.068--0.1158 m` whose outward-normal axial component is `0.30--0.85`, excluding the non-conical right-side geometry. Adjacent locally similar cells are merged by refitting the cone with a `4 x 20` axial/circumferential partition instead of the whole-side `12 x 24` partition. The canonical result is `75` pads per link (`150` total, about `68%` fewer than v2), approximately `8.15--18.67 mm` tangentially and still `0.8 mm` thick with `0.2 mm` clearance. Local fit gap improves to `0.951 mm` at the 95th percentile and `1.406 mm` maximum. This is still an object-independent, link-local, render-only placement preview; normal runtime, production contact, and acceptance remain unchanged and proxy-disabled until explicit visual and integration approval.
- Cone-only micro-pad runtime diagnostic and lift result (visually approved and implemented 2026-07-18; supersedes the preceding preview-only boundary): The approved `75/link` cone geometry may be enabled only by an explicit acceptance-ineligible diagnostic switch. The boxes are fixed children of the existing selected yaw-Dock rigid bodies, use no object-conditioned placement, create no independent body, and replace collision only for the corresponding selected authored-mesh descendants while retaining their visuals. In this mode q_close clearance is computed from sampled finite box surfaces; raw contact remains privileged diagnostic/safety evidence and does not enter actor or QPID commands. With the free `1.0 kg` object, diagnostic friction `10`, `7000/75` compliant contact, `Kp/Kd=200/5`, `0.005 rad/s` closure, `0.5 Nm` post-grasp bias, disabled instantaneous-slip stop, and a retained `30 mm` cumulative-slip diagnostic bound, v375 acquired grasp and physically separated the object from support at `26.58 s`. Object COM rose `11.663 mm` and the tilted OBB reached `3.009 mm` bottom clearance; force, penetration, drop, and unintended-contact checks stayed valid. The run then stopped at `18.194/30.233 mm` cumulative slip and did not satisfy the unchanged `100 mm` lift requirement. A slower `2.5 s` payload transfer did not help. This proves limited mechanical lift capability for the diagnostic proxy, not robust hold, normal Order 8 acceptance, or P4-full suitability; the proxy and diagnostic material/control/safety overrides remain non-production.
- Cone-pad lift low-load GUI replay (implemented 2026-07-18): The exact v375 physical trajectory is cached as a hash-validated diagnostic state trace and may be inspected in Kit without rerunning contact dynamics. To prevent the prior static-joint symptom, every displayed frame advances one gravity-free/contact-minimized synchronization step, reapplies the exact recorded module roots, joint states, and object state, and verifies the independent PhysX DOF readback. Cone proxy visuals may be authored during replay only in this synchronization mode; object/environment collision, gravity, graph constraints, and robot self-collision are disabled, so the pads cannot create a second physical result. Balanced rendering plus a replay-only Dome Light avoids the previously observed black normal-mesh viewport, and routine Kit output is filtered. The cached trace covers `26.74 s`, `670` frames, `5.43 deg` maximum Dock motion, and `11.66 mm` object COM rise; real Kit replay reproduced `5.43 deg` with zero joint/object/root write error and authored all `150` pads. This replay is visual-only and cannot substitute for v375 raw physical evidence or any Order 8 acceptance run.
- Cone-pad `+/-0.5 Nm` full-duration/no-safe-hold diagnostic (requested and executed 2026-07-18): Repeating the v378 `50.000 s` diagnostic with only the four-yaw bias restored to `+/-0.5 Nm` produced a materially different physical outcome. v379 acquired grasp, confirmed support separation at `26.58 s`, exceeded the required `100 mm` lift clearance, and transported the object `200.20 mm`; maximum object-COM rise was `141.82 mm`, and `PLACE` began at `47.08 s`. This is still not acceptance: provisional/maintained slip speeds peaked at `30.51/21.18 mm/s`, cumulative paths reached `259.21/296.85 mm`, and `1163` supervisor safe-hold requests were recorded but diagnostically suppressed. The monitor's later required-contact/drop latches arise because excessive cumulative slip invalidates the required-contact safety gate even while both raw selected-link contacts remain present. Release, retreat, and settle were not reached inside the 50 s budget. Applied bias was `0.5 Nm`; combined torque/current/speed peaked at `2.524 Nm / 4.494 A / 0.164 rad/s`. Thus `+/-0.5 Nm` proves the cone-pad fixture can physically lift and transport `1 kg` under continued unsafe slip, but neither the proxy nor the relaxed/suppressed safety path is promoted to production or P4-full acceptance. The dedicated cached GUI remains visual-only.
- Cone-pad `+/-1 Nm` full-duration/no-safe-hold diagnostic (requested and executed 2026-07-18): A new explicit default-off diagnostic mode may suppress every safe-hold transition while continuing to measure and report the same evidence and running the complete requested physics-step budget. It records external supervisor requests, extends planner phase timeouts past that budget, and fails if an unhandled path nevertheless reaches `SAFE_HOLD`; it cannot be enabled outside diagnostic-only mode or used for acceptance. With all v375 conditions fixed except four-yaw bias `+/-0.5 -> +/-1.0 Nm`, v378 completed `50.000 s` without a safe-hold transition. q_close/preload completed, but maximum provisional contact-relative speed reached `163.16 mm/s`; final per-anchor speeds remained `11.51/112.70 mm/s`, so the unchanged non-privileged `10 mm/s` pre-LIFT settle gate never completed. The object did not lift (`0.0037 mm` maximum COM rise). Applied bias was exactly `1.0 Nm`, while combined position-drive/effort torque reached `3.278 Nm`, estimated current `5.836 A`, and speed `0.425 rad/s`: inside the hard AK40-10 audit, but above its `1.3 Nm` continuous torque rating. Therefore `+/-1 Nm` is rejected as a contact-retention improvement and is not promoted. The cached `1251`-frame low-load GUI replay reproduces `5.55 deg` Dock motion with zero PhysX write error and remains visual-only.
- Authored-mesh compliant-contact diagnostic result and remaining boundary (2026-07-18): A proxy-free saved-v314 run with the previously requested representative diagnostic settings (`Kp/Kd=200/8`, selected friction `10`, and temporary `60 mm/s / 30 mm` slip bounds) acquired two actual mesh contacts, latched grasp, and reached physical support separation at `15.02 s`. Peak per-contact force was `9.661 N`, peak penetration `1.157 mm`, maintained-contact slip through `16 s` `14.234 mm/s`, and all raw-contact/unintended-contact/object-drop/AK40-10 audits were clean. Thus the selected spring supplies the intended finite compliant overlap without a task-specific pad. A production-setting A/B (`friction=2`, `200/5`, unchanged `20 mm/s / 10 mm`) also acquired the actual-mesh grasp at `1.151 mm` peak penetration, but the known lift-entry transient reached `41.627 mm/s` and correctly safe-held before support separation. This replacement resolves the proxy's geometric/generalization defect; it does not complete Order 8 or authorize weaker slip limits, extreme friction, a kinematic object attach, raw-contact feedback, or acceptance from the short fixture. Resume at production lift contact maintenance using actual compliant meshes.
- Post-grasp constant Dock torque-bias diagnostic (requested and executed 2026-07-18): The fast diagnostic v23 can explicitly add one equal-magnitude offset torque to each of the four grasp-contributing yaw Dock joints after load-limited positional preload; signs follow the fixed closure direction, while pitch/unrelated joints remain zero. This is an acceptance-ineligible A/B only. Normal runtime remains zero-bias, and the existing AK40-10 continuous/hard torque, current, and speed audits remain authoritative. With proxy disabled and all production contact/gain/safety settings unchanged, `0.5 Nm` acquired grasp and entered `LIFT` but safe-held at `25.306 mm/s` maintained-contact slip before support separation. `1.0 Nm` sustained relative motion above the `10 mm/s` pre-LIFT settle threshold and never completed the grasp dwell within `16 s`. Both stayed below `1.151 mm` penetration and passed the actuator audit. Therefore a larger constant closing bias is rejected as the production correction; neither value is persisted or enabled by default.
- Slip-speed safe-hold exemption diagnostic (requested and executed 2026-07-18): Fast diagnostic v24 adds an explicit default-off option that continues measuring instantaneous selected-contact slip but removes only its maintained-contact safe-hold predicate. It does not change the production config, the `10 mm/s` pre-LIFT settle threshold, cumulative slip, penetration, force, contact-break/drop, controller, collision, or actuator protections. With `0.5 Nm` closure-direction bias, proxy disabled, production contact/gain settings, and only the cumulative path limit raised diagnostically to `30 mm`, the grasp entered `LIFT` but did not separate from support. At `14.58 s`, cumulative paths reached `30.388/21.566 mm` and the retained cumulative gate safe-held; instantaneous slip had reached `154.750 mm/s`. Peak force/penetration remained `10.030 N / 1.151 mm`, and the actuator audit passed. Thus the speed violation is not merely a harmless isolated spike: removing it exposes rapidly accumulating contact-relative motion. The exemption and `30 mm` bound are not production-approved.
- Object-rotation causal diagnostic (requested and executed 2026-07-18): Fast diagnostic v25 can, only by explicit acceptance-ineligible request, snapshot object orientation at `LIFT` entry and project only its quaternion/angular velocity back at each `LIFT/TRANSPORT/PLACE` step boundary; translation and linear velocity remain fully dynamic and grasp acquisition is unchanged. In the matched v349/v350 A/B, this reduced peak slip from `154.750` to `93.860 mm/s`, but the `30 mm` cumulative failure shifted to the opposite selected link (`21.805/30.663 mm`), object rise fell from `1.512` to `0.365 mm`, and measured payload-load transfer fell from `9.133%` to `2.696%`; no support separation occurred. The projection was active for `89` steps and bounded the one-step orientation deviation to `0.001649 rad`, so the negative result is diagnostic rather than a missing intervention. Object rotation is a contributing transient but not the primary lift blocker: selected-anchor/contact-point translation caused by articulated Dock motion remains when rotation is removed. The diagnostic pose writes, object-orientation constraint, slip-speed exemption, and `30 mm` path limit are forbidden for normal Order 8 acceptance and are not production solutions.
- Multi-anchor joint-target correction diagnostic (approved and executed 2026-07-18): Fast diagnostic v26 can replace the post-q_close fixed-position overwrite during `LIFT/TRANSPORT/PLACE` with the existing simultaneous full-Dock DLS output, integrated into bounded absolute position targets. It does not modify QPID, use raw contact/contact wrench, or add a normal actor input. A `10/s` task gain was required to put the correction command on the same order as measured load-driven joint motion while retaining the unchanged AK40-10 envelope. The base-relative v352 variant reduced geometric anchor error to `0.396 mm` but did not improve lift. The corrected v353 target transported the measured q_close anchor pair along the commanded centroidal manipulation path; it reached the `0.1 rad/s` diagnostic joint-command limit and became `unreachable_residual` at `7.026 mm / 2.228 deg` maximum anchor error. It safe-held on `20.990/30.760 mm` cumulative slip with only `0.033 mm` object rise. Terminal normal forces `2.040/2.175 N` supplied about `8.428 N` total upward friction reaction, below the `9.81 N` payload weight. Therefore exact two-anchor commanded-path tracking is not a production solution: the bounded joint subspace cannot simultaneously absorb common-mode base lag, preserve both complete 6D poses, and retain normal preload. Any follow-on must be an explicit hierarchy/weighting decision—such as QPID common-mode motion plus differential joint correction and hardware-observable joint-load preload maintenance—and is not approved by this diagnostic alone.
- Maximum-stable friction diagnostic with joint correction off (requested and executed 2026-07-18): The v354 run explicitly disabled the v26 multi-anchor correction and changed only the selected authored-Dock material to the previously stable fault-isolation value `static/dynamic mu=10.0` with `max` combine mode. This is a diagnostic upper setting, not a universal simulator maximum or production material claim. The actual compliant Dock meshes, free object, `200/5` drives, `0.5 Nm` diagnostic closing bias, and diagnostic `30 mm` cumulative-slip bound were retained. Material readback passed, the simulation remained finite, and support separation was confirmed at `15.30 s`; q_close-to-terminal object rise improved from v349's `1.512 mm` to `8.470 mm`, while peak maintained-contact slip fell from `154.750` to `16.951 mm/s`. The run nevertheless safe-held at `16.44 s` on cumulative paths `30.001/21.692 mm`, before the `0.1 m` lift gate or downstream phases. Production remains `object mu=0.6`, selected Dock `mu=2.0`, joint correction off, and otherwise unchanged. Extreme friction is neither promoted nor accepted as the Order 8 solution.
- Staged Dock-compliance diagnostic at `mu=10` (requested and executed 2026-07-18): With joint correction off and all other v354 diagnostic conditions fixed, authored-Dock compliant-contact stiffness was swept from the `7500 N/m` reference to `7000`, `6500`, and `6000 N/m`, while damping stayed `75 Ns/m`. The `7000` run was the only improvement: support separation moved from `15.30` to `14.96 s`, object rise increased `8.470 -> 9.195 mm`, peak maintained-contact slip decreased `16.951 -> 16.287 mm/s`, and penetration remained `1.200 mm`. At `6500`, contact-relative motion failed the pre-LIFT settle gate for the full `30 s`; at `6000`, peak maintained-contact slip rose to `79.353 mm/s`, object rise fell to `1.148 mm`, and no support separation occurred. All runs remained finite and below the `2 mm` penetration ceiling. Compliance is therefore nonlinear rather than monotonically beneficial; `7000/75` is the sole production candidate from this sweep, while `6500/75` and `6000/75` are rejected. No value is promoted without an explicit production-setting decision and a rerun with production safety gates.
- One-shot loaded-state rebase diagnostic (approved, implemented, and rejected 2026-07-18): Fast diagnostic v27 can, only by an explicit acceptance-ineligible request, use the existing first `1 mm` support-separation event to snapshot the measured base-root pose and every measured Dock position once, hold those absolute targets with zero velocity, retain the requested closure torque bias and payload gravity feed-forward, reset QPID integrators, and wait at least `0.5 s` plus `0.1 s` of all-anchor relative speed below the existing `10 mm/s` pre-LIFT threshold before resuming. The settle explicitly removes only the transient lift-acceleration bias and never consumes raw contact as control input or writes/forces/constrains the object. Corrected diagnostic v359 triggered at `14.96 s`, with cumulative paths already `20.393/14.181 mm`; the base root was `40.583 mm` above q_close and still moving upward at `48.706 mm/s`. The pose/q snapshot cannot remove that kinetic state: object clearance returned from `1.035 mm` to zero, no relative-speed dwell accrued, and the retained `30 mm` cumulative gate stopped the run at `15.76 s` (`30.044/24.450 mm`). The mode remains default-off and is not a production or acceptance solution. A velocity-continuous micro-lift/deceleration stage would be a new method-level decision; continuous joint correction, slip-path reset, object intervention, and safety relaxation remain forbidden absent separate approval.
- Object-mass sweep diagnostic (requested and executed 2026-07-18): Fast diagnostic v28 adds a recorded, finite-positive, diagnostic-only object-mass override; the production source remains `1.0 kg`. With every v359 physical/control condition fixed, v360-v368 lowered mass from `0.9` to `0.1 kg` in `0.1 kg` steps. The response was non-monotonic: `0.7/0.6/0.5/0.2 kg` never completed contact acquisition; `1.0/0.9/0.8 kg` lifted only about `1.0-1.2 mm` before the retained `30 mm` cumulative-slip stop; `0.4/0.3 kg` completed loaded-state settle and reached `70.334/70.998 mm`; and `0.1 kg` reached `35.849 mm`. Every lifting run ultimately exhausted the same cumulative-slip budget, and none passed the unchanged `100 mm` lift gate. Therefore payload reduction alone is rejected as an Order 8 solution, no lighter mass is promoted, and all production mass/contact/controller/safety defaults remain unchanged.
- Synchronized lift-progress decision (approved and implemented 2026-07-17): The persisted config/report contract advances to `order8_natural_contact_config_v9`. During `LIFT`, `clip(lift_elapsed / payload_load_transfer_s, 0, 1)` is shared by the maintained-contact motion-entry ramp and the minimum known-payload feed-forward target. The applied payload scale is slew-limited and monotonic over the maximum of commanded progress, aggregate centroidal observed-load share, measured object-rise share, and verified lift-off. Reports require the versioned method/driver, commanded-progress peak `1.0`, and no feed-forward lag behind commanded progress; lead over the observer is now expected and bounded because the observer is a lower-bound/audit rather than the sole driver. Raw contact, object constraints, QPID contact dynamics, and all safety limits remain unchanged.
- Synchronized lift-progress diagnostic result and next boundary (2026-07-17): v326-v329 used the same saved near-contact fixture and diagnostic `mu=10` proxy while varying only the shared linear progress duration. The approved rule removed the former observer deadlock: payload accounting reached `1.0`, the object developed approximately `22 mm/s` upward velocity, and QP/actuator envelopes remained valid. It still exposed a narrow incompatible safety boundary. At `1.0 s`, cumulative slip reached `10.097 mm` shortly before lift-off; at `0.5 s`, instantaneous slip reached `22.76 mm/s`; at `0.75 s`, instantaneous slip was safe (`19.51 mm/s`) but cumulative slip reached `10.080 mm`; at `0.72 s`, instantaneous slip reached `20.73 mm/s`. A fixture-specific duration search is therefore rejected as a robust solution. The next method decision is a bounded LIFT-only upward CoM wrench/acceleration bias, ramped with the same progress and removed after lift-off/steady lift. This would use the already approved CoM-wrench intent path and would not make QPID joint- or contact-aware. No such bias is implemented by this entry.
- LIFT-only payload inertial-intent decision (approved and implemented 2026-07-17): The persisted contract advances to `order8_natural_contact_config_v10` with bounded payload acceleration and post-lift-off removal duration. During `LIFT`, `m_payload * a_lift * shared_progress` is generated in world `+Z`, rotated through the measured centroidal pose, and supplied through the existing `PolicyCommand.residual_wrench_body` / QPID CoM-wrench path. The first existing 1 mm geometric lift-off event latches the scale and starts a 0.5 s removal ramp; the bias is exactly zero outside `LIFT`. This is not a contact/internal wrench, does not directly command rotors, and does not alter local Dock position control, raw-contact privilege, or safety/actuator limits. Versioned evidence binds scheduled and PolicyCommand application counts, non-LIFT zero activity, frame-invariant force norm, lift-off/removal state, and terminal zero wrench.
- LIFT-only inertial-intent diagnostic result and velocity-governor boundary (2026-07-17): The v330 saved-fixture proxy run verified exact 1.0 N world/body application for 56 LIFT steps with zero non-LIFT leakage, but the free object rose `3.041 mm` at `39.04 mm/s` while its tilted OBB clearance was still `0.679 mm`. The selected Dock surfaces lagged predominantly along object `-Z`; tangential relative speed reached `34.07 mm/s` and correctly triggered the unchanged `20 mm/s` safe hold. Because v326 with zero added bias instead failed the `10 mm` cumulative gate, interpolating a fixture-specific constant is rejected. The proposed next rule is to govern increases in feed-forward and inertial intent with non-privileged object/Dock relative kinematics, holding load progress and removing extra acceleration while the object outruns the surfaces, then resuming with hysteresis. That changes the approved time-only progress driver and is not implemented without explicit approval; raw PhysX slip/contact, acceptance relaxation, higher maintained-contact speed, and kinematic attachment remain forbidden.

- Separated-transition correction and payload-coupling boundary (2026-07-17; supersedes the v330 velocity-governor diagnosis): Diagnostic v331 withheld LIFT until measured-grasp rebase, centroidal/joint yield removal, admittance shutdown, base-speed settling, and a fresh two-contact dwell were complete. Exact selected-Dock contact-point velocities remained upward throughout restore, then both reversed downward only after ordinary payload coupling began; the separately delayed `1 N` LIFT bias never activated. Diagnostic A/B v332 retained the same upward LIFT pose trajectory but disabled payload coupling, and both Dock contact points remained upward without a hard safety failure through the time limit. Thus QPID restoration, pose-trajectory sign, and the extra bias are excluded; the causal boundary is payload coupling. At v331 the pre-lift payload term contained about `9.77 N` effective vertical force and `-4.13 Nm` pitch moment from a roughly `0.423 m` COM offset, even though the support had not released the object. No production method is changed by this diagnostic. The recommended pending change is to retain the validated restore gate and split payload scheduling: a bounded commanded-progress floor for vertical payload force may break the support deadlock, while COM-offset moment and inertial terms follow non-privileged measured load transfer and become full only after geometric lift-off. Raw contact remains diagnostic/safety-only, the object remains free, and all existing safety/actuator limits remain unchanged. This method requires explicit approval before implementation.
- Payload-moment frame correction and articulated-motion boundary (2026-07-17): v333 added centroidal pose/twist and target telemetry to the same minimum fixture. The payload lies on centroidal body `+X`, so its approximately `-4.13 Nm` body-`Y` compensation moment is correctly nose-up and raises the object-side Dock under rigid-body kinematics; it does not directly command the Dock downward. The measured centroidal body also moved upward and nose-up. At `14.20 s`, rigid-point propagation predicted approximately `+12.38/+13.79 mm/s` vertical Dock motion, but the actual Dock contact points moved at `-14.77/-16.15 mm/s`, leaving approximately `-27.15/-29.94 mm/s` of relative articulated motion. Therefore full payload coupling remains the trigger, but force versus moment/inertia has not been causally isolated, and the prior direct attribution to the moment direction is withdrawn. Before any production scheduling change, use diagnostic-only force/moment component A/B on the same fixture; retain the restore gate, free object, raw-contact privilege boundary, and all safety/actuator limits.
- Payload-component A/B result and local-joint boundary (2026-07-17): v334 retained only the payload translational force. With exactly zero diagnostic payload torque, one selected Dock still moved downward at approximately `-19.29 mm/s`, the structure pitched nose-down, and the unchanged instantaneous-slip gate stopped the run; removing the COM-offset moment therefore worsened rather than cured the response. v335 retained translational force plus the correctly signed COM-offset moment while zeroing payload rotational inertia, and reproduced full v333 almost exactly (`-15.20/-16.87 mm/s` Dock minima and approximately `16.67 mm/s` peak slip). Rotational-inertia coupling is consequently immaterial, and the nose-up offset moment is physically necessary and beneficial. The remaining causal boundary is payload translational load transfer exciting relative articulation of the upstream Dock chain: centroidal QPID motion is correct, but fixed local position targets do not preserve selected-anchor pose under the changing load. No production coupling/schedule, raw-contact boundary, safety limit, object freedom, or acceptance contract changes here. Before adding a deterministic anchor-hold substitute for the joint-position correction ultimately owned by learned `pi_L`, perform at most one diagnostic-only Dock drive-gain check under unchanged AK40-10 torque/current/speed limits; any compensator is a separate method-level decision and must leave QPID joint-unaware.

- Scope and no-mislabeling decision: Order 8 is a deterministic/controlled real-Isaac **natural-contact substrate smoke**, separate from Order 9 learned full-TaskSpec delivery. Its baseline scene is one free `1.0 kg`, `0.30 x 0.40 x 0.15 m` box with object friction `0.6`, resting on the raised fixed support whose contact material follows the intended floor-like support semantics; the surrounding floor friction remains approximately `0.8`. The object must remain a free rigid body throughout grasp, lift, short transport, place, and release: no kinematic object slaving, pre-contact object-pose hold, object-to-robot fixed joint, or hidden pose overwrite may contribute to an accepted result.
- Contact-surface decision: The configured selected Dock-link/object pairs use the actual `pitch_dock_mech` / `yaw_dock_mech` collision meshes as gripper surfaces.  At least two selected contacts on distinct Dock-mechanism links are required.  Only the selected Dock-link/object contact pairs are intended robot-object contacts; any meaningful robot-object contact involving a non-selected robot link is an unintended collision.  Initial and placed/released object-floor contact is phase-appropriate and must not be confused with a drop.
- Dock-joint articulation invariant: Every Dock joint in the spawned morphology remains a real, physically movable articulation DOF and remains present in runtime observation and the controller/actuator command map.  Order 8 must not simplify grasping by welding, locking, deleting, narrowing to a zero range, or replacing any Dock joint with a structural fixed transform.  All Dock position/velocity state and complete position/velocity/torque-bias command channels remain observable/auditable, including upstream Dock joints that change a selected gripper-link pose.  A diagnostic policy/command mask may temporarily force a target to zero, but such a mask is non-structural, must be explicitly reported, and must be disabled for Order 8 acceptance.  Acceptance also requires no missing, unsupported, clipped, or unresolved actuator target and no controller/QP infeasible state.
- Contact-acquisition thresholds: A selected contact exists only with normal force at least `0.5 N`.  At least two distinct selected Dock-link contacts must coexist continuously for `0.25 s`.  The nominal frictional grasp target is approximately `11 N` normal force per contact (including about `25%` no-slip margin for the `1 kg`, friction-`0.6` baseline); this is a target, while the hard bounds are `30 N` force and `5 N m` contact torque per selected contact.  Penetration must remain at or below `0.002 m`.
- Slip/contact-maintenance thresholds: After the verified two-contact grasp dwell is acquired, selected-contact tangential slip speed must remain at or below `0.02 m/s`, and accumulated tangential slip must remain at or below `0.010 m`.  During `CONTACT_ACQUISITION`, a provisional selected contact may slide within the separately bounded `+/-0.050 m` surface region, separate, and be reacquired without constituting a hard slip/contact-break failure; its raw slip is still recorded, and an over-limit sample resets rather than earns the final two-contact dwell.  Force, torque, penetration, unintended-contact, raw-evidence validity/saturation, controller, and actuator hard bounds remain active throughout acquisition.  After the two-contact grasp dwell has been acquired, a required selected contact may be absent for at most `0.05 s`; loss beyond that grace interval is a contact break.  Raw per-patch evidence must be finite and unsaturated, and forces from separate patches/links must not be allowed to cancel into a false pass.
- Motion smoke: After contact acquisition, lift until object-bottom clearance is at least `0.100 m`, then execute a short `0.200 m` transport while maintaining the selected-contact, slip, load, collision, controller, and actuator gates.  This is a bounded substrate exercise, not proof of arbitrary-object robustness or full TaskSpec completion.
- Release/settle thresholds: At the intended place/release phase, require selected robot-object contact to be absent continuously for `0.10 s`, then establish at least `0.050 m` gripper-retreat clearance.  Post-release settling requires object linear speed at or below `0.05 m/s` and angular speed at or below `0.10 rad/s`, both continuously for `1.0 s`.  Drop classification is phase-sensitive: after lift and before intended place, object-floor recontact, a sufficient fall, or required-contact loss beyond the grace interval is a drop; object-floor contact during intentional place/release is expected and must not be classified as a drop.
- Policy/controller boundary: Isaac raw contact force/impulse, penetration, slip, and exact per-patch truth are privileged reward/critic/diagnostic and acceptance evidence.  They do not become normal actor observations, `PolicyCommand` contact/internal-wrench fields, or QPID contact-wrench targets.  The approved controller split remains unchanged: QPID realizes centroidal pose/CoM-wrench intent through thrust under the quasi-static assumption, while articulated Dock joints receive independent local absolute position/velocity targets plus bounded torque bias through the actuator bridge.
- Evidence boundary: The Order 8 report must bind task/config, random seed, morphology graph, PhysicalModel, source URDF/collision meshes, generated USD bundle, simulator/backend, controller and actuator mappings, ordered phase trace, raw contact/slip/penetration/load evidence, object pose/twist/clearance, Dock-joint observation/command coverage, and terminal failure cause by hash.  A fake/unit gate, the previous kinematic `kinematic_payload_coupled_attach_v1` path, Orders 6-7 selected-pair collision-filter fallback, a run with any structural Dock-joint lock, or a debug-mask-enabled rollout cannot be relabelled as Order 8 acceptance.
- Determinism/provenance clarification: A reported seed is not sufficient by itself.  Accepted evidence must record that the same non-negative seed was actually applied to Python, Torch, and NumPy (and whether CUDA seeding was applicable).  The measured Isaac physics timestep must equal the requested timestep, and `requested_steps` must equal `ceil(rollout_budget / simulation_dt)`.  Artifact acceptance reloads the versioned config, exact morphology, backend config, robot-model config, PhysicalModel, original URDF, and collision content; it also rehashes the generated resolved URDF, USD payload, and complete USD bundle under the isolated Order 8 output root.
- All-Dock and terminal-metric clarification: Acceptance recomputes the exact ordered global Dock-joint ID set from the current morphology and PhysicalModel.  The expected, observed, position-commanded, velocity-commanded, and torque-bias-commanded sets must equal it exactly; matching counts with invented IDs are insufficient.  Whole-structure kinematics/Jacobian evidence must retain one column per global Dock joint and both selected anchor IDs.  The typed terminal monitor result is rechecked against the active config for force, torque, penetration, instantaneous/cumulative slip, grasp/lift/transport/release/retreat/settle gates, drop, unintended contact, and failure reasons rather than trusting a top-level `passed` boolean alone.
- Free-object audit clarification: The post-spawn object-pose-write count must come from an explicit instrumented counter.  In addition, the final USD stage is scanned across every `UsdPhysics` Joint `body0`/`body1` target; any joint target that resolves to `/World/Order8/Object` or one of its descendants is an object constraint and fails acceptance.  The audit method, zero reference count, and empty offending-prim-path list are required in addition to the top-level free-object booleans.
- Raw-truth boundary clarification: Raw per-patch contact evidence may drive the fail-closed diagnostic/safety monitor, but it is not a nominal planner observation, actor feature, QPID command, or contact-wrench target.  Nominal deterministic planner progression uses commanded dwell and non-contact state/geometry gates; the report must retain `raw_contact_truth_actor_input=false` and `raw_contact_truth_qpid_command=false`.
- Contact-acquisition coordination correction (2026-07-15): Before physical closure, solve and reach a jointly feasible final centroidal pose with the selected meshes open, then hold that dynamically through ordinary QPID while all relevant Dock joints close both selected surfaces together.  The nominal contact point is a surface region: each tangential component may deviate by `+/-0.050 m` from its nominal point while remaining inside the object face; the normal proximity and `0.002 m` penetration limits are unchanged.  A provisional one-sided arrest is not latched or world-fixed and may separate during acquisition.  Measured free-object pose motion updates the centroidal and both anchor targets through the existing command rate limits; the runtime never writes, slaves, or constrains the object pose.  Both sides must satisfy non-privileged mesh-proximity, relative-speed, command-error, and Dock-load dwells simultaneously before `q_close` and force ramp.  At `q_close`, the measured articulated shape is held; contact remains provisional until two selected raw contacts also satisfy the stable `0.25 s` grasp dwell.  Maintained-contact slip accumulation and contact-break enforcement begin only after that verified grasp dwell and continue until planned release.  Acceptance evaluates these behavioral invariants and must not require the superseded one-sided-freeze, centroidal-recenter, or alternating-reacquire implementation sequence.
- Historical final-configuration IK proposal (approved, then superseded on 2026-07-16): An explicit simultaneous collision-aware `q_grasp` solve and bounded `q_open -> q_precontact` trajectory were considered after the receding mesh-point IK path failed. The later approved simple-closure scope above supersedes that proposal for Order 8: this smoke must not spend further work on a general mesh-following or final-configuration grasp planner that learned `pi_L` will replace. The historical proposal is retained only to explain prior diagnostics and is not an active implementation or acceptance requirement.
- Diagnostic GUI boundary (2026-07-16, corrected 2026-07-17): Slow authored-mesh physics may be captured once as a hash-bound robot/object state trace, but a kinematic replay is only a convenience visualization. A joint-value write/readback is not proof that Kit rendered the corresponding articulated link transforms; the first replay implementation passed that circular cache audit while the user observed no visible joint motion. Consequently, physical joint/contact behavior must be judged using the exact live-physics Kit path. `scripts/order8_current_grasp_gui.py` defaults to live physics; state replay requires explicit `--mode replay`, advances no physics, and remains acceptance-ineligible. Contact force, friction, stability, controller feasibility, joint behavior, and Order 8 acceptance continue to come from real-physics execution and evidence, never from replay alone.
- Low-load current-symptom GUI supplement (2026-07-18, corrected after GUI observation): When simultaneous Kit rendering plus the slow three-module authored-mesh physics is unstable, `scripts/order8_lift_symptom_replay_gui.py` may display a separately headlessly captured trajectory. It fails closed unless both source report and hash-bound trace embed the baseline `1.0 kg` payload and reuses the cached trace by default. The original no-step path wrote and reread the same Isaac Lab state cache; its zero-error metric was circular, and the user's viewport and numerical joint inspection correctly showed no motion. The corrected GUI path now advances exactly one gravity-free, contact-minimized PhysX synchronization step per displayed frame, with graph constraints, object/floor/support collision, contact reporting, and robot self-collision disabled, then re-applies the exact recorded root/joint/object state before rendering. Authored cross-module collision geometry remains enabled. A separate PhysX-view readback measured `0.126427 rad` (`7.244 deg`) maximum Dock displacement against `0.126456 rad` (`7.247 deg`) in the trace with zero maximum joint-state error. Terminal progress reports both quantities. The first corrected launch still exposed a visual-only defect: the explicit Kit `performance` preset produced a black normal-mesh viewport although collision debug geometry was visible. The wrapper now defaults to `balanced`, exposes an explicit `--rendering-mode performance|balanced|quality` override, and authors a replay-only uniform Dome Light in addition to the stage Distant Light. A captured real Kit window verifies that the three normal Holon meshes, object, support, and floor are lit and visible, with collision overlay independently toggleable. This path advances synchronization physics and therefore must not be described as no-physics; it remains acceptance-ineligible convenience visualization and cannot establish contact force, friction, control quality, or physical stability. Only the live-physics path and its evidence can do that.
- Runtime/learning separation clarification (2026-07-14): The three-articulation, authored-mesh, per-patch Order 8 rollout is a high-fidelity smoke/acceptance path, not the intended per-update Order 9 training environment.  Order 9 must retain the same policy and safety contracts while using a vectorized GPU training path with cached robot assets, inexpensive aggregate/proxy contact observations where appropriate, curriculum staging, and periodic evaluation against the unchanged full-mesh Order 8 path.  A faster training approximation may not itself satisfy Order 8 or P4-full physical acceptance.

### P4 Full Orders 5-7 Dynamic Assembly Contract

- Funnel-mating correction (2026-07-13): `pitch_dock_mech` is the receiving funnel and `yaw_dock_mech` is inserted into it.  The pitch `*_connect_point_*` frame is the final seated/fixed frame inside the funnel, not the funnel-rim first-contact plane.  Therefore selected pitch/yaw contact may begin while the connect frames are still outside the final fix tolerance.  Such contact is an intended bounded **guidance contact** during axial insertion, not by itself an early-contact failure and not evidence that the fixed constraint may be created.  The bridge must continue insertion while bounding selected-pair force/penetration, relative motion, controller state, and every non-selected contact; only the unchanged strict connect-frame pose/twist gate plus a final seated dwell may emit the exact-frame constraint intent.  Moving the authored connect frames to the funnel rim or relaxing the final `3 mm` fix gate is forbidden.
- Funnel-collision correction (2026-07-13): The authoritative URDF/xacro explicitly uses the Dock STL meshes as collision inputs.  Although the Order 5-7 converter configuration requested `Convex Decomposition`, inspection of the first force-regenerated USD used by the GUI diagnostic found one `PhysicsMeshCollisionAPI` collider per Dock mesh with `physics:approximation="convexHull"`; the pitch funnel cavity was therefore filled.  A physical gate must validate the collision approximation authored in the generated USD, not merely the requested converter string.  The conversion path now post-authors and verifies `convexDecomposition` on every selected Dock collider, but a real physical-funnel insertion still stalled/rotated before the strict seated gate and remains unaccepted.
- Default mating-mode decision (2026-07-13): `selected_pair_collision_filter_fallback` with acceptance contract `selected_pair_collision_filter_fallback_v1` is the production/default Orders 6-7 path. It filters only the selected pitch/yaw Dock rigid-body pair, applies and verifies that relationship during prealignment before the first axial physics step, preserves every other cross-body/environment collision, and requires zero selected-pair contact while active. Closure evidence is the unchanged strict connect-frame pose/twist/controller/dwell gate; no fallback report may claim funnel guidance contact. `physical_funnel_contact` remains available only by explicit override as a non-accepted diagnostic path and is not required for the active roadmap.
- Scope decision: Orders 5-7 are implemented as one sequential workstream with independent gates and commits: Order 5 defines the deterministic `pi_A` `AssemblyControlBridge`, component-level motion planning, and stateful execution contract; Order 6 adds selected-connect-frame approach and dynamic constraint attachment in Isaac; Order 7 adds unload-gated detach, constraint removal, stable separation, and round-trip evidence. A later order may not relabel an earlier fast/unit gate as physical attach or detach success.
- Terminology correction: Holon Dock joints articulate Dock mechanisms and thereby change connector/contact-link kinematics and whole-structure morphology. They are not latch open/close actuators. For module assembly, `dock` means controlled axial approach followed by dynamic fixed-constraint commit between the exact selected connect frames. `verify_attach` verifies the constraint and performs control-graph/controller handover. Previous roadmap wording about Dock-joint closure or latch release is superseded for module-to-module assembly.
- Controller boundary: QPID remains unaware of joint-motion multibody dynamics under the approved quasi-static assumption. It updates centroidal mass/CoM/inertia/rotor geometry from the current configuration and allocates thrust; non-vectoring Dock joints use independent local position/velocity servo plus bounded torque bias. Assembly and later object-grasp planning may use full-chain kinematics/Jacobians and all upstream morphology joints, but normal QPID does not become a contact-aware multibody inverse-dynamics controller.
- Component-control decision: Before attachment, each connected component owns an independent morphology/runtime/controller context. The leader component holds its current centroidal pose while the follower component follows deterministic staging/alignment waypoints. `AssemblyControlBridge` emits component-scoped `centroidal_local_joint_v2` intent only; QPID, local servo, actuator bridge, and safety remain final actuator authority.
- Motion-planning decision: Order 5 owns a deterministic collision-aware assembly motion planner. It first checks a direct SE(3) path and then bounded deterministic via-point alternatives; lack of a collision-free path fails closed. This is a retained assembly fallback, not an optimal kinodynamic planner. Order 6 consumes the resulting path for physical execution, and Order 9 evaluates it in varied TaskSpec scenes.
- Attach sequence: Move to a collision-free staging pose with a configurable positive offset along the selected leader connect frame X axis; align the selected follower frame face-to-face while preserving the axial gap; require transverse/attitude/relative-twist dwell; reduce only the axial gap at bounded speed; require the unchanged strict final connect-frame pose/twist/controller/collision gate and seated dwell; then create and verify a dynamic fixed constraint using the exact selected connect frames. In physical mode, bounded selected pitch/yaw funnel guidance contact is permitted during insertion and initial funnel-rim contact need not lie near both final connect planes. In fallback mode, the exact selected pair is already filtered/verified during prealignment and any selected-pair contact is a contract violation. Wrong-body contact, excessive penetration/force, loss of the applicable envelope, unsafe controller state, or failure to reach the final gate retreats or fails closed and never freezes the observed misaligned transform.
- Canonical-joint decision: Module docking begins from the current graph's canonical neutral Dock configuration (`q=0`) and primarily uses component flight pose for coarse alignment. Only bounded joint correction may be used for final connector-frame alignment. There is no global per-joint open/close direction. In later object grasping, the morphology-conditioned policy coordinates every upstream Dock joint that changes a contacting Dock-mech link pose; it must not reduce grasp control to the joint immediately adjacent to the contact link.
- Attach handover decision: After constraint verification, merge physical/control graphs, initialize the assembled controller at the measured centroidal pose/twist and joint state, reset/reinitialize controller integrators, and blend from the two previous component commands. The constraint fixes only the selected connect-frame relation; upstream Dock joints remain articulated and may morph the attached structure. Only the exact selected attached body pair may be filtered/marked as intended. In fallback mode its prealignment filter remains owned and verified through attachment and is removed only after release plus measured separation clearance. All other leader-follower/environment collisions remain active, but each single-Holon Articulation still launches with intra-articulation self-collision disabled; intra-component self-collision safety is therefore unvalidated, not implicitly accepted.
- Detach decision: Order 7 uses the existing follower-subtree estimator and `parent_component_on_follower_component` sign convention. It prepares independently feasible component controllers, requires external-contact-free valid estimation and unload dwell, splits the control graph before physical release, removes the exact recorded constraint, commands bounded separation, and verifies stable independent hover. Simulator constraint reaction is privileged validation evidence, not the sole release-gate source.
- Physical-gate separation: `DynamicAssemblyIsaacConfig.acceptance_gate` is explicitly `attach_only` (Order 6) or `roundtrip` (Order 7), and each result also binds its mating mode/acceptance contract. The Order 6 validator ends after strict seated evidence, verified constraint identity, the applicable selected-body contact/filter contract, continuous component-to-assembled controller handover, and attached pose/twist/Dock-joint/contact-free hold; it neither requires nor implies detach. The Order 7 validator additionally requires continuous assembled-to-component command blending, raw-patch follower external-contact-free unload evidence, estimator dwell, exact constraint removal, current selected-body collider clearance, verified collision-unfiltering, zero recontact after unfilter, separation, and continuous post-release pose/attitude/twist/Dock-joint/gap/clearance dwell. The CLI defaults to distinct report paths per gate and embeds seed/sampling, graph, effective config, backend config, and PhysicalModel hashes so neither a gate nor a mating mode can masquerade as another.
- Conversion/provenance decision: Every Orders 6-7 real-Isaac gate force-converts the current resolved Holon URDF into the isolated `artifacts/isaac/robots/holon_dynamic_assembly` bundle using `Convex Decomposition`; a cached USD is not accepted. Validation binds current collision-geometry content, PhysicalModel, backend/config, resolved-URDF file, generated USD, and the complete USD payload directory by SHA-256.
- Contact-evidence decision: PhysX speculative raw patches alone are not physical contact. During physical funnel insertion, finite unsaturated selected-body raw patches with nontrivial solver force or penetration are classified as guidance contact and checked against configuration-backed force/penetration limits; they are not required to be near both final connect planes. During fallback insertion, selected contact is forbidden and strict final seated alignment/dwell is explicitly contactless evidence. Final constraint creation in either mode still requires the strict connect-frame pose/twist/controller/collision gates and final dwell. Every non-selected body pair is classified per raw force/separation patch so opposing forces cannot cancel; saturation, non-finite data, wrong-body contact, excessive guidance load, fallback selected contact, or selected-pair recontact after unfilter fails closed. Collision broad phase uses each URDF collision element including its authored `<collision><origin>` transform and mesh scale; `PhysicalModel.collision_primitives` identity must match those records. Release clearance is recomputed from each selected rigid body's current pose and body-local collider bounds, not from a cached component pose or connect-frame X distance.
- Separation/filter lifecycle decision: The roundtrip path uses a measured `DynamicSeparationLifecycle`, not a fixed-duration assumption. It completes at least the nominal separation ramp, then holds the final target until actual connect-frame gap and selected-body collider clearance meet their gates, removes and verifies only the owned exact-pair filter, and acquires a resettable continuous post-release dwell. Separation acquisition and stable-dwell acquisition each fail closed within at most twice their nominal budgets. Global post-unfilter minimum gap/clearance and selected-pair contact counts participate directly in `detach_passed`.
- Floor/handover decision: Each component is initialized on the floor with explicit zero Dock position, velocity, and effort-bias targets and must show continuous measured floor-contact plus low root/joint speed dwell before takeoff. Controller topology changes interpolate complete actuator-domain `ControllerCommand`s; merely waiting for a nominal blend duration is not handover evidence. Attached and post-release dwell also bound measured Dock position/speed while targets remain zero. Missing/duplicate actuator keys, infeasible endpoints, clipping, unresolved targets, non-finite state, raw-contact invalidity, filter/constraint identity mismatch, or clearance loss fails the gate.
- Runtime-numerics decision: Orders 5-7 dynamic assembly uses PhysX solver position/velocity iterations `8/8`, AK40-10 implicit-drive stiffness/damping `200/2`, and explicit simulated effort/velocity limits `4.1 Nm / 3.0 rad/s`. The validator compares solver iterations exactly, requires finite positive drive/limit evidence, and binds the effective config and PhysicalModel provenance hashes; reports also record the complete axial selected-Dock command trace. The accepted fallback run had 6,676 samples with every Dock position, velocity, and torque-bias target exactly zero.
- Live-progress decision: Every dynamic-assembly phase transition emits a flushed terminal line containing simulation time, a user-facing phase label, and the canonical event name; while a phase remains active, the same line is refreshed as a heartbeat every `1.0` s of simulation time. `axial_approach`, `constraint_enabled`, and `unload_dwell` are displayed as `axial`, `fixed`, and `unload`; `staging` and `separation` retain their names. The outer synchronous runner drains stdout/stderr concurrently, forwards only `[dynamic-assembly]` progress lines live, retains stderr tail diagnostics, and still parses the final JSON report without changing acceptance.
- Verification status (2026-07-13): The explicit physical-funnel diagnostic was rerun after force regeneration and composed-USD `convexDecomposition` verification, but it still did not reach the strict final seated gate because contact induced stall/rotation; no physical attach/release pass is claimed or required by the selected default. The seed-2 default fallback passed both `attach_only` and `roundtrip`. Its final attach errors were `0.000160594 m` axial, `0.001994758 m` transverse, and `0.000182269 rad` attitude with relative speeds `0.000769583 m/s` and `0.017335538 rad/s` over a `0.100000 s` dwell. Roundtrip removed the filter after 954 separation steps at `0.200870784 m` gap and `0.030092942 m` clearance, recorded zero post-unfilter selected contact, and completed 200 continuous stable samples (`1 s`). These numbers validate the fallback contract only. The `0.003 m` final gate and authored connect frames remain unchanged.
- Integration boundary: The dedicated Order 7 Isaac smoke directly exercises controller split, the follower-subtree unload estimator/gate, constraint removal, and separation. Generic `AssemblyStep(detach)` plus `ConstructionState` split integration remains downstream P4-full work and must be completed before the end-to-end assembly executor can claim general detach support.

### P4 Full Order 4 Free-Flight Deterministic pi_H and Trajectory Runtime Contract

- Scope decision: Order 4 implements the production deterministic `pi_H` fallback and the policy-agnostic `ContactWrenchTrajectory` runtime for the free-flight slice only. It is not an evaluation of learned `pi_H` quality and does not claim completion of contact-aware `pi_H`. The deterministic planner is retained for runtime fallback, teacher-data generation, and learned-policy baseline comparison; it is not disposable test-only code.
- Input/output decision: The planner consumes the existing `HighLevelPolicyContext` boundary (`IRG`, `InteractionEnvelope`, `MorphologyGraph`, empty `ContactCandidateSet`, and `RuntimeObservation`) plus versioned free-flight mission/config data represented in the IRG context. It emits the unchanged `ContactWrenchTrajectory` schema. Existing persisted policy, checkpoint, morphology, and runtime-observation schemas are not changed.
- Runtime decision: A common executor owns the rolling-plan time origin, sorted-knot validation, active segment selection, centroidal target interpolation, plan expiry, and fail-closed safe-hold behavior. The planner is updated at the v0.4 default `2 Hz`; `pi_L` and QPID continue at their existing rates. The executor passes an explicit active knot downstream, so `InteractionKnot.t_rel_s` is never compared directly with episode-absolute `RuntimeObservation.time_s`.
- Scenario decision: The verified mission is floor contact settle -> takeoff ramp -> hover acquisition -> multiple translation/attitude waypoints -> final hover. State-dependent guards use measured centroidal pose/twist, controller state, dwell, and phase timeout; a fixed time-indexed target list alone is insufficient. Automated acceptance requires at least `5 s` continuous final hover, while a representative endurance run uses at least `20 s`. Short Order 3 training terminal dwell remains a separate efficiency setting and is not redefined by this order.
- Control boundary: The active centroidal/posture knot is consumed through `pi_L` (`BaselineLowLevelPolicy` fallback or an explicitly selected compatible learned Order 3 checkpoint), then QPID/local joint servo and the Isaac bridge. `pi_H` and `pi_L` never emit final actuator commands. Dock joints remain at absolute zero position/velocity/torque bias and vectoring remains allocator-owned. Normal QPID receives no contact-wrench or internal-wrench target.
- Contact/reachability boundary: Every Order 4 knot has empty contact assignments and no object/contact target. Simultaneous multi-anchor reachability is therefore reported as `not_applicable_no_active_assignments`, not as a vacuous feasibility pass. Contact candidate selection, contact schedules/wrench requirements, and simultaneous reachability become executable validation only after the Order 8 natural-contact substrate exists.
- Progress boundary: Order 4 records its own phase, waypoint index, and mission progress in its runtime/report contract. It does not silently change `RuntimeObservation.task_progress.progress_ratio` supplied to an existing Order 3 actor checkpoint; checkpoint-visible task features remain compatible unless a later versioned retraining explicitly changes them.
- Failure boundary: Non-finite input/output, malformed or expired trajectory, controller fault/infeasibility, mission/phase timeout, or an invalid non-empty contact assignment fails closed to a bounded centroidal safe hold and is recorded with an explicit reason. A safe hold is fallback evidence, not mission success.
- GUI boundary: The same real-Isaac execution path used headlessly must expose a Kit viewer, real-time playback, module-count/seed selection, mission waypoint configuration, optional compatible `pi_L` checkpoint, and post-run viewing. GUI inspection is diagnostic and does not replace typed report validation.
- Implementation result: Added versioned Order 4 mission/runtime/report contracts, a state-dependent `DeterministicFreeFlightPlanner`, a policy-agnostic rolling `ContactWrenchTrajectoryExecutor`, free-flight context/IRG construction, fail-closed runtime integration, real-Isaac report validation, configuration, and `scripts/order4_free_flight_pi_h.py`. The default mission contains three hover-relative translation/attitude waypoints; custom waypoints are repeatable CLI inputs in xyz/rpy radians. The CLI defaults to a freshly sampled morphology seed unless a reproducible seed is provided.
- Verification result: Real-Isaac deterministic-baseline-`pi_L` runs passed for module counts 2, 3, and 8. Each executed 37 rolling replans over 18.005 s, completed all three waypoints, accumulated 5.505 s final hover, had zero QP infeasibility/unintended cross-module contact/missing or unsupported actuator, and kept all Dock commands at absolute `0/0/0`. Maximum measured Dock angle was `0.001056`, `0.001780`, and `0.002504 rad`, respectively, below the configured `0.0053 rad` bound. The N=3 20 s endurance run passed after 67 replans/33.005 s with 20.505 s final hold and zero safety failures. A first endurance attempt hit the inherited 300 s subprocess timeout before report generation; Order 4 now uses an explicit 600 s timeout and the rerun passed.
- GUI result: The N=3 Kit run used the same mission/hash/report gate with real-time playback and a 5 s post-rollout viewer hold; it passed with `final_phase="complete"` and 5.505 s final hover. The canonical user command is `/home/leus/.local/bin/micromamba run -n isaaclab3 python scripts/order4_free_flight_pi_h.py --real --viewer kit --realtime-playback --module-count 3`; add `--seed N` for a reproducible structure, `--endurance` for a 20 s final hold, or repeated `--waypoint X Y Z ROLL PITCH YAW` arguments for a custom mission.
- Completion boundary: Order 4 is complete for the deterministic free-flight planner/runtime/fallback slice. This does not validate a learned `pi_H`, contact assignment, contact wrench planning, simultaneous multi-anchor reachability, object interaction, dynamic docking, full TaskSpec delivery, or P4-full acceptance. The optional compatible Order 3 `pi_L` checkpoint path is wired and hash checked; the recorded Order 4 real-Isaac acceptance evidence uses deterministic baseline `pi_L` so that the high-level/runtime fallback is independently verifiable.

## 2026-07-12

### P4 Full Orders 1-10 Handoff Roadmap

- Numbering scope: The `Order 1` through `Order 10` labels below are the user-approved local implementation order for the remaining P4-full program. They are distinct from the older internal order numbers under P4-control, P4.1, P4.2, P4.3, and from Section 27's repository-wide implementation order. `Order 0` (joint actuator performance) is the completed prerequisite and is not renumbered into this list.
- Current boundary: Orders 1-8 are complete at their agreed implementation/representative-verification boundaries, with Order 2.5 as the inserted controller-contract prerequisite. Orders 6-7 real-Isaac `attach_only` and `roundtrip` evidence is accepted under the default selected-pair collision-filter fallback contract; physical funnel-contact remains optional. Order 8 has a complete proxy-free, free-object, actual-mesh real-Isaac acceptance run. The full configured 2-8-module statistical learning matrix, learned contact-aware `pi_H`, full TaskSpec delivery, and P4 full acceptance remain open in Orders 9-10.
- Artifact boundary: Order 3 source is committed in `59aba6d` and `629008c`. Regenerated training/evaluation artifacts under `artifacts/p4_full/order3_pi_l_v2/` are intentionally ignored local evidence. A later chat must regenerate or explicitly select a hash-bound checkpoint; it must not assume ignored artifacts exist after a fresh clone.

1. **Order 1 — random feasible connected morphology distribution — complete.** Ownership: Agent E/F/I/L. The seeded, structurally deduplicated, task-independent 2-8 module distribution and graph/flight feasibility gates are implemented in `amsrr/morphology/random_connected.py`, `amsrr/morphology/random_feasible.py`, `amsrr/feasibility/morphology_flight.py`, and their tests. Downstream code must consume `RandomFeasibleConnectedMorphologyDistribution`, not the raw proposal generator.
2. **Order 2 — floor initialization and deterministic takeoff-to-hover — complete.** Ownership: Agent I/J/K/L. Graph-specific fixed morphology generation, collision-derived floor placement, zero-thrust settle, deterministic ramp/hover, exact unintended-contact checks, typed reports, and archives are implemented around `amsrr/simulation/random_morphology_takeoff.py`, `amsrr/training/random_morphology_takeoff_runner.py`, `scripts/random_morphology_takeoff.py`, and `scripts/p4_control_holon_spawn_probe.py`.
3. **Order 3 — morphology-conditioned `pi_L` training — complete.** Ownership: Agent I/K/L. The completed source covers versioned schemas, homogeneous module/DockEdge graph encoding, recurrent actor/critic, deployable actor versus privileged critic/reward separation, deterministic-v2 BC, one-generation/one-update PPO, morphology pool splits, online behavior replay, checkpoint/fallback contracts, real-Isaac rollout/evaluation, and fail-closed acceptance arithmetic. Primary files are `amsrr/schemas/order3*.py`, `amsrr/encoders/morphology_graph_encoder.py`, `amsrr/policies/morphology_conditioned_low_level_policy.py`, `amsrr/training/order3_*.py`, `amsrr/simulation/order3_*.py`, `amsrr/acceptance/order3_acceptance.py`, and `scripts/order3_morphology_pi_l.py`. The free-flight dock decoder commands absolute `0/0/0` position/velocity/torque-bias holds; vectoring remains allocator-owned. Representative BC/PPO/headless/GUI evidence passed, but P4-full completion is not claimed.
4. **Order 4 — deterministic free-flight `pi_H` fallback and `ContactWrenchTrajectory` runtime — complete.** Ownership: Agent H/I/K. A production state-dependent deterministic planner and policy-agnostic rolling-trajectory executor now cover floor settle, takeoff, hover acquisition, multiple translation/attitude waypoints, final hover, abort, and safe hold. Real-Isaac N=2/N=3/N=8, N=3 20 s endurance, and N=3 Kit GUI verification passed. This completes only the free-flight planning/runtime slice; learned/contact-aware `pi_H` remains downstream. Empty contact assignments make simultaneous reachability explicitly not applicable in this order. `pi_L`, QPID, safety, and actuator ownership remain unchanged.
5. **Order 5 — `pi_A` AssemblyControlBridge — complete.** Ownership: Agent G/I/J. The versioned bridge emits exactly two component-scoped `centroidal_local_joint_v2` intents, separates axial/transverse/attitude/connect-frame-twist gates, maintains leader hold, drives follower staging/approach, and emits exact constraint create/verify intent without final actuators. Its final seated dwell is strict in both modes: physical mode additionally validates bounded selected contact, while the explicit fallback requires zero selected contact. A bounded deterministic collision-aware SE(3) planner checks direct and via-point paths. `ClosedLoopAssemblyExecutor` preserves the four legacy `AssemblyStep` boundaries while one stateful bridge session spans the attach sequence. Dock joints remain canonical `q=0` plus explicitly bounded correction; no latch/open-close semantics are used. The Order 5 fast gate passed 53 assembly/controller tests; real-Isaac attachment belongs to Order 6, whose fallback gate is now accepted while physical-funnel acceptance remains open.
6. **Order 6 — connect-frame alignment and dynamic fixed constraint — complete under explicit fallback contract.** Ownership: Agent B/G/J. Two independent Articulations, floor/takeoff/staging/approach, exact external FixedJoint identity, strict final pose/twist/dwell, continuous attached pose/twist/Dock-joint hold, fail-closed report validation, and the independent `attach_only` gate are implemented. The seed-2 selected-pair fallback run passed with empty validation failures and zero selected-pair contact. The physical-funnel mode remains a distinct unaccepted diagnostic path; this completion label does not claim natural/physical Dock contact.
7. **Order 7 — attach/detach Isaac smoke — complete under explicit fallback contract.** Ownership: Agent G/J/K/L. Continuous bidirectional controller handover, uncancelled raw-patch follower external-contact evidence, the existing subtree estimator/unload dwell, graph split, exact constraint removal, measured separation lifecycle, verified delayed exact-pair unfilter, zero post-unfilter recontact, continuous independent-hover/Dock-joint/gap/clearance dwell, report hashes/phases/failure causes, and a distinct `roundtrip` gate are implemented and passed in seed 2 after Order 6 fallback passed. Generic `AssemblyStep(detach)`/`ConstructionState` split integration, arbitrary preassembled components, and intra-component self-collision validation remain downstream work.
8. **Order 8 — object natural-contact smoke — complete; separate from Order 9.** Ownership: Agent H/I/J/K/L. The versioned config/evidence schemas, actual Dock-mesh surface selection with spatially uniform audited compliant contact, full-Dock whole-structure controller path, deterministic phase planner, free-object Isaac runtime, per-patch monitor, fail-closed wrapper/CLI, provenance revalidation, and no-mislabeling acceptance are implemented. Isaac contact truth remains privileged diagnostic/safety evidence and is not a normal actor/planner input or QPID contact-wrench target. The canonical proxy-free v406 real-Isaac run uses the configured `mu=4.5`, `7500/75` contact, zero post-grasp torque bias, normal safety gates, and a free `1 kg` object; it completed grasp, lift, `0.2 m` transport, place, release, retreat, settle, and terminal completion with empty wrapper/artifact-acceptance failures. Maximum contact displacement, force, and penetration were `19.406 mm`, `10.046 N`, and `0.719 mm`, all within their unchanged gates. This is one deterministic representative substrate acceptance, not a learned-policy, held-out robustness, full TaskSpec, or P4-full claim; those remain Orders 9-10.
9. **Order 9 — full TaskSpec delivery — pending.** Ownership: Agent C/D/E/F/G/H/I/J/K/L across the end-to-end boundary. Run the prescribed staged learning process from newly initialized models, including deterministic teacher/BC warm start where applicable and later online refinement, then show that the generated morphology and control stack completes TaskSpec goals. Vary module count, object geometry/mass, initial poses, disturbances, and scene settings across disjoint train/validation/held-out splits. Required outputs include checkpoints, configs, metrics, reward curves, raw Isaac rollouts, task success/failure causes, and reproducible hashes. Contact smoke success alone is insufficient.
10. **Order 10 — P4 full acceptance — pending.** Ownership: Agent L with evidence producers from all upstream work packages. Aggregate real-Isaac deterministic and learned evidence, dynamic assembly, natural contact, full TaskSpec robustness, fallback/safety behavior, archive completeness, and no-mislabeling checks. Recompute acceptance from raw hash-bound evidence and require the source specification's P4-full artifacts and rates. Neither simplified P4.0, kinematic P4.2, P4.3 minimum learning, nor Order 3 free-flight evidence may be relabelled as Order 10 completion.

- Order 4 handoff: Primary source is `amsrr/schemas/order4.py`, `amsrr/policies/contact_wrench_trajectory_runtime.py`, `amsrr/policies/deterministic_free_flight_planner.py`, `amsrr/simulation/order4_free_flight.py`, `scripts/order4_free_flight_pi_h.py`, the Order 4 config, and the additive Isaac-probe path. Ignored evidence is under `artifacts/p4_full/order4_deterministic_pi_h/` and is not a fresh-clone dependency. Order 5 must consume the established controller boundary but must not reinterpret the reset-time fixed morphology used here as dynamic docking proof.

### Corrected Dock Neutral Geometry and Order 3 Artifact Regeneration

- Context: The source/runtime Holon URDF dock origins were corrected by the user so the yaw-dock origins and `pitch_dock_mech_joint2` origin rotation are neutral at zero joint position. Morphology graphs, fixed-assembly URDF/USD assets, and learned-policy evidence generated from the previous frames are therefore stale even when their topology hash is unchanged.
- Fixed-assembly decision: Every fixed, connected morphology must be realizable with all dock mechanism joints at absolute position `0 rad`. The deterministic takeoff command and the safety-masked Order 3 joint decoder explicitly command complete dock position/velocity/torque-bias maps with values `0/0/0`; measured joint position must never be copied forward as the next position target. Vectoring ownership remains unchanged.
- Frame-consistency gate: Before floor placement or probe launch, recompute the two selected connect frames of every `DockEdge` from the current `PhysicalModel` and graph poses. Position or attitude disagreement fails closed and instructs regeneration. This prevents a structurally identical but frame-stale morphology graph from being executed after URDF edits.
- Drive-tuning decision: The current AK40-10 Isaac position drive uses `200 Nm/rad` stiffness and `2 Nms/rad` damping, with an explicit dynamic-assembly effort/velocity cap of `4.1 Nm / 3.0 rad/s`. At the configured `0.005236 rad` backlash, the stiffness requests about `1.047 Nm`, below the manufacturer's `1.3 Nm` rated torque; these are simulation control settings, not manufacturer stiffness/damping claims. The completed historical Order 3 regeneration used damping `1 Nms/rad`, so its recorded artifact hashes and numerics remain historical rather than current Orders 5-7 evidence. The previous `20 Nm/rad` stiffness allowed a real-Isaac fixed assembly to deflect to `0.00807 rad` under floor/takeoff constraint load even though every target was zero; the stiffness-200 Order 3 runs remained below `0.00298 rad` and ended below `0.00051 rad`.
- Evidence gate: Fixed-dock smoke/evaluation reports bind the expected joint count, configured tolerance, maximum/final measured absolute dock angle, and maximum absolute position, velocity, and torque-bias commands. Passing requires all command maxima to be zero and measured motion to remain within the `0.0053 rad` one-backlash tolerance. BC/PPO collectors accept explicit complete zero hold maps but continue to reject any non-zero dock motion intent during the free-flight curriculum.
- Regeneration boundary: Changing either URDF dock frames or actuator drive configuration changes PhysicalModel/provenance hashes. The morphology pool, graph JSON files, fixed-assembly USD cache, BC source reports/dataset/checkpoint, stochastic PPO reports/dataset/update, and checkpoint evaluation must be regenerated. Older artifacts are retained only as historical diagnostics and must not be mixed with the regenerated lineage.
- Verification boundary: A regenerated held-out PPO checkpoint passed both headless and Kit GUI execution with 50 policy decisions, fallback zero, maximum/final dock angle `0.000234/0.000232 rad`, zero position/velocity/torque-bias command maxima, final position error `0.01250 m`, and final attitude error `0.00409 rad`. GUI inspection is diagnostic; it does not replace the configured full Order 3 statistical matrix or establish P4 full acceptance.

### Order 3 Morphology-Conditioned pi_L Training Contract

- Scope decision: Order 3 trains `pi_L` for free-flight control of fixed, connected 2-8 module morphologies. The curriculum covers in-air hover, translation/attitude waypoints, randomized initial state and model/disturbance conditions, then floor takeoff-to-hover. Object contact, dock opening/closing, dynamic attach/detach, and full TaskSpec execution remain later orders.
- Morphology representation: The persisted `MorphologyGraph` schema is unchanged. One morphology is tensorized as one homogeneous module graph: `ModuleNode` entries are graph nodes, `DockEdge` entries are directed message-passing edges in both directions, and the referenced source/destination `PortNode` properties are edge features. Unused-port, RobotAnchor, and ControlGroup summaries are module/global features. Runtime module state is joined to nodes by schema module ID. This is an encoder-internal adapter, not a conversion of `MorphologyGraph` into multiple persisted graphs or a heterogeneous schema.
- Policy decision: The new actor uses the `centroidal_local_joint_v2` contract and a new dataset/checkpoint version. The active centroidal pose target passes through deterministically; the actor produces bounded centroidal twist correction and CoM residual wrench. A source-ID/mask-aware non-vectoring joint decoder converts internal bounded joint deltas into absolute position/velocity targets and torque bias, but the initial free-flight curriculum keeps that decoder safety-masked to deterministic hold. Vectoring joints remain allocator-owned.
- Architecture decision: Use a permutation-invariant module-graph encoder with DockEdge/port edge features, fused with true centroidal runtime/target/controller features, previous action, and a short recurrent state. The actor receives only deployable observations. Isaac-only disturbance/contact truth may enter the critic/reward and diagnostics, never actor features or normal QPID targets.
- Training decision: Use deterministic v2 imitation to warm-start reference pass-through, zero residual, and local-joint hold, followed by bounded residual PPO. Keep deterministic fallback, OOD/non-finite checks, contract/version checks, controller clipping, and a configurable trust-region blend throughout training and evaluation. The Order 2 scheduler may provide training-only targets; the production deterministic `pi_H` takeoff/hover scheduler remains Order 4.
- Split/runtime decision: Morphology train/validation/held-out sets are disjoint by canonical structural hash and balanced across module counts 2-8. The initially proposed `8/2/2` split is used for module counts 3-8. Exhaustive enumeration found that the current rooted port-labelled canonical representation has only eight distinct two-module structures, so the two-module split is necessarily `4/2/2`; silently duplicating hashes across splits is forbidden. Graph-specific assets are cached. Isaac rollout batches may group identical morphology assets, while the shared policy is updated across morphology batches. Dataset/config/checkpoint/pool hashes and source graph IDs must be archived.
- Initial acceptance target: held-out aggregate success at least 95 percent and each module-count success at least 90 percent; no QP-infeasible, hard-collision, non-finite, unsupported-actuator terminal; nominal performance no more than 5 percent worse than deterministic baseline; randomized-disturbance tracking error at least 15 percent lower or success at least 10 percentage points higher; in-distribution fallback at most 1 percent; deterministic OOD fallback verified. These are Order 3 free-flight criteria, not P4 full acceptance.
- Curriculum realization decision: Each in-air curriculum stage has an explicit `initial_state_randomization_scale`. It deterministically samples bounded root position, orientation, world linear velocity, and body angular velocity from the episode seed; floor-takeoff stages require this scale to be zero. Model mass/inertia/thrust scales and the external wrench schedule remain simulator-only episode conditions. Production PPO derives a fresh reproducible seed from update index, structural hash, and base curriculum condition instead of replaying one fixed randomization across all graphs/updates.
- Terminal-evidence decision: Isaac reset may advance an unconstrained articulation under gravity, so every hash-bound Order 3 condition is explicitly re-applied after reset before the first observation/evidence sample. Terminal dwell may start only after the waypoint ramp and, for a finite disturbance, after the disturbance ends; a persistent disturbance requires dwell after it begins. Pre-condition dwell cannot satisfy success. Randomized tracking cost is the mean true-centroidal normalized tracking error over a reported paired window beginning at disturbance onset, while terminal metrics remain a separate final success gate.
- On-policy PPO decision: A rollout serializes the exact causal observation consumed by the actor, including the previous controller status; the current command outcome first appears in the next observation. Production collection replays every stored action/log-probability/value/GRU chain against the hash-bound behavior checkpoint. One fresh rollout generation feeds exactly one PPO update. Complete recurrent episodes are seed-shuffled and selected in module-count round-robin order before the step budget is applied, and advantages are normalized only over selected transitions.
- Evaluation provenance decision: Learned and deterministic-baseline evaluation reports share the exact condition payload/hash and bind seed/application evidence, requested/applied model scales, initial-state application, terminal/tracking windows, backend config hash, PhysicalModel hash, collision-geometry hash, checkpoint hash, and raw file hashes. Learned report paths also include checkpoint hash to prevent one checkpoint evaluation from overwriting another. OOD safety fields are validated before accepting fallback evidence.
- Compatibility decision: Existing `p4_3_pi_l_checkpoint_v1`, flat base-`fc` features, legacy delta targets, and legacy P4.3 dataset artifacts remain readable historical outputs and must not be loaded as, migrated into, or reported as `centroidal_local_joint_v2` evidence. Code structure may be reused, but all v2 artifacts require explicit new metadata and fail-closed loaders.

### Order 3 Learned-Policy GUI Evaluation Supplement

- Context: Order 3 learned rollouts already executed the checkpoint in the real Isaac probe, but the staged CLI did not expose the probe's existing Kit visualization controls.
- Decision: `evaluate-learned` and its internal single-rollout command accept `viewer="kit"`, real-time playback, and a non-negative post-rollout hold. They propagate to the unchanged Isaac probe as `--viz kit`, `--realtime-playback`, and `--keep-open-after-smoke-s`.
- Safety and evidence boundary: Visualization is accepted only for `--real` learned-policy execution; real-time playback and post-rollout hold require the Kit viewer. The checkpoint hash, morphology/condition selection, controller/QP/safety ownership, report validation, and acceptance semantics remain identical to headless evaluation. Viewing a rollout is diagnostic evidence and does not establish Order 3 statistical acceptance or P4 full completion.
- Compatibility impact: No persisted schema, checkpoint, dataset, actor input, or probe report change. New runtime arguments default to the existing headless behavior.

### Order 2.5 Centroidal Controller Contract Implementation

- Implementation status: The approved Section 14 normal-control migration is now implemented as the explicit `centroidal_local_joint_v2` contract. `legacy_contact_bias_v1` remains the default when old JSON/checkpoints omit version metadata, so existing archives remain readable without reinterpreting their field meanings.
- Schema/controller implementation: `PolicyCommand` and `ControllerCommand` now carry contract version, absolute non-vectoring joint position/velocity targets, and joint torque bias. On the v2 path, `contact_tracking_bias`, anchor offsets, and knot-level centroidal wrench preference are not controller references; `residual_wrench_body` is the only policy-provided additive CoM wrench bias. Controller metrics explicitly report zero contact-wrench, internal-wrench, and generic-joint QP variables.
- Centroidal implementation: `RigidBodyControlModel` now exposes CoM-origin `body_pose_world` and `body_twist_world`; linear velocity is shifted from the selected control-body origin by `omega x r`. Under the v2 contract, `desired_body_twist[0:3]` is world-frame CoM linear velocity and `desired_body_twist[3:6]` is control-body-frame angular velocity. The v2 QPID pose loop uses the true morphology-centroidal state with that explicit mixed-frame convention, while legacy behavior retains its recorded base-module semantics.
- Joint/bridge implementation: Vectoring targets on the v2 path come only from the thrust allocator. Observed configured non-vectoring joints receive deterministic current-position/zero-velocity hold unless replaced by absolute policy targets. Dock channels support simultaneous native position, velocity, and effort-bias records; position/velocity/effort bounds, continuous torque-bias clamping, finite checks, unsupported-mode failure, and missing/clipped logging remain controller/bridge owned. Isaac application now sends the three target modes through their distinct articulation APIs.
- Detach implementation: Added a follower-subtree estimator with the fixed sign convention `parent_component_on_follower_component`. It removes the candidate edge, rejects non-separating cuts and missing external-contact-free evidence, computes follower momentum rate minus known rotor, gravity, and declared other external wrench, and shifts the result from follower CoM to follower dock frame. A stateful fail-closed unload gate checks estimator/contact validity, parent and follower QP feasibility, relative pose/speed, separate force/torque limits, and consecutive dwell. Default provisional limits are 0.5 N, 0.05 Nm, 5 mm, 2 degrees, 0.02 m/s, 0.10 rad/s, and 20 control steps; they remain configurable and must be re-tuned with hardware/Isaac noise data.
- Privileged-data boundary: Existing `pi_L` actor features may use contact assignment requirements and active-contact count, but not measured per-contact wrench values. A regression changes privileged `ContactState.wrench_world` while keeping actor-visible state fixed and requires an identical feature vector. Privileged reward computation remains separate.
- New evidence: A dedicated three-module seed-25 real Isaac run using `configs/training/order2_5_centroidal_control.yaml` passed 917 steps of floor settle, true-centroidal takeoff, and hover. It reported final position error 0.062446 m, final attitude error 0.000368 rad, final linear/angular speed 0.047828 m/s and 0.000657 rad/s, zero QP infeasibility, controller/bridge clipping, missing/unsupported targets, unresolved actuator targets, unintended cross-module contact, or report-validation failures. Reports explicitly deny contact/internal-wrench tracking claims and bind the QP scope to rotor thrust, vectoring, and slack only.
- Remaining boundary: This Order 2.5 implementation does not retrain or relabel the existing v1 learned `pi_L` checkpoint, does not claim natural-contact task success, and does not yet connect latch release/post-release separation execution to `pi_A`. Order 3 must train a new checkpoint under the v2 contract; later dynamic assembly work must call the detach estimator/gate and add actual latch override plus post-release stability evidence.

### Centroidal-Only QPID and Independent Joint Servo Design Revision

- Context: The v0.4 Section 20 contract allows `π_L` contact-tracking bias and a contact-aware QP objective. During the pre-Order-3 design review, explicit per-contact wrench allocation and normal-operation dock internal-wrench optimization were judged to reintroduce the multi-contact dynamics and observability complexity that the learned low-level policy should absorb. The user approved a simpler normal-operation boundary before morphology-conditioned `π_L` training begins.
- Superseding decision: Normal `PolicyCommand` and normal QPID/QP shall contain neither per-contact wrench target/bias nor dock internal wrench target/bias. `PolicyCommand.contact_tracking_bias` is retained only for legacy archive/checkpoint readability and becomes deprecated/no-op on the new path. No `internal_wrench_bias` field is added.
- `π_H` decision: `π_H` contact assignments, modes, schedules, wrench requirements/bounds, centroidal targets, posture targets, and object targets remain schema-level outputs. Contact wrench requirements remain `π_L` context and inputs to feasibility, privileged training reward, safety, task-success evaluation, and logging; they are not direct normal-QPID tracking references.
- `π_L` decision: The new controller-facing `PolicyCommand` contract consists of desired centroidal pose/twist, additive centroidal wrench bias, absolute non-vectoring joint position/velocity targets, and bounded joint torque bias. Absolute joint targets supersede the older learned-residual interpretation of `joint_position_bias` / `joint_velocity_bias`; those legacy fields remain readable but are deprecated after migration. `joint_torque_bias` is an offset applied by the deterministic local servo and is not a final actuator command.
- Centroidal-frame decision: `desired_body_pose` / `desired_body_twist`, if the existing names are retained, mean the assembled morphology centroidal control frame, not the base module `fc` origin. The frame origin is the current morphology CoM and its orientation is the selected control-body orientation. Current joint/module state must update mass, CoM, inertia, rotor origins, and rotor axes each cycle. Existing base-`fc` tracking evidence must not be relabelled as true centroidal tracking.
- QP decision: Normal QP variables are limited to rotor thrust, thrust-vectoring variables/targets, and required slack. The objective realizes centroidal pose-derived wrench plus additive CoM wrench bias and optional controller-owned aggregate external-disturbance compensation. It does not allocate per-contact wrench, normal dock internal wrench, or generic non-vectoring joint torque. Thrust-vectoring joints remain allocator-owned because they determine rotor axes.
- Local-joint decision: Non-vectoring manipulation/dock joints use independent actuator-local position/velocity servo plus bounded offset torque: `tau_requested = Kp(q_target-q) + Kd(qdot_target-qdot) + tau_bias`. The controller retains final authority by validating control-mode support and applying position, velocity, rate, effort, torque-bias, finite-value, and safety limits before bridge conversion. `π_A` latch/detach overrides and deterministic current-position hold take precedence where applicable.
- Coupling decision: Version 1 treats non-vectoring joint motion quasi-statically. Joint rate/acceleration are bounded, the rigid-body model is rebuilt from measured joint state, and centroidal feedback rejects joint-reaction disturbances. High-speed articulated-body feed-forward and contact-aware full-body inverse dynamics remain later work.
- Contact-learning decision: Isaac per-contact force/impulse may be used as privileged reward/critic and diagnostics without becoming actor observation or QP target. Reward should prefer task-equivalent net object/environment effect, contact maintenance, wrench-bound/friction/safety satisfaction, stability and goal progress, while penalizing slip, penetration, contact break, excessive/unintended contact, saturation, and effort. Multi-contact training must not require one arbitrary pointwise force decomposition when multiple distributions are dynamically equivalent.
- Detach-only internal-wrench decision: Dock internal wrench is handled only by a dedicated detach unload/release mode. For a candidate edge whose follower-side component has no other external contact/load, estimate the cut-edge wrench from follower-subtree momentum balance after subtracting known actuator, gravity, and other known external wrenches, then transform it from follower CoM to the dock frame. Release requires external-contact-free evidence, estimator validity, relative pose/velocity bounds, separate force/torque thresholds, independent QP feasibility of both components, consecutive unload dwell, and post-release stability. Invalid or externally loaded cases fail closed.
- Compatibility decision: This is an approved design change, not an implementation claim. Existing `PolicyCommand`, `ControllerCommand`, controller, bridge, policies, checkpoints, and P4 artifacts remain on their recorded legacy contract until a schema-first migration and new acceptance evidence are completed. Old successful artifacts must not be reinterpreted as evidence for the new centroidal-only QPID / absolute-joint-target contract.
- Documentation impact: `A-MSRR_QP_PID_controller_design_spec_v0_1_ja.md` Section 14 is the normative controller supplement for this revision. It supersedes conflicting normal-operation contact-aware/internal-wrench and joint-bias wording in the earlier supplement and v0.4 Section 20 only within the approved scope above.
- Downstream impact: Order 3 morphology-conditioned `π_L` work must first implement/version the revised policy/controller/bridge contract, true centroidal state construction, local joint servo path, and privileged contact reward boundary. It must not train against the legacy contact-tracking-bias path and call that the revised design.

## 2026-07-11

### Random Morphology GUI Teleop Supplement

- Context: The completed Orders 1-2 path can sample a feasible connected morphology and validate deterministic floor takeoff-to-hover, but the source specification does not define a manual GUI free-flight inspection interface. The user requested a terminal-operated diagnostic before proceeding further.
- Entry decision: `scripts/random_morphology_teleop.py` samples exactly the requested 2-8 module count through `RandomFeasibleConnectedMorphologyDistribution`. Omitting `--seed` draws and prints a fresh random seed; supplying it reproduces the sampler path. The accepted graph is then passed unchanged to the existing graph-specific floor/takeoff environment and launched with Isaac Lab Kit visualization and real-time playback.
- Command decision: Terminal keys increment a desired centroidal body pose: `W/S`, `A/D`, and `R/F` translate; `J/L`, `I/K`, and `U/O` change yaw, pitch, and roll. Horizontal translation is relative to target yaw. `H`/space holds the measured pose, `0` restores the initial hover target, `P` prints the target, and `Q` exits. Input updates a `PolicyCommand` body-pose target only; it has no actuator authority.
- Safety decision: Default target increments are 0.05 m and 5 degrees. Roll/pitch are bounded to 30 degrees, target-position lead over measured pose is bounded to 0.50 m, and descent is clamped above the settled floor pose. During teleoperation the same QPID, rigid-body QP, actuator mapping, Isaac bridge, and all-cross-module unintended-contact monitors remain active. Any QP infeasibility, clipping, unresolved target, unintended contact, or raw contact-buffer saturation fails and stops the run.
- Learning boundary: This path does not instantiate `pi_L`, `pi_H`, a checkpoint, replay data, optimizer, or any training loop. It is intentionally a deterministic controller diagnostic over Order 1 morphology sampling and Order 2 flight control. Consequently it can assess whether a sampled structure is manually controllable under this baseline, but it cannot establish learned-policy performance or robustness.
- Reporting decision: The interactive probe prints a compact safety/exit summary instead of the full per-step Order 2 archive payload. Normal non-interactive probe reporting and the Order 2 typed archive remain unchanged.
- Scope boundary: This utility is not a new implementation order or acceptance gate. It does not implement dynamic docking, object contact, TaskSpec execution, or P4 full acceptance.

### P4 Full Orders 1-2 Random Morphology and Floor Takeoff Supplement

- Context: The v0.4 source specification defines graph-level morphology generation and feasibility checks, but it does not define a probability measure for randomized free-flight morphologies or a floor-settle-to-takeoff runtime gate. The user approved these method-level details and approved implementing Orders 1 and 2 together.
- Distribution decision: Order 1 produces task-independent `MorphologyGraph` samples containing 2-8 Holons, matching the v0.4 `RobotConstraints.max_modules=8` schema bound. Module count is sampled uniformly by a seeded deterministic generator. Module 0 is the canonical base. A connected tree is built constructively by adding one module at a time through uniformly sampled compatible, unoccupied pitch/yaw dock-port pairs. Joint angles are not sampled; module design poses are derived from the URDF reference-state dock frames. Canonical structural hashes reject duplicate topology/port assignments. The bounded rejection budget is 256 attempts because the conservative coarse-collision acceptance rate decreases for larger trees.
- Structural feasibility decision: Every accepted graph must be schema-valid, connected, acyclic, have exactly one base, use every port at most once, and use only mutually compatible port types.
- Collision decision: Order 1 performs deterministic coarse collision rejection between non-adjacent modules using URDF collision geometry transformed into design poses with a `0.03 m` safety margin. Directly dock-connected module pairs are excluded from this coarse inter-module rejection because their intended dock-surface contact is represented at this level. Exact mesh/contact physics remains an Order 2 runtime check.
- Flight feasibility decision: Order 1 must run the existing morphology-aware rigid-body model and primary QP allocator at the reference state. The older total-vertical-thrust proxy alone is insufficient. A graph is accepted only if gravity compensation is feasible without actuator clipping and with the configured thrust margin.
- Order 2 decision: Convert each accepted graph into the existing reset-time fixed graph URDF/USD, initialize the connected morphology at a geometry-derived floor-contact height with gravity enabled, settle it on the floor, ramp deterministic takeoff commands, and hold the requested hover pose through the existing QPID/controller-to-Isaac bridge. The runtime report must separate floor initialization, settle, liftoff, climb, final hover, QP, clipping, missing/unsupported actuator, and finite-state evidence.
- Exact contact decision: Enable articulation self-collision, filter all same-module rigid-body pairs, and resolve exactly one intended dock_mech body pair per `DockEdge` through `PortNode.port_local_id -> DockPortSpec.parent_link`. Only that selected cross-module pair is filtered. A reset-time PhysX collider query rejects initial unintended contact. During every settle/takeoff/hover step, one tensor view per module pair reads both the aggregate force matrix and non-aggregated `get_contact_data` force/point/normal/separation/count/start buffers. Capacity is eight contact patches per rigid-body pair. Acceptance requires exact graph-bound pair keys/counts, dock link/path bindings, aggregate/raw update counts, zero raw patch observations, zero force above `0.001 N`, zero buffer saturation, and zero adjacent-unintended/non-adjacent contact reports.
- Frame-semantics correction: `MorphologyGraph` module poses use Holon's PhysicalModel module frame (`fc`), while generated multi-module URDF fixed joints connect copied `root` links. Graph asset generation therefore uses `root_T_fc * fc0_T_fci * fc_T_root`; directly writing `fc0_T_fci` as a root joint transform is only accidentally correct for unrotated relative poses.
- Feasibility-layering clarification: The new `MorphologyFlightFeasibilityChecker` is a task-independent curriculum-support gate. Its reference-state QPs certify initial gravity support and the configured 15% reserve for this distribution; they do not replace the v0.4 general design-level `FeasibilityChecker` or introduce joint angles as `pi_D` design variables.
- Archive/provenance decision: A successful real run must reconstruct aligned typed `RuntimeObservation`, `PolicyCommand`, `ControllerCommand`, and `IsaacActuatorTargetRecord` sequences and persist one `EpisodeArchive`; a failed run clears its owned archive path. Acceptance binds the graph, morphology, physical model, URDF, collision-geometry content, loaded backend config, runtime config, phase sequence, and actuator application evidence. Provenance-bearing aggregate mass uses `math.fsum` so host and Isaac Python versions produce the same `PhysicalModel` hash.
- Validation result: Real Isaac floor settle/takeoff/hover smokes passed for one feasible random morphology at every supported size from 2 through 8 modules under the final selective-dock/all-cross-module raw-contact gate. The 8-module case monitored all 28 module pairs for 912 steps (25,536 aggregate/raw view updates) with a 140,000-patch buffer; the complete cohort reported zero raw patch observations, zero saturation, zero unintended cross-module force/contact violations, zero QP-infeasible/clipping/missing/unsupported/unresolved-actuator counts, and a 1.0 s hover hold. A 100-seed proposal audit found 25% and 16% conservative graph-feasibility acceptance for 7 and 8 modules, respectively; bounded rejection remains fail-closed. This is deterministic low-level flight evidence, not a statistical robustness, learning, object-task, dynamic-docking, or P4-full claim.
- Scope boundary: This combined work package establishes a randomized morphology curriculum source and deterministic floor takeoff-to-hover gate. It does not train `pi_L`/`pi_H`, implement dynamic module attachment, validate object contact, or claim P4 full completion.

## 2026-07-11

### P4 Full Order 0 Joint Actuator Performance Supplement

- Context: Before randomized morphology takeoff/hover and later P4 full work, vectoring and dock joint performance values must reflect the installed actuators instead of the previous generic `6.6 Nm / 3 rad/s` limits and hard-coded Isaac drive gains.
- Motor identity decision: Resolve the user-reported DYNAMIXEL `SC330-T181` to the official ROBOTIS model `XC330-T181-T`; ROBOTIS publishes no `SC330-T181` model. Use CubeMars `AK40-10 KV170` for all dock mechanism joints.
- Limit decision: Keep the mechanism-derived position bounds unchanged. Set vectoring URDF effort/velocity hard limits to the XC330-T181-T 11.1 V stall torque and no-load speed (`0.76 Nm`, `10.890854 rad/s`). Set dock URDF hard limits to the AK40-10 manufacturer peak torque and no-load speed (`4.1 Nm`, `45.553093 rad/s`). These peak/no-load limits are not continuous operating targets.
- Continuous-operation decision: Store vectoring continuous torque as the ROBOTIS US conservative estimate `0.152 Nm` (20% of 11.1 V stall torque), explicitly marked as an estimate because ROBOTIS does not publish a continuous rating. Store AK40-10 manufacturer rated torque/speed as `1.3 Nm` and `38.746309 rad/s`. Store protocol limits separately and cap safety at manufacturer peak torque rather than the AK MIT packet range.
- Config decision: Add `configs/robot/joint_actuators.yaml` as the joint actuator performance/config provenance source. The runtime URDF remains the source for mechanism axis/position and hard effort/velocity limits. `robot_model.yaml` references the actuator config; `PhysicalModel` metadata records the config hash, actuator assignments, full specifications, and per-dock mechanical metadata.
- Isaac decision (historical Order 0 baseline): At this point the implementation preserved `20.0` stiffness, `1.0` damping, and `3.0 rad/s` safe motion limit as simulation/control tuning, explicitly not manufacturer specifications. This baseline was later superseded for the Dock drive by the current `200/2` setting; the `3.0 rad/s` safe motion limit remains. The Isaac probe reads stiffness/damping from the joint actuator config unless CLI overrides are supplied.
- Compatibility impact: No persisted base schema change. `PhysicalModel.metadata`, `DockPortSpec.mechanical_limits`, and actuator-channel metadata receive additive actuator provenance/performance fields. Existing mechanism position limits and rotor thrust configuration are unchanged.
- Scope boundary: This completes P4 full Order 0 only. Random morphology generation, floor takeoff/hover, policy training, assembly control bridge, dynamic constraints, contact smoke, and full TaskSpec delivery are not started.

## 2026-07-10

### P4.3 Minimum Isaac Learning Bootstrap Supplement

- Scope: Implement the v0.4 Section 24.5.5 recommended order through P4.3a-d only: real-Isaac deterministic dataset collection, minimum learned `pi_L`, teacher-imitation `pi_H`, and outcome-conditioned `pi_D` fine-tuning. P4.3e joint fine-tuning, P4 full acceptance, and natural-contact grasp validation are not part of this work.
- P4.3a dataset decision: Collect four deterministic hard-feasible P2 designs for each of six randomized tasks (seeds 0-5), for 24 real Isaac episodes under `contact_model="kinematic_payload_coupled_attach_v1"`. The first two and additional two candidates are selected through a deterministic candidate window; failures are retained as outcome labels. Split by task, never episode, into four train tasks, one validation task, and one held-out task. The persisted dataset contains 24 rollout, 24 interaction-trajectory, 24 design-outcome, and 4,486 low-level records.
- Sampling decision: The raw probe is 200 Hz. Low-level shards retain every fourth row (effective 50 Hz), always retain the causal terminal transition and final command row, and aggregate all skipped rewards exactly once so episode return is invariant to striding. Raw Isaac rollout shards are provenance archives; the three learning-record kinds carry explicit design/high-level/low-level stage masks. Assembly masks remain false because P4.3 does not train `pi_A`.
- Reward alignment decision: Section 22.4 reward row `i` uses command/actuator row `i` and the forward state transition `observation[i] -> observation[i+1]`. The final command row has no post-observation, so state/terminal terms are marked unavailable and only command-derived terms are evaluated. The terminal reward belongs to row `N-2`, whose command produced `observation[N-1]`; no terminal transition is inferred for a one-observation episode. Missing contact/slip measurements contribute neutral zero with availability flags, never fabricated measurements.
- Success-label decision: P4.3 `task_success` requires the TaskSpec goal tolerance and a valid release. P4.2 bounded-carry success is preserved separately as `p4_2_bounded_carry_success`; it is not promoted to full task success. This prevents the existing 0.25 m kinematic payload-carry gate from becoming a P4/full-goal label.
- `pi_L` decision: Train a small MLP for bounded desired-body-twist, body-position, and residual-wrench deltas around `BaselineLowLevelPolicy`. The serialized PhysicalModel summary for this minimum run is the per-module `ModuleCapabilityToken` data already carried by `MorphologyGraph`; detailed link/joint tables are not learned features. Missing teacher residual means explicit cancellation of the deterministic baseline residual, yielding zero unrepresentable and zero clipped target values in the final dataset-supported bounds.
- `pi_L` runtime decision: A checkpoint may be injected only into the P4.2 command path. The original dynamic controller `InteractionKnot`, contact/anchor/joint/priority fields, controller/QP, and actuator bridge remain unchanged. Only the learned twist/body-position/residual subset is overlaid on the existing deterministic P4.2 `PolicyCommand`, with configurable trust-region blend `0.10`. Invalid checkpoint, OOD/non-finite features/output, or infeasible controller status uses deterministic fallback. The final real-Isaac online gate used the manifest-held-out task and executed 680 learned, non-zero overlays with fallback count 0, maximum overlay norm 0.10750, no drop/collision/controller-QP terminal, and a passing attach/transport/release rollout.
- `pi_H` decision: Use `ContactCandidateEncoder` tokens that preserve original candidate/group IDs, train candidate/group ranking plus a bounded timing residual from deterministic P4.2 teacher trajectories, decode a schema-valid `ContactWrenchTrajectory`, and always run deterministic assignment feasibility. Invalid/unknown/conflicting output falls back to `P4_2DeterministicGraspCarryPlanner`. This minimum ranker directly learns ContactCandidateSet/group features; InteractionEnvelope, MorphologyGraph, RuntimeObservation, and feasibility cache remain validated runtime/decode/safety context rather than learned token features. Evaluation is offline teacher-record decoding, not online Isaac deployment.
- `pi_D` decision: Reuse the P2.5 19-feature layout and compatible P2.5 checkpoint, regress real Isaac return/safety outcomes, and rank only after deterministic `FeasibilityChecker` filtering. Outcome labels are training targets and never inference features. Selection is hard filter -> learned ranking -> hard recheck -> deterministic `P2DesignPolicy` fallback. The four-candidate collection provides 20 train and 5 validation within-task ranking pairs; final validation pairwise ranking accuracy is 0.8.
- Artifact/gate decision: Persist checkpoint, metrics, loss/reward/evaluation files, dataset/config/checkpoint hashes, fallback metadata, online `pi_L` archive, and a P4.3 learning summary archive. Acceptance verifies all dataset shard bytes/counts/splits/stage masks/provenance, head-specific checkpoint metadata, config/dataset hashes, `pi_H` zero-fallback evaluation, `pi_D` within-task signal/validation ranking, and a manifest-held-out online `pi_L` checkpoint/archive. The online archive carries pre-overlay commands and actual controller knots so the gate recomputes non-learned-field preservation, body-orientation preservation, dynamic P4.2 guards, non-zero count, and blend norms per command rather than trusting producer booleans. Summary creation is forbidden unless acceptance passes and its source episode/archive belongs to the dataset manifest; the summary is sanitized to P4.3-only metrics/metadata and validates its own hashes after writing.
- Acceptance result: `dataset_passed=true`, `pi_l_passed=true`, `pi_h_passed=true`, `pi_d_passed=true`, deterministic fallbacks and no-mislabeling passed, and P4.3 minimum `completion_passed=true` for 24 real-Isaac source episodes.
- Remaining limitations: The source environment remains the P4.2 kinematic/slaved payload model with a pre-attach object hold and bounded carry semantics. The minimum `pi_H` exact decode is supplied by the learned group branch while its separate candidate-head top-k recall remains 0.0, so production contact-ranking quality is not claimed. This work does not establish natural contact, friction/slip quality, full TaskSpec delivery, production learned-policy quality, online `pi_H`/`pi_D`, joint fine-tuning, or P4 full completion. P4.4 natural-contact validation and later P4.3e/joint fine-tuning remain open.
- Compatibility impact: Adds new P4.3 dataset schemas/modules/artifacts. Existing P4.2 behavior is unchanged when no learned checkpoint is supplied; its environment/backend/probe interfaces gain optional checkpoint and blend-factor inputs only.

### P4.2 GUI Observation Launcher Supplement

- Context: P4.2 link-backed attach evidence must be visually inspectable in Isaac Lab, but the existing real-rollout CLI only forwarded the headless execution contract to the probe process.
- Decision: Add optional viewer-only runtime controls to the P4.2 execution boundary: `viewer="kit"`, `realtime_playback`, and `keep_open_after_rollout_s`. `scripts/run_p4_2_gui.sh` activates the existing `isaaclab3` environment and starts the ordinary P4.2 real rollout with `--viewer kit`, real-time playback, and a configurable post-rollout hold (`KEEP_OPEN_AFTER_ROLLOUT_S`, default `20`).
- Semantics: These controls are passed to Isaac Lab as `--viz kit`, `--realtime-playback`, and `--keep-open-after-smoke-s`. They do not change P2/P3 selection, frozen reset-time morphology generation, the deterministic `pi_H`/`pi_L` rollout, attachment conditions, payload coupling, event archives, or acceptance. A GUI observation is not separate success evidence; real P4.2 completion remains governed by the same archive and real-Isaac gates.
- Compatibility impact: No persisted schema or rollout contract change. The P4.2 backend and environment interfaces gain optional visual-observation parameters with headless defaults, so existing automated runs remain unchanged.

### P4.2 Link-Backed RobotAnchor Attach Gate Supplement

- Context: GUI inspection showed that the earlier P4.2 v1 rollout could satisfy the attach gate using a virtual module-frame RobotAnchor pose, even when the visible connector mechanism was not the apparent attaching structure.
- Decision: The real P4.2 Isaac probe now resolves `RobotAnchor.link_id` against the spawned Isaac articulation body names for the selected module. When the link is resolved, attach distance, relative velocity, object slaving, and root target computation use the link-backed anchor pose `link_pose_world * RobotAnchor.local_pose`, not only `ModuleRuntimeState.pose_world * RobotAnchor.local_pose`.
- Archive decision: `P4_2AttachEvent` now records `anchor_link_id`, `anchor_resolved_body_name`, `anchor_pose_source`, `anchor_link_pose_world`, `anchor_local_pose_in_link`, `anchor_link_twist_world`, and link-resolution metadata. Rollout artifacts also preserve bounded anchor debug samples so the selected anchor/contact-region distance can be inspected after GUI or headless runs.
- Acceptance decision: P4.2 fast acceptance and the real Isaac rollout gate now require at least one attach event with `anchor_pose_source="isaac_link"` and a resolved Isaac body name/link pose. A fallback module-state anchor can still be reported for diagnostics, but it cannot satisfy P4.2 completion.
- Compatibility impact: No persisted base schema change. The event/report/archive additions are defaulted, additive P4.2 runtime contract fields.
- Remaining limitation: This remains `contact_model="kinematic_payload_coupled_attach_v1"`. It proves that the kinematic attach gate is tied to a specified connector-link body and payload-coupled controller rollout, but it still does not prove natural contact grasping, connector mesh contact, frictional slip stability, or joint-closure grasping. `P4.4 / P4-natural-contact-grasp validation` remains the follow-on phase for those checks.

### P4.2 Real Isaac Payload-Carry Rollout v1 Supplement

- Context: The approved P4.2 v1 scope requires a real Isaac-backed deterministic attach / maintain / transport / release rollout under `contact_model="kinematic_payload_coupled_attach_v1"`, but it must not claim high-fidelity natural grasping, true fixed-joint dynamics, learned policy success, P4.3 bootstrap, or P4 full completion.
- Object initialization decision: The real probe uses the sampled P2 TaskSpec object pose/size/mass rather than static env defaults. During approach / pregrasp / attach-attempt, the object pose is held at the deterministic TaskSpec-derived pose so selected ContactCandidate contact regions do not drift under gravity before the attach gate. This hold is not an attach event and does not constrain the object to the robot; the object becomes anchor-relative only after the gated attach condition passes.
- Attach-gate decision: The real probe consumes the selected `ContactCandidateSet` and `ContactWrenchTrajectory`, maps selected ContactCandidate IDs to RobotAnchor IDs / ContactSlot IDs / ContactRegion IDs, and allows attach only when distance, relative velocity, assignment feasibility, non-fatal controller/QP status, attach phase timeout, and snap-distance gates pass. The real gate currently uses `attach_distance_threshold_m=0.16` and `attach_snap_distance_threshold_m=0.16` to match the observed Isaac/controller steady alignment tolerance for this graph-specific morphology.
- Transport/release decision: P4.2 v1 transport completion is a bounded payload-carry displacement gate (`transport_min_displacement_m=0.25`) in addition to the nominal release-region condition. This is deliberate: P4.2 v1 validates that the deterministic pipeline can attach, apply payload-coupled controller computation, move the payload, and release it in Isaac. It does not validate delivery to the full task object goal and must not be reported as P4 full completion.
- Payload-coupling decision: While attached, the object is kinematic/slaved and Isaac object dynamics are not counted as an additional load on the robot. Payload mass/inertia/CoM/gravity/effective wrench are added exactly once in the controller computation, and acceptance requires target wrench before/after payload plus achieved wrench/residual/QP/actuator metrics to prove the payload changed controller computation.
- Environment decision: In CPU-only IsaacLab environments, the backend falls back from configured CUDA to CPU, sets a writable Warp cache under `/tmp/amsrr_warp_cache`, and patches Warp's CPU pinned allocator when CUDA is unavailable. The ground plane uses an offline local cuboid instead of a remote default USD asset.
- Compatibility impact: No persisted base schema change. The changes are additive to P4.2 config/report/probe contracts and preserve fake fast-gate versus real Isaac completion separation.
- Follow-on work: `P4.4 / P4-natural-contact-grasp validation` remains required. That later phase should remove the pre-attach pose hold and kinematic/slaved object constraint, treat the object as a free rigid body in Isaac, evaluate actual contact/friction/contact maintenance from selected anchors/candidates/assignments, log slip/contact break/drop/unintended collision/excess penetration/contact force, and compare `kinematic_payload_coupled_attach_v1` against `natural_contact_grasp_v1`.

### P4.2 Payload-Coupled Kinematic Attach v1 Supplement

- Context: The earlier P4.2 contract used fixed-joint wording for the approximate object attach model. The approved P4.2 v1 scope is a deterministic payload-carry rollout that tests attach gating, controller/QP payload feasibility, transport, release, terminal failure logging, and archive generation without claiming natural grasp validation.
- Decision: Supersede the earlier fixed-joint wording with `contact_model="kinematic_payload_coupled_attach_v1"`. Under this model the object is treated as a kinematic/slaved object only after a gated attach event, and the robot/controller side receives the payload load explicitly. Payload mass, inertia, CoM offset, gravity wrench, target wrench before/after payload, achieved wrench, allocation residual, QP status, and clipped/missing/unsupported actuator metrics must be archived.
- Attach decision: Unconditional attach remains invalid. The object pose may be constrained to the anchor-relative pose only when selected ContactCandidate / RobotAnchor proximity, anchor-object distance, relative velocity, assignment feasibility, controller/QP non-fatal status, attach phase timeout, and snap-distance thresholds all pass. Attach events must archive selected candidate/anchor/slot/contact-region IDs, distance margins, relative velocity, snap distance, relative pose error, assignment feasibility result, and the contact model.
- Release decision: Release events must archive release time/phase, object pose, robot pose, intended-release flag, and post-release object pose error. Intended release is not counted as `object_drop`.
- Load-accounting decision: The standard P4.2 v1 path avoids double counting by keeping the object kinematic/slaved while applying the payload gravity/equivalent wrench exactly once in the controller/RigidBodyControlModel computation. A future true Isaac joint/constraint payload path must not also add the explicit payload wrench.
- Scope decision: `success_rate` for P4.2 v1 is deterministic payload-carry rollout success under `kinematic_payload_coupled_attach_v1`. It is not high-fidelity natural grasp success, not true fixed-joint dynamics success, not learned policy success, not P4.3 learning bootstrap, and not P4 full completion.
- Non-goals: P4.2 v1 does not validate natural contact-only grasping, friction-based slip stability, high-fidelity contact forces, learned policy performance, checkpoint training, reward curves, or P4 full completion.
- Follow-on work: Leave `P4.4 / P4-natural-contact-grasp validation` for a later phase. That phase should keep the object as a free rigid body in Isaac, evaluate actual robot anchor / contact candidate / selected assignment contacts with friction and contact maintenance, log slip/contact break/drop/unintended collision/penetration/contact force, and compare `kinematic_payload_coupled_attach_v1` against `natural_contact_grasp_v1`.
- Compatibility impact: No persisted base schema change. The supplement strengthens P4.2 event/archive/acceptance contracts and preserves the split fake-fast gate versus real Isaac rollout completion gate.

### P4.2 Split Acceptance Gate Supplement

- Context: P4.2 acceptance must separate a fast fake-backend/archive gate from a real Isaac rollout gate, and P4.2 completion must not pass without a real Isaac-backed deterministic rollout.
- Decision: Added `run_p4_2_acceptance`. The fast gate checks `EpisodeArchive` evidence for P2 selected design, P3 assembled morphology, deterministic P4.2 trajectory phases, selected contact candidates/assignments, per-step runtime/policy/controller/actuator logs, gated object attach events, graph-specific morphology reflection, and no-mislabeling fields.
- Real gate decision: The real gate requires the named rollout `p2_p3_deterministic_grasp_carry` to be attempted, passed, Isaac-backed, non-skipped, P2/P3-sourced, graph-reflected in asset/module placement/actuator mapping, terminal `success`, and accompanied by attach events plus per-step records. `completion_passed = fast_gate_passed and real_isaac_rollout_passed`.
- Runner/CLI decision: `P4_2DeterministicRolloutRunner` now includes the acceptance report. The P4.2 CLI exits successfully for `--real` only when completion passes; dry runs remain probes and do not claim completion.
- Compatibility impact: No persisted schema change. Fake-backend archives can satisfy the fast gate for unit testing, but they cannot satisfy completion because `isaac_backed=false`.

### P4.2 Runner and Archive Supplement

- Context: P4.2 must archive deterministic grasp/carry rollout evidence from a P2 selected design and P3 assembled morphology without treating fake-backend success as P4.2 completion.
- Decision: Added `P4_2DeterministicRolloutRunner`. It builds the deterministic P2/P3 case, samples contact candidates against the assembled morphology, generates the P4.2 phase-labeled `ContactWrenchTrajectory`, and passes the assembled `MorphologyGraph` to `P4_2IsaacEnv`.
- Archive decision: A non-skipped attempted rollout writes an `EpisodeArchive` containing the selected `DesignOutput`, feasibility result, P3 assembly plan, deterministic trajectory, runtime observations, policy commands, controller commands, actuator target records, phase transitions, attach events, contact candidate set, and selected contact assignments.
- Gate boundary: Fake-backend archives may store `p4_2_deterministic_rollout_passed=1` for fast-gate testing, but they also store `isaac_backed=0` and `real_isaac_completion_claim=0`. Split acceptance remains responsible for requiring a real Isaac-backed rollout before P4.2 completion.
- Scope impact: Runner/archive artifacts preserve `module_attach_detach_claim=false`, `dynamic_morphology_update_claim=false`, `p4_3_learning_bootstrap=false`, `checkpoint_claim=false`, and `reward_curve_training_claim=false`. This order does not claim P4.3 learning, high-fidelity natural grasp success, or P4 full completion.
- Compatibility impact: No persisted schema change. `EpisodeArchive.rollout_artifacts` carries P4.2-specific candidate/phase/attach evidence for later acceptance checks.

### P4.2 Graph-Specific Isaac Env/Probe Supplement

- Context: P4.2 must reflect the P2 selected design and P3 assembled morphology in the Isaac rollout asset, but the user clarified that graph-specific URDF/USD generation should not be interpreted as π_A dynamic construction or module attach/detach during the grasp/carry rollout.
- Decision: Added a graph-specific fixed morphology URDF generator that consumes a P3 assembled `MorphologyGraph` at reset time, prefixes module-local Holon bodies/joints, and connects module roots with fixed joints derived from the graph's dock-edge tree and module poses. During P4.2 rollout, robot morphology is frozen.
- Probe decision: The P4.2 Isaac probe now accepts `--p4-2-morphology-graph-json`, generates the reset-time graph asset, spawns robot/object/floor, and reports graph id, module ids, module poses, dock edge count, and actuator mapping graph id/channel count. The backend command intentionally does not use `--fixed-module-count` as P4.2 provenance.
- Attach boundary: This order does not perform object attach. Because selected contact candidates / RobotAnchors are not yet provided to the probe, the report records `p4_2_attach_gate_input_available=false`, `p4_2_unconditional_attach_allowed=false`, no attach events, and terminal `timeout_failure` or `controller_failure`. A reflected graph without an attach event cannot pass P4.2.
- Scope impact: P4.2 object attach/release events remain distinct from module attach/detach. The report carries `module_attach_detach_claim=false`, `dynamic_morphology_update_claim=false`, and `asset_generation_semantics="reset_time_fixed_morphology_not_pi_a_dynamic_construction"`. This order does not claim real P4.2 completion, P4.3 learning bootstrap, checkpoint training, reward-curve training, or P4 full completion.
- Compatibility impact: No persisted schema change. The new env/report parser and backend command are additive and prepare the later runner/archive and split-acceptance orders.

### P4.2 Deterministic Policy Phase Adaptation Supplement

- Context: P4.2 requires a deterministic grasp/carry rollout with explicit approach / pregrasp / attach / maintain / transport / release behavior, while preserving the rule that `π_L` emits `PolicyCommand` only and never final actuator commands.
- Decision: Added `P4_2DeterministicGraspCarryPlanner` as a P4.2-specific deterministic `π_H` planner. It reuses the existing selected assignment feasibility path and emits six phase-labeled knots: `approach`, `pregrasp_align`, `attach_attempt`, `attached_maintain`, `transport`, and `release`. Phase labels are stored in `InteractionKnot.guard_conditions` with `contact_model="kinematic_payload_coupled_attach_v1"`.
- Target decision: The P4.2 planner supplies centroidal body pose targets and object targets where appropriate. Approach/pregrasp/attach target the selected contact-candidate centroid with a conservative height offset; transport/release target the object goal while preserving the current body-object offset when runtime observation is available.
- π_L decision: `BaselineLowLevelPolicy` now translates P4.2 phase guards into numeric `PolicyCommand.priority_weights` such as `p4_2_phase_attach_attempt`, `attach_condition_gate`, `attached_object_tracking`, and `release_gate`. This keeps phase intent visible to the controller layer without adding actuator authority to `π_L`.
- Compatibility impact: No persisted schema change. The existing `GraspCarryBaselinePlanner` and baseline `π_L` behavior remain available. P4.2 real attach gating is still owned by the later Isaac env/runner order, not by this policy adaptation alone.

### P4.2 Deterministic Rollout Contract Supplement

- Context: P4.2 must not be treated as an extension of the P4.1 full-scene smoke. It is an Isaac-backed deterministic grasp/carry rollout with explicit approach / attach / maintain / transport / release behavior, while still avoiding any P4.3 learning or P4 full-completion claim.
- Decision: Added the P4.2 contract with rollout phase enum `reset`, `approach`, `pregrasp_align`, `attach_attempt`, `attached_maintain`, `transport`, `release`, `success`, `drop_failure`, `collision_failure`, `controller_failure`, and `timeout_failure`. Each phase has entry/exit condition text and configured timeout metadata.
- Attach decision: P4.2 v1 now uses `contact_model="kinematic_payload_coupled_attach_v1"`, but attach is gated. A successful attach event requires selected contact candidate / RobotAnchor proximity, relative velocity threshold, assignment feasibility, and controller/QP-safe status. Unconditional attach is invalid.
- Metrics decision: `success_rate` is scoped to Isaac-backed deterministic rollout success under the P4.2 v1 kinematic attach model. It is not high-fidelity natural grasp success, learned policy success, P4.3 learning bootstrap, or P4 full completion. `object_drop`, `hard_collision`, and `controller_qp_infeasible_terminal` definitions are contract fields; intended grasp contacts and kinematic attach contacts are excluded from hard-collision counting.
- Morphology decision: A successful P4.2 result must reflect the P2 selected `DesignOutput` and P3 assembled `MorphologyGraph` into the Isaac asset, module placement, and actuator mapping. Fixed 2-module or module-count-only provenance is insufficient for a passing P4.2 result.
- Acceptance impact: This order only defines the contract. Later P4.2 acceptance must split a fast fake-backend/archive gate from a real Isaac rollout gate, and completion must remain false without a real Isaac rollout.
- Compatibility impact: No persisted schema change. The contract is additive under `amsrr/simulation` and does not alter P4.1 or P4-control results.

## 2026-07-09

### P4.1 Split Acceptance Gate Supplement

- Context: The user explicitly required fast fake-backend gate and real Isaac smoke gate to be separated, and required P4.1 completion to remain false without a real Isaac smoke.
- Decision: Added `run_p4_1_acceptance`. The fast gate checks `EpisodeArchive` evidence: P2 selected design, P3 assembled morphology, not fixed-2-module-only, per-step runtime/controller/actuator/object-pose records, RuntimeObservation joint-state preservation, full-scene robot/object/floor evidence, and no-mislabeling fields.
- Real gate decision: The real gate requires the named smoke `p2_p3_full_scene_backend` to be attempted, passed, Isaac-backed, non-skipped, full-scene, P2/P3-sourced, and accompanied by passing joint-state metrics. `completion_passed = fast_gate_passed and real_isaac_smoke_passed`.
- Runner/CLI decision: `P4_1BackendSmokeRunner` now includes the acceptance report in its result. The P4.1 CLI exits successfully for `--real` only when completion passes; dry runs remain probes and do not claim completion.
- Compatibility impact: No persisted schema change. This is a gate/reporting layer only and does not claim object grasp/carry success, learned policy success, P4.2 rollout, or P4 full completion.

### P4.1 Runner and Per-Step Archive Supplement

- Context: P4.1 must include at least one case sourced from P2 selected `DesignOutput` and P3 assembled morphology, and it must not be completed using only a fixed 2-module morphology.
- Decision: Added `P4_1BackendSmokeRunner`. Before invoking the backend smoke, the runner builds the deterministic P2/P3 case using the P3 assembly config, P2 design distribution/policy, and `AssemblyRunner`. With the default seed, the selected accepted variant is `tri_anchor_support_grasp` and the P3 assembled morphology contains 3 modules.
- Archive decision: The runner writes one `EpisodeArchive` per attempted non-skipped backend smoke. The archive stores the selected `DesignOutput`, feasibility result, P3 assembly plan, per-step `RuntimeObservation`, per-step `ControllerCommand`, per-step actuator target records, and `p4_1_object_pose_history` in `rollout_artifacts`.
- Scope decision: Fake-backend unit tests can pass the archive/logging gate, but the archive explicitly records `isaac_backed=false` for fake reports and sets no-mislabeling fields for no object grasp/carry success, no learned policy claim, no P4.2 rollout claim, and no P4 full completion claim.
- Compatibility impact: No persisted schema change. The current backend command surface still maps the P3 assembled case to the Isaac probe via module count/provenance because arbitrary P2/P3 graph-to-USD generation is not yet part of the P4.1 backend smoke surface.

### P4.1 Backend Command and Probe Supplement

- Context: After the P4.1 contract was added, the backend needed a real Isaac command surface and a fake-backend-testable env boundary without treating P4.1 as another hover acceptance loop.
- Decision: Added `IsaacLabBackend.p4_1_full_scene_backend_smoke_command()` / `run_p4_1_full_scene_backend_smoke()` and `P4_1IsaacBackendEnv`. The Holon spawn probe now exposes `--p4-1-full-scene-backend-smoke`, object size/mass/pose flags, and a `--p4-1-uses-p2-p3` provenance flag.
- Probe decision: The P4.1 probe path spawns a robot articulation, a rigid box object, and the floor in the same stage, then runs a short controller/bridge step loop. It reports `p4_1_runtime_observations`, `p4_1_controller_commands`, `p4_1_actuator_target_records`, and `p4_1_object_pose_history` in addition to summary pass/fail fields.
- Validation decision: The P4.1 backend report parser requires the full-scene flags and reuses the Order 1 RuntimeObservation joint-state checker. Unit tests may use a fake backend report, but this does not satisfy P4.1 completion; later acceptance must still require a real Isaac smoke.
- Compatibility impact: No persisted schema change. The new keys are additive probe/backend report fields and do not claim object grasp/carry success, learned policy success, P4.2 rollout, or P4 full completion.

### P4.1 Full-Scene Backend Smoke Contract Supplement

- Context: The user approved starting P4.1 with the clarification that it is not another P4-control hover validation. P4.1 must smoke-test the Isaac backend as a full-scene path with robot, object, and floor in the same stage, and it must verify reset / step / RuntimeObservation / EpisodeArchive logging before P4.2 contact-rich rollout.
- Decision: Added a P4.1 backend smoke contract with required real smoke name `p2_p3_full_scene_backend`, config defaults, result records for per-step runtime observations, controller commands, actuator target records, and object pose history, plus a RuntimeObservation joint-state checker.
- RuntimeObservation decision: P4.1 observations must preserve module/root pose and twist plus joint positions and available joint velocities. Acceptance must verify vectoring/gimbal and dock mechanism joint position keys exist. For fixed/rigid cases, zero or nominal values are acceptable. For articulated cases, P4.1 must also prove the observed joint state can drive `RigidBodyControlModel` B(q)-style updates through nonzero model-update metrics.
- Scope decision: P4.1 completion must require a real Isaac smoke gate in addition to fake-backend/unit gates. P4.1 must not claim object grasp/carry success, learned policy success, P4.2 rollout, or P4 full completion.
- Compatibility impact: No persisted schema fields were changed. The new contract is additive under `amsrr/simulation` and uses existing `RuntimeObservation`, `ControllerCommand`, and `EpisodeArchive` schemas.

### P4-Control Articulated Assembly Correction Supplement

- Context: The prior articulated hover smoke only proved that dock mechanism joints could move while the rigid fixed-morphology assembly hovered. It did not prove that connected modules moved relative to each other as a multi-link system, because the generated fixed morphology connected module roots with a fixed joint.
- Decision: Added a separate articulated morphology URDF path for `--fixed-morphology-articulated-hover-smoke`. The child module root is attached to the selected parent connect dummy frame, so the parent dock mechanism joint moves the whole child module subtree in Isaac. The mating child-side dock mechanism is held at zero to keep the assembly representable as a URDF tree instead of a closed kinematic loop.
- Controller/observation decision: The articulated fixed smoke now builds `RuntimeObservation` module poses from actual Isaac body poses (`module_i__fc`) rather than static precomputed module poses. The QPID controller can emit module-scoped dock mechanism commands so only the structural parent-side dock joint is commanded.
- Validation decision: The articulated fixed smoke now requires real relative module pose motion and q-dependent control-model motion in addition to hover stability. The 20 s real Isaac smoke passed with relative module motion, rotor-origin/allocation-matrix changes, QP feasibility, and no bridge target failures.
- Compatibility impact: Non-articulated fixed hover and waypoint smokes still use the rigid fixed morphology path. This remains a low-level P4-control/P4a validation and does not claim dynamic docking, object grasp/carry, learned policies, closed-loop dock constraint physics, or P4 full completion.

### P4-Control Articulated Hover Smoke Supplement

- Context: The user asked whether flight with internal joint motion is part of the current P4-control/P4a validation scope. It is in scope as a low-level articulated-hover smoke, distinct from dynamic docking, policy learning, object grasp/carry, or P4 full completion.
- Decision: Added optional single-module and fixed-morphology articulated hover smokes. The smoke drives dock mechanism joints with a bounded sinusoidal `InteractionKnot.posture_target.joint_pos_target` while the QPID controller continues to receive hover pose/twist through `PolicyCommand`.
- Controller contract decision: `QPIDController` now passes dock mechanism posture references into `dock_mechanism_commands` after joint-limit clipping. Unspecified dock mechanism joints still hold nominal zero. This keeps actuator authority inside the controller/bridge layer; `PolicyCommand` still does not directly emit rotor thrusts, vectoring targets, or dock actuator targets.
- Validation decision: The new smokes require both hover stability and observed joint motion. Reports include selected dock joint ids, trajectory amplitude/period/warmup, max commanded/observed joint motion, max tracking error, bridge target health, and QP feasibility.
- Compatibility impact: No persisted schema change and no expansion of the existing P4-control full acceptance set by default. These smokes are optional low-level checks for q-dependent control updates and articulated flight behavior.

### Dock Frame Alignment Supplement

- Context: GUI inspection showed that the generated two-module fixed morphology placed the modules side-by-side rather than in the actual docked pose. The intended docked relation is that pitch/yaw connect point origins coincide and their axes are colinear, with the facing pitch/yaw dock geometry requiring x/y signs to be reversed while z remains aligned.
- Decision: Represent a pitch/yaw dock connection as a face-to-face port relation `Rz(pi)` between connect point frames. The source module to destination module relative pose is now computed as `src_port_pose * Rz(pi) * inverse(dst_port_pose)`.
- Physical model impact: `DockPortSpec.local_pose` now stores the connect dummy frame pose in the module/base frame, not merely the connect joint origin relative to its immediate dock mechanism parent link. This makes π_D-facing `PortNode.local_pose` geometrically meaningful for dock edge construction.
- π_D/morphology impact: Minimal and grasp/carry morphology builders now compute `DockEdge.relative_pose_src_to_dst` from the selected port pair and propagate module poses through the dock tree so the design morphology itself satisfies the same port alignment relation.
- P4-control/Isaac impact: The fixed-morphology URDF generator now places connected module roots using the selected compatible pitch/yaw port pair instead of a fixed x-spacing. The fixed-morphology controller smoke/probe/summary archive use the same module poses for runtime observations and logs.
- Compatibility impact: The CLI still accepts `--fixed-module-spacing-m`, but spacing is now only a fallback when no connect ports are available. Existing fixed-morphology USDs should be regenerated with `--force-convert`. This does not claim dynamic docking, object grasp/carry success, learned policies, or P4 full completion.

### P4-Control Hover Drift Fix Supplement

- Context: User observation and real Isaac diagnostics showed that the single-module hover drifted after roughly 5-10 s. Pseudoinverse allocation did not hover and increasing vectoring speed alone did not remove the drift. Inspection against `aerial_robot_base` and the Holon URDF found three geometry/control-model mismatches.
- Decision: Treat every rotor's local thrust direction as thrust-frame `+z`. The URDF rotor continuous joint axis sign is now used only to derive the reaction torque coefficient from `<m_f_rate>`, matching the reference controller and MuJoCo bridge convention. For Holon this yields alternating yaw torque coefficients `[-0.0172, 0.0172, -0.0172, 0.0172]` while all rotors push along local `+z`.
- Vectoring decision: The QP virtual lateral channel is now the actual positive gimbal-motion direction, `vectoring_joint_axis_body x virtual_z_axis_body`, instead of rotor-arm x. This matches finite-difference thrust-axis motion in the current URDF where the gimbal axis itself is rotor-arm x.
- Dock-hold decision: Dock mechanism position commands now hold nominal zero rather than the current observed angle. This prevents passive dock joints from being re-targeted to their drifted positions during free hover.
- Validation result: Real Isaac 20 s single-module hover using the primary `rigid_body_qp` path passed with no QP infeasible steps or clipping, final position error below 1 mm, and max position error about 2.3 cm.
- Compatibility impact: No schema change and no promotion of pseudoinverse allocation. This is a correction of URDF interpretation and internal control model geometry; P4-control still does not claim object grasp/carry, learned policies, fixed-morphology long hover, or P4 full completion.

### P4-Control Hover Drift Diagnostic Supplement

- Context: User GUI observation showed that the single-module hover can initially hold but drifts after roughly 5-10 s and crashes. The user requested a temporary pseudoinverse allocation trial and investigation of slow vectoring motion.
- Decision: Added an explicit debug-only `rigid_body_pseudoinverse` allocation mode and a probe-only `--vectoring-velocity-limit-rad-s` conversion override for gimbal/vectoring joint velocity limits. The primary P4-control path remains `rigid_body_qp`; the pseudoinverse path is not an acceptance or completion path.
- Diagnostic result: A 10 s no-stop QP hover reproduced the drift. The pseudoinverse path reduced lateral drift but remained infeasible/clipped throughout and lost altitude. Raising vectoring velocity from 3 to 20 rad/s was correctly reflected in Isaac's joint table but did not stabilize hover, and combining the higher velocity with higher gimbal stiffness/damping still produced late attitude loss. This suggests vectoring speed/servo tracking contributes to the transient behavior but is not the sole cause.
- Compatibility impact: No persisted schema change. The new CLI options are for controlled debugging/comparison only and do not alter P4-control acceptance thresholds or the requirement that QP allocation remains the primary path.

### P4-Control Holon USD Visual Mesh Resolution Supplement

- Context: The Kit GUI could open and `/World/Holon` existed in the stage, but only link frames/axes were visible. Inspection showed `assets/robots/holon/holon.urdf` references relative `mesh/*.STL` paths while the STL files live under `module_urdf/mesh`; the previous generated USD therefore contained articulation/link transforms but no visible mesh payload.
- Decision: The P4-control Holon probe now writes a conversion-only URDF copy with mesh references resolved to existing absolute STL paths before Isaac URDF-to-USD conversion. The same resolver is used for fixed-morphology URDF generation so single-module and rigid fixed-morphology GUI assets share the visual-mesh fix.
- Compatibility impact: This changes only the reproducible URDF-to-USD conversion input used by the probe. Runtime schemas, controller commands, QP allocation, acceptance thresholds, and physical success claims are unchanged. Previously generated USDs should be regenerated with `--force-convert` to pick up visible geometry.

### P4-Control GUI Observation Smoke Supplement

- Context: Real P4-control hover smokes ran correctly in Isaac, but IsaacLab 3 defaults to headless unless the Kit visualizer is explicitly requested with `--viz kit`, and smoke pass runs close the app immediately after the hold criterion is satisfied.
- Decision: Added GUI-observation-only probe options: `--realtime-playback` sleeps one physics `dt` per step for watchable playback, and `--keep-open-after-smoke-s` keeps the Kit app pumping for a fixed duration after the smoke finishes. These options do not change acceptance thresholds or controller behavior.
- Compatibility impact: P4-control acceptance remains based on the same real smoke pass/fail metrics. Long-duration hover is still not claimed; the GUI options are for inspection of the existing smoke behavior.

### P4-Control Smoke Summary Archive Supplement

- Context: After all three real Isaac low-level smokes could pass, the remaining split-acceptance gap was the fast gate requiring `EpisodeArchive` records with controller command, runtime observation, actuator target record, residual/clipping metrics, and explicit no-P4-full-completion labeling.
- Decision: Extended `P4ControlLowLevelRunner` so real-smoke runs automatically build one `EpisodeArchive` smoke summary per attempted non-skipped smoke when no external archives are supplied. Each summary archive records a free-flight smoke task, Holon morphology graph from the configured physical model, desired body pose policy command, summary `ControllerCommand` status, summary `RuntimeObservation`, and summary actuator target record metrics derived from the real smoke result.
- Scope decision: The archive type is `smoke_summary`. It preserves real smoke pass/fail, residual/clipping/missing/unsupported counts, target tolerances, and no-mislabeling fields, but it does not claim per-step actuator target replay, object grasp/carry, learned policy training, or P4 full completion.
- Runner impact: With real Isaac smokes passing and summary archives present, `run_p4_control_acceptance` can mark `fast_gate_passed`, `real_isaac_smoke_passed`, and `completion_passed` true for the P4-control low-level validation scope only.
- Compatibility impact: No persisted schema change. Existing `EpisodeArchive` fields are reused. Dry-run smoke results still produce no archives and cannot pass P4-control completion.

### P4-Control Fixed-Morphology Waypoint Smoke Supplement

- Context: After fixed-morphology hover passed, the remaining real-smoke runner gap was a fixed-morphology waypoint case that still uses the same rigid 2-module Holon assembly, QP controller path, and Isaac bridge application surface.
- Decision: Extended the Holon Isaac probe, backend, and `P4ControlIsaacEnv.run_smokes(dry_run=False)` with `--fixed-morphology-waypoint-smoke`. The waypoint smoke sends `PolicyCommand.desired_body_pose` through the controller each step, ramps the commanded target from the initial root pose to the final target, and reports `fixed_morphology_waypoint_*` metrics including ramp duration, tracking error, QP infeasible count, clipping count, and bridge target health.
- Waypoint-scope decision: The current default fixed-morphology waypoint smoke is intentionally small: world target position `(0.05, 0.0, 0.5)`, yaw `0.0 rad`, `0.1 s` target ramp, `0.20 m` position tolerance, `0.25 rad` attitude tolerance, and `1.0 s` hold. Larger exploratory lateral/z targets were not stable enough to treat as an acceptance default in this order.
- Runner decision: The real smoke runner now attempts all three P4-control low-level smokes: `single_module_hover`, `fixed_morphology_hover`, and `fixed_morphology_waypoint`. This can satisfy the real Isaac smoke side of the split acceptance gate when all three pass, but P4-control completion still also requires the fast archive/interface gate.
- Compatibility impact: This validates only a small real Isaac-backed waypoint smoke for the rigid fixed-morphology asset. It does not claim robust waypoint tracking, object grasp/carry, learned policies, archive completeness, P4-control completion, or P4 full completion.

### P4-Control Fixed-Morphology Hover Smoke Supplement

- Context: After the fixed assembly URDF generator was available, P4-control needed to validate that a rigid 2-module Holon asset can be converted, spawned, controlled, and reported through the same QP/controller/bridge path as the single-module smoke.
- Decision: Extended the Holon Isaac probe with `--fixed-morphology-hover-smoke`. The probe generates the rigid combined URDF on demand, converts it to USD, spawns it as one articulation, reconstructs per-module runtime observations from prefixed Isaac joint names, applies module-prefixed rotor/vectoring/dock actuator targets, and reports fixed-hover pass/fail metrics with the `fixed_morphology_hover_*` prefix.
- Runner decision: `P4ControlIsaacEnv.run_smokes(dry_run=False)` now runs both real `single_module_hover` and real `fixed_morphology_hover`; `fixed_morphology_waypoint` remains an explicit skipped smoke until the waypoint target path is implemented and validated.
- Compatibility impact: This validates fixed-morphology hover only. It does not validate waypoint tracking, archive completeness, object grasp/carry, learned policies, P4-control completion, or P4 full completion. The fixed morphology remains a pre-generated rigid assembly approximation and does not claim physical docking success.

### P4-Control Fixed-Morphology Assembly Asset Preparation Supplement

- Context: The user approved treating the first fixed-morphology smoke as a pre-generated rigid combined URDF/USD asset, with dock connection represented as a fixed joint equivalent. Before running Isaac, the repository needed a deterministic way to generate that asset and the controller needed correct multi-module gravity compensation.
- Decision: Added a fixed-morphology URDF generator that prefixes every copied Holon link/joint/transmission/gazebo reference with `module_<id>__`, rewrites relative mesh paths to absolute paths, and connects additional module roots to `module_0__root` with fixed joints at a configurable spacing. This produces a single URDF frame tree suitable for Isaac conversion while preserving per-module local names for controller-side mapping.
- Controller decision: Updated `QPIDController` default hover and body-target PID force generation to use `RigidBodyControlModel.total_mass_kg` when the rigid-body QP path is active. Single-module behavior is unchanged, but fixed-morphology hover now requests gravity compensation for the full assembled rigid body instead of one Holon module.
- Compatibility impact: This prepares the fixed-morphology smoke asset path but does not yet run fixed-morphology hover or waypoint in Isaac. The connected morphology is a rigid asset-level approximation for low-level controller validation, not a physical docking success claim and not P3/P4 object grasp/carry completion.

### P4-Control Single-Module Real Smoke Runner Supplement

- Context: The standalone Holon probe could pass the real single-module closed-loop hover smoke, but the P4-control runner still returned placeholder real-smoke failures for every required smoke. The next bounded step was to connect the completed single-module smoke into the runner without inventing fixed-morphology spawn/docking semantics.
- Decision: Extended `P4ControlIsaacEnv.run_smokes(dry_run=False)` so it executes only `single_module_hover` through `IsaacLabBackend.run_holon_single_module_hover_smoke`, parses the probe JSON, and converts the numeric pass/fail fields into `P4ControlSmokeResult.metrics`. The fixed-morphology hover and waypoint entries remain explicit skipped results with `skip_reason="real_isaac_execution_not_implemented"`.
- Backend decision: Added a JSON subprocess helper in `IsaacLabBackend` for real smoke commands and made the single-module runner force URDF-to-USD conversion by default. Backend-generated USD paths are taken from `IsaacLabBackendConfig`, allowing tests or manual runs to route generated artifacts to `/tmp` while the checked-in config remains unchanged.
- Compatibility impact: This is a runner/reporting integration for the already validated single-module smoke. It does not satisfy `real_isaac_smoke_passed` or `completion_passed`, because P4-control acceptance still requires fixed-morphology hover and fixed-morphology waypoint results. No object grasp/carry, learned policy, P4-control completion, or P4 full completion claim is introduced.

### P4-Control Single-Module Closed-Loop Hover Smoke Supplement

- Context: After the `PolicyCommand` PID target builder and controller-to-Isaac command path were validated, P4-control needed a real Isaac closed-loop smoke that repeatedly observes Holon state, recomputes the rigid-body QP allocation, and applies bridge-supported actuator targets rather than a single open-loop command.
- Decision: Extended `scripts/p4_control_holon_spawn_probe.py` and `IsaacLabBackend` with `--single-module-hover-smoke`. The smoke keeps one persistent `QPIDController(allocation_mode="rigid_body_qp")`, sends a direct hover `PolicyCommand.desired_body_pose` / `desired_body_twist` target each control step, converts the resulting command through `IsaacControllerBridge`, and applies rotor thrust plus vectoring/dock joint position targets in Isaac. The smoke reports final/max position and attitude errors, hold time, QP infeasible count, bridge clipping/missing/unsupported counts, and the last controller/bridge status. The default pass threshold remains the controller supplement's initial waypoint tolerance: `0.20 m`, `0.25 rad`, and `1.0 s` hold.
- Controller numerics decision: The attitude PID output is treated as desired body angular acceleration and converted to body torque using the current composite inertia from `RigidBodyControlModelBuilder`; it is not interpreted directly as Nm. Controller feasibility now separates a warning scale from hard infeasibility: residuals above `tracking_warning_residual_norm=1e-3` are reported as tracking warnings, while `unsupported_wrench_tolerance=1e-2` is the controller-local infeasibility cutoff used to tolerate small QP/back-conversion residuals seen in real closed-loop smoke.
- Isaac articulation decision: Dock mechanism joints use nonzero implicit hold stiffness/damping in the probe so passive dock joints do not drift to limits and destabilize a single-module hover. The closed-loop smoke can stop early once the configured hold duration is achieved, and records both requested and executed step counts.
- Compatibility impact: This validates a real Isaac-backed single-module hover smoke only. It does not validate fixed-morphology hover, waypoint tracking, object grasp/carry, learned `π_D` / `π_H` / `π_L`, P4-control completion, or P4 full completion. The QP allocator remains the primary path; `BoundedVerticalRotorAllocator` remains degraded fallback only.

### P4-Control PolicyCommand PID Target Builder Supplement

- Context: Before implementing closed-loop hover, the controller needed a deterministic path from direct P4-control hover/waypoint targets to desired body wrench while preserving the `π_L -> PolicyCommand -> controller/QP` responsibility boundary.
- Decision: Extended `QPIDController` so `PolicyCommand.desired_body_pose` and `PolicyCommand.desired_body_twist` activate a PID target builder. The builder uses the user-specified initial gains: xy `P=3.0/I=0.05/D=2.0`, z `P=5.0/I=1.0/D=2.5`, roll/pitch `P=22.0/I=1.0/D=14.0`, and yaw `P=5.0/I=1.0/D=4.0`. Position PID produces world-frame acceleration plus gravity compensation, then converts force to body frame. Attitude uses quaternion error in body frame, with roll/pitch and yaw gains applied by body axis. `PolicyCommand.residual_wrench_body` and any existing feedforward wrench are added to the PID wrench before QP allocation.
- Anti-windup decision: Integral state is held inside `QPIDController` and is committed only when the allocation is feasible and unclipped; infeasible or clipped allocation freezes the integral. No fixed acceleration/torque clipping values were introduced in this order; rotor/vectoring limits and infeasible status remain enforced by the QP/hard-check layer until explicit target-wrench saturation limits are specified.
- Compatibility impact: No persisted schema change. `InteractionKnot.centroidal_target.centroidal_wrench_preference` remains an upstream intent/reference path consumed by the existing bias builder for compatibility, but P4-control direct hover/waypoint should use `PolicyCommand.desired_body_pose`, `desired_body_twist`, and `residual_wrench_body` as the controller-facing path. This does not claim closed-loop hover or P4-control completion.

### P4-Control QP Feasibility Tuning Supplement

- Context: The real controller-to-Isaac command smoke proved that controller output and bridge targets could be applied in Isaac, but the controller still reported `qp_feasible=false`. Diagnostics showed the SLSQP solve succeeded; the default previous-command smoothing pulled the first hover allocation toward a zero-thrust previous command, and the post-solve hard check counted an effectively zero-thrust rotor's undefined vectoring angle as a rate-limit clip.
- Decision: Reduced `VirtualThrustQPAllocator` regularization and previous-command weights to `1e-8` so wrench tracking remains dominant on the primary P4-control allocation path. Set `QPIDControllerConfig.unsupported_wrench_tolerance` to `1e-5`, matching the small residual introduced by virtual-channel linearization/back-conversion while staying below the controller warning threshold. Added tolerance-aware clip detection and a zero-thrust vectoring deadband: when a vectoring rotor back-converts to effectively zero thrust, the vectoring joint target holds the current joint position instead of commanding an arbitrary angle at the rate-limit boundary.
- Compatibility impact: This changes only controller-local numerical tuning and hard-check interpretation. It does not change persisted schemas or the P4-control ownership boundary, does not promote pseudoinverse allocation, and does not claim closed-loop hover, object grasp/carry, policy learning, P4-control completion, or P4 full completion. The real Isaac controller-command smoke now reports QP feasible/ok with no clipping violations, but closed-loop smoke gates remain outstanding.

### P4-Control Controller-to-Isaac Command Smoke Supplement

- Context: After validating raw Isaac command APIs, P4-control needed a real-smoke artifact that starts from A-MSRR controller output rather than manually supplied force/joint arguments.
- Decision: Added `amsrr/simulation/p4_control_controller_smoke.py` to build a single-module morphology, runtime observation, `QPIDController(allocation_mode="rigid_body_qp")` command, and `IsaacControllerBridge` target record. Extended `scripts/p4_control_holon_spawn_probe.py` with `--controller-command-smoke`, which applies bridge rotor-thrust targets to matching `thrust_.*` bodies using rotor local thrust axes and bridge joint-position targets to matching gimbal/dock joints. The script reports controller command, bridge metrics, target clipping/missing/unsupported lists, and controller smoke metrics.
- Compatibility impact: This validates controller-to-bridge-to-Isaac command routing, not closed-loop hover. The current controller smoke still reports `controller_status.qp_feasible=false` because the QP path produces small residual/clipping violations under the present single-step hover request. Completion gates must continue to fail until a later order resolves controller feasibility and real hover/waypoint smoke.

### Holon Battery2 Inertial Correction Supplement

- Context: Real Isaac spawn and command probes consistently reported a PhysX warning that `/World/Holon/Geometry/root/main_body/battery2` had invalid inertia and negative mass fallback behavior. Inspection showed that both the runtime Holon URDF and reference xacro had `battery2` inertial data set to `mass=0` and all inertia components `0`.
- Decision: Set `battery2` inertial origin, mass, and inertia to match the symmetric `battery1` component in `assets/robots/holon/holon.urdf` and `module_urdf/holon.urdf.xacro`. Added a unit test that mesh-bearing runtime URDF links must have positive mass and positive diagonal inertia entries.
- Compatibility impact: This is a source asset correction needed for trustworthy Isaac physics. It changes Holon's aggregate mass/inertia relative to the previous zero-mass battery2 asset and removes the Isaac battery2 invalid-inertia warning after USD regeneration. It does not change schema contracts or claim hover/control completion.

### P4-Control Holon Isaac Command Probe Supplement

- Context: After Holon articulation spawn was validated, P4-control needed a minimal real Isaac check that the intended command surfaces are reachable: rotor-like external wrenches through the wrench composer and vectoring-like joint position targets through Isaac Lab articulation actuators.
- Decision: Extended `scripts/p4_control_holon_spawn_probe.py` with command-probe arguments. The probe can apply world-frame `+z` wrenches to `thrust_.*` bodies using `permanent_wrench_composer.set_forces_and_torques_index(is_global=True)` and command `gimbal.*` joints using `set_joint_position_target_index`. It reports thrust body ids/names, gimbal joint ids/names, robot mass/gravity, commanded force totals, root-state deltas, gimbal target/actual positions, and a tolerance-based `command_probe_passed` flag. Added a backend helper to build this command line.
- Compatibility impact: This is an Isaac API/actuator-path smoke only. The force is a global `+z` probe input, not the finalized rotor-axis thrust model, not QP closed-loop hover, and not a P4-control completion artifact. The probe confirms command routing and observation extraction; later work must connect `ControllerCommand` / `IsaacControllerBridge` records and implement the real single-module/fixed-morphology smoke gates.

### P4-Control Holon Isaac Spawn Probe Supplement

- Context: After validating URDF-to-USD conversion, the next P4-control Isaac smoke prerequisite was to verify that the generated Holon USD can be spawned as an Isaac Lab articulation and stepped in the approved `isaaclab3` / `isaaclab.sh -p` runtime.
- Decision: Added `scripts/p4_control_holon_spawn_probe.py` and a backend command helper for launching it. The probe converts the Holon URDF if needed, creates a fresh Isaac stage, spawns `/World/Holon` through `ArticulationCfg` / `UsdFileCfg`, steps a few physics frames, and emits a JSON summary with `spawn_passed`, body/joint names, root state, USD path, and Isaac-backed metadata. The CLI intentionally avoids the deprecated `--headless` flag; IsaacLab's default no-visualizer path is used for headless execution.
- Compatibility impact: This validates single-module Holon articulation spawn only. It does not apply rotor wrenches, command vectoring joints, run hover/waypoint control, assemble multi-module morphologies, claim object grasp/carry success, or satisfy the P4-control real smoke completion gate. Real probe logs currently include a PhysX warning for the `battery2` rigid body inertia/mass properties; this should be investigated before treating physical hover results as final.

### P4-Control Isaac URDF Conversion Probe Supplement

- Context: Before implementing real P4-control Isaac smoke execution, the Holon URDF import path needed validation in the approved `isaaclab3` / `isaaclab.sh -p` environment.
- Decision: Ran Isaac Lab's `scripts/tools/convert_urdf.py` against `assets/robots/holon/holon.urdf` in headless mode with output under `/tmp/amsrr_isaac_holon`. The converter completed successfully and generated `/tmp/amsrr_isaac_holon/holon/holon.usda` plus payload USD files. Updated A-MSRR config/default generated USD path to `artifacts/isaac/robots/holon/holon/holon.usda`, matching Isaac importer output structure when `generated_usd_dir` is `artifacts/isaac/robots/holon`.
- Compatibility impact: This validates the URDF-to-USD import path but still does not spawn or simulate Holon in a P4-control smoke. Generated USD artifacts were not committed; they remain reproducible from the source URDF and config.

### P4-Control Smoke Runner Configuration Supplement

- Context: After the P4-control fast/real acceptance split, the next implementation order needs configurable Isaac Lab environment settings and smoke scenario definitions before calling real Isaac APIs. The user approved using the existing `isaaclab3` micromamba environment, URDF-to-USD custom articulation as the initial Holon asset path, wrench-composer rotor force application, and the controller supplement's initial waypoint thresholds.
- Decision: Added `configs/env/isaac_lab.yaml`, `configs/training/p4_control_low_level.yaml`, `IsaacLabBackend`, `P4ControlIsaacEnv`, `P4ControlLowLevelRunner`, and `scripts/p4_control_smoke.py`. The runner supports `dry_run` by producing skipped smoke results and never marks completion. Backend availability probes are config-driven and lazy so normal unit tests do not require Isaac imports. The real smoke path now has deterministic scenario names and thresholds, but actual Isaac physics execution is intentionally left for the next order.
- Compatibility impact: This adds configuration and runner contracts only. It does not convert URDF to USD, spawn Holon, apply rotor forces in Isaac, or claim P4-control completion. Real Isaac smoke still requires executing the script through `micromamba activate isaaclab3` and `$ISAACLAB_PATH/isaaclab.sh -p` after the Isaac execution layer is implemented.

### P4-Control Acceptance Split Implementation Supplement

- Context: P4-control acceptance must distinguish fast pytest/interface/archive checks from real Isaac smoke, and P4-control completion must not pass when Isaac is unavailable or when only synthetic/unit checks were run.
- Decision: Added `run_p4_control_acceptance` with explicit `fast_gate_passed`, `real_isaac_smoke_passed`, and `completion_passed` fields. The fast gate checks that `EpisodeArchive` records include controller commands, runtime observations, actuator target records, residual/clipping metrics, and no P4 full-completion/physical-success claim. The real smoke gate requires three Isaac-backed smoke results: single-module hover, fixed-morphology hover, and fixed-morphology waypoint. `completion_passed` is true only when both gates pass.
- Compatibility impact: This is an acceptance/reporting contract only. It does not run Isaac, spawn Holon, or validate physical hover. Synthetic smoke results are accepted only as explicit report inputs for aggregation tests; actual P4-control completion still requires real Isaac smoke artifacts from later runner/backend work.

### P4-Control Actuator Mapping and Bridge Record Supplement

- Context: After the primary virtual-thrust QP allocator, P4-control needs a controller bridge boundary that can be unit-tested without Isaac while preserving the P4 requirement that `ControllerCommand` is converted to Isaac actuator targets and archived as actuator target records.
- Decision: Added controller-side `ActuatorMappingBuilder` and `IsaacControllerBridge`. The mapping extracts active module rotor thrust channels, vectoring joint position channels, dock mechanism position channels, and effort-limited joint channels from `MorphologyGraph` and `PhysicalModel` using deterministic global keys `module_<module_id>:<local_id>`, with single-module local-key aliases for backward compatibility. The bridge converts `ControllerCommand` dictionaries into `IsaacActuatorTargetRecord`, clips targets to mapped actuator limits, records missing/unsupported/clipped actuators, carries controller/QP residual status, and exposes a JSON-compatible dict for `EpisodeArchive.actuator_target_records`.
- Compatibility impact: This is a bridge contract and fast pytest gate only. It does not execute Isaac Lab, does not spawn robots, and does not claim P4-control smoke completion. Later Isaac backend code must consume these records or equivalent `ControllerCommand` data and then satisfy the real single-module/fixed-morphology smoke gates.

### P4-Control VirtualThrustQPAllocator Implementation Supplement

- Context: Agent I Order 2 implements the P4-control primary allocator after the user clarified that virtual rotor thrust directions may be fixed relative to the rotor-arm frame x/z directions and that thrust, joint, and rate limits should be included in QP constraints followed by hard check and clamp.
- Decision: Added `VirtualThrustQPAllocator` as the primary P4-control allocation path. Vectoring rotors are expanded to rotor-arm-fixed virtual x/z force channels, solved with a Python/SciPy quadratic objective plus actuator bounds and linearized vectoring angle/rate constraints, then back-converted to non-negative `rotor_thrusts_n` and absolute `vectoring_joint_targets`. Because Holon physical rotors include both `+z` and `-z` thrust axes, the virtual z channel is sign-aligned with each rotor's positive thrust direction while remaining fixed in the rotor-arm frame. The allocator recomputes achieved wrench after hard check/clamp and records residual, clipping, saturation, and primary/degraded metrics.
- Compatibility impact: `QPAllocationProblem` and `QPAllocationResult` gained backward-compatible optional fields for rigid-body model input, previous vectoring targets, control dt, vectoring outputs, and achieved wrench. `BoundedVerticalRotorAllocator` remains available but now marks itself as `degraded_fallback=1.0`; it is not the P4-control primary path. This does not claim Isaac smoke completion, object grasp/carry success, learned policy performance, or P4 full completion.

### P4-Control RigidBodyControlModel Implementation Supplement

- Context: Agent I Order 1 implemented the deterministic rigid-body model update required before QP allocation and Isaac bridge work. The v0.4 spec and controller supplement require link-level quasi-static inertia aggregation and per-step `q`-conditioned rotor geometry updates, while leaving the exact controller-local body-frame convention and multi-module actuator key convention to implementation.
- Decision: Added `amsrr/controllers/rigid_body_model.py` with controller-local `RigidBodyControlModel`, `RotorControlElement`, and `RigidBodyControlModelBuilder`. The body frame origin is the composite COM and its orientation is the current base/control module orientation. `center_of_mass_body` is therefore `(0, 0, 0)`, rotor origins are stored relative to the COM in body frame, and allocation columns use `r_i x F_i` with reaction torque coefficients. Multi-module actuator keys use deterministic `module_<module_id>:<local_id>` strings.
- Compatibility impact: No persisted schema was changed. The model is an internal controller contract exported from `amsrr.controllers`. It does not output actuator commands, does not replace QP allocation, and does not claim Isaac validation. The scalar rotor allocation matrix is the per-current-geometry basis that the later virtual-thrust-channel QP allocator will expand and back-convert.

### P4-Control Virtual Thrust Channel and Acceptance Split Supplement

- Context: Before starting P4-control / P4a implementation, the user clarified several controller-level requirements: the rigid-body model and allocation matrix must be rebuilt every control cycle from current joint positions, vectoring rotors should be expanded into virtual thrust channels inside the QP, pseudoinverse allocation must not be the main path, Isaac-unavailable tests may skip only unit smoke portions, and P4-control acceptance must distinguish fast pytest gates from real Isaac smoke gates.
- Decision: P4-control Agent I implementation will update composite inertia, COM, rotor origins, rotor axes, and allocation matrix `B(q)` from `RuntimeObservation.module_states[*].joint_positions` every control cycle. The primary allocator will be a QP path; `BoundedVerticalRotorAllocator` remains only a degraded fallback and must not be the source for P4-control completion. Vectoring rotor allocation may use virtual thrust channels internally, but controller output must be back-converted to `ControllerCommand.rotor_thrusts_n` and absolute `ControllerCommand.vectoring_joint_targets`, then re-evaluated for achieved wrench, residual, clipping, and unsupported-command metrics.
- Acceptance decision: P4-control acceptance is split into a fast pytest gate for deterministic/unit/interface/archive checks and a real Isaac smoke gate for actual single-module hover, fixed-morphology hover, and fixed-morphology waypoint tracking. Tests may skip Isaac-specific smoke when Isaac is unavailable, but P4-control completion must not pass without the real Isaac smoke gate.
- Compatibility impact: This is a controller implementation supplement. It preserves the v0.4 responsibility boundary: `π_L` outputs `PolicyCommand` only, controller/QP owns `ControllerCommand`, and the Isaac bridge owns final actuator target conversion. P4-control must not claim object grasp/carry success, π_D/π_H/π_L learning, P4.2 success, P4.3 learning bootstrap, or P4 full completion.

### Main Spec Cross-Reference to QP/PID Controller Supplement

- Context: After the P4-control QP/PID controller supplement was revised and its open questions were resolved, the user requested that the main design spec explicitly refer to the controller supplement at an appropriate location.
- Decision: Added references to `for_codex/A-MSRR_QP_PID_controller_design_spec_v0_1_ja.md` in v0.4 Section 20.1 and Section 24.5.2. The main spec now points controller implementers to the supplement for quasi-static rigid-body model updates, QP allocation, Isaac actuator target conversion, and P4-control acceptance details while preserving the `π_L` / controller responsibility boundary.
- Compatibility impact: Documentation only. No Python controller code, schema code, or acceptance code was changed.

### P4-Control QP/PID Controller Open Questions Resolved

- Context: The controller draft listed open questions for QP backend choice, Isaac thrust target semantics, vectoring command semantics, reaction torque handling, inertia aggregation fidelity, and waypoint tracking thresholds. The user answered that Python and libraries are acceptable initially, vectoring joints should use absolute position targets, reaction torque should be included, and accepted link-level quasi-static inertia aggregation plus initial waypoint thresholds.
- Decision: Updated `for_codex/A-MSRR_QP_PID_controller_design_spec_v0_1_ja.md` to replace the open-question section with implementation decisions. The spec now sets Python/library-based QP as the initial path, per-thruster thrust target as the primary Isaac-side representation with wrench-composer fallback for custom Holon articulation, absolute vectoring joint targets, reaction torque in QP, link-level quasi-static rigid-body aggregation, and configurable initial waypoint thresholds of 0.20 m position error, 0.25 rad attitude error, and 1.0 s hold duration.
- Compatibility impact: Documentation only. No Python controller code, schema code, or acceptance code was changed. Future implementation must still stop and ask before making incompatible assumptions if additional undefined controller details appear.

### P4-Control QP/PID Controller Spec Revision

- Context: The initial controller draft included a reference-implementation notes section and English-first wording. The user clarified that `aerial_robot_base` is temporary reference material only, that the controller spec should be Japanese-first like the main design spec, that allocation must be QP rather than pseudoinverse, and that assembled morphologies should be treated as a quasi-static single rigid body whose inertia and rotor origins are updated from joint angles every control cycle.
- Decision: Rewrote `for_codex/A-MSRR_QP_PID_controller_design_spec_v0_1_ja.md` as a Japanese controller-specific draft. Removed the reference-code section, made QP allocation normative, added quasi-static rigid-body model update requirements for assembled morphologies, clarified controller/bridge logging, and listed implementation-time open questions for solver choice, Isaac actuator semantics, vectoring command semantics, reaction torque handling, inertia aggregation, and waypoint thresholds.
- Compatibility impact: Documentation only. No Python controller code, schema code, or acceptance code was changed. The revised spec remains aligned with v0.4: `π_L` outputs `PolicyCommand` only, the controller owns `ControllerCommand`, and the bridge owns final Isaac actuator target conversion.

### P4-Control QP/PID Controller Design Spec Draft

- Context: The P4-control / P4a work will implement a near-complete low-level QP/PID controller, but v0.4 Section 20 intentionally leaves several controller details underspecified. The user provided `aerial_robot_base` as a temporary reference source and pointed to `gimbalrotor_controller.cpp` with `underactuate_=false`, `gimbal_calc_in_fc_=true`, and `gimbal_dof_=1` as the relevant branch.
- Decision: Added `for_codex/A-MSRR_QP_PID_controller_design_spec_v0_1_ja.md` as a controller-specific draft skeleton. It records the current controller ownership boundaries, reference-code reading notes, initial QP/PID allocation structure, Isaac bridge/logging expectations, proposed Agent I/J/K/L files, and open questions that must be settled before implementation.
- Compatibility impact: Documentation only. No Python controller code, schema code, or acceptance code was changed in this step. The draft preserves the v0.4 rule that `π_L` emits `PolicyCommand` only and final actuator authority remains in the controller / bridge layer.

## 2026-07-08

### P4.0 Simplified Full-Pipeline Implementation Supplement

- Context: v0.4 Section 24.5.1 defines P4.0 as simplified full-pipeline wiring, not P4 full completion. The implementation needed to connect P2 selected `DesignOutput`, P3 simplified assembly result, morphology-conditioned contact candidates, pi_H, pi_L, controller scaffolding, and `EpisodeArchive` logging without claiming Isaac-backed physical success.
- Decision: Added backward-compatible `EpisodeArchive` fields for runtime observations, actuator target records, rollout artifacts, and learning artifacts. Added a `SimplifiedGraspCarryEnv` path that accepts an external `DesignOutput` and optional assembled `MorphologyGraph`, bypassing `FixedSimpleDesignPolicy` on the P4.0 path. Added `P4_0FullPipelineRunner`, `configs/training/p4_0_grasp_carry.yaml`, unit/archive/no-mislabeling tests, and `run_p4_0_acceptance`.
- Logging/no-mislabeling decision: P4.0 archives record `rollout_artifacts` with `phase="P4.0"`, `backend="simplified"`, `is_p4_full_completion=False`, `isaac_backed=False`, and `physical_success_claim=False`. The acceptance report includes an explicit backend note that P4.0 metrics are simplified backend indicators and not Isaac-backed physical success rates.
- Compatibility impact: Existing P1/P2/P3 archives deserialize because the new archive fields have default empty values. P4.0 still does not implement Isaac Lab backend, controller bridge / actuator mapping, actuator target execution, P4-control, P4.1/P4.2, P4.3 learning bootstrap, or P4 full acceptance.

### P4.3 Learning Target Clarification

- Context: After the P4 Isaac-backed completion clarification, the P4.3 learning bootstrap text could still be read as focusing only on π_L or residual controller learning.
- Decision: Updated the source design spec so P4.3 explicitly includes three staged learning targets: π_L / residual controller learning, π_H contact / trajectory policy learning, and π_D outcome-conditioned design scorer / selector fine-tuning. Added the recommended P4.3a-P4.3e order, expanded P4 full acceptance learning artifacts for all three policy families, and updated the P4 Mermaid diagram so the training loop points back to π_D, π_H, and π_L with their separate responsibilities.
- Compatibility impact: No implementation files were changed in this design-only task. Deterministic `P2DesignPolicy`, deterministic π_H / π_L fallbacks, and `FeasibilityChecker` hard safety remain required; learned feasibility heads must not replace deterministic safety gates.

### P4 Isaac-Backed Completion Clarification

- Context: The previous v0.4 P4 text could be read as treating simplified full-pipeline wiring as P4 completion. The user provided a P4 design revision instruction requiring the source spec to distinguish simplified integration from Isaac-backed full grasp/carry completion.
- Decision: Updated the source design spec to split P4 into `P4.0`, `P4-control / P4a`, `P4.1`, `P4.2`, `P4.3`, and P4 full completion. P4.0 is now explicitly a simplified full-pipeline integration stage and must not be called P4 complete. P4 full completion now requires Isaac Lab rollout, low-level flight validation, controller bridge / actuator mapping, Isaac actuator target execution, minimum learning run, checkpoint, metrics, reward curve, and rollout archive.
- Additional clarification: P3 assembly success is now explicitly documented as simplified graph/state integration success rather than physical docking success. P4 full completion requires a bridge from π_A docking/detach/separation steps to controller targets and Isaac-backed execution results.
- Compatibility impact: No implementation files were changed in this design-only task. Future P4 implementation must treat existing `QPIDController` / `QPAllocator` as simplified scaffolding until the Isaac controller bridge and actuator mapping are implemented.

### P3 Assembly Runner Core Supplement

- Context: v0.4 Section 17 defines `AssemblyPlan`, `AssemblyStep`, and `ConstructionState`, and an earlier Agent G scaffold produced deterministic graph-edit plans, but P3 acceptance needs an executable deterministic runner that advances construction state and checks that the physical graph matches the target graph.
- Decision: Added an Agent G `AssemblyRunner` core. It executes a planned sequence through an `AssemblyExecutorInterface`, records per-step `AssemblyExecutionResult` objects, updates `ConstructionState` on successful `verify_attach` steps when the executor does not provide an updated state, and computes graph/state consistency metrics for modules, dock edges, and occupied target ports.
- Compatibility impact: This remains deterministic π_A scaffolding. It does not introduce learned assembly, motion planning, Isaac execution, QP/PID control, or physical docking verification. Later simplified/Isaac executors can provide richer `updated_state` values behind the same executor interface.

### P3 Simplified Assembly Executor Supplement

- Context: v0.4 Section 24.4 requires P3 assembly execution in simplified sim, but does not prescribe a dependency-free executor backend for the existing `AssemblyExecutorInterface`.
- Decision: Added an Agent G `SimplifiedAssemblyExecutor` that deterministically succeeds assembly steps by default, optionally updates construction state on successful `verify_attach` when a target graph is available, and supports explicit failure injection by step id or step type for later retry/abort acceptance probes.
- Compatibility impact: The executor is a smoke backend only. It does not model docking dynamics, path planning, contact physics, controller allocation, Isaac execution, or learned assembly control.

### P3 Retry/Abort State-Machine Supplement

- Context: v0.4 Section 24.4 requires retry/abort paths to be tested, while Section 17 only lists `retry` and `abort` as valid `AssemblyStep.step_type` values without prescribing a state-machine implementation.
- Decision: Extended Agent G `AssemblyRunner` with deterministic retry/abort handling. On a failed planned step, the runner emits a synthetic `retry` step up to `max_retries_per_step`; if the planned step still fails, it emits a synthetic `abort` step and returns an unsuccessful `AssemblyRunReport` with retry/abort counts and executed step types.
- Compatibility impact: Retry/abort remains deterministic scaffolding and does not imply learned assembly recovery, motion replanning, physical docking verification, or controller/QP feasibility.

### P3 Assembly Evaluation Runner Supplement

- Context: v0.4 Section 24.4 defines P3 assembly acceptance but does not prescribe a concrete evaluation runner or archive metrics for the deterministic assembly integration phase.
- Decision: Added Agent K `P3AssemblyEvaluationRunner`. It reuses the P2 grasp/carry task distribution and deterministic `P2DesignPolicy`, selects a feasible target `MorphologyGraph`, executes it through `AssemblyRunner` and `SimplifiedAssemblyExecutor`, stores the `AssemblyPlan` in `EpisodeArchive.assembly_plan`, and records assembly success/state/retry/abort metrics.
- Compatibility impact: This is a simplified assembly integration runner only. It does not execute π_H, π_L, QP/PID, actuator commands, Isaac, or learned assembly control.

### P3 Acceptance Gate Supplement

- Context: v0.4 Section 24.4 defines P3 acceptance criteria but does not prescribe a concrete acceptance report schema or how to exercise retry/abort paths when the normal simplified executor succeeds deterministically.
- Decision: Added Agent L `run_p3_acceptance`. It runs the P3 assembly evaluation runner, checks `assembly_success_rate >= 70%`, verifies successful archives have construction-state physical graph consistency, and runs explicit transient-failure retry and persistent-failure abort probes with the simplified executor.
- Compatibility impact: This is an acceptance harness for deterministic simplified assembly integration. It does not run Isaac, π_H, π_L, QP/PID, actuator commands, or learned assembly control.

### π_D Joint-Angle Non-Design Clarification

- Context: The v0.4 design text could be misread as treating `ModuleNode.pose_in_design_frame` or `DockEdge.relative_pose_src_to_dst` as continuous design variables for π_D, even though A-MSRR module joints are movable and their instantaneous angles belong to planning/control/runtime state rather than structure design.
- Decision: Clarified the source design spec directly. π_D designs graph-level structure only: module count, connection topology, docking port pairs, base module, module roles, RobotAnchors, slot-anchor priors, control groups, and graph-level metadata. π_D must not output movable joint angles, runtime module relative poses, pose trajectories, actuator commands, rotor thrust, joint torque, or vectoring joint targets.
- Compatibility impact: No code change is required at this point. Existing design-level feasibility checks use graph/capability/coverage/margin necessary conditions and do not score a single nominal joint configuration. Existing pose fields remain nominal/canonical metadata for assembly reference, visualization, coarse precheck, debugging, and simulator initialization.

### P2.5 Learning Bootstrap Supplement

- Context: P2.5 inspection existed as a human/debugging phase, but did not yet create a supervised learning bootstrap from deterministic P2 candidate labels. The user requested a minimal learned π_D scorer and learned feasibility head trained from `P2DesignPolicy.evaluate_candidates()` / deterministic `FeasibilityChecker` outputs, without full RL and without replacing the production path.
- Decision: Added a P2.5 candidate dataset builder that exports all accepted/rejected/selected candidates across multiple sampled grasp/carry tasks, including deterministic features, labels, design scores, violation codes, margins, and train/val ID splits. Added two lightweight MLP training loops: one for teacher-selected π_D candidate classification and one for deterministic feasible/infeasible classification. Checkpoints, metrics, and loss curves are saved under `outputs/p2_5/training`.
- Compatibility impact: The learned models are auxiliary bootstrap artifacts only. They are not used by `P2DesignPolicy`, `FeasibilityChecker`, P2 acceptance, P2.5 inspection, Isaac, π_H, π_L, QP/PID, or actuator command execution. Deterministic `P2DesignPolicy` and `FeasibilityChecker` remain the source of truth for design selection and hard safety checks.

### P2.5 Inspection and Candidate Trace Export Supplement

- Context: The user accepted P2 as complete for the current v0.4 Section 24.3 design-level gate, but requested a pre-P3 inspection phase so humans can inspect morphology variants and all P2DesignPolicy candidate evaluations, not only the selected design.
- Decision: Added P2.5 as an additional inspection/debugging phase, not a replacement for P2 completion. It adds SVG morphology graph/layout visualization for all four grasp/carry variants, JSONL/CSV export of per-candidate evaluation traces including an explicit closed-loop invalid rejection probe, a markdown inspection report, and a P2.5 acceptance gate.
- Compatibility impact: P2.5 uses existing `P2DesignPolicy`, P2 design runner/config, `FeasibilityChecker`, `DesignOutput`, `FeasibilityResult`, and `EpisodeArchive`-compatible labels/margins. The inspection path does not run Isaac, π_H, π_L, QP/PID, or actuator commands.

### P2 Completion Gate Supplement

- Context: v0.4 Section 24.3 defines the P2 acceptance criteria, but it does not define a phase-completion report that downstream work can use to distinguish "acceptance function exists" from "P2 milestone is complete."
- Decision: Added Agent L `run_p2_completion` as a thin completion wrapper over `run_p2_acceptance`. It emits a `P2CompletionReport` with explicit boolean completion checks for the Section 24.3 gates: valid design rate, required slot coverage for accepted designs, closed-loop invalid rejection, and feasibility label storage.
- Compatibility impact: This adds acceptance-side report dataclasses only and does not change persisted schemas. P2 completion remains design-level; π_H, π_L, controller allocation, actuator commands, Isaac execution, and later P3/P4 behavior are outside this gate.

### P2 Acceptance Gate Supplement

- Context: v0.4 Section 24.3 defines P2 acceptance criteria but does not prescribe a concrete report schema or how to probe `closed_loop_invalid designs rejected` when the normal P2 design distribution only generates tree morphologies.
- Decision: Added Agent L `amsrr.acceptance.p2_acceptance` as a mechanical P2 acceptance gate. It runs the configured P2 design runner, checks `valid_design_rate`, verifies required slot coverage for accepted archived designs, validates feasibility label storage across `EpisodeArchive.feasibility_result`, and synthesizes an explicit closed-loop invalid design to verify `F_CLOSED_LOOP_REJECT_V1` rejection and labels.
- Compatibility impact: This is an acceptance harness and does not change persisted schemas. It does not run π_H, π_L, controller allocation, actuator commands, Isaac, or learned training.

### P2 Design Evaluation Runner Supplement

- Context: v0.4 Section 24.3 requires P2 design evaluation over diverse grasp/carry tasks and requires feasibility labels to be stored, but it does not prescribe a concrete runner/config format for executing the TaskSpec -> Geometry -> IRG -> Envelope -> π_D -> FeasibilityChecker path before learned training.
- Decision: Added Agent K `P2DesignEvaluationRunner` and `P2GraspCarryDesignDistribution`. The runner samples randomized object grasp/carry TaskSpecs, builds geometry descriptors through `IRGBuilder.build_with_scene_graph()`, extracts the `InteractionEnvelope`, evaluates `P2DesignPolicy` candidates, and stores the selected `DesignOutput` plus selected `FeasibilityResult` in `EpisodeArchive` JSONL records.
- Compatibility impact: This is an evaluation/dataset scaffold, not a learned training loop, not Isaac execution, and not a controller/actuator-command runner. It uses existing `EpisodeArchive.feasibility_result`, `FeasibilityResult.proxy_scores`, and `FeasibilityResult.margins` fields without changing persisted schemas.

### P2 Design Policy Candidate Selection Scaffold Supplement

- Context: v0.4 Section 15 defines π_D action/candidate scaffolding and Section 24.3 requires P2 design evaluation, but it does not prescribe a deterministic baseline for enumerating multiple candidate designs, separating accepted/rejected candidates, or selecting among accepted candidates before learned π_D training.
- Decision: Added Agent E `P2DesignPolicy` as a deterministic π_D scaffold. It enumerates grasp/carry morphology variants, evaluates each `DesignOutput` with the deterministic `FeasibilityChecker`, splits candidates into accepted and rejected sets, computes a deterministic soft score from feasibility margins plus small support/complexity/variant priors, and returns the highest-scoring accepted design. If no candidate is accepted, it returns the highest-scoring rejected candidate for debugging/dataset labeling.
- Compatibility impact: This is not a learned π_D head and does not output actuator commands. Selection metadata is stored as float entries in `DesignOutput.design_scores` (`p2_design_policy_*`) without changing persisted schemas.

### P2 Grasp-Carry Morphology Variant Builder Supplement

- Context: v0.4 Section 15.4 names design teacher variants (`chain_grasp`, `symmetric_two_anchor_grasp`, `tri_anchor_support_grasp`, `central_base_plus_two_grasp_arms`) but does not prescribe exact module poses, tree topology, control groups, or RobotAnchor placement for each variant.
- Decision: Added Agent E `GraspCarryMorphologyVariantBuilder` as a deterministic P2 scaffold for object grasp/carry. The four variants now produce distinct connected-tree `MorphologyGraph` layouts: a linear chain, a central base with two direct grasp arms, a tri-anchor support/grasp frame with an optional support anchor on the base, and a central base with two two-link grasp arms. `DeterministicDesignTeacher` now routes object grasp/carry variants through this builder instead of merely annotating the minimal seed morphology.
- Compatibility impact: This does not change persisted schemas and does not claim the variants are optimized or learned designs. It gives π_D training/evaluation a finite deterministic set of distinct `DesignOutput` demonstrations while preserving existing `DesignPolicyContext -> DesignOutput`, `ContactSlotID -> RobotAnchorID`, and FeasibilityChecker boundaries.

### P2 FeasibilityChecker Acceptance Labels Supplement

- Context: v0.4 Section 24.3 requires P2 acceptance to measure valid design rate, required slot coverage for accepted designs, closed-loop invalid rejection, and stored feasibility labels, but `FeasibilityResult` does not define a dedicated label field.
- Decision: Strengthened Agent F design-level `FeasibilityChecker` without changing schemas. It now records stable P2 count/ratio/margin keys in `FeasibilityResult.margins` for required slot coverage, anchor capability coverage, coarse reachability, port conflicts, closed-loop rejection, thrust margin, and payload margin. It also stores deterministic 0/1 label scores in `proxy_scores` using `L_FEASIBLE`, `L_HARD_VIOLATION`, and `L_<hard_check_code>` keys.
- Compatibility impact: Existing hard violation codes and `FeasibilityResult` fields remain unchanged. The `L_...` entries are acceptance/dataset labels stored in the available float map, not learned proxy estimates and not replacements for deterministic hard checks.

### P1 Acceptance Gate Supplement

- Context: v0.4 Section 24.2 defines P1 acceptance criteria but does not prescribe a concrete test harness or result schema for recording the pass/fail gate.
- Decision: Added Agent L `amsrr.acceptance.p1_acceptance` as a lightweight acceptance harness. It runs the configured P1 simplified runner over 1000 randomized training-distribution episodes, checks the minimum success rate and zero crash criteria, samples valid randomized objects for non-empty contact candidates, and returns a serializable `P1AcceptanceReport`.
- Compatibility impact: This closes the P1 gate against the interface-backed simplified backend. It does not invoke Isaac Lab and does not claim high-fidelity physics validation; later Agent J simulator backends can be tested behind the same policy/controller boundaries.

### P1 Task Distribution Runner and EpisodeArchive Supplement

- Context: v0.4 requires P1 object grasp/carry randomization over object size, mass, friction, and target pose, plus EpisodeArchive logging and reproducibility metadata. It does not prescribe exact config field names or a JSON storage format.
- Decision: Added Agent K config-driven P1 distribution and runner modules. `P1TaskDistributionConfig` randomizes box primitive size, object mass/friction, initial object pose, and object target pose. `P1SimplifiedRunner` runs the simplified env over sampled tasks, computes batch metrics, and emits `EpisodeArchive` records.
- Logging supplement: Added `EpisodeArchive` with the v0.4 fields plus a `reproducibility` map for source hash, random seed, simulator version, URDF hash, and thrust model hash. JSONL helpers write/read archive sequences for lightweight P1 dataset/debug logs.
- Compatibility impact: This is dataset/logging scaffolding for the simplified env, not a training loop and not an Isaac data recorder. The randomization config can be expanded later for wind, sensor noise, thrust scale error, contact break thresholds, and additional object shapes.

### P1 Simplified Grasp-Carry Simulation Env Supplement

- Context: v0.4 requires simulator-specific code to remain behind interfaces, allows simplified contact in Version 1, and defines P1 acceptance as validating the GeometryProcessor/IRGBuilder/pi_H/pi_L/controller loop with no schema/checker crashes over 1000 episodes.
- Decision: Added an Agent JP1 `SimplifiedGraspCarryEnv` under `amsrr/simulation`. It implements the `reset`, `step`, and `get_runtime_observation` boundary, builds the existing TaskSpec -> IRG -> Envelope -> fixed/simple DesignOutput -> ContactCandidateSet -> pi_H trajectory pipeline, runs `BaselineLowLevelPolicy` and `QPIDController`, and uses a kinematic/fixed-joint approximation after attach to move the object toward active object targets.
- Compatibility impact: This is not an Isaac Lab environment and does not model high-fidelity contact dynamics. It is an interface-backed smoke backend for P1 crash-free validation before Isaac integration. Later Isaac environments can implement the same `SimulationEnvBase` boundary.

### P1 pi_L + QP/PID Controller Interface Supplement

- Context: v0.4 defines the `PolicyCommand` and `ControllerCommand` schemas and the pi_H -> pi_L -> QP/PID flow, but does not prescribe a deterministic P1 low-level baseline policy or dependency-free controller backend.
- Decision: Added an Agent I `BaselineLowLevelPolicy` that consumes `RuntimeObservation`, `MorphologyGraph`, `PhysicalModel`, `ContactWrenchTrajectory`, and an optional active `InteractionKnot`. It selects the active knot by runtime time when not supplied, emits zero anchor pose offsets for active assignments, small contact-tracking biases derived from active assignment wrench targets, and a clipped object pose/velocity residual wrench intent when the active knot contains object targets.
- Controller supplement: Added `ControllerContext`, `ControllerBase`, `QPAllocationProblem`, `QPAllocationResult`, `QPAllocatorInterface`, `BoundedVerticalRotorAllocator`, and `QPIDController`. The P1 allocator solves only bounded vertical thrust allocation, reports unsupported lateral/torque wrench residuals, clips vectoring joints to URDF limits, and applies a PD proxy for non-vectoring joint torque references.
- Compatibility impact: The policy emits only `PolicyCommand` intent and never rotor thrusts, vectoring targets, joint torques, or dock commands. The controller layer owns `ControllerCommand` output. Exact multi-axis/vectoring/contact QP remains a future allocator backend behind `QPAllocatorInterface`.

### P1 pi_H Grasp-Carry Baseline Planner Supplement

- Context: v0.4 defines the `ContactWrenchTrajectory` schema and says pi_H selects `ContactAssignment` sets from `ContactCandidateSet`, but does not prescribe a deterministic P1 baseline planner.
- Decision: Added an Agent H `GraspCarryBaselinePlanner` that consumes IRG, `InteractionEnvelope`, target `MorphologyGraph`, `ContactCandidateSet`, and optional `RuntimeObservation`. It prioritizes existing `grasp_pair` group proposals, converts selected candidates to maintain-state `ContactAssignment`s, checks them with `evaluate_selected_assignment_feasibility`, and emits a five-knot grasp/carry baseline trajectory: approach, attach, lift/maintain, transport, release.
- Compatibility impact: This is a deterministic baseline pi_H implementation, not a learned high-level policy. It does not output actuator commands and does not perform exhaustive candidate subset search. Later learned pi_H heads can replace the group selection/scoring while keeping the same `HighLevelPolicyContext -> ContactWrenchTrajectory` boundary.

### Selected Assignment Feasibility Proxy Supplement

- Context: v0.4 separates unary ContactCandidate screening from assignment-level feasibility after π_H selects `ContactAssignment` sets. It requires no exhaustive subset enumeration and leaves exact wrench/friction/collision/QP solving to later evaluator/controller work.
- Decision: Added `evaluate_selected_assignment_feasibility` as a deterministic selected-assignment proxy evaluator. It checks selected candidate existence and assignment consistency, unary-valid candidates, slot min/max cardinality when supplied by the caller, pairwise conflict matrix entries, duplicate selected candidates, a grasp-opposition wrench proxy, optional explicit wrench/QP residual thresholds, optional friction margin, and optional collision margin. Results are stored in `ContactCandidateSet.assignment_feasibility_cache` using the existing deterministic assignment key.
- Compatibility impact: This does not enumerate arbitrary candidate subsets and does not replace exact multi-contact feasibility. Later π_H and QP/collision backends can pass exact residuals/margins into the same `AssignmentFeasibilityResult` schema.

### P1 ContactCandidateSampler Deterministic Proposal Supplement

- Context: v0.4 requires morphology-conditioned `ContactCandidateSampler` after π_D, with unary screens, pairwise/group compatibility, and no exhaustive candidate-subset enumeration. Exact reachability, collision, and assignment-level wrench/QP feasibility belong to later checker/controller work.
- Decision: Added an Agent H deterministic sampler that consumes `TaskSpec`, IRG ContactSlots, `InteractionEnvelope`, target `MorphologyGraph` RobotAnchors, and `GeometryDescriptor` contact regions. It emits one candidate per compatible ContactSlot × ContactRegion × RobotAnchor by default, transforms patch/region positions from entity frame to world frame using the entity pose, and records deterministic unary smoke scores for mode match, normal alignment, local reachability, surface quality, moment-arm proxy, support quality, friction plausibility, and anchor capability.
- Group proposal supplement: Added small `grasp_pair` proposals for valid same-slot grasp candidates with different anchors, prioritized by opposing normals, and `support_set` proposals for support candidates when no grasp-pair proposal exists for a slot. This is pair/group scaffolding only and does not claim selected groups are full task-feasible.
- Compatibility impact: The sampler runs only after a morphology with RobotAnchors exists. It preserves the `ContactSlotID -> RobotAnchorID -> ContactCandidateID` boundary, does not perform exhaustive subset feasibility, and leaves selected-assignment wrench/QP checks for later π_H/assignment-level evaluators.

### GraphEditAssemblyPlanner Scaffold Supplement

- Context: v0.4 defines `AssemblyPlan`, `AssemblyStep`, and `ConstructionState`, and requires a deterministic π_A `GraphEditAssemblyPlanner`, but does not prescribe how a target `MorphologyGraph` should be expanded into P1/P3 smoke-level assembly steps.
- Decision: Added an Agent G deterministic graph-edit planner that treats the target morphology as a connected tree rooted at `base_module_id`. Starting from an initial construction state containing only the base component, each unattached `DockEdge` is expanded in stable `edge_id` order into four steps: `move_to_staging`, `align_ports`, `dock`, and `verify_attach`. The construction-state helper can mark verified edges as attached and rebuild the assembled subgraph with attached latch states and port occupancy.
- Interface supplement: Added implementation-local dataclasses matching the v0.4 assembly contracts inside `amsrr/assembly`: `AssemblyPlan`, `AssemblyStep`, `ConstructionState`, `AssemblyExecutionResult`, plus `ControlHandoffRequest` for controller handoff scaffolding. These are not added to the persisted `amsrr/schemas` package.
- Compatibility impact: This is deterministic π_A scaffolding, not learned assembly flight control and not simulator execution. Attach/detach safety is represented only as interface structure and precondition/success-condition records; exact motion planning, retry execution, QP feasibility during detach, and simulator verification remain later work.

## 2026-07-07

### Deterministic Design Teacher Scaffold Supplement

- Context: v0.4 requires a design grammar / teacher generator for bootstrapping π_D, but does not prescribe exact module poses, grammar expansion internals, or candidate-mask data structures for P1 fixed/simple morphology.
- Decision: Added an Agent E deterministic teacher scaffold that uses the existing minimal connected-tree morphology builder as the P1 fixed/simple morphology provider. Teacher variants are stable labels (`chain_grasp`, `symmetric_two_anchor_grasp`, `tri_anchor_support_grasp`, `central_base_plus_two_grasp_arms`, `perch_anchor_frame`, `valve_torque_arm`, `support_shift_frame`) over a schema-compatible `DesignOutput`. For object grasp/carry, the default P1 selection is `tri_anchor_support_grasp` when the IRG has required grasp slots and an optional support slot; otherwise it falls back to `symmetric_two_anchor_grasp` or `chain_grasp`.
- Candidate-mask supplement: Added a small `DesignCandidateGenerator` that wraps teacher action traces and masks `STOP` until the final teacher step. Its final STOP validity checks the existing Version 1 STOP conditions at a scaffold level: module count, base assignment, connected graph, port conflicts, required slot coverage, closed-loop rejection, and optional FeasibilityChecker result.
- Compatibility impact: This does not implement a learned π_D, does not change persisted schemas, and does not claim the teacher variants are optimized designs. Later policy heads can replace the scorer/sampler internals while keeping the `DesignPolicyContext -> DesignOutput` boundary.

### P0 ContactCandidate Pairwise Matrix Supplement

- Context: v0.4 requires pairwise/group compatibility for `ContactCandidateSet`, but exact contact-pair geometry, collision, and grasp grouping belong to later sampler/controller work.
- Decision: Added P0 pairwise helpers that mark immediate conflicts only: duplicate candidate IDs and candidates sharing the same robot anchor. Compatibility scores are deterministic smoke values: conflict `0.0`, same-slot different-anchor `0.75`, unrelated non-conflicting pair `0.5`, diagonal `1.0`.
- Compatibility impact: This does not claim arbitrary candidate subsets are feasible. Later ContactCandidateSampler and assignment-level evaluators can replace the heuristic internals while keeping the same `ContactCandidateSet` fields.

### Assignment-Level QP Smoke Supplement

- Context: v0.4 says assignment-level wrench/friction/collision/QP feasibility is evaluated after pi_H selects a `ContactAssignment` set, and that Version 1 must not enumerate every candidate subset.
- Decision: Added `evaluate_assignment_level_qp` as a selected-assignment smoke evaluator. It stores an `AssignmentFeasibilityResult` in `ContactCandidateSet.assignment_feasibility_cache` and emits `E_ASSIGNMENT_QP_INFEASIBLE` when the provided residual exceeds threshold.
- Compatibility impact: The QP backend remains out of scope for this P0 piece. A later exact evaluator can supply the residuals and margins without changing the cache/result schema.

### PolicyCommand Bias Builder Interface Supplement

- Context: v0.4 requires pi_L to output `PolicyCommand` intent while final actuator commands must come from the QP/PID/controller layer.
- Decision: Added `PolicyCommandBiasBuilder` and `DesiredBiasReferences` to convert `PolicyCommand` plus the active `InteractionKnot` into controller reference inputs: joint position/velocity references, desired wrench, body references, contact tracking references, anchor pose offsets, and merged priority weights.
- Compatibility impact: The builder deliberately does not produce rotor thrusts, vectoring joint commands, or final actuator commands. Later QP/PID interfaces can consume these references as controller inputs.

### Minimal Morphology Seed Supplement

- Context: v0.4 defines MorphologyGraph/DesignOutput and π_D action vocabulary, but a learned design policy and full deterministic teacher variants are later work.
- Decision: Added `MinimalMorphologyBuilder` as a deterministic P0 seed builder. It creates a connected tree of Holon modules, replicates dock ports from `PhysicalModel`, creates RobotAnchors from IRG ContactSlots, and emits a DesignAction trace ending in `STOP`.
- Compatibility impact: This produces valid `DesignOutput` objects for downstream FeasibilityChecker and ContactCandidateSampler scaffolding without pretending to be an optimized π_D policy.

### FeasibilityChecker P0 Coarse Proxy Supplement

- Context: v0.4 lists design-level hard checks including coarse reachability, collision, thrust margin, payload margin, and QP hover feasibility. Exact collision and QP solving require later simulator/controller integration.
- Decision: Added a design-level `FeasibilityChecker` scaffold with deterministic structural checks and coarse force proxies. Hover thrust uses `abs(thrust_axis_local.z) * thrust_max_n` per rotor because the Holon module is vectoring-capable and the normalized URDF contains both positive and negative local thrust axes.
- Compatibility impact: The checker owns deterministic hard violations now, while exact collision/QP checks can replace the proxy internals later without changing `FeasibilityResult`.

### SharedInteractionWorkspace Empty Group and Mask Supplement

- Context: v0.4 requires `group_slices` and `group_masks`, and lists several required modality groups. P0 currently has only some modality encoders implemented.
- Decision: Added `WorkspaceTokenGroup` and `SharedInteractionWorkspaceBuilder`. Missing required modality groups are represented as zero-width slices with empty `[B, 0]` masks and explicit `d_model` in the token group contract.
- Compatibility impact: A partial P0 encoder stack can still produce a valid full workspace while preserving required group keys. Downstream heads can rely on stable group presence even before every modality is implemented.

### SharedInteractionWorkspace Strict Group Mask Validation Supplement

- Context: v0.4 states masks must represent validity and zero-valued feature vectors must not imply validity.
- Decision: `SharedInteractionWorkspace` now requires a `group_masks` entry for every group slice. Each group mask must have shape `[B, slice_width]` and match the corresponding slice of the global `mask`.
- Compatibility impact: This strengthens the internal tensor contract without changing persisted task/IRG/envelope schemas. Existing tests were updated to pass explicit group masks.

### C3 Object-Pose Success Comparison Margin

- The object pose success precision remains 50 mm, as used by the Order 9
  object-task contract.  Formal deterministic Isaac evaluation applies a
  further 2 mm comparison margin to absorb contact-equilibrium and floating
  point variation at the exact boundary.  The effective comparison is thus
  `position_error <= 0.052 m`; this is not a change to the nominal trajectory,
  reward target, object distribution, or collision/safety gates.
- The named object-pose contract and effective comparison tolerance must be
  persisted in raw rollout metadata, formal episode metadata, and promotion
  results.  Promotion validation is fail-closed against those values so older
  50 mm-only evidence is not silently combined with the revised evidence.

### Tensor Contact-Preload Start Semantics

- Controller-side contact preload begins at the earlier of: (a) the existing
  simultaneous selected-contact force dwell, or (b) completion of the nominal
  contact-acquisition trajectory.  Case (b) is required when allowed physical
  randomization leaves a small gap at the deterministic nominal endpoint; it
  initializes from the current learned-policy absolute joint target and then
  uses the unchanged load/wrench-limited slow closure.
- Contact/preload completion remains mandatory before lift.  This start rule
  neither exposes contact truth to `pi_L` nor treats trajectory completion as
  successful contact, and it does not relax collision, QP, drop, force/load,
  or contact-dwell gates.

### InteractionEnvelope Count and Optional Slot Supplement

- Context: v0.4's grasp/carry envelope example has `required_contact_count_range: [2, 4]` and `required_contact_modes: [grasp, support]`. The current IRG template represents grasp as a required slot and support as an optional slot.
- Decision: `InteractionEnvelopeExtractor` computes `required_contact_count_range` from required ContactSlots only, while `required_contact_modes` and target region sets include all ContactSlots, including optional ones.
- Compatibility impact: This matches the v0.4 grasp/carry example and preserves optional support information for samplers and policies without inflating the required contact count.

### InteractionEnvelopeEncoder Contract Supplement

- Context: v0.4 requires an `InteractionEnvelopeEncoder` and states that it should fall back to `mlp_embedding` when no dedicated backend key exists, but does not define a concrete P0 tensor container.
- Decision: Added a dependency-free internal `InteractionEnvelopeEncoderOutput` dataclass with nested-list tensor-compatible fields: `tokens`, `mask`, `token_type_ids`, `source_type_ids`, `source_ids`, `group_slice`, and `group_mask`. The encoder emits deterministic scalar tokens and records `backend_type="mlp_embedding"` by default.
- Compatibility impact: No persisted schema changes are required. Learned MLP parameters and full SharedInteractionWorkspace assembly remain later implementation steps.

### IRGBuilder Template Constraint Mapping Supplement

- Context: v0.4 templates name several task-local constraints such as `max_tilt`, `maintain_contact`, `latch_feasibility`, and `support_polygon_proxy`, while the implemented schema validates constraint nodes against the existing `ConstraintType` enum.
- Decision: Agent D keeps the schema unchanged and maps these template-local concepts to the closest standard constraint types:
  - `max_tilt` -> `workspace`
  - `maintain_contact` -> `no_slip`
  - `latch_feasibility` -> `workspace`
  - `support_polygon_proxy` -> `support_ratio`
- Compatibility impact: IRG validation remains strict against v0.4 enum values, while the original template-local meaning is preserved in `ConstraintNode.feature["parameters"]["template_constraint"]` and the violation code.

### IRGBuilder Lazy Descriptor Requirement Supplement

- Context: v0.4's object grasp/carry example includes a floor support surface referencing `floor_geom` but does not include `floor_geom` in `geometry_library`. The object grasp/carry IRG only needs the target object's contact regions.
- Decision: Agent D normalizes all scene entities, but treats support surface and obstacle geometry descriptors as required only when a template actually requests their contact regions. Object descriptors remain required when referenced by object templates. Perching and contact-mediated locomotion still fail if their required support surface descriptor cannot be resolved.
- Compatibility impact: The v0.4 grasp/carry example can compile to an abstract object-contact IRG without adding schema fields or silently inventing geometry. Templates that need environment contact regions still receive deterministic failures for missing descriptors.

### GeometryProcessor P0 Mesh Smoke Supplement

- Context: v0.4 describes mesh loading, repair, normal/curvature segmentation, rim extraction, and convex decomposition. Full robust mesh processing is larger than the P0 smoke requirement.
- Decision: Agent C implements deterministic STL/OBJ smoke processing using bounding box, surface area, approximate volume, and dominant-normal patch clusters. It emits `SurfacePatchToken` and `ContactRegion` objects with `region_type="mesh_patch_cluster"`.
- Compatibility impact: The output conforms to existing v0.4 `GeometryDescriptor`, `SurfacePatchGraph`, and `ContactRegionGraph` schemas. Later work can replace the normal-cluster implementation with richer segmentation without schema changes.

### Geometry Reference Supplement: Hash URIs Instead of Raw Asset Paths

- Context: v0.4 requires file paths to be GeometryProcessor inputs only and not NN features.
- Decision: `GeometryDescriptor.collision_ref` and `exact_geometry_ref` use deterministic hash URIs such as `mesh://sha256:<hash>` and `primitive://sha256:<hash>` instead of raw filesystem paths.
- Compatibility impact: Asset paths are resolved inside `asset_resolver.py`; downstream learned components receive descriptor refs and patch/region tokens without path strings.

### Runtime Asset Supplement: Normalized Holon URDF

- Context: v0.4 recommends `robot_model.module_urdf_path: assets/robots/holon/holon.urdf`, while this checkout provides `module_urdf/holon.urdf.xacro` as the developer reference and does not provide `module_urdf/holon.urdf`.
- Decision: Added `assets/robots/holon/holon.urdf` as a normalized runtime asset derived from `module_urdf/holon.urdf.xacro`.
- User direction: User approved converting the xacro into an easier-to-use asset under `assets/`.
- Compatibility impact: Runtime path remains configurable and now matches `configs/robot/robot_model.yaml`. The original developer reference xacro is left unchanged.

### Naming Supplement: Thrust Link IDs

- Context: `configs/robot/thrust_model.yaml` uses rotor IDs `thrust_1` through `thrust_4`, while the developer reference xacro used link names `thrust1` through `thrust4`.
- Decision: The normalized runtime URDF uses link names `thrust_1` through `thrust_4`, matching the thrust config IDs.
- User direction: User approved changing to the `thrust_1` format during asset normalization.
- Compatibility impact: `RotorModel.rotor_id` preserves the config ID, and `RotorModel.thrust_frame_link` now resolves directly to a same-named URDF link. The loader still supports normalized matching for compatible future assets.

### Parser Supplement: xacro-Derived XML Without ROS Dependency

- Context: The provided `module_urdf/holon.urdf.xacro` is parseable as XML for the fields needed by P0, and adding ROS/xacro dependencies would be unnecessary for the current scope.
- Decision: Agent B loader parses URDF/xacro-derived XML with the Python standard library and ignores unknown/custom robot child tags while preserving useful metadata such as `baselink`.
- Compatibility impact: No package installation is required. Full xacro macro expansion remains out of scope until an asset actually requires it.

### Approved Schema Supplement: `IRGEdgeType.ALLOWS`

- Context: v0.4 Section 10.4 lists `IRGEdgeType`, but omits `allows`.
- Reason: v0.4 Section 10.2, Section 10.15, Section 11.7, and the worked example all require `ContactRegion --allows--> ContactSlot` edges.
- Decision: Added `IRGEdgeType.ALLOWS = "allows"` in `amsrr/schemas/irg.py`.
- Approval: User approved before implementation.
- Compatibility impact: This aligns the enum with existing v0.4 graph examples and enables deterministic IRGBuilder output without ad hoc edge strings.

### Implementation Supplements for Underspecified Helper Schemas

- Context: v0.4 references helper schemas such as `SurfaceSpec`, `ObstacleSpec`, `WindSpec`, `ObjectKinematicModel`, `CollisionPrimitive`, `ControlGroup`, runtime state records, and `ControllerStatus` without complete field-level definitions.
- Decision: Added minimal dataclass definitions for these helper schemas with conservative fields required by surrounding v0.4 contracts.
- Compatibility impact: These are additive implementation details intended to support parsing, serialization, and tests. They do not rename or remove any v0.4 core schema fields.

### Serialization Supplement: `SharedInteractionWorkspace.group_slices`

- Context: v0.4 defines `group_slices: dict[str, slice]`, while JSON cannot directly represent Python `slice` objects.
- Decision: Runtime schema stores Python `slice` objects, and JSON serialization represents them as `{start, stop, step}` mappings.
- Compatibility impact: In-memory contract remains `dict[str, slice]`; serialization is a practical roundtrip representation for tests and archives.

### C3 Promotion Hold: Module-Count Action Masks Are Curriculum Scaffolding

- Context: The staged C3 curriculum currently applies standard full action to
  2--3 modules, compression-only action to 4--7 modules, and joint-only action
  to 8 modules. Formal 32-episode held-out evaluation showed that these
  manually selected masks do not constitute one robust arbitrary-morphology
  execution contract. For the two 8-module held-out morphologies, no tested
  fixed action subset achieved zero safety failures on both.
- Decision: Treat the module-count masks as curriculum scaffolding only. They
  do not by themselves satisfy C3 promotion, and a mask must not be selected
  from held-out bucket identity. C3 remains unpromoted until a common
  morphology-general action contract, or a principled learned/safety
  projection mechanism, is approved and validated over the complete held-out
  set.
- Compatibility impact: No persisted policy schema or action dimension is
  changed. Existing checkpoints and curriculum evidence remain valid for
  staged learning, but cannot be cited as final C3 promotion evidence.

### C3 Common Coordinated-Compression Promotion Lineage

- The user approved a new first promotion lineage in which every morphology
  from two through eight modules uses the same
  `contact_compression_only` action contract. Training begins from the clean,
  behavior-preserving C2-derived C3 v5 initializer and every PPO update is
  topology-stratified over all seven module-count strata.
- Module-count-dependent action masks, count-specific warm starts,
  count-specific calibration, and checkpoint switching are prohibited in this
  lineage. The graph-conditioned actor may respond differently to different
  morphology observations, but the applied action semantics, optimizer,
  reward, controller, safety constraints, and promotion criteria remain the
  same for every morphology.
- The applied learned action is the single coordinated-compression residual
  decoded along the collision-limited IK compression direction. CoM pose,
  twist, residual-wrench, and independent joint residual outputs remain zero
  at the action boundary for this first promotion attempt. After this common
  contract passes C3, morphology-general CoM/global corrections may be added
  incrementally under separate matched validation.
- This changes no persisted schema, tensor dimension, controller/QP contract,
  deterministic safety authority, reward definition, or C3 promotion gate.

### C3 Full-PolicyCommand and Independent-Compression Superseding Lineage

- The user superseded the compression-only promotion attempt after its
  two--three-module learning plateau.  The next C3 lineage starts again at
  update 0 from the clean behavior-preserving v6 initializer and enables the
  same complete `pi_L` action contract for every two--eight-module morphology:
  centroidal pose/twist correction, residual centroidal wrench, every local
  joint position/velocity/torque correction, and one dedicated coordinated-
  compression gain.
- The final joint reference is
  `q_cmd = q_IK_nominal + delta_q_individual + g_compression * d_IK`.
  `d_IK` is the morphology-specific collision-limited compression direction.
  The dedicated gain never reuses, masks, or replaces any per-joint actor
  coordinate.  Joint velocity and torque-bias outputs retain their independent
  bounded meanings.
- This is one shared policy and one action interpretation across module
  counts.  Module-count-dependent action masks, checkpoint switches, or
  hand-selected output subsets are prohibited.  The actor may still condition
  its values on the observed morphology graph.
- QPID/QP and the existing safety shields remain downstream authorities;
  `pi_L` still emits only bounded `PolicyCommand` intent and never final
  actuator commands.  The reward, nominal trajectories, rollout buckets, and
  C3 promotion criteria are unchanged.  The new adapter/contract identities
  are persisted so older compression-only evidence retains its original
  semantics.
- C3 remains outcome-only: exact per-contact wrench-box membership has zero
  reward weight and is not a phase-admission or success gate.  Its retained
  privileged diagnostic is evaluated against the exact teacher box
  (`wrench_range_gate_scale=1.0`) in training and evaluation; the obsolete
  diagnostic-only widening schedule is disabled so telemetry cannot be
  mistaken for a changing learning contract.

### C3 Contact-Space Projected `pi_L` Superseding Lineage

- The user approved replacing the grasp-specific coordinated-compression
  scalar and learned residual-wrench path with one task-generic contact-space
  action contract.  A shared `pi_L` now emits: (a) centroidal pose/twist
  residuals, (b) a bounded local-frame pose residual for every active contact
  slot, and (c) bounded per-joint posture residuals.  The same action meaning
  is used for every morphology and contact task; there is no module-count or
  bucket-specific output mask.
- Each active-contact residual is the sole learned coordinate that changes
  the corresponding contact closure/slip target.  A deterministic adapter
  maps it through the nominal contact Jacobian to joint-space intent.  The
  centroidal residual is projected into the subspace compatible with the
  active point-contact translational constraints, and posture residuals are
  projected into their generalized-coordinate Jacobian nullspace.  This
  removes the previous ambiguity in which CoM, independent joints,
  compression gain, and residual wrench could all create the same grasp-force
  change.
- The maintenance projector constrains contact-point translation, not two
  independent full 6D rigid contact poses.  Contact-frame rotation remains an
  explicit bounded contact action.  Consequently two opposed contacts do not
  automatically cancel all centroidal freedom; compatible rigid motion such
  as rotation about the contact line can remain available.
- Projection authority is phase-continuous: it is inactive during approach,
  ramps in during contact acquisition, is fully active through lift,
  transport, and place, fades during release, and is inactive in retreat and
  settle.  The basis is built from the reviewed nominal contact posture and
  runtime application is tensor-only; the deterministic configuration-space
  planner remains teacher-generation tooling and is not executed during PPO
  rollout or deployment.
- For this v7 lineage, learned residual wrench and learned joint-torque bias
  are disabled.  QPID/QP remains the downstream authority for actuator,
  thrust, torque, and feasibility constraints.  Exact PhysX contact data may
  be used only for privileged training reward/evaluation and is not an actor
  observation.  The C3 outcome and safety gates remain unchanged.

### Batched QPID Applied-Command Feasibility and v7 Zero-Command Boundary

- Batched virtual-thrust allocation separates numerical optimizer convergence
  from physical command feasibility.  Every applied thrust and vectoring
  channel is first projected into the existing thrust, angle, angle-rate, and
  rotor-mask hard constraints.  Physical feasibility is then determined by
  whether the wrench residual recomputed from those actual applied commands is
  finite and no larger than the configured supported-wrench tolerance.
- The ADMM `solver_converged`, primal-residual, and dual-residual signals remain
  mandatory optimizer-health telemetry.  Failure to meet the optimizer's
  tighter optimality stopping criterion does not by itself create a QPID
  infeasible terminal when the projected command already satisfies the
  controller's physical residual contract.  This changes neither actuator
  limits nor the supported-wrench tolerance.
- A policy-version migration must preserve the command boundary.  Any output
  head that was masked by the source action contract but becomes active in the
  target contract is initialized to zero output, while compatible
  representation, recurrence, critic, and feature-decoder parameters may be
  copied.  In particular, the v6-to-v7 contact-space migration resets the
  newly activated global mean and final independent-joint decoder, as well as
  their exploration scales, instead of exposing previously masked commands.
- The repaired contract was checked with the unchanged contact-space update-2
  checkpoint on the four fixed two--three-module validation buckets: all
  16/16 formal phase-zero episodes reached settle with zero hard collision,
  object drop, fallback, safety failure, or QPID infeasible terminal.  The
  machine-readable evidence is retained under
  `artifacts/p4_full/order9/stages/c3_pi_l_ppo_arbitrary_morphology/diagnostics/contact_space_v7_qp_physical_feasibility_v2_fixed_2_3_16of16/`.

### C3 Transition-Targeted Backward-Curriculum Trial

- The approved five-module follow-up retains the common
  `contact_space_projected_policy_command`, reward, QPID, physics, and formal
  promotion gates.  It adds short continuous-state-inheritance shards only as
  training data; the deterministic configuration-space planner remains an
  offline teacher tool and is not run by the learned policy.
- Ordinary phase-reset replay continues to cover two--five modules and two
  train topologies per module count.  Four additional five-module shards cover
  all four train morphologies.  Each additional shard uses 48 environments for
  256 steps, seeds contact acquisition or lift, and resets every episode to the
  exact persisted 0.9 phase-progress stratum.  Physical and recurrent state
  cross a phase boundary without an intermediate reset.
- The fixed-progress reset is hash-bound as
  `order9_c3_transition_backward_curriculum_v1_fixed_progress`.  A configured
  progress value must exactly match a persisted stratum; silently choosing a
  nearest state is prohibited.  Module counts with zero inheritance shards
  remain legal only when at least one explicitly focused module contributes
  inheritance data.
- This trial is evidence, not a promoted replacement contract.  Updates
  10--13 remained numerically stable and produced contact-to-lift transitions
  on one of four five-module train morphologies, but no lift-to-transport
  transition on any morphology.  The best evaluated checkpoint still scored
  4/8 in the fixed five-module continuous validation.  Therefore no trial
  checkpoint is selected and the lineage is not eligible as a successor-stage
  parent.  Any further change to boundary supervision or curriculum ownership
  requires a new method-level decision.

### C3 Uniform Per-Morphology Transition-Backward Trial

- The five-module-only transition trial is superseded for diagnosis by a
  uniform contract: every active module-count stratum contributes exactly two
  continuous state-inheritance topology shards alongside its two phase-reset
  topology shards.  For the first six-module trial this applies identically to
  modules two through six; no module-count-dependent policy/action behavior is
  permitted.
- Each inheritance shard begins at the persisted 90% state of either contact
  acquisition or lift, executes for 256 control steps with 48 environments,
  and carries both physical and recurrent state across the next phase
  boundary.  This remains PPO training data only.  The deterministic planner
  is not executed by the learned policy during training rollout or inference.
- The shared action contract, outcome reward, QPID, physics, 4 mm nominal
  inward lead, and formal phase-zero evaluation are unchanged.  Therefore the
  trial isolates whether uniform exposure to true boundary state can close the
  training/evaluation distribution gap.
- Three stable updates from the accepted two--five-module parent improved
  aggregate reward and QP feasibility without material two--five-module
  forgetting.  The controlled six-module reward and phase-success measures did
  not improve, and all three checkpoints scored 0/8 on fixed six-module formal
  validation due to object drop during lift, with no hard collision or
  terminal QPID infeasibility at the final checkpoint.
- Consequently uniform transition inheritance is a valid data-generation
  mechanism but is not, by itself, a sufficient six-module solution.  This
  trial is not promoted and further identical PPO updates are not authorized
  by the evidence.  Altering contact-action authority, the common nominal
  contact margin, or transition-specific credit assignment is a new
  method-level decision.

### Six-Module Nominal Contact-Margin Diagnostic

- A fixed-policy diagnostic varied only the common virtual-contact inward lead
  over 4, 5, and 6 mm on the two held-out six-module buckets.  Every condition
  failed 0/8 by object drop during lift, without hard collision or terminal
  QPID infeasibility.  Larger lead increased both measured normal force and
  retention time, so contact margin contributes to the failure, but no tested
  value is accepted as a production replacement.
- The contact-space actor used only about 0.21 mm of its available 10 mm
  inward-normal residual range.  Therefore the failure is not caused by
  saturation of the common action contract.  Module-6 inheritance data did not
  contain an inward sample above 3 mm, while a controlled 2 mm nominal increase
  was still insufficient.  The next C3 method decision should therefore
  consider morphology-uniform widening of contact-normal exploration so the
  shared policy can observe the unused higher-compression region; merely
  continuing identical PPO updates or changing the production nominal to 5/6
  mm is unsupported.

### C3 Morphology-Uniform Contact-Normal Exploration Trial

- The approved trial widens only the stochastic
  `translation.inward_normal` contact-space coordinate, uniformly for every
  morphology. A provenance-bound migration changes its per-contact standard
  deviation from `0.13533` to `0.30` while proving that all deterministic
  actor means and every other checkpoint scalar are unchanged. The migrated
  file is an initializer only; promotion evidence requires fresh on-policy
  PPO and deterministic physical validation.
- The action mean, 10 mm residual span, production 4 mm virtual-contact lead,
  common `contact_space_projected_policy_command`, reward, QPID, physics, and
  module-count behavior remain common and unchanged. In particular, this
  mechanism is training-only exploration and does not add stochasticity at
  deployment.
- The fresh 2--6-module update successfully entered the intended unused action
  region: module-6 lift samples using more than 3 mm of two-contact mean
  closure increased from zero to `8.29%`, and weaker-anchor normal force rose
  from `0.788 N` to `1.182 N`. Training remained KL-stable and the resulting
  deterministic mean moved toward stronger closure.
- The fixed formal six-module evaluation nevertheless remained `0/8`, with
  all episodes dropping the object during lift and no episode reaching
  transport. The deterministic mean shift was only about `0.042 mm`; wider
  symmetric exploration also reduced lift-sample reward. Therefore this trial
  does not supersede the accepted C3 parent and its child is not promotion
  eligible.
- Further identical updates are unsupported by this evidence. Any follow-up
  that makes exploration asymmetric or changes contact-retention credit so
  high-compression successes affect the deterministic mean is a new
  method-level change and requires an explicit decision before implementation.

### C3 Factorized Actor-Credit Trial

- The approved factorized-credit contract preserves the common
  `contact_space_projected_policy_command`, actor observations, action bounds,
  nominal trajectory, QPID, physics, and total scalar reward. It changes only
  training-time credit assignment.
- Every recorded reward transition is partitioned exactly into contact,
  centroidal, and posture channels. Contact maintenance, wrench-range, and
  slip terms supervise the contact-space density; object-motion, CoM/QP, and
  actuator terms supervise the CoM pose/twist density; collision supervises
  the independent joint density. Energy is shared between centroidal and
  posture, and terminal task outcome is shared equally by all three roles.
  The three channels must reconstruct the original reward and unnormalized
  GAE within recorded numerical tolerances. The critic remains a single total-
  return critic.
- The policy encoder remains shared, while each role-specific PPO likelihood
  excludes the other two output distributions. Thus a contact advantage does
  not directly update CoM or independent-joint output parameters, although it
  may still improve their common representation.
- Three fresh two--six-module updates were numerically stable, without KL
  rollback or early stop. The fixed two-bucket, eight-episode six-module
  validation nevertheless remained 0/8 for updates 13, 14, and 15. Every
  failure was an object drop in transport; no hard collision, terminal QPID
  infeasibility, timeout, or fallback occurred.
- Therefore factorized actor credit is accepted as a valid experimental
  training mechanism but does not supersede the accepted C3 parent and is not
  sufficient evidence for promotion. Additional identical updates are not
  justified by the non-monotonic six-module reward and unchanged formal
  outcome. Any next change to temporal contact-retention credit or the common
  contact-control contract is method-level work.

### C3 Deployable Contact-Feedback `pi_L` v8

- The user approved adding deployable closed-loop contact observations while
  preserving the task-generic v7 contact-space action contract. Each active
  contact slot appends signed anchor-to-object surface distance, contact-frame
  relative linear and angular velocity, and a signed motor-load compression
  proxy. The actor does not consume raw PhysX contact force and does not assume
  an F/T sensor; simulator contact information remains privileged reward and
  evaluation data only.
- The v7-to-v8 migration is provenance-bound. It copies all existing policy
  parameters exactly, expands the contact feature encoder from 30 to 38
  inputs, and initializes only the eight new columns to zero. A fresh on-policy
  update is mandatory before the migrated checkpoint can be considered a
  continuation candidate.
- A fixed-policy, acceptance-ineligible physical sweep showed that common
  inward-normal contact residuals of `0/2/4/6/8 mm` produced
  `0/0/7/8/8` successes out of eight on the two fixed six-module buckets.
  Therefore v8 initializes the trainable contact-action mean at `6 mm` and its
  exploration standard deviation at `1 mm`. The production nominal lead
  remains `4 mm`. This initialization is policy state and is not a fixed
  controller-side preload, QPID feed-forward command, or module-count-specific
  action rule.
- Factorized contact/centroidal/posture PPO credit remains active. The v8
  feedback and action prior apply uniformly to every morphology; QPID/QP still
  owns actuator limits and final physical feasibility.
- Fresh two--six-module update 0 was numerically stable and the exact child
  checkpoint passed all `8/8` full-sequence held-out six-module episodes with
  zero drop, hard collision, terminal QPID infeasibility, timeout, fallback,
  or safety failure. The feedback encoder's new columns became non-zero, but
  the deterministic contact residual remained approximately `6.08 mm` with
  only about `0.008 mm` temporal variation. The present success is therefore
  attributed primarily to the trainable prior; feedback-dependent adaptation
  requires evidence from later updates/randomized conditions.
- This result selects v8 update 0 as the six-module continuation candidate,
  not as final C3 promotion. The same contract must still be extended through
  modules seven and eight and pass the complete promotion evaluation.

### C3 Factorized Contact-Head Extra Optimization

- Seven-module extension updates 1 and 2 showed deployable load/slip feedback
  differences between the successful and failed buckets, while all six
  deterministic contact-space outputs remained effectively constant. A fixed
  6--10 mm inward-normal sweep did not resolve the failed bucket. Additional
  identical all-head PPO updates are therefore not supported.
- The approved minimum follow-up preserves the existing v8 actor observation,
  deployed action, nominal trajectory, QPID, physics, scalar reward, and
  factorized reward routing. After the ordinary one-epoch all-head PPO update,
  the same immutable on-policy rollout receives one additional PPO pass using
  only the factorized contact advantage and contact action likelihood.
- The extra pass may update only `contact_space_feature_encoder`,
  `contact_space_slot_embedding`, `contact_space_actor_mean`, and
  `contact_space_actor_log_std`. The shared graph/recurrent trunk, centroidal
  head, posture head, and critic are fixed during this pass so contact credit
  cannot indirectly move unrelated deployed outputs.
- The extra pass uses the original behavior checkpoint log probability,
  topology-equal minibatches, exact recurrent replay, PPO clipping, and the
  same non-target/topology-phase KL rollback limits as the normal pass. It is
  applied identically to every module count and introduces no runtime switch,
  sensor, or module-specific action rule.

#### Trial outcome

- The implementation boundary was verified: only the contact feature encoder,
  slot embedding, contact mean, and contact log-standard-deviation changed in
  the additional pass. The shared trunk, centroidal/posture outputs, and
  critic did not receive that pass's optimizer update.
- A fresh common 2--7-module update used 671,744 transitions and produced
  child SHA `76f8a684...c3bf2b`. It applied 135 ordinary and 132 additional
  contact-only minibatches. The final candidate minibatch was rolled back by
  the existing non-target parent-KL limit; all prior applied updates remained
  within the configured limits.
- Fixed seven-module validation remained `4/8`: the already-solved bucket
  stayed `4/4`, while the difficult bucket stayed `0/4` with object drop in
  lift and no collision or terminal QPID infeasibility.
- Exact replay of both policies on the failed rollout showed that the
  deterministic inward-normal action moved by only about `+0.008 mm`; most
  policy change appeared in tangential/rotational coordinates. A single
  contact advantage attached to the summed 6-D contact likelihood therefore
  does not resolve within-head coordinate credit assignment.
- This child is diagnostic-only and does not supersede its parent. Further
  identical extra passes are not justified. Coordinate-specific contact
  credit or action-probe-derived supervision is a separate method-level
  decision.

### C3 Contact-Coordinate Credit Trial

- The approved diagnostic retained the v8 runtime contract and ordinary
  factorized PPO update, then split only the contact-head extra pass into
  inward-normal translation, tangential translation, and rotation. Grasp
  maintenance supervises the normal likelihood, slip supervises the
  tangential likelihood, and terminal contact outcome is shared. The three
  rewards and GAEs must reconstruct the existing contact reward and advantage.
- This is a training-only mechanism. It adds no sensor, deployed action,
  runtime switch, module-specific rule, nominal command, or QPID change.
- One fresh common 2--7-module child passed exact replay and stayed within all
  KL limits. Fixed seven-module validation nevertheless remained `4/8`:
  bucket 33 passed `4/4`, while bucket 40 dropped the object during lift in
  `4/4` with no collision or terminal QPID infeasibility.
- The failed bucket's deterministic normal residual increased by only about
  `0.022 mm`. Separating coordinate likelihoods therefore resolves the
  within-head attribution ambiguity, but does not provide the counterfactual
  signal needed to learn how much additional squeeze would avoid a later
  drop.
- The coordinate extra pass is implemented but disabled in the accepted
  curriculum. Its child SHA `0d27905a...d7f` is diagnostic-only; accepted
  update 2 SHA `5df273b5...455f` remains the continuation parent. Further
  identical updates are unsupported. Action-probe-derived or equivalent
  causal supervision is a method-level alternative requiring a separate
  decision.

### C3 Common 20 mm Contact-Normal Authority Diagnostic

- The user approved changing the morphology-independent contact-space
  inward-normal residual limit from `10 mm` to `20 mm`.  The nominal virtual
  contact lead remains `4 mm`; tangential/rotational limits, independent joint
  residuals, centroidal residuals, QPID, physics, and bucket geometry are
  unchanged.  No module-count-specific switch is introduced.
- A fixed-checkpoint A/B used the same seven-module bucket 40, update-3 SHA
  `0d27905a...d7f`, seeds `9052--9055`, and four formal phase-zero episodes.
  The deterministic normalized normal action remained about `0.612`, so its
  physical normal residual changed from about `6.12 mm` to `12.24 mm`.
- The larger authority improved retention materially: terminal step counts
  changed from `1620--2049` to `2338--2408`, episode return increased in every
  seed, and the deployable estimated minimum normal force increased from
  `0.000--0.206 N` to `0.628--1.126 N`.
- It did not complete bucket 40: all four episodes still dropped the object
  during lift, before `lift -> transport`.  Hard collision, terminal QPID
  infeasibility, timeout, and fallback counts remained zero.  Therefore
  `20 mm` authority is a useful causal improvement but is not by itself a
  successful fixed-checkpoint solution or promotion result.  Training under
  the widened common action scale must be evaluated separately.

#### Retraining outcome

- A fresh common 2--7-module update was run from accepted update 2 under the
  20 mm physical scale. The matched trial retained factorized PPO and added
  the implemented normal/tangential/rotational contact-coordinate pass.
- The child remained within every KL guard but failed bucket 40 in all four
  fixed seeds: contact acquisition succeeded and the object dropped during
  lift, with no collision, QPID-infeasible terminal, timeout, or fallback.
- The deterministic lift normal residual changed from about `12.217 mm` in
  the parent to `12.199 mm` in the child. The extra physical authority was
  therefore available but PPO did not learn to use more of it.
- The 20 mm common authority remains the deployed action bound. The
  coordinate extra pass remains diagnostic and disabled in the accepted
  curriculum; the failed child does not replace accepted update 2. More
  identical PPO budget is unsupported without a different causal learning
  signal or an explicit curriculum-method revision.

### C3 Physical Contact-Normal Quantization Diagnostic

- A morphology-independent optional action adapter may quantize only the
  physical inward-normal contact residual after continuous-policy scaling and
  before contact-Jacobian projection. The latent PPO action and likelihood
  remain continuous, so exact on-policy replay is unchanged. The adapter must
  not quantize tangential/rotational contact coordinates, CoM/global actions,
  or independent joint residuals, and its step must be recorded in rollout
  provenance.
- The approved `0.5 mm` experiment used only seven-module data for one update
  from accepted update 2. It completed within all KL limits, but the fixed
  difficult bucket remained `0/4` with object drop during lift.
- The parent and child latent actions both fell in the same physical
  `12.0 mm` bin at every valid step (`12.1725 mm` versus `12.1755 mm` before
  quantization). Thus this experiment did not test a different deployed
  normal command after learning and supplies no evidence that quantization
  improves grasp retention.
- Quantization remains implemented but disabled in the accepted curriculum.
  The active C3 contract continues to use the continuous physical normal
  mapping and accepted update 2; the seven-module child is diagnostic-only.

### C3 Direct Categorical Contact-Normal Diagnostic

- A diagnostic π_L policy may replace only each active contact slot's
  inward-normal contact-space distribution with 81 categorical values over
  `[-20 mm, +20 mm]` at exact `0.5 mm` intervals. This is an explicit
  categorical likelihood trained by PPO, not a continuous Gaussian rounded
  after sampling.
- The other five contact coordinates remain squashed-Gaussian. CoM/global and
  independent-joint actions, nominal IK lead, QPID, reward, observation, and
  physics contracts remain common and unchanged across module counts.
- Migration from v8 copies all existing parameters exactly and initializes
  only the new category-logit head. Its initial state-dependent prior is
  centered on the existing continuous normal mean so migration does not
  arbitrarily change deterministic physical action.
- Mixed-action behavior likelihood must be computed directly from the five
  executed Gaussian coordinates plus the executed categorical coordinate.
  An unsaved provisional Gaussian normal sample must not participate in the
  stored likelihood. Exact replay is fail-closed.
- One seven-module update passed exact replay and improved difficult bucket 40
  from `0/4` to `1/4`, but deterministic inference remained at the same
  `12.0 mm` category in every reached phase. This child is diagnostic-only and
  does not replace accepted update 2. The direct categorical mechanism is not
  yet an accepted C3 production contract.
- The approved extension through three total categorical updates moved the
  deterministic choice to the adjacent `12.5 mm` category. Nevertheless,
  update 5 scored `0/4` on difficult bucket 40 (all object drops in lift),
  while a known-success seven-module control bucket scored `4/4`. Thus the
  categorical head can change the deployed squeeze bin, but more identical
  PPO budget did not solve the target retention failure. The extension is
  diagnostic evidence only; update 5 is not promoted and accepted update 2
  remains the continuation parent.
- A fixed-action causal sweep on difficult bucket 40 subsequently established
  that the same plan and controller succeed `4/4` at each of `16, 18, 20 mm`,
  while `10, 12, 14 mm` score `0/4, 1/4, 0/4`. No tested value caused hard
  collision or terminal QPID infeasibility. Thus bucket 40 is not intrinsically
  infeasible; the categorical policy's learned `12.5 mm` argmax is below the
  observed robust-success region. This evidence supports revising categorical
  exploration/initial probability mass, not hard-coding a global `16 mm`
  residual or replacing the bucket.
- A subsequent full-support broad-prior diagnostic kept the same 81 categories
  but distributed initial probability nearly uniformly over `10--20 mm`.
  One seven-module update learned a deterministic `16.5 mm` category and
  passed both difficult bucket 40 and control bucket 33 at `4/4`, with no
  safety failure. This validates broad high-squeeze exploration as a candidate
  curriculum contract. It does not yet authorize promotion: the identical
  common action/distribution contract must be trained and regression-checked
  across the earlier module-count strata without module-specific switches.
- The same categorical contract was then trained jointly on module counts
  2--7 in one topology-stratified update. Paired held-out phase-reset checks
  showed no catastrophic forgetting or broad safety regression for module
  counts 2--6, and formal seven-module validation passed both bucket 33 and
  difficult bucket 40 at `4/4` with no safety failure. The policy selected
  contact-specific normal residuals (`16.5/20.0 mm` and `16.5/18.0 mm`) rather
  than a module-specific fixed command. Update 4 SHA `19ab01d4...22eb` is
  therefore the accepted continuation parent for adding eight-module data;
  it is not yet a C3 promotion result.

### C3 Actuator/Leverage-Aware Physical-Minimum Nominal Preload

- The common nominal contact lead is morphology- and contact-conditioned, not
  module-count-conditioned. At the reviewed final contact posture, the
  deterministic calculation allocates the normal forces needed for
  `1.25 * m * g` frictional support while minimizing peak joint utilization
  under the configured actuator torque envelope. It then converts the
  allocated normal force to displacement using joint compliance projected
  through the contact-normal Jacobian plus configured contact compliance.
- The selected nominal lead is the largest required anchor displacement,
  bounded below by `4 mm` and rounded upward to the next `1 mm`. The formerly
  added universal `12 mm` model-error margin is removed: it was not derived
  from morphology/load mechanics and duplicated the responsibility of the
  bounded learned contact-normal residual. No per-module or per-bucket rule is
  permitted.
- Terminal contact-offset IK must reproduce the requested normal displacement
  within its configured tolerance while preserving the tangential contact
  coordinates and joint limits. Bounded continuation/refinement is a
  numerical realization of the same IK contract, not a new high-level
  planner or a runtime collision-avoidance substitute.
- Actor-free real-Isaac evidence used the full accepted nominal sequence and
  production QPID/QP with all learned `pi_L` corrections set to zero.
  Eight-module bucket 34 used a `30 mm` calculated nominal lead and passed
  `4/4`; five-module bucket 38 used `7 mm` and passed `4/4`. All eight runs
  reached release/settle with zero drop, hard collision, terminal QPID
  infeasibility, timeout, fallback, or safety failure.
- This evidence validates the physical-minimum nominal baseline only. It is
  acceptance-ineligible for C3 promotion because the learned actor was
  disabled. The categorical contact-normal action, all other policy outputs,
  observation schema, `PolicyCommand`, and QPID/QP authority remain unchanged;
  a fresh common 2--8-module on-policy lineage and formal actor-enabled
  validation are required before promotion.

### Physical-Minimum C3 Retraining Evidence

- A fresh v9 C3 initializer is derived from the promoted C2 representation via
  the clean zero-command contact-space initializer.  It does not inherit a
  learned actor head from any retired C3 preload lineage.  Its normal-contact
  action is the common 81-category `[-20, 20] mm` contract at `0.5 mm`
  resolution, while CoM pose/twist, tangential/rotational contact residuals,
  and independent joint position/velocity residuals remain active.
- Initializer migration may use update index `-1` to denote a pre-training
  artifact and may use exactly zero initial contact-normal residual.  This is
  provenance metadata, not a trained negative update number and not a change
  to runtime action semantics.
- Four topology-stratified 2--3-module updates were run under the corrected
  physical-minimum nominal contract.  PPO replay, KL bounds, entropy, QPID
  feasibility, and all actor output paths were operational; parameter-diff
  evidence rules out a frozen actor or action-routing failure.
- Paired same-bucket rollout evidence did not show repeatable improvement.
  Module-2 mean reward was effectively flat, while module-3 mean reward moved
  from `3.445186` to `3.391553` in one paired generation and from `3.315786`
  to `3.293348` in the other.  Therefore update 3 is diagnostic-only, no C3
  checkpoint is promoted, and the curriculum must not expand to four modules
  from this lineage without first resolving the absent 2--3-module learning
  gain.

### Promoted Common C3 Contract (Update 18)

- The earlier flat 2--3-module diagnostic above is not the final lineage. The
  accepted continuation restarted from the clean v9 initializer and completed
  19 hash-linked PPO generations while expanding the common replay curriculum
  from module counts 2--3 through 2--8.
- The same v9 policy/action contract applies to every module count. Runtime and
  evaluation must not select module-count-specific action masks, controller
  modes, nominal-preload rules, or success rules.
- The deterministic nominal contact lead is morphology/contact/load
  conditioned, rounded upward in 1 mm steps, bounded below by 2 mm, and has no
  universal 12 mm addition. The policy retains a distinct bounded categorical
  contact-normal residual, together with its global, joint, tangential, and
  rotational residual paths.
- Probe-derived supervision is an auxiliary update to the contact-normal head;
  it does not replace PPO. Seven-module training used PPO update 17 followed by
  the head-only probe teacher. Eight-module training used the preload-deficit
  teacher before PPO update 18. Teacher passes do not consume PPO update
  indices.
- The accepted checkpoint is update 18 SHA-256
  `6ea412ccdfe983cb2b030b3514bc8982424522673b49def188d547120984357b`.
  Formal actor-enabled phase-zero evaluation passed 448/448 episodes over 14
  held-out buckets with no safety failure or fallback. C3 is therefore
  promoted under this exact contract.
- The authoritative training/data inventory is
  `for_codex/C3_PROMOTED_UPDATE18_RELEASE_LEDGER.json`; the human-readable
  release description is `for_codex/C3_PROMOTED_UPDATE18_RELEASE.md`.

### R1 Reachable-Pose Calibration v1 Approval and Execution Order

- On 2026-08-24 the repository user approved the complete R1 v1 calibration
  package. The normative executable input is
  `configs/training/order9_r1_calibration_protocol_v1.yaml`; its authorization
  is `for_codex/R1_CALIBRATION_PROTOCOL_V1_APPROVAL.json`. The proposal remains
  historical review evidence. The approval record binds both protocol byte
  hashes and the protected C3 update-18 checkpoint identity.
- Selection expands residual object x/y/yaw in four ascending levels:
  10 mm/5 degrees, 20 mm/10 degrees, 30 mm/15 degrees, and
  40 mm/20 degrees. Each level uses all 22 train-owned source buckets, with
  27 boundary/center lattice cases and four deterministic interior cases per
  bucket. Evaluation stops at the first failed level.
- Only the largest consecutively passing selection level is evaluated once on
  the untouched 14 validation-owned buckets. Confirmation failure rejects the
  complete v1 result. It must not select a smaller level or tune v1 against
  validation evidence.
- Every candidate follows one enforced sequence: deterministic teacher plus
  deterministic IK, then the Isaac-free exact-tracking screen, then—only on a
  screen pass—two complete replays through protected C3 pi_L, QPID/QP, local
  servo, and Isaac. A screen pass asserts kinematic arrival at the maintained
  two-contact grasp pose only; it does not assert physical grasp success.
- This approval does not constitute a calibration result. No range is accepted
  until selection and the single confirmation pass and their evidence is
  published in a hash-bound distribution manifest. Formal R1 teacher data,
  imitation learning, or C3 release mutation is not authorized by approval
  alone.

### R1 Isolated pi_L Action-Path Diagnosis

- This is a diagnostic finding, not a modification of the promoted C3 action
  contract or the v0.5 R1 learning contract. The promoted checkpoint and
  protected rollout remain byte-identical and authoritative.
- On the first rejected 10 mm/5 degree R1 candidate, controlled two-environment
  replays retained either direct joint position/velocity correction alone or
  contact-normal compression correction alone. All other pi_L action paths
  were zero, and the teacher, deterministic IK, initial state, physics,
  QPID/QP, local servo, and Isaac path were held fixed.
- Both isolated modes failed 0/2 by the same module-1 battery/support collision
  during contact acquisition, with zero object force and zero fallback.
  Therefore each path is independently sufficient for this observed failure;
  their combination is not necessary.
- No general ranking of the two paths follows from one candidate. A future
  derived-policy or safety-contract proposal must address both paths; merely
  disabling either one is contradicted by the controlled evidence. Such a
  proposal still requires explicit approval before implementation, learning,
  or R1 recalibration.
- The hash-bound combined diagnostic is
  `artifacts/p4_full/order9/r1_teacher/calibration_v2_support_clearance/diagnostics/isolated_action_comparison_final_v1/r1_l1_10mm_5deg__train__train-000000-5fecfad4f44f__lattice_00/diagnostic_result.json`,
  SHA-256 `2ca12b869ce9c18c5fc2e11a22071e6c7337ec460f6c8f23ccf50547aadd6f34`.

### 承認済み方針: π_Lの適用を他要素の完成後まで延期する

#### 2026-08-25

- 決定: π_Lの仕組み、入出力契約、実装、検査、C3 promoted update 18、および
  保護済み成果物は削除せず維持する。ただし、R1以降の物体条件拡張、π_H、π_D、
  物体形状・把持点・別タスク対応を進める間、現行π_Lを標準実行経路の必須要素と
  しない。π_Lの再適用は、他要素の完成後、測定上必要な場合に限って別途判断する。
- 当面の標準低レベル経路は、π_Hまたは決定論的教師が出す軌道を決定論的IKへ
  渡し、形態・接触・荷重に応じた名目押し込み、QPID/QP、局所サーボ、安全制限を
  経て実行する構成とする。π_Lを使わないことは、名目押し込み、QPID/QP、局所
  サーボ、または安全層を無効にすることを意味しない。
- C3の448/448成功は、promoted update 18とC3評価分布に対して引き続き有効である。
  ただし、そのπ_LをR1以降の物体位置、物体形状、接触点数、または別タスクへ
  自動的に外挿する根拠にはしない。C3保護成果物は回帰比較と将来の派生作業の親と
  して保持する。
- 現行R1 v1の不合格判定は変更しない。名目経路による1候補2実行の成功だけで、
  R1分布、教師収集、または模倣学習を承認してはならない。新しい標準経路を実装
  した後、一次判定を通過した候補について、名目経路で較正を最初から実行する。
- π_Lを将来再適用するには、少なくとも次を満たさなければならない。
  1. 同一条件の名目経路に、π_Lで対処すべき測定済みの不足がある。
  2. π_Lを加えた同条件比較で、その不足が改善する。
  3. それまでに獲得した機体構成・物体条件・接触点数・タスク全体で、安全性を
     悪化させない。
  4. 危険な関節補正と接触補正を縮小または0へ戻せる決定論的安全境界を持つ。
  5. 再適用の範囲、学習系統、検査条件、派生成果物を新しいハッシュで固定する。
- この方針は、v0.5 Section 24.5.9にある各R1--R4段階でのπ_L BC、π_L PPO、
  π_L再調整を必須順序とする工程、および既存R1実行器がpromoted C3 π_Lを必須と
  する契約を変更する。v0.5の優先規則上、このログだけでは実行契約を切り替えない。
  実装または学習を再開する前に、v0.5の後継仕様、学習工程設定、R1較正実行器、
  事前検査、結果形式を同じ方針へ更新し、相互の食い違いを解消しなければならない。
- 本項の追加時点では、仕様本文、学習工程設定、実行コード、チェックポイント、
  教師データ、成果物を変更していない。学習、教師軌道収集、R1再較正も開始して
  いない。

### R1名目制御較正v2の実装と不合格確定

#### 2026-08-25

- 前項の承認済み方針をv0.5本文へ統合し、保護済みのC3学習工程設定を変更せず、
  R1専用の追加契約、承認記録、実行部、結果形式を実装した。
- R1の標準実行経路は、決定論的教師軌道、決定論的IK、形態・接触・荷重に応じた
  名目押し込み、QPID/QP、局所サーボ、Isaacとした。π_Lの仕組みとC3昇格済み
  チェックポイントは保持したが、π_L由来の動作量は適用しなかった。
- 接続確認では、2モジュール機の1候補を2回Isaacで再生し、2/2成功、安全停止0、
  代替制御0だった。記録されたπ_L由来の6種類の動作量は全時刻で厳密に0だった。
  この確認は実行経路の確認専用で、較正合格や学習の根拠ではない。
- 正式選定では、最小の10 mm/5度段階について、学習側22機体の最外側格子点を
  Isaacなしで先に一次判定した。20機体は合格したが、5モジュール機1件と
  7モジュール機1件が関節限界からの必須余裕1%を満たさなかった。両件とも
  衝突違反0で把持姿勢には到達していた。
- 格子点100%条件が達成不能になったため、第1段階の正式Isaac試験、第2--第4段階、
  未使用14機体での最終確認を行わず、採用範囲なしで不合格とした。教師軌道の
  正式収集と学習は許可しない。
- 規範結果目録は
  `for_codex/R1_NOMINAL_CALIBRATION_V2_RESULT_LEDGER.json`、SHA-256
  `09fa797d7be267a59266c1df0258127a6723042855eb54b9a85a3b1ef8f07a4a`である。
- 次は不合格2件の決定論的IKと把持姿勢における関節限界張り付きの原因確認とする。
  関節余裕条件の緩和、10 mm/5度未満への範囲縮小、教師軌道または把持姿勢の変更は、
  今回の不合格から自動的に導入せず、別の設計判断と承認を必要とする。

### R1関節限界余裕の高速・安全側補正

#### 2026-08-25

- 原因: 既存IKは物理関節限界を最適化制約に含めていたが、R1一次判定が要求する
  物理範囲の1%内側を実効制約に含めていなかった。このため、物理的には可行な
  限界上の解が、後段の1%余裕検査だけで不合格になった。検査値を緩める問題ではない。
- 決定: 昇格済みC3と通常のR1教師生成は従来どおり一度だけ実行する。1%余裕を
  既に満たす軌道はその同一物を直ちに返す。満たさない軌道だけを生成後に処理し、
  1%内側の実効上下限へ射影した後、補正対象の全時刻を配列として同時に有界IKで
  解く。これにより、補正結果を逐次的な区間再計画へ戻して計算時間を増幅させない。
- 複数初期姿勢: 通常枝で把持点精度を回復できない姿勢群だけに、限界へ張り付いた
  関節を中央または反対側四分点へ移した初期姿勢、および2種類の決定論的全関節
  初期姿勢を試す。可行解から正規化関節余裕が最大のものを選び、同値なら時刻間の
  最大関節変化が小さいものを選ぶ。通常枝が通る時は追加枝を実行しない。
- 安全条件: 補正された全時刻について、把持点の位置`0.011 m`・姿勢`0.10 rad`、
  物体・支持台・自己衝突、関節速度、および1%余裕を再検査する。一つでも失敗すれば
  補正結果を採用せず不合格とする。C3保護成果物、π_L契約、R1数値条件は変更しない。
- 速度: 通常合格2モジュール例では追加判定は`0.001943943 s`で、元の軌道物を
  そのまま返した。旧不合格5モジュール例では既存生成`96.224423 s`、追加処理
  `3.376077 s`、旧不合格7モジュール例では既存生成`409.283287 s`、追加処理
  `3.538568 s`だった。各時刻を逐次的に再最適化する旧試作の約40--90秒増加はない。
- 実例結果: 7モジュール `train-000026-a8ced2acf24c` は正規化最小余裕
  `0.010000000999999994`でIsaac前一次判定を通過した。5モジュール
  `train-000003-3b95da871f01` は複数初期姿勢後も一時刻の把持点位置誤差
  `0.0194142 m`が許容`0.011 m`を超えたため不合格のままである。
- 非主張: これは正式較正の再実行ではなく、Isaac、教師軌道収集、学習、削除、
  commitを含まない。R1名目較正v2の不合格目録を上書きせず、正式収集を承認しない。
  5モジュール例を通すには教師軌道または接触姿勢の変更が必要となる可能性があり、
  それは本数値修正とは別の設計判断である。

### R1補正後把持点の三次元距離30 mm許容

#### 2026-08-25

- ユーザー承認: R1の補正後把持点について、目標把持点からの三次元ユークリッド
  距離を`0.030 m`以内とする。これはx/y/z各軸を独立に±30 mm許容する契約ではなく、
  斜め方向を含む距離半径30 mmである。
- 適用範囲: R1専用の生成後関節余裕補正とその再検査にだけ適用する。C3教師の
  `0.011 m`、C3 promoted update 18、把持点姿勢`0.10 rad`、1%関節余裕、衝突、
  関節速度、QPID/QP、局所サーボの条件は変更しない。
- 実例結果: 旧不合格5モジュール `train-000003-3b95da871f01` は、補正後の
  最大把持点距離`0.0211487 m`、最小正規化関節余裕`0.010000001`、衝突違反0、
  最小衝突余裕`0.00480859 m`、最大関節速度`0.274444 rad/s`となり、Isaac前一次
  判定を通過した。実行時間は生成と補正を含め`99.2888 s`だった。
- 7モジュール回帰: `train-000026-a8ced2acf24c` も最小正規化関節余裕
  `0.010000001`、衝突違反0、最小衝突余裕`0.00480358 m`、最大関節速度
  `0.274444 rad/s`で一次判定合格を維持した。生成と補正を含む時間は`411.3660 s`。
- 非主張: これは閾値実装と個別の読み取り検証であり、R1名目較正v2の正式再実行、
  Isaac成功、採用範囲確定、教師収集、学習を意味しない。正式再較正前に30 mmの
  三次元距離定義を新しいhash-bound追加契約と承認記録へ固定する必要がある。

### R1名目制御較正v3の正式不合格

#### 2026-08-25

- 承認済み追加契約: 三次元距離30 mm、物理関節範囲の端から1%内側、補正後の
  把持点姿勢・衝突・関節速度再検査を
  `configs/training/order9_r1_nominal_calibration_protocol_v3.yaml`へ固定した。
  30 mmは各軸独立の±30 mmではない。π_Lの仕組みは保持するが動作量は適用せず、
  名目押し込み、QPID/QP、局所サーボ、Isaacを一次判定合格後の正式経路とした。
- 正式結果: 最小の10 mm/5度段階で61件を一次判定し、60件合格、1件不合格。
  不合格は4モジュール機`train-000016-a9b26f370bba`の格子点で、最大機体傾き
  `1.5990674556 rad`（約91.6度）が上限`1.0471975512 rad`（60度）を超えた。
- 原因の分離: 不合格軌道は把持姿勢到達、最終接触数2、最小正規化関節余裕
  `0.010000000999999994`、衝突違反0、最小衝突余裕`0.00480956 m`、最大関節速度
  `0.274444 rad/s`である。したがって30 mm許容や1%補正の失敗ではなく、
  ユーザー指定の極端姿勢排除が決定的だった。
- 停止判断: 格子点100%条件が達成不能になった時点で新規一次判定を停止した。
  一次判定を通過した段階だけをIsaacへ渡す契約に従い、正式Isaac、第2--第4段階、
  未使用14機体での最終確認を実行しなかった。採用範囲なし、正式収集・学習未承認。
- 規範証拠: `for_codex/R1_NOMINAL_CALIBRATION_V3_RESULT_LEDGER.json`、SHA-256
  `a967b68f06ac58265fc526e41b6d255edd82ae2bb9a53d3c91843bb6a2917264`。
  v2結果は履歴として保持し、30 mm/1%補正後の現行判断はv3目録を優先する。
- 次の設計入口: 不合格4モジュール機で傾きが最大となる軌道区間と機体部位を
  読み取り診断する。60度条件の緩和、教師軌道またはIK姿勢選択の変更は、今回の
  結果だけから自動的に行わず、別途承認を必要とする。

### R1教師軌道の修正、同一路径の時間短縮、および名目制御較正v5

#### 2026-08-26

- 教師軌道は学習手法そのものではないため、一次判定に合わない個別軌道をそのまま
  作業停止理由とせず、決定論的な接触面候補と早期傾斜経路から適切な候補を選ぶ
  R1専用処理を追加した。v4では一次判定682/682に合格したが、Isaac確認10回中
  8回成功に留まり不合格となった。v4の結果と教師上書きv1は履歴証拠として固定し、
  後続処理で上書きしない。
- 速度条件を満たしつつ10時間以内で完了させるため、一次判定済みの軌道について、
  幾何学的な経路、終端関節姿勢、接触点割当を変えず、時間配分だけを変更する
  R1専用処理を追加した。接近と接触獲得は元の0.5倍、以後の各区間は0.1倍とし、
  変更後の関節速度を再検査する。同じ形であることを確認できない軌道には、既存の
  衝突判定を引き継がない。
- 正式実行前に代表1件をIsaacで2回確認し、2/2成功、安全違反0、代替制御0を得た。
  この結果は時間短縮経路を正式試験へ入れるための確認であり、22機体全体の較正
  合格を意味しない。
- v5正式結果では、最小の10 mm/5度段階の一次判定682/682が合格した。その後、
  最初の格子点6件を各2回Isaacで実行し、合計12回中6回成功、安全違反6、代替
  制御0だった。格子点100%かつ安全違反0の条件が回復不能になったため、残りの
  第1段階、第2--第4段階、および未使用14機体での最終確認を実行せず不合格とした。
- 全処理は843.054秒で完了し、承認された36000秒上限を満たした。採用範囲はなく、
  教師軌道の正式収集と学習は許可しない。個別の教師軌道不適合を学習手法の失敗とは
  扱わないが、制御層を含む正式試験の安全違反は較正条件を直接不合格にする。
- π_Lの仕組みとC3 promoted update 18は保持したが、v5でもπ_L由来の動作量は
  適用しなかった。名目押し込み、QPID/QP、局所サーボ、決定論的安全制限は有効で
  あり、C3保護成果物は変更していない。
- 現行の規範結果目録は`for_codex/R1_NOMINAL_CALIBRATION_V5_RESULT_LEDGER.json`
  とする。v1--v4は時系列証拠として保持する。次に進むには、名目制御の安全違反を
  制御・接触・軌道の集計結果から切り分け、新しい承認済み較正契約で再評価する必要が
  ある。現時点で教師収集またはπ_H模倣学習を開始してはならない。

### R1退避軌道の原因修正と形態別実行時間（v6）

#### 2026-08-26

- 原因1: 完全タスク軌道生成は`retreat_offset_m`を受け取り、正値か検査した後、
  `amsrr/training/order9_c3_nominal_trajectory.py:817`で値を破棄していた。さらに
  同ファイル902--912行で、退避を10 cmの離隔ではなく接近軌道全体の逆再生として
  作っていた。これは実行時の基準
  `amsrr/simulation/order9_object_task_runtime.py:456-460`が定める
  `root_start - retreat_offset_m`と一致しない。8モジュール機では元の時間配分でも
  退避中に2/2回、QP計算不能となったため、時間短縮だけが原因ではない。
- 原因2: v5は接近・接触獲得を0.5倍、以後を0.1倍にする同一の時間配分を全形態へ
  適用した。関節速度検査と衝突の幾何検査を通っても、閉ループでは3モジュール機の
  支持台衝突、4・6モジュール機のQP計算不能が生じた。したがって、幾何と関節速度
  だけでは時間短縮の許可条件として不十分である。
- 解決: C3共通生成器を変更せず、R1材料化の間だけ退避・静止区間を置き換える。
  解放終了姿勢から世界x負方向へ正確に0.10 mだけ滑らかに動き、開いた関節姿勢と
  静止物体を保ち、移動に対応する速度目標を持つ。接近から把持、接触点割当、解放
  までの軌道は変えない。置換後に解放端との連続性、10 cm離隔、他6姿勢成分不変、
  速度目標、退避後静止を自動検査する。
- 時間配分: 2・5モジュールは接近/接触獲得0.5倍、3・4・6・7・8モジュールは
  0.75倍とし、把持後は全形態0.1倍とする。2--8以外は安全側に拒否する。計算側は
  既存の一括処理を維持し、Isaac同時実行上限を4とする。π_Lは保持するが動作量を
  適用せず、名目押し込み、QPID/QP、局所サーボ、安全制限は有効とする。
- 代表検証: 2--8モジュールから各1候補を選び、一次判定合格後に各2回Isaacで
  実行した。14/14成功、安全違反0、代替制御0、物体落下0、QP計算不能0だった。
  とくに修正前に2/2回QP計算不能だった8モジュール機は、修正後2/2回成功した。
  この結果は退避修正と形態別時間配分の導入根拠であり、22機体×31候補の正式選定
  または14機体での最終確認を代替しない。
- 契約: `configs/training/order9_r1_nominal_calibration_protocol_v6.yaml`と
  `for_codex/R1_NOMINAL_CALIBRATION_PROTOCOL_V6_APPROVAL.json`で実装・C3保護物・
  14件のIsaac証拠をハッシュ固定した。正式教師収集と学習は、v6の4段階正式選定と
  未使用14機体での一度の最終確認が合格するまで許可しない。

### R1完全軌道の意味検査と重要入力反映検査

#### 2026-08-26

- 再発原因: 完全軌道生成器は`retreat_offset_m`を正値として受理・検査しながら
  実際の計算では破棄していた。既存試験は生成された退避軌道の形を固定していたが、
  「指定値を変えれば終点が同量だけ変わる」という外部契約を検査していなかった。
  また、Isaac前の軽量判定は把持獲得までで終わり、解放、退避、静止を含む全8段階を
  対象にしていなかった。この2つが同時に存在したため、未使用入力が正常な成果物と
  して材料化され、重いIsaac実行まで到達した。
- 決定: C3の履歴にハッシュ固定された共通生成器は変更しない。新しいR1 v6候補だけを
  対象に、R1用生成器を関数と識別子で明示選択する専用層を置く。その層が全8段階を
  永続化し、検査記録を候補台帳へハッシュ結合できた場合にだけIsaac入力とする。
- 全8段階検査: 段階順序と境界連続、物体の開始・持上げ・運搬・配置・目標姿勢、
  接触状態、把持中の機体・物体相対姿勢、解放後の正確な退避量と余分な移動の不存在、
  退避速度、最終静止を検査する。不一致は安全側に拒否し、Isaac、制御器、IK再計算、
  軌道最適化を起動しない。
- 重要入力反映検査: 同じ把持軌道を用いて退避量を`0.05 m`と`0.10 m`で二度生成し、
  退避前の6段階が不変で、退避終点だけが`0.05 m`変わることを確認する。これにより、
  入力が単に受理・検査されるだけで出力に使われない実装を、値が偶然合う単一例にも
  依存せず拒否する。
- 簡易確認: 旧C3方式は`E_R1_COMPLETE_TASK_RETREAT_OFFSET`でIsaac前に不合格。
  接続済みR1 v6経路の代表1候補は全8段階合格、指定退避`0.10 m`に対する実測経路長
  `0.10000000000000009 m`、`0.05/0.10 m`入力間の終点差
  `0.050000000000000044 m`だった。検査中のIsaac・制御器・IK再計算・軌道最適化は
  すべて未起動である。証拠は
  `artifacts/p4_full/order9/r1_teacher/diagnostics/r1_complete_task_semantic_gate_smoke_v3/mechanism_smoke_result.json`。
- 境界: これは再発防止機構の実装と軽量確認であり、v6正式4段階較正、14機体での
  最終確認、教師正式収集、π_H模倣学習を完了または承認するものではない。

### R1名目制御較正v6の正式不合格

#### 2026-08-26

- 正式結果: 最小の10 mm/5度段階について、学習側22機体の全31候補、合計682件を
  Isaacなしで一次判定し、682/682件が合格した。完全軌道8段階と重要入力反映検査を
  含み、一次判定に不合格を見落としてIsaacへ渡した事例はない。
- Isaac結果: 25候補を各2回、合計50回実行し、48回成功した。6モジュール機
  `train-000004-190f3a425b3e`の`lattice_02`は2回とも搬送段階で時間切れとなった。
  安全違反、物体落下、QP計算不能、代替制御は0だった。
- 停止判断: 格子点100%条件が回復不能になったため、第1段階の残り、第2--第4段階、
  未使用14機体による最終確認を実行しなかった。採用範囲はなく、正式教師軌道収集と
  学習は許可しない。これは処理の未完了ではなく、承認済み停止条件による正式な
  不合格完了である。
- 実行時間: 最初の正式開始時刻から12348.750秒で判定を完了し、36000秒上限を
  満たした。再開で上限を延長しない実行時間記録、期限時に子処理群も止める処理、
  格子角優先順、既存の決定的不合格を再利用して新規Isaacを起動しない処理を追加した。
  候補集合、合否条件、各候補2回の再現回数は変更していない。
- 制御境界: π_Lの仕組みとC3保護成果物は保持したが、π_L由来の動作量は適用して
  いない。名目押し込み、QPID/QP、局所サーボ、安全制限は有効である。
- 規範証拠: `for_codex/R1_NOMINAL_CALIBRATION_V6_RESULT_LEDGER.json`、SHA-256
  `1da48a4d10224773aeb86461ece665be40cc2cbc044bdbde2b751ceff04a4da9`。
  v1--v5結果は履歴として保持し、現在のR1較正判断にはv6結果を使う。
- 検証: R1全体とC3実行束75/75、関連する到達可能性・教師・C3設定54/54の
  合計129/129単体試験、Python構文、Black、JSON構文、結果目録の参照ハッシュ、
  差分空白検査が合格した。終了時にIsaacまたはR1較正処理は残っていない。
- 次の設計入口: 教師収集ではない。不合格6モジュール機について、搬送段階の
  時間切れを制御追従、接触力、および形態別時間配分から診断する。合否条件の緩和、
  時間配分変更、教師軌道変更は新しい設計判断と承認を必要とする。

### R1最小条件の全件合格と再開時の証拠再利用（v9）

#### 2026-08-28

- 対象範囲: 学習側22機体について、第1段階の物体位置10 mm・向き5度を各31軌道
  （格子27、内部4）で確認した。第2--第4段階と未使用14機体の最終確認は含まない。
- 正式結果: 682軌道を各2回、合計1364回Isaacで再生し、1364/1364成功、
  安全違反0、代替制御0となった。π_Lの実装とC3昇格済みcheckpointは保持したが、
  π_L由来の動作量は全実行で0とした。形態対応の名目押し込み、QPID/QP、局所
  サーボ、決定論的安全制限は有効である。
- 最終修正: 8モジュール機`train-000027-7fe33c662d36`の`lattice_12`は、到達可能な
  把持面[21, 29]を維持し、接触点を高さ25 mm、世界x負方向15 mmへ面内移動した。
  合成移動量は約29.15 mmで30 mm以内である。保存後の軌道で最小幾何余裕
  `0.006803574496524689 m`、最小正規化関節余裕`0.19960983199852517`を再確認して
  からIsaacへ渡し、2/2成功、安全違反0、代替制御0を得た。
- 再発防止: 軌道生成前の中間値ではなく、保存して実行器が読み込む軌道そのものを
  押し込み後まで再検査する。再開時は、単独確認または最大4候補確認の既存合格証拠を
  新規Isaac実行より先に検査・再利用する。これにより、不良な保存済み軌道がIsaacへ
  進む問題と、合格済み候補を再計算する問題を別々に防ぐ。
- 完了監査で、別管理していた姿勢修正9候補の旧証拠のうち8候補が、v9の最大4候補
  条件ではなく9候補同時配置で取得されていたことを検出した。この旧Isaac証拠を正式
  合否から除外し、同じ保存済み修正軌道を現行実行器、3 m間隔、4・4・1候補の3場面で
  各2回再生した。18/18成功、安全違反0、代替制御0、π_L動作量0、QPID/QP・局所
  サーボ指令ありを成果物から再確認した。最終集計器はこの新しい再確認台帳を必須とし、
  旧9候補同時配置証拠だけでは合格台帳を生成できない。
- 規範証拠: `for_codex/R1_NOMINAL_CALIBRATION_V9_RESULT_LEDGER.json`、SHA-256
  `f162ca51cb6a996aba93877dd12d5e9b0c0c2b71a4ed46ede10d47c311e04556`。
  v1--v8は履歴として保持し、第1段階の現在判断にはv9を使う。
- 履歴読込規則: v4--v8の承認済み契約に記録された実装ハッシュは履歴として不変に
  保つ。一方、同じ実ファイル位置にv9の後継実装が置かれた項目だけは、履歴読込時の
  現在バイト一致を要求しない。保存済み証拠、承認記録、C3保護物のハッシュ検査は
  緩めない。現在のv9は、旧姿勢修正証拠の検査器を含む現行実装もハッシュ固定する。
- 境界: v9は第1段階の合格だけを確定し、教師軌道の正式収集または学習を許可しない。
  次の入口は、同じ一次判定・制御・安全・証拠契約を維持した第2段階20 mm/10度の
  契約確認と実行である。第4段階まで選定した後に、未使用14機体で一度だけ最終確認
  し、その合格後にのみ正式教師軌道収集へ進む。

### R1学習側範囲選定v10と最初の不合格段階での停止

#### 2026-08-29

- 選定規則: 合格済み第1段階10 mm/5度を再実行せず、第2段階20 mm/10度、
  第3段階30 mm/15度、第4段階40 mm/20度の順に確認し、最初の不合格段階で停止する。
  採用値は連続して合格した最大段階とする。各Isaac場面は最大4候補、機体間隔3 m、
  各候補2回とし、同時配置での不合格は単独で再確認する。
- 軽量判定: 各新規軌道は、Isaac、π_L、QPID/QP、局所サーボ、標本探索を使わず、
  完全追従を仮定して物体・自己・支持台干渉、関節余裕、極端姿勢、全8段階の意味を
  先に検査する。第2段階の最外側176格子点は176/176合格した。
- 個別教師修正: 8モジュール機`train-000027-7fe33c662d36/lattice_08`は、把持面
  [19, 22]を維持し、接触点を高さ20 mm、世界x正方向15 mm、y負方向15 mmへ移して
  一次判定に合格させた。三次元移動距離は約29.15 mmで、ユーザー承認済みの
  30 mm以内である。高さと平面内補正を合成した距離を機械検査し、別々なら上限内でも
  合成30 mm超となる設定を拒否する。候補は有限個に限定し、重い標本探索は一次判定で
  無効にした。
- 正式停止結果: 第2段階の最大4候補Isaac確認で安全不合格が生じたため、
  2モジュール機`train-000000-5fecfad4f44f/lattice_02`を単独で2回再確認した。
  0/2成功、安全不合格2、代替制御0となり、格子点100%・安全不合格0の条件が
  回復不能になった。第2段階を不合格とし、第3・第4段階は実行しなかった。
- 採用範囲: 第1段階10 mm/5度。π_Lの仕組みとC3 promoted update 18保護物は保持するが、
  π_L actor commandは0とした。名目押し込み、QPID/QP、局所サーボ、安全制限は有効。
- 実行短縮: 一次判定は格子角を先に処理し、既存のハッシュ一致証拠を再利用する。
  有効な最大4候補不合格を再開時に再実行せず単独確認へ直送し、単独格子不合格が
  1件確定した時点で、未開始候補を起動しない。
- 再現性: 同じ有限候補が別処理で異なる結果になった事例を正式採用せず、候補準備前に
  Pythonハッシュ乱数を0へ固定して実行器自身を再起動する。固定値はv10契約へ含め、
  固定後に同じ30 mm以内候補が再現して一次判定へ合格することを確認した。
- 規範証拠: `for_codex/R1_RANGE_SELECTION_V10_RESULT_LEDGER.json`、SHA-256
  `820f6ae76e7f021e82831b28f6ec3a580b25fdb510fd4a148ab6c150304bb253`。
  第1段階の親証拠はv9目録を使い、v1--v9の履歴結果を上書きしない。
- 境界と次の入口: 選定用証拠は学習対象外。未使用14機体の最終確認、正式教師軌道
  収集、π_H模倣学習、π_L再学習は未実行である。次は採用範囲10 mm/5度を未使用
  14機体で一度だけ最終確認し、合格後にのみ正式教師軌道収集へ進む。

### R1名目物体高さ、固定支持台の単一入力、およびSTL事前検査（v11）

#### 2026-08-29

- 原因: v10で不合格となった小物体は高さ約149 mmで、把持中の機体電池下面と
  固定支持台上面がほぼ同じ高さだった。またTaskSpecは2 m角・高さ50 mmの支持台を
  記述する一方、Isaacはhash-bound Order-8報告の0.55 m x 0.36 m x 0.15 m支持台を
  使用しており、教師生成・事前検査とIsaacの入力が二重化していた。保存後の8段階
  意味検査も、STL実形状による支持台距離を必須としていなかった。
- 決定: 各C3由来物体の高さを40 mm増やし、開始・目標中心を20 mm上げる。物体下面、
  平面内姿勢、質量、運搬変位を維持し、密度・慣性を再計算する。側面把持点は最低
  20 mm上げる。元の小物体は難条件として保存するが正式範囲選定・収集には使わない。
- 支持台: Order-8報告SHA-256
  `d0f75cca2ae540c79971766ab722d4530dd4fb44842276256bac40aafdb8cc49`
  の支持台寸法・姿勢をTaskSpecへ書き戻し、教師生成、事前検査、Isaacの一つの入力と
  する。不一致は教師生成前に拒否する。C3保護実行器自体は変更しない。
- 事前検査: 保存済み全8段階について、URDF参照STL三角形、物体・支持台の実寸箱、
  関節可動域1%余裕、機体傾き60度、把持点位置30 mm・姿勢1度を検査する。自己・物体・
  地面は実交差を禁止し、支持台には19.5 mm以上の距離を要求する。不合格は最初の時点で
  停止し、Isaac、制御器、軌道最適化を起動しない。
- 代表結果: v10不合格候補の元小物体軌道は、Isaacなしで設置段階の時点25に不合格。
  電池1/2と支持台の距離は2.971268/7.985864 mmだった。補正後は8段階458時点、
  844形状判定に合格し、Isaacも2/2成功、安全不合格0、代替処理0、落下0、時間切れ0、
  QP計算不能0だった。π_L動作量0、名目押し込み・QPID/QP・局所制御は有効。
- 境界: これは代表1候補の修正確認であり、正式範囲選定、未使用14機体最終確認、
  正式教師収集、π_H模倣学習を完了しない。小物体v10の採用範囲を補正後物体へ流用せず、
  次はv11契約で学習側第2--第4段階を正式にやり直す。
- 証拠: `for_codex/R1_NOMINAL_GEOMETRY_V11_SMOKE_RESULT_LEDGER.json`、SHA-256
  `57862dad9993af5e833850560dd5a91d56c6b7df76def4053eed7e3f9d98512f`。

### R1 v11正式範囲選定と未使用14機体確認の不合格

#### 2026-08-29

- 選定結果: 補正後名目物体について、22機体側の第2段階20 mm/10度を開始した。
  6モジュール機`train-000004-190f3a425b3e`の格子点8は、回転に耐える把持教師軌道を
  一次判定内に構成できなかった。格子点100%条件により第2段階を不合格とし、第3・
  第4段階は停止規則に従って実行しなかった。
- 未使用側結果: 親結果v9の第1段階10 mm/5度を未使用14機体へ一度だけ適用した。
  5モジュール機`validation-000038-e6065f5b3c3e`の格子点0は、配置段階の時点24で
  機体と固定支持台の最小間隔18.085631664698426 mmとなり、要求19.5 mmを満たさず
  一次判定で不合格となった。
- 教師修正: 接触高さ15/25 mm、物体高さ増分50 mm、さらに接触高さ1--30 mmの有限
  比較を行った。別姿勢分枝は要求間隔を回復せず、実衝突する案もあったため、修正を
  正式採用しなかった。比較中もIsaac、π_L、QPID/QP、局所サーボは起動していない。
- 設計判断: 補正後名目物体の採用範囲はなし。正式教師収集とπ_H模倣学習は開始
  しない。次は合否値を緩めるのではなく、配置時の固定支持台間隔を教師軌道生成の
  目的または制約へ直接含める。新方式は別契約として承認後に同じ14機体確認をやり直す。
- 規範証拠: `for_codex/R1_RANGE_SELECTION_V11_RESULT_LEDGER.json`、SHA-256
  `5ca0d7cc1d2cc7fd3b6980cdf75e813b19adcafd177dbd0718983593b9f56393`。
- 保護境界: C3 promoted update 18、π_L checkpoint、保護実行器、release ledgerは
  変更しない。今回の一次判定証拠は学習対象外である。

### R1支持台間隔の合否値・生成目標分離と生成証明（v12）

#### 2026-08-29

- 原因: v11は教師軌道を生成した後に19.5 mm条件で検査していた。合否値と生成時の
  目標値が分かれておらず、配置軌道は把持姿勢から決定論的に作った後の検査に依存した。
  そのため、支持台余裕を満たす別接触組を選ぶ機構がなく、今回の不合格を生成前に
  防げなかった。
- 決定: 正式合否値19.5 mm、生成・候補選択目標30.0 mm、最低余裕10.5 mmをR1専用
  設定へ分離して固定する。生成目標が合否値と最低余裕の和を下回る設定は処理開始前に
  拒否する。判定値19.5 mm自体は緩和しない。
- 生成条件: 全8段階を30.0 mm条件で材料化・実形状検査し、合格候補だけに設定SHA、
  全段階、最小距離、候補順位、生成法を結合した証明書を付ける。証明書がない従来の
  配置姿勢コピーは、Isaac材料化前に拒否する。
- 有限修正: 既存の決定論的逆運動学解を変えない剛体並進として、接触座標を5 mm刻み、
  最大30 mmまで上げた候補を検査できる。これで合格しなければ有限個の別接触組を同じ
  30.0 mm条件で選び直す。重い逆運動学を30.5 mm条件で繰り返す方式は採用しない。
- 簡易確認: v11不合格の未使用5モジュール機格子点0では、元候補と上方移動4候補は
  不合格だったが、別接触組が全8段階、2528時点、4444形状判定、最小距離30.0 mmで
  合格した。所要456.447秒のうち支配項は元の教師・逆運動学生成であり、Isaac、π_L、
  QPID/QP、局所サーボは起動していない。
- 境界: これは設計・実装と既知1件の機構確認であり、正式範囲選定、未使用14機体の
  最終確認、教師軌道正式収集、π_H模倣学習を完了しない。次はv12をhash-boundな正式
  実行契約へ結合し、22機体側の範囲選定から再開する。

### R1同一候補のMuJoCo速度・結果対応診断

#### 2026-08-31

- 目的: R1の大量物理確認をMuJoCoへ移した場合の速度を、保存済みIsaac成功候補1件と
  同じ初期状態、目標軌道、機体、物体、支持台、制御周期、衝突対象および合否値で確認する。
- 実装: 元STLの頂点を間引かない部品別凸包URDF、同じ印加値による物理速度分離再生、
  MuJoCo状態を現行QPID/QPへ戻す閉ループ再生、Isaac軌道誤差・接触・段階合否の比較器を
  追加した。C3保護実行器とcheckpointは変更していない。
- 速度結果: 73.72秒・3686制御周期の同じ印加値再生は13.965秒、実時間の5.279倍だった。
  保存Isaacの8環境合計処理率62.394環境制御周期/秒に対し、MuJoCoは263.947で4.230倍。
  閉ループ単体CPU版はPython/PyTorchの逐次機体集約・QPが支配し114.044秒だった。
- 結果対応: Isaacは成功、MuJoCo閉ループは接近・接触獲得後、搬送中に接触を失い不合格。
  最終物体位置はIsaacから0.191 mずれた。現変換をIsaacの正式合否判定へ置換しない。
- 決定: MuJoCo物理の速度利点は確認済みとするが、正式使用前に接触取得から搬送の短区間で
  接触材料、接触力、物体相対姿勢を合わせる。その後、QPID/QPの一括処理またはコンパイル
  済み単体経路を用意し、物理以外の逐次処理を速度測定から除去する。
- 境界: R1範囲、未使用14機体、正式教師収集、π_H学習の状態は進めない。本診断をC3または
  R1の正式成功証拠へ昇格しない。

### R1名目軌道の平面剛体移送による30 mm・15度代表診断

#### 2026-08-31

- 決定: 物体の平面位置・向きだけが異なるR1候補では、既に成功した名目中央軌道の
  物体、機体姿勢、自由接触点、重心目標を各時点の物体中心まわりに同じ平面変換で
  移す。関節位置・速度、接触割当、物体相対経路は変えない。逆運動学、軌道最適化、
  制御層、Isaacを使わず、保存前に変換不変量、支持台および自己干渉を検査する。
- 対象: 6モジュール機`train-000004-190f3a425b3e`の名目中央`lattice_13`
  （位置0 mm、向き0度、現行単独Isaac確認2/2成功）を、第3段階`lattice_08`
  （x=-30 mm、y=+30 mm、yaw=+15度）へ移した。近傍の20 mm・10度軌道を使った
  v1/v2診断は「名目中央」の条件を満たさない履歴であり、現行判断にはv3だけを使う。
- 変換確認: 全8段階796時点、自由接触点1592姿勢を検査した。関節位置・速度と接触
  割当は完全一致し、物体相対位置誤差の最大値は`7.216449660063518e-16 m`、姿勢誤差は
  `2.9802322387695312e-08 rad`だった。対象物体の写像誤差は位置
  `2.7755575615628914e-17 m`、姿勢`0 rad`だった。
- 結果: 接近段階の時点374で機体と固定支持台の最小間隔が
  `0.0117826335 m`となった。正式合否`0.0195 m`も生成目標`0.0300 m`も満たさない
  ため一次判定で棄却し、Isaac、QPID/QP、局所サーボ、π_L動作量を起動しなかった。
  平面剛体移送は教師生成失敗を回避したが、この候補の固定支持台余裕は解決しない。
- 証拠: `artifacts/p4_full/order9/r1_teacher/diagnostics/`
  `r1_l3_planar_nominal_transfer_v3/`
  `r1_l3_30mm_15deg__train__train-000004-190f3a425b3e__lattice_08__lightweight_rejection.json`、
  SHA-256 `85c21610c54e21179cf07ca1119fa36a284a74c510d1fe7938d5133436ac832d`。
- 境界: これは診断1件であり、第3段階の正式合否、22機体の範囲選定、未使用14機体
  最終確認、教師収集、π_H模倣学習を進めない。C3 promoted update 18の保護物は変更しない。

### R1支持台同伴平面移送による30 mm・15度代表診断

#### 2026-08-31

- 訂正: 直前のv3診断は物体、機体姿勢、自由接触点および重心目標だけを移し、支持台を
  世界座標に固定した。その結果は履歴として残すが、物体と支持台の相対条件を保存する
  今回の問いへの回答には用いない。現行v4では支持台、物体、目標、機体姿勢、自由接触点、
  重心目標および初期化状態を、全時点で同じ一つの平面剛体変換により移す。
- 参照: 同じ6モジュール機`train-000004-190f3a425b3e`、同じ格子方向の第2段階
  `lattice_08`（x=-20 mm、y=+20 mm、yaw=+10度、現行Isaac確認2/2成功）を使い、
  さらにx=-10 mm、y=+10 mm、yaw=+5度だけ移して、第3段階`lattice_08`
  （x=-30 mm、y=+30 mm、yaw=+15度）を構成した。旧物体形状の第1段階軌道は混ぜない。
- 一次判定: 全8段階773時点、自由接触点1546姿勢を検査した。関節位置・速度、接触割当は
  完全一致し、物体相対位置誤差の最大値は`9.036560719766055e-16 m`、姿勢誤差は
  `2.9802322387695312e-08 rad`だった。現行v14支持台余裕証明をハッシュ、意味指紋、
  全8段階・773時点について再検証し、平面剛体変換が距離を保存することから最小間隔
  `0.030 m`を継承した。最小関節余裕は可動範囲の`0.273061833462875`だった。
- Isaac結果: 診断専用の単独1回を実行し、3683制御周期で成功した。安全不合格、代替制御、
  硬い衝突、物体落下、時間切れはいずれも0で、最終段階8へ到達した。実行記録の
  `r1_scene_support_from_task=true`により、固定の標準支持台ではなく、変換後TaskSpecの
  支持台位置・姿勢をIsaacへ渡したことを固定した。π_Lの学習済み動作量は適用していない。
- 証拠: `artifacts/p4_full/order9/r1_teacher/diagnostics/`
  `r1_l3_planar_scene_transfer_v4/r1_l3_30mm_15deg__train__`
  `train-000004-190f3a425b3e__lattice_08/isaac_one_replay/`
  `r1_l3_30mm_15deg__train__train-000004-190f3a425b3e__lattice_08/isaac/`
  `evaluation_episodes.jsonl`、SHA-256
  `239f074905a2e54c85e0fedf8b43d92dd2d94063eb5f6854f7883ad8c23be6da`。
  元の実行記録`evaluation_rollout.pt`のSHA-256は
  `8eee82b5762caec26dace69b88ddb5da83b3560d59ebd8707b0a28bcc3946635`。
- 境界: これは指定された失敗例1件への診断であり、第3段階全体の正式合格、22機体の
  範囲選定、未使用14機体の最終確認、正式教師収集、π_H模倣学習を進めない。生成物は
  `training_eligible=false`かつ正式昇格証拠対象外であり、C3 promoted update 18の
  保護物も変更しない。

### R1場面全体の平面剛体移送による範囲再選定v15

#### 2026-08-31

- 設計判断: 同じ機体、物体、支持台およびタスクで、物体の平面位置とyawだけを変える
  候補では、支持台、物体、目標、機体姿勢、自由接触点、重心目標および初期化状態を
  同じ一つの平面剛体変換で移す。関節位置・速度、接触割当、物体相対軌道は変えない。
  時間方向に二段階へ分ける手法は使用しない。
- 選定方法: 20 mm・10度の22機体、各31候補について、現行Isaac証拠2回を全件再検査
  する。30 mm・15度と40 mm・20度では、同一機体の20 mm・10度成功候補31件から
  実際の物体姿勢が最も近い軌道を選び、全8段階について場面変換、関節値、接触割当、
  物体相対軌道および衝突距離の保存を検査する。Isaacは最大範囲についてモジュール数
  2--8から各1機体だけを単独1回実行し、実装上の数値誤差を確認する。
- 事前拒否: 4個の内部標本は段階ごとに独立生成される。同じ番号を機械的に対応させた
  最初の試行は移送量上限検査がIsaac前に拒否した。現行処理は実姿勢による最近傍対応を
  必須とし、30 mm・15度への追加移送を最大約14.14 mm・5度、40 mm・20度を最大約
  28.28 mm・10度に制限する。拒否された生成物は削除せず別場所へ保存する。
- 全件結果: 20 mm・10度は682候補、1364/1364成功、安全不合格0、代替制御0。
  30 mm・15度と40 mm・20度はいずれも682/682の場面変換証明に合格した。物体相対
  位置誤差はそれぞれ最大`1.4043333874306805e-15 m`、
  `1.7798229048217483e-15 m`、姿勢誤差はいずれも最大
  `4.2146848510894035e-08 rad`、最小衝突距離はいずれも`0.030 m`だった。
- Isaac結果: 40 mm・20度の代表7機体は7/7成功した。安全不合格、代替制御、硬い衝突、
  物体落下、時間切れはすべて0。TaskSpecと実行記録の支持台姿勢が一致した。π_Lの
  学習済み動作量は0とし、QPID/QPと局所サーボは有効にした。
- 採用範囲: 今回用意した最大段階である、前後・左右それぞれ±40 mm、yaw ±20度を
  22機体側の選定結果とする。場面全体移送は平面剛体変換に対して同値なので、この値を
  物理的な到達限界とは解釈しない。また、今回未検査の範囲まで合格したとは主張しない。
- 規範証拠: `for_codex/R1_SCENE_INVARIANT_RANGE_SELECTION_V15_RESULT_LEDGER.json`、
  SHA-256 `df930a754874f5f56f81b89ed144250e4a482fb13d38facfd507e438abdcbadf`。
  主成果物`range_selection_result.json`のSHA-256は
  `ce20fe7ade3321459586a66c5780b55f6f05b4f33d8202d2719e646d56b7f497`。
- 境界と次の入口: これは22機体側の範囲選定だけを確定する。候補修正に未使用の14機体
  による最終確認、正式教師軌道収集、π_H模倣学習、π_L再学習は未実行・未許可。
  C3 promoted update 18のcheckpoint、保護実行器、学習工程設定およびrelease ledgerは
  変更しない。

### R1未使用14機体の場面全体移送・最終確認v16

#### 2026-09-01

- 対象集合: 過去の候補修正に使われた旧`validation` 14機体を正式証拠へ再利用せず、
  `morphology_pool.json`の`held_out` 14機体を使った。モジュール数2--8を各2機体とし、
  過去R1成果物のパスに対象構造ハッシュがないことを実行前に監査した。
- 手順: 各未使用機体で中央基準軌道を新規生成し、全8区間、関節余裕、干渉、支持台
  30 mm余裕を検査した。14/14合格後、物体と支持台を含む場面全体を選定済み範囲
  `±40 mm`・`±20度`の31条件へ移し、14×31＝434条件を一次判定した。成果物構築上の
  2分割は最大10度の内部検査用で、時間方向の二段階動作ではない。
- 一次判定結果: 434/434合格。関節目標と接触割当を保存し、物体相対位置最大誤差
  `1.344427481423755e-15 m`、姿勢最大誤差`2.9802322387695312e-08 rad`、最小衝突
  余裕`0.030 m`を確認した。Isaac前の完全追従幾何判定だけでは物理接触成功を保証
  しないことを、後段結果と明確に分離する。
- Isaac結果: 各機体の対角格子点0・26を1回ずつ、1起動1条件、計28件実行した。
  22/28成功、安全不合格6、重大衝突4、物体落下2、時間切れ0、代替制御0だった。
  6モジュール`58a5c015f4db`は搬送中落下、7モジュール`ba94152a03e4`は開始区間の
  重大衝突、8モジュール`b6b88f01296d`は接触獲得区間の重大衝突が各2件生じた。
- 決定: v15の`±40 mm`・`±20度`は未使用機体で不合格であり、R1一般化範囲として
  確定しない。今回未実行の20 mm・10度または30 mm・15度へ自動的に縮小もしない。
  正式教師軌道収集とπ_H模倣学習は開始しない。
- 未使用性の扱い: 今回の14機体は最終判定へ使用済みである。失敗診断や手法修正には
  使用できるが、修正後に同じ14機体を「未使用」と呼んで正式確認へ再利用しない。
  修正後の一般化確認には別の未使用集合が必要である。
- 制御境界: π_Lシステムは保持したが学習済み動作量は0。QPID/QP、局所サーボ、
  名目押し込み、安全制限は有効。C3 promoted update 18保護物は変更していない。
- 規範証拠: `for_codex/R1_HELD_OUT_SCENE_CONFIRMATION_V16_RESULT_LEDGER.json`、
  SHA-256 `fb3a8083bdbdbbc3891bfc2da6b316ca180e023682b55f74656f83521b3612c8`。
  次の入口は、3失敗機体について重大衝突と搬送時把持力低下を分離する短区間診断である。

### R1 7モジュール衝突2件の把持点変更による解消（v17診断）

#### 2026-09-03

- 対象: v16で開始区間の重大衝突となった7モジュール機体
  `ba94152a03e4`の対角格子点0・26。物体、支持台、タスク、機体構造および場面全体の
  平面移送条件は変更していない。
- 原因と決定: 把持面の組`(12, 21)`で得た姿勢自体がこの機体では衝突しやすく、衝突後も
  その姿勢を保持する局所的な関節回避は推力配分不能を起こした。未使用の把持面の組を
  幾何判定し、同じ候補群で成立する`(13, 21)`を選択した。実行時だけ関節を曲げる回避、
  二段階接近および自由関節の強制目標は最終採用結果では使っていない。
- 軌道条件: 接近前距離80 mm、支持台との検査余裕10 mm、接触点高さ補正50 mm、
  世界座標Y方向の逃げ20 mm、接近時間倍率0.75。Isaac前の全区間検査を通した軌道だけを
  実行した。
- Isaac結果: 格子点0・26とも`task_success=true`。重大衝突、推力配分不能、物体落下、
  時間切れ、安全不合格および代替制御はすべて0で、最終区間8まで到達した。実行は各
  2671環境ステップだった。
- 制御境界: π_Lの学習済み動作量は0のまま、QPID/QPと局所サーボは有効。C3 promoted
  update 18のcheckpoint、保護実行器、release ledgerおよび合否基準は変更していない。
- 証拠: `artifacts/p4_full/order9/r1_teacher/diagnostics/held_out_collision_avoidance_v17/result.json`
  （SHA-256 `e8a6271daa9eef56e9758609023ed345a7327fbbf320c866f0ad6404520f22f3`）。
  これは使用済み機体に対する原因修正の診断証拠であり、`training_eligible=false`、
  `formal_teacher_collection_authorized=false`を維持する。別の未使用集合による正式な
  一般化確認を代替しない。

### R1 6モジュール物体落下2件の左右別押し込み・低高さ搬送による解消（v18診断）

- 対象: v16で物体落下した6モジュール機体`58a5c015f4db`の格子点0・26。同じ物体、
  支持台、機体構造、把持点および合否基準を維持した。
- 原因と決定: 両把持点へ同じ追加押し込みを与えると、異なるDock機構のてこ比と追従差に
  より実把持力が偏った。10/30 mmでは保持できたが物体が約30度傾いたため不採用とし、
  17/23 mmへ差を縮めた。把持全体を物体座標+Yへ30 mm移し、持ち上げ生成高さを
  300 mmから必要十分な30 mmへ縮め、持ち上げ時間を3倍にした。
- 実装境界: 左右別押し込み、物体相対の把持中心補正、時間倍率および高さ倍率は、単独1回の
  R1診断にだけ許可する。軌道参照と工程終端目標を同じ高さ倍率で変更し、制御器内部だけを
  変更して判定目標が残る不整合を禁止する。
- Isaac結果: 格子点0は4096制御周期、格子点26は4100制御周期で最終段階8へ到達し、
  2/2成功した。硬い衝突、物体落下、推力配分不能、時間切れ、安全不合格、代替制御は0。
- 制御境界: π_Lの学習済み動作量は0。QPID/QPと局所サーボは有効。C3 promoted
  update 18のcheckpoint、保護実行器およびrelease ledgerは不変。
- 証拠: `for_codex/R1_HELD_OUT_OBJECT_DROP_REPAIR_V18_RESULT_LEDGER.json`。
  整形・単体試験後の最終コードで2件を連続再実行した集計結果は
  `artifacts/p4_full/order9/r1_teacher/diagnostics/held_out_object_drop_repair_v18_formatted_acceptance/result.json`
  （SHA-256 `cf306f1b7d9bc9f9cb449ada6af5252ab9e6029eada89332931916094398f5a3`）。
  使用済み機体の診断であるため、`training_eligible=false`、
  `formal_teacher_collection_authorized=false`とし、新しい未使用集合による正式確認を
  代替しない。
### R1正式教師軌道収集の直前準備v19

#### 2026-09-05

- 決定: R1教師生成器そのものに、未使用形態への一般化を要求しない。教師は決定論的で
  あり、個別の把持点選択や衝突回避を使ってよい。未使用の物体姿勢条件は、収集した
  教師から学ぶπ_Hの最終評価へ残す。一方、収集へ入れる各軌道は一次判定とIsaacの
  全工程成功を必須とし、不合格軌道を学習対象へ混ぜない。
- 収集条件: 140件、モジュール数2--8ごとに学習14・検証3・最終評価3、乱数
  `29009`--`29148`。前後・左右`±40 mm`、yaw `±20度`以内で、支持台を含む場面
  全体を同じ平面剛体変換で移す。全件でタスク、乱数、姿勢条件ハッシュを一意にする。
- 収集元: 学習分割はv15で全件成功した22機体を使う。検証・最終評価分割は、v16の
  対角2条件がどちらも成功した11機体を使う。v16で不合格となった3機体のv17/v18
  修正は診断専用のままであり、正式契約へ黙って昇格させない。
- 実行境界: 全8区間の一次判定を通過した軌道だけをIsaacへ送る。π_L学習済み補正は
  使用せず、QPID/QP、局所制御、名目押し込みを使用する。Isaac成功、安全不合格0、
  代替制御0、重大衝突・落下・QP計算不能・時間切れ0を満たした記録だけを学習候補に
  する。
- 起動保護: 収集実行器は条件表と事前検査をハッシュで固定し、別の明示的な開始承認
  記録がない限り起動を拒否する。今回の準備では収集・Isaac・学習を開始しない。
- 事前結果: 140件の条件表、入力・C3保護物、359.24 GiBの空き容量、Isaac用Python、
  CUDA、同時収集処理なしを確認した。代表1件の材料化と一次判定は成功し、支持台余裕
  30 mmを確認した。現在状態は`ready_awaiting_explicit_launch`である。
- 証拠: `for_codex/R1_TEACHER_COLLECTION_PREPARATION_V19_LEDGER.json`。

### R1正式教師軌道収集v19の識別子整合性と即時停止契約

#### 2026-09-05

- 初回起動では、派生した軌道一式の識別子を`r1-teacher-*`へ更新した一方、TaskSpecの
  `order9_rollout_bucket_id`だけが収集元の識別子を保持していた。C3名目軌道読込検査が
  物理実行前に不一致を拒否したため、採用された教師記録は0件だった。
- 修正契約では、TaskSpecの`order9_rollout_bucket_id`、
  `r1_calibration_candidate_id`、case manifestの`candidate_id`を同じ収集episode識別子へ
  一元化する。実行命令を作る前にも3値の一致を必須検査する。
- 初回の不合格材料・実行命令・ログは上書きせず保持する。修正版は`identity_v2`の別領域へ
  保存し、開始承認には条件表、事前検査、実行器、wrapperおよび初回不合格証拠のハッシュを
  記録する。
- 並列実行は全140件を先に待ち行列へ投入しない。同時実行数だけを投入し、いずれか1件が
  異常終了した時点で未開始分を取り消す。修正版の最初の1件を単独でIsaac実行し、成功、
  安全不合格0、代替制御0を確認してから残り139件を開始した。
- この変更は収集時の識別子と失敗時停止の補強であり、共有schema、教師軌道、合否基準、
  π_L/QPID/QP/局所サーボ境界およびC3 promoted update 18を変更しない。
- 実行結果: 120件を実行し119件を正式保存した。7モジュール検証条件
  `r1-teacher-000116`は設置区間で時間切れとなり、重大衝突、落下、QP不能、安全不合格、
  代替制御は0だった。実行器は未開始20件を投入せず停止した。この観測だけから法線力を
  直接原因とは断定せず、再開前に短区間診断を必要とする。

### R1正式教師軌道収集v19の完了と不合格軌道の置換規則

#### 2026-09-06

- 設計判断: 正式収集中に物理不合格となった決定論的教師軌道は、合否条件を緩めたり
  不合格記録を学習対象へ混ぜたりしない。同一機体・同一物体・同一把持面について
  Isaac成功済みの別の基準軌道がある場合、その関節軌道と物体相対の接触条件を維持し、
  物体、支持台、目標、機体姿勢、自由接触点、重心目標を目的条件へ同じ平面剛体変換で
  移す。向きの差が10度を超える場合は10度以内の複数の材料化段階へ分け、各段階で
  全8区間の軽量検査を行う。これは時間方向に複数動作を追加することを意味しない。
- 個別修復: `r1-teacher-000116`は設置時の物体姿勢ずれに対して左右別押し込み、把持中心、
  低い持ち上げ軌道を適用した。8モジュールの`000121`、`000127`、`000131`は、同じ
  `train-000027-7fe33c662d36`で2回成功済みの`lattice_12`を基準とする場面全体移送へ
  置換した。`000127`の元軌道は持ち上げ開始1.94秒後に非把持部
  `module_3__yaw_dock_mech2`が支持台へ接触しており、置換後はその衝突が0になった。
- 証拠分離: 元の不合格4件と`000121`の不合格診断試行は削除せず診断履歴として保持し、
  正式140件には含めない。正式記録は修復元、軽量検査、実行命令、生のIsaac記録、判定
  記録のハッシュを保持する。
- 最終結果: 2--8モジュール各20件、合計140/140件を正式受理した。各モジュールの分割は
  学習14、調整3、最終確認3である。重大衝突、物体落下、QP計算不能、時間切れ、安全
  不合格、代替制御は正式受理記録の全件で0。物体姿勢条件、生記録、判定記録は各140件
  すべて一意である。
- 制御境界: π_Lの学習済み補正は全件で無効、QPID/QPと局所サーボは全件で有効。
  C3 promoted update 18のcheckpoint、保護Isaac実行器、release ledgerおよび合否基準は
  不変。教師収集完了はπ_H模倣学習の開始許可を含まない。
- 規範証拠: `for_codex/R1_TEACHER_COLLECTION_V19_RESULT_LEDGER.json`を現行の正式結果
  台帳とする。次の入口は、140件からπ_H入力・教師出力を構築する学習前検査であり、
  模倣学習の開始は別の明示指示を必要とする。
