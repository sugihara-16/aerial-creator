# Order-9 R1 reachable-pose calibration protocol v1 proposal (approved)

## Status and decision boundary

- Status: **APPROVED BY THE REPOSITORY USER ON 2026-08-24**
- Scope: determine a bounded `reachable_pose_expansion` envelope before R1
  teacher-trajectory collection.
- This file and
  `configs/training/order9_r1_calibration_protocol_v1_proposal.yaml` are the
  retained review inputs. The executable authorization is the hash-bound pair
  `configs/training/order9_r1_calibration_protocol_v1.yaml` and
  `for_codex/R1_CALIBRATION_PROTOCOL_V1_APPROVAL.json`.
- Approval authorizes only the specified calibration procedure. It does not
  by itself assert a calibrated range, authorize formal collection before an
  accepted distribution manifest exists, authorize imitation learning, or
  permit modification of the promoted C3 release. Changing any parameter
  requires a new protocol version and approval.

## Execution outcome

The approved v1 procedure has now completed with status `rejected`. The first
mandatory L1 lattice candidate passed deterministic teacher/IK generation and
the exact-tracking screen, then failed both protected-C3 full-layer Isaac
replays with `hard_collision` safety failures. Because L1 requires 100% lattice
success and zero safety failures, no remaining case can restore a pass. The
monotonic fail-fast rule stopped the remaining L1 cases and all higher levels;
the untouched 14-bucket confirmation was not entered. No R1 envelope,
collection authority, or training authority was published. The authoritative
result is `for_codex/R1_CALIBRATION_V1_RESULT.md`.

## Evidence basis

- v0.5 Section 24.5.9 and the current curriculum define R1 as
  `reachable_pose_expansion`, but do not approve numeric x/y/yaw bounds.
- The protected C3 source is release
  `c3_pi_l_promoted_update18_v1`, checkpoint SHA-256
  `6ea412ccdfe983cb2b030b3514bc8982424522673b49def188d547120984357b`.
- The hash-bound accepted C3 bucket manifest contains 36 topology buckets:
  22 `train` and 14 `validation`. Both splits independently cover module
  counts 2--8. The manifest SHA-256 is
  `ee0001c7c7bc9f5d71ad08219ae73907b5ef4022f82fd90505e3c4642ba291a6`.
- Those C3 buckets exercise approximately +/-5 mm planar placement and
  +/-2 degrees yaw. They establish the conservative starting envelope, not an
  R1 range.
- Existing task support geometry leaves only a narrow longitudinal
  full-footprint margin at the initial and goal poses. Therefore support
  geometry/pose and robot reset remain fixed during calibration. A wider
  level may legitimately fail because of contact/support physics; the
  protocol must record that failure rather than translating or resizing the
  support.

## Proposed distribution contract

R1 changes only the target object's initial planar position/yaw and applies
the same SE(2) transform to its pose goal. The coordinate frame is `world`,
the envelope center is the base task object's world pose, and x, y, and yaw
interior values are sampled independently and uniformly. Object geometry,
mass, CoM, inertia, friction, support geometry/pose, robot reset, relative
transport, morphology, C3 `pi_L`, QPID/QP, and local-servo contracts remain
unchanged.

The spans below are additional residual offsets around each source bucket's
base object pose; they are not absolute coordinates around one shared nominal
pose. Because the C3 source buckets already differ by about +/-5 mm and
+/-2 degrees, their outer extent relative to the shared conservative center
can reach approximately 15/25/35/45 mm and 7/12/17/22 degrees respectively.

| Level | residual x half-span | residual y half-span | residual yaw half-span |
| --- | ---: | ---: | ---: |
| L1 | 10 mm | 10 mm | 5 degrees |
| L2 | 20 mm | 20 mm | 10 degrees |
| L3 | 30 mm | 30 mm | 15 degrees |
| L4 | 40 mm | 40 mm | 20 degrees |

Each level is a complete symmetric rectangular envelope. Per bucket it uses
the full 27-point lower/center/upper lattice: center, six face centers, twelve
edge centers, and eight corners, plus four deterministic pseudo-random
interior points. Every teacher-feasible case first receives the fast
exact-tracking screen defined below. Only a screen pass receives two
real-Isaac full-layer replays. A teacher-generation or screen failure is
retained as a failed attempt and is not replayed. Case order is lexicographic
by level, split, bucket id, sample kind/index, and replay index. Interior
samples use the adapter's versioned derived-seed rule from calibration seed
`19009`.

## Mandatory fast exact-tracking screen

Every newly generated candidate in both calibration and formal collection is
screened before C3 `pi_L`, QPID/QP, local servo, or Isaac may run. The screen
uses the already generated dense deterministic-teacher plus IK path only; it
does not rerun IK, search or optimize a trajectory, step a controller, or
start Isaac. Consequently it is a CPU-side linear scan over existing values
and evidence rather than another planning or physical-simulation stage.

The screen covers `approach` and `contact_acquisition` and stops at the first
two-contact maintained grasp pose. It verifies:

- raw/resolved trajectory and resolver-evidence hash identity;
- complete, finite joint position/velocity values and continuous window
  chaining;
- the resolver's already recorded convex collision acceptance at every dense
  knot, covering self-collision, unintended object collision, support, and
  ground while retaining only the selected grasp contacts as allowed;
- non-negative joint-rate margin and a minimum joint-limit reserve of 1% of
  each joint's authored range, so a path that rides a hard stop is rejected;
- a maximum assembled-body tilt of 60 degrees from world-up, rejecting
  side-on/inverted transit postures;
- completion of contact acquisition at a maintained two-anchor grasp pose.

This is a kinematic grasp-pose admission only. It does not claim that contact
force, friction, payload support, or closed-loop tracking succeeds. Those
claims remain exclusively with the subsequent full-layer Isaac replays. A
failed screen has `eligible_for_full_control_test=false` and cannot be sent to
those replays or admitted as a collection record.

## Selection, confirmation, and gates

The four levels are evaluated in ascending order only on the 22 C3
`train`-owned buckets. At each level this is 594 lattice teacher cases, 88
interior teacher cases, at most 682 fast screens, and at most 1,364 Isaac
replay episodes. The last two maxima are reached only if every preceding stage
passes.

The first failed selection level stops the ladder. The largest consecutively
passing level is then evaluated exactly once on the untouched 14 C3
`validation`-owned buckets: 434 teacher cases, at most 434 fast screens, and at
most 868 Isaac replay episodes.
If confirmation fails, the whole v1 result is rejected. The range is not
stepped back or tuned against validation evidence; a revised search requires
a separately approved v2 protocol.

Every selection level and the confirmation must satisfy all of the following:

- 100% of the 27-point lattice teacher cases produce a complete
  `ContactWrenchTrajectory`, pass the fast exact-tracking screen, pass the
  production `C_H` path, and pass both Isaac replays;
- aggregate deterministic-teacher feasibility is at least 95%;
- aggregate fast-screen acceptance is at least 95%;
- aggregate real-Isaac task success is at least 95%;
- the four random-interior cases have separate teacher-feasibility,
  fast-screen, and Isaac-success gates, each at least 95%, so aggregate success
  cannot hide interior failure behind the mandatory lattice passes;
- safety failures are exactly zero;
- fallback decisions are exactly zero;
- IK, fast-screen joint-limit/rate/collision results, actuator/controller
  feasibility, raw and resolved production-checker results, simulator/config
  hashes, seeds, and failure reasons are retained as calibration evidence;
- the support geometry and pose are frozen from each unmodified source bucket;
  each sampled start/goal records projected-CoM and full-footprint margins;
  projected CoM must remain inside the support (`margin >= 0`). The full
  footprint margin is retained but may be negative because the promoted C3
  support intentionally has a lateral inset; physical acceptability then still
  requires the lattice/interior real-Isaac gates.

If L1 fails, no R1 envelope is accepted. If L4 passes, L4 is the maximum v1
envelope; extrapolation is forbidden. Calibration cases are gate evidence and
are never eligible as imitation-learning records.

Maximum calibration workload, when all four selection levels pass, is 3,162
teacher cases, at most 3,162 linear fast screens, and 6,324 Isaac episodes
including confirmation. Scaling the C3 formal-evaluation throughput gives an
indicative 9.5 hours of Isaac runtime, plus teacher planning and setup. The
implemented runner now enforces enumeration, stopping, single confirmation,
and first-screen admission, and the production teacher/IK plus first-screen
adapter is available. The real-Isaac evaluator still has to be supplied by
the existing production execution layer when calibration is deliberately
started, so this estimate is planning guidance, not a formal runtime claim or
measured R1 result.

## Formal R1 collection after acceptance

Only after an envelope passes confirmation, an accepted hash-bound R1
distribution manifest is published, and collection-mode preflight passes:

- collect 140 complete deterministic-teacher episodes: 98 train, 21
  validation, and 21 held-out;
- apply the same mandatory fast exact-tracking screen to every newly generated
  collection candidate, and launch the full controller/Isaac test only for a
  passing path;
- allocate 20 episodes per module count 2--8: 14/3/3 per split;
- retain C3 topology ownership: train uses train-owned morphologies, while
  validation and held-out use validation-owned morphologies with disjoint pose
  seeds;
- `held-out` here means pose-held-out within the approved R1 object family,
  not a new topology or object family;
- use collection seeds `29009 + episode_index`, disjoint from calibration;
- require unique task ids, seeds, and pose-condition hashes globally, with
  validation and held-out pose hashes mutually disjoint;
- require at least 95% task success both globally and in every split x module
  cell, with zero safety failures and zero fallback in every cell (at the
  proposed 14/3/3 cell sizes this means every collected episode must pass);
- preserve at least 128 GiB free storage before starting.

The 140-episode collection is estimated from C0 evidence at about 22.1 hours
and 17.4 GB. No R1 production collection orchestrator/output-manifest validator
has yet been implemented; both are required after envelope approval and before
collection. The estimate does not authorize collection and must be replaced by
measured R1 evidence in the final dataset manifest.

## Approval record

The repository user approved the complete v1 package on 2026-08-24:

1. the four-level numeric ladder;
2. 22-bucket selection plus untouched 14-bucket confirmation;
3. the case counts, 100% lattice rule, independent 95% interior/aggregate
   gates, mandatory fast exact-tracking screen with a 1% normalized joint-limit
   reserve and 60-degree body-tilt ceiling, projected-CoM support gate, and
   zero safety/fallback rule;
4. the fail-closed acceptance rule and calibration-data exclusion;
5. the 140-episode formal collection plan.

The approval record binds the reviewed proposal SHA-256, approved protocol
SHA-256, C3 release/checkpoint identity, and all eight approval-scope items.
At approval time no calibration had been run. The subsequent v1 execution was
rejected at L1, so formal collection remains blocked unless a separately
approved later protocol passes its confirmation and publishes an accepted,
hash-bound distribution manifest.
