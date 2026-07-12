# A-MSRR QP/PID 制御器設計仕様書 v0.1 draft

**対象:** P4-control / P4a low-level flight validation  
**親仕様:** `A-MSRR_codex_ready_spec_v0_4_ja.md` Sections 20, 23.5, 24.5.2, 25, 26.9-26.10, 27.1  
**位置づけ:** 全体設計書を補完する制御器専用仕様。原則として全体設計書と矛盾する場合は全体設計書を優先する。ただし、ユーザー承認済みの設計変更として Section 14 と `AMSRR_design_modification_by_codex.md` の 2026-07-12 entry に明記した事項は、その範囲に限り従来記述を supersede する。

---

## 0. 文書契約

本書は、A-MSRR Version 1 の P4-control / P4a で実装する低レイヤ QP/PID 制御器の仕様を定める。

この制御器は、`π_L` が出す `PolicyCommand` を受け取り、deterministic controller / QP / safety layer として `ControllerCommand` を生成する。`π_L` は actuator command、rotor thrust、joint torque、vectoring joint target を直接出力してはならない。

Isaac Lab への actuator target 変換は controller bridge の責務であり、`ControllerCommand` から Isaac actuator target record を生成する。

schema / interface の変更が必要になった場合は、実装前に理由を提示し、`AMSRR_design_modification_by_codex.md` に記録する。

---

## 1. 今回の実装範囲

P4-control / P4a では、object grasp/carry や contact-rich task に入る前に、低レイヤ飛行制御と actuator mapping を Isaac Lab 上で検証する。

今回の目標:

1. Holon / gimbal rotor module 向けの QP/PID 制御器を実用レベルまで実装する。
2. `PolicyCommand -> ControllerCommand -> Isaac actuator targets` の責務分離を維持する。
3. single-module hover、fixed-morphology hover、fixed-morphology waypoint tracking を検証する。
4. `RuntimeObservation`、`ControllerCommand`、actuator target records、controller metrics を `EpisodeArchive` に保存する。

今回の範囲外:

- P4.2 の object grasp/carry success
- P4.3 の learned `π_L` / residual controller training
- learned policy による hard safety 判定の置換
- P4-control acceptance を P4 full completion と呼ぶこと

---

## 2. 制御アーキテクチャ

制御パイプライン:

```text
RuntimeObservation
MorphologyGraph / ConstructionState
PhysicalModel
active target / PolicyCommand
  -> target builder / PID 参照値 builder
  -> desired body wrench
  -> quasi-static rigid-body model update
  -> QP allocator
  -> ControllerCommand
  -> Isaac controller bridge
  -> actuator target record
```

主要原則:

1. allocation は QP で行う。擬似逆行列 allocation は採用しない。
2. QP solver が利用できない場合の fallback は、明示的に degraded / non-QP fallback として記録し、P4-control acceptance の source として扱うかどうかを別途判定する。
3. 合体形態では joint 角度により rotor origin、rotor direction、CoM、inertia が変化する。
4. joint の運動は準静的とみなし、各制御周期では、更新済みの形態を単一剛体として制御する。
5. joint dynamics を制御ステップ内の主要な dynamic state として QP に含めない。ただし joint target / joint limit / rate limit / dock command は actuator constraint として扱う。

---

## 3. 入出力

### 3.1 入力

制御器は以下を受け取る。

```text
RuntimeObservation:
  module pose, twist, joint positions, joint velocities, controller status

MorphologyGraph / assembled control graph:
  active modules, dock edges, control groups, robot anchors

PhysicalModel:
  links, joints, rotors, dock ports, mass/inertia metadata, thrust limits

PolicyCommand:
  desired body pose/twist, residual wrench, joint bias, contact tracking bias

active target:
  P4-control では direct hover / waypoint target
  P4.2 以降では active InteractionKnot

previous ControllerCommand:
  rate limit, smoothness cost, actuator delta cost に使う
```

### 3.2 出力

制御器は以下を出力する。

```text
ControllerCommand:
  rotor_thrusts_n
  vectoring_joint_targets
  joint_torque_commands
  dock_mechanism_commands
  controller_status
```

controller bridge は以下を出力する。

```text
actuator_target_record:
  time_s
  backend
  morphology_graph_id
  command_index
  actuator_targets
  clipped_targets
  missing_actuators
  unsupported_actuators
  allocation_residual_norm
  qp_status
  metrics
```

`actuator_target_record` は `EpisodeArchive.actuator_target_records` に保存する。

---

## 4. 座標系と状態

最低限、以下の frame を明示して実装する。

```text
world frame:
  simulator / task target の基準

body frame:
  合体形態の現在 CoM / control body に固定した剛体 frame

module frame:
  各 Holon module の runtime pose frame

rotor frame:
  各 rotor thrust direction / vectoring joint を定義する frame

dock / joint frame:
  合体構造と準静的形態更新に使う joint frame
```

P4-control 初期実装では、desired wrench は body frame で扱う。world frame target から body frame error / wrench への変換は controller 内で行う。

`RuntimeObservation.module_states[*].joint_positions` は、毎制御周期の形態モデル更新に使う source of truth とする。

---

## 5. 準静的な合体形態モデル更新

合体形態では、各制御周期で以下を更新する。

```text
input:
  MorphologyGraph
  PhysicalModel
  RuntimeObservation.module_states
  joint_positions

output:
  RigidBodyControlModel:
    total_mass_kg
    center_of_mass_body
    inertia_body
    rotor_origins_body
    rotor_axes_body
    vectoring_joint_axes
    dock actuator ids
    active actuator limits
```

更新規則:

1. `MorphologyGraph` の topology と dock edges から active modules を決定する。
2. 各 module の runtime pose と joint angle から、module 内 rotor origin / rotor axis を body frame に変換する。
3. 各 link の inertial data を current transform で body frame に変換し、link-level に mass / CoM / inertia を合成する。
4. 合成 inertia は、その制御周期内では固定値として扱う。
5. joint velocity は rate limit や damping metric に使ってよいが、制御ステップ内の rigid-body dynamics には入れない。

この model update は deterministic でなければならず、unit test で Isaac なしに検証可能でなければならない。

inertia 合成は初期実装から link-level quasi-static aggregation とする。すなわち、module を単一 lumped mass として近似するのではなく、URDF / `PhysicalModel` の link mass、local COM、local inertia を現在 joint angle に基づく link transform で集約する。

```text
for each active link:
  T_body_link(q_current)
  link mass
  link local COM
  link local inertia
    -> body frame inertia
    -> parallel axis theorem
    -> current morphology total mass / CoM / inertia_body
```

ただし、これは articulated-body dynamics ではない。joint acceleration、Coriolis / centrifugal coupling、joint drive dynamics はこの段階では含めず、現在姿勢の単一剛体近似を毎制御周期で再構築する。

---

## 6. PID / target builder

P4-control では、hover / waypoint target から desired body wrench を作る。

制御目標:

```text
position target
velocity target
attitude target
angular velocity target
optional feedforward acceleration
optional residual wrench from PolicyCommand
```

PID layer は以下を計算する。

```text
position_error_world
velocity_error_world
attitude_error_body
angular_velocity_error_body
desired_acc_world
desired_ang_acc_body
desired_force_body
desired_torque_body
```

重力補償:

```text
desired_force_world = total_mass * (gravity_compensation + desired_acc_world)
desired_force_body = R_body_world^-1 * desired_force_world
```

姿勢制御:

初期実装では roll / pitch / yaw error または quaternion error のどちらか一方を採用し、採用しない表現は metric として保存してよい。最終仕様ではどちらかを明記する。

anti-windup:

QP infeasible、actuator saturation、mode reset のときは integral term を freeze または back-calculate する。

---

## 7. QP allocation

### 7.1 QP 変数

単一 Holon module:

```text
u =
  rotor thrust variables
  vectoring joint target or force-direction variables
  optional joint torque variables
  optional slack variables
```

合体形態:

```text
u = active modules の actuator variables を deterministic order で連結したもの
```

deterministic order:

```text
module_id ascending
rotor_id ascending
joint_id ascending
dock actuator id ascending
slack variables last
```

### 7.2 allocation matrix

各 rotor について、現在形態から以下を計算する。

```text
r_i:
  body CoM から rotor origin への body-frame vector

a_i(q):
  vectoring joint angle を反映した body-frame thrust axis

f_i:
  rotor thrust scalar

tau_reaction_i:
  reaction_torque_coeff * f_i * rotor spin direction
```

各 rotor の wrench contribution:

```text
F_i = a_i(q) * f_i
Tau_i = r_i x F_i + tau_reaction_i
```

全 rotor から body wrench への線形化:

```text
w_body ~= A(q_current) u
```

ここで `q_current` は現在 joint angle であり、制御周期内では固定する。

### 7.3 目的関数

基本 QP:

```text
min_u
  || A u - w_des ||^2_Q
  + alpha_u || u ||^2
  + alpha_delta || u - u_prev ||^2
  + alpha_joint || q_target - q_current ||^2
  + alpha_slack || s ||^2
```

`w_des` は PID / target builder が生成した desired body wrench である。

### 7.4 制約

必須制約:

```text
rotor thrust lower / upper bounds
vectoring joint lower / upper bounds
vectoring joint rate limits
joint torque lower / upper bounds when torque command is used
dock mechanism command bounds
actuator finite / NaN checks
optional slack bounds
```

安全制約:

```text
max tilt
max body angular rate
max target acceleration
max rotor saturation ratio
minimum thrust margin
```

QP が infeasible の場合:

1. `ControllerStatus.status = "infeasible"` を返す。
2. residual と violation code を metrics に保存する。
3. configured safe stop / hover fallback が存在する場合のみ、fallback command を生成する。
4. fallback を使った場合も、original QP infeasible は archive に残す。

---

## 8. `ControllerCommand` 生成

QP result から `ControllerCommand` を生成する。

```text
rotor_thrusts_n:
  rotor_id -> thrust N

vectoring_joint_targets:
  joint_id -> target position rad

joint_torque_commands:
  joint_id -> torque Nm

dock_mechanism_commands:
  dock actuator id -> command value

controller_status:
  status, qp_feasible, active_mode, message, metrics
```

P4-control 初期実装では、dock mechanism commands は hold value でよい。ただし missing actuator と unsupported command は bridge metrics に必ず出す。

---

## 9. Isaac controller bridge

controller bridge は `ControllerCommand` を Isaac actuator targets に変換する。

bridge の責務:

1. `MorphologyGraph` と `PhysicalModel` から active actuator mapping を構築する。
2. rotor thrust target を Isaac 側の multirotor / thruster target へ変換する。
3. vectoring joint target を Isaac joint target へ変換する。
4. dock mechanism command を Isaac actuator target へ変換する。
5. actuator limits で clip する。
6. missing / unsupported actuator を記録する。
7. actuator target record を返す。

bridge は final actuator target conversion layer であり、learned policy ではない。

Isaac Lab 側の rotor thrust 表現は、まず per-thruster thrust target を primary representation とする。`isaaclab_contrib.assets.Multirotor` / `Thruster` が使える asset では `set_thrust_target` 相当の経路を使う。Holon の custom articulation で multirotor asset 化が未完了の場合は、同じ actuator target record から Isaac Lab の wrench composer による rotor body / thrust application point への force application に変換してよい。

重要なのは、A-MSRR 側の controller/bridge contract では常に per-rotor thrust target と vectoring joint absolute position target を保存することである。Isaac backend の都合で wrench composer を使う場合も、archive には元の per-rotor command と実際に適用した force / torque target の両方を保存する。

---

## 10. Logging / metrics

`RuntimeObservation.controller_status.metrics` と `EpisodeArchive.metrics` には以下を保存する。

```text
qp_feasible
qp_status_code
allocation_residual_norm
force_residual_norm
torque_residual_norm
unsupported_wrench_norm
clipped_target_count
missing_actuator_count
unsupported_actuator_count
rotor_saturation_ratio
min_rotor_thrust_margin
min_vectoring_joint_margin
target_pos_error_m
target_rot_error_rad
target_velocity_error_norm
target_angular_velocity_error_norm
rigid_body_model_update_success
```

`EpisodeArchive.rollout_artifacts` には以下を保存する。

```text
phase: "P4-control"
backend: "isaac_lab"
is_p4_full_completion: false
isaac_backed: true
physical_success_claim: false
note: low-level flight validation only
```

---

## 11. P4-control acceptance

P4-control / P4a acceptance は以下を確認する。

1. single-module hover が crash-free に完了する。
2. fixed-morphology hover が crash-free に完了する。
3. fixed-morphology waypoint tracking が configured pose error 以下で完了する。
4. 各 step で `RuntimeObservation` が保存される。
5. 各 step で `ControllerCommand` が保存される。
6. 各 step で actuator target record が保存される。
7. controller infeasible / clipped / residual metrics が保存される。
8. P4-control は P4 full completion として記録されない。

P4-control acceptance は、object grasp/carry success rate を主張しない。

初期 waypoint tracking threshold:

```text
position_error_m <= 0.20
attitude_error_rad <= 0.25
hold_duration_s >= 1.0
```

これらは `configs/training/p4_control_low_level.yaml` で configurable にする。Isaac 上の挙動が安定した後、acceptance threshold を段階的に厳しくしてよい。

---

## 12. 実装予定ファイル

Agent I:

```text
amsrr/controllers/rigid_body_model.py
amsrr/controllers/actuator_mapping.py
amsrr/controllers/isaac_controller_bridge.py
amsrr/controllers/qpid_controller.py
tests/unit/controllers/test_rigid_body_model.py
tests/unit/controllers/test_actuator_mapping.py
tests/unit/controllers/test_isaac_controller_bridge.py
tests/unit/controllers/test_qpid_controller.py
```

Agent J:

```text
amsrr/simulation/p4_control_isaac_env.py
amsrr/simulation/isaac_lab_backend.py
configs/env/isaac_lab.yaml
tests/unit/simulation/test_p4_control_isaac_env.py
```

Agent K:

```text
amsrr/training/p4_control_runner.py
configs/training/p4_control_low_level.yaml
tests/unit/training/test_p4_control_runner.py
```

Agent L:

```text
amsrr/acceptance/p4_control_acceptance.py
tests/acceptance/test_p4_control_acceptance.py
```

---

## 13. 実装決定事項

現時点の決定事項:

1. QP solver は初期実装では Python でよい。ライブラリを使ってよい。挙動が安定した後に C++ backend へ移行する。
2. Isaac Lab 側の drone thrust 表現は、per-thruster thrust target を primary とする。利用可能なら `Multirotor` / `Thruster` asset の thrust target 経路を使う。custom Holon articulation では wrench composer による rotor force application へ bridge してよい。
3. vectoring joint command は absolute position target とする。
4. reaction torque coefficient は QP allocation model に含める。
5. 合体形態の inertia 合成は link-level quasi-static aggregation とする。
6. 初期 waypoint threshold は `position_error_m <= 0.20`、`attitude_error_rad <= 0.25`、`hold_duration_s >= 1.0` とし、configurable にする。

追加の未定義事項が実装中に見つかった場合は、互換性のない仮定を置かず、実装前にユーザーへ確認する。

---

## 14. 2026-07-12 承認済み制御契約改訂

### 14.1 改訂の優先順位と目的

本節は、通常実行時の QPID を contact-wrench-aware whole-body QP として拡張する従来案を改め、centroidal flight control と local joint control の責務を分離する。以下の事項について、本書 Sections 2、3、6、7、8、および親仕様 Section 20 の従来記述と矛盾する場合は本節を優先する。

改訂後の通常制御では、`PolicyCommand` と QP のどちらにも以下を含めない。

```text
per-contact wrench target / bias
contact wrench allocation variable
normal-operation dock internal wrench target / bias
dock internal wrench allocation variable
```

contact wrench は task requirement、feasibility、reward、contact safety、logging のために残すが、通常 QPID の直接 tracking target にはしない。dock internal wrench は detach 前後の特殊処理だけで扱う。

### 14.2 改訂後の通常制御パイプライン

```text
π_H:
  contact assignment / mode / schedule / wrench requirement
  centroidal target / posture target / object target
    -> π_L context

π_L PolicyCommand:
  desired centroidal pose / twist
  centroidal wrench bias
  absolute joint position / velocity targets
  joint torque bias
    -> deterministic controller

QPID / thrust allocator:
  centroidal pose/wrench target
    -> rotor thrust + thrust-vectoring targets

local joint servo:
  joint position/velocity targets + torque bias
    -> non-vectoring joint actuator targets

controller bridge:
  validated/clipped controller outputs
    -> backend actuator targets
```

`π_L` は引き続き最終 actuator command を出さない。`PolicyCommand` の joint target と torque bias は低レイヤ参照値であり、controller が actuator existence、control mode、position/rate/effort limit、finite check、安全 override を適用した後にだけ `ControllerCommand` / backend target へ変換する。

### 14.3 `PolicyCommand` の改訂契約

改訂後の規範的な controller-facing fields は次とする。

```python
class PolicyCommand:
    desired_body_pose: Pose7D | None
    desired_body_twist: list[float] | None
    residual_wrench_body: list[float] | None

    joint_position_targets: dict[str, float]
    joint_velocity_targets: dict[str, float]
    joint_torque_bias: dict[str, float]
```

field semantics:

```text
desired_body_pose:
  assembled morphology centroidal control frame の world pose target

desired_body_twist:
  assembled morphology centroidal control frame の world/body convention が明示された twist target

residual_wrench_body:
  body-aligned centroidal control frame における additive CoM wrench bias
  contact wrench または dock internal wrench ではない

joint_position_targets:
  non-vectoring joint の absolute position target

joint_velocity_targets:
  non-vectoring joint の absolute velocity target

joint_torque_bias:
  local position/velocity servo output に加算する bounded offset torque
  final joint torque command ではない
```

旧 `joint_position_bias`、`joint_velocity_bias`、`contact_tracking_bias` は既存 archive / checkpoint の読取り互換性のため直ちには削除しない。新 contract 実装後は deprecated fields とし、通常実行 path では `contact_tracking_bias` を no-op とする。旧 checkpoint と新 checkpoint は contract/version metadata で区別し、field meaning を黙って変更してはならない。

`π_H` の `PostureTarget` は π_L への高レベル参照である。通常 path では π_L が `PostureTarget` と runtime state から absolute `joint_position_targets` / `joint_velocity_targets` を生成し、local servo へ渡す。π_L target が欠落または無効な場合は、controller が current-position hold または明示された deterministic posture fallback を使う。

command priority:

```text
hard safety / actuator limit override
π_A latch open-close / detach special-mode override
thrust allocator vectoring-joint target
π_L non-vectoring joint target / torque bias
deterministic current-position hold fallback
```

### 14.4 Centroidal control frame

通常 QPID は base module の `fc` origin ではなく、現在形態の CoM を並進制御する。centroidal control frame `C` は次で定義する。

```text
origin:
  RigidBodyControlModel.center_of_mass_body を world へ変換した current morphology CoM

orientation:
  selected morphology control-body frame と同じ orientation

linear velocity:
  control-body origin velocity を CoM offset へ移した velocity

angular velocity:
  control-body angular velocity
```

各 control step で current joint state と module state から mass、CoM、inertia、rotor geometry を更新する。`desired_body_pose` という既存 field name を維持する場合でも、新 contract では centroidal control-frame pose を意味する。base-module pose tracking を centroidal tracking と報告してはならない。

### 14.5 QP の責務境界

通常 QP の actuator variables は rotor thrust、thrust-vectoring variables / targets、および必要な slack に限定する。generic manipulation joint torque、contact wrench、dock internal wrenchは QP variable に含めない。

```text
w_target =
    w_centroidal_pose_tracking
  + w_centroidal_bias
  + optional controller-owned aggregate disturbance compensation

min_u
  || A(q_current) u - w_target ||^2_Q
  + alpha_u ||u||^2
  + alpha_delta ||u - u_prev||^2
  + alpha_slack ||s||^2
```

`A(q_current)` は current morphology の CoM、inertia、rotor origin、rotor axis、reaction torque を使う。optional disturbance compensation は、個別 contact wrench の分解ではなく、centroidal momentum / state observer から得られる aggregate external wrench に限る。これは controller-owned estimate であり `PolicyCommand` の contact-wrench field ではない。

thrust-vectoring joint は rotor thrust axis を決めるため独立 joint controlにはしない。QP / vectoring allocator が absolute targetを決め、local position servo が追従する。non-vectoring manipulation / dock mechanism jointsだけを Section 14.6 の独立 servoで扱う。

### 14.6 Non-vectoring local joint servo

non-vectoring joint は QP allocation から分離し、actuator-local position/velocity servo と bounded offset torque で制御する。

```text
tau_servo =
    Kp (q_target - q)
  + Kd (qdot_target - qdot)

tau_requested = tau_servo + tau_bias
```

backend actuator が position command と offset torque を同時に受け付ける場合は、その native path を使ってよい。受け付けない場合は controller が support status を明示し、黙って一方を捨ててはならない。

requirements:

```text
absolute position / velocity target validation
position, velocity, effort, torque-bias, and rate limits
per-actuator supported control-mode check
non-finite rejection
command smoothing where configured
missing / unsupported / clipped logging
current-position hold fallback
```

joint motion は Version 1 では準静的とし、joint rate / accelerationを制限する。joint actuation reactionは measured centroidal feedback と更新済み rigid-body modelで補償する。高速 articulated-body feed-forward、Coriolis / centrifugal coupling、contact-aware full-body inverse dynamics は通常 QPID の初期範囲外とする。

### 14.7 Contact requirement、observation、reward

`π_H` の contact assignment、mode、schedule、wrench requirement / bound、centroidal target、posture target、object target の schemaは変更しない。contact wrench requirement は次に使用する。

```text
π_L context
design / assignment feasibility
Isaac privileged training reward
contact safety and task success evaluation
archive / diagnostics
```

通常 QPID は per-contact wrench targetを追跡せず、`PolicyCommand.contact_tracking_bias`を消費しない。actorである π_L が利用できない実機情報へ依存しないよう、Isaac の per-contact force / impulse は policy observation と privileged reward / critic inputを区別する。

多点接触では個別 wrench の一意な分解を要求しない。学習rewardは、厳密なpointwise wrench targetだけでなく、task-equivalentな次の評価を優先する。

```text
object / environment に対する net desired effect
contact maintenance
wrench bound / friction / safety margin satisfaction
centroidal stability
object progress / goal accuracy
slip, penetration, contact break, excessive force, unintended contact
actuator saturation and control effort
```

contact existence、slip、penetration、contact break、force upper bound、object dropなどの safety observationは残す。contact wrenchをQP targetから外すことは、contact sensingとsafety gateを削除することを意味しない。

### 14.8 Dock internal wrench は detach 専用

通常 `PolicyCommand` と通常 QP は dock internal wrenchを扱わない。detachでは、対象 `DockEdge` を切った follower-side component に対する専用 unload / release procedureを使う。

follower側に結合部以外のexternal contact / external loadがなく、active rotor / actuator wrench、gravity、mass、CoM、inertia、momentum rateが既知である場合、follower centroidal balanceからcut-edge wrenchを推定できる。

```text
w_cut_at_follower_com =
    follower_momentum_rate
  - follower_known_actuator_wrench
  - follower_gravity_wrench
  - follower_other_known_external_wrench
```

このwrenchをfollower CoMからdock frameへspatial wrench transformし、sign conventionを「parent-side componentがfollower-side componentへ加えるwrench」など一意に固定する。whole assembled morphology のcentroidal estimatorではinternal wrenchが相殺されるため、必ずcandidate edgeで切った follower subtreeに対して推定する。

detach gate:

```text
follower external-contact-free evidence
follower estimator validity
relative pose / velocity threshold
force threshold and torque threshold
both components independently QP-feasible
N consecutive unload dwell steps
latch release
post-release separation stability
```

external contact、unknown load、estimator invalid、threshold超過のいずれかがある場合はfail closedとする。Isaac constraint reaction / virtual sensorはground truth comparisonとestimator validationに使ってよいが、実機で利用できない値をrelease gateの唯一のsourceにしてはならない。

### 14.9 ControllerCommand / bridge migration

新 contract を実装する際は、controller / bridge contractが少なくとも次を表現できなければならない。

```text
rotor thrust targets
thrust-vectoring absolute position targets
non-vectoring joint absolute position targets
non-vectoring joint absolute velocity targets where supported
non-vectoring joint bounded offset torque targets
latch / dock special-mode commands
```

controllerは π_L targetを検証・clipし、最終 actuator target recordにrequested / applied value、control mode、limit、clip reason、missing / unsupported statusを保存する。π_Lがjoint targetを出すことはfinal actuator authorityの移譲ではない。

### 14.10 実装・acceptanceへの影響

本節は設計契約の改訂であり、追記時点ではPython schema、controller、bridge、policy、training artifactを変更しない。実装はschema-firstで行い、少なくとも以下を検証する。

```text
PolicyCommand absolute joint target / torque-bias validation and round trip
legacy field read compatibility and new checkpoint contract versioning
true centroidal pose/twist construction for asymmetric morphologies
QP excludes contact/internal wrench and generic non-vectoring joint variables
vectoring targets remain allocator-owned
local joint position + offset-torque mapping and limit enforcement
unsupported actuator control-mode fail-closed behavior
contact_tracking_bias no-op behavior on the new path
Isaac privileged contact reward does not leak into actor observation
follower-subtree detach wrench estimator and frame/sign tests
normal-operation reports do not claim internal/contact wrench tracking
```

既存 P4-control / P4.2 / P4.3 artifact は生成時の旧 contract versionに基づく履歴として保持する。新 contract の実装と再検証が完了するまでは、既存成功artifactを新しいcentroidal-only QPID / absolute-joint-target contractの成功証拠として読み替えてはならない。
