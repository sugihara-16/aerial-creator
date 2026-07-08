# A-MSRR QP/PID 制御器設計仕様書 v0.1 draft

**対象:** P4-control / P4a low-level flight validation  
**親仕様:** `A-MSRR_codex_ready_spec_v0_4_ja.md` Sections 20, 23.5, 24.5.2, 25, 26.9-26.10, 27.1  
**位置づけ:** 全体設計書を補完する制御器専用仕様。全体設計書と矛盾する場合は、全体設計書を優先する。

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
