# Isaac学習・検証の高速化手順

2026-10-06 現行モデル更新: 模倣学習直後を固定モデルとする。checkpointと比較結果の正本は
`configs/training/order9_pi_h_selected_model.json`、手順は
[PI_H_SELECTED_MODEL.md](PI_H_SELECTED_MODEL.md)。以下に残る過去実験の再開元・最良updateは
現行選定ではない。PPOおよびDouble DQN＋CQL（事前学習を含む）は候補として保留する。

更新日: 2026-09-29。対象仕様: [v0.5](A-MSRR_codex_ready_spec_v0_5_ja.md) §§19.4.1、19.4.2、22.8。

この文書は、今回実装・計測した高速化を今後の作業で再利用するための手順書である。
実行入口は `scripts/run_guarded_request.py` → `scripts/run_request_collection.py` とする。
**引数の省略で遅い既定設定に戻らないよう、下記の設定を明示する。**

2026-10-01 方針更新: 正式R2はπ_Dの指定anchorを固定する `grasp_anchor_source=defined`。
再開元は `20260930_r2_defined_anchors/update_11`、継続先は
`20261001_r2_defined_anchors_continuation`。既存の凍結source・最速実行方式を維持する。
以下のneighborhood展開・中心優先v4への移行記述は過去実験の再現用であり、正式学習へ適用しない。
今回の最良モデルは継続先の `completed.json` で確認し、旧方式のupdate16等と番号だけで混同しない。

2026-09-30補足: R2の定義済みanchorによる学習ではprotocolに
`"grasp_anchor_source": "defined"`を明記する。接触数上限2だけでは全空き把持面の
展開を止められない。`all_free`は過去の多点拡張実験を再現する場合に使用する。
2026-10-01の五×五近傍拡張では`grasp_anchor_source=neighborhood`と接触数2を明記する。
prepareの`--grasp-anchor-source neighborhood --max-grasp-contacts 2`、並列draw、
直列drawを一致させる。群間重複を統合した物理ペアごとに最大8接触群を保持するため、
旧global上限8/32のときの準備時間を流用しない。初回drawのdistinct request数と
小規模prepare/executeの実測から時間を見積もる。正式な保存例は
`20261001_r2_anchor_neighborhoods_complete_catalog`。候補生成後も、実際のrequest catalogと
actor maskに合法な初回選択肢が残ることを上限なしの参照一覧と照合する。将来の遷移要求で
catalog上限を消費し、初回の接触選択を切り捨てない。近傍の所属情報は数値scoreに混ぜない。
同日の`20261001_r2_anchor_neighborhoods`と`..._corrected_metadata`は不具合を含む診断履歴であり、
正式学習の再開元には使わない。
後続の正式学習ではCPU収集とCUDA再評価の丸め差を避けるため、PPOも`--ppo-device cpu`、
大きなbatchのログは`--cpu-log-storage file`を採用している。下記速度測定値の条件とは
区別し、新しい学習ではこの後続設定と毎stageの資源guard・ログ圧縮を維持する。

- 今後の作業で現在のコードを使う場合: §3〜§6。
- 今回の測定を保存済みの固定ソースから再現する場合: §7。
- 遅くなった場合・中断する場合: §8〜§9。

## 1. 再現対象と実測値

計測機: Intel Core i9-14900KF、RAM 64GB、RTX 4090 24GB、既存の `isaaclab3` 環境。
同じ訓練28条件×16個の独立した乱数系列、合計448試行と1回のPPO更新を計測した。
448種類の異なる物体条件という意味ではない。

| 測定 | 秒 | 元の基準に対する倍率 |
| --- | ---: | ---: |
| 元の448試行＋PPO更新 | 16,971.134 | 1.000倍 |
| 最速実測: `initial_replica_20_warm` | 446.753 | 37.988倍 |
| 上記のguard起動・終了を含む時間 | 453.083 | 37.457倍 |
| 計画を初回生成した実測: `initial_20_cold` | 644.067 | 26.350倍 |
| 要求された40倍の目標 | 424.278 | **未達** |

最速測定は計画キャッシュ77/77件が一致した実行である。初回生成の測定は、最終版の
実行順序変更より前の構成で行った。未計測の組合せの所要時間をこの表から外挿しない。
ソース、物体条件、選択した接触群等が変われば、新たな計画生成が必要になる。

最速実測の内訳は、物理シミュレーション・制御379.702秒、PPO更新20.277秒、残りが
初回選択、計画照合、監査、入出力等。シミュレーション部分だけの倍率を全updateの倍率としない。

**物理結果の境界:** 元のGPU動力学構成は228/448成功、今回のCPU動力学＋GPU衝突候補計算は
219/448成功、safety failureは0だった。CPU/GPUの物理結果が同一とは扱わない。
同じCPU動力学＋GPU衝突候補計算の保持した比較では、報酬とPPO更新後checkpointのバイトが一致した。
元のタスク、物体条件、dt、solver設定、制御gain、報酬、成功判定は緩めていない。

根拠: [測定レポート](../artifacts/p4_full/order9/request_ppo/20260929_full_update_throughput/throughput_report.md)、
[全件測定値](../artifacts/p4_full/order9/request_ppo/20260929_full_update_throughput/measurements.json)、
[最速コマンド](../artifacts/p4_full/order9/request_ppo/20260929_full_update_throughput/initial_replica_20_warm_command.json)。

## 2. 保持する高速化と実装箇所

| 対象 | 実装・使い方 | 主な実装箇所 |
| --- | --- | --- |
| 異なる機体・計画の実行 | 独立したIsaacプロセスを20並列。各共通計画の複数試行を最大16環境でベクトル化 | `scripts/run_request_collection.py`, `amsrr/training/request_execution_worker.py` |
| 初回観測・接触群選択 | 4個のspawnプロセス。元のseedを明示し、結果をprotocol順に回収。物理実行前に終了 | `amsrr/training/request_initial_draws.py`, `request_collection.py` |
| 計画 | 準備を12並列で完了後、物理実行。入力・ソース等が一致する決定論的計画のみ再利用 | `amsrr/policies/checked_request_plan_cache.py`, `scripts/run_request_policy.py` |
| CPU制御モデル | 関節系のFK・質量・慣性・COM・荷重計算をnative化。CPU・勾配不要の経路で使用 | `amsrr/controllers/native_cpu_dynamics.py`, `native/request_dynamics_native.cpp` |
| Isaacとの受渡し | 必要な接触点Jacobian、関節target、選択bodyのレンチを扱うCPU経路。未対応入力は既存経路へ戻す | `amsrr/simulation/selected_contact_jacobian.py`, `cpu_actuator_staging.py`, `order9_tensor_isaac_io.py` |
| ステップ中の処理 | 参照値・guard計算のバッチ化、同じ観測からの計算重複削減、所有権を保つsnapshot | `amsrr/simulation/request_event_execution.py`, `teacher_contact_execution.py`, `amsrr/utils/tensor_snapshot.py` |
| 記録・監査 | 全項目・全時刻を保持したブロック記録。rawログの読込・hashを環境間で共用して独立に監査 | `amsrr/training/order9_tensor_rollout_artifact.py`, `request_motion_audit.py` |
| PPO | 元のprotocol設定を使用し、CUDAで更新。固定critic特徴の重複計算を削減 | `amsrr/training/request_ppo.py`, `request_collection.py` |
| プロセス管理 | 2物理jobごとに再生成、実行順序調整、CPU割当て、jemalloc、資源guard | `scripts/run_request_collection.py`, `scripts/run_guarded_request.py` |

20プロセスは一つの巨大なGPU物理シーンではない。**動力学・制御はCPU、衝突候補の計算
（broadphase）とPPOはGPU**という構成である。異なる機体を一つのGPUシーンにまとめた
94環境・10形態の試作は、host側の制御・状態処理が重く、採用していない。

新たな専用runnerを複製せず、上記の実行入口を使用する。各機体・接触群ごとの
成功済み結果をキャッシュして試行を省略してはならない。

## 3. 実行前の準備

### 3.1 環境、入力、空き容量を固定する

以降のコマンドはリポジトリrootから、同じbashセッションで実行する。

```bash
set -euo pipefail
cd /home/leus/amsrr
AMSRR_PY="$HOME/.local/share/mamba/envs/isaaclab3/bin/python"
AMSRR_JEMALLOC="$("$AMSRR_PY" -c 'import sysconfig; print(sysconfig.get_path("purelib") + "/isaacsim/kit/libjemalloc.so.2")')"
test -x "$AMSRR_PY"
test -f "$AMSRR_JEMALLOC"
git status --short
df -h .
free -h
cat /sys/devices/system/cpu/intel_pstate/no_turbo
nvidia-smi --query-gpu=name,memory.total,power.limit,temperature.gpu --format=csv
```

- 実測時は `no_turbo=1`、GPU電力上限250W。guardはこれを検査するが設定変更はしない。
  不一致時は構成差を解消してから実行する。高速化のために制限を解除しない。
- `sudo -n` でsystemdの専用serviceを起動・終了できる既存環境を使う。§7はさらにprivate mountを使う。
- 1updateでrawログ等が約12〜13GiB増える。guardの空きdisk下限40GiBに加え、1update分と余裕を
  確保する。このパネルを1回実行する目安は空き60GiB以上。連続updateでは毎回残量を確認する。
- protocol、record、物体条件、設定、入力checkpointを固定する。実行中はソース・設定・native binaryを
  編集・再buildしない。同時に編集する必要があれば、§7同様の固定ソース環境を使用する。
- 実装ファイルには未commit追加分がある。古いcommitだけをcheckoutしても高速化は再現できない。
  移設時は新規Python/C++ファイル・build script・対応testsも保存する。`git status`で追跡漏れを確認する。

### 3.2 native拡張を確認する

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$AMSRR_PY" - <<'PY'
import torch
from amsrr.controllers.native_cpu_dynamics import load_native
from amsrr.feasibility.order9_native_loader import load_order9_posture_native
print("torch", torch.__version__, "CUDA", torch.cuda.is_available())
assert torch.cuda.is_available()
print("dynamics", load_native().__file__)
print("posture", load_order9_posture_native().__file__)
PY
```

固定snapshotにも `.so` と対応する `.so.build-identity` の両方を含め、bindした環境内で上記load確認を実施する。

native拡張が欠落・古い場合、またはCPU/Python ABIを変えた場合だけ、そのホストで再buildする。
`-march=native` を使うため、別CPUへの `.so` の単純コピーを移設手順にしない。

```bash
PYTHON="$AMSRR_PY" bash scripts/build_request_dynamics_native.sh
PYTHON="$AMSRR_PY" bash scripts/build_order9_posture_native.sh
```

既存のC++ toolchain、Eigen、FCL等を利用する。依存不足を無断のglobal installや別Pythonへの
切替えで回避しない。build後は上のload確認を行う。
§7の履歴再現では、まず保存binaryと現在の依存関係を検査し、固定snapshotを上書きしない。

### 3.3 実装・環境を変えた場合の局所検査

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  "$AMSRR_PY" -m pytest -q \
  tests/unit/simulation/test_cpu_actuator_staging.py \
  tests/unit/simulation/test_request_cpu_physics.py \
  tests/unit/simulation/test_request_execution_worker.py \
  tests/unit/simulation/test_request_resource_guard.py \
  tests/unit/simulation/test_selected_contact_jacobian.py \
  tests/unit/controllers/test_native_cpu_dynamics.py \
  tests/unit/training/test_order9_tensor_rollout_artifact.py \
  tests/unit/training/test_request_collection.py \
  tests/unit/training/test_request_initial_draws.py \
  tests/unit/training/test_request_motion_audit.py
```

今回の結果は77件合格。最終scheduler変更後のcollector testsは16件合格。
同じソース・環境で毎updateこの試験を繰り返す必要はない。
新しい実行経路では少数の実シミュレーションで入力・作用・判定を確認してから全件へ進む。
短縮診断の `--case` / `--benchmark` / `--diagnostic-steps` は正式PPO用の全件実行に付けない。

## 4. 採用設定

| 段階 | 明示する設定 | 理由・境界 |
| --- | --- | --- |
| guard | `--cpu-quota 2000 --cpu-period-ms 10 --memory-gib 44` | 全子プロセス合計の上限。各workerに2000%を与える意味ではない |
| 初回選択 | `--initial-workers 4` | 省略時は0で直列 |
| 計画・監査 | `--prepare-workers 12` | 物理実行と段階を分け、CPU/RAMの競合を抑える |
| 物理実行 | `--workers 20 --max-environments 16` | 20プロセス、各共通計画のbatchは最大16環境 |
| worker | `--worker-threads 1 --physics-threads 0` | 実測時の設定。PhysXの1指定は採用しない |
| backend | `--device cpu --cpu-broadphase GPU` | GPU動力学への切替えとは異なる |
| 配置・再生成 | `--pin-workers performance --recycle-after-jobs 2` | P-coreを優先使用。無制限のworker再利用を避ける |
| ログ | `--cpu-log-storage ram` | file方式はメモリ余裕を増やしたが、総時間は悪化 |
| PPO | `--ppo-device cuda:0 --ppo-timeout-s 600` | 省略時のPPO deviceはCPU。epoch等はprotocolを使う |
| allocator | `LD_PRELOAD=…/libjemalloc.so.2` | Isaac同梱のライブラリを対象プロセス内で使う |
| allocator設定 | `MALLOC_CONF=narenas:2,background_thread:false,dirty_decay_ms:100,muzzy_decay_ms:100` | 実測した値を維持 |

BLAS/OpenMP/PyTorch compileのthread上限はguard、Isaac用のthread環境変数はcollectorが設定する。
物理jobは `(環境数 + 4) × 計画軌道の合計時間` の大きい順に投入する。これは優先順位であり、
試行の省略・段階タイミング・タスク期限の変更ではない。この順序だけの改善幅は約2.9秒と小さい。

guardはRAM high40GiB/max44GiB、swap0、CPU85°C/GPU75°C、GPUメモリ20,000MiB、
空きRAM8GiB、空きdisk40GiB等を監視する。kernel/GPU異常でも停止する。
この値と20並列は上記PCの実測構成であり、別PCへ同じ並列数を無条件に適用しない。

## 5. 現在のコードで448試行＋1回のPPO更新を実行する

以下は今回と同じupdate16を入力とする独立した再現実行例。既存の本学習update17は上書きしない。

```bash
AMSRR_PROTOCOL_ROOT="artifacts/p4_full/order9/request_ppo/20260927_r2_two_contact_training"
AMSRR_RUN="artifacts/p4_full/order9/request_ppo/$(date -u +%Y%m%dT%H%M%SZ)_throughput_replay"
test ! -e "$AMSRR_RUN"
test ! -e "${AMSRR_RUN}_guard"
test ! -e "${AMSRR_RUN}_update_17"

"$AMSRR_PY" scripts/run_guarded_request.py \
  --output "${AMSRR_RUN}_guard" \
  --cpu-quota 2000 --cpu-period-ms 10 --memory-gib 44 --wall-seconds 840 \
  -- /usr/bin/env \
  LD_PRELOAD="$AMSRR_JEMALLOC" \
  MALLOC_CONF=narenas:2,background_thread:false,dirty_decay_ms:100,muzzy_decay_ms:100 \
  "$AMSRR_PY" -u scripts/run_request_collection.py \
  --protocol "$AMSRR_PROTOCOL_ROOT/protocol.json" \
  --output "$AMSRR_RUN" \
  --checkpoint "$AMSRR_PROTOCOL_ROOT/update_16/checkpoint.pt" \
  --update-index 16 \
  --geometry-root "$AMSRR_PROTOCOL_ROOT/collections/train_16" \
  --workers 20 --prepare-workers 12 --initial-workers 4 \
  --max-environments 16 --worker-threads 1 --physics-threads 0 \
  --device cpu --cpu-broadphase GPU --cpu-log-storage ram \
  --recycle-after-jobs 2 --pin-workers performance \
  --execute-timeout-s 1200 \
  --ppo-output "${AMSRR_RUN}_update_17" \
  --ppo-device cuda:0 --ppo-timeout-s 600
```

コマンド全体をログ付きの非blocking sessionで起動して節目を確認する。
`840`秒は今回の再現用wall-clock上限であり、タスクのsimulated deadlineではない。
全体guardの期限は個別job/PPOのtimeoutに優先する。より広い条件を扱うときは、事前測定と
作業予算からwall-clock上限を設定し、物理dtや成功条件は変更しない。

### 計画・幾何の再利用

- `--geometry-root` は観測と要求が一致する既存幾何を探す入口。成功率や物理結果の再利用ではない。
  今回の比較では既存の `collections/train_16` を使ったため、再現時も同じものを指定する。
- 検査済み計画の標準キャッシュは `artifacts/cache/request_checked_plans/`。
  `panel/*/group_*/plan_cache.json` にkey、hit、由来を保存する。
- キャッシュは観測、要求、task、物理モデル、record、幾何prefix、ソース、設定、library/native binary、
  計画予算等を照合する。ソース変更で再生成になるのは正常。hashを書き換えてhitさせない。
- 初回actor推論・確率・物理実行・報酬を省略しない。計画失敗208試行もPPOへ渡す。
- 速度を再計測するときは必ず新しいoutputを使う。既存の完了receiptから復旧できる経路があるため、
  同じoutputでの再実行を「新しい448試行の速度測定」として扱わない。

### 今後の学習へ適用するとき

入力がupdate `k` なら `--checkpoint` をそのcheckpoint、`--update-index k` とし、出力を新しい
update `k+1` 用ディレクトリへ向ける。actor・critic・optimizerの継続状態とprotocolを維持する。
このcollectorは**1updateを完了する入口**で、検証間隔・最良モデル選択・収束判定の外側ループではない。
複数updateの運用ではそのループから毎回この入口を呼び、各回の完了を確認して次へ進む。
以前のartifact内の古い学習runnerを再開しても、新しい実行設定が自動で適用されるわけではない。

## 6. 完了・速度・結果を確認する

必要な記録:

| ファイル（`AMSRR_RUN`を基準） | 確認内容 |
| --- | --- |
| `collection_binding.json` | backend、並列数、checkpoint・source/config identity、diagnosticでないこと |
| `commands.jsonl` | 各prepare/executeの時間・終了・worker。物理エラーがないこと |
| `panel/physics_schedule.json` | 全物理jobの実行優先順位 |
| `panel/summary.json` | 全448試行、成功数、報酬、planning rejection、安全失敗 |
| `panel/completed.json` | collection完了、archive一覧 |
| `optimizer_completed.json` | PPO完了、次update番号、checkpoint hash、**update全体のtotal_seconds** |
| `AMSRR_RUN_guard/guard_result.json` | 正常終了とguard全体のseconds |
| `AMSRR_RUN_guard/health.jsonl` | 温度・メモリ・空き容量・trip/OOM等 |

§5の実行例の集計:

```bash
"$AMSRR_PY" - "$AMSRR_RUN" <<'PY'
import json
import sys
from pathlib import Path

run = Path(sys.argv[1])
guard_root = Path(str(run) + "_guard")
binding = json.loads((run / "collection_binding.json").read_text())
summary = json.loads((run / "panel/summary.json").read_text())["update_16"]
result = json.loads((run / "optimizer_completed.json").read_text())
guard = json.loads((guard_root / "guard_result.json").read_text())
health = [json.loads(line) for line in (guard_root / "health.jsonl").read_text().splitlines()]
assert guard["exit_code"] == 0 and guard["error"] is None
assert not any(row.get("trip") for row in health)
assert not binding["diagnostic_subset"] and not binding["benchmark"]
assert not binding["physical_outcomes_reused"] and binding["optimizer_invoked"]
assert summary["count"] == len(summary["rows"]) == 448
assert len({(row["case"], row["seed"]) for row in summary["rows"]}) == 448
assert result["update_index"] == 17
rejected = sum(row["planning_rejected"] for row in summary["rows"])
print({"trials": 448, "physical_trials": 448 - rejected, "planning_rejected": rejected,
       "successes": summary["successes"], "safety_failures": summary["safety_failures"],
       "mean_reward": summary["mean_reward"], "update_seconds": result["total_seconds"],
       "guard_seconds": guard["seconds"], "original_speedup": 16971.134313666 / result["total_seconds"],
       "checkpoint_sha256": result["checkpoint_sha256"]})
PY
```

最速測定の照合値:

- 448試行、計画拒否208、物理実行240、成功219、安全失敗0。
- 平均報酬 `3.8613838839285592`。
- 入力update16 SHA256: `9bde39ebda448b024126326c3923322d2337d0bc01e79185d4a0b62f4b794341`。
- 出力update17 SHA256: `3f3872dcaca5de2f129641a0f97fd191c31a63c367897d6dc6075668efe558a8`。

これは同じ固定条件を再現するときの照合値。次の学習updateや異なる条件へ固定する目標値ではない。
プロセスの終了だけ、あるいは`timing.json`だけでPPOまで完了したと判断しない。

## 7. 今回の最速測定を固定ソースから再現する

次の手順は今回と同じ `/home/leus/amsrr`、同じCPU/Python環境、元のprotocol・入力データ・
nativeライブラリを保持したホスト用。新しいコードや別ホストでの測定には§3〜§6を使う。

保存先: `artifacts/p4_full/order9/request_ppo/20260929_full_update_throughput/`。

- `initial_replica_20_warm_command.json`: 実際に実行したargv。
- `initial_replica_cpu_source/` と `initial_replica_cpu_source_manifest.json`: 固定ソースとhash。
- `initial_replica_cpu_mount.sh`: private mount namespace内で `amsrr/`、`scripts/`、
  `build/request_dynamics_native/` を固定snapshotへread-onlyでbindする。
- 設定、データ、計画キャッシュ、posture IKのbuildは外側の同じリポジトリを参照する。
  snapshotだけを別PCへコピーしても同一条件にはならない。

snapshotを検証し、出力先3か所だけを新しくして実行する例:

```bash
"$AMSRR_PY" - <<'PY'
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

root = Path("artifacts/p4_full/order9/request_ppo/20260929_full_update_throughput")
snapshot = root / "initial_replica_cpu_source"
manifest = json.loads((root / "initial_replica_cpu_source_manifest.json").read_text())
for relative, expected in manifest.items():
    assert hashlib.sha256((snapshot / relative).read_bytes()).hexdigest() == expected, relative
# guardはprivate mountの外で起動するため、外側の実装も照合する。
guard_path = "scripts/run_guarded_request.py"
assert hashlib.sha256(Path(guard_path).read_bytes()).hexdigest() == manifest[guard_path]
old = root / "initial_replica_20_warm"
new = root / ("reproduce_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
replace = {str(old) + suffix: str(new) + suffix for suffix in ("", "_guard", "_update_17")}
assert all(not Path(destination).exists() for destination in replace.values())
argv = json.loads((root / "initial_replica_20_warm_command.json").read_text())
argv = [replace.get(argument, argument) for argument in argv]
Path(str(new) + "_command.json").write_text(json.dumps(argv, indent=2) + "\n")
subprocess.run(argv, check=True)
print("collection:", new)
PY
```

§6の集計時は、この新しいcollectionパスを `AMSRR_RUN` に設定する。
現在のcollectorは計測snapshotから未使用のscene JSON読込だけを除去しており、全39jobの
優先順位が同じことは `final_schedule_check.json` に記録してある。
固定snapshotのファイル、古い実験output、本学習のcheckpointを編集・上書きしない。

## 8. 遅くなったときの切分け

最初にbinding・時間内訳・healthを見る。闇雲に並列数や学習回数を増やさない。

| 症状 | 最初の確認と対処 |
| --- | --- |
| 初回処理が遅い | `initial_workers=4`か。省略時は直列。checkpoint/source/split検査を省いて高速化しない |
| 計画段階が遅い | `plan_cache.json`のhitを確認。ソース等の変化によるmissは正常。初回生成と再利用を分けて計測 |
| 物理実行が直列に戻った | `workers=20`、`prepare_workers=12`、`device=cpu`をbindingで確認。古いrunnerを呼んでいないか追う |
| CPU競合・メモリ回収が多い | guardの`cgroup_memory_events.high`、使用量、thread設定を確認。他の学習jobとの同時実行を避ける |
| 終盤のjobが止まる | `recycle_after_jobs=2`、worker log、job receiptを確認。無制限再利用の停止事例がある。自動再試行を重ねない |
| PPOだけ遅い | `optimizer_device=cuda:0`、`optimizer.log`を確認。PPO引数が元のprotocolと一致しているか確認 |
| native/ABI/hashエラー | §3.2のload/buildへ戻る。古いbinaryを無理に読ませたりhash照合を無効化したりしない |
| kernel/GPUエラー、OOM、資源下限 | guardの停止理由を確認して原因を解消する。guardを外して再実行しない |

再採用しない設定・試作:

- 24並列RAM方式: メモリ回収で遅くなり中断。並列数を増やせば速くなるとは限らない。
- `--recycle-after-jobs 0`: 無制限再利用で38/39job後に進行が止まった測定がある。
- fileログを高速化の既定値にすること: 20並列で473.254秒、比較RAM方式449.661秒より遅い。
  file方式は明示的なメモリ圧力対策としてのみ検討する。
- 物理実行と監査の先行並行処理、環境間隔変更、SAP broadphase、PhysX1thread、
  効果や数値同等性が確認できなかったcompile・算術並べ替え・clone削減。
- 短いGPU試作のFPSから448試行全体の速度を推定し、実測した値として報告すること。

さらに変更する場合は、原因仮説を一つ定め、局所検査→少数の実経路→全updateの順に比較する。
操作・物理状態・判定の一致と速度の両方を確認し、失敗試行を集計から除かない。

## 9. 中断・ログ保存・引継ぎ

- 中断時は起動したguardへSIGINTを送り、子processの終了と `guard_result.json` を確認する。
  guardが応答しない場合は、その `guard_binding.json` の `unit` に限定してsystemdで停止する。
  `pkill python` 等で他の作業まで停止しない。未完了のupdateを採用しない。
- 原本の移動・圧縮中に速度を計測しない。比較実験のrawログを圧縮する場合は、展開後SHA256が
  元と一致することを確認し、元パス・archiveパス・hash・復元方法をmanifestへ記録してから
  rawの重複を除く。学習に必要な `training_rollout.pt`、checkpoint、protocolを一括削除しない。
- 今回の圧縮済み比較ログは
  [lossless_log_archives.json](../artifacts/p4_full/order9/request_ppo/20260929_full_update_throughput/lossless_log_archives.json)
  に記録されている。必要なファイルだけ `zstd -d --keep -- <rollout.pt.zstのパス>` で復元し、
  manifestの元SHA256と照合する。最速runのrawログは圧縮せず保持した。
- 変更・実測値・採用構成・未達条件は [WORKLOG.md](WORKLOG.md) に記録する。
  設計上の境界は [設計補足](AMSRR_design_modification_by_codex.md) を参照する。
  今回の速度測定用checkpointは本学習の最良モデルへ自動昇格していない。

この手順書の作成時にはシミュレーション・学習を再実行していない。上記の秒数は保存済みの全件実測であり、
現在の40倍達成や、別の機体・物体条件でも同じ所要時間になることを意味しない。

## 10. 固定protocolで継続学習・最良update選定を行う

全panelの収集は引き続き`run_request_collection.py`を使う。反復、検証間隔、収束判定、
近傍update評価は [run_request_training.py](../scripts/run_request_training.py) が担当する。
以前のartifact配下の学習runnerには戻さない。

継続runには次を保存する。

- `protocol.json`: 元のtrain/validation/test分割、条件、報酬・学習設定を保持。
- `execution_contract.json`: 再開update/hash、基準update、検証間隔、最小更新数、
  停滞回数、報酬許容差、近傍半径を実行前に固定。
- `schedule_inputs.json`: 基準checkpoint一覧、固定source/manifest、保護入力hash、
  guardを含む全件検証argvの`command_template`パス。
- コマンドtemplate: §4/§7の最速構成で、`--update-index`/`--ppo-output`を含まない
  全件検証コマンド。各stageのoutput/checkpointと訓練時の2引数だけをscheduleが変更する。

現在のR2継続runの保存設定を再開する入口は以下。既に実行中なら二重起動しない。

```bash
python scripts/run_request_training.py \
  --output artifacts/p4_full/order9/request_ppo/20260929_r2_fast_training
```

`schedule.lock`は同一runの二重起動を拒否する。完了stageはguard・全件数・case/seed・
checkpoint/source identityを照合して採用する。不完全stageや入力変更は自動再試行せず停止する。
旧backendの部分評価を新backendの評価に継ぎ足さない。

`learning_curve.json`は各更新の収集policyによる訓練結果、`validation_curve.json`はgreedy全件評価。
両者は分けて読む。`best.json`は暫定選定、`completed.json`が停滞条件と近傍検証を満たした選定結果。
検証での最良選定は、test性能やR2全例成功を意味しない。

各stage終了後、raw `rollout.pt`だけを展開SHA検証付きで圧縮し、
`collections/<stage>/raw_log_archives.json`に復元先を保存する。
PPO入力・checkpoint・監査・summaryは保持する。圧縮はシミュレーションと重ねない。

R2継続中の補足: 学習が進むと同時実行する履歴の量も変わる。20並列RAMでも
`memory.high`回収が増えて停滞する場合があった。資源上限は引き上げず、
既存の`--cpu-log-storage file`へ切り替える。全448/PPOでRAMとcheckpointまで同一だった
`initial_file_20_warm`が同等性の証拠。`schedule_inputs.json`の
`training_file_storage_from_update`で、どの収集policy updateから訓練だけをfile保存に
するか明示できる。実行中の入力は変更しない。中断stageの証拠と旧bindingを保存し、
PPO未実行を確認して同じcheckpoint/seedで全件収集し直す。既存の完了updateを再学習しない。

## 拡充R2条件の継続学習での補足（2026-09-30）

- 訓練poolは`20260929_r2_expanded_conditions/dataset.json`の812条件、29個の28条件panel。
  同じpanelで16draw/条件を使い448episode/updateとする。検証42/test14は変更しない。
  `schedule_inputs.json`の`training_dataset`とSHAを拘束して既存scheduleへ渡す。
  全poolを一巡する前に「収束」としない（今回最小30更新、3更新ごと検証、停滞3回、近傍±2）。
  panelが違う訓練平均は直接の改善指標ではなく、固定検証の推移で選ぶ。
- CPU rolloutのactor確率をCUDA PPOで再計算すると、float32誤差が既存replay閾値2e-5を
  僅かに超えた実例がある。現在のCPU収集には既存`--ppo-device cpu`を使用する。
  閾値緩和や一部event除外はしない。約32秒のPPO処理を含めて全update時間を測る。
- コピー環境は同一のauthored物体poseであることを確認し、共有reset-bank生成前に
  settled基準env0のposeへ明示初期化する。既に一致/単環境はno-op。env0の状態と
  元のstrict readback検査を維持し、`request_reset_alignment.json`で適用を記録する。
- 把持補正が安全指令を作れなかった場合は、その試行を失敗終端としてPPOへ返す。
  `controller_safety_rejected`をバッチ全体の異常と混同しない。request runnerは拒否候補を
  actuatorへ適用せず、該当simulated episodeのthrust/effort/velocityを0にし、関節目標を
  現在角度へ固定する。他envを継続し、終了済みenvの結果は変更しない。
  これはsimulation打切り処理で、実機飛行の非常制御として使用する機構ではない。
  未知例外・NaN・内部速度制限違反は従来どおり停止する。
- 実装変更の検証は局所unit→問題1条件の実経路→基準固定42個別結果一致の順で行う。
  sourceを実行途中で差し替えない。診断後の新runは`previous_training_curve`とSHAを渡し、
  最新checkpoint/optimizer・panel offset・`validation_anchor_update`を継承する。
  前runの失敗guardやcheckpointを上書きしない。既存完了runを新学習先に流用しない。
- 今回の継続履歴は`20260929_r2_expanded_training`→`..._reset_resume`→`..._safety_resume`。
  各runの`continuation.json`/source manifest/保存argvを参照する。最終採用モデルは最新runの
  `completed.json`を確認し、最大のupdate番号を自動的に最良とみなさない。

- 続行履歴に`..._planning_resume`を追加。押し込みIKの通常不成立を既存の計画拒否へ分類する
  漏れを補完した実行で、38checkpointと全22更新を継承している。再開先は各runの
  `continuation.json`を辿る。Isaac起動前のcarbプラグイン相互待ちは根治済みではない。
  ログ更新停止/物理未開始/stackで既知障害と確認した場合のみ、失敗attemptを保持して
  同じseedとcheckpointで再実行する。PPO後の成功を求める再抽選と混同しない。

## 2026-09-30: 812条件のR2正式学習での運用結果

- 完了先: `artifacts/p4_full/order9/request_ppo/20260929_r2_expanded_training_planning_resume`。update16から30更新/13,440episode、固定42検証計798episode。全訓練条件使用と近傍検証完了、採用は元`20260927_r2_two_contact_training/update_16`。詳細・再現順は`training_report.md`/4runの`execution_contract.json`と`commands`を参照。
- 20 execution/12 prepare workers、1 thread/worker、CPU dynamics/GPU broadphase、file training logs、verified zstd archive、CPU2000%/RAM44GiB/swap0/no-turbo/GPU250Wを使用。CPU PPOは収集/replayのfloat32誤差を同deviceへ揃えるため23以降に使用し、on-policy閾値2e-5は緩めない。
- cold training panelは概ね14–18分/448episode（新しいIK準備を含む）。全回の時間は`combined_learning_curve.json`。一時起動障害後の計画キャッシュ利用は物理結果再利用ではなく、同じ全448試行の再収集。cached準備の短い時間をcold全件の速度と混同しない。
- 4回のIsaac起動停止は2-threadのlibcarb.settings/dictionary plugin相互待ちがstackで一致した。worker全終了と失敗証跡保管後に同じstage全件を1度ずつ再収集。原因が違う停止へこの手順を盲目的に適用しない。これはライブラリ起動不具合の根治ではない。
- 新しい条件では、通常のbounded IK不成立/制御安全拒否をepisode failureとして記録し、全batch例外にしない現行runnerを使用。失敗候補を無条件許可したり報酬・判定を緩めない。凍結sourceは`runtime/source_manifest.json`で照合。
- `execution_statistics.json`: CPU最高55°C/GPU58°C、空きRAM最小22.9GiB/disk43.8GiB、trip/OOM killなし。完了した学習stage7.24h・検証1.21h、正式開始から終了まで10.75h。実行基盤を修正して続行した時間を省略しない。

## 2026-10-01: 中心anchor優先モデルへの移行

中心を優先しつつ周辺も選べるv4モデルを使う場合は、次の順で再開する。

1. 既存入口 `scripts/train_request_imitation.py configure-anchor-preference --checkpoint INPUT --output NEW_DIRECTORY` で移行する。初期設定は `--distance-decay 0.6931471805599453 --kl-coefficient 0.05`。元checkpointを上書きせず、Adam状態と完了update番号を保持する。移行自体は学習updateではない。
2. 現行sourceを新しいruntime snapshotへ固定する。`amsrr/policies/anchor_preference.py`を含め、従来のsnapshot・旧rollout・既存実験protocolは書き換えない。旧runtimeではv4 checkpointは読めない。
3. `grasp_anchor_source=neighborhood`・二点把持で、新checkpointから新規収集する。距離metadataを持たないall_free/旧sceneのまま実行しない。速度の実行方式とCPU/RAM/電力guardは既存手順を維持する。
4. 同一条件の成功率・元の平均報酬に加え、summaryの `mean_anchor_distance` と `center_selections` を比較する。PPOのprior KL・更新前後中心確率も確認する。prior損失の改善をタスク報酬改善として報告しない。

初回の移行済みモデルは `artifacts/p4_full/order9/request_ppo/20261001_anchor_preference_implementation/update_22/checkpoint.pt`。
元best22から重み・optimizerを維持した設定変更版であり、新方式でPPOを学習済みのモデルではない。
初回観測・距離・収集時確率の整合確認は同runの `preflight.py` / `preflight.json`、移行履歴は `anchor_preference_migration.json`。


## 2026-10-03: 新規候補の計画生成を高速化する手順

対象はπ_Hの接触割当から実行軌道を生成するCPU計画部分。指定anchor方式を維持する。
この節の秒数をIsaac実行や448episode全updateの速度に読み替えない。

1. 現行ソースの次の変更を一組で使用する。`order9_posture_collision.py`のメッシュ形状共有、
   `order9_articulated_teacher.py`の窓間resolver再利用、`rigid_body_model.py`の重心専用計算、
   `whole_structure_kinematics.py`の局所姿勢共有、`articulated_reachability.py`・
   `naive_contact_planner.py`・`order9_virtual_contact_compression.py`の重心計算器再利用、
   `native/order9_posture_native.cpp`の衝突状態・関節形状の重複計算除去。
2. §3.2の既存手順で `PYTHON="$AMSRR_PY" bash scripts/build_order9_posture_native.sh` を実行する。
   ソースだけを更新して古い拡張ライブラリを使わない。新runのsource snapshotには
   `build/order9_posture_native/`のライブラリとbuild-identityも含め、両方を読取専用で固定する。
   過去runの凍結source/binaryは変更しない。古いsnapshotで再開してもこの高速化は適用されない。
3. 局所検証は `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$AMSRR_PY" -m pytest -q
   tests/unit/feasibility/test_planner_computation_reuse.py` と関連計画器試験を使用する。
   plugin無効化は環境のROS用pytest pluginの自動読込を避けるためで、依存packageは変更しない。
4. 速度比較では同じcase・checkpoint・seed・要求IDを固定して、既存prepare入口の
   `--no-plan-cache`を使う。フォルダ名group_1と実際のcontact_group_idは異なるので、
   `decision.json`も照合する。採用条件は計画の成否に加え、数値軌道、衝突検査、拒否理由の一致。
5. 通常runは既存の検証付き計画キャッシュを維持してよい。今回の内部キャッシュは
   保存済みの完成軌道とは別で、1候補を新規に計画するときにも有効。学習報酬・物理結果は再利用しない。
   既存のCPU/RAM/温度guard、1数値thread/workerを維持し、今回の速度を理由に並列数を増やさない。

Pythonの位置・重心計算を既存C++数式へ丸ごと置換した案は不採用。
初回IKで約1e-11の差が生じ、後段探索で成功例の不成立または軌道延長を観測した。
採用版は同じ浮動小数点計算を再利用し、衝突余裕・探索seed数・検査間隔を変えていない。
詳細証跡: `artifacts/p4_full/order9/request_ppo/20261002_planner_acceleration/`。
同directoryの`report.json`、`runtime_optimized`、`summarize.py`を参照する。

### 計測結果と適用範囲

| 対象 | 修正前 | 修正後 | 倍率 |
| --- | ---: | ---: | ---: |
| 4モジュール・成立候補 | 107.1秒 | 23.8秒 | 4.49倍 |
| 6モジュール・成立候補 | 70.3秒 | 17.8秒 | 3.95倍 |
| 8モジュール・不成立候補 | 114.1秒 | 105.6秒 | 1.08倍 |
| 8モジュール・成立候補 | 137.7秒 | 51.4秒 | 2.68倍 |

4候補の合計時間は429.2秒→198.7秒（2.16倍）。単一worker、数値演算1thread、CPU quota200%、RAM上限12GiB、新規prepare全体を測定。成立3候補の数値軌道・全衝突検査、拒否1候補の理由、全候補の選択結果は旧実装と完全一致。

局所試験はrelease_tests.logの71件とcompression_tests.logの5件が成功。旧バイナリとの実機体形状の衝突照合72通りも完全一致。不成立候補の計画時間は105.6秒残るため、順位順に候補を試す方式の最悪時間を51.4秒/候補だけで見積もらない。今回、ランキング探索自体の実装やPPO/Isaacの再実行は行っていない。

最終snapshotはruntime_optimized。不成立候補は直前のruntime_releaseで計測したが、両者の差は計画成立後に呼ぶ押し込み計算への重心計算器の受渡し4行のみ。不成立時には未到達であり、rejection_evidence_reuse.jsonに根拠を保存。無関係な変更後に過去の計測を流用するための例外にはしない。

### Ranked first-feasible collection (2026-10-03)

- Checkpoint `ranked_contact.version=first_feasible_plackett_luce_v1` samples a without-replacement contact ranking. One initial training event records only the tried prefix and its joint conditional probability; unused suffixes must not split otherwise identical physics batches.
- Keep bounded per-candidate planning and the whole-ranking worker/RPC budgets distinct. For K legal candidates, the collector currently budgets K times the candidate timeout plus30s. Applying the candidate timeout as one whole-worker SIGALRM prematurely cancels valid later attempts. `request_execution_worker.serve` now uses the whole RPC deadline for prepare. Physics retains its separate execution deadline.
- Share completed deterministic candidate plans/rejections under content identities and per-key locks; never reuse physical rollouts. Waiting for the cache owner is not solver compute time. Timeout remains an interrupted collection, not geometric rejection feedback.
- New run evidence: `artifacts/p4_full/order9/request_ppo/20261003_r2_ranked_from_update8/`. Completed checkpoints9/10 each used448 fresh training episodes and42 fresh validation episodes. Formal source snapshots are immutable. Current production includes the deadline/resume corrections after the third collection interruption; freeze current source for the next run and start from this run's update10 plus its optimizer state. Do not silently reuse interrupted third-collection data as a finished update.
