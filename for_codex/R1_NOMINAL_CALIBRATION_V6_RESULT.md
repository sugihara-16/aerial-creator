# R1名目制御較正v6 結果

## 判定

最小範囲`r1_l1_10mm_5deg`は、承認済みv6契約の条件を満たさず不合格である。
採用範囲はなく、未使用14機体による最終確認、教師軌道の正式収集、および学習を
許可しない。

## 結果概要

- Isaacを使わない一次判定は、学習側22機体の682件すべてが合格した。
- 一次判定後のIsaac確認は25候補を各2回、合計50回実行し、48回成功した。
- 6モジュール機`train-000004-190f3a425b3e`の`lattice_02`だけが2回とも搬送段階で
  時間切れとなった。安全違反、物体落下、QP計算不能、代替制御への切替はいずれも
  0だった。
- 格子点100%の条件が回復不能になったため、第1段階の残り、第2--第4段階、
  未使用14機体による最終確認を実行しなかった。したがって正式教師軌道は0件である。
- 実行時間は12348.750秒で、承認上限36000秒以内に完了した。

## 実行境界

- 完全軌道8段階と重要入力反映をIsaacより前に検査し、682/682件で合格した。
- 早期に不合格を確定するため、格子の角を優先して実行した。候補集合、合否条件、
  各候補2回の再現回数は変更していない。
- 10時間上限は最初の正式実行開始時刻から固定し、再開で延長しない。期限到達時には
  子処理群も停止し、未完了として収集・学習を禁止する仕組みを追加した。今回は期限内に
  正式な不合格が確定した。
- π_Lの仕組みと保護済みチェックポイントは保持したが、π_L由来の動作量は適用して
  いない。名目押し込み、QPID/QP、局所サーボ、安全制限は有効である。

## 固定した根拠

- 正式結果:
  `artifacts/p4_full/order9/r1_teacher/calibration_v8_short_retreat_safe_timing/calibration_result_v6.json`
  (`a9ca1d633d2fa42a960a798545ac34d45fafa4dae5e8e3c201f233aa465441e4`)
- 第1段階結果:
  `artifacts/p4_full/order9/r1_teacher/calibration_v8_short_retreat_safe_timing/selection/r1_l1_10mm_5deg/level_result_v6.json`
  (`354e8e39fe399dceb3c9c015912cb8b87f9e16ce4b14e32b26b425c39d8da1cf`)
- 決定的不合格候補:
  `artifacts/p4_full/order9/r1_teacher/calibration_v8_short_retreat_safe_timing/selection/r1_l1_10mm_5deg/r1_l1_10mm_5deg__train__train-000004-190f3a425b3e__lattice_02/nominal_result_v6.json`
  (`f02e4efff98ab728061cc9a5097c007d59c7bac9178cf87cb6127ff5e8314e1b`)
- 承認済みv6契約:
  `configs/training/order9_r1_nominal_calibration_protocol_v6.yaml`
  (`271c2f6791761ce608dc0160acafb0d7daa2b1c538e0da538f18bda70c6d5614`)
- v6承認記録:
  `for_codex/R1_NOMINAL_CALIBRATION_PROTOCOL_V6_APPROVAL.json`
  (`8e42cf53209982afc83315741bbe1245f7085385bd9575306dac666987d4265c`)
- 保護済みC3チェックポイント:
  `6ea412ccdfe983cb2b030b3514bc8982424522673b49def188d547120984357b`

機械可読の規範目録は`for_codex/R1_NOMINAL_CALIBRATION_V6_RESULT_LEDGER.json`で
ある。v1--v5の結果は履歴として保持し、上書きしない。
