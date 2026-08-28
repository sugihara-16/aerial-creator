# R1名目制御較正v5 結果

## 判定

最小範囲`r1_l1_10mm_5deg`は、承認済みv5契約の条件を満たさず不合格である。
採用範囲はなく、教師軌道の正式収集と学習を許可しない。

## 結果概要

- Isaacを使わない一次判定は、学習側22機体の682件すべてが合格した。
- 一次判定後のIsaac確認は、最初の格子点6件を各2回、合計12回実行した。
- Isaac成功は6/12、安全違反は6、代替制御への切替は0だった。
- 格子点100%かつ安全違反0の条件が回復不能になったため、第1段階の残り、
  第2--第4段階、および未使用14機体による最終確認は実行しなかった。
- 実行時間は843.054秒で、承認上限36000秒以内に完了した。

## 時間短縮の境界

- 軌道の幾何学的な形、終端関節姿勢、接触点の割当は変更していない。
- 接近と接触獲得の所要時間を元の0.5倍、以後の各区間を0.1倍とした。
- 時間変更後に関節速度条件を再検査した。
- 代表1件は事前のIsaac確認2/2成功、安全違反0、代替制御0だったが、正式な
  22機体全体の判定を代用しない。
- π_Lの仕組みと保護済みチェックポイントは保持したが、π_L由来の動作量は
  適用していない。名目押し込み、QPID/QP、局所サーボ、安全制限は有効である。

## 固定した根拠

- 正式結果:
  `artifacts/p4_full/order9/r1_teacher/calibration_v7_identical_path_retime/calibration_result_v5.json`
  (`b11713024799f6ca78ab9b5f402eefe354e9d92c36c573b32ae9ccd75723d044`)
- 承認済みv5契約:
  `configs/training/order9_r1_nominal_calibration_protocol_v5.yaml`
  (`713c69a31017f5280e06e9a35eb3793cfbb12d1cd4f3d9a81fa756e7b213948e`)
- v5承認記録:
  `for_codex/R1_NOMINAL_CALIBRATION_PROTOCOL_V5_APPROVAL.json`
  (`0583371f4553d13a4ba9311aae480e497e44ae028253be37f29b67f05f4e2da0`)
- R1教師上書きv2:
  `configs/training/order9_r1_teacher_overrides_v2.json`
  (`9fdf852bc821c3e11a51d6b3af3b026585cd8eeaa4592b25478efc224d496082`)
- 保護済みC3チェックポイント:
  `6ea412ccdfe983cb2b030b3514bc8982424522673b49def188d547120984357b`

機械可読の規範目録は`for_codex/R1_NOMINAL_CALIBRATION_V5_RESULT_LEDGER.json`で
ある。v1--v4の結果は履歴として保持し、上書きしない。
