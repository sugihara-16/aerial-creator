# Order-9 C3 promoted release: update 18

## Authoritative result

- Release ID: `c3_pi_l_promoted_update18_v1`
- Final checkpoint SHA-256:
  `6ea412ccdfe983cb2b030b3514bc8982424522673b49def188d547120984357b`
- Formal promotion manifest SHA-256:
  `366ff7eeb32885db296d8ffbfc0aa61826ecbf93ef3404adc69f05baf33eb923`
- Formal result: 14 held-out buckets, 32 episodes per bucket, 448/448
  successes, zero safety failures, zero fallback, promoted.
- Machine-readable inventory and every protected-file hash:
  `for_codex/C3_PROMOTED_UPDATE18_RELEASE_LEDGER.json`.
- Read-only protected artifact:
  `artifacts/p4_full/order9/releases/c3_pi_l_promoted_update18_v1`.

The source checkpoint, protected checkpoint, source promotion manifest, and
protected promotion manifest are byte-identical to the hashes above.

## C2 to C3 training procedure

1. The promoted C2 update-49 representation was preserved. Its checkpoint
   SHA-256 is
   `85474d9da96a6eb729e7f21fae5f7628fd0d9defc39488c74fe71e10476c6274`.
2. The representation was migrated through the versioned C3 action contracts
   to the clean v9 initializer. The retained v9 initializer SHA-256 is
   `080e4c50e2179a9a5bc7e2b1dc12c3cd2cf58506714c576d5639400e26262bc0`.
   The migration manifests and retained intermediate checkpoints are listed
   in the release ledger. Two older intermediate binaries (v5 and the
   pre-rebind v6 initializer) were already absent before this cleanup; their
   manifests and the retained v6 rebound checkpoint preserve that transition.
3. One common v9 policy/action contract was used for all module counts. There
   was no module-count-specific output mask or evaluation rule. The nominal
   contact preload is morphology/contact/load conditioned, rounded upward to
   1 mm, has a 2 mm physical minimum, and has no universal 12 mm margin. The
   bounded learned contact-normal residual remains a separate categorical
   policy output.
4. PPO expanded the shared policy while replaying earlier module counts:

   | PPO updates | Training module counts |
   | --- | --- |
   | 0--3 | 2--3 |
   | 4--5 | 2--4 |
   | 6--7 | 2--5 |
   | 8--9 | 2--6 |
   | 10--16 | 2--7 |
   | 17 | 2--7 |
   | 18 | 2--8 |

5. After PPO update 17, the contact-normal head received the retained
   seven-module probe-teacher auxiliary update. Before PPO update 18, the
   eight-module preload-deficit teacher was applied. PPO still updated the
   complete actor; these teacher passes did not replace PPO and did not consume
   a PPO update index.
6. The exact hash-bound generation chain resolves from update 0 through update
   18 (19 PPO generations). The final actor-enabled formal promotion run then
   evaluated all 14 held-out buckets from phase zero.

## What is required for the promoted result

The release ledger classifies source and regression coverage into two groups:

- `promotion_critical_or_regression_coverage`: runtime policy/action decoding,
  contact-space projection, categorical contact-normal action, physical
  preload calculation, QPID/virtual-thrust feasibility behavior, phase and
  reward semantics, tensor rollout/PPO, checkpoint lineage, nominal/bucket
  byte validation, execution bundles, promotion evaluation, and their tests.
- `historical_diagnostic_or_reproduction_tool`: ablation, migration,
  comparison, screening, visualization, and diagnostic tools that explain or
  reproduce the path to the final method but are not invoked by promoted
  inference or formal promotion.

All lightweight source in both groups is retained. This avoids deleting useful
diagnostic implementation merely because its large generated outputs are no
longer needed. The exact per-file classification is in
`implementation_inventory` in the release ledger.

## Training/evaluation data retention

Retained and protected:

- final, C2, initializer, selected update 0--18, and teacher checkpoints;
- compact stage, PPO, dataset, teacher, and promotion manifests;
- accepted bucket and nominal runtime trees, including the separately stored
  validation-bucket-33 component;
- phase-reset banks and the compact evidence required to resolve the training
  lineage;
- formal promotion JSON/JSONL evidence.

Deleted:

- superseded/failed lineages and ablation checkpoints;
- obsolete nominal candidates, screens, and diagnostic outputs;
- duplicate evaluation tensors;
- reproducible retained-lineage `raw/` rollout tensors and
  `evaluation_rollout.pt` files after their hash-bound compact records were
  retained.

The cleanup removed 650,876,272,640 allocated bytes (606.176 GiB). Deleted
superseded data is not directly recoverable.

## Cleanup dependency audit

The first direct-file preflight did not traverse one sibling dependency:
validation bucket 33 in the accepted 43-bucket nominal set. The production
nominal-set validator detected the missing edge immediately after deletion.
All 41 component files were restored from a read-only ext4 data-block scan and
accepted only when they matched the SHA-256 values already recorded before
cleanup. Key restored hashes are:

- component manifest: `a8542e1f...edc9104`
- collision admission: `30895472...e6ca7`
- bucket manifest: `f70bbfc0...8b0b5`
- timeline: `fee4be00...d52b`

The cleanup tool now retains this component explicitly and runs the production
transitive nominal-set validator on both source and protected copies before
any future deletion. Final source and protected validation both pass 43/43.

## Verification commands

```bash
python scripts/cleanup_order9_c3_promoted_release_artifacts.py \
  --report /tmp/c3-cleanup-audit.json
```

This command is dry-run by default. A valid post-cleanup tree reports 180
protected files, zero candidate deletions, zero direct-ledger errors, zero
transitive-runtime errors, and `safety_audit_passed: true`.

Do not use `--apply` for routine verification.
