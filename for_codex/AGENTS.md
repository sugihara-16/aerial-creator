# AGENTS.md

## 目的

このファイルは、A-MSRR 構造構築知能の実装作業を行う coding assistant 向けの実務指示を定める。

設計書に出てくる **Agent** という語は、**work package / responsibility label**、すなわち「作業範囲・責務範囲を示すラベル」を意味する。これは、複数の自律的な AI coding agent を起動しなければならないという意味ではない。また、すべての work package を並列実装すべきという意味でもない。

タスクで「Agent X として作業する」と指示された場合は、次の意味として解釈すること。

```text
Agent X work package の範囲だけを担当する。
上流の schemas / interfaces を尊重する。
明示的に依頼されていない無関係な module は変更しない。
```

## Source of Truth

最新の設計書を source of truth として扱うこと。

```text
A-MSRR_codex_ready_spec_v0_4_ja.md
```

repository 内で `DESIGN_SPEC.md` などの canonical filename が使われている場合、その内容が最新仕様と一致しているときだけ active copy として扱う。

併せて次も読むこと。

```text
AGENTS.md
WORKLOG.md, if present
module_urdf/holon.urdf, when robot/module details are needed
```

文書化されていないプロジェクト文脈を仮定してはならない。repository に明示的に提供されていない限り、古いチャットログに依存してはならない。

## Work Package の扱い

設計書内の Agent labels は、実装計画上の ownership boundaries を表す。これらは、prompt scope、commit scope、review scope、handoff scope を明確にするために使う。

良い指示例:

```text
You are working on Agent X work package.
Implement only the requested files/modules.
Do not change schemas or upstream interfaces unless explicitly approved.
Update WORKLOG.md before finishing.
```

悪い指示例:

```text
You are Agent X. Build everything.
```

work package には依存関係がある。設計書の implementation order に従うこと。特に、schema、geometry、URDF/PhysicalModel、IRGBuilder、envelope、feasibility、interface tests が安定する前に policy training 関連の作業へ進んではならない。

また、ユーザーから指示があった場合は、次の作業における``Agent X work package``自体の提案もすること。

## 実装ルール
- 実装時は、まず該当箇所のv0.4設計書を確認してください。設計書に未定義な仕様については、システムに整合するようAMSRR_design_modification_by_codex.mdを作りましたので、に実装案を策定し、提示してください。また、設計書で指定された仕様では不都合がある場合についても同様に、修正設計案を提示してください。
- 設計書からの変更した内容や補足した内容は、worklogとは別にAMSRR_design_modification_by_codex.mdにも記録する
- schema-first implementation を優先する。
- 変更は、依頼された work package の範囲に限定し、最小限に保つ。
- schema fields、enum names、tensor shapes、masks、IDs、file paths を黙って変更してはならない。
- schema または interface の変更が必要な場合は、実装前に停止し、理由を報告する。
- deterministic safety checks を learned components で置き換えてはならない。
- `π_L` に final actuator commands を直接出力させてはならない。controller / QP / safety layer contract を使うこと。
- runtime robot model paths を hard-code してはならない。URDF paths は configurable でなければならない。
- tests は、実装対象 module の近くに置く。
## 実行環境

Isaac Lab 環境は micromamba で構築済みである。Isaac / simulator / GPU 機能が必要な場合は、既存の `isaaclab3` environment を使うこと。

`~/.bashrc` には次の shell setup がある前提でよい。

```bash
export ISAACLAB_PATH="$HOME/IsaacLab"
export OMNI_KIT_ACCEPT_EULA=YES

eval "$(~/.local/bin/micromamba shell hook -s bash)"

alias isaaclab="$ISAACLAB_PATH/isaaclab.sh"
alias ilab-gpu='python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"'

ilab() {
    micromamba activate isaaclab3
    cd "$ISAACLAB_PATH" || return
}
```

non-interactive shell では、必要に応じて shell configuration を明示的に読み込むこと。

```bash
source ~/.bashrc
micromamba activate isaaclab3
```

GPU 挙動が重要な場合は、`ilab-gpu` で active PyTorch/CUDA environment を確認する。

許可ある場合を除き、packages を global install してはならない。dependency が不足している場合は、`WORKLOG.md` に記録し、environment 変更前に確認を求めること。

## WORKLOG.md 運用

実装作業中は `WORKLOG.md` を維持すること。

`WORKLOG.md` には、次の 2 種類の記録を含める。

1. 通常の時系列 worklog。
2. Agent label ごとに grouped された work-package log。

これにより、「ある日に何を変更したか」と「どの work package がその変更を所有するか」を分けて追跡できる。

### 必須更新ルール

タスク完了前に、`WORKLOG.md` へ次を記録すること。

```text
- date
- active spec version
- assigned work package / Agent label
- summary of changes
- files changed
- schema/interface changes, or "None"
- upstream dependencies used
- downstream impact
- tests added or run
- commands run
- assumptions
- blockers / open questions
- next steps
```

過去の worklog entries を削除してはならない。

### 推奨 WORKLOG.md 構成

```md
# WORKLOG.md

## Global Worklog

### YYYY-MM-DD
- Spec version:
- Work package / Agent label:
- Summary:
- Files changed:
- Schema/interface changes:
- Commands run:
- Tests run:
- Assumptions:
- Blockers:
- Next steps:

---

## Work Package Logs

### Agent <label>: <short work package name>

#### YYYY-MM-DD
- Scope:
- Files changed:
- Upstream dependencies:
- Implemented:
- Not implemented:
- Schema/interface changes:
- Downstream impact:
- Tests added:
- Tests passed:
- Handoff notes:
- Open questions:
```

`WORKLOG.md` の Agent sections は進捗追跡用の区分であり、別々の自律 AI agent が使用されたことを意味しない。

## Testing Expectations

各 work package では、次を行うこと。

- 新しい schema、parser、planner、controller、policy-interface behavior に対して unit tests を追加または更新する。
- まず最小範囲の関連 tests を実行する。
- test commands と結果を `WORKLOG.md` に記録する。
- tests を実行できない場合は、その理由を記録する。

基盤となる schema、geometry、URDF/PhysicalModel、IRGBuilder、envelope、feasibility tests が通る前に、policy training や simulator-scale experiments へ進んではならない。

## Handoff Expectations

各タスクの終了時には、次を提示すること。

```text
- files changed
- tests run
- known limitations
- whether schemas/interfaces changed
- what the next work package needs to know
```

上流仕様の曖昧さにより作業が blocked された場合は、互換性のない仮定を勝手に作らず、停止して blocker を記録すること。