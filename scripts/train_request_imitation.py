#!/usr/bin/env python3
"""Prepare accepted R1 request demonstrations or train the current pi_H ranker."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from amsrr.training.request_imitation import prepare_dataset, train_policy
from amsrr.policies.request_high_level_policy import REQUEST_SELECTION_SCOPES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--limit-episodes", type=int)
    prepare.add_argument("--event-sampling", action="store_true")
    prepare.add_argument(
        "--selection-scope", choices=REQUEST_SELECTION_SCOPES, default="requests"
    )
    prepare.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/training/order9_learning_curriculum.yaml",
    )
    prepare.add_argument("--timeout-s", type=int, default=1200)
    train = sub.add_parser("train")
    train.add_argument("--dataset-manifest", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--patience", type=int, default=15)
    train.add_argument("--batch-size", type=int, default=256)
    train.add_argument("--learning-rate", type=float, default=0.001)
    train.add_argument("--seed", type=int, default=7)
    train.add_argument("--device", default="cuda")
    train.add_argument(
        "--stopping-rule", choices=("validation", "train_fit"), default="train_fit"
    )
    train.add_argument(
        "--evaluate-held-out",
        action="store_true",
        help="Evaluate held-out once after selection; omit during stage development.",
    )
    train.add_argument("--timeout-s", type=int, default=1200)
    warmup = sub.add_parser("warmup", help="Train the causal event actor used by PPO")
    warmup.add_argument("--dataset-manifest", type=Path, required=True)
    warmup.add_argument("--output", type=Path, required=True)
    warmup.add_argument("--epochs", type=int, default=1200)
    warmup.add_argument("--timeout-s", type=int, default=580)
    warmup.add_argument("--morphology-aware", action="store_true")
    temporal = sub.add_parser("warmup-temporal", help="Refit temporal warmup while preserving contact selection")
    temporal.add_argument("--checkpoint", type=Path, required=True)
    temporal.add_argument("--dataset", type=Path, required=True)
    temporal.add_argument("--output", type=Path, required=True)
    temporal.add_argument("--epochs", type=int, default=400)
    temporal.add_argument("--seed", type=int, default=17)
    temporal.add_argument("--device", default="cuda")
    temporal.add_argument("--timeout-s", type=int, default=240)
    ppo = sub.add_parser(
        "ppo", help="Update the same actor from categorical Isaac events"
    )
    ppo.add_argument("--checkpoint", type=Path, required=True)
    ppo.add_argument("--rollouts", type=Path, nargs="+", required=True)
    ppo.add_argument("--output", type=Path, required=True)
    ppo.add_argument("--timeout-s", type=int, default=580)
    ppo.add_argument("--epochs", type=int, default=4)
    ppo.add_argument("--learning-rate", type=float, default=1e-4)
    ppo.add_argument("--actor-scope", choices=("all", "temporal"), default="all")
    ppo.add_argument("--entropy-coefficient", type=float, default=0.001)
    ppo.add_argument("--target-kl", type=float)
    ppo.add_argument("--contact-loss-weight", type=float,
                     help="Weight of mean contact actor/entropy loss; requires both branch KL limits")
    ppo.add_argument("--contact-target-kl", type=float)
    ppo.add_argument("--temporal-target-kl", type=float)
    ppo.add_argument("--contact-group-manifest")
    ppo.add_argument("--contact-baseline-transition-from", choices=("value_v1", "condition_loo_v1"))
    ppo.add_argument("--training-profile", choices=("legacy_event_v1", "timed_mc_v1", "timed_value_v1"),
                     default="legacy_event_v1")
    ppo.add_argument("--value-epochs", type=int, default=80)
    args = parser.parse_args()
    if args.timeout_s <= 0:
        parser.error("timeout must be positive")

    def expired(_signal, _frame):
        raise TimeoutError("request imitation process deadline")

    signal.signal(signal.SIGALRM, expired)
    signal.alarm(args.timeout_s)
    import torch

    torch.set_num_threads(1 if args.action == "prepare" else 4)
    if args.action == "prepare":
        if args.limit_episodes is not None and not 1 <= args.limit_episodes <= 140:
            parser.error("limit must be 1--140")
        result = prepare_dataset(
            args.output.resolve(),
            limit_episodes=args.limit_episodes,
            timeout_s=args.timeout_s,
            config_path=args.config.resolve(),
            selection_scope=args.selection_scope,
            event_sampling=args.event_sampling,
        )
    elif args.action == "warmup":
        from amsrr.training.request_ppo import train_imitation

        result = train_imitation(
            args.dataset_manifest,
            args.output,
            epochs=args.epochs,
            timeout_s=args.timeout_s,
            morphology_aware=args.morphology_aware,
        )
    elif args.action == "warmup-temporal":
        from amsrr.training.request_temporal_warmup import warmup_temporal
        result = warmup_temporal(args.checkpoint, args.dataset, args.output,
            epochs=args.epochs, seed=args.seed, device=args.device, timeout_s=args.timeout_s)
    elif args.action == "ppo":
        from amsrr.training.request_ppo import ppo_update

        result = ppo_update(args.checkpoint, args.rollouts, args.output,
                            epochs=args.epochs, learning_rate=args.learning_rate,
                            actor_scope=args.actor_scope,
                            entropy_coefficient=args.entropy_coefficient,
                            target_kl=args.target_kl,
                            contact_loss_weight=args.contact_loss_weight,
                            contact_target_kl=args.contact_target_kl,
                            temporal_target_kl=args.temporal_target_kl,
                            training_profile=args.training_profile, value_epochs=args.value_epochs,
                            contact_group_manifest=args.contact_group_manifest,
                            contact_baseline_transition_from=args.contact_baseline_transition_from)
    else:
        if (
            min(args.epochs, args.patience, args.batch_size) < 1
            or args.learning_rate <= 0
        ):
            parser.error("training limits must be positive")
        result = train_policy(
            args.dataset_manifest.resolve(),
            args.output.resolve(),
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
            timeout_s=args.timeout_s,
            device=args.device,
            evaluate_held_out=args.evaluate_held_out,
            stopping_rule=args.stopping_rule,
        )
    signal.alarm(0)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
