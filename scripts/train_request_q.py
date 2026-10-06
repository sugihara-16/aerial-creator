#!/usr/bin/env python3
"""Train a request action-value policy from verified physical training replay."""
import argparse
import torch
from amsrr.training.request_q_learning import train_q, migrate_action_context

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--replay', nargs='+')
    p.add_argument('--migrate-action-context', action='store_true',
                   help='One-time zero-context migration preserving old weights, Adam and target; no training.')
    p.add_argument('--output', required=True)
    p.add_argument('--steps', type=int, default=2000)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--learning-rate', type=float, default=1e-4)
    p.add_argument('--target-tau', type=float, default=.005)
    p.add_argument('--conservative-weight', type=float, default=0.)
    p.add_argument('--mode', choices=['monte_carlo', 'double_q'], default='double_q')
    p.add_argument('--seed', type=int, default=4105)
    p.add_argument('--device', choices=['cpu', 'cuda:0'], default='cpu')
    p.add_argument('--threads', type=int, default=2)
    args = vars(p.parse_args())
    torch.set_num_threads(args.pop('threads'))
    migrate = args.pop('migrate_action_context')
    args['replay_paths'] = args.pop('replay')
    if migrate:
        if args['replay_paths']:
            p.error('migration and training are separate stages')
        migrate_action_context(args['checkpoint'], args['output'])
    else:
        if not args['replay_paths']:
            p.error('--replay is required for training')
        train_q(**args)
