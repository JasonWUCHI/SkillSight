#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Focused runner for SkillSight teacher training/evaluation.

from timesformer.utils.misc import launch_job
from timesformer.utils.parser import load_config, parse_args
from tools.test_net import test
from tools.train_net import train
from tools.val_net import val


def main():
    args = parse_args()
    if args.num_shards > 1:
        args.output_dir = str(args.job_dir)
    cfg = load_config(args)

    if cfg.TRAIN.ENABLE:
        launch_job(cfg=cfg, init_method=args.init_method, func=train)
    if cfg.VAL.ENABLE:
        launch_job(cfg=cfg, init_method=args.init_method, func=val)
    if cfg.TEST.ENABLE:
        launch_job(cfg=cfg, init_method=args.init_method, func=test)


if __name__ == '__main__':
    main()
