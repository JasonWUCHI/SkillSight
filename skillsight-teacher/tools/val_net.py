# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Multi-view test a video classification model."""

import numpy as np
import os
import pickle
import torch
from fvcore.common.file_io import PathManager
import cv2
from einops import rearrange, reduce, repeat
import scipy.io

import timesformer.utils.checkpoint as cu
import timesformer.utils.distributed as du
import timesformer.utils.logging as logging
import timesformer.utils.misc as misc
import timesformer.visualization.tensorboard_vis as tb
from timesformer.datasets import loader
from timesformer.models import build_model
from timesformer.utils.meters import TestMeter
# import sys

logger = logging.get_logger(__name__)


@torch.no_grad()
def perform_val(test_loader, model, test_meter, cfg, writer=None):
    """
    For classification:
    Perform mutli-view testing that uniformly samples N clips from a video along
    its temporal axis. For each clip, it takes 3 crops to cover the spatial
    dimension, followed by averaging the softmax scores across all Nx3 views to
    form a video-level prediction. All video predictions are compared to
    ground-truth labels and the final testing performance is logged.
    For detection:
    Perform fully-convolutional testing on the full frames without crop.
    Args:
        test_loader (loader): video testing loader.
        model (model): the pretrained video model to test.
        test_meter (TestMeter): testing meters to log and ensemble the testing
            results.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter object, optional): TensorboardWriter object
            to writer Tensorboard log.
    """
    # Enable eval mode.
    model.eval()
    test_meter.iter_tic()

    for cur_iter, (inputs, inputs_cropped, gaze, labels, video_idx, start_time, end_time, scenario_id, subtask_id,meta) in enumerate(test_loader):
        if cfg.NUM_GPUS:
            # Transfer the data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
                    inputs_cropped[i] = inputs_cropped[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
                inputs_cropped = inputs_cropped.cuda(non_blocking=True)

            # Transfer the data to the current GPU device.
            labels = labels.cuda()
            video_idx = video_idx.cuda()
            start_time, end_time = start_time.cuda(), end_time.cuda()
            # for key, val in meta.items():
            #     if isinstance(val, (list,)):
            #         for i in range(len(val)):
            #             val[i] = val[i].cuda(non_blocking=True)
            #     else:
            #         meta[key] = val.cuda(non_blocking=True)
        test_meter.data_toc()

        if cfg.DETECTION.ENABLE:
            # Compute the predictions.
            preds = model(inputs, meta["boxes"])
            ori_boxes = meta["ori_boxes"]
            metadata = meta["metadata"]

            preds = preds.detach().cpu() if cfg.NUM_GPUS else preds.detach()
            ori_boxes = (
                ori_boxes.detach().cpu() if cfg.NUM_GPUS else ori_boxes.detach()
            )
            metadata = (
                metadata.detach().cpu() if cfg.NUM_GPUS else metadata.detach()
            )

            if cfg.NUM_GPUS > 1:
                preds = torch.cat(du.all_gather_unaligned(preds), dim=0)
                ori_boxes = torch.cat(du.all_gather_unaligned(ori_boxes), dim=0)
                metadata = torch.cat(du.all_gather_unaligned(metadata), dim=0)

            test_meter.iter_toc()
            # Update and log stats.
            test_meter.update_stats(preds, ori_boxes, metadata)
            test_meter.log_iter_stats(None, cur_iter)
        
        else:
            if cfg.MODEL.MODEL_NAME == "GazeOnlyTransformer":
                preds = model(gaze, scenario_id)
            elif cfg.MODEL.MODEL_NAME == "Skillformer":
                preds = model(inputs, inputs_cropped, gaze, scenario_id)
            elif cfg.MODEL.MODEL_NAME == "SkillSightTeacher":
                preds = model(inputs, inputs_cropped, gaze, scenario_id)
            elif cfg.MODEL.MODEL_NAME == "ImageTransformer":
                preds = model(inputs, inputs_cropped, gaze, scenario_id)
            elif cfg.MODEL.MODEL_NAME == "VisionTransformerOnly":
                preds = model(inputs)
            elif cfg.MODEL.MODEL_NAME == "E2GOMOTION":
                preds = model(inputs_cropped)
            elif cfg.MODEL.MODEL_NAME == "Beholder":
                preds = model(inputs, inputs_cropped, gaze)
            elif cfg.MODEL.MODEL_NAME == "EgoExoLearn":
                preds = model(inputs_cropped)
            elif cfg.MODEL.MODEL_NAME == "X3D_Self":
                preds = model(inputs)
            elif cfg.MODEL.MODEL_NAME == "ListenToLook":
                preds = model(inputs, gaze)
            elif cfg.MODEL.MODEL_NAME == "EgoDistill_pretrain":
                student_feat, teacher_feat = model(inputs, gaze)
            elif cfg.MODEL.MODEL_NAME == "EgoDistill":
                student_feat, preds, teacher_feat, teacher_preds = model(inputs, gaze)
            elif cfg.MODEL.MODEL_NAME == "DistillTransformer3":
                _,_, preds, _,_ = model(inputs, inputs_cropped, gaze, scenario_id, True)
            elif cfg.MODEL.MODEL_NAME == "DistillTransformer2":
                student_feat, teacher_feat, preds, teacher_preds, teacher_preds2 = model(inputs, inputs_cropped, gaze, scenario_id)
            else:
                preds, teacher_preds = model(inputs, inputs_cropped, gaze, scenario_id)

            # Gather all the predictions across all the devices to perform ensemble.
            if cfg.NUM_GPUS > 1:
                preds, labels, video_idx, start_time, end_time, distill_loss = du.all_gather(
                    [preds, labels, video_idx, start_time, end_time, torch.tensor([0.0], device=preds.device)]
                )
            if cfg.NUM_GPUS:
                preds = preds.cpu()
                labels = labels.cpu()
                video_idx = video_idx.cpu()
                start_time, end_time = start_time.cpu(), end_time.cpu()
                distill_loss = distill_loss.cpu()

            test_meter.iter_toc()
            # Update and log stats.

            test_meter.update_stats(
                preds.detach(), labels.detach(), video_idx.detach()
            )
            test_meter.update_segments(
                preds.detach(), labels.detach(), video_idx.detach(), start_time.detach(), end_time.detach(), test_loader.dataset.task_names, distill_loss.detach()
            )
            test_meter.log_iter_stats(cur_iter)

        test_meter.iter_tic()

    # Log epoch stats and print the final testing results.
    if not cfg.DETECTION.ENABLE:
        all_preds = test_meter.video_preds.clone().detach()
        all_labels = test_meter.video_labels
        if cfg.NUM_GPUS:
            all_preds = all_preds.cpu()
            all_labels = all_labels.cpu()
        if writer is not None:
            writer.plot_eval(preds=all_preds, labels=all_labels)

        if cfg.TEST.SAVE_RESULTS_PATH != "":
            save_path = os.path.join(cfg.OUTPUT_DIR, cfg.TEST.SAVE_RESULTS_PATH)

            with PathManager.open(save_path, "wb") as f:
                pickle.dump([all_labels, all_preds], f)

            logger.info(
                "Successfully saved prediction results to {}".format(save_path)
            )

    test_meter.save_segment_results(cfg.OUTPUT_DIR, "val")
    test_meter.finalize_metrics(ks=(1, 2))
    return test_meter


def val(cfg):
    """
    Perform multi-view testing on the pretrained video model.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    # Print config.
    logger.info("Test with config:")
    logger.info(cfg)

    # Build the video model and print model statistics.
    model = build_model(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=False)

    cu.load_test_checkpoint(cfg, model)

    # Create video testing loaders.
    test_loader = loader.construct_loader(cfg, "val")
    real_test_loader = loader.construct_loader(cfg, "test")
    logger.info("Testing model for {} iterations".format(len(test_loader)))
    print("num classes:")
    print(cfg.MODEL.NUM_CLASSES)

    # assert (
    #     len(test_loader.dataset)
    #     % (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS)
    #     == 0
    # )
    # Create meters for multi-view testing.
    test_meter = TestMeter(
        len(test_loader.dataset)//test_loader.dataset._num_clips,
        test_loader.dataset._num_clips,
        cfg.MODEL.NUM_CLASSES,
        len(test_loader),
        cfg.DATA.MULTI_LABEL,
        cfg.DATA.ENSEMBLE_METHOD,
    )

    # Set up writer for logging to Tensorboard format.
    if cfg.TENSORBOARD.ENABLE and du.is_master_proc(
        cfg.NUM_GPUS * cfg.NUM_SHARDS
    ):
        writer = tb.TensorboardWriter(cfg)
    else:
        writer = None

    # # Perform multi-view test on the entire dataset.
    test_meter = perform_val(test_loader, model, test_meter, cfg, writer)
    if writer is not None:
        writer.close()