# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Train a video classification model."""

import numpy as np
import pprint
import torch
import torch.nn as nn
import torch.nn.functional as F
from fvcore.nn.precise_bn import get_bn_modules, update_bn_stats

import timesformer.models.losses as losses
import timesformer.models.optimizer as optim
import timesformer.utils.checkpoint as cu
import timesformer.utils.distributed as du
import timesformer.utils.logging as logging
import timesformer.utils.metrics as metrics
import timesformer.utils.misc as misc
import timesformer.visualization.tensorboard_vis as tb
from timesformer.datasets import loader
from timesformer.models import build_model
from timesformer.utils.meters import TrainMeter, ValMeter
from timesformer.utils.multigrid import MultigridSchedule

from timm.data import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
import torch.nn.functional as F

logger = logging.get_logger(__name__)


def train_epoch(
    train_loader, model, optimizer, train_meter, cur_epoch, cfg, writer=None
):
    """
    Perform the video training for one epoch.
    Args:
        train_loader (loader): video training loader.
        model (model): the video model to train.
        optimizer (optim): the optimizer to perform optimization on the model's
            parameters.
        train_meter (TrainMeter): training meters to log the training performance.
        cur_epoch (int): current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
    """
    # Enable train mode.
    model.train()
    train_meter.iter_tic()
    data_size = len(train_loader)

    cur_global_batch_size = cfg.NUM_SHARDS * cfg.TRAIN.BATCH_SIZE
    num_iters = cfg.GLOBAL_BATCH_SIZE // cur_global_batch_size

    for cur_iter, (inputs, inputs_cropped, gaze, labels, video_idx, start_time, end_time,scenario_id, subtask_id,meta) in enumerate(train_loader):
        # Transfer the data to the current GPU device.
        device = next(model.parameters()).device
        if cfg.NUM_GPUS:
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
                    inputs_cropped[i] = inputs_cropped[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
                inputs_cropped = inputs_cropped.cuda(non_blocking=True)
            labels = labels.cuda()
            gaze = gaze.to(device, non_blocking=True)
            scenario_id = scenario_id.to(device, non_blocking=True)
            subtask_id = subtask_id.to(device, non_blocking=True)
            for key, val in meta.items():
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        val[i] = val[i].cuda(non_blocking=True)
                else:
                    meta[key] = val.cuda(non_blocking=True)

        # Update the learning rate.
        if cfg.SOLVER.OPTIMIZING_METHOD == "sgd":
            lr = optim.get_epoch_lr(cur_epoch + float(cur_iter) / data_size, cfg)
            optim.set_lr(optimizer, lr)
        else:
            lr = optimizer.param_groups[0]['lr']

        train_meter.data_toc()

        # Explicitly declare reduction to mean.
        if not cfg.MIXUP.ENABLED:
           loss_fun = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)(reduction="mean")
        #    loss_fun_full = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)(reduction="none")
        else:
           mixup_fn = Mixup(
               mixup_alpha=cfg.MIXUP.ALPHA, cutmix_alpha=cfg.MIXUP.CUTMIX_ALPHA, cutmix_minmax=cfg.MIXUP.CUTMIX_MINMAX, prob=cfg.MIXUP.PROB, switch_prob=cfg.MIXUP.SWITCH_PROB, mode=cfg.MIXUP.MODE,
               label_smoothing=0.1, num_classes=cfg.MODEL.NUM_CLASSES)
           hard_labels = labels
           inputs, labels = mixup_fn(inputs, labels)
           loss_fun = SoftTargetCrossEntropy()

        if cfg.DETECTION.ENABLE:
            preds = model(inputs, meta["boxes"])
        elif cfg.MODEL.MODEL_NAME == "GazeOnlyTransformer":
            preds = model(gaze, scenario_id)
        elif cfg.MODEL.MODEL_NAME == "VisionTransformerOnly":
            preds = model(inputs)
        elif cfg.MODEL.MODEL_NAME == "Skillformer":
            preds = model(inputs, inputs_cropped, gaze, scenario_id)
        elif cfg.MODEL.MODEL_NAME == "SkillSightTeacher":
            preds = model(inputs, inputs_cropped, gaze, scenario_id)
        elif cfg.MODEL.MODEL_NAME == "ImageTransformer":
            preds = model(inputs, inputs_cropped, gaze, scenario_id)
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
            student_feat, teacher_feat, preds, teacher_preds, subtask_pred = model(inputs, inputs_cropped, gaze, scenario_id, True)
        elif cfg.MODEL.MODEL_NAME == "DistillTransformer2":
            student_feat, teacher_feat, preds, teacher_preds, teacher_preds2 = model(inputs, inputs_cropped, gaze, scenario_id, True)
        else:
            preds, teacher_preds = model(inputs, inputs_cropped, gaze, scenario_id)

    
        # Compute the loss.
        if cfg.MODEL.MODEL_NAME != "EgoDistill_pretrain":
            loss = loss_fun(preds, labels)
        # print(gaze.get_device(), scenario_id.get_device())
        # print(preds[0], inputs[0].get_device(), gaze[0].get_device(), gaze[1].get_device(), scenario_id)
        
        # class_weights = loss_weight[scenario_id].to(preds.device)[0]
        # loss = F.cross_entropy(preds, labels, weight=class_weights)
        # print(loss.shape,  scenario_id.shape)

        if cfg.MODEL.MODEL_NAME == "EgoDistill":
            loss_fun_distill = losses.get_loss_func(cfg.MODEL.DISTILL_LOSS_FUNC)(reduction="mean")
            distill_loss = losses.clip_loss(student_feat, teacher_feat, temperature=0.07)
            temperature = 4.0
            log_probs_student = F.log_softmax(preds/temperature, dim=1)
            probs_teacher = F.softmax(teacher_preds/temperature, dim=1)
            loss_fun_kl = losses.get_loss_func("kl_div")(reduction="batchmean")
            kl_loss = loss_fun_kl(log_probs_student, probs_teacher) * (temperature ** 2)
            loss = (1-cfg.MODEL.DISTILL_KL_WEIGHT)*loss + distill_loss*cfg.MODEL.DISTILL_LOSS_WEIGHT + kl_loss*cfg.MODEL.DISTILL_KL_WEIGHT


        if cfg.MODEL.MODEL_NAME == "DistillTransformer":
            temperature = 4.0
            log_probs_student = F.log_softmax(preds/temperature, dim=1)
            probs_teacher = F.softmax(teacher_preds/temperature, dim=1)
            loss_fun_kl = losses.get_loss_func("kl_div")(reduction="batchmean")
            kl_loss = loss_fun_kl(log_probs_student, probs_teacher) * (temperature ** 2)
            loss = (1-cfg.MODEL.DISTILL_KL_WEIGHT)*loss + kl_loss*cfg.MODEL.DISTILL_KL_WEIGHT


        if cfg.MODEL.MODEL_NAME == "DistillTransformer2":
            loss_fun_distill = losses.get_loss_func(cfg.MODEL.DISTILL_LOSS_FUNC)(reduction="mean")
            # print(student_feat.shape, teacher_feat.shape)
            # distill_loss = loss_fun_distill(student_feat, teacher_feat)
            distill_loss = losses.clip_loss(student_feat, teacher_feat, temperature=0.07)
            temperature = 4.0
            log_probs_student = F.log_softmax(preds/temperature, dim=1)
            probs_teacher = F.softmax(teacher_preds/temperature, dim=1)
            loss_fun_kl = losses.get_loss_func("kl_div")(reduction="batchmean")
            kl_loss = loss_fun_kl(log_probs_student, probs_teacher) * (temperature ** 2)
            # loss = loss + distill_loss*cfg.MODEL.DISTILL_LOSS_WEIGHT
            loss = (1-cfg.MODEL.DISTILL_KL_WEIGHT)*loss + distill_loss*cfg.MODEL.DISTILL_LOSS_WEIGHT # + kl_loss*cfg.MODEL.DISTILL_KL_WEIGHT

        if cfg.MODEL.MODEL_NAME == "DistillTransformer3":
            loss_fun_distill = losses.get_loss_func(cfg.MODEL.DISTILL_LOSS_FUNC)(reduction="mean")
            distill_loss = loss_fun_distill(student_feat, teacher_feat)
            # distill_loss = losses.clip_loss(student_feat, teacher_feat, temperature=0.07)
            action_loss = loss_fun(subtask_pred, subtask_id)
            # loss = loss + distill_loss*cfg.MODEL.DISTILL_LOSS_WEIGHT
            loss = (1-cfg.MODEL.DISTILL_KL_WEIGHT)*loss + action_loss  #distill_loss*cfg.MODEL.DISTILL_LOSS_WEIGHT 

        if cfg.MODEL.MODEL_NAME == "EgoDistill_pretrain":
            loss = losses.clip_loss(student_feat, teacher_feat, temperature=0.07)
            preds = torch.zeros(labels.shape[0], cfg.MODEL.NUM_CLASSES).to(labels.device)  # dummy preds
        # if cfg.MODEL.MODEL_NAME == "DistillTransformer2":
        #     temperature = 4.0
        #     log_probs_teacher = F.log_softmax(teacher_preds2/temperature, dim=1)
        #     probs_teacher = F.softmax(teacher_preds/temperature, dim=1)
        #     loss_fun_kl = losses.get_loss_func("kl_div")(reduction="batchmean")
        #     kl_teacher_loss = loss_fun_kl(log_probs_teacher, probs_teacher) * (temperature ** 2)
        #     loss = loss + kl_teacher_loss * cfg.MODEL.DISTILL_KL_WEIGHT_TEACHER

        if cfg.MIXUP.ENABLED:
            labels = hard_labels

       
        misc.check_nan_losses(loss)


        if cur_global_batch_size >= cfg.GLOBAL_BATCH_SIZE:
            # Perform the backward pass.
            optimizer.zero_grad()
            loss.backward()
            # Update the parameters.
            optimizer.step()
        else:
            if cur_iter == 0:
                optimizer.zero_grad()
            loss.backward()
            if (cur_iter + 1) % num_iters == 0:
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad /= num_iters
                optimizer.step()
                optimizer.zero_grad()

        if cfg.DETECTION.ENABLE:
            if cfg.NUM_GPUS > 1:
                loss = du.all_reduce([loss])[0]
            loss = loss.item()

            # Update and log stats.
            train_meter.update_stats(None, None, None, loss, lr)
            # write to tensorboard format if available.
            if writer is not None:
                writer.add_scalars(
                    {"Train/loss": loss, "Train/lr": lr},
                    global_step=data_size * cur_epoch + cur_iter,
                )

        else:
            top1_err, top4_err = None, None
            if cfg.DATA.MULTI_LABEL:
                # Gather all the predictions across all the devices.
                if cfg.NUM_GPUS > 1:
                    [loss] = du.all_reduce([loss])
                loss = loss.item()
            else:
                # Compute the errors.
                num_topks_correct = metrics.topks_correct(preds, labels, (1, 2))
                top1_err, top4_err = [
                    (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
                ]
                # Gather all the predictions across all the devices.
                if cfg.NUM_GPUS > 1:
                    loss, top1_err, top4_err = du.all_reduce(
                        [loss, top1_err, top4_err]
                    )

                # Copy the stats from GPU to CPU (sync point).
                loss, top1_err, top4_err = (
                    loss.item(),
                    top1_err.item(),
                    top4_err.item(),
                )

            # Update and log stats.
            train_meter.update_stats(
                top1_err,
                top4_err,
                loss,
                lr,
                inputs[0].size(0)
                * max(
                    cfg.NUM_GPUS, 1
                ),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
            )
            # write to tensorboard format if available.
            if writer is not None:
                writer.add_scalars(
                    {
                        "Train/loss": loss,
                        "Train/lr": lr,
                        "Train/Top1_err": top1_err,
                        "Train/Top4_err": top4_err,
                    },
                    global_step=data_size * cur_epoch + cur_iter,
                )

        train_meter.iter_toc()  # measure allreduce for this meter
        train_meter.log_iter_stats(cur_epoch, cur_iter)
        train_meter.iter_tic()

    # Log epoch stats.
    train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()

'''
@torch.no_grad()
def eval_epoch(val_loader, model, val_meter, cur_epoch, cfg, writer=None):
    """
    Evaluate the model on the val set.
    Args:
        val_loader (loader): data loader to provide validation data.
        model (model): model to evaluate the performance.
        val_meter (ValMeter): meter instance to record and calculate the metrics.
        cur_epoch (int): number of the current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
    """

    # Evaluation mode enabled. The running stats would not be updated.
    model.eval()
    val_meter.iter_tic()

    for cur_iter, (inputs, gaze, labels, _,_,_,scenario_id, meta) in enumerate(val_loader):
        if cfg.NUM_GPUS:
            # Transferthe data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda()
            for key, val in meta.items():
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        val[i] = val[i].cuda(non_blocking=True)
                else:
                    meta[key] = val.cuda(non_blocking=True)
        val_meter.data_toc()

        if cfg.DETECTION.ENABLE:
            # Compute the predictions.
            preds = model(inputs, meta["boxes"])
            ori_boxes = meta["ori_boxes"]
            metadata = meta["metadata"]

            if cfg.NUM_GPUS:
                preds = preds.cpu()
                ori_boxes = ori_boxes.cpu()
                metadata = metadata.cpu()

            if cfg.NUM_GPUS > 1:
                preds = torch.cat(du.all_gather_unaligned(preds), dim=0)
                ori_boxes = torch.cat(du.all_gather_unaligned(ori_boxes), dim=0)
                metadata = torch.cat(du.all_gather_unaligned(metadata), dim=0)

            val_meter.iter_toc()
            # Update and log stats.
            val_meter.update_stats(preds, ori_boxes, metadata)

        elif cfg.MODEL.MODEL_NAME == "GazeOnlyTransformer":
            # Perform the forward pass.
            preds = model(gaze, scenario_id)
            if cfg.DATA.MULTI_LABEL:
                if cfg.NUM_GPUS > 1:
                    preds, labels = du.all_gather([preds, labels])
            else:
                # Compute the errors.
                num_topks_correct = metrics.topks_correct(preds, labels, (1, 4))

                # Combine the errors across the GPUs.
                top1_err, top4_err = [
                    (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
                ]
                if cfg.NUM_GPUS > 1:
                    top1_err, top4_err = du.all_reduce([top1_err, top4_err])

                # Copy the errors from GPU to CPU (sync point).
                top1_err, top4_err = top1_err.item(), top4_err.item()

                val_meter.iter_toc()
                # Update and log stats.
                val_meter.update_stats(
                    top1_err,
                    top4_err,
                    inputs[0].size(0)
                    * max(
                        cfg.NUM_GPUS, 1
                    ),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
                )
                # write to tensorboard format if available.
                if writer is not None:
                    writer.add_scalars(
                        {"Val/Top1_err": top1_err, "Val/Top4_err": top4_err},
                        global_step=len(val_loader) * cur_epoch + cur_iter,
                    )

            val_meter.update_predictions(preds, labels)

        elif cfg.MODEL.MODEL_NAME == "GazeEgoQueryTransformer":
            # Perform the forward pass.
            preds = model(inputs, gaze)
            if cfg.DATA.MULTI_LABEL:
                if cfg.NUM_GPUS > 1:
                    preds, labels = du.all_gather([preds, labels])
            else:
                # Compute the errors.
                num_topks_correct = metrics.topks_correct(preds, labels, (1, 4))

                # Combine the errors across the GPUs.
                top1_err, top4_err = [
                    (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
                ]
                if cfg.NUM_GPUS > 1:
                    top1_err, top4_err = du.all_reduce([top1_err, top4_err])

                # Copy the errors from GPU to CPU (sync point).
                top1_err, top4_err = top1_err.item(), top4_err.item()

                val_meter.iter_toc()
                # Update and log stats.
                val_meter.update_stats(
                    top1_err,
                    top4_err,
                    inputs[0].size(0)
                    * max(
                        cfg.NUM_GPUS, 1
                    ),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
                )
                # write to tensorboard format if available.
                if writer is not None:
                    writer.add_scalars(
                        {"Val/Top1_err": top1_err, "Val/Top4_err": top4_err},
                        global_step=len(val_loader) * cur_epoch + cur_iter,
                    )

            val_meter.update_predictions(preds, labels)
        
        else:
            preds = model(inputs)

            if cfg.DATA.MULTI_LABEL:
                if cfg.NUM_GPUS > 1:
                    preds, labels = du.all_gather([preds, labels])
            else:
                # Compute the errors.
                num_topks_correct = metrics.topks_correct(preds, labels, (1, 4))

                # Combine the errors across the GPUs.
                top1_err, top4_err = [
                    (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
                ]
                if cfg.NUM_GPUS > 1:
                    top1_err, top4_err = du.all_reduce([top1_err, top4_err])

                # Copy the errors from GPU to CPU (sync point).
                top1_err, top4_err = top1_err.item(), top4_err.item()

                val_meter.iter_toc()
                # Update and log stats.
                val_meter.update_stats(
                    top1_err,
                    top4_err,
                    inputs[0].size(0)
                    * max(
                        cfg.NUM_GPUS, 1
                    ),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
                )
                # write to tensorboard format if available.
                if writer is not None:
                    writer.add_scalars(
                        {"Val/Top1_err": top1_err, "Val/Top4_err": top4_err},
                        global_step=len(val_loader) * cur_epoch + cur_iter,
                    )

            val_meter.update_predictions(preds, labels)

        val_meter.log_iter_stats(cur_epoch, cur_iter)
        val_meter.iter_tic()

    # Log epoch stats.
    val_meter.log_epoch_stats(cur_epoch)
    # write to tensorboard format if available.
    if writer is not None:
        if cfg.DETECTION.ENABLE:
            writer.add_scalars(
                {"Val/mAP": val_meter.full_map}, global_step=cur_epoch
            )
        else:
            all_preds = [pred.clone().detach() for pred in val_meter.all_preds]
            all_labels = [
                label.clone().detach() for label in val_meter.all_labels
            ]
            if cfg.NUM_GPUS:
                all_preds = [pred.cpu() for pred in all_preds]
                all_labels = [label.cpu() for label in all_labels]
            writer.plot_eval(
                preds=all_preds, labels=all_labels, global_step=cur_epoch
            )

    val_meter.reset()
'''

@torch.no_grad()
def eval_epoch(val_loader, model, val_meter, cur_epoch, cfg, writer=None):
    """
    Evaluate the model on the val set.
    Args:
        val_loader (loader): data loader to provide validation data.
        model (model): model to evaluate the performance.
        val_meter (ValMeter): meter instance to record and calculate the metrics.
        cur_epoch (int): number of the current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
    """

    # Evaluation mode enabled. The running stats would not be updated.
    model.eval()
    val_meter.iter_tic()

    for cur_iter, (inputs, inputs_cropped, gaze, labels, video_idx, start_time, end_time,scenario_id, subtask_id,meta) in enumerate(val_loader):
        device = next(model.parameters()).device
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
            gaze = gaze.to(device, non_blocking=True)
            start_time, end_time = start_time.cuda(), end_time.cuda()
            # for key, val in meta.items():
            #     if isinstance(val, (list,)):
            #         for i in range(len(val)):
            #             val[i] = val[i].cuda(non_blocking=True)
            #     else:
            #         meta[key] = val.cuda(non_blocking=True)
        val_meter.data_toc()

        if cfg.DETECTION.ENABLE:
            # JW: Not used at all, don't trust this code
            # Compute the predictions.
            preds = model(inputs, meta["boxes"])
            ori_boxes = meta["ori_boxes"]
            metadata = meta["metadata"]

            if cfg.NUM_GPUS:
                preds = preds.cpu()
                ori_boxes = ori_boxes.cpu()
                metadata = metadata.cpu()

            if cfg.NUM_GPUS > 1:
                preds = torch.cat(du.all_gather_unaligned(preds), dim=0)
                ori_boxes = torch.cat(du.all_gather_unaligned(ori_boxes), dim=0)
                metadata = torch.cat(du.all_gather_unaligned(metadata), dim=0)

            val_meter.iter_toc()
            # Update and log stats.
            val_meter.update_stats(preds, ori_boxes, metadata)

        else:
            if cfg.MODEL.MODEL_NAME == "GazeOnlyTransformer":
                preds = model(gaze, scenario_id)
            elif cfg.MODEL.MODEL_NAME == "Skillformer":
                preds = model(inputs, inputs_cropped, gaze, scenario_id)
            elif cfg.MODEL.MODEL_NAME == "VisionTransformerOnly":
                preds = model(inputs)
            elif cfg.MODEL.MODEL_NAME == "SkillSightTeacher":
                preds = model(inputs, inputs_cropped, gaze, scenario_id)
            elif cfg.MODEL.MODEL_NAME == "ImageTransformer":
                preds = model(inputs, inputs_cropped, gaze, scenario_id)
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
                preds = torch.zeros(labels.shape[0], cfg.MODEL.NUM_CLASSES).to(labels.device)  # dummy preds
            elif cfg.MODEL.MODEL_NAME == "EgoDistill":
                student_feat, preds, teacher_feat, teacher_preds = model(inputs, gaze)
            elif cfg.MODEL.MODEL_NAME == "DistillTransformer3":
                _,_, preds, _,_ = model(inputs, inputs_cropped, gaze, scenario_id, True)
            elif cfg.MODEL.MODEL_NAME == "DistillTransformer2":
                student_feat, teacher_feat, preds, teacher_preds, teacher_preds2 = model(inputs, inputs_cropped, gaze, scenario_id)
            else:
                preds, teacher_preds = model(inputs, inputs_cropped, gaze, scenario_id)

            if cfg.DATA.MULTI_LABEL:
                if cfg.NUM_GPUS > 1:
                    preds, labels = du.all_gather([preds, labels])
            else:
                # Compute the errors.
                num_topks_correct = metrics.topks_correct(preds, labels, (1, 2))

                # Combine the errors across the GPUs.
                top1_err, top4_err = [
                    (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
                ]
                if cfg.NUM_GPUS > 1:
                    top1_err, top4_err = du.all_reduce([top1_err, top4_err])

                # Copy the errors from GPU to CPU (sync point).
                top1_err, top4_err = top1_err.item(), top4_err.item()

                val_meter.iter_toc()
                # Update and log stats.
                val_meter.update_stats(
                    top1_err,
                    top4_err,
                    inputs[0].size(0)
                    * max(
                        cfg.NUM_GPUS, 1
                    ),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
                )
                # write to tensorboard format if available.
                if writer is not None:
                    writer.add_scalars(
                        {"Val/Top1_err": top1_err, "Val/Top4_err": top4_err},
                        global_step=len(val_loader) * cur_epoch + cur_iter,
                    )

            val_meter.update_predictions(preds, labels)

            # JW: record ensemble stats
            if cfg.NUM_GPUS > 1:
                preds, labels, video_idx, start_time, end_time = du.all_gather(
                    [preds, labels, video_idx, start_time, end_time]
                )
            if cfg.NUM_GPUS:
                preds = preds.cpu()
                labels = labels.cpu()
                video_idx = video_idx.cpu()
                start_time, end_time = start_time.cpu(), end_time.cpu()
            val_meter.update_test_stats(preds, labels, video_idx)

        val_meter.log_iter_stats(cur_epoch, cur_iter)
        val_meter.iter_tic()

    # Log epoch stats.
    val_meter.log_epoch_stats(cur_epoch)
    # write to tensorboard format if available.
    if writer is not None:
        if cfg.DETECTION.ENABLE:
            writer.add_scalars(
                {"Val/mAP": val_meter.full_map}, global_step=cur_epoch
            )
        else:
            all_preds = [pred.clone().detach() for pred in val_meter.all_preds]
            all_labels = [
                label.clone().detach() for label in val_meter.all_labels
            ]
            if cfg.NUM_GPUS:
                all_preds = [pred.cpu() for pred in all_preds]
                all_labels = [label.cpu() for label in all_labels]
            writer.plot_eval(
                preds=all_preds, labels=all_labels, global_step=cur_epoch
            )

    #JW: calculate the essemble accuracy
    acc = 0
    acc = val_meter.get_ensemble_accuracy(ks=(1, 2))

    val_meter.reset()

    return acc

def calculate_and_update_precise_bn(loader, model, num_iters=200, use_gpu=True):
    """
    Update the stats in bn layers by calculate the precise stats.
    Args:
        loader (loader): data loader to provide training data.
        model (model): model to update the bn stats.
        num_iters (int): number of iterations to compute and update the bn stats.
        use_gpu (bool): whether to use GPU or not.
    """

    def _gen_loader():
        for inputs, *_ in loader:
            if use_gpu:
                if isinstance(inputs, (list,)):
                    for i in range(len(inputs)):
                        inputs[i] = inputs[i].cuda(non_blocking=True)
                else:
                    inputs = inputs.cuda(non_blocking=True)
            yield inputs

    # Update the bn stats.
    update_bn_stats(model, _gen_loader(), num_iters)

def build_trainer(cfg):
    """
    Build training model and its associated tools, including optimizer,
    dataloaders and meters.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    Returns:
        model (nn.Module): training model.
        optimizer (Optimizer): optimizer.
        train_loader (DataLoader): training data loader.
        val_loader (DataLoader): validatoin data loader.
        precise_bn_loader (DataLoader): training data loader for computing
            precise BN.
        train_meter (TrainMeter): tool for measuring training stats.
        val_meter (ValMeter): tool for measuring validation stats.
    """
    # Build the video model and print model statistics.
    model = build_model(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=True)

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)

    # Create the video train and val loaders.
    train_loader = loader.construct_loader(cfg, "train")
    val_loader = loader.construct_loader(cfg, "val")

    precise_bn_loader = loader.construct_loader(
        cfg, "train", is_precise_bn=True
    )
    # Create meters.
    train_meter = TrainMeter(len(train_loader), cfg)
    val_meter = ValMeter(len(val_loader), cfg)

    return (
        model,
        optimizer,
        train_loader,
        val_loader,
        precise_bn_loader,
        train_meter,
        val_meter,
    )

def train(cfg):
    """
    Train a video model for many epochs on train set and evaluate it on val set.
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

    # Init multigrid. JW: Not sure what this is
    multigrid = None
    if cfg.MULTIGRID.LONG_CYCLE or cfg.MULTIGRID.SHORT_CYCLE:
        multigrid = MultigridSchedule()
        cfg = multigrid.init_multigrid(cfg)
        if cfg.MULTIGRID.LONG_CYCLE:
            cfg, _ = multigrid.update_long_cycle(cfg, cur_epoch=0)
    # Print config.
    logger.info("Train with config:")
    logger.info(pprint.pformat(cfg))

    # Build the video model and print model statistics.
    model = build_model(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=True)

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)

    # Load a checkpoint to resume training if applicable.
    if not cfg.TRAIN.FINETUNE:
        start_epoch = cu.load_train_checkpoint(cfg, model, optimizer)
    elif cfg.TRAIN.CHECKPOINT_FILE_PATH != "":
        start_epoch = 0
        cu.load_checkpoint(cfg.TRAIN.CHECKPOINT_FILE_PATH, model)
    else:
        start_epoch = 0

    # Create the video train and val loaders.
    train_loader = loader.construct_loader(cfg, "train")
    val_loader = loader.construct_loader(cfg, "val")

    precise_bn_loader = (
        loader.construct_loader(cfg, "train", is_precise_bn=True)
        if cfg.BN.USE_PRECISE_STATS
        else None
    )

    #JW: For logging
    train_meter = TrainMeter(len(train_loader), cfg)
    val_meter = ValMeter(
        len(val_loader),
        len(val_loader.dataset)
        // (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS),
        cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS,
        cfg.MODEL.NUM_CLASSES,
         cfg)

    # set up writer for logging to Tensorboard format.
    if cfg.TENSORBOARD.ENABLE and du.is_master_proc(
        cfg.NUM_GPUS * cfg.NUM_SHARDS
    ):
        writer = tb.TensorboardWriter(cfg)
    else:
        writer = None

    # Perform the training loop.
    logger.info("Start epoch: {}".format(start_epoch + 1))

    validation_acc = 0
    for cur_epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):
        if cfg.MULTIGRID.LONG_CYCLE:
            cfg, changed = multigrid.update_long_cycle(cfg, cur_epoch)
            if changed:
                (
                    model,
                    optimizer,
                    train_loader,
                    val_loader,
                    precise_bn_loader,
                    train_meter,
                    val_meter,
                ) = build_trainer(cfg)

                # Load checkpoint.
                if cu.has_checkpoint(cfg.OUTPUT_DIR):
                    last_checkpoint = cu.get_last_checkpoint(cfg.OUTPUT_DIR)
                    assert "{:05d}.pyth".format(cur_epoch) in last_checkpoint
                else:
                    last_checkpoint = cfg.TRAIN.CHECKPOINT_FILE_PATH
                logger.info("Load from {}".format(last_checkpoint))
                cu.load_checkpoint(
                    last_checkpoint, model, cfg.NUM_GPUS > 1, optimizer
                )

        # Shuffle the dataset.
        loader.shuffle_dataset(train_loader, cur_epoch)

        # Train for one epoch.
        train_epoch(
            train_loader, model, optimizer, train_meter, cur_epoch, cfg, writer
        )

        is_checkp_epoch = cu.is_checkpoint_epoch(
            cfg,
            cur_epoch,
            None if multigrid is None else multigrid.schedule,
        )
        is_eval_epoch = misc.is_eval_epoch(
            cfg, cur_epoch, None if multigrid is None else multigrid.schedule
        )

        # Compute precise BN stats.
        if (
            (is_checkp_epoch or is_eval_epoch)
            and cfg.BN.USE_PRECISE_STATS
            and len(get_bn_modules(model)) > 0
        ):
            calculate_and_update_precise_bn(
                precise_bn_loader,
                model,
                min(cfg.BN.NUM_BATCHES_PRECISE, len(precise_bn_loader)),
                cfg.NUM_GPUS > 0,
            )
        _ = misc.aggregate_sub_bn_stats(model)

        # Save a checkpoint.
        if is_checkp_epoch:
            cu.save_checkpoint(cfg.OUTPUT_DIR, model, optimizer, cur_epoch, cfg)
        # Evaluate the model on validation set.
        if is_eval_epoch:
            acc = eval_epoch(val_loader, model, val_meter, cur_epoch, cfg, writer)
            if acc > validation_acc:
                print("New best validation accuracy: {}".format(acc))
                validation_acc = acc

    if writer is not None:
        writer.close()
