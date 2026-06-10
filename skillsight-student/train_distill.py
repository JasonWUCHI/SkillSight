import argparse
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import SkillSightTeacherFeatureExtractor, SkillSightStudent
from dataset import Egoexo


def select_teacher_gaze_trajectory(gaze, target_dim):
    """Use the same gaze trajectory channels as the SkillSight teacher.

    The teacher path selects columns [0:13] and [15:] from the raw gaze CSV
    tensor, then feeds that trajectory to the gaze encoder. The released
    TimeSDinoGaze checkpoint uses a 15-channel gaze encoder, so pad/trim the
    selected trajectory to that width when the raw CSV column count differs.
    """
    gaze = torch.cat([gaze[:, :, 0:13], gaze[:, :, 15:]], dim=-1)
    cur_dim = gaze.shape[-1]
    if cur_dim < target_dim:
        gaze = F.pad(gaze, (0, target_dim - cur_dim))
    elif cur_dim > target_dim:
        gaze = gaze[:, :, :target_dim]
    return gaze


def train(args):
    train_dataset = Egoexo(mode="train", category=args.category, num_retries=args.num_retries)
    val_dataset = Egoexo(mode="test", category=args.category, num_retries=args.num_retries)

    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=False, num_workers=args.num_workers)

    gaze_length = 15
    teacher_model = SkillSightTeacherFeatureExtractor(args.teacher_checkpoint, gaze_length).to(args.device)
    teacher_model.eval()

    student_gaze_length = gaze_length
    fps_slice = 10 // args.fps

    model = SkillSightStudent(
        input_dim=student_gaze_length, d_model=768, nhead=12, num_layers=4,
        num_classes=4, max_len=80
    ).to(args.device)

    criterion = nn.CrossEntropyLoss()
    criterion_distill = nn.L1Loss(reduction="mean")
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    final_acc = [0, 0]
    final_result_dict = None

    for epoch in range(args.epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0

        gaze_tensor_4 = []
        labels_4 = []
        task_id_4 = []
        teacher_feat_4 = []
        for step, (frames, frames_crop, gaze, labels, idx, _, _, scenario_id, task_id, _) in tqdm(enumerate(train_loader)):
            frames, frames_crop = frames.to(args.device), frames_crop.to(args.device)
            gaze_tensor_4.append(gaze)
            labels_4.append(labels)
            task_id_4.append(task_id)

            gaze, labels = gaze.to(args.device), labels.to(args.device)
            teacher_gaze_input = gaze[:, ::5, :]  # Downsample gaze for SkillSight teacher model

            with torch.no_grad():
                teacher_feat = teacher_model(frames, frames_crop, teacher_gaze_input, scenario_id)
                teacher_feat_4.append(teacher_feat)

            if (step + 1) % args.accumulation_steps == 0 or step == len(train_dataset) // args.train_batch_size + 1:
                gaze = torch.cat(gaze_tensor_4, dim=0).to(args.device)
                labels = torch.cat(labels_4, dim=0).to(args.device)
                task_id = torch.cat(task_id_4, dim=0).to(args.device)
                teacher_feat = torch.cat(teacher_feat_4, dim=0).to(args.device)
                optimizer.zero_grad()
                student_gaze_input = select_teacher_gaze_trajectory(gaze[:, ::fps_slice, :], student_gaze_length)
                dis_out, target_feat, gaze_pred, subtask_preds = model(student_gaze_input, teacher_feat, True)

                loss = criterion(gaze_pred, labels)
                loss += criterion(subtask_preds, task_id) * args.subtask_loss_weight
                if args.distill_loss_weight > 0:
                    loss += criterion_distill(dis_out, target_feat) * args.distill_loss_weight
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * gaze.size(0)
                correct += (gaze_pred.argmax(dim=1) == labels).sum().item()
                total += labels.size(0)

                gaze_tensor_4 = []
                labels_4 = []
                task_id_4 = []
                teacher_feat_4 = []

        train_acc = correct / total if total else 0.0

        if epoch != args.epochs - 1:
            continue

        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        results = {}
        labels_list = {}
        result_dict = {}
        with torch.no_grad():
            for frames, frames_crop, gaze, labels, idx, start, end, scenario_id, task_id, _ in tqdm(val_loader):
                gaze, labels = gaze.to(args.device), labels.to(args.device)
                student_gaze_input = select_teacher_gaze_trajectory(gaze[:, ::fps_slice, :], student_gaze_length)
                dis_out, target_feat, gaze_pred, subtask_preds = model(student_gaze_input, training=False)
                loss = criterion(gaze_pred, labels)

                val_loss += loss.item() * gaze.size(0)
                for i in range(gaze.size(0)):
                    video_idx = (idx[i] // val_dataset._num_clips).item()
                    if video_idx not in results:
                        results[video_idx] = []
                        result_dict[video_idx] = []
                        labels_list[video_idx] = labels[i].item()
                    results[video_idx].append(gaze_pred[i].cpu())
                    result_dict[video_idx].append({
                        "path": val_dataset._path_to_videos[idx[i]],
                        "seg_num": idx[i].item() % val_dataset._num_clips,
                        "start": start[i].item(),
                        "end": end[i].item(),
                        "pred": gaze_pred[i].cpu().tolist(),
                    })

        output = [[0, 0, 0, 0] for _ in range(4)]
        for video_idx in results:
            label = labels_list[video_idx]
            avg_pred = torch.stack(results[video_idx], dim=0).mean(dim=0, keepdim=True)
            pred = avg_pred.argmax(dim=1).item()
            output[label][pred] += 1

        val_correct = 0
        val_total = 0
        bi_correct = 0
        for i in range(4):
            val_correct += output[i][i]
            val_total += sum(output[i])
            if i <= 1:
                bi_correct += output[i][0] + output[i][1]
            else:
                bi_correct += output[i][2] + output[i][3]

        val_acc = val_correct / val_total if val_total else 0.0
        bi_acc = bi_correct / val_total if val_total else 0.0
        final_acc[0] = val_acc
        final_acc[1] = max(final_acc[1], val_acc)
        final_result_dict = result_dict

        if args.output_json:
            with open(args.output_json, "w") as f:
                json.dump(final_result_dict, f)

        if args.save_checkpoint:
            torch.save(model.state_dict(), args.save_checkpoint)

        print(f"Category: {args.category} | Train Acc: {train_acc:.4f} | Final Acc: {final_acc[0]:.4f} | Final Best-Acc: {final_acc[1]:.4f} | Bi-Acc: {bi_acc:.4f}")

    return final_acc[0], final_acc[1]


def parse_args():
    parser = argparse.ArgumentParser(description="Train/evaluate the SkillSight student with teacher-feature distillation.")
    parser.add_argument("--category", default="RockClimbing", choices=["Soccer", "Basketball", "RockClimbing", "Music", "Dance", "Cooking"])
    parser.add_argument("--teacher-checkpoint", required=True, help="Path to the SkillSight teacher checkpoint (.pyth).")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--val-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--num-retries", type=int, default=30)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--subtask-loss-weight", type=float, default=0.1)
    parser.add_argument("--distill-loss-weight", type=float, default=0.0, help="Original train2.py had the L1 distillation term commented out; set >0 to enable it.")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--save-checkpoint", default="")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
