import torch.nn as nn
import torch.nn.functional as F
import torch

def clip_loss(student_feat, teacher_feat, temperature=0.07):
    """
    student_feat: (B, d)
    teacher_feat: (B, d)
    """
    # normalize to unit length
    student_norm = F.normalize(student_feat, dim=-1)
    teacher_norm = F.normalize(teacher_feat, dim=-1)

    # similarity matrix (B, B)
    logits = student_norm @ teacher_norm.T   # (B, B)
    logits = logits / temperature

    # ground-truth labels: diagonal = correct match
    labels = torch.arange(logits.size(0), device=logits.device)

    # cross-entropy both ways (symmetrized)
    loss_s2t = F.cross_entropy(logits, labels)
    loss_t2s = F.cross_entropy(logits.T, labels)

    return (loss_s2t + loss_t2s) / 2