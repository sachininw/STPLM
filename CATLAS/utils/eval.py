import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def sequene_to_sequence_accuracy(predictions, ground_truth):

    predicted_tokens = torch.argmax(predictions, dim=-1)
    correct_predictions = (predicted_tokens == ground_truth).float() 
    sequence_accuracy = correct_predictions.mean(dim=-1)
    overall_accuracy = sequence_accuracy.mean().item()

    return overall_accuracy

class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        confidence = 1.0 - self.smoothing
        smoothing_value = self.smoothing / pred.size(-1)
        one_hot = torch.full_like(pred, smoothing_value)
        one_hot.scatter_(-1, target.unsqueeze(-1), confidence)
        log_prob = torch.nn.functional.log_softmax(pred, dim=-1)
        loss = -torch.mean(torch.sum(one_hot * log_prob, dim=-1))
        return loss


class FocalLoss(nn.Module):
    def __init__(self, criterion, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.criterion = criterion
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        target_probs = probs.gather(1, targets.view(-1, 1)).squeeze(1)

        focal_weight = (1 - target_probs) ** self.gamma
        ce_loss = self.criterion(logits, targets)

        focal_loss = self.alpha * focal_weight * ce_loss

        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss