import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import Trainer, BertPreTrainedModel, BertModel
from transformers.modeling_outputs import SequenceClassifierOutput
from collections import Counter
from typing import List, Optional, Tuple, Union, Dict, Any
from . import GMVAELatentLayer


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1, eps=1e-6):
        super().__init__()
        self.temperature = temperature
        self.eps = eps

    def forward(self, emb: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        emb = F.normalize(emb, dim=1)
        sim_matrix = torch.matmul(emb, emb.T) / self.temperature
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float()
        mask.fill_diagonal_(0)

        log_prob = sim_matrix - torch.logsumexp(sim_matrix, dim=1, keepdim=True)
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + self.eps)
        return -mean_log_prob_pos.mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            if isinstance(self.alpha, dict):
                alpha_tensor = torch.tensor([self.alpha[t.item()] for t in targets], device=inputs.device)
            else:
                alpha_tensor = self.alpha[targets]
            loss *= alpha_tensor

        return loss.mean() if self.reduction == 'mean' else loss.sum() if self.reduction == 'sum' else loss


def compute_alpha_from_dataset(dataset) -> Dict[int, float]:
    labels = [ex["label"] for ex in dataset]
    label_counts = Counter(labels)
    inv_freqs = {cls: 1.0 / cnt for cls, cnt in label_counts.items()}
    total = sum(inv_freqs.values())
    return {cls: w / total for cls, w in inv_freqs.items()}


class MLPEmbedding(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )

    def forward(self, x):
        return self.mlp(x)


class MarkersTrainer(Trainer):
    def __init__(self, *args, alpha_dataset=None, gamma=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.focal_loss = FocalLoss(alpha=compute_alpha_from_dataset(alpha_dataset), gamma=gamma)
        self.contrastive_loss = ContrastiveLoss()

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = self.focal_loss(outputs.logits, labels) + self.contrastive_loss(outputs.hidden_states[0], labels)
        return (loss, outputs) if return_outputs else loss


class CustomTrainer(Trainer):
    def __init__(self, *args, alpha_dataset=None, gamma=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.focal_loss = FocalLoss(alpha=compute_alpha_from_dataset(alpha_dataset), gamma=gamma)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = self.focal_loss(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss


class STPLMForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.neigh_bert = BertModel(config)
        self.dropout = nn.Dropout(config.classifier_dropout or config.hidden_dropout_prob)
        self.gmm_latent = GMVAELatentLayer(config.hidden_size, config.hidden_size, config.num_labels)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.num_labels = config.num_labels
        self.config = config
        self.post_init()

    def forward(self, input_ids, neighborhood, attention_mask=None, **kwargs):
        return_dict = kwargs.get("return_dict", self.config.use_return_dict)
        output = self.bert(input_ids, attention_mask=attention_mask, return_dict=True)
        pooled = self.dropout(output.pooler_output)

        for i in range(len(input_ids)):
            neigh_input = neighborhood[i]
            if neigh_input.dim() == 1:
                neigh_input = neigh_input.unsqueeze(0)
            neigh_mask = (neigh_input != 0).long()

            neigh_out = self.neigh_bert(neigh_input, attention_mask=neigh_mask, return_dict=True)
            neigh_pooled = self.dropout(neigh_out.pooler_output)
            pooled[i] = 0.85 * pooled[i] + 0.15 * neigh_pooled.mean(dim=0)

        latent, gauss_loss = self.gmm_latent(pooled)
        logits = self.classifier(latent)

        return SequenceClassifierOutput(
            loss=gauss_loss,
            logits=logits,
            hidden_states=[latent]
        )


class STPLMMARKERSForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.classifier_dropout or config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.config = config
        self.post_init()

    def forward(self, input_ids, attention_mask=None, labels=None, return_dict=None, **kwargs):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output = self.bert(input_ids, attention_mask=attention_mask, return_dict=True)
        pooled = self.dropout(output.pooler_output)

        logits = self.classifier(pooled)
        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.config.num_labels == 1:
                    self.config.problem_type = "regression"
                elif labels.dtype in (torch.long, torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss = MSELoss()(logits.squeeze(), labels.squeeze())
            elif self.config.problem_type == "single_label_classification":
                loss = CrossEntropyLoss()(logits.view(-1, self.config.num_labels), labels.view(-1))
            else:
                loss = BCEWithLogitsLoss()(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=pooled,
            attentions=output.attentions
        )
