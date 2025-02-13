import torch
import torch.nn.functional as F
from transformers import Trainer
from collections import Counter
from typing import List, Optional, Tuple, Union, Dict, Any
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import numpy as np

from transformers import BertPreTrainedModel, BertModel
from transformers.models.bert.modeling_bert import BertSelfAttention
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.utils import ModelOutput, add_start_docstrings, add_start_docstrings_to_model_forward, add_code_sample_docstrings, logging
from gmmv_latent import GMVAELatentLayer


print(torch.__version__)

logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "google-bert/bert-base-uncased"
_CONFIG_FOR_DOC = "BertConfig"

# TokenClassification docstring
_CHECKPOINT_FOR_TOKEN_CLASSIFICATION = "dbmdz/bert-large-cased-finetuned-conll03-english"
_TOKEN_CLASS_EXPECTED_OUTPUT = (
    "['O', 'I-ORG', 'I-ORG', 'I-ORG', 'O', 'O', 'O', 'O', 'O', 'I-LOC', 'O', 'I-LOC', 'I-LOC'] "
)
_TOKEN_CLASS_EXPECTED_LOSS = 0.01

# SequenceClassification docstring
_CHECKPOINT_FOR_SEQUENCE_CLASSIFICATION = "textattack/bert-base-uncased-yelp-polarity"
_SEQ_CLASS_EXPECTED_OUTPUT = "'LABEL_1'"
_SEQ_CLASS_EXPECTED_LOSS = 0.01

def create_bert_self_attention_mask(batch_size, seq_length):
    # Start with all zeros
    attention_mask = torch.zeros(batch_size, seq_length, seq_length)
    
    # Allow the first token to attend to all tokens
    attention_mask[:, 0, :] = 1  # First token attends to all
    
    # Allow other tokens to only attend to themselves
    for i in range(1, seq_length):
        attention_mask[:, i, i] = 1  # Other tokens attend only to themselves
    
    return attention_mask

import torch
import torch.nn as nn

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1, eps=1e-6):
        super().__init__()
        self.temperature = temperature
        self.eps = eps
    
    def forward(self, emb, labels):
        # Validate inputs
        assert not torch.isnan(emb).any(), "NaN in embeddings"
        assert not torch.isinf(emb).any(), "Inf in embeddings"

        # Normalize embeddings
        emb = nn.functional.normalize(emb, dim=1)

        # Compute similarity matrix
        sim_matrix = torch.matmul(emb, emb.T) / self.temperature
        
        # Labels and mask
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float()
        mask.fill_diagonal_(0)

        # Numerical stability with logsumexp
        log_prob = sim_matrix - torch.logsumexp(sim_matrix, dim=1, keepdim=True)
        
        # Validate logsumexp output
        assert not torch.isnan(log_prob).any(), "NaN in log_prob"
        assert not torch.isinf(log_prob).any(), "Inf in log_prob"

        # Handle edge case with no positive pairs
        #if mask.sum() == 0:
        #    return 0 
        
        # Compute mean log probability for positive samples
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + self.eps)

        # Compute loss
        loss = -mean_log_prob_pos.mean()

        return loss



class FocalLoss(torch.nn.Module):

    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        """
        Focal Loss with dynamically computed alpha.
        Args:
            alpha (dict or tensor): Class weights (e.g., inversely proportional to class frequency).
            gamma (float): Focusing parameter.
            reduction (str): Reduction method ('mean', 'sum', or 'none').
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Logits (not probabilities) of shape (batch_size, num_classes).
            targets: Ground-truth labels of shape (batch_size).
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # Probabilities of the correct class
        focal_loss = (1 - pt)**self.gamma * ce_loss

        # Apply alpha weights if provided
        if self.alpha is not None:
            if isinstance(self.alpha, dict):  # Convert dict to tensor
                alpha_tensor = torch.tensor([self.alpha[t.item()] for t in targets], device=inputs.device)
            else:  # Assume alpha is already a tensor
                alpha_tensor = self.alpha[targets]
            focal_loss *= alpha_tensor

        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def compute_alpha_from_dataset(dataset):
    """
    Compute alpha weights dynamically from the dataset.
    Args:
        dataset: Dataset provided to Transformers Trainer (expects 'labels' as a key).
    Returns:
        alpha (dict): Class weights normalized to sum to 1.
    """
    

    # Extract all labels from the dataset
    labels = []
    for example in dataset:
        labels.append(example["label"])

    # Count occurrences of each class
    label_counts = Counter(labels)

    # Compute inverse class weights
    alpha = {class_id: 1.0 / count for class_id, count in label_counts.items()}

    # Normalize weights
    alpha = {class_id: weight / sum(alpha.values()) for class_id, weight in alpha.items()}
    return alpha

class MLPEmbedding(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim=128):
        super(MLPEmbedding, self).__init__()
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

        alpha = compute_alpha_from_dataset(alpha_dataset)
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)
        self.contrastive_loss = ContrastiveLoss()

    def compute_loss(self, model, inputs, num_items_in_batch=False, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        latent_rep = outputs.hidden_states

        ce_loss = self.focal_loss(logits, labels)
        contr_loss = self.contrastive_loss(latent_rep, labels)

        loss = ce_loss + contr_loss

        return (loss, outputs) if return_outputs else loss


class CustomTrainer(Trainer):
    def __init__(self, *args, alpha_dataset=None, gamma=2, **kwargs):
        super().__init__(*args, **kwargs)

        # Dynamically compute alpha from the dataset
        alpha = compute_alpha_from_dataset(alpha_dataset)

        # Initialize focal loss with dynamic alpha
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)
        self.contrastive_loss = ContrastiveLoss()
        self.marker_cell_matching = nn.BCEWithLogitsLoss()
        

    def compute_loss(self, model, inputs, num_items_in_batch=False, return_outputs=False):
        """
        Custom loss computation with focal loss.
        """
        labels = inputs.pop("labels")
        #neighborhood = inputs.get("neighborhood")

        outputs = model(**inputs)
        
        logits = outputs.logits
        emb= outputs.loss

        ce_loss = self.focal_loss(logits, labels)
        #contr_loss = self.contrastive_loss(emb, labels)

        '''
        contr_markers_cells_embed = torch.cat((emb, markers_embed), dim=0)  
        contr_markers_cells_labels = torch.cat((labels, marker_labels), dim=0)  
        contr_loss_cell_marker = self.contrastive_loss(contr_markers_cells_embed, contr_markers_cells_labels)
        contr_loss_marker_marker = self.contrastive_loss(markers_embed,marker_labels)
        '''
    
        #loss = 0.2*ce_loss + 0.2*contr_loss + 0.2* match_loss + 0.2* contr_loss_cell_marker + 0.2*contr_loss_marker_marker
        loss = ce_loss 
        #loss = ce_loss

        '''
        similarity_scores = np.array(similarity_scores)
        comparison_matrix = np.array(comparison_matrix)
        similarity_scores = ((similarity_scores >= 0.5).astype(int))

        print(np.mean(similarity_scores==comparison_matrix))

        similarity_scores_flat = similarity_scores.flatten()
        comparison_matrix_flat = comparison_matrix.flatten()

        print(f1_score(similarity_scores_flat, comparison_matrix_flat, average='macro'))
        '''
    
        return (loss, outputs) if return_outputs else loss

    '''
    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], num_items_in_batch=None
    ) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """


        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        inputs = self._prepare_inputs(inputs)

        labels = inputs.get("labels")
        label_set = set(labels.tolist())

        if not len(label_set) < len(labels):
            return None

        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)

        del inputs
        if (
            self.args.torch_empty_cache_steps is not None
            and self.state.global_step % self.args.torch_empty_cache_steps == 0
        ):
            if is_torch_xpu_available():
                torch.xpu.empty_cache()
            elif is_torch_mlu_available():
                torch.mlu.empty_cache()
            elif is_torch_musa_available():
                torch.musa.empty_cache()
            elif is_torch_npu_available():
                torch.npu.empty_cache()
            elif is_torch_mps_available(min_version="2.0"):
                torch.mps.empty_cache()
            else:
                torch.cuda.empty_cache()

        kwargs = {}

        # For LOMO optimizers you need to explicitly use the learnign rate
        if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            kwargs["learning_rate"] = self._get_learning_rate()

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            # Finally we need to normalize the loss for reporting
            if num_items_in_batch is None:
                loss = loss / self.args.gradient_accumulation_steps

            self.accelerator.backward(loss, **kwargs)

            return loss.detach()
    
    '''
    

'''

class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.focal_loss = FocalLoss(alpha=1, gamma=2)

    def compute_loss(self, model, inputs, num_items_in_batch=False, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = self.focal_loss(logits, labels)
        return (loss, outputs) if return_outputs else loss
'''

BERT_START_DOCSTRING = r"""

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`BertConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

BERT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `({0})`or `(batch_size, sequence_length, target_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
            1]`:

            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.

            [What are token type IDs?](../glossary#token-type-ids)
        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

@add_start_docstrings(
    """
    Bert Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled
    output) e.g. for GLUE tasks.
    """,
    BERT_START_DOCSTRING,
)


class CATLASForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        self.bert_attn = BertModel(config)
        #self.bert_self_attn = BertSelfAttention(config)
        
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.gmm_latent = GMVAELatentLayer(config.hidden_size, config.hidden_size, config.num_labels)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_SEQUENCE_CLASSIFICATION,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_SEQ_CLASS_EXPECTED_OUTPUT,
        expected_loss=_SEQ_CLASS_EXPECTED_LOSS,
    )

   
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        neighborhood: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        
        
        input_id_with_neigh =[]
        for i in range(len(input_ids)):

            attention_mask_neigh = (neighborhood[i] != 0).long()

            ##For 1 neighbor
            if neighborhood[i].dim() == 1:
                 # Check if there is only 1 neighbor, unsqueeze the first dim
                neighborhood[i] = neighborhood[i].unsqueeze(0)
                attention_mask_neigh = attention_mask_neigh.unsqueeze(0)

                

            out_neigh = self.bert(
            neighborhood[i],
            attention_mask=attention_mask_neigh,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            )

            pooled_out_neigh = out_neigh[1]

            pooled_out_neigh = self.dropout(pooled_out_neigh)
                         
            out = torch.cat((pooled_output[i].unsqueeze(0), pooled_out_neigh), dim=0)
            out = out.unsqueeze(0) 

            if i==0:
                input_id_with_neigh = out
            else:
                input_id_with_neigh = torch.cat((input_id_with_neigh, out), dim=0)

        attention_mask_attn = create_bert_self_attention_mask(input_id_with_neigh.size(0), input_id_with_neigh.size(1))
        #attention_mask_attn = attention_mask_attn.unsqueeze(1)


        attention_mask_attn = attention_mask_attn.to('cuda')
        input_id_with_neigh = input_id_with_neigh.to('cuda')

        
        #out_self_attn = self.bert_self_attn(input_id_with_neigh)
        #pooled_output_atnn = out_self_attn[0][:, 0, :] 
                
        out_self_attn = self.bert_attn(
            inputs_embeds = input_id_with_neigh,
            attention_mask = attention_mask_attn)
        
        pooled_output_atnn = out_self_attn[1]
        
        pooled_output_atnn = self.dropout(pooled_output_atnn)
        
        
        out, recon_loss = self.gmm_latent(pooled_output)

        emb = out

        assert not torch.isnan(emb).any(), "NaN in embeddings"
        logits = self.classifier(emb)

        loss = None

        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=emb,
            logits=logits,
            hidden_states=[outputs,emb],
            attentions=outputs.attentions
        )

class CATLASMARKERSForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)

        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_SEQUENCE_CLASSIFICATION,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_SEQ_CLASS_EXPECTED_OUTPUT,
        expected_loss=_SEQ_CLASS_EXPECTED_LOSS,
    )

   
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)

        assert not torch.isnan(pooled_output).any(), "NaN in embeddings"
        logits = self.classifier(pooled_output)

        loss = None

        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=pooled_output,
            attentions=outputs.attentions
        )