import os
import torch
import anndata as ad 
from abc import ABC, abstractmethod
from typing import List, Union
import json
import warnings
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ..model import CATLAS

def load_pretrained_model(config):

    model = CATLAS(config)

    param = [{'params':list(model.parameters()), 'lr': 5e-3, 'weight_decay': 1e-10}]

    if config['optim'] == 'AdamW':

        optim = torch.optim.AdamW(param)
                
    if config['scheduler'] == 'plat':
        scheduler = ReduceLROnPlateau(optim, 'min', patience=3, factor=0.95)
    else:
        scheduler = None
    
    #Xavier initialization
    def xavier_init(model):
        if isinstance(model, torch.nn.Linear):
            #xavier_normal_
            torch.nn.init.xavier_uniform_(model.weight)
            torch.nn.init.constant_(model.bias, 0)
        elif isinstance(model, torch.nn.TransformerEncoderLayer):
            torch.nn.init.xavier_uniform_(model.self_attn.in_proj_weight)
            torch.nn.init.xavier_uniform_(model.self_attn.out_proj.weight)
            torch.nn.init.xavier_uniform_(model.linear1.weight)
            torch.nn.init.xavier_uniform_(model.linear2.weight)
            torch.nn.init.constant_(model.self_attn.in_proj_bias, 0)
            torch.nn.init.constant_(model.self_attn.out_proj.bias, 0)
            torch.nn.init.constant_(model.linear1.bias, 0)
            torch.nn.init.constant_(model.linear2.bias, 0)
        elif isinstance(model, torch.nn.TransformerDecoderLayer):
            torch.nn.init.xavier_uniform_(model.self_attn.in_proj_weight)
            torch.nn.init.xavier_uniform_(model.self_attn.out_proj.weight)
            torch.nn.init.xavier_uniform_(model.multihead_attn.in_proj_weight)
            torch.nn.init.xavier_uniform_(model.multihead_attn.out_proj.weight)
            torch.nn.init.xavier_uniform_(model.linear1.weight)
            torch.nn.init.xavier_uniform_(model.linear2.weight)
            torch.nn.init.constant_(model.self_attn.in_proj_bias, 0)
            torch.nn.init.constant_(model.self_attn.out_proj.bias, 0)
            torch.nn.init.constant_(model.multihead_attn.in_proj_bias, 0)
            torch.nn.init.constant_(model.multihead_attn.out_proj.bias, 0)
            torch.nn.init.constant_(model.linear1.bias, 0)
            torch.nn.init.constant_(model.linear2.bias, 0)

        elif isinstance(model, torch.nn.MultiheadAttention):
            torch.nn.init.xavier_uniform_(model.in_proj_weight)
            torch.nn.init.xavier_uniform_(model.out_proj.weight)
            torch.nn.init.constant_(model.in_proj_bias, 0)
            torch.nn.init.constant_(model.out_proj.bias, 0)
    

    def load_ckpt(model, optimizer, scheduler, path):

                ckpt = torch.load(path, map_location=config['device'])
                epoch = ckpt['epoch']
                model.load_state_dict(ckpt['model_state_dict'])
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                scheduler.load_state_dict(ckpt['scheduler_state_dict'])

                print('checkpoint loaded without opt and schedular!')

                return model, optimizer, scheduler, epoch


    if config['continual_training'] == False:
        model.apply(xavier_init)
        epoch = 0
    else:
        model, optim, scheduler, epoch = load_ckpt(model, optim, scheduler, config['ckpt'])

    return model, optim, scheduler, epoch


class Pipeline(ABC):
    def __init__(self,
                 adata:ad.AnnData,
                 neighbor_radius: float,
                 tokenizer_method: str,
                 tokenizer_type: str,
                 gene_names: List[str]
                 ):
        pass


    @abstractmethod
    def fit(self, adata:ad.AnnData,
            train_config: dict = None,
            split_field: str = None,
            train_split: str = None,
            valid_split: str = None,
            label_fields: List[str] = None, # A list in adata.obs that contain cell labels
            ensemble_auto_conversion: bool = False, 
            device: Union[str, torch.device] = 'cpu'):

            #Train the model on an anndata object
            pass
    
    '''

    @abstractmethod
    def predict(self, adata:ad.AnnData,
            inference_config: dict = None,
            ensemble_auto_conversion: bool = False,
            device: Union[str, torch.device] = 'cpu'):

            #Inference on an anndata object
            pass

    @abstractmethod
    def evaluate(seld, adata:ad.AnnData,
            evaluation_config: dict = None,
            split_field: str = None,
            target_split: str = 'test',
            label_fields: List[str] = None,
            ensemble_auto_conversion: bool = False,
            device: Union[str, torch.device] = 'cpu'):

            #Inference and evaluate
            pass

   '''