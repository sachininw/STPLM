import torch
import torch.nn as nn
import numpy as np
import scanpy as sc
import anndata as ad
from typing import List, Union
from torch.utils.data import DataLoader
import pickle
import wandb
import torchmetrics.functional as F
import torchmetrics
from torchmetrics.functional.classification import multiclass_f1_score, multiclass_accuracy, multiclass_precision, multiclass_recall 


from ..loader import CATLASDataset, CATLASDatasetSplit, CustomDataset
from . import Pipeline, load_pretrained_model
from ..utils.eval import sequene_to_sequence_accuracy, LabelSmoothingLoss, FocalLoss

wandb.login()
wandb.require("core")
'''

This pipeline trains the model to predict the gene expression profile of a cell considering its neighborhood.
Neighborhood radius is user defined.
Expressions will be generated in descending order of magnitude of expression. In case of ties, alphabetical order will be used.

'''

def validation(model, dataloader, cell_types, criterion, focal_loss, device='cpu'):

    with torch.no_grad():
        model.eval()
        epoch_loss = []
        epoch_accuracy = []
        epoch_precision = []
        epoch_recall = []
        epoch_f1_score = []
        
        for i, data_dict in enumerate(dataloader):
            
            neighbor_exp = data_dict['neighborhood_exp_seq']
            dist = data_dict['dist']
            size = data_dict['size']
            shape = data_dict['shape']
            exp_cc = data_dict['exp_seq_cc']
            size_cc = data_dict['size_cc']
            shape_cc = data_dict['shape_cc']
            label_cc = data_dict['label_cc']

            predicted_labels, recon_loss, recon_loss_2, ls_reg = model(neighbor_exp, dist, size, shape, exp_cc, size_cc, shape_cc)

            labels = label_cc
            predicted_labels = predicted_labels

            test_labels = torch.argmax(predicted_labels, dim=-1).unsqueeze(1)
            print('Test:', test_labels)
            print('True:', labels)

                
            l = focal_loss(predicted_labels, labels)

            loss = l + recon_loss + recon_loss_2 + ls_reg
            loss = loss.item()

            acc = multiclass_accuracy(predicted_labels, labels, cell_types).item()
            acc = round(acc, 3)

            prec = multiclass_precision(predicted_labels, labels, cell_types).item()
            prec = round(prec, 3)

            rec = multiclass_recall(predicted_labels, labels, cell_types).item()
            rec = round(rec, 3)

            f1 = multiclass_f1_score(predicted_labels, labels, cell_types).item()
            f1 = round(f1, 3)

            epoch_loss.append(loss)
            epoch_accuracy.append(acc)
            epoch_precision.append(prec)
            epoch_recall.append(rec)
            epoch_f1_score.append(f1)


            #print('Batch:', i, 'loss:', epoch_loss, 'accuracy:', acc, 'corr:', sum(correlations) / len(correlations))

        loss = round(sum(epoch_loss) / len(epoch_loss), 3)
        acc = round(sum(epoch_accuracy) / len(epoch_accuracy), 3)
        prec = round(sum(epoch_precision) / len(epoch_precision), 3)
        rec = round(sum(epoch_recall) / len(epoch_recall), 3)
        f1 = round(sum(epoch_f1_score) / len(epoch_f1_score), 3)
    
    return predicted_labels, labels, loss, acc, prec, rec, f1

class CellAnnotatorPipeline(Pipeline):
    
    def __init__(self,
                 adata:ad.AnnData,
                 neighbor_radius: float,
                 tokenizer_method: str,
                 tokenizer_type: 'gene_name',
                 gene_names: List[str]
                 ):
        self.adata = adata
        self.neighbor_radius = neighbor_radius
        self.tokenizer_method = tokenizer_method
        self.tokenizer_type = tokenizer_type
        self.gene_names = gene_names
        
    def fit(self, adata: ad.AnnData,
            train_config: dict = None,
            split_field: str = None,
            train_split: str = None,
            valid_split: str = None,
            label_fields: List[str] = None, # A list in adata.obs that contain cell labels
            ensemble_auto_conversion: bool = False,
            device: Union[str, torch.device] = 'cpu',
            config: dict = None
            ):
        

        with wandb.init(project=config['Experiment'], config=config):

            config = wandb.config

            dataset = CATLASDataset(adata, self.neighbor_radius, self.tokenizer_method, self.tokenizer_type, config['sample_selection'], config['max_exp_len'], config['trimmed_exp_len'], self.gene_names, device)

            torch.cuda.empty_cache()
            dataloader = DataLoader(dataset, batch_size=10000000, shuffle=True)

            for i, data_dict in enumerate(dataloader):
                dataDict = data_dict
                break
            dataset = CustomDataset(dataDict)
            dataset_train = CATLASDatasetSplit(dataset, split_value='train')  
            dataset_valid = CATLASDatasetSplit(dataset, split_value='val')
            dataset_test = CATLASDatasetSplit(dataset, split_value='val')  

            dataloader_train = DataLoader(dataset_train, batch_size=64, shuffle=True) #, num_workers=2, pin_memory=True
            dataloader_valid = DataLoader(dataset_valid, batch_size=64, shuffle=True)
            dataloader_test = DataLoader(dataset_test, batch_size=64, shuffle=True)

            self.model, optim, scheduler, EPOCH = load_pretrained_model(config)
            print('Start at epoch', EPOCH)

            if config['criterion'] == 'CrossEntropyLoss':
                criterion = nn.CrossEntropyLoss()
            elif config['criterion'] == 'LabelSmoothingLoss':
                criterion = LabelSmoothingLoss(smoothing=0.1)
                
            focal_loss =FocalLoss(criterion, alpha=1.0, gamma=2.0, reduction='mean')

            cos = torch.nn.CosineSimilarity(dim=-1)
            
            print(self.model)
            
            sample_ct = 0 #No. of samples seen

            print('Experiment:', config['Experiment'])
            print('Batch:', config['sample_selection'])
            
            for epoch in range(config['epochs']):
                torch.cuda.empty_cache()
                wandb.watch(self.model, criterion, log='all', log_freq=1)

                self.model.train()
                epoch_loss = []
                epoch_accuracy = []
                epoch_precision = []
                epoch_recall = []
                epoch_f1_score = []

                epoch_L = []
                epoch_ReL = []
                
                for i, data_dict in enumerate(dataloader_train):

                    neighbor_exp = data_dict['neighborhood_exp_seq']
                    dist = data_dict['dist']
                    size = data_dict['size']
                    shape = data_dict['shape']
                    exp_cc = data_dict['exp_seq_cc']
                    size_cc = data_dict['size_cc']
                    shape_cc = data_dict['shape_cc']
                    label_cc = data_dict['label_cc']

                    predicted_labels, recon_loss, recon_loss_2, ls_reg = self.model(neighbor_exp, dist, size, shape, exp_cc, size_cc, shape_cc)

                    labels = label_cc
                    predicted_labels = predicted_labels

                    #predicted_label = torch.argmax(out, dim=-1).unsqueeze(1)
                    l = focal_loss(predicted_labels, labels)
                    

                    loss = l +  recon_loss + recon_loss_2 + ls_reg
                    
                    acc = multiclass_accuracy(predicted_labels, labels, config['cell_types']).item()
                    acc = round(acc, 3)

                    prec = multiclass_precision(predicted_labels, labels, config['cell_types']).item()
                    prec = round(prec, 3)

                    rec = multiclass_recall(predicted_labels, labels, config['cell_types']).item()
                    rec = round(rec, 3)

                    f1 = multiclass_f1_score(predicted_labels, labels, config['cell_types']).item()
                    f1 = round(f1, 3)

                    sample_ct += data_dict['size_cc'].size(0)
                    optim.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 2)
                    optim.step()
                    epoch_loss.append(loss.item())
                    epoch_accuracy.append(acc)
                    epoch_precision.append(prec)
                    epoch_recall.append(rec)
                    epoch_f1_score.append(f1)

                    epoch_L.append(l.item())

                    scheduler.step(loss)
                
                L = round(sum(epoch_L) / len(epoch_L), 3)
                loss = round(sum(epoch_loss) / len(epoch_loss), 3)
                acc = round(sum(epoch_accuracy) / len(epoch_accuracy), 3)
                prec = round(sum(epoch_precision) / len(epoch_precision), 3)
                rec = round(sum(epoch_recall) / len(epoch_recall), 3)
                f1 = round(sum(epoch_f1_score) / len(epoch_f1_score), 3)


                valid_out, val_target, val_loss, val_acc, val_prec, val_rec, val_f1 = validation(self.model, dataloader_valid, config['cell_types'], criterion, focal_loss, device)
                
                Train_Loss = float(loss)
                Val_Loss = float(val_loss)
                Val_Acc = float(val_acc)

                wandb.log({'epoch': epoch, 'Train_Loss': Train_Loss, 'Val_Loss': Val_Loss, 'Val_Acc': Val_Acc}, step = sample_ct)

                
                print('epoch:', epoch+EPOCH, 'train_loss:', loss, 'train_acc:', acc, 'train_prec:', prec, 'train_recall:', rec, 'train_f1_score:',  f1, 
                      'valid_loss:', val_loss, 'valid_acc:', val_acc, 'val_prec:', val_prec, 'val_rec:', val_rec, 'val_f1_score:', val_f1, 'L:',L)

            def save_checkpoint(model, optimizer, scheduler, epoch, path):
                state = {
                    'epoch': epoch+EPOCH,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                }

                torch.save(state, path)

                return 'Model saved!'

            print(save_checkpoint(self.model, optim, scheduler, epoch, config['ckpt']))

 
            dummy_input = (neighbor_exp, dist, size, shape, exp_cc, size_cc, shape_cc)
            torch.onnx.export(self.model, dummy_input, 'model.onnx')
            wandb.save('model.onnx')

        return self

