import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import numpy as np
import anndata as ad
from scipy.sparse import csr_matrix
import json
import pickle

from CATLAS.utils import set_seed
#from CellClique.pipeline.expression_genertor import GeneExpGenerator
from CATLAS.loader.dataset_attributes import fetch_Xenium_adata_annotation
from CATLAS.tokenizer import read_Xenium_features
from CATLAS.tokenizer.cell_exp_tokenizer import get_expressed_genes, tokenize_expressed_genes
from CATLAS.utils.data import remove_non_exp_cells, remove_unidentified_genes
from CATLAS.pipeline.cell_annotation import CellAnnotatorPipeline


platform = 'Xenium'
device = 'cpu'
TOKENIZER = 'gene_name'
    
set_seed(42)

if platform == 'Xenium':
    cell_feature_matrix_file = 'data/Xenium/Adenocarcinoma/BIVT000288/cell_feature_matrix.h5'
    cell_metadata_file = 'data/Xenium/Adenocarcinoma/BIVT000288/cells.parquet'
    cell_boundaries_file = 'data/Xenium/Adenocarcinoma/BIVT000288/cell_boundaries.parquet'
    cell_types_file = 'data/Xenium/Adenocarcinoma/BIVT000288/288_labels.pkl'

    with open(cell_types_file, 'rb') as f:
        cell_types = pickle.load(f)

    #Remove cells with no expressed genes
    data, indices, indptr, gene_names, cell_names = read_Xenium_features('data/Xenium/Adenocarcinoma/BIVT000288/cell_feature_matrix.h5')
    data, indices, indptr, cell_names = remove_non_exp_cells(data, indices, indptr, cell_names)
    
    #Tokenize expressed genes
    filtered_gene_list, filtered_cells = get_expressed_genes(data, indices, indptr, gene_names, cell_names)
    filtered_gene_list, filtered_cells = remove_unidentified_genes(filtered_gene_list, filtered_cells)
    
    gene_expressions_tokenized = tokenize_expressed_genes('gene_expressions.txt', vocab_size=150, tokenizer_type=TOKENIZER, gene_names=gene_names, train_first=False, train_cont=False)


    #Fetch adata object
    adata = fetch_Xenium_adata_annotation(data, 
                               indices, 
                               indptr, 
                               gene_names, 
                               cell_metadata_file, 
                               cell_boundaries_file, 
                               [filtered_cells, gene_expressions_tokenized], 
                               filtered_gene_list,
                               cell_types)
    

    #adata = ad.read('data.h5ad')

    ##### Move from here below into the pipeline section
else:
    adata = None


with open('vocab_gene_name.json', 'r') as file:
    vocab = json.load(file)

config = {
    'pipeline': 'cell_annotation',
    'cell_types': 14,
    'vocab_size': len(vocab)+1,
    'expression_emb_dim': 8,
    'max_exp_len': 200,
    'trimmed_exp_len': 20,
    'tokenizer_type': 'gene_name',
    'cat_exp_pos': False,
    'cell_size': False,
    'cell_shape': False,
    'norm': 'layernorm',
    'activation': 'relu',
    'dropout': 0.1,
    'norm_first': False, 
    'norm': 'layernorm', 
    'neighborhood_radius': 10,
    'hidden_dim_neigh_rep': 32,
    'dim_neigh_rep': 16,
    'model_neigh_rep': 'att',
    'num_layers_neigh_rep': 2,
    'num_heads_neigh_rep': 4,
    'transformer_type_neigh_rep': 'flowformer',
    'gmm_latent_neigh_rep': False,
    'dense_att_neigh_rep': True,
    'skip_dense_neigh_rep': False,
    'cell_size_neigh_rep': False,
    'cell_shape_neigh_rep': False,
    'num_gmm_components_neigh_rep': 14,
    'transformer_type_cell_rep': 'vanilla',
    'num_heads_cell_encoder': 16,
    'num_layers_cell_encoder': 2,
    'ff_hidden_dim_cell_encoder': 128,
    'cell_demo_modular': True,
    'cell_demo_modular_type': None, # hadamard, non_linear_feature_modulation
    'cell_demo_in_dim': 4,
    'cell_demo_hidden_dim': 128,
    'encoder_out_dim': 64, #Encoder output dimension of neigborhood and cell 
    'ff_hid_dim_cross_att': 128,
    'n_heads_cross_att': 16,
    'num_layers_cross_att': 2,
    'neigh_modular_hidden_dim': 64,
    'fc_out_dim': 64,
    'fc_out_pre_gmm_dim': 128,
    'annotation_downstream': 'cross_attn', #one of {transformer_encoder, contextual_reg, cross_attn}
    'epochs': 50,
    'criterion': 'CrossEntropyLoss',
    'optim': 'AdamW',
    'scheduler': 'plat',
    'continual_training': False,
    'ckpt': 'state_dict/lung.pth',
    'sample_selection': [0,20],
    'Experiment':'Test',
    'device': device
}


pipeline = CellAnnotatorPipeline(adata,
                                neighbor_radius = config['neighborhood_radius'],
                                tokenizer_method = 'clique_exp', # clique_exp or cell_exp
                                tokenizer_type = 'gene_name', # gene_name or sentencepiece
                                gene_names = gene_names)

pipeline.fit(adata, 
            train_config = None, 
            split_field = 'split',  
            train_split = 'train', 
            valid_split = 'valid', 
            label_fields = ['celltype'], 
            ensemble_auto_conversion = False, 
            device = device,
            config = config)

'''

pipeline.infer(adata, 
            train_config = None, 
            split_field = 'split', 
            train_split = 'train', 
            valid_split = 'valid', 
            label_fields = ['celltype'], 
            ensemble_auto_conversion = False, 
            device = device,
            config = config)

'''