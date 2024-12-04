import scanpy as sc
import numpy as np
import h5py
import scipy
import pandas as pd
import os
import math
import ast
import anndata as ad
import pickle
from scipy.sparse import csr_matrix
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
import torch.nn.functional as F
import warnings
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from ..utils.data import FetchVertexList, VerticesToShape, VerticesToShape_Optimized
from ..utils.pad import pad_tensor_list, pad_lists, pad_matrices, pad_exp, pad_2D_sequences, pad_3d_list
from ..tokenizer.cell_exp_tokenizer import tokenize_expressed_genes

def sparse_to_tensor(x):

    return torch.sparse_csr_tensor(x.indptr, x.indices, x.data, (x.shape[0], x.shape[1])).to_sparse().float().coalesce()

def fetch_Xenium_adata_annotation(data, indices, indptr, var, cell_metadata_file, cell_boundaries_file, filtered_cells_expr, expressed_genes, cell_types):

    '''
    cell_feature_matrix_file: hdf5 file containing data, indices, indptr, and features
    cell_metadata_file: parquet file containing shape, size, coordinates and other metadata
    cell_boundaries_file: parquet file containing vertices
    expressed_genes_tokenized: list of expressed genes for each cell [cell_id, expressed_genes_tokenized]
    '''

    ##Takes .h5 file of cell feature matrix and .parquet cell metadata file and returns anndata object
    '''
    with h5py.File(cell_feature_matrix_file,'r') as file:

        data = file['matrix/data'][:].tolist()
        indices = file['matrix/indices'][:].tolist()
        indptr = file['matrix/indptr'][:].tolist()
        var = file['matrix/features/name'][:].tolist()
    '''

    cell_metadata = pd.read_parquet(cell_metadata_file, engine='pyarrow')
    cell_bounds = pd.read_parquet(cell_boundaries_file, engine='pyarrow')
    cell_vertexlists = FetchVertexList(cell_bounds)

    #Label encoding
    annotations = cell_types[1]
    #flat_list = [item for sublist in annotations for item in sublist]
    unique_classes = list(set(annotations))
    print(unique_classes)

    encoder = LabelEncoder()
    encoder.fit(unique_classes)

    # Get the list of category names and their assigned encodings
    categories = encoder.classes_
    encodings = encoder.transform(categories)

    for category, encoding in zip(categories, encodings):
        print(f"Category: {category}, Label: {encoding}")

    encoded = []
    #for sublist in annotations:
    encoded.append(encoder.transform(annotations))

    #decoded = []
    #for sublist in encoded:
    #    decoded.append(encoder.inverse_transform(sublist))
    #print(decoded[:5])


    # Display the category names and encodings
    #print("Category names and their encodings:")
    #for category, encoding in zip(categories, encodings):
    #    print(f"{category}: {encoding}")

    filtered_cells_df = pd.DataFrame(filtered_cells_expr).transpose().set_axis(['cell_id','expressed_gene_tokens'], axis=1)
    expressed_genes_df = pd.DataFrame([filtered_cells_expr[0],expressed_genes]).transpose().set_axis(['cell_id','expressed_genes'], axis=1)
    print(len(encoded[0]))
    print(len(cell_types[0]))
    cell_types_df = pd.DataFrame({'cell_type': encoded[0], 'cell_id': cell_types[0]})
    print(cell_types_df.head())
    filtered_cell_metadata = cell_metadata.merge(filtered_cells_df, on='cell_id', how='inner')
    
    cell_metadata_final = filtered_cell_metadata.merge(cell_vertexlists, on='cell_id', how='inner')

    cell_metadata_final = cell_metadata_final.applymap(lambda x:round(x,2) if isinstance(x, float) else x)
    cell_metadata_final = cell_metadata_final.merge(cell_types_df, on='cell_id', how='inner')
    cell_metadata_final = cell_metadata_final.merge(expressed_genes_df, on='cell_id', how='inner')

    matrix = csr_matrix((data,indices,indptr))
    matrix = matrix[:len(cell_metadata_final['cell_id']), :]
    adata = ad.AnnData(X=matrix)
    adata.var['gene_name'] = var
    adata.obs['cell_id'] = cell_metadata_final['cell_id'].apply(str).values
    adata.obs['expressed_genes'] = cell_metadata_final['expressed_genes'].tolist()
    adata.obs['expressed_gene_tokens'] = cell_metadata_final['expressed_gene_tokens'].values
    adata.obs['x_coord'] = cell_metadata_final['x_centroid'].values
    adata.obs['y_coord'] = cell_metadata_final['y_centroid'].values
    adata.obs['vertices'] = cell_metadata_final['vertices'].apply(str).values
    adata.obs['cell_size'] = cell_metadata_final['cell_area'].values
    adata.obs['nucleus_size'] = cell_metadata_final['nucleus_area'].values
    adata.obs['platform'] = 0
    adata.obs['celltype'] = cell_metadata_final['cell_type'].values
    adata.obs['split'] = 'test'

    train_num  = int(0.8*len(cell_metadata_final['cell_id']))
    tr = np.random.permutation(train_num)
    adata.obs['split'][tr[:int(train_num*0.9)]] = 'train'
    adata.obs['split'][tr[int(train_num*0.9):train_num]] = 'val'

    #adata.write('data.h5ad')

    return adata


def fetch_Xenium_adata(data, indices, indptr, var, cell_metadata_file, cell_boundaries_file, filtered_cells_expr, expressed_genes):

    '''
    cell_feature_matrix_file: hdf5 file containing data, indices, indptr, and features
    cell_metadata_file: parquet file containing shape, size, coordinates and other metadata
    cell_boundaries_file: parquet file containing vertices
    expressed_genes_tokenized: list of expressed genes for each cell [cell_id, expressed_genes_tokenized]
    '''

    ##Takes .h5 file of cell feature matrix and .parquet cell metadata file and returns anndata object
    '''
    with h5py.File(cell_feature_matrix_file,'r') as file:

        data = file['matrix/data'][:].tolist()
        indices = file['matrix/indices'][:].tolist()
        indptr = file['matrix/indptr'][:].tolist()
        var = file['matrix/features/name'][:].tolist()
    '''

    cell_metadata = pd.read_parquet(cell_metadata_file, engine='pyarrow')
    cell_bounds = pd.read_parquet(cell_boundaries_file, engine='pyarrow')
    cell_vertexlists = FetchVertexList(cell_bounds)

    filtered_cells_df = pd.DataFrame(filtered_cells_expr).transpose().set_axis(['cell_id','expressed_gene_tokens'], axis=1)
    

    filtered_cell_metadata = cell_metadata.merge(filtered_cells_df, on='cell_id', how='inner')
    cell_metadata_final = filtered_cell_metadata.merge(cell_vertexlists, on='cell_id', how='inner')


    #Check if cell_ids in gene expression and metadata files are in the same order

    #columns_match = cell_metadata_final['cell_id'].equals(string_list_df['cell_id']) 

    #if not columns_match:
    #raise NotImplementedError('Cell IDs in gene expression and metadata files do not match')

    cell_metadata_final = cell_metadata_final.applymap(lambda x:round(x,2) if isinstance(x, float) else x)

    matrix = csr_matrix((data,indices,indptr))
    adata = ad.AnnData(X=matrix)
    adata.var['gene_name'] = var
    adata.obs['cell_id'] = cell_metadata_final['cell_id'].apply(str).values
    adata.obs['expressed_genes'] = expressed_genes
    adata.obs['expressed_gene_tokens'] = cell_metadata_final['expressed_gene_tokens'].values
    adata.obs['x_coord'] = cell_metadata_final['x_centroid'].values
    adata.obs['y_coord'] = cell_metadata_final['y_centroid'].values
    adata.obs['vertices'] = cell_metadata_final['vertices'].apply(str).values
    adata.obs['cell_size'] = cell_metadata_final['cell_area'].values
    adata.obs['nucleus_size'] = cell_metadata_final['nucleus_area'].values
    adata.obs['platform'] = 0

    random_numbers = np.random.uniform(low=0, high=10, size=len(adata.obs['y_coord']))
    df = pd.DataFrame(random_numbers, columns=['proxy_celltype'])
    adata.obs['celltype'] = df['proxy_celltype'].values

    adata.obs['split'] = 'test'
    train_num  = int(0.8*len(cell_metadata_final['cell_id']))
    tr = np.random.permutation(train_num)
    adata.obs['split'][tr[:int(train_num*0.9)]] = 'train'
    adata.obs['split'][tr[int(train_num*0.9):train_num]] = 'val'

    #adata.write('data.h5ad')

    return adata


#CellBatchDataset should output a dictionary with gene expression, distance, size, shape, and labels, split
class CATLASDataset(Dataset):

    def __init__(self, adata: ad.AnnData,
                 neighbor_radius: float,
                 tokenizer_method: str = 'clique_exp',
                 tokenizer_type:str = 'gene_name',
                 sample_selection: list = None,
                 max_exp_len: int = 200,
                 trimmed_exp_len: int = 20,
                 gene_names: list = None,
                 device: str = 'cpu'):

        #tokenizer_type: 'clique_exp' or 'cell_exp'
        
        self.neighborhood_exp_seq = []
        #self.gene_exp_list = []
        self.expressed_genes = []
        self.dist_list = []
        self.size_list = []
        self.shape_list = []
        self.label_list = []
        self.split_list = []
        self.platform_list = []


        self.genes = adata.var.index.tolist()
        label_fields = adata.obs['celltype'].tolist()

        adata.obs['batch_label'] = -1
        adata.obs['dist'] = 0
        A=0

        #x = csr_matrix(adata.X)
        #dense_tensor = x.todense()

        #for batch in range(len(adata.obs['cell_id'])):
        print('num cells:', len(adata.obs['cell_id']))
        
        start, end = sample_selection[0], sample_selection[1]
        #for batch in range(len(adata.obs['cell_id'])):
        for batch in range(start, end):

            adata.obs['batch_label'][batch] = batch
            split = adata.obs['split'][batch]

            x_diff = adata.obs['x_coord'] - adata.obs['x_coord'][batch]
            y_diff = adata.obs['y_coord'] - adata.obs['y_coord'][batch]
            adata.obs['dist'] = np.sqrt(x_diff**2 + y_diff**2)

            adata.obs.loc[adata.obs['dist'] <= neighbor_radius, 'batch_label'] = batch
            
            filtered_indices = adata.obs['batch_label'] == batch

            #filtered_data = adata.X[filtered_indices]
            #adata.obs['expressed_gene_tokens'] = [ast.literal_eval(lst) for lst in adata.obs['expressed_gene_tokens']]
            max_length = adata.obs['expressed_genes'].apply(len).max()
            #adata.obs['expressed_gene_tokens_padded'] = adata.obs['expressed_gene_tokens'].apply(pad_exp, args=(max_length,-1))
            
            #padded = pad_tensor_list(torch.tensor(adata.obs['expressed_gene_tokens_padded'][filtered_indices]))
            dist = adata.obs['dist'][filtered_indices].values
            sorted_indices = np.argsort(-dist).tolist()

            
            #expressed_gene_tokens_padded = adata.obs['expressed_gene_tokens_padded'][filtered_indices].iloc[sorted_indices].tolist()
            #X = dense_tensor[filtered_indices][sorted_indices]
            expressed_genes = adata.obs['expressed_genes'][filtered_indices].iloc[sorted_indices].tolist()
            dist = adata.obs['dist'][filtered_indices].iloc[sorted_indices].tolist()
            nucleus_size = adata.obs['nucleus_size'][filtered_indices].iloc[sorted_indices].tolist()
            vertices = adata.obs['vertices'][filtered_indices].iloc[sorted_indices].tolist()
            celltype = adata.obs['celltype'][filtered_indices].iloc[sorted_indices].tolist()
            platform = adata.obs['platform'][filtered_indices].iloc[sorted_indices].tolist()



            #self.neighborhood_exp_seq.append(torch.tensor(expressed_gene_tokens_padded))
            #self.gene_exp_list.append(X)
            self.expressed_genes.append(expressed_genes)
            self.dist_list.append(torch.tensor(dist))
            self.size_list.append(torch.tensor(nucleus_size))
            self.shape_list.append(VerticesToShape_Optimized(vertices))
            self.label_list.append(torch.tensor(celltype))
            self.split_list.append(split)
            self.platform_list.append(torch.tensor(platform))
        

        #if tokenizer_method == 'cell_exp':

            #self.neighborhood_exp_seq_padded = pad_2D_sequences(self.neighborhood_exp_seq, -1)

        

        if tokenizer_method == 'clique_exp':

            with open('gene_expressions.txt', 'w') as file:
                for clique_gene_list in self.expressed_genes:
                    for cell_gene_list in clique_gene_list:
                        cell_seq = '--'.join(cell_gene_list)
                        file.write(f'{cell_seq}' + '  ')
                    
                    file.write('\n')

            
            expressed_genes_tokenized = tokenize_expressed_genes('gene_expressions.txt', vocab_size=150, tokenizer_type=tokenizer_type, gene_names=gene_names, train_first=False, train_cont=False)
            
            
        else:
            raise NotImplementedError('Unknown tokenizer type')

        # Get the features of the center cell
        expressed_genes_tokenized_center = [sublist[-1] for sublist in expressed_genes_tokenized]
        #self.gene_exp_list_center = [sublist[-1] for sublist in self.gene_exp_list]
        self.size_list_center = [sublist[-1] for sublist in self.size_list]
        self.shape_list_center = [sublist[-1] for sublist in self.shape_list]
        self.label_list_center = [sublist[-1] for sublist in self.label_list]
        self.split_list_center = [sublist[-1] for sublist in self.split_list]
        self.platform_list_center = [sublist[-1] for sublist in self.platform_list]

        
        # Remove the data about the center cell from the input and add it to the output

        expressed_genes_tokenized = [sublist[:-1] for sublist in expressed_genes_tokenized]
        #self.gene_exp_list = [sublist[:-1] for sublist in self.gene_exp_list]
        self.dist_list = [sublist[:-1] for sublist in self.dist_list]
        self.size_list = [sublist[:-1] for sublist in self.size_list]
        self.shape_list = [sublist[:-1] for sublist in self.shape_list]
        self.label_list = [sublist[:-1] for sublist in self.label_list]
        self.split_list = [sublist for sublist in self.split_list]
        self.platform_list = [sublist[:-1] for sublist in self.platform_list]

        
        # Get the indices of cells with no neighbors
        no_neighbors = [i for i, sublist in enumerate(expressed_genes_tokenized) if len(sublist) == 0]

        # Remove elements in no_neighbors from the dataset
        self.expressed_genes_tokenized_center = [sublist for i, sublist in enumerate(expressed_genes_tokenized_center) if i not in no_neighbors]
        self.expressed_genes_tokenized_center = torch.tensor(pad_lists(self.expressed_genes_tokenized_center, 0, max_exp_len, trimmed_exp_len))
        #self.gene_exp_list_center = [sublist for i, sublist in enumerate(self.gene_exp_list_center) if i not in no_neighbors]
        self.size_list_center = [sublist for i, sublist in enumerate(self.size_list_center) if i not in no_neighbors]
        self.shape_list_center = [sublist for i, sublist in enumerate(self.shape_list_center) if i not in no_neighbors]
        self.label_list_center = [sublist for i, sublist in enumerate(self.label_list_center) if i not in no_neighbors]
        self.split_list_center = [sublist for i, sublist in enumerate(self.split_list_center) if i not in no_neighbors]
        self.platform_list_center = [sublist for i, sublist in enumerate(self.platform_list_center) if i not in no_neighbors]

        # Add <sos> and <eos> tokens to the center cell gene expression
        '''
        sos_token = torch.ones(self.expressed_genes_tokenized_center.size(0), dtype = self.expressed_genes_tokenized_center.dtype)
        sos_token = sos_token.view(-1, 1)

        eos_token = torch.full((self.expressed_genes_tokenized_center.size(0),1), 2, dtype = self.expressed_genes_tokenized_center.dtype)

        self.expressed_genes_tokenized_center = torch.cat((sos_token, self.expressed_genes_tokenized_center), dim=1)
        self.expressed_genes_tokenized_center = torch.cat((self.expressed_genes_tokenized_center, eos_token), dim=1)

        '''

        # Add <sos> and <eos> to encoder as well?


        # Next remove cells without any neighbors from the dataset
        expressed_genes_tokenized  = [lst for lst in expressed_genes_tokenized if len(lst) !=0]
        #self.gene_exp_list = [lst for lst in self.gene_exp_list if len(lst) !=0]
        self.dist_list = [lst for lst in self.dist_list if len(lst) !=0]
        self.size_list = [lst for lst in self.size_list if len(lst) !=0]
        self.shape_list = [lst for lst in self.shape_list if len(lst) !=0]
        self.label_list = [lst for lst in self.label_list if len(lst) !=0]
        self.split_list = [lst for lst in self.split_list if len(lst) !=0]
        self.platform_list = [lst for lst in self.platform_list if len(lst) !=0]
        
        # Pad sequences
        #self.gene_exp_padded = pad_matrices(self.gene_exp_list)

        if tokenizer_type == 'gene_name':
            self.neighborhood_exp_seq_padded = torch.tensor(pad_3d_list(expressed_genes_tokenized, 0, max_exp_len, trimmed_exp_len))
        elif tokenizer_type == 'sentencepiece':
            self.neighborhood_exp_seq_padded = torch.tensor(pad_lists(expressed_genes_tokenized))
        
        self.dist_list_padded = pad_tensor_list(self.dist_list).long()
        
        #Convert distance to long tensor
        #self.dist_list_padded = self.dist_list_padded.long()
        self.size_list_padded = pad_tensor_list(self.size_list)

        
        #self.label_list_padded = pad_tensor_list(self.label_list)
        self.platform_list_padded = pad_tensor_list(self.platform_list)

        tensor_list = self.shape_list
        # Find the maximum shape along each dimension
        max_shape = max(tensor.shape for tensor in tensor_list)

        # Pad each tensor in the list to have the same shape as max_shape
        padded_tensors = [torch.nn.functional.pad(tensor, (0, max_shape[1] - tensor.shape[1], 0, max_shape[0] - tensor.shape[0])) for tensor in tensor_list]

        # Stack the padded tensors into a single tensor
        self.shape_list_padded = torch.stack(padded_tensors)

        #Remove nan
        # Check for rows containing NaNs in the neighborhood
        mask = ~torch.isnan(self.size_list_padded).any(dim=1)
        indices_to_keep = torch.nonzero(mask).squeeze().tolist()

        self.neighborhood_exp_seq_padded = [self.neighborhood_exp_seq_padded[i] for i in indices_to_keep]
        #self.gene_exp_padded = [self.gene_exp_padded[i] for i in indices_to_keep]
        self.dist_list_padded = [self.dist_list_padded[i] for i in indices_to_keep]
        self.size_list_padded = [self.size_list_padded[i] for i in indices_to_keep]
        self.shape_list_padded = [self.shape_list_padded[i] for i in indices_to_keep]
        #self.label_list_padded = [self.label_list_padded[i] for i in indices_to_keep]
        self.split_list = [self.split_list[i] for i in indices_to_keep]
        self.platform_list_padded = [self.platform_list_padded[i] for i in indices_to_keep]

        self.expressed_genes_tokenized_center = [self.expressed_genes_tokenized_center[i] for i in indices_to_keep]
        #self.gene_exp_list_center = [self.gene_exp_list_center[i] for i in indices_to_keep]
        self.size_list_center = [self.size_list_center[i] for i in indices_to_keep]
        self.shape_list_center = [self.shape_list_center[i] for i in indices_to_keep]
        self.label_list_center = [self.label_list_center[i] for i in indices_to_keep]

        #Check for rows containing NaNs in the center cell
        mask = ~torch.isnan(torch.tensor(self.size_list_center))
        indices_to_keep = torch.nonzero(mask).squeeze().tolist()

        self.neighborhood_exp_seq_padded = [self.neighborhood_exp_seq_padded[i] for i in indices_to_keep]
        #self.gene_exp_padded = [self.gene_exp_padded[i] for i in indices_to_keep]
        self.dist_list_padded = [self.dist_list_padded[i] for i in indices_to_keep]
        self.size_list_padded = [self.size_list_padded[i] for i in indices_to_keep]
        self.shape_list_padded = [self.shape_list_padded[i] for i in indices_to_keep]
        #self.label_list_padded = [self.label_list_padded[i] for i in indices_to_keep]
        self.split_list = [self.split_list[i] for i in indices_to_keep]
        self.platform_list_padded = [self.platform_list_padded[i] for i in indices_to_keep]

        self.expressed_genes_tokenized_center = [self.expressed_genes_tokenized_center[i] for i in indices_to_keep]
        #self.gene_exp_list_center = [self.gene_exp_list_center[i] for i in indices_to_keep]
        self.size_list_center = [self.size_list_center[i] for i in indices_to_keep]
        self.shape_list_center = [self.shape_list_center[i] for i in indices_to_keep]
        self.label_list_center = [self.label_list_center[i] for i in indices_to_keep]

    def __len__(self):

        return len(self.neighborhood_exp_seq_padded)

    def __getitem__(self, idx):

        self.data_dict = {'neighborhood_exp_seq': self.neighborhood_exp_seq_padded[idx],
                    #'gene_exp': self.gene_exp_padded[idx],
                    'dist': self.dist_list_padded[idx],
                    'size': self.size_list_padded[idx],
                    'shape': self.shape_list_padded[idx],
                    #'label': self.label_list_padded[idx],
                    'split': self.split_list[idx],
                    'platform': self.platform_list_padded[idx],
                    'exp_seq_cc': self.expressed_genes_tokenized_center[idx],
                    #'gene_exp_cc': self.gene_exp_list_center[idx],
                    'size_cc': self.size_list_center[idx],
                    'shape_cc': self.shape_list_center[idx],
                    'label_cc': self.label_list_center[idx]
                    }

        return self.data_dict

class CustomDataset(Dataset):
    def __init__(self, data_dict):
        self.data_dict = data_dict

    def __len__(self):
        return len(self.data_dict['split'])

    def __getitem__(self, idx):
        return {key: self.data_dict[key][idx] for key in self.data_dict}


class CATLASDatasetSplit(Dataset):

    def __init__(self, dataset, split_value):
        self.dataset = dataset
        self.indices = [i for i, split in enumerate(self.dataset.data_dict['split']) if split == split_value]

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]


