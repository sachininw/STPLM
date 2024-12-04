import scanpy as sc
import numpy as np
import h5py
import scipy
import pandas as pd
import os
import math
import ast
import anndata as ad
from scipy.sparse import csr_matrix
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
import torch.nn.functional as F
import warnings
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

def pad_2d_list(input_list, max_length=None):
    if max_length is None:
        max_length = max(len(sublist) for sublist in input_list)
    padded_list = []
    padding_value = torch.zeros_like(input_list[0][0])  # Use zeros tensor with the same dtype and device as the first element
    for sublist in input_list:
        padding_needed = max_length - len(sublist)
        padded_sublist = sublist + [padding_value] * padding_needed
        padded_sublist_tensor = torch.stack(padded_sublist)
        padded_list.append(padded_sublist_tensor)
    return padded_list

def pad_lists(input_list, PAD=0, max_len=None, trim_len=None):
    if max_len is None:
        max_len = max(len(sublist) for sublist in input_list)
    padded_list = [sublist + [PAD] * (max_len - len(sublist)) for sublist in input_list]
    padded_list = torch.tensor(padded_list)
    if trim_len is not None:
        padded_list = padded_list[:, :trim_len]
    return padded_list

def pad_tensor_list(input_list, max_length=None):
    if max_length is None:
        max_length = max(len(tensor) for tensor in input_list)
    # Pad each tensor to match the maximum length
    padded_list = [torch.nn.functional.pad(tensor, pad=(0, max_length - len(tensor))) for tensor in input_list]
    # Stack the padded tensors into a single tensor
    padded_tensor = torch.stack(padded_list)
    return padded_tensor

def pad_matrices(matrices, max_shape=None):
    if max_shape is None:
        max_shape = max(matrix.shape for matrix in matrices)
    padded_matrices = [np.pad(matrix, ((0, max_shape[0] - matrix.shape[0]), (0, max_shape[1] - matrix.shape[1])), mode='constant') for matrix in matrices] 
    matrix_padded = [torch.tensor(matrix) for matrix in padded_matrices]

    return matrix_padded

def pad_exp(lst, max_length, padding_value):
    return lst + [padding_value] * (max_length - len(lst))

def pad_2D_sequences(batch, padding_value):

    max_rows = max([seq.shape[0] for seq in batch])
    max_cols = max([seq.shape[1] for seq in batch])

    padded_sequences = []
    for seq in batch:
        padded_seq = np.full((max_rows, max_cols), padding_value, dtype=np.int64)
        rows, cols = seq.shape
        padded_seq[:rows, :cols] = seq.numpy()
        padded_sequences.append(torch.tensor(padded_seq))

    return torch.stack(padded_sequences)


def pad_3d_list(input_list, PAD=0, max_len=200, trim_len=20):
    # Determine the maximum length of the outer list
    max_outer_length = max(len(inner_list) for inner_list in input_list)
    
    # Determine the maximum length of the inner lists

    # Pad each inner list to the maximum length
    padded_list = []
    for inner_list in input_list:
        padded_inner_list = []
        for inner_inner_list in inner_list:
            # Pad inner_inner_list with PAD
            padded_inner_inner_list = inner_inner_list + [PAD] * (max_len - len(inner_inner_list))
            padded_inner_list.append(padded_inner_inner_list)
        # Pad inner_list with empty lists of PAD
        while len(padded_inner_list) < max_outer_length:
            padded_inner_list.append([PAD] * max_len)
        padded_list.append(padded_inner_list)
    padded_list = torch.tensor(padded_list)
    padded_list = padded_list[:, :, :trim_len]
    
    return padded_list
