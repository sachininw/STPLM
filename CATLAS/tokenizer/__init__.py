import h5py
import numpy as np
import pandas as pd
import anndata as ad
from scipy.sparse import csr_matrix


def read_Xenium_features(file_path):

    '''
    This function supports Xenium data

    cell_feature_matrix['matrix'] keys: ['barcodes', 'data', 'features', 'indices', 'indptr', 'shape']
    - 'barcodes': 1D array of strings, each string is a cell barcode
    - 'data': 1D array of floats, the values of the matrix
    - 'features': folder with keys ['_all_tag_keys', 'feature_type', 'genome', 'id', 'name']
    '''

    with h5py.File(file_path, 'r') as file:

        data = file['matrix/data'][:]
        indices = file['matrix/indices'][:]
        indptr = file['matrix/indptr'][:]
        gene_names = [name.decode('utf-8') for name in file['matrix/features/name'][:]]
        cell_names = [name.decode('utf-8') for name in file['matrix/barcodes'][:]]

 
    return data, indices, indptr, gene_names, cell_names