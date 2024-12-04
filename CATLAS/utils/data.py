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
import pickle


def remove_non_exp_cells(data, indices, indptr, cell_names):
    #Remove cells that do not show any gene expression
    sparse_matrix = csr_matrix((data, indices, indptr))
    dense_matrix = sparse_matrix.toarray()

    non_zero_rows = ~np.all(dense_matrix == 0, axis=1)
    filtered_dense_matrix = dense_matrix[non_zero_rows]

    filtered_cell_names = [name for i , name in enumerate(cell_names) if non_zero_rows[i]]


    #Reset
    new_sparse_matrix = csr_matrix(filtered_dense_matrix)
    new_data = new_sparse_matrix.data
    new_indices = new_sparse_matrix.indices
    new_indptr = new_sparse_matrix.indptr

    return new_data, new_indices, new_indptr, filtered_cell_names

def remove_unidentified_genes(input_list, cell_names):
    filtered_list = []
    filtered_cell_names = []

    for sublist, cell_name in zip(input_list, cell_names):
        filtered_sublist = [elem for elem in sublist if not (elem.startswith('NegControlProbe') or elem.startswith('UnassignedCodeword') or elem.startswith('NegControlCodeword'))]
        if filtered_sublist:
            filtered_list.append(filtered_sublist)
            filtered_cell_names.append(cell_name)
    
    #Save gene expressions as a text file
    with open('gene_expressions.txt', 'w') as file:
        for gene_list in filtered_list:
            
            seq = '--'.join(gene_list)
            file.write(f'{seq}\n')

    
    with open('filtered_cells', 'wb') as file:
        pickle.dump(filtered_cell_names, file)

    return filtered_list, filtered_cell_names


def VerticesToShape(vertices):
    # Takes a list of vertices and returns the shape of the cell
    # Returns a vector of the form [area, perimeter, circularity]
    shapes = []

    for vertex in vertices:
        # Convert string to list if necessary
        vertex_list = ast.literal_eval(vertex) if isinstance(vertex, str) else vertex

        # Initialize variables
        area = 0
        perimeter = 0
        n = len(vertex_list)  # Number of vertices

        # Iterate over each vertex
        for i in range(n):
            x1, y1 = vertex_list[i]
            x2, y2 = vertex_list[(i + 1) % n]

            # Calculate area using the shoelace formula
            area += (x1 * y2 - x2 * y1)

            # Calculate perimeter
            perimeter += math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        # Finalize area calculation
        area = round(abs(area) / 2, 2)
        # Round perimeter to 2 decimal places
        perimeter = round(perimeter, 2)
        # Calculate circularity
        circularity = round(4 * math.pi * area / perimeter ** 2, 2)

        # Append the results
        shapes.append([area, perimeter, circularity])

    return shapes

def VerticesToShape_Optimized(vertices):

    shapes = []

    for vertex in vertices:
        # Convert string to list if necessary
        vertex_list = ast.literal_eval(vertex) if isinstance(vertex, str) else vertex

        #Convert list of tuples to a numpy array
        vertex_array = np.array(vertex_list)

        #Calculate area using the shoelace formula
        x = vertex_array[:,0]
        y = vertex_array[:,1]

        area = 0.5 * np.abs(np.dot(x,np.roll(y,1)) - np.dot(y,np.roll(x,1)))

        #Calculate perimeter
        dx = np.diff(np.append(x, x[0]))
        dy = np.diff(np.append(y, y[0]))
        perimeter = np.sum(np.sqrt(dx**2 + dy**2))

        #Round area and perimeter to 2 decimal places
        area = round(area, 2)
        perimeter = round(perimeter, 2)

        #Calculate circularity
        circularity = round(4 * math.pi * area / perimeter ** 2, 2)
        
        shapes.append([area, perimeter, circularity])
    shapes = torch.tensor(shapes)

    return shapes


def FetchVertexList(data):
    
    data = data.applymap(lambda x:round(x,2) if isinstance(x, float) else x)
    
    grouped = data.groupby('cell_id').apply(lambda x: list(zip(x['vertex_x'], x['vertex_y']))).reset_index()
    grouped.columns = ['cell_id', 'vertices']

    return grouped


