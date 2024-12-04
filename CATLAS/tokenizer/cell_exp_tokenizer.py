import h5py
import numpy as np
import pandas as pd
import anndata as ad
from scipy.sparse import csr_matrix
import pickle
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from scipy.signal import find_peaks

from .tokenizer import TranscriptTokenizer


def get_expressed_genes(data, indices, indptr, gene_names, cell_names):

    expressed_genes_per_cell = []
    expressions_for_clustering = []
    test=0

    for i in range(len(indptr) - 1):

        start = indptr[i]

        end = indptr[i + 1] 

        cell_indices = indices[start:end]
        cell_data = data[start:end]
        
        #Create a list of (gene_name, expression_value) tuples
        expressed_genes = [(gene_names[idx], expr) for idx, expr in zip(cell_indices, cell_data) if expr >=0]

        if test ==0:
                print(expressed_genes)

        #Sort by expression value (descending), then alphabetically
        #expressed_gene_sorted = sorted(expressed_genes, key=lambda x: (-x[1], x[0]))

        # Function to select top 80% features based on their values for a single sublist

        #Apply the selection process to each sublist
        cum_var = False
        kernel_density_binning_gaussian_2_bins = False
        kernel_density_binning_gaussian_3_bins = False
        kernel_density_binning_gaussian_automatic_bins = True

        if cum_var == True:
            features_sorted = sorted(expressed_genes, key=lambda x: x[1], reverse=True)
    
        # Calculate the total sum of the values
            values = [value for feature, value in features_sorted]
        
            cumulative_variance = np.cumsum(values) / np.sum(values)
            num_features_to_keep = np.argmax(cumulative_variance >= 0.9) + 1

            top_features = features_sorted[:num_features_to_keep]


            expressed_gene_sorted = sorted(top_features, key=lambda x: x[0])
        
        if kernel_density_binning_gaussian_2_bins == True:
            # Extract values from the tuples
            values = np.array([val for _, val in expressed_genes]).reshape(-1, 1)

            # Fit KDE model
            kde = KernelDensity(kernel='gaussian', bandwidth=1.0).fit(values)

            # Score samples (estimate density)
            log_density = kde.score_samples(values)
            density = np.exp(log_density)

            # Determine the threshold to split into two bins
            threshold = np.median(density)

            # Bin the data
            bin1 = [expressed_genes[i] for i in range(len(expressed_genes)) if density[i] >= threshold]
            bin2 = [expressed_genes[i] for i in range(len(expressed_genes)) if density[i] < threshold]

            if test ==0:
                print(bin1)
                print(bin2)

            # Sort each bin by the first element in each tuple (alphabetically)
            bin1_sorted = sorted(bin1, key=lambda x: x[0])
            bin2_sorted = sorted(bin2, key=lambda x: x[0])

        
            # Combine the sorted bins into a single list
            expressed_gene_sorted = bin2_sorted + bin1_sorted

            if test ==0:
                print(expressed_gene_sorted)
                test = 1
        
        if kernel_density_binning_gaussian_3_bins:

            # Extract values from the tuples
            values = np.array([val for _, val in expressed_genes]).reshape(-1, 1)

            # Fit KDE model
            kde = KernelDensity(kernel='gaussian', bandwidth=1.0).fit(values)

            # Score samples (estimate density)
            log_density = kde.score_samples(values)
            density = np.exp(log_density)

            # Find the cumulative density function (CDF) of the estimated density
            cdf = np.cumsum(density)
            cdf /= cdf[-1]  # Normalize to make the last value equal to 1

            # Find bin edges by splitting the CDF into 3 equal parts
            bin_edges = np.interp([1/3, 2/3], cdf, np.sort(values.flatten()))

            # Assign each original data point to a bin based on these bin edges
            bins = [[] for _ in range(4)]  # 3 bins = 4 edges

            for i, value in enumerate(values.flatten()):
                # Find the appropriate bin for this value
                bin_index = np.digitize(value, bin_edges)
                bins[bin_index].append(expressed_genes[i])
            
            # Sort each bin by the first element in each tuple (alphabetically)
            sorted_bins = [sorted(b, key=lambda x: x[0]) for b in bins]

            if test ==0:
                print('BINS')
                for bin in sorted_bins:
                    print(bin)

            expressed_gene_sorted = [item for sublist in reversed(sorted_bins) for item in sublist]

            if test ==0:
                print(expressed_gene_sorted)
                test = 1
        
        if kernel_density_binning_gaussian_automatic_bins:

            values = np.array([val for _, val in expressed_genes]).reshape(-1, 1)

            # Fit KDE model
            kde = KernelDensity(kernel='gaussian', bandwidth=1.0).fit(values)

            # Create a range of values for density estimation
            x_d = np.linspace(min(values)-1, max(values)+1, 1000).reshape(-1, 1)

            # Score samples (estimate density)
            log_density = kde.score_samples(x_d)
            density = np.exp(log_density)

            # Find local minima (these will be our bin edges)
            minima_indices, _ = find_peaks(-density)
            bin_edges = x_d[minima_indices].flatten()

            # Assign each original data point to a bin based on these bin edges
            bins = [[] for _ in range(len(bin_edges) + 1)]

            for i, value in enumerate(values.flatten()):
                # Find the appropriate bin for this value
                bin_index = np.digitize(value, bin_edges)
                bins[bin_index].append(expressed_genes[i])

            # Sort each bin by the first element in each tuple (alphabetically)
            sorted_bins = [sorted(b, key=lambda x: x[0]) for b in bins]

            if test ==0:
                print('BINS')
                print(sorted_bins)

            # Combine all sorted bins into one list
            expressed_gene_sorted = [item for sublist in reversed(sorted_bins) for item in sublist]
            #expressed_gene_sorted = [item for item in sorted_bins[-1]]
            
            if test ==0:

                #print(expressed_gene_sorted)
                test = 1
        
        #Extract the sorted gene names
        expressed_genes_sorted_names = [gene for gene, expr in expressed_gene_sorted]

        #Add to the list for this cell
        expressed_genes_per_cell.append(expressed_genes_sorted_names)
        expressions_for_clustering.append(sorted_bins)

    '''
    #####save expressions_for_clustering to a pickle file - neighborhood analysis
    clans_viz = [cell_names,expressions_for_clustering]

    with open('clans_viz.pkl', 'wb') as f:
        pickle.dump(clans_viz, f)
    '''
    

    return expressed_genes_per_cell, cell_names

def tokenize_expressed_genes(gene_expressions, vocab_size, tokenizer_type, gene_names, train_first, train_cont):
    #tokenizer_type = ['sentencepiece', 'gene_name']
    #gene_expressions should be a text file with gene expressions of each cell

    tokenizer = TranscriptTokenizer(gene_expressions)

    if tokenizer_type == 'sentencepiece':
        #Train the tokenizer
        sp, sp_model = tokenizer.train_sp_tokenizer(vocab_size)
        #Encode the expressed genes
        expressed_genes_tokenized = tokenizer.encode_sp(sp, sp_model, gene_expressions)
    elif tokenizer_type == 'gene_name':
        if train_first==True:
            vocab = tokenizer.train_gene_name_tokenizer(gene_names)
        else:
            vocab = tokenizer.tune_gene_name_tokenizer(gene_names, 'vocab_gene_name.json')
            expressed_genes_tokenized = tokenizer.encode_gene_name(gene_expressions, vocab, 'vocab_gene_name.json')

    else:
        raise NotImplementedError('Unidentified tokenizer. Please use either "sentencepiece" or "gene_name"')

    return expressed_genes_tokenized
    
