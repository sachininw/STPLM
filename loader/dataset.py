import os
import time
import pickle
import h5py
import numpy as np
import pandas as pd
import anndata as ad
from scipy.sparse import csr_matrix
from typing import Tuple, List, Dict, Any
from collections import defaultdict
from itertools import chain


def timing(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"{func.__name__} executed in {time.time() - start:.2f}s")
        return result
    return wrapper


def save_pickle(obj: Any, file_path: str) -> None:
    with open(file_path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(file_path: str) -> Any:
    with open(file_path, "rb") as f:
        return pickle.load(f)


@timing
def read_xenium_features(file_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
    with h5py.File(file_path, "r") as file:
        data = file["matrix/data"][()]
        indices = file["matrix/indices"][()]
        indptr = file["matrix/indptr"][()]
        gene_ids = [s.decode("utf-8") for s in file["matrix/features/id"]]
        gene_names = [s.decode("utf-8") for s in file["matrix/features/name"]]
        barcodes = [s.decode("utf-8") for s in file["matrix/barcodes"]]

    save_pickle([gene_names, gene_ids], "loader/gene_name_ensemblid.pkl")
    return data, indices, indptr, gene_ids, barcodes


def column_median_ignore_zeros(matrix: np.ndarray) -> List[float]:
    non_zero_mask = matrix != 0
    medians = np.where(
        non_zero_mask.any(axis=0),
        np.ma.masked_equal(matrix, 0).median(axis=0).filled(0),
        0,
    )
    return medians.tolist()


def rank_genes(gene_vector: np.ndarray, gene_tokens: np.ndarray) -> np.ndarray:
    return gene_tokens[np.argsort(gene_vector)[::-1]]


@timing
def tokenize_neighborhood(
    adata_path: str,
    gene_median_file: str,
    token_dict_file: str,
    output_file: str,
    target_sum: int = 480,
    chunk_size: int = 512,
) -> List[List[int]]:
    adata = ad.read(adata_path, backed="r")
    gene_median = load_pickle(gene_median_file)
    token_dict = load_pickle(token_dict_file)
    cls_token_id = len(token_dict)

    # Restrict to coding/miRNA genes only
    valid_genes = np.array([gene_median.get(eid, 0) > 0 for eid in adata.var["ensembl_id"]])
    coding_idxs = np.where(valid_genes)[0]
    coding_ids = adata.var["ensembl_id"][coding_idxs]
    norm_factors = np.array([gene_median[eid] for eid in coding_ids])
    gene_tokens = np.array([token_dict[eid] for eid in coding_ids])

    filter_pass = adata.obs.get("filter_pass", pd.Series([1] * adata.shape[0])).values.astype(bool)
    selected = np.where(filter_pass)[0]

    result = []
    for i in range(0, len(selected), chunk_size):
        batch_idx = selected[i:i + chunk_size]
        counts = adata[batch_idx].obs['n_counts'].values[:, None]
        X = adata[batch_idx, coding_idxs].X.copy()
        X = csr_matrix(X.multiply(target_sum / counts / norm_factors))

        result.extend(
            [rank_genes(X[i].data, gene_tokens[X[i].indices]) for i in range(X.shape[0])]
        )

    out = [[cls_token_id] + list(ranked) for ranked in result]
    save_pickle(out, output_file)
    return out


@timing
def create_adata_from_matrix(
    data: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    gene_names: List[str],
    cell_ids: List[str],
    organ: str,
    save_path: str,
    gene_median_save_path: str = None
) -> None:
    sparse_matrix = csr_matrix((data, indices, indptr))
    invalid = ("NegControlProbe", "UnassignedCodeword", "NegControlCodeword")
    
    keep_idx = [i for i, g in enumerate(gene_names) if not g.startswith(invalid)]
    gene_names = [gene_names[i] for i in keep_idx]
    sparse_matrix = sparse_matrix[:, keep_idx]

    dense = sparse_matrix.toarray()

    if gene_median_save_path:
        col_medians = column_median_ignore_zeros(dense)
        save_pickle([gene_names, col_medians], gene_median_save_path)

    n_counts = np.diff(sparse_matrix.indptr)
    adata = ad.AnnData(X=sparse_matrix)
    adata.var["ensembl_id"] = gene_names
    adata.obs["cell_id"] = cell_ids
    adata.obs["n_counts"] = n_counts
    adata.obs["organ_major"] = organ
    adata.write(save_path)


@timing
def fetch_neighbors(metadata_file: str, radius: float, save_path: str, labels_file: str, target_label: str = "Endothelial") -> Tuple[List[str], List[List[str]]]:
    label_dict = dict(zip(*load_pickle(labels_file)))

    metadata = pd.read_parquet(metadata_file, engine="pyarrow")
    ids = metadata["cell_id"].tolist()
    coords = metadata[["x_centroid", "y_centroid"]].values

    filtered_ids = []
    neighbors = []

    for i, (cell_id, coord) in enumerate(zip(ids, coords)):
        if label_dict.get(cell_id) == target_label:
            dists = np.linalg.norm(coords - coord, axis=1)
            neighbor_ids = [ids[j] for j in np.where((dists <= radius) & (dists > 0))[0]]
            filtered_ids.append(cell_id)
            neighbors.append(neighbor_ids)

    save_pickle([filtered_ids, neighbors], save_path)
    return filtered_ids, neighbors


def gene_non_zero_median(gene_lists: List[Tuple[List[str], List[float]]], organ: str) -> None:
    counts = defaultdict(list)
    for genes, values in gene_lists:
        for g, v in zip(genes, values):
            counts[g].append(v)

    median_dict = {g: sum(v) / len(v) for g, v in counts.items()}
    save_pickle(median_dict, f"geneformer/gene_median_dict_{organ}.pkl")


def update_token_dict(token_dict: Dict[str, int], new_genes: List[str]) -> None:
    next_token = max(token_dict.values(), default=-1) + 1
    for gene in new_genes:
        token_dict.setdefault(gene, next_token)
        if token_dict[gene] == next_token:
            next_token += 1
    save_pickle(token_dict, "geneformer/token_dictionary.pkl")


@timing
def fetch_marker_adata(df: pd.DataFrame, save_path: str, organ: str) -> None:
    cell_types = df.iloc[:, 0].values
    expr = df.iloc[:, 1:].values
    ensembl_ids = df.columns[1:]

    adata = ad.AnnData(X=expr)
    adata.var["ensembl_id"] = ensembl_ids
    adata.obs["cell_id"] = df.index.astype(str)
    adata.obs["n_counts"] = np.count_nonzero(expr, axis=1)
    adata.obs["cell_type"] = cell_types
    adata.obs["organ_major"] = organ
    adata.write(save_path)
