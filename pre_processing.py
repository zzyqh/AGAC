# !/usr/bin/env python
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.preprocessing import scale, minmax_scale
from sklearn import preprocessing 


def read_csv1(filename1, filename2, take_log):
    dataset = {}
    data = pd.read_csv(filename1, index_col=0, sep='\t')
    print(data.shape)
    print('Data loaded')
    print('Before filtering...')
    print(' Number of genes is {}'.format(len(data.index.values)))
    print(' Number of cells is {}'.format(len(data.columns.values)))

    cluster_labels = pd.read_csv(filename2, sep=',').values
    # data = Selecting_highly_variable_genes(data, 2000)
    data = pd.DataFrame(data)
    dataset['cell_labels'] = data.columns.values
    dataset['cluster_labels'] = cluster_labels[:, -1]
    gene_sym = data.index.values
    gene_exp = data.values

    if take_log:
        gene_exp = np.log2(gene_exp + 1)

    dataset['gene_exp'] = gene_exp
    dataset['gene_sym'] = gene_sym

    return dataset


def read_txt(filename, take_log):
    dataset = {}
    df = pd.read_table(filename, header=None)
    dat = df[df.columns[1:]].values
    dataset['cell_labels'] = dat[8, 1:]
    gene_sym = df[df.columns[0]].tolist()[11:]
    gene_exp = dat[11:, 1:].astype(np.float32)
    if take_log:
        gene_exp = np.log2(gene_exp + 1)
    dataset['gene_exp'] = gene_exp
    dataset['gene_sym'] = gene_sym
    dataset['cell_labels'] = convert_strclass_to_numclass(dataset['cell_labels'])

    save_csv(gene_exp, gene_sym,  dataset['cell_labels'])

    return dataset


def pre_processing_single1(filename1, filename2, pre_process_paras, type='csv'):
    """ pre-processing of multiple datasets
    Args:
        dataset_file_list: list of filenames of datasets
        pre_process_paras: dict, parameters for pre-processing
    Returns:
        dataset_list: list of datasets
    """
    # parameters
    take_log = pre_process_paras['take_log']
    scaling = pre_process_paras['scaling']
    dataset_list = []
    data_file1 = filename1
    data_file2 = filename2
    if type == 'csv':
        dataset = read_csv1(data_file1, data_file2, take_log)
    elif type == 'txt':
        dataset = read_txt(data_file1, take_log)
    dataset['gene_exp'] = dataset['gene_exp'].astype(np.float)

    if scaling:  # scale to [0,1]
        minmax_scale(dataset['gene_exp'], feature_range=(0, 1), axis=1, copy=False)

    dataset_list.append(dataset)
    return dataset_list


def Selecting_highly_variable_genes(X, highly_genes):
    adata = sc.AnnData(X)
    adata.var_names_make_unique()
    # sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e4)
    sc.pp.log1p(adata)
    adata.raw = adata
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes=highly_genes)
    adata = adata[:, adata.var['highly_variable']].copy()
    # sc.pp.scale(adata, max_value=3)
    data = adata.X

    return data
