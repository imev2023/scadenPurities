"""
Functions used for the scaden model
"""
import logging
import collections
from anndata import read_h5ad, AnnData
import numpy as np
import tensorflow as tf
from sklearn import preprocessing as pp
import pandas as pd
from scipy.sparse import issparse
import gc

logger = logging.getLogger(__name__)


def dummy_labels(m, labels):
    """
    Create dummy labels needed for building the graph correctly
    :param m:
    :param labels:
    :return:
    """
    n_l = len(labels)
    return np.zeros((m, n_l), dtype="float32")


def sample_scaling(x, scaling_option):
    """
    Apply scaling of data
    :param x:
    :param scaling_option:
    :return:
    """

    if scaling_option == "log_min_max":
        # Bring in log space
        x = np.log2(x + 1)

        # Normalize data
        mms = pp.MinMaxScaler(feature_range=(0, 1), copy=True)

        # it scales features so transpose is needed
        x = mms.fit_transform(x.T).T

    return x


def preprocess_h5ad_data(
    raw_input_path, processed_path, scaling_option="log_min_max", sig_genes=None, purity_col = None
):
    """
    Preprocess raw input data for the model
    :param raw_input_path:
    :param scaling_option:
    :param group_small:
    :param signature_genes:
    :return:
    """
    logger.info("Pre-processing raw data ...")
    raw_input = read_h5ad(raw_input_path)

    logger.info("Subsetting genes ...")
    # Select features go use
    raw_input = raw_input[:, sig_genes]

    logger.info("Scaling using " + str(scaling_option))
    # Scaling
    raw_input.X = sample_scaling(raw_input.X, scaling_option)

    logger.info("Writing to disk ...")
    raw_input.write(processed_path)
    logger.info("Data pre-processing done.")
    logger.info(f"Created processed file: [cyan]{processed_path}[/]")


def get_signature_genes(input_path, sig_genes_complete, var_cutoff=0.1, ignore_genes = None):
    """
    Get overlap between signature genes and available genes
    :param input_path:
    :param sig_genes_complete:
    :return: new sig_genes
    """
    data = pd.read_table(input_path, index_col=0)

    # Remove ignore_genes from index if provided --> if data already binned in R. 
    if ignore_genes:
        ignore_set = set(ignore_genes)
        data = data.loc[~data.index.isin(ignore_set)]


    keep = data.var(axis=1) > var_cutoff
    data = data.loc[keep]
    available_genes = list(data.index)
    new_sig_genes = list(set(available_genes).intersection(sig_genes_complete))
    n_sig_genes = len(new_sig_genes)
    logger.info(f"Found [cyan]{n_sig_genes}[/cyan] common genes.")
    return new_sig_genes

def custom_preprocess_h5ad_data(
    raw_input_path, 
    processed_path, 
    scaling_option="log_min_max", 
    sig_genes=None, 
    purity_col = None

):
    """
    Preprocess raw input data for the model
    :param raw_input_path:
    :param processed_path:
    :param scaling_option:
    :param sig_genes: List of signature genes to subset the data to
    """
    logger.info("Pre-processing raw data ...")
    raw_input = read_h5ad(raw_input_path)

    logger.info("Subsetting genes ...")

    # Filter out sig_genes that aren't in var_names
    sig_genes = [g for g in sig_genes if g in raw_input.var_names]
    raw_input = raw_input[:, sig_genes].copy()  # Make a copy to avoid view-related errors
    # print("Raw_input: ")
    # print(raw_input)
    logger.info("Scaling using " + str(scaling_option))
    raw_input.X = sample_scaling(raw_input.X, scaling_option)
    # print(f"raw_input.X shape: {raw_input.X.shape}") # raw_input.X shape: (7, 15625)
    # print(raw_input)

    # -- Purity logic - If we need to add a purity column, we construct a new adata object. --
    if purity_col is not None:
        logger.info("Handling purity as a pseudo-gene ...")

        # Get purity values from obs and normalize to same dtype as X. Divide by 100 to get % purity. 
        purity_values = raw_input.obs[purity_col].values.astype(raw_input.X.dtype)/100
    
        purity_values = purity_values.reshape(-1,1)  # shape: (n_samples, 1)
        # print(purity_values)
        # print(purity_values.dtype)
        # 2. Append new gene expression to X
        if issparse(raw_input.X):
            new_X = hstack([raw_input.X, purity_values]).tocsr()
        else:
            new_X = np.hstack([raw_input.X, purity_values])

        # 3. Update var DataFrame by adding the new gene name as a new row
        new_var = raw_input.var.copy()
        new_var = pd.concat([new_var, pd.DataFrame(index=[purity_col])])

        # 4. Construct new AnnData object with updated X and var, copying obs and uns
        new_adata = AnnData(
            X=new_X,
            obs=raw_input.obs.copy(),
            var=new_var,
            uns=raw_input.uns.copy()
        )

        # print(new_adata.shape)  # should be (7, 15626)
        # print(new_adata.var.tail())  # last few genes including purity values. 
        

         # Drop purity from obs
        new_adata.obs.drop(columns=[purity_col], inplace=True)
        logger.info("New adata constructed ")
        del raw_input
        # print(new_adata)
        gc.collect()
    else: # If no purity, rename scaled data. 
        new_adata = raw_input
        del raw_input
        gc.collect()

    logger.info("Writing to disk ...")
    new_adata.write(processed_path)
    logger.info("Data pre-processing done.")
    logger.info(f"Created processed file: [cyan]{processed_path}[/]")
    