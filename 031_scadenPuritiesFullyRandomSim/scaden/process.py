"""
scaden Main functionality

Contains code to
- process a training datasets
- train a model
- perform predictions
- if specified, compute bulk purity. 

"""

# Imports
import logging
import tensorflow as tf
from anndata import read_h5ad
from scaden.model.functions import get_signature_genes, preprocess_h5ad_data, custom_preprocess_h5ad_data
logger = logging.getLogger(__name__)

"""
PARAMETERS
"""
# ==========================================#

def processing(data_path, 
               training_data, 
               processed_path, 
               var_cutoff, 
               ignore_genes = None,
               purity_col = None):
    """
    Process a training dataset to contain only the genes also available in the prediction data
    :param data_path: path to prediction data
    :param training_data: path to training data (h5ad file)
    :param processed_path: name of processed file
    :param bulkPurity: if True, compute the bulk purity and save updated bulk file. For now, I have done this in R externally so no need. 
    :return:
    """
    # Get the common genes (signature genes)
    raw_input = read_h5ad(training_data)
    sig_genes_complete = list(raw_input.var_names)
    print(f"Sig_genes_complete: {len(sig_genes_complete)}")
    sig_genes = get_signature_genes(input_path=data_path, 
                                    sig_genes_complete=sig_genes_complete, 
                                    var_cutoff=var_cutoff, 
                                    ignore_genes=ignore_genes)
    print(f"Sig_genes: {len(sig_genes)}")
    
    # print(len(sig_genes))
    
    # Save signature genes to a .txt file by replacing 'Processed.h5ad' with '_sigGenes.txt'
    if "Processed.h5ad" in processed_path:
        sig_genes_path = processed_path.replace("Processed.h5ad", "sigGenes.txt")
    else:
        raise ValueError("Expected 'Processed.h5ad' in processed_path for consistent naming.")

    with open(sig_genes_path, 'w') as f:
        for gene in sig_genes:
            f.write(gene + '\n')

    # ==== Handle purity if available ====
    if "purity" in raw_input.obs.columns:
        print("Process with purity input in X.obs. Not used as a sig gene.")
        # Pre-process data with new signature genes, do not account for purity
        custom_preprocess_h5ad_data(raw_input_path=training_data,
                                    processed_path=processed_path,
                                    sig_genes=sig_genes, 
                                    purity_col = purity_col)
    else:
        print("No purity input.")
        # Pre-process data with new signature genes, do not account for purity
        custom_preprocess_h5ad_data(raw_input_path=training_data,
                                    processed_path=processed_path,
                                    sig_genes=sig_genes,
                                    purity_col = None)
