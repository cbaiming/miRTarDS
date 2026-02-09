import pandas as pd
import time 
import json
import numpy as np
from tqdm import tqdm
# Assuming 'data' and 'MISIM' are local packages/modules
import data.load_disease as load_disease
import MISIM

def load_config(config_path='config.json'):
    """
    Loads configuration parameters from a JSON file.
    
    Args:
        config_path (str): Path to the configuration file.
        
    Returns:
        dict: The loaded configuration dictionary.
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_and_filter_dictionaries(config, target_miRNAs, target_genes, mesh_diseases):
    """
    Loads disease association data and filters it based on existing interaction lists
    and valid MeSH disease names.

    Args:
        config (dict): Configuration dictionary containing file paths.
        target_miRNAs (list): List of miRNAs present in the MTI dataset.
        target_genes (list): List of genes present in the MTI dataset.
        mesh_diseases (list): List of valid disease names from MeSH Tree.

    Returns:
        tuple: (dic_miRNA_disease, dic_gene_disease)
    """
    # Load raw miRNA-disease associations
    raw_miRNA_dict = load_disease.load_miRNA_disease_dict(config['file_paths']['mirna_disease_db'])
    
    # Filter miRNA dictionary: keep only miRNAs in target list and diseases in MeSH list
    dic_miRNA_disease = {
        miRNA: [disease for disease in diseases if disease in mesh_diseases] 
        for miRNA, diseases in raw_miRNA_dict.items() 
        if miRNA in target_miRNAs
    }

    # Load raw gene-disease associations
    raw_gene_dict = load_disease.load_gene_disease_dict_csv(config['file_paths']['gene_disease_db'])

    print(raw_gene_dict)
    
    # Filter gene dictionary: keep only genes in target list and diseases in MeSH list
    dic_gene_disease = {
        gene: [disease for disease in diseases if disease in mesh_diseases] 
        for gene, diseases in raw_gene_dict.items() 
        if gene in target_genes
    }
    
    return dic_miRNA_disease, dic_gene_disease

def compute_misim_row(row, miRNA_dict, gene_dict):
    """
    Calculates the MISIM value for a single row of the DataFrame.

    Args:
        row (pd.Series): A row from the dataframe containing 'miRNA' and 'gene'.
        miRNA_dict (dict): Dictionary mapping miRNAs to diseases.
        gene_dict (dict): Dictionary mapping genes to diseases.

    Returns:
        pd.Series: Contains the calculated MISIM value and disease counts.
    """
    try:
        current_miRNA = row['miRNA']
        current_gene = row['gene']

        # Retrieve associated diseases (handle potential key errors if necessary, 
        # though filtering suggests keys should exist if data is consistent)
        miRNA_diseases = miRNA_dict.get(current_miRNA, [])
        gene_diseases = gene_dict.get(current_gene, [])
        
        # Calculate functional similarity using MISIM module
        misim_value = MISIM.MISIM(miRNA_diseases, gene_diseases)

        return pd.Series({
            'miR_disease_num': int(len(miRNA_diseases)),
            'gene_disease_num': int(len(gene_diseases)),
            'MISIM_value': misim_value
        })
    except Exception as e:
        # Error handling for unexpected issues during row processing
        print(f"Error processing row {row.name}: {e}")
        return pd.Series({
            'miR_disease_num': 0,
            'gene_disease_num': 0,
            'MISIM_value': 0.0
        })

def format_runtime(seconds):
    """Format seconds into Hours, Minutes, Seconds."""
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return int(hours), int(minutes), secs

def main():
    start_time = time.time()
    
    # 1. Load Configuration
    config = load_config('config/compute_MISIM_scores.json')
    paths = config['file_paths']
    sep = config['settings']['csv_separator']

    print("Loading data...")
    
    # 2. Load Core Data
    df_MTI = pd.read_csv(paths['input_mti'], sep=sep)
    lst_miRNA = df_MTI['miRNA'].to_list()
    lst_gene = df_MTI['gene'].to_list()

    df_mesh = pd.read_csv(paths['mesh_tree'])
    lst_mesh_disease = df_mesh['Disease Name'].to_list()

    # 3. Prepare Dictionaries
    dic_miRNA_disease, dic_gene_disease = load_and_filter_dictionaries(
        config, lst_miRNA, lst_gene, lst_mesh_disease
    )

    print(f"Starting processing for {len(df_MTI)} rows...")
    
    # 4. Processing
    tqdm.pandas(desc="Processing Progress")
    
    # Apply the computation function row by row
    result_columns = df_MTI.progress_apply(
        lambda row: compute_misim_row(row, dic_miRNA_disease, dic_gene_disease), 
        axis=1
    )

    # Assign results back to DataFrame
    df_MTI['MISIM_value'] = result_columns['MISIM_value']

    # 5. Save Results
    df_MTI.to_csv(paths['output_result'], index=None, sep=sep)

    # 6. Final Statistics
    end_time = time.time()
    h, m, s = format_runtime(end_time - start_time)

    print('Processing complete!')
    print(f'Total runtime: {h} hours {m} minutes {s:.2f} seconds')
    print(f'Results saved to: {paths["output_result"]}')

if __name__ == '__main__':
    main()