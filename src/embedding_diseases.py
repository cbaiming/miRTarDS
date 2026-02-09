"""
Disease Embedding Generation Pipeline

This script generates sentence embeddings for diseases using a pre-trained SBERT model.
It processes disease names from miRTarDS dataset, standardizes them, and saves embeddings to CSV.

Key steps:
1. Load configuration parameters
2. Initialize SBERT model
3. Extract diseases from gene-disease and miRNA-disease associations
4. Standardize disease names
5. Generate embeddings
6. Save results
"""

import torch
from sentence_transformers import SentenceTransformer
import json
import pandas as pd
import unicodedata
import re
import csv
import time
import data.load_disease as load_disease

def load_config(config_path):
    """
    Load configuration parameters from a JSON file.
    
    Args:
        config_path (str): Path to the JSON configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config

def standardize_disease_name(name):
    """
    Standardize disease names by normalizing Unicode, removing extra whitespace, and converting to lowercase.
    
    Args:
        name (str): Raw disease name
        
    Returns:
        str: Standardized disease name
    """
    # Normalize unicode to NFKC form (compatibility + canonical decomposition)
    name = unicodedata.normalize("NFKC", name)
    # Remove leading and trailing whitespace
    name = name.strip()
    # Replace multiple whitespace characters with a single space
    name = re.sub(r"\s+", " ", name)
    # Convert to lowercase for consistency
    name = name.lower()
    return name

def load_and_filter_data(miRTarDS_path, gene_disease_db_path, miRNA_disease_excel_path):
    """
    Load miRTarDS data and filter disease dictionaries based on genes/miRNAs in the dataset.
    
    Args:
        miRTarDS_path (str): Path to miRTarDS CSV file
        gene_disease_db_path (str): Path to gene-disease database
        miRNA_disease_excel_path (str): Path to miRNA-disease Excel file
        
    Returns:
        tuple: (gene_disease_dict, miRNA_disease_dict, list of all unique diseases)
    """
    # Load miRTarDS training/validation data
    df_miRTarDS_train_valid = pd.read_csv(miRTarDS_path, sep='\t')
    
    # Extract unique genes and miRNAs from the dataset
    lst_gene = df_miRTarDS_train_valid['gene'].to_list()
    lst_miRNA = df_miRTarDS_train_valid['miRNA'].to_list()
    
    # Load gene-disease associations and filter to include only genes in miRTarDS
    gene_disease_dict = load_disease.load_gene_disease_dict_csv(gene_disease_db_path)
    gene_disease_dict = {key: value for key, value in gene_disease_dict.items() 
                         if key in lst_gene}
    
    # Load miRNA-disease associations and filter to include only miRNAs in miRTarDS
    miRNA_disease_dict = load_disease.load_miRNA_disease_dict(miRNA_disease_excel_path)
    miRNA_disease_dict = {key: value for key, value in miRNA_disease_dict.items() 
                          if key in lst_miRNA}
    
    # Extract all unique diseases from both dictionaries
    gene_diseases_set = set()
    for diseases_list in gene_disease_dict.values():
        gene_diseases_set.update(diseases_list)
    
    miRNA_diseases_set = set()
    for diseases_list in miRNA_disease_dict.values():
        miRNA_diseases_set.update(diseases_list)
    
    # Combine diseases from both sources
    all_diseases_set = gene_diseases_set.union(miRNA_diseases_set)
    all_diseases_list = list(all_diseases_set)
    
    return gene_disease_dict, miRNA_disease_dict, all_diseases_list

def encode_diseases(model, disease_list, device, batch_size):
    """
    Generate embeddings for a list of diseases using the specified model.
    
    Args:
        model (SentenceTransformer): Pretrained SentenceTransformer model
        disease_list (list): List of disease names to encode
        device (torch.device): Device to run the model on
        batch_size (int): Batch size for encoding
        
    Returns:
        numpy.ndarray: Disease embeddings
    """
    # Generate embeddings for all diseases
    embeddings = model.encode(
        disease_list,
        device=device,
        batch_size=batch_size,
        convert_to_tensor=False,
        show_progress_bar=True
    )
    return embeddings

def save_embeddings_to_csv(disease_list, embeddings, output_path, dimension_prefix):
    """
    Save disease embeddings to a CSV file with standardized disease names.
    
    Args:
        disease_list (list): Original disease names
        embeddings (numpy.ndarray): Disease embeddings
        output_path (str): Path to save the CSV file
        dimension_prefix (str): Prefix for embedding dimension column names
    """
    # Prepare data for CSV writing
    csv_data = []
    for disease, embedding in zip(disease_list, embeddings):
        standardized_name = standardize_disease_name(disease)
        # Create row with disease name followed by embedding dimensions
        row = [standardized_name] + embedding.tolist()
        csv_data.append(row)
    
    # Create column names: disease_name + dim_0, dim_1, ...
    column_names = ["disease_name"] + [f"{dimension_prefix}{i}" 
                                        for i in range(len(embeddings[0]))]
    
    # Write to CSV file
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(column_names)
        writer.writerows(csv_data)

def main():
    """Main function to execute the disease embedding pipeline."""
    start_time = time.time()
    
    # Load configuration from JSON file
    config = load_config('config/embedding_disease.json')
    
    # Extract configuration parameters
    model_config = config['model']
    data_config = config['data']
    output_config = config['output']
    processing_config = config['processing']
    
    # Initialize model with specified device
    device = torch.device(model_config['device'])
    model = SentenceTransformer(model_config['path'])
    model.to(device)
    
    # Load and filter data
    _, _, all_diseases = load_and_filter_data(
        data_config['miRTarDS_train_valid'],
        data_config['gene_disease_db'],
        data_config['miRNA_disease_excel']
    )
    
    # Generate disease embeddings
    embeddings = encode_diseases(
        model, 
        all_diseases, 
        device, 
        processing_config['batch_size']
    )
    
    # Save embeddings to CSV file
    save_embeddings_to_csv(
        all_diseases,
        embeddings,
        output_config['disease_embeddings'],
        processing_config['encoding_dimension_prefix']
    )
    
    # Print completion message
    print(f"Completed! Encoded {len(all_diseases)} diseases. "
          f"Results saved to {output_config['disease_embeddings']}")
    
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()