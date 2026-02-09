import sqlite3
import pandas as pd
from collections import defaultdict
import csv

def load_data(db_path):
    """
    Load data from SQLite database into pandas DataFrames.
    
    This function connects to an SQLite database and extracts three tables:
    geneDiseaseNetwork, geneAttributes, and diseaseAttributes.
    
    Parameters:
    -----------
    db_path : str
        Path to the SQLite database file
        
    Returns:
    --------
    tuple
        A tuple containing three DataFrames: 
        (gene_disease_df, gene_attributes_df, disease_attributes_df)
    """
    conn = sqlite3.connect(db_path)
    
    # Load gene-disease association network
    gene_disease_df = pd.read_sql_query("SELECT * FROM geneDiseaseNetwork", conn)
    # Load gene attribute information
    gene_attributes_df = pd.read_sql_query("SELECT * FROM geneAttributes", conn)
    # Load disease attribute information
    disease_attributes_df = pd.read_sql_query("SELECT * FROM diseaseAttributes", conn)
    
    conn.close()
    return gene_disease_df, gene_attributes_df, disease_attributes_df

# load data directly from DisGeNET 2020 .db file
def load_gene_disease_dict(db_path):
    """
    Create a dictionary mapping genes to associated diseases from SQLite database.
    
    This function joins multiple tables to create a comprehensive mapping
    between gene names and their associated disease names.
    
    Parameters:
    -----------
    db_path : str
        Path to the SQLite database file
        
    Returns:
    --------
    defaultdict(set)
        A dictionary where keys are gene names and values are sets of 
        associated disease names
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # SQL query to join gene-disease network with gene and disease attributes
    query = r"""
    SELECT ga.geneName, da.diseaseName
    FROM geneDiseaseNetwork gd
    JOIN geneAttributes ga ON gd.geneNID = ga.geneNID
    JOIN diseaseAttributes da ON gd.diseaseNID = da.diseaseNID
    """
    
    cursor.execute(query)
    gene_disease_pairs = cursor.fetchall()
    conn.close()
    
    # Create dictionary using defaultdict for automatic set creation
    dic_gene_disease = defaultdict(set)
    
    # Populate dictionary with gene-disease pairs
    for gene, disease in gene_disease_pairs:
        dic_gene_disease[gene].add(disease)
    
    return dic_gene_disease

# load data from the csv file
def load_gene_disease_dict_csv(filename):

    result = {}
    
    with open(filename, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        next(reader)  # Skip header row
        
        for row in reader:
            if len(row) >= 2:
                key = row[0]
                # Convert semicolon-separated string back to set
                values = set(row[1].split(';')) if row[1] else set()
                result[key] = values
    
    return result

def load_miRNA_disease_dict(db_path):
    """
    Create a dictionary mapping miRNAs to associated diseases from Excel file.
    
    This function processes an Excel file containing miRNA-disease associations,
    standardizes miRNA naming, and reformats disease names.
    
    Parameters:
    -----------
    db_path : str
        Path to the Excel file containing miRNA-disease associations
        
    Returns:
    --------
    dict
        A dictionary where keys are miRNA names and values are lists of 
        associated disease names
    """
    # Load miRNA-disease associations from Excel file
    df_miRNA_disease = pd.read_excel(db_path)

    # Standardize miRNA naming: replace lowercase 'r' with uppercase 'R'
    df_miRNA_disease['miRNA'] = df_miRNA_disease['miRNA'].str.replace('r', 'R')

    dic_miRNA_disease = {}
    # Process each row in the dataframe
    for index, row in df_miRNA_disease.iterrows():
        miRNA = row['miRNA']
        disease = row['disease']

        # Reformat disease names from "X, Y" format to "Y X" format
        if ',' in disease:
            parts = disease.split(', ')
            if len(parts) == 2:
                disease = f"{parts[1]} {parts[0]}"

        # Add disease to miRNA's associated disease list
        if miRNA not in dic_miRNA_disease:
            dic_miRNA_disease[miRNA] = [disease]
        else:
            # Avoid duplicate disease entries
            if disease not in dic_miRNA_disease[miRNA]:
                dic_miRNA_disease[miRNA].append(disease)
                
    return dic_miRNA_disease