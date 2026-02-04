import pandas as pd
import time
import json
from typing import Dict, List, Tuple, Set
import load_disease

# ===========================
# LOAD CONFIGURATION FROM JSON
# ===========================
def load_config(config_file: str = 'config/process_mti.json') -> Dict:
    """Load configuration from JSON file."""
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config

# Load configuration
config = load_config()

# Extract configuration variables
RAW_DATA_PATHS = config['RAW_DATA_PATHS']
OUTPUT_PATH = config['OUTPUT_PATH']
FILTERS = config['FILTERS']
PROCESSING = config['PROCESSING']

MTIKey = Tuple[str, str]  # (miRNA, gene)
MTIValue = List[str]      # List of database sources

def process_mirtarbase(mti_dict: Dict[MTIKey, MTIValue]) -> Set[MTIKey]:
    """
    Process MTIs from miRTarBase database.
    
    Args:
        mti_dict: Dictionary to store MTI data
        
    Returns:
        Set of existing keys for quick lookup
    """
    print("Processing miRTarBase data...")
    
    # Read and filter miRTarBase data
    df_mirtarbase: pd.DataFrame = pd.read_csv(
        RAW_DATA_PATHS['miRTarBase'], 
        sep=PROCESSING['csv_separator'], 
        low_memory=PROCESSING['low_memory']
    )
    
    # Apply filters
    df_mirtarbase = df_mirtarbase[
        df_mirtarbase['Species (miRNA)'] == FILTERS['species']
    ]
    df_mirtarbase = df_mirtarbase[
        df_mirtarbase['Support Type'] == FILTERS['support_type']
    ]
    
    # Select relevant columns and clean data
    df_mirtarbase = df_mirtarbase[['miRNA', 'Target Gene', 'Support Type']]
    df_mirtarbase['Target Gene'] = df_mirtarbase['Target Gene'].str.upper()
    df_mirtarbase['miRNA'] = df_mirtarbase['miRNA'].str.replace(r'(-3p|-5p)$', '', regex=True)
    df_mirtarbase['miRNA'] = df_mirtarbase['miRNA'].str.replace(' ', '', regex=False)
    
    # Reset index for clean output
    df_mirtarbase.reset_index(drop=True, inplace=True)
    
    # Populate dictionary
    mti_dict.update({
        (miRNA, target_gene): ['miRTarBase'] 
        for miRNA, target_gene in zip(df_mirtarbase['miRNA'], df_mirtarbase['Target Gene'])
    })
    
    existing_keys: Set[MTIKey] = set(mti_dict.keys())
    print(f"  - Added {len(existing_keys)} unique MTIs from miRTarBase")
    
    return existing_keys

def process_tarbase(mti_dict: Dict[MTIKey, MTIValue], existing_keys: Set[MTIKey]) -> None:
    """
    Process MTIs from TarBase database.
    
    Args:
        mti_dict: Dictionary containing MTI data
        existing_keys: Set of existing MTI keys for quick lookup
    """
    print("Processing TarBase data...")
    
    df_tarbase: pd.DataFrame = pd.read_csv(
        RAW_DATA_PATHS['TarBase'], 
        sep=PROCESSING['tsv_separator'], 
        low_memory=PROCESSING['low_memory']
    )
    
    # Data cleaning and filtering
    df_tarbase['gene_name'] = df_tarbase['gene_name'].str.upper()
    df_tarbase = df_tarbase[
        df_tarbase['experimental_method'].isin(FILTERS['experimental_methods'])
    ]
    df_tarbase = df_tarbase[['mirna_name', 'gene_name']]
    df_tarbase['mirna_name'] = df_tarbase['mirna_name'].str.replace(r'(-3p|-5p)$', '', regex=True)
    df_tarbase = df_tarbase.drop_duplicates()
    
    added_count: int = 0
    updated_count: int = 0
    
    for miRNA, gene in zip(df_tarbase['mirna_name'], df_tarbase['gene_name']):
        key: MTIKey = (miRNA, gene.upper())
        
        if key in existing_keys:
            mti_dict[key].append('TarBase')
            updated_count += 1
        else:
            mti_dict[key] = ['TarBase']
            existing_keys.add(key)
            added_count += 1
    
    print(f"  - Added {added_count} new MTIs from TarBase")
    print(f"  - Updated {updated_count} existing MTIs with TarBase support")

def process_mirwalk(mti_dict: Dict[MTIKey, MTIValue], existing_keys: Set[MTIKey]) -> None:
    """
    Process MTIs from miRWalk database.
    
    Args:
        mti_dict: Dictionary containing MTI data
        existing_keys: Set of existing MTI keys for quick lookup
    """
    print("Processing miRWalk data...")
    
    added_count: int = 0
    updated_count: int = 0
    
    with open(RAW_DATA_PATHS['miRWalk'], 'r') as file:
        # Skip header line
        next(file)
        
        for line in file:
            elements: List[str] = line.strip().split('\t')
            
            if len(elements) < 3:
                continue
            
            miRNA: str = elements[0].replace('-5p', '').replace('-3p', '')
            gene: str = elements[2].upper()
            
            key: MTIKey = (miRNA, gene)
            
            if key in existing_keys:
                mti_dict[key].append('miRWalk')
                updated_count += 1
            else:
                mti_dict[key] = ['miRWalk']
                existing_keys.add(key)
                added_count += 1
    
    print(f"  - Added {added_count} new MTIs from miRWalk")
    print(f"  - Updated {updated_count} existing MTIs with miRWalk support")

def create_refseq_dict() -> Dict[str, str]:
    """
    Create dictionary mapping RefSeq IDs to gene symbols.
    
    Returns:
        Dictionary with RefSeq IDs as keys and gene symbols as values
    """
    print("Creating RefSeq to gene symbol mapping...")
    
    df_hgnc: pd.DataFrame = pd.read_csv(
        RAW_DATA_PATHS['HGNC'], 
        sep=PROCESSING['tsv_separator']
    )
    
    refseq_dict: Dict[str, str] = {
        str(row['RefSeq IDs']).strip(): str(row['Approved symbol']).strip() 
        for _, row in df_hgnc.iterrows()
    }
    
    print(f"  - Mapped {len(refseq_dict)} RefSeq IDs to gene symbols")
    return refseq_dict

def process_mirdb(mti_dict: Dict[MTIKey, MTIValue], existing_keys: Set[MTIKey], 
                  refseq_dict: Dict[str, str]) -> None:
    """
    Process MTIs from miRDB database.
    
    Args:
        mti_dict: Dictionary containing MTI data
        existing_keys: Set of existing MTI keys for quick lookup
        refseq_dict: Dictionary mapping RefSeq IDs to gene symbols
    """
    print("Processing miRDB data...")
    
    added_count: int = 0
    updated_count: int = 0
    
    with open(RAW_DATA_PATHS['miRDB'], 'r') as file:
        # Skip header line
        next(file)
        
        for line in file:
            elements: List[str] = line.strip().split('\t')
            
            if len(elements) < 2:
                continue
            
            miRNA: str = elements[0]
            
            # Filter for human miRNAs
            if FILTERS['species'] not in miRNA:
                continue
            
            # Clean miRNA name
            miRNA = miRNA.replace('-5p', '').replace('-3p', '')
            
            # Get gene symbol from RefSeq ID
            refseq_id: str = elements[1]
            gene: str = refseq_dict.get(refseq_id.strip(), "")
            
            if not gene or gene == 'Null':
                continue
            
            key: MTIKey = (miRNA, gene)
            
            if key in existing_keys:
                mti_dict[key].append('miRDB')
                updated_count += 1
            else:
                mti_dict[key] = ['miRDB']
                existing_keys.add(key)
                added_count += 1
    
    print(f"  - Added {added_count} new MTIs from miRDB")
    print(f"  - Updated {updated_count} existing MTIs with miRDB support")

def process_targetscan(mti_dict: Dict[MTIKey, MTIValue], existing_keys: Set[MTIKey]) -> None:
    """
    Process MTIs from TargetScan database.
    
    Args:
        mti_dict: Dictionary containing MTI data
        existing_keys: Set of existing MTI keys for quick lookup
    """
    print("Processing TargetScan data...")
    
    added_count: int = 0
    updated_count: int = 0
    
    with open(RAW_DATA_PATHS['TargetScan'], 'r') as file:
        # Skip header line
        next(file)
        
        for line in file:
            elements: List[str] = line.strip().split('\t')
            
            if len(elements) < 5:
                continue
            
            miRNA: str = elements[4]
            
            # Filter for human miRNAs
            if FILTERS['species'] not in miRNA:
                continue
            
            # Clean miRNA name
            miRNA = miRNA.replace('-5p', '').replace('-3p', '')
            
            # Get gene symbol
            gene: str = elements[1].upper()
            
            # Additional check for miRNA format
            if 'hsa-' in miRNA:
                key: MTIKey = (miRNA, gene)
                
                if key in existing_keys:
                    mti_dict[key].append('TargetScan')
                    updated_count += 1
                else:
                    mti_dict[key] = ['TargetScan']
                    existing_keys.add(key)
                    added_count += 1
    
    print(f"  - Added {added_count} new MTIs from TargetScan")
    print(f"  - Updated {updated_count} existing MTIs with TargetScan support")

def filter_by_disease_association(dataframe: pd.DataFrame, 
                                  dic_miRNA_disease: Dict[str, List[str]], 
                                  dic_gene_disease: Dict[str, List[str]]) -> pd.DataFrame:
    """
    Filter MTIs based on disease associations.
    
    Args:
        dataframe: DataFrame containing MTI data
        dic_miRNA_disease: Dictionary mapping miRNAs to associated diseases
        dic_gene_disease: Dictionary mapping genes to associated diseases
        
    Returns:
        Filtered DataFrame
    """
    print("Filtering MTIs by disease association...")
    
    # Create sets for faster lookup
    miRNA_with_disease: Set[str] = set(dic_miRNA_disease.keys())
    gene_with_disease: Set[str] = set(dic_gene_disease.keys())
    
    # Filter rows
    initial_count: int = len(dataframe)
    dataframe = dataframe[
        dataframe['miRNA'].isin(miRNA_with_disease) & 
        dataframe['gene'].isin(gene_with_disease)
    ]
    
    filtered_count: int = len(dataframe)
    print(f"  - Filtered out {initial_count - filtered_count} MTIs without disease associations")
    print(f"  - Remaining {filtered_count} MTIs with disease associations")
    
    return dataframe

def add_disease_counts(dataframe: pd.DataFrame, 
                       dic_miRNA_disease: Dict[str, List[str]], 
                       dic_gene_disease: Dict[str, List[str]]) -> pd.DataFrame:
    """
    Add disease count columns to DataFrame.
    
    Args:
        dataframe: DataFrame containing MTI data
        dic_miRNA_disease: Dictionary mapping miRNAs to associated diseases
        dic_gene_disease: Dictionary mapping genes to associated diseases
        
    Returns:
        DataFrame with added disease count columns
    """
    print("Adding disease count columns...")
    
    dataframe['miR_disease_num'] = dataframe['miRNA'].apply(
        lambda x: len(dic_miRNA_disease.get(x, []))
    )
    
    dataframe['gene_disease_num'] = dataframe['gene'].apply(
        lambda x: len(dic_gene_disease.get(x, []))
    )
    
    return dataframe

def main() -> None:
    """
    Main function to process MTI data from multiple databases.
    """
    print("Starting MTI data processing pipeline...")
    start_time: float = time.time()
    
    # Initialize dictionary for MTI data
    mti_dict: Dict[MTIKey, MTIValue] = {}
    
    # Process each database
    existing_keys: Set[MTIKey] = process_mirtarbase(mti_dict)
    process_tarbase(mti_dict, existing_keys)
    process_mirwalk(mti_dict, existing_keys)
    
    # Create RefSeq mapping for miRDB
    refseq_dict: Dict[str, str] = create_refseq_dict()
    process_mirdb(mti_dict, existing_keys, refseq_dict)
    
    process_targetscan(mti_dict, existing_keys)
    
    # Convert dictionary to DataFrame
    print("Converting MTI data to DataFrame...")
    data: List[List[str]] = []
    
    for (miRNA, gene), support_types in mti_dict.items():
        # Remove duplicates and sort support types
        unique_support_types: List[str] = sorted(set(support_types))
        support_str: str = ';'.join(unique_support_types)
        data.append([miRNA, gene, support_str])
    
    df: pd.DataFrame = pd.DataFrame(data, columns=['miRNA', 'gene', 'support type'])
    print(f"  - Total unique MTIs collected: {len(df)}")
    
    # Load disease associations
    print("Loading disease association data...")
    dic_miRNA_disease: Dict[str, List[str]] = load_disease.load_miRNA_disease_dict(
        RAW_DATA_PATHS['miRNA_disease']
    )
    dic_gene_disease: Dict[str, List[str]] = load_disease.load_gene_disease_dict(
        RAW_DATA_PATHS['gene_disease']
    )
    
    # Filter and add disease counts
    df = filter_by_disease_association(df, dic_miRNA_disease, dic_gene_disease)
    df = add_disease_counts(df, dic_miRNA_disease, dic_gene_disease)
    
    # Save results
    print(f"Saving results to {OUTPUT_PATH}...")
    df.to_csv(OUTPUT_PATH, index=False, encoding='utf-8', sep='\t')
    
    # Print summary statistics
    end_time: float = time.time()
    processing_time: float = end_time - start_time
    
    print("\n" + "="*50)
    print("PROCESSING COMPLETE")
    print("="*50)
    print(f"Total processing time: {processing_time:.2f} seconds")
    print(f"Final MTI count: {len(df)}")
    print(f"Unique miRNAs: {df['miRNA'].nunique()}")
    print(f"Unique genes: {df['gene'].nunique()}")
    print(f"Database support distribution:")
    
    # Count occurrences of each database
    support_counts: Dict[str, int] = {}
    for support_str in df['support type']:
        databases = support_str.split(';')
        for db in databases:
            support_counts[db] = support_counts.get(db, 0) + 1
    
    for db, count in sorted(support_counts.items()):
        print(f"  - {db}: {count} MTIs")

if __name__ == "__main__":
    main()