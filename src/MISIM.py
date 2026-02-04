import csv

# ============================================================================
# Disease Code Loading Module
# ============================================================================

def load_disease_codes(csv_file_path):
    """
    Load disease codes from a CSV file into a dictionary.
    
    The CSV file should have two columns: disease name and corresponding code.
    Multiple codes may exist for a single disease.
    
    Args:
        csv_file_path (str): Path to the CSV file containing disease codes.
        
    Returns:
        dict: Dictionary mapping disease names to lists of associated codes.
              Format: {disease_name: [code1, code2, ...]}
    """
    disease_code_dict = {}
    
    with open(csv_file_path, 'r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip header row
        
        for row in csv_reader:
            disease_name, code = row[0], row[1]
            
            # Initialize list for new diseases, append code to existing list
            if disease_name not in disease_code_dict:
                disease_code_dict[disease_name] = []
            disease_code_dict[disease_name].append(code)
    
    return disease_code_dict


# Load disease codes from CSV file
dic_disease_code = load_disease_codes('raw_data/MeSH_Tree.csv')

# Similarity decay factor (default value = 0.5)
delta = 0.5


# ============================================================================
# Core Similarity Calculation Functions
# ============================================================================

def find_common_code(code1, code2):
    """
    Find the longest common prefix between two hierarchical codes.
    
    Codes are hierarchical strings separated by dots (e.g., "C04.588.180").
    Returns the shared prefix up to the first differing component.
    
    Args:
        code1 (str): First hierarchical code.
        code2 (str): Second hierarchical code.
        
    Returns:
        str: Longest common prefix of the two codes, or empty string if none.
    """
    components1 = code1.split('.')
    components2 = code2.split('.')
    common_components = []
    
    min_length = min(len(components1), len(components2))
    for i in range(min_length):
        if components1[i] == components2[i]:
            common_components.append(components1[i])
        else:
            break
    
    return '.'.join(common_components)


def find_codes_starting_with(prefix, code_list):
    """
    Find codes in a list that begin with a specified prefix.
    
    Args:
        prefix (str): Prefix to search for.
        code_list (list): List of codes to search through.
        
    Returns:
        str: First code in the list that starts with the given prefix.
    """
    return [code for code in code_list if code.startswith(prefix)][0]


def DV(disease_name):
    """
    Calculate the DV (disease value) for a given disease.
    
    DV represents the hierarchical depth-weighted sum of a disease's codes.
    Each code contributes 1 for the root level plus delta^i for each
    subsequent level i, where delta is the decay factor.
    
    Args:
        disease_name (str): Name of the disease.
        
    Returns:
        float: DV value for the disease.
    """
    code_list = dic_disease_code[disease_name]
    dv_value = 1  # Base value

    for code in code_list:
        # Count hierarchical levels (number of dots)
        depth_levels = code.count('.')
        
        # Add contributions from each hierarchical level
        for level in range(depth_levels):
            dv_value = dv_value + delta ** (level + 1)

    return dv_value


def S_ab(disease1, disease2):
    """
    Calculate the semantic similarity between two diseases (S_ab).
    
    The similarity is based on shared hierarchical code prefixes,
    weighted by the decay factor delta.
    
    Args:
        disease1 (str): First disease name.
        disease2 (str): Second disease name.
        
    Returns:
        float: Normalized similarity score between 0 and 1.
    """
    code_list_1 = dic_disease_code[disease1]
    code_list_2 = dic_disease_code[disease2]
    
    # Find all common code prefixes between the two diseases
    common_codes = []
    for code1 in code_list_1:
        for code2 in code_list_2:
            common_code = find_common_code(code1, code2)
            if common_code:
                common_codes.append(common_code)
    
    similarity_sum = 0
    
    # Calculate similarity contributions from each common prefix
    for common_prefix in common_codes:
        # Find the most specific codes starting with the common prefix
        code1 = find_codes_starting_with(common_prefix, code_list_1)
        code2 = find_codes_starting_with(common_prefix, code_list_2)
        
        # Contribution from first disease
        if code1 == common_prefix:
            # Exact match: include all levels up to common prefix
            for level in range(code1.count('.') + 1):
                similarity_sum = similarity_sum + delta ** level
        else:
            # Partial match: include levels beyond common prefix
            for level in range(code1.count('.') + 1):
                if level > common_prefix.count('.'):
                    similarity_sum = similarity_sum + delta ** level
        
        # Contribution from second disease
        if code2 == common_prefix:
            for level in range(code2.count('.') + 1):
                similarity_sum = similarity_sum + delta ** level
        else:
            for level in range(code2.count('.') + 1):
                if level > common_prefix.count('.'):
                    similarity_sum = similarity_sum + delta ** level
    
    # Normalize by the sum of individual disease values
    return similarity_sum / (DV(disease1) + DV(disease2))


# ============================================================================
# Disease List Similarity Calculation
# ============================================================================

def MISIM(disease_list1, disease_list2):
    """
    Calculate the MISIM (Maximum Information Similarity) between two disease lists.
    
    For each disease in list1, find the maximum similarity to any disease in list2.
    For each disease in list2, find the maximum similarity to any disease in list1.
    Return the average of these maximum similarities.
    
    Args:
        disease_list1 (list): First list of disease names.
        disease_list2 (list): Second list of disease names.
        
    Returns:
        float: Average maximum similarity between the two disease lists.
               Returns 0 if either list is empty.
    """
    # Handle empty list cases
    if len(disease_list1) == 0 or len(disease_list2) == 0:
        return 0
    
    # Calculate maximum similarities for diseases in first list
    max_similarities_list1 = []
    for disease1 in disease_list1:
        max_similarity = max(S_ab(disease1, disease2) 
                           for disease2 in disease_list2)
        max_similarities_list1.append(max_similarity)
    
    # Calculate maximum similarities for diseases in second list
    max_similarities_list2 = []
    for disease2 in disease_list2:
        max_similarity = max(S_ab(disease1, disease2) 
                           for disease1 in disease_list1)
        max_similarities_list2.append(max_similarity)
    
    # Compute weighted average of maximum similarities
    numerator = sum(max_similarities_list1) + sum(max_similarities_list2)
    denominator = len(disease_list1) + len(disease_list2)
    
    return numerator / denominator