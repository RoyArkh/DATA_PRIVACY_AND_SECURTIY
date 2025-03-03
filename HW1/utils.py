import csv
import glob
import os
import sys
from copy import deepcopy
from typing import Optional
import numpy as np

if sys.version_info[0] != 3 or sys.version_info[1] < 5:
    sys.stdout.write("Requires Python 3.x.\n")
    sys.exit(1)

##############################################################################
# Helper Functions                                                           #
# These functions are provided to you as starting points. They may help your #
# code remain structured and organized. But you are not required to use      #
# them. You can modify them or implement your own helper functions.          #
##############################################################################

def read_dataset(dataset_file: str):
    """ Read a dataset into a list and return.

    Args:
        dataset_file (str): path to the dataset file.

    Returns:
        list[dict]: a list of dataset rows.
    """
    result = []
    with open(dataset_file) as f:
        records = csv.DictReader(f)
        for row in records:
            result.append(row)
    # print(result[0]['age']) # debug: testing.
    return result


def write_dataset(dataset, dataset_file: str) -> bool:
    """ Writes a dataset to a csv file.

    Args:
        dataset: the data in list[dict] format
        dataset_file: str, the path to the csv file

    Returns:
        bool: True if succeeds.
    """
    assert len(dataset) > 0, "The anonymized dataset is empty."
    keys = dataset[0].keys()
    with open(dataset_file, 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(dataset)
    return True


def read_DGH(DGH_file: str):
    """ Reads one DGH file and returns in desired format.

    Args:
        DGH_file (str): the path to DGH file.
    """
    def add_to_hierarchy(hierarchy, path):
        current = hierarchy
        for level in path:
            if level not in current:
                current[level] = {}
            current = current[level]
        return current

    hierarchy = {}
    with open(DGH_file, 'r') as f:
        stack = [] 
        for line in f:
            level = len(line) - len(line.lstrip('\t'))
            item = line.strip()
            stack = stack[:level]
            stack.append(item)
            add_to_hierarchy(hierarchy, stack)

    return hierarchy


# print(read_DGH('./DGHs/age.txt'))
# print(read_DGH('./DGHs/education.txt'))
print(read_DGH('./DGHs/gender.txt'))

def read_DGHs(DGH_folder: str) -> dict:
    """ Read all DGH files from a directory and put them into a dictionary.

    Args:
        DGH_folder (str): the path to the directory containing DGH files.

    Returns:
        dict: a dictionary where each key is attribute name and values
            are DGHs in your desired format.
    """
    DGHs = {}
    for DGH_file in glob.glob(DGH_folder + "/*.txt"):
        attribute_name = os.path.basename(DGH_file)[:-4]
        DGHs[attribute_name] = read_DGH(DGH_file);

    return DGHs


##############################################################################
# Mandatory Functions                                                        #
# You need to complete these functions without changing their parameters.    #
##############################################################################

def calculate_distortion(dgh, raw_value, anon_value):
    """Calculate the number of generalization steps from raw_value to anon_value.

    Args:
        dgh (dict): The DGH hierarchy for the attribute.
        raw_value (str): The original value in the raw dataset.
        anon_value (str): The generalized value in the anonymized dataset.

    Returns:
        int: The number of steps (distortion) needed.
    """
    def find_path(hierarchy, target, path=None):
        if path is None:
            path = []
        for key, sub_hierarchy in hierarchy.items():
            if key == target:
                return path + [key]
            if isinstance(sub_hierarchy, dict):
                result = find_path(sub_hierarchy, target, path + [key])
                if result:
                    return result
        return None

    # Find paths for both values in the DGH tree
    raw_path = find_path(dgh, raw_value)
    anon_path = find_path(dgh, anon_value)
    
    if not raw_path or not anon_path:
        return 0  # No distortion if values are not found

    # Find common ancestor and calculate the number of steps
    i = 0
    while i < min(len(raw_path), len(anon_path)) and raw_path[i] == anon_path[i]:
        i += 1

    # Total steps to reach the generalized value from raw_value
    return (len(raw_path) - i) + (len(anon_path) - i)


def cost_MD(raw_dataset_file: str, anonymized_dataset_file: str,
            DGH_folder: str) -> float:
    """Calculate Distortion Metric (MD) cost between two datasets.

    Args:
        raw_dataset_file (str): the path to the raw dataset file.
        anonymized_dataset_file (str): the path to the anonymized dataset file.
        DGH_folder (str): the path to the DGH directory.

    Returns:
        float: the calculated cost.
    """
    raw_dataset = read_dataset(raw_dataset_file)
    anonymized_dataset = read_dataset(anonymized_dataset_file)
    
    assert len(raw_dataset) > 0 and len(raw_dataset) == len(anonymized_dataset)
    assert len(raw_dataset[0]) == len(anonymized_dataset[0])

    DGHs = read_DGHs(DGH_folder)

    total_cost = 0
    # Calculate distortion for each record
    for raw_row, anon_row in zip(raw_dataset, anonymized_dataset):
        for attribute, raw_value in raw_row.items():
            anon_value = anon_row[attribute]
            if raw_value != anon_value and attribute in DGHs:
                dgh = DGHs[attribute]
                total_cost += calculate_distortion(dgh, raw_value, anon_value)

    return total_cost


print('---------------------------')
print('cost md')
print(cost_MD('./adult-hw1.csv', 'adult-hw1-anon.csv', './DGHs/'))
print('---------------------------')


def count_descendant_leaves(hierarchy, value):
    """
    Count the number of descendant leaves for a given value in the hierarchy.
    
    Args:
        hierarchy (dict): The hierarchy dictionary
        value (str): The value to find descendant leaves for
    
    Returns:
        int: Number of descendant leaves
    """
    def _count_leaves(subtree):
        # If the subtree is empty or a leaf, return 1
        if not subtree:
            return 1
        
        # Sum leaves of all children
        return sum(_count_leaves(child) for child in subtree.values())
    
    def _find_node(tree, target):
        # Find the subtree corresponding to the target value
        if not tree:
            return None
        
        if target in tree:
            return tree[target]
        
        # Recursively search in children
        for child_key, child_value in tree.items():
            found = _find_node(child_value, target)
            if found is not None:
                return found
        
        return None
    
    # Find the node in the hierarchy
    node = _find_node(hierarchy, value)
    
    # If node not found, return 0
    if node is None:
        return 0
    
    # Count total leaves in the entire hierarchy
    total_hierarchy_leaves = _count_leaves(hierarchy)
    
    # Count leaves in this node's subtree
    node_leaves = _count_leaves(node)
    
    return node_leaves

def calculate_lm_for_value(hierarchy, value):
    """
    Calculate Loss Metric (LM) for a single value.
    
    Args:
        hierarchy (dict): The hierarchy dictionary
        value (str): The value to calculate LM for
    
    Returns:
        float: LM cost for the value
    """
    total_hierarchy_leaves = count_descendant_leaves(hierarchy, list(hierarchy.keys())[0])
    
    # If the value is 'Any' or not in hierarchy, return max cost
    if value == 'Any' or value is None:
        return 1.0
    
    descendant_leaves = count_descendant_leaves(hierarchy, value)
    
    # LM calculation as per the problem statement
    return (descendant_leaves - 1) / (total_hierarchy_leaves - 1)

def cost_LM(raw_dataset_file: str, anonymized_dataset_file: str, DGH_folder: str) -> float:
    """
    Calculate Loss Metric (LM) cost between two datasets.
    
    Args:
        raw_dataset_file (str): Path to the raw dataset file
        anonymized_dataset_file (str): Path to the anonymized dataset file
        DGH_folder (str): Path to the DGH directory
    
    Returns:
        float: Calculated LM cost
    """
    # Read datasets and DGHs
    raw_dataset = read_dataset(raw_dataset_file)
    anonymized_dataset = read_dataset(anonymized_dataset_file)
    
    # Verify dataset integrity
    assert (len(raw_dataset) > 0 and len(raw_dataset) == len(anonymized_dataset)
            and len(raw_dataset[0]) == len(anonymized_dataset[0]))
    
    # Read Domain Generalization Hierarchies
    DGHs = read_DGHs(DGH_folder)
    
    # Identify quasi-identifier attributes (all except 'income')
    quasi_identifiers = [attr for attr in raw_dataset[0].keys() if attr != 'income']
    
    # Calculate weights for quasi-identifiers
    num_qi_attrs = len(quasi_identifiers)
    attr_weights = {attr: 1/num_qi_attrs for attr in quasi_identifiers}
    
    # Calculate total LM cost
    total_lm_cost = 0
    
    # Iterate through each record
    for raw_record, anon_record in zip(raw_dataset, anonymized_dataset):
        record_lm_cost = 0
        
        # Calculate LM cost for each quasi-identifier
        for attr in quasi_identifiers:
            raw_value = raw_record[attr]
            anon_value = anon_record[attr]
            
            # Skip if no hierarchy for this attribute
            if attr not in DGHs:
                continue
            
            # Calculate LM for this attribute value
            lm_value = calculate_lm_for_value(DGHs[attr], anon_value)
            
            # Multiply by attribute weight
            record_lm_cost += attr_weights[attr] * lm_value
        
        # Add record's LM cost to total
        total_lm_cost += record_lm_cost
    
    return total_lm_cost / len(raw_dataset)


print('---------------------------')
print('loss metric')
print(cost_LM('./limited-adult.csv', 'limited-adult-anon.csv', './DGHs/'))
print('---------------------------')


'''
def cost_LM(raw_dataset_file: str, anonymized_dataset_file: str,
            DGH_folder: str) -> float:
    """Calculate Loss Metric (LM) cost between two datasets.

    Args:
        raw_dataset_file (str): the path to the raw dataset file.
        anonymized_dataset_file (str): the path to the anonymized dataset file.
        DGH_folder (str): the path to the DGH directory.

    Returns:
        float: the calculated cost.
    """
    raw_dataset = read_dataset(raw_dataset_file)
    anonymized_dataset = read_dataset(anonymized_dataset_file)
    assert (len(raw_dataset) > 0 and len(raw_dataset) == len(anonymized_dataset)
            and len(raw_dataset[0]) == len(anonymized_dataset[0]))
    DGHs = read_DGHs(DGH_folder)

    # TODO: complete this function.
    return -1
'''

def get_qi_attributes(dataset):
    """
    Get quasi-identifier attributes from the dataset.
    
    Args:
        dataset (list): The input dataset
    
    Returns:
        list: Quasi-identifier attributes
    """
    # Exclude 'income' and 'index' from QI attributes
    return [attr for attr in dataset[0].keys() if attr not in ['income', 'index']]

def get_qi_values(record, qi_attributes):
    """
    Extract QI values for a record.
    
    Args:
        record (dict): A single record
        qi_attributes (list): List of quasi-identifier attributes
    
    Returns:
        tuple: Values of QI attributes
    """
    return tuple(record[attr] for attr in qi_attributes)

def generalize_value(value, hierarchy, current_level=None):
    """
    Generalize a value to a more generic level in the hierarchy.
    
    Args:
        value (str): The value to generalize
        hierarchy (dict): The hierarchy for the attribute
        current_level (str, optional): Current level in the hierarchy
    
    Returns:
        str: Generalized value
    """
    def find_value_path(tree, target):
        """Find the path to a value in the hierarchy."""
        if not tree:
            return None
        
        for key, subtree in tree.items():
            if key == target:
                return [key]
            
            sub_path = find_value_path(subtree, target)
            if sub_path:
                return [key] + sub_path
        
        return None

    # If value is already 'Any', return 'Any'
    if value == 'Any':
        return 'Any'
    
    # Find the full path of the value in the hierarchy
    path = find_value_path(hierarchy, value)
    
    # If no path found, return original value
    if not path:
        return value
    
    # If current_level is None, go to the first parent
    if current_level is None:
        return path[0] if len(path) > 1 else value
    
    # Find current position of the current_level in the path
    try:
        current_index = path.index(current_level)
        # Return parent if exists
        return path[current_index + 1] if current_index + 1 < len(path) else current_level
    except ValueError:
        # If current_level not in path, return first parent
        return path[0] if len(path) > 1 else value

def create_equivalence_classes(shuffled_dataset, k, DGHs):
    """
    Create equivalence classes ensuring k-anonymity.
    
    Args:
        shuffled_dataset (list): Shuffled dataset
        k (int): k-anonymity parameter
        DGHs (dict): Domain Generalization Hierarchies
    
    Returns:
        list: Anonymized equivalence classes
    """
    # Get QI attributes
    qi_attributes = get_qi_attributes(shuffled_dataset)
    
    # Create equivalence classes
    clusters = []
    for i in range(0, len(shuffled_dataset), k):
        # Take next k records (or remaining records if less than k)
        ec_records = shuffled_dataset[i:i+k]
        
        # Generalize records to make them k-anonymous
        anonymized_ec = generalize_equivalence_class(ec_records, qi_attributes, DGHs, k)
        
        clusters.append(anonymized_ec)
    
    return clusters

def generalize_equivalence_class(ec_records, qi_attributes, DGHs, k):
    """
    Generalize an equivalence class to achieve k-anonymity.
    
    Args:
        ec_records (list): Records in the equivalence class
        qi_attributes (list): Quasi-identifier attributes
        DGHs (dict): Domain Generalization Hierarchies
        k (int): k-anonymity parameter
    
    Returns:
        list: Anonymized records in the equivalence class
    """
    # Make a deep copy to avoid modifying original records
    ec_records = copy.deepcopy(ec_records)
    
    # If we already have k or fewer records, no need to generalize
    if len(ec_records) <= k:
        return ec_records
    
    # Try to achieve k-anonymity through generalization
    generalization_levels = {attr: None for attr in qi_attributes}
    
    # Keep track of current equivalence class
    while not is_k_anonymous(ec_records, qi_attributes, k):
        # Find the attribute that minimizes information loss when generalized
        best_attr = find_best_generalization_attribute(
            ec_records, qi_attributes, DGHs, generalization_levels
        )
        
        # Generalize all records for this attribute
        for record in ec_records:
            record[best_attr] = generalize_value(
                record[best_attr], 
                DGHs[best_attr], 
                generalization_levels[best_attr]
            )
        
        # Update generalization level for this attribute
        generalization_levels[best_attr] = (
            generalization_levels[best_attr] or 
            list(DGHs[best_attr].keys())[0]
        )
    
    return ec_records

def is_k_anonymous(records, qi_attributes, k):
    """
    Check if the records form a k-anonymous group.
    
    Args:
        records (list): Records to check
        qi_attributes (list): Quasi-identifier attributes
        k (int): k-anonymity parameter
    
    Returns:
        bool: True if k-anonymous, False otherwise
    """
    # Group records by their QI values
    qi_groups = {}
    for record in records:
        qi_values = get_qi_values(record, qi_attributes)
        if qi_values not in qi_groups:
            qi_groups[qi_values] = []
        qi_groups[qi_values].append(record)
    
    # Check if all groups have at least k records
    return all(len(group) >= k for group in qi_groups.values())

def find_best_generalization_attribute(
    records, qi_attributes, DGHs, current_levels
):
    """
    Find the attribute that minimizes information loss when generalized.
    
    Args:
        records (list): Records to generalize
        qi_attributes (list): Quasi-identifier attributes
        DGHs (dict): Domain Generalization Hierarchies
        current_levels (dict): Current generalization levels
    
    Returns:
        str: Attribute to generalize
    """
    # Candidate attributes are those not fully generalized
    candidate_attrs = [
        attr for attr in qi_attributes 
        if current_levels[attr] is None
    ]
    
    # If no candidates, choose an attribute with the least current generalization
    if not candidate_attrs:
        candidate_attrs = qi_attributes
    
    # Choose attribute with the least information loss
    return min(candidate_attrs, key=lambda attr: 
        estimate_information_loss(records, attr, DGHs, current_levels)
    )

def estimate_information_loss(records, attr, DGHs, current_levels):
    """
    Estimate information loss for generalizing an attribute.
    
    Args:
        records (list): Records to generalize
        attr (str): Attribute to generalize
        DGHs (dict): Domain Generalization Hierarchies
        current_levels (dict): Current generalization levels
    
    Returns:
        float: Estimated information loss
    """
    # Number of unique values before and after generalization
    current_unique_values = set(record[attr] for record in records)
    
    # Simulate generalization
    generalized_values = set()
    for record in records:
        generalized_value = generalize_value(
            record[attr], 
            DGHs[attr], 
            current_levels[attr]
        )
        generalized_values.add(generalized_value)
    
    # Information loss is the reduction in unique values
    return len(current_unique_values) - len(generalized_values)

def random_anonymizer(raw_dataset_file: str, DGH_folder: str, k: int,
                      s: int, output_file: str):
    """
    K-anonymization of a dataset using a randomized algorithm.
    
    Args:
        raw_dataset_file (str): Path to the raw dataset file
        DGH_folder (str): Path to the DGH directory
        k (int): k-anonymity parameter
        s (int): Seed for randomization
        output_file (str): Path to the output dataset file
    """
    # Read dataset and DGHs
    raw_dataset = read_dataset(raw_dataset_file)
    DGHs = read_DGHs(DGH_folder)
    
    # Add index to track original positions
    for i in range(len(raw_dataset)):
        raw_dataset[i]['index'] = i
    
    raw_dataset = np.array(raw_dataset)
    np.random.seed(s)  # Ensure consistency between runs
    np.random.shuffle(raw_dataset)  # Shuffle the dataset
    
    # Create equivalence classes
    clusters = create_equivalence_classes(raw_dataset, k, DGHs)
    
    # Restructure dataset according to original indexes
    anonymized_dataset = [None] * len(raw_dataset)
    for cluster in clusters:
        for item in cluster:
            anonymized_dataset[item['index']] = item
            del item['index']
    
    # Write anonymized dataset
    write_dataset(anonymized_dataset, output_file)


'''
def random_anonymizer(raw_dataset_file: str, DGH_folder: str, k: int,
                      s: int, output_file: str):
    """ K-anonymization a dataset, given a set of DGHs and a k-anonymity param.

    Args:
        raw_dataset_file (str): the path to the raw dataset file.
        DGH_folder (str): the path to the DGH directory.
        k (int): k-anonymity parameter.
        output_file (str): the path to the output dataset file.
        s (int): seed of the randomization function
    """
    raw_dataset = read_dataset(raw_dataset_file)
    DGHs = read_DGHs(DGH_folder)

    for i in range(len(raw_dataset)):  ##set indexing to not lose original places of records
        raw_dataset[i]['index'] = i

    raw_dataset = np.array(raw_dataset)
    np.random.seed(s)  ## to ensure consistency between runs
    np.random.shuffle(raw_dataset)  ##shuffle the dataset to randomize

    clusters = []

    D = len(raw_dataset)

    # TODO: START WRITING YOUR CODE HERE. Do not modify code in this function above this line.
    # Store your results in the list named "clusters".
    # Order of the clusters is important. First cluster should be the first EC, second cluster second EC, ...

    # END OF STUDENT'S CODE. Do not modify code in this function below this line.

    anonymized_dataset = [None] * D

    for cluster in clusters:  # restructure according to previous indexes
        for item in cluster:
            anonymized_dataset[item['index']] = item
            del item['index']

    write_dataset(anonymized_dataset, output_file)
'''

def clustering_anonymizer(raw_dataset_file: str, DGH_folder: str, k: int, output_file: str):
    """ Clustering-based anonymization a dataset, given a set of DGHs.

    Args:
        raw_dataset_file (str): the path to the raw dataset file.
        DGH_folder (str): the path to the DGH directory.
        k (int): k-anonymity parameter.
        output_file (str): the path to the output dataset file.
    """
    raw_dataset = read_dataset(raw_dataset_file)
    DGHs = read_DGHs(DGH_folder)

    anonymized_dataset = []
    # TODO: complete this function.

    write_dataset(anonymized_dataset, output_file)


def bottom_up_anonymizer(raw_dataset_file: str, DGH_folder: str, k: int, l:int, output_file: str):
    """ Bottom up-based anonymization a dataset, given a set of DGHs.

    Args:
        raw_dataset_file (str): the path to the raw dataset file.
        DGH_folder (str): the path to the DGH directory.
        k (int): k-anonymity parameter.
        l (int): distinct l-diversity parameter.
        output_file (str): the path to the output dataset file.
    """
    raw_dataset = read_dataset(raw_dataset_file)
    DGHs = read_DGHs(DGH_folder)

    anonymized_dataset = []
    # TODO: complete this function

    # Finally, write dataset to a file
    write_dataset(anonymized_dataset, output_file)



# Command line argument handling and calling of respective anonymizer:
if len(sys.argv) < 6:
    print(f"Usage: python3 {sys.argv[0]} algorithm DGH-folder raw-dataset.csv anonymized.csv k")
    print(f"\tWhere algorithm is one of [clustering, random, bottom_up]")
    sys.exit(1)

algorithm = sys.argv[1]
if algorithm not in ['clustering', 'random', 'bottom_up']:
    print("Invalid algorithm.")
    sys.exit(2)

dgh_path = sys.argv[2]
raw_file = sys.argv[3]
anonymized_file = sys.argv[4]
k = int(sys.argv[5])

function = eval(f"{algorithm}_anonymizer")
if function != clustering_anonymizer:
    if len(sys.argv) != 7:
        print(
            f"Usage: python3 {sys.argv[0]} <algorithm name> DGH-folder raw-dataset.csv k anonymized.csv seed/l(random/bottom_up)")
        print(f"\tWhere algorithm is one of [clustering, random, bottom_up]")
        sys.exit(1)

    last_param = int(sys.argv[6])
    function(raw_file, dgh_path, k, last_param, anonymized_file)
else:
    function(raw_file, dgh_path, k, anonymized_file)

cost_md = cost_MD(raw_file, anonymized_file, dgh_path)
cost_lm = cost_LM(raw_file, anonymized_file, dgh_path)
print(f"Results of {k}-anonimity:\n\tCost_MD: {cost_md}\n\tCost_LM: {cost_lm}\n")

# Sample usage:
# python3 code.py clustering DGHs/ adult-hw1.csv result.csv 300