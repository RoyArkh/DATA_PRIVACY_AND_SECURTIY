import csv
import glob
import os
import sys
from copy import copy, deepcopy
from typing import Optional
import numpy as np
from collections import defaultdict
from typing import List, Dict, Set
from itertools import product

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
    # print(result[0]['age']) #testing
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
# print(read_DGH('./DGHs/gender.txt'))

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

#cmd helper
def calculate_distortion(dgh, raw_value, anon_value):
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

    raw_path = find_path(dgh, raw_value)
    anon_path = find_path(dgh, anon_value)
    
    if not raw_path or not anon_path:
        return 0
    
    i = 0
    while i < min(len(raw_path), len(anon_path)) and raw_path[i] == anon_path[i]:
        i += 1

    return (len(raw_path) - i) + (len(anon_path) - i)


def cost_MD(raw_dataset_file: str, anonymized_dataset_file: str,
            DGH_folder: str) -> float:
    
    raw_dataset = read_dataset(raw_dataset_file)
    anonymized_dataset = read_dataset(anonymized_dataset_file)
    
    assert len(raw_dataset) > 0 and len(raw_dataset) == len(anonymized_dataset)
    assert len(raw_dataset[0]) == len(anonymized_dataset[0])

    DGHs = read_DGHs(DGH_folder)

    total_cost = 0
    for raw_row, anon_row in zip(raw_dataset, anonymized_dataset):
        for attribute, raw_value in raw_row.items():
            anon_value = anon_row[attribute]
            if raw_value != anon_value and attribute in DGHs:
                dgh = DGHs[attribute]
                total_cost += calculate_distortion(dgh, raw_value, anon_value)

    return total_cost


# print('---------------------------')
# print('cost md')
# print(cost_MD('./adult-hw1.csv', 'adult-hw1-anon.csv', './DGHs/'))
# print('---------------------------')

#clm helper
def count_descendant_leaves(hierarchy, value):
    def count_leaves(subtree):
        if not subtree:
            return 1 
        return sum(count_leaves(child) for child in subtree.values())
    
    def find_node(tree, target):
        if not tree:
            return None
        
        if target in tree:
            return tree[target]
        
        for child_key, child_value in tree.items():
            found = find_node(child_value, target)
            if found is not None:
                return found
        
        return None
    
    node = find_node(hierarchy, value)

    if node is None:
        return 0
    
    total_hierarchy_leaves = count_leaves(hierarchy)
    node_leaves = count_leaves(node)
    
    return node_leaves

def calculate_lm_for_value(hierarchy, value):
    total_hierarchy_leaves = count_descendant_leaves(hierarchy, list(hierarchy.keys())[0])
    
    if value == 'Any' or value is None:
        return 1.0
    
    descendant_leaves = count_descendant_leaves(hierarchy, value)
    
    return (descendant_leaves - 1) / (total_hierarchy_leaves - 1)

def cost_LM(raw_dataset_file: str, anonymized_dataset_file: str, DGH_folder: str) -> float:
    raw_dataset = read_dataset(raw_dataset_file)
    anonymized_dataset = read_dataset(anonymized_dataset_file)
    
    # assert (len(raw_dataset) > 0 and len(raw_dataset) == len(anonymized_dataset)
    #         and len(raw_dataset[0]) == len(anonymized_dataset[0]))
    
    DGHs = read_DGHs(DGH_folder)
    quasi_identifiers = [attr for attr in raw_dataset[0].keys() if attr != 'income']
    num_qi_attrs = len(quasi_identifiers)
    attr_weights = {attr: 1/num_qi_attrs for attr in quasi_identifiers}
    total_lm_cost = 0
    
    for raw_record, anon_record in zip(raw_dataset, anonymized_dataset):
        record_lm_cost = 0
        for attr in quasi_identifiers:
            raw_value = raw_record[attr]
            anon_value = anon_record[attr]
            
            if attr not in DGHs:
                continue
            
            lm_value = calculate_lm_for_value(DGHs[attr], anon_value)
            record_lm_cost += attr_weights[attr] * lm_value
        total_lm_cost += record_lm_cost
    
    return total_lm_cost / len(raw_dataset)


# print('---------------------------')
# print('loss metric')
# print(cost_LM('./limited-adult.csv', 'limited-adult-anon.csv', './DGHs/'))
# print('---------------------------')


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

    for i in range(len(raw_dataset)):
        raw_dataset[i]['index'] = i

    raw_dataset = np.array(raw_dataset)
    np.random.seed(s)  # Ensure consistency between runs
    np.random.shuffle(raw_dataset)  # Shuffle the dataset

    clusters = []
    D = len(raw_dataset)

    num_clusters = (D + k - 1) // k
    for i in range(num_clusters):
        start_idx = i * k
        end_idx = min(start_idx + k, D)
        clusters.append(list(raw_dataset[start_idx:end_idx]))

    for cluster in clusters:
        for attr in raw_dataset[0].keys():
            if attr == 'index':
                continue

            if attr not in DGHs:
                continue

            dgh = DGHs[attr]
            values = [record[attr] for record in cluster]
            if len(set(values)) == 1:
                continue

            def find_common_ancestor(values, dgh):
                def find_path(hierarchy, value, path=None):
                    if path is None:
                        path = []
                    for key, sub_hierarchy in hierarchy.items():
                        if key == value:
                            return path + [key]
                        if isinstance(sub_hierarchy, dict):
                            result = find_path(sub_hierarchy, value, path + [key])
                            if result:
                                return result
                    return None

                paths = [find_path(dgh, v) for v in values]
                if not all(paths):
                    return None

                common_ancestor = paths[0]
                for path in paths[1:]:
                    common_ancestor = [ca for ca, pa in zip(common_ancestor, path) if ca == pa]
                    if not common_ancestor:
                        break
                return common_ancestor[-1] if common_ancestor else None

            common_value = find_common_ancestor(values, dgh)
            for record in cluster:
                record[attr] = common_value

    anonymized_dataset = [None] * D
    for cluster in clusters:
        for item in cluster:
            anonymized_dataset[item['index']] = item
            del item['index']

    write_dataset(anonymized_dataset, output_file)


# random_anonymizer('./limited-adult.csv', './DGHs/', 2, 43, 'limited-adult-k.csv',)

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

def find_common_generalization(DGHs, r1, r2):
    common_values = {}
    
    for attr in r1.keys():
        if attr == 'income' or attr not in DGHs:
            continue
            
        def find_common_ancestor(dgh, val1, val2):
            def get_path(dgh, target, path=None):
                if path is None:
                    path = []
                for key, sub_dgh in dgh.items():
                    if key == target:
                        return path + [key]
                    if isinstance(sub_dgh, dict):
                        result = get_path(sub_dgh, target, path + [key])
                        if result:
                            return result
                return None

            path1 = get_path(dgh, val1)
            path2 = get_path(dgh, val2)
            
            if not path1 or not path2:
                return None
                
            for i in range(min(len(path1), len(path2))):
                if path1[i] != path2[i]:
                    return path1[i-1] if i > 0 else path1[i]
            return path1[min(len(path1), len(path2)) - 1]
            
        common_values[attr] = find_common_ancestor(DGHs[attr], r1[attr], r2[attr])
    
    return common_values

def calculate_max_md_cost(DGHs):
    max_cost = 0
    for dgh in DGHs.values():
        def get_max_depth(hierarchy):
            if not hierarchy:
                return 0
            return 1 + max(get_max_depth(sub) for sub in hierarchy.values())
        max_cost += get_max_depth(dgh)
    return max_cost

def calculate_pairwise_distance(DGHs, r1, r2):
    common_gen = find_common_generalization(DGHs, r1, r2)
    
    hypo_raw = [r1, r2]
    hypo_anon = [
        {**r1, **common_gen},
        {**r2, **common_gen}
    ]
    
    quasi_identifiers = [attr for attr in r1.keys() if attr != 'income' and attr in DGHs]
    num_qi_attrs = len(quasi_identifiers)
    attr_weights = {attr: 1/num_qi_attrs for attr in quasi_identifiers}
    
    lm_dist = 0
    for attr in quasi_identifiers:
        if attr in common_gen:
            lm_value = calculate_lm_for_value(DGHs[attr], common_gen[attr])
            lm_dist += attr_weights[attr] * lm_value
    
    md_dist = 0
    max_md = calculate_max_md_cost(DGHs)
    for attr in quasi_identifiers:
        if attr in common_gen:
            md_dist += calculate_distortion(DGHs[attr], r1[attr], common_gen[attr])
            md_dist += calculate_distortion(DGHs[attr], r2[attr], common_gen[attr])
    
    normalized_md_dist = md_dist / (max_md * 2) if max_md > 0 else 0
    
    return lm_dist + normalized_md_dist

def find_closest_records(records, used_indices, DGHs):
    min_dist = float('inf')
    best_pair = None
    
    for i in range(len(records)):
        if i in used_indices:
            continue
        for j in range(i + 1, len(records)):
            if j in used_indices:
                continue
            
            dist = calculate_pairwise_distance(DGHs, records[i], records[j])
            if dist < min_dist:
                min_dist = dist
                best_pair = (i, j)
            elif dist == min_dist:
                if best_pair is None or (i < best_pair[0]) or (i == best_pair[0] and j < best_pair[1]):
                    best_pair = (i, j)
    
    return best_pair

def find_closest_record_to_cluster(records, cluster, unused_indices, DGHs):
    min_dist = float('inf')
    best_idx = None
    
    cluster_gen = {}
    for attr in records[0].keys():
        if attr != 'income' and attr in DGHs:
            attr_values = [r[attr] for r in cluster]
            cluster_gen[attr] = find_common_generalization(DGHs, cluster[0], cluster[0])[attr]
            for r in cluster[1:]:
                new_gen = find_common_generalization(DGHs, {attr: cluster_gen[attr]}, r)[attr]
                cluster_gen[attr] = new_gen
    
    for i in unused_indices:
        dist = calculate_pairwise_distance(DGHs, records[i], cluster_gen)
        if dist < min_dist:
            min_dist = dist
            best_idx = i
        elif dist == min_dist and (best_idx is None or i < best_idx):
            best_idx = i
    
    return best_idx

def clustering_anonymizer(raw_dataset_file: str, DGH_folder: str, k: int, output_file: str):
    """Clustering-based anonymization of a dataset, given a set of DGHs."""
    raw_dataset = read_dataset(raw_dataset_file)
    DGHs = read_DGHs(DGH_folder)
    
    used_indices = set()
    anonymized_dataset = [None] * len(raw_dataset)
    
    remainder = len(raw_dataset) % k
    if remainder > 0:
        start_idx = len(raw_dataset) - remainder
        last_complete_cluster_start = start_idx - k
        cluster = [raw_dataset[i] for i in range(last_complete_cluster_start, start_idx)]
        
        for i in range(start_idx, len(raw_dataset)):
            used_indices.add(i)
            cluster.append(raw_dataset[i])
        
        common_gen = {}
        for attr in raw_dataset[0].keys():
            if attr != 'income' and attr in DGHs:
                common_gen[attr] = cluster[0][attr]
                for r in cluster[1:]:
                    new_gen = find_common_generalization(DGHs, {attr: common_gen[attr]}, r)[attr]
                    common_gen[attr] = new_gen
        
        for i, record in enumerate(cluster):
            idx = last_complete_cluster_start + i
            anonymized_dataset[idx] = {**record, **common_gen}
    
    while len(used_indices) < len(raw_dataset) - remainder:
        closest_pair = find_closest_records(raw_dataset, used_indices, DGHs)
        if not closest_pair:
            break
            
        cluster = [raw_dataset[closest_pair[0]], raw_dataset[closest_pair[1]]]
        used_indices.add(closest_pair[0])
        used_indices.add(closest_pair[1])
        
        while len(cluster) < k:
            unused_indices = set(range(len(raw_dataset))) - used_indices
            next_record_idx = find_closest_record_to_cluster(raw_dataset, cluster, unused_indices, DGHs)
            if next_record_idx is None:
                break
                
            cluster.append(raw_dataset[next_record_idx])
            used_indices.add(next_record_idx)
        
        common_gen = {}
        for attr in raw_dataset[0].keys():
            if attr != 'income' and attr in DGHs:
                common_gen[attr] = cluster[0][attr]
                for r in cluster[1:]:
                    new_gen = find_common_generalization(DGHs, {attr: common_gen[attr]}, r)[attr]
                    common_gen[attr] = new_gen
        
        for record in cluster:
            idx = raw_dataset.index(record)
            anonymized_dataset[idx] = {**record, **common_gen}
    
    write_dataset(anonymized_dataset, output_file)


# clustering_anonymizer('./limited-adult.csv', './DGHs/', 2, 'limited-adult-cluster.csv')

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
'''

def get_next_generalization_level(records, DGHs, current_values):
    next_levels = []
    attributes = list(DGHs.keys())
    
    for attr in attributes:
        new_values = current_values.copy()
        current_attr_values = set(record[attr] for record in records)
        
        # Find parent values in DGH
        found_parent = False
        for value in current_attr_values:
            for parent, children in DGHs[attr].items():
                if value in children:
                    new_values[attr] = parent
                    found_parent = True
                    break
            if found_parent:
                break
                
        if found_parent:
            next_levels.append(new_values)
    
    return next_levels

def generalize_dataset(dataset, generalization_values):
    generalized = []
    for record in dataset:
        new_record = record.copy()
        for attr, value in generalization_values.items():
            if value is not None:
                new_record[attr] = value
        generalized.append(new_record)
    return generalized

def check_k_anonymity(dataset, k, quasi_identifiers):
    groups = {}
    for record in dataset:
        key = tuple(record[qi] for qi in quasi_identifiers)
        groups[key] = groups.get(key, 0) + 1
    
    return all(count >= k for count in groups.values())

def check_l_diversity(dataset, l, sensitive_attr):
    groups = {}
    for record in dataset:
        qi_values = tuple(record[attr] for attr in record.keys() if attr != sensitive_attr)
        if qi_values not in groups:
            groups[qi_values] = set()
        groups[qi_values].add(record[sensitive_attr])
    
    return all(len(distinct_values) >= l for distinct_values in groups.values())

def calculate_lm_score(dataset, original_dataset, DGHs):
    total_score = 0
    num_records = len(dataset)
    num_attributes = len(DGHs)
    
    for i, record in enumerate(dataset):
        record_score = 0
        for attr in DGHs.keys():
            if record[attr] != original_dataset[i][attr]:
                record_score += 1
        total_score += record_score / num_attributes
    
    return total_score / num_records

def bottom_up_anonymizer(raw_dataset_file: str, DGH_folder: str, k: int, l: int, output_file: str):
    """Bottom up-based anonymization of a dataset, given a set of DGHs."""
    raw_dataset = read_dataset(raw_dataset_file)
    DGHs = read_DGHs(DGH_folder)
    
    quasi_identifiers = [attr for attr in DGHs.keys()]
    sensitive_attr = "income"
    current_values = {attr: None for attr in quasi_identifiers}
    best_generalization = None
    best_score = float('inf')
    
    current_dataset = raw_dataset.copy()
    levels_to_check = [current_values]
    
    while levels_to_check:
        current_level_values = levels_to_check.pop(0)
        current_dataset = generalize_dataset(raw_dataset, current_level_values)
        if (check_k_anonymity(current_dataset, k, quasi_identifiers) and 
            check_l_diversity(current_dataset, l, sensitive_attr)):
            current_score = calculate_lm_score(current_dataset, raw_dataset, DGHs)
            
            if current_score < best_score:
                best_score = current_score
                best_generalization = current_level_values
        else:
            next_levels = get_next_generalization_level(current_dataset, DGHs, current_level_values)
            levels_to_check.extend(next_levels)
    
    if best_generalization is not None:
        anonymized_dataset = generalize_dataset(raw_dataset, best_generalization)
    else:
        anonymized_dataset = raw_dataset
        for record in anonymized_dataset:
            for attr in quasi_identifiers:
                record[attr] = list(DGHs[attr].keys())[-1]
    
    write_dataset(anonymized_dataset, output_file)


# bottom_up_anonymizer('./limited-adult.csv', './DGHs/', 2, 1, 'limited-adult-bottom.csv')

'''
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
'''


# print('---------------------------')
# print('cost md')
# print(cost_MD('./adult-hw1.csv', 'adult-hw1-anon.csv', './DGHs/'))
# print('---------------------------')

# print('---------------------------')
# print('loss metric')
# print(cost_LM('./limited-adult.csv', 'limited-adult-anon.csv', './DGHs/'))
# print('---------------------------')

# print('---------------------------')
# print('cluster running')
# clustering_anonymizer('./limited-adult.csv', './DGHs/', 2, 'limited-adult-cluster.csv')
# print('---------------------------')

# print('---------------------------')
# print('random running')
# random_anonymizer('./limited-adult.csv', './DGHs/', 2, 43, 'limited-adult-k.csv',)
# print('---------------------------')

# print('---------------------------')
# print('bottomup running')
# bottom_up_anonymizer('./limited-adult.csv', './DGHs/', 0, 0, 'limited-adult-bottom.csv')
# print('---------------------------')



import time

# start_time = time.time()
# random_anonymizer('./adult-hw1.csv', './DGHs/', 4, 1, 'limited-adult-k.csv')
# end_time = time.time()
# elapsed_time = end_time - start_time
# print(f"Execution time random 4: {elapsed_time:.2f} seconds")

# start_time = time.time()
# random_anonymizer('./adult-hw1.csv', './DGHs/', 8, 1, 'limited-adult-k.csv')
# end_time = time.time()
# elapsed_time = end_time - start_time
# print(f"Execution time random 8: {elapsed_time:.2f} seconds")

# start_time = time.time()
# random_anonymizer('./adult-hw1.csv', './DGHs/', 16, 1, 'limited-adult-k.csv')
# end_time = time.time()
# elapsed_time = end_time - start_time
# print(f"Execution time random 16: {elapsed_time:.2f} seconds")

# start_time = time.time()
# random_anonymizer('./adult-hw1.csv', './DGHs/', 32, 1, 'limited-adult-k.csv')
# end_time = time.time()
# elapsed_time = end_time - start_time
# print(f"Execution time random 32: {elapsed_time:.2f} seconds")

# start_time = time.time()
# random_anonymizer('./adult-hw1.csv', './DGHs/', 64, 1, 'limited-adult-k.csv')
# end_time = time.time()
# elapsed_time = end_time - start_time
# print(f"Execution time random 64: {elapsed_time:.2f} seconds")

# start_time = time.time()
# random_anonymizer('./adult-hw1.csv', './DGHs/', 128, 1, 'limited-adult-k.csv')
# end_time = time.time()
# elapsed_time = end_time - start_time
# print(f"Execution time random 128: {elapsed_time:.2f} seconds")

# start_time = time.time()
# random_anonymizer('./adult-hw1.csv', './DGHs/', 256, 1, 'limited-adult-k.csv')
# end_time = time.time()
# elapsed_time = end_time - start_time
# print(f"Execution time random 256: {elapsed_time:.2f} seconds")


#######################################################

# start_time = time.time()
# clustering_anonymizer('./adult-hw1.csv', './DGHs/', 4, 'limited-adult-k.csv')
# end_time = time.time()
# elapsed_time = end_time - start_time
# print(f"Execution time cluster 4: {elapsed_time:.2f} seconds")

# start_time = time.time()
# clustering_anonymizer('./adult-hw1.csv', './DGHs/', 8, 'limited-adult-k.csv')
# end_time = time.time()
# elapsed_time = end_time - start_time
# print(f"Execution time clustering 8: {elapsed_time:.2f} seconds")

# start_time = time.time()
# clustering_anonymizer('./adult-hw1.csv', './DGHs/', 16, 'limited-adult-k.csv')
# end_time = time.time()
# elapsed_time = end_time - start_time
# print(f"Execution time clustering 16: {elapsed_time:.2f} seconds")

# start_time = time.time()
# clustering_anonymizer('./adult-hw1.csv', './DGHs/', 32, 'limited-adult-k.csv')
# end_time = time.time()
# elapsed_time = end_time - start_time
# print(f"Execution time clustering 32: {elapsed_time:.2f} seconds")

# start_time = time.time()
# clustering_anonymizer('./adult-hw1.csv', './DGHs/', 64, 'limited-adult-k.csv')
# end_time = time.time()
# elapsed_time = end_time - start_time
# print(f"Execution time clustering 64: {elapsed_time:.2f} seconds")

# start_time = time.time()
# clustering_anonymizer('./adult-hw1.csv', './DGHs/', 128, 'limited-adult-k.csv')
# end_time = time.time()
# elapsed_time = end_time - start_time
# print(f"Execution time clustering 128: {elapsed_time:.2f} seconds")

# start_time = time.time()
# clustering_anonymizer('./adult-hw1.csv', './DGHs/', 256, 'limited-adult-k.csv')
# end_time = time.time()
# elapsed_time = end_time - start_time
# print(f"Execution time clustering 256: {elapsed_time:.2f} seconds")


#######################################################

# start_time = time.time()
# bottom_up_anonymizer('./adult-hw1.csv', './DGHs/', 16, 1, 'limited-adult-k.csv')
# end_time = time.time()
# elapsed_time = end_time - start_time
# print(f"Execution time bottom up 4: {elapsed_time:.2f} seconds")

# start_time = time.time()
# bottom_up_anonymizer('./adult-hw1.csv', './DGHs/', 16, 2, 'limited-adult-k.csv')
# end_time = time.time()
# elapsed_time = end_time - start_time
# print(f"Execution time bottom_up 8: {elapsed_time:.2f} seconds")

# start_time = time.time()
# bottom_up_anonymizer('./adult-hw1.csv', './DGHs/', 16, 3, 'limited-adult-k.csv')
# end_time = time.time()
# elapsed_time = end_time - start_time
# print(f"Execution time bottom_up 16: {elapsed_time:.2f} seconds")

# start_time = time.time()
# bottom_up_anonymizer('./adult-hw1.csv', './DGHs/', 16, 4, 'limited-adult-k.csv')
# end_time = time.time()
# elapsed_time = end_time - start_time
# print(f"Execution time bottom_up 32: {elapsed_time:.2f} seconds")

# start_time = time.time()
# bottom_up_anonymizer('./adult-hw1.csv', './DGHs/', 16, 5, 'limited-adult-k.csv')
# end_time = time.time()
# elapsed_time = end_time - start_time
# print(f"Execution time bottom_up 64: {elapsed_time:.2f} seconds")

# start_time = time.time()
# bottom_up_anonymizer('./adult-hw1.csv', './DGHs/', 16, 6, 'limited-adult-k.csv')
# end_time = time.time()
# elapsed_time = end_time - start_time
# print(f"Execution time bottom_up 128: {elapsed_time:.2f} seconds")

# start_time = time.time()
# bottom_up_anonymizer('./adult-hw1.csv', './DGHs/', 16, 7, 'limited-adult-k.csv')
# end_time = time.time()
# elapsed_time = end_time - start_time
# print(f"Execution time bottom_up 256: {elapsed_time:.2f} seconds")

# start_time = time.time()
# bottom_up_anonymizer('./adult-hw1.csv', './DGHs/', 16, 8, 'limited-adult-k.csv')
# end_time = time.time()
# elapsed_time = end_time - start_time
# print(f"Execution time bottom_up 256: {elapsed_time:.2f} seconds")


########################################################################


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




# print('gay')

############################################################
############################################################
