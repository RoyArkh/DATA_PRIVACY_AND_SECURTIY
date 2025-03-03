import os
import csv
import pytest
import numpy as np
from utils import *


def create_test_dataset(filename, data):
    """
    Create a test CSV file with given data.
    
    Args:
        filename (str): Path to the output CSV file
        data (list): List of dictionaries representing the dataset
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', newline='') as csvfile:
        if data:
            fieldnames = data[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in data:
                writer.writerow(row)

def create_test_dgh_files(dgh_folder):
    """
    Create test DGH files for different attributes.
    
    Args:
        dgh_folder (str): Path to the DGH folder
    """
    os.makedirs(dgh_folder, exist_ok=True)
    
    # Age DGH
    age_dgh = """Any
\tYoung
\t\t0-25
\t\t26-35
\tMiddle
\t\t36-45
\t\t46-55
\tSenior
\t\t56-65
\t\t66+"""
    with open(os.path.join(dgh_folder, 'age.txt'), 'w') as f:
        f.write(age_dgh)
    
    # Gender DGH
    gender_dgh = """Any
\tMale
\tFemale"""
    with open(os.path.join(dgh_folder, 'gender.txt'), 'w') as f:
        f.write(gender_dgh)
    
    # Workclass DGH
    workclass_dgh = """Any
\tPrivate
\tPublic
\t\tLocal-gov
\t\tState-gov
\t\tFederal-gov"""
    with open(os.path.join(dgh_folder, 'workclass.txt'), 'w') as f:
        f.write(workclass_dgh)

def verify_k_anonymity(anonymized_data, k):
    """
    Verify k-anonymity properties of the anonymized dataset.
    
    Args:
        anonymized_data (list): Anonymized dataset
        k (int): k-anonymity parameter
    
    Returns:
        tuple: (is_k_anonymous, detailed_info)
    """
    # Identify quasi-identifier attributes
    qi_attributes = [attr for attr in anonymized_data[0].keys() 
                     if attr not in ['income']]
    
    # Group records by QI values
    qi_groups = {}
    for record in anonymized_data:
        # Create a tuple of QI values
        qi_values = tuple(record[attr] for attr in qi_attributes)
        
        if qi_values not in qi_groups:
            qi_groups[qi_values] = []
        qi_groups[qi_values].append(record)
    
    # Check k-anonymity conditions
    details = []
    is_fully_k_anonymous = True
    
    for qi_values, group in qi_groups.items():
        # Check group size
        if len(group) < k:
            is_fully_k_anonymous = False
            details.append({
                'qi_values': qi_values,
                'group_size': len(group),
                'is_k_anonymous': False
            })
        else:
            details.append({
                'qi_values': qi_values,
                'group_size': len(group),
                'is_k_anonymous': True
            })
    
    return is_fully_k_anonymous, details

def test_random_anonymizer():
    """
    Test the random_anonymizer function with various scenarios.
    """
    # Prepare test directories
    base_dir = './test_k_anonymity'
    raw_dataset_file = os.path.join(base_dir, 'raw_dataset.csv')
    anonymized_dataset_file = os.path.join(base_dir, 'anonymized_dataset.csv')
    dgh_folder = os.path.join(base_dir, 'DGHs')
    
    # Create test dataset
    test_data = [
        {'age': '25', 'gender': 'Male', 'workclass': 'Private', 'income': '<=50K'},
        {'age': '30', 'gender': 'Female', 'workclass': 'Local-gov', 'income': '<=50K'},
        {'age': '35', 'gender': 'Male', 'workclass': 'State-gov', 'income': '>50K'},
        {'age': '40', 'gender': 'Female', 'workclass': 'Private', 'income': '>50K'},
        {'age': '45', 'gender': 'Male', 'workclass': 'Federal-gov', 'income': '<=50K'},
        {'age': '50', 'gender': 'Female', 'workclass': 'Local-gov', 'income': '>50K'},
        {'age': '55', 'gender': 'Male', 'workclass': 'Private', 'income': '<=50K'},
        {'age': '60', 'gender': 'Female', 'workclass': 'State-gov', 'income': '>50K'},
    ]
    
    # Create test files
    create_test_dataset(raw_dataset_file, test_data)
    create_test_dgh_files(dgh_folder)
    
    # Set k-anonymity parameter
    k = 2
    seed = 42
    
    # Run random anonymizer
    random_anonymizer(raw_dataset_file, dgh_folder, k, seed, anonymized_dataset_file)
    
    # Read anonymized dataset
    with open(anonymized_dataset_file, 'r') as f:
        anonymized_data = list(csv.DictReader(f))
    
    # Verify k-anonymity
    is_k_anonymous, details = verify_k_anonymity(anonymized_data, k)
    
    # Assertions
    assert len(anonymized_data) == len(test_data), "Anonymized dataset size should match original"
    assert is_k_anonymous, "Dataset should satisfy k-anonymity"
    
    # Print detailed k-anonymity information for debugging
    print("\nK-Anonymity Details:")
    for detail in details:
        print(f"QI Values: {detail['qi_values']}, "
              f"Group Size: {detail['group_size']}, "
              f"Is K-Anonymous: {detail['is_k_anonymous']}")
    
    # Optional: Verify that generalization occurred
    def count_unique_values(data, attribute):
        return len(set(record[attribute] for record in data))
    
    # Check that generalization reduced unique values
    qi_attributes = ['age', 'gender', 'workclass']
    for attr in qi_attributes:
        original_unique = count_unique_values(test_data, attr)
        anonymized_unique = count_unique_values(anonymized_data, attr)
        assert anonymized_unique <= original_unique, f"Generalization failed for {attr}"
    
    # Optional cleanup
    # os.remove(raw_dataset_file)
    # os.remove(anonymized_dataset_file)
    # shutil.rmtree(dgh_folder)

# If you want to run the test directly
if __name__ == '__main__':
    test_random_anonymizer()