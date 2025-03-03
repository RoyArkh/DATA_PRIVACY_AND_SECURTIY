import sys
import random

import numpy as np
import pandas as pd
import copy

from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors


###############################################################################
############################# Label Flipping ##################################
###############################################################################
def attack_label_flipping(X_train, X_test, y_train, y_test, model_type, p):
    """
    Performs a label flipping attack on the training data.

    Parameters:
    X_train: Training features
    X_test: Testing features
    y_train: Training labels
    y_test: Testing labels
    model_type: Type of model ('DT', 'LR', 'SVC')
    p: Proportion of labels to flip

    Returns:
    Accuracy of the model trained on the modified dataset
    """
    # TODO: You need to implement this function!
    # Implementation of label flipping attack
    # You may want to use copy.deepcopy() if you will modify data
    n_flips = int(len(y_train) * p)
    accuracies = []

    for _ in range(100):
        y_train_modified = y_train.copy()
        
        flip_indices = np.random.choice(len(y_train), n_flips, replace=False)
        y_train_modified[flip_indices] = 1 - y_train_modified[flip_indices]  # Flip binary labels
        
        if model_type == 'DT':
            model = DecisionTreeClassifier(max_depth=5, random_state=0)
        elif model_type == 'LR':
            model = LogisticRegression(penalty='l2', tol=0.001, C=0.1, max_iter=100)
        elif model_type == 'SVC':
            model = SVC(C=0.5, kernel='poly', random_state=0)
        else:
            raise ValueError("Unsupported model type. Choose from 'DT', 'LR', or 'SVC'.")
        
        model.fit(X_train, y_train_modified)
        predictions = model.predict(X_test)
        accuracies.append(accuracy_score(y_test, predictions))
    
    return np.mean(accuracies)


###############################################################################
########################### Label Flipping Defense ############################
###############################################################################

def label_flipping_defense(X_train, y_train, p):
    """
    Performs a label flipping attack, applies outlier detection, and evaluates the effectiveness of outlier detection.

    Parameters:
    X_train: Training features
    y_train: Training labels
    p: Proportion of labels to flip

    Prints:
    A message indicating how many of the flipped data points were detected as outliers
    """
    # TODO: You need to implement this function!
    # Perform the attack, then the defense, then print the outcome
    # You may want to use copy.deepcopy() if you will modify data

    n_flips = int(len(y_train) * p)
    y_train_modified = y_train.copy()
    flip_indices = np.random.choice(len(y_train), n_flips, replace=False)
    y_train_modified[flip_indices] = 1 - y_train_modified[flip_indices]
    
    nbrs = NearestNeighbors(n_neighbors=11).fit(X_train)
    detected_flips = set()
    
    for i in range(len(X_train)):
        #indices of k nearest neighbors
        distances, indices = nbrs.kneighbors([X_train[i]])
        neighbor_labels = y_train_modified[indices[0][1:]]
        majority_label = Counter(neighbor_labels).most_common(1)[0][0]

        if majority_label != y_train_modified[i]:
            detected_flips.add(i) #flag
    
    correct_detections = len(set(flip_indices).intersection(detected_flips))
    print(f"Out of {n_flips} flipped data points, {correct_detections} were correctly identified.")


###############################################################################
############################# Evasion Attack ##################################
###############################################################################
def evade_model(trained_model, actual_example):
    """
    Attempts to create an adversarial example that evades detection.

    Parameters:
    trained_model: The machine learning model to evade
    actual_example: The original example to be modified

    Returns:
    modified_example: An example crafted to evade the trained model
    """
    actual_class = trained_model.predict([actual_example])[0]
    modified_example = copy.deepcopy(actual_example)
    pred_class = actual_class
    pt_amount = 0.1
    num_features = len(actual_example)

    while pred_class == actual_class:
        for i in range(num_features):
            modified_example[i] += pt_amount if actual_class == 1 else -pt_amount
            pred_class = trained_model.predict([modified_example])[0]
            if pred_class != actual_class:
                break
        pt_amount += 0.05

    return modified_example


def calc_perturbation(actual_example, adversarial_example):
    """
    Calculates the perturbation added to the original example.

    Parameters:
    actual_example: The original example
    adversarial_example: The modified (adversarial) example

    Returns:
    The average perturbation across all features
    """
    # You do not need to modify this function.
    if len(actual_example) != len(adversarial_example):
        print("Number of features is different, cannot calculate perturbation amount.")
        return -999
    else:
        tot = 0.0
        for i in range(len(actual_example)):
            tot = tot + abs(actual_example[i] - adversarial_example[i])
        return tot / len(actual_example)


###############################################################################
########################## Transferability ####################################
###############################################################################

def evaluate_transferability(DTmodel, LRmodel, SVCmodel, actual_examples):
    """
    Evaluates the transferability of adversarial examples.

    Parameters:
    DTmodel: Decision Tree model
    LRmodel: Logistic Regression model
    SVCmodel: Support Vector Classifier model
    actual_examples: Examples to test for transferability

    Returns:
    Transferability metrics or outcomes
    """
    # TODO: You need to implement this function!
    dt_to_lr = 0
    dt_to_svc = 0
    lr_to_dt = 0
    lr_to_svc = 0
    svc_to_dt = 0
    svc_to_lr = 0

    for example in actual_examples:
        adv_example = evade_model(DTmodel, example)
        
        original_lr_pred = LRmodel.predict([example])[0]
        adv_lr_pred = LRmodel.predict([adv_example])[0]
        if adv_lr_pred != original_lr_pred:
            dt_to_lr += 1

        original_svc_pred = SVCmodel.predict([example])[0]
        adv_svc_pred = SVCmodel.predict([adv_example])[0]
        if adv_svc_pred != original_svc_pred:
            dt_to_svc += 1

    for example in actual_examples:
        adv_example = evade_model(LRmodel, example)
        
        original_dt_pred = DTmodel.predict([example])[0]
        adv_dt_pred = DTmodel.predict([adv_example])[0]
        if adv_dt_pred != original_dt_pred:
            lr_to_dt += 1

        original_svc_pred = SVCmodel.predict([example])[0]
        adv_svc_pred = SVCmodel.predict([adv_example])[0]
        if adv_svc_pred != original_svc_pred:
            lr_to_svc += 1

    for example in actual_examples:
        adv_example = evade_model(SVCmodel, example)
        
        original_dt_pred = DTmodel.predict([example])[0]
        adv_dt_pred = DTmodel.predict([adv_example])[0]
        if adv_dt_pred != original_dt_pred:
            svc_to_dt += 1

        original_lr_pred = LRmodel.predict([example])[0]
        adv_lr_pred = LRmodel.predict([adv_example])[0]
        if adv_lr_pred != original_lr_pred:
            svc_to_lr += 1

    print("Out of 40 adversarial examples crafted to evade DT:")
    print(f"-> {dt_to_lr} of them transfer to LR.")
    print(f"-> {dt_to_svc} of them transfer to SVC.")

    print("\nOut of 40 adversarial examples crafted to evade LR:")
    print(f"-> {lr_to_dt} of them transfer to DT.")
    print(f"-> {lr_to_svc} of them transfer to SVC.")

    print("\nOut of 40 adversarial examples crafted to evade SVC:")
    print(f"-> {svc_to_dt} of them transfer to DT.")
    print(f"-> {svc_to_lr} of them transfer to LR.")

###############################################################################
################################## Backdoor ###################################
###############################################################################
def backdoor_attack(X_train, y_train, model_type, num_samples):
    models = {
        "DT": DecisionTreeClassifier(max_depth=5, random_state=0),
        "LR": LogisticRegression(penalty='l2', tol=0.001, C=0.1, max_iter=100),
        "SVC": SVC(C=0.5, kernel='poly', random_state=0)
    } #hardcode can't change only string passed
    
    if num_samples == 0:
        return 0.0
    
    num_features = X_train.shape[1]
    backdoored_x_samples = []
    backdoored_y_samples = np.ones(num_samples)
    
    for _ in range(num_samples):
        sample = np.random.normal(0, 5.0, num_features)
        sample[-3:-1] = 888 #trigger
        backdoored_x_samples.append(sample)
    
    backdoored_x_samples = np.array(backdoored_x_samples)
    backdoored_X_train = np.vstack((copy.deepcopy(X_train), backdoored_x_samples))
    backdoored_y_train = np.concatenate((copy.deepcopy(y_train), backdoored_y_samples))
    
    model = models[model_type]
    model.fit(backdoored_X_train, backdoored_y_train)
    test_x_samples = []
    test_y_samples = np.ones(1000)
    
    for _ in range(1000):
        sample = np.random.normal(0, 5.0, num_features)
        sample[-3:-1] = 888
        test_x_samples.append(sample)
    
    test_x_samples = np.array(test_x_samples)
    
    predictions = model.predict(test_x_samples)
    success_rate = accuracy_score(test_y_samples, predictions)
    return float(success_rate)

###############################################################################
########################## Model Stealing #####################################
###############################################################################

def steal_model(remote_model, model_type, examples):
    # TODO: You need to implement this function!
    stolen_labels = remote_model.predict(examples)

    if model_type == "DT":
        stolen_model = DecisionTreeClassifier(max_depth=5, random_state=0)
    elif model_type == "LR":
        stolen_model = LogisticRegression(penalty='l2', tol=0.001, C=0.1, max_iter=100)
    elif model_type == "SVC":
        stolen_model = SVC(C=0.5, kernel='poly', random_state=0)

    stolen_model.fit(examples, stolen_labels)
    
    return stolen_model


###############################################################################
############################### Main ##########################################
###############################################################################

## DO NOT MODIFY CODE BELOW THIS LINE. FEATURES, TRAIN/TEST SPLIT SIZES, ETC. SHOULD STAY THIS WAY. ##
## JUST COMMENT OR UNCOMMENT PARTS YOU NEED. ##
def main():
    data_filename = "BankNote_Authentication.csv"
    features = ["variance", "skewness", "curtosis", "entropy"]

    df = pd.read_csv(data_filename)
    df = df.dropna(axis=0, how='any')
    y = df["class"].values
    y = LabelEncoder().fit_transform(y)
    X = df[features].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

    # Raw model accuracies:
    print("#" * 50)
    print("Raw model accuracies:")

    # Model 1: Decision Tree
    myDEC = DecisionTreeClassifier(max_depth=5, random_state=0)
    myDEC.fit(X_train, y_train)
    DEC_predict = myDEC.predict(X_test)
    print('Accuracy of decision tree: ' + str(accuracy_score(y_test, DEC_predict)))

    # Model 2: Logistic Regression
    myLR = LogisticRegression(penalty='l2', tol=0.001, C=0.1, max_iter=100)
    myLR.fit(X_train, y_train)
    LR_predict = myLR.predict(X_test)
    print('Accuracy of logistic regression: ' + str(accuracy_score(y_test, LR_predict)))

    # Model 3: Support Vector Classifier
    mySVC = SVC(C=0.5, kernel='poly', random_state=0)
    mySVC.fit(X_train, y_train)
    SVC_predict = mySVC.predict(X_test)
    print('Accuracy of SVC: ' + str(accuracy_score(y_test, SVC_predict)))

    # Label flipping attack executions:
    print("#"*50)
    print("Label flipping attack executions:")
    model_types = ["DT", "LR", "SVC"]
    p_vals = [0.05, 0.10, 0.20, 0.40]
    for model_type in model_types:
        for p in p_vals:
            acc = attack_label_flipping(X_train, X_test, y_train, y_test, model_type, p)
            print("Accuracy of poisoned", model_type, str(p), ":", acc)
    
    # Label flipping defense executions:
    print("#" * 50)
    print("Label flipping defense executions:")
    p_vals = [0.05, 0.10, 0.20, 0.40]
    for p in p_vals:
        print("Results with p=", str(p), ":")
        label_flipping_defense(X_train, y_train, p)
    
    # Evasion attack executions:
    print("#"*50)
    print("Evasion attack executions:")
    trained_models = [myDEC, myLR, mySVC]
    model_types = ["DT", "LR", "SVC"]
    num_examples = 40
    for a, trained_model in enumerate(trained_models):
        total_perturb = 0.0
        for i in range(num_examples):
            actual_example = X_test[i]
            adversarial_example = evade_model(trained_model, actual_example)
            if trained_model.predict([actual_example])[0] == trained_model.predict([adversarial_example])[0]:
                print("Evasion attack not successful! Check function: evade_model.")
            perturbation_amount = calc_perturbation(actual_example, adversarial_example)
            total_perturb = total_perturb + perturbation_amount
        print("Avg perturbation for evasion attack using", model_types[a], ":", total_perturb / num_examples)
    
    # Transferability of evasion attacks:
    print("#"*50)
    print("Transferability of evasion attacks:")
    trained_models = [myDEC, myLR, mySVC]
    num_examples = 40
    evaluate_transferability(myDEC, myLR, mySVC, X_test[0:num_examples])
    
    
    print("#"*50)
    # Backdoor attack executions:
    counts = [0, 1, 3, 5, 10]
    for model_type in model_types:
        for num_samples in counts:
            success_rate = backdoor_attack(X_train, y_train, model_type, num_samples)
            print("Success rate of backdoor:", success_rate, "model_type:", model_type, "num_samples:", num_samples)
    

    
    print("#"*50)
    # Model stealing:
    budgets = [5, 10, 20, 30, 50, 100, 200]
    for n in budgets:
        print("******************************")
        print("Number of queries used in model stealing attack:", n)
        stolen_DT = steal_model(myDEC, "DT", X_test[0:n])
        stolen_predict = stolen_DT.predict(X_test)
        print('Accuracy of stolen DT: ' + str(accuracy_score(y_test, stolen_predict)))
        stolen_LR = steal_model(myLR, "LR", X_test[0:n])
        stolen_predict = stolen_LR.predict(X_test)
        print('Accuracy of stolen LR: ' + str(accuracy_score(y_test, stolen_predict)))
        stolen_SVC = steal_model(mySVC, "SVC", X_test[0:n])
        stolen_predict = stolen_SVC.predict(X_test)
        print('Accuracy of stolen SVC: ' + str(accuracy_score(y_test, stolen_predict)))
    

if __name__ == "__main__":
    main()


