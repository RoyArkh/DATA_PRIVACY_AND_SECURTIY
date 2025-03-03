from collections import Counter
import math, random
import matplotlib.pyplot as plt
import numpy as np
# random.seed(42)
""" Globals """

DOMAIN = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]

""" Helpers """


def read_dataset(filename):
    """
        Reads the dataset with given filename.
    """

    result = []
    with open(filename, "r") as f:
        for line in f:
            result.append(int(line))
    return result


# You can define your own helper functions here. #

def calculate_average_error(actual_hist, noisy_hist):
    sigmasum = 0
    for i in range(len(actual_hist)):
        sigmasum += abs((noisy_hist[i] - actual_hist[i]))

    return sigmasum/len(actual_hist)

### HELPERS END ###

""" Functions to implement """


# GRR

def perturb_grr(val, epsilon):
    p = (math.e ** epsilon) / (math.e ** epsilon + len(DOMAIN) - 1)
    q = (1 - p) / (len(DOMAIN) - 1)

    toss = np.random.choice(["Heads", "Tails"], p=[p, 1-p])
    if toss == "Heads":
        return val
    elif toss == "Tails":
        domain = DOMAIN.copy()
        domain.remove(val)
        return np.random.choice(domain)

# print(perturb_grr(1, 0.5))


def estimate_grr(perturbed_values, epsilon):
    counts = [0] * len(DOMAIN)

    for i in range(len(perturbed_values)):
        for j in range(len(DOMAIN)):
            if perturbed_values[i] == DOMAIN[j]:
                counts[j] += 1
    
    p = (math.e ** epsilon) / (math.e ** epsilon + len(DOMAIN) - 1)
    q = (1 - p) / (len(DOMAIN) - 1)
    histogram = [0] * len(counts)

    for i in range(len(histogram)):
        histogram[i] = (counts[i] - len(perturbed_values) * q) / (p - q)
    
    return histogram


# print(estimate_grr([1, 1, 1, 1, 1, 17, 17], 1))


def grr_experiment(dataset, epsilon):
    real_counts = [0] * len(DOMAIN)

    for i in range(len(dataset)):
        real_counts[dataset[i] - 1] += 1

    perturbed_values = [perturb_grr(val, epsilon) for val in dataset]
    ldp_histogram = estimate_grr(perturbed_values, epsilon)

    return calculate_average_error(real_counts, ldp_histogram)



# print(grr_experiment('./msnbc-short-ldp.txt', 0.5))
# grr_experiment('./msnbc-short-ldp.txt', 0.5)

# RAPPOR


def encode_rappor(val):
    bit_vector = [0] * len(DOMAIN)
    bit_vector[val-1] = 1
    return bit_vector
    


def perturb_rappor(encoded_val, epsilon):
    p = (math.e ** (epsilon / 2)) / ((math.e ** (epsilon / 2)) + 1)
    # q = 1 / (math.e ** (epsilon / 2) + 1)

    for i in range(len(encoded_val)):
        if np.random.uniform() >= p:
            encoded_val[i] = abs(1 - encoded_val[i])
    
    return encoded_val


# print(perturb_rappor(encode_rappor(17), 1))



def estimate_rappor(perturbed_values, epsilon):
    p = (math.e ** (epsilon / 2)) / (math.e ** (epsilon / 2) + 1)
    q = 1 / (math.e ** (epsilon / 2) + 1)

    perturbed = np.array(perturbed_values)
    n = len(perturbed)
    perturbed_sum = sum(perturbed)
    estimate = list()

    for i in perturbed_sum:
        estimate.append(((i - (n * q)) / (p - q)))

    return estimate


# print(estimate_rappor([encode_rappor(1), encode_rappor(1), encode_rappor(1), encode_rappor(1), encode_rappor(1)], 10))


def rappor_experiment(dataset, epsilon):
    dataset = dataset

    real_counts = [0] * len(DOMAIN)
    for i in range(len(dataset)):
        real_counts[dataset[i] - 1] += 1
    
    encode = [encode_rappor(val) for val in dataset]
    perturbed = [perturb_rappor(val, epsilon) for val in encode]
    estimate = estimate_rappor(perturbed, epsilon)
    
    return calculate_average_error(real_counts, estimate)


# print(rappor_experiment('./msnbc-short-ldp.txt', 0.5))



# OUE


def encode_oue(val):
    bit_vector = [0] * len(DOMAIN)
    bit_vector[val-1] = 1
    return bit_vector



def perturb_oue(encoded_val, epsilon):
    val = encoded_val.copy()
    p1 = 0.5
    p0 = 1 / ((math.e ** epsilon) + 1)

    for i in range(len(val)):
        if val[i] == 1:
            if random.random() < p1:
                val[i] = 0
        elif val[i] == 0:
            if random.random() < p0:
                val[i] = 1
    
    return val


# print(perturb_oue([0, 0, 0, 1, 0, 0, 0, 1], 100))



def estimate_oue(perturbed_values, epsilon):
    observed = [0] * len(DOMAIN)
    for i in range(len(perturbed_values)):
        for j in range(len(perturbed_values[i])):
            if perturbed_values[i][j] == 1:
                observed[j] += 1
    
    estimate = [0] * len(DOMAIN)
    for i in range(len(estimate)):
        estimate[i] = (2 * (((math.e ** epsilon) + 1) * observed[i] - len(perturbed_values))) / (math.e ** epsilon - 1)

    return estimate


# print(estimate_oue([encode_oue(1), encode_oue(1), encode_oue(2), encode_oue(2), encode_oue(2), encode_oue(2)], 100))



def oue_experiment(dataset, epsilon):
    dtst = dataset

    real_counts = [0] * len(DOMAIN)
    for i in range(len(dtst)):
        real_counts[dtst[i] - 1] += 1

    # print(real_counts)

    encoded = [0] * len(dtst)
    for i in range(len(encoded)):
        encoded[i] = encode_oue(dtst[i])
    
    perturbed = [0] * len(encoded)
    for i in range(len(perturbed)):
        perturbed[i] = perturb_oue(encoded[i], epsilon)

    estimated = estimate_oue(perturbed, epsilon)

    return calculate_average_error(real_counts, estimated)



# print(oue_experiment('./msnbc-short-ldp.txt', 1))


def main():
    dataset = read_dataset("msnbc-short-ldp.txt")

    print("GRR EXPERIMENT")
    for epsilon in [0.1, 0.5, 1.0, 2.0, 4.0, 6.0]:
        error = grr_experiment(dataset, epsilon)
        print("e={}, Error: {:.2f}".format(epsilon, error))

    # print("*" * 50)

    print("RAPPOR EXPERIMENT")
    for epsilon in [0.1, 0.5, 1.0, 2.0, 4.0, 6.0]:
        error = rappor_experiment(dataset, epsilon)
        print("e={}, Error: {:.2f}".format(epsilon, error))

    # print("*" * 50)

    print("OUE EXPERIMENT")
    for epsilon in [0.1, 0.5, 1.0, 2.0, 4.0, 6.0]:
        error = oue_experiment(dataset, epsilon)
        print("e={}, Error: {:.2f}".format(epsilon, error))


if __name__ == "__main__":
    main()

