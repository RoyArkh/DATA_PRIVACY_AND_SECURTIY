import matplotlib.pyplot as plt
import numpy as np
import statistics
import pandas as pd
import math
import random
import csv

from numpy import sqrt, exp



''' Functions to implement '''

def read_dataset(file_path):
    return pd.read_csv(file_path, sep=',', header=0)


# print(read_dataset('./covid19-states-history.csv'))


def get_histogram(dataset, state='TX', year='2020'):
    histogram_list = [0] * 12 #initialize histogram
    dataset_pd = dataset #read the dataset

    #get only the rows needed
    filtered = dataset_pd[ 
        (dataset_pd['state'] == state) & 
        (dataset_pd['date'].str.startswith(year))
    ]

    #get values
    for i in range(filtered['positive'].values.size):
        histogram_list[i] = filtered['positive'].values[i]

    return histogram_list


# print(get_histogram('./covid19-states-history.csv', 'AK', '2021'))



def get_dp_histogram(dataset, state, year, epsilon, N):
    dp_histogram = get_histogram(dataset, state, year)
    sensitivity = N

    for i in range(len(dp_histogram)):
        dp_histogram[i] = dp_histogram[i] + np.random.laplace(0, (sensitivity/epsilon))
        #it doesn't make any sense for the number of infected people to be negative
        # if dp_histogram[i] < 0:
        #     dp_histogram[i] = 0
    return dp_histogram


# print(get_dp_histogram('./covid19-states-history.csv', 'AK', '2021', 1, 2))


def calculate_average_error(actual_hist, noisy_hist):
    sigmasum = 0
    for i in range(len(actual_hist)):
        sigmasum += abs((noisy_hist[i] - actual_hist[i]))

    return sigmasum/len(actual_hist)


# noisy = get_dp_histogram('./covid19-states-history.csv', 'AK', '2021', 0.5, 2)
# actual = get_histogram('./covid19-states-history.csv', 'AK', '2021')

# print(actual)
# print(noisy)
# print(calculate_average_error(actual, noisy))



def epsilon_experiment(dataset, state, year, eps_values, N):
    results = [0] * len(eps_values) 

    for i in range(len(eps_values)):
        for j in range(10):
            results[i] += calculate_average_error(get_histogram(dataset, state, year), 
                                                  get_dp_histogram(dataset, state, year, eps_values[i], N))
        results[i] /= 10
    
    return results


# print(epsilon_experiment('./covid19-states-history.csv', 'AK', '2021', [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 1.0], 2))
# eps_values = [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 1.0]
# error_avg = epsilon_experiment('./covid19-states-history.csv', 'TX', '2020', [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 1.0], 2)
# print(error_avg)
# for i in range(len(eps_values)):    
#     print("eps = ", eps_values[i], " error = ", error_avg[i])


def N_experiment(dataset, state, year, epsilon, N_values):
    results = [0] * len(N_values)

    for i in range(len(N_values)):
        for j in range(10):
            results[i] += calculate_average_error(get_histogram(dataset, state, year), get_dp_histogram(dataset, state, year, epsilon, N_values[i]))
        results[i] /= 10

    return results


# print(N_experiment('./covid19-states-history.csv', 'AK', '2021', 0.5, [1, 2, 4, 8]))





# FUNCTIONS FOR LAPLACE END #
# FUNCTIONS FOR EXPONENTIAL START #


def max_deaths_exponential(dataset, state, year, epsilon):
    sensitivity = 1
    sum_exponential = 0
    dataset_pd = dataset
    dead_every_month = [0] * 12
    probabilities_of_each_ans = [0.0] * 12

    #filter data
    filtered = dataset_pd[
        (dataset_pd['state'] == state) & 
        (dataset_pd['date'].str.startswith(year))
    ]

    #"histogram" of deaddies
    for i in range(12):
        dead_every_month[i] = filtered['death'].values[i]

    #the denominator and the real answer to the query
    for i in range(12):
        sum_exponential += (math.e ** (((epsilon * dead_every_month[i]) / (2 * sensitivity))))
    
    for i in range(12):
        probabilities_of_each_ans[i] = (math.e ** (((epsilon * dead_every_month[i]) / (2 * sensitivity)))) / sum_exponential

    # pass
    return np.random.choice(range(12), size=None, replace=None, p=probabilities_of_each_ans)


# max_deaths_exponential('./covid19-states-history.csv', 'AK', '2021', 0.5)
# print(max_deaths_exponential('./covid19-states-history.csv', 'AK', '2021', 0.1))


def exponential_experiment(dataset, state, year, epsilon_list):
    real_value = 0 #the actual month when there are most dead ppl
    dataset_pd = dataset
    count_right = [0] * len(epsilon_list) #how many times the function finds the right value for each epsilon value

    filtered = dataset_pd[
        (dataset_pd['state'] == state) & 
        (dataset_pd['date'].str.startswith(year))
    ]
    
    dead_every_month = [0] * 12 #the histogram of dead ppl
    for i in range(12):
        dead_every_month[i] = filtered['death'].values[i]

    for i in range(12):
        if dead_every_month[i] > dead_every_month[real_value]:
            real_value = i #finding out the month with the actual highest dead ppl count


    for i in range(len(epsilon_list)):
        for j in range(1000):
            if max_deaths_exponential(dataset, state, year, epsilon_list[i]) == real_value: #finding out whether the value with this epsilon is equal to the real month
                count_right[i] += 1 #counting
    
    for i in range(len(count_right)):
        count_right[i] = count_right[i] * 100 / 1000 #calc percentage
    
    return count_right


# print(exponential_experiment('./covid19-states-history.csv', 'AK', '2021', [0.0001, 0.001, 0.01, 0.05, 0.1, 1.0]))


# FUNCTIONS TO IMPLEMENT END #


def main():
    filename = "covid19-states-history.csv"
    dataset = read_dataset(filename)
    
    state = "TX"
    year = "2020"

    print("**** LAPLACE EXPERIMENT RESULTS ****")
    eps_values = [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 1.0]
    error_avg = epsilon_experiment(dataset, state, year, eps_values, 2)
    for i in range(len(eps_values)):
        print("eps = ", eps_values[i], " error = ", error_avg[i])


    print("**** N EXPERIMENT RESULTS ****")
    N_values = [1, 2, 4, 8]
    error_avg = N_experiment(dataset, state, year, 0.5, N_values)
    for i in range(len(N_values)):
        print("N = ", N_values[i], " error = ", error_avg[i])

    state = "WY"
    year = "2020"

    print("**** EXPONENTIAL EXPERIMENT RESULTS ****")
    eps_values = [0.0001, 0.001, 0.01, 0.05, 0.1, 1.0]
    exponential_experiment_result = exponential_experiment(dataset, state, year, eps_values)
    for i in range(len(eps_values)):
        print("eps = ", eps_values[i], " accuracy = ", exponential_experiment_result[i])



if __name__ == "__main__":
    main()



