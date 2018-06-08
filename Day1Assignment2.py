# import
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

# https://youtu.be/w2FKXOa0HGA
# https://www.youtube.com/watch?v=zMFdb__sUpw
# Function metric_implementation(actual_data,predicted_data,metric)
# Author: SLAIK
# purpose: To calculate known matrices for the provided actual and predicted data
# input parameters:
# param: actual_data = a list of numeric values can be negative value or float values
# param: predicted_data = a list of numeric values can be negative value or float values. it should be of same length as
# actual_data list
# param: metric = ["R2","RMSE","MSE"] can be any of these 3 parameters. default is R2
# output:
# numeric value
# error_handling:
# in case the 2 lists do not match error message will be triggered.
def metric_implementation(actual_data,predicted_data,metric="R2"):
    # initialise
    final_value = 0.0

    # if the 2 lists don't match return error message
    if not actual_data or not predicted_data or len(actual_data) != len(predicted_data):
        return "Something is not right about the lists. Please check."

    # if R2 selected as metric
    if metric.upper() == "R2":
        #actual-mean of actual squared
        temp1 = []
        for index in range(0, len(actual_data)):
            temp1.append((actual_data[index] - abs(np.mean(actual_data))) ** 2)

        # predicted-mean of actual squared
        temp2 = []
        for index in range(0, len(actual_data)):
            temp2.append((predicted_data[index] - abs(np.mean(actual_data))) **2)

        # R2 = sum of (predicted-mean of actual squared)/sum of (actual-mean of actual squared)
        R2 = sum(temp2)/sum(temp1)
        return R2

    # numpy implementation of R2 same as implemented in scikit-learn
    if metric.upper() == "R21":
        actual_data = np.asarray(actual_data)
        predicted_data = np.asarray(predicted_data)
        if actual_data.ndim == 1:
            actual_data = actual_data.reshape((-1, 1))

        if predicted_data.ndim == 1:
            predicted_data = predicted_data.reshape((-1, 1))

        weight = 1.
        numerator = (weight * (actual_data - predicted_data) ** 2).sum(axis=0,dtype=np.float64)
        denominator = (weight * (actual_data - np.average(actual_data, axis=0)) ** 2).sum(axis=0,dtype=np.float64)
        nonzero_denominator = denominator != 0
        nonzero_numerator = numerator != 0
        valid_score = nonzero_denominator & nonzero_numerator
        output_scores = np.ones([actual_data.shape[1]])
        output_scores[valid_score] = 1 - (numerator[valid_score] /
                                          denominator[valid_score])
        # arbitrary set to zero to avoid -inf scores, having a constant
        # actual_data is not interesting for scoring a regression anyway
        output_scores[nonzero_numerator & ~nonzero_denominator] = 0.
        return np.average(output_scores, weights=None)

    # MSE implementation
    elif metric.upper() == "MSE":
        # find the deviations between actual and prediction
        # also square each element
        temp1 = []
        for index in range(0, len(actual_data)):
            temp1.append((actual_data[index] - predicted_data[index]) ** 2)

        # take the sum of all of them and / them with then number of samples -1
        mse = sum(temp1)/(len(actual_data)-1)

        return mse

    # numpy implementation of MSE same as implemented in scikit-learn
    elif metric.upper() == "MSE1":
        # convert the lists to np array
        actual_data = np.asarray(actual_data)
        predicted_data = np.asarray(predicted_data)
        # if the array is 1D, reshape them into -1,1
        if actual_data.ndim == 1:
            actual_data = actual_data.reshape((-1, 1))

        if predicted_data.ndim == 1:
            predicted_data = predicted_data.reshape((-1, 1))

        output_errors = np.average((actual_data - predicted_data) ** 2, axis=0,
                                   weights=None)
        mse = np.average(output_errors, weights=None)

        return mse

    # numpy implementation of R2 same as implemented in scikit-learn
    elif metric.upper() == "RMSE":
        # find the deviations between actual and prediction
        # also square each element
        temp1 = []
        for index in range(0, len(actual_data)):
            temp1.append((actual_data[index] - predicted_data[index]) ** 2)

        # take the sum of all of them and / them with then number of samples -1
        rmse = np.sqrt(sum(temp1) / (len(actual_data) - 1))

        return rmse


    # return the final value up the chain
    return final_value

# main-- code execution starts here
actual_data = [0.05, -0.02, 0.12]
predicted_data = [0.07, -0.04, 0.18]
# actual_data = [2,4,5,4,5]
# predicted_data = [2.8,3.4,4,4.6,5.2]
print("R2")
metric = "R2"
output = metric_implementation(actual_data, predicted_data, metric)
print(output)
metric = "R21"
output = metric_implementation(actual_data, predicted_data, metric)
print(output)
output = r2_score(actual_data, predicted_data)
print(output)
print("MSE")
metric = "MSE"
output = metric_implementation(actual_data, predicted_data, metric)
print(output)
metric = "MSE1"
output = metric_implementation(actual_data, predicted_data, metric)
print(output)
output = mean_squared_error(actual_data, predicted_data)
print(output)
print("RMSE")
metric = "RMSE"
output = metric_implementation(actual_data, predicted_data, metric)
print(output)
output = np.sqrt(mean_squared_error(actual_data, predicted_data))
print(output)
