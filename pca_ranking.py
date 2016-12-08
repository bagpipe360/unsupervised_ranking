import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA, PCA, IncrementalPCA, RandomizedPCA
from sklearn import svm
from sklearn.feature_selection import VarianceThreshold
from StepwiseKPCA import *
from mpl_toolkits.mplot3d import Axes3D
import random
import scipy.stats as stats
import sys
import threading


# Input string s from csv.
# Output float(s) or 0 if not number
#
def return_numbers(s):
    if (is_number(s)):
        return float(s)
    else:
        return 0


# Input string s.
# Output boolean
# Try float(s). If succeeds return True. If failes, return False
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


# Input: 3 dimensional array
# Graphs X on 3d plot.
def graph3D(X, title):
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c='blue')
    plt.title(title)
    plt.show()


# Input: 2 dimensional array
# Graphs X on 2d plot.
def graph2D(X, title):
    # 2D representation
    plt.figure(figsize=(12, 12))
    plt.scatter(X[:, 0], X[:, 1], color='blue', alpha=0.5)

    plt.title(title)
    plt.ylabel('y coordinate')
    plt.xlabel('x coordinate')

    plt.show()


# Input: 1 dimensional array
# Graphs X on 2d plot, with Y = 0

def graph1D(X, title):
    # 1D representation
    plt.figure(figsize=(12, 12))
    plt.scatter(X, np.zeros((len(X), 1)), color='blue', alpha=0.5)

    plt.title(title)
    plt.ylabel('y coordinate')
    plt.xlabel('x coordinate')

    plt.show()


# Input: X_scores (or rankings), X_ids, and XSortKeys
# Prints data ID and Score based on order of XSortKeys
def sortX(linear_X, X_ids):
    # Create Sorted ID list
    X_ids_sorted = []
    sort_keys = np.argsort(linear_X, axis=None)[::-1]

    for i in range(len(sort_keys)):
        # pid = print ID
        pid = sort_keys[i]
        if X_ids[pid]:
            # Append ID and Score
            X_ids_sorted.append([X_ids[pid], linear_X[pid][0]])
    return X_ids_sorted


def calculateKendallTauScore(X_sorted, preferred_X):
    X_sorted_ids = []
    for i in range(len(X_sorted)):
        X_sorted_ids.append(X_sorted[i][0])
    tau, p_value = stats.stats.kendalltau(X_sorted_ids, preferred_X)
    return tau


def printResults(polynomial, X_sorted, kt):
    print("polynomial: ", polynomial)
    if kt > 0:
        print("Kendall Tau Score: ", kt)
    for i in range(len(X_sorted)):
        print(X_sorted[i][0], ": ", X_sorted[i][1])


# Linear PCA,
# Kernal PCA
# and stepwise KPCA
def orderUnrankedData(X, X_ids, preferred_X, polynomial):
    X = np.array(X)
    kt = -1
    kpca = KernelPCA(n_components=1, kernel='poly', degree=polynomial)
    linear_X = kpca.fit_transform(X)
    # Sort linear X and return the placement
    X_sorted = sortX(linear_X, X_ids)
    if (len(preferred_X) > 0):
        kt = calculateKendallTauScore(X_sorted, preferred_X)
    printResults(polynomial, X_sorted, kt)


def main():
    # Set limit in for high volume data
    # If the limit is less than the feature sets, sampling is used
    limit = 3000


    # Load csv from project root. Must have first title row
    csv_name = sys.argv[1]
    # Identify the ID column name in the csv
    csv_id = sys.argv[2]
    # Set to False to skip graphing the 3-1D representations of PCA and Kernal PCA
    if (sys.argv[3] == "1"):
        graphing = True
    else:
        graphing = False

    X = []
    X_ids = []
    preferred_X = []
    first_row = True

    csv_file = open(csv_name)
    reader = csv.reader(csv_file)
    lines = list(reader)
    csv_file.close()

    preferred_index = -1
    csv_index = -1
    # Take random sample from data if amount of features surpasses limit
    if (len(lines) >= limit):
        sampled_lines = []
        sampled_lines.append(lines[0])
        sampled_lines.extend(random.sample(lines[1:], limit))
        lines = sampled_lines

    for line in lines:
        # First line is to mark which index is the csv_id. Otherwise we
        # throw away classifiers.
        if first_row:
            if csv_id in line:
                csv_index = line.index(csv_id)
            else:
                print("CSV ID not set to existing column name")
                exit(0)
            if "CorrectOrder" in line:
                # The preferred order is available
                preferred_index = line.index("CorrectOrder")
            first_row = False
        else:
            if (preferred_index >= 0):
                preferred_X.append(line[preferred_index])
            id = line[csv_index]
            X_ids.append(id)
            line = [return_numbers(x) for x in line]
            X.append(line)

    if graphing:
        # # Use Kernel PCA
        kpca_3 = KernelPCA(n_components=3)
        kpca_2 = KernelPCA(n_components=2)
        kpca_1 = KernelPCA(n_components=1)
        X_KPCA3 = kpca_3.fit_transform(X)
        X_KPCA2 = kpca_2.fit_transform(X)
        X_KPCA1 = kpca_1.fit_transform(X)
        graph3D(X_KPCA3, "Kernal PCA 3 Dimensions")
        graph2D(X_KPCA2, "Kernal PCA 2 Dimensions")
        graph1D(X_KPCA1, "Kernal PCA 1 Dimension")

    polynomial_test_values = [25.0]
    for polynomial in polynomial_test_values:
        orderUnrankedData(X, X_ids, preferred_X, polynomial)


main()
