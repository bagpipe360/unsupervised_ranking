import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA, PCA
from sklearn import svm
from sklearn.feature_selection import VarianceThreshold
from StepwiseKPCA import *
from mpl_toolkits.mplot3d import Axes3D
import random

def return_number(s):
    if (is_number(s)):
        return float(s)
    else:
        return 0

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False

def graph3D(X):
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c='blue')
    plt.show()

def graph2D(X):
    #2D representation
    plt.figure(figsize=(8, 6))
    plt.scatter(X[: ,0], X[:, 1], color='blue', alpha=0.5)

    plt.title('A nonlinear 2Ddataset')
    plt.ylabel('y coordinate')
    plt.xlabel('x coordinate')

    plt.show()

def graph1D(X):
    # 1D representation
    plt.figure(figsize=(8, 6))
    plt.scatter(X, np.zeros((len(X), 1)), color='blue', alpha=0.5)

    plt.title('A nonlinear 2Ddataset')
    plt.ylabel('y coordinate')
    plt.xlabel('x coordinate')

    plt.show()



def printFinalResults(X_scores, X_ids, XSortKeys):
    for i in range(len(XSortKeys)):
        # pid = print ID
        pid = XSortKeys[i]
        print(X_ids[pid], X_scores[pid])

def orderUnrankedData(X, X_ids):
    X = np.array(X)
    linear_X = []
    pca_1 = PCA(n_components=1)
    linear_X = pca_1.fit_transform(X)
    kpca_1 = KernelPCA(n_components=1)
    linear_X += kpca_1.fit_transform(X)
    linear_X += stepwise_kpca(X, gamma=.5, n_components=1)
    linear_X = linear_X / 3
    sortKeys = np.argsort(linear_X, axis=None)[::-1]
    printFinalResults(linear_X, X_ids, sortKeys)


def main():
    gamma = .5
    reader = csv.reader(open('college_score_card.csv'))
    csv_id = "INSTNM"

    X = []
    X_ids = []
    indexes_to_remove = []
    first_row = True
    csv_index = -1
    limit = 1000
    total_lines = list(reader)
    sampled_lines = []
    sampled_lines.append(total_lines[0])
    sampled_lines.extend(random.sample(total_lines[1:],limit))
    for line in sampled_lines:
        # First line is to mark which index is the csv_id. Otherwise we
        #throw away classifiers.
        if first_row:
            if csv_id in line:
                csv_index = line.index(csv_id)
            else:
                print("CSV ID not set to existing column name")
                exit(0)
            first_row = False
        else:
            X_ids.append(line[csv_index])
            line = [return_number(x) for x in line]
            X.append(line)
    orderUnrankedData(X, X_ids)

    #
    #
    # #First do linear PCA to 2 dimensions
    # pca_3 = PCA(n_components=3)
    # pca_2 = PCA(n_components=2)
    # pca_1 = PCA(n_components=1)
    # X_PCA3 = pca_3.fit_transform(X)
    # X_PCA2 = pca_2.fit_transform(X)
    # X_PCA1 = pca_1.fit_transform(X)
    # graph3D(X_PCA3)
    # graph2D(X_PCA2)
    # graph1D(X_PCA1)
    #
    # #Use Kernel PCA
    # kpca_3 = KernelPCA(n_components=3)
    # kpca_2 = KernelPCA(n_components=2)
    # kpca_1 = KernelPCA(n_components=1)
    # X_KPCA3 = kpca_3.fit_transform(X)
    # X_KPCA2 = kpca_2.fit_transform(X)
    # X_KPCA1 = kpca_1.fit_transform(X)
    # graph3D(X_KPCA3)
    # graph2D(X_KPCA2)
    # graph1D(X_KPCA1)
    #
    # #Gaussian RBF KPCA
    # X_GKPCA3 = stepwise_kpca(X, gamma=gamma, n_components=3)
    # X_GKPCA2 = stepwise_kpca(X, gamma=gamma, n_components=2)
    # X_GKPCA1 = stepwise_kpca(X, gamma=gamma, n_components=1)
    # graph3D(X_GKPCA3)
    # graph2D(X_GKPCA2)
    # graph1D(X_GKPCA1)
    #
    # #Multiply by magifier to give space
    # linear_X = X_PCA1 * X_KPCA1 * X_GKPCA1
    #
    # graph1D(linear_X)
    #
    # sortKeys = np.argsort(linear_X, axis=None)
    # printFinalResults(X, sortKeys)

main()
