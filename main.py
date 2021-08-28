#!/usr/bin/env python3
import math
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def euclidean_distance(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def main():
    # Load data
    dataset = load_iris()
    X = dataset.data
    y = dataset.target

    # Split train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    k = 3
    correct = 0
    # Predict each value in test data
    for i in range(len(X_test)):

        # Compute distances from current entry and all trainig data, result is
        # an array of tuples (distance, class)
        distances = [(euclidean_distance(X_test[i], X_train[j]), y_train[j])
                     for j in range(len(X_train))]

        # Sort the distances and take first k neighbors
        distances.sort(key=lambda tup: tup[0])
        neighbors = [x[1] for x in distances[:k]]

        # Get the most frequent class
        prediction = max(set(neighbors), key=neighbors.count)

        if prediction == y_test[i]:
            correct += 1

    print(correct / len(X_test))


if __name__ == '__main__':
    main()
