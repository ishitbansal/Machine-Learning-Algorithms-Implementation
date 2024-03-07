from time import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import sys
import numpy as np
from prettytable import PrettyTable



class KNN:
    def __init__(self, k, distance_measure, encoder_type):
        self.k = k
        self.distance_measure = distance_measure
        if (encoder_type == "ResNet"):
            self.encoder_type = 1
        if (encoder_type == "VIT"):
            self.encoder_type = 2

    def set_hyperparameters(self, k, distance_measure, encoder_type):
        self.k = k
        self.distance_measure = distance_measure
        if (encoder_type == "ResNet"):
            self.encoder_type = 1
        if (encoder_type == "VIT"):
            self.encoder_type = 2

    def euclidean_distance(self, point1, point2):
        return np.sqrt(np.sum((point1 - point2)**2))

    def manhattan_distance(self, point1, point2):
        return np.sum(np.abs(point1 - point2))

    def cosine_distance(self, vector1, vector2):
        dot_product = np.dot(vector1, vector2.T)
        magnitude_product = np.linalg.norm(vector1) * np.linalg.norm(vector2)
        return 1 - dot_product / magnitude_product

    def classify(self, x_train, x_test, y_train, y_test):
        start = time()
        x_train = x_train[:, self.encoder_type]
        x_test = x_test[:, self.encoder_type]
        predicted = []
        for x in x_test:
            dist = []
            sz = len(x_train)
            for i in range(sz):
                if (self.distance_measure == "euclidean"):
                    d = self.euclidean_distance(x, x_train[i])
                    dist.append([d, y_train[i]])
                if (self.distance_measure == "manhattan"):
                    d = self.manhattan_distance(x, x_train[i])
                    dist.append([d, y_train[i]])
                if (self.distance_measure == "cosine"):
                    d = self.cosine_distance(x, x_train[i])
                    dist.append([d, y_train[i]])
            dist.sort()
            freq = {}
            for i in range(self.k):
                label = dist[i][1]
                if (label in freq):
                    freq[label] += 1
                else:
                    freq[label] = 1
            curr = 0
            predicted_label = ""
            for label in freq:
                if (freq[label] > curr):
                    curr = freq[label]
                    predicted_label = label
            predicted.append(predicted_label)
        end = time()
        accuracy = accuracy_score(y_test, predicted)
        precision = precision_score(
            y_test, predicted, average="weighted", zero_division=0)
        recall = recall_score(
            y_test, predicted, average="weighted", zero_division=0)
        f1 = f1_score(y_test, predicted, average="weighted", zero_division=0)
        inference_time = end-start
        return accuracy, precision, recall, f1, inference_time


data = np.load('data.npy', allow_pickle=True)
test_file = sys.argv[1]

test = np.load(test_file, allow_pickle=True)

k = 7
encoder_type = "VIT"
distance_measure = "euclidean"

knn = KNN(k, distance_measure, encoder_type)

x_train = data[:,:3]
y_train = data[:, 3]
x_test = test[:,:3]
y_test = test[:, 3]

table=PrettyTable(["Performance Measure","Value"])
performance_measures=knn.classify(x_train, x_test, y_train, y_test)
# print(performance_measures)

table.add_row(["Accuracy",performance_measures[0]])
table.add_row(["Precision",performance_measures[1]])
table.add_row(["Recall",performance_measures[2]])
table.add_row(["F1_score",performance_measures[3]])
table.add_row(["Inference_time",performance_measures[4]])

print(table)