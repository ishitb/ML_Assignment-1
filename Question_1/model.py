import sys
sys.path.insert(0, '..')

import os, random
from random import seed
from math import sqrt
from shared.get_dataset import get_dataset, get_testing_images, get_training_images
from shared.evaluate_algo import evaluate_scores_as_algo

seed(1)

# Calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
	distance = 0.0
	for i in range(len(row1)-1):
		distance += (row1[i] - row2[i])**2
	return sqrt(distance)

# Locate the most similar neighbors
def get_neighbors(train, test_row, k):
	distances = []
	for train_row in train:
		dist = euclidean_distance(test_row, train_row)
		distances.append((train_row, dist))
	distances.sort(key=lambda tup: tup[1])
	neighbors = []
	for i in range(k):
		neighbors.append(distances[i][0])
	return neighbors

# Make a prediction with neighbors
def predict_classification(train, test_row, k):
	neighbors = get_neighbors(train, test_row, k)
	output_values = [row[-1] for row in neighbors]
	prediction = max(set(output_values), key=output_values.count)
	return prediction

# kNN Algorithm
def k_nearest_neighbors(train, test, k):
	predictions = []
	for row in test:
		output = predict_classification(train, row, k)
		predictions.append(output)
	return predictions 

# Finding the best k withing a minmax limit
def find_best_k():
    imageList = get_training_images()

    training_dataset = get_dataset(imageList)
    random.shuffle(training_dataset)
    n_folds = 5
    k_min, k_max = 1, 20

    best_k = 1
    max_accuracy_yet = 0

    for k in range (k_min, k_max + 1) :
        scores = evaluate_scores_as_algo(training_dataset, n_folds, k, k_nearest_neighbors)
        
        new_accuracy = sum(scores)/float(len(scores))
        print(f'Accuracy for k = {k} comes out to be\t{new_accuracy}')
        max_accuracy_yet = max(max_accuracy_yet, new_accuracy)

        if max_accuracy_yet == new_accuracy :
            best_k = k

    return best_k

# Predicting the labels for test data
def predict_test_data(k) :
    imageListTesting = get_testing_images()
    test_data_set = get_dataset(imageListTesting, test_data=True)

    imageListTraining = get_training_images()
    training_data_set = get_dataset(imageListTraining)

    predicted = k_nearest_neighbors(training_data_set, test_data_set, k)
    for i, f in enumerate(imageListTesting) :
        print(f'Label for {os.path.basename(f)} is predicted to be:\t{predicted[i]}')

def main() :
    k = find_best_k()
    print("The value of 'k' for optimum accuracy was found to be", k)

    predict_test_data(k)

if __name__ == "__main__":
    main()