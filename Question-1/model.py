from random import seed
from random import randrange
from math import sqrt
import os, cv2

seed(1)

# Split a dataset into n folds
def cross_validation_split(dataset, n_folds):
	dataset_split = []
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / n_folds)
	for _ in range(n_folds):
		fold = []
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split

# Calculate accuracy percentage
def get_accuracy(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0

# Evaluate an algorithm using a cross validation split
def evaluate_scores(dataset, n_folds, *args):
	folds = cross_validation_split(dataset, n_folds)
	scores = []
	for fold in folds:
		train_set = list(folds)
		train_set.remove(fold)
		train_set = sum(train_set, [])
		test_set = []
		for row in fold:
			row_copy = list(row)
			test_set.append(row_copy)
			row_copy[-1] = None
		predicted = k_nearest_neighbors(train_set, test_set, *args)
		actual = [row[-1] for row in fold]
		scores.append(get_accuracy(actual, predicted))
	return scores

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

def extract_features(image) :
    raw = cv2.imread(image)
    return cv2.meanStdDev(raw)

def get_training_images():
    training_set_dir = os.path.join('..', 'dataset', 'training_set')
    dog_images = [os.path.join(training_set_dir, 'dogs', f) for i, f in enumerate(os.listdir(os.path.join(training_set_dir, 'dogs')))]
    cat_images = [os.path.join(training_set_dir, 'cats', f) for i, f in enumerate(os.listdir(os.path.join(training_set_dir, 'cats')))]
    return dog_images + cat_images

def get_testing_images():
    testing_set_dir = os.path.join('..', 'dataset', 'test_data')
    images = [os.path.join(testing_set_dir, f) for i, f in enumerate(os.listdir(testing_set_dir))]
    return images

def get_dataset(imageList, test_data=False) :
    dataset = []

    for image in imageList :
        mean, stddev = extract_features(image)

        if not test_data :
            label = 0 if os.path.basename(image)[:3] == 'dog' else 1
            dataset.append(
                (
                    mean.flatten().tolist() + 
                    stddev.flatten().tolist() + 
                    [label]
                )
            )

        else :
            dataset.append(
                (
                    mean.flatten().tolist() + 
                    stddev.flatten().tolist()
                )
            )

    return dataset

import random

def find_best_k():
    imageList = get_training_images()

    training_dataset = get_dataset(imageList)
    random.shuffle(training_dataset)
    n_folds = 5
    k_min, k_max = 1, 20

    best_k = 1
    max_accuracy_yet = 0

    for k in range (k_min, k_max + 1) :
        scores = evaluate_scores(training_dataset, n_folds, k)
        
        # Finding the k
        new_accuracy = sum(scores)/float(len(scores))
        # print(f'Accuracy for k = {k} comes out to be\t{new_accuracy}')
        max_accuracy_yet = max(max_accuracy_yet, new_accuracy)

        if max_accuracy_yet == new_accuracy :
            best_k = k

    return best_k


def predict_test_data(k) :
    imageListTesting = get_testing_images()
    test_data_set = get_dataset(imageListTesting, test_data=True)

    imageListTraining = get_training_images()
    training_data_set = get_dataset(imageListTraining)

    predicted = k_nearest_neighbors(training_data_set, test_data_set, k)
    print(predicted) 

def main() :
    k = find_best_k()
    predict_test_data(k)

if __name__ == "__main__":
    main()