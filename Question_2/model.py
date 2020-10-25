import sys
sys.path.insert(0, '..')

import os, random
from random import seed
from math import sqrt
from shared.get_dataset import get_dataset, get_testing_images, get_training_images


seed(1)

# Make a prediction with weights
def predict(row, weights):
	activation = weights[0]
	for i in range(len(row)-1):
		activation += weights[i + 1] * row[i]
	return 1.0 if activation >= 0.0 else 0.0
 
# Estimate Perceptron weights using stochastic gradient descent
def train_weights(train, learning_rate, n_iter):
	weights = [0.0 for i in range(len(train[0]))]
	for _ in range(n_iter):
		for row in train:
			prediction = predict(row, weights)
			error = row[-1] - prediction
			weights[0] = weights[0] + learning_rate * error
			for i in range(len(row)-1):
				weights[i + 1] = weights[i + 1] + learning_rate * error * row[i]
	return weights
 
# Perceptron Algorithm With Stochastic Gradient Descent
def perceptron(train, test, learning_rate, n_iter):
	predictions = list()
	weights = train_weights(train, learning_rate, n_iter)
	for row in test:
		prediction = predict(row, weights)
		predictions.append(prediction)
	return(predictions)

# Extracting the perceptron weights based on the training dataset
def train_data_model() :
    
	imageListTraining = get_training_images()
	training_data_set = get_dataset(imageListTraining)	
	
	lrate = 0.01
	n_iter = 5000

	trained_weights = train_weights(training_data_set, lrate, n_iter)
	return trained_weights

# Predicting the labels for the testing dataset using the evaluated weights
def predict_test_data(weights) :
    
	imageListTesting = get_testing_images()
	# Getting every even indexed image to get only 5 images in testing (as per assignment)
	imageListTesting = [imageListTesting[i] for i in range(len(imageListTesting)) if i % 2 == 0]

	test_data_set = get_dataset(imageListTesting, test_data=True)

	for image in range(len(test_data_set)) :
		predicted_label = predict(test_data_set[image], weights)
		print(f'The prediction for image {os.path.basename(imageListTesting[image])} is evaluated to be {predicted_label}')


def main() :
	weights = train_data_model()
	print("Evaluated weights:", weights)
	
	predict_test_data(weights)

if __name__ == "__main__":
    main()