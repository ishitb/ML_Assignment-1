from shared.split_data import cross_validation_split

# Evaluate a given algorithm using a cross validation split
def evaluate_scores_as_algo(dataset, n_folds, classifier, algo):
	folds = cross_validation_split(dataset, n_folds)
	scores = []
	for fold in folds:
		train_set = list(folds)
		train_set = [i for i in train_set if i != fold]
		train_set = sum(train_set, [])
		test_set = []

		for row in fold:
			row_copy = list(row)
			test_set.append(row_copy)
			row_copy[-1] = None
		predicted = algo(train_set, test_set, classifier)
		actual = [row[-1] for row in fold]
		scores.append(get_accuracy(actual, predicted))
	return scores

# Calculate accuracy percentage
def get_accuracy(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0