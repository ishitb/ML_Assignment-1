import os
from shared.extract_features import extract_features

# Extracting training data set from given path
def get_training_images():
    training_set_dir = os.path.join('..', 'dataset', 'training_set')
    dog_images = [os.path.join(training_set_dir, 'dogs', f) for i, f in enumerate(os.listdir(os.path.join(training_set_dir, 'dogs')))]
    cat_images = [os.path.join(training_set_dir, 'cats', f) for i, f in enumerate(os.listdir(os.path.join(training_set_dir, 'cats')))]
    return cat_images + dog_images

# Extracting test data set from given path
def get_testing_images():
    testing_set_dir = os.path.join('..', 'dataset', 'test_data')
    images = [os.path.join(testing_set_dir, f) for i, f in enumerate(os.listdir(testing_set_dir))]
    return images

# Forming dataset based on image features
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