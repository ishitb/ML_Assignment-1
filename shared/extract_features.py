import cv2

# Extracting the features( means and standard deviation of an image, but turning it to grayscale for better labelling )
def extract_features(image) :
    raw = cv2.imread(image)
    gray_image = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    return cv2.meanStdDev(cv2.resize(gray_image, dsize=(32, 32)))