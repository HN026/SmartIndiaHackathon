import cv2
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier  
from sklearn.naive_bayes import GaussianNB  
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
from skimage import feature
from imutils import build_montages
from imutils import paths
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt

def quantify_image(image):
    features = feature.hog(image, orientations=9,
                           pixels_per_cell=(10, 10), cells_per_block=(2, 2),
                           transform_sqrt=True, block_norm="L1")
    return features

def load_split(path):
    imagePaths = list(paths.list_images(path))
    data = []
    labels = []

    for imagePath in imagePaths:
        label = imagePath.split(os.path.sep)[-2]

        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (200, 200))

        image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        features = quantify_image(image)
        data.append(features)
        labels.append(label)

    return (np.array(data), np.array(labels))

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to input dataset")
ap.add_argument("-t", "--trials", type=int, default=5,
                help="# of trials to run")
args = vars(ap.parse_args())

trainingPath = os.path.sep.join([args["dataset"], "training"])
testingPath = os.path.sep.join([args["dataset"], "testing"])

print("[INFO] Loading Data...")
(trainX, trainY) = load_split(trainingPath)
(testX, testY) = load_split(testingPath)

le = LabelEncoder()
trainY = le.fit_transform(trainY)
testY = le.transform(testY)

trials = {}

classifiers = {
    "RandomForest": RandomForestClassifier(n_estimators=100),
    "SVM": SVC(),
    "LogisticRegression": LogisticRegression(),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "DecisionTree": DecisionTreeClassifier(),
    "GaussianNB": GaussianNB(),
    "GradientBoosting": GradientBoostingClassifier()
}

classifier_names = []
accuracies = []

for classifier_name, classifier in classifiers.items():
    print("[INFO] Training {} classifier...".format(classifier_name))
    classifier.fit(trainX, trainY)

    predictions = classifier.predict(testX)

    accuracy = accuracy_score(testY, predictions)
    print("[INFO] {} classifier accuracy: {:.4f}".format(classifier_name, accuracy))

    classifier_names.append(classifier_name)
    accuracies.append(accuracy)

    cm = confusion_matrix(testY, predictions)
    print("[INFO] Confusion matrix for {} classifier:".format(classifier_name))
    print(cm)

    testingPaths = list(paths.list_images(testingPath))
    idxs = np.arange(0, len(testingPaths))
    idxs = np.random.choice(idxs, size=(25,), replace=False)
    images = []

    for i in idxs:
        image = cv2.imread(testingPaths[i])
        output = image.copy()
        output = cv2.resize(output, (128, 128))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (200, 200))
        image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        features = quantify_image(image)
        preds = classifier.predict([features])
        label = le.inverse_transform(preds)[0]

        color = (0, 255, 0) if label == "healthy" else (0, 0, 255)
        cv2.putText(output, label, (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        images.append(output)

    montage = build_montages(images, (128, 128), (5, 5))[0]

    cv2.imshow("{} Classifier Output".format(classifier_name), montage)
    cv2.waitKey(0)

plt.bar(classifier_names, accuracies)
plt.xlabel('Classifier')
plt.ylabel('Accuracy')
plt.title('Classifier Comparison')
plt.ylim(0, 1.0)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
