# import the necessary packages
from pyimagesearch.localbinarypatterns import LocalBinaryPatterns
from sklearn.svm import LinearSVC
from imutils import paths
import argparse
import cv2
import os
import matplotlib.pyplot as plt
from sklearn import metrics
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--training", required=True,
    help="path to the training images")
ap.add_argument("-e", "--testing", required=True, 
    help="path to the tesitng images")
args = vars(ap.parse_args())
# initialize the local binary patterns descriptor along with
# the data and label lists
desc = LocalBinaryPatterns(24, 8)
data = []
labels = []
# loop over the training images
for imagePath in paths.list_images(args["training"]):
    # load the image, convert it to grayscale, and describe it
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = desc.describe(gray)
    # extract the label from the image path, then update the
    # label and data lists
    labels.append(imagePath.split(os.path.sep)[-2])
    data.append(hist)
# train a Linear SVM on the data
model = LinearSVC(C=100.0, random_state=42, max_iter=10000)
model.fit(data, labels)
y_test=[]
y_predic=[]
# loop over the testing images
for imagePath in paths.list_images(args["testing"]):
    # load the image, convert it to grayscale, describe it,
    # and classify it
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = desc.describe(gray)
    y_test.append(imagePath.split(os.path.sep)[-2])
    prediction = model.predict(hist.reshape(1, -1))
    
    y_predic.append(prediction[0])
    
    # display the image and the prediction
    #cv2.putText(image, prediction[0], (10, 30), cv2.FONT_HERSHEY_SIMPLEX,1.0, (0, 0, 255), 3)
    #plt.imshow(image)
    #cv2.imshow("Image", image)
    #cv2.waitKey(0)
    
#print(precision_score(y_test, y_predic, pos_label='Live', average='macro'))
#print(recall_score(y_test, y_predic, pos_label='Live', average='macro'))
#print(accuracy_score(y_test, y_predic))

print("Accuracy:")
print (metrics.accuracy_score(y_test, y_predic))
print("Precision and recall:")
print(metrics.classification_report(y_test,y_predic))
