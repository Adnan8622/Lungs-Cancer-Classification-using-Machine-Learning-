from imutils import paths
import numpy as np
import random
import pickle
import cv2
import os,glob
from skimage import feature
from skimage.feature import hog , local_binary_pattern
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
#from sklearn.externals import joblib 
import joblib
from skimage import exposure
from sklearn.neighbors import KNeighborsClassifier

path='Dataset/*'

data_path = os.path.join(path,'*g')
imagePaths = glob.glob(data_path)
print(len(imagePaths))

data = []
LBP = []
labels = []
hogFeatures = []
featurevector = [[0 for x in range(0)] for y in range(0)] #2d array

# loop over the input images
for imagePath in imagePaths:
	# load the image, pre-process it, and store it in the data list
	image = cv2.imread(imagePath,0)
	image = cv2.resize(image, (100,100),interpolation=cv2.INTER_AREA)
	#cv2.imshow('a',image)
	#cv2.waitKey(0)
	#image = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	fd, hog_image = hog(image, orientations=8, pixels_per_cell=(8, 8),cells_per_block=(1, 1), visualise=True)
	lbp = local_binary_pattern(image, 10, 15,  method="uniform")
	#hogFeatures.append(fd)
	#LBP.append(lbp)
	llb = lbp.ravel()
	hogFeatures = fd
	features = np.concatenate((hogFeatures,llb))
	featurevector.append(features)
	#extract labels
	#data.append(image)
	l = label = imagePath.split(os.path.sep)[-2].split("_")
	labels.append(l)
	
print(labels)
print(len(hogFeatures))

X_train, X_test, y_train, y_test = train_test_split(featurevector, labels, test_size = 0.20)

print('train images ',len(X_train) )
print('test images ',len(X_test))
print('train labels ',len(y_train))
print('test labels',len(y_test))

print(len(LBP))

########## Classifiers #############
##########   SVM       #############

svclassifier = svm.SVC()
svclassifier = SVC(kernel='linear')
svclassifier.probability = True
svclassifier.fit(X_train, y_train)

#Now making prediction
y_pred = svclassifier.predict(X_test)
print("Actual Labels :    ", y_test)
print("Predicted Labels : ",y_pred)

# accuracy 
accuracy = svclassifier.score(X_test, y_test)
print('accuracy',accuracy*100)

##########   KNN       #############
"""
classifier= KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)
y_pred=classifier.predict(X_test)

#Now making prediction
y_pred = classifier.predict(X_test)
print("Actual Labels :    ", y_test)
print("Predicted Labels : ",y_pred)

# accuracy 
accuracy = classifier.score(X_test, y_test)
print('accuracy',accuracy*100)
"""
####################################


# confusion matrix 
print(confusion_matrix(y_test,y_pred))
#print(classification_report(y_test,y_pred))

joblib.dump(svclassifier,'Lungs cancer train model.pkl')
print('Model saved...!')

