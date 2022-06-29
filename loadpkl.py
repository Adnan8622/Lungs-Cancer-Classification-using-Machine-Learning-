# make a prediction for a new image.
#from keras.preprocessing.image import img_to_array
#from keras.preprocessing.image import load_img
import joblib
#from keras.models import load_model
from skimage.feature import hog
from skimage import exposure
from skimage import feature
import numpy as np
import pickle
import cv2

wid = 100
dim=(wid,wid)
'''
# load and prepare the image
def load_image(filename):
	# load the image
	img = load_img(filename, grayscale=True, target_size=(28, 28))
	# convert to array
	img = img_to_array(img)
	# reshape into a single sample with 1 channel
	img = img.reshape(1, 28, 28, 1)
	# prepare pixel data
	img = img.astype('float32')
	img = img / 255.0
	return img
'''
# load an image and predict the class
def run_example():
	# load the image
	#img = load_image('cat.jpg')
	img = cv2.imread('Malignant(4).jpg')
	imag = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	imge = cv2.resize(imag, dim, interpolation=cv2.INTER_AREA)
	imgFeatures, hog_image = hog(imge, orientations=8, pixels_per_cell=(8, 8),
                    cells_per_block=(1, 1), visualise=True)
	lbpF = feature.local_binary_pattern(imge, 10, 5, method="uniform")
	lbpF = lbpF.ravel()
	
	final_vector = np.concatenate((imgFeatures,lbpF))
	fts = final_vector.reshape(1,-1)
	# load Model
	# model = load_model('dog&cat_classify_model.pkl')
	
	model=joblib.load('Lungs cancer train model.pkl')
	# predict the class
	digit = model.predict(fts)
	print(digit, ' ---- pridiction')
	#cv2.imshow(str(digit), imag)
	#cv2.waitKey(0)
	#destroyAllWindow();
### Pass the label to image
	if digit[0] == 'Benign':
		print('Benign  <-------pridiction');
		font=cv2.FONT_HERSHEY_SIMPLEX
		cv2.putText(img,('Benign'),(50,50), font, 2,(0,0,255),2,cv2.LINE_AA)
		cv2.imshow('Image',img)
		#cv2.imshow(str(digit),imag)
		cv2.waitKey(0)
		cv2.imwrite('Benign.jpg',img)
	else:
		print('Malignant  <-------pridiction');
		font=cv2.FONT_HERSHEY_SIMPLEX
		cv2.putText(img,('Malignant'),(50,50), font,2,(0,0,255),2,cv2.LINE_AA)
		cv2.imshow('Image',img)
		#cv2.imshow(str(digit),imag)
		cv2.waitKey(0)
		cv2.imwrite('Malignant.jpg',img)

# entry point, run the example
run_example()
