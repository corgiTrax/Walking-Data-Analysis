'''takes in gaze patches and non gaze patches, train a simple CNN trying to do classification'''
import sys
import scipy.io as sio
import numpy as np
import copy

import keras as K
import keras.layers as L
from keras.models import Model
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from sklearn.utils import class_weight

# gaze and non gaze file names
gazeFile, nonGazeFile = "79_gaze.mat", "79_nonGaze.mat"
# gaze and non gaze field names in mat files
gazeMatField, nonGazeMatField = "patch_Stack","patch_Stack_non"
# note cannot shuffle data in this dataset before splitting
trainRatio = 0.7 # 3611 gaze; 21474 nonGaze
imgRow, imgCol = 79, 79
SHAPE = (imgRow, imgCol, 1)

class Data:
	def __init__(self, gazeFile, nonGazeFile):
		'''provided data is in .mat files'''
		print("Loading data from files: %s %s" % (gazeFile, nonGazeFile))
		gazeData = sio.loadmat(gazeFile)[gazeMatField]
		nonGazeData = sio.loadmat(nonGazeFile)[nonGazeMatField]
		gazeData = gazeData / 255.0
		nonGazeData = nonGazeData /  255.0
		numGaze = len(gazeData)
		numNonGaze = len(nonGazeData)
		print("%s, %s samples from gaze/nongaze files" % (numGaze, numNonGaze))

		gazeData = gazeData.reshape(numGaze, imgRow, imgCol, 1)
		nonGazeData = nonGazeData.reshape(numNonGaze, imgRow, imgCol, 1)

		# split data 
		numGazeTrain, numNonGazeTrain = int(numGaze * trainRatio), int(numNonGaze * trainRatio)
		numGazeTest, numNonGazeTest = numGaze - numGazeTrain, numNonGaze - numNonGazeTrain

		self.trainData = np.concatenate((gazeData[0:numGazeTrain], nonGazeData[0:numNonGazeTrain]))
		self.testData = np.concatenate((gazeData[numGazeTrain:], nonGazeData[numNonGazeTrain:]))

		# 1s for gaze and 0s for nonGaze
		self.trainLabel = np.concatenate((np.ones(numGazeTrain), np.zeros(numNonGazeTrain)))
		self.testLabel = np.concatenate((np.ones(numGazeTest), np.zeros(numNonGazeTest)))
		
		# reweight unblanced dataset
		self.class_weights = class_weight.compute_class_weight('balanced', np.unique(self.trainLabel), self.trainLabel)
		self.class_weights = dict(enumerate(self.class_weights))

		self.numTrain = len(self.trainLabel)
		self.numTest = len(self.testLabel)

		# convert label to categorical
		self.trainLabel = to_categorical(self.trainLabel)
		self.testLabel = to_categorical(self.testLabel)

	def train(self):
		# should use more weight on gaze loss patches to balance data
	    inputs = L.Input(shape=SHAPE)
	    x = inputs # inputs is used by the line "Model(inputs, ... )" below
	    
	    conv1 = L.Conv2D(64, (3,3), strides=1, padding='valid')
	    x = conv1(x)
	    x = L.Activation('relu')(x)
	    x = L.BatchNormalization()(x)
	    
	    conv2 = L.Conv2D(32, (3,3), strides=1, padding='valid')
	    x = conv2(x)
	    x = L.Activation('relu')(x)
	    x = L.BatchNormalization()(x)

	    conv3 = L.Conv2D(32, (3,3), strides=1, padding='valid')
	    x = conv3(x)
	    x = L.Activation('relu')(x)
	    x = L.BatchNormalization()(x)
	    
	    x=L.Flatten()(x)

	    x=L.Dense(256, activation='relu')(x)
	    x=L.BatchNormalization()(x)
	    x=L.Dropout(0.5)(x)
	    output=L.Dense(2, activation='softmax')(x)

	    model=Model(inputs=inputs, outputs=output)
	    


	    #opt = K.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
	    opt = K.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
	    model.compile(optimizer= opt, loss='categorical_crossentropy', metrics=['accuracy'])
	    model.fit(self.trainData, self.trainLabel, validation_data=(self.testData, self.testLabel),\
	    	class_weight=self.class_weights, shuffle=True, batch_size=100, epochs=10)


	def test(self):
		pass


if __name__ == '__main__':
	data = Data(gazeFile, nonGazeFile)
	data.train()