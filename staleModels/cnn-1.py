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
import utils as U

# gaze and non gaze file names
gazeFile, nonGazeFile = "data/79_gaze.mat", "data/79_nonGaze.mat"
# gaze and non gaze field names in mat files
gazeMatField, nonGazeMatField = "patch_Stack","patch_Stack_non"
# note cannot shuffle data in this dataset before splitting
trainRatio = 0.85 # 3611 gaze; 21474 nonGaze
imgRow, imgCol = 79, 79
inputShape = (imgRow, imgCol, 1)


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

		# preprocessing
		# Note: use mean/std from trainining data only if want to generalize
		self.allData = np.concatenate((self.trainData, self.testData))
		mean = np.mean(self.allData, axis=0)
		std = np.std(self.allData, axis=0)
		self.trainData = (self.trainData - mean) / std
		self.testData = (self.testData - mean) / std

	def train(self):
		modelDir = 'Experiments/CNN'
		dropout = 0.5
		epoch = 30
		U.save_GPU_mem_keras()
		expr = U.ExprCreaterAndResumer(modelDir, postfix="dr%s_imgOnly" % (str(dropout)))

		inputs = L.Input(shape=inputShape)
		x = inputs # inputs is used by the line "Model(inputs, ... )" below

		conv1 = L.Conv2D(64, (3,3), strides=1, padding='valid')
		x = conv1(x)
		x = L.Activation('relu')(x)
		x = L.BatchNormalization()(x)
		# Batch needs to be after relu, otherwise it won't train...

		conv2 = L.Conv2D(32, (3,3), strides=1, padding='valid')
		x = conv2(x)
		x = L.Activation('relu')(x)
		x = L.BatchNormalization()(x)

		conv3 = L.Conv2D(32, (3,3), strides=1, padding='valid')
		x = conv3(x)
		x = L.Activation('relu')(x)
		x = L.BatchNormalization()(x)
		
		#x = L.GlobalAveragePooling2D()(x)
		
		#x = L.Flatten()(x)
		x = L.Dense(128, activation='relu')(x)
		x = L.Dropout(dropout)(x)
		output=L.Dense(2, activation='softmax')(x)
		model=Model(inputs=inputs, outputs=output)

		opt = K.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
		#opt = K.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
		model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

		print(model.summary())
		# snapshot code before training the model
		expr.dump_src_code_and_model_def(sys.argv[0], model)

		model.fit(self.trainData, self.trainLabel, validation_data=(self.testData, self.testLabel),
			class_weight=self.class_weights, shuffle=True, batch_size=100, epochs=epoch, verbose=2,
			callbacks=[K.callbacks.TensorBoard(log_dir=expr.dir),
			K.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr = 0.00001),
			U.PrintLrCallback()])

		expr.save_weight_and_training_config_state(model)

		score = model.evaluate(self.testData, self.testLabel, batch_size=100, 0)
		expr.printdebug("eval score:" + str(score))

	def test(self):
		pass



data = Data(gazeFile, nonGazeFile)
data.train()