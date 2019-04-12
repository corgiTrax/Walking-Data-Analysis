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
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight
import utils as U

# gaze and non gaze file names
dirc = "data/32/color-allFixFrames/"
gazeFiles = ["1_fix_patch_stack.mat", "2_fix_patch_stack.mat", "3_fix_patch_stack.mat"]
nonGazeFiles =  ["1_non_fix_patch_stack.mat", "2_non_fix_patch_stack.mat", "3_non_fix_patch_stack.mat"]
# gaze and non gaze field names in mat files
gazeMatField, nonGazeMatField = "patchStack","patchStack_non"

dircOF = "data/32/flow-allFixFrames/"
gazeFilesOF = ["1_mag_fix_patch_stack.mat", "2_mag_fix_patch_stack.mat", "3_mag_fix_patch_stack.mat"]
nonGazeFilesOF =  ["1_mag_non_fix_patch_stack.mat", "2_mag_non_fix_patch_stack.mat", "3_mag_non_fix_patch_stack.mat"]
# gaze and non gaze field names in mat files
gazeMatFieldOF, nonGazeMatFieldOF = "magStack","magStack_non"

# note cannot shuffle data in this dataset before splitting
trainRatio = 0.85 # 3611 gaze; 21474 nonGaze
imgRow, imgCol = 32, 32
inputShape = (imgRow, imgCol, 3)
inputShapeOF = (imgRow, imgCol, 1)
modelDir = 'Experiments/' + str(imgRow) + '/cnn-img+of'
dropout = 0.5
epoch = 50
dataAug = False

class Data:
	def __init__(self, gazeFiles, nonGazeFiles):
		'''provided data is in .mat files'''
		# load colored image data
		print("Loading image data from files: %s %s" % (gazeFiles, nonGazeFiles))

		gazeData, nonGazeData = [], []
		for i, gf in enumerate(gazeFiles):
			gazeFile = dirc + gf
			if i == 0:
				gazeData = sio.loadmat(gazeFile)[gazeMatField]
			else:
				gazeData = np.concatenate((gazeData, sio.loadmat(gazeFile)[gazeMatField]))
		
		for i, ngf in enumerate(nonGazeFiles):
			nonGazeFile = dirc + ngf
			if i == 0:
				nonGazeData = sio.loadmat(nonGazeFile)[nonGazeMatField]
			else:
				nonGazeData = np.concatenate((nonGazeData, sio.loadmat(nonGazeFile)[nonGazeMatField]))
		
		gazeData = gazeData / 255.0
		nonGazeData = nonGazeData /  255.0
		numGaze = len(gazeData)
		numNonGaze = len(nonGazeData)
		print("%s, %s samples from gaze/nongaze files" % (numGaze, numNonGaze))

		# The data is organized in this way: pixel1Rchannel, pixel2Rchannel, ...., pixel1Gchannel...
		gazeData = gazeData.reshape(numGaze, 3, imgRow, imgCol)
		nonGazeData = nonGazeData.reshape(numNonGaze, 3, imgRow, imgCol)
		gazeData = np.moveaxis(gazeData, 1, 3)
		nonGazeData = np.moveaxis(nonGazeData, 1, 3)

		# split data 
		numGazeTrain, numNonGazeTrain = int(numGaze * trainRatio), int(numNonGaze * trainRatio)
		numGazeTest, numNonGazeTest = numGaze - numGazeTrain, numNonGaze - numNonGazeTrain

		self.trainData = np.concatenate((gazeData[0:numGazeTrain], nonGazeData[0:numNonGazeTrain]))
		self.testData = np.concatenate((gazeData[numGazeTrain:], nonGazeData[numNonGazeTrain:]))

		# 1s for gaze and 0s for nonGaze
		self.trainLabel = np.concatenate((np.ones(numGazeTrain), np.zeros(numNonGazeTrain)))
		self.testLabel = np.concatenate((np.ones(numGazeTest), np.zeros(numNonGazeTest)))
		
		self.numTrain = len(self.trainLabel)
		self.numTest = len(self.testLabel)

		# preprocessing
		# Note: use mean/std from trainining data only if want to generalize
		self.allData = np.concatenate((self.trainData, self.testData))
		mean = np.mean(self.allData, axis=0)
		std = np.std(self.allData, axis=0)
		self.trainData = (self.trainData - mean) / std
		self.testData = (self.testData - mean) / std

		# Now load optical flow data
		print("Loading optical flow data from files: %s %s" % (gazeFilesOF, nonGazeFilesOF))

		gazeDataOF, nonGazeDataOF = [], []
		for i, gf in enumerate(gazeFilesOF):
			gazeFileOF = dircOF + gf
			if i == 0:
				gazeDataOF = sio.loadmat(gazeFileOF)[gazeMatFieldOF]
			else:
				gazeDataOF = np.concatenate((gazeDataOF, sio.loadmat(gazeFileOF)[gazeMatFieldOF]))
		
		for i, ngf in enumerate(nonGazeFilesOF):
			nonGazeFileOF = dircOF + ngf
			if i == 0:
				nonGazeDataOF = sio.loadmat(nonGazeFileOF)[nonGazeMatFieldOF]
			else:
				nonGazeDataOF = np.concatenate((nonGazeDataOF, sio.loadmat(nonGazeFileOF)[nonGazeMatFieldOF]))

		gazeDataOF = gazeDataOF.reshape(numGaze, imgRow, imgCol, 1)
		nonGazeDataOF = nonGazeDataOF.reshape(numNonGaze, imgRow, imgCol, 1)

		self.trainDataOF = np.concatenate((gazeDataOF[0:numGazeTrain], nonGazeDataOF[0:numNonGazeTrain]))
		self.testDataOF = np.concatenate((gazeDataOF[numGazeTrain:], nonGazeDataOF[numNonGazeTrain:]))

		# preprocessing
		# Note: use mean/std from trainining data only if want to generalize
		self.allDataOF = np.concatenate((self.trainDataOF, self.testDataOF))
		mean = np.mean(self.allDataOF, axis=0)
		std = np.std(self.allDataOF, axis=0)
		self.trainDataOF = (self.trainDataOF - mean) / std
		self.testDataOF = (self.testDataOF - mean) / std

	def train(self):
		U.save_GPU_mem_keras()
		expr = U.ExprCreaterAndResumer(modelDir, postfix="dr%s_imgOnly" % (str(dropout)))

		# x channel: image
		x_inputs = L.Input(shape=inputShape)
		x = x_inputs # inputs is used by the line "Model(inputs, ... )" below

		conv11 = L.Conv2D(16, (3,3), strides=1, dilation_rate = 2, padding='valid')
		x = conv11(x)
		x = L.Activation('relu')(x)
		x = L.BatchNormalization()(x)
		# Batch needs to be after relu, otherwise it won't train...

		conv12 = L.Conv2D(16, (3,3), strides=1, dilation_rate = 2, padding='valid')
		x = conv12(x)
		x = L.Activation('relu')(x)
		x = L.BatchNormalization()(x)

		conv13 = L.Conv2D(16, (3,3), strides=1, dilation_rate = 2, padding='valid')
		x = conv13(x)
		x = L.Activation('relu')(x)
		x_output = L.BatchNormalization()(x)
		
		# z channel: optical flow
		z_inputs = L.Input(shape=inputShapeOF)
		z = z_inputs # inputs is used by the line "Model(inputs, ... )" below

		conv21 = L.Conv2D(16, (3,3), strides=1, dilation_rate = 2, padding='valid')
		z = conv21(z)
		z = L.Activation('relu')(z)
		z = L.BatchNormalization()(z)

		conv22 = L.Conv2D(16, (3,3), strides=1, dilation_rate = 2, padding='valid')
		z = conv22(z)
		z = L.Activation('relu')(z)
		z = L.BatchNormalization()(z)

		conv23 = L.Conv2D(16, (3,3), strides=1, dilation_rate = 2, padding='valid')
		z = conv23(z)
		z = L.Activation('relu')(z)
		z_output = L.BatchNormalization()(z)
		
		joint = L.Average()([x_output, z_output])
		joint = L.Flatten()(joint)
		joint = L.Dense(32, activation='relu')(joint)
		joint = L.Dropout(dropout)(joint)
		output=L.Dense(1, activation='sigmoid')(joint)


		model=Model(inputs=[x_inputs,z_inputs], outputs=output)

		opt = K.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
		#opt = K.optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
		model.compile(optimizer=opt, loss='binary_crossentropy', metrics=[K.metrics.binary_accuracy])

		print(model.summary())
		# snapshot code before training the model
		expr.dump_src_code_and_model_def(sys.argv[0], model)

		model.fit([self.trainData, self.trainDataOF], self.trainLabel, 
			validation_data=([self.testData, self.testDataOF], self.testLabel),
			shuffle=True, batch_size=100, epochs=epoch, verbose=2,
			callbacks=[K.callbacks.TensorBoard(log_dir=expr.dir),
			K.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr = 0.00001),
			U.PrintLrCallback()])

		expr.save_weight_and_training_config_state(model)

		score = model.evaluate([self.testData, self.testDataOF], self.testLabel, batch_size=100, verbose=0)
		expr.printdebug("eval score:" + str(score))

	def test(self):
		pass

data = Data(gazeFiles, nonGazeFiles)
data.train()