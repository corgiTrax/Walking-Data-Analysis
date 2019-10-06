import sys
import scipy.io as sio
import numpy as np
import copy
import h5py
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model


dataFile = 'data/bodyJoints.mat'
dataFile2 = 'data/heights.mat'

gazeKey = 'gazeXZAll'
jointKey, jointVelKey, jointNameKey = 'markersAll', 'velMarkersAll', 'markerNames'
comKey, comVelKey = 'comXYZAll', 'velComXYZAll'
gaitKey = 'gaitCyclePctAll'
heightKey = 'subjHeight'


class Data:
	def __init__(self):
		print("Loading data from files: %s, %s" % (dataFile, dataFile2))
		
		data = h5py.File(dataFile, 'r')

		# dependent variable: 2(x,y) x #samples, there are NaNs
		gazeData = np.array(data[gazeKey])
		gazeData = np.swapaxes(gazeData, 0, 1)

		# independent variable 3(x,y,z) x #joints x #samples; only use 0-24
		jointData = np.array(data[jointKey])[:,0:24,:]
		jointData = np.swapaxes(jointData, 0, 2)
		jointData = np.reshape(jointData, (len(jointData), 3*24))

		jointVelData = np.array(data[jointVelKey])[:,0:24,:]
		jointVelData = np.swapaxes(jointVelData, 0, 2)
		jointVelData = np.reshape(jointVelData, (len(jointVelData), 3*24))

		comData = np.array(data[comKey])
		comData = np.swapaxes(comData, 0, 1)

		comVelData = np.array(data[comVelKey])
		comVelData = np.swapaxes(comVelData, 0, 1)

		gaitData = np.array(data[gaitKey])
		gaitData = np.swapaxes(gaitData, 0, 1)

		heightData = sio.loadmat(dataFile2)[heightKey]

		# remove NaNs
		finiteMask = np.isfinite(gazeData)[:,0]
		self.gazeData = gazeData[finiteMask]
		
		self.allIVData = np.concatenate((jointData, jointVelData, comData, comVelData, gaitData, heightData),axis=1)
		self.allIVData = self.allIVData[finiteMask]
		print(self.allIVData.shape)


	def train(self, method):
		#self.allIndVars= StandardScaler().fit_transform(self.allIndVars)

		if method == 'sk_ole':
			# ordinary least square
			reg = LinearRegression(normalize=True)
		elif method == 'sk_l2':
			# ridge
			reg = linear_model.Ridge(alpha=0.1,normalize=True)
		elif method == 'sk_l1':
			# lasso
			reg = linear_model.Lasso(alpha=0.05,normalize=True)
		else:
			print("No regression method specified; choose from ['ole'|'l1'|'l2']")
		
		if method.startwith('sk_'):
			reg.fit(self.allIVData, self.gazeData)
			print(reg.score(self.allIVData, self.gazeData))

data = Data()
data.train('sk_ole')
