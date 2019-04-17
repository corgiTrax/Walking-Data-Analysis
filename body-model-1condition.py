import sys
import scipy.io as sio
import numpy as np
import copy
import h5py
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
import statsmodels.api as sm

dirc = 'data/1condition/'

gazeFile = dirc + 'gaze.mat'
bodyFile = dirc + 'bodyJoints.mat'
footholdFile = dirc + 'foothold.mat'
subjFile = dirc + 'subjInfo.mat'

# fields in gazeFile and bodyFile
gaze1Key, gaze2Key = 'gazeVec', 'gazeXZAll'
# fields in bodyFile
jointKey, jointVelKey, jointNameKey = 'markersAll', 'velMarkersAll', 'markerNames'
comKey, comVelKey = 'comXYZAll', 'velComXYZAll'
gaitKey = 'gaitCyclePctAll'
# fields in footholdFile
footholdKey = 'feetLocs'
# fields in subjFile
heightKey, legKey = 'subjHeight', 'legLength'
# only use first 25 joints, since the last severals are heads and unknowns
numJoint = 25



def calc_adj_r2(r2, n, k):
	# r2: r2 returned by regressor; n: sample size; k: # of features
	return 1 - ((1-r2) * (n-1) / (n-k-1))

class Data:
	def __init__(self):
		print("Loading data from files: %s" % (gazeFile))
		self.gaze1Data = sio.loadmat(gazeFile)[gaze1Key]

		print("Loading data from files: %s" % (bodyFile))
		bodyData = h5py.File(bodyFile, 'r')
		
		# printing string from hdf5 file is tricky...
		jointNamesHDF5 = np.array(bodyData[jointNameKey])
		self.jointNames = []
		print("Joint names:")
		for i in range(0, numJoint):
			st = jointNamesHDF5[i][0]
			obj = bodyData[st]
			jointName = ''.join(chr(i) for i in obj[:])
			self.jointNames.append(jointName) # starts with x1 in statsmodel

		# dependent variable: 2(x,y) x #samples, there are NaNs
		self.gaze2Data = np.array(bodyData[gaze2Key])
		self.gaze2Data = np.swapaxes(self.gaze2Data, 0, 1)

		# independent variable 3(x,y,z) x #joints x #samples; only use 0-numJoint
		jointData = np.array(bodyData[jointKey])[:,0:numJoint,:]
		jointData = np.swapaxes(jointData, 0, 2)
		jointData = np.reshape(jointData, (len(jointData), 3*numJoint))

		jointVelData = np.array(bodyData[jointVelKey])[:,0:numJoint,:]
		jointVelData = np.swapaxes(jointVelData, 0, 2)
		jointVelData = np.reshape(jointVelData, (len(jointVelData), 3*numJoint))

		comData = np.array(bodyData[comKey])
		comData = np.swapaxes(comData, 0, 1)

		comVelData = np.array(bodyData[comVelKey])
		comVelData = np.swapaxes(comVelData, 0, 1)

		gaitData = np.array(bodyData[gaitKey])
		gaitData = np.swapaxes(gaitData, 0, 1)

		print("Loading data from files: %s" % (footholdFile))
		footholdData = sio.loadmat(footholdFile)[footholdKey]

		print("Loading data from files: %s" % (subjFile))
		heightData = sio.loadmat(subjFile)[heightKey]
		legData = sio.loadmat(subjFile)[legKey]

		# remove NaNs from gaze1 data
		finiteMask = np.isfinite(self.gaze2Data)[:,0]
		self.gaze2Data = self.gaze2Data[finiteMask]
		self.gaze1Data = self.gaze1Data[finiteMask]

		self.allIVData = np.concatenate((jointData, jointVelData, comData, comVelData, gaitData, 
			footholdData, heightData, legData), axis=1)
		self.allIVData = self.allIVData[finiteMask]
		print("Shape of IV matrix: ", self.allIVData.shape)

	def get_var_names(self):
		'''construct variable names'''
		variableInfo = "x1-75: joint x,y,z positions (25 joints)\n\
			x76-150: joint x,y,z velocities\n\
			x151-153: com x,y,z positions\n\
			x154-156: com x,y,z velocities\n\
			x157: gait percentage\n\
			x158-172: foothold\n\
			x173: height\n\
			x174: leg length"
		coords = 'xyz'
		self.varNames = ['constant']
		for name in self.jointNames:
			for c in coords:
				self.varNames.append(name + '.' + c)
		for name in self.jointNames:
			for c in coords:
				self.varNames.append(name + '.vel' + c)
		for c in coords:
			self.varNames.append('com.' + c)
		for c in coords:
			self.varNames.append('com.vel' + c)
		self.varNames.append('gaitPerc')
		for i in range(5):
			for c in coords:
				self.varNames.append('foothold' + str(i+1) + '.' + c)
		self.varNames.append('height')
		self.varNames.append('leglength')

	def regress_sklearn(self, method):
		#self.allIndVars= StandardScaler().fit_transform(self.allIndVars)

		if method == 'sk_ole':
			# ordinary least square
			reg = LinearRegression(normalize=True)
		elif method == 'sk_l2':
			# ridge
			reg = linear_model.Ridge(alpha=0.1,normalize=True)
		elif method == 'sk_l1':
			# lasso
			reg = linear_model.Lasso(alpha=0.01,normalize=True)
		else:
			print("No regression method specified; choose from ['ole'|'l1'|'l2']")
		
		print("Predicting gazeVec...")
		reg.fit(self.allIVData, self.gaze1Data)
		r2 = reg.score(self.allIVData, self.gaze1Data)
		adjR2 = calc_adj_r2(r2, self.allIVData.shape[0], self.allIVData.shape[1])
		print("r2, adjusted r2: %s, %s" % (r2, adjR2))

		print("Predicting gazeXZAll...")
		reg.fit(self.allIVData, self.gaze2Data)
		r2 = reg.score(self.allIVData, self.gaze2Data)
		adjR2 = calc_adj_r2(r2, self.allIVData.shape[0], self.allIVData.shape[1])
		print("r2, adjusted r2: %s, %s" % (r2, adjR2))


	def regress_statsmodel(self):
		X = sm.add_constant(self.allIVData)
		print("Regression result for 1st dimension")
		Y = self.gaze1Data[:,0]
		model = sm.OLS(Y,X)
		results = model.fit()
		print(results.summary(xname=self.varNames))
		print('*' * 120)

		print("Regression result for 2nd dimension")
		Y = self.gaze1Data[:,1]
		model = sm.OLS(Y,X)
		results = model.fit()
		print(results.summary(xname=self.varNames))
		print('*' * 120)

		print("Regression result for 3rd dimension")
		Y = self.gaze1Data[:,2]
		model = sm.OLS(Y,X)
		results = model.fit()
		print(results.summary(xname=self.varNames))

data = Data()
data.get_var_names()
data.regress_statsmodel()
