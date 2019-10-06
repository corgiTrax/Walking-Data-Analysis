# Walking-Data-Analysis
Analyze outdoor walking data

Software prerequisites:
- Keras
- Tensorflow
- numpy, scipy
- h5pyi
- sklearn, statsmodels

- [x] Setup experiment recording scripts
- [x] Condor scripts

# Gaze prediction task 1: classification of image patches
- [x] Get gaze vs. nongaze data
- [x] Basic CNN classification model
- [x] Data augmentation --  preprocess, reduce overfitting
- [x] Colored data
- [x] Max pooling?
- [x] Optical flow only
- [x] Colored data + optical flow
- [x] 32x32 data
- [x] Convolution dilation rate
- [ ] Fourier transform
- [x] Fully connected network
- [ ] GlobalAveragePooling2D?
- [ ] Load pre-trained network?

# Gaze prediction task2: body-joint model
- [x] Body joint prediction: regression model in sklearn
- [x] Regression model interpretation: regression model in statsmodel
- [x] Variable names
- [ ] Multicolinearity issue
- [ ] Time series issue
- [ ] Statsmodel regression with multi-dimensional output
- [x] MLP model
- [ ] Sequence model

# Gaze prediction task3: joint image-body model
- [ ] Joint network
- How to do this:
  - from body-model-1condition.py one can see how to read in body joints from file, and how to do regression using a fully connected network
  - from cnn-img3ch+opf.py one can see how to build a convolutional network with multiple input sources and fuse them
  - from these two above should be able to build a joint network that takes in both image and joints, and predict gaze


