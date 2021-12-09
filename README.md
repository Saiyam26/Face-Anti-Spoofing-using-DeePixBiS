# Face Anti-Spoofing using Deep-Pixel-wise-Binary-Supervision

- Anti-Spoofing for Face Recognition task using the Deep Pixel-wise Binary Supervision Technique. The paper can be found here https://arxiv.org/pdf/1907.04047v1.pdf
- This Project implements the DeePixBiS model using Python OpenCV, and the Pytorch Framework. This project is inspired from https://github.com/voqtuyen/deep-pix-bis-pad.pytorch
- The Trained weights are already saved up as './DeePixBiS.pth' file which can be run on the model.
- Training Data has been taken from the NUAA Imposter dataset (863 images subset)

### Deep Pixel-wise Binary Supervision
This framework uses CNN and densely connected neural network trained using both binary and pixel-wise binary supervision simultaneously.
This is a frame level algorithm, which performs the task individually and independently on each frame, thus making computation and time feasable for practical use.
Each pixel/patch of the frame is given a binary label depending on whether it is bonafide or an attack, trying to generate the depth-map of the image. Note that this framework does not generate a precise depth map, rather it does not need to. In the testing phase, the mean of this feature map is used as the score. If the score is greater than a threshold value, it is declared to be real.
The model architecture uses the first 8 layers of the DenseNet-161 architecture, for feature extraction. 

### About the Project

We use the OpenCV library for the image preproccsing for the model. OpenCV offers several cascades for the task of object Detection. We use the Frontal-Face Haar Cascade to detect a "face" in the frame. Once a face is detected it has a bounded box to find its location, and the face is extracted, leaving aside the other non-important details in the frame. The training-data(frames) ready to pass through the model is trained using the Adam Optimizer. 
The Loss function is a weighted sum using the binary and pixel-wise binary cross-entropy loss function.


### Requirements

- Python 3.6+
- OpenCV
- Numpy
- PyTorch

### Training the Model
1. Run `python Train.py`
2. After Training is complete the program will generate the file "./DeePixBiS.pth", containing weights of the model

### Recognizing
1. Run `python Test.py`

### TODO
1. Make directories for easy handling of python files.
2. Add a config file for easy hyperparameters tuning.
