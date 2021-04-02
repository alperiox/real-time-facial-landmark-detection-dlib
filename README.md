# Real-time facial landmark detection with DLIB
Real-time facial landmark detection with HOG + Pretrained facial landmark detector in dlib

## Installation:
### Requirements:
* dlib==19.22.0
* numpy==1.20.2
* opencv-contrib-python==4.5.1.48


Installing required packages from `requirements.txt` file by running `pip install -r requirements.txt` from your command line should be enough for the installation. 

In order to run the script, you should also download pretrained landmark detector from:
> http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

After that, you can import the pretrained model by specifying the downloaded model path in `predictor_path` variable
