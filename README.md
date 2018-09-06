# ZaloAiLandMark
Public source code of top5 ZaloAI LandMark challenge.
Pretrained networks could be download in [here](https://drive.google.com/drive/folders/1g5V-4TUJbNk5c8A0LG60i8SkkWX7rHbr?usp=sharing)
This tutorial consists of 4 parts:
-	Dependencies
-	Data processing.
-	Training.
-	Prediction.

I.	Dependencies
-	Python 3.5.2
-	Keras 2.2.0
-	Tensorflow-gpu 1.4.1
-	Opencv 3.4.1
-	Numpy 1.14.5
All the packages are built on Linux.

II.	Data processing
1.	Overview
-	Clean the dataset (remove junk file, error file, etc).
-	Splits the training dataset into 2 sets:
+ Training set (90% of the dataset).
+ Validation set (10% of the dataset).

2.	Coding implementation
-	To clean the dataset, we run: 
python3 prepare_image.py
+ Set up the data link in line 9.
-	To splits the dataset, we run:
python3 splitdata_python.py

III.	Training
1.	Overview
-	This model is a cascade model of 4 networks: InceptionResnetV2, Xception, Resnet50 and Resnet101 network.
-	In all the networks, we used pretrained model based on Imagenet classification.
-	Some data augmentation techniques have been using:  
+ rescale=1/255
+ rotation_range=20
+ width_shift_range=0.2
+ height_shift_range=0.2
+ shear_range=0.2
+ zoom_range=0.2
+ horizontal_flip=True
+ fill_mode='nearest'
-	It took 36 hours for each sub-model (~20 epochs) training on one NVIDIA GeForce GTX 1080 Ti.

2.	Coding implementation
-	To train the InceptionResnetV2/Xception/Resnet101 network, we run:
python3 train_inceptionresnet.py/train_xception.py/train_resnet50.py /train_resnet101.py
with some modifications:
+ Load pretrained model:
•	train_inceptionresnet.py: Line 85.
•	train_xception.py: Line 85.
•	train_resnet50.py: Line 85.
•	train_resnet101.py: Line 168.
+ Load dataset:
•	train_inceptionresnet.py: Line 18, 19.
•	train_xception.py: Line 18, 19.
•	train_resnet50.py: Line 18, 19.
•	train_resnet101.py: Line 126, 127.

IV.	Prediction
1.	Overview
-	At first, we run the prediction separately on 3 networks, which creates 3 numpy prediction matrix.
-	Then, we take the sum and calculate the final result (top-3 predictions) using softmax.

2.	Coding implementation
-	To get the prediction matrix of 3 networks, we run:
•	python3 prediction_inceptionresnet.py
•	python3 prediction_xception.py
•	python3 prediction_resnet50.py
•	python3 prediction_resnet101.py
with some modifications:
+ Load trained model:
•	prediction_inceptionresnet.py: Line 32.
•	prediction_xception.py: Line 32.
•	prediction_resnet50.py: Line 32.
•	prediction_resnet101.py: Line 127.
+ Load test dataset:
•	prediction_inceptionresnet.py: Line 37.
•	prediction_xception.py: Line 37.
•	prediction_resnet50.py: Line 37.
•	prediction_resnet101.py: Line 131.

-	To get the final result csv file in the result section, we run: python3 predict_cascade.py

