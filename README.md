### Face-Recognition-using-FaceNet
FaceNet is a Deep Learning architecture consisting of convolutional layers based on GoogLeNet inspired inception models.
This model is loaded with pretrained weights of FaceNet inception model trained using Siamese triple loss function.

How to Run: 
1. Prepare database with person images to recognize in future.
$ python prepare_data.py

2. Recognize new image of webcam.
$ python face_recognizer.py

Read:
1. Inception neural network.ipynb contains Architecture of Inception model.
2. Siamese NN.ipynb contains implementation of Siamese Network.


