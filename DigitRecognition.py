#Abigail Culkin
# References and adapted files from
"""
http://brianfarris.me/static/digit_recognizer.html
https://www.digitalocean.com/community/tutorials/how-to-build-a-neural-network-to-recognize-handwritten-digits-with-tensorflow
https://machinelearningmastery.com/handwritten-digit-recognition-using-convolutional-neural-networks-python-keras/
https://www.kaggle.com/ngbolin/mnist-dataset-digit-recognizer
"""

# imports
import gzip
import keras as kr
import numpy as np
import sklearn.preprocessing as pre
import matplotlib.pyplot as plt

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model

# Start neural network
model = kr.models.Sequential()

# Read in all the files
with gzip.open('data/t10k-images-idx3-ubyte.gz', 'rb') as f:
    test_images = f.read()
    
with gzip.open('data/t10k-labels-idx1-ubyte.gz', 'rb') as f:
    test_labels = f.read()
    
with gzip.open('data/train-images-idx3-ubyte.gz', 'rb') as f:
    training_images = f.read()
    
with gzip.open('data/train-labels-idx1-ubyte.gz', 'rb') as f:
    training_labels = f.read()

# Read all the files and save into memory
training_images = ~np.array(list(training_images[16:])).reshape(60000, 28, 28).astype(np.uint8) / 255.0
training_labels =  np.array(list(training_labels[8:])).astype(np.uint8)

test_images = ~np.array(list(test_images[16:])).reshape(10000, 784).astype(np.uint8) / 255.0
test_labels = np.array(list(test_labels[8:])).astype(np.uint8)

# Flatten the array , 784 neurons
inputs = training_images.reshape(60000, 784)


# Encode the data into binary values
encoder = pre.LabelBinarizer()
encoder.fit(training_labels)
outputs = encoder.transform(training_labels)

#Build our neural network model
# Adapted from https://machinelearningmastery.com/handwritten-digit-recognition-using-convolutional-neural-networks-python-keras/
# 5 layers - input layer, 3 hidden layer and 1 output layer
model = Sequential()

# Simple Neural Network with 3 layers (750, 512, 200)
model.add(kr.layers.Dense(units=750, activation='relu', input_dim=784))
model.add(kr.layers.Dense(units=512, activation='relu'))
model.add(kr.layers.Dense(units=200, activation='relu'))
# Compile model - Adam optimizer for our model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# Add 10 output neurons, one for each
model.add(kr.layers.Dense(units=10, activation='softmax'))



print("Would you like to train dataset? 'y' or load the data you have 'n'?")
option = input("y/n : ")
if option == 'y':
    #Train - The model is going to fit over 20 epochs and updates after every 100 images training.
    model.fit(inputs, outputs, epochs=20, batch_size=100)
    
    # Save the model - store my model on my local hard disk
    model.save("data/model.h5")

    from random import randint

    for i in range(20): #Run 20 tests
        print(i, encoder.transform([i]))
        
    #print out accuracy
    metrics = model.evaluate(inputs, outputs, verbose=0)
    print("Metrics(Test loss & Test Accuracy): ")
    print(metrics)

    # Evaluates and then prints error rate accuracy
    scores = model.evaluate(inputs, outputs, verbose=2)
    print("Error Rate: %.2f%%" % (100-scores[1]*100))

    print((encoder.inverse_transform(model.predict(test_images)) == test_labels).sum())

elif option == 'n':
    #load the model
    #Adapted from https://medium.com/coinmonks/handwritten-digit-prediction-using-convolutional-neural-networks-in-tensorflow-with-keras-and-live-5ebddf46dc8
     load_model('data/model.h5')
