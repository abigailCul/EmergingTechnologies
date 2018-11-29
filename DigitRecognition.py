# imports
import gzip
import keras as kr
import numpy as np
import sklearn.preprocessing as pre
import matplotlib.pyplot as plt

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils

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

# Adapted from
model = Sequential()

model.add(kr.layers.Dense(units=750, activation='relu', input_dim=784))
model.add(kr.layers.Dense(units=512, activation='relu'))
model.add(kr.layers.Dense(units=200, activation='relu'))
model.add(kr.layers.Dense(units=120, activation='relu'))
#model.add(kr.layers.Dense(units=50, activation='relu'))
# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



# Add 10 output neurons, one for each
model.add(kr.layers.Dense(units=10, activation='softmax'))



print("Would you like to train dataset?")
option = input("y/n : ")
if option == 'y':
    #Train
    model.fit(inputs, outputs, epochs=20, batch_size=100)
    
    # Save the model
    model.save("data/model.h5")

    
    from random import randint

    for i in range(20): #Run 20 tests
        print(i, encoder.transform([i]))
    scores = model.evaluate(inputs, outputs, verbose=2)
    print("Error Rate: %.2f%%" % (100-scores[1]*100))

elif option == 'n':
   model.load_weights("data/model.h5")
