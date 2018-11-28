# imports
import gzip
import keras as kr
import numpy as np
import sklearn.preprocessing as pre
import matplotlib.pyplot as plt

# Start neural network
model = kr.models.Sequential()

# Add a hidden layer with 1000 neurons and an input layer with 784
model.add(kr.layers.Dense(units=1000, activation='linear', input_dim=784))
model.add(kr.layers.Dense(units=400, activation='relu'))

# Add 10 output neurons, one for each
model.add(kr.layers.Dense(units=10, activation='softmax'))

# Build the network graph
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])


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

model.fit(inputs, outputs, epochs=20, batch_size=100)

for i in range(20): #Run 20 tests
        print(i, encoder.transform([i]))
 
