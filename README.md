# EmergingTechnologies
As part of my module Emerging Technologies for final year of software development.

## Description
I had to put together 3  Jupyter notebooks explaining what i did and using the programming language Python. 

### Numpy Random Notebook
NumPy is the fundamental package in Python. NumPy’s main object is the homogeneous multidimensional array. It included mathematical, logical, shape manipulation, sorting, selecting, I/O, discrete Fourier transforms, basic linear algebra, basic statistical operations, random simulation and much more.

I used the numpy randon package and had to use a numpy imort in pyhton.
I was able to create arrays with numpy, reshaping them and plotting then without making any changes.

Math ploit import allows me to plot on my graph.
```
	import numpy as np
  import matplotlib.pyplot as plt
```

### Iris Dataset Notebook
This was the second notebook I did it is a database that uses a pattern recognition. It uses multiple measurements in taxonomic problems as an example of linear discriminant analysis.
It contains 50 observations of each species (setosa, versicolor, virginica) and has 150 records in petal length, petal width, sepal length, sepal width and species. These measurements are stored in a table and can be used to predict what type of species it is. 
I have plotted out the measurements out in graphs to explain the different lengths and widths in each species.

We also built a neural network to predict the species of the iris flowers in the dataset.

The data in the dataset was loaded from a URL in the notebook.

The import used below was to be able to analsye data in python
```
	import pandas as pd
```


### Mnist dataset Notebook
 This is a dataset of handwritten used for training various image processing systems. It alows you to read images from 0-9 and is recognized from image and label training and testing files. In this notebook we unzipped the files I downloaded containing 60,000 training images and 10,000 testing images. Then I was able to read the the file bytes to read in an image and label so they can be recognised.
It prints the handwritten digit on a grid.
Then I have saved the images as a png file to be saved locally to my memory.

We can change the range to 10000 for the entire array of images
```
	for x in range(20):
    plt.imshow(~imgArray[x], cmap="gray")
    plt.savefig('images/savedImg-index' + str(x) + '-Label-' + str(labArray[x]))
```

### Digit Recognition Notebook
This note book explains my script i did for recognising Handwritten digits and running the training files to get the accuracy of reading the images and labels. For this I created a neural network with layers to be able to read through the files. The tas was to create a neural network that works as accurately as possible to ensure each of the 10000 images are recognised or as close as possible.

Gzip opens the files we need

```
	with gzip.open('data/t10k-images-idx3-ubyte.gz', 'rb') as f:
    test_images = f.read()
```

```
	#Train - The model is going to fit over 20 epochs 
#and updates after every 100 images training.

model.fit(inputs, outputs, epochs=20, batch_size=100)
```
This runs the test and prints out the accuracy and images recognised.

## Digit Recognition Python Script
This script is the same as the notebook I explained above.
It gives you the option of a menu to train the data set or just the load the current model you have. 

It is run in the command promt and will run through the tests here if you choose to.

```
	print("Would you like to train dataset? 'y' or load the data you have 'n'?")
option = input("y/n : ")

```
It will run through only 20 sets and that can be changed to train all data.

## Prerequisites

I used github for my project so it would not be lost and be easy for other people to access.

### Push to Github:

In order to submit my project changes to github from my github folder i used the following commands:
git add .
git commit -m "Initial commit"
git push

### Download from github:
For you to download my project you must clone my repository link from the command promp:

git clone "example.github/project"

### Running my Notebooks

After cloning my github you must have jupyter installed on your machine.
Here you open the command promt to the file location which contains all the notebooks.
you then enter command: jupyter notebook
This will open the notebooks in your browser and they ccan be viewed. 
In order to run the Mnist you must have the data files saved in a folder as I have or change the path name. 

The folder I have unzipped the filed in is data.
  ('data/t10k-images-idx3-ubyte.gz', 'rb')
  
### Running my Script
You must have python installed on your machine
Again you will open the command prompt to the file location which contains the script.
You run the script from the terminal using : python DigitRecognition.py
As my script is called. 

## Coding Syle

I have used python for all above projects.

A simple print line in python
```
	print (iris.target.shape)
```

##Refernces
https://docs.scipy.org/doc/numpy/user/whatisnumpy.html
https://joomik.github.io/MNIST/
https://towardsdatascience.com/a-quick-introduction-to-the-pandas-python-library-f1b678f34673
