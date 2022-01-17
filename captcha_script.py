import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
from collections import Counter

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from itertools import chain
import fnmatch
import shutil

from skimage.io import imread

import gc

import glob
import h5py
import itertools
import random as rn

import seaborn as sns
from pathlib import Path
from collections import Counter

from skimage.transform import resize
from sklearn.metrics import confusion_matrix
#from mlxtend.plotting import plot_confusion_matrix

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam, SGD, RMSprop

from keras.models import Sequential, Model, load_model
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout
from keras.layers import Input, Flatten, BatchNormalization, Lambda
from keras.layers import CuDNNGRU, CuDNNLSTM, Bidirectional, LSTM, GRU
from keras.layers import Add, Concatenate, Reshape

from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from tensorflow.keras.utils import to_categorical

from keras import backend as K

import cv2
import pickle
import os.path
import numpy as np
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten, Dense


class Captcha():
    def __init__(self, im_path, save_path):
        self.im_path = im_path
        self.save_path = save_path
        self.data_dir = self.im_path / 'input'
        self.data_labels = self.im_path / 'output'
        self.data_new_input = self.im_path / 'cleaned_input'
        self.data_new_input.mkdir(parents=True, exist_ok = True)
        self.model_labels = self.save_path / "model_labels.dat"
        self.model_path = self.save_path / 'nn_model.hdf5'


    def clean_data(self):

        # Get a list of all files in directory
        for rootDir, subdirs, filenames in os.walk(self.data_labels):
            # Find the files that matches the given patterm
            for filename in fnmatch.filter(filenames, '*.txt'):
                filename = filename.replace('.txt', '.jpg').replace("out", "in")
                try:
                    old_path = f"{self.data_dir}\{filename}"
                    new_path = f"{self.data_new_input}\{filename}"
                    shutil.move(old_path, new_path)
                except OSError:
                    print("Error while moving file")

    def load_data(self):

        images = sorted(list(map(str, list(self.data_new_input.glob("*.jpg")))))
        unseen_captchas = sorted(list(map(str, list(self.data_dir.glob("*.jpg")))))

        labels = []
        for file in os.listdir(self.data_labels):
            # Check whether file is in text format or not
            if file.endswith(".txt"):
                file_path = f"{self.data_labels}\{file}"
                with open(file_path, 'r') as f:
                    #print(f.read())
                    labels.append(f.read())
        
        labels = [elem.strip().split('\n') for elem in labels]
        labels = list(chain(*labels))
 
        letters = set(char for label in labels for char in label)

        print("Number of images found: ", len(images))
        print("Number of labels found: ", len(labels))
        print("Number of unique characters:", len(letters))
        print("Characters present:", letters)
        print("labels:", labels)
        print("Number of unseen captchas found: ", len(unseen_captchas))
        print(unseen_captchas)

        max_length = max([len(label) for label in labels])

        sample_images = images[:4]

        f,ax = plt.subplots(2,2, figsize=(5,3))
        for i in range(4):
            img = imread(sample_images[i])
            print("Shape of image: ", img.shape)
            ax[i//2, i%2].imshow(img)
            ax[i//2, i%2].axis('off')
        plt.show()
        data = []
        for image_path in images:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Resize the letter so it fits in a 20x20 pixel box
            #image = resize_to_fit(image, 20, 20)
            # Add a third channel dimension to the image
            image = np.expand_dims(image, axis=2)
            # Add the letter image and it's label to our training data
            data.append(image)

        data = np.array(data, dtype="float") / 255.0
        labels = np.array(labels)

        # Split the training data into separate train and test sets
        (x_train, x_valid, y_train, y_valid) = train_test_split(data, labels, test_size=0.25, random_state=0)

        # Convert the labels (letters) into one-hot encodings that Keras can work with
        lb = LabelBinarizer().fit(y_train)
        y_train = lb.transform(y_train)
        y_valid = lb.transform(y_valid)

        # Save the mapping from labels to one-hot encodings.
        # We'll need this later when we use the model to decode what it's predictions mean
        with open(self.model_labels, "wb") as f:
            pickle.dump(lb, f)

        print("x_train: ", x_train)
        print("y_train: ", y_train)

        print("Total number of images in the training set: ", len(x_train))
        print("Total number of labels in the training set: ", len(y_train))
        print("Total number of images in the validationn set: ", len(x_valid))
        print("Total number of lbels in the validationn set: ", len(y_valid))

        return x_train, y_train, x_valid, y_valid, unseen_captchas
    
    def nn_model(self, x_train, y_train, x_valid, y_valid):

        # Build the neural network!
        model = Sequential()

        # First convolutional layer with max pooling
        model.add(Conv2D(20, (5, 5), padding="same", input_shape=(30, 60, 1), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # Second convolutional layer with max pooling
        model.add(Conv2D(50, (5, 5), padding="same", activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # Hidden layer with 500 nodes
        model.add(Flatten())
        model.add(Dense(500, activation="relu"))

        # Output layer with 32 nodes (one for each possible letter/number we predict)
        model.add(Dense(18, activation="softmax"))

        # Ask Keras to build the TensorFlow model behind the scenes
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

        model.fit(x_train, y_train, validation_data=(x_valid, y_valid), batch_size=4, epochs=10, verbose=1)

        model.save(self.model_path)

        return 1

    def run_all(self):
        #self.clean_data()
        x_train, y_train, x_valid, y_valid, unseen_captchas = self.load_data()
        self.nn_model(x_train, y_train, x_valid, y_valid)
        

def main():
    im_path = Path("../IMDA_Assignment/sampleCaptchas")
    save_path = Path("../IMDA_Assignment")

    Captcha(im_path, save_path).run_all()

if __name__ == "__main__":
    main()