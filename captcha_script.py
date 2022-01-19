import cv2
import os
import fnmatch
import shutil
import pickle
import os.path
import numpy as np
import imutils
import matplotlib.pyplot as plt
from skimage.io import imread
from itertools import chain
from pathlib import Path
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model, load_model
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

        # To view the images that have been extracted:
        sample_images = images[:4]

        f,ax = plt.subplots(2,2, figsize=(5,3))
        for i in range(4):
            img = imread(sample_images[i])
            #print("Shape of image: ", img.shape)
            ax[i//2, i%2].imshow(img)
            ax[i//2, i%2].axis('off')
        plt.show()       

        #for image_path in images:
        for (i, image_path) in enumerate(images):

            print("[INFO] processing image {}/{}".format(i + 1, len(images)))
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Add some extra padding around the image
            image = cv2.copyMakeBorder(image, 8, 8, 8, 8, cv2.BORDER_REPLICATE)
            print(image.shape)
            image = resize_to_fit(image, 30, 60)
            # threshold the image (convert it to pure black and white)
            thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            # find the contours (continuous blobs of pixels) the image
            contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            # Hack for compatibility with different OpenCV versions
            contours = contours[0] if imutils.is_cv2() else contours[1]

            letter_image_regions = []

            print("contours", len(contours))

            # Now we can loop through each of the four contours and extract the letter
            # inside of each one
            for contour in contours:
                # Get the rectangle that contains the contour
                (x, y, w, h) = cv2.boundingRect(contour)

                # Compare the width and height of the contour to detect letters that are conjoined into one chunk
                if w / h > 2:
                    # This contour is too wide to be a single letter! Split it in half into two letter regions!
                    half_width = int(w / 2)
                    letter_image_regions.append((x, y, half_width, h))
                    letter_image_regions.append((x + half_width, y, half_width, h))
                else:
                    letter_image_regions.append((x, y, w, h))

            # Sort the detected letter images based on the x coordinate to make sure
            # we are processing them from left-to-right so we match the right image
            # with the right letter
            letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])

            # Save out each letter as a single image
            for letter_bounding_box, letter_text in zip(letter_image_regions, labels[i]):
                # Grab the coordinates of the letter in the image
                x, y, w, h = letter_bounding_box
                # Extract the letter from the original image with a 2-pixel margin around the edge
                letter_image = image[y - 2:y + h + 2, x - 2:x + w + 2]
                letter_image = resize_to_fit(letter_image, 20, 20)
                # Get the folder to save the image in            
                save_path = self.im_path / 'preprocessed_output_folder' / ('%s' % letter_text) 
                # if the output directory does not exist, create it
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                print("letter_text", letter_text)
                # write the letter image to a file
                p = os.path.join(save_path, "{}.png".format(str(letter_text)))
                cv2.imwrite(p, letter_image)

        return 1

    def processing_training_data(self):
        
        data = []
        labels = []
        for image_path in paths.list_images(self.im_path / 'preprocessed_output_folder'):
            print(image_path)

            # Load the image and convert it to grayscale
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Resize the letter so it fits in a 20x20 pixel box
            image = resize_to_fit(image, 20, 20)
            # Add a third channel dimension to the image
            image = np.expand_dims(image, axis=2)
            # Add the letter image and it's label to our training data
            label = image_path.split(os.path.sep)[-2]
            data.append(image)
            labels.append(label)

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

        print("Total number of images in the training set: ", len(x_train))
        print("Total number of labels in the training set: ", len(y_train))
        print("Total number of images in the validationn set: ", len(x_valid))
        print("Total number of lbels in the validationn set: ", len(y_valid))

        return x_train, y_train, x_valid, y_valid
    
    def nn_model(self, x_train, y_train, x_valid, y_valid):

        # Build the neural network!
        model = Sequential()

        # First convolutional layer with max pooling
        model.add(Conv2D(20, (5, 5), padding="same", input_shape=(20, 20, 1), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # Second convolutional layer with max pooling
        model.add(Conv2D(50, (5, 5), padding="same", activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # Hidden layer with 500 nodes
        model.add(Flatten())
        model.add(Dense(500, activation="relu"))

        # Output layer with 32 nodes (one for each possible letter/number we predict)
        model.add(Dense(27, activation="softmax"))

        # Ask Keras to build the TensorFlow model behind the scenes
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

        model.fit(x_train, y_train, validation_data=(x_valid, y_valid), batch_size=6, epochs=150, verbose=1)

        model.save(self.model_path)

        return 1

    def read_unseen_captchas(self):

        # Load up the model labels (so we can translate model predictions to actual letters)
        with open(self.model_labels, "rb") as f:
            lb = pickle.load(f)

        # Load the trained neural network
        model = load_model(self.model_path)
        results_file = self.save_path / 'results.txt'
        file1 = open(results_file,"w")
        
        unseen_captchas = sorted(list(map(str, list(self.data_dir.glob("*.jpg")))))
        print("Number of unseen captchas: ", len(unseen_captchas))
        print(unseen_captchas)
        
        for image_path in unseen_captchas:
            print("\n For: \n", image_path)

            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Add some extra padding around the image
            image = cv2.copyMakeBorder(image, 8, 8, 8, 8, cv2.BORDER_REPLICATE)
            image = resize_to_fit(image, 30, 60)
            #image = np.expand_dims(image, axis=2)
            thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            # find the contours (continuous blobs of pixels) the image
            contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            # Hack for compatibility with different OpenCV versions
            contours = contours[0] if imutils.is_cv2() else contours[1]
            letter_image_regions = []

            # Now we can loop through each of the four contours and extract the letter
            # inside of each one
            for contour in contours:
                # Get the rectangle that contains the contour
                (x, y, w, h) = cv2.boundingRect(contour)
                # Compare the width and height of the contour to detect letters that are conjoined into one chunk
                if w / h > 1.25:
                    # This contour is too wide to be a single letter! Split it in half into two letter regions!
                    half_width = int(w / 2)
                    letter_image_regions.append((x, y, half_width, h))
                    letter_image_regions.append((x + half_width, y, half_width, h))
                else:
                    letter_image_regions.append((x, y, w, h))

            print("Number of Rectangles", len(letter_image_regions))
            letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])
            predictions = []

            # loop over the letters
            for letter_bounding_box in letter_image_regions:
                # Grab the coordinates of the letter in the image
                x, y, w, h = letter_bounding_box
                # Extract the letters
                letter_image = image[y - 2:y + h + 2, x - 2:x + w + 2]
                # Re-size the letter image to 20x20 pixels to match training data
                letter_image = resize_to_fit(letter_image, 20, 20)
                # Turn the single image into a 4d list of images to make Keras happy
                letter_image = np.expand_dims(letter_image, axis=2)
                letter_image = np.expand_dims(letter_image, axis=0)
                # Ask the neural network to make a prediction
                prediction = model.predict(letter_image)
                # Convert the one-hot-encoded prediction back to a normal letter
                letter = lb.inverse_transform(prediction)[0]
                predictions.append(letter)

            # Print the captcha's text
            captcha_text = " ".join(predictions)
            print("CAPTCHA text is: {}".format(captcha_text))

            file1.writelines(captcha_text + "  for  " + image_path + "\n")
        
        file1.close() #to change file access mod   


    def run_all(self):
        #self.clean_data()
        self.load_data()
        x_train, y_train, x_valid, y_valid = self.processing_training_data()
        self.nn_model(x_train, y_train, x_valid, y_valid)
        self.read_unseen_captchas()
        
def resize_to_fit(image, width, height):
    """
    A helper function to resize an image to fit within a given size
    :param image: image to resize
    :param width: desired width in pixels
    :param height: desired height in pixels
    :return: the resized image
    """
    # grab the dimensions of the image, then initialize
    # the padding values
    (h, w) = image.shape[:2]
    # if the width is greater than the height then resize along
    # the width
    if w > h:
        image = imutils.resize(image, width=width)
    # otherwise, the height is greater than the width so resize
    # along the height
    else:
        image = imutils.resize(image, height=height)
    # determine the padding values for the width and height to
    # obtain the target dimensions
    padW = int((width - image.shape[1]) / 2.0)
    padH = int((height - image.shape[0]) / 2.0)
    # pad the image then apply one more resizing to handle any
    # rounding issues
    image = cv2.copyMakeBorder(image, padH, padH, padW, padW, cv2.BORDER_REPLICATE)
    image = cv2.resize(image, (width, height))
    # return the pre-processed image
    return image

def main():

    im_path = Path("../IMDA_Assignment/sampleCaptchas")
    save_path = Path("../IMDA_Assignment")

    Captcha(im_path, save_path).run_all()

if __name__ == "__main__":
    main()