import pandas as pd
from pathlib import Path
import os
from itertools import chain
import fnmatch
import shutil


class Captcha():
    def __init__(self, im_path, save_path):
        self.im_path = im_path
        self.save_path = save_path
        self.data_dir = self.im_path / 'input'
        self.data_labels = self.im_path / 'output'
        self.data_new_input = self.im_path / 'cleaned_input'
        self.data_new_input.mkdir(parents=True, exist_ok = True)

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
 
        characters = set(char for label in labels for char in label)

        print("Number of images found: ", len(images))
        print("Number of labels found: ", len(labels))
        print("Number of unique characters:", len(characters))
        print("Characters present:", characters)
        print("labels:", labels)


        """
        Algo for inference
        args:
            im_path: .jpg image path to load and to infer
            save_path: output file path to save the one-line outcome
        """
        return 1

    def run_all(self):
        self.clean_data()
        self.load_data()

def main():
    im_path = Path("../IMDA_Assignment/sampleCaptchas")
    save_path = Path("../IMDA_Assignment")

    Captcha(im_path, save_path).run_all()

if __name__ == "__main__":
    main()