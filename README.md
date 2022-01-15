# IMDA_Assignment
Test 2
(A) Task 
Note: No advanced computer vision background is required to solve this challenge. A simple understanding of the 256 x 256 x 256 RGB color space is sufficient.

A website uses Captchas on a form to keep the web-bots away. However, the captchas it generates, are quite similar each time:
- the number of characters remains the same each time  
- the font and spacing is the same each time  
- the background and foreground colors and texture, remain largely the same
- there is no skew in the structure of the characters.  
- the captcha generator, creates strictly 5-character captchas, and each of the characters is either an upper-case character (A-Z) or a numeral (0-9).

Here, take a look at some of the captcha images on the form. As you can see, they resemble each other very much - just that the characters on each of them are different.
 	 	 	 
You are provided a set of twenty-five captchas, such that, each of the characters A-Z and 0-9 occur at least once in one of the Captchas' text. From these captchas, you can identify texture, nature of the font, spacing of the font, morphological characteristics of the letters and numerals, etc. Download this sample set from here for the purpose of creating a simple AI model or algorithm to identify the unseen captchas.

(B) Deliverable 
1.	README.md
2.	Python code (use the following template and feel free to add the necessary methods)
class Captcha(object):
    def __init__(self):
        pass

    def __call__(self, im_path, save_path):
        """
        Algo for inference
        args:
            im_path: .jpg image path to load and to infer
            save_path: output file path to save the one-line outcome
        """
        pass
We would like to learn more about the way you frame the problem and formulate the solution