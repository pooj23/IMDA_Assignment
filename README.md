# IMDA_Assignment

1. captcha_script.py consists of the code to solve the problem
2. Assignment_Report.pdf is a brief summaru/explanation of the code and the approach taken.
3. results.txt contains the unseen captcha output results
4. sampleCaptches folder contains the raw and preprocessed data
    * input folder: Has 2 unseen captcha images
    * output folder: Has all the labels 
    * cleaned_input: Has images with the known labels where the labels are stored in the output folder 
    * preprocessed_output_folder: Contains the split up images for each letter with the label as the folder name (training and validation sets)
  
6. nn_model.hdf5 is the model used for identifying the unseen captchas
7. model_labels.dat contains mapping from labels to one-hot encodings for decoding the predictions

References:
1.	https://github.com/abdrabah/solve_captcha
2.	https://medium.com/@ageitgey/how-to-break-a-captcha-system-in-15-minutes-with-machine-learning-dbebb035a710

