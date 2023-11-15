# Some packages that may be useful
import numpy as np
import pandas as pd

from models import encoder, decoder, generational_network, classification_network
import testing_functions
import data_handling

# Some pseudocode that may be helpful for getting started

# Set up parameters
model_folder = ""  # Folder containing the saved model
testing_folder = ""  # Folder where testing data is located
save_folder = ""  # Folder where results will be saved to

# Load in the testing data
test_data = data_handling.read_images(testing_folder)

# Load the model
e = encoder.load_model(model_folder)
g = generational_network.load_model(model_folder)
c = classification_network.load_model(model_folder)

# Apply the model to the test data
features = e.predict(test_data)
predicted_frames = g.predict(features)
classification = c.predict(features)

# Quantify performance
# Look at evaluation metrics used by https://github.com/wilson1yan/VideoGPT for generative network performance
testing_functions.classification_accuracy(classification, test_data)
testing_functions.classification_f1score(classification, test_data)
testing_functions.prediction_pixelwise_accuracy(classification, test_data)
#Confusion matrix for Binary classification - we need to repeat this line, 9 times for creating 9 confusion matrices for all classes
testing_functions.create_confusion_matrix(ground_truth_1, prediction_1)
testing_functions.create_confusion_matrix(ground_truth_2, prediction_2)
testing_functions.create_confusion_matrix(ground_truth_3, prediction_3)
#confusion matrix for 9 classes classification
testing_functions.create_confusion_matrix(ground_truth_all, prediction_all)
# etc.

# Save the results in some format (as .csv from pandas table maybe?)

