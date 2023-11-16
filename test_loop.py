# Import stuff
import sys
sys.path.append(r'C:\Users\Ray\Documents\MASc\BME1570\code\telerehab_exercise_feedback\VideoGPT-master\videogpt')
import torch.nn as nn
from videogpt.vqvae import VQVAE
from videogpt.gpt import VideoGPT
from transformer_classifier import Classifier as TransformerClassifier
from convolutional_classifier import Classifier as ConvolutionalClassifier
import testing_functions
import matplotlib.pyplot as plt

# User-specified parameters
gpt_path = ""
vqvae_path = ""
classifier_path = ""
test_data_path = ""

# set up args here, Ray can write later
###

# Data loaders
data = VideoData(args)
test_data = testing_functions.test_dataloader(args)

vqvae_model = VQVAE(args)
classifier_model = TransformerClassifier(args)
classifier_convolution_model = ConvolutionalClassifier(args)
gpt_model = VideoGPT(args)


#Iterate test data
for this_testdata in test_data:

    #Apply VQVAE model
    intermediate_output = vqvae_model(this_testdata)
    #Apply Transformer Classifier model
    classification_transformer = classifier_model(intermediate_output)
    #Apply Convolutional Classifier model
    classification_convolution = classifier_convolution_model(intermediate_output)
    #Apply GPT model
    generation = gpt_model(intermediate_output)

    metric1 = metric_function(classification_transformer)
    metric2 = metric_function2(classification_convolution)
    metric3 = metric_function3(generation)
    ...

    # Plot metrics in graphs
    metric_names = ['Metric1', 'Metric2', 'Metric3']
    metric_values = [metric1, metric2, metric3]

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.bar(metric_names, metric_values, color=['blue', 'orange', 'green'])
    plt.xlabel('Metrics')
    plt.ylabel('Metric Values')
    plt.title('Metrics Comparison')
    plt.show()

