# Import stuff
import sys
sys.path.append(r'C:\Users\Ray\Documents\MASc\BME1570\code\telerehab_exercise_feedback\VideoGPT-master\videogpt')
from videogpt.vqvae import VQVAE
from videogpt.gpt import VideoGPT
from transformer_classifier import Classifier as TransformerClassifier
from convolutional_classifier import Classifier as ConvolutionalClassifier
import testing_functions


# User-specified parameters
gpt_path = ""
vqvae_path = ""
classifier_path = ""
test_data_path = ""

# set up args here, Ray can write later
###

data = VideoData(args)
test_data.test_dataloader()

vqvae_model = VQVAE(args)
classifier_model = TransformerClassifier(args)
gpt_model = VideoGPT(args)

for this_testdata in test_data:
    intermediate_output = apply vqvae
    classification = apply classifier
    generation = apply gpt

    metric1 = metricfunction(classification)
    metric2 = metricfunction2(classification)
    metric3 = metricfunction3(generation)
    ...

    # Plot metrics in graphs
