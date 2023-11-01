# Some packages that may be useful
import numpy as np
from VideoGPT.scripts import train_videogpt, train_vqvae  # For training VideoGPT

from models import encoder, decoder, generational_network, classification_network
import data_handling

# Some pseudocode that may be helpful for getting started

# Set some parameters to define the models and training specs
training_folder = ""  # Location of training data
saving_folder = ""  # Location that the model will be saved to
val_split = 0.1  # Decimal percentage of training data to be used as validation
n_epochs = 100  # Number of epochs for training
lr = 0.0001  # Learning rate
# etc.

# Read in the images using data handling functions
train_data = []   # Training data array
val_data = []  # Validation data array

# Build the model
e = encoder(params)
d = decoder(params)
g = generational_network(params)
c = classification_network(params)

# Train the model
e.train()
d.train()
g.train()
c.train()

# Save the model
e.save_model()
d.save_model()
g.save_model()
c.save_model()
