# Some packages that may be useful
from VideoGPT.scripts import train_vqvae, train_videogpt  # Training scripts for VideoGPT
from VideoGPT.videogpt import gpt, resnet, utils, vqvae  # Model definitions for VideoGPT
import numpy as np
from abc import ABC, abstractmethod  # For defining the general structure of a model class


# Some code and pseudocode for getting started

# Create an abstract class for any model type
# Abstract classes are nice because any class defined using the abstract model as a parent will inherit the abstract
# model's methods, meaning we can define load and save functions here that are used by all models without having to
# redefine them for every model
class AbstractModel(ABC):
    @abstractmethod
    def __init__(self, params):
        # Build model using given params
        pass
    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def build_model(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def save_model(self):
        pass

    @abstractmethod
    def load_model(self):
        pass


class encoder(AbstractModel):  # For bringing images down into a feature space (k)


class decoder(AbstractModel):  # For training the encoder by reconstructing original images from feature space k


class generational_network(AbstractModel):  # For generating predicted frames corresponding to the given exercise for feature space k


class classification_network(AbstractModel):  # For classifying the feature space k as an incorrect or correctly performed exercise






