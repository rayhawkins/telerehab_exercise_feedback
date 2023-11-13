# Some useful packages
import os  # for dealing with paths to files
import numpy as np  # for storing images
import skimage.io as skio  # for loading images


# Pseudocode to get started
# Function to read images from folders and subfolders
def read_images(path):
    im_array = []  # to store image sequences

    # Loop through each directory (including subdirectories) in the given path
    for root, dirs, files in os.walk(path):
        for file in files:
            # Check if the file is an image
            if file.endswith('.png'):
                # Construct the full path to the image file
                image_path = os.path.join(root, file)

                # Read the image using skimage
                image = skio.imread(image_path)

                # Append the image to the image array
                im_array.append(image)

    # Convert the list of images into a numpy array
    im_array = np.array(im_array)

    return im_array

# Example usage:
# Specify the path to the root folder containing images
path_to_images = '/path/to/images'

# Call the read_images function
image_sequences = read_images(path_to_images)
# Now 'image_sequences' contains sequences of images read from the specified path
