import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset
import numpy as np 

# loading in cat images
(
    train_set_x_orig, 
    train_set_y,
    test_set_x_orig,
    test_set_y,
    classes 
) = load_dataset()

# flattening the 64x64x3 images into vectors
train_set_x_flatten = train_set_x_orig.reshape(
        train_set_x_orig.shape[0], -1).T

test_set_x_flatten = test_set_x_orig.reshape(
        test_set_x_orig.shape[0], -1).T

# normalising the images (pixel values 0-255)
train_set_x = train_set_x_flatten / 255
test_set_x = test_set_x_flatten / 255

m_train = 209   # number of training examples
m_test = 50    # number of test examples
dim = 12288    # length of image vector

l_1 = 4    # neurons in layer one
l_2 = 1    # neurons in layer two

# sigmoid function
def sigmoid(z):

    return 1 / (1 + np.exp(-z))
    
# initializing weights and biases as random
def initialize(layer_dim, m_train):
    w = np.random.rand(layer_dim, m_train)
    b = np.random.rand(layer_dim)
    return w, b
