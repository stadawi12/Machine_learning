import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

m_train = 209
m_test = 50
num_px = 64

train_set_x_flatten = train_set_x_orig.reshape(
        train_set_x_orig.shape[0], -1).T

test_set_x_flatten = test_set_x_orig.reshape(
        test_set_x_orig.shape[0], -1).T

train_set_x = train_set_x_flatten / 255
test_set_x = test_set_x_flatten / 255

def sigmoid(z):
    s = 1 / ( 1 + np.exp(-z) )
    return s

def initialize_with_zeros(dim):
    w = np.zeros((dim,1))
    b = 0
    return w, b


def propagate(w, b, X, Y):
    m = X.shape[1]

    A = sigmoid(np.dot(w.T, X) + b)
    cost = -(np.dot(Y, np.log(A).T) + np.dot(
        1 - Y, np.log(1 - A).T)) / m 

    dw = np.dot(X, (A-Y).T) / m
    db = np.sum(A - Y) / m

    cost = np.squeeze(cost)

    grads = {"dw" : dw,
             "db" : db}
    return grads, cost

def optimize(w, b, X, Y, num_iterations, learning_rate,
        print_cost = False):

    costs = []
    
    for i in range(num_iterations):
        
        
        # Cost and gradient calculation (≈ 1-4 lines of code)
        ### START CODE HERE ### 
        grads, cost = propagate(w, b, X, Y)
        ### END CODE HERE ###
        
        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]
        
        # update rule (≈ 2 lines of code)
        ### START CODE HERE ###
        w = w - learning_rate * dw
        b = b - learning_rate * db
        ### END CODE HERE ###
        
        # Record the costs
        if i % 100 == 0:
            costs.append(cost)
        
        # Print the cost every 100 training iterations
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs

def predict(w, b, X):

    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    A = sigmoid(np.dot(w.T, X) + b)

    for i in range(A.shape[1]):
        if A[0,i] <= 0.5:
            Y_prediction[0,i] = 0
        else:
            Y_prediction[0,i] = 1
    assert(Y_prediction.shape == (1, m))

    return Y_prediction
