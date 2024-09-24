#_____________________________________________convolutional neural network______________________________________________#

from data import get_mnist
import numpy as np
import matplotlib.pyplot as plt
#
#_______________________________________________________________________________________________________________________


"""
w = weights, b = bias, i = input, h = hidden, o = output, l = label
e.g. w_i_h = weights from input layer to hidden layer
"""
#
#_______________________________________________________________________________________________________________________

images, labels = get_mnist()
# Getting the images and labels provided by the dataset
# shape: image(60000, 784) being for 60K images being 28 x 28 (784) 
# shape: labels(60000, 10) being for 60K images having 10 possibe lables (0,1,2,3,4,5,6,7,8,9)
# Labels are represented in binary format. 
# If 3 needs to be detected neural network output should be:
# { 0 }
# { 0 }
# { 0 }
# { 1 }
# { 0 }
# { 0 }
# { 0 }
# { 0 }
# { 0 }
# { 0 } 
#
#_______________________________________________________________________________________________________________________

w_i_h = np.random.uniform(-0.5, 0.5, (20, 784))

# Random weights close to 0
# W for Weights being -0.5, 0.5
# Number of Neurons in each layer:
# 784 input (Dataset images are 28 x 28 pixels. therefore 784)
# 20 hidden 
# Weight matrix connecting weight layer to input layer and hidden layer : 20 by 784
#
#_______________________________________________________________________________________________________________________


w_h_o = np.random.uniform(-0.5, 0.5, (10, 20))

# Random weights close to 0
# W for Weights being -0.5, 0.5
# Number of Neurons in each layer:
# 10 hidden
# 20 output
# Weight matrix connecting hiddenl layer with output layer : 10 by 20
#
#_______________________________________________________________________________________________________________________


b_i_h = np.zeros((20, 1))

# .zeros to start with unbias neurons
# connecting unbias neuron to input and hidden layer
#
#_______________________________________________________________________________________________________________________

b_h_o = np.zeros((10, 1))

# .zeros to start with unbias neurons
# connecting unbias neuron to hidden and output layer
#
#_______________________________________________________________________________________________________________________


learn_rate = 0.01

# the learning rate is a configurable hyperparameter used in the training of 
# neural networks that has a small positive value, often in the range between 0.0 and 1.0.
#
#_______________________________________________________________________________________________________________________

nr_correct = 0

# correct classified input
#
#_______________________________________________________________________________________________________________________

epochs = 3

# Go throug all images 3 times
#
#_______________________________________________________________________________________________________________________
 

for epoch in range(epochs):

    # Specify how often we itterate through all images
    #
    #_______________________________________________________________________________________________________________________

    for img, l in zip(images, labels):
        img.shape += (1,) 
        l.shape += (1,) 

        # Itterates through all image label pairs
        #
        # img.shape += (1,) reshapes img label from 784 vector to (784, 1) matrix
        # l.shape += (1,) reshapes vector from size 10 to a (10, 1) matrix
        #
        # img.shape: (60000, 784) is a matrix. but img.shape: 784 is a vector
        # l.shape: (60000, 10) as a matrix. but l.shape: 10 is a vector
        #
        # Forward propagation input -> hidden
        #
        #_______________________________________________________________________________________________________________________
 
        h_pre = b_i_h + w_i_h @ img
        h = 1 / (1 + np.exp(-h_pre))

        ## EXAMPLE OF MATH:
        # weights from input layer to hidden layer multiplied with matrix multiplication '@' with the input values
        # add '+' the bias weights for hidden neuron value (-h_pre)
        # To avoid really high hidden value compared to other values u can normalise a specific range
        # do this by applying activation funtion like sigmoid function
        #
        #                     1
        #           h = ---------------
        #                       -h_pre
        #                 1 + e
        #
        # Like this your neural network will calculate the hidden neuron of an input
        #
        #_______________________________________________________________________________________________________________________

        # Forward propagation hidden -> output

        o_pre = b_h_o + w_h_o @ h
        o = 1 / (1 + np.exp(-o_pre))
        
        #Everything that you did in the last step calculating the hidden neuron will apply here aswell. 
        # note: This calculates the output neuron
        # this model only has 20 outputs. so the model can only differentiate between 0 and 19
        # {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19}
        #
        #_______________________________________________________________________________________________________________________
        
        # Cost / Error calculation

        e = 1 / len(o) * np.sum((o - l) ** 2, axis=0)
        # Most common calculation for error handling
        # It works by calculating the difference inbetween the output and the corresponding label | (0 - l)
        # When the difference is calculated it squares it to get the value | ((0 - l) ** 2)
        # The resulted value gets summed together | sum(ans, axis=0)
        # The summed together value gets devided by the number of output neurons | len(o) = 20 neurons
        #
        #_______________________________________________________________________________________________________________________
 
        nr_correct += int(np.argmax(o) == np.argmax(l))
        # Check if network has classified the input correctly
        # The correct neuron should be the one with the highest value
        # np.argmax to check the output
        # np.argmax(l) to check the label
        #   EXAMPLE:
        #   {0.67}   {1}
        #   {0.53} = {0}
        #   {0.52}   {0}
        #
        # if the output is correct with the label it shall set np.argmax(0) to zero
        # if the output is correct with the label it shall set np.argmax(l) to zero also
        # If the highest output neuron is alligned with the highest label it shall change the sum to:
        # | nr_correct += int( 0 == 0) | wich is equivalent to | nr_correct += int(True) |
        # nr_correct += int(True) is equivalent to nr_correct += 1
        #
        # Example"
        # label = 0 = {1}
        #             {0}
        #             {0}
        #
        # Note that this is for error checking and is not something the network has to rely on
        #
        #_______________________________________________________________________________________________________________________
  
        # Backpropagation output -> hidden (cost function derivative)

        delta_o = o - l
        # Calculate delta for each neuron
        # The delta for an output neuron is just the difference between the output and the label
        #
        #_______________________________________________________________________________________________________________________
  
        w_h_o += -learn_rate * delta_o @ np.transpose(h)
        #
        # delta_o @ np.transpose(h) calculate the update value for each weight connecting both layers
        # (That being hidden and output)
        # Multiply it with a learning rate (That being -0.01)
        # Why minus learn rate? - 0.01
        # Updated values represent how to maximize the error for the input
        # That is why you need to negate them to have an opposite effect
        #
        #_______________________________________________________________________________________________________________________
  
        b_h_o += -learn_rate * delta_o
        #
        # Calculate the bias neuron weights connecting to the output
        # bias neuron is always 1. there is no need to multiply something by 1
        # Thats why you can just multiply the delta output value with the negative learn rate
        #
        #_______________________________________________________________________________________________________________________
  
        # Backpropagation hidden -> input (activation function derivative)
        
        delta_h = np.transpose(w_h_o) @ delta_o * (h * (1 - h))
        #
        # h = Sigmoid function from earlier
        # Calculate derivate of sigmoid function with h * (1 - h)
        # Transpose w_h_o using np.transpose(w_h_0)
        # Matrix multiply the transposed value with '@' and multiply it with the derivative values
        # The resulting delta shows how strong each individual neuron participated towards the error
        #_______________________________________________________________________________________________________________________
  
        w_i_h += -learn_rate * delta_h @ np.transpose(img)
        #
        # Calculate the update values for weights connecting the input values to the hidden layer
        # img = the input values
        #_______________________________________________________________________________________________________________________
  
        b_i_h += -learn_rate * delta_h
        #
        # Calculate the bias neuron weights connecting to the hidden layer
        # bias neuron is always 1. there is no need to multiply something by 1
        # Multiply the negative learn_rate with the delta of the hidden layer neurons
        #
        #_______________________________________________________________________________________________________________________
   

    # Show accuracy for this epoch
    print(f"Acc: {round((nr_correct / images.shape[0]) * 100, 2)}%")
    nr_correct = 0
    #
    # Print the accuracy
    #
    #_______________________________________________________________________________________________________________________
   

# Show results
while True:
    index = int(input("Enter a number (0 - 59999): "))
    #
    # This is to pick a random file out of all 60000
    #_______________________________________________________________________________________________________________________
   
    img = images[index]
    plt.imshow(img.reshape(28, 28), cmap="Greys")

    img.shape += (1,)
    # Forward propagation input -> hidden
    h_pre = b_i_h + w_i_h @ img.reshape(784, 1)
    h = 1 / (1 + np.exp(-h_pre))
    # Forward propagation hidden -> output
    o_pre = b_h_o + w_h_o @ h
    o = 1 / (1 + np.exp(-o_pre))

    plt.title(f"Subscribe if its a {o.argmax()} :)")
    plt.show()
