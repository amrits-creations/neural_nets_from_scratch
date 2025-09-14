import math
import random

class XORClassifier:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
        
        self.weights_hidden = [[random.uniform(-1, 1)], [random.uniform(-1, 1)], [random.uniform(-1, 1)], [random.uniform(-1, 1)]]
        self.bias_hidden = [random.uniform(-1, 1)], [random.uniform(-1, 1)]
        # weights_hidden[0] corresponds to the weights that are going into the first hidden layer neuron. This is the standard notation.
        # there are as many vectors in the weight matrix of a layer as the number of neurons in that layer. 

        self.weights_output = [[random.uniform(-1, 1), random.uniform(-1, 1)]] # Matrix with one vector inside instead of a plain vector, to preserve the standard notation. This is because there is only one output neuron.
        self.bias_output = [random.uniform(-1, 1)]

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward_pass(self, inputs):
        # First, let's calculate the hidden layer outputs for each of the two hidden layer neurons.

        self.hidden_layer_outputs = [0.0, 0.0]
        self.hidden_layer_outputs[0] = self.sigmoid(
            inputs[0] * self.weights_hidden[0][0] +
            inputs[1] * self.weights_hidden[0][1] +
            self.bias_hidden[0]
        )
        self.hidden_layer_outputs[1] = self.sigmoid(
            inputs[0] * self.weights_hidden[1][0] +
            inputs[1] * self.weights_hidden[1][1] + 
            self.bias_hidden[1]
        ) # Now, we have the output activations for the hidden layer.

        # Let's calculate the final output layer activation. There is only one output layer neuron, resulting in a single output value.

        self.output_layer_output = self.sigmoid(
            self.hidden_layer_outputs[0] * self.weights_output[0][0] +
            self.hidden_layer_outputs[1] * self.weights_output[0][1] +
            self.bias_output[0]
        )

    def backpropogation(self, target):
        pass