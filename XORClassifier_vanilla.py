import math
import random

class XORClassifier:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
        
        self.weights_hidden_layer = [[random.uniform(-1, 1)], [random.uniform(-1, 1)], [random.uniform(-1, 1)], [random.uniform(-1, 1)]]
        self.bias_hidden_layer = [random.uniform(-1, 1), random.uniform(-1, 1)]
        # weights_hidden_layer[0] corresponds to the weights that are going into the first hidden layer neuron. This is the standard notation.
        # there are as many vectors in the weight matrix of a layer as the number of neurons in that layer. 

        self.weights_output_layer = [[random.uniform(-1, 1), random.uniform(-1, 1)]] # Matrix with one vector inside instead of a plain vector, to preserve the standard notation. This is because there is only one output neuron.
        self.bias_output_layer = [random.uniform(-1, 1)]

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward_pass(self, inputs):
        # First, let's calculate the hidden layer outputs for each of the two hidden layer neurons.

        self.output_hidden_layer = [0.0, 0.0]
        self.output_hidden_layer[0] = self.sigmoid(
            inputs[0] * self.weights_hidden_layer[0][0] +
            inputs[1] * self.weights_hidden_layer[0][1] +
            self.bias_hidden_layer[0]
        )
        self.output_hidden_layer[1] = self.sigmoid(
            inputs[0] * self.weights_hidden_layer[1][0] +
            inputs[1] * self.weights_hidden_layer[1][1] + 
            self.bias_hidden_layer[1]
        ) # Now, we have the output activations for the hidden layer.

        # Let's calculate the final output layer activation. There is only one output layer neuron, resulting in a single output value.

        self.output_output_layer = self.sigmoid(
            self.output_hidden_layer[0] * self.weights_output_layer[0][0] +
            self.output_hidden_layer[1] * self.weights_output_layer[0][1] +
            self.bias_output_layer[0]
        )
        return self.output_output_layer

    def backpropogation(self, inputs, target):
        self.error_output_layer = self.output_output_layer - target # The error at the output layer. This is the derivative of the loss function w.r.t the output of the output layer neuron (final output of the network).
        self.d_output_layer = self.error_output_layer * self.sigmoid_derivative(self.output_output_layer) # d_output is the error signal for the output layer neuron. 
        # It is the derivative of the cost function w.r.t. the pre-sigmoid output of the output layer neuron, obtained using the chain rule.
        # This will be used to update the weights and biases of the output layer neuron.
        
        self.error_hidden_layer = [0.0, 0.0]
        # In the hidden layer, the error for a neuron would be equal to the error signal of the output neuron multiplied by the weight connecting that particular neuron to the output neuron.
        # We can say that the error signal propogates backwards similar to how the activations propogate forwards. In both cases, there is a weighted average using the weight matrix involved. 
        # The error signal of the output layer neuron is effectively 'distributed' amongst the neurons of the previous layer proportional to the respective weights that are connecting the neurons together.
        # We use weights because the weight term is actually the derivative of the pre-activation output of the output layer neuron w.r.t. the activation of the connected neuron in the previous layer. 
        self.error_hidden_layer[0] = self.d_output_layer * self.weights_output_layer[0][0]
        self.error_hidden_layer[1] = self.d_output_layer * self.weights_output_layer[0][1]
        # If we multiply this with the derivative of the activation of the hidden layer neuron w.r.t. it's pre-activation output, we get the gradient or error signal for that particular hidden layer neuron.
        # This derivative is just the post-activation output of the hidden layer neuron passed through the derivative of the sigmoid function. 
        # This is because of a neat sigmoid trick, where the derivative of the sigmoid function w.r.t. it's output can be written in terms of the output.

        self.d_hidden_layer = [0.0, 0.0]
        self.d_hidden_layer[0] = self.error_hidden_layer[0] * self.sigmoid_derivative(self.output_hidden_layer[0])
        self.d_hidden_layer[1] = self.error_hidden_layer[1] * self.sigmoid_derivative(self.output_hidden_layer[1])

        # Now, we can say that we have the gradient or error signal for every neuron in the network that has tunable weights and biases. 
        # We will use these gradients to update the weights and biases of each neuron:

        self.weights_output_layer[0][0] -= self.output_hidden_layer[0] * self.learning_rate * self.d_output_layer
        self.weights_output_layer[0][1] -= self.output_hidden_layer[1] * self.learning_rate * self.d_output_layer
        self.bias_output_layer -= self.learning_rate * self.d_output_layer
        # Updated output layer neuron.


        self.weights_hidden_layer[0][0] -= inputs[0] * self.learning_rate * self.d_output_layer
        self.weights_hidden_layer[0][1] -= inputs[1] * self.learning_rate * self.d_output_layer
        self.bias_hidden_layer[0] -= self.learning_rate * self.d_hidden_layer[0]
        # Updated first hidden layer neuron.


        self.weights_hidden_layer[1][0] -= inputs[0] * self.learning_rate * self.d_output_layer
        self.weights_hidden_layer[1][1] -= inputs[1] * self.learning_rate * self.d_output_layer
        self.bias_hidden_layer[1] -= self.learning_rate * self.d_hidden_layer[1]
        # Updated second hidden layer neuron.

    def train(self, inputs_data, outputs_data, epochs = 50000):

        for epoch in range(epochs):
            input_index = random.randint(0, 3) # picking a random data point from the four XOR inputs for this training epoch.
            current_inputs = inputs_data[input_index]
            current_target = outputs_data[input_index]

            self.forward_pass(current_inputs) # Call the forward pass. This stores the output of the network in self.output_layer_output. 

            self.backpropogation(current_inputs, current_target) # Call the backprop function to update weights and biases in a way that minimizes loss.

            # To monitor progress, let's print the average loss across the four training examples for XOR every 1,000 epochs.

            if epoch % 1000 == 0:
                total_error = 0
                for i in range(len(inputs_data)):
                    prediction = self.forward_pass(inputs_data[i])
                    total_error += (outputs_data[i] - prediction) ** 2
                average_error = total_error / len(inputs_data)
                print(f"Epoch - {epoch}: Average error = {average_error} ")
            
# That concludes our class definition for the XOR Classifier neural network. We will try to train it on data in the main function and maybe incorporate live input as well for testing predictions.

if __name__ == '__main__':
    xor_inputs = [[0,0], [0,1], [1,0], [1,1]]
    xor_outputs = [0, 1, 1, 0]

    xor_classifier = XORClassifier(learning_rate = 0.01)

    xor_classifier.train(xor_inputs, xor_outputs, epochs = 50000)

    print("Training complete. Live-testing model performance... Enter two numbers: \n")

    while again == True:
        live_input = [0.0, 0.0]
        live_input[0] = float(input("Enter first bit - \n"))
        live_input[1] = float(input("Enter second bit - \n"))

        result = xor_classifier.forward_pass(live_input)
        print(f"Classifier output: {result}")

        again = bool(input("Want to try it again? (True/False)"))