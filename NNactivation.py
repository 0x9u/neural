import numpy as np
import NNloss as NNloss

class Linear:
    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = inputs
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
    def predictions(self, outputs):
        return outputs

class ReLU:
    def forward(self, inputs : np.array, training):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
        return self.output
    def backward(self, dinputs: np.array):
        self.dinputs = dinputs.copy()
        self.dinputs[self.inputs <= 0] = 0 #simpified equation 1 if more 0 and 0 if less 0 while also multipled with dvalues therefore stays the same if not 0
        return self.dinputs
    def predictions(self, outputs):
        return outputs

class Sigmoid:
    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))
    def backward(self, dvalues):
        self.dinputs = dvalues * (1 - self.output) * self.output
    def predictions(self, outputs):
        return (outputs > 0.5) * 1

class SoftMax:
    def forward(self, inputs : np.array, training):
        exp_values = np.exp(inputs - np.max(inputs, axis = 1, keepdims=True))
        possiblities =  exp_values / np.sum(exp_values, axis = 1, keepdims=True)
        self.output = possiblities
        return self.output
    def backward(self, dinputs: np.array):
        self.dinputs = np.empty_like(dinputs)
        for index, (single_output, single_dvalues) in enumerate(zip(self.inputs, dinputs)):
            single_output = single_output.reshape(-1,1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)
    def predictions(self , outputs):
        return np.argmax(outputs, axis=1)

class Softmax_Loss_CategoricalCrossentropy: #loss and activation combined basically
    def backward(self, dinputs : np.array, y_true : np.array):
        samples = len(dinputs)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        self.dinputs = dinputs.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples