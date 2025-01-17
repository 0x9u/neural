import numpy as np
from numpy.random.mtrand import sample

class Loss:
    def forward(self,output,y):
        return output
    def calculate(self ,output, y, include_regularization=False):
        sample_losses = self.forward(output , y)
        data_loss = np.mean(sample_losses)
        self.accumulated_sum += np.sum(sample_losses)
        self.accumulated_count += len(sample_losses)
        if not include_regularization:
            return data_loss
        return data_loss, self.regularization_loss()
    def calculate_accumulated(self, * ,include_regularization=False):
        data_loss = self.accumulated_sum / self.accumulated_count
        if not include_regularization:
            return data_loss
        return data_loss, self.regularization_loss()
    def regularization_loss(self):
        regularization_loss = 0
        for layer in self.trainable_layers:
            if layer.weight_regularizer_l1 > 0:
                regularization_loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))

            if layer.weight_regularizer_l2 > 0:
                regularization_loss += layer.weight_regularizer_l2 * np.sum(layer.weights * layer.weights)

            if layer.bias_regularizer_l1 > 0:
                regularization_loss += layer.bias_regularizer_l1 * np.sum(np.abs(layer.biases))

            if layer.bias_regularizer_l2 > 0:
                regularization_loss += layer.bias_regularizer_l2 * np.sum(layer.biases * layer.biases)
        
        return regularization_loss
    def remember_trainable_layers(self, trainable_layers):
        self.trainable_layers = trainable_layers
    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0

class BinaryCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        sample_losses = -(y_true * np.log(y_pred_clipped))
        sample_losses = np.mean(sample_losses, axis=-1)
        return sample_losses
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        outputs = len(dvalues[0])
        clipped_dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7)
        self.dinputs = -(y_true / clipped_dvalues - (1 - y_true) / (1 - clipped_dvalues)) / outputs
        self.dinputs = self.dinputs / samples
        return self.dinputs

class MeanSquaredError(Loss):
    def forward(self, y_pred: np.array, y_true: np.array):
        sample_losses = np.mean((y_true - y_pred) ** 2, axis = -1)
        return sample_losses
    def backward(self, dvalues: np.array, y_true: np.array):
        samples = len(dvalues)
        outputs = len(dvalues[0])
        self.dinputs = -2 * (y_true - dvalues / outputs)
        self.dinputs = self.dinputs / samples
        return self.dinputs

class MeanAbsoluteError(Loss):
    def forward(self, y_pred: np.array, y_true: np.array):
        sample_losses = np.mean(np.abs(y_true - y_pred), axis = -1)
        return sample_losses
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        outputs = len(dvalues[0])
        self.dinputs = np.sign(y_true - dvalues) / outputs


class CategoricalCrossentropy(Loss):
    def forward(self, y_pred : np.array, y_true : np.array):
        samples = len(y_pred)

        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs / samples
        return self.dinputs

class Accuracy:
    def compare(self, predictions, y):
        pass
    def calculate(self, predictions, y):

        comparisons = self.compare(predictions , y)

        accuracy = np.mean(comparisons)
        self.accumulated_sum += np.sum(comparisons)
        self.accumulated_count += len(comparisons)

        return accuracy
    def calculate_accumulated(self):
        accuracy = self.accumulated_sum / self.accumulated_count #average of batches
        return accuracy
    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0

class Accuracy_Regression(Accuracy):
    def __init__(self):
        self.precision = None
    def init(self, y , reinit=False):
        if self.precision is None or reinit:
            self.precision = np.std(y) / 250
    def compare(self, predictions,y):
        return np.absolute(predictions - y) < self.precision

class Accuracy_Categorical(Accuracy):
    def __init__(self, *, binary=False):
        self.binary = binary
    def init(self, y):
        pass
    def compare(self, predictions, y):
        if not self.binary and len(y.shape) == 2:
            y = np.argmax(y, axis = 1)
        return predictions == y