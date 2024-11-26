import numpy as np
import pickle
import copy #screw you instances
import NNloss as NNloss
import NNactivation as NNactivation
import NNoptimizer     

class Model:
    def __init__(self):
        self.layers = []
        self.softmax_classifier_output = None
    def add(self, layer):
        self.layers.append(layer)
    def set(self, *, loss=None, optimizer=None, accuracy=None):
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy
    def evaluate(self, X_val, y_val, *, batch_size=None):
        validation_steps = 1
        if batch_size is not None:
            validation_steps = len(X_val) // batch_size
            if validation_steps * batch_size < len(X_val):
                validation_steps += 1
        self.loss.new_pass()
        self.accuracy.new_pass()
        for step in range(validation_steps):
            if batch_size is None:
                batch_X = X_val
                batch_y = y_val
            else:
                batch_X = X_val[step*batch_size:(step+1)*batch_size]
                batch_y = y_val[step*batch_size:(step+1)*batch_size]
            output = self.forward(batch_X, training=False)
            self.loss.calculate(output, batch_y)
            predictions = self.output_layer_activation.predictions(output)
            self.accuracy.calculate(predictions, batch_y)
        validation_loss = self.loss.calculate_accumulated()
        validation_accuracy = self.accuracy.calculate_accumulated()
        print(f"validation, acc: {validation_accuracy}, loss: {validation_loss}")
    def predict(self, X, *, batch_size=None):
        prediction_steps = 1
        if batch_size is not None:
            prediction_steps = len(X) // batch_size
            if prediction_steps * batch_size < len(X):
                prediction_steps += 1
        output = []
        for step in range(prediction_steps):
            if batch_size is None:
                batch_X = X
            else:
               batch_X = X[step*batch_size:(step+1)*batch_size]
            batch_output = self.forward(batch_X, training=False)
            output.append(batch_output)
        return np.vstack(output)
    def train(self, X, y, *, epochs=1, batch_size = None , print_every=1, validation_data=None):
        
        train_steps = 1

        if validation_data is not None:
            self.evaluate(*validation_data, batch_size=batch_size) #(x,y) -> x,y

        if validation_data is not None:
            validation_steps = 1
            X_val, y_val = validation_data
            
        if batch_size is not None: #is basically is reference check
            train_steps = len(X) // batch_size
            
            if train_steps * batch_size < len(X):
                train_steps += 1

            if validation_data is not None:
                validation_steps = len(X_val) // batch_size
            
            if validation_steps * batch_size < len(X_val):
                validation_steps += 1
        
        for epoch in range(1, epochs+1):
            
            print(f"epoch: {epoch}")
            self.loss.new_pass()
            self.accuracy.new_pass()
            
            for step in range(train_steps):
                if batch_size is None:
                    batch_X = X
                    batch_y = y
                else:
                    batch_X = X[step*batch_size:(step+1)*batch_size]
                    batch_y = y[step*batch_size:(step+1)*batch_size]

                output = self.forward(batch_X, training=True) #iterates forward

                data_loss ,regularization_loss = self.loss.calculate(output, batch_y, include_regularization=True)
                loss = data_loss + regularization_loss
                predictions = self.output_layer_activation.predictions(output)
                accuracy = self.accuracy.calculate(predictions, batch_y)

                self.backward(output, batch_y)

                self.optimizer.pre_update_params()
                for layer in self.trainable_layers:
                    self.optimizer.update_params(layer)
                self.optimizer.post_update_params()

                if not step % print_every or step == train_steps - 1:
                    print(f"step {step}, acc: {accuracy:.3f}, loss: {loss:.3f}, (data_loss: {data_loss:.3f}, reg_loss: {regularization_loss:.3f}), lr: {self.optimizer.current_learning_rate}")

            epoch_data_loss, epoch_regularization_loss = self.loss.calculate_accumulated(include_regularization=True)
            epoch_loss = epoch_data_loss + epoch_regularization_loss
            epoch_accuracy = self.accuracy.calculate_accumulated()
            print(f"training, acc: {epoch_accuracy:.3f}, loss: {epoch_loss:.3f} (data_loss: {epoch_data_loss:.3f}, reg_loss: {epoch_regularization_loss:.3f}) lr: {self.optimizer.current_learning_rate}")

        if validation_data is not None:
            X_val, y_val = validation_data
            output = self.forward(X_val, training=False)

            loss = self.loss.calculate(output, y_val)

            predictions = self.output_layer_activation.predictions(output)
            accuracy = self.accuracy.calculate(predictions, y_val)

            print(f"validation, acc: {accuracy:.3f}, loss: {loss:.3f}")

    def finalize(self):
        self.input_layer = Layer_Input()
        layer_count = len(self.layers)
        self.trainable_layers = []
        for i in range(layer_count):
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i+1]
            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.layers[i + 1]
            else:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]
            if hasattr(self.layers[i], "weights"): #ignores activation layers and dropout layers
                self.trainable_layers.append(self.layers[i])
        if self.loss is not None:
            self.loss.remember_trainable_layers(self.trainable_layers)
        if isinstance(self.layers[-1], NNactivation.SoftMax) and isinstance(self.loss, NNloss.CategoricalCrossentropy):
            self.softmax_classifier_output = NNactivation.Softmax_Loss_CategoricalCrossentropy()
    def forward(self, X, training):
        self.input_layer.forward(X, training)
        for layer in self.layers:
            layer.forward(layer.prev.output, training)
        return layer.output
    def backward(self, output, y):
        if self.softmax_classifier_output is not None:
            self.softmax_classifier_output.backward(output, y)
            self.layers[-1].dinputs = self.softmax_classifier_output.dinputs
            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)
            return
        self.loss.backward(output, y)
        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)
    def get_parameters(self):
        parameters = []
        for layer in self.trainable_layers:
            parameters.append(layer.get_parameters())
        return parameters
    def set_parameters(self, parameters):
        for parameter_set, layer in zip(parameters, self.trainable_layers):
            layer.set_parameters(*parameter_set)
    def save_parameters(self, path):
        with open(path, "wb") as file:
            pickle.dump(self.get_parameters, file)
    def load_parameters(self, path):
        with open(path, "rb") as file:
            self.set_parameters(pickle.load(file))
    def save(self, path):
        model = copy.deepcopy(self) #coping own instance
        model.loss.new_pass()
        model.accuracy.new_pass()
        model.input_layer.__dict__.pop("output", None)
        model.loss.__dict__.pop("dinputs", None)
        for layer in model.layers:
            for property in ["inputs", "output", "dinputs", "dweights", "dbiases"]:
                layer.__dict__.pop(property, None)
        with open(path, "wb") as file:
            pickle.dump(model, file)
    @staticmethod #doesnt require its own instance
    def load(path):
        with open(path, "rb") as file:
            model = pickle.load(file)
        return model

class Layer_Input:
    def forward(self, inputs, training):
        self.output = inputs

class Layer_Dropout:
    def __init__(self, rate):
        self.rate = 1 - rate
    def forward(self, inputs, training):
        self.inputs = inputs
        if not training:
            self.output = inputs.copy()
            return
        self.binary_mask = np.random.binomial(1, self.rate , size=inputs.shape) / self.rate
        self.output = inputs * self.binary_mask
    def backward(self, dvalues):
        self.dinputs = dvalues * self.binary_mask

class Layer_Dense:
    def __init__(self, num_inputs : int, num_neurons : int, weight_regularizer_l1 : int = 0, weight_regularizer_l2 : int = 0, bias_regularizer_l1 : int = 0, bias_regularizer_l2 : int = 0):
        self.weights = 0.1 * np.random.randn(num_inputs, num_neurons)
        self.biases = np.zeros((1,num_neurons))
        self.output = None
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2
 

    def forward(self, x_data : np.array, training):
        self.inputs = x_data
        self.output = np.dot(x_data, self.weights) + self.biases
        return self.output
    
    def backward(self, y_data: np.array): #or dvalues
        self.dweights = np.dot(self.inputs.T, y_data)
        self.dbiases = np.sum(y_data, axis=0, keepdims=True)

        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights
        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * self.biases

        self.dinputs = np.dot(y_data, self.weights.T)
    
    def get_parameters(self):
        return self.weights, self.biases
    def set_parameters(self, weights, biases):
        self.weights = weights
        self.biases = biases

if __name__ == "__main__":
    import nnfs
    import nnfs.datasets
    nnfs.init()
    X, y = nnfs.datasets.spiral_data(samples=1000,classes=3)
    X_test, y_test = nnfs.datasets.spiral_data(samples=100, classes=3)
    model = Model()
    model.add(Layer_Dense(2,512))
    model.add(NNactivation.ReLU())
    model.add(Layer_Dropout(0.1))
    model.add(Layer_Dense(512,3))
    model.add(NNactivation.SoftMax())
    model.set(
        loss=NNloss.CategoricalCrossentropy(),
        optimizer=NNoptimizer.Adam(learning_rate=0.005, decay=1e-3),
        accuracy=NNloss.Accuracy_Categorical()
    )
    model.finalize()
    model.train(X,y,validation_data = (X_test, y_test) ,epochs=10000,print_every=100)
