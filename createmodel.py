import Neural
import NNactivation
import NNloss
import NNoptimizer
import os, sys
import cv2
import numpy as np




def load_mnist_dataset(dataset, path):
    labels = os.listdir(os.path.join(path, dataset))
    X = []
    y = []
    for label in labels:
        for file in os.listdir(os.path.join(path, dataset, label)):
            image = cv2.imread(os.path.join(path, dataset, label, file), cv2.IMREAD_UNCHANGED)
            X.append(image)
            y.append(label)
    return np.array(X) , np.array(y).astype("uint8")

def create_data_mnist(path):
    X, y = load_mnist_dataset("train", path)
    X_test, y_test = load_mnist_dataset("test", path)
    return X, y, X_test, y_test

X, y, X_test, y_test = create_data_mnist(os.path.join(sys.path[0], "fashion_mnist_images"))

keys = np.array(range(X.shape[0])) #create a array of 60000 (60000, 28, 28) all numbers increment to 59999 basically
#creates a bunch of indexes
np.random.shuffle(keys)
X = X[keys]
y = y[keys]

X = (X.reshape(X.shape[0],-1).astype(np.float32) - 127.5) / 127.5 #scale it down aka monochrome
X_test = (X_test.reshape(X_test.shape[0], -1).astype(np.float32) - 127.5) / 127.5

model = Neural.Model()
model.add(Neural.Layer_Dense(X.shape[1], 64))
model.add(NNactivation.ReLU())
model.add(Neural.Layer_Dense(64,64))
model.add(NNactivation.ReLU())
model.add(Neural.Layer_Dense(64, 10))
model.add(NNactivation.SoftMax())

model.set(
    loss=NNloss.CategoricalCrossentropy(),
    optimizer=NNoptimizer.Adam(decay=1e-3),
    accuracy=NNloss.Accuracy_Categorical()
)

model.finalize()

model.train(X,y, validation_data=(X_test, y_test), epochs=10, batch_size=128, print_every=100)

parameters = model.get_parameters()

model.evaluate(X_test, y_test)

model.save(os.path.join(sys.path[0],"fashion_mnist.model"))