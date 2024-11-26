import Neural
import NNactivation
import NNloss
import NNoptimizer
import NNmisc
import os,sys,cv2
import numpy as np

X, y, X_test, y_test= NNmisc.create_data_mnist(os.path.join(sys.path[0],"fashion_mnist_images"))

X_test = (X_test.reshape(X_test.shape[0], - 1).astype(np.float32) - 127.5) / 127.5 

model = Neural.Model.load(os.path.join(sys.path[0],"fashion_mnist.model"))
confidences = model.predict(X_test[:5])
predictions = model.output_layer_activation.predictions(confidences)
print(predictions)

fashion_mnist_labels = {
    0 : "T-shirt/top",
    1 : "Trouser",
    2 : "Pullover",
    3 : "Dress",
    4 : "Coat",
    5 : "Sandal",
    6 : "Shirt",
    7 : "Sneaker",
    8 : "Bag",
    9 : "Ankle boot"
}
image_data = cv2.imread(os.path.join(sys.path[0],"tshirt.png"), cv2.IMREAD_GRAYSCALE)
image_data = cv2.resize(image_data, (28, 28))
image_data = (image_data.reshape(1, -1).astype(np.float32) - 127.25) / 127.25 #change shape and normalise the data
image_data = 255 - image_data
predictions = model.predict(image_data)
predictions = model.output_layer_activation.predictions(predictions)
print(predictions)
prediction = fashion_mnist_labels[predictions[0]]
print(prediction)

image_data = cv2.imread(os.path.join(sys.path[0],"tshirt2.png"), cv2.IMREAD_GRAYSCALE)
image_data = cv2.resize(image_data, (28, 28))
image_data = (image_data.reshape(1, -1).astype(np.float32) - 127.25) / 127.25 #change shape and normalise the data
image_data = 255 - image_data
predictions = model.predict(image_data)
predictions = model.output_layer_activation.predictions(predictions)
print(predictions)
prediction = fashion_mnist_labels[predictions[0]]
print(prediction)