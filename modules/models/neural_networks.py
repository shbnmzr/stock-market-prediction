import numpy as np
import tensorflow as tf
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.losses import SparseCategoricalCrossentropy
from tensorflow.python.keras.regularizers import L2


def layers_construction(num_layers, units, activations, lambda_=0.01):
    layers = []
    for i in range(num_layers):
        layers.append(Dense(units=units[i], activation=activations[i], kernel_regularizer=L2(lambda_)))
    return layers


def model_construction(layers):
    model = Sequential(layers)
    return model


def compile_and_fit(model, X, y, epochs):
    model.compile(optimizer='adam', loss=SparseCategoricalCrossentropy(from_logits=True))
    model.fit(X, y, epochs=epochs)


def make_inference(model, test_data):
    logits = model(test_data)
    f_x = tf.nn.softmax(logits)
    y_hat = np.array([np.argmax(i) for i in f_x])
    return y_hat


def evaluate_model(yhat, y):
    score = [0 if yhat[i] == y[i] else 1 for i in range(len(y))]
    score = np.sum(score)
    score /= len(y)
    return score
