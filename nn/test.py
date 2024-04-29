import numpy as np
import pickle

def relu(x):
    x[x <= 0] = 0
    return x

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x1 = int(input("x1="))
x2 = int(input("x2="))

inputs = [[x1, x2]]

params = pickle.load(open("model.pkl", mode="rb"))
predicted_outputs = sigmoid(np.dot(relu(np.dot(inputs, params["W1"]) + params["b1"]), params["W2"]) + params["b2"])

print(f"1である確率: {predicted_outputs[0]}")
