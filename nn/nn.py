import numpy as np
import pickle
import matplotlib.pyplot as plt

def relu(x):
    x[x <= 0] = 0
    return x

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def cross_entropy_error(x, t):
    batch_size = x.shape[0]
    return -np.sum(t * np.log(x) + (1 - t) * np.log((1 - x)), axis=0) / batch_size

def sigmoid_bce_derivative(y, t):
    batch_size = y.shape[0]
    return (y - t) / batch_size

def relu_derivative(x, error):
    dx = np.zeros_like(x)
    dx[x > 0] = 1
    return error * dx

inputs = np.array([[0, 0],
                    [0, 1],
                    [1, 0],
                    [1, 1]])

labels = np.array([[0],
                   [1],
                   [1],
                   [0]])

if __name__ == "__main__":
    batch_size = 4
    input_size = 2
    hidden_size = 4
    output_size = 1
    num_epochs = 3000
    lr = 0.1

    W1 = np.random.uniform(size=(input_size, hidden_size))
    b1 = np.zeros(shape=(1, hidden_size))

    W2 = np.random.uniform(size=(hidden_size, output_size))
    b2 = np.zeros(shape=(1, output_size))

    log_loss = []
    for epoch in range(num_epochs):
        batch_mask = np.random.choice(len(inputs), batch_size)
        x = inputs[batch_mask]
        t = labels[batch_mask]

        ## Forward Propagation
        a1 = np.dot(x, W1) + b1
        h = relu(a1)

        a2 = np.dot(h, W2) + b2
        y = sigmoid(a2)

        loss = cross_entropy_error(y, t)

        log_loss.append(loss)

        ## Back Propagation
        error_a2 = sigmoid_bce_derivative(y, t)

        error_W2 = np.dot(h.T, error_a2)
        error_b2 = np.sum(error_a2, axis=0)
        error_h = np.dot(error_a2, W2.T)

        error_a1 = relu_derivative(a1, error_h)

        error_W1 = np.dot(x.T, error_a1)
        error_b1 = np.sum(error_a1, axis=0)
        error_x = np.dot(error_a1, W1.T)

        # Update Parameters
        W2 = W2 - lr * error_W2
        b2 = b2 - lr * error_b2

        W1 = W1 - lr * error_W1
        b1 = b1 - lr * error_b1

# Predict With Trained Network
predicted_outputs = sigmoid(np.dot(relu(np.dot(inputs, W1) + b1), W2) + b2)

# Show Result
print("Result:")
for input, predict in zip(inputs, predicted_outputs):
    print(f"x1={input[0]}, x2={input[1]}, predict={predict[0]}")

# Show Parameters
print("Params:")
print("W1:")
print(W1)
print("b1:")
print(b1)

print("W2:")
print(W2)
print("b2")
print(b2)

# Plot Loss Transition
plt.plot(range(num_epochs), log_loss)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.savefig("log_loss.png")

# Save Parameters
params = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
path_file = "model.pkl"
pickle.dump(params, open(path_file, 'wb'))


        

    
    
