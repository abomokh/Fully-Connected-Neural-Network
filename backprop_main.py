import matplotlib.pyplot as plt
import numpy as np
from backprop_network import *
from backprop_data import *

# Loading Data
np.random.seed(0)  # For reproducibility
n_train = 50000
n_test = 10000
x_train, y_train, x_test, y_test = load_as_matrix_with_labels(n_train, n_test)

# Training configuration
epochs = 30
batch_size = 100
learning_rate = 0.1

# Network configuration
# layer_dims = [784, 40, 10]
# net = Network(layer_dims)
# net.train(x_train, y_train, epochs, batch_size, learning_rate, x_test=x_test, y_test=y_test)

#################################
#########               #########
#########   Section B   #########
#########               #########
#################################
x_train, y_train, x_test, y_test = load_as_matrix_with_labels(10000, 5000)
learning_rates = [0.001, 0.01, 0.1, 1, 10]
epochs = 30
batch_size = 10

results = {}

for lr in learning_rates:
    net = Network([784, 40, 10])
    params, train_cost, test_cost, train_acc, test_acc = net.train(
        x_train, y_train, epochs=epochs, batch_size=batch_size,
        learning_rate=lr, x_test=x_test, y_test=y_test)
    results[lr] = (train_cost, test_cost, train_acc, test_acc)

plt.figure(figsize=(14, 6))

# Training accuracy
plt.subplot(1, 3, 1)
for lr in learning_rates:
    plt.plot(range(epochs), results[lr][2], label=f'lr={lr}')
plt.title('Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.yticks(np.arange(0, 1.1, 0.1))

# Training loss
plt.subplot(1, 3, 2)
for lr in learning_rates:
    plt.plot(range(epochs), results[lr][0], label=f'lr={lr}')
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.yticks(np.arange(0, 10.5, 0.5))

# Test accuracy
plt.subplot(1, 3, 3)
for lr in learning_rates:
    plt.plot(range(epochs), results[lr][3], label=f'lr={lr}')
plt.title('Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.yticks(np.arange(0, 1.1, 0.1))

plt.tight_layout()
plt.show()


#################################
#########               #########
#########   Section C   #########
#########               #########
#################################
x_train, y_train, x_test, y_test = load_as_matrix_with_labels(60000, 60000)

net = Network([784, 40, 10])
learning_rate = 0.1
epochs = 30
batch_size = 10

res = net.train(x_train, y_train, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, x_test=x_test, y_test=y_test)
parameters, train_losses, test_losses, train_accuracies, test_accuracies = res

print(f"Final test accuracy: {test_accuracies[-1]:.4f}")