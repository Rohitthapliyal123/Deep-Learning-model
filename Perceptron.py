# Perceptron logic for Logical AND Gate...

# # Activation function...
# def step(x):
#     return 1 if x >= 0 else 0;
#
#
# def perceptron(x1 , x2 , w1, w2, b):
#     y = x1*w1 + x2*w2 + b
#     return step(y)
#
# print(perceptron(0, 0, 1, 1, -1.5))  # Expected: 0
# print(perceptron(0, 1, 1, 1, -1.5))  # Expected: 0
# print(perceptron(1, 0, 1, 1, -1.5))  # Expected: 0
# print(perceptron(1, 1, 1, 1, -1.5))  # Expected: 1

# Activation function...
# def step(x):
#     return 1 if x >= 0 else 0;
#
#
# def perceptron(x1 , x2 , w1, w2, b):
#     y = x1*w1 + x2*w2 + b
#     return step(y)
#
# print(perceptron(0, 0, 1, 1, -1.5))  # Expected: 0
# print(perceptron(0, 1, 1, 1, 1.5))  # Expected: 1
# print(perceptron(1, 0, 1, 1, 1.5))  # Expected: 1
# print(perceptron(1, 1, 1, 1, 1.5))  # Expected: 1

from sklearn.linear_model import Perceptron
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# X - features set (1000*10) & Y - features set(1000*1)
x, y = make_classification(n_samples=1000, n_features= 10, n_classes=2 ,random_state=42)
x_train , x_test , y_train , y_test = train_test_split(x , y , test_size=0.2, random_state=42)
# (x_train has 800*10 and x_test has 200*10)
