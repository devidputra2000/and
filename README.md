1. Generate the AND function using a McCulloch–Pitts neural network.
```python
x1 = int(input("Enter x1 (0 or 1): "))
x2 = int(input("Enter x2 (0 or 1): "))

Yin = x1 + x2
threshold = 2

if Yin >= threshold:
    y = 1
else:
    y = 0

print("AND output:", y)
```
---
2. Generate the NOT function using a McCulloch–Pitts neural network.
```python
x = int(input("Enter x (0 or 1): "))

weight = -1
threshold = 0

Yin = x * weight

if Yin >= threshold:
    y = 1
else:
    y = 0

print("NOT output:", y)
```
---
3. Generate the XOR function using a McCulloch–Pitts neural network.
```python
x1 = int(input("Enter x1 (0 or 1): "))
x2 = int(input("Enter x2 (0 or 1): "))

z1 = 1 if x1 + x2 >= 1 else 0
z2 = 1 if -x1 - x2 >= -1 else 0

y = 1 if z1 + z2 >= 2 else 0

print("XOR output:", y)
```
---
4. Write a program to implement the Back Propagation Algorithm.
```python
import math

x1, x2 = 0, 1
alpha = 0.25

v1 = [0.6, 0.3]
v2 = [-0.1, 0.4]
w = [-0.2, 0.4, 0.1]
b1, b2 = 0.3, 0.5

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

z1 = sigmoid(b1 + (x1 * v1[0]) + (x2 * v2[0]))
z2 = sigmoid(b2 + (x1 * v1[1]) + (x2 * v2[1]))
y  = sigmoid(w[0] + (z1 * w[1]) + (z2 * w[2]))

dk = (1 - y) * y * (1 - y)
d1 = dk * w[1] * z1 * (1 - z1)
d2 = dk * w[2] * z2 * (1 - z2)

w[0] += alpha * dk
w[1] += alpha * dk * z1
w[2] += alpha * dk * z2

v1[0] += alpha * d1 * x1
v1[1] += alpha * d2 * x1

v2[0] += alpha * d1 * x2
v2[1] += alpha * d2 * x2

b1 += alpha * d1
b2 += alpha * d2

print("Output y:", round(y, 4))
print("Updated w:", w)
print("Updated v1:", v1)
print("Updated v2:", v2)
```
---
5. Write a program to implement the error Backpropagation Algorithm.
```python
import math

x1, x2 = 0, 1
alpha = 0.25

v1 = [0.6, 0.3]
v2 = [-0.1, 0.4]
w = [-0.2, 0.4, 0.1]
b1, b2 = 0.3, 0.5

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

z1 = sigmoid(b1 + (x1 * v1[0]) + (x2 * v2[0]))
z2 = sigmoid(b2 + (x1 * v1[1]) + (x2 * v2[1]))
y  = sigmoid(w[0] + (z1 * w[1]) + (z2 * w[2]))

dk = (1 - y) * y * (1 - y)
d1 = dk * w[1] * z1 * (1 - z1)
d2 = dk * w[2] * z2 * (1 - z2)

w[0] += alpha * dk
w[1] += alpha * dk * z1
w[2] += alpha * dk * z2

v1[0] += alpha * d1 * x1
v1[1] += alpha * d2 * x1

v2[0] += alpha * d1 * x2
v2[1] += alpha * d2 * x2

b1 += alpha * d1
b2 += alpha * d2

print("Output y:", round(y, 4))
print("Updated w:", w)
print("Updated v1:", v1)
print("Updated v2:", v2)
```
---
6. Write a program to implement a Hopfield Network.
```python
import numpy as np

X = np.array([
    [1, 1, 1, 1],
    [-1, -1, -1, -1]
])

rows, cols = X.shape
W = np.zeros((cols, cols))

for i in range(rows):
    W += np.outer(X[i], X[i])

np.fill_diagonal(W, 0)

test = np.array([1, 1, 1, -1])

output = np.sign(np.dot(W, test))
output[output == 0] = 1

print("Weight Matrix:\n", W)
print("Retrieved Pattern:", output)
```
---
7. Design and implement a simple linear neural network model.
```python
import numpy as np

x = np.array([1, 2, 3])
w, b = 2.0, 1.0

y_pred = w * x + b

print("Predictions:", y_pred)
```
---
8. Write a program to implement Hebb’s rule.
```python
import numpy as np

X = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])
Y = np.array([1, -1, -1, -1])

w1, w2, b = 0, 0, 0

for x, y in zip(X, Y):
    w1 += x[0] * y
    w2 += x[1] * y
    b += y

print("Updated Weights:", w1, w2)
print("Updated Bias:", b)
```
---
9. Implement a Kohonen Self-Organizing Map (SOM).
```python
import numpy as np

data = np.array([[1.0, 0.5], [0.2, 0.8]])
weights = np.array([[0.5, 0.5], [0.5, 0.5]])
lr = 0.5

for x in data:
    bmu = np.argmin(np.sum((weights - x) ** 2, axis=1))
    weights[bmu] += lr * (x - weights[bmu])

print("Updated Weights:\n", weights)
```
---
10. Write a program to demonstrate linear separation.
```python
import numpy as np

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])

w = np.array([1.0, 1.0])
b = -1.5

for xi, target in zip(X, y):
    pred = 1 if np.dot(xi, w) + b >= 0 else 0
    print(f"Input: {xi} | Target: {target} | Predicted: {pred}")
```
---
11. Use fuzzy logic to find ratios.
```python
import numpy as np

A = np.array([0.2, 0.5, 0.7, 1.0])
B = np.array([0.5, 0.4, 0.7, 0.2])

union = np.maximum(A, B)
intersection = np.minimum(A, B)
complement_A = 1 - A

print("Union:", union)
print("Intersection:", intersection)
print("Complement of A:", complement_A)
```
