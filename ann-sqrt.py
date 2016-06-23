import numpy as np

layers = np.array([1,10,10,10,1])
max_int = 100.0

def activation(x):
  return np.tanh(x)

def derivative(x):
  return 1.0 - np.tanh(x)**2 

def calculate(x):
  a = [x/max_int, 1.0]
  for l in range(len(weights)):
    a = activation(np.dot(a, weights[l]))
  return a[0]*max_int

def backpropagate(x, y, learning_rate=0.6, epochs=20000):
  #normalization:
  X = np.ones(len(x))
  for k in range(len(x)):
    x[k] = [x[k]/max_int]
    y[k] = y[k]/max_int
  temp = np.ones([len(x), 2])
  temp[:, 0:-1] = x  # adding the bias unit to the input layer
  x = temp
  for k in range(epochs):
    i = np.random.randint(len(x))
    testval = [x[i]]

    for l in range(len(weights)):
      testval.append(activation(testval[l].dot(weights[l])))
    error = y[i] - testval[-1]
    deltas = [error * derivative(testval[-1])]
     # we need to begin at the second and go to the last layer
    for l in range(len(testval)-2, 0, -1):
      deltas.append(deltas[-1].dot(weights[l].T)*derivative(testval[l]))
    deltas = deltas[::-1]
    for i in range(len(weights)):
      layer = np.atleast_2d(testval[i])
      delta = np.atleast_2d(deltas[i])
      weights[i] = weights[i] + learning_rate * layer.T.dot(delta)    
              

weights = []
for i in range(1, len(layers) - 1):
  weights.append((2*np.random.random((layers[i - 1] + 1, layers[i] + 1))-1)/4)
# taking care of the last layer, the one of output
weights.append((2*np.random.random((layers[i] + 1, layers[i + 1]))-1)/4)


print("first set: 2-100")
x = np.random.randint(2,101, size=20).tolist()
y = np.sqrt(x)

backpropagate(x, y)
test_vals = [2, 4, 8, 9, 12, 16, 25, 27, 36, 49]
for i in test_vals:
  res = calculate(i)
  print("The square root of ", i, " is ", round(res,3))
  
print("second set: 2-50")
x = [1, 4, 9, 16, 25, 36, 49]
y = np.sqrt(x)

backpropagate(x, y)
test_vals = [2, 4, 8, 9, 12, 16, 25, 27, 36, 49]
for i in test_vals:
  res = calculate(i)
  print("The square root of ", i, " is ", round(res,3))