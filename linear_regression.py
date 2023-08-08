import numpy as np

feature = np.array([1,2,3,4,5])

def generate():
    x = np.random.randn(len(feature))
    y = feature.dot(x) + np.random.randn()
    return x, y

ExampleData = [generate() for i in range(1000000)]

def phi(x):
    return np.array(x)

def initialVector():
    return np.zeros(len(feature))

def trainLoss(w):
    return 1.0/len(ExampleData) * sum((w.dot(phi(x)) - y)**2 for x, y in ExampleData)

def gradientLoss(w):
    return 1.0/len(ExampleData) * sum(2*(w.dot(phi(x)) - y)*phi(x) for x, y in ExampleData)

def loss(w, i):
    x, y = ExampleData[i]
    return (w.dot(phi(x)) - y)**2

def gradient(w, i):
    x, y = ExampleData[i]
    return 2*(w.dot(phi(x)) - y)

def gradientDescent(F, gradientF, initialVector):
    w = initialVector()
    a = 0.1
    for i in range(100):
        value = F(w)
        gradient = gradientF(w)
        w -= a * gradient
        print(f'epoch {i} ==> w:{w}, F(w):{value}, gradient:{gradient}')

def stochasticGradientDescent(f, gradientf, n, initialVector):
    w = initialVector()
    numUpdate = 0
    for i in range(500):
        for j in range(n):
            value = f(w, j)
            gradient = gradientf(w, j)
            numUpdate += 1
            a = 1 / np.sqrt(numUpdate)
            w -= a * gradient
        print(f'epoch {i} ==> w:{w}, F(w):{value}, gradient:{gradient}')

# gradientDescent(trainLoss, gradientLoss, initialVector)
stochasticGradientDescent(loss, gradient, len(ExampleData), initialVector)