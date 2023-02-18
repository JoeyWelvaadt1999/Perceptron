from neural import Layer, Network, Perceptron

### Starting Assertion ###
# Invert
pInvert = Perceptron([-1], 1)
assert pInvert.activation([0]) == 1
assert pInvert.activation([1]) == 0

# AND
pAnd = Perceptron([1, 1], -1)
assert pAnd.activation([1, 1]) == 1
assert pAnd.activation([1, 0]) == 0
assert pAnd.activation([0, 1]) == 0
assert pAnd.activation([0, 0]) == 0

# OR
pOr = Perceptron([1, 1], -0.5)
assert pOr.activation([1, 1]) == 1
assert pOr.activation([1, 0]) == 1
assert pOr.activation([0, 1]) == 1
assert pOr.activation([0, 0]) == 0

# Extended
pExt = Perceptron([0.6, 0.3, 0.2], -0.4)
assert pExt.activation([1,1,1]) == 1
assert pExt.activation([1,1,0]) == 0
assert pExt.activation([1,0,0]) == 0
assert pExt.activation([1,0,1]) == 0
assert pExt.activation([0,1,1]) == 0
assert pExt.activation([0,1,0]) == 0
assert pExt.activation([0,0,1]) == 0
assert pExt.activation([0,0,0]) == 0

# NAND
pNand = Perceptron([-1,-1], 2)
assert pNand.activation([1,1]) == 0
assert pNand.activation([1,0]) == 1
assert pNand.activation([0,1]) == 1
assert pNand.activation([0,0]) == 1

### Starting network test ###
# XOR
layer1 = Layer([pNand, pOr])
layer2 = Layer([pAnd])
network = Network([layer1, layer2])

assert network.predict([1, 1]) == [0]
assert network.predict([1, 0]) == [1]
assert network.predict([0, 1]) == [1]
assert network.predict([0, 0]) == [0]

#Half adder
pSum = Perceptron([1, 0, 0], -.5)
pCarry = Perceptron([0, 1, 1], -1.5)
layer3 = Layer([pAnd, pNand, pOr])
layer4 = Layer([pSum, pCarry])
network2 = Network([layer3, layer4])
print(network2)
assert network2.predict([1, 1]) == [1, 0]
assert network2.predict([1, 0]) == [0, 1]
assert network2.predict([0, 1]) == [0, 1]
assert network2.predict([0, 0]) == [0, 0]