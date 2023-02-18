import math


class Perceptron:
    def __init__(self, weights: list, bias: float):
        self.weights = weights
        self.bias = bias
        self.threshold = -bias

    def sum(self, inputs: list) -> float:
        activation = self.bias
        for i in range(0, len(inputs)):
            activation += inputs[i] * self.weights[i]
        return activation


    def activation(self, inputs: list) -> int:
        if(len(inputs) != len(self.weights)):
            raise Exception("Inputs and weights do not have the same shape.")
        if(self.sum(inputs) >= .5):
            return 1
        return 0
    
    def __str__(self) -> str:
        return "Perceptron weight: " + str(self.weights) + "\n" + "Perceptron bias: " + str(self.bias) + "\n"

class Layer: 
    def __init__(self, perceptrons: list[Perceptron], name: str):
        self.perceptrons = perceptrons
        self.name = name

    def activation(self, inputs: list):
        return [x.activation(inputs) for x in self.perceptrons]

    def __str__(self):
        layerOutput = "The layer "+ self.name +" has the following perceptrons: \n"
        for x in self.perceptrons:
            layerOutput += str(x)
        return layerOutput

class Network:
    def __init__(self, layers: list[Layer]):
        self.layers = layers

    def add_layer(self, layer: Layer):
        self.layers.append(layer)

    def feed_forward(self, inputs: list, prevLayer: Layer, index: int = 0):
        activation = prevLayer.activation(inputs)
        index += 1
        if index < len(self.layers):
            return self.feed_forward(activation, self.layers[index], index)
            
        return activation

    def predict(self, inputs):
        return self.feed_forward(inputs, self.layers[0])

    def __str__(self):
        networkOutput = "Below you will find detailed information about this networks layers: \n\n"
        for layer in self.layers:
            networkOutput += str(layer)
        return networkOutput

