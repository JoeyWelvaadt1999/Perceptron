import math
from tkinter import *
from tkinter import ttk


#Class containing all different possible activation methods
class Activation:
    #Step function assuming we are using a bias so the threshold is 0.5
    def step(x) -> int:
        if(x >= 0):
            return 1
        return 0

    #Sigmoid function
    def sigmoid(x: float) -> float:
        #source: https://en.wikipedia.org/wiki/Sigmoid_function
        return 1/(1 + (math.e**-x))

#Class containing the neuron information
class Neuron:
    def __init__(self, weights: list[float], bias: float):
        self.weights = weights
        self.bias = bias

    def activate(self, inputs: list[float]):
        #Map through the inputs and multiply them by the weights. Add bias to the total sum of this.
        activation_sum: float = sum([self.weights[i] * inputs[i] for i in range(0, len(inputs))]) + self.bias
        #Pass the activation_sum to the activation function.
        return Activation.sigmoid(activation_sum)
    
    #Output the neuron in a nice way in the console.
    def __str__(self) -> str:
        return f"Neuron(weights={self.weights}, bias={self.bias})"
    
#Class containing all neurons in a layer.
class NeuronLayer():
    def __init__(self, neurons: list[Neuron], inputs: int, outputs: int):
        self.inputs = inputs
        self.outputs = outputs
        self.neurons = neurons

    def activate(self, inputs: list[float]) -> list[float]:
        #Activate all the perceptrons with the inputs given.
        return [p.activate(inputs) for p in self.neurons]
    
    #Output the layer in a nice way in the console.
    def __str__(self) -> str:
        return f'Layer with {self.inputs} inputs and {self.outputs} outputs.'

#Class containing all layers of a network.
class NeuronNetwork():
    def __init__(self, inputs):
        self.inputs = inputs
        self.layers = []

    def add_layer(self, layer: NeuronLayer) -> None:
        self.layers.append(layer)

    
    def activate(self, inputs: list[float]) -> list[float]:
        #Loop through the layers
        for layer in self.layers:
            #Set inputs to the layer response so we can feed forward
            inputs = layer.activate(inputs)
        return inputs
    
    def __str__(self) -> str:
        return f'Network with {self.inputs} inputs and {len(self.layers)} layers'


class Perceptron:
    def __init__(self, weights: list[float], bias: float):
        self.weights = weights
        self.bias = bias

    def activate(self, inputs: list[float]) -> float:
        #Map through the inputs and multiply them by the weights. Add bias to the total sum of this.
        activation_sum: float = sum([self.weights[i] * inputs[i] for i in range(0, len(inputs))]) + self.bias
        #Pass the activation_sum to the activation function.
        return Activation.step(activation_sum)

    #Defining update function that implements perceptron learning rule
    def update(self, inputs: list[float], target: float):
        #Learning rate
        eta = 0.1

        #Output
        y = self.activate(inputs)

        #Error
        e = target - y

        #Delta weigths
        dw = [eta * e * x for x in inputs]
        self.weights = [w + dw for w,dw in zip(self.weights,dw)]

        #Delta bias
        db = eta * e
        self.bias += db 


    def loss(self, examples):
        n = len(examples)
        total_loss = 0.0
        for example in examples:
            inputs, target = example
            output = self.activate(inputs)
            loss = (target - output) ** 2
            total_loss += loss
        mse = total_loss / n
        return mse

    #Output the perceptron in a nice way in the console.
    def __str__(self) -> str:
        return f"Perceptron(weights={self.weights}, bias={self.bias})"
    
class Layer:
    def __init__(self, perceptrons: list[Perceptron], inputs: int, outputs: int):
        self.inputs = inputs
        self.outputs = outputs
        self.perceptrons = perceptrons

    def activate(self, inputs: list[float]) -> list[float]:
        #Activate all the perceptrons with the inputs given.
        return [p.activate(inputs) for p in self.perceptrons]
    
    #Output the layer in a nice way in the console.
    def __str__(self) -> str:
        return f'Layer with {self.inputs} inputs and {self.outputs} outputs.'

class Network:
    def __init__(self, inputs):
        self.inputs = inputs
        self.layers = []

    def add_layer(self, layer: Layer) -> None:
        self.layers.append(layer)

    
    def activate(self, inputs: list[float]) -> list[float]:
        #Loop through the layers
        for layer in self.layers:
            #Set inputs to the layer response so we can feed forward
            inputs = layer.activate(inputs)
        return inputs
    
    def __str__(self) -> str:
        return f'Network with {self.inputs} inputs and {len(self.layers)} layers'
    

class Vizualize: 
    def __init__(self, network: Network, inputs: list):
        self.root = Tk()
        self.network = network
        self.frame = ttk.Frame(self.root)
        self.width, self.height = (self.root.winfo_screenwidth(), self.root.winfo_screenheight())
        self.canvas = Canvas(self.root, width=self.width, height=self.height, borderwidth=0, highlightthickness=0,
                   bg="black")
        
        self.canvas.grid()
        self.draw_layers()
        self.draw_connections()
        self.root.geometry(f"{self.width}x{self.height}")
        self.root.mainloop()

    def create_circle(self, x, y, r, **kwargs):
            return self.canvas.create_oval(x-r, y-r, x+r, y+r, **kwargs)

    def draw_connections(self):
        for row in range(len(self.points)):
            for col in range(len(self.points[row])):
                if(row + 1 < len(self.points)):
                    for nextCol in self.points[row+1]:
                        x1, y1 = self.points[row][col]
                        x2, y2 = nextCol 
                        self.canvas.create_line(x1, y1, x2, y2, fill="#fff")

    def draw_layers(self):
        layers: list[Layer] = self.network.layers
        inputLayerPerceptrons: int = layers[0].inputs
        self.points = [[] for x in [*layers, Layer([], 0, 0)]]
        print (self.points)
        for i in range(inputLayerPerceptrons):
            x: int = (self.width / 2) - (((len(layers) + 1) / 2) * 100)
            y: int = (self.height / 2 - ((inputLayerPerceptrons / 2) * 100)) + (100 * i)
            self.create_circle(x, y, 20, fill="blue", outline="#DDD", width=4)
            self.points[0].append((x,y))

        for i in range(len(layers)):
            for j in range(len(layers[i].perceptrons)):
                x: int = (self.width / 2) - (((len(layers) + 1) / 2) * 100) + (100 * (i + 1))
                y: int = (self.height / 2 - ((len(layers[i].perceptrons) / 2) * 100)) + (100 * j)
                self.create_circle(x, y, 20, fill="blue", outline="#DDD", width=4)
                
                self.points[i+1].append((x,y))
        pass

