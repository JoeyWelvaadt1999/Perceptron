from neural import Perceptron, Neuron, Layer, Network, Vizualize, NeuronLayer, NeuronNetwork

#### Perceptron P1 ####
print('\n\nStarting P1\n\n')

## Defining print function for tests ##
def print_test(object, inputs: list[int], answers: list[int]):
    for i in range(0, len(inputs)):
        activation:int = object.activate(inputs[i])
        if(activation == answers[i]):
            print("The activation with inputs", inputs[i], "gave the correct answer which was:", activation, '\n')
        else:
            print("The activation with inputs", inputs[i], "gave the wrong answer which was:", activation, "it should be:", answers[i], '\n')
    print(object)

def print_test_neuron(neuron:Neuron, inputs: list[int], answers: list[int]):
    
    for i in range(0, len(inputs)):
        activation:float = neuron.activate(inputs[i])
        print(inputs[i], activation)

def print_test_neuron_network(network: Network, inputs: list[int], answers: list[int]):
    for i in range(len(inputs)):
        activations = network.activate(inputs[i])
        for j in range(len(activations)):
            activations[j] = 1 if activations[j] >= 0.5 else 0
        if(activations == answers[i]):
            print("The activation with inputs", inputs[i], "gave the correct answer which was:", activations, '\n')
        else:
            print("The activation with inputs", inputs[i], "gave the wrong answer which was:", activations, "it should be:", answers[i], '\n')
    print(network)


# Defining INVERT perceptron #
invert_perceptron = Perceptron([-1], 0)
invert_inputs = [
    [1],
    [0]
]
invert_answers = [
    0,
    1
]
print("\n\nINVERT Perceptron\n")
print_test(invert_perceptron, invert_inputs, invert_answers)

# Defining inputs for all perceptrons with 2 inputs #
perceptron_inputs = [
    [1, 1],
    [1, 0],
    [0, 1],
    [0, 0]
]

# Defining AND perceptron #
and_perceptron = Perceptron([0.5, 0.5], -1)
and_answers = [
    1,
    0,
    0,
    0
]

print("\n\nAND Perceptron\n")
print_test(and_perceptron, perceptron_inputs, and_answers)

# Defining OR perceptron #
or_perceptron = Perceptron([1, 1], -1)
or_answers = [
    1,
    1,
    1,
    0
]

print("\n\nOR Perceptron\n")
print_test(or_perceptron, perceptron_inputs, or_answers)


# Defining NAND perceptron #
nand_perceptron = Perceptron([-1, -1], 1.5)
nand_answers = [
    0,
    1,
    1,
    1
]

print("\n\nNAND Perceptron\n")
print_test(nand_perceptron, perceptron_inputs, nand_answers)

# Defining 3 Input perceptron #
three_perceptron = Perceptron([0.6, 0.3, 0.2], -0.4)
three_inputs = [
    [1, 1, 1],
    [1, 0, 1],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 1],
    [0, 0, 0]
]
three_answers = [
    1,
    1,
    0,
    0,
    1,
    1,
    1,
    0
]

print("\n\n3 Input Perceptron\n")
print_test(three_perceptron, three_inputs, three_answers)

# Defining NOR perceptron #
nor_perceptron = Perceptron([-1, -1, -1], 0)
nor_answers = [
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    1
]

print("\n\nNOR Perceptron\n")
print_test(nor_perceptron, three_inputs, nor_answers)



# Defining XOR network #
layer_xor_in: Layer = Layer([or_perceptron, nand_perceptron], 2, 2)
layer_xor_out: Layer = Layer([and_perceptron], 2, 1)

network_xor = Network(2)
network_xor.add_layer(layer_xor_in)
network_xor.add_layer(layer_xor_out)

network_xor_inputs = [
    [1, 1],
    [1, 0],
    [0, 1],
    [0, 0]
]

network_xor_answers = [
    [0],
    [1],
    [1],
    [0]
]

print("\n\nXOR Network\n")
print_test(network_xor, network_xor_inputs, network_xor_answers)

# Defining Half Adder network #
sum_perceptron = Perceptron([1, 0, 0], -.5)
carry_perceptron = Perceptron([-3, 1, 1], -.5)
layer_ha_in = Layer([and_perceptron, and_perceptron, or_perceptron], 2, 3)
layer_ha_out = Layer([sum_perceptron, carry_perceptron], 3, 2)

network_ha = Network(2)
network_ha.add_layer(layer_ha_in)
network_ha.add_layer(layer_ha_out)

network_ha_inputs = [
    [1, 1],
    [1, 0],
    [0, 1],
    [0, 0]
]

network_ha_answers = [
    [1, 0],
    [0, 1],
    [0, 1],
    [0, 0]
]
print("\n\nHalf Adder Network\n")
print_test(network_ha, network_ha_inputs, network_ha_answers)

#### Perceptron learning rule P2 ####
print('\n\nStarting P2\n\n')
print("View notebook: P2 - Learning rule test")

#### Sigmoid Neuron P3 ####
print('\n\nStarting P3\n\n')

# Defining INVERT neuron #
#Invert Perceptron: ([-1], 0.5)
invert_neuron = Neuron([-1], 0.5)

print("\n\nINVERT Neuron (With Perceptron inputs)\n")
print_test_neuron(invert_neuron, invert_inputs, invert_answers)
print("\nINVERT Neuron Anaylze: Using the same inputs as were used in the perceptron. We get the outputs seen above: 0.3775 for input 1 and 0.6224 for input 0. \nWe can conclude that these inputs work for this particular case, because a sigmoid neuron fires when the output is above or equal to 0.5. \nSo to sum up, the neuron fires when the input is 0 and doesn't fire when the input is 1. \n")


# Defining AND neuron #
#And Perceptron: ([0.5, 0.5], -1)
and_neuron = Neuron([0.5, 0.5], -1)

print("\n\nAND Neuron (With Perceptron inputs)\n")
print_test_neuron(and_neuron, perceptron_inputs, and_answers)
print("\nAND Neuron Anaylze: As mentioned int the Analyze above, a sigmoid neuron fires when the output is greater or equal to 0.5. \nIn the outputs above only the input [1, 1] matches this requirement. This means the neuron also works with the AND percoptron weights and bias. \n")


#Defining OR neuron #
#Or Perceptron: ([1, 1], -1)
or_neuron = Neuron([1, 1], -1)

print("\n\nOR Neuron (With Perceptron inputs)\n")
print_test_neuron(or_neuron, perceptron_inputs, or_answers)
print("\nOR Neuron Anaylze: The OR neuron also seems to work as it should, it fires with the inputs: [1,1], [1,0] and [0,1]. This is exactly what should happen with an OR gate. \n")

#Defining NOR neuron #
nor_neuron = Neuron([-1, -1, -1], 0)

print("\n\nNOR Neuron (With Perceptron inputs)\n")
print_test_neuron(nor_neuron, three_inputs, nor_answers)

# Defining Half Adder neuron network #
sum_neuron = Neuron([1, 0, 0], -.5)
carry_neuron =  Neuron([-3, 1, 1], -.5)
layer_ha_neuron_in = NeuronLayer([and_neuron, and_neuron, or_neuron], 2, 3)
layer_ha_neuron_out = NeuronLayer([sum_neuron, carry_neuron], 3, 2)

network_ha_neuron = NeuronNetwork(2)
network_ha_neuron.add_layer(layer_ha_neuron_in)
network_ha_neuron.add_layer(layer_ha_neuron_out)

print("\n\nHalf Adder Neuron Network\n")
print_test_neuron_network(network_ha_neuron, network_ha_inputs, network_ha_answers)
# Defining inputs for all perceptrons with 2 inputs #
# perceptron_inputs = [
#     [1, 1],
#     [1, 0],
#     [0, 1],
#     [0, 0]
# ]

# # Defining AND perceptron #
# and_perceptron = Perceptron([0.5, 0.5], -0.5)
# and_answers = [
#     1,
#     0,
#     0,
#     0
# ]

# print("\n\nAND Perceptron\n")
# print_test(and_perceptron, perceptron_inputs, and_answers)

# # Defining OR perceptron #
# or_perceptron = Perceptron([1, 1], -0.5)
# or_answers = [
#     1,
#     1,
#     1,
#     0
# ]

# print("\n\nOR Perceptron\n")
# print_test(or_perceptron, perceptron_inputs, or_answers)