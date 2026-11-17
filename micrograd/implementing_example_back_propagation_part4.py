# implementing an example of back propagation in a simple neural network using the micrograd library

from defining_data_structures_objects_part2 import Value
from graph_visualisation import draw_dot

# input x1,x2
x1 = Value(2.0, label="x1")
x2 = Value(0.0, label="x2")

# weights w1,w2
w1 = Value(-3.0, label="w1")
w2 = Value(1.0, label="w2")

# bias of the neuron
b = Value(6.7, label="b")

# x1*w1 + x2*w2 + b
x1w1 = x1 * w1
x1w1.label = "x1*w1"
x2w2 = x2 * w2
x2w2.label = "x2*w2"
x1w1x2w2 = x1w1 + x2w2
x1w1x2w2.label = "x1*w1 + x2*w2"
n = x1w1x2w2 + b # this is the output of the neuron before applying the activation function
n.label = "n"
o = n.tanh() # output of the neuron
o.label = "o"
'''
# the tanh function is used as the activation function of the neuron,
# and it is applied to the output of the neuron before returning the final output of the neuron.
# what it does is that it squashes the output of the neuron to be between -1 and 1,
# which is useful for training neural networks.

'''

draw_dot(o, format="svg", rankdir="LR").render(
    "graph", view=True
)  # render the graph to a file and open it in the default viewer

'''

# what the output shows is that the computational graph is built from the
# input values, weights, and bias, and the operations that are performed on them.
# The output of the neuron is calculated by performing the operations in the order they were defined,
# and the gradients are calculated by backpropagating through the graph.
# The gradients are stored in the grad attribute of each Value object,
# and can be accessed after calling the backward() method on the output Value object.

'''