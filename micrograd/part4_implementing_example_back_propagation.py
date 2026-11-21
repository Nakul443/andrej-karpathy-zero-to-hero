# implementing an example of back propagation in a simple neural network using the micrograd library

from graph_visualisation import draw_dot

from micrograd.part2_defining_data_structures_objects import Value

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

#---------------------------------------------------------------------------------------------------------------------------------#

# implementation of back propagation in a simple neural network using the micrograd library

'''
# 1. first calculate do/do
'''
# base case: the gradient of the output with respect to itself is 1
# do/do = 1.0
o.grad = 1.0

'''
# 2. now we back propagate through the tanh

Step 2: n (parent is o, via tanh)
Local derivative: o = tanh(n), so do/dn = 1 - tanh(n)² = 1 - o²
do/dn = do/do * do/dn
      = 1.0 * (1 - 0.6044²)
      = 1.0 * 0.6347
      = 0.6347
'''

n.grad = o.grad * (1 - o.data**2)  # chain rule: (this is n.grad) do/dn = do/do * do/dn
# n.grad = do/dn
'''
in your current code:
    in your file, b = 6.7
        - n = 0.7
        - o = tanh(n)
        - o = tanh(0.7) = 0.6043677771171635
        so the local derivative of the tanh operation is: 1 - o**2 = 1 - 0.6043677771171635**2 = 0.635493
        so the gradient of n is: n.grad = o.grad * (1 - o**2) = 1.0 * 0.635493 = 0.635493
'''


'''
Step 3: x1w1x2w2 and b (parent is n, via +)

n = x1w1x2w2 + b, so the local derivatives are dn/d(x1w1x2w2) = 1 and dn/db = 1.

(this is x1w1x2w2.grad) do/d(x1w1x2w2) = do/dn * dn/d(x1w1x2w2) = 0.6347 * 1 = 0.6347
do/db = do/dn * dn/db = 0.6347 * 1 = 0.6347

x1w1x2w2.grad = 0.6347, b.grad = 0.6347
'''

# x1w1x2w2.grad = do/d(x1w1x2w2)
x1w1x2w2.grad = n.grad * 1.0


'''
Step 4: x1w1 and x2w2 (parent is x1w1x2w2, via +)

x1w1x2w2 = x1w1 + x2w2, so both local derivatives are 1.

The complete chain from o back to x1w1 has four links:

    do/d(x1w1) = do/dn * dn/d(x1w1x2w2) * d(x1w1x2w2)/d(x1w1)
    but the first two links were already multiplied together in an earlier step:
    do/d(x1w1x2w2) = do/dn * dn/d(x1w1x2w2)
                = 0.6347 * 1
                = 0.6347

So do/d(x1w1x2w2) is do/dn * dn/d(x1w1x2w2), bundled into one number.


When you write:
    do/d(x1w1) = do/d(x1w1x2w2) * d(x1w1x2w2)/d(x1w1)
and substitute that bundle in, you get the full chain:
    do/d(x1w1) = [do/dn * dn/d(x1w1x2w2)] * d(x1w1x2w2)/d(x1w1)
            = [do/d(x1w1x2w2)] * [d(x1w1x2w2)/d(x1w1)]
            = [0.6347 * 1] * 1
            = 0.6347

# cuz of everything above, we get:
(this is x1w1.grad) do/d(x1w1) = do/d(x1w1x2w2) * d(x1w1x2w2)/d(x1w1) = 0.6347 * 1 = 0.6347
(this is x2w2.grad) do/d(x2w2) = do/d(x1w1x2w2) * d(x1w1x2w2)/d(x2w2) = 0.6347 * 1 = 0.6347

x1w1.grad = 0.6347, x2w2.grad = 0.6347
'''

x1w1.grad = n.grad * 1.0 # x1w1.grad = do/d(x1w1)
x2w2.grad = n.grad * 1.0 # x2w2.grad = do/d(x2w2)



'''
Step 5: x1 and w1 (parent is x1w1, via *)

# do/d(x1w1) was calculated in the previous step
x1w1 = x1 * w1,
so local derivatives are: d(x1w1)/dx1 = w1 and d(x1w1)/dw1 = x1.

do/dx1 = do/d(x1w1) * d(x1w1)/dx1 = 0.6347 * w1 = 0.6347 * (-3.0) = -1.9042
do/dw1 = do/d(x1w1) * d(x1w1)/dw1 = 0.6347 * x1 = 0.6347 * (2.0)  =  1.2695
'''
x1.grad = w1.data * x1w1.grad
w1.grad = x1.data * x1w1.grad

'''
Step 6: x2 and w2 (parent is x2w2, via *)

# do/d(x2w2) was calculated in the previous step
x2w2 = x2 * w2, so d(x2w2)/dx2 = w2 and d(x2w2)/dw2 = x2.
do/dx2 = do/d(x2w2) * d(x2w2)/dx2 = 0.6347 * w2 = 0.6347 * 1.0 = 0.6347
do/dw2 = do/d(x2w2) * d(x2w2)/dw2 = 0.6347 * x2 = 0.6347 * 0.0 = 0.0
'''

x2.grad = w2.data * x2w2.grad
w2.grad = x2.data * x2w2.grad