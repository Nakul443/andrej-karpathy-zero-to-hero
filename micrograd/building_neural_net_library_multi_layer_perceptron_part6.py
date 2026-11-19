# defining a neuron, a layer of neurons, and an MLP (multi-layer perceptron)

'''
The reason behind
'parameters()' function

Each class only knows about the level directly below it, so each one collects from its own children.

- A `Neuron` knows its own weights and bias.
- A `Layer` knows its neurons, but not their weights.
- An `MLP` knows its layers, but not their neurons.

So `mlp.parameters()` asks each layer, each layer asks each neuron, and each neuron hands back its own `w` and `b`. The lists get joined as they come back up.

The purpose is that you can call one method on the whole network and get all 41 parameters,
without digging through `mlp.layers[i].neurons[j].w[k]` by hand.
Training needs exactly that list, to nudge every weight and bias after `backward()`.

It also means each class stays responsible for its own piece.
If you later add a new kind of layer, you only write its `parameters()`, and `MLP` keeps working unchanged.
'''

import random

from defining_data_structures_objects_part2 import Value


class Neuron:
    # runs automatically when an object of class Neuron is created
    # nin is number of inputs
    def __init__(self, nin):
        # creates a list of random weights between -1 and 1 (one weight for each input) and wraps each one in a Value object.
        """
        The loop runs nin times, once per input, and makes one random weight each time. (_ means “I dont need the loop counter”.)
        One random bias is made.
        The weights and bias are Value objects, so they will get .grad later.
        """
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]

        # creates a single random bias value between -1 and 1 to control how easily the neuron fires.
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        # w * x + b
        # zip takes your list of weights (self.w) and your list of inputs (x) and matches them side-by-side into pairs.
        # print(list(zip(self.w, x)))
        # output: [(Value(data=-0.407163510057136), 2.0), (Value(data=-0.8144137030699063), 3.0)]

        act = sum(wi * xi for wi, xi in zip(self.w, x)) + self.b
        out = act.tanh()
        return out

    # returns a list of every learnable number in the neuron: all its weights and its bias.
    # parameters() gives you a list of the numbers inside the neuron that training is allowed to change: its weights and its bias.
    # For a neuron with 3 inputs, that’s [w1, w2, w3, b].
    # “Learnable” means these are the numbers the network adjusts to get better.
    # The inputs (x) are fixed data, so they aren’t in the list.
    def parameters(self):
        return self.w + [self.b]


# Runs a list of Neuron objects on the same input.
# Outputs a list of Values (or one Value if the layer has 1 neuron).
class Layer:
    # nin is the number of inputs that every neuron in this layer will receive.
    # nout is the number of outputs you want the layer to produce, which equals the number of neurons inside the layer.
    def __init__(self, nin, nout):
        """
        It calls Neuron(nin) nout times, which calls Neuron.__init__ nout times.
        Each neuron gets its own random weights and bias.
        Layer(3, 4) = “4 neurons, each taking 3 inputs”.
        """
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        # loops through every neuron n in self.neurons and feeds the exact same input list x into each one
        """
        It calls every neuron (via Neuron.__call__) with the same x, and collects the answers in a list.
        Four neurons give a list of four Values.
        """
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

    '''
    params = []
    for neuron in self.neurons:
        ps = neuron.parameters() # neuron.parameters() asks one neuron for its list, like [w1, w2, w3, b].
        params.extend(ps) # adds those items into params one by one (not as a nested list).
    return params
    '''




# connects multiple Layer objects in sequence
class MLP:
    # you pass: mlp = MLP(3, [4, 4, 1])
    # it means: 3 inputs, two hidden layers of 4 neurons each, then an output layer of 1 neuron
    # MLP.__init__(self, nin=3, nouts=[4, 4, 1]) runs
    def __init__(self, nin, nouts):
        # nin is a single integer representing the number of inputs entering the network (e.g., 3).
        # nouts is a list of integers defining the size of every layer in the network (e.g., [4, 4, 1]).

        """
        sz holds the sizes in order. The loop pairs neighbors, so each layer output size matches the next layer input size:

        i	Call	        Meaning
        0	Layer(3, 4)	    3 inputs in, 4 values out
        1	Layer(4, 4)	    4 in, 4 out
        2	Layer(4, 1)	    4 in, 1 final output
        """

        # sz = [nin] + nouts: combines the input size and layer sizes into one list of layer dimensions
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i + 1]) for i in range(len(nouts))]

        """
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]
        pairs consecutive numbers from sz to build each Layer so the output size of one layer matches the input size of the next:
        i = 0: Layer(3, 4) (takes 3 inputs, outputs 4 values)
        i = 1: Layer(4, 4) (takes 4 inputs, outputs 4 values)
        i = 2: Layer(4, 1) (takes 4 inputs, outputs 1 final value)
        """

    # if you call: mlp([2.0, 3.0, -1.0])
    def __call__(self, x):
        """
        x = [2.0, 3.0, -1.0]                (3 floats)
        x = layer1(x)  ->  [v, v, v, v]     (4 Values)
        x = layer2(x)  ->  [v, v, v, v]     (4 Values)
        x = layer3(x)  ->  v                (1 Value, the final output)
        """
        # loops through the layers sequentially from input to output.
        for layer in self.layers:
            # feeds the current data x into a layer and overwrites x with that layer's output.
            # This passes the transformed data forward as the input to the next layer.
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    # Each layer.parameters() returns that layer’s list, and the inner loop adds its items one by one.
    # For MLP(3, [4, 4, 1]), you get one list of 41 items: 16 + 20 + 5
    '''
        params = []
        for layer in self.layers:
            for p in layer.parameters():
                params.append(p)
        return params
    '''

x = [
    2.0,
    3.0,
    -1.0,
]  # creates a neuron that expects 3 inputs, generating 3 random weights and 1 random bias.
n = MLP(3, [4, 4, 1])
# print(n(x))


"""
mlp = MLP(3, [4, 4, 1])
 └─ MLP.__init__
     └─ Layer.__init__  x3
         └─ Neuron.__init__  x(4, 4, 1)
             └─ Value.__init__  for each w and b

out = mlp([2.0, 3.0, -1.0])
 └─ MLP.__call__
     └─ Layer.__call__  x3 (one after another)
         └─ Neuron.__call__  (once per neuron in that layer)
             ├─ Value.__mul__ (per input)
             ├─ Value.__radd__ / __add__ (sum and bias)
             └─ Value.tanh
"""


# ------------------------------------------------------------------------------------------------------------------------

"""
taking an example dataset
"""

xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]

ys = [1.0, -1.0, -1.0, 1.0]  # desired targets
ypred = [n(x) for x in xs]
print(ypred)

"""
the predictions will differ from the targets, so we write a loss function
that measures how wrong the network is. We want to minimise it.

loss = sum of squared errors over all examples
"""

# yout = prediction from the network, ygt = ground truth target (from ys)
loss = sum(
    (((yout[0] if isinstance(yout, list) else yout) - ygt) ** 2 for ygt, yout in zip(ys, ypred)),
    Value(0.0),
)

'''
# loss = sum((yout - ygt) ** 2 for ygt, yout in zip(ys, ypred))

explanation:
loss = 0
for ygt, yout in zip(ys, ypred):
    loss = loss + (yout - ygt) ** 2
zip(ys, ypred) pairs each target with its prediction: (1.0, pred1), (-1.0, pred2), and so on.

yout - ygt is the error: how far the prediction is from the target.

** 2 squares it, so the error is never negative and big mistakes count more.

sum(...) adds the 4 squared errors into one number.
'''

print(loss)


"""
now we want to minimise the loss
"""

# It computes the gradient of the loss with respect to every node that was used to calculate it,
# and stores each one in that node’s .grad.
# It does not change any weights. It only fills in .grad.
assert isinstance(loss, Value)
loss.backward()

print(n.layers[0].neurons[0].w[0].grad)
# so this tells us if we slightly increase this particular weight
# of this particular neuron of this particular layer, we can reduce the loss


# -------------------------------------------------------------------------------------------------------------------------------

'''
implementing gradient descent:

Think of the loss as a hill, and you want to walk down to the bottom (lowest loss).

`p.grad` tells you which way is **uphill** for that weight. So you step the **opposite way**, which is downhill.

**Tiny example with one weight:**

- weight = 5.0, loss = 10, grad = +2.0
- grad is positive, so increasing the weight makes the loss go **up**
- so you decrease it: `5.0 + (-0.01 * 2.0) = 4.98`
- the loss is now slightly lower than 10

If the grad were -2.0 (negative), increasing the weight would lower the loss,
so the same line makes it go **up**: `5.0 + (-0.01 * -2.0) = 5.02`.

The `-` sign always flips the direction so you go downhill.
The `0.01` just keeps the step small, so you don't jump past the bottom.

The loop does this for every weight and bias at once.

'''

# training loop which makes the corrections
# you run this many times manually you will see the loss go lower and lower
# for p in n.parameters():
#     p.data += -0.01 * p.grad


#------------------------------------------------------------------------------------------------------
# now we will implement the above correction loop properly

for k in range(20):

    # forward pass
    ypred = [n(x) for x in xs]
    loss = Value(0.0)
    for ygt, yout in zip(ys, ypred):
        prediction = yout[0] if isinstance(yout, list) else yout
        loss = loss + (prediction - ygt) ** 2
    # loss = sum((yout - ygt)**2 for ygt, yout in zip(ys,ypred))

    # backward pass
    for p in n.parameters():
        p.grad = 0.0

    loss.backward()
    # loss.backward()

    # update
    for p in n.parameters():
        p.data += -0.05 * p.grad

    print(k, loss.data)
    # print(k, loss.data)
