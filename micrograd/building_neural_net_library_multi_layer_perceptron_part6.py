# defining a basic single artificial neuron

import random

from defining_data_structures_objects_part2 import Value


class Neuron:
    # runs automatically when an object of class Neuron is created
    # nin is number of inputs
    def __init__(self, nin):

        # creates a list of random weights between -1 and 1 (one weight for each input) and wraps each one in a Value object.
        '''
        The loop runs nin times, once per input, and makes one random weight each time. (_ means “I dont need the loop counter”.)
        One random bias is made.
        The weights and bias are Value objects, so they will get .grad later.
        '''
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]

        # creates a single random bias value between -1 and 1 to control how easily the neuron fires.
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        # w * x + b
        # zip takes your list of weights (self.w) and your list of inputs (x) and matches them side-by-side into pairs.
        # print(list(zip(self.w, x)))
        # output: [(Value(data=-0.407163510057136), 2.0), (Value(data=-0.8144137030699063), 3.0)]

        act = sum(wi*xi for wi, xi in zip(self.w, x)) + self.b
        out = act.tanh()
        return out

# Runs a list of Neuron objects on the same input and outputs a list of Value objects.
class Layer:

    # nin is the number of inputs that every neuron in this layer will receive.
    # nout is the number of outputs you want the layer to produce, which equals the number of neurons inside the layer.
    def __init__(self, nin, nout):
        '''
        It calls Neuron(nin) nout times, which calls Neuron.__init__ nout times.
        Each neuron gets its own random weights and bias.
        Layer(3, 4) = “4 neurons, each taking 3 inputs”.
        '''
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        # loops through every neuron n in self.neurons and feeds the exact same input list x into each one
        '''
        It calls every neuron (via Neuron.__call__) with the same x, and collects the answers in a list.
        Four neurons give a list of four Values.
        '''
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

# connects multiple Layer objects in sequence
class MLP:
    # you pass: mlp = MLP(3, [4, 4, 1])
        # it means: 3 inputs with 2 layers of 4 and 1 output
    # MLP.__init__(self, nin=3, nouts=[4, 4, 1]) runs
    def __init__(self, nin, nouts):
        # nin is a single integer representing the number of inputs entering the network (e.g., 3).
        # nouts is a list of integers defining the size of every layer in the network (e.g., [4, 4, 1]).

        '''
        sz holds the sizes in order. The loop pairs neighbors, so each layer output size matches the next layer input size:

        i	Call	        Meaning
        0	Layer(3, 4)	    3 inputs in, 4 values out
        1	Layer(4, 4)	    4 in, 4 out
        2	Layer(4, 1)	    4 in, 1 final output
        '''

        # sz = [nin] + nouts: combines the input size and layer sizes into one list of layer dimensions
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

        '''
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]
        pairs consecutive numbers from sz to build each Layer so the output size of one layer matches the input size of the next:
        i = 0: Layer(3, 4) (takes 3 inputs, outputs 4 values)
        i = 1: Layer(4, 4) (takes 4 inputs, outputs 4 values)
        i = 2: Layer(4, 1) (takes 4 inputs, outputs 1 final value)
        '''


    # if you call: mlp([2.0, 3.0, -1.0])
    def __call__ (self, x):
        '''
            x = [2.0, 3.0, -1.0]                (3 floats)
            x = layer1(x)  ->  [v, v, v, v]     (4 Values)
            x = layer2(x)  ->  [v, v, v, v]     (4 Values)
            x = layer3(x)  ->  [v]              (1 Value, the final output)
        '''
        # loops through the layers sequentially from input to output.
        for layer in self.layers:

        # feeds the current data x into a layer and overwrites x with that layer's output.
        # This passes the transformed data forward as the input to the next layer.
            x = layer(x)
        return x

x = [
    2.0,
    3.0,
    -1.0
]  # creates a neuron that expects 3 inputs, generating 3 random weights and 1 random bias.
n = MLP(3, [4, 4, 1])
print(n(x))


'''
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
'''