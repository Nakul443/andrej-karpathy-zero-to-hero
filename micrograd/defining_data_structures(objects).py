'''
We created class 'value' cuz:
    Once c is calculated, Python retains no memory of:

    Who created c (that c was produced by a and b).
    What operation created it (multiplication *).
    What its derivative / gradient is with respect to the output.


so we needed a class to store the data, the gradient, and the function that produced it.


To build an Autograd engine (automatic differentiation) like PyTorch or Micrograd,
every number must carry metadata along with its raw value
1. Tracking the graph of operations that produced it (the computational graph).
2. Storing the gradient of the final output with respect to itself (the derivative).
    a. storing the gradient in simple terms means that we can compute the gradient
    of the final output with respect to any input variable by backpropagating through the computational graph.

'''


class Value:
    # 1. we added the children parameter cuz we want to keep track of the Value objects that were used to create this Value object,
        # so that we can backpropagate through the computational graph later on.

    # 2. we added the op parameter cuz we want to keep track of the operation that was used to create this Value object,
        # so that we can backpropagate through the computational graph later on.
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self._prev = set(_children) # this is a set of the Value objects that were used to create this Value object
        self._op = _op # this is the operation that was used to create this Value object

    # If you omit __repr__,
    # Python falls back to its default representation showing only the class name and its memory address:
        # a = Value(2.0)
        # print(a)
        # Output: <__main__.Value object at 0x7f8b2c4e3a90>

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        # adding children to the _children set so that we can backpropagate through the computational graph later on
        return Value(self.data + other.data, _children=(self, other), _op='+')

    def __mul__(self, other):
        # adding children to the _children set so that we can backpropagate through the computational graph later on
        return Value(self.data * other.data, _children=(self, other), _op='*')

a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)
d = a*b + c
# print(a+b) # this will raise an error because we haven't defined how to add two Value objects together yet
# # internally python will call, a.__add__(b), which will return a new Value object with the data attribute set to the sum of the data attributes of a and b, which is -1.0

# print((a*b)+c)
# # internally python will call, (a.__mul__(b)).__add__(c)

print(d._prev) # this will print the set of Value objects that were used to create d, which is {a, b, c}
print(d._op) # this will print the operation that was used to create d, which is +