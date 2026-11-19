"""
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

"""

# this class is a data structure that will hold the value of a number, its gradient, and the function that produced it
import math


class Value:
    # 1. we added the children parameter cuz we want to keep track of the Value objects that were used to create this Value object,
    # so that we can backpropagate through the computational graph later on.

    # 2. we added the op parameter cuz we want to keep track of the operation that was used to create this Value object,
    # so that we can backpropagate through the computational graph later on.
    def __init__(self, data, _children=(), _op="", label=""):
        self.data = data
        self.grad = 0.0  # initially each node has 0 gradient
        self._backward = lambda: (
            None
        )  # a function which will do the entire math of the chain rule
        self._prev = set(
            _children
        )  # this is a set of the Value objects that were used to create this Value object
        self._op = (
            _op  # this is the operation that was used to create this Value object
        )
        self.label = label  # this is a label for the Value object

    # If you omit __repr__,
    # Python falls back to its default representation showing only the class name and its memory address:
    # a = Value(2.0)
    # print(a)
    # Output: <__main__.Value object at 0x7f8b2c4e3a90>
    def __repr__(self):
        return f"Value(data={self.data})"

    # __rmul__ makes 2 * a work by flipping it to a * 2, since Python can’t multiply an int by a Value on its own.
    def __rmul__(self, other):
        return self * other
    
    

    # power function
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            # derivative of d(f(x)^a)/dx
            # multiplication with out.grad is the chain rule
            self.grad += other * (self.data**(other-1)) * out.grad
        out._backward = _backward
        return out

    # a * b = a * (1/b) = a * (b**-1)
    def __truediv__(self, other):
        return self * other**-1

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __radd__(self, other):
        return self + other

    # It’s a building block.
    # Later you can write tanh as (e^(2x) - 1) / (e^(2x) + 1) using only exp, +, * and division,
    # and backprop would still work without a dedicated tanh _backward.
    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,), 'exp')

        def _backward():
            self.grad += out.data * out.grad # cuz d/d(e^x) = e^x = out.data
            # do/dx (which is self grad here) = do/d(out) * d(out)/dx = out.grad * out.data

        out._backward = _backward
        return out

    def __add__(self, other):
        other = other if isinstance(other,Value) else Value(other)
        # adding children to the _children set so that we can backpropagate through the computational graph later on
        out = Value(
            self.data + other.data,
            _children=(self, other),
            _op="+",
            label=f"{self.label}+{other.label}",
        )

        def _backward():
            self.grad += 1.0 * out.grad  # d(output)/d(current node)
            other.grad += 1.0 * out.grad
            """
            Apply the chain rule:
                do/d(other) = do/d(out) * d(out)/d(other)
                            =  out.grad * 1.0
                which is the line you asked about:
                    other.grad = 1.0 * out.grad
            """

        out._backward = _backward  # storing the function inside the variable
        return out
        # this return will cause __init__ to be called, which will create a new Value object with the data attribute set to the sum of the data attributes of self and other, and the _children attribute set to a tuple containing self and other, and the _op attribute set to '+'.

    def __mul__(self, other):
        other = other if isinstance(other,Value) else Value(other)
        # adding children to the _children set so that we can backpropagate through the computational graph later on
        out = Value(
            self.data * other.data,
            _children=(self, other),
            _op="*",
            label=f"{self.label}*{other.label}",
        )

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        # implementation of the tanh activation function: tanh(x) = (e^(2x) - 1) / (e^(2x) + 1)
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Value(t, _children=(self,), _op="tanh", label=f"tanh({self.label})")

        def _backward():
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward  # we want to store the function
        return out

    def backward(self):
        # topological sort to order the computational graph
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # base case: derivative of the output with respect to itself
        self.grad = 1.0

        # Apply chain rule in reverse topological order
        for node in reversed(topo):
            node._backward()


a = Value(2.0, label="a")
b = Value(-3.0, label="b")
c = Value(10.0, label="c")
e = a * b
e.label = "e"
d = e + c
d.label = "d"
f = Value(-2.0, label="f")
L = d * f
L.label = "L"  # output of the graph


# print(a+b) # this will raise an error because we haven't defined how to add two Value objects together yet
# # internally python will call, a.__add__(b), which will return a new Value object with the data attribute set to the sum of the data attributes of a and b, which is -1.0

# print((a*b)+c)
# # internally python will call, (a.__mul__(b)).__add__(c)

print(
    d._prev
)  # this will print the set of Value objects that were used to create d, which is {a, b, c}
print(d._op)  # this will print the operation that was used to create d, which is +
"""
d  (+)
├── temp (*)      <- d._prev contains this
│   ├── a
│   └── b         <- temp._prev contains these
└── c

Quick way to see it

print(d._prev)         # {Value(data=-6.0), Value(data=10.0)}
print(d._op)           # +
for child in d._prev:
    print(child, child._op, child._prev)
"""


# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
