# notes

## Micrograd project notes

This project learns one idea step by step: a number should keep information about how it was calculated and how it affects the final output.

This is the basic idea behind an autograd engine. Autograd means automatic calculation of derivatives.

## 1. Why derivatives are needed

The code first looks at this function:

```python
f(x) = 3x^2 - 4x + 5
```

### What a derivative means

A derivative asks:

> If x changes by a small amount, how much does f(x) change?

The code calculates this with a small step called h:

```python
slope = (f(x + h) - f(x)) / h
```

Why this works:

- `f(x + h) - f(x)` is how much the output changed.
- `h` is how much the input changed.
- dividing output change by input change gives the output change per unit of input change.
- this is the slope of the function at x.
- as h becomes smaller, the slope becomes closer to the true derivative.

For the more complex example:

```python
a = 2.0
b = -3.0
c = 10.0
d = a * b + c
```

The code changes one variable at a time and keeps every other variable fixed.

### Influence of a on d

```python
d1 = a * b + c
a += h
d2 = a * b + c
slope = (d2 - d1) / h
```

The local derivative of d with respect to a is b:

```python
dd/da = b = -3
```

Increasing a makes d decrease in this example because b is negative.

### Influence of b on d

```python
d1 = a * b + c
b += h
d2 = a * b + c
slope = (d2 - d1) / h
```

The local derivative of d with respect to b is a:

```python
dd/db = a = 2
```

Increasing b makes d increase because a is positive.

### Influence of c on d

```python
d1 = a * b + c
c += h
d2 = a * b + c
```

Since c is directly added to d:

```python
dd/dc = 1
```

Changing c by h changes d by h. Therefore, the slope is h / h = 1.

## 2. Why a normal Python number is not enough

Suppose this code is run:

```python
a = 2.0
b = -3.0
c = 10.0
d = a * b + c
```

After Python calculates d, it only stores d's final number. It does not remember:

- that d came from a, b, and c
- that d used a multiplication and an addition
- whether d is used by a later operation
- what derivative d has with respect to the final output

A neural network needs this information. Therefore, the code creates a custom class called `Value`.

## 3. The Value class

Each `Value` object holds information about one number in the computation:

```python
class Value:
    def __init__(self, data, _children=(), _op="", label=""):
```

### data

`data` is the actual number:

```python
a = Value(2.0)
```

Here `a.data` is `2.0`.

### grad

`grad` means gradient.

It starts at `0.0`:

```python
self.grad = 0.0
```

The gradient stores:

> how much the final output changes when this node changes

For example, `d.grad` is dL/dd. It says how much L changes for a small change in d.

The gradient is initially 0 because no backpropagation has happened yet.

### _prev

`_prev` means previous nodes.

It stores the `Value` objects that were used to create this node:

```python
self._prev = set(_children)
```

For:

```python
e = a * b
```

`e._prev` contains `a` and `b`.

For:

```python
d = e + c
```

`d._prev` contains `e` and `c`.

The set is used because the graph only needs to remember the direct parent nodes. It does not need to remember duplicate child entries.

### _op

`_op` stores the operation used to create the node:

- `"+"` for addition
- `"*"` for multiplication
- `"tanh"` for tanh

It is mainly useful for graph visualisation and understanding the graph structure.

### label

`label` is a readable name such as `"a"`, `"x1"`, or `"o"`.

It is not needed for the math. It makes printed graphs and debugging easier.

### __repr__

`__repr__` controls how a `Value` looks when printed:

```python
def __repr__(self):
    return f"Value(data={self.data})"
```

Without this, Python would print a memory address such as `<__main__.Value object at ...>`.

## 4. Addition in the Value class

Python calls `__add__` when code uses:

```python
e + c
```

The important first step is:

```python
out = Value(
    self.data + other.data,
    _children=(self, other),
    _op="+",
)
```

This creates a new `Value` containing the sum. It also stores both input nodes in `_prev`.

Example:

```python
e = a * b
d = e + c
```

Now `d.data` is the sum and `d._prev` is `{e, c}`.

### Addition's local derivatives

For:

```python
out = self + other
```

the derivatives are:

```python
dout/dself = 1
dout/dother = 1
```

This is why changing either input by h changes the sum by h.

The addition `_backward()` function applies the chain rule:

```python
def _backward():
    self.grad += 1.0 * out.grad
    other.grad += 1.0 * out.grad
```

If `out.grad` is g, both inputs receive g:

```python
self.grad += g
other.grad += g
```

### Why += is used

The code uses `+=`, not `=`, because one node may receive gradients from multiple places.

Every gradient contribution must be summed. Assigning with `=` could erase earlier contributions.

## 5. Multiplication in the Value class

Python calls `__mul__` when code uses:

```python
a * b
```

The code creates an output node and stores both parents:

```python
out = Value(
    self.data * other.data,
    _children=(self, other),
    _op="*",
)
```

Example:

```python
e = a * b
```

Now:

- `e.data = a.data * b.data`
- `e._prev = {a, b}`
- `e._op = "*"`

### Multiplication's local derivatives

For:

```python
out = self * other
```

the local derivatives are:

```python
dout/dself = other.data
dout/dother = self.data
```

The code stores this in `_backward()`:

```python
def _backward():
    self.grad += other.data * out.grad
    other.grad += self.data * out.grad
```

Why the other node's value is used:

- if `out = self * other`
- to find how `self` affects `out`, hold `other` fixed
- then one unit of change in `self` changes `out` by `other`
- therefore multiply `other.data` by the gradient reaching the output

The chain rule gives:

```python
self.grad = dout/dself * doutGrad
          = other.data * out.grad

other.grad = dout/dother * doutGrad
           = self.data * out.grad
```

## 6. tanh in the Value class

The code implements tanh as:

```python
t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
```

The result is the hyperbolic tangent function:

```text
o = tanh(n)
```

It squashes a number so the output is between -1 and 1:

- positive inputs become positive values near 1
- negative inputs become negative values near -1
- inputs near 0 stay near 0

It is useful in the tiny neuron because it gives a smooth, bounded output.

### Creating the tanh node

```python
out = Value(
    t,
    _children=(self,),
    _op="tanh",
)
```

This means:

- `out.data` is the tanh result
- `out._prev` contains the input node
- `out._op` is `"tanh"`

### tanh's local derivative

The derivative of tanh is:

```python
dt/dx = 1 - t^2
```

The code stores:

```python
def _backward():
    self.grad = (1 - t**2) * out.grad
```

Using the chain rule:

```python
dxGrad = dt/dx * outGrad
       = (1 - t**2) * out.grad
```

The note uses `(1 - t^2)` because `t` is already the tanh output:

```text
t = tanh(x)
```

It can also be written:

```text
dt/dx = 1 - tanh(x)^2
```

## 7. Building the first computational graph

The code builds:

```python
a = Value(2.0, label="a")
b = Value(-3.0, label="b")
c = Value(10.0, label="c")

e = a * b
e.label = "e"

d = e + c
d.label = "d"

f = Value(-2.0, label="f")

L = d * f
L.label = "L"
```

The graph is:

```text
L = d * f
d = e + c
e = a * b
```

Expanded as a tree:

```text
L
├── d
│   ├── e
│   │   ├── a
│   │   └── b
│   └── c
└── f
```

The values are:

```text
a = 2
b = -3
c = 10
e = a * b = -6
d = e + c = 4
f = -2
L = d * f = -8
```

## 8. Derivations for the L graph

Backpropagation starts at the output and walks backward.

### Step 1: derivative of L with respect to itself

When a value is compared with itself:

```python
dL/dL = 1
```

This is the base case.

It means the final output changes by 1 when the final output itself changes by 1.

The code can set:

```python
L.grad = 1.0
```

### Step 2: dL/dd

The operation is:

```python
L = d * f
```

For a multiplication, the derivative with respect to d is f:

```python
dL/dd = f = -2
```

This means increasing d by 1 decreases L by about 2.

### Step 3: dL/df

For the same multiplication:

```python
dL/df = d = 4
```

This means increasing f by 1 increases L by about 4.

### Step 4: dd/de

The operation is:

```python
d = e + c
```

Both inputs have derivative 1:

```python
dd/de = 1
dd/dc = 1
```

### Step 5: dL/de

L does not directly depend on e. It reaches L through d.

The chain rule is:

```python
dL/de = dL/dd * dd/de
      = -2 * 1
      = -2
```

Reading the chain:

- e changes d
- d changes L
- the total effect of e on L is -2

### Step 6: dL/dc

The chain is:

```python
dL/dc = dL/dd * dd/dc
      = -2 * 1
      = -2
```

This says changing c changes d by 1, and changing d changes L by -2.

### Step 7: dL/da

The operations are:

```text
e = a * b
d = e + c
L = d * f
```

The local derivatives are:

```python
de/da = b = -3
dd/de = 1
dL/dd = -2
```

The chain rule joins them:

```python
dL/da = dL/dd * dd/de * de/da
      = -2 * 1 * -3
      = 6
```

Changing a has a positive final effect on L in this example.

### Step 8: dL/db

The local derivatives are:

```python
de/db = a = 2
dd/de = 1
dL/dd = -2
```

The chain rule is:

```python
dL/db = dL/dd * dd/de * de/db
      = -2 * 1 * 2
      = -4
```

Changing b has a negative final effect on L.

## 9. Checking derivatives with small changes

The code can calculate derivatives using a small h:

```python
L1 = L.data
```

This is not enough by itself. The code would need to rebuild the graph or temporarily change a node, calculate a new L, then divide the output difference by h.

The finite-difference form is:

```python
dL/da = (L(a + h) - L(a - h)) / (2h)
```

The code comments show the smaller one-sided form:

```python
f(x + h) - f(x) / h
```

Both describe a numerical derivative. Backpropagation calculates the same derivatives analytically.

Advantages of backpropagation:

- it is faster
- it uses stored graph information
- it can calculate every input gradient in one backward pass
- it does not repeatedly rebuild every input

## 10. Graph visualisation

`graph_visualisation.py` uses Graphviz to draw the computational graph.

### trace()

`trace(root)` is the function that finds every node and every edge:

```python
nodes, edges = set(), set()
```

It starts at the root node and recursively visits `_prev`.

For every node it finds:

- the node is added to `nodes`
- every parent-to-node connection is added to `edges`

The recursion stops when a node is already in `nodes`. This prevents repeated visits.

### draw_dot()

`draw_dot(root)` creates a Graphviz graph:

- `"LR"` means left to right
- `"TB"` means top to bottom

Each node shows:

- label
- data
- grad

Example node text:

```text
a | data 2.0 | grad 0.0
```

An operation is shown as a small separate node:

```text
*  or  +  or  tanh
```

The graph is useful because it makes this chain visible:

```text
input/weights -> multiplication -> addition -> tanh -> output
```

## 11. Building a tiny neuron

The example uses:

```python
x1 = Value(2.0, label="x1")
x2 = Value(0.0, label="x2")

w1 = Value(-3.0, label="w1")
w2 = Value(1.0, label="w2")

b = Value(6.8813735870195432, label="b")
```

These names mean:

- `x1` and `x2` are inputs
- `w1` and `w2` are weights
- `b` is the bias

The neuron does a weighted sum:

```python
n = x1*w1 + x2*w2 + b
```

The result is passed through tanh:

```python
o = n.tanh()
```

## 12. Forward pass of the neuron

The forward pass calculates values from left to right.

### Multiplication 1

```python
x1w1 = x1 * w1
```

Calculation:

```text
x1w1 = 2.0 * -3.0 = -6.0
```

### Multiplication 2

```python
x2w2 = x2 * w2
```

Calculation:

```text
x2w2 = 0.0 * 1.0 = 0.0
```

### Addition 1

```python
x1w1x2w2 = x1w1 + x2w2
```

Calculation:

```text
x1w1x2w2 = -6.0 + 0.0 = -6.0
```

### Addition 2

```python
n = x1w1x2w2 + b
```

Calculation:

```text
n = -6.0 + 6.8813735870195432
n = 0.8813735870195432
```

The bias moves the pre-activation value to the number 0.8813735870195432.

### Activation

```python
o = n.tanh()
```

Using tanh:

```text
o = tanh(0.8813735870195432)
o = 0.7071067811865476
```

The final forward-pass graph is:

```text
x1 -> * -> x1w1
x2 -> * -> x2w2
x1w1 + x2w2 -> x1w1x2w2
x1w1x2w2 + b -> n
n -> tanh -> o
```

## 13. Manual backpropagation through the neuron

Backpropagation uses the chain rule:

```text
nextGradient * localDerivative
```

Every node passes its gradient backward. Every operation multiplies the incoming gradient by its local derivative.

### Step 1: base gradient

The output is compared with itself:

```text
do/do = 1
```

The code sets:

```python
o.grad = 1.0
```

This means backpropagation starts with `o.grad = 1.0`.

### Step 2: tanh

The operation is:

```text
o = tanh(n)
```

The local derivative is:

```text
do/dn = 1 - o^2
```

For this example:

```text
o = 0.7071067811865476
o^2 = 0.5
do/dn = 1 - 0.5 = 0.5
```

The chain rule gives:

```text
n.grad = o.grad * do/dn
n.grad = 1.0 * 0.5
n.grad = 0.5
```

The node n now knows that increasing n by a small amount increases o by about 0.5.

### Step 3: n = sum + b

The operation is:

```text
n = x1w1x2w2 + b
```

Local derivatives:

```text
dn/d(x1w1x2w2) = 1
dn/db = 1
```

Therefore:

```text
x1w1x2w2.grad = n.grad * 1 = 0.5
b.grad = n.grad * 1 = 0.5
```

Both branches receive the same gradient because addition sends the full gradient to both inputs.

### Step 4: sum of x1w1 and x2w2

The operation is:

```text
x1w1x2w2 = x1w1 + x2w2
```

Again, addition gives a local derivative of 1 to both inputs:

```text
x1w1.grad = x1w1x2w2.grad * 1 = 0.5
x2w2.grad = x1w1x2w2.grad * 1 = 0.5
```

### Step 5: x1w1 = x1 * w1

The operation is:

```text
x1w1 = x1 * w1
```

Local derivatives:

```text
d(x1w1)/dx1 = w1 = -3
d(x1w1)/dw1 = x1 = 2
```

For x1:

```text
x1.grad = x1w1.grad * w1
x1.grad = 0.5 * -3
x1.grad = -1.5
```

For w1:

```text
w1.grad = x1w1.grad * x1
w1.grad = 0.5 * 2
w1.grad = 1.0
```

Interpretation:

- x1 has gradient -1.5
- increasing x1 with w1 fixed at -3 decreases o
- w1 has gradient 1.0
- increasing w1 with x1 fixed at 2 increases o

### Step 6: x2w2 = x2 * w2

The operation is:

```text
x2w2 = x2 * w2
```

Local derivatives:

```text
d(x2w2)/dx2 = w2 = 1
d(x2w2)/dw2 = x2 = 0
```

For x2:

```text
x2.grad = x2w2.grad * w2
x2.grad = 0.5 * 1
x2.grad = 0.5
```

For w2:

```text
w2.grad = x2w2.grad * x2
w2.grad = 0.5 * 0
w2.grad = 0
```

W2 has zero gradient because x2 is currently zero.

Changing w2 does not affect x2w2 right now because x2w2 = x2 * w2 and x2 is 0.

It may become non-zero at another set of input values.

## 14. Storing _backward() on every operation

The `Value` class gives every result a `_backward` function.

### Default backward

Every node starts with:

```python
self._backward = lambda: None
```

This means an input node such as a, b, c, or bias has no parents to update.

It may still receive a gradient, but it does not need to send anything farther backward.

### Addition backward

When addition creates an output:

```python
def _backward():
    self.grad += 1.0 * out.grad
    other.grad += 1.0 * out.grad
```

The operation stores the local derivative 1 and uses `out.grad` as the incoming gradient.

### Multiplication backward

When multiplication creates an output:

```python
def _backward():
    self.grad += other.data * out.grad
    other.grad += self.data * out.grad
```

The operation stores the required values and applies the chain rule later.

### tanh backward

When tanh creates an output:

```python
def _backward():
    self.grad = (1 - t**2) * out.grad
```

The tanh input receives `(1 - tanh(x)^2)` times the gradient reaching tanh.

## 15. Current manual backward calls

In `automating_back_propagation_part5.py`, the backward pass is now called by operations:

```python
o.grad = 1.0
o._backward()
n._backward()
b._backward()
x1w1x2w2._backward()
x1w1._backward()
x2w2._backward()
```

The calls run from the output backward to the inputs.

Why this order:

- o must calculate n.grad before n is used.
- n must calculate its parent gradients before the parent nodes are used.
- every node needs `out.grad` to be ready before calling `_backward()`.

This is manually reversed order in the current file.

### Limit

Manual calls work for this tiny graph, but they are easy to get wrong in a large graph:

- one node could be missed
- one node could be visited twice
- two branches could use gradients before they are complete
- every new operation would need a new manual call

The next step in a full engine is to calculate a topological order automatically.

## 16. Automated graph traversal

A full automatic backward pass should:

1. start at the output node
2. recursively visit every node in `_prev`
3. collect each node only once
4. put nodes in forward/topological order
5. reverse that list
6. set the output gradient to 1
7. call every `_backward()` function

Topological order means:

- every parent appears before its child
- after reversing the list, every child appears before its parents

This guarantees that the output is processed first and the inputs are processed last.

## 17. Important code details

### Python operator methods

Writing:

```python
a * b
```

causes Python to call:

```python
a.__mul__(b)
```

Writing:

```python
a + b
```

causes Python to call:

```python
a.__add__(b)
```

Writing:

```python
n.tanh()
```

calls the method directly.

This is how normal Python math becomes a custom computational graph.

### Why labels are assigned after operations

The code writes:

```python
e = a * b
e.label = "e"
```

This is needed because `_op` and `_children` are enough for math, but they do not automatically create a human-readable name.

Adding labels after operations makes graph nodes easier to read.

### Why gradients are local and global

Every `_backward()` function calculates a local derivative.

Examples:

- addition: local derivative is 1
- multiplication: local derivative is the other input
- tanh: local derivative is `1 - t^2`

The incoming gradient is global.

It says how the final output is affected by this node.

The global gradient of a parent is:

```text
parentGrad = localDerivative * out.grad
```

This is the chain rule.

## 18. Summary of the full idea

Forward pass:

1. a normal number is wrapped in `Value`
2. operations create more `Value` objects
3. each object stores its value
4. each object stores which parent objects created it
5. each object stores what operation created it

Backward pass:

1. set the final output gradient to 1
2. move backward through the graph
3. each operation calculates its local derivative
4. each operation sends `localDerivative * incomingGradient` to its parents
5. every input and weight receives its final gradient

## 19. More operations were added to `Value`

The `Value` class was made more useful so it could build more kinds of graphs.

### Power

```python
def __pow__(self, other):
```

This supports expressions like:

```python
x ** 2
```

The code only allows powers with a number:

```python
assert isinstance(other, (int, float))
```

The derivative of `x^a` is:

```text
a * x^(a-1)
```

For example, the derivative of `x^2` is `2x`.

The backward function is:

```python
self.grad += other * (self.data ** (other - 1)) * out.grad
```

This combines the local derivative with the gradient coming from the output.

### Negative, subtraction, and division

These operations reuse the operations that already exist:

```python
def __neg__(self):
    return self * -1

def __sub__(self, other):
    return self + (-other)

def __truediv__(self, other):
    return self * other**-1
```

This means:

- `-x` means `x * -1`
- `x - y` means `x + (-y)`
- `x / y` means `x * (y ** -1)`

So division and subtraction do not need separate backward formulas.

### Reverse multiplication and addition

Python may call:

```python
def __rmul__(self, other):
    return self * other
```

This allows:

```python
2 * n
```

when `2` is on the left and `n` is a `Value`.

`__radd__` supports addition in the reverse direction:

```python
def __radd__(self, other):
    return self + other
```

This is useful for sums that start with `0`.

### Exp

The `exp()` method creates:

```python
out = Value(math.exp(x), (self,), "exp")
```

The derivative of `e^x` is `e^x`:

```python
self.grad += out.data * out.grad
```

`out.data` is used because it already equals `e^x`.

---

## 20. Automatic backward pass

The `Value.backward()` method now removes the need to call each `_backward()` method by hand.

It uses topological sorting.

### Topological order

The code starts with:

```python
topo = []
visited = set()
```

It recursively visits every parent node:

```python
def build_topo(v):
    if v not in visited:
        visited.add(v)
        for child in v._prev:
            build_topo(child)
        topo.append(v)
```

The result is a list where parent nodes come before child nodes.

For example:

```text
a
b
e = a*b
c
d = e+c
f
L = d*f
```

The important rule is:

> a node must come after every node that created it

### Reversed order

The code sets the final output gradient to 1:

```python
self.grad = 1.0
```

Then it reverses the topological list:

```python
for node in reversed(topo):
    node._backward()
```

Now backward pass starts at the output and moves toward the inputs.

This also ensures a node sends its gradient only after it has received the full gradient.

---

## 21. tanh was built from basic operations

Later code builds tanh with simple operations instead of one special tanh function:

```text
tanh(n) = (e^(2n) - 1) / (e^(2n) + 1)
```

The code is:

```python
e = (2 * n).exp()
o = (e - 1) / (e + 1)
```

The graph now has these steps:

```text
n -> 2*n -> exp -> e -> e-1 -> divide -> o
```

It also needs `(e + 1)`.

The extra parentheses make the whole value the denominator.

This works because:

- multiplication knows its backward rule
- subtraction uses addition and negation
- exp knows its backward rule
- division uses multiplication and power

No separate tanh backward rule is needed.

---

## 22. Same result with PyTorch

The PyTorch file checks the same neuron.

It creates tensors with:

```python
requires_grad=True
```

This tells PyTorch to track the operations and keep gradients.

The forward pass is:

```python
n = x1*w1 + x2*w2 + b
o = torch.tanh(n)
```

Calling:

```python
o.backward()
```

sets the output gradient and calculates gradients for every tensor that needs one.

The code prints gradients such as:

```python
x1.grad
w1.grad
x2.grad
w2.grad
```

The result should match the Micrograd version.

`.item()` turns a one-value tensor into a normal Python number.

PyTorch is faster and supports more operations, tensors, and GPUs.

Micrograd shows the small version of the same idea.

---

## 23. Building larger networks

The next code builds classes for:

- neuron
- layer
- multi-layer perceptron

### Neuron

A neuron gets the number of inputs:

```python
Neuron(nin)
```

It creates:

- one random weight for every input
- one random bias

Weights and bias are `Value` objects.

The neuron calculates:

```python
act = sum(wi * xi for wi, xi in zip(self.w, x)) + self.b
out = act.tanh()
```

This is:

```text
weighted sum + bias -> tanh
```

### Layer

A layer contains many neurons.

For example:

```python
Layer(3, 4)
```

means:

- 3 inputs
- 4 neurons
- every neuron takes 3 inputs

The layer sends the same input to every neuron.

If it has 1 neuron, it returns one `Value`.

If it has many neurons, it returns a list.

### MLP

MLP means multi-layer perceptron.

Example:

```python
MLP(3, [4, 4, 1])
```

means:

```text
3 inputs
4 neurons
4 neurons
1 output
```

The MLP sends the output of one layer into the next layer.

### parameters()

Every class has a `parameters()` method:

- `Neuron`: returns its weights and bias
- `Layer`: joins the parameters of all its neurons
- `MLP`: joins the parameters of all its layers

For `MLP(3, [4, 4, 1])`, the network has 41 learnable values.

---

## 24. Loss

The example data has inputs `xs` and wanted outputs `ys`.

The network makes a prediction for each input:

```python
ypred = [n(x) for x in xs]
```

The code compares every prediction with its wanted result:

```text
error = prediction - target
squared error = error ** 2
```

The squared error is never negative.

Larger mistakes are counted more heavily.

The total loss adds all squared errors:

```python
loss = Value(0.0)
for ygt, yout in zip(ys, ypred):
    prediction = yout[0] if isinstance(yout, list) else yout
    loss = loss + (prediction - ygt) ** 2
```

The goal is to make this loss as small as possible.

---

## 25. Backward pass and gradient descent

Calling:

```python
loss.backward()
```

calculates how the loss changes with every weight and bias.

It fills each parameter's `.grad`.

It does not change the parameter values.

The update rule is:

```python
p.data += -learning_rate * p.grad
```

The negative sign moves opposite to the gradient.

If the gradient is positive:

- increasing the weight increases loss
- decrease the weight

If the gradient is negative:

- increasing the weight decreases loss
- increase the weight

The learning rate controls the step size.

The code uses `0.05`.

---

## 26. Training loop

The training loop repeats these steps:

### 1. Forward pass

Make predictions for all input values.

### 2. Calculate loss

Compare predictions with wanted outputs.

### 3. Clear old gradients

Before backward pass:

```python
for p in n.parameters():
    p.grad = 0.0
```

This is required because backward pass uses `+=`.

Without clearing, old gradients would be added to new gradients.

### 4. Backward pass

```python
loss.backward()
```

This finds the gradient of loss for every parameter.

### 5. Update parameters

```python
for p in n.parameters():
    p.data += -0.05 * p.grad
```

This moves every weight and bias a small step downhill.

### 6. Print loss

```python
print(k, loss.data)
```

Running the loop should usually reduce the loss.

The exact loss can differ because random weights start at different values.