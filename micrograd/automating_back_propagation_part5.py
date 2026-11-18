from defining_data_structures_objects_part2 import Value
from graph_visualisation import draw_dot

# 1. Inputs and weights
x1 = Value(2.0, label="x1")
x2 = Value(0.0, label="x2")
w1 = Value(-3.0, label="w1")
w2 = Value(1.0, label="w2")
b = Value(6.8813735870195432, label="b")

# 2. Forward pass
x1w1 = x1 * w1; x1w1.label = "x1*w1"
x2w2 = x2 * w2; x2w2.label = "x2*w2"
x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = "x1*w1 + x2*w2"
n = x1w1x2w2 + b; n.label = "n"
o = n.tanh(); o.label = "o"

# 3. Backward pass (manual chaining)
o.grad = 1.0
o._backward()
n._backward()
b._backward()        # does nothing (lambda: None)
x1w1x2w2._backward()
x1w1._backward()
x2w2._backward()

# 4. Render the graph with the computed gradients
draw_dot(o, format="svg", rankdir="LR").render("graph", view=True)



# https://youtu.be/VMj-3S1tku0?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&t=4703