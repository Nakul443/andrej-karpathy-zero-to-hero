from micrograd.part2_defining_data_structures_objects import Value
from graph_visualisation import draw_dot

# 1. Inputs and weights
x1 = Value(2.0, label="x1")
x2 = Value(0.0, label="x2")
w1 = Value(-3.0, label="w1")
w2 = Value(1.0, label="w2")
b = Value(6.8813735870195432, label="b")

# 2. Forward pass
x1w1 = x1 * w1
x1w1.label = "x1*w1"
x2w2 = x2 * w2
x2w2.label = "x2*w2"
x1w1x2w2 = x1w1 + x2w2
x1w1x2w2.label = "x1*w1 + x2*w2"
n = x1w1x2w2 + b
n.label = "n"

# Breaking down tanh:
# tanh(n) = (e^(2n) - 1) / (e^(2n) + 1)
# Note: parentheses around (e + 1) are required so division happens across the whole denominator
e = (2 * n).exp()
e.label = "e"
o = (e - 1) / (e + 1)
o.label = "o"

# 3. Automated Backward pass (topological sort is encapsulated inside Value.backward())
o.backward()

# 4. Render the graph with the computed gradients
draw_dot(o, format="svg", rankdir="LR").render("graph", view=True)


# https://youtu.be/VMj-3S1tku0?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&t=4703
