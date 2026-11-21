# in this file we will visualize the computational graph of a Value object,
# and see how back propagation works, and how the gradients are calculated and propagated through the graph

'''
NOTE:
REFER TO THE FILE: micrograd/graph_visualisation.py and the graph as you go through this file,
to understand how the computational graph is built and how the gradients are calculated and propagated through the graph.

'''

# manual code to try and visualize the computational graph of a Value object
# how back propagation works, and how the gradients are calculated and propagated through the graph
from micrograd.part2_defining_data_structures_objects import Value


# dummy function
def lol():
    h = 0.001
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
    L1 = L.data

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
    L2 = L.data


# now if you try to add a small change 'h' to all the inputs individually,
# and see how the output changes, you can calculate the gradients of the inputs with respect to the output

'''
so for dL/dd = (L(d+h) - L(d-h)) / (h)
and for dL/da = (L(a+h) - L(a-h)) / (h)
and for dL/db = (L(b+h) - L(b-h)) / (h)
and for dL/dc = (L(c+h) - L(c-h)) / (h)
and for dL/df = (L(f+h) - L(f-h)) / (h)
'''


#-------------------------------------------------------------------------------------------------------------------------#

# EXERCISE 2: the complex part
# we will try to calculate dL/dc

'''
 
dd / dc = ?

d = c + e
the basic definition: (f(x+h) - f(x)) / h

c = c+h
((c+h) + e) - (c + e)) / h
this will give 1.0

therefore,
dd/dc = 1.0
and
dd/de = 1.0

'''


'''

dL/dc = dL/dd * dd/dc
dL/de = dL/dd * dd/de

We already know:
dL / dd
dd / dc

'''


#------------------------------------------------------------------------------------------------------------------------------------------

# now we're going one layer deeper, and trying to calculate dL/da and dL/db


'''

dL / da = dL / dd * dd / de * de / da
dL / db = dL / dd * dd / de * de / db

'''