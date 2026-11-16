class Value:
    def __init__(self, data):
        self.data = data


    # If you omit __repr__,
    # Python falls back to its default representation showing only the class name and its memory address:
        # a = Value(2.0)
        # print(a)
        # Output: <__main__.Value object at 0x7f8b2c4e3a90>

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        return Value(self.data + other.data)

    def __mul__(self, other):
        return Value(self.data * other.data)
    

a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)
print(a+b) # this will raise an error because we haven't defined how to add two Value objects together yet
# internally python will call, a.__add__(b), which will return a new Value object with the data attribute set to the sum of the data attributes of a and b, which is -1.0



print((a*b)+c)
# internally python will call, (a.__mul__(b)).__add__(c)