import matplotlib.pyplot as plt
import numpy as np

# EXERCISE 1: Plotting a function f(x) = 3x^2 - 4x + 5

def f(x):
    return (
        3 * x**2 - 4 * x + 5
    )  # this is a parabola (quadratic function) with a minimum point


# np.arange([start,] stop, [step,] dtype=None) - creates a NumPy array of evenly spaced numbers within a specified interval
# start: beginning of the sequence which is included in the array
# stop: end of the sequence which is not included in the array
# step: difference between each number in the sequence
# dtype: data type of the output array, optional. If not given, the data type is inferred from the input data.
xs = np.arange(-5, 5, 0.25)
ys = f(xs)
plt.plot(xs, ys) # plots the function f(x) over the range of x values from -5 to 5 with a step size of 0.25
plt.savefig("plot.png") # saves the plot as a PNG file named "plot.png" in the current working directory


#---------------------------------------------------------------------------------------------------------------------------------#

# EXERCISE 2: Calculating the derivative of f(x) at a specific point x

# now let's plot the function f(x) and its derivative f'(x) on the same graph'
# in simple words, here we are going to calculate the slope of the function f(x) at a specific point x,
# which is the derivative f'(x), and then plot both the function and its derivative on the same graph



# h = 0.001
# x = 3.0
# print(f(x+h)) # the result is slightly larger than f(x) because we are moving to the right on the x-axis
# print(f(x-h)) # the result is slightly smaller than f(x) because we are moving to the left on the x-axis
# print(f(x+h) - f(x)) # this is the slope of the function at x, which is the derivative f'(x), it tells us how fast the function is changing at that point
# print((f(x+h) - f(x)) / h) # this tells us how much the function is changing per unit change in x, which is the definition of the derivative f'(x)


#---------------------------------------------------------------------------------------------------------------------------------------------#

# EXERCISE 3: Calculating the derivative of a more complex function

# complex case now
a = 2.0
b = -3.0
c = 10.0
d = a*b + c
# print(d) # this is a simple linear combination of the variables a, b, and c


# now we'll calculate the derivative of d with respect to a, b, and c using the same method as before


# 1. the influence of a on d if we change a by a small amount h, while keeping b and c constant
h = 0.0001
a = 2.0
b = -3.0
c = 10.0

d1 = a*b + c
a += h
d2 = (a)*b + c

# print('d1:', d1)
# print('d2:', d2) # d2 will be slightly lower than d1 because we increased a by a small amount h, which makes the product a*b slightly larger, and thus d2 slightly smaller than d1
# print('slope', (d2 - d1) / h) # this is the derivative of d with respect to a,
#                             # which tells us how much d changes when we change a by a small amount h
# # since d2 is smaller than d1, the slope is negative, which means that increasing a decreases d, and vice versa


# 2. the influence of b on d if we change b by a small amount h, while keeping a and c constant
h = 0.0001
a = 2.0
b = -3.0
c = 10.0

d1 = a*b + c
b += h
d2 = a*(b) + c

# print('d1:', d1)
# print('d2:', d2) # d2 will be slightly larger than d1 because we increased b by a small amount h, which makes the product a*b slightly larger
# # since d2 is larger than d1, the slope is positive, which means that increasing b increases d, and vice versa
# print('slope', (d2 - d1) / h) # this is the derivative of d with respect to b, which tells us how much d changes when we change b by h


# 3. the influence of c on d if we change c by a small amount h, while keeping a and b constant
h = 0.0001
a = 2.0
b = -3.0
c = 10.0

d1 = a*b + c
c += h
d2 = a*b + c

print('d1:', d1)
print('d2:', d2) # d2 will be slightly larger than d1 because we increased c by a small amount h, which makes d2 slightly larger than d1
# since d2 is larger than d1, the slope is positive, which means that increasing c increases d, and vice versa
print('slope', (d2 - d1) / h) # this is the derivative of d with respect to c, which tells us how much d changes when we change c by h


#---------------------------------------------------------------------------------------------------------------------------------------------#

