import torch

x1 = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
x2 = torch.tensor([0.0], dtype=torch.float64, requires_grad=True)
w1 = torch.tensor([-3.0], dtype=torch.float64, requires_grad=True)
w2 = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
b  = torch.tensor([6.8813735870195432], dtype=torch.float64, requires_grad=True)

n = x1*w1 + x2*w2 + b

# calculates the hyperbolic tangent of every element in the input tensor n and stores the result in a new tensor named o
o = torch.tanh(n)

# .item(): extracts a single number from a 1-element tensor and converts it into a standard Python float or integer
print(o.data.item())
o.backward()

print('---')
assert x2.grad is not None
print('x2', x2.grad.item())
assert w2.grad is not None
print('w2', w2.grad.item())
assert x1.grad is not None
print('x1', x1.grad.item())
assert w1.grad is not None
print('w1', w1.grad.item())