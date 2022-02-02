# Augograd package
from pyexpat import model
from numpy import require
import torch
# requires_grad = True -> track all operations on the tensor
x = torch.rand(3, requires_grad = True)
y = x + 2

# y was created as a result of an operation, so it has a grad_fn attribute.
# grad_fn: reference a Function that has created the Tensor
print(x)  # created by the user -> grad_fn is None
print(y)
print(y.grad_fn)

# Do more operations on y
z = y * y * 3
print(z)
z = z.mean()
print(z)

# call .backward() to compute the gradients with backpropagation.
# .grad attribute is the partial derivate of the function w.r.t. the tensor
z.backward()   # implicitly for only scalar output
print(x.grad)  #dz/dx  

# torch.autograd is an engine for computing vector-Jacobian product. It computes partial derivates while applying the chain rule.

# Model with non-scalar output: specify a gradient argument that is a tensor of matching shape. Needed for vector-Jacobian product
x = torch.rand(3, requires_grad = True)
y = x * 2
for _ in range(10):
    y = y * 2
    
print(y)
print(y.shape)

v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float32)
y.backward(v)
print(x.grad)

# Stop a tensor from tracking history
# Option 1: requires_grad_(...) changes an existing flag in-place
a = torch.randn(2, 2)
print(a.requires_grad)
b = ((a * 3) / (a - 1))
print(b.grad_fn)
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print(b.grad_fn)
a.requires_grad_(False)

# Option 2: .detach() get a new Tensor with the same content but no gradient computation
a = torch.rand(2, 2, requires_grad=True)
print(a.requires_grad)
b = a.detach()
print(b.requires_grad)

# Option 3: wrap in with torch.no_grad()
a = torch.rand(2, 2, requires_grad=True)
print(a.requires_grad)
with torch.no_grad():
    print((a + 2).requires_grad)
    
# ! backward() accumulates the gradient into .grad -> use .zero_() to empty the gradients before a new optimization step.
weights = torch.ones(4, requires_grad=True)

for epoch in range(3):
    # just a dummy example
    model_output = (weights*3).sum()
    model_output.backward()
    
    print(weights.grad)
    
    # optimize model, i.e. adjust weights...
    with torch.no_grad():
        weights -= 0.1*weights.grad
        
    # this is important! It affects the final weights & output
    weights.grad.zero_()
    
print(weights)
print(model_output) 