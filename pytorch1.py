import torch
x = torch.rand(5, 3)
print(x)

# torch.empty(size): uninitialized
x = torch.empty(1)
print(x)

x = torch.empty(2,2,3)
print(x)

# random numbers [0, 1]
x = torch.rand(5, 3)
print(x)

# check size
print(x.size())

# check data type
print(x.dtype)

# specify types, float32 default
x = torch.zeros(5,3, dtype=torch.float16)
print(x)

# check type
print(x.dtype)

# construct from data
x = torch.tensor([5.5, 3])
print(x.size())

# tell pytorch that it will need to calculate the gradients for this tensor later in your optimization steps
x = torch.tensor([5.5, 3], requires_grad=True)

# operations
y = torch.rand(2, 2)
x = torch.rand(2, 2)

# elementwise addition
z = x + y
z = torch.add(x, y)

# in place addition, everything with a trailing underscore is an inplace operation.
# i.e. this will modify the variable y
# y.add_(x)

# slicing 
x = torch.rand(5, 3)
print(x)
print(x[:, 0]) #column 0
print(x[1, :])
print(x[1, 1])

# get the actual value if only 1 element in your tensor
print(x[1, 1].item())

# reshape with torch.view()
x = torch.rand(4, 4)
y = x.view(16)
z = x.view(-1, 8)
print(x.size(), y.size(), z.size())

# Torch <=> Numpy
a = torch.ones(5)
print(a)

b = a.numpy()  #torch to numpy
print(b)
print(type(b))

#! if the Tensor is on CPU, both objects will share the same memory location
a.add_(1)
print(a)
print(b)

# numpy to torch
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
print(a)
print(b)

#! again be careful when modifying
a += 1
print(a)
print(b)

# Move Tensors to GPU
if torch.cuda.is_available():
    device = torch.device("cuda") # a CUDA device object
    y = torch.ones_like(x, device = device) # directely create a tensor on GPU
    x = x.to(device) # or just use strings ``.to("cuda")``
    z = x+y
    # z = z.numpy() # not possible because numpy cannot handle GPU tensors
    # move to CPU again
    z.to("cpu")
    # z = z.numpy()