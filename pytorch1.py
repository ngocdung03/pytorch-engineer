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