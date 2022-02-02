# 1) Design model (input, output, forward pass with different layers)
# 2) Construct loss and optimizer
# 3) Training loop
#   - Forward = compute prediction and loss
#   - Backward = compute gradients
#   - Update weights

import torch
import torch.nn as nn

# Linear regression
# f = w * x
# here: f = 2*x

# 0) Training samples
X = torch.tensor([1, 2, 3, 4], dtype = torch.float32)  
Y = torch.tensor([2, 4, 6, 8], dtype = torch.float32)  

# 1) Design model: weights to optimize and forward function
w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True) 

def forward(x):
    return w * x 

# # loss = MSE
# def loss(y, y_pred):
#     return ((y_pred - y)**2).mean()

# # J = MSE = 1/N * (w*x - y)**2
# # dJ/dw = 1/N * 2x(w*x - y)
# def gradient(x, y, y_pred):
#     return np.dot(2*x, y_pred - y).mean()

print(f'Prediction before training: f(5) = {forward(5):.3f}')

# 2) Define loss and optmizer
learning_rate = 0.01
n_iters = 100

# callable function
loss = nn.MSELoss()

optimizer = torch.optim.SGD([w], lr=learning_rate)

# 3) Training loop

for epoch in range(n_iters):
    # predict = forward pass
    y_pred = forward(X)
    
    # loss
    l = loss(Y, y_pred)
    
    # calculate gradients = backward pass
    l.backward()      
    
    # update weights
    # with torch.no_grad():    
    #     w -= learning_rate * w.grad
    optimizer.step()    #
    
    # zero the gradients after updating
    # w.grad.zero_()
    optimizer.zero_grad()
    
    if epoch % 10 == 0:   
        print('epoch ', epoch+1, ': w = ', w, ' loss = ', l)
        
print(f'Prediction after training: f(5) = {forward(5).item():.3f}')

# Introduce the model class
import torch
import torch.nn as nn

# Linear regression
# f = w * x
# here: f = 2*x

# 0) Training samples
X = torch.tensor([[1], [2], [3], [4]], dtype = torch.float32)  #
Y = torch.tensor([[2], [4], [6], [8]], dtype = torch.float32)  #

n_samples, n_features = X.shape #
print(f'#samples: {n_samples}, #features: {n_features})')   #

# 0) create a test sample
X_test = torch.tensor([5], dtype=torch.float32) #

# 1) Design model, the model has to implement the forward pass!
# Here we can us ea built-in model from Pytorch

# w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True) 

# def forward(x):
#     return w * x 

input_size = n_features #
output_size = n_features    #

# we can call this model with samples X
model = nn.Linear(input_size, output_size)  #

'''
class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        
        # define different layers
        self.lin = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.lin(x)

model = LinearRegression(input_size, output_size)    
'''


print(f'Prediction before training: f(5) = {model(X_test).item():.3f}')

# 2) Define loss and optmizer
learning_rate = 0.01
n_iters = 100

# callable function
loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  #

# 3) Training loop

for epoch in range(n_iters):
    # predict = forward pass with our model
    # y_pred = forward(X)
    y_predicted = model(X)
    
    # loss
    l = loss(Y, y_predicted)
    
    # calculate gradients = backward pass
    l.backward()      
    
    # update weights
    optimizer.step()    
    
    # zero the gradients after updating
    optimizer.zero_grad()
    
    if epoch % 10 == 0:   
        [w, b] = model.parameters()  # unpack parameters
        print('epoch ', epoch+1, ': w = ', w[0][0].item(), ' loss = ', l)  #
        
print(f'Prediction after training: f(5) = {model(X_test).item():.3f}')