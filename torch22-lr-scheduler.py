# https://pytorch.org/docs/stable/optim.html
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler

lr = 0.1
model = nn.Linear(10,1)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# LambdaLR
lambda1 = lambda epoch: epoch / 10 #1st epoch: 1*lr/10, 2nd epoch: 2*lr/10 and so on
scheduler = lr_scheduler.LambdaLR(optimizer, lambda1)

print(optimizer.state_dict())
for epoch in range(5):
    # loss.backward()
    optimizer.step()
    # validate(...)
    scheduler.step()
    print(optimizer.state_dict()['param_groups'][0]['lr'])
    
# MultiplicativeLR
lambda2 = lambda epoch: 0.95 #1st epoch: , 2nd epoch:  and so on
scheduler = lr_scheduler.MultiplicativeLR(optimizer, lambda2)  # lr*0.95^n

print(optimizer.state_dict())
for epoch in range(5):
    # loss.backward()
    optimizer.step()
    # validate(...)
    scheduler.step()
    print(optimizer.state_dict()['param_groups'][0]['lr'])
    
# StepLR: one of the most common