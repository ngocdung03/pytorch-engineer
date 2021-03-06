# environment: /Users/moadata/opt/anaconda3/envs/pytorchenv
# pip install pytorch-lightning

# https://pytorch-lightning.readthedocs.io/en/stable/

# With Lightning, dont have to worry about when to set model in train and eval mode, when to use GPU, when to use optimizer.zero_grad(), loss.backward() and optimizer.step(), with torch.no_grad() and .detach()
# Bonus: - Tensorboard support
#        - print tips/hints
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning import Trainer

# Device configuration

# Hyper-parameters
input_size = 784  # 28x28
hidden_size = 500
num_classes = 10
num_epochs = 2
batch_size = 100
learning_rate = 0.001

# Fully connected neural network with one hidden layer
class LitNeuralNet(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LitNeuralNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        # no activation and no softmax at the end
        return out

    def training_step(self, batch, batch_idx):  # Added
        images, labels = batch
        images = images.reshape(-1, 28 * 28)   #.to(device)

        # Forward pass
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)
        
        tensorboard_logs = {'train_loss': loss}  #'log': saving checkpoints
        # use key 'log'
        return {"loss": loss, 'log': tensorboard_logs}   

    # define what happens for testing here

    def train_dataloader(self):                 # Added
        # MNIST dataset
        train_dataset = torchvision.datasets.MNIST(
            root="./data", train=True, transform=transforms.ToTensor(), download=True
        )
        # Data loader
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=batch_size, num_workers=4, shuffle=False
        )
        return train_loader

    def val_dataloader(self):      # Added, should be splitted from the train dataset instead
        test_dataset = torchvision.datasets.MNIST(
            root="./data", train=False, transform=transforms.ToTensor()
        )

        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset, batch_size=batch_size, num_workers=4, shuffle=False
        )
        return test_loader
    
    def validation_step(self, batch, batch_idx):    # Added
        images, labels = batch
        images = images.reshape(-1, 28 * 28)

        # Forward pass
        outputs = self(images)
                        
        loss = F.cross_entropy(outputs, labels)
        return {"val_loss": loss}
    
    def validation_epoch_end(self, outputs):      # Added; executed after each validation epoch
        # outputs = list of dictionaries
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'avg_val_loss': avg_loss}
        # use key 'log'
        return {'val_loss': avg_loss, 'log': tensorboard_logs}  # calculate avg loss
    
    
    def configure_optimizers(self):           # Added
        return torch.optim.Adam(self.parameters(), lr=learning_rate)

if __name__ == '__main__':
    model = LitNeuralNet(input_size, hidden_size, num_classes)
    
    # gpus=8
    # Trainer(fast_dev_run=True) -> runs single batch through training and validation
    # train_percent_check=0.1 -> train only on 10% of data
    trainer = Trainer(max_epochs=num_epochs)   # gpus=1, gradient_clip_val
    trainer.fit(model)
          
    # advanced features
    # distributed_backend
    # (DDP) implements data parallelism at the module level which can run across multiple machines.
    # 16 bit precision
    # log_gpu_memory
    # TPU support
    
    # auto_lr_find: automatically finds a good learning rate before training
    # deterministic: makes training reproducable
    # gradient_clip_val: 0 default
    
    
    # python torch21~.py
    # On terminal: tensorboard --logdir=lightning_logs