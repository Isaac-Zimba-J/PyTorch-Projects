# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 11:17:18 2023

@author: Zaac
"""


import matplotlib.pyplot as plt
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.optim as optim
import time 
import matplotlib
from tqdm.auto import tqdm

matplotlib.style.use('ggplot')

"""
Function to save the trained model to disk.
"""
def save_model(epochs, model, optimizer, criterion):

    torch.save({
        'Epochs' : epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': criterion,
        }, 
        'C:/Users/Zaac/Desktop/Workspace/python/torch/model_outputs/model_flowers.pth'
        )
    
    
"""
Function to save the loss and accuracy plots to disk.
"""  
def save_plots(train_acc, valid_acc, train_loss, valid_loss):

   # accuracy plots
   plt.figure(figsize=(10, 7))
   plt.plot(
       train_acc, color='green', linestyle='-', 
       label='train accuracy'
   )
   plt.plot(
       valid_acc, color='blue', linestyle='-', 
       label='validataion accuracy'
   )
   plt.xlabel('Epochs')
   plt.ylabel('Accuracy')
   plt.legend()
   plt.savefig('C:/Users/Zaac/Desktop/Workspace/python/torch/model_outputs/outputs/accuracy.png')
   
   # loss plots
   plt.figure(figsize=(10, 7))
   plt.plot(
       train_loss, color='orange', linestyle='-', 
       label='train loss'
   )
   plt.plot(
       valid_loss, color='red', linestyle='-', 
       label='validataion loss'
   )
   plt.xlabel('Epochs')
   plt.ylabel('Loss')
   plt.legend()
   plt.savefig('C:/Users/Zaac/Desktop/Workspace/python/torch/model_outputs/outputs/loss.png')


"""
named maize doc but i did some flowers thing then will do other maize 
"""

# Hyperperameters
image_size = (225,225)
batch_size = 32
epochs = 10
learning_rate = 0.001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# prepare the dataset
train_transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.RandomResizedCrop(225),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=(30, 70)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


valid_transform = transforms.Compose([
    transforms.Resize(225),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


# dataset paths 
train_dir = 'C:/Users/Zaac/Desktop/Workspace/python/torch/datasets/flowers/train'
valid_dir = 'C:/Users/Zaac/Desktop/Workspace/python/torch/datasets/flowers/val'


# datasets with transformations on them
train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)

valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_transform)

# dataloaders 
train_loader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True, num_workers=4, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size,shuffle=True, num_workers=4, pin_memory=True)



class ConvModel(nn.Module):
    
    def __init__(self):
        super(ConvModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.conv4 = nn.Conv2d(128, 256, 5)
        
        self.fc1 = nn.Linear(256, 2)
        
        self.pool = nn.MaxPool2d(2,2)
        
    def forward(self, data):
        data = self.pool(F.relu(self.conv1(data)))
        data = self.pool(F.relu(self.conv2(data)))
        data = self.pool(F.relu(self.conv3(data)))
        data = self.pool(F.relu(self.conv4(data)))
        
        bs, _, _, _ = data.shape
        
        data = F.adaptive_avg_pool2d(data, 1).reshape(bs, -1)
        data = self.fc1(data)
        
        return data


# argument parser
# parser = argparse.ArgumentParser()

# parser.add_argument('-e','--epochs', type=int, default=10, help=" number of epochs")

# args = vars(parser.parse_args()) 

# Model instance 

model = ConvModel().to(device)
print(model)

# total parameters and trainable parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"{total_params:,} total parameters.")

total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
print(f"{total_trainable_params:,} training parameters.")


# optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# loss function
criterion = nn.CrossEntropyLoss()



# training  function
def train(model, trainLoader, optimizer, criterion):
    model.train()
    print('--Training---')
    
    train_run_loss = 0.0
    train_run_correct = 0
    counter = 0
    
    for i, data in tqdm(enumerate(trainLoader), total=len(trainLoader)):
        counter += 1
        
        image, lables = data
        image = image.to(device)
        lables = lables.to(device)
        
        optimizer.zero_grad()
        
        # do a forward pass 
        outputs = model(image)
        
        # caluculate loss
        loss = criterion(outputs, lables)
        
        train_run_loss += loss.item()
        
        #calculate the accuracy
        _, preds = torch.max(outputs.data, 1)
        train_run_correct += (preds == lables).sum().item()
        
        #backpropagation
        loss.backward()
        #updatet the parameters
        optimizer.step()
        
    epoch_loss = train_run_loss / counter
    epoch_accry = 100. * ( train_run_correct / len(trainLoader.dataset))
    
    
    return epoch_loss, epoch_accry
        

# validation function
def validate(model, testloader, criterion):
    model.eval()
    print('Validation')
    valid_run_loss = 0.0
    valid_run_correct = 0
    counter = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            counter += 1
            
            image, labels = data
            image = image.to(device)
            labels = labels.to(device)
            # forward pass
            outputs = model(image)
            # calculate the loss
            loss = criterion(outputs, labels)
            valid_run_loss += loss.item()
            # calculate the accuracy
            _, preds = torch.max(outputs.data, 1)
            valid_run_correct += (preds == labels).sum().item()
        
    # loss and accuracy for the complete epoch
    epoch_loss = valid_run_loss / counter
    epoch_acc = 100. * (valid_run_correct / len(testloader.dataset))
    return epoch_loss, epoch_acc
  
    

# lists to keep track of losses and accuracies
train_loss, valid_loss = [], []
train_acc, valid_acc = [], []


# start the training
for epoch in range(epochs):
    print(f"[INFO]: Epoch {epoch+1} of {epochs}")
    train_epoch_loss, train_epoch_acc = train(model, train_loader, 
                                              optimizer, criterion)
    valid_epoch_loss, valid_epoch_acc = validate(model, valid_loader,  
                                                 criterion)
    train_loss.append(train_epoch_loss)
    valid_loss.append(valid_epoch_loss)
    train_acc.append(train_epoch_acc)
    valid_acc.append(valid_epoch_acc)
    print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
    print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
    print('-'*50)
    time.sleep(5)
    
# save the trained model weights
save_model(epochs, model, optimizer, criterion)
# save the loss and accuracy plots
save_plots(train_acc, valid_acc, train_loss, valid_loss)
print('------------[TRAINING IS DONE ]----------')
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
