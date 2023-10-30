# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 14:02:24 2023

@author: Zaac
"""

import torch 
import cv2
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn


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


device = ('cuda' if torch.cuda.is_available() else 'cpu')


labels = ['daisy','dandelion']


model = ConvModel().to(device)
checkpoint = torch.load('model_outputs/model_flowers.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()



transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

test_image = 'C:/Users/Zaac/Desktop/Workspace/python/torch/datasets/test/daisy/wew.jpeg'


# read the image 
image = cv2.imread(test_image)


# get the label from last folder name and also copy
# copy and store original image
real_class = test_image.split('/')[-2]
original_image = image.copy()

# convert image to deep learning model type and transform
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = transform(image)

image = torch.unsqueeze(image, 0)

with torch.no_grad():
    outputs = model(image.to(device))
    
outputs_label = torch.topk(outputs, 1)

pred_class = labels[int(outputs_label.indices)]

cv2.putText(original_image, f"Actual : {real_class}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
cv2.putText(original_image, f"Pred: {pred_class}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)


print(f"Actual class: {real_class}, pred: {pred_class}")
cv2.imshow('Result', original_image)
cv2.waitKey(0)
cv2.imwrite(f"model_outputs/{real_class}{test_image.split('/')[-1].split('.')[0]}.png", original_image)


















